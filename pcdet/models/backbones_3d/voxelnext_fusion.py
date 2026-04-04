import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv
from pcdet.models.backbones_3d.spconv_backbone import replace_feature
import torchvision.models as models
from functools import partial

# =====================================================================
# 1. Base Sparse Modules (kept aligned with the VoxelNeXt sparse design)
# =====================================================================
def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))
        return out

class SimpleImageEncoder(nn.Module):
    def __init__(self, out_dim=128, freeze_bn=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.conv_out = nn.Conv2d(256, out_dim, kernel_size=1)
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images):
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        x = (images - mean) / std
        feat = self.backbone(x) 
        feat = self.conv_out(feat)
        return feat

# =====================================================================
# 2. Custom Fusion Modules
# =====================================================================
class InstanceAwareGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.obj_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid() 
        )
        self.scale_proj = nn.Linear(1, channels) 

    def forward(self, x_lidar):
        p_obj = self.obj_head(x_lidar)
        gate_scale = 1.0 + torch.tanh(self.scale_proj(p_obj))
        return p_obj, gate_scale


class DeformableRectifier(nn.Module):
    def __init__(self, dim_lidar, dim_img, stride=8, model_cfg=None):
        super().__init__()
        self.stride = stride
        self.num_heads = 4
        self.scale = (dim_lidar // self.num_heads) ** -0.5
        model_cfg = model_cfg or {}

        self.num_points = int(model_cfg.get('DEFORMABLE_NUM_POINTS', 17))
        self.offset_range = float(model_cfg.get('DEFORMABLE_OFFSET_RANGE', 1.5))
        self.use_deformable_residual = bool(model_cfg.get('USE_DEFORMABLE_RESIDUAL', True))
        self.mask_invalid_samples = bool(model_cfg.get('MASK_INVALID_SAMPLES', True))
        self.register_buffer('base_offsets', self.build_base_offsets(self.num_points), persistent=False)

        self.offset_head = nn.Linear(dim_lidar, self.num_points * 2)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        self.q_proj = nn.Linear(dim_lidar, dim_lidar)
        self.k_proj = nn.Linear(dim_img, dim_lidar)
        self.v_proj = nn.Linear(dim_img, dim_lidar)
        self.out_proj = nn.Linear(dim_lidar, dim_lidar)

    @staticmethod
    def build_base_offsets(num_points):
        preset_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
            (-3, -3), (-3, 0), (-3, 3),
            (0, -3), (0, 3),
            (3, -3), (3, 0), (3, 3),
        ]
        if num_points <= len(preset_offsets):
            return torch.tensor(preset_offsets[:num_points], dtype=torch.float32)

        offsets = list(preset_offsets)
        ring = 4
        while len(offsets) < num_points:
            candidates = []
            for x in range(-ring, ring + 1):
                for y in range(-ring, ring + 1):
                    if max(abs(x), abs(y)) != ring:
                        continue
                    candidates.append((x, y))
            candidates.sort(key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1]), item[0], item[1]))
            for candidate in candidates:
                offsets.append(candidate)
                if len(offsets) >= num_points:
                    break
            ring += 1

        return torch.tensor(offsets, dtype=torch.float32)

    def project_to_image(self, indices, batch_dict, img_shape):
        voxel_size = batch_dict['voxel_size']
        pc_range = batch_dict['point_cloud_range']
        spatial_indices = indices[:, 1:][:, [2, 1, 0]].float() 
        current_voxel_size = voxel_size * self.stride
        physical_xyz = spatial_indices * current_voxel_size + pc_range[:3] + current_voxel_size / 2
        N = physical_xyz.shape[0]
        xyz_hom = torch.cat([physical_xyz, torch.ones((N, 1), device=indices.device)], dim=1)
        trans_mats = batch_dict['trans_lidar_to_img']
        batch_ids = indices[:, 0].long()
        if 'lidar_aug_matrix' in batch_dict:
            aug_mats = batch_dict['lidar_aug_matrix'].float()
            point_aug_inv = torch.inverse(aug_mats[batch_ids])
            xyz_hom = torch.matmul(point_aug_inv, xyz_hom.unsqueeze(-1)).squeeze(-1)
        point_trans_mats = trans_mats[batch_ids].float()

        img_pts_hom = torch.matmul(point_trans_mats, xyz_hom.unsqueeze(-1)).squeeze(-1)
        depth = img_pts_hom[:, 2].clamp(min=1e-5)
        u_img = img_pts_hom[:, 0] / depth
        v_img = img_pts_hom[:, 1] / depth

        _, _, H_feat, W_feat = img_shape
        _, _, H_img, W_img = batch_dict['images'].shape
        scale_x = float(W_feat) / float(W_img)
        scale_y = float(H_feat) / float(H_img)
        u_feat = u_img * scale_x
        v_feat = v_img * scale_y

        denom_w = max(W_feat - 1, 1)
        denom_h = max(H_feat - 1, 1)
        u_norm = 2.0 * (u_feat / denom_w) - 1.0
        v_norm = 2.0 * (v_feat / denom_h) - 1.0
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        return uv_norm, batch_ids

    def forward(self, x_lidar, indices, img_feats, batch_dict):
        centers_uv, batch_inds = self.project_to_image(indices, batch_dict, img_feats.shape)

        B, C, H, W = img_feats.shape
        delta_u = 2.0 / max(W - 1, 1)
        delta_v = 2.0 / max(H - 1, 1)
        base_offsets = self.base_offsets.to(device=x_lidar.device, dtype=x_lidar.dtype)
        base_offsets_norm = torch.stack([
            base_offsets[:, 0] * delta_u,
            base_offsets[:, 1] * delta_v
        ], dim=-1)

        if self.use_deformable_residual:
            learned_offsets = self.offset_head(x_lidar).view(-1, self.num_points, 2)
            learned_offsets = torch.tanh(learned_offsets)
            learned_offsets = torch.stack([
                learned_offsets[..., 0] * (self.offset_range * delta_u),
                learned_offsets[..., 1] * (self.offset_range * delta_v)
            ], dim=-1)
        else:
            learned_offsets = centers_uv.new_zeros((x_lidar.shape[0], self.num_points, 2))

        grid = centers_uv.unsqueeze(1) + base_offsets_norm.unsqueeze(0) + learned_offsets
        valid_mask = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)

        sampled_feats = img_feats.new_zeros((x_lidar.shape[0], self.num_points, C))
        for b in range(B):
            mask = (batch_inds == b)
            if mask.sum() == 0:
                continue
            grid_b = grid[mask].view(1, -1, self.num_points, 2)
            feat_b = F.grid_sample(img_feats[b:b+1], grid_b, align_corners=True, padding_mode='zeros')
            sampled_feats[mask] = feat_b.squeeze(0).permute(1, 2, 0)

        q = self.q_proj(x_lidar).unsqueeze(1)
        k = self.k_proj(sampled_feats)
        v = self.v_proj(sampled_feats)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.mask_invalid_samples:
            attn = attn.masked_fill(~valid_mask.unsqueeze(1), -1e4)
            attn = attn.softmax(dim=-1)
            attn = attn * valid_mask.unsqueeze(1).to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            attn = attn.softmax(dim=-1)

        out = (attn @ v).squeeze(1)

        offset_scale = self.offset_range * max((delta_u ** 2 + delta_v ** 2) ** 0.5, 1e-6)
        offset_instability = learned_offsets.norm(dim=-1).mean(dim=-1, keepdim=True) / max(offset_scale, 1e-6)
        offset_stability = torch.exp(-offset_instability).clamp(min=0.0, max=1.0)
        valid_sample_ratio = valid_mask.float().mean(dim=-1, keepdim=True)
        attn_entropy = -(attn.squeeze(1).clamp(min=1e-6) * torch.log(attn.squeeze(1).clamp(min=1e-6))).sum(dim=-1, keepdim=True)
        attn_focus = 1.0 - attn_entropy / np.log(max(self.num_points, 2))
        attn_focus = attn_focus.clamp(min=0.0, max=1.0)

        rectifier_stats = {
            'valid_sample_ratio': valid_sample_ratio,
            'offset_stability': offset_stability,
            'attn_focus': attn_focus,
        }
        return self.out_proj(out), rectifier_stats


class FusionRectifierBlock(nn.Module):
    def __init__(self, d_model, stride=8, model_cfg=None):
        super().__init__()
        self.rectifier = DeformableRectifier(d_model, d_model, stride=stride, model_cfg=model_cfg)
        self.gate = InstanceAwareGate(d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_sparse, batch_dict, img_feats):
        x = x_sparse.features
        indices = x_sparse.indices
        x_rect, rectifier_stats = self.rectifier(x, indices, img_feats, batch_dict)
        p_obj, scale = self.gate(x)
        x_in = self.fusion_proj(torch.cat([x, x_rect], dim=-1))
        x_in = x_in * scale
        x_out = self.norm(x + x_in)
        x_sparse = x_sparse.replace_feature(x_out)

        fusion_gain = x_in.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6)
        fusion_gain = (1.0 - torch.exp(-fusion_gain)).clamp(min=0.0, max=1.0)

        if 'fusion_gate_scores' not in batch_dict:
            batch_dict['fusion_gate_scores'] = []
            batch_dict['fusion_valid_sample_ratio'] = []
            batch_dict['fusion_offset_stability'] = []
            batch_dict['fusion_attn_focus'] = []
            batch_dict['fusion_gain'] = []
        batch_dict['fusion_gate_scores'].append(p_obj)
        batch_dict['fusion_valid_sample_ratio'].append(rectifier_stats['valid_sample_ratio'])
        batch_dict['fusion_offset_stability'].append(rectifier_stats['offset_stability'])
        batch_dict['fusion_attn_focus'].append(rectifier_stats['attn_focus'])
        batch_dict['fusion_gain'].append(fusion_gain)
        
        if self.training:
            if 'aux_preds' not in batch_dict: batch_dict['aux_preds'] = []
            batch_dict['aux_preds'].append(p_obj)
            if 'aux_indices' not in batch_dict:
                batch_dict['aux_indices'] = indices
        return x_sparse


# =====================================================================
# 3. Sparse VoxelNeXt Backbone with the Custom Fusion Blocks
# =====================================================================
class VoxelNeXtFusion(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.register_buffer('voxel_size', torch.tensor(model_cfg.VOXEL_SIZE).float())
        self.register_buffer('point_cloud_range', torch.tensor(model_cfg.POINT_CLOUD_RANGE).float())

        # Image backbone used by the fusion blocks.
        self.use_fusion = model_cfg.get('USE_DEFORM_RECTIFY', False)
        if self.use_fusion:
            freeze_img_bn = model_cfg.get('FREEZE_IMAGE_BN', True)
            self.img_backbone = SimpleImageEncoder(out_dim=channels[3], freeze_bn=freeze_img_bn)
            print(
                f"[VoxelNeXtFusion] "
                f"freeze image BN: {freeze_img_bn}, "
                f"deformable points: {model_cfg.get('DEFORMABLE_NUM_POINTS', 17)}, "
                f"offset range: {model_cfg.get('DEFORMABLE_OFFSET_RANGE', 1.5)}"
            )
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        # Inject the image-LiDAR fusion blocks at conv4 (stride = 8).
        num_blocks = model_cfg.get('NUM_BLOCKS', 3)
        self.fusion_blocks = nn.ModuleList([
            FusionRectifierBlock(d_model=channels[3], stride=8, model_cfg=model_cfg) for _ in range(num_blocks)
        ])

        # Keep conv5 / conv6 after fusion to enlarge the receptive field.
        self.conv5 = spconv.SparseSequential(
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        self.conv6 = spconv.SparseSequential(
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )

        # Keep the official 2D BEV post-processing block.
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.num_point_features = out_channel

    def bev_out(self, x_conv):
        # Sparse sum-pooling from 3D sparse features to BEV sparse features.
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['point_cloud_range'] = self.point_cloud_range

        if self.use_fusion:
            if 'image_features' not in batch_dict:
                batch_dict['image_features'] = self.img_backbone(batch_dict['images'])
            img_feats = batch_dict['image_features']

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size']
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # Perform image-LiDAR fusion at the stride-8 sparse feature stage.
        if self.use_fusion:
            for block in self.fusion_blocks:
                x_conv4 = block(x_conv4, batch_dict, img_feats)

            if 'fusion_gate_scores' in batch_dict:
                quality_features = torch.cat([
                    torch.stack(batch_dict.pop('fusion_gate_scores'), dim=0).mean(dim=0),
                    torch.stack(batch_dict.pop('fusion_valid_sample_ratio'), dim=0).mean(dim=0),
                    torch.stack(batch_dict.pop('fusion_offset_stability'), dim=0).mean(dim=0),
                    torch.stack(batch_dict.pop('fusion_attn_focus'), dim=0).mean(dim=0),
                    torch.stack(batch_dict.pop('fusion_gain'), dim=0).mean(dim=0),
                ], dim=-1)
                batch_dict['fusion_quality_features'] = quality_features
                batch_dict['fusion_quality_indices'] = x_conv4.indices
                batch_dict['fusion_quality_stride'] = 8

        # Continue extracting large-receptive-field features after fusion.
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        # Multi-scale feature aggregation.
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        # BEV output and refinement.
        out = self.bev_out(x_conv4)
        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict['encoded_spconv_tensor'] = out
        batch_dict['encoded_spconv_tensor_stride'] = 8
        
        return batch_dict



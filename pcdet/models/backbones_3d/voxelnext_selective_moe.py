import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial

from pcdet.models.backbones_3d.spconv_backbone import replace_feature
from .voxelnext_fusion import (
    DeformableRectifier,
    SimpleImageEncoder,
    SparseBasicBlock,
    post_act_block,
)


class ObservationRouter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.fg_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.router = nn.Sequential(
            nn.Linear(channels + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, 0.0)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([2.0, 0.0, -1.0]))

    def forward(self, x_lidar, range_norm, projection_valid):
        p_fg = self.fg_head(x_lidar)
        router_input = torch.cat([x_lidar, range_norm, p_fg, projection_valid], dim=-1)
        weights = self.router(router_input).softmax(dim=-1)
        return p_fg, weights


class LightFusionExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x_lidar, sampled_img_feats, valid_mask):
        delta = self.fusion(torch.cat([x_lidar, sampled_img_feats], dim=-1))
        return delta * valid_mask.to(delta.dtype)


class SelectiveMoEFusionBlock(nn.Module):
    def __init__(self, channels, stride=8, model_cfg=None, max_bev_range=1.0):
        super().__init__()
        model_cfg = model_cfg or {}
        self.stride = stride
        self.rectify_ratio = float(model_cfg.get('RECTIFY_RATIO', 0.25))
        self.router = ObservationRouter(channels)
        self.light_expert = LightFusionExpert(channels)
        self.rectified_expert = DeformableRectifier(channels, channels, stride=stride, model_cfg=model_cfg)
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('max_bev_range', torch.tensor(float(max_bev_range)), persistent=False)

    def get_voxel_centers(self, indices, batch_dict):
        voxel_size = batch_dict['voxel_size'].to(indices.device).float()
        pc_range = batch_dict['point_cloud_range'].to(indices.device).float()
        spatial_indices = indices[:, 1:][:, [2, 1, 0]].float()
        current_voxel_size = voxel_size * self.stride
        return spatial_indices * current_voxel_size + pc_range[:3] + current_voxel_size / 2

    def project_to_image(self, centers_xyz, indices, batch_dict, img_shape):
        xyz_hom = torch.cat(
            [centers_xyz, torch.ones((centers_xyz.shape[0], 1), device=centers_xyz.device)],
            dim=1,
        )
        batch_ids = indices[:, 0].long()
        if 'lidar_aug_matrix' in batch_dict:
            aug_mats = batch_dict['lidar_aug_matrix'].float()
            point_aug_inv = torch.inverse(aug_mats[batch_ids])
            xyz_hom = torch.matmul(point_aug_inv, xyz_hom.unsqueeze(-1)).squeeze(-1)

        trans_mats = batch_dict['trans_lidar_to_img'][batch_ids].float()
        img_pts_hom = torch.matmul(trans_mats, xyz_hom.unsqueeze(-1)).squeeze(-1)
        depth = img_pts_hom[:, 2]
        safe_depth = depth.clamp(min=1e-5)
        u_img = img_pts_hom[:, 0] / safe_depth
        v_img = img_pts_hom[:, 1] / safe_depth

        _, _, H_feat, W_feat = img_shape
        _, _, H_img, W_img = batch_dict['images'].shape
        u_feat = u_img * (float(W_feat) / float(W_img))
        v_feat = v_img * (float(H_feat) / float(H_img))

        denom_w = max(W_feat - 1, 1)
        denom_h = max(H_feat - 1, 1)
        u_norm = 2.0 * (u_feat / denom_w) - 1.0
        v_norm = 2.0 * (v_feat / denom_h) - 1.0
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        valid = (depth > 1e-5) & (uv_norm[:, 0].abs() <= 1.0) & (uv_norm[:, 1].abs() <= 1.0)
        return uv_norm, valid.unsqueeze(-1), batch_ids

    @staticmethod
    def sample_image_features(img_feats, uv_norm, batch_ids):
        B, C, _, _ = img_feats.shape
        sampled = img_feats.new_zeros((uv_norm.shape[0], C))
        for b in range(B):
            mask = batch_ids == b
            if mask.sum() == 0:
                continue
            grid = uv_norm[mask].view(1, -1, 1, 2)
            feat = F.grid_sample(img_feats[b:b + 1], grid, align_corners=True, padding_mode='zeros')
            sampled[mask] = feat.squeeze(0).squeeze(-1).transpose(0, 1)
        return sampled

    def select_rectified_tokens(self, score, valid_mask, batch_ids):
        selected = torch.zeros(score.shape[0], device=score.device, dtype=torch.bool)
        if self.rectify_ratio <= 0:
            return selected

        for b in batch_ids.unique(sorted=True):
            cur_mask = (batch_ids == b) & valid_mask.squeeze(-1)
            num_valid = int(cur_mask.sum().item())
            if num_valid == 0:
                continue
            k = max(1, int(torch.ceil(score.new_tensor(num_valid * self.rectify_ratio)).item()))
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(-1)
            k = min(k, cur_idx.numel())
            top_idx = torch.topk(score[cur_idx], k=k, largest=True).indices
            selected[cur_idx[top_idx]] = True
        return selected

    def forward(self, x_sparse, batch_dict, img_feats):
        x = x_sparse.features
        indices = x_sparse.indices
        centers_xyz = self.get_voxel_centers(indices, batch_dict)
        range_norm = torch.linalg.norm(centers_xyz[:, :2], dim=-1, keepdim=True)
        range_norm = (range_norm / self.max_bev_range.clamp(min=1e-6)).clamp(min=0.0, max=1.0)

        uv_norm, projection_valid, batch_ids = self.project_to_image(centers_xyz, indices, batch_dict, img_feats.shape)
        p_fg, route_weights = self.router(x, range_norm, projection_valid.float())

        sampled_img_feats = self.sample_image_features(img_feats, uv_norm, batch_ids)
        delta_light = self.light_expert(x, sampled_img_feats, projection_valid)
        w_light = route_weights[:, 1:2]
        w_rectified = route_weights[:, 2:3]
        delta = w_light * delta_light

        rect_score = (p_fg.detach() * range_norm * projection_valid.float()).squeeze(-1)
        selected_mask = self.select_rectified_tokens(rect_score, projection_valid.bool(), batch_ids)
        rect_valid_sample_ratio = x.new_zeros((x.shape[0], 1))
        rect_offset_stability = x.new_zeros((x.shape[0], 1))
        rect_attn_focus = x.new_zeros((x.shape[0], 1))

        if selected_mask.any():
            rect_out, rect_stats = self.rectified_expert(
                x[selected_mask],
                indices[selected_mask],
                img_feats,
                batch_dict,
            )
            delta[selected_mask] = delta[selected_mask] + w_rectified[selected_mask] * rect_out
            rect_valid_sample_ratio[selected_mask] = rect_stats['valid_sample_ratio']
            rect_offset_stability[selected_mask] = rect_stats['offset_stability']
            rect_attn_focus[selected_mask] = rect_stats['attn_focus']
        elif self.training:
            dummy = x.new_tensor(0.0)
            for param in self.rectified_expert.parameters():
                dummy = dummy + param.sum() * 0.0
            delta = delta + dummy

        fusion_gain = delta.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6)
        fusion_gain = (1.0 - torch.exp(-fusion_gain)).clamp(min=0.0, max=1.0)
        x_sparse = x_sparse.replace_feature(self.norm(x + delta))

        quality_features = torch.cat([
            route_weights,
            p_fg,
            range_norm,
            projection_valid.float(),
            selected_mask.float().unsqueeze(-1),
            fusion_gain,
            rect_valid_sample_ratio,
            rect_offset_stability,
            rect_attn_focus,
        ], dim=-1).clamp(min=0.0, max=1.0)

        batch_dict['fusion_quality_features'] = quality_features
        batch_dict['fusion_quality_indices'] = x_sparse.indices
        batch_dict['fusion_quality_stride'] = self.stride
        batch_dict['moe_route_weights'] = route_weights.detach()

        if self.training:
            batch_dict.setdefault('aux_preds', []).append(p_fg)
            batch_dict['aux_indices'] = indices

        return x_sparse


class VoxelNeXtSelectiveMoE(nn.Module):
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

        pc_range = torch.tensor(model_cfg.POINT_CLOUD_RANGE).float()
        max_x = torch.max(pc_range[0].abs(), pc_range[3].abs())
        max_y = torch.max(pc_range[1].abs(), pc_range[4].abs())
        max_bev_range = torch.sqrt(max_x ** 2 + max_y ** 2).item()

        freeze_img_bn = model_cfg.get('FREEZE_IMAGE_BN', True)
        self.img_backbone = SimpleImageEncoder(out_dim=channels[3], freeze_bn=freeze_img_bn)
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
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.selective_moe_fusion = SelectiveMoEFusionBlock(
            channels=channels[3],
            stride=8,
            model_cfg=model_cfg,
            max_bev_range=max_bev_range,
        )

        self.conv5 = spconv.SparseSequential(
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        self.conv6 = spconv.SparseSequential(
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )
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
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        return spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size,
        )

    def forward(self, batch_dict):
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['point_cloud_range'] = self.point_cloud_range
        if 'image_features' not in batch_dict:
            batch_dict['image_features'] = self.img_backbone(batch_dict['images'])
        img_feats = batch_dict['image_features']

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size'],
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv4 = self.selective_moe_fusion(x_conv4, batch_dict, img_feats)

        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)
        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict['encoded_spconv_tensor'] = out
        batch_dict['encoded_spconv_tensor_stride'] = 8
        return batch_dict

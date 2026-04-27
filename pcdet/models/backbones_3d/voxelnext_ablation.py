import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial

from pcdet.models.backbones_3d.spconv_backbone import replace_feature
from .voxelnext_fusion import DeformableRectifier, SimpleImageEncoder, SparseBasicBlock, post_act_block
from .voxelnext_selective_moe import LightFusionExpert, ObservationRouter


class ForegroundHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_lidar):
        return self.head(x_lidar)


class ConventionalFusionExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x_lidar, img_token, valid_mask=None):
        delta = self.fusion(torch.cat([x_lidar, img_token], dim=-1))
        if valid_mask is not None:
            delta = delta * valid_mask.to(delta.dtype)
        return delta


class ProjectedFusionMixin:
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

        _, _, h_feat, w_feat = img_shape
        _, _, h_img, w_img = batch_dict['images'].shape
        u_feat = u_img * (float(w_feat) / float(w_img))
        v_feat = v_img * (float(h_feat) / float(h_img))

        denom_w = max(w_feat - 1, 1)
        denom_h = max(h_feat - 1, 1)
        u_norm = 2.0 * (u_feat / denom_w) - 1.0
        v_norm = 2.0 * (v_feat / denom_h) - 1.0
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        valid = (depth > 1e-5) & (uv_norm[:, 0].abs() <= 1.0) & (uv_norm[:, 1].abs() <= 1.0)
        return uv_norm, valid.unsqueeze(-1), batch_ids

    @staticmethod
    def sample_image_features(img_feats, uv_norm, batch_ids):
        batch_size, channels, _, _ = img_feats.shape
        sampled = img_feats.new_zeros((uv_norm.shape[0], channels))
        for b in range(batch_size):
            mask = batch_ids == b
            if mask.sum() == 0:
                continue
            grid = uv_norm[mask].view(1, -1, 1, 2)
            feat = F.grid_sample(img_feats[b:b + 1], grid, align_corners=True, padding_mode='zeros')
            sampled[mask] = feat.squeeze(0).squeeze(-1).transpose(0, 1)
        return sampled

    @staticmethod
    def build_candidate_offsets(num_points):
        return DeformableRectifier.build_base_offsets(num_points)

    def sample_candidate_features(self, img_feats, centers_uv, batch_ids, offsets_xy):
        batch_size, channels, h_feat, w_feat = img_feats.shape
        delta_u = 2.0 / max(w_feat - 1, 1)
        delta_v = 2.0 / max(h_feat - 1, 1)
        offsets_xy = offsets_xy.to(device=centers_uv.device, dtype=centers_uv.dtype)
        offsets_norm = torch.stack([offsets_xy[:, 0] * delta_u, offsets_xy[:, 1] * delta_v], dim=-1)
        grid = centers_uv.unsqueeze(1) + offsets_norm.unsqueeze(0)
        valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)

        sampled = img_feats.new_zeros((centers_uv.shape[0], offsets_xy.shape[0], channels))
        for b in range(batch_size):
            mask = batch_ids == b
            if mask.sum() == 0:
                continue
            grid_b = grid[mask].view(1, -1, offsets_xy.shape[0], 2)
            feat = F.grid_sample(img_feats[b:b + 1], grid_b, align_corners=True, padding_mode='zeros')
            sampled[mask] = feat.squeeze(0).permute(1, 2, 0)
        return sampled, valid

    def build_quality_stats(
        self,
        route_weights,
        p_fg,
        range_norm,
        projection_valid,
        rect_selected,
        fusion_gain,
        rect_valid_sample_ratio,
        rect_offset_stability,
        rect_attn_focus,
    ):
        return {
            'route_weights': route_weights,
            'p_fg': p_fg,
            'range_norm': range_norm,
            'projection_valid': projection_valid.float(),
            'rect_selected': rect_selected.float(),
            'fusion_gain': fusion_gain,
            'rect_valid_sample_ratio': rect_valid_sample_ratio,
            'rect_offset_stability': rect_offset_stability,
            'rect_attn_focus': rect_attn_focus,
        }


class DirectFusionBlock(nn.Module, ProjectedFusionMixin):
    def __init__(self, channels, stride=8, model_cfg=None, max_bev_range=1.0):
        super().__init__()
        self.stride = stride
        self.fg_head = ForegroundHead(channels)
        self.expert = ConventionalFusionExpert(channels)
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('max_bev_range', torch.tensor(float(max_bev_range)), persistent=False)

    def forward(self, x_sparse, batch_dict, img_feats):
        x = x_sparse.features
        indices = x_sparse.indices
        centers_xyz = self.get_voxel_centers(indices, batch_dict)
        range_norm = torch.linalg.norm(centers_xyz[:, :2], dim=-1, keepdim=True)
        range_norm = (range_norm / self.max_bev_range.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        uv_norm, projection_valid, batch_ids = self.project_to_image(centers_xyz, indices, batch_dict, img_feats.shape)
        sampled_img = self.sample_image_features(img_feats, uv_norm, batch_ids)
        p_fg = self.fg_head(x)
        delta = self.expert(x, sampled_img, projection_valid)
        fusion_gain = delta.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6)
        fusion_gain = (1.0 - torch.exp(-fusion_gain)).clamp(min=0.0, max=1.0)

        route_weights = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        route_weights[:, 1] = 1.0
        ones = projection_valid.float().new_ones(projection_valid.shape)
        zeros = projection_valid.float().new_zeros(projection_valid.shape)
        quality_stats = self.build_quality_stats(
            route_weights=route_weights,
            p_fg=p_fg,
            range_norm=range_norm,
            projection_valid=projection_valid,
            rect_selected=zeros,
            fusion_gain=fusion_gain,
            rect_valid_sample_ratio=projection_valid.float(),
            rect_offset_stability=ones,
            rect_attn_focus=ones,
        )
        x_sparse = x_sparse.replace_feature(self.norm(x + delta))
        return x_sparse, quality_stats


class DeformableFusionBlock(nn.Module, ProjectedFusionMixin):
    def __init__(self, channels, stride=8, model_cfg=None, max_bev_range=1.0):
        super().__init__()
        self.stride = stride
        self.fg_head = ForegroundHead(channels)
        self.rectifier = DeformableRectifier(channels, channels, stride=stride, model_cfg=model_cfg)
        self.expert = ConventionalFusionExpert(channels)
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('max_bev_range', torch.tensor(float(max_bev_range)), persistent=False)

    def forward(self, x_sparse, batch_dict, img_feats):
        x = x_sparse.features
        indices = x_sparse.indices
        centers_xyz = self.get_voxel_centers(indices, batch_dict)
        range_norm = torch.linalg.norm(centers_xyz[:, :2], dim=-1, keepdim=True)
        range_norm = (range_norm / self.max_bev_range.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        _, projection_valid, _ = self.project_to_image(centers_xyz, indices, batch_dict, img_feats.shape)
        x_rect, rect_stats = self.rectifier(x, indices, img_feats, batch_dict)
        p_fg = self.fg_head(x)
        delta = self.expert(x, x_rect, projection_valid)
        fusion_gain = delta.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6)
        fusion_gain = (1.0 - torch.exp(-fusion_gain)).clamp(min=0.0, max=1.0)

        route_weights = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        route_weights[:, 2] = 1.0
        quality_stats = self.build_quality_stats(
            route_weights=route_weights,
            p_fg=p_fg,
            range_norm=range_norm,
            projection_valid=projection_valid,
            rect_selected=projection_valid.float(),
            fusion_gain=fusion_gain,
            rect_valid_sample_ratio=rect_stats['valid_sample_ratio'],
            rect_offset_stability=rect_stats['offset_stability'],
            rect_attn_focus=rect_stats['attn_focus'],
        )
        x_sparse = x_sparse.replace_feature(self.norm(x + delta))
        return x_sparse, quality_stats


class RuleRouteMoEBlock(nn.Module, ProjectedFusionMixin):
    def __init__(self, channels, stride=8, model_cfg=None, max_bev_range=1.0):
        super().__init__()
        self.stride = stride
        self.fg_head = ForegroundHead(channels)
        self.light_expert = LightFusionExpert(channels)
        self.rectifier = DeformableRectifier(channels, channels, stride=stride, model_cfg=model_cfg)
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('max_bev_range', torch.tensor(float(max_bev_range)), persistent=False)

    def forward(self, x_sparse, batch_dict, img_feats):
        x = x_sparse.features
        indices = x_sparse.indices
        centers_xyz = self.get_voxel_centers(indices, batch_dict)
        distance = torch.linalg.norm(centers_xyz[:, :2], dim=-1, keepdim=True)
        range_norm = (distance / self.max_bev_range.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        uv_norm, projection_valid, batch_ids = self.project_to_image(centers_xyz, indices, batch_dict, img_feats.shape)
        sampled_img = self.sample_image_features(img_feats, uv_norm, batch_ids)
        p_fg = self.fg_head(x)

        route_weights = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        near_mask = distance.squeeze(-1) < 50.0
        mid_mask = (distance.squeeze(-1) >= 50.0) & (distance.squeeze(-1) < 100.0)
        far_mask = distance.squeeze(-1) >= 100.0
        route_weights[near_mask, 0] = 1.0
        route_weights[mid_mask, 1] = 1.0
        route_weights[far_mask, 2] = 1.0

        delta_light = self.light_expert(x, sampled_img, projection_valid)
        delta_rect, rect_stats = self.rectifier(x, indices, img_feats, batch_dict)
        delta = route_weights[:, 1:2] * delta_light + route_weights[:, 2:3] * delta_rect
        fusion_gain = delta.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6)
        fusion_gain = (1.0 - torch.exp(-fusion_gain)).clamp(min=0.0, max=1.0)

        quality_stats = self.build_quality_stats(
            route_weights=route_weights,
            p_fg=p_fg,
            range_norm=range_norm,
            projection_valid=projection_valid,
            rect_selected=route_weights[:, 2:3] * projection_valid.float(),
            fusion_gain=fusion_gain,
            rect_valid_sample_ratio=rect_stats['valid_sample_ratio'],
            rect_offset_stability=rect_stats['offset_stability'],
            rect_attn_focus=rect_stats['attn_focus'],
        )
        x_sparse = x_sparse.replace_feature(self.norm(x + delta))
        return x_sparse, quality_stats


class AllSampleRectifiedExpert(nn.Module, ProjectedFusionMixin):
    def __init__(self, channels, stride=8, model_cfg=None):
        super().__init__()
        self.stride = stride
        model_cfg = model_cfg or {}
        self.num_points = int(model_cfg.get('DEFORMABLE_NUM_POINTS', 17))
        self.register_buffer('candidate_offsets', self.build_candidate_offsets(self.num_points), persistent=False)
        self.fusion = ConventionalFusionExpert(channels)

    def forward(self, x_lidar, centers_uv, batch_ids, projection_valid, img_feats):
        sampled_feats, valid_mask = self.sample_candidate_features(
            img_feats=img_feats,
            centers_uv=centers_uv,
            batch_ids=batch_ids,
            offsets_xy=self.candidate_offsets,
        )
        valid_float = valid_mask.float()
        pooled_img = (sampled_feats * valid_float.unsqueeze(-1)).sum(dim=1)
        pooled_img = pooled_img / valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        delta = self.fusion(x_lidar, pooled_img, projection_valid)

        valid_sample_ratio = valid_float.mean(dim=-1, keepdim=True)
        ones = valid_sample_ratio.new_ones(valid_sample_ratio.shape)
        return delta, {
            'valid_sample_ratio': valid_sample_ratio,
            'offset_stability': ones,
            'attn_focus': ones,
        }


class AllSampleMoEBlock(nn.Module, ProjectedFusionMixin):
    def __init__(self, channels, stride=8, model_cfg=None, max_bev_range=1.0):
        super().__init__()
        self.stride = stride
        self.router = ObservationRouter(channels)
        self.light_expert = LightFusionExpert(channels)
        self.rectified_expert = AllSampleRectifiedExpert(channels, stride=stride, model_cfg=model_cfg)
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('max_bev_range', torch.tensor(float(max_bev_range)), persistent=False)

    def forward(self, x_sparse, batch_dict, img_feats):
        x = x_sparse.features
        indices = x_sparse.indices
        centers_xyz = self.get_voxel_centers(indices, batch_dict)
        range_norm = torch.linalg.norm(centers_xyz[:, :2], dim=-1, keepdim=True)
        range_norm = (range_norm / self.max_bev_range.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        centers_uv, projection_valid, batch_ids = self.project_to_image(centers_xyz, indices, batch_dict, img_feats.shape)

        p_fg, route_weights = self.router(x, range_norm, projection_valid.float())
        sampled_img = self.sample_image_features(img_feats, centers_uv, batch_ids)
        delta_light = self.light_expert(x, sampled_img, projection_valid)
        delta_rect, rect_stats = self.rectified_expert(x, centers_uv, batch_ids, projection_valid, img_feats)

        delta = route_weights[:, 1:2] * delta_light + route_weights[:, 2:3] * delta_rect
        fusion_gain = delta.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6)
        fusion_gain = (1.0 - torch.exp(-fusion_gain)).clamp(min=0.0, max=1.0)
        rect_selected = (route_weights[:, 2:3] >= route_weights[:, 1:2]).float() * projection_valid.float()

        quality_stats = self.build_quality_stats(
            route_weights=route_weights,
            p_fg=p_fg,
            range_norm=range_norm,
            projection_valid=projection_valid,
            rect_selected=rect_selected,
            fusion_gain=fusion_gain,
            rect_valid_sample_ratio=rect_stats['valid_sample_ratio'],
            rect_offset_stability=rect_stats['offset_stability'],
            rect_attn_focus=rect_stats['attn_focus'],
        )
        x_sparse = x_sparse.replace_feature(self.norm(x + delta))
        return x_sparse, quality_stats


class VoxelNeXtAblationBase(nn.Module):
    fusion_block_cls = None

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        assert self.fusion_block_cls is not None
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

        num_blocks = model_cfg.get('NUM_BLOCKS', 3)
        self.fusion_blocks = nn.ModuleList([
            self.fusion_block_cls(channels[3], stride=8, model_cfg=model_cfg, max_bev_range=max_bev_range)
            for _ in range(num_blocks)
        ])

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

    @staticmethod
    def bev_out(x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, inv, features_cat)

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

        quality_stats_list = []
        aux_preds = []
        for block in self.fusion_blocks:
            x_conv4, block_stats = block(x_conv4, batch_dict, img_feats)
            quality_stats_list.append(block_stats)
            aux_preds.append(block_stats['p_fg'])

        if len(quality_stats_list) > 0:
            route_weights = torch.stack([item['route_weights'] for item in quality_stats_list], dim=0).mean(dim=0)
            p_fg = torch.stack([item['p_fg'] for item in quality_stats_list], dim=0).mean(dim=0)
            range_norm = torch.stack([item['range_norm'] for item in quality_stats_list], dim=0).mean(dim=0)
            projection_valid = torch.stack([item['projection_valid'] for item in quality_stats_list], dim=0).mean(dim=0)
            rect_selected = torch.stack([item['rect_selected'] for item in quality_stats_list], dim=0).mean(dim=0)
            fusion_gain = torch.stack([item['fusion_gain'] for item in quality_stats_list], dim=0).mean(dim=0)
            rect_valid_sample_ratio = torch.stack([item['rect_valid_sample_ratio'] for item in quality_stats_list], dim=0).mean(dim=0)
            rect_offset_stability = torch.stack([item['rect_offset_stability'] for item in quality_stats_list], dim=0).mean(dim=0)
            rect_attn_focus = torch.stack([item['rect_attn_focus'] for item in quality_stats_list], dim=0).mean(dim=0)
            batch_dict['fusion_quality_features'] = torch.cat([
                route_weights,
                p_fg,
                range_norm,
                projection_valid,
                rect_selected,
                fusion_gain,
                rect_valid_sample_ratio,
                rect_offset_stability,
                rect_attn_focus,
            ], dim=-1).clamp(min=0.0, max=1.0)
            batch_dict['fusion_quality_indices'] = x_conv4.indices
            batch_dict['fusion_quality_stride'] = 8

        if self.training:
            batch_dict['aux_preds'] = aux_preds
            batch_dict['aux_indices'] = x_conv4.indices

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


class VoxelNeXtDirectFusion(VoxelNeXtAblationBase):
    fusion_block_cls = DirectFusionBlock


class VoxelNeXtDeformFusion(VoxelNeXtAblationBase):
    fusion_block_cls = DeformableFusionBlock


class VoxelNeXtRuleRouteMoE(VoxelNeXtAblationBase):
    fusion_block_cls = RuleRouteMoEBlock


class VoxelNeXtAllSampleMoE(VoxelNeXtAblationBase):
    fusion_block_cls = AllSampleMoEBlock

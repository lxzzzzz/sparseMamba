import torch
import torch.nn.functional as F

from .detector3d_template import Detector3DTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class SelectiveMoEVoxelNeXt(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.enable_aux_loss = model_cfg.get('ENABLE_AUX_LOSS', True)
        self.aux_loss_weight = model_cfg.get('AUX_LOSS_WEIGHT', 0.1)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            return {'loss': loss}, tb_dict, disp_dict

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_head, tb_dict = self.dense_head.get_loss()
        loss_aux = self.get_aux_loss(batch_dict) if self.enable_aux_loss else loss_head.new_tensor(0.0)
        loss = loss_head + self.aux_loss_weight * loss_aux

        tb_dict['loss_head'] = loss_head.item()
        tb_dict['loss_aux'] = loss_aux.item()
        tb_dict['aux_loss_weight'] = float(self.aux_loss_weight if self.enable_aux_loss else 0.0)
        tb_dict['loss_total'] = loss.item()
        return loss, tb_dict, disp_dict

    def get_aux_loss(self, batch_dict):
        preds_list = batch_dict.get('aux_preds', None)
        if preds_list is None or len(preds_list) == 0:
            return batch_dict['gt_boxes'].new_tensor(0.0)

        indices = batch_dict['aux_indices']
        gt_boxes = batch_dict['gt_boxes']
        voxel_centers = self.get_voxel_centers(indices, batch_dict, stride=8)
        labels = self.assign_voxel_labels(voxel_centers, indices[:, 0], gt_boxes)

        loss = voxel_centers.new_tensor(0.0)
        for pred in preds_list:
            loss = loss + F.binary_cross_entropy(pred.squeeze(-1), labels)
        return loss / len(preds_list)

    @staticmethod
    def get_voxel_centers(indices, batch_dict, stride=8):
        voxel_size = torch.as_tensor(batch_dict['voxel_size'], device=indices.device, dtype=torch.float32)
        pc_range = torch.as_tensor(batch_dict['point_cloud_range'], device=indices.device, dtype=torch.float32)
        spatial_indices = indices[:, 1:][:, [2, 1, 0]].float()
        current_voxel_size = voxel_size * stride
        return spatial_indices * current_voxel_size + pc_range[:3] + current_voxel_size / 2.0

    @staticmethod
    def assign_voxel_labels(points, batch_idxs, gt_boxes):
        labels = points.new_zeros(points.shape[0])
        batch_size = gt_boxes.shape[0]

        for b in range(batch_size):
            mask = batch_idxs == b
            if mask.sum() == 0:
                continue

            cur_points = points[mask]
            cur_boxes = gt_boxes[b]
            in_boxes_result = roiaware_pool3d_utils.points_in_boxes_gpu(
                cur_points.unsqueeze(0),
                cur_boxes[:, :7].unsqueeze(0),
            ).squeeze(0)

            if in_boxes_result.ndim == 1:
                cur_labels = (in_boxes_result >= 0).float()
            else:
                cur_labels = (in_boxes_result.max(dim=1)[0] >= 0).float()
            labels[mask] = cur_labels

        return labels

    def attach_observation_quality(self, pred_boxes, pred_scores, batch_index, batch_dict):
        if pred_boxes.shape[0] == 0:
            empty_quality = pred_boxes.new_zeros((0, 11))
            empty_reliability = pred_scores.new_zeros((0,))
            return empty_quality, empty_reliability

        quality_features = batch_dict.get('fusion_quality_features', None)
        quality_indices = batch_dict.get('fusion_quality_indices', None)
        quality_stride = int(batch_dict.get('fusion_quality_stride', 8))
        if quality_features is None or quality_indices is None or quality_features.shape[0] == 0:
            empty_quality = pred_boxes.new_zeros((pred_boxes.shape[0], 11))
            return empty_quality, pred_scores.clone()

        voxel_centers = self.get_voxel_centers(quality_indices, batch_dict, stride=quality_stride)
        batch_mask = quality_indices[:, 0] == batch_index
        voxel_centers = voxel_centers[batch_mask]
        voxel_quality = quality_features[batch_mask]

        if voxel_centers.shape[0] == 0:
            empty_quality = pred_boxes.new_zeros((pred_boxes.shape[0], quality_features.shape[1]))
            return empty_quality, pred_scores.clone()

        topk = min(8, voxel_centers.shape[0])
        distances = torch.cdist(pred_boxes[:, :3], voxel_centers)
        nearest_dist, nearest_idx = distances.topk(k=topk, dim=1, largest=False)
        weights = 1.0 / (nearest_dist + 1e-3)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled_quality = (voxel_quality[nearest_idx] * weights.unsqueeze(-1)).sum(dim=1)
        pooled_quality = pooled_quality.clamp(min=0.0, max=1.0)

        w_lidar = pooled_quality[:, 0]
        w_light = pooled_quality[:, 1]
        w_rect = pooled_quality[:, 2]
        p_fg = pooled_quality[:, 3]
        projection_valid = pooled_quality[:, 5]
        rect_selected = pooled_quality[:, 6]
        fusion_gain = pooled_quality[:, 7]
        reliability = (
            0.30 * pred_scores
            + 0.15 * w_lidar
            + 0.10 * w_light
            + 0.15 * w_rect
            + 0.15 * p_fg
            + 0.05 * projection_valid
            + 0.05 * rect_selected
            + 0.05 * fusion_gain
        ).clamp(min=0.0, max=1.0)
        return pooled_quality, reliability

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}

        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            pred_scores = final_pred_dict[index]['pred_scores']
            obs_quality_vec, reliability_scores = self.attach_observation_quality(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                batch_index=index,
                batch_dict=batch_dict,
            )
            final_pred_dict[index]['obs_quality_vec'] = obs_quality_vec
            final_pred_dict[index]['reliability_scores'] = reliability_scores

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict,
                batch_index=index,
                data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
            )

        return final_pred_dict, recall_dict

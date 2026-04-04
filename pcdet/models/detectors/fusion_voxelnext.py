import torch
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class FusionVoxelNeXt(Detector3DTemplate):
    """
    基于 VoxelNeXt 全稀疏架构的 Detector
    融合了可学习跨模态校正与辅助门控 Loss 计算
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.enable_aux_loss = model_cfg.get('ENABLE_AUX_LOSS', True)
        self.aux_loss_weight = model_cfg.get('AUX_LOSS_WEIGHT', 0.1)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # 1. 逐个运行模块 (VFE -> Sparse Backbone (带融合) -> VoxelNeXtHead)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # 2. 训练阶段：计算总 Loss
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        
        # 3. 测试/推理阶段：后处理
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        # --- A. 获取 VoxelNeXt Head 的主要检测 Loss ---
        # VoxelNeXt Head 的 get_loss 函数已经把总 loss 和 tb_dict 放入了 batch_dict
        loss_head, tb_dict = self.dense_head.get_loss()
        
        # --- B. 计算创新点 2 的辅助 Loss (Instance-Aware Gate) ---
        loss_aux = self.get_aux_loss(batch_dict) if self.enable_aux_loss else loss_head.new_tensor(0.0)
        
        # --- C. 总 Loss 求和 ---
        # 给辅助 Loss 一个权重 (可以根据训练情况微调，通常 0.1~0.5)
        loss = loss_head + self.aux_loss_weight * loss_aux
        
        # 记录到 Tensorboard
        tb_dict['loss_head'] = loss_head.item()
        tb_dict['loss_aux'] = loss_aux.item()
        tb_dict['aux_loss_weight'] = float(self.aux_loss_weight if self.enable_aux_loss else 0.0)
        tb_dict['loss_total'] = loss.item()

        return loss, tb_dict, disp_dict

    def get_aux_loss(self, batch_dict):
        """
        计算创新点 2 (Instance Aware Gate) 的辅助监督 Loss
        目标：让稀疏特征网络在融合增强前学会分辨前景与背景
        """
        # 1. 获取辅助预测输出 (由 Backbone 中的 gate 模块产生)
        preds_list = batch_dict.get('aux_preds', None)
        if preds_list is None or len(preds_list) == 0:
            return torch.tensor(0.0).to(batch_dict['gt_boxes'].device)

        # 获取对应的稀疏索引 [N, 4] -> (batch_idx, z, y, x)
        indices = batch_dict['aux_indices']
        gt_boxes = batch_dict['gt_boxes'] # [B, M, 8]
        
        # 2. 计算稀疏特征在真实世界的物理中心坐标
        # 注意：这里的中心坐标必须考虑当前的下采样步长 (stride)
        voxel_centers = self.get_voxel_centers(indices, batch_dict, stride=8)
        
        # 3. 生成标签: 利用 CUDA Op 判断点是否在任意 GT 框内
        labels = self.assign_voxel_labels(voxel_centers, indices[:, 0], gt_boxes)
        
        # 4. 计算多层辅助损失的平均值 (BCE Loss)
        loss = 0
        for pred in preds_list:
            loss += F.binary_cross_entropy(pred.squeeze(-1), labels)
            
        return loss / len(preds_list)

    def get_voxel_centers(self, indices, batch_dict, stride=8):
        """
        工具函数：将降采样后的 Grid Index 转换为真实的 3D 物理坐标 XYZ
        【核心修复】：必须乘以 stride，否则坐标范围全错！
        """
        voxel_size = torch.as_tensor(batch_dict['voxel_size'], device=indices.device, dtype=torch.float32)
        pc_range = torch.as_tensor(batch_dict['point_cloud_range'], device=indices.device, dtype=torch.float32)
        
        # indices 是 (batch_idx, z, y, x)
        # 转换成 (x, y, z)
        spatial_indices = indices[:, 1:][:, [2, 1, 0]].float()
        
        # 计算该特征层级下的实际 voxel size
        current_voxel_size = voxel_size * stride
        
        # 物理坐标 = 索引 * 当前尺寸 + 起点偏移 + 半个尺寸偏移(求中心)
        physical_xyz = spatial_indices * current_voxel_size + pc_range[:3] + (current_voxel_size / 2.0)
        
        return physical_xyz

    def assign_voxel_labels(self, points, batch_idxs, gt_boxes):
        """
        工具函数：利用 CUDA Op 快速判断点是否在框内，生成前景/背景 Label
        """
        batch_size = gt_boxes.shape[0]
        labels_list = []
        
        for b in range(batch_size):
            mask = (batch_idxs == b)
            if mask.sum() == 0:
                continue
            
            cur_points = points[mask] # [N_b, 3]
            cur_boxes = gt_boxes[b]   # [M, 8]
            
            # 使用 roiaware_pool3d 判断点是否在框内
            # 返回: [Batch=1, N_points] 内部值为命中框的 idx, 未命中为 -1
            in_boxes_result = roiaware_pool3d_utils.points_in_boxes_gpu(
                cur_points.unsqueeze(0), 
                cur_boxes[:, :7].unsqueeze(0)
            ).squeeze(0) 
            
            if in_boxes_result.ndim == 1:
                cur_labels = (in_boxes_result >= 0).float()
            else:
                # 兼容性防御
                cur_labels = (in_boxes_result.max(dim=1)[0] > 0).float()

            labels_list.append(cur_labels)
            
        if len(labels_list) > 0:
            return torch.cat(labels_list, dim=0)
        else:
            return torch.zeros(points.shape[0], device=points.device)

    def attach_observation_quality(self, pred_boxes, pred_scores, batch_index, batch_dict):
        if pred_boxes.shape[0] == 0:
            empty_quality = pred_boxes.new_zeros((0, 5))
            empty_reliability = pred_scores.new_zeros((0,))
            return empty_quality, empty_reliability

        quality_features = batch_dict.get('fusion_quality_features', None)
        quality_indices = batch_dict.get('fusion_quality_indices', None)
        quality_stride = int(batch_dict.get('fusion_quality_stride', 8))
        if quality_features is None or quality_indices is None or quality_features.shape[0] == 0:
            empty_quality = pred_boxes.new_zeros((pred_boxes.shape[0], 5))
            return empty_quality, pred_scores.clone()

        voxel_centers = self.get_voxel_centers(quality_indices, batch_dict, stride=quality_stride)
        batch_mask = quality_indices[:, 0] == batch_index
        voxel_centers = voxel_centers[batch_mask]
        voxel_quality = quality_features[batch_mask]

        if voxel_centers.shape[0] == 0:
            empty_quality = pred_boxes.new_zeros((pred_boxes.shape[0], 5))
            return empty_quality, pred_scores.clone()

        topk = min(8, voxel_centers.shape[0])
        distances = torch.cdist(pred_boxes[:, :3], voxel_centers)
        nearest_dist, nearest_idx = distances.topk(k=topk, dim=1, largest=False)
        weights = 1.0 / (nearest_dist + 1e-3)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled_quality = (voxel_quality[nearest_idx] * weights.unsqueeze(-1)).sum(dim=1)
        pooled_quality = pooled_quality.clamp(min=0.0, max=1.0)

        reliability = (
            0.25 * pred_scores
            + 0.20 * pooled_quality[:, 0]
            + 0.15 * pooled_quality[:, 1]
            + 0.15 * pooled_quality[:, 2]
            + 0.10 * pooled_quality[:, 3]
            + 0.15 * pooled_quality[:, 4]
        ).clamp(min=0.0, max=1.0)
        return pooled_quality, reliability

    def post_processing(self, batch_dict):
        """
        利用 Detector3DTemplate 中自带的标准后处理逻辑计算 Recall
        """
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
                batch_dict=batch_dict
            )
            final_pred_dict[index]['obs_quality_vec'] = obs_quality_vec
            final_pred_dict[index]['reliability_scores'] = reliability_scores

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

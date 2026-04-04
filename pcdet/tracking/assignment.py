import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pcdet.utils import box_utils


def bev_iou_matrix(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 7)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 7)
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    iou = box_utils.boxes3d_nearest_bev_iou(
        torch.from_numpy(boxes_a), torch.from_numpy(boxes_b)
    )
    return iou.cpu().numpy().astype(np.float32)


def hungarian_assign(cost_matrix, valid_mask=None):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float32)
    if cost_matrix.size == 0:
        return []

    if valid_mask is None:
        valid_mask = np.ones_like(cost_matrix, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    masked_cost = np.where(valid_mask, cost_matrix, 1e6)
    row_ind, col_ind = linear_sum_assignment(masked_cost)
    matches = []
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if valid_mask[row, col]:
            matches.append((row, col))
    return matches


def match_detections_to_gt(det_boxes, det_labels, gt_boxes, gt_labels, iou_threshold):
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 7)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
    det_labels = np.asarray(det_labels, dtype=np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    det_to_gt = -np.ones(det_boxes.shape[0], dtype=np.int64)
    gt_to_det = -np.ones(gt_boxes.shape[0], dtype=np.int64)
    if det_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return det_to_gt, gt_to_det, np.zeros((det_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    iou = bev_iou_matrix(det_boxes, gt_boxes)
    valid = det_labels[:, None] == gt_labels[None, :]
    valid &= iou >= float(iou_threshold)
    matches = hungarian_assign(1.0 - iou, valid)

    for det_idx, gt_idx in matches:
        det_to_gt[det_idx] = gt_idx
        gt_to_det[gt_idx] = det_idx

    return det_to_gt, gt_to_det, iou

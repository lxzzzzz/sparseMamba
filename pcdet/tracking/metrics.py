import numpy as np

from .assignment import bev_iou_matrix, hungarian_assign


class TrackingMetrics:
    def __init__(self, iou_threshold=0.1):
        self.iou_threshold = float(iou_threshold)
        self.total_gt = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.id_switches = 0
        self._gt_to_pred = {}

    def update(self, sequence_id, gt_boxes, gt_ids, gt_labels, pred_boxes, pred_ids, pred_labels):
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
        gt_ids = np.asarray(gt_ids, dtype=np.int64)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 7)
        pred_ids = np.asarray(pred_ids, dtype=np.int64)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)

        self.total_gt += gt_boxes.shape[0]
        if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
            return
        if gt_boxes.shape[0] == 0:
            self.false_positive += pred_boxes.shape[0]
            return
        if pred_boxes.shape[0] == 0:
            self.false_negative += gt_boxes.shape[0]
            return

        iou = bev_iou_matrix(gt_boxes, pred_boxes)
        valid = (gt_labels[:, None] == pred_labels[None, :]) & (iou >= self.iou_threshold)
        matches = hungarian_assign(1.0 - iou, valid)

        matched_gt = set()
        matched_pred = set()
        for gt_idx, pred_idx in matches:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            self.true_positive += 1

            gt_key = (str(sequence_id), int(gt_ids[gt_idx]))
            pred_id = int(pred_ids[pred_idx])
            if gt_key in self._gt_to_pred and self._gt_to_pred[gt_key] != pred_id:
                self.id_switches += 1
            self._gt_to_pred[gt_key] = pred_id

        self.false_negative += gt_boxes.shape[0] - len(matched_gt)
        self.false_positive += pred_boxes.shape[0] - len(matched_pred)

    def summary(self):
        precision = self.true_positive / max(self.true_positive + self.false_positive, 1)
        recall = self.true_positive / max(self.total_gt, 1)
        mota = 1.0 - (self.false_positive + self.false_negative + self.id_switches) / max(self.total_gt, 1)
        return {
            'total_gt': int(self.total_gt),
            'tp': int(self.true_positive),
            'fp': int(self.false_positive),
            'fn': int(self.false_negative),
            'id_switches': int(self.id_switches),
            'precision': float(precision),
            'recall': float(recall),
            'mota': float(mota),
        }

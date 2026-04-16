from collections import defaultdict

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
        self.id_switch_events = []
        self.matched_iou_sum = 0.0
        self.matched_iou_count = 0
        self._gt_to_pred = {}
        self._seq_stats = defaultdict(lambda: {
            'gt_counts': defaultdict(int),
            'pred_counts': defaultdict(int),
            'pair_counts': defaultdict(int),
            'gt_matched_counts': defaultdict(int),
            'gt_last_matched': {},
            'frag': defaultdict(int),
        })

    def update(self, sequence_id, gt_boxes, gt_ids, gt_labels, pred_boxes, pred_ids, pred_labels, frame_idx=None):
        sequence_id = str(sequence_id)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
        gt_ids = np.asarray(gt_ids, dtype=np.int64)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 7)
        pred_ids = np.asarray(pred_ids, dtype=np.int64)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        seq_stats = self._seq_stats[sequence_id]

        for gt_id in gt_ids.tolist():
            seq_stats['gt_counts'][int(gt_id)] += 1
        for pred_id in pred_ids.tolist():
            seq_stats['pred_counts'][int(pred_id)] += 1

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
            self.matched_iou_sum += float(iou[gt_idx, pred_idx])
            self.matched_iou_count += 1

            gt_id = int(gt_ids[gt_idx])
            pred_id = int(pred_ids[pred_idx])
            gt_key = (sequence_id, gt_id)
            prev_pred_id = self._gt_to_pred.get(gt_key, None)
            if prev_pred_id is not None and prev_pred_id != pred_id:
                self.id_switches += 1
                self.id_switch_events.append({
                    'sequence_id': sequence_id,
                    'frame_idx': None if frame_idx is None else int(frame_idx),
                    'gt_id': gt_id,
                    'prev_pred_id': int(prev_pred_id),
                    'new_pred_id': pred_id,
                })
            self._gt_to_pred[gt_key] = pred_id
            seq_stats['pair_counts'][(gt_id, pred_id)] += 1
            seq_stats['gt_matched_counts'][gt_id] += 1

        matched_gt_ids = {int(gt_ids[idx]) for idx in matched_gt}
        for gt_id in gt_ids.tolist():
            gt_id = int(gt_id)
            is_matched = gt_id in matched_gt_ids
            prev_matched = seq_stats['gt_last_matched'].get(gt_id, None)
            if is_matched and prev_matched is False:
                seq_stats['frag'][gt_id] += 1
            seq_stats['gt_last_matched'][gt_id] = is_matched

        self.false_negative += gt_boxes.shape[0] - len(matched_gt)
        self.false_positive += pred_boxes.shape[0] - len(matched_pred)

    def summary(self):
        precision = self.true_positive / max(self.true_positive + self.false_positive, 1)
        recall = self.true_positive / max(self.total_gt, 1)
        mota = 1.0 - (self.false_positive + self.false_negative + self.id_switches) / max(self.total_gt, 1)
        motp = self.matched_iou_sum / max(self.matched_iou_count, 1)

        idtp = 0
        idfp = 0
        idfn = 0
        mostly_tracked = 0
        partially_tracked = 0
        mostly_lost = 0
        fragments = 0
        weighted_ass_iou = 0.0

        for seq_stats in self._seq_stats.values():
            gt_ids = sorted(seq_stats['gt_counts'].keys())
            pred_ids = sorted(seq_stats['pred_counts'].keys())
            total_gt_dets = sum(seq_stats['gt_counts'].values())
            total_pred_dets = sum(seq_stats['pred_counts'].values())

            if len(gt_ids) > 0 and len(pred_ids) > 0:
                overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
                valid = np.zeros((len(gt_ids), len(pred_ids)), dtype=bool)
                for gt_idx, gt_id in enumerate(gt_ids):
                    for pred_idx, pred_id in enumerate(pred_ids):
                        count = float(seq_stats['pair_counts'].get((gt_id, pred_id), 0))
                        overlap[gt_idx, pred_idx] = count
                        valid[gt_idx, pred_idx] = count > 0
                max_overlap = overlap.max() if overlap.size > 0 else 0.0
                matches = hungarian_assign(max_overlap - overlap, valid)
                seq_idtp = sum(int(overlap[gt_idx, pred_idx]) for gt_idx, pred_idx in matches)
            else:
                seq_idtp = 0

            for gt_id, pred_id in seq_stats['pair_counts'].keys():
                pair_tp = int(seq_stats['pair_counts'][(gt_id, pred_id)])
                if pair_tp <= 0:
                    continue
                pair_fn = int(seq_stats['gt_counts'][gt_id]) - pair_tp
                pair_fp = int(seq_stats['pred_counts'][pred_id]) - pair_tp
                assoc_iou = pair_tp / max(pair_tp + pair_fn + pair_fp, 1)
                weighted_ass_iou += pair_tp * assoc_iou

            idtp += seq_idtp
            idfn += total_gt_dets - seq_idtp
            idfp += total_pred_dets - seq_idtp

            for gt_id, total_count in seq_stats['gt_counts'].items():
                matched_count = seq_stats['gt_matched_counts'].get(gt_id, 0)
                tracked_ratio = matched_count / max(total_count, 1)
                if tracked_ratio >= 0.8:
                    mostly_tracked += 1
                elif tracked_ratio >= 0.2:
                    partially_tracked += 1
                else:
                    mostly_lost += 1
                fragments += seq_stats['frag'].get(gt_id, 0)

        id_precision = idtp / max(idtp + idfp, 1)
        id_recall = idtp / max(idtp + idfn, 1)
        idf1 = 2.0 * idtp / max(2 * idtp + idfp + idfn, 1)
        det_a = self.true_positive / max(self.true_positive + self.false_positive + self.false_negative, 1)
        ass_a = weighted_ass_iou / max(self.true_positive, 1)
        hota = float(np.sqrt(det_a * ass_a))
        det_pr = precision
        det_re = recall
        ass_pr = weighted_ass_iou / max(self.true_positive + self.false_positive, 1)
        ass_re = weighted_ass_iou / max(self.true_positive + self.false_negative, 1)
        return {
            'total_gt': int(self.total_gt),
            'tp': int(self.true_positive),
            'fp': int(self.false_positive),
            'fn': int(self.false_negative),
            'id_switches': int(self.id_switches),
            'id_switch_events': list(self.id_switch_events),
            'precision': float(precision),
            'recall': float(recall),
            'mota': float(mota),
            'motp': float(motp),
            'idtp': int(idtp),
            'idfp': int(idfp),
            'idfn': int(idfn),
            'id_precision': float(id_precision),
            'id_recall': float(id_recall),
            'idf1': float(idf1),
            'mostly_tracked': int(mostly_tracked),
            'partially_tracked': int(partially_tracked),
            'mostly_lost': int(mostly_lost),
            'fragments': int(fragments),
            'deta': float(det_a),
            'assa': float(ass_a),
            'detpr': float(det_pr),
            'detre': float(det_re),
            'asspr': float(ass_pr),
            'assre': float(ass_re),
            'hota': hota,
        }

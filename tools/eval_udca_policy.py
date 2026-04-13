import _init_path
import argparse
import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcdet.tracking.utils import (
    association_quality,
    compose_quality_scalar,
    get_cache_obs_quality,
    get_cache_reliability,
)


def clamp01(values):
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def bev_iou_matrix(boxes_a, boxes_b):
    from pcdet.tracking.assignment import bev_iou_matrix as _bev_iou_matrix

    return _bev_iou_matrix(boxes_a, boxes_b)


def hungarian_assign(cost_matrix, valid_mask=None):
    from pcdet.tracking.assignment import hungarian_assign as _hungarian_assign

    return _hungarian_assign(cost_matrix, valid_mask)


def load_frame_cache(cache_root, sequence_id, frame_idx):
    from pcdet.tracking.cache import load_frame_cache as _load_frame_cache

    return _load_frame_cache(cache_root, sequence_id, frame_idx)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def class_names_to_labels(class_names, names):
    name_to_label = {name: idx + 1 for idx, name in enumerate(class_names)}
    return np.asarray([name_to_label.get(name, -1) for name in names], dtype=np.int64)


def filter_boxes_by_distance(boxes, *extra_arrays, max_distance=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    if max_distance is None or boxes.shape[0] == 0:
        return (boxes, *extra_arrays)

    ranges = np.linalg.norm(boxes[:, 0:2], axis=1)
    keep = ranges <= float(max_distance)
    filtered = [boxes[keep]]
    for arr in extra_arrays:
        arr = np.asarray(arr)
        filtered.append(arr[keep])
    return tuple(filtered)


def enrich_metric_dict(metric_dict, total_frames, num_sequences, elapsed, stage_stats):
    metric_dict = dict(metric_dict)
    elapsed = max(float(elapsed), 1e-6)
    metric_dict['fps'] = float(total_frames / elapsed)
    metric_dict['num_frames'] = int(total_frames)
    metric_dict['num_sequences'] = int(num_sequences)
    metric_dict['MOTA'] = metric_dict.get('mota', 0.0)
    metric_dict['MOTP'] = metric_dict.get('motp', 0.0)
    metric_dict['Rcll'] = metric_dict.get('recall', 0.0)
    metric_dict['Prcn'] = metric_dict.get('precision', 0.0)
    metric_dict['MT'] = metric_dict.get('mostly_tracked', 0)
    metric_dict['PT'] = metric_dict.get('partially_tracked', 0)
    metric_dict['ML'] = metric_dict.get('mostly_lost', 0)
    metric_dict['FP'] = metric_dict.get('fp', 0)
    metric_dict['FN'] = metric_dict.get('fn', 0)
    metric_dict['IDsw'] = metric_dict.get('id_switches', 0)
    metric_dict['Frag'] = metric_dict.get('fragments', 0)
    metric_dict['TP'] = metric_dict.get('tp', 0)
    metric_dict['IDF1'] = metric_dict.get('idf1', 0.0)
    metric_dict['IDP'] = metric_dict.get('id_precision', 0.0)
    metric_dict['IDR'] = metric_dict.get('id_recall', 0.0)
    metric_dict['HOTA'] = metric_dict.get('hota', 0.0)
    metric_dict['DetA'] = metric_dict.get('deta', 0.0)
    metric_dict['AssA'] = metric_dict.get('assa', 0.0)
    metric_dict['DetPr'] = metric_dict.get('detpr', 0.0)
    metric_dict['DetRe'] = metric_dict.get('detre', 0.0)
    metric_dict['AssPr'] = metric_dict.get('asspr', 0.0)
    metric_dict['AssRe'] = metric_dict.get('assre', 0.0)
    metric_dict['FPS'] = metric_dict.get('fps', 0.0)
    metric_dict.update({key: int(value) for key, value in stage_stats.items()})
    return metric_dict


@dataclass
class TrackState:
    track_id: int
    label: int
    score: float
    hits: int
    missed: int
    last_box: np.ndarray
    velocity_xy: np.ndarray
    last_visual_conf: float
    last_u3d: float
    status: str


class UDCAPolicyTracker:
    def __init__(
        self,
        max_age=2,
        min_hits=2,
        score_thresh=0.1,
        match_iou=0.1,
        center_gate=8.0,
        rescue_center_gate=12.0,
        fallback_center_gate=8.0,
        max_distance=None,
        u3d_high=0.55,
        u2d_high=0.55,
        rescue_min_visual_conf=0.35,
        disable_stage2=False,
        disable_stage3=False,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.score_thresh = float(score_thresh)
        self.match_iou = float(match_iou)
        self.center_gate = float(center_gate)
        self.rescue_center_gate = float(rescue_center_gate)
        self.fallback_center_gate = float(fallback_center_gate)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.u3d_high = float(u3d_high)
        self.u2d_high = float(u2d_high)
        self.rescue_min_visual_conf = float(rescue_min_visual_conf)
        self.disable_stage2 = bool(disable_stage2)
        self.disable_stage3 = bool(disable_stage3)
        self.next_track_id = 1
        self.tracks = []
        self.stage_stats = defaultdict(int)

    def reset(self):
        self.next_track_id = 1
        self.tracks = []
        self.stage_stats = defaultdict(int)

    def _predict_boxes(self):
        if len(self.tracks) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        predicted = []
        for track in self.tracks:
            box = track.last_box.copy()
            box[0:2] += track.velocity_xy
            predicted.append(box)
        return np.stack(predicted, axis=0).astype(np.float32)

    def _filter_det_by_distance(self, det_boxes, det_scores, det_labels, det_reliability, det_obs_quality):
        if self.max_distance is None or det_boxes.shape[0] == 0:
            return det_boxes, det_scores, det_labels, det_reliability, det_obs_quality
        ranges = np.linalg.norm(det_boxes[:, 0:2], axis=1)
        keep = ranges <= self.max_distance
        return (
            det_boxes[keep],
            det_scores[keep],
            det_labels[keep],
            det_reliability[keep],
            det_obs_quality[keep],
        )

    def _extract_detections(self, frame_cache):
        det_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        det_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
        det_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)
        det_reliability = get_cache_reliability(frame_cache).reshape(-1)
        det_obs_quality = get_cache_obs_quality(frame_cache, len(det_scores))

        keep = det_scores >= self.score_thresh
        det_boxes = det_boxes[keep]
        det_scores = det_scores[keep]
        det_labels = det_labels[keep]
        det_reliability = det_reliability[keep]
        det_obs_quality = det_obs_quality[keep]
        det_boxes, det_scores, det_labels, det_reliability, det_obs_quality = self._filter_det_by_distance(
            det_boxes, det_scores, det_labels, det_reliability, det_obs_quality
        )

        quality_scalar = compose_quality_scalar(det_scores, det_reliability, det_obs_quality)
        assoc_quality = association_quality(det_scores, det_reliability, det_obs_quality, quality_scalar=quality_scalar)
        visual_conf = clamp01(det_obs_quality.mean(axis=1) if det_obs_quality.shape[0] > 0 else np.zeros((0,), dtype=np.float32))
        visual_available = np.any(np.abs(det_obs_quality) > 1e-6, axis=1) if det_obs_quality.shape[0] > 0 else np.zeros((0,), dtype=bool)
        u2d = np.where(visual_available, 1.0 - visual_conf, 1.0).astype(np.float32)
        geom_conf = clamp01(0.5 * det_scores + 0.5 * det_reliability)
        u3d = (1.0 - geom_conf).astype(np.float32)

        return {
            'boxes': det_boxes,
            'scores': det_scores,
            'labels': det_labels,
            'reliability': det_reliability,
            'obs_quality': det_obs_quality,
            'quality_scalar': quality_scalar,
            'assoc_quality': assoc_quality,
            'visual_conf': visual_conf,
            'visual_available': visual_available,
            'u2d': u2d,
            'u3d': u3d,
        }

    def _candidate_mask(self, pred_boxes, track_labels, det_boxes, det_labels, iou_gate, center_gate):
        if pred_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
            return np.zeros((pred_boxes.shape[0], det_boxes.shape[0]), dtype=bool)
        iou = bev_iou_matrix(pred_boxes, det_boxes)
        same_class = track_labels[:, None] == det_labels[None, :]
        center_dist = np.linalg.norm(pred_boxes[:, None, 0:2] - det_boxes[None, :, 0:2], axis=-1)
        valid = same_class & ((iou >= iou_gate) | (center_dist <= center_gate))
        return valid

    @staticmethod
    def _select_subset(cost, valid, row_indices=None, col_indices=None):
        if row_indices is not None:
            row_indices = np.asarray(row_indices, dtype=np.int64)
            cost = cost[row_indices]
            valid = valid[row_indices]
        if col_indices is not None:
            col_indices = np.asarray(col_indices, dtype=np.int64)
            cost = cost[:, col_indices]
            valid = valid[:, col_indices]
        return cost, valid, row_indices, col_indices

    def _match_subset(self, cost, valid, row_indices=None, col_indices=None):
        sub_cost, sub_valid, row_indices, col_indices = self._select_subset(cost, valid, row_indices, col_indices)
        matches = []
        for row_idx, col_idx in hungarian_assign(sub_cost, sub_valid):
            global_row = int(row_indices[row_idx]) if row_indices is not None else int(row_idx)
            global_col = int(col_indices[col_idx]) if col_indices is not None else int(col_idx)
            matches.append((global_row, global_col))
        return matches

    def _joint_cost(self, pred_boxes, dets):
        det_boxes = dets['boxes']
        if pred_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
            return np.zeros((pred_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)

        iou = bev_iou_matrix(pred_boxes, det_boxes)
        center_dist = np.linalg.norm(pred_boxes[:, None, 0:2] - det_boxes[None, :, 0:2], axis=-1)
        center_score = 1.0 - np.clip(center_dist / max(self.center_gate, 1e-3), 0.0, 1.0)

        track_visual = np.asarray([track.last_visual_conf for track in self.tracks], dtype=np.float32)
        track_u3d = np.asarray([track.last_u3d for track in self.tracks], dtype=np.float32)

        visual_consistency = 1.0 - np.abs(track_visual[:, None] - dets['visual_conf'][None, :])
        visual_consistency = clamp01(visual_consistency)

        track_conf = clamp01(1.0 - track_u3d)
        det_geom_conf = clamp01(1.0 - dets['u3d'])
        w3d = track_conf[:, None] + det_geom_conf[None, :]
        w2d = dets['visual_conf'][None, :] + (1.0 - dets['u2d'][None, :])
        norm = np.clip(w3d + w2d, 1e-6, None)
        w3d = w3d / norm
        w2d = w2d / norm

        cost_3d = 0.7 * (1.0 - iou) + 0.3 * (1.0 - center_score)
        cost_2d = 1.0 - visual_consistency
        return (w3d * cost_3d + w2d * cost_2d).astype(np.float32)

    def _rescue_cost(self, pred_boxes, dets):
        det_boxes = dets['boxes']
        if pred_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
            return np.zeros((pred_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)

        iou = bev_iou_matrix(pred_boxes, det_boxes)
        center_dist = np.linalg.norm(pred_boxes[:, None, 0:2] - det_boxes[None, :, 0:2], axis=-1)
        center_score = 1.0 - np.clip(center_dist / max(self.rescue_center_gate, 1e-3), 0.0, 1.0)
        track_visual = np.asarray([track.last_visual_conf for track in self.tracks], dtype=np.float32)
        visual_consistency = 1.0 - np.abs(track_visual[:, None] - dets['visual_conf'][None, :])
        visual_consistency = clamp01(visual_consistency)

        cost_3d = 0.5 * (1.0 - iou) + 0.5 * (1.0 - center_score)
        cost_2d = 1.0 - visual_consistency
        return (0.35 * cost_3d + 0.65 * cost_2d).astype(np.float32)

    def _fallback_cost(self, pred_boxes, dets):
        det_boxes = dets['boxes']
        if pred_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
            return np.zeros((pred_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)
        iou = bev_iou_matrix(pred_boxes, det_boxes)
        center_dist = np.linalg.norm(pred_boxes[:, None, 0:2] - det_boxes[None, :, 0:2], axis=-1)
        center_score = 1.0 - np.clip(center_dist / max(self.fallback_center_gate, 1e-3), 0.0, 1.0)
        return (0.8 * (1.0 - iou) + 0.2 * (1.0 - center_score)).astype(np.float32)

    def _update_track(self, track_idx, det_idx, dets, status):
        track = self.tracks[track_idx]
        prev_box = track.last_box.copy()
        new_box = dets['boxes'][det_idx].copy()
        track.velocity_xy = new_box[0:2] - prev_box[0:2]
        track.last_box = new_box
        track.score = float(dets['scores'][det_idx])
        track.hits += 1
        track.missed = 0
        track.last_visual_conf = float(dets['visual_conf'][det_idx])
        track.last_u3d = float(dets['u3d'][det_idx])
        track.status = status

    def _append_new_track(self, det_idx, dets):
        visual_conf = float(dets['visual_conf'][det_idx])
        self.tracks.append(
            TrackState(
                track_id=self.next_track_id,
                label=int(dets['labels'][det_idx]),
                score=float(dets['scores'][det_idx]),
                hits=1,
                missed=0,
                last_box=dets['boxes'][det_idx].copy(),
                velocity_xy=np.zeros((2,), dtype=np.float32),
                last_visual_conf=visual_conf,
                last_u3d=float(dets['u3d'][det_idx]),
                status='stable' if dets['u3d'][det_idx] <= self.u3d_high else 'fragile',
            )
        )
        self.next_track_id += 1
        self.stage_stats['births'] += 1

    def update(self, frame_cache):
        dets = self._extract_detections(frame_cache)
        det_boxes = dets['boxes']
        det_labels = dets['labels']
        det_scores = dets['scores']

        for track in self.tracks:
            track.missed += 1
            if track.missed > 0 and track.status == 'stable':
                track.status = 'fragile'

        matched_track_ids = set()
        matched_det_ids = set()

        pred_boxes = self._predict_boxes()
        track_labels = np.asarray([track.label for track in self.tracks], dtype=np.int64)

        # Stage 1: high-confidence joint matching
        stage1_det_ids = np.flatnonzero((dets['u3d'] <= self.u3d_high) & (dets['u2d'] <= self.u2d_high))
        if pred_boxes.shape[0] > 0 and stage1_det_ids.size > 0:
            valid1 = self._candidate_mask(pred_boxes, track_labels, det_boxes, det_labels, self.match_iou, self.center_gate)
            cost1 = self._joint_cost(pred_boxes, dets)
            for track_idx, det_idx in self._match_subset(cost1, valid1, col_indices=stage1_det_ids):
                if track_idx in matched_track_ids or det_idx in matched_det_ids:
                    continue
                self._update_track(track_idx, det_idx, dets, status='stable')
                matched_track_ids.add(track_idx)
                matched_det_ids.add(det_idx)
                self.stage_stats['stage1_matches'] += 1

        # Stage 2: visual rescue under 3D degradation
        if not self.disable_stage2:
            track_candidates = np.asarray(
                [idx for idx, track in enumerate(self.tracks) if idx not in matched_track_ids and track.missed > 0],
                dtype=np.int64,
            )
            det_candidates = np.flatnonzero(
                (dets['u3d'] > self.u3d_high)
                & (dets['u2d'] <= self.u2d_high)
                & (dets['visual_conf'] >= self.rescue_min_visual_conf)
                & (~np.isin(np.arange(det_boxes.shape[0]), list(matched_det_ids)))
            )
            if track_candidates.size > 0 and det_candidates.size > 0:
                valid2 = self._candidate_mask(
                    pred_boxes, track_labels, det_boxes, det_labels, 0.0, self.rescue_center_gate
                )
                cost2 = self._rescue_cost(pred_boxes, dets)
                for track_idx, det_idx in self._match_subset(cost2, valid2, row_indices=track_candidates, col_indices=det_candidates):
                    if track_idx in matched_track_ids or det_idx in matched_det_ids:
                        continue
                    self._update_track(track_idx, det_idx, dets, status='recovering')
                    matched_track_ids.add(track_idx)
                    matched_det_ids.add(det_idx)
                    self.stage_stats['stage2_matches'] += 1

        # Stage 3: pure 3D fallback
        if not self.disable_stage3:
            track_candidates = np.asarray(
                [idx for idx in range(len(self.tracks)) if idx not in matched_track_ids],
                dtype=np.int64,
            )
            det_candidates = np.asarray(
                [idx for idx in range(det_boxes.shape[0]) if idx not in matched_det_ids],
                dtype=np.int64,
            )
            if track_candidates.size > 0 and det_candidates.size > 0:
                valid3 = self._candidate_mask(
                    pred_boxes, track_labels, det_boxes, det_labels, self.match_iou, self.fallback_center_gate
                )
                cost3 = self._fallback_cost(pred_boxes, dets)
                for track_idx, det_idx in self._match_subset(cost3, valid3, row_indices=track_candidates, col_indices=det_candidates):
                    if track_idx in matched_track_ids or det_idx in matched_det_ids:
                        continue
                    self._update_track(track_idx, det_idx, dets, status='fragile')
                    matched_track_ids.add(track_idx)
                    matched_det_ids.add(det_idx)
                    self.stage_stats['stage3_matches'] += 1

        survivors = []
        for idx, track in enumerate(self.tracks):
            if idx not in matched_track_ids and track.missed > self.max_age:
                self.stage_stats['deleted_tracks'] += 1
                continue
            survivors.append(track)
        self.tracks = survivors

        for det_idx in range(det_boxes.shape[0]):
            if det_idx in matched_det_ids:
                continue
            self._append_new_track(det_idx, dets)

        outputs = []
        for track in self.tracks:
            if track.hits < self.min_hits and track.missed > 0:
                continue
            outputs.append({
                'track_id': int(track.track_id),
                'pred_box': track.last_box.copy(),
                'pred_label': int(track.label),
                'pred_score': float(track.score),
                'missed': int(track.missed),
                'status': track.status,
            })
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description='Uncertainty-Driven Cross-Modal Cascaded Association policy evaluation')
    parser.add_argument('--cache_dir', type=str, required=True, help='detector cache root')
    parser.add_argument('--gt_pkl', type=str, required=True, help='ground-truth pkl, e.g. tracking_infos_val.pkl')
    parser.add_argument('--class_names', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--match_iou', type=float, default=0.1)
    parser.add_argument('--center_gate', type=float, default=8.0)
    parser.add_argument('--rescue_center_gate', type=float, default=12.0)
    parser.add_argument('--fallback_center_gate', type=float, default=8.0)
    parser.add_argument('--max_age', type=int, default=2)
    parser.add_argument('--min_hits', type=int, default=2)
    parser.add_argument('--max_distance', type=float, default=100.0, help='evaluate and track targets within XY range only')
    parser.add_argument('--u3d_high', type=float, default=0.55, help='high 3D uncertainty threshold')
    parser.add_argument('--u2d_high', type=float, default=0.55, help='high 2D uncertainty threshold')
    parser.add_argument('--rescue_min_visual_conf', type=float, default=0.35, help='minimum visual confidence for stage-2 rescue')
    parser.add_argument('--disable_stage2', action='store_true', help='disable stage-2 visual rescue')
    parser.add_argument('--disable_stage3', action='store_true', help='disable stage-3 pure 3D fallback')
    parser.add_argument('--save_dir', type=str, required=True, help='output dir for metrics and tracking results')
    return parser.parse_args()


def main():
    args = parse_args()
    from pcdet.tracking.metrics import TrackingMetrics

    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_infos = load_pickle(args.gt_pkl)
    frame_infos.sort(key=lambda info: (str(info['sequence_id']), int(info['frame_idx'])))
    sequence_to_infos = defaultdict(list)
    for info in frame_infos:
        sequence_to_infos[str(info['sequence_id'])].append(info)

    tracker = UDCAPolicyTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        score_thresh=args.score_thresh,
        match_iou=args.match_iou,
        center_gate=args.center_gate,
        rescue_center_gate=args.rescue_center_gate,
        fallback_center_gate=args.fallback_center_gate,
        max_distance=args.max_distance,
        u3d_high=args.u3d_high,
        u2d_high=args.u2d_high,
        rescue_min_visual_conf=args.rescue_min_visual_conf,
        disable_stage2=args.disable_stage2,
        disable_stage3=args.disable_stage3,
    )
    metrics = TrackingMetrics(iou_threshold=args.match_iou)
    results = {}
    total_frames = 0
    stage_stats = defaultdict(int)
    start_time = time.perf_counter()

    for sequence_id, infos in sequence_to_infos.items():
        tracker.reset()
        seq_results = []

        for info in infos:
            total_frames += 1
            frame_cache = load_frame_cache(cache_dir, sequence_id, info['frame_idx'])
            outputs = tracker.update(frame_cache)
            seq_results.append({
                'frame_idx': int(info['frame_idx']),
                'tracks': outputs,
            })

            gt_annos = info.get('annos', {})
            gt_boxes = np.asarray(gt_annos.get('gt_boxes_lidar', []), dtype=np.float32).reshape(-1, 7)
            gt_ids = np.asarray(gt_annos.get('track_id', []), dtype=np.int64)
            gt_labels = class_names_to_labels(args.class_names, gt_annos.get('name', []))
            gt_boxes, gt_ids, gt_labels = filter_boxes_by_distance(gt_boxes, gt_ids, gt_labels, max_distance=args.max_distance)

            pred_boxes = np.asarray([item['pred_box'] for item in outputs], dtype=np.float32).reshape(-1, 7)
            pred_ids = np.asarray([item['track_id'] for item in outputs], dtype=np.int64)
            pred_labels = np.asarray([item['pred_label'] for item in outputs], dtype=np.int64)
            pred_boxes, pred_ids, pred_labels = filter_boxes_by_distance(
                pred_boxes, pred_ids, pred_labels, max_distance=args.max_distance
            )
            metrics.update(sequence_id, gt_boxes, gt_ids, gt_labels, pred_boxes, pred_ids, pred_labels)

        results[sequence_id] = seq_results
        for key, value in tracker.stage_stats.items():
            stage_stats[key] += int(value)

    metric_dict = enrich_metric_dict(
        metrics.summary(),
        total_frames=total_frames,
        num_sequences=len(sequence_to_infos),
        elapsed=time.perf_counter() - start_time,
        stage_stats=stage_stats,
    )
    with open(save_dir / 'udca_policy_metrics.json', 'w') as f:
        json.dump(metric_dict, f, indent=2)
    with open(save_dir / 'udca_policy_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('================ UDCA Policy Evaluation ================')
    print(f'cache_dir: {cache_dir}')
    print(f'gt_pkl: {args.gt_pkl}')
    print(f'score_thresh: {args.score_thresh}')
    print(f'match_iou: {args.match_iou}')
    print(f'center_gate: {args.center_gate}')
    print(f'rescue_center_gate: {args.rescue_center_gate}')
    print(f'fallback_center_gate: {args.fallback_center_gate}')
    print(f'max_age: {args.max_age}')
    print(f'min_hits: {args.min_hits}')
    print(f'u3d_high: {args.u3d_high}')
    print(f'u2d_high: {args.u2d_high}')
    print(f'rescue_min_visual_conf: {args.rescue_min_visual_conf}')
    print(f'disable_stage2: {args.disable_stage2}')
    print(f'disable_stage3: {args.disable_stage3}')
    print(
        'Summary | '
        f"MOTA={metric_dict.get('mota', 0.0):.4f} "
        f"MOTP={metric_dict.get('motp', 0.0):.4f} "
        f"Rcll={metric_dict.get('recall', 0.0):.4f} "
        f"Prcn={metric_dict.get('precision', 0.0):.4f} "
        f"MT={metric_dict.get('mostly_tracked', 0)} "
        f"ML={metric_dict.get('mostly_lost', 0)} "
        f"FP={metric_dict.get('fp', 0)} "
        f"FN={metric_dict.get('fn', 0)} "
        f"IDsw={metric_dict.get('id_switches', 0)} "
        f"Frag={metric_dict.get('fragments', 0)} "
        f"FPS={metric_dict.get('fps', 0.0):.2f}"
    )
    print(
        'Extended | '
        f"HOTA={metric_dict.get('hota', 0.0):.4f} "
        f"DetA={metric_dict.get('deta', 0.0):.4f} "
        f"AssA={metric_dict.get('assa', 0.0):.4f} "
        f"IDF1={metric_dict.get('idf1', 0.0):.4f} "
        f"IDP={metric_dict.get('id_precision', 0.0):.4f} "
        f"IDR={metric_dict.get('id_recall', 0.0):.4f} "
        f"Stage1={metric_dict.get('stage1_matches', 0)} "
        f"Stage2={metric_dict.get('stage2_matches', 0)} "
        f"Stage3={metric_dict.get('stage3_matches', 0)} "
        f"Births={metric_dict.get('births', 0)} "
        f"Deleted={metric_dict.get('deleted_tracks', 0)} "
        f"Seq={metric_dict.get('num_sequences', 0)} "
        f"Frames={metric_dict.get('num_frames', 0)}"
    )


if __name__ == '__main__':
    main()

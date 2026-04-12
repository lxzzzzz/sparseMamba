import _init_path
import argparse
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcdet.tracking.assignment import bev_iou_matrix, hungarian_assign
from pcdet.tracking.cache import load_frame_cache
from pcdet.tracking.metrics import TrackingMetrics


@dataclass
class TrackState:
    track_id: int
    label: int
    score: float
    hits: int
    missed: int
    last_box: np.ndarray
    velocity_xy: np.ndarray


class AB3DMOTBaseline:
    def __init__(self, max_age=2, min_hits=2, match_iou=0.1, score_thresh=0.1, center_gate=8.0):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.match_iou = float(match_iou)
        self.score_thresh = float(score_thresh)
        self.center_gate = float(center_gate)
        self.next_track_id = 1
        self.tracks = []

    def _predict_boxes(self):
        if len(self.tracks) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        predicted = []
        for track in self.tracks:
            box = track.last_box.copy()
            box[0:2] += track.velocity_xy
            predicted.append(box)
        return np.stack(predicted, axis=0).astype(np.float32)

    def _candidate_mask(self, pred_boxes, pred_labels, det_boxes, det_labels):
        if pred_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
            return np.zeros((pred_boxes.shape[0], det_boxes.shape[0]), dtype=bool)
        iou = bev_iou_matrix(pred_boxes, det_boxes)
        same_class = pred_labels[:, None] == det_labels[None, :]
        center_dist = np.linalg.norm(pred_boxes[:, None, 0:2] - det_boxes[None, :, 0:2], axis=-1)
        valid = same_class & ((iou >= self.match_iou) | (center_dist <= self.center_gate))
        return valid

    def update(self, frame_cache):
        det_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        det_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
        det_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

        keep = det_scores >= self.score_thresh
        det_boxes = det_boxes[keep]
        det_scores = det_scores[keep]
        det_labels = det_labels[keep]

        for track in self.tracks:
            track.missed += 1

        matched_track_ids = set()
        matched_det_ids = set()

        pred_boxes = self._predict_boxes()
        track_labels = np.asarray([track.label for track in self.tracks], dtype=np.int64)
        valid = self._candidate_mask(pred_boxes, track_labels, det_boxes, det_labels)
        iou = bev_iou_matrix(pred_boxes, det_boxes) if pred_boxes.shape[0] > 0 and det_boxes.shape[0] > 0 else np.zeros(
            (pred_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32
        )
        matches = hungarian_assign(1.0 - iou, valid)

        for track_idx, det_idx in matches:
            if iou[track_idx, det_idx] < self.match_iou:
                pred_center = pred_boxes[track_idx, 0:2]
                det_center = det_boxes[det_idx, 0:2]
                if np.linalg.norm(pred_center - det_center) > self.center_gate:
                    continue

            track = self.tracks[track_idx]
            prev_box = track.last_box.copy()
            new_box = det_boxes[det_idx].copy()
            track.velocity_xy = new_box[0:2] - prev_box[0:2]
            track.last_box = new_box
            track.score = float(det_scores[det_idx])
            track.hits += 1
            track.missed = 0
            matched_track_ids.add(track_idx)
            matched_det_ids.add(det_idx)

        survivors = []
        for track in self.tracks:
            if track.missed <= self.max_age:
                survivors.append(track)
        self.tracks = survivors

        for det_idx in range(det_boxes.shape[0]):
            if det_idx in matched_det_ids:
                continue
            self.tracks.append(
                TrackState(
                    track_id=self.next_track_id,
                    label=int(det_labels[det_idx]),
                    score=float(det_scores[det_idx]),
                    hits=1,
                    missed=0,
                    last_box=det_boxes[det_idx].copy(),
                    velocity_xy=np.zeros((2,), dtype=np.float32),
                )
            )
            self.next_track_id += 1

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
            })
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT-style baseline on detector cache')
    parser.add_argument('--cache_dir', type=str, required=True, help='detector cache root')
    parser.add_argument('--gt_pkl', type=str, required=True, help='ground-truth pkl, e.g. tracking_infos_val.pkl')
    parser.add_argument('--class_names', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--match_iou', type=float, default=0.1)
    parser.add_argument('--center_gate', type=float, default=8.0)
    parser.add_argument('--max_age', type=int, default=2)
    parser.add_argument('--min_hits', type=int, default=2)
    parser.add_argument('--save_dir', type=str, required=True, help='output dir for metrics and tracking results')
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def class_names_to_labels(class_names, names):
    name_to_label = {name: idx + 1 for idx, name in enumerate(class_names)}
    return np.asarray([name_to_label.get(name, -1) for name in names], dtype=np.int64)


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_infos = load_pickle(args.gt_pkl)
    frame_infos.sort(key=lambda info: (str(info['sequence_id']), int(info['frame_idx'])))
    sequence_to_infos = defaultdict(list)
    for info in frame_infos:
        sequence_to_infos[str(info['sequence_id'])].append(info)

    tracker = AB3DMOTBaseline(
        max_age=args.max_age,
        min_hits=args.min_hits,
        match_iou=args.match_iou,
        score_thresh=args.score_thresh,
        center_gate=args.center_gate,
    )
    metrics = TrackingMetrics(iou_threshold=args.match_iou)
    results = {}

    for sequence_id, infos in sequence_to_infos.items():
        tracker.next_track_id = 1
        tracker.tracks = []
        seq_results = []

        for info in infos:
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

            pred_boxes = np.asarray([item['pred_box'] for item in outputs], dtype=np.float32).reshape(-1, 7)
            pred_ids = np.asarray([item['track_id'] for item in outputs], dtype=np.int64)
            pred_labels = np.asarray([item['pred_label'] for item in outputs], dtype=np.int64)
            metrics.update(sequence_id, gt_boxes, gt_ids, gt_labels, pred_boxes, pred_ids, pred_labels)

        results[sequence_id] = seq_results

    metric_dict = metrics.summary()
    with open(save_dir / 'ab3dmot_baseline_metrics.json', 'w') as f:
        json.dump(metric_dict, f, indent=2)
    with open(save_dir / 'ab3dmot_baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('================ AB3DMOT-Style Baseline ================')
    print(f'cache_dir: {cache_dir}')
    print(f'gt_pkl: {args.gt_pkl}')
    print(f'score_thresh: {args.score_thresh}')
    print(f'match_iou: {args.match_iou}')
    print(f'center_gate: {args.center_gate}')
    print(f'max_age: {args.max_age}')
    print(f'min_hits: {args.min_hits}')
    print(
        'Summary | '
        f"MOTA={metric_dict.get('mota', 0.0):.4f} "
        f"IDF1={metric_dict.get('idf1', 0.0):.4f} "
        f"HOTA={metric_dict.get('hota', 0.0):.4f} "
        f"DetA={metric_dict.get('deta', 0.0):.4f} "
        f"AssA={metric_dict.get('assa', 0.0):.4f} "
        f"Pr={metric_dict.get('precision', 0.0):.4f} "
        f"Re={metric_dict.get('recall', 0.0):.4f} "
        f"IDSW={metric_dict.get('id_switches', 0)} "
        f"MT={metric_dict.get('mostly_tracked', 0)} "
        f"PT={metric_dict.get('partially_tracked', 0)} "
        f"ML={metric_dict.get('mostly_lost', 0)} "
        f"Frag={metric_dict.get('fragments', 0)}"
    )


if __name__ == '__main__':
    main()

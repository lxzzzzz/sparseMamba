import _init_path
import argparse
import ast
import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.tracking.assignment import bev_iou_matrix, hungarian_assign
from pcdet.tracking.cache import load_frame_cache
from pcdet.tracking.metrics import TrackingMetrics
from pcdet.tracking.utils import filter_boxes_by_spatial_range, normalize_bev_range


@dataclass
class TrackState:
    track_id: int
    label: int
    score: float
    hits: int
    missed: int
    last_box: np.ndarray
    velocity_xy: np.ndarray
    prev_velocity_xy: np.ndarray


class AB3DMOTBaseline:
    def __init__(
        self,
        max_age=2,
        min_hits=2,
        match_iou=0.1,
        score_thresh=0.1,
        center_gate=8.0,
        max_distance=None,
        bev_range=None,
        motion_model='constant_velocity',
        motion_horizon=1.0,
        velocity_momentum=0.0,
        accel_gain=0.0,
        max_speed=100.0,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.match_iou = float(match_iou)
        self.score_thresh = float(score_thresh)
        self.center_gate = float(center_gate)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.bev_range = normalize_bev_range(bev_range)
        self.motion_model = str(motion_model)
        self.motion_horizon = float(motion_horizon)
        self.velocity_momentum = float(velocity_momentum)
        self.accel_gain = float(accel_gain)
        self.max_speed = float(max_speed)
        self.next_track_id = 1
        self.tracks = []

    def _clip_velocity(self, velocity_xy):
        velocity_xy = np.asarray(velocity_xy, dtype=np.float32)
        speed = np.linalg.norm(velocity_xy)
        if speed <= self.max_speed or speed <= 1e-6:
            return velocity_xy
        return velocity_xy * (self.max_speed / speed)

    def _predict_boxes(self):
        if len(self.tracks) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        predicted = []
        for track in self.tracks:
            box = track.last_box.copy()
            steps_ahead = max(float(track.missed), 1.0) * self.motion_horizon
            velocity_xy = self._clip_velocity(track.velocity_xy)
            motion_delta = velocity_xy * steps_ahead
            if self.motion_model == 'const_accel':
                accel_xy = self._clip_velocity(track.velocity_xy - track.prev_velocity_xy)
                motion_delta = motion_delta + 0.5 * self.accel_gain * accel_xy * (steps_ahead ** 2)
            box[0:2] += motion_delta
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

    def _filter_det_by_distance(self, det_boxes, det_scores, det_labels):
        return filter_boxes_by_spatial_range(
            det_boxes,
            det_scores,
            det_labels,
            max_distance=self.max_distance,
            bev_range=self.bev_range,
        )

    def update(self, frame_cache):
        det_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        det_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
        det_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

        keep = det_scores >= self.score_thresh
        det_boxes = det_boxes[keep]
        det_scores = det_scores[keep]
        det_labels = det_labels[keep]
        det_boxes, det_scores, det_labels = self._filter_det_by_distance(det_boxes, det_scores, det_labels)

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
            measured_velocity = self._clip_velocity(new_box[0:2] - prev_box[0:2])
            prev_velocity_xy = track.velocity_xy.copy()
            smoothed_velocity = (
                self.velocity_momentum * prev_velocity_xy
                + (1.0 - self.velocity_momentum) * measured_velocity
            )
            track.prev_velocity_xy = prev_velocity_xy
            track.velocity_xy = self._clip_velocity(smoothed_velocity)
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
                    prev_velocity_xy=np.zeros((2,), dtype=np.float32),
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
    parser.add_argument('--data_cfg', type=str, default=None, help='optional data/detector yaml used to resolve default BEV range')
    parser.add_argument(
        '--dataset_preset',
        type=str,
        default='default',
        choices=['default', 'v2x_xian_2hz'],
        help='optional dataset-specific eval preset; only applied when explicitly requested',
    )
    parser.add_argument('--class_names', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--match_iou', type=float, default=0.1)
    parser.add_argument('--center_gate', type=float, default=8.0)
    parser.add_argument('--max_age', type=int, default=2)
    parser.add_argument('--min_hits', type=int, default=2)
    parser.add_argument('--motion_model', type=str, default='constant_velocity', choices=['constant_velocity', 'const_accel'])
    parser.add_argument('--motion_horizon', type=float, default=1.0, help='prediction horizon in frame units')
    parser.add_argument('--velocity_momentum', type=float, default=0.0, help='EMA momentum for measured velocity updates')
    parser.add_argument('--accel_gain', type=float, default=0.0, help='acceleration gain used only for const_accel motion model')
    parser.add_argument('--max_speed', type=float, default=100.0, help='maximum per-frame XY displacement allowed in motion prediction')
    parser.add_argument('--max_distance', type=float, default=100.0, help='evaluate and track targets within XY range only')
    parser.add_argument(
        '--bev_range',
        type=str,
        default=None,
        help='optional BEV range, e.g. "[-40.96, -28.16, 40.96, 28.16]"; overrides --max_distance when provided',
    )
    parser.add_argument('--save_dir', type=str, required=True, help='output dir for metrics and tracking results')
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def class_names_to_labels(class_names, names):
    name_to_label = {name: idx + 1 for idx, name in enumerate(class_names)}
    return np.asarray([name_to_label.get(name, -1) for name in names], dtype=np.int64)


def parse_bev_range_arg(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        return normalize_bev_range(raw_value)

    text = str(raw_value).strip()
    if not text:
        return None
    if text[0] in '[(':
        values = ast.literal_eval(text)
    else:
        values = [float(item) for item in text.replace(',', ' ').split()]
    return normalize_bev_range(values)


def load_default_bev_range_from_cfg(cfg_file):
    if cfg_file is None:
        return None
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    pc_range = None
    if 'POINT_CLOUD_RANGE' in cfg:
        pc_range = cfg.POINT_CLOUD_RANGE
    elif 'DATA_CONFIG' in cfg and 'POINT_CLOUD_RANGE' in cfg.DATA_CONFIG:
        pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    if pc_range is None:
        return None
    return normalize_bev_range([pc_range[0], pc_range[1], pc_range[3], pc_range[4]])


def resolve_spatial_filter(args):
    cli_bev_range = parse_bev_range_arg(args.bev_range)
    if cli_bev_range is not None:
        return None, cli_bev_range
    return args.max_distance, load_default_bev_range_from_cfg(args.data_cfg)


def apply_dataset_preset(args):
    if args.dataset_preset == 'default':
        return args

    if args.dataset_preset == 'v2x_xian_2hz':
        # Tuned from label_val GT motion statistics: constant velocity is notably more stable than acceleration.
        args.match_iou = 0.01
        args.center_gate = 20.0
        args.max_age = 4
        args.min_hits = 2
        args.motion_model = 'constant_velocity'
        args.motion_horizon = 1.0
        args.velocity_momentum = 0.60
        args.accel_gain = 0.0
        args.max_speed = 30.0
        args.score_thresh = max(float(args.score_thresh), 0.1)
        return args

    raise ValueError(f'Unsupported dataset_preset: {args.dataset_preset}')


def main():
    args = parse_args()
    args = apply_dataset_preset(args)
    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    effective_max_distance, effective_bev_range = resolve_spatial_filter(args)

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
        max_distance=effective_max_distance,
        bev_range=effective_bev_range,
        motion_model=args.motion_model,
        motion_horizon=args.motion_horizon,
        velocity_momentum=args.velocity_momentum,
        accel_gain=args.accel_gain,
        max_speed=args.max_speed,
    )
    metrics = TrackingMetrics(iou_threshold=args.match_iou)
    results = {}
    total_frames = 0
    start_time = time.perf_counter()

    for sequence_id, infos in sequence_to_infos.items():
        tracker.next_track_id = 1
        tracker.tracks = []
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
            gt_boxes, gt_ids, gt_labels = filter_boxes_by_spatial_range(
                gt_boxes, gt_ids, gt_labels, max_distance=effective_max_distance, bev_range=effective_bev_range
            )

            pred_boxes = np.asarray([item['pred_box'] for item in outputs], dtype=np.float32).reshape(-1, 7)
            pred_ids = np.asarray([item['track_id'] for item in outputs], dtype=np.int64)
            pred_labels = np.asarray([item['pred_label'] for item in outputs], dtype=np.int64)
            pred_boxes, pred_ids, pred_labels = filter_boxes_by_spatial_range(
                pred_boxes, pred_ids, pred_labels, max_distance=effective_max_distance, bev_range=effective_bev_range
            )
            metrics.update(sequence_id, gt_boxes, gt_ids, gt_labels, pred_boxes, pred_ids, pred_labels)

        results[sequence_id] = seq_results

    metric_dict = metrics.summary()
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    metric_dict['fps'] = float(total_frames / elapsed)
    metric_dict['max_distance'] = None if effective_max_distance is None else float(effective_max_distance)
    metric_dict['bev_range'] = None if effective_bev_range is None else [float(v) for v in effective_bev_range.tolist()]
    metric_dict['num_frames'] = int(total_frames)
    metric_dict['num_sequences'] = int(len(sequence_to_infos))
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
    with open(save_dir / 'ab3dmot_baseline_metrics.json', 'w') as f:
        json.dump(metric_dict, f, indent=2)
    with open(save_dir / 'ab3dmot_baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('================ AB3DMOT-Style Baseline ================')
    print(f'cache_dir: {cache_dir}')
    print(f'gt_pkl: {args.gt_pkl}')
    print(f'dataset_preset: {args.dataset_preset}')
    print(f'score_thresh: {args.score_thresh}')
    print(f'match_iou: {args.match_iou}')
    print(f'center_gate: {args.center_gate}')
    print(f'motion_model: {args.motion_model}')
    print(f'motion_horizon: {args.motion_horizon}')
    print(f'velocity_momentum: {args.velocity_momentum}')
    print(f'accel_gain: {args.accel_gain}')
    print(f'max_speed: {args.max_speed}')
    print(f'max_age: {args.max_age}')
    print(f'min_hits: {args.min_hits}')
    print(f'max_distance: {args.max_distance}')
    print(f'bev_range: {None if args.bev_range is None else [float(v) for v in normalize_bev_range(args.bev_range).tolist()]}')
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
        f"DetPr={metric_dict.get('detpr', 0.0):.4f} "
        f"DetRe={metric_dict.get('detre', 0.0):.4f} "
        f"AssPr={metric_dict.get('asspr', 0.0):.4f} "
        f"AssRe={metric_dict.get('assre', 0.0):.4f} "
        f"TP={metric_dict.get('tp', 0)} "
        f"PT={metric_dict.get('partially_tracked', 0)} "
        f"Seq={metric_dict.get('num_sequences', 0)} "
        f"Frames={metric_dict.get('num_frames', 0)}"
    )


if __name__ == '__main__':
    main()

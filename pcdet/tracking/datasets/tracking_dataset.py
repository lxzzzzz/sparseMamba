import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..assignment import match_detections_to_gt
from ..cache import load_frame_cache
from ..utils import (
    GEOM_DIM,
    QUALITY_DIM,
    TIME_DIM,
    TRACK_CONTEXT_DIM,
    build_geometry_tokens,
    build_quality_tokens,
    build_time_token,
    build_track_context,
    get_annos,
    get_cache_obs_quality,
    get_cache_reliability,
)


class TrackingDataset(Dataset):
    def __init__(self, dataset_cfg, class_names, training=True):
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.class_name_to_label = {name: idx + 1 for idx, name in enumerate(class_names)}
        self.training = training
        self.mode = 'train' if training else 'test'
        self.root_path = Path(dataset_cfg.ROOT_DIR)
        self.cache_root = self._resolve_cache_root(dataset_cfg, self.mode)
        self.history_len = int(dataset_cfg.get('HISTORY_LEN', 8))
        self.match_iou = float(dataset_cfg.get('DET_GT_MATCH_IOU', 0.1))
        self.min_track_history = int(dataset_cfg.get('MIN_TRACK_HISTORY', 1))

        self.frame_infos = []
        for info_path in dataset_cfg.INFO_PATH[self.mode]:
            with open(self.root_path / info_path, 'rb') as f:
                self.frame_infos.extend(pickle.load(f))

        self.frame_infos.sort(key=lambda info: (str(info['sequence_id']), int(info['frame_idx'])))
        self.sequence_to_infos = defaultdict(list)
        for info in self.frame_infos:
            self.sequence_to_infos[str(info['sequence_id'])].append(info)

        self.samples = []
        for sequence_id, infos in self.sequence_to_infos.items():
            for frame_pos, info in enumerate(infos):
                self.samples.append((sequence_id, frame_pos, info))

        self._frame_cache = {}
        self._obs_cache = {}

    @staticmethod
    def _resolve_cache_root(dataset_cfg, mode):
        if mode == 'train' and 'TRAIN_CACHE_DIR' in dataset_cfg:
            return Path(dataset_cfg.TRAIN_CACHE_DIR)
        if mode == 'test' and 'VAL_CACHE_DIR' in dataset_cfg:
            return Path(dataset_cfg.VAL_CACHE_DIR)
        if 'CACHE_DIR' in dataset_cfg:
            return Path(dataset_cfg.CACHE_DIR)
        raise KeyError('Expected CACHE_DIR or split-specific TRAIN_CACHE_DIR / VAL_CACHE_DIR in tracking config')

    def __len__(self):
        return len(self.samples)

    def _label_array(self, names):
        return np.asarray([self.class_name_to_label.get(name, -1) for name in names], dtype=np.int64)

    @staticmethod
    def _frame_key(info):
        return str(info['sequence_id']), int(info['frame_idx'])

    def _history_window(self, sequence_id, frame_pos):
        start = max(0, frame_pos - self.history_len)
        return self.sequence_to_infos[sequence_id][start:frame_pos]

    def _load_frame_cache(self, info):
        key = self._frame_key(info)
        if key not in self._frame_cache:
            self._frame_cache[key] = load_frame_cache(self.cache_root, info['sequence_id'], info['frame_idx'])
        return self._frame_cache[key]

    def _gt_track_map(self, info):
        annos = get_annos(info)
        gt_labels = self._label_array(annos['name'])
        track_map = {}
        for idx, track_id in enumerate(annos['track_id'].tolist()):
            track_map[int(track_id)] = {
                'box': annos['gt_boxes_lidar'][idx],
                'label': gt_labels[idx],
            }
        return annos, gt_labels, track_map

    def _frame_detections(self, info):
        frame_cache = self._load_frame_cache(info)
        boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32)
        labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64)
        reliability = get_cache_reliability(frame_cache)
        obs_quality_vec = get_cache_obs_quality(frame_cache, len(scores))
        geom_tokens = build_geometry_tokens(boxes)
        quality_tokens, quality_scalar = build_quality_tokens(scores, reliability, obs_quality_vec)
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'reliability': reliability,
            'obs_quality_vec': obs_quality_vec,
            'geom_tokens': geom_tokens,
            'quality_tokens': quality_tokens,
            'quality_scalar': quality_scalar,
        }

    def _matched_observations(self, info):
        key = self._frame_key(info)
        if key in self._obs_cache:
            return self._obs_cache[key]

        annos, gt_labels, _ = self._gt_track_map(info)
        detections = self._frame_detections(info)
        det_to_gt, _, _ = match_detections_to_gt(
            det_boxes=detections['boxes'],
            det_labels=detections['labels'],
            gt_boxes=annos['gt_boxes_lidar'],
            gt_labels=gt_labels,
            iou_threshold=self.match_iou,
        )

        matched = {}
        for det_idx, gt_idx in enumerate(det_to_gt.tolist()):
            if gt_idx < 0:
                continue
            track_id = int(annos['track_id'][gt_idx])
            matched[track_id] = {
                'box': detections['boxes'][det_idx],
                'label': int(detections['labels'][det_idx]),
                'geom_token': detections['geom_tokens'][det_idx],
                'quality_token': detections['quality_tokens'][det_idx],
                'quality_scalar': float(detections['quality_scalar'][det_idx]),
                'det_idx': int(det_idx),
            }
        self._obs_cache[key] = matched
        return matched

    def _build_track_history(self, track_id, history_steps):
        geom_history = np.zeros((self.history_len, GEOM_DIM), dtype=np.float32)
        quality_history = np.zeros((self.history_len, QUALITY_DIM), dtype=np.float32)
        time_history = np.zeros((self.history_len, TIME_DIM), dtype=np.float32)
        history_mask = np.zeros((self.history_len,), dtype=np.float32)

        offset = self.history_len - len(history_steps)
        hit_count = 0
        last_obs_step = None
        last_quality = 0.0
        prev_quality = None
        last_box = np.zeros((7,), dtype=np.float32)
        quality_values = []
        observed_flags = []

        for step_idx, observations in enumerate(history_steps):
            obs = observations.get(track_id, None)
            observed_flags.append(obs is not None)
            if obs is None:
                continue

            slot = offset + step_idx
            geom_history[slot] = obs['geom_token']
            quality_history[slot] = obs['quality_token']
            history_mask[slot] = 1.0

            hit_count += 1
            missing_gap = step_idx if last_obs_step is None else (step_idx - last_obs_step - 1)
            quality_trend = 0.0 if prev_quality is None else obs['quality_scalar'] - prev_quality
            time_history[slot] = build_time_token(
                age_delta=float(len(history_steps) - step_idx) / float(max(self.history_len, 1)),
                missing_gap=float(missing_gap) / float(max(self.history_len, 1)),
                hit_count=float(hit_count) / float(max(self.history_len, 1)),
                quality_trend=quality_trend,
            )

            prev_quality = obs['quality_scalar']
            last_quality = obs['quality_scalar']
            last_obs_step = step_idx
            last_box = obs['box']
            quality_values.append(obs['quality_scalar'])

        current_missing_gap = len(history_steps) if last_obs_step is None else (len(history_steps) - last_obs_step - 1)
        observed_ratio = float(sum(observed_flags)) / float(max(len(history_steps), 1)) if len(history_steps) > 0 else 0.0
        track_context = build_track_context(
            current_missing_gap=float(current_missing_gap) / float(max(self.history_len, 1)),
            hit_count=float(hit_count) / float(max(self.history_len, 1)),
            last_quality=last_quality,
            observed_ratio=observed_ratio,
        )

        return {
            'geom_history': geom_history,
            'quality_history': quality_history,
            'time_history': time_history,
            'history_mask': history_mask,
            'track_context': track_context,
            'last_box': last_box.astype(np.float32),
            'quality_values': quality_values,
            'observed_flags': observed_flags,
            'current_missing_gap': current_missing_gap,
            'hit_count': hit_count,
        }

    def __getitem__(self, index):
        sequence_id, frame_pos, current_info = self.samples[index]
        history_infos = self._history_window(sequence_id, frame_pos)
        history_steps = [self._matched_observations(info) for info in history_infos]

        track_labels = {}
        track_obs_count = defaultdict(int)
        for observations in history_steps:
            for track_id, obs in observations.items():
                track_labels[track_id] = obs['label']
                track_obs_count[track_id] += 1

        active_track_ids = [
            track_id for track_id in sorted(track_labels.keys()) if track_obs_count[track_id] >= self.min_track_history
        ]

        current_annos, current_gt_labels, current_gt_map = self._gt_track_map(current_info)
        current_dets = self._frame_detections(current_info)
        current_obs = self._matched_observations(current_info)

        next_gt_map = {}
        if frame_pos + 1 < len(self.sequence_to_infos[sequence_id]):
            _, _, next_gt_map = self._gt_track_map(self.sequence_to_infos[sequence_id][frame_pos + 1])

        num_tracks = len(active_track_ids)
        num_dets = current_dets['boxes'].shape[0]
        track_geom_history = np.zeros((num_tracks, self.history_len, GEOM_DIM), dtype=np.float32)
        track_quality_history = np.zeros((num_tracks, self.history_len, QUALITY_DIM), dtype=np.float32)
        track_time_history = np.zeros((num_tracks, self.history_len, TIME_DIM), dtype=np.float32)
        track_mask = np.zeros((num_tracks, self.history_len), dtype=np.float32)
        track_context = np.zeros((num_tracks, TRACK_CONTEXT_DIM), dtype=np.float32)
        track_boxes = np.zeros((num_tracks, 7), dtype=np.float32)
        track_class_ids = np.zeros((num_tracks,), dtype=np.int64)
        next_geom_targets = np.zeros((num_tracks, GEOM_DIM), dtype=np.float32)
        next_geom_mask = np.zeros((num_tracks,), dtype=np.float32)

        assoc_targets = np.zeros((num_tracks, num_dets), dtype=np.float32)
        assoc_mask = np.zeros((num_tracks, num_dets), dtype=bool)

        det_to_gt, _, _ = match_detections_to_gt(
            det_boxes=current_dets['boxes'],
            det_labels=current_dets['labels'],
            gt_boxes=current_annos['gt_boxes_lidar'],
            gt_labels=current_gt_labels,
            iou_threshold=self.match_iou,
        )

        track_id_to_pos = {track_id: idx for idx, track_id in enumerate(active_track_ids)}
        for track_pos, track_id in enumerate(active_track_ids):
            history_desc = self._build_track_history(track_id, history_steps)
            track_geom_history[track_pos] = history_desc['geom_history']
            track_quality_history[track_pos] = history_desc['quality_history']
            track_time_history[track_pos] = history_desc['time_history']
            track_mask[track_pos] = history_desc['history_mask']
            track_context[track_pos] = history_desc['track_context']
            track_boxes[track_pos] = history_desc['last_box']
            track_class_ids[track_pos] = track_labels[track_id]
            assoc_mask[track_pos, current_dets['labels'] == track_labels[track_id]] = True
            if track_id in next_gt_map:
                next_geom_targets[track_pos] = build_geometry_tokens(next_gt_map[track_id]['box'][None, :])[0]
                next_geom_mask[track_pos] = 1.0

        for det_idx, gt_idx in enumerate(det_to_gt.tolist()):
            if gt_idx < 0:
                continue
            track_id = int(current_annos['track_id'][gt_idx])
            if track_id not in track_id_to_pos:
                continue
            assoc_targets[track_id_to_pos[track_id], det_idx] = 1.0

        return {
            'sequence_id': str(sequence_id),
            'frame_id': str(current_info['frame_id']),
            'frame_idx': int(current_info['frame_idx']),
            'track_ids': np.asarray(active_track_ids, dtype=np.int64),
            'track_geom_history': track_geom_history,
            'track_quality_history': track_quality_history,
            'track_time_history': track_time_history,
            'track_mask': track_mask,
            'track_context': track_context,
            'track_boxes': track_boxes,
            'track_class_ids': track_class_ids,
            'candidate_det_geom': current_dets['geom_tokens'].astype(np.float32),
            'candidate_det_quality': current_dets['quality_tokens'].astype(np.float32),
            'candidate_det_boxes': current_dets['boxes'].astype(np.float32),
            'candidate_det_scores': current_dets['scores'].astype(np.float32),
            'candidate_det_labels': current_dets['labels'].astype(np.int64),
            'assoc_targets': assoc_targets.astype(np.float32),
            'assoc_mask': assoc_mask,
            'next_geom_targets': next_geom_targets.astype(np.float32),
            'next_geom_mask': next_geom_mask.astype(np.float32),
            'gt_boxes': current_annos['gt_boxes_lidar'].astype(np.float32),
            'gt_track_ids': current_annos['track_id'].astype(np.int64),
            'gt_labels': current_gt_labels.astype(np.int64),
        }


def collate_tracking_batch(batch_list):
    batch_size = len(batch_list)
    history_len = max(item['track_geom_history'].shape[1] for item in batch_list)
    max_tracks = max(1, max(item['track_geom_history'].shape[0] for item in batch_list))
    max_dets = max(1, max(item['candidate_det_geom'].shape[0] for item in batch_list))
    max_gt = max(1, max(item['gt_boxes'].shape[0] for item in batch_list))

    ret = {
        'sequence_id': [],
        'frame_id': [],
        'frame_idx': torch.zeros(batch_size, dtype=torch.long),
        'track_ids': torch.full((batch_size, max_tracks), -1, dtype=torch.long),
        'track_geom_history': torch.zeros(batch_size, max_tracks, history_len, GEOM_DIM, dtype=torch.float32),
        'track_quality_history': torch.zeros(batch_size, max_tracks, history_len, QUALITY_DIM, dtype=torch.float32),
        'track_time_history': torch.zeros(batch_size, max_tracks, history_len, TIME_DIM, dtype=torch.float32),
        'track_mask': torch.zeros(batch_size, max_tracks, history_len, dtype=torch.float32),
        'track_context': torch.zeros(batch_size, max_tracks, TRACK_CONTEXT_DIM, dtype=torch.float32),
        'track_boxes': torch.zeros(batch_size, max_tracks, 7, dtype=torch.float32),
        'track_class_ids': torch.zeros(batch_size, max_tracks, dtype=torch.long),
        'candidate_det_geom': torch.zeros(batch_size, max_dets, GEOM_DIM, dtype=torch.float32),
        'candidate_det_quality': torch.zeros(batch_size, max_dets, QUALITY_DIM, dtype=torch.float32),
        'candidate_det_boxes': torch.zeros(batch_size, max_dets, 7, dtype=torch.float32),
        'candidate_det_scores': torch.zeros(batch_size, max_dets, dtype=torch.float32),
        'candidate_det_labels': torch.zeros(batch_size, max_dets, dtype=torch.long),
        'assoc_targets': torch.zeros(batch_size, max_tracks, max_dets, dtype=torch.float32),
        'assoc_mask': torch.zeros(batch_size, max_tracks, max_dets, dtype=torch.bool),
        'next_geom_targets': torch.zeros(batch_size, max_tracks, GEOM_DIM, dtype=torch.float32),
        'next_geom_mask': torch.zeros(batch_size, max_tracks, dtype=torch.float32),
        'gt_boxes': torch.zeros(batch_size, max_gt, 7, dtype=torch.float32),
        'gt_track_ids': torch.full((batch_size, max_gt), -1, dtype=torch.long),
        'gt_labels': torch.zeros(batch_size, max_gt, dtype=torch.long),
    }

    for batch_idx, item in enumerate(batch_list):
        num_tracks = item['track_geom_history'].shape[0]
        num_dets = item['candidate_det_geom'].shape[0]
        num_gt = item['gt_boxes'].shape[0]

        ret['sequence_id'].append(item['sequence_id'])
        ret['frame_id'].append(item['frame_id'])
        ret['frame_idx'][batch_idx] = item['frame_idx']

        if num_tracks > 0:
            ret['track_ids'][batch_idx, :num_tracks] = torch.from_numpy(item['track_ids'])
            ret['track_geom_history'][batch_idx, :num_tracks] = torch.from_numpy(item['track_geom_history'])
            ret['track_quality_history'][batch_idx, :num_tracks] = torch.from_numpy(item['track_quality_history'])
            ret['track_time_history'][batch_idx, :num_tracks] = torch.from_numpy(item['track_time_history'])
            ret['track_mask'][batch_idx, :num_tracks] = torch.from_numpy(item['track_mask'])
            ret['track_context'][batch_idx, :num_tracks] = torch.from_numpy(item['track_context'])
            ret['track_boxes'][batch_idx, :num_tracks] = torch.from_numpy(item['track_boxes'])
            ret['track_class_ids'][batch_idx, :num_tracks] = torch.from_numpy(item['track_class_ids'])
            ret['assoc_targets'][batch_idx, :num_tracks, :num_dets] = torch.from_numpy(item['assoc_targets'])
            ret['assoc_mask'][batch_idx, :num_tracks, :num_dets] = torch.from_numpy(item['assoc_mask'])
            ret['next_geom_targets'][batch_idx, :num_tracks] = torch.from_numpy(item['next_geom_targets'])
            ret['next_geom_mask'][batch_idx, :num_tracks] = torch.from_numpy(item['next_geom_mask'])

        if num_dets > 0:
            ret['candidate_det_geom'][batch_idx, :num_dets] = torch.from_numpy(item['candidate_det_geom'])
            ret['candidate_det_quality'][batch_idx, :num_dets] = torch.from_numpy(item['candidate_det_quality'])
            ret['candidate_det_boxes'][batch_idx, :num_dets] = torch.from_numpy(item['candidate_det_boxes'])
            ret['candidate_det_scores'][batch_idx, :num_dets] = torch.from_numpy(item['candidate_det_scores'])
            ret['candidate_det_labels'][batch_idx, :num_dets] = torch.from_numpy(item['candidate_det_labels'])

        if num_gt > 0:
            ret['gt_boxes'][batch_idx, :num_gt] = torch.from_numpy(item['gt_boxes'])
            ret['gt_track_ids'][batch_idx, :num_gt] = torch.from_numpy(item['gt_track_ids'])
            ret['gt_labels'][batch_idx, :num_gt] = torch.from_numpy(item['gt_labels'])

    return ret

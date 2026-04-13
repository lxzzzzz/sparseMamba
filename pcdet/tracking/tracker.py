from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .assignment import bev_iou_matrix, hungarian_assign
from .utils import (
    GEOM_DIM,
    QUALITY_DIM,
    TIME_DIM,
    TRACK_CONTEXT_DIM,
    association_quality,
    build_geometry_tokens,
    build_quality_tokens,
    build_time_token,
    build_track_context,
    get_cache_obs_quality,
)


@dataclass
class TrackState:
    track_id: int
    label: int
    score: float
    hits: int
    missed: int
    last_box: np.ndarray
    geom_history: deque = field(default_factory=lambda: deque(maxlen=8))
    quality_history: deque = field(default_factory=lambda: deque(maxlen=8))
    time_history: deque = field(default_factory=lambda: deque(maxlen=8))
    quality_scalar_history: deque = field(default_factory=lambda: deque(maxlen=8))

    def append(self, geom_token, quality_token, time_token, quality_scalar, box, score):
        self.geom_history.append(np.asarray(geom_token, dtype=np.float32))
        self.quality_history.append(np.asarray(quality_token, dtype=np.float32))
        self.time_history.append(np.asarray(time_token, dtype=np.float32))
        self.quality_scalar_history.append(float(quality_scalar))
        self.last_box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.hits += 1
        self.missed = 0


class OnlineTracker:
    def __init__(self, model, tracker_cfg, class_names, device):
        self.model = model
        self.tracker_cfg = tracker_cfg
        self.class_names = class_names
        self.device = device
        self.history_len = int(tracker_cfg.get('HISTORY_LEN', 8))
        self.max_age = int(tracker_cfg.get('MAX_AGE', 3))
        self.min_hits = int(tracker_cfg.get('MIN_HITS', 2))
        self.match_threshold = float(tracker_cfg.get('MATCH_THRESHOLD', 0.5))
        self.recovery_match_threshold = float(tracker_cfg.get('RECOVERY_MATCH_THRESHOLD', self.match_threshold + 0.1))
        self.high_quality_thresh = float(tracker_cfg.get('HIGH_QUALITY_THRESH', 0.3))
        self.low_quality_thresh = float(tracker_cfg.get('LOW_QUALITY_THRESH', 0.1))
        self.allow_low_quality_birth = bool(tracker_cfg.get('ALLOW_LOW_QUALITY_BIRTH', False))
        self.assoc_iou_gate = float(tracker_cfg.get('ASSOC_IOU_GATE', 0.01))
        self.assoc_center_dist = float(tracker_cfg.get('ASSOC_CENTER_DIST', 8.0))
        self.recovery_assoc_iou_gate = float(tracker_cfg.get('RECOVERY_ASSOC_IOU_GATE', self.assoc_iou_gate))
        self.recovery_assoc_center_dist = float(tracker_cfg.get('RECOVERY_ASSOC_CENTER_DIST', max(self.assoc_center_dist * 0.5, 1.0)))
        self.reset()

    def reset(self):
        self.next_track_id = 1
        self.tracks = []

    def _build_track_batch(self):
        num_tracks = len(self.tracks)
        track_geom_history = np.zeros((1, num_tracks, self.history_len, GEOM_DIM), dtype=np.float32)
        track_quality_history = np.zeros((1, num_tracks, self.history_len, QUALITY_DIM), dtype=np.float32)
        track_time_history = np.zeros((1, num_tracks, self.history_len, TIME_DIM), dtype=np.float32)
        track_mask = np.zeros((1, num_tracks, self.history_len), dtype=np.float32)
        track_context = np.zeros((1, num_tracks, TRACK_CONTEXT_DIM), dtype=np.float32)
        track_boxes = np.zeros((num_tracks, 7), dtype=np.float32)
        track_labels = np.zeros((num_tracks,), dtype=np.int64)

        for idx, track in enumerate(self.tracks):
            geom_hist = list(track.geom_history)[-self.history_len:]
            quality_hist = list(track.quality_history)[-self.history_len:]
            time_hist = list(track.time_history)[-self.history_len:]
            offset = self.history_len - len(geom_hist)

            if geom_hist:
                track_geom_history[0, idx, offset:] = np.stack(geom_hist, axis=0)
                track_quality_history[0, idx, offset:] = np.stack(quality_hist, axis=0)
                track_time_history[0, idx, offset:] = np.stack(time_hist, axis=0)
                track_mask[0, idx, offset:] = 1.0

            last_quality = track.quality_scalar_history[-1] if len(track.quality_scalar_history) > 0 else 0.0
            observed_ratio = len(geom_hist) / float(max(self.history_len, 1))
            track_context[0, idx] = build_track_context(
                current_missing_gap=float(track.missed) / float(max(self.history_len, 1)),
                hit_count=float(track.hits) / float(max(self.history_len, 1)),
                last_quality=last_quality,
                observed_ratio=observed_ratio,
            )
            track_boxes[idx] = track.last_box
            track_labels[idx] = track.label

        batch = {
            'track_geom_history': torch.from_numpy(track_geom_history).to(self.device),
            'track_quality_history': torch.from_numpy(track_quality_history).to(self.device),
            'track_time_history': torch.from_numpy(track_time_history).to(self.device),
            'track_mask': torch.from_numpy(track_mask).to(self.device),
            'track_context': torch.from_numpy(track_context).to(self.device),
        }
        return batch, track_boxes, track_labels

    def _candidate_mask(self, track_boxes, track_labels, det_boxes, det_labels, iou_gate, center_dist_gate):
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return np.zeros((len(track_boxes), len(det_boxes)), dtype=bool)

        iou = bev_iou_matrix(track_boxes, det_boxes)
        same_class = track_labels[:, None] == det_labels[None, :]
        track_centers = np.asarray(track_boxes, dtype=np.float32)[:, None, 0:2]
        det_centers = np.asarray(det_boxes, dtype=np.float32)[None, :, 0:2]
        center_dist = np.linalg.norm(track_centers - det_centers, axis=-1)
        return same_class & ((iou >= iou_gate) | (center_dist <= center_dist_gate))

    def _detection_tokens(self, frame_data):
        det_boxes = np.asarray(frame_data.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        det_scores = np.asarray(frame_data.get('pred_scores', []), dtype=np.float32)
        det_labels = np.asarray(frame_data.get('pred_labels', []), dtype=np.int64)
        reliability = np.asarray(frame_data.get('reliability_scores', det_scores), dtype=np.float32)
        obs_quality_vec = get_cache_obs_quality(frame_data, len(det_scores))
        det_geom = build_geometry_tokens(det_boxes)
        det_quality, det_quality_scalar = build_quality_tokens(det_scores, reliability, obs_quality_vec)
        det_assoc_quality = association_quality(det_scores, reliability, obs_quality_vec, quality_scalar=det_quality_scalar)
        return det_boxes, det_scores, det_labels, det_geom, det_quality, det_quality_scalar, det_assoc_quality

    @staticmethod
    def _run_matching(pair_scores, candidate_mask, threshold, track_indices=None, det_indices=None):
        if pair_scores.size == 0:
            return []
        if track_indices is not None:
            track_indices = np.asarray(track_indices, dtype=np.int64)
            pair_scores = pair_scores[track_indices]
            candidate_mask = candidate_mask[track_indices]
        if det_indices is not None:
            det_indices = np.asarray(det_indices, dtype=np.int64)
            pair_scores = pair_scores[:, det_indices]
            candidate_mask = candidate_mask[:, det_indices]

        matches = []
        for track_pos, det_pos in hungarian_assign(1.0 - pair_scores, candidate_mask):
            if pair_scores[track_pos, det_pos] < threshold:
                continue
            global_track_idx = int(track_indices[track_pos]) if track_indices is not None else int(track_pos)
            global_det_idx = int(det_indices[det_pos]) if det_indices is not None else int(det_pos)
            matches.append((global_track_idx, global_det_idx))
        return matches

    def _append_new_track(self, label, score, det_box, det_geom, det_quality, det_quality_scalar):
        initial_time = build_time_token(age_delta=0.0, missing_gap=0.0, hit_count=1.0 / float(max(self.history_len, 1)), quality_trend=0.0)
        track = TrackState(
            track_id=self.next_track_id,
            label=int(label),
            score=float(score),
            hits=1,
            missed=0,
            last_box=det_box.copy(),
            geom_history=deque(maxlen=self.history_len),
            quality_history=deque(maxlen=self.history_len),
            time_history=deque(maxlen=self.history_len),
            quality_scalar_history=deque(maxlen=self.history_len),
        )
        track.append(det_geom, det_quality, initial_time, det_quality_scalar, det_box, score)
        self.tracks.append(track)
        self.next_track_id += 1

    def _update_track_with_detection(self, track_idx, det_idx, det_boxes, det_scores, det_geom, det_quality, det_quality_scalar):
        missing_gap = float(self.tracks[track_idx].missed) / float(max(self.history_len, 1))
        hit_count = float(self.tracks[track_idx].hits + 1) / float(max(self.history_len, 1))
        quality_trend = det_quality_scalar[det_idx] - (
            self.tracks[track_idx].quality_scalar_history[-1] if len(self.tracks[track_idx].quality_scalar_history) > 0 else det_quality_scalar[det_idx]
        )
        time_token = build_time_token(
            age_delta=0.0,
            missing_gap=missing_gap,
            hit_count=hit_count,
            quality_trend=quality_trend,
        )
        self.tracks[track_idx].append(
            det_geom[det_idx],
            det_quality[det_idx],
            time_token,
            det_quality_scalar[det_idx],
            det_boxes[det_idx],
            det_scores[det_idx],
        )

    def update(self, frame_data):
        det_boxes, det_scores, det_labels, det_geom, det_quality, det_quality_scalar, det_assoc_quality = self._detection_tokens(frame_data)
        for track in self.tracks:
            track.missed += 1

        matched_tracks = set()
        matched_dets = set()
        high_quality_mask = det_assoc_quality >= self.high_quality_thresh
        low_quality_mask = (det_assoc_quality >= self.low_quality_thresh) & ~high_quality_mask
        if self.tracks and len(det_boxes) > 0:
            track_batch, track_boxes, track_labels = self._build_track_batch()
            model_input = dict(track_batch)
            model_input['candidate_det_geom'] = torch.from_numpy(det_geom).unsqueeze(0).to(self.device)
            model_input['candidate_det_quality'] = torch.from_numpy(det_quality).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(model_input)
                pair_scores = torch.sigmoid(output['association_logits'][0]).cpu().numpy()

            primary_mask = self._candidate_mask(
                track_boxes, track_labels, det_boxes, det_labels, self.assoc_iou_gate, self.assoc_center_dist
            )
            primary_matches = self._run_matching(
                pair_scores, primary_mask, self.match_threshold, det_indices=np.flatnonzero(high_quality_mask)
            )
            for track_idx, det_idx in primary_matches:
                self._update_track_with_detection(
                    track_idx, det_idx, det_boxes, det_scores, det_geom, det_quality, det_quality_scalar
                )
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

            recovery_track_indices = np.asarray(
                [idx for idx, track in enumerate(self.tracks) if idx not in matched_tracks and track.missed > 0],
                dtype=np.int64,
            )
            recovery_det_indices = np.flatnonzero(low_quality_mask & ~np.isin(np.arange(len(det_boxes)), list(matched_dets)))
            if recovery_track_indices.size > 0 and recovery_det_indices.size > 0:
                recovery_mask = self._candidate_mask(
                    track_boxes,
                    track_labels,
                    det_boxes,
                    det_labels,
                    self.recovery_assoc_iou_gate,
                    self.recovery_assoc_center_dist,
                )
                recovery_matches = self._run_matching(
                    pair_scores,
                    recovery_mask,
                    self.recovery_match_threshold,
                    track_indices=recovery_track_indices,
                    det_indices=recovery_det_indices,
                )
                for track_idx, det_idx in recovery_matches:
                    self._update_track_with_detection(
                        track_idx, det_idx, det_boxes, det_scores, det_geom, det_quality, det_quality_scalar
                    )
                    matched_tracks.add(track_idx)
                    matched_dets.add(det_idx)

        survivors = []
        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks and track.missed > self.max_age:
                continue
            survivors.append(track)
        self.tracks = survivors

        for det_idx in range(len(det_boxes)):
            if det_idx in matched_dets:
                continue
            if det_assoc_quality[det_idx] < self.high_quality_thresh and not self.allow_low_quality_birth:
                continue
            self._append_new_track(
                label=det_labels[det_idx],
                score=det_scores[det_idx],
                det_box=det_boxes[det_idx],
                det_geom=det_geom[det_idx],
                det_quality=det_quality[det_idx],
                det_quality_scalar=det_quality_scalar[det_idx],
            )

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

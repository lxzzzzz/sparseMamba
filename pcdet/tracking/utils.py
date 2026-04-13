import pickle
from pathlib import Path

import numpy as np

GEOM_DIM = 8
QUALITY_DIM = 7
TIME_DIM = 4
TRACK_CONTEXT_DIM = 4


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def frame_cache_rel_path(sequence_id, frame_idx):
    return Path(str(sequence_id)) / f'{int(frame_idx):06d}.pkl'


def ensure_quality_vec(obs_quality_vec, length):
    obs_quality_vec = np.asarray(obs_quality_vec, dtype=np.float32)
    if obs_quality_vec.size == 0:
        return np.zeros((length, 5), dtype=np.float32)
    return obs_quality_vec.reshape(length, -1)


def build_geometry_tokens(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    if boxes.shape[0] == 0:
        return np.zeros((0, GEOM_DIM), dtype=np.float32)
    geom = np.zeros((boxes.shape[0], GEOM_DIM), dtype=np.float32)
    geom[:, 0:6] = boxes[:, 0:6]
    geom[:, 6] = np.sin(boxes[:, 6])
    geom[:, 7] = np.cos(boxes[:, 6])
    return geom


def compose_quality_scalar(scores, reliability, obs_quality_vec):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    reliability = np.asarray(reliability, dtype=np.float32).reshape(-1)
    obs_quality_vec = ensure_quality_vec(obs_quality_vec, len(scores))
    quality_mean = obs_quality_vec.mean(axis=1) if obs_quality_vec.shape[1] > 0 else np.zeros_like(scores)
    quality_scalar = 0.40 * scores + 0.35 * reliability + 0.25 * quality_mean
    return np.clip(quality_scalar.astype(np.float32), 0.0, 1.0)


def build_quality_tokens(scores, reliability, obs_quality_vec):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    reliability = np.asarray(reliability, dtype=np.float32).reshape(-1)
    obs_quality_vec = ensure_quality_vec(obs_quality_vec, len(scores))
    if len(scores) == 0:
        return np.zeros((0, QUALITY_DIM), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    quality_scalar = compose_quality_scalar(scores, reliability, obs_quality_vec)
    quality_tokens = np.concatenate([
        scores[:, None],
        reliability[:, None],
        obs_quality_vec
    ], axis=1).astype(np.float32)
    return quality_tokens, quality_scalar


def has_meaningful_quality(scores, reliability, obs_quality_vec):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    reliability = np.asarray(reliability, dtype=np.float32).reshape(-1)
    obs_quality_vec = ensure_quality_vec(obs_quality_vec, len(scores))
    if len(scores) == 0:
        return False
    if np.any(np.abs(obs_quality_vec) > 1e-6):
        return True
    return not np.allclose(reliability, scores, atol=1e-6)


def association_quality(scores, reliability, obs_quality_vec, quality_scalar=None):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    reliability = np.asarray(reliability, dtype=np.float32).reshape(-1)
    obs_quality_vec = ensure_quality_vec(obs_quality_vec, len(scores))
    if quality_scalar is None:
        quality_scalar = compose_quality_scalar(scores, reliability, obs_quality_vec)
    else:
        quality_scalar = np.asarray(quality_scalar, dtype=np.float32).reshape(-1)
    if has_meaningful_quality(scores, reliability, obs_quality_vec):
        return quality_scalar.astype(np.float32)
    return scores.astype(np.float32)


def build_time_token(age_delta, missing_gap, hit_count, quality_trend):
    return np.asarray([
        float(age_delta),
        float(missing_gap),
        float(hit_count),
        float(quality_trend),
    ], dtype=np.float32)


def build_track_context(current_missing_gap, hit_count, last_quality, observed_ratio):
    return np.asarray([
        float(current_missing_gap),
        float(hit_count),
        float(last_quality),
        float(observed_ratio),
    ], dtype=np.float32)


def get_annos(info):
    annos = info.get('annos', {})
    return {
        'name': np.asarray(annos.get('name', [])),
        'track_id': np.asarray(annos.get('track_id', []), dtype=np.int64),
        'gt_boxes_lidar': np.asarray(annos.get('gt_boxes_lidar', []), dtype=np.float32).reshape(-1, 7),
    }


def get_cache_reliability(frame_cache):
    if 'reliability_scores' in frame_cache:
        return np.asarray(frame_cache['reliability_scores'], dtype=np.float32)
    return np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32)


def get_cache_obs_quality(frame_cache, length=None):
    obs_quality_vec = np.asarray(frame_cache.get('obs_quality_vec', []), dtype=np.float32)
    if length is None:
        if obs_quality_vec.ndim > 0 and obs_quality_vec.shape[0] > 0:
            length = obs_quality_vec.shape[0]
        else:
            length = len(frame_cache.get('pred_scores', []))
    return ensure_quality_vec(obs_quality_vec, length)


def normalize_bev_range(bev_range):
    if bev_range is None:
        return None
    bev_range = np.asarray(bev_range, dtype=np.float32).reshape(-1)
    if bev_range.shape[0] != 4:
        raise ValueError(f'bev_range must have 4 values [x_min, y_min, x_max, y_max], got {bev_range.tolist()}')
    x_min, y_min, x_max, y_max = bev_range.tolist()
    if x_min > x_max or y_min > y_max:
        raise ValueError(f'invalid bev_range ordering: {bev_range.tolist()}')
    return np.asarray([x_min, y_min, x_max, y_max], dtype=np.float32)


def build_box_filter_mask(boxes, max_distance=None, bev_range=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    keep = np.ones((boxes.shape[0],), dtype=np.bool_)
    if boxes.shape[0] == 0:
        return keep

    if max_distance is not None:
        ranges = np.linalg.norm(boxes[:, 0:2], axis=1)
        keep &= ranges <= float(max_distance)

    bev_range = normalize_bev_range(bev_range)
    if bev_range is not None:
        x_min, y_min, x_max, y_max = bev_range.tolist()
        keep &= (
            (boxes[:, 0] >= x_min)
            & (boxes[:, 0] <= x_max)
            & (boxes[:, 1] >= y_min)
            & (boxes[:, 1] <= y_max)
        )
    return keep


def filter_boxes_by_spatial_range(boxes, *extra_arrays, max_distance=None, bev_range=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    keep = build_box_filter_mask(boxes, max_distance=max_distance, bev_range=bev_range)
    filtered = [boxes[keep]]
    for arr in extra_arrays:
        arr = np.asarray(arr)
        filtered.append(arr[keep])
    return tuple(filtered)

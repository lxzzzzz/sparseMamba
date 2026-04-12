import _init_path
import argparse
from pathlib import Path

import numpy as np

from pcdet.tracking.cache import load_cache_index, load_frame_cache, save_cache_index, save_frame_cache


def parse_args():
    parser = argparse.ArgumentParser(description='Filter tracking detector cache by BEV box area')
    parser.add_argument('--src_cache_dir', type=str, required=True, help='source cache root')
    parser.add_argument('--dst_cache_dir', type=str, required=True, help='filtered cache root')
    parser.add_argument(
        '--keep_labels',
        type=int,
        nargs='*',
        default=None,
        help='only keep these label ids, e.g. 1 for Car, or 1 2 for Car/Pedestrian',
    )
    parser.add_argument('--min_area', type=float, default=None, help='minimum BEV area dx*dy to keep')
    parser.add_argument('--max_area', type=float, default=None, help='maximum BEV area dx*dy to keep')
    parser.add_argument('--verbose_every', type=int, default=200, help='print progress every N frames')
    return parser.parse_args()


def ensure_vector_length(vec, length, width=None):
    arr = np.asarray(vec)
    if arr.size == 0:
        if width is None:
            return np.zeros((length,), dtype=np.float32)
        return np.zeros((length, width), dtype=np.float32)
    if width is None:
        return arr.reshape(-1)
    return arr.reshape(length, width)


def build_keep_mask(frame_cache, keep_labels=None, min_area=None, max_area=None):
    boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
    labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

    if boxes.shape[0] == 0:
        return boxes, np.zeros((0,), dtype=np.bool_), np.zeros((0,), dtype=np.float32)

    bev_area = boxes[:, 3] * boxes[:, 4]
    keep = np.ones((boxes.shape[0],), dtype=np.bool_)

    if keep_labels is not None and len(keep_labels) > 0:
        keep &= np.isin(labels, np.asarray(keep_labels, dtype=np.int64))
    if min_area is not None:
        keep &= bev_area >= float(min_area)
    if max_area is not None:
        keep &= bev_area <= float(max_area)

    return boxes, keep, bev_area


def filter_frame_cache(frame_cache, keep_mask):
    num_boxes = int(keep_mask.shape[0])
    pred_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
    pred_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
    pred_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)
    reliability = ensure_vector_length(frame_cache.get('reliability_scores', pred_scores), num_boxes)
    obs_quality = ensure_vector_length(frame_cache.get('obs_quality_vec', []), num_boxes, width=5)

    filtered = dict(frame_cache)
    filtered['pred_boxes'] = pred_boxes[keep_mask]
    filtered['pred_scores'] = pred_scores[keep_mask]
    filtered['pred_labels'] = pred_labels[keep_mask]
    filtered['reliability_scores'] = reliability[keep_mask]
    filtered['obs_quality_vec'] = obs_quality[keep_mask]
    return filtered


def main():
    args = parse_args()
    src_cache_dir = Path(args.src_cache_dir)
    dst_cache_dir = Path(args.dst_cache_dir)
    dst_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_index = load_cache_index(src_cache_dir)
    new_index = []

    total_before = 0
    total_after = 0

    for idx, item in enumerate(cache_index):
        frame_cache = load_frame_cache(src_cache_dir, item['sequence_id'], item['frame_idx'])
        boxes, keep_mask, bev_area = build_keep_mask(
            frame_cache,
            keep_labels=args.keep_labels,
            min_area=args.min_area,
            max_area=args.max_area,
        )
        filtered_cache = filter_frame_cache(frame_cache, keep_mask)
        rel_path = save_frame_cache(dst_cache_dir, filtered_cache)
        new_index.append({
            'sequence_id': filtered_cache['sequence_id'],
            'frame_id': filtered_cache['frame_id'],
            'frame_idx': filtered_cache['frame_idx'],
            'cache_path': str(rel_path),
        })

        total_before += int(boxes.shape[0])
        total_after += int(keep_mask.sum())

        if args.verbose_every > 0 and (idx + 1) % args.verbose_every == 0:
            print(
                f'processed {idx + 1}/{len(cache_index)} frames | '
                f'boxes {total_before} -> {total_after} | '
                f'last frame kept {int(keep_mask.sum())}/{int(boxes.shape[0])}'
            )

    save_cache_index(dst_cache_dir, new_index)

    kept_ratio = 0.0 if total_before == 0 else total_after / total_before
    print('================ Filter Summary ================')
    print(f'src_cache_dir: {src_cache_dir}')
    print(f'dst_cache_dir: {dst_cache_dir}')
    print(f'frames: {len(cache_index)}')
    print(f'boxes_before: {total_before}')
    print(f'boxes_after: {total_after}')
    print(f'keep_ratio: {kept_ratio:.4f}')
    print(f'keep_labels: {args.keep_labels}')
    print(f'min_area: {args.min_area}')
    print(f'max_area: {args.max_area}')


if __name__ == '__main__':
    main()

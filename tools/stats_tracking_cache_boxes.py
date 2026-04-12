import _init_path
import argparse
from pathlib import Path

import numpy as np

from pcdet.tracking.cache import load_cache_index, load_frame_cache


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize box stats from tracking detector cache')
    parser.add_argument('--cache_dir', type=str, required=True, help='cache root')
    parser.add_argument(
        '--labels',
        type=int,
        nargs='*',
        default=None,
        help='only include these label ids, e.g. 1 for Car',
    )
    parser.add_argument('--score_thresh', type=float, default=None, help='optional score threshold')
    parser.add_argument('--verbose_every', type=int, default=200, help='print progress every N frames')
    return parser.parse_args()


def describe_array(name, arr):
    if arr.size == 0:
        print(f'{name}: empty')
        return
    percentiles = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(
        f'{name}: '
        f'min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}'
    )
    print(
        f'{name} percentiles: '
        f'p01={percentiles[0]:.4f}, p05={percentiles[1]:.4f}, p10={percentiles[2]:.4f}, '
        f'p25={percentiles[3]:.4f}, p50={percentiles[4]:.4f}, p75={percentiles[5]:.4f}, '
        f'p90={percentiles[6]:.4f}, p95={percentiles[7]:.4f}, p99={percentiles[8]:.4f}'
    )


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_index = load_cache_index(cache_dir)

    all_scores = []
    all_dx = []
    all_dy = []
    all_dz = []
    all_area = []
    label_counter = {}

    for idx, item in enumerate(cache_index):
        frame_cache = load_frame_cache(cache_dir, item['sequence_id'], item['frame_idx'])
        boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
        labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

        if boxes.shape[0] == 0:
            continue

        keep = np.ones((boxes.shape[0],), dtype=np.bool_)
        if args.labels is not None and len(args.labels) > 0:
            keep &= np.isin(labels, np.asarray(args.labels, dtype=np.int64))
        if args.score_thresh is not None:
            keep &= scores >= float(args.score_thresh)

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.shape[0] == 0:
            continue

        for label in labels.tolist():
            label_counter[label] = label_counter.get(label, 0) + 1

        all_scores.append(scores)
        all_dx.append(boxes[:, 3])
        all_dy.append(boxes[:, 4])
        all_dz.append(boxes[:, 5])
        all_area.append(boxes[:, 3] * boxes[:, 4])

        if args.verbose_every > 0 and (idx + 1) % args.verbose_every == 0:
            print(f'processed {idx + 1}/{len(cache_index)} frames')

    if len(all_dx) == 0:
        print('No boxes matched the filters.')
        return

    scores = np.concatenate(all_scores, axis=0)
    dx = np.concatenate(all_dx, axis=0)
    dy = np.concatenate(all_dy, axis=0)
    dz = np.concatenate(all_dz, axis=0)
    area = np.concatenate(all_area, axis=0)

    print('================ Cache Box Stats ================')
    print(f'cache_dir: {cache_dir}')
    print(f'frames: {len(cache_index)}')
    print(f'num_boxes: {dx.shape[0]}')
    print(f'labels filter: {args.labels}')
    print(f'score_thresh: {args.score_thresh}')
    print(f'label_counts: {label_counter}')

    describe_array('score', scores)
    describe_array('dx', dx)
    describe_array('dy', dy)
    describe_array('dz', dz)
    describe_array('bev_area', area)


if __name__ == '__main__':
    main()

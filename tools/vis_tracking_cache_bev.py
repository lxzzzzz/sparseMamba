import _init_path
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pcdet.tracking.cache import load_frame_cache
from pcdet.utils import box_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize one tracking cache frame in BEV')
    parser.add_argument('--data_root', type=str, required=True, help='tracking dataset root')
    parser.add_argument('--cache_dir', type=str, required=True, help='detector cache root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument(
        '--tracking_info',
        type=str,
        default=None,
        help='tracking infos pickle, defaults to <data_root>/tracking_infos_<split>.pkl',
    )
    parser.add_argument('--sequence_id', type=str, required=True, help='sequence id')
    parser.add_argument('--frame_idx', type=int, required=True, help='frame index inside the sequence')
    parser.add_argument('--score_thresh', type=float, default=0.1, help='prediction score threshold')
    parser.add_argument(
        '--point_cloud_range',
        type=float,
        nargs=6,
        default=[0.0, -51.2, -5.0, 204.8, 51.2, 3.0],
        help='minx miny minz maxx maxy maxz',
    )
    parser.add_argument('--fig_w', type=float, default=16.0)
    parser.add_argument('--fig_h', type=float, default=8.0)
    parser.add_argument('--point_size', type=float, default=0.3)
    parser.add_argument('--draw_gt', action='store_true', help='draw gt boxes from tracking info')
    parser.add_argument('--out', type=str, required=True, help='output png path')
    return parser.parse_args()


def load_tracking_infos(info_path):
    with open(info_path, 'rb') as f:
        return pickle.load(f)


def find_frame_info(tracking_infos, sequence_id, frame_idx):
    for info in tracking_infos:
        if str(info['sequence_id']) == str(sequence_id) and int(info['frame_idx']) == int(frame_idx):
            return info
    raise KeyError(f'Cannot find frame: sequence_id={sequence_id}, frame_idx={frame_idx}')


def load_points(data_root, info):
    lidar_rel = info['point_cloud']['lidar_path']
    num_features = int(info['point_cloud'].get('num_features', 4))
    lidar_path = Path(data_root) / lidar_rel
    return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, num_features)


def filter_points_by_range(points, point_cloud_range):
    pc_range = np.asarray(point_cloud_range, dtype=np.float32)
    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    )
    return points[mask]


def bev_corners(boxes_lidar):
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    corners3d = box_utils.boxes_to_corners_3d(boxes_lidar)
    return corners3d[:, :4, :2].astype(np.float32)


def draw_boxes(ax, boxes_lidar, color, linewidth=1.5, labels=None):
    corners = bev_corners(boxes_lidar)
    for i in range(corners.shape[0]):
        poly = np.concatenate([corners[i], corners[i, :1]], axis=0)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth)
        front = corners[i, [0, 3]].mean(axis=0)
        center = corners[i].mean(axis=0)
        ax.plot([center[0], front[0]], [center[1], front[1]], color=color, linewidth=linewidth)
        if labels is not None:
            ax.text(center[0], center[1], labels[i], color=color, fontsize=7)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    tracking_info_path = Path(args.tracking_info) if args.tracking_info else data_root / f'tracking_infos_{args.split}.pkl'
    tracking_infos = load_tracking_infos(tracking_info_path)
    info = find_frame_info(tracking_infos, args.sequence_id, args.frame_idx)

    points = load_points(data_root, info)
    points = filter_points_by_range(points, args.point_cloud_range)

    frame_cache = load_frame_cache(args.cache_dir, args.sequence_id, args.frame_idx)
    pred_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
    pred_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
    pred_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

    keep = pred_scores >= args.score_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h), dpi=180)
    ax.scatter(points[:, 0], points[:, 1], s=args.point_size, c=points[:, 2], cmap='viridis', alpha=0.8, linewidths=0)

    pred_text = [f'{int(lbl)}:{score:.2f}' for lbl, score in zip(pred_labels, pred_scores)]
    draw_boxes(ax, pred_boxes, color='red', linewidth=1.4, labels=pred_text)

    if args.draw_gt and 'annos' in info and 'gt_boxes_lidar' in info['annos']:
        gt_boxes = np.asarray(info['annos']['gt_boxes_lidar'], dtype=np.float32).reshape(-1, 7)
        draw_boxes(ax, gt_boxes, color='lime', linewidth=1.1)

    pc_range = np.asarray(args.point_cloud_range, dtype=np.float32)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(
        f'Seq {args.sequence_id} Frame {args.frame_idx} | '
        f'points={points.shape[0]} pred={pred_boxes.shape[0]} score>={args.score_thresh}'
    )
    ax.grid(True, linestyle='--', alpha=0.25)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f'Saved BEV image to: {out_path}')


if __name__ == '__main__':
    main()

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
    parser = argparse.ArgumentParser(description='Visualize two consecutive tracking frames in BEV for tuning association')
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
    parser.add_argument('--frame_idx', type=int, required=True, help='first frame index')
    parser.add_argument(
        '--next_frame_idx',
        type=int,
        default=None,
        help='second frame index, defaults to frame_idx + 1',
    )
    parser.add_argument('--score_thresh', type=float, default=0.1, help='detector score threshold')
    parser.add_argument(
        '--point_cloud_range',
        type=float,
        nargs=6,
        default=[0.0, -51.2, -5.0, 204.8, 51.2, 3.0],
        help='minx miny minz maxx maxy maxz',
    )
    parser.add_argument('--fig_w', type=float, default=16.0)
    parser.add_argument('--fig_h', type=float, default=9.0)
    parser.add_argument('--point_size', type=float, default=0.25)
    parser.add_argument('--draw_gt', action='store_true', help='draw GT boxes and GT track links')
    parser.add_argument('--draw_det', action='store_true', help='draw detector boxes from cache', default=True)
    parser.add_argument(
        '--tracker_results',
        type=str,
        default=None,
        help='optional tracker results pickle from eval_ab3dmot_baseline.py or eval_udca_policy.py',
    )
    parser.add_argument(
        '--tracker_score_thresh',
        type=float,
        default=0.0,
        help='optional score threshold for tracker outputs',
    )
    parser.add_argument(
        '--annotate_ids',
        action='store_true',
        help='draw GT / tracker ids near box centers',
    )
    parser.add_argument(
        '--draw_points_both',
        action='store_true',
        help='overlay both frame point clouds instead of only the first frame',
    )
    parser.add_argument('--out', type=str, required=True, help='output png path')
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_tracking_infos(info_path):
    return load_pickle(info_path)


def load_tracker_results(path):
    if path is None:
        return None
    return load_pickle(path)


def find_frame_info(tracking_infos, sequence_id, frame_idx):
    for info in tracking_infos:
        if str(info['sequence_id']) == str(sequence_id) and int(info['frame_idx']) == int(frame_idx):
            return info
    raise KeyError(f'Cannot find frame: sequence_id={sequence_id}, frame_idx={frame_idx}')


def find_tracker_frame(tracker_results, sequence_id, frame_idx):
    if tracker_results is None:
        return None
    seq_results = tracker_results.get(str(sequence_id), None)
    if seq_results is None:
        return None
    for item in seq_results:
        if int(item['frame_idx']) == int(frame_idx):
            return item
    return None


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


def box_centers(boxes_lidar):
    boxes_lidar = np.asarray(boxes_lidar, dtype=np.float32).reshape(-1, 7)
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return boxes_lidar[:, 0:2]


def draw_boxes(ax, boxes_lidar, color, linewidth=1.5, labels=None, linestyle='-'):
    corners = bev_corners(boxes_lidar)
    for i in range(corners.shape[0]):
        poly = np.concatenate([corners[i], corners[i, :1]], axis=0)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth, linestyle=linestyle)
        front = corners[i, [0, 3]].mean(axis=0)
        center = corners[i].mean(axis=0)
        ax.plot([center[0], front[0]], [center[1], front[1]], color=color, linewidth=linewidth, linestyle=linestyle)
        if labels is not None:
            ax.text(center[0], center[1], labels[i], color=color, fontsize=7)


def extract_gt(info):
    annos = info.get('annos', {})
    boxes = np.asarray(annos.get('gt_boxes_lidar', []), dtype=np.float32).reshape(-1, 7)
    track_ids = np.asarray(annos.get('track_id', []), dtype=np.int64)
    names = np.asarray(annos.get('name', []))
    return boxes, track_ids, names


def extract_det(cache, score_thresh):
    boxes = np.asarray(cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
    labels = np.asarray(cache.get('pred_labels', []), dtype=np.int64).reshape(-1)
    keep = scores >= float(score_thresh)
    return boxes[keep], scores[keep], labels[keep]


def extract_tracks(frame_result, score_thresh):
    if frame_result is None:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    tracks = frame_result.get('tracks', [])
    boxes = []
    track_ids = []
    scores = []
    for item in tracks:
        score = float(item.get('pred_score', 0.0))
        if score < float(score_thresh):
            continue
        boxes.append(np.asarray(item['pred_box'], dtype=np.float32))
        track_ids.append(int(item['track_id']))
        scores.append(score)
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.stack(boxes, axis=0), np.asarray(track_ids, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def draw_links(ax, centers_a, ids_a, centers_b, ids_b, color, linewidth=1.2, alpha=0.9):
    id_to_center_b = {int(track_id): centers_b[idx] for idx, track_id in enumerate(ids_b.tolist())}
    common = 0
    for idx, track_id in enumerate(ids_a.tolist()):
        center_b = id_to_center_b.get(int(track_id), None)
        if center_b is None:
            continue
        center_a = centers_a[idx]
        ax.plot(
            [center_a[0], center_b[0]],
            [center_a[1], center_b[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        common += 1
    return common


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    tracking_info_path = Path(args.tracking_info) if args.tracking_info else data_root / f'tracking_infos_{args.split}.pkl'
    next_frame_idx = int(args.next_frame_idx) if args.next_frame_idx is not None else int(args.frame_idx) + 1

    tracking_infos = load_tracking_infos(tracking_info_path)
    tracker_results = load_tracker_results(args.tracker_results)

    info_a = find_frame_info(tracking_infos, args.sequence_id, args.frame_idx)
    info_b = find_frame_info(tracking_infos, args.sequence_id, next_frame_idx)

    points_a = filter_points_by_range(load_points(data_root, info_a), args.point_cloud_range)
    points_b = filter_points_by_range(load_points(data_root, info_b), args.point_cloud_range)

    cache_a = load_frame_cache(args.cache_dir, args.sequence_id, args.frame_idx)
    cache_b = load_frame_cache(args.cache_dir, args.sequence_id, next_frame_idx)

    det_boxes_a, det_scores_a, det_labels_a = extract_det(cache_a, args.score_thresh)
    det_boxes_b, det_scores_b, det_labels_b = extract_det(cache_b, args.score_thresh)

    gt_boxes_a, gt_ids_a, gt_names_a = extract_gt(info_a)
    gt_boxes_b, gt_ids_b, gt_names_b = extract_gt(info_b)

    tracker_frame_a = find_tracker_frame(tracker_results, args.sequence_id, args.frame_idx)
    tracker_frame_b = find_tracker_frame(tracker_results, args.sequence_id, next_frame_idx)
    track_boxes_a, track_ids_a, track_scores_a = extract_tracks(tracker_frame_a, args.tracker_score_thresh)
    track_boxes_b, track_ids_b, track_scores_b = extract_tracks(tracker_frame_b, args.tracker_score_thresh)

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h), dpi=180)
    ax.scatter(points_a[:, 0], points_a[:, 1], s=args.point_size, c='0.7', alpha=0.35, linewidths=0, label=f'points t={args.frame_idx}')
    if args.draw_points_both:
        ax.scatter(points_b[:, 0], points_b[:, 1], s=args.point_size, c='tab:blue', alpha=0.20, linewidths=0, label=f'points t={next_frame_idx}')

    if args.draw_det:
        det_text_a = [f'A:{score:.2f}' for score in det_scores_a.tolist()]
        det_text_b = [f'B:{score:.2f}' for score in det_scores_b.tolist()]
        draw_boxes(ax, det_boxes_a, color='red', linewidth=1.2, labels=det_text_a if args.annotate_ids else None)
        draw_boxes(ax, det_boxes_b, color='orange', linewidth=1.2, labels=det_text_b if args.annotate_ids else None, linestyle='--')

    gt_link_count = 0
    if args.draw_gt:
        gt_labels_a = [f'gt:{int(track_id)}' if args.annotate_ids else None for track_id in gt_ids_a.tolist()]
        gt_labels_b = [f'gt:{int(track_id)}' if args.annotate_ids else None for track_id in gt_ids_b.tolist()]
        draw_boxes(ax, gt_boxes_a, color='lime', linewidth=1.3, labels=gt_labels_a if args.annotate_ids else None)
        draw_boxes(ax, gt_boxes_b, color='cyan', linewidth=1.3, labels=gt_labels_b if args.annotate_ids else None, linestyle='--')
        gt_link_count = draw_links(ax, box_centers(gt_boxes_a), gt_ids_a, box_centers(gt_boxes_b), gt_ids_b, color='deepskyblue', linewidth=1.0, alpha=0.8)

    track_link_count = 0
    if tracker_results is not None:
        track_labels_a = [f'id:{int(track_id)}' if args.annotate_ids else None for track_id in track_ids_a.tolist()]
        track_labels_b = [f'id:{int(track_id)}' if args.annotate_ids else None for track_id in track_ids_b.tolist()]
        draw_boxes(ax, track_boxes_a, color='magenta', linewidth=1.0, labels=track_labels_a if args.annotate_ids else None)
        draw_boxes(ax, track_boxes_b, color='purple', linewidth=1.0, labels=track_labels_b if args.annotate_ids else None, linestyle='--')
        track_link_count = draw_links(ax, box_centers(track_boxes_a), track_ids_a, box_centers(track_boxes_b), track_ids_b, color='magenta', linewidth=1.0, alpha=0.7)

    pc_range = np.asarray(args.point_cloud_range, dtype=np.float32)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.25)

    title_parts = [
        f'Seq {args.sequence_id}',
        f'frames {args.frame_idx}->{next_frame_idx}',
        f'det_t={det_boxes_a.shape[0]}',
        f'det_t1={det_boxes_b.shape[0]}',
    ]
    if args.draw_gt:
        title_parts.append(f'gt_links={gt_link_count}')
    if tracker_results is not None:
        title_parts.append(f'track_links={track_link_count}')
    title_parts.append(f'score>={args.score_thresh}')
    ax.set_title(' | '.join(title_parts))

    legend_handles = [
        plt.Line2D([0], [0], color='red', lw=1.2, label=f'det frame {args.frame_idx}'),
        plt.Line2D([0], [0], color='orange', lw=1.2, linestyle='--', label=f'det frame {next_frame_idx}'),
    ]
    if args.draw_gt:
        legend_handles.extend([
            plt.Line2D([0], [0], color='lime', lw=1.3, label=f'gt frame {args.frame_idx}'),
            plt.Line2D([0], [0], color='cyan', lw=1.3, linestyle='--', label=f'gt frame {next_frame_idx}'),
            plt.Line2D([0], [0], color='deepskyblue', lw=1.0, label='gt track link'),
        ])
    if tracker_results is not None:
        legend_handles.extend([
            plt.Line2D([0], [0], color='magenta', lw=1.0, label=f'track frame {args.frame_idx}'),
            plt.Line2D([0], [0], color='purple', lw=1.0, linestyle='--', label=f'track frame {next_frame_idx}'),
            plt.Line2D([0], [0], color='magenta', lw=1.0, label='tracker id link'),
        ])
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f'Saved pair BEV image to: {out_path}')


if __name__ == '__main__':
    main()

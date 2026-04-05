import _init_path
import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d

from pcdet.tracking.cache import load_frame_cache


GT_CLASS_COLORS = {
    'Car': (0.0, 1.0, 0.0),
    'Pedestrian': (0.0, 1.0, 1.0),
    'Cyclist': (1.0, 1.0, 0.0),
}

PRED_CLASS_COLORS = {
    1: (1.0, 0.3, 0.3),
    2: (1.0, 0.5, 1.0),
    3: (1.0, 0.7, 0.2),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tracking cache predictions against GT')
    parser.add_argument('--data_root', type=str, required=True, help='tracking dataset root')
    parser.add_argument('--cache_dir', type=str, required=True, help='tracking cache directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='which info split to load')
    parser.add_argument('--sequence_id', type=str, default=None, help='optional sequence id filter')
    parser.add_argument('--frame_idx', type=int, default=None, help='optional frame idx filter')
    parser.add_argument('--frame_id', type=str, default=None, help='optional frame id filter')
    parser.add_argument('--score_thresh', type=float, default=0.1, help='prediction score threshold')
    parser.add_argument('--max_frames', type=int, default=50, help='maximum number of frames to visualize when iterating')
    return parser.parse_args()


def load_infos(data_root, split):
    info_path = Path(data_root) / f'tracking_infos_{split}.pkl'
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    infos.sort(key=lambda info: (str(info['sequence_id']), int(info['frame_idx'])))
    return infos


def filter_infos(infos, args):
    selected = []
    for info in infos:
        if args.sequence_id is not None and str(info['sequence_id']) != str(args.sequence_id):
            continue
        if args.frame_idx is not None and int(info['frame_idx']) != int(args.frame_idx):
            continue
        if args.frame_id is not None and str(info['frame_id']) != str(args.frame_id):
            continue
        selected.append(info)
    if args.frame_idx is None and args.frame_id is None:
        selected = selected[:args.max_frames]
    return selected


def load_points(data_root, info):
    lidar_rel = info['point_cloud']['lidar_path']
    num_features = int(info['point_cloud'].get('num_features', 4))
    lidar_path = Path(data_root) / lidar_rel
    return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, num_features)


def make_point_cloud(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:, :3])
    if points.shape[0] > 0:
        z = points[:, 2]
        min_z, max_z = np.percentile(z, 1), np.percentile(z, 99)
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
        norm_z = np.clip((z - min_z) / (max_z - min_z + 1e-6), 0.0, 1.0)
        colors[:, 0] = 0.45 * norm_z
        colors[:, 1] = 0.45 * norm_z
        colors[:, 2] = 0.80
        pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


def box_to_lineset(box, color):
    center = box[0:3]
    size = box[3:6]
    axis_angles = np.array([0.0, 0.0, box[6]], dtype=np.float32)
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, size)
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    line_set.paint_uniform_color(color)
    return line_set


def draw_frame(points, gt_boxes, gt_names, pred_boxes, pred_labels, pred_scores, title):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1440, height=900)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)
    vis.add_geometry(make_point_cloud(points))

    for idx in range(gt_boxes.shape[0]):
        class_name = str(gt_names[idx])
        color = GT_CLASS_COLORS.get(class_name, (0.6, 0.6, 0.6))
        vis.add_geometry(box_to_lineset(gt_boxes[idx], color))

    for idx in range(pred_boxes.shape[0]):
        color = PRED_CLASS_COLORS.get(int(pred_labels[idx]), (1.0, 1.0, 1.0))
        vis.add_geometry(box_to_lineset(pred_boxes[idx], color))

    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()
    infos = load_infos(args.data_root, args.split)
    infos = filter_infos(infos, args)

    if len(infos) == 0:
        raise ValueError('No frames matched the provided filters')

    print('Ground-truth colors: Car=green, Pedestrian=cyan, Cyclist=yellow')
    print('Prediction colors: Car=red, Pedestrian=magenta, Cyclist=orange')
    print("Press 'Q' in the Open3D window to move to the next frame.")

    for info in infos:
        points = load_points(args.data_root, info)
        annos = info.get('annos', {})
        gt_boxes = np.asarray(annos.get('gt_boxes_lidar', []), dtype=np.float32).reshape(-1, 7)
        gt_names = np.asarray(annos.get('name', []))

        frame_cache = load_frame_cache(args.cache_dir, info['sequence_id'], info['frame_idx'])
        pred_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        pred_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64)
        pred_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32)

        keep = pred_scores >= float(args.score_thresh)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        title = (
            f"cache vis | split={args.split} seq={info['sequence_id']} "
            f"frame_idx={info['frame_idx']} frame_id={info['frame_id']} "
            f"gt={gt_boxes.shape[0]} pred={pred_boxes.shape[0]} score>={args.score_thresh:.2f}"
        )
        print(title)
        draw_frame(points, gt_boxes, gt_names, pred_boxes, pred_labels, pred_scores, title)


if __name__ == '__main__':
    main()

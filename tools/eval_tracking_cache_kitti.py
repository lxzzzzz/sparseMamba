import _init_path
import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
from pcdet.tracking.cache import load_frame_cache
from pcdet.utils import box_utils, calibration_kitti

from create_tracking_infos import get_frame_annos


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate tracking detector cache with KITTI metrics')
    parser.add_argument('--data_root', type=str, required=True, help='KITTI tracking dataset root')
    parser.add_argument('--cache_dir', type=str, required=True, help='detector cache root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='split name')
    parser.add_argument(
        '--tracking_info',
        type=str,
        default=None,
        help='tracking infos pickle, defaults to <data_root>/tracking_infos_<split>.pkl',
    )
    parser.add_argument(
        '--format_cfg',
        type=str,
        default='tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_sequence.yaml',
        help='tracking layout config used to recover GT from raw labels',
    )
    parser.add_argument(
        '--class_names',
        nargs='+',
        default=['Car', 'Pedestrian', 'Cyclist'],
        help='classes to evaluate',
    )
    parser.add_argument(
        '--dummy_bbox_height',
        type=float,
        default=50.0,
        help='2D bbox height used when --use_dummy_bbox is enabled',
    )
    parser.add_argument(
        '--use_dummy_bbox',
        action='store_true',
        default=True,
        help='ignore projected 2D bbox and use a fixed-size dummy bbox for dt annotations',
    )
    parser.add_argument(
        '--project_bbox',
        dest='use_dummy_bbox',
        action='store_false',
        help='project 3D boxes to image to build dt bbox; use only if calibration is trustworthy',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='optional output directory to save det_annos/result text',
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_tracking_infos(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_gt_annos(tracking_infos, data_root, format_cfg, class_names):
    gt_annos = []
    cache = {}
    class_name_set = set(class_names)

    for info in tracking_infos:
        sequence_id = str(info['sequence_id'])
        frame_id = str(info['frame_id'])
        frame_idx = int(info['frame_idx'])
        frame_annos = get_frame_annos(data_root, sequence_id, frame_id, frame_idx, format_cfg, cache)

        filtered = [anno for anno in frame_annos if anno['name'] in class_name_set]
        if len(filtered) == 0:
            gt_annos.append({
                'name': np.zeros((0,), dtype='<U1'),
                'truncated': np.zeros((0,), dtype=np.float32),
                'occluded': np.zeros((0,), dtype=np.int32),
                'alpha': np.zeros((0,), dtype=np.float32),
                'bbox': np.zeros((0, 4), dtype=np.float32),
                'dimensions': np.zeros((0, 3), dtype=np.float32),
                'location': np.zeros((0, 3), dtype=np.float32),
                'rotation_y': np.zeros((0,), dtype=np.float32),
                'score': np.zeros((0,), dtype=np.float32),
            })
            continue

        gt_annos.append({
            'name': np.asarray([anno['name'] for anno in filtered]),
            'truncated': np.asarray([anno['truncated'] for anno in filtered], dtype=np.float32),
            'occluded': np.asarray([anno['occluded'] for anno in filtered], dtype=np.int32),
            'alpha': np.asarray([anno['alpha'] for anno in filtered], dtype=np.float32),
            'bbox': np.stack([anno['bbox'] for anno in filtered], axis=0).astype(np.float32),
            'dimensions': np.stack([anno['dimensions'] for anno in filtered], axis=0).astype(np.float32),
            'location': np.stack([anno['location'] for anno in filtered], axis=0).astype(np.float32),
            'rotation_y': np.asarray([anno['rotation_y'] for anno in filtered], dtype=np.float32),
            'score': np.asarray([anno.get('score', -1.0) for anno in filtered], dtype=np.float32),
        })

    return gt_annos


def build_dummy_bbox(num_boxes, height):
    if num_boxes == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.tile(np.asarray([0.0, 0.0, height, height], dtype=np.float32), (num_boxes, 1))


def build_dt_annos(tracking_infos, cache_root, class_names, use_dummy_bbox, dummy_bbox_height):
    dt_annos = []

    for info in tracking_infos:
        sequence_id = str(info['sequence_id'])
        frame_idx = int(info['frame_idx'])
        image_shape = np.asarray(info['image']['image_shape'], dtype=np.int32)
        calib = calibration_kitti.Calibration(info['calib'])
        frame_cache = load_frame_cache(cache_root, sequence_id, frame_idx)

        pred_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        pred_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
        pred_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

        valid_mask = (pred_labels > 0) & (pred_labels <= len(class_names))
        pred_boxes = pred_boxes[valid_mask]
        pred_scores = pred_scores[valid_mask]
        pred_labels = pred_labels[valid_mask]

        if pred_boxes.shape[0] == 0:
            dt_annos.append({
                'name': np.zeros((0,), dtype='<U1'),
                'truncated': np.zeros((0,), dtype=np.float32),
                'occluded': np.zeros((0,), dtype=np.int32),
                'alpha': np.zeros((0,), dtype=np.float32),
                'bbox': np.zeros((0, 4), dtype=np.float32),
                'dimensions': np.zeros((0, 3), dtype=np.float32),
                'location': np.zeros((0, 3), dtype=np.float32),
                'rotation_y': np.zeros((0,), dtype=np.float32),
                'score': np.zeros((0,), dtype=np.float32),
                'boxes_lidar': np.zeros((0, 7), dtype=np.float32),
            })
            continue

        pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        if use_dummy_bbox:
            pred_bbox = build_dummy_bbox(pred_boxes.shape[0], dummy_bbox_height)
        else:
            pred_bbox = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            ).astype(np.float32)

        pred_names = np.asarray(class_names)[pred_labels - 1]
        alpha = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]

        dt_annos.append({
            'name': pred_names,
            'truncated': np.zeros((pred_boxes.shape[0],), dtype=np.float32),
            'occluded': np.zeros((pred_boxes.shape[0],), dtype=np.int32),
            'alpha': alpha.astype(np.float32),
            'bbox': pred_bbox,
            'dimensions': pred_boxes_camera[:, 3:6].astype(np.float32),
            'location': pred_boxes_camera[:, 0:3].astype(np.float32),
            'rotation_y': pred_boxes_camera[:, 6].astype(np.float32),
            'score': pred_scores.astype(np.float32),
            'boxes_lidar': pred_boxes.astype(np.float32),
        })

    return dt_annos


def keep_bev_3d_lines(result_str):
    kept = []
    for line in result_str.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if 'AP@' in stripped or stripped.startswith('bev  AP:') or stripped.startswith('3d   AP:'):
            kept.append(line)
    return '\n'.join(kept)


def keep_bev_3d_metrics(result_dict):
    filtered = {}
    for key, value in result_dict.items():
        if '_bev/' in key or '_3d/' in key:
            filtered[key] = value
    return filtered


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_dir)
    tracking_info_path = Path(args.tracking_info) if args.tracking_info else data_root / f'tracking_infos_{args.split}.pkl'
    format_cfg = load_yaml(args.format_cfg)
    tracking_infos = load_tracking_infos(tracking_info_path)

    gt_annos = build_gt_annos(tracking_infos, data_root, format_cfg, args.class_names)
    dt_annos = build_dt_annos(
        tracking_infos=tracking_infos,
        cache_root=cache_root,
        class_names=args.class_names,
        use_dummy_bbox=args.use_dummy_bbox,
        dummy_bbox_height=args.dummy_bbox_height,
    )

    result_str, result_dict = get_official_eval_result(gt_annos, dt_annos, args.class_names)
    bev_3d_str = keep_bev_3d_lines(result_str)
    bev_3d_dict = keep_bev_3d_metrics(result_dict)

    print('================ KITTI BEV/3D Evaluation ================')
    print(f'split: {args.split}')
    print(f'tracking_info: {tracking_info_path}')
    print(f'cache_dir: {cache_root}')
    print(f'use_dummy_bbox: {args.use_dummy_bbox}')
    print(bev_3d_str)

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'cache_eval_{args.split}_det_annos.pkl', 'wb') as f:
            pickle.dump(dt_annos, f)
        with open(save_dir / f'cache_eval_{args.split}_gt_annos.pkl', 'wb') as f:
            pickle.dump(gt_annos, f)
        with open(save_dir / f'cache_eval_{args.split}_bev3d.txt', 'w') as f:
            f.write(bev_3d_str + '\n')
        with open(save_dir / f'cache_eval_{args.split}_bev3d_metrics.pkl', 'wb') as f:
            pickle.dump(bev_3d_dict, f)


if __name__ == '__main__':
    main()

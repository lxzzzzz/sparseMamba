import _init_path
import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets.kitti import kitti_utils
from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
from pcdet.tracking.cache import load_frame_cache
from pcdet.tracking.utils import filter_boxes_by_spatial_range, normalize_bev_range


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate tracking detector cache with KITTI BEV/3D metrics')
    parser.add_argument('--data_root', type=str, required=True, help='tracking dataset root')
    parser.add_argument('--cache_dir', type=str, required=True, help='detector cache root')
    parser.add_argument('--data_cfg', type=str, default=None, help='optional data/detector yaml used to resolve default BEV range')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument(
        '--tracking_info',
        type=str,
        default=None,
        help='tracking infos pickle, defaults to <data_root>/tracking_infos_<split>.pkl',
    )
    parser.add_argument('--gt_pkl', type=str, required=True, help='raw GT pkl containing lidar-frame labels')
    parser.add_argument('--class_names', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--gt_boxes_key', type=str, default='annos.gt_boxes_lidar')
    parser.add_argument('--gt_names_key', type=str, default='annos.name')
    parser.add_argument('--gt_sequence_key', type=str, default='sequence_id')
    parser.add_argument('--gt_frame_idx_key', type=str, default='frame_idx')
    parser.add_argument('--gt_frame_id_key', type=str, default='frame_id')
    parser.add_argument('--max_distance', type=float, default=100.0, help='evaluate targets within XY range only')
    parser.add_argument(
        '--bev_range',
        type=str,
        default=None,
        help='optional BEV range, e.g. "[-40.96, -28.16, 40.96, 28.16]"; overrides --max_distance when provided',
    )
    parser.add_argument('--save_dir', type=str, default=None)
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_by_path(data, key_path, default=None):
    cur = data
    for key in key_path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_sequence_id(value):
    return '' if value is None else str(value)


def normalize_frame_idx(value):
    if value is None:
        return None
    return int(value)


def normalize_frame_id(value):
    if value is None:
        return None
    return str(value)


def frame_key_from_info(info):
    return normalize_sequence_id(info.get('sequence_id')), normalize_frame_idx(info.get('frame_idx'))


def frame_key_from_gt_info(info, args):
    return (
        normalize_sequence_id(get_by_path(info, args.gt_sequence_key)),
        normalize_frame_idx(get_by_path(info, args.gt_frame_idx_key)),
    )


def frame_id_key_from_gt_info(info, args):
    return (
        normalize_sequence_id(get_by_path(info, args.gt_sequence_key)),
        normalize_frame_id(get_by_path(info, args.gt_frame_id_key)),
    )


def build_gt_index(gt_infos, args):
    by_frame_idx = {}
    by_frame_id = {}
    for info in gt_infos:
        key_idx = frame_key_from_gt_info(info, args)
        key_id = frame_id_key_from_gt_info(info, args)
        if key_idx[1] is not None:
            by_frame_idx[key_idx] = info
        if key_id[1] is not None:
            by_frame_id[key_id] = info
    return by_frame_idx, by_frame_id


def find_gt_info(info, gt_by_idx, gt_by_id):
    key_idx = frame_key_from_info(info)
    if key_idx in gt_by_idx:
        return gt_by_idx[key_idx]
    key_id = (normalize_sequence_id(info.get('sequence_id')), normalize_frame_id(info.get('frame_id')))
    if key_id in gt_by_id:
        return gt_by_id[key_id]
    raise KeyError(f'Cannot find GT for sequence_id={key_idx[0]}, frame_idx={key_idx[1]}, frame_id={info.get("frame_id")}')


def filter_class_mask(names, class_names):
    class_name_set = set(class_names)
    return np.asarray([name in class_name_set for name in names], dtype=np.bool_)


def build_empty_anno():
    return {
        'name': np.zeros((0,), dtype='<U1'),
        'gt_boxes_lidar': np.zeros((0, 7), dtype=np.float32),
    }


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


def build_gt_annos(tracking_infos, gt_infos, args, max_distance=None, bev_range=None):
    gt_by_idx, gt_by_id = build_gt_index(gt_infos, args)
    gt_annos = []

    for info in tracking_infos:
        gt_info = find_gt_info(info, gt_by_idx, gt_by_id)
        names = np.asarray(get_by_path(gt_info, args.gt_names_key, []))
        boxes = np.asarray(get_by_path(gt_info, args.gt_boxes_key, []), dtype=np.float32).reshape(-1, 7)

        if names.shape[0] != boxes.shape[0]:
            raise ValueError(
                f'GT name/box count mismatch for sequence_id={info.get("sequence_id")} frame_idx={info.get("frame_idx")}: '
                f'{names.shape[0]} names vs {boxes.shape[0]} boxes'
            )

        if boxes.shape[0] == 0:
            gt_annos.append(build_empty_anno())
            continue

        keep = filter_class_mask(names, args.class_names)
        filtered_boxes, filtered_names = filter_boxes_by_spatial_range(
            boxes[keep].astype(np.float32),
            names[keep],
            max_distance=max_distance,
            bev_range=bev_range,
        )
        gt_annos.append({
            'name': filtered_names,
            'gt_boxes_lidar': filtered_boxes,
        })

    return gt_annos


def build_dt_annos(tracking_infos, cache_root, class_names, max_distance=None, bev_range=None):
    dt_annos = []
    for info in tracking_infos:
        frame_cache = load_frame_cache(cache_root, info['sequence_id'], info['frame_idx'])
        pred_boxes = np.asarray(frame_cache.get('pred_boxes', []), dtype=np.float32).reshape(-1, 7)
        pred_scores = np.asarray(frame_cache.get('pred_scores', []), dtype=np.float32).reshape(-1)
        pred_labels = np.asarray(frame_cache.get('pred_labels', []), dtype=np.int64).reshape(-1)

        valid = (pred_labels > 0) & (pred_labels <= len(class_names))
        pred_boxes = pred_boxes[valid]
        pred_scores = pred_scores[valid]
        pred_labels = pred_labels[valid]
        pred_boxes, pred_scores, pred_labels = filter_boxes_by_spatial_range(
            pred_boxes, pred_scores, pred_labels, max_distance=max_distance, bev_range=bev_range
        )

        if pred_boxes.shape[0] == 0:
            dt_annos.append({
                'name': np.zeros((0,), dtype='<U1'),
                'boxes_lidar': np.zeros((0, 7), dtype=np.float32),
                'score': np.zeros((0,), dtype=np.float32),
            })
            continue

        dt_annos.append({
            'name': np.asarray(class_names)[pred_labels - 1],
            'boxes_lidar': pred_boxes.astype(np.float32),
            'score': pred_scores.astype(np.float32),
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
    return {k: v for k, v in result_dict.items() if '_bev/' in k or '_3d/' in k}


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_dir)
    tracking_info_path = Path(args.tracking_info) if args.tracking_info else data_root / f'tracking_infos_{args.split}.pkl'
    effective_max_distance, effective_bev_range = resolve_spatial_filter(args)

    tracking_infos = load_pickle(tracking_info_path)
    gt_infos = load_pickle(args.gt_pkl)

    gt_annos = build_gt_annos(tracking_infos, gt_infos, args, max_distance=effective_max_distance, bev_range=effective_bev_range)
    dt_annos = build_dt_annos(tracking_infos, cache_root, args.class_names, max_distance=effective_max_distance, bev_range=effective_bev_range)

    map_name_to_kitti = {name: name for name in args.class_names}
    eval_gt_annos = kitti_utils.transform_annotations_to_kitti_format(gt_annos, map_name_to_kitti=map_name_to_kitti)
    eval_dt_annos = kitti_utils.transform_annotations_to_kitti_format(dt_annos, map_name_to_kitti=map_name_to_kitti)

    result_str, result_dict = get_official_eval_result(eval_gt_annos, eval_dt_annos, args.class_names)
    bev_3d_str = keep_bev_3d_lines(result_str)
    bev_3d_dict = keep_bev_3d_metrics(result_dict)

    print('================ KITTI BEV/3D Evaluation ================')
    print(f'split: {args.split}')
    print(f'tracking_info: {tracking_info_path}')
    print(f'gt_pkl: {args.gt_pkl}')
    print(f'cache_dir: {cache_root}')
    print(f'max_distance: {effective_max_distance}')
    print(f'bev_range: {None if effective_bev_range is None else [float(v) for v in effective_bev_range.tolist()]}')
    print(bev_3d_str)

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'cache_eval_{args.split}_det_annos.pkl', 'wb') as f:
            pickle.dump(eval_dt_annos, f)
        with open(save_dir / f'cache_eval_{args.split}_gt_annos.pkl', 'wb') as f:
            pickle.dump(eval_gt_annos, f)
        with open(save_dir / f'cache_eval_{args.split}_bev3d.txt', 'w') as f:
            f.write(bev_3d_str + '\n')
        with open(save_dir / f'cache_eval_{args.split}_bev3d_metrics.pkl', 'wb') as f:
            pickle.dump(bev_3d_dict, f)


if __name__ == '__main__':
    main()

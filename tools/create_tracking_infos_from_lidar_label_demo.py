import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage import io

from pcdet.datasets.kitti import kitti_utils
from pcdet.utils import calibration_kitti

from create_tracking_infos import (
    find_image_file,
    find_sequence_dir,
    get_frame_calib,
    group_entries_by_sequence,
    infer_frame_idx,
    list_frame_ids_from_sequence_dir,
    load_format_cfg,
    parse_split_entries,
    resolve_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo: create tracking infos from KITTI-tracking labels whose 3D fields are already in lidar coordinates'
    )
    parser.add_argument('--data_path', type=str, required=True, help='tracking dataset root')
    parser.add_argument('--save_path', type=str, default=None, help='output directory, defaults to data_path')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument(
        '--format_cfg',
        type=str,
        default='tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_sequence.yaml',
        help='dataset layout config for indexing lidar/image/calib files',
    )
    parser.add_argument(
        '--label_layout',
        choices=['sequence', 'framewise'],
        default='sequence',
        help='sequence: training/label_02/<seq>.txt, framewise: training/label_02/<seq>/<frame>.txt',
    )
    parser.add_argument(
        '--with_score',
        action='store_true',
        help='set if each label line has an extra score field before track_id (framewise) or after yaw (sequence)',
    )
    parser.add_argument('--point_features', type=int, default=4)
    return parser.parse_args()


def parse_sequence_lidar_tracking_labels(label_file, with_score=False):
    frame_to_annos = defaultdict(list)
    if not label_file.exists():
        return frame_to_annos

    with open(label_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            min_len = 18 if with_score else 17
            if len(fields) < min_len:
                continue

            frame_idx = int(fields[0])
            track_id = int(fields[1])
            cls_name = fields[2]
            if cls_name == 'DontCare':
                continue

            score = float(fields[17]) if with_score and len(fields) > 17 else -1.0
            boxes_lidar = np.array([float(x) for x in fields[10:17]], dtype=np.float32)

            frame_to_annos[frame_idx].append({
                'track_id': track_id,
                'name': cls_name,
                'truncated': float(fields[3]),
                'occluded': int(float(fields[4])),
                'alpha': float(fields[5]),
                'bbox': np.array([float(x) for x in fields[6:10]], dtype=np.float32),
                'score': score,
                'gt_boxes_lidar': boxes_lidar,
            })

    return frame_to_annos


def parse_framewise_lidar_tracking_labels(label_file, with_score=False):
    frame_annos = []
    if not label_file.exists():
        return frame_annos

    with open(label_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            min_len = 16 if with_score else 15
            if len(fields) < min_len:
                continue

            cls_name = fields[0]
            if cls_name == 'DontCare':
                continue

            boxes_lidar = np.array([float(x) for x in fields[8:15]], dtype=np.float32)
            if with_score:
                score = float(fields[15])
                track_id = int(fields[16])
            else:
                score = -1.0
                track_id = int(fields[15])

            frame_annos.append({
                'track_id': track_id,
                'name': cls_name,
                'truncated': float(fields[1]),
                'occluded': int(float(fields[2])),
                'alpha': float(fields[3]),
                'bbox': np.array([float(x) for x in fields[4:8]], dtype=np.float32),
                'score': score,
                'gt_boxes_lidar': boxes_lidar,
            })

    return frame_annos


def build_annos_from_lidar_labels(frame_annos):
    if not frame_annos:
        return {
            'name': np.zeros((0,), dtype='<U1'),
            'track_id': np.zeros((0,), dtype=np.int64),
            'bbox': np.zeros((0, 4), dtype=np.float32),
            'dimensions': np.zeros((0, 3), dtype=np.float32),
            'location': np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.zeros((0,), dtype=np.float32),
            'alpha': np.zeros((0,), dtype=np.float32),
            'score': np.zeros((0,), dtype=np.float32),
            'gt_boxes_lidar': np.zeros((0, 7), dtype=np.float32),
        }

    names = np.asarray([anno['name'] for anno in frame_annos])
    track_ids = np.asarray([anno['track_id'] for anno in frame_annos], dtype=np.int64)
    bbox = np.stack([anno['bbox'] for anno in frame_annos], axis=0).astype(np.float32)
    alpha = np.asarray([anno['alpha'] for anno in frame_annos], dtype=np.float32)
    score = np.asarray([anno['score'] for anno in frame_annos], dtype=np.float32)
    gt_boxes_lidar = np.stack([anno['gt_boxes_lidar'] for anno in frame_annos], axis=0).astype(np.float32)

    pseudo_annos = [{
        'name': names.copy(),
        'gt_boxes_lidar': gt_boxes_lidar.copy(),
    }]
    kitti_utils.transform_annotations_to_kitti_format(
        pseudo_annos,
        map_name_to_kitti={str(name): str(name) for name in np.unique(names).tolist()}
    )
    pseudo = pseudo_annos[0]

    return {
        'name': names,
        'track_id': track_ids,
        'bbox': bbox,
        'dimensions': pseudo['dimensions'].astype(np.float32),
        'location': pseudo['location'].astype(np.float32),
        'rotation_y': pseudo['rotation_y'].astype(np.float32),
        'alpha': alpha if alpha.shape[0] == gt_boxes_lidar.shape[0] else pseudo['alpha'].astype(np.float32),
        'score': score,
        'gt_boxes_lidar': gt_boxes_lidar,
    }


def get_sequence_lidar_label_cache(root_path, sequence_id, format_cfg, cache, with_score):
    key = ('label_lidar', sequence_id)
    if key in cache:
        return cache[key]

    label_dir_candidates = format_cfg['PATHS']['LABEL_DIR_CANDIDATES']
    label_filename = f'{sequence_id}.txt'
    label_file = resolve_file(root_path, [str(Path(candidate) / label_filename) for candidate in label_dir_candidates])
    cache[key] = parse_sequence_lidar_tracking_labels(label_file, with_score=with_score)
    return cache[key]


def get_frame_lidar_annos(root_path, sequence_id, frame_id, frame_idx, format_cfg, cache, label_layout, with_score):
    label_dir_candidates = format_cfg['PATHS']['LABEL_DIR_CANDIDATES']

    if label_layout == 'sequence':
        labels_by_frame = get_sequence_lidar_label_cache(root_path, sequence_id, format_cfg, cache, with_score)
        return labels_by_frame.get(frame_idx, [])

    label_file = resolve_file(
        root_path,
        [str(Path(candidate) / sequence_id / f'{frame_id}.txt') for candidate in label_dir_candidates],
    )
    return parse_framewise_lidar_tracking_labels(label_file, with_score=with_score)


def collect_infos_for_split(root_path, split, format_cfg, args):
    entries = parse_split_entries(root_path, split, format_cfg)
    grouped_entries = group_entries_by_sequence(entries, format_cfg)
    split_infos = []
    cache = {}

    point_dir_candidates = format_cfg['PATHS']['POINT_CLOUD_DIR_CANDIDATES']
    image_dir_candidates = format_cfg['PATHS']['IMAGE_DIR_CANDIDATES']
    image_suffixes = format_cfg.get('IMAGE_SUFFIXES', ['.png', '.jpg', '.jpeg'])
    lidar_suffix = format_cfg.get('POINT_CLOUD_SUFFIX', '.bin')

    for sequence_id in sorted(grouped_entries.keys()):
        point_dir = find_sequence_dir(root_path, point_dir_candidates, sequence_id)
        image_dir = find_sequence_dir(root_path, image_dir_candidates, sequence_id)

        frame_ids = grouped_entries[sequence_id]
        if frame_ids is None:
            frame_ids = list_frame_ids_from_sequence_dir(point_dir, lidar_suffix)

        for fallback_idx, frame_id in enumerate(frame_ids):
            lidar_file = point_dir / f'{frame_id}{lidar_suffix}'
            if not lidar_file.exists():
                raise FileNotFoundError(f'Missing lidar for sequence {sequence_id}, frame {frame_id}')

            image_file = find_image_file(image_dir, frame_id, image_suffixes)
            image_shape = np.array(io.imread(image_file).shape[:2], dtype=np.int32)
            frame_idx = infer_frame_idx(frame_id, fallback_idx)
            calib_dict, _ = get_frame_calib(root_path, sequence_id, frame_id, format_cfg, cache)
            frame_annos = get_frame_lidar_annos(
                root_path, sequence_id, frame_id, frame_idx, format_cfg, cache, args.label_layout, args.with_score
            )
            annos = build_annos_from_lidar_labels(frame_annos)

            split_infos.append({
                'sequence_id': str(sequence_id),
                'frame_id': str(frame_id),
                'frame_idx': int(frame_idx),
                'point_cloud': {
                    'lidar_idx': f'{sequence_id}_{frame_id}',
                    'lidar_path': str(lidar_file.relative_to(root_path)),
                    'num_features': int(args.point_features),
                },
                'image': {
                    'image_idx': str(frame_id),
                    'image_path': str(image_file.relative_to(root_path)),
                    'image_shape': image_shape,
                },
                'calib': {
                    'P2': calib_dict['P2'],
                    'R0': calib_dict['R0'],
                    'Tr_velo2cam': calib_dict['Tr_velo2cam'],
                },
                'annos': annos,
            })

    split_infos.sort(key=lambda info: (info['sequence_id'], int(info['frame_idx'])))
    return split_infos


def main():
    args = parse_args()
    root_path = Path(args.data_path)
    save_path = Path(args.save_path) if args.save_path is not None else root_path
    save_path.mkdir(parents=True, exist_ok=True)

    format_cfg = load_format_cfg(args.format_cfg)

    for split in args.splits:
        infos = collect_infos_for_split(root_path, split, format_cfg, args)
        out_file = save_path / f'tracking_infos_{split}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump(infos, f)
        print(f'Saved {len(infos)} frame infos to {out_file}')


if __name__ == '__main__':
    main()

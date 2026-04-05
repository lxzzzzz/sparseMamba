import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from skimage import io

from pcdet.utils import box_utils, calibration_kitti


def parse_args():
    parser = argparse.ArgumentParser(description='Create generic KITTI-style tracking frame infos')
    parser.add_argument('--data_path', type=str, required=True, help='tracking dataset root')
    parser.add_argument('--save_path', type=str, default=None, help='output directory, defaults to data_path')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='split names under ImageSets')
    parser.add_argument(
        '--format_cfg',
        type=str,
        default='tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_sequence.yaml',
        help='dataset layout config for tracking info generation',
    )
    return parser.parse_args()


def load_format_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_dir(base_path, candidates):
    for candidate in candidates:
        path = base_path / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f'Unable to resolve any directory from: {candidates}')


def resolve_file(base_path, candidates):
    for candidate in candidates:
        path = base_path / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f'Unable to resolve any file from: {candidates}')


def infer_frame_idx(frame_id, fallback_idx):
    try:
        return int(frame_id)
    except ValueError:
        return int(fallback_idx)


def parse_split_entries(root_path, split, format_cfg):
    imagesets_dir = root_path / format_cfg.get('IMAGESETS_DIR', 'ImageSets')
    split_file = imagesets_dir / f'{split}.txt'
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def group_entries_by_sequence(entries, format_cfg):
    split_mode = format_cfg.get('SPLIT_ENTRY_MODE', 'sequence')
    entry_sep = format_cfg.get('ENTRY_SEPARATOR', '/')

    if split_mode == 'sequence':
        return {entry: None for entry in entries}

    if split_mode != 'frame':
        raise ValueError(f'Unsupported SPLIT_ENTRY_MODE: {split_mode}')

    grouped = defaultdict(list)
    for entry in entries:
        parts = entry.split(entry_sep)
        if len(parts) < 2:
            raise ValueError(f'Frame split entry must contain sequence and frame id: {entry}')
        sequence_id = parts[0]
        frame_id = parts[1]
        grouped[sequence_id].append(frame_id)

    for sequence_id in grouped:
        grouped[sequence_id] = sorted(set(grouped[sequence_id]))
    return dict(grouped)


def parse_sequence_tracking_labels(label_file):
    frame_to_annos = defaultdict(list)
    if not label_file.exists():
        return frame_to_annos

    with open(label_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) < 17:
                continue

            frame_idx = int(fields[0])
            track_id = int(fields[1])
            cls_name = fields[2]
            if cls_name == 'DontCare':
                continue

            frame_to_annos[frame_idx].append({
                'track_id': track_id,
                'name': cls_name,
                'truncated': float(fields[3]),
                'occluded': int(float(fields[4])),
                'alpha': float(fields[5]),
                'bbox': np.array([float(x) for x in fields[6:10]], dtype=np.float32),
                'dimensions': np.array([float(fields[12]), float(fields[10]), float(fields[11])], dtype=np.float32),
                'location': np.array([float(x) for x in fields[13:16]], dtype=np.float32),
                'rotation_y': float(fields[16]),
                'score': float(fields[17]) if len(fields) > 17 else -1.0,
            })
    return frame_to_annos


def parse_frame_detection_labels_with_track_id(label_file):
    frame_annos = []
    if not label_file.exists():
        return frame_annos

    with open(label_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) < 16:
                continue

            cls_name = fields[0]
            if cls_name == 'DontCare':
                continue

            # Supported frame-level label layouts:
            # 1) class truncated occluded alpha bbox h w l x y z ry track_id
            # 2) class truncated occluded alpha bbox h w l x y z ry score track_id
            has_score = len(fields) >= 17
            track_id = int(fields[16] if has_score else fields[15])
            frame_annos.append({
                'track_id': track_id,
                'name': cls_name,
                'truncated': float(fields[1]),
                'occluded': int(float(fields[2])),
                'alpha': float(fields[3]),
                'bbox': np.array([float(x) for x in fields[4:8]], dtype=np.float32),
                'dimensions': np.array([float(fields[10]), float(fields[8]), float(fields[9])], dtype=np.float32),
                'location': np.array([float(x) for x in fields[11:14]], dtype=np.float32),
                'rotation_y': float(fields[14]),
                'score': float(fields[15]) if has_score else -1.0,
            })
    return frame_annos


def build_annos(calib, frame_annos):
    if not frame_annos:
        return {
            'name': np.zeros((0,), dtype='<U1'),
            'track_id': np.zeros((0,), dtype=np.int64),
            'bbox': np.zeros((0, 4), dtype=np.float32),
            'dimensions': np.zeros((0, 3), dtype=np.float32),
            'location': np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.zeros((0,), dtype=np.float32),
            'gt_boxes_lidar': np.zeros((0, 7), dtype=np.float32),
        }

    names = np.asarray([anno['name'] for anno in frame_annos])
    track_ids = np.asarray([anno['track_id'] for anno in frame_annos], dtype=np.int64)
    bbox = np.stack([anno['bbox'] for anno in frame_annos], axis=0).astype(np.float32)
    dimensions = np.stack([anno['dimensions'] for anno in frame_annos], axis=0).astype(np.float32)
    location = np.stack([anno['location'] for anno in frame_annos], axis=0).astype(np.float32)
    rotation_y = np.asarray([anno['rotation_y'] for anno in frame_annos], dtype=np.float32)
    gt_boxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1).astype(np.float32)
    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

    return {
        'name': names,
        'track_id': track_ids,
        'bbox': bbox,
        'dimensions': dimensions,
        'location': location,
        'rotation_y': rotation_y,
        'gt_boxes_lidar': gt_boxes_lidar.astype(np.float32),
    }


def find_sequence_dir(root_path, dir_candidates, sequence_id):
    base_dir = resolve_dir(root_path, dir_candidates)
    sequence_dir = base_dir / sequence_id
    if not sequence_dir.exists():
        raise FileNotFoundError(f'Missing sequence directory: {sequence_dir}')
    return sequence_dir


def list_frame_ids_from_sequence_dir(sequence_dir, suffix):
    return sorted(path.stem for path in sequence_dir.glob(f'*{suffix}'))


def find_image_file(image_dir, frame_id, suffixes):
    for suffix in suffixes:
        path = image_dir / f'{frame_id}{suffix}'
        if path.exists():
            return path
    raise FileNotFoundError(f'Missing image for frame {frame_id} under {image_dir}')


def load_calib(calib_file):
    calib_dict = calibration_kitti.get_calib_from_file(str(calib_file))
    calib = calibration_kitti.Calibration(calib_dict)
    return calib_dict, calib


def get_sequence_label_cache(root_path, sequence_id, format_cfg, cache):
    key = ('label', sequence_id)
    if key in cache:
        return cache[key]

    label_dir_candidates = format_cfg['PATHS']['LABEL_DIR_CANDIDATES']
    label_filename = f'{sequence_id}.txt'
    label_file = resolve_file(root_path, [str(Path(candidate) / label_filename) for candidate in label_dir_candidates])
    label_format = format_cfg.get('LABEL_FORMAT', 'kitti_tracking_sequence')
    if label_format != 'kitti_tracking_sequence':
        raise ValueError(f'LABEL_FORMAT {label_format} is not valid for sequence_file label mode')
    cache[key] = parse_sequence_tracking_labels(label_file)
    return cache[key]


def get_frame_calib(root_path, sequence_id, frame_id, format_cfg, cache):
    layout_cfg = format_cfg['LAYOUT']
    calib_mode = layout_cfg.get('CALIB', 'sequence_file')
    calib_dir_candidates = format_cfg['PATHS']['CALIB_DIR_CANDIDATES']

    if calib_mode == 'sequence_file':
        key = ('calib', sequence_id)
        if key not in cache:
            calib_file = resolve_file(
                root_path,
                [str(Path(candidate) / f'{sequence_id}.txt') for candidate in calib_dir_candidates],
            )
            cache[key] = load_calib(calib_file)
        return cache[key]

    if calib_mode == 'sequence_frame':
        calib_file = resolve_file(
            root_path,
            [str(Path(candidate) / sequence_id / f'{frame_id}.txt') for candidate in calib_dir_candidates],
        )
        return load_calib(calib_file)

    raise ValueError(f'Unsupported calib layout mode: {calib_mode}')


def get_frame_annos(root_path, sequence_id, frame_id, frame_idx, format_cfg, cache):
    layout_cfg = format_cfg['LAYOUT']
    label_mode = layout_cfg.get('LABEL', 'sequence_file')
    label_format = format_cfg.get('LABEL_FORMAT', 'kitti_tracking_sequence')
    label_dir_candidates = format_cfg['PATHS']['LABEL_DIR_CANDIDATES']

    if label_mode == 'sequence_file':
        labels_by_frame = get_sequence_label_cache(root_path, sequence_id, format_cfg, cache)
        return labels_by_frame.get(frame_idx, [])

    if label_mode == 'sequence_frame':
        label_file = resolve_file(
            root_path,
            [str(Path(candidate) / sequence_id / f'{frame_id}.txt') for candidate in label_dir_candidates],
        )
        if label_format == 'kitti_det_with_track_id':
            return parse_frame_detection_labels_with_track_id(label_file)
        raise ValueError(f'Unsupported frame label format: {label_format}')

    raise ValueError(f'Unsupported label layout mode: {label_mode}')


def collect_infos_for_split(root_path, split, format_cfg):
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
            calib_dict, calib = get_frame_calib(root_path, sequence_id, frame_id, format_cfg, cache)
            frame_annos = get_frame_annos(root_path, sequence_id, frame_id, frame_idx, format_cfg, cache)
            annos = build_annos(calib, frame_annos)

            split_infos.append({
                'sequence_id': str(sequence_id),
                'frame_id': str(frame_id),
                'frame_idx': int(frame_idx),
                'point_cloud': {
                    'lidar_idx': f'{sequence_id}_{frame_id}',
                    'lidar_path': str(lidar_file.relative_to(root_path)),
                    'num_features': int(format_cfg.get('POINT_FEATURES', 4)),
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
        infos = collect_infos_for_split(root_path, split, format_cfg)
        out_file = save_path / f'tracking_infos_{split}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump(infos, f)
        print(f'Saved {len(infos)} frame infos to {out_file}')


if __name__ == '__main__':
    main()

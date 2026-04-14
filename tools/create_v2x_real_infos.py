import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


CAMERA_IDS = ('02', '03', '04', '05')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create detector and tracking infos for the v2x_real multi-camera tracking dataset'
    )
    parser.add_argument('--data_path', type=str, required=True, help='dataset root, e.g. data/v2x_real')
    parser.add_argument('--save_path', type=str, default=None, help='output directory, defaults to data_path')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='split names under ImageSets')
    parser.add_argument(
        '--format_cfg',
        type=str,
        default='tools/cfgs/dataset_configs/tracking_info_builder/v2x_real_sequence_multicam.yaml',
        help='dataset layout config',
    )
    parser.add_argument('--reference_camera', type=str, default='02', choices=list(CAMERA_IDS))
    parser.add_argument(
        '--lidar_tolerance',
        type=float,
        default=1e-3,
        help='max allowed abs diff for shared lidar fields when merging camera labels',
    )
    return parser.parse_args()


def load_format_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f) or {}


def parse_split_entries(root_path, split, format_cfg):
    imagesets_dir = root_path / format_cfg.get('IMAGESETS_DIR', 'ImageSets')
    split_file = imagesets_dir / f'{split}.txt'
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def find_sequence_dir(root_path, relative_dir, sequence_id):
    sequence_dir = root_path / relative_dir / sequence_id
    if not sequence_dir.exists():
        raise FileNotFoundError(f'Missing sequence directory: {sequence_dir}')
    return sequence_dir


def list_frame_ids(sequence_dir, suffix):
    return sorted(path.stem for path in sequence_dir.glob(f'*{suffix}'))


def load_multicam_calib(calib_file, reference_camera):
    entries = {}
    with open(calib_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ':' not in line:
                continue
            key, values = line.split(':', 1)
            vals = np.asarray([float(x) for x in values.strip().split()], dtype=np.float32)
            entries[key] = vals

    cam_key = f'{int(reference_camera):02d}'
    p_key = f'P_rect_{cam_key}'
    tr_key = f'Tr_velo_to_cam_{cam_key}'

    if p_key not in entries or tr_key not in entries or 'R0_rect' not in entries:
        raise KeyError(f'Calibration file missing {p_key}, {tr_key}, or R0_rect: {calib_file}')

    return {
        'P2': entries[p_key].reshape(3, 4),
        'R0': entries['R0_rect'].reshape(3, 3),
        'Tr_velo2cam': entries[tr_key].reshape(3, 4),
    }


def infer_image_shape_from_projection(calib_dict):
    p2 = calib_dict['P2']
    cx = float(p2[0, 2])
    cy = float(p2[1, 2])
    width = max(int(round(cx * 2.0)), 1)
    height = max(int(round(cy * 2.0)), 1)
    return np.asarray([height, width], dtype=np.int32)


def parse_sequence_camera_labels(label_file):
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
                'alpha': float(fields[5]),
                'bbox': np.array([float(x) for x in fields[6:10]], dtype=np.float32),
                'gt_boxes_lidar': np.array([float(x) for x in fields[10:17]], dtype=np.float32),
            })

    return frame_to_annos


def build_label_cache(root_path, sequence_id, format_cfg):
    label_root = root_path / format_cfg['PATHS']['LABEL_DIRNAME']
    cache = {}
    for cam_id in CAMERA_IDS:
        label_file = label_root / f'label_{cam_id}' / f'{sequence_id}.txt'
        cache[cam_id] = parse_sequence_camera_labels(label_file)
    return cache


def merge_frame_annos(labels_by_camera, frame_idx, tolerance):
    merged = {}

    for cam_id in CAMERA_IDS:
        for anno in labels_by_camera[cam_id].get(frame_idx, []):
            key = (int(anno['track_id']), str(anno['name']))
            if key not in merged:
                merged[key] = {
                    'track_id': int(anno['track_id']),
                    'name': str(anno['name']),
                    'alpha': float(anno['alpha']),
                    'bbox': anno['bbox'].astype(np.float32),
                    'gt_boxes_lidar': anno['gt_boxes_lidar'].astype(np.float32),
                    'source_camera': cam_id,
                }
                continue

            existing = merged[key]
            if not np.allclose(existing['gt_boxes_lidar'], anno['gt_boxes_lidar'], atol=tolerance, rtol=0.0):
                raise ValueError(
                    f'Inconsistent lidar box for frame={frame_idx}, track_id={anno["track_id"]}, '
                    f'class={anno["name"]}, cameras={existing["source_camera"]}/{cam_id}'
                )

    return [merged[key] for key in sorted(merged.keys(), key=lambda item: item[0])]


def convert_lidar_boxes_to_reference_camera(gt_boxes_lidar, calib_dict):
    if gt_boxes_lidar.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar = gt_boxes_lidar[:, 0:3].astype(np.float32).copy()
    l = gt_boxes_lidar[:, 3].astype(np.float32)
    w = gt_boxes_lidar[:, 4].astype(np.float32)
    h = gt_boxes_lidar[:, 5].astype(np.float32)
    heading = gt_boxes_lidar[:, 6].astype(np.float32)

    xyz_lidar[:, 2] -= h / 2.0
    xyz_lidar_hom = np.hstack((xyz_lidar, np.ones((xyz_lidar.shape[0], 1), dtype=np.float32)))
    xyz_cam = np.dot(xyz_lidar_hom, np.dot(calib_dict['Tr_velo2cam'].T, calib_dict['R0'].T))

    location = xyz_cam.astype(np.float32)
    dimensions = np.stack([l, h, w], axis=1).astype(np.float32)
    rotation_y = (-heading - np.pi / 2.0).astype(np.float32)
    return location, dimensions, rotation_y


def build_annos(merged_annos, calib_dict):
    if not merged_annos:
        return {
            'name': np.zeros((0,), dtype='<U1'),
            'track_id': np.zeros((0,), dtype=np.int64),
            'bbox': np.zeros((0, 4), dtype=np.float32),
            'alpha': np.zeros((0,), dtype=np.float32),
            'dimensions': np.zeros((0, 3), dtype=np.float32),
            'location': np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.zeros((0,), dtype=np.float32),
            'rotation_z': np.zeros((0,), dtype=np.float32),
            'gt_boxes_lidar': np.zeros((0, 7), dtype=np.float32),
        }

    names = np.asarray([anno['name'] for anno in merged_annos])
    track_ids = np.asarray([anno['track_id'] for anno in merged_annos], dtype=np.int64)
    bbox = np.stack([anno['bbox'] for anno in merged_annos], axis=0).astype(np.float32)
    alpha = np.asarray([anno['alpha'] for anno in merged_annos], dtype=np.float32)
    gt_boxes_lidar = np.stack([anno['gt_boxes_lidar'] for anno in merged_annos], axis=0).astype(np.float32)
    location, dimensions, rotation_y = convert_lidar_boxes_to_reference_camera(gt_boxes_lidar, calib_dict)

    return {
        'name': names,
        'track_id': track_ids,
        'bbox': bbox,
        'alpha': alpha,
        'dimensions': dimensions,
        'location': location,
        'rotation_y': rotation_y,
        'rotation_z': gt_boxes_lidar[:, 6].astype(np.float32),
        'gt_boxes_lidar': gt_boxes_lidar,
    }


def build_frame_info(root_path, sequence_id, frame_id, frame_idx, format_cfg, reference_camera, labels_by_camera, tolerance):
    point_dir = find_sequence_dir(root_path, format_cfg['PATHS']['POINT_CLOUD_DIRNAME'], sequence_id)
    calib_dir = find_sequence_dir(root_path, format_cfg['PATHS']['CALIB_DIRNAME'], sequence_id)

    lidar_file = point_dir / f'{frame_id}{format_cfg.get("POINT_CLOUD_SUFFIX", ".bin")}'
    if not lidar_file.exists():
        raise FileNotFoundError(f'Missing lidar file: {lidar_file}')

    calib_file = calib_dir / f'{frame_id}.txt'
    if not calib_file.exists():
        raise FileNotFoundError(f'Missing calib file: {calib_file}')

    calib_dict = load_multicam_calib(calib_file, reference_camera=reference_camera)
    image_shape = infer_image_shape_from_projection(calib_dict)
    merged_annos = merge_frame_annos(labels_by_camera, frame_idx=frame_idx, tolerance=tolerance)
    annos = build_annos(merged_annos, calib_dict)

    return {
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
            'image_path': '',
            'image_shape': image_shape,
        },
        'calib': calib_dict,
        'annos': annos,
    }


def collect_infos_for_split(root_path, split, format_cfg, reference_camera, tolerance):
    sequence_ids = sorted(parse_split_entries(root_path, split, format_cfg))
    infos = []

    for sequence_id in sequence_ids:
        point_dir = find_sequence_dir(root_path, format_cfg['PATHS']['POINT_CLOUD_DIRNAME'], sequence_id)
        frame_ids = list_frame_ids(point_dir, format_cfg.get('POINT_CLOUD_SUFFIX', '.bin'))
        labels_by_camera = build_label_cache(root_path, sequence_id, format_cfg)

        for frame_id in frame_ids:
            frame_idx = int(frame_id)
            infos.append(
                build_frame_info(
                    root_path=root_path,
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    frame_idx=frame_idx,
                    format_cfg=format_cfg,
                    reference_camera=reference_camera,
                    labels_by_camera=labels_by_camera,
                    tolerance=tolerance,
                )
            )

    infos.sort(key=lambda info: (info['sequence_id'], int(info['frame_idx'])))
    return infos


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def main():
    args = parse_args()
    root_path = Path(args.data_path)
    save_path = Path(args.save_path) if args.save_path is not None else root_path
    save_path.mkdir(parents=True, exist_ok=True)
    format_cfg = load_format_cfg(args.format_cfg)

    for split in args.splits:
        infos = collect_infos_for_split(
            root_path=root_path,
            split=split,
            format_cfg=format_cfg,
            reference_camera=args.reference_camera,
            tolerance=args.lidar_tolerance,
        )
        save_pickle(save_path / f'detect_infos_{split}.pkl', infos)
        save_pickle(save_path / f'tracking_infos_{split}.pkl', infos)
        print(f'Saved {len(infos)} frames for split={split} into {save_path}')


if __name__ == '__main__':
    main()

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage import io

from pcdet.utils import box_utils, calibration_kitti


def parse_args():
    parser = argparse.ArgumentParser(description='Create KITTI-tracking-style frame infos')
    parser.add_argument('--data_path', type=str, required=True, help='tracking dataset root')
    parser.add_argument('--save_path', type=str, default=None, help='output directory, defaults to data_path')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='split names under ImageSets')
    return parser.parse_args()


def resolve_first(base_path, candidates):
    for candidate in candidates:
        path = base_path / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f'Unable to resolve any of: {candidates}')


def load_sequence_ids(root_path, split):
    split_file = root_path / 'ImageSets' / f'{split}.txt'
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def parse_tracking_labels(label_file):
    frame_to_annos = defaultdict(list)
    if not label_file.exists():
        return frame_to_annos

    with open(label_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split(' ')
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


def collect_infos_for_split(root_path, split):
    sequence_ids = load_sequence_ids(root_path, split)
    split_infos = []

    for sequence_id in sequence_ids:
        point_dir = resolve_first(root_path, [
            Path('training/velodyne') / sequence_id,
            Path('testing/velodyne') / sequence_id,
        ])
        image_dir = resolve_first(root_path, [
            Path('training/image_02') / sequence_id,
            Path('training/image_2') / sequence_id,
            Path('testing/image_02') / sequence_id,
            Path('testing/image_2') / sequence_id,
        ])
        calib_file = resolve_first(root_path, [
            Path('training/calib') / f'{sequence_id}.txt',
            Path('testing/calib') / f'{sequence_id}.txt',
        ])
        label_file_candidates = [
            Path('training/label_02') / f'{sequence_id}.txt',
            Path('training/label_2') / f'{sequence_id}.txt',
        ]
        label_file = next((root_path / candidate for candidate in label_file_candidates if (root_path / candidate).exists()), None)

        calib_dict = calibration_kitti.get_calib_from_file(str(calib_file))
        calib = calibration_kitti.Calibration(calib_dict)
        labels_by_frame = parse_tracking_labels(label_file) if label_file is not None else defaultdict(list)

        lidar_files = sorted(point_dir.glob('*.bin'))
        for frame_idx, lidar_file in enumerate(lidar_files):
            frame_id = lidar_file.stem
            image_file = next(
                (image_dir / f'{frame_id}{suffix}' for suffix in ['.png', '.jpg', '.jpeg'] if (image_dir / f'{frame_id}{suffix}').exists()),
                None,
            )
            if image_file is None:
                raise FileNotFoundError(f'Missing image for sequence {sequence_id}, frame {frame_id}')

            image_shape = np.array(io.imread(image_file).shape[:2], dtype=np.int32)
            frame_annos = build_annos(calib, labels_by_frame.get(int(frame_id), []))
            split_infos.append({
                'sequence_id': str(sequence_id),
                'frame_id': str(frame_id),
                'frame_idx': int(frame_idx),
                'point_cloud': {
                    'lidar_idx': f'{sequence_id}_{frame_id}',
                    'lidar_path': str(lidar_file.relative_to(root_path)),
                    'num_features': 4,
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
                'annos': frame_annos,
            })

    split_infos.sort(key=lambda info: (info['sequence_id'], int(info['frame_idx'])))
    return split_infos


def main():
    args = parse_args()
    root_path = Path(args.data_path)
    save_path = Path(args.save_path) if args.save_path is not None else root_path
    save_path.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        infos = collect_infos_for_split(root_path, split)
        out_file = save_path / f'tracking_infos_{split}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump(infos, f)
        print(f'Saved {len(infos)} frame infos to {out_file}')


if __name__ == '__main__':
    main()

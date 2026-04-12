import argparse
import pickle
from pathlib import Path

import numpy as np
from skimage import io

from create_tracking_infos import (
    find_image_file,
    find_sequence_dir,
    get_frame_calib,
    group_entries_by_sequence,
    infer_frame_idx,
    list_frame_ids_from_sequence_dir,
    load_format_cfg,
    parse_split_entries,
)
from create_tracking_infos_from_lidar_label_demo import (
    build_annos_from_lidar_labels,
    get_frame_lidar_annos,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo: create detector infos from KITTI-tracking labels whose 3D fields are already lidar boxes'
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
        help='set if each label line has an extra score field',
    )
    parser.add_argument('--point_features', type=int, default=4)
    return parser.parse_args()


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
        out_file = save_path / f'detect_infos_{split}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump(infos, f)
        print(f'Saved {len(infos)} detector infos to {out_file}')


if __name__ == '__main__':
    main()

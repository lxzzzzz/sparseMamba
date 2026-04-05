import _init_path
import argparse
import datetime
from pathlib import Path

import yaml
import torch
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.tracking.cache import save_cache_index, save_frame_cache
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Generate detector caches for tracking')
    parser.add_argument('--detector_cfg', type=str, required=True, help='detector yaml')
    parser.add_argument('--data_cfg', type=str, required=True, help='tracking frame dataset yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='detector checkpoint')
    parser.add_argument('--save_dir', type=str, required=True, help='cache output root')
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='which tracking info split to run detector inference on',
    )
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=2, help='dataloader workers')
    return parser.parse_args()


def load_cfg(cfg_file):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def resolve_cfg_split_key(data_cfg, split):
    if 'INFO_PATH' not in data_cfg or 'DATA_SPLIT' not in data_cfg:
        raise KeyError('Data config must contain INFO_PATH and DATA_SPLIT')

    if split in data_cfg.INFO_PATH and split in data_cfg.DATA_SPLIT:
        return split

    for cfg_key, split_name in data_cfg.DATA_SPLIT.items():
        if str(split_name) == split and cfg_key in data_cfg.INFO_PATH:
            return cfg_key

    raise KeyError(f'Split "{split}" is not present in INFO_PATH / DATA_SPLIT')


def configure_data_cfg_for_split(data_cfg, split):
    cfg_split_key = resolve_cfg_split_key(data_cfg, split)

    # Cache generation always uses inference mode (training=False), which maps
    # datasets to their "test" branch. Redirect that branch to the requested split.
    data_cfg.INFO_PATH['test'] = list(data_cfg.INFO_PATH[cfg_split_key])
    data_cfg.DATA_SPLIT['test'] = data_cfg.DATA_SPLIT[cfg_split_key]
    return data_cfg


def main():
    args = parse_config()
    detector_cfg = load_cfg(args.detector_cfg)
    data_cfg = load_cfg(args.data_cfg)
    data_cfg = configure_data_cfg_for_split(data_cfg, args.split)
    detector_cfg.DATA_CONFIG = data_cfg
    detector_cfg.TAG = Path(args.detector_cfg).stem

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / f'generate_cache_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    logger = common_utils.create_logger(log_file)
    log_config_to_file(detector_cfg, logger=logger)
    logger.info('Cache generation split: %s', args.split)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=detector_cfg.DATA_CONFIG,
        class_names=detector_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=detector_cfg.MODEL, num_class=len(detector_cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    cache_index = []
    with torch.no_grad():
        for batch_dict in dataloader:
            sequence_ids = [str(item) for item in batch_dict['sequence_id']]
            frame_ids = [str(item) for item in batch_dict['frame_id']]
            frame_indices = [int(item) for item in batch_dict['frame_idx']]

            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            for batch_idx, pred in enumerate(pred_dicts):
                frame_cache = {
                    'sequence_id': sequence_ids[batch_idx],
                    'frame_id': frame_ids[batch_idx],
                    'frame_idx': frame_indices[batch_idx],
                    'pred_boxes': pred['pred_boxes'].detach().cpu().numpy(),
                    'pred_scores': pred['pred_scores'].detach().cpu().numpy(),
                    'pred_labels': pred['pred_labels'].detach().cpu().numpy(),
                    'reliability_scores': pred.get('reliability_scores', pred['pred_scores']).detach().cpu().numpy(),
                    'obs_quality_vec': pred.get(
                        'obs_quality_vec',
                        torch.zeros((pred['pred_boxes'].shape[0], 5), device=pred['pred_boxes'].device)
                    ).detach().cpu().numpy(),
                }
                rel_path = save_frame_cache(save_dir, frame_cache)
                cache_index.append({
                    'sequence_id': frame_cache['sequence_id'],
                    'frame_id': frame_cache['frame_id'],
                    'frame_idx': frame_cache['frame_idx'],
                    'cache_path': str(rel_path),
                })

    save_cache_index(save_dir, cache_index)
    logger.info('Saved %d cached frames into %s', len(cache_index), save_dir)


if __name__ == '__main__':
    main()

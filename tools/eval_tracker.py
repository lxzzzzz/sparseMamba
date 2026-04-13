import _init_path
import argparse
import datetime
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.tracking.cache import load_frame_cache
from pcdet.tracking.metrics import TrackingMetrics
from pcdet.tracking.models import TrackMamba
from pcdet.tracking.tracker import OnlineTracker
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Tracker evaluation')
    parser.add_argument('--cfg_file', type=str, required=True, help='tracking config')
    parser.add_argument('--ckpt', type=str, required=True, help='tracker checkpoint')
    parser.add_argument('--extra_tag', type=str, default='default', help='eval tag')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def build_model(model_cfg):
    return TrackMamba(
        geom_dim=int(model_cfg.get('GEOM_DIM', 8)),
        quality_dim=int(model_cfg.get('QUALITY_DIM', 7)),
        time_dim=int(model_cfg.get('TIME_DIM', 4)),
        context_dim=int(model_cfg.get('CONTEXT_DIM', 4)),
        hidden_dim=int(model_cfg.get('HIDDEN_DIM', 128)),
        num_blocks=int(model_cfg.get('NUM_BLOCKS', 2)),
        dropout=float(model_cfg.get('DROPOUT', 0.1)),
        pos_weight=float(model_cfg.get('POS_WEIGHT', 6.0)),
        assoc_weight=float(model_cfg.get('ASSOC_WEIGHT', 1.0)),
        motion_weight=float(model_cfg.get('MOTION_WEIGHT', 0.4)),
    )


def class_names_to_labels(class_names, names):
    name_to_label = {name: idx + 1 for idx, name in enumerate(class_names)}
    return np.asarray([name_to_label.get(name, -1) for name in names], dtype=np.int64)


def resolve_eval_cache_dir(data_cfg):
    if 'VAL_CACHE_DIR' in data_cfg:
        return data_cfg.VAL_CACHE_DIR
    return data_cfg.CACHE_DIR


def enrich_metric_dict(metric_dict, total_frames, num_sequences, elapsed):
    metric_dict = dict(metric_dict)
    elapsed = max(float(elapsed), 1e-6)
    metric_dict['fps'] = float(total_frames / elapsed)
    metric_dict['num_frames'] = int(total_frames)
    metric_dict['num_sequences'] = int(num_sequences)
    metric_dict['MOTA'] = metric_dict.get('mota', 0.0)
    metric_dict['MOTP'] = metric_dict.get('motp', 0.0)
    metric_dict['Rcll'] = metric_dict.get('recall', 0.0)
    metric_dict['Prcn'] = metric_dict.get('precision', 0.0)
    metric_dict['MT'] = metric_dict.get('mostly_tracked', 0)
    metric_dict['PT'] = metric_dict.get('partially_tracked', 0)
    metric_dict['ML'] = metric_dict.get('mostly_lost', 0)
    metric_dict['FP'] = metric_dict.get('fp', 0)
    metric_dict['FN'] = metric_dict.get('fn', 0)
    metric_dict['IDsw'] = metric_dict.get('id_switches', 0)
    metric_dict['Frag'] = metric_dict.get('fragments', 0)
    metric_dict['TP'] = metric_dict.get('tp', 0)
    metric_dict['IDF1'] = metric_dict.get('idf1', 0.0)
    metric_dict['IDP'] = metric_dict.get('id_precision', 0.0)
    metric_dict['IDR'] = metric_dict.get('id_recall', 0.0)
    metric_dict['HOTA'] = metric_dict.get('hota', 0.0)
    metric_dict['DetA'] = metric_dict.get('deta', 0.0)
    metric_dict['AssA'] = metric_dict.get('assa', 0.0)
    metric_dict['DetPr'] = metric_dict.get('detpr', 0.0)
    metric_dict['DetRe'] = metric_dict.get('detre', 0.0)
    metric_dict['AssPr'] = metric_dict.get('asspr', 0.0)
    metric_dict['AssRe'] = metric_dict.get('assre', 0.0)
    metric_dict['FPS'] = metric_dict.get('fps', 0.0)
    return metric_dict


def main():
    args, cfg = parse_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag / 'eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f'eval_tracker_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    logger = common_utils.create_logger(log_file)
    logger.info('Start tracker evaluation')
    log_config_to_file(cfg, logger=logger)

    model = build_model(cfg.MODEL).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    tracker_cfg = dict(cfg.TRACKER)
    tracker_cfg['HISTORY_LEN'] = int(cfg.DATA_CONFIG.get('HISTORY_LEN', 8))
    cache_dir = resolve_eval_cache_dir(cfg.DATA_CONFIG)

    data_root = Path(cfg.DATA_CONFIG.ROOT_DIR)
    info_file = data_root / cfg.DATA_CONFIG.INFO_PATH['test'][0]
    with open(info_file, 'rb') as f:
        frame_infos = pickle.load(f)

    frame_infos.sort(key=lambda info: (str(info['sequence_id']), int(info['frame_idx'])))
    sequence_to_infos = defaultdict(list)
    for info in frame_infos:
        sequence_to_infos[str(info['sequence_id'])].append(info)

    metrics = TrackingMetrics(iou_threshold=float(cfg.EVAL.get('IOU_THRESHOLD', cfg.DATA_CONFIG.get('DET_GT_MATCH_IOU', 0.1))))
    results = {}
    total_frames = 0
    start_time = time.perf_counter()

    for sequence_id, infos in sequence_to_infos.items():
        tracker = OnlineTracker(model, tracker_cfg, cfg.CLASS_NAMES, device)
        seq_results = []

        progress_bar = tqdm(infos, desc=f'eval {sequence_id}', leave=True)
        for info in progress_bar:
            total_frames += 1
            frame_cache = load_frame_cache(cache_dir, sequence_id, info['frame_idx'])
            outputs = tracker.update(frame_cache)
            seq_results.append({
                'frame_idx': int(info['frame_idx']),
                'tracks': outputs,
            })

            gt_annos = info.get('annos', {})
            gt_boxes = np.asarray(gt_annos.get('gt_boxes_lidar', []), dtype=np.float32).reshape(-1, 7)
            gt_ids = np.asarray(gt_annos.get('track_id', []), dtype=np.int64)
            gt_labels = class_names_to_labels(cfg.CLASS_NAMES, gt_annos.get('name', []))

            pred_boxes = np.asarray([item['pred_box'] for item in outputs], dtype=np.float32).reshape(-1, 7)
            pred_ids = np.asarray([item['track_id'] for item in outputs], dtype=np.int64)
            pred_labels = np.asarray([item['pred_label'] for item in outputs], dtype=np.int64)
            metrics.update(sequence_id, gt_boxes, gt_ids, gt_labels, pred_boxes, pred_ids, pred_labels)
            progress_bar.set_postfix(tracks=len(outputs))

        results[sequence_id] = seq_results

    metric_dict = enrich_metric_dict(
        metrics.summary(), total_frames=total_frames, num_sequences=len(sequence_to_infos), elapsed=time.perf_counter() - start_time
    )
    with open(output_dir / 'tracker_metrics.json', 'w') as f:
        json.dump(metric_dict, f, indent=2)
    with open(output_dir / 'tracker_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    logger.info('Tracking metrics: %s', metric_dict)
    logger.info(
        'Summary | MOTA=%.4f MOTP=%.4f Rcll=%.4f Prcn=%.4f MT=%d ML=%d FP=%d FN=%d IDsw=%d Frag=%d FPS=%.2f',
        metric_dict.get('mota', 0.0),
        metric_dict.get('motp', 0.0),
        metric_dict.get('recall', 0.0),
        metric_dict.get('precision', 0.0),
        metric_dict.get('mostly_tracked', 0),
        metric_dict.get('mostly_lost', 0),
        metric_dict.get('fp', 0),
        metric_dict.get('fn', 0),
        metric_dict.get('id_switches', 0),
        metric_dict.get('fragments', 0),
        metric_dict.get('fps', 0.0),
    )
    print('================ TrackMamba Evaluation ================')
    print(f'cfg_file: {args.cfg_file}')
    print(f'ckpt: {args.ckpt}')
    print(f'cache_dir: {cache_dir}')
    print(f'info_file: {info_file}')
    print(
        'Summary | '
        f"MOTA={metric_dict.get('mota', 0.0):.4f} "
        f"MOTP={metric_dict.get('motp', 0.0):.4f} "
        f"Rcll={metric_dict.get('recall', 0.0):.4f} "
        f"Prcn={metric_dict.get('precision', 0.0):.4f} "
        f"MT={metric_dict.get('mostly_tracked', 0)} "
        f"ML={metric_dict.get('mostly_lost', 0)} "
        f"FP={metric_dict.get('fp', 0)} "
        f"FN={metric_dict.get('fn', 0)} "
        f"IDsw={metric_dict.get('id_switches', 0)} "
        f"Frag={metric_dict.get('fragments', 0)} "
        f"FPS={metric_dict.get('fps', 0.0):.2f}"
    )
    print(
        'Extended | '
        f"HOTA={metric_dict.get('hota', 0.0):.4f} "
        f"DetA={metric_dict.get('deta', 0.0):.4f} "
        f"AssA={metric_dict.get('assa', 0.0):.4f} "
        f"IDF1={metric_dict.get('idf1', 0.0):.4f} "
        f"IDP={metric_dict.get('id_precision', 0.0):.4f} "
        f"IDR={metric_dict.get('id_recall', 0.0):.4f} "
        f"DetPr={metric_dict.get('detpr', 0.0):.4f} "
        f"DetRe={metric_dict.get('detre', 0.0):.4f} "
        f"AssPr={metric_dict.get('asspr', 0.0):.4f} "
        f"AssRe={metric_dict.get('assre', 0.0):.4f} "
        f"TP={metric_dict.get('tp', 0)} "
        f"PT={metric_dict.get('partially_tracked', 0)} "
        f"Seq={metric_dict.get('num_sequences', 0)} "
        f"Frames={metric_dict.get('num_frames', 0)}"
    )


if __name__ == '__main__':
    main()

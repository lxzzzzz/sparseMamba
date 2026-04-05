import _init_path
import argparse
import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.tracking.datasets import TrackingDataset, collate_tracking_batch
from pcdet.tracking.models import TrackMamba
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Tracker training')
    parser.add_argument('--cfg_file', type=str, required=True, help='tracking config file')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--epochs', type=int, default=None, help='num epochs')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--extra_tag', type=str, default='default', help='run tag')
    parser.add_argument('--ckpt', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


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
        recovery_weight=float(model_cfg.get('RECOVERY_WEIGHT', 0.5)),
        survival_weight=float(model_cfg.get('SURVIVAL_WEIGHT', 0.3)),
        motion_weight=float(model_cfg.get('MOTION_WEIGHT', 0.4)),
    )


def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_tb = {}
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='val', leave=False)
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss, tb_dict = model.get_loss(batch, outputs)
            total_loss += float(loss.item())
            total_steps += 1
            for key, value in tb_dict.items():
                total_tb[key] = total_tb.get(key, 0.0) + float(value)
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_steps = max(total_steps, 1)
    avg_tb = {key: value / avg_steps for key, value in total_tb.items()}
    avg_tb['loss'] = total_loss / avg_steps
    return avg_tb


def main():
    args, cfg = parse_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    epochs = args.epochs or cfg.OPTIMIZATION.NUM_EPOCHS

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f'train_tracker_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    logger = common_utils.create_logger(log_file)
    logger.info('Start tracker training')
    log_config_to_file(cfg, logger=logger)
    logger.info('Train cache dir: %s', cfg.DATA_CONFIG.get('TRAIN_CACHE_DIR', cfg.DATA_CONFIG.get('CACHE_DIR', '')))
    logger.info('Val cache dir: %s', cfg.DATA_CONFIG.get('VAL_CACHE_DIR', cfg.DATA_CONFIG.get('CACHE_DIR', '')))

    train_set = TrackingDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_tracking_batch,
        pin_memory=True,
    )

    val_loader = None
    if 'test' in cfg.DATA_CONFIG.INFO_PATH and len(cfg.DATA_CONFIG.INFO_PATH['test']) > 0:
        val_set = TrackingDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_tracking_batch,
            pin_memory=True,
        )

    model = build_model(cfg.MODEL).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.OPTIMIZATION.LR),
        weight_decay=float(cfg.OPTIMIZATION.get('WEIGHT_DECAY', 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.OPTIMIZATION.get('LR_STEP', 8)),
        gamma=float(cfg.OPTIMIZATION.get('LR_GAMMA', 0.5)),
    )

    start_epoch = 0
    best_val = float('inf')
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        best_val = float(checkpoint.get('best_val', best_val))
        logger.info('Resumed from %s', args.ckpt)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tb = {}
        progress_bar = tqdm(train_loader, desc=f'train {epoch + 1}/{epochs}', leave=False)
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss, tb_dict = model.get_loss(batch, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            for key, value in tb_dict.items():
                epoch_tb[key] = epoch_tb.get(key, 0.0) + float(value)
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        tb_log.add_scalar('train/loss', avg_train_loss, epoch)
        avg_train_tb = {key: value / max(len(train_loader), 1) for key, value in epoch_tb.items()}
        for key, value in epoch_tb.items():
            tb_log.add_scalar(f'train/{key}', value / max(len(train_loader), 1), epoch)
        logger.info(
            'Epoch %d train_loss=%.6f assoc=%.6f recovery=%.6f survival=%.6f motion=%.6f',
            epoch,
            avg_train_loss,
            avg_train_tb.get('loss_assoc', 0.0),
            avg_train_tb.get('loss_recovery', 0.0),
            avg_train_tb.get('loss_survival', 0.0),
            avg_train_tb.get('loss_motion', 0.0),
        )

        val_loss = None
        if val_loader is not None:
            val_metrics = evaluate_epoch(model, val_loader, device)
            val_loss = val_metrics['loss']
            for key, value in val_metrics.items():
                tb_log.add_scalar(f'val/{key}', value, epoch)
            logger.info(
                'Epoch %d val_loss=%.6f assoc=%.6f recovery=%.6f survival=%.6f motion=%.6f',
                epoch,
                val_metrics['loss'],
                val_metrics.get('loss_assoc', 0.0),
                val_metrics.get('loss_recovery', 0.0),
                val_metrics.get('loss_survival', 0.0),
                val_metrics.get('loss_motion', 0.0),
            )

        ckpt_path = ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val': best_val if val_loss is None else min(best_val, val_loss),
        }, ckpt_path)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val': best_val,
            }, ckpt_dir / 'best_model.pth')

    tb_log.close()
    logger.info('Tracker training finished')


if __name__ == '__main__':
    main()

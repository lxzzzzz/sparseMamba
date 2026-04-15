# V2X-Xian 运行说明

## 1. 数据放置

将数据集根目录放到：

```text
data/v2x_xian
```

目录形态应为：

```text
data/v2x_xian/
├── training/
├── ImageSets/
```

## 2. 生成 tracking infos

```bash
python tools/create_v2x_xian_infos.py \
  --data_path data/v2x_xian \
  --save_path data/v2x_xian \
  --splits train val trainval all
```

说明：

- `label_02/<seq>.txt` 使用雷达坐标系的 `x y z dx dy dz yaw`
- 这里不要用通用的 `tools/create_tracking_infos.py`，因为它按相机坐标 KITTI label 解析，不适用于这套 lidar label
- 生成的 `tracking_infos_*.pkl` 同时可用于 detector training、val cache、tracking eval

## 3. 训练纯雷达检测器

训练：

```bash
python tools/train.py \
  --cfg_file tools/cfgs/dair_v2x_models/voxelnext_v2x_xian_lidar.yaml \
  --extra_tag v2x_xian_lidar
```

验证：

```bash
python tools/test.py \
  --cfg_file tools/cfgs/dair_v2x_models/voxelnext_v2x_xian_lidar.yaml \
  --ckpt output/dair_v2x_models/voxelnext_v2x_xian_lidar/v2x_xian_lidar/ckpt/checkpoint_epoch_30.pth \
  --eval_tag val
```

## 4. 训练多模态检测器

训练：

```bash
python tools/train.py \
  --cfg_file tools/cfgs/dair_v2x_models/fusion_voxelnext_v2x_xian.yaml \
  --extra_tag v2x_xian_fusion
```

验证：

```bash
python tools/test.py \
  --cfg_file tools/cfgs/dair_v2x_models/fusion_voxelnext_v2x_xian.yaml \
  --ckpt output/dair_v2x_models/fusion_voxelnext_v2x_xian/v2x_xian_fusion/ckpt/checkpoint_epoch_30.pth \
  --eval_tag val
```

## 5. 生成 val cache

纯雷达 detector：

```bash
python tools/generate_tracking_cache.py \
  --detector_cfg tools/cfgs/dair_v2x_models/voxelnext_v2x_xian_lidar.yaml \
  --data_cfg tools/cfgs/dataset_configs/v2x_xian_tracking_frame_dataset.yaml \
  --ckpt output/dair_v2x_models/voxelnext_v2x_xian_lidar/v2x_xian_lidar/ckpt/checkpoint_epoch_30.pth \
  --save_dir cache/v2x_xian/lidar_val \
  --split val
```

多模态 detector：

```bash
python tools/generate_tracking_cache.py \
  --detector_cfg tools/cfgs/dair_v2x_models/fusion_voxelnext_v2x_xian.yaml \
  --data_cfg tools/cfgs/dataset_configs/v2x_xian_tracking_frame_fusion_dataset.yaml \
  --ckpt output/dair_v2x_models/fusion_voxelnext_v2x_xian/v2x_xian_fusion/ckpt/checkpoint_epoch_30.pth \
  --save_dir cache/v2x_xian/fusion_val \
  --split val
```

## 6. 做 AB3DMOT 跟踪评测

```bash
python tools/eval_ab3dmot_baseline.py \
  --cache_dir cache/v2x_xian/fusion_val \
  --gt_pkl data/v2x_xian/tracking_infos_val.pkl \
  --data_cfg tools/cfgs/dataset_configs/v2x_xian_tracking_frame_dataset.yaml \
  --dataset_preset v2x_xian_2hz \
  --class_names Car \
  --save_dir output/tracking_eval/v2x_xian/ab3dmot_fusion
```

## 7. 做 UDCA 跟踪评测

```bash
python tools/eval_udca_policy.py \
  --cache_dir cache/v2x_xian/fusion_val \
  --gt_pkl data/v2x_xian/tracking_infos_val.pkl \
  --data_cfg tools/cfgs/dataset_configs/v2x_xian_tracking_frame_dataset.yaml \
  --dataset_preset v2x_xian_2hz \
  --class_names Car \
  --save_dir output/tracking_eval/v2x_xian/udca_fusion
```

## 8. 可视化

BEV cache 可视化：

```bash
python tools/vis_tracking_cache_bev.py \
  --cache_dir cache/v2x_xian/fusion_val \
  --gt_pkl data/v2x_xian/tracking_infos_val.pkl
```

如果需要单帧投影检查，可基于 `tools/check_projection.py` 再指定 `data/v2x_xian/training` 路径做调试。

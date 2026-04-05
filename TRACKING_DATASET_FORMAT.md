# 通用跟踪数据格式

这套 tracking 代码的核心接口不是某个固定数据集名字，而是 `tracking_infos_*.pkl` 的统一结构。

因此后面你更换检测数据集或跟踪数据集时，建议遵循下面的原则：

## 1. 检测阶段

只要数据能被现有检测数据集读取，或者你补一个新的检测 dataset 配置 / dataset 类，就可以训练 detector。

检测阶段最终只需要产出：

- detector checkpoint

## 2. 跟踪阶段真正依赖的输入

跟踪阶段不直接依赖原始检测训练格式，而依赖两部分：

1. `tracking_infos_train.pkl` / `tracking_infos_val.pkl`
2. detector cache

其中 `tracking_infos_*.pkl` 每一帧至少需要这些字段：

```python
{
    'sequence_id': str,
    'frame_id': str,
    'frame_idx': int,
    'point_cloud': {
        'lidar_idx': str,
        'lidar_path': str,
        'num_features': int,
    },
    'image': {
        'image_idx': str,
        'image_path': str,
        'image_shape': np.ndarray,
    },
    'calib': {
        'P2': np.ndarray,
        'R0': np.ndarray,
        'Tr_velo2cam': np.ndarray,
    },
    'annos': {
        'name': np.ndarray,
        'track_id': np.ndarray,
        'bbox': np.ndarray,
        'dimensions': np.ndarray,
        'location': np.ndarray,
        'rotation_y': np.ndarray,
        'gt_boxes_lidar': np.ndarray,
    }
}
```

## 3. 你以后换数据集时最推荐的做法

不要每来一个新数据集就改 tracker 主体逻辑。

正确做法是：

1. 把数据整理成 KITTI-style tracking 结构
2. 给 `create_tracking_infos.py` 提供一个新的 `format_cfg`
3. 新增对应的 tracking dataset yaml
4. 其余训练、cache、eval 脚本不动

## 4. create_tracking_infos.py 现在支持的两类组织方式

### A. 序列级 KITTI tracking

适用配置：

`tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_sequence.yaml`

适用特点：

- `ImageSets/train.txt` 里写的是序列 id
- 标定是 `training/calib/<seq>.txt`
- 标注是 `training/label_02/<seq>.txt`
- 标注每行格式是：

```text
frame_id track_id class ...
```

### B. 帧级 KITTI tracking 变体

适用配置：

`tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_framewise.yaml`

适用特点：

- `ImageSets/train.txt` 里写的是 `seq/frame`
- 标定是 `training/calib/<seq>/<frame>.txt`
- 标注是 `training/label_02/<seq>/<frame>.txt`
- 标注每行格式是：

```text
class truncated occluded alpha bbox h w l x y z ry track_id
```

## 5. 你当前这份数据该用哪个

你当前这份：

`/media/lx/LY/Roadside/V2X-Seq-train_val/kitti_tracking`

应使用：

`tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_framewise.yaml`

生成命令：

```bash
python tools/create_tracking_infos.py \
  --data_path /media/lx/LY/Roadside/V2X-Seq-train_val/kitti_tracking \
  --save_path /media/lx/LY/Roadside/V2X-Seq-train_val/kitti_tracking \
  --splits train val \
  --format_cfg tools/cfgs/dataset_configs/tracking_info_builder/kitti_tracking_framewise.yaml
```

固定 detector data config：

`tools/cfgs/dataset_configs/kitti_tracking_dataset.yaml`

固定 tracker config：

`tools/cfgs/tracking_models/track_mamba_kitti_tracking.yaml`

生成 detector cache：

```bash
python tools/generate_tracking_cache.py \
  --detector_cfg <你的detector.yaml> \
  --data_cfg tools/cfgs/dataset_configs/kitti_tracking_dataset.yaml \
  --ckpt <你的detector.ckpt> \
  --save_dir /media/lx/LY/Roadside/V2X-Seq-train_val/kitti_tracking/cache/<cache_name>
```

训练 tracker：

```bash
python tools/train_tracker.py \
  --cfg_file tools/cfgs/tracking_models/track_mamba_kitti_tracking.yaml \
  --set DATA_CONFIG.CACHE_DIR /media/lx/LY/Roadside/V2X-Seq-train_val/kitti_tracking/cache/<cache_name>
```

## 6. 以后新增数据集时你只需要改什么

通常只改两处：

1. 新增一个 `format_cfg`
2. 新增一个 dataset yaml

只要你把数据整理成上面两类之一，或者非常接近这两类，主训练代码就不用再动。

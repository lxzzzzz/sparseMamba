# TODO

## 1. 当前已完成的进度

### 1.1 检测主线已经收敛
- 已明确当前检测方法的主线不是 `Mamba detector`，而是：
  - 远距离稀疏目标的跨模态检测增强
  - 面向弱观测的前景感知融合抑噪
- 检测侧 Mamba 已从设计上和代码上移除，不再作为 detector/backbone 的一部分保留。

### 1.2 检测侧命名与注册已经清理完成
- 检测器类已经重命名为 `FusionVoxelNeXt`
  - 文件：`pcdet/models/detectors/fusion_voxelnext.py`
- 3D 主干类已经重命名为 `VoxelNeXtFusion`
  - 文件：`pcdet/models/backbones_3d/voxelnext_fusion.py`
- 注册入口已更新：
  - `pcdet/models/detectors/__init__.py`
  - `pcdet/models/backbones_3d/__init__.py`
- 检测配置已改为：
  - `tools/cfgs/dair_v2x_models/fusion_voxelnext.yaml`

### 1.3 DeformableRectifier 已升级为可学习 deformable 采样
- 当前 `DeformableRectifier` 不再是固定 17 点硬采样。
- 已实现：
  - 基于投影中心的图像采样
  - `base_offsets` 作为固定局部先验
  - `offset_head` 预测 residual offsets
  - `tanh + offset_range` 控制偏移范围
  - `grid_sample` 做图像特征采样
  - `valid_mask` 屏蔽越界采样点
  - 注意力聚合采样到的图像特征
- 相关配置已加入 `fusion_voxelnext.yaml`：
  - `DEFORMABLE_NUM_POINTS`
  - `DEFORMABLE_OFFSET_RANGE`
  - `MASK_INVALID_SAMPLES`
  - `USE_DEFORMABLE_RESIDUAL`

### 1.4 InstanceAwareGate 和辅助监督已保留并接通
- `InstanceAwareGate` 仍保留在融合块中。
- 当前训练逻辑：
  - backbone 产生 `aux_preds`
  - detector 中根据稀疏体素中心和 GT boxes 生成前景/背景标签
  - 通过 BCE 计算辅助监督
- 检测总损失：
  - `loss = loss_head + aux_loss_weight * loss_aux`

### 1.5 整体检测架构已稳定
- 当前 detector 路线：
  - `MeanVFE`
  - `VoxelNeXtFusion` 稀疏主干
  - `conv4` 处插入图像-点云融合块
  - 保留 `conv5/conv6`
  - 稀疏 BEV 聚合
  - `VoxelNeXtHead`
- 当前检测头仍然是 **VoxelNeXt 稀疏头**，不是 CenterPoint dense head。

### 1.6 论文定位已初步统一
- 当前文章的合理总立意已经明确为：
  - 面向远距离稀疏弱观测场景的稳健 3D 多目标跟踪
- 检测端职责：
  - 提升单帧目标可见性
  - 增强远距离弱目标的检测稳定性
- 跟踪端职责：
  - 利用时序信息缓解漏检、分数波动、ID switch 和轨迹断裂


## 2. 当前核心代码的设计思路

### 2.1 检测器：`FusionVoxelNeXt`
文件：
- `pcdet/models/detectors/fusion_voxelnext.py`

设计职责：
- 作为总检测器串联各模块
- 训练时汇总：
  - 主检测损失
  - `InstanceAwareGate` 的辅助损失
- 推理时沿用标准 VoxelNeXt 后处理

核心思想：
- 不把检测故事讲成 “Mamba detector”
- 而是聚焦在：
  - 稀疏体素层的跨模态证据增强
  - 基于前景感知的融合抑噪

### 2.2 主干：`VoxelNeXtFusion`
文件：
- `pcdet/models/backbones_3d/voxelnext_fusion.py`

设计职责：
- 保持 VoxelNeXt 全稀疏主干范式
- 在 `conv4` 的 stride=8 稀疏特征层插入融合块
- 之后继续使用 `conv5/conv6` 扩大感受野

这样设计的原因：
- 太早融合，图像语义不够强
- 太晚融合，空间分辨率太低
- `conv4` 是较好的折中位置

### 2.3 图像编码器：`SimpleImageEncoder`
设计职责：
- 以 `ResNet18` 截断 backbone 提取图像特征
- 输出与融合层通道对齐的 2D feature map

当前作用：
- 给稀疏 LiDAR token 提供补充的纹理和边界信息
- 特别用于远距离点云稀疏目标的单帧增强

### 2.4 可学习跨模态校正模块：`DeformableRectifier`
设计职责：
- 将稀疏 voxel 中心恢复为物理坐标
- 投影到图像平面
- 以投影中心为 anchor，在图像特征图上进行可学习 deformable 采样
- 对采样结果做注意力聚合，形成图像补充特征

当前实现要点：
- `base_offsets`：局部采样先验
- `offset_head`：预测残差偏移
- `valid_mask`：屏蔽越界采样点
- `q/k/v + softmax`：对候选图像证据加权聚合

方法意义：
- 不是把它单独作为“全新可变形模块”来讲
- 更合理的表述是：
  - 将可学习局部采样嵌入到全稀疏 VoxelNeXt 的跨模态融合过程中
  - 提升远距离弱观测目标的局部证据提取稳定性

### 2.5 前景感知门控：`InstanceAwareGate`
设计职责：
- 从 LiDAR 稀疏特征预测当前位置的 objectness / foreground tendency
- 将该标量映射为通道级 gate scale
- 用于调制融合后的特征

当前作用：
- 抑制背景区域被无差别注入图像信息
- 让融合更聚焦于潜在目标区域

### 2.6 融合块：`FusionRectifierBlock`
设计职责：
- 组合：
  - `DeformableRectifier`
  - `InstanceAwareGate`
  - `fusion_proj`
  - `LayerNorm`
- 输出新的稀疏特征

当前数据流：
- LiDAR sparse feature -> rectifier 获取图像补充特征
- 与原始 LiDAR 特征拼接并线性融合
- 用 gate 做通道调制
- 残差归一化后写回 sparse tensor

### 2.7 检测侧目前不再保留 Mamba
结论已经确定：
- 单帧远距离稀疏目标检测里，Mamba 不是当前最优增量点
- detector 中再加 Mamba 会使方法故事分散、变量过多
- 因此 Mamba 被明确留给后续 tracking 侧使用


## 3. 当前整篇文章的整体立意

当前最顺的表述：
- 面向远距离稀疏弱观测场景的稳健 3D 多目标跟踪

问题定义：
- 远距离目标点云稀疏
- 单帧观测弱
- 图像与点云存在投影误差和局部不对齐
- 检测分数容易波动
- 在跟踪时容易出现：
  - 漏检后轨迹断裂
  - 遮挡后重关联失败
  - 邻近目标交互时 ID switch

检测端解决的问题：
- 让目标更容易在单帧中被“看见”
- 提高弱观测场景中的单帧检测稳定性

跟踪端将要解决的问题：
- 让已经被看见过的目标不容易“丢”
- 通过历史时序上下文提升关联鲁棒性


## 4. 当前尚未完成的部分

### 4.1 跟踪模块还没有开始落地
当前仓库中尚未新增以下内容：
- tracking dataset class
- detector prediction cache 生成脚本
- Track-Mamba 模型
- online tracker
- `train_tracker.py`
- `eval_tracker.py`
- tracking 配置文件

### 4.2 跟踪数据集还未接入
后续计划是加入符合 KITTI tracking 风格的数据集。

预期需要的数据字段：
- `sequence_id`
- `frame_id`
- `frame_idx`
- `track_id`
- `gt_boxes`
- 可选：
  - `points`
  - `images`
  - `calib`


## 5. 下一步需要实施的 plan

### 5.1 总体路线
采用两阶段、模块化的 `tracking-by-detection` 路线：

第一阶段：
- 训练检测器 `FusionVoxelNeXt`

第二阶段：
- 用冻结 detector 在 tracking 数据上生成逐帧预测缓存
- 基于缓存训练 `Track-Mamba`

推理阶段：
- detector 逐帧输出检测结果
- tracker 基于历史轨迹和当前检测做在线关联

### 5.2 跟踪部分的核心创新方向
不要把创新点写成：
- “把 Mamba 用到跟踪上”

更合理的创新表述应当是：
- 将检测端的弱观测信息显式传递给时序关联模块
- 用 Mamba 建模轨迹历史中的观测退化与恢复过程
- 在远距离稀疏弱观测条件下提升跨帧身份一致性

建议的方法关键词：
- Detector-Aware Track-Mamba
- Weak-Observation-Aware Association
- Reliability-Guided Temporal Association

### 5.3 推荐的跟踪 token 设计
每个 detection / track token 至少包含：
- 几何信息：
  - center
  - size
  - yaw
- 运动信息：
  - 与上一帧的位移
  - 时间间隔
  - 可选简单速度近似
- 检测信息：
  - `pred_score`
  - `pred_label`
- 弱观测 / 可靠性信息：
  - detector 的置信度统计
  - 可选 box 内点数
  - 可选 heatmap / objectness 相关统计

第一版建议：
- 先使用 geometry + score + class + 简单 reliability
- 不要一开始就加复杂外观 re-id 分支

### 5.4 推荐的工程拆分
建议新增如下模块：

1. tracking dataset
- 新目录建议：
  - `pcdet/datasets/kitti_tracking/`
  - 或 `pcdet/datasets/dair_tracking/`
- 新类建议：
  - `KittiTrackingDataset`
  - 或 `DairTrackingDataset`

2. tracking package
- 新目录建议：
  - `pcdet/tracking/`
- 其中包括：
  - `track_mamba.py`
  - `online_tracker.py`
  - `tracking_dataset_utils.py`
  - 可选 `tracking_metrics.py`

3. tools 脚本
- `tools/generate_track_cache.py`
- `tools/train_tracker.py`
- `tools/eval_tracker.py`

### 5.5 配置文件的模块化方案
需要把 detector 和 tracker 完全解耦，通过配置连接。

建议新增：
- `tools/cfgs/dataset_configs/dair_v2x_tracking_dataset.yaml`
- `tools/cfgs/dair_v2x_trackers/track_mamba_online.yaml`

tracker 顶层配置建议包含以下 section：
- `DATA_CONFIG`
- `DETECTOR`
- `TRACKER`
- `ASSOCIATION`
- `TRAINING`
- `INFERENCE`

其中：

`DETECTOR` 建议包含：
- `CFG_FILE`
- `CKPT`
- `CACHE_DIR`
- `USE_PRECOMPUTED_PREDICTIONS`

`TRACKER` 建议包含：
- `NAME: TrackMamba`
- `HISTORY_LEN`
- `TOKEN_DIM`
- `USE_RELIABILITY_FEATURES`
- `MAMBA_D_MODEL`
- `MAMBA_D_STATE`
- `NUM_LAYERS`

`ASSOCIATION` 建议包含：
- `CLASS_CONSISTENT_MATCH`
- `MAX_CENTER_DIST`
- `MIN_IOU_FOR_LABEL`
- `MATCH_SCORE_THRESH`
- `UNMATCHED_TTL`
- `MIN_HITS_TO_CONFIRM`

`INFERENCE` 建议包含：
- `INIT_TRACK_THRESH`
- `UPDATE_TRACK_THRESH`
- `KILL_TRACK_AGE`
- `MAX_ACTIVE_TRACKS`

### 5.6 推荐的实施顺序
建议严格按下面顺序落地：

1. 新增 tracking dataset class
- 先把 KITTI tracking 风格序列数据读通
- 确保能读到 `sequence_id/frame_idx/track_id`

2. 新增 detector cache 生成脚本
- 用 detector 对 tracking 数据逐帧推理
- 保存每帧检测结果与轻量 `track_features`

3. 新增 tracker config
- 先把配置结构搭好，不急着实现全部逻辑

4. 实现 Track-Mamba
- 先做关联打分模块
- 不急着做复杂 motion model

5. 实现 online tracker
- 采用 active tracks + Hungarian matching
- 支持 birth / update / kill

6. 实现 `train_tracker.py`
- 使用 detector cache + GT track id 构造正负关联样本

7. 实现 `eval_tracker.py`
- 在线逐帧输出 track 结果

8. 做基础 ablation
- 几何匹配 baseline
- 非 Mamba baseline
- Track-Mamba
- reliability-aware Track-Mamba


## 6. 实施时的重要注意事项

### 6.1 不要回到 detector-side Mamba
- 检测端的 Mamba 已经明确删除
- 后续 Mamba 只放在 tracking 侧

### 6.2 训练流程必须保持两阶段
- 先 detector
- 再 tracker
- 第一版不要做 det-track end-to-end 联训

### 6.3 跟踪训练要尽量使用 detector predictions
- 不要只拿 GT boxes 训练 tracker
- 应该用 detector 的实际输出构造关联样本
- 这样训练分布才和部署阶段一致

### 6.4 第一版先做简单、稳的 tracker
- 先把 online association 跑通
- 先不要引入重型 image appearance 分支
- 先把“弱观测鲁棒的时序关联”讲清楚

### 6.5 论文贡献组织建议
当前最合理的贡献结构：
- Contribution 1:
  - 面向远距离稀疏目标的可学习跨模态局部校正融合
- Contribution 2:
  - 实例感知门控及辅助监督，抑制背景噪声
- Contribution 3:
  - 面向弱观测时序关联的 Detector-Aware Track-Mamba


## 7. 当前关键文件清单

检测侧核心文件：
- `pcdet/models/detectors/fusion_voxelnext.py`
- `pcdet/models/backbones_3d/voxelnext_fusion.py`
- `tools/cfgs/dair_v2x_models/fusion_voxelnext.yaml`
- `tools/cfgs/dataset_configs/dair_v2x_dataset.yaml`
- `pcdet/models/detectors/__init__.py`
- `pcdet/models/backbones_3d/__init__.py`

下一步将要新增的关键文件：
- `tools/cfgs/dataset_configs/dair_v2x_tracking_dataset.yaml`
- `tools/cfgs/dair_v2x_trackers/track_mamba_online.yaml`
- `tools/generate_track_cache.py`
- `tools/train_tracker.py`
- `tools/eval_tracker.py`
- `pcdet/tracking/track_mamba.py`
- `pcdet/tracking/online_tracker.py`
- `pcdet/datasets/kitti_tracking/...` 或 `pcdet/datasets/dair_tracking/...`


## 8. 一句话续接提示

如果后续要继续实现，默认从下面这一步开始：

先新增 **tracking dataset class + tracking config skeleton**，把 `sequence_id / frame_idx / track_id` 的数据链路先打通，再实现 detector cache 和 Track-Mamba。

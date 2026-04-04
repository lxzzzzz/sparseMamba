#!/bin/bash
# 开启报错即停止模式
set -e 

# ==========================================
# 1. 在这里集中定义你的所有参数
# ==========================================
CONFIG_FILE="tools/cfgs/dair_v2x_models/fusion_voxelnext_100m.yaml"
EXTRA_TAG="fusion_det_100m"
WORKERS=0
EPOCHS=1          # 新增：设置训练的 epoch 数量
BATCH_SIZE=4       # 新增：设置 batch size

TIME=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="train_${EXTRA_TAG}_${TIME}.log"

echo "========================================================="
echo "开始训练融合模型: ${EXTRA_TAG}"
echo "使用配置文件: ${CONFIG_FILE}"
echo "设定的 Epochs: ${EPOCHS}"
echo "设定的 Batch Size: ${BATCH_SIZE}"
echo "日志将实时保存到: ${LOG_FILE}"
echo "========================================================="

# ==========================================
# 2. 在执行命令中把变量传进去
# ==========================================
python tools/train.py \
  --cfg_file ${CONFIG_FILE} \
  --extra_tag ${EXTRA_TAG} \
  --workers ${WORKERS} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  2>&1 | tee ${LOG_FILE}

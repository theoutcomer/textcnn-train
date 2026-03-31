#!/bin/bash
# 阿里云 PAI-DSW 训练启动脚本

set -e

echo "========================================"
echo "TextCNN PAI 训练启动"
echo "========================================"

# 设置环境变量
export PYTHONPATH=/mnt/workspace/textcnn:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 检查 GPU
echo "检查 GPU..."
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 创建输出目录
mkdir -p /mnt/output/logs
mkdir -p /mnt/output/checkpoints
mkdir -p /mnt/output/exported

echo ""
echo "========================================"
echo "开始训练"
echo "========================================"

# 启动训练
python scripts/cloud_train.py \
    --config configs/pai_config.yaml \
    --data_path /mnt/data/cnfanews_training.json \
    2>&1 | tee /mnt/output/logs/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "训练完成"
echo "========================================"
echo "检查点: /mnt/output/checkpoints/"
echo "日志: /mnt/output/logs/"

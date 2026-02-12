#!/bin/bash

# 1. 检查并创建虚拟环境 (如果服务器已有环境可跳过)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. 安装依赖 (服务器通常已有 torch，若无则安装)
# 尝试导入 torch，失败则安装
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    # 假设服务器 CUDA 版本 >= 12.1
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
fi

# 3. 准备全量数据
echo "Generating Full Sudoku Dataset..."
# 确保存放数据的目录存在
mkdir -p data_source
# 如果服务器能联网下载，直接跑；如果不能，需确保 data_source 下有 csv 文件
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-full \
  --num-aug 1000 \
  --local-dir data_source

# 4. 启动 Tree-TRM 训练 (适配服务器高性能卡)
# 注意：这里去掉了 DISABLE_COMPILE 和 forward_dtype=float32
# 并且恢复了 global_batch_size 为 512 (如果显存够大如 A100/H100 可设为 1024)

RUN_NAME="server_tree_trm_sudoku_full"

echo "Starting training: $RUN_NAME"

# 自动判断 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."

if [ "$NUM_GPUS" -gt 1 ]; then
    # 多卡训练
    torchrun --nproc_per_node=$NUM_GPUS pretrain.py \
        arch=tree_trm \
        data_paths="[data/sudoku-extreme-full]" \
        evaluators="[]" \
        epochs=50000 \
        eval_interval=5000 \
        lr=1e-4 \
        puzzle_emb_lr=1e-4 \
        weight_decay=1.0 \
        puzzle_emb_weight_decay=1.0 \
        arch.mlp_t=True \
        arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.H_cycles=3 \
        arch.L_cycles=6 \
        +run_name=${RUN_NAME} \
        ema=True \
        global_batch_size=512
else
    # 单卡训练
    python pretrain.py \
        arch=tree_trm \
        data_paths="[data/sudoku-extreme-full]" \
        evaluators="[]" \
        epochs=50000 \
        eval_interval=5000 \
        lr=1e-4 \
        puzzle_emb_lr=1e-4 \
        weight_decay=1.0 \
        puzzle_emb_weight_decay=1.0 \
        arch.mlp_t=True \
        arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.H_cycles=3 \
        arch.L_cycles=6 \
        +run_name=${RUN_NAME} \
        ema=True \
        global_batch_size=512
fi


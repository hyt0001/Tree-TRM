#!/bin/bash
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# Use a larger batch size (1024) as requested
# Using the updated tree_trm config (with increased cycles, branching factor, max steps)
# Using TF32 for speed
# Using Cross-Attention based tree selection (from previous code changes)

run_name="server_tree_trm_sudoku_large_v2"

echo "Starting training with larger capacity and Cross-Attention..."

nohup python pretrain.py \
    arch=tree_trm \
    data_paths="[data/sudoku-extreme-full]" \
    evaluators="[]" \
    epochs=100000 \
    eval_interval=2000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True \
    arch.pos_encodings=none \
    +run_name=${run_name} \
    ema=True \
    global_batch_size=1024 \
    > train_large_v2.log 2>&1 &

echo "Training started in background. Monitor with: tail -f train_large_v2.log"


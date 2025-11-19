#!/bin/bash
# Multi-turn SFT training script for Qwen 7B model
# This script trains a Qwen 7B model using FSDP (Fully Sharded Data Parallel) strategy
#
# Usage:
#   1. Update the configuration variables below (save_path, model path, data paths, etc.)
#   2. Set your WANDB_API_KEY environment variable or update it in this script
#   3. Run: bash run_qwen_7B.sh

set -e  # Exit on error
set -x  # Print commands

# ==================== Configuration ====================
# Project directory
PROJECT_DIR=$(pwd)

# Checkpoint save path
save_path=/path/to/checkpoints

# Model path (pretrained model directory)
model_path=/path/to/model

# Data paths
train_data_path=$PROJECT_DIR/data/sft/your_train_data.parquet
val_data_path=$PROJECT_DIR/data/sft/your_val_data.parquet

# WANDB configuration
export WANDB_API_KEY=your_wandb_api_key_here

# Training configuration
nproc_per_node=8  # Number of GPUs per node
nnodes=1          # Number of nodes

# ==================== Setup ====================
# Install dependencies
pip install -r requirements.txt

# Create log directory
log_dir=$save_path/logs
mkdir -p $log_dir
log_path=$log_dir/master.log

# ==================== Training ====================
echo "=== Training started at $(date) ===" | tee -a ${log_path}
echo "Model path: $model_path" | tee -a ${log_path}
echo "Save path: $save_path" | tee -a ${log_path}
echo "Train data: $train_data_path" | tee -a ${log_path}
echo "Val data: $val_data_path" | tee -a ${log_path}

torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data_path \
    data.val_files=$val_data_path \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=18000 \
    data.truncation=left \
    data.train_batch_size=256 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-7b-instruct \
    trainer.logger='[console,wandb]' \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    trainer.test_freq=-1 \
    trainer.save_freq=36 \
    trainer.total_epochs=6 \
    optim.lr=6e-6 2>&1 | tee -a ${log_path}

echo "=== Training completed at $(date) ===" | tee -a ${log_path}

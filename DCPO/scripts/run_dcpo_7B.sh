#!/bin/bash
# DCPO (Direct Contrastive Preference Optimization) training script for 7B model
# This script trains a 7B model using DCPO algorithm on a single node with 8x H100 GPUs
#
# Usage:
#   1. Update the configuration variables below (model path, data paths, log directory, etc.)
#   2. Set your WANDB_API_KEY environment variable or update it in this script
#   3. Make sure your current working directory is the root of the project
#   4. Run: bash run_dcpo_7B.sh

set -e  # Exit on error
set -x  # Print commands

# ==================== Configuration ====================
# Project directory
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/DCPO/config"

# Model configuration
model_size="7B"
model_path=/path/to/model  # Path to pretrained 7B model

# WANDB configuration
export WANDB_API_KEY="your_wandb_api_key_here"

# Environment variables
export VLLM_USE_V1=1

# Training configuration
train_batch_size=128
num_workers=64  # Number of rollout workers

# Dynamic batch size and token length configuration
use_dynamic_bsz=True
max_prompt_length=$((1024 * 4))      # 4096 tokens
max_response_length=$((1024 * 20))  # 20480 tokens
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 10 / 10))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))

# Project and logging
project_name="memagent_dcpo_single_node_7B"
logdir="/path/to/logs/${project_name}"
mkdir -p ${logdir}
logpath="${logdir}/master.log"

# Data paths
train_single=$PROJECT_DIR/data/Asearcher/your_train_single.parquet
val_single=$PROJECT_DIR/data/Asearcher/your_val_single.parquet
train_multi_object=$PROJECT_DIR/data/Asearcher/your_train_multi_object.parquet
val_multi_object=$PROJECT_DIR/data/Asearcher/your_val_multi_object.parquet

train_files="['$train_single', '$train_multi_object']"
test_files="['$val_single','$val_multi_object']"

# Tool configuration
tool_config_path=$PROJECT_DIR/DCPO/config/mem_search_tool_config_single.yaml

# ==================== Setup ====================
# Install dependencies
pip install shortuuid uuid
pip install -r requirements_sglang.txt
pip install numpy==1.26.4

# Increase file descriptor limit
ulimit -n 65535

# ==================== Training ====================
echo "=== Training started at $(date) ===" | tee -a ${logpath}
echo "Model size: ${model_size}" | tee -a ${logpath}
echo "Model path: ${model_path}" | tee -a ${logpath}
echo "Log directory: ${logdir}" | tee -a ${logpath}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mem_agent_loop_config' \
    algorithm.adv_estimator=dcpo \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=768 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +model.trust_remote_code=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=40 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=12 \
    +actor_rollout_ref.rollout.random_no_prune=True \
    +actor_rollout_ref.rollout.random_no_prune_prob=0.5 \
    +actor_rollout_ref.rollout.actual_n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.num_workers=$num_workers \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name='mem_agent_loop' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$logdir \
    reward_model.reward_manager=mem_reward \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    trainer.total_epochs=2 2>&1 | tee -a ${logpath}

echo "=== Training completed at $(date) ===" | tee -a ${logpath}

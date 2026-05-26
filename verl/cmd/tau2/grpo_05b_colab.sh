#!/usr/bin/env bash
# Qwen2.5-0.5B + GRPO + T3 on Tau2Bench-Telecom
# Tuned for Colab single A100 40GB, target ~4h for 100 steps.
set -euo pipefail

DATA_NAME="Tau2Bench"
DOMAIN_NAME="${DOMAIN_NAME:-telecom}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/${DATA_NAME}/${DOMAIN_NAME}}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_full_minus_test_9009_solo_think_short.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val_full_minus_test_9009_solo_think_short.parquet}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export USE_AREW_BONUS=false
export AREW_BONUS_MODE=step_abs_sum
export AREW_BONUS_SCALE=5.0
export AREW_NEGATIVE_STEP_MODE=full
export AREW_START_STEP=0
export AREW_END_STEP=-1

# ===== T3 settings =====
EARLY_CUT="${EARLY_CUT:-true}"            # set false to run vanilla GRPO baseline
TRUNC_STRENGTH="${TRUNC_STRENGTH:-8}"
HARD_TOLERATE="${HARD_TOLERATE:-999}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"

project_name="${DATA_NAME}-${DOMAIN_NAME}-Train"
experiment_name="${EXP_NAME:-solo-05B-T3}"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"
mkdir -p "${default_local_dir}"

python3 -m verl.trainer.main_ppo \
    data_name="${DATA_NAME}" \
    max_turns=20 \
    use_interactions=false \
    early_cut="${EARLY_CUT}" \
    +trunc_strength="${TRUNC_STRENGTH}" \
    +hard_tolerate_num="${HARD_TOLERATE}" \
    +tau2_strict_progress_label=true \
    +algorithm.arew_negative_step_mode="${AREW_NEGATIVE_STEP_MODE}" \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=64 \
    data.val_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=384 \
    data.max_start_length=4096 \
    data.truncation=middle \
    data.max_obs_length=1024 \
    \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.use_arew_bonus=false \
    algorithm.arew_bonus_mode=step_abs_sum \
    algorithm.arew_bonus_scale=5.0 \
    algorithm.arew_start_step=0 \
    algorithm.arew_end_step=-1 \
    \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    \
    trainer.logger='["console"]' \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_local_dir="${default_local_dir}" \
    2>&1 | tee "${default_local_dir}.log"

echo "Done: ${experiment_name}"

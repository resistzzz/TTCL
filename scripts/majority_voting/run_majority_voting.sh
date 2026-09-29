#!/usr/bin/env bash
set -euo pipefail
set -x

# Self-consistency majority-voting training. The training reward has no access
# to ground truth; labels in the parquet are retained only for W&B monitoring.
export WANDB_API_KEY="Your Wandb API Key"
export WANDB_MODE="online"
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
unset RAY_ADDRESS

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

LLM_PATH="${LLM_PATH:-path_to_Qwen3-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-data/amc23/amc23.parquet}"
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
DATA_NAME="${DATA_NAME:-AMC23}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
VAL_N="${VAL_N:-8}"
N="${N:-32}"
TRAIN_TEMP="${TRAIN_TEMP:-0.7}"
TEST_TEMP="${TEST_TEMP:-0.7}"

N_GPUS="${N_GPUS:-4}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-20}"

NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-False}"
LOSS_MODE="${LOSS_MODE:-vanilla}"
PROJECT_NAME="${PROJECT_NAME:-Calibration}"
EXP_NAME="${EXP_NAME:-MajorityVoting_${DATA_NAME}_${MODEL_NAME}_K${N}_MiniBs${PPO_MINI_BATCH_SIZE}_MicBs${PPO_MICRO_BATCH_SIZE}_TrTemp${TRAIN_TEMP}}"

mkdir -p wandb "rollout_log/${EXP_NAME}/train" "rollout_log/${EXP_NAME}/val"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.apply_chat_template_kwargs='{enable_thinking:false}' \
    actor_rollout_ref.model.path="${LLM_PATH}" \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="${TRAIN_TEMP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n="${N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=majority_voting \
    +reward_model.reward_kwargs.monitor_correctness=true \
    +reward_model.val_reward_manager=rlcr \
    +reward_model.val_reward_kwargs.format_reward=0.0 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="${TEST_TEMP}" \
    trainer.rollout_data_dir="rollout_log/${EXP_NAME}/train" \
    trainer.validation_data_dir="rollout_log/${EXP_NAME}/val" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    2>&1 | tee "wandb/${EXP_NAME}.log"

#!/usr/bin/env bash
set -euo pipefail
set -x

# Joint pseudo-correctness + calibration training.  Correctness uses the
# in-group majority answer; calibration uses agreement targets. Dataset labels
# are monitoring-only during training, while validation still uses RLCR.
export WANDB_API_KEY="Your Wandb API Key"
export WANDB_MODE="online"
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
unset RAY_ADDRESS

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

LLM_PATH="${LLM_PATH:-merged_hf_model/Calibration/RLCR_AdvNoStdNorm_Qwen3-4B_DAPO14k/global_step_330}"
MODEL_NAME="${MODEL_NAME:-RLCR-Qwen3-4B-Epoch3}"
TRAIN_FILES="${TRAIN_FILES:-data/aime24/aime24.parquet}"
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
DATA_NAME="${DATA_NAME:-AIME24}"

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
VAL_ECE_BINS="${VAL_ECE_BINS:-10}"

# frequency | loo | entropy_frequency | entropy_loo
TARGET_MODE="${TARGET_MODE:-frequency}"
case "${TARGET_MODE}" in
  frequency|loo|entropy_frequency|entropy_loo) ;;
  *) echo "Invalid TARGET_MODE: ${TARGET_MODE}" >&2; exit 1 ;;
esac
# Entropy modes shrink a noisy local support target toward an EMA global
# support prior: (1 - rho * H/log K) * support + rho * H/log K * prior.
ENTROPY_SHRINKAGE_RHO="${ENTROPY_SHRINKAGE_RHO:-0.5}"
# Historical weight in prior_t = beta * prior_(t-1) + (1-beta) * batch_prior.
PRIOR_EMA_BETA="${PRIOR_EMA_BETA:-0.95}"
# online: the historical behaviour, using the current rollout group's support.
# epoch_ema: initialize q_0 with one frozen-policy rollout pass, then use a
# per-question target cache that is only updated after each training epoch.
TARGET_CACHE_MODE="${TARGET_CACHE_MODE:-epoch_ema}"
TARGET_CACHE_EMA_RATE="${TARGET_CACHE_EMA_RATE:-0.01}"
case "${TARGET_CACHE_MODE}" in
  online|epoch_ema) ;;
  *) echo "Invalid TARGET_CACHE_MODE: ${TARGET_CACHE_MODE}" >&2; exit 1 ;;
esac
# confidence: malformed confidences get no calibration gradient.
# confidence_or_all: they receive the calibration format-learning fallback.
LOSS_MASK_MODE="${LOSS_MASK_MODE:-confidence_or_all}"
case "${LOSS_MASK_MODE}" in
  confidence|confidence_or_all) ;;
  *) echo "Invalid LOSS_MASK_MODE: ${LOSS_MASK_MODE}" >&2; exit 1 ;;
esac

# Correctness labels: policy is the in-group majority baseline;
# target_cache_sample samples one label from frozen per-question q_e(y | x);
# target_cache_majority deterministically selects argmax_y q_e(y | x).
# The cache is updated only at epoch boundaries by TARGET_CACHE_EMA_RATE.
PSEUDO_LABEL_SOURCE="${PSEUDO_LABEL_SOURCE:-target_cache_sample}"
case "${PSEUDO_LABEL_SOURCE}" in
  policy|target_cache_sample|target_cache_majority) ;;
  *) echo "Invalid PSEUDO_LABEL_SOURCE: ${PSEUDO_LABEL_SOURCE}" >&2; exit 1 ;;
esac
if [[ "${PSEUDO_LABEL_SOURCE}" != "policy" && "${TARGET_CACHE_MODE}" != "epoch_ema" ]]; then
    echo "cache-based pseudo labels require TARGET_CACHE_MODE=epoch_ema" >&2
    exit 1
fi

NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-False}"
LOSS_MODE="${LOSS_MODE:-vanilla}"
PROJECT_NAME="${PROJECT_NAME:-Calibration}"
if [[ "${TARGET_MODE}" == entropy_* ]]; then
    ENTROPY_EXP_SUFFIX="_Rho${ENTROPY_SHRINKAGE_RHO}-EmaBeta${PRIOR_EMA_BETA}"
else
    ENTROPY_EXP_SUFFIX=""
fi
if [[ "${TARGET_CACHE_MODE}" == "epoch_ema" ]]; then
    TARGET_CACHE_EXP_SUFFIX="_EpochTargetEma${TARGET_CACHE_EMA_RATE}"
else
    TARGET_CACHE_EXP_SUFFIX="_OnlineTarget"
fi
if [[ "${PSEUDO_LABEL_SOURCE}" == "target_cache_sample" ]]; then
    PSEUDO_LABEL_EXP_SUFFIX="_PSamp"
elif [[ "${PSEUDO_LABEL_SOURCE}" == "target_cache_majority" ]]; then
    PSEUDO_LABEL_EXP_SUFFIX="_PMajor"
else
    PSEUDO_LABEL_EXP_SUFFIX="_PPol"
fi
EXP_NAME="${EXP_NAME:-MajorityTTCL_${DATA_NAME}_${MODEL_NAME}_${TARGET_MODE}${ENTROPY_EXP_SUFFIX}${TARGET_CACHE_EXP_SUFFIX}${PSEUDO_LABEL_EXP_SUFFIX}_${LOSS_MASK_MODE}_K${N}}"

mkdir -p wandb "rollout_log/${EXP_NAME}/train" "rollout_log/${EXP_NAME}/val"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
    algorithm.use_kl_in_reward=false \
    reward_model.launch_reward_fn_async=false \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    +data.apply_chat_template_kwargs='{enable_thinking:false}' \
    actor_rollout_ref.model.path="${LLM_PATH}" \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="${TRAIN_TEMP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n="${N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    reward_model.reward_manager=majority_ttcl \
    +reward_model.ece_bins="${VAL_ECE_BINS}" \
    +reward_model.reward_kwargs.target_mode="${TARGET_MODE}" \
    +reward_model.reward_kwargs.entropy_shrinkage_rho="${ENTROPY_SHRINKAGE_RHO}" \
    +reward_model.reward_kwargs.prior_ema_beta="${PRIOR_EMA_BETA}" \
    +reward_model.reward_kwargs.target_cache_mode="${TARGET_CACHE_MODE}" \
    +reward_model.reward_kwargs.target_cache_ema_rate="${TARGET_CACHE_EMA_RATE}" \
    +reward_model.reward_kwargs.pseudo_label_source="${PSEUDO_LABEL_SOURCE}" \
    +reward_model.reward_kwargs.loss_mask_mode="${LOSS_MASK_MODE}" \
    +reward_model.reward_kwargs.monitor_correctness=true \
    +reward_model.val_reward_manager=rlcr \
    +reward_model.val_reward_kwargs.format_reward=0.0 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=true \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
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

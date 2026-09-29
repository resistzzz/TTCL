set -x

export WANDB_API_KEY="Your Wandb API Key"
export WANDB_MODE="online"
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

# This is a single-node job.  Do not inherit a previously exported Ray head.
unset RAY_ADDRESS

LLM_PATH=Path_to_Qwen3-4B
MODEL_NAME=Qwen3-4B
EXP_NAME=GRPO_Qwen3-4B_DAPO14k

train_files=data/dapo17k/train.parquet

val_files=data/amc23/amc23.parquet

reward_manager=naive
n=8
n_gpus=4
lr=1e-6

total_epochs=3

PROJ_NAME="Calibration"
LOSS_MODE=vanilla


mkdir -p wandb
mkdir -p rollout_log

# Force a fresh local Ray cluster instead of /tmp/ray/ray_current_cluster.
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs='{enable_thinking:false}' \
    actor_rollout_ref.model.path=$LLM_PATH \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$reward_manager \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    trainer.rollout_data_dir=rollout_log/$EXP_NAME/train \
    trainer.validation_data_dir=rollout_log/$EXP_NAME/val \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.save_freq=60 \
    trainer.test_freq=20 \
    trainer.total_epochs=$total_epochs \
    2>&1 | tee wandb/$EXP_NAME.log

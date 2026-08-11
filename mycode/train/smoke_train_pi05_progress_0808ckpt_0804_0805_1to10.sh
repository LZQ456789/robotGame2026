#!/usr/bin/env bash
# Smoke test for Experiment A (100 steps). Same hyperparams as full run except steps/log/save.
set -euo pipefail

eval "$(/root/anaconda3/bin/conda shell.posix hook)"
conda activate lerobot

export http_proxy="${http_proxy:-http://192.168.32.28:18000}"
export https_proxy="${https_proxy:-http://192.168.32.28:18000}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="localhost,127.0.0.1,::1,${MASTER_ADDR:-}"
export NO_PROXY="$no_proxy"

export HF_HOME=/media/jushen/kaka-liu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /root/workspace/lerobot

export TOKENIZERS_PARALLELISM=false
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Local single-node fallback when distributed env vars are absent.
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

num_processes=$((WORLD_SIZE * NPROC_PER_NODE))
# On a 1-GPU machine, do not request 8 processes.
visible_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "$WORLD_SIZE" == "1" && "$visible_gpus" -gt 0 && "$visible_gpus" -lt "$NPROC_PER_NODE" ]]; then
  NPROC_PER_NODE="$visible_gpus"
  num_processes=$((WORLD_SIZE * NPROC_PER_NODE))
fi

PRETRAINED_PATH=/media/jushen/kaka-liu/outputs/pi05_training0808_multigpu_step10000_bs32_data_0806_from_raw_0806_02_0807_from_raw_0807_pre_1/checkpoints/last/pretrained_model
DATASET_ROOT=/media/jushen/kaka-liu/datasets/lerobot_style/0804_0805_progress_1to10
OUTPUT_DIR=/media/jushen/kaka-liu/outputs/pi05_progress_0808ckpt_0804_0805_1to10_smoke100
JOB_NAME=kaka-liu_pi05_progress_0808ckpt_0804_0805_1to10_smoke100

echo "SMOKE A"
echo "WORLD_SIZE=$WORLD_SIZE NPROC_PER_NODE=$NPROC_PER_NODE num_processes=$num_processes"
echo "PRETRAINED_PATH=$PRETRAINED_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

accelerate launch \
  --same_network \
  --multi_gpu \
  --gpu_ids all \
  --num_machines "$WORLD_SIZE" \
  --num_processes "$num_processes" \
  --machine_rank "$RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  "$(which lerobot-train)" \
  --dataset.repo_id=lcarus3000/game260731 \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.discover_packages_path=lerobot.policies.pi05_progress \
  --policy.type=pi05_progress \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.repo_id=lcarus3000/pi05_progress_0808ckpt_0804_0805_1to10_smoke \
  --policy.pretrained_path="$PRETRAINED_PATH" \
  --policy.progress_target_mode=task_fraction \
  --policy.total_task_units=10 \
  --policy.boxes_per_segment=2 \
  --policy.progress_loss_weight=0.1 \
  --policy.enable_error_recovery=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --steps=100 \
  --policy.device=cuda \
  --batch_size=32 \
  --log_freq=10 \
  --save_freq=100 \
  --num_workers=2

#!/usr/bin/env bash
# Experiment B: official pi05 base + pi05_progress on 0804+0805 (1~10 boxes)
# Only pretrained_path differs from Experiment A.
set -euo pipefail

eval "$(/root/anaconda3/bin/conda shell.posix hook)"
conda activate lerobot

export http_proxy="${http_proxy:-http://192.168.32.28:18000}"
export https_proxy="${https_proxy:-http://192.168.32.28:18000}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="localhost,127.0.0.1,::1,${MASTER_ADDR:-}"
export NO_PROXY="$no_proxy"

# Do NOT hardcode HF_TOKEN here. Use the environment / existing HF login.
export HF_HOME=/media/jushen/kaka-liu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /root/workspace/lerobot

export TOKENIZERS_PARALLELISM=false
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

: "${WORLD_SIZE:?WORLD_SIZE is required (injected by the distributed launcher)}"
: "${RANK:?RANK is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

num_processes=$((WORLD_SIZE * NPROC_PER_NODE))

# Historically used Hub id in previous successful runs (e.g. 0806_..._base).
# Local cache snapshot (same revision):
#   /media/jushen/kaka-liu/hf_cache/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
PRETRAINED_PATH=lerobot/pi05_base
DATASET_ROOT=/media/jushen/kaka-liu/datasets/lerobot_style/0804_0805_progress_1to10
OUTPUT_DIR=/media/jushen/kaka-liu/outputs/pi05_progress_base_0804_0805_1to10_step10000_bs32
JOB_NAME=kaka-liu_pi05_progress_base_0804_0805_1to10_step10000_bs32

echo "WORLD_SIZE=$WORLD_SIZE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "num_processes=$num_processes"
echo "PRETRAINED_PATH=$PRETRAINED_PATH"
echo "DATASET_ROOT=$DATASET_ROOT"
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
  --policy.repo_id=lcarus3000/pi05_progress_base_0804_0805_1to10 \
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
  --steps=10000 \
  --policy.device=cuda \
  --batch_size=32 \
  --log_freq=20 \
  --save_freq=2000

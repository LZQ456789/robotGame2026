#!/usr/bin/env bash
# Task1 Subtask-only V1 CONTROL — BASE init, subtask prompts DISABLED (pair-only).
# Same data / sampler / steps / batch / optimizer / model as V1; only language differs.
# DO NOT execute this script as part of implementation validation.
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

: "${WORLD_SIZE:?WORLD_SIZE is required (injected by the distributed launcher)}"
: "${RANK:?RANK is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

num_processes=$((WORLD_SIZE * NPROC_PER_NODE))

PRETRAINED_PATH=lerobot/pi05_base
META_DIR=/media/jushen/kaka-liu/datasets/analysis/task1_subtask_v1
DATASET_ROOT=/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0804
OUTPUT_DIR=/media/jushen/kaka-liu/outputs/pi05_task1_subtask_v1_control_base_0804_0805_0806_step10000_bs32
JOB_NAME=kaka-liu_pi05_task1_subtask_v1_control_base_0804_0805_0806_step10000_bs32

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR already exists: $OUTPUT_DIR"
    echo "Refuse to start with resume=False over an existing directory."
    exit 1
fi

echo "WORLD_SIZE=$WORLD_SIZE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "num_processes=$num_processes"
echo "PRETRAINED_PATH=$PRETRAINED_PATH"
echo "META_DIR=$META_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "task1_subtask_enable=false (CONTROL)"

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
  --dataset.repo_id=task1_subtask_v1 \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.task1_subtask_meta_dir="$META_DIR" \
  --dataset.task1_subtask_enable=false \
  --dataset.task1_subtask_pair_balance=true \
  --policy.type=pi05 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.repo_id=lcarus3000/pi05_task1_subtask_v1_control_base \
  --policy.pretrained_path="$PRETRAINED_PATH" \
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

#!/usr/bin/env python3
"""Single-process smoke: load pi05 weights into pi05_progress and run a few train steps.

Checks:
  - plugin discovery
  - non-strict checkpoint load (only progress_head missing)
  - progress targets present via episode_index/task_index/(frame_index|index)
  - action_loss / progress_loss finite
  - progress_head receives gradients and parameter updates

Usage examples:
  # Experiment A
  python mycode/checks/smoke_pi05_progress_train_step.py \
    --pretrained_path /media/jushen/kaka-liu/outputs/.../checkpoints/last/pretrained_model \
    --steps 3 --batch_size 1

  # Experiment B
  python mycode/checks/smoke_pi05_progress_train_step.py \
    --pretrained_path lerobot/pi05_base \
    --steps 3 --batch_size 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.parser import load_plugin
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument(
        "--dataset_root",
        default="/media/jushen/kaka-liu/datasets/lerobot_style/0804_0805_progress_1to10",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_plugin("lerobot.policies.pi05_progress")
    cfg_cls = PreTrainedConfig.get_choice_class("pi05_progress")

    root = Path(args.dataset_root).resolve()
    ds_meta = LeRobotDatasetMetadata(repo_id=root.name, root=str(root))

    cfg = cfg_cls(
        device=args.device,
        dtype="bfloat16" if args.device.startswith("cuda") else "float32",
        gradient_checkpointing=True,
        compile_model=False,
        pretrained_path=args.pretrained_path,
        progress_target_mode="task_fraction",
        total_task_units=10,
        boxes_per_segment=2,
        progress_loss_weight=0.1,
        enable_error_recovery=False,
        freeze_vision_encoder=False,
        train_expert_only=False,
        push_to_hub=False,
        repo_id="local/pi05_progress_smoke",
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        "observation.images.camera_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        "observation.images.camera_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        "observation.images.camera_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))}

    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=str(root),
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
    )

    print(f"Loading policy from {args.pretrained_path}")
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.train()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        dataset_stats=dataset.meta.stats,
        pretrained_path=args.pretrained_path,
        dataset_meta=dataset.meta,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
    )
    it = iter(loader)

    # Only optimize progress_head for memory-light update verification; full
    # optimizer for all params is done in the accelerate smoke scripts.
    progress_params = [p for n, p in policy.named_parameters() if "progress_head" in n and p.requires_grad]
    if not progress_params:
        raise RuntimeError("No trainable progress_head parameters found")
    optim = torch.optim.AdamW(progress_params, lr=2.5e-5)

    before = {n: p.detach().float().clone() for n, p in policy.named_parameters() if "progress_head" in n}
    pred_means = []

    for step in range(1, args.steps + 1):
        batch = next(it)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        # Ensure metadata still present after processor round-trip.
        for key in ("episode_index", "task_index"):
            if key not in batch:
                raise KeyError(f"{key} missing after preprocessor; progress targets will fail")
        if "frame_index" not in batch and "index" not in batch:
            raise KeyError("neither frame_index nor index present after preprocessor")
        # Processor may leave some tensors on CPU depending on loaded device config.
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(args.device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        loss, loss_dict = policy.forward(batch)
        loss.backward()

        grads = []
        for p in progress_params:
            if p.grad is None:
                raise RuntimeError("progress_head grad is None")
            grads.append(float(p.grad.detach().float().norm()))
        optim.step()

        pred_means.append(loss_dict["progress_pred_mean"])
        print(
            f"step={step} loss={loss_dict['loss']:.6f} "
            f"action_loss={loss_dict['action_loss']:.6f} "
            f"progress_loss={loss_dict['progress_loss']:.6f} "
            f"progress_pred_mean={loss_dict['progress_pred_mean']:.6f} "
            f"progress_target_mean={loss_dict['progress_target_mean']:.6f} "
            f"progress_grad_norm={sum(grads):.6f}"
        )
        for name, value in loss_dict.items():
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                raise RuntimeError(f"NaN/Inf in {name}: {value}")

    after = {n: p.detach().float() for n, p in policy.named_parameters() if "progress_head" in n}
    deltas = {n: float((after[n] - before[n]).abs().max()) for n in before}
    print("progress_head max abs param delta:", deltas)
    if not any(v > 0 for v in deltas.values()):
        raise RuntimeError("progress_head parameters did not update")
    if len(set(round(x, 6) for x in pred_means)) == 1 and args.steps > 1:
        print("WARNING: progress_pred_mean stayed constant across smoke steps (may be OK for tiny steps)")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"cuda max_memory_allocated_gb={mem_gb:.2f}")
    print("SMOKE TRAIN STEP PASSED")


if __name__ == "__main__":
    main()

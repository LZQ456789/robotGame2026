#!/usr/bin/env python3
"""Read-only Progress target sanity check for medicine-box 1~10 experiments.

Usage:
    python mycode/checks/progress_target_sanity_check.py \
        --dataset_root /media/jushen/kaka-liu/datasets/lerobot_style/0804_0805_progress_1to10 \
        --total_task_units 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch

from lerobot.configs.parser import load_plugin
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi05_progress.modeling_pi05_progress import PI05ProgressPolicy
from lerobot.policies.pi05_progress.task_text import parse_task_text, pair_bounds_from_parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/media/jushen/kaka-liu/datasets/lerobot_style/0804_0805_progress_1to10",
    )
    parser.add_argument("--total_task_units", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.02)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    load_plugin("lerobot.policies.pi05_progress")
    cfg_cls = PreTrainedConfig.get_choice_class("pi05_progress")
    meta = LeRobotDatasetMetadata(repo_id=root.name, root=str(root))
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    eps = pq.read_table(sorted((root / "meta" / "episodes").rglob("*.parquet"))[0]).to_pandas()

    print("=== tasks ===")
    for task_text, row in tasks.iterrows():
        parsed = parse_task_text(str(task_text))
        bounds = pair_bounds_from_parsed(parsed)
        print(
            f"task_index={int(row['task_index'])} | {task_text!r} | "
            f"first={parsed.first_box} second={parsed.second_box} "
            f"before={parsed.completed_before} after={parsed.completed_after} bounds={bounds}"
        )
        if bounds is None:
            raise SystemExit(f"Unparseable task: {task_text!r}")

    stub = type("Stub", (), {})()
    stub.config = cfg_cls(
        device="cpu",
        total_task_units=args.total_task_units,
        boxes_per_segment=2,
        progress_loss_weight=0.1,
        enable_error_recovery=False,
        progress_target_mode="task_fraction",
        push_to_hub=False,
        repo_id="local/progress_sanity",
    )
    stub._episode_lengths, stub._episode_starts = PI05ProgressPolicy._read_episode_metadata(meta)
    stub._task_progress_bounds = PI05ProgressPolicy._read_task_progress_bounds(meta)
    stub._progress_q01 = torch.empty(0)
    stub._progress_q99 = torch.empty(0)

    expected = {
        0: (0.0, 0.1, 0.2),
        1: (0.2, 0.3, 0.4),
        2: (0.4, 0.5, 0.6),
        3: (0.6, 0.7, 0.8),
        4: (0.8, 0.9, 1.0),
    }
    task_by_idx = {int(r["task_index"]): t for t, r in tasks.iterrows()}

    print("\n=== start / middle / end targets ===")
    for ti in range(5):
        task_text = task_by_idx[ti]
        ep_rows = eps[eps["tasks"].apply(lambda xs, tt=task_text: tt in xs)]
        ep_idx = int(ep_rows.iloc[0]["episode_index"])
        length = int(ep_rows.iloc[0]["length"])
        frames = [0, length // 2, length - 1]
        batch = {
            "episode_index": torch.tensor([ep_idx] * 3),
            "frame_index": torch.tensor(frames),
            "task_index": torch.tensor([ti] * 3),
        }
        targets = PI05ProgressPolicy._get_progress_targets(stub, batch, 3).tolist()
        print(f"\n{task_text} (task_index={ti}, ep={ep_idx}, len={length})")
        for label, fi, tgt, want in zip(
            ["start", "middle", "end"], frames, targets, expected[ti], strict=True
        ):
            frac = fi / (length - 1)
            ok = abs(tgt - want) <= args.tol
            print(
                f"  {label}: episode_index={ep_idx} frame_index={fi} "
                f"episode_fraction={frac:.4f} progress_target={tgt:.4f} "
                f"want≈{want:.2f} OK={ok}"
            )
            if not ok:
                raise SystemExit("Progress target out of tolerance; do not start training.")

    print("\nPROGRESS TARGET SANITY CHECK PASSED")


if __name__ == "__main__":
    main()

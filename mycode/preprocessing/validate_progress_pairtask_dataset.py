#!/usr/bin/env python3
"""Read-only validation for Progress pairtask LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.policies.pi05_progress.task_text import parse_task_text  # noqa: E402


def _as_scalar(value):
    """Unwrap nested list/array scalars from parquet stats columns."""
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return value


def load_tasks(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta/tasks.parquet")
    data = table.to_pydict()
    if "task_index" in data and "task" in data:
        return {int(i): str(t) for i, t in zip(data["task_index"], data["task"], strict=True)}
    # fallback: task as index
    raise RuntimeError(f"Unexpected tasks schema: {table.column_names}")


def load_episodes(root: Path):
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise RuntimeError("No episode parquet files")
    tables = [pq.read_table(f) for f in files]
    # concatenate manually
    cols = tables[0].column_names
    merged = {c: [] for c in cols}
    for t in tables:
        d = t.to_pydict()
        for c in cols:
            merged[c].extend(d[c])
    return merged


def progress_target(completed_before: float, completed_after: float, frac: float, total: float = 18.0) -> float:
    low = completed_before / total
    high = completed_after / total
    return low + (high - low) * frac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--total_task_units", type=float, default=18.0)
    args = parser.parse_args()
    root = args.root

    info = json.loads((root / "meta/info.json").read_text())
    tasks = load_tasks(root)
    eps = load_episodes(root)

    print("=== DATASET ===")
    print("path:", root)
    print("episodes:", info.get("total_episodes"))
    print("frames:", info.get("total_frames"))
    print()
    print("=== TASKS ===")
    for idx in sorted(tasks):
        print(f"{idx} | {tasks[idx]}")
    print()

    print("=== PARSE_TASK_TEXT ===")
    parse_ok = True
    for idx in sorted(tasks):
        text = tasks[idx]
        parsed = parse_task_text(text)
        print(
            f"{text}\n"
            f"  first_box={parsed.first_box} second_box={parsed.second_box} "
            f"completed_before={parsed.completed_before} completed_after={parsed.completed_after}"
        )
        if not parsed.has_pair or parsed.completed_before is None or parsed.completed_after is None:
            parse_ok = False
            print("  PARSE FAILED")
    print("parse_all_ok:", parse_ok)
    print()

    # episode task consistency via meta episodes.tasks (list) and stats/task_index
    print("=== EPISODE TASK CHECK ===")
    n = len(eps["episode_index"])
    pair_counts: Counter[str] = Counter()
    mismatches = 0
    for i in range(n):
        ep_idx = int(eps["episode_index"][i])
        task_list = eps["tasks"][i]
        if isinstance(task_list, list):
            task_text = task_list[0] if task_list else ""
        else:
            task_text = str(task_list)
        # find task_index
        matches = [k for k, v in tasks.items() if v == task_text]
        if not matches:
            mismatches += 1
            print(f"episode {ep_idx}: unknown task {task_text!r}")
            continue
        task_index = matches[0]
        tmin = int(_as_scalar(eps["stats/task_index/min"][i]))
        tmax = int(_as_scalar(eps["stats/task_index/max"][i]))
        if tmin != tmax or tmin != task_index:
            mismatches += 1
            print(
                f"episode {ep_idx}: task_index mismatch meta={task_index} "
                f"stats_min={tmin} stats_max={tmax} task={task_text}"
            )
        parsed = parse_task_text(task_text)
        if parsed.has_pair:
            pair_counts[f"{parsed.first_box}-{parsed.second_box}"] += 1
    print("episodes:", n, "mismatches:", mismatches)
    print("pair distribution:")
    for k in sorted(pair_counts, key=lambda x: int(x.split("-")[0])):
        print(f"  {k}: {pair_counts[k]}")
    print()

    print("=== PROGRESS GT SANITY (task_fraction, total=18) ===")
    # one episode per pair
    seen_pairs: set[str] = set()
    for i in range(n):
        task_list = eps["tasks"][i]
        task_text = task_list[0] if isinstance(task_list, list) else str(task_list)
        parsed = parse_task_text(task_text)
        if not parsed.has_pair:
            continue
        pair = f"{parsed.first_box}-{parsed.second_box}"
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        length = int(eps["length"][i])
        ep_idx = int(eps["episode_index"][i])
        for label, frame in [
            ("start", 0),
            ("middle", max(0, length // 2)),
            ("end", max(0, length - 1)),
        ]:
            frac = 0.0 if length <= 1 else frame / (length - 1)
            tgt = progress_target(
                parsed.completed_before,
                parsed.completed_after,
                frac,
                total=args.total_task_units,
            )
            print(
                f"{pair} ep={ep_idx} {label}: frame={frame}/{length-1} "
                f"frac={frac:.3f} progress={tgt:.3f}"
            )
        print()

    expected = {f"{i}-{i+1}" for i in range(1, 18, 2)}
    missing = sorted(expected - set(pair_counts), key=lambda x: int(x.split("-")[0]))
    print("missing pairs:", missing or "(none)")
    print("validation_done")


if __name__ == "__main__":
    main()

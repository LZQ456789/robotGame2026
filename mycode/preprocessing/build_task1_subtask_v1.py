#!/usr/bin/env python3
"""Build Task1 Subtask-only V1 offline labels + analysis sidecars.

Does NOT modify original 0804/0805/0806 LeRobot datasets.
Writes metadata under /media/jushen/kaka-liu/datasets/analysis/task1_subtask_v1/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycode.task1_subtask_v1.common import (  # noqa: E402
    ACTION_HORIZON_DEFAULT,
    DATASETS,
    DEFAULT_META_DIR,
    FPS,
    VALID_PAIRS,
    parse_pair,
)
from mycode.task1_subtask_v1.labeling import (  # noqa: E402
    apply_action_chunk_safety,
    label_episode,
)


def load_episode_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet under {root}/meta/episodes")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index")


def load_states(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet under {root}/data")
    dfs = [
        pd.read_parquet(
            f,
            columns=["episode_index", "frame_index", "index", "observation.state", "task_index"],
        )
        for f in files
    ]
    return pd.concat(dfs, ignore_index=True)


def task_text_from_ep(ep_row: pd.Series) -> str:
    t = ep_row.get("tasks")
    if isinstance(t, (list, np.ndarray)):
        return str(t[0]) if len(t) else ""
    return str(t)


def verify_dataset_assumptions(root: Path, expected_episodes: int, expected_frames: int) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    errors = []
    if info.get("fps") != FPS:
        errors.append(f"fps={info.get('fps')} != {FPS}")
    st = info["features"]["observation.state"]
    ac = info["features"]["action"]
    if list(st["shape"]) != [16]:
        errors.append(f"state shape {st['shape']} != [16]")
    if list(ac["shape"]) != [16]:
        errors.append(f"action shape {ac['shape']} != [16]")
    names = st.get("names") or []
    if len(names) >= 16:
        if names[14] != "end_effector_left_position_align":
            errors.append(f"state[14] name={names[14]!r}")
        if names[15] != "end_effector_right_position_align":
            errors.append(f"state[15] name={names[15]!r}")
    cams = sorted(k for k in info["features"] if k.startswith("observation.images"))
    expected_cams = [
        "observation.images.camera_head",
        "observation.images.camera_left",
        "observation.images.camera_right",
    ]
    if cams != expected_cams:
        errors.append(f"cameras={cams}")
    if info.get("total_episodes") != expected_episodes:
        errors.append(f"episodes={info.get('total_episodes')} != {expected_episodes}")
    if info.get("total_frames") != expected_frames:
        errors.append(f"frames={info.get('total_frames')} != {expected_frames}")
    if errors:
        raise RuntimeError(f"Dataset assumption failure for {root}: " + "; ".join(errors))
    return {
        "fps": info["fps"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "cameras": cams,
        "state_dim": 16,
        "action_dim": 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_META_DIR))
    parser.add_argument("--action_horizon", type=int, default=ACTION_HORIZON_DEFAULT)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    action_horizon = int(args.action_horizon)

    frame_rows: list[dict] = []
    ep_rows: list[dict] = []
    dist_counter: dict[tuple, dict] = defaultdict(lambda: {"operation_frames": 0, "valid_train_anchors": 0})
    short_subtasks_report: list[dict] = []
    assumption_report = {}
    global_offset = 0  # cumulative frames across sources for combined indexing helpers

    for ds in DATASETS:
        date, source = ds["date"], ds["source"]
        root = Path(ds["root"])
        print(f"==> Labeling {date} ({source})")
        assumption_report[date] = verify_dataset_assumptions(
            root, ds["expected_episodes"], ds["expected_frames"]
        )
        ep_table = load_episode_table(root)
        df = load_states(root)
        if len(df) != ds["expected_frames"]:
            raise RuntimeError(f"{date}: loaded {len(df)} frames != expected {ds['expected_frames']}")

        for ep_idx, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            states = np.stack(g["observation.state"].to_numpy())
            ep_len = len(g)
            ep_row = ep_table.loc[ep_table.episode_index == int(ep_idx)].iloc[0]
            original_task = task_text_from_ep(ep_row)
            try:
                pair_lo, pair_hi = parse_pair(original_task)
            except ValueError as e:
                ep_rows.append(
                    {
                        "source_date": date,
                        "source_episode_index": int(ep_idx),
                        "pair": "",
                        "num_frames": ep_len,
                        "num_valid_operation_frames": 0,
                        "num_valid_train_anchors": 0,
                        "num_cycles": 0,
                        "valid": False,
                        "reason": str(e),
                    }
                )
                continue

            result = label_episode(states, original_task, pair_lo, pair_hi)
            if not result.valid:
                ep_rows.append(
                    {
                        "source_date": date,
                        "source_episode_index": int(ep_idx),
                        "pair": f"{pair_lo}-{pair_hi}",
                        "num_frames": ep_len,
                        "num_valid_operation_frames": 0,
                        "num_valid_train_anchors": 0,
                        "num_cycles": len(result.cycles),
                        "valid": False,
                        "reason": result.reason,
                    }
                )
                # Still emit frames with invalid flags for auditability
                for local_i, (_, row) in enumerate(g.iterrows()):
                    frame_rows.append(
                        {
                            "source_date": date,
                            "source_root": str(root),
                            "source_episode_index": int(ep_idx),
                            "frame_index": int(row["frame_index"]),
                            "global_index_in_source": int(row["index"]),
                            "combined_frame_offset": global_offset + int(row["index"]),
                            "original_task": original_task,
                            "pair_lo": pair_lo,
                            "pair_hi": pair_hi,
                            "box_id": int(result.box_id[local_i]),
                            "mode": str(result.mode[local_i]),
                            "subtask_id": str(result.subtask_id[local_i]),
                            "prompt": str(result.prompt_baseline[local_i]),
                            "prompt_subtask": str(result.prompt_subtask[local_i]),
                            "prompt_baseline": str(result.prompt_baseline[local_i]),
                            "confidence": "INVALID",
                            "is_terminal_idle": bool(result.is_terminal_idle[local_i]),
                            "is_post_place_variation": bool(result.is_post_place_variation[local_i]),
                            "is_valid_operation_frame": False,
                            "is_valid_train_anchor": False,
                            "episode_valid": False,
                        }
                    )
                continue

            anchors, safety = apply_action_chunk_safety(
                result.is_valid_operation_frame,
                result.box_id,
                result.subtask_id,
                result.is_terminal_idle,
                result.is_post_place_variation,
                action_horizon=action_horizon,
            )
            for sid, st in safety["subtask_stats"].items():
                if st["shorter_than_horizon"]:
                    short_subtasks_report.append(
                        {
                            "source_date": date,
                            "episode_index": int(ep_idx),
                            "pair": f"{pair_lo}-{pair_hi}",
                            "subtask_id": sid,
                            **st,
                            "action_horizon": action_horizon,
                        }
                    )

            n_op = int(result.is_valid_operation_frame.sum())
            n_anc = int(anchors.sum())
            ep_rows.append(
                {
                    "source_date": date,
                    "source_episode_index": int(ep_idx),
                    "pair": f"{pair_lo}-{pair_hi}",
                    "num_frames": ep_len,
                    "num_valid_operation_frames": n_op,
                    "num_valid_train_anchors": n_anc,
                    "num_cycles": len(result.cycles),
                    "valid": True,
                    "reason": "OK",
                    "terminal_idle_frames": int(result.is_terminal_idle.sum()),
                    "post_place_frames": int(result.is_post_place_variation.sum()),
                    "chunk_safety_violations_if_unfiltered": safety["violations_if_unfiltered"],
                }
            )

            for local_i, (_, row) in enumerate(g.iterrows()):
                sid = str(result.subtask_id[local_i])
                bid = int(result.box_id[local_i])
                is_op = bool(result.is_valid_operation_frame[local_i])
                is_anc = bool(anchors[local_i])
                prompt = (
                    str(result.prompt_subtask[local_i])
                    if is_op and sid
                    else str(result.prompt_baseline[local_i])
                )
                frame_rows.append(
                    {
                        "source_date": date,
                        "source_root": str(root),
                        "source_episode_index": int(ep_idx),
                        "frame_index": int(row["frame_index"]),
                        "global_index_in_source": int(row["index"]),
                        "combined_frame_offset": global_offset + int(row["index"]),
                        "original_task": original_task,
                        "pair_lo": pair_lo,
                        "pair_hi": pair_hi,
                        "box_id": bid,
                        "mode": str(result.mode[local_i]),
                        "subtask_id": sid,
                        "prompt": prompt,
                        "prompt_subtask": str(result.prompt_subtask[local_i]),
                        "prompt_baseline": str(result.prompt_baseline[local_i]),
                        "confidence": str(result.confidence[local_i]) or "NONE",
                        "is_terminal_idle": bool(result.is_terminal_idle[local_i]),
                        "is_post_place_variation": bool(result.is_post_place_variation[local_i]),
                        "is_valid_operation_frame": is_op,
                        "is_valid_train_anchor": is_anc,
                        "episode_valid": True,
                    }
                )
                if is_op and sid and bid > 0:
                    key = (f"{pair_lo}-{pair_hi}", bid, sid, date)
                    dist_counter[key]["operation_frames"] += 1
                    if is_anc:
                        dist_counter[key]["valid_train_anchors"] += 1

        global_offset += int(assumption_report[date]["total_frames"])

    frames_df = pd.DataFrame(frame_rows)
    eps_df = pd.DataFrame(ep_rows)
    dist_rows = [
        {
            "pair": k[0],
            "box_id": k[1],
            "subtask_id": k[2],
            "source_date": k[3],
            "operation_frames": v["operation_frames"],
            "valid_train_anchors": v["valid_train_anchors"],
        }
        for k, v in sorted(dist_counter.items())
    ]
    dist_df = pd.DataFrame(dist_rows)

    # Aggregate summary
    valid_eps = eps_df[eps_df["valid"]]
    invalid_eps = eps_df[~eps_df["valid"]]
    valid_frames = frames_df[frames_df["episode_valid"]]
    op_frames = int(valid_frames["is_valid_operation_frame"].sum())
    anc_frames = int(valid_frames["is_valid_train_anchor"].sum())
    term_frames = int(valid_frames["is_terminal_idle"].sum())
    post_frames = int(valid_frames["is_post_place_variation"].sum())

    retention_by = {}
    for col in ["source_date", "pair_lo", "box_id", "subtask_id"]:
        if col == "pair_lo":
            tmp = valid_frames.copy()
            tmp["pair"] = tmp["pair_lo"].astype(str) + "-" + tmp["pair_hi"].astype(str)
            group_col = "pair"
        else:
            tmp = valid_frames
            group_col = col
        rows = []
        for key, g in tmp.groupby(group_col):
            if group_col in {"box_id", "subtask_id"} and (key == -1 or key == ""):
                continue
            op_n = int(g["is_valid_operation_frame"].sum())
            an_n = int(g["is_valid_train_anchor"].sum())
            rows.append(
                {
                    group_col: key if not hasattr(key, "item") else (key.item() if hasattr(key, "item") else key),
                    "operation_frames": op_n,
                    "valid_train_anchors": an_n,
                    "retention_pct": (100.0 * an_n / op_n) if op_n else 0.0,
                }
            )
        retention_by[group_col] = rows

    # Global short-subtask check: any subtask type with ZERO anchors and all runs < horizon
    subtask_global = defaultdict(lambda: {"operation_frames": 0, "valid_train_anchors": 0, "max_run": 0})
    for r in dist_rows:
        sg = subtask_global[r["subtask_id"]]
        sg["operation_frames"] += r["operation_frames"]
        sg["valid_train_anchors"] += r["valid_train_anchors"]

    # Compute max contiguous run duration per subtask globally from frames
    for sid, g in valid_frames[valid_frames["is_valid_operation_frame"]].groupby("subtask_id"):
        # approximate max duration via episode-wise runs
        max_run = 0
        for (_, _ep), gg in g.groupby(["source_date", "source_episode_index"]):
            idx = np.sort(gg["frame_index"].to_numpy())
            if len(idx) == 0:
                continue
            breaks = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks + 1, len(idx)]
            for a, b in zip(starts, ends, strict=True):
                max_run = max(max_run, int(idx[b - 1] - idx[a] + 1))
        subtask_global[sid]["max_run"] = max_run

    unusable_short = {
        sid: st
        for sid, st in subtask_global.items()
        if st["operation_frames"] > 0 and st["valid_train_anchors"] == 0 and st["max_run"] < action_horizon
    }

    summary = {
        "action_horizon": action_horizon,
        "pi05_chunk_size": action_horizon,
        "action_delta_indices": list(range(action_horizon)),
        "datasets": assumption_report,
        "episodes_total": int(len(eps_df)),
        "episodes_valid": int(len(valid_eps)),
        "episodes_invalid": int(len(invalid_eps)),
        "invalid_episode_reasons": invalid_eps["reason"].value_counts().to_dict() if len(invalid_eps) else {},
        "valid_box_cycles": int(valid_eps["num_cycles"].sum()) if len(valid_eps) else 0,
        "operation_frames": op_frames,
        "valid_train_anchors": anc_frames,
        "anchor_retention_pct": (100.0 * anc_frames / op_frames) if op_frames else 0.0,
        "terminal_idle_frames": term_frames,
        "post_place_variation_frames": post_frames,
        "chunk_safety_rule": (
            "anchor t valid iff frames t..t+H-1 share same (box_id, subtask_id), "
            "are operation frames, and stay inside the episode"
        ),
        "retention_by": retention_by,
        "subtask_global": {k: dict(v) for k, v in subtask_global.items()},
        "unusable_short_subtasks": unusable_short,
        "supported_pairs": [f"{a}-{b}" for a, b in sorted(VALID_PAIRS)],
        "source_roots": [d["root"] for d in DATASETS],
    }

    frames_path = out_dir / "task1_subtask_v1_frames.parquet"
    eps_path = out_dir / "task1_subtask_v1_episodes.csv"
    dist_path = out_dir / "task1_subtask_v1_distribution.csv"
    summary_path = out_dir / "task1_subtask_v1_summary.json"

    frames_df.to_parquet(frames_path, index=False)
    # Keep source_date as zero-padded strings in CSV (avoid 0804 -> 804).
    eps_df = eps_df.copy()
    eps_df["source_date"] = eps_df["source_date"].astype(str).str.zfill(4)
    if "source_date" in dist_df.columns:
        dist_df = dist_df.copy()
        dist_df["source_date"] = dist_df["source_date"].astype(str).str.zfill(4)
    eps_df.to_csv(eps_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n===== SUMMARY =====")
    print(json.dumps({k: summary[k] for k in [
        "episodes_total", "episodes_valid", "episodes_invalid", "valid_box_cycles",
        "operation_frames", "valid_train_anchors", "anchor_retention_pct",
        "terminal_idle_frames", "post_place_variation_frames", "unusable_short_subtasks",
    ]}, indent=2))
    print(f"Wrote {frames_path}")
    print(f"Wrote {eps_path}")
    print(f"Wrote {dist_path}")
    print(f"Wrote {summary_path}")

    if unusable_short:
        raise SystemExit(
            "FATAL: one or more semantic subtasks have max duration < action_horizon "
            f"and zero valid train anchors: {unusable_short}"
        )
    if int(summary["episodes_invalid"]) > 0:
        print("WARNING: some episodes invalid — see episodes.csv")
    if anc_frames == 0:
        raise SystemExit("FATAL: zero valid train anchors after chunk-safety filtering")


if __name__ == "__main__":
    main()

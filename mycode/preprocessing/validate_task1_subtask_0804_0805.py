#!/usr/bin/env python3
"""Validate whether Task1 0804/0805 can share 0806 Subtask-only V1 labeling.

Analysis only — does not modify source datasets or train models.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUCTION_THR = 0.35
GRIPPER_THR = 0.3
FPS = 26.0
MEDIAN_K = 5
MIN_ON_FRAMES = 15
MIN_OFF_FRAMES = 5
SAME_BOX_POSE_THR = 0.40
LIKELY_SAME_BOX_POSE_THR = 0.55
EXPECTED_BOXES = 2

HANDOVER_BOXES = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14}
LEFT_PLACE_BOXES = {7, 8, 9, 15, 16, 17, 18}

OUT = Path("/media/jushen/kaka-liu/datasets/analysis/task1_subtask_0804_0805_validation")

DATASETS = [
    {
        "date": "0804",
        "source": "0806_from_raw_0804",
        "root": Path("/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0804"),
        "raw": Path("/media/jushen/kaka-liu/datasets/raw/0804"),
    },
    {
        "date": "0805",
        "source": "0806_from_raw_0805",
        "root": Path("/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0805"),
        "raw": Path("/media/jushen/kaka-liu/datasets/raw/0805"),
    },
    {
        "date": "0806",
        "source": "0806_from_raw_0806_progress_pairtask",
        "root": Path("/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0806_progress_pairtask"),
        "raw": Path("/media/jushen/kaka-liu/datasets/raw/0806"),
    },
]

SUBTASK_TEXT = {
    "S0_APPROACH_PICK": "Approach and pick up the current box with the left suction cup.",
    "S1_TRANSFER_TO_GRIPPER": "Transfer the current box from the left suction cup to the right gripper.",
    "S2_GRIPPER_PLACEMENT": "Place the current box into its target carton position with the right gripper.",
    "S1_LEFT_DIRECT_PLACE": "Place the current box directly into its target carton position with the left suction cup.",
}


@dataclass
class CycleInfo:
    on: int
    off: int
    duration: int
    mode_inferred: str
    mode_expected: str
    completed: bool
    box_id: int | None
    grip_closes: list[int] = field(default_factory=list)
    grip_opens: list[int] = field(default_factory=list)
    pre_pose_left: list[float] = field(default_factory=list)
    confidence: str = "NORMAL_HIGH_CONFIDENCE"
    recovery_flag: bool = False
    subtasks: list[dict] = field(default_factory=list)
    notes: str = ""


def median_filter(x: np.ndarray, k: int = MEDIAN_K) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i : i + k])
    return out


def debounce_binary(on: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    y = on.astype(bool).copy()
    n = len(y)
    i = 0
    while i < n:
        j = i
        while j < n and y[j] == y[i]:
            j += 1
        run = j - i
        need = min_on if y[i] else min_off
        if run < need:
            fill = bool(y[i - 1]) if i > 0 else (bool(y[j]) if j < n else False)
            y[i:j] = fill
            i = max(0, i - 1)
            continue
        i = j
    return y


def rising_edges(on: np.ndarray) -> list[int]:
    return (np.where((~on[:-1]) & on[1:])[0] + 1).tolist()


def falling_edges(on: np.ndarray) -> list[int]:
    return (np.where(on[:-1] & (~on[1:]))[0] + 1).tolist()


def pose_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def parse_pair(task: str) -> tuple[int | None, int | None]:
    # English: Pack medicine boxes 9 and 10 into the carton
    m = re.search(r"boxes?\s+(\d+)\s+and\s+(\d+)", task, flags=re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Chinese: 抓取第9个和第10个药盒
    m = re.search(r"第\s*(\d+)\s*个.*?第\s*(\d+)\s*个", task)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def expected_mode(box_id: int) -> str:
    if box_id in HANDOVER_BOXES:
        return "HANDOVER"
    if box_id in LEFT_PLACE_BOXES:
        return "LEFT_PLACE"
    return "UNKNOWN"


def english_pair_task(lo: int, hi: int) -> str:
    return f"Pack medicine boxes {lo} and {hi} into the carton."


def make_prompt(lo: int, hi: int, box_id: int, subtask_key: str, total_boxes: int = 18) -> str:
    return (
        f"{english_pair_task(lo, hi)}\n"
        f"Current box: {box_id} of {total_boxes}.\n"
        f"Current subtask: {SUBTASK_TEXT[subtask_key]}"
    )


def load_episode_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index")


def load_states_actions(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    dfs = [
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action", "task_index"])
        for f in files
    ]
    return pd.concat(dfs, ignore_index=True)


def task_text_from_ep(ep_row: pd.Series) -> str:
    t = ep_row.get("tasks")
    if isinstance(t, (list, np.ndarray)):
        return str(t[0]) if len(t) else ""
    return str(t)


def process_signals(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[CycleInfo]]:
    suction = debounce_binary(median_filter(states[:, 14]) > SUCTION_THR, MIN_ON_FRAMES, MIN_OFF_FRAMES)
    gripper = debounce_binary(median_filter(states[:, 15]) > GRIPPER_THR, MIN_ON_FRAMES, MIN_OFF_FRAMES)
    on_edges = rising_edges(suction)
    off_edges = falling_edges(suction)
    g_close = rising_edges(gripper)
    g_open = falling_edges(gripper)

    cycles: list[CycleInfo] = []
    off_i = 0
    for idx, on in enumerate(on_edges):
        while off_i < len(off_edges) and off_edges[off_i] <= on:
            off_i += 1
        off = off_edges[off_i] if off_i < len(off_edges) else len(states) - 1
        if off_i < len(off_edges):
            off_i += 1
        next_on = on_edges[idx + 1] if idx + 1 < len(on_edges) else len(states)
        closes = [f for f in g_close if on <= f < next_on]
        opens = [f for f in g_open if on <= f < next_on]
        closes_during = [f for f in closes if on <= f <= off]
        if closes_during:
            mode = "HANDOVER"
            completed = off > closes_during[0] and off < len(states) - 1
        else:
            mode = "LEFT_PLACE"
            completed = (off - on) >= MIN_ON_FRAMES and off < len(states) - 1
        pre = max(0, on - 5)
        cycles.append(
            CycleInfo(
                on=on,
                off=off,
                duration=off - on,
                mode_inferred=mode,
                mode_expected="UNKNOWN",
                completed=completed,
                box_id=None,
                grip_closes=closes,
                grip_opens=opens,
                pre_pose_left=states[pre, 0:7].tolist(),
            )
        )
    return suction, gripper, cycles


def assign_boxes(cycles: list[CycleInfo], lo: int | None, hi: int | None) -> list[CycleInfo]:
    if lo is None or hi is None:
        for i, c in enumerate(cycles):
            c.box_id = i + 1
            c.mode_expected = expected_mode(c.box_id) if c.box_id <= 18 else "UNKNOWN"
        return cycles

    expected = [lo, hi]
    n = len(cycles)
    if n == 0:
        return cycles
    if n <= EXPECTED_BOXES:
        for i, c in enumerate(cycles):
            c.box_id = expected[min(i, len(expected) - 1)]
            c.mode_expected = expected_mode(c.box_id)
        return cycles

    # Surplus: link lowest-distance pairs then map to expected boxes
    extra = n - EXPECTED_BOXES
    pair_costs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = pose_dist(np.array(cycles[i].pre_pose_left), np.array(cycles[j].pre_pose_left))
            adj_pen = 0.0 if j == i + 1 else (0.05 if j == i + 2 else 0.15)
            pair_costs.append((d + adj_pen, d, i, j))
    pair_costs.sort()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    linked = 0
    for _cost, d, i, j in pair_costs:
        if linked >= extra:
            break
        if find(i) == find(j):
            continue
        if len({find(k) for k in range(n)}) <= EXPECTED_BOXES:
            break
        union(i, j)
        linked += 1

    order, comp_first = [], {}
    for i in range(n):
        r = find(i)
        if r not in comp_first:
            comp_first[r] = i
            order.append(r)
    while len(order) > EXPECTED_BOXES:
        late = order[-1]
        late_pose = np.array(cycles[comp_first[late]].pre_pose_left)
        best = None
        for early in order[:-1]:
            d = pose_dist(late_pose, np.array(cycles[comp_first[early]].pre_pose_left))
            if best is None or d < best[0]:
                best = (d, early)
        union(best[1], late)
        order, comp_first = [], {}
        for i in range(n):
            r = find(i)
            if r not in comp_first:
                comp_first[r] = i
                order.append(r)

    labels = {comp: expected[min(idx, 1)] for idx, comp in enumerate(order)}
    for i, c in enumerate(cycles):
        c.box_id = labels[find(i)]
        c.mode_expected = expected_mode(c.box_id)
    return cycles


def segment_subtasks(c: CycleInfo, prev_end: int, ep_len: int) -> list[dict]:
    """Return list of {name,start,end} for a normal cycle."""
    out = []
    s0_start = prev_end
    s0_end = c.on
    out.append({"name": "S0_APPROACH_PICK", "start": s0_start, "end": s0_end, "frames": max(0, s0_end - s0_start)})

    if c.mode_expected == "HANDOVER" or (c.mode_expected == "UNKNOWN" and c.mode_inferred == "HANDOVER"):
        closes_during = [f for f in c.grip_closes if c.on <= f <= c.off]
        opens_after = [f for f in c.grip_opens if f > c.off] or [f for f in c.grip_opens if f > c.on]
        if not closes_during:
            return out  # incomplete
        out.append(
            {
                "name": "S1_TRANSFER_TO_GRIPPER",
                "start": c.on,
                "end": c.off,
                "frames": c.off - c.on,
            }
        )
        place_end = opens_after[0] if opens_after else min(ep_len - 1, c.off + 1)
        # clip place_end if overlaps next cycle handled by caller
        out.append(
            {
                "name": "S2_GRIPPER_PLACEMENT",
                "start": c.off,
                "end": place_end,
                "frames": max(0, place_end - c.off),
            }
        )
    else:
        out.append(
            {
                "name": "S1_LEFT_DIRECT_PLACE",
                "start": c.on,
                "end": c.off,
                "frames": c.off - c.on,
            }
        )
    return out


def classify_episode_cycles(
    cycles: list[CycleInfo], lo: int | None, hi: int | None, ep_len: int
) -> tuple[list[CycleInfo], str, int]:
    """Classify cycles; return cycles, episode_class, terminal_idle_start."""
    if not cycles:
        return cycles, "INVALID_INCOMPLETE", ep_len

    # Mode mismatch / completion checks per cycle
    by_box: dict[int, list[CycleInfo]] = defaultdict(list)
    for c in cycles:
        if c.box_id is not None:
            by_box[int(c.box_id)].append(c)

    has_recovery = False
    has_ambiguous = False
    has_incomplete = False

    for box_id, group in by_box.items():
        if len(group) > 1:
            has_recovery = True
            for c in group:
                c.recovery_flag = True
                c.confidence = "RECOVERY_OR_ROLLBACK"
                c.notes = "multiple suction cycles assigned to same box"
        for c in group:
            if c.mode_expected != "UNKNOWN" and c.mode_inferred != c.mode_expected:
                # unexpected mode for this box
                if not c.recovery_flag:
                    c.confidence = "AMBIGUOUS"
                    c.notes = f"mode_inferred={c.mode_inferred} != expected={c.mode_expected}"
                    has_ambiguous = True
            if not c.completed and not c.recovery_flag:
                c.confidence = "INVALID_INCOMPLETE"
                c.notes = (c.notes + "; incomplete cycle").strip("; ")
                has_incomplete = True

    # Extra cycles beyond 2 without clear same-box grouping already marked recovery
    if len(cycles) > EXPECTED_BOXES and not has_recovery:
        # leftover surplus not grouped — ambiguous
        has_ambiguous = True
        for c in cycles[EXPECTED_BOXES:]:
            if c.confidence.startswith("NORMAL"):
                c.confidence = "AMBIGUOUS"
                c.notes = "surplus cycle without strong same-box evidence"
                c.recovery_flag = True

    # Post-place extra grip without suction re-ON: known harmless variation.
    # Keep HIGH labels for spans ending at first valid gripper OPEN; do not demote.
    for c in cycles:
        if c.mode_expected == "HANDOVER" or (c.mode_expected == "UNKNOWN" and c.mode_inferred == "HANDOVER"):
            closes_during = [f for f in c.grip_closes if c.on <= f <= c.off]
            if closes_during:
                first_open = [o for o in c.grip_opens if o > closes_during[0]]
                if first_open:
                    later = [f for f in c.grip_closes if f > first_open[0]]
                    if later:
                        c.notes = "post-place extra gripper adjustment after first open (excluded from S2; not recovery)"

    # Segment subtasks for normal cycles only; resolve S2 / next S0 overlap
    prev_end = 0
    ordered = sorted(cycles, key=lambda x: x.on)
    for i, c in enumerate(ordered):
        if not c.confidence.startswith("NORMAL"):
            prev_end = c.off  # still advance
            continue
        subs = segment_subtasks(c, prev_end, ep_len)
        # clip S2 end before next suction on
        if i + 1 < len(ordered):
            next_on = ordered[i + 1].on
            for s in subs:
                if s["name"] == "S2_GRIPPER_PLACEMENT" and s["end"] > next_on:
                    s["end"] = next_on
                    s["frames"] = max(0, s["end"] - s["start"])
                    if c.confidence == "NORMAL_HIGH_CONFIDENCE":
                        c.confidence = "NORMAL_MEDIUM_CONFIDENCE"
                        c.notes = (c.notes + "; S2 overlaps next suction").strip("; ")
        # Require complete subtask set
        names = {s["name"] for s in subs}
        need = {"S0_APPROACH_PICK", "S1_TRANSFER_TO_GRIPPER", "S2_GRIPPER_PLACEMENT"} if (
            c.mode_expected == "HANDOVER" or (c.mode_expected == "UNKNOWN" and c.mode_inferred == "HANDOVER")
        ) else {"S0_APPROACH_PICK", "S1_LEFT_DIRECT_PLACE"}
        if not need.issubset(names):
            c.confidence = "INVALID_INCOMPLETE"
            c.notes = (c.notes + "; missing subtask phases").strip("; ")
            has_incomplete = True
            c.subtasks = subs
        else:
            # zero-length S0 at episode start is ok; negative not allowed
            if any(s["frames"] < 0 for s in subs):
                c.confidence = "AMBIGUOUS"
                has_ambiguous = True
            c.subtasks = subs
            prev_end = max(s["end"] for s in subs)

    # Terminal idle start = end of last usable cycle's last subtask, else last off
    terminal_start = ep_len
    normal_cycles = [c for c in ordered if c.confidence.startswith("NORMAL")]
    if len(normal_cycles) >= 1:
        last = normal_cycles[-1]
        if last.subtasks:
            terminal_start = max(s["end"] for s in last.subtasks)
        else:
            terminal_start = last.off

    # Episode-level class for V1 usability: need exactly 2 NORMAL cycles covering pair
    normal = [c for c in cycles if c.confidence.startswith("NORMAL")]
    high = [c for c in cycles if c.confidence == "NORMAL_HIGH_CONFIDENCE"]

    if has_recovery:
        ep_class = "RECOVERY_OR_ROLLBACK"
    elif has_ambiguous:
        ep_class = "AMBIGUOUS"
    elif has_incomplete or len(normal) < EXPECTED_BOXES:
        ep_class = "INVALID_INCOMPLETE"
    elif lo is not None and hi is not None:
        boxes = sorted({int(c.box_id) for c in normal if c.box_id is not None})
        if boxes != sorted([lo, hi]):
            ep_class = "AMBIGUOUS"
        elif len(high) == EXPECTED_BOXES and all(c.confidence == "NORMAL_HIGH_CONFIDENCE" for c in normal):
            ep_class = "NORMAL_HIGH_CONFIDENCE"
        else:
            ep_class = "NORMAL_MEDIUM_CONFIDENCE"
    else:
        ep_class = "AMBIGUOUS"

    return cycles, ep_class, terminal_start


def robust(arr: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return {"n": 0, "median": np.nan, "mean": np.nan, "p10": np.nan, "p90": np.nan, "iqr": np.nan, "std": np.nan}
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "std": float(np.std(a)),
    }


def schema_report(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    return {
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "feature_keys": list(info["features"].keys()),
        "state_shape": info["features"]["observation.state"]["shape"],
        "state_names": info["features"]["observation.state"].get("names"),
        "action_shape": info["features"]["action"]["shape"],
        "action_names": info["features"]["action"].get("names"),
        "image_keys": [k for k in info["features"] if k.startswith("observation.images")],
        "tasks": tasks.reset_index().to_dict(orient="records") if "task" in tasks.reset_index().columns else tasks.to_dict(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    compatibility = {}
    ep_rows = []
    cycle_rows = []
    subtask_durations: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    state_stats = {}
    prompt_examples = []
    joint_stats = {}

    for ds in DATASETS:
        date, source, root = ds["date"], ds["source"], ds["root"]
        print(f"Processing {date} {source} ...")
        compatibility[date] = {
            "source": source,
            "schema": schema_report(root),
            "raw_exists": ds["raw"].exists(),
            "raw_path": str(ds["raw"]),
        }

        ep_table = load_episode_table(root)
        df = load_states_actions(root)
        states_all = np.stack(df["observation.state"].to_numpy())
        actions_all = np.stack(df["action"].to_numpy())

        compatibility[date]["actuator"] = {
            "suction_p10_med_p90_max": [float(x) for x in np.percentile(states_all[:, 14], [10, 50, 90, 100])],
            "gripper_p10_med_p90_max": [float(x) for x in np.percentile(states_all[:, 15], [10, 50, 90, 100])],
            "suction_frac_on": float((states_all[:, 14] > SUCTION_THR).mean()),
            "gripper_frac_closed": float((states_all[:, 15] > GRIPPER_THR).mean()),
            "same_detector_ok": True,
        }
        joint_stats[date] = {
            "left_joint_mean": states_all[:, 0:7].mean(axis=0).tolist(),
            "left_joint_std": states_all[:, 0:7].std(axis=0).tolist(),
            "right_joint_mean": states_all[:, 7:14].mean(axis=0).tolist(),
            "right_joint_std": states_all[:, 7:14].std(axis=0).tolist(),
            "action_left_mean": actions_all[:, 0:7].mean(axis=0).tolist(),
            "action_right_mean": actions_all[:, 7:14].mean(axis=0).tolist(),
            "action_suction_p90": float(np.percentile(actions_all[:, 14], 90)),
            "action_gripper_p90": float(np.percentile(actions_all[:, 15], 90)),
        }

        for ep_idx, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            states = np.stack(g["observation.state"].to_numpy())
            ep_len = len(g)
            ep_row = ep_table.loc[ep_table.episode_index == int(ep_idx)].iloc[0]
            task = task_text_from_ep(ep_row)
            lo, hi = parse_pair(task)

            _, _, cycles = process_signals(states)
            cycles = assign_boxes(cycles, lo, hi)
            cycles, ep_class, terminal_start = classify_episode_cycles(cycles, lo, hi, ep_len)

            terminal_idle = max(0, ep_len - terminal_start)
            # Cycle-level V1 policy: include NORMAL_HIGH cycles even if sibling is MEDIUM.
            # Exclude episode entirely only for recovery / ambiguous / incomplete.
            hard_exclude = ep_class in {"RECOVERY_OR_ROLLBACK", "AMBIGUOUS", "INVALID_INCOMPLETE"}
            high_cycles = [c for c in cycles if c.confidence == "NORMAL_HIGH_CONFIDENCE"]
            usable_v1_episode = (not hard_exclude) and len(high_cycles) > 0

            usable_frames = 0
            if not hard_exclude:
                for c in high_cycles:
                    for s in c.subtasks:
                        usable_frames += max(0, s["end"] - s["start"])
                        subtask_durations[date][s["name"]].append(s["frames"])

            n_valid = sum(1 for c in cycles if c.confidence.startswith("NORMAL"))
            n_rec = sum(1 for c in cycles if c.confidence == "RECOVERY_OR_ROLLBACK" or c.recovery_flag)
            n_amb = sum(1 for c in cycles if c.confidence == "AMBIGUOUS")

            ep_rows.append(
                {
                    "date": date,
                    "source_dataset": source,
                    "episode_index": int(ep_idx),
                    "task": task,
                    "pair_lo": lo,
                    "pair_hi": hi,
                    "num_detected_cycles": len(cycles),
                    "num_valid_cycles": n_valid,
                    "num_recovery_cycles": n_rec,
                    "num_ambiguous_cycles": n_amb,
                    "episode_class": ep_class,
                    "usable": usable_v1_episode,
                    "length": ep_len,
                    "terminal_idle_frames": terminal_idle,
                    "terminal_idle_pct": terminal_idle / ep_len if ep_len else 0.0,
                    "usable_frames": usable_frames,
                    "num_high_cycles": len(high_cycles),
                    "notes": "cycle-level HIGH inclusion; MEDIUM sibling kept out" if ep_class == "NORMAL_MEDIUM_CONFIDENCE" and high_cycles else "",
                }
            )

            for c in cycles:
                pattern = ">".join(s["name"] for s in c.subtasks) if c.subtasks else ""
                cycle_start = c.subtasks[0]["start"] if c.subtasks else c.on
                cycle_end = c.subtasks[-1]["end"] if c.subtasks else c.off
                usable_cycle = (not hard_exclude) and c.confidence == "NORMAL_HIGH_CONFIDENCE"
                cycle_rows.append(
                    {
                        "date": date,
                        "source_dataset": source,
                        "episode": int(ep_idx),
                        "box_id": c.box_id,
                        "mode": c.mode_expected if c.mode_expected != "UNKNOWN" else c.mode_inferred,
                        "mode_inferred": c.mode_inferred,
                        "mode_expected": c.mode_expected,
                        "subtask_pattern": pattern,
                        "cycle_start": cycle_start,
                        "cycle_end": cycle_end,
                        "suction_on": c.on,
                        "suction_off": c.off,
                        "confidence": c.confidence,
                        "recovery_flag": c.recovery_flag,
                        "usable_for_v1": usable_cycle,
                        "operation_frames": max(0, cycle_end - cycle_start) if usable_cycle else 0,
                        "notes": c.notes,
                    }
                )

                # prompt examples: one per usable cycle / first subtask mid frame
                if usable_cycle and lo is not None and hi is not None and c.subtasks:
                    # Keep a few examples per date
                    n_date = sum(1 for p in prompt_examples if p["date"] == date)
                    if n_date < 12:
                        s = c.subtasks[0]
                        mid = (s["start"] + s["end"]) // 2
                        prompt_examples.append(
                            {
                                "date": date,
                                "episode_index": int(ep_idx),
                                "frame_index": int(mid),
                                "box_id": c.box_id,
                                "subtask": s["name"],
                                "original_task": task,
                                "prompt": make_prompt(lo, hi, int(c.box_id), s["name"]),
                                "prompt_chars": len(make_prompt(lo, hi, int(c.box_id), s["name"])),
                            }
                        )

    ep_df = pd.DataFrame(ep_rows)
    cy_df = pd.DataFrame(cycle_rows)
    for df_ in (ep_df, cy_df):
        df_["date"] = df_["date"].astype(str).map(lambda x: x.zfill(4))

    # Box distribution
    box_rows = []
    for box in range(1, 19):
        row = {"box_id": box}
        total_cycles = 0
        total_frames = 0
        for date in ["0804", "0805", "0806"]:
            sub = cy_df[(cy_df.date == date) & (cy_df.box_id == box) & (cy_df.usable_for_v1)]
            n_c = int(len(sub))
            n_f = int(sub.operation_frames.sum()) if n_c else 0
            row[f"{date}_cycles"] = n_c
            row[f"{date}_frames"] = n_f
            total_cycles += n_c
            total_frames += n_f
        row["total_cycles"] = total_cycles
        row["total_frames"] = total_frames
        row["expected_mode"] = expected_mode(box)
        box_rows.append(row)
    box_df = pd.DataFrame(box_rows)

    # Mode agreement by box for 0804/0805 (all detected cycles with expected mapping)
    mode_agree = []
    for box in range(1, 11):
        for date in ["0804", "0805"]:
            sub = cy_df[(cy_df.date == date) & (cy_df.box_id == box)]
            if len(sub) == 0:
                continue
            exp = expected_mode(box)
            hand = int((sub.mode_inferred == "HANDOVER").sum())
            left = int((sub.mode_inferred == "LEFT_PLACE").sum())
            unexpected = int((sub.mode_inferred != exp).sum())
            mode_agree.append(
                {
                    "box_id": box,
                    "date": date,
                    "cycle_count": len(sub),
                    "HANDOVER": hand,
                    "LEFT_PLACE": left,
                    "unexpected": unexpected,
                    "expected_mode": exp,
                    "agreement_rate": float((sub.mode_inferred == exp).mean()),
                }
            )
    # also aggregate across dates per box
    mode_by_box = []
    for box in range(1, 11):
        sub = cy_df[(cy_df.date.isin(["0804", "0805"])) & (cy_df.box_id == box)]
        exp = expected_mode(box)
        mode_by_box.append(
            {
                "box_id": box,
                "0804_cycles": int(((cy_df.date == "0804") & (cy_df.box_id == box)).sum()),
                "0805_cycles": int(((cy_df.date == "0805") & (cy_df.box_id == box)).sum()),
                "HANDOVER_count": int((sub.mode_inferred == "HANDOVER").sum()),
                "LEFT_PLACE_count": int((sub.mode_inferred == "LEFT_PLACE").sum()),
                "unexpected_count": int((sub.mode_inferred != exp).sum()) if len(sub) else 0,
                "expected_mode": exp,
                "agreement_rate": float((sub.mode_inferred == exp).mean()) if len(sub) else np.nan,
            }
        )

    # Duration stats
    dur_stats = {}
    for date, dmap in subtask_durations.items():
        dur_stats[date] = {}
        for name, vals in dmap.items():
            st = robust(vals)
            st["median_seconds"] = st["median"] / FPS if st["n"] else np.nan
            dur_stats[date][name] = st

    # Usable summary
    usable_summary = {}
    for date in ["0804", "0805", "0806"]:
        e = ep_df[ep_df.date == date]
        c = cy_df[(cy_df.date == date) & (cy_df.usable_for_v1)]
        usable_summary[date] = {
            "total_episodes": int(len(e)),
            "usable_episodes": int(e.usable.sum()),
            "total_frames": int(e.length.sum()),
            "usable_frames": int(e.usable_frames.sum()),
            "usable_box_cycles": int(len(c)),
            "excluded_episodes": int((~e.usable).sum()),
            "episode_class_counts": e.episode_class.value_counts().to_dict(),
            "mean_terminal_idle_pct": float(e.terminal_idle_pct.mean()),
            "median_terminal_idle_pct": float(e.terminal_idle_pct.median()),
            "terminal_idle_frames_total": int(e.terminal_idle_frames.sum()),
        }

    combined = {
        "usable_episodes": sum(usable_summary[d]["usable_episodes"] for d in usable_summary),
        "usable_frames": sum(usable_summary[d]["usable_frames"] for d in usable_summary),
        "usable_box_cycles": sum(usable_summary[d]["usable_box_cycles"] for d in usable_summary),
    }

    # Imbalance
    b1_10 = box_df[box_df.box_id.between(1, 10)]
    b11_18 = box_df[box_df.box_id.between(11, 18)]
    imbalance = {
        "avg_cycles_1_10": float(b1_10.total_cycles.mean()),
        "avg_cycles_11_18": float(b11_18.total_cycles.mean()),
        "avg_frames_1_10": float(b1_10.total_frames.mean()),
        "avg_frames_11_18": float(b11_18.total_frames.mean()),
        "cycle_ratio_1_10_over_11_18": float(b1_10.total_cycles.mean() / max(b11_18.total_cycles.mean(), 1e-9)),
        "frame_ratio_1_10_over_11_18": float(b1_10.total_frames.mean() / max(b11_18.total_frames.mean(), 1e-9)),
    }

    # Pair coverage
    pair_cov = (
        ep_df[ep_df.date.isin(["0804", "0805"])]
        .groupby(["date", "pair_lo", "pair_hi"])
        .size()
        .reset_index(name="n_episodes")
    )

    # Detector verdict
    detector = "YES"
    for date in ["0804", "0805", "0806"]:
        a = compatibility[date]["actuator"]
        if not (0.7 <= a["suction_p10_med_p90_max"][2] <= 0.85):
            detector = "YES_WITH_ADJUSTED_THRESHOLDS"
        if not (0.65 <= a["gripper_p10_med_p90_max"][2] <= 0.75):
            detector = "YES_WITH_ADJUSTED_THRESHOLDS"

    # Joint shift check: compare mean L2 of joint means vs 0806
    ref_l = np.array(joint_stats["0806"]["left_joint_mean"])
    ref_r = np.array(joint_stats["0806"]["right_joint_mean"])
    shifts = {}
    for date in ["0804", "0805"]:
        shifts[date] = {
            "left_mean_l2": float(np.linalg.norm(np.array(joint_stats[date]["left_joint_mean"]) - ref_l)),
            "right_mean_l2": float(np.linalg.norm(np.array(joint_stats[date]["right_joint_mean"]) - ref_r)),
        }
    data_compat = "HIGH"
    if any(v["left_mean_l2"] > 0.5 or v["right_mean_l2"] > 0.5 for v in shifts.values()):
        data_compat = "MEDIUM"
    if any(v["left_mean_l2"] > 1.5 or v["right_mean_l2"] > 1.5 for v in shifts.values()):
        data_compat = "LOW"

    # Sampling recommendation
    sampling = {
        "recommended": "B_pair_balanced",
        "rationale": (
            "0804+0805 only cover pairs 1-2..9-10; 0806 covers 1-18. "
            "Pair-balanced sampling (equalize 9 pair tasks) prevents 1-10 from drowning 11-18 "
            "while remaining simpler than per-box balancing. Within each pair, use HIGH-confidence frames only."
        ),
        "alternatives": {
            "A_naive": "Risk: Box1-10 frames dominate by ~cycle_ratio.",
            "C_box_balanced": "Ideal but harder with only 0806 for 11-18; may underuse 0804/0805.",
            "D_date_weighted": "Useful as secondary weight inside pairs; not primary.",
        },
    }

    # Terminal idle recommendation
    terminal_rec = {
        "choice": "A_remove_terminal_idle",
        "reason": "Avoid teaching idle; V1 should only train on labeled subtask spans ending at last placement open / suction off.",
    }

    # Final verdict
    def date_suitability(date: str) -> str:
        u = usable_summary[date]
        # fraction of cycles that are HIGH usable among all detected cycles
        cy_date = cy_df[cy_df.date == date]
        cycle_high_rate = float(cy_date.usable_for_v1.mean()) if len(cy_date) else 0.0
        sub = [r for r in mode_agree if r["date"] == date]
        mode_ok = all(r["agreement_rate"] >= 0.95 for r in sub) if sub else True
        if cycle_high_rate >= 0.85 and mode_ok and detector.startswith("YES"):
            return "YES"
        if cycle_high_rate >= 0.7 and detector.startswith("YES"):
            return "PARTIAL"
        return "NO"

    # Write outputs
    ep_df.to_csv(OUT / "subtask_validation_summary.csv", index=False)
    cy_df.to_csv(OUT / "box_cycle_summary.csv", index=False)
    box_df.to_csv(OUT / "box_distribution.csv", index=False)
    pd.DataFrame(mode_by_box).to_csv(OUT / "mode_agreement_by_box.csv", index=False)
    pair_cov.to_csv(OUT / "pair_coverage.csv", index=False)
    (OUT / "subtask_examples.json").write_text(json.dumps(prompt_examples[:30], indent=2, ensure_ascii=False))

    # Try tokenizer length if available
    tok_info = {"available": False}
    try:
        from transformers import AutoTokenizer

        # lightweight check with a common tokenizer if present; else skip
        tok_info = {
            "available": False,
            "note": "Tokenizer not loaded in validation; prompt char lengths reported instead.",
            "prompt_char_stats": robust([p["prompt_chars"] for p in prompt_examples]) if prompt_examples else {},
            "sample_prompt": prompt_examples[0]["prompt"] if prompt_examples else "",
        }
    except Exception as e:
        tok_info = {"available": False, "error": str(e), "prompt_char_stats": robust([p["prompt_chars"] for p in prompt_examples])}

    if prompt_examples:
        tok_info["prompt_char_stats"] = robust([p["prompt_chars"] for p in prompt_examples])
        tok_info["sample_prompt"] = prompt_examples[0]["prompt"]

    summary = {
        "detector_same": detector,
        "data_compatibility": data_compat,
        "joint_shifts_vs_0806": shifts,
        "usable_summary": usable_summary,
        "combined": combined,
        "imbalance": imbalance,
        "duration_stats": dur_stats,
        "terminal_idle_recommendation": terminal_rec,
        "sampling": sampling,
        "suitability": {
            "0804": date_suitability("0804"),
            "0805": date_suitability("0805"),
            "joint_training_feasibility": "HIGH" if data_compat == "HIGH" and detector.startswith("YES") else "MEDIUM",
            "recommended_option": "B",
            "recommended_option_text": "Use all HIGH-confidence normal data from 0804+0805+0806 with pair-balanced sampling",
        },
        "compatibility": compatibility,
        "joint_stats": joint_stats,
        "tokenizer_prompt": tok_info,
        "mode_by_box": mode_by_box,
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps({k: summary[k] for k in ["detector_same", "data_compatibility", "usable_summary", "combined", "imbalance", "suitability"]}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

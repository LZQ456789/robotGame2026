#!/usr/bin/env python3
"""Mine Task1 error-recovery candidates from 0806/0807 without modifying source data.

Primary evidence: observation.state[14] (left suction), state[15] (right gripper).
Arm joints and sampled video frames are supporting evidence only.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image

SUCTION_THR = 0.35
GRIPPER_THR = 0.3
FPS = 26.0
MIN_ON_FRAMES = 15  # debounce short pulses
MIN_OFF_FRAMES = 5
MEDIAN_K = 5
HANDOVER_BOXES_0806 = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14}
LEFT_PLACE_BOXES_0806 = {7, 8, 9, 15, 16, 17, 18}

OUT_DIR = Path("/media/jushen/kaka-liu/datasets/analysis/task1_recovery_0806_0807")
FRAME_DIR = OUT_DIR / "visual_samples"

DATASETS = [
    {
        "date": "0806",
        "source_dataset": "0806_from_raw_0806_progress_pairtask",
        "root": Path("/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0806_progress_pairtask"),
        "has_box_ids": True,
        "raw_note": "converted from /media/jushen/kaka-liu/datasets/raw/0806 (27 nested trajectory.hdf5)",
    },
    {
        "date": "0806",
        "source_dataset": "0806_from_raw_0806_02",
        "root": Path("/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0806_02"),
        "has_box_ids": False,
        "raw_note": "converted from /media/jushen/kaka-liu/datasets/raw/0806_02 flat HDF5 (no pair labels)",
    },
    {
        "date": "0807",
        "source_dataset": "0807_from_raw_0807",
        "root": Path("/media/jushen/kaka-liu/datasets/lerobot_style/0807_from_raw_0807"),
        "has_box_ids": False,
        "raw_note": "raw/0807 missing on this machine; LeRobot-only (46 eps, generic task text)",
    },
]


@dataclass
class Event:
    kind: str
    frame: int


@dataclass
class SuctionCycle:
    on: int
    off: int
    duration: int
    grip_closes: list[int] = field(default_factory=list)
    grip_opens: list[int] = field(default_factory=list)
    mode: str = "UNKNOWN"
    completed: bool = False
    box_id: int | str | None = None
    left_motion: float = 0.0
    right_motion: float = 0.0
    pre_pose_left: list[float] = field(default_factory=list)
    pre_pose_right: list[float] = field(default_factory=list)


@dataclass
class RecoveryCandidate:
    date: str
    source_dataset: str
    episode_index: int
    task_text: str
    box_id: Any
    mode: str
    recovery_type: str
    start_frame: int
    end_frame: int
    first_attempt_frame: int
    retry_frame: int
    suction_event_pattern: str
    gripper_event_pattern: str
    confidence: str
    visual_support: str
    notes: str
    arm_support: str = ""
    signal_evidence: str = ""
    events: list[dict] = field(default_factory=list)


def median_filter(x: np.ndarray, k: int = MEDIAN_K) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i : i + k])
    return out


def debounce_binary(on: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    """Enforce minimum ON/OFF run lengths."""
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
            # merge into surrounding state if possible
            fill = False if y[i] else True
            if i == 0 and j < n:
                fill = bool(y[j])
            elif i > 0:
                fill = bool(y[i - 1])
            y[i:j] = fill
            # restart near previous boundary
            i = max(0, i - 1)
            continue
        i = j
    return y


def rising_edges(on: np.ndarray) -> list[int]:
    return (np.where((~on[:-1]) & on[1:])[0] + 1).tolist()


def falling_edges(on: np.ndarray) -> list[int]:
    return (np.where(on[:-1] & (~on[1:]))[0] + 1).tolist()


def parse_boxes(task_text: str) -> list[int]:
    # "Pack medicine boxes 1 and 2 into the carton"
    m = re.search(r"boxes?\s+(\d+)\s+and\s+(\d+)", task_text, flags=re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return []


def mode_for_box_0806(box_id: int) -> str:
    if box_id in HANDOVER_BOXES_0806:
        return "HANDOVER"
    if box_id in LEFT_PLACE_BOXES_0806:
        return "LEFT_PLACE"
    return "UNKNOWN"


def robust_stats(arr: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return {"n": 0, "median": np.nan, "p10": np.nan, "p90": np.nan, "iqr": np.nan, "mad": np.nan}
    med = float(np.median(a))
    return {
        "n": int(len(a)),
        "median": med,
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "mad": float(np.median(np.abs(a - med))),
    }


def joint_path_length(joints: np.ndarray) -> float:
    if len(joints) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    return float(d.sum())


def pose_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def load_episode_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index")


def load_states(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    dfs = [pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    states = np.stack(df["observation.state"].to_numpy())
    df = df.assign(
        suction=states[:, 14].astype(float),
        gripper=states[:, 15].astype(float),
    )
    # keep full state for joints
    df["_state"] = list(states)
    return df


def process_episode_signals(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Event], list[SuctionCycle]]:
    suction_raw = states[:, 14].astype(float)
    gripper_raw = states[:, 15].astype(float)
    suction = debounce_binary(median_filter(suction_raw) > SUCTION_THR, MIN_ON_FRAMES, MIN_OFF_FRAMES)
    gripper = debounce_binary(median_filter(gripper_raw) > GRIPPER_THR, MIN_ON_FRAMES, MIN_OFF_FRAMES)

    events: list[Event] = []
    for f in rising_edges(suction):
        events.append(Event("SUCTION_ON", int(f)))
    for f in falling_edges(suction):
        events.append(Event("SUCTION_OFF", int(f)))
    for f in rising_edges(gripper):
        events.append(Event("GRIPPER_CLOSE", int(f)))
    for f in falling_edges(gripper):
        events.append(Event("GRIPPER_OPEN", int(f)))
    events.sort(key=lambda e: (e.frame, e.kind))

    on_edges = rising_edges(suction)
    off_edges = falling_edges(suction)
    cycles: list[SuctionCycle] = []
    off_i = 0
    for on in on_edges:
        while off_i < len(off_edges) and off_edges[off_i] <= on:
            off_i += 1
        if off_i >= len(off_edges):
            off = len(states) - 1
        else:
            off = off_edges[off_i]
            off_i += 1
        # gripper events associated with this cycle: from on to next on (or end),
        # including short post-off window for placement open
        next_on = None
        idx = on_edges.index(on)
        if idx + 1 < len(on_edges):
            next_on = on_edges[idx + 1]
        end_assoc = next_on if next_on is not None else len(states)
        g_closes = [e.frame for e in events if e.kind == "GRIPPER_CLOSE" and on <= e.frame < end_assoc]
        g_opens = [e.frame for e in events if e.kind == "GRIPPER_OPEN" and on <= e.frame < end_assoc]

        # Mode inference from actuators inside [on, off]
        closes_during_on = [f for f in g_closes if on <= f <= off]
        if closes_during_on:
            mode = "HANDOVER"
            # Transfer completed once suction turns OFF after a grip-close during ON.
            # Gripper OPEN may occur later during placement and is not required here.
            completed = off > closes_during_on[0] and off < len(states) - 1
        else:
            mode = "LEFT_PLACE"
            # completed if meaningful ON duration and OFF occurred before episode end
            completed = (off - on) >= MIN_ON_FRAMES and off < len(states) - 1

        left = states[on:off, 0:7]
        right = states[on:off, 7:14]
        pre = max(0, on - 5)
        cycles.append(
            SuctionCycle(
                on=on,
                off=off,
                duration=off - on,
                grip_closes=g_closes,
                grip_opens=g_opens,
                mode=mode,
                completed=completed,
                left_motion=joint_path_length(left),
                right_motion=joint_path_length(right),
                pre_pose_left=states[pre, 0:7].tolist(),
                pre_pose_right=states[pre, 7:14].tolist(),
            )
        )
    return suction, gripper, events, cycles


# Empirically, distinct-box left pre-pose distances on 0806 pairtask are ~0.32–0.91.
# Distances << that strongly suggest returning to the same pick configuration.
SAME_BOX_POSE_THR = 0.40
LIKELY_SAME_BOX_POSE_THR = 0.55
EXPECTED_BOXES_DEFAULT = 2


def assign_boxes(cycles: list[SuctionCycle], expected_boxes: list[int], date: str, has_box_ids: bool) -> list[SuctionCycle]:
    """Assign box IDs. Pose-based same-box merging only when surplus cycles exist."""
    expected_n = len(expected_boxes) if expected_boxes else EXPECTED_BOXES_DEFAULT
    n = len(cycles)
    if n == 0:
        return cycles

    if n <= expected_n:
        if has_box_ids and len(expected_boxes) == 2:
            for i, c in enumerate(cycles):
                c.box_id = expected_boxes[min(i, len(expected_boxes) - 1)]
                c.mode = mode_for_box_0806(int(c.box_id))
        else:
            for i, c in enumerate(cycles):
                c.box_id = f"box_{i + 1}"
        return cycles

    # Surplus: link the (n-expected_n) lowest-distance ordered pairs, then label components.
    extra = n - expected_n
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
        comps = len({find(k) for k in range(n)})
        if comps <= expected_n:
            break
        union(i, j)
        linked += 1

    order = []
    comp_first = {}
    for i in range(n):
        r = find(i)
        if r not in comp_first:
            comp_first[r] = i
            order.append(r)

    while len(order) > expected_n:
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

    if has_box_ids and len(expected_boxes) == 2:
        box_labels = {comp: expected_boxes[min(idx, len(expected_boxes) - 1)] for idx, comp in enumerate(order)}
        for i, c in enumerate(cycles):
            c.box_id = box_labels[find(i)]
            c.mode = mode_for_box_0806(int(c.box_id))
    else:
        box_labels = {comp: f"box_{idx + 1}" for idx, comp in enumerate(order)}
        for i, c in enumerate(cycles):
            c.box_id = box_labels[find(i)]
    return cycles


def event_pattern(events: list[Event], start: int, end: int) -> tuple[str, str]:
    sub = [e for e in events if start <= e.frame <= end]
    suct = [f"{e.kind}@{e.frame}" for e in sub if e.kind.startswith("SUCTION")]
    grip = [f"{e.kind}@{e.frame}" for e in sub if e.kind.startswith("GRIPPER")]
    return ">".join(suct) or "NONE", ">".join(grip) or "NONE"


def classify_same_box_retry(
    group: list[SuctionCycle],
    templates: dict[str, dict],
) -> list[tuple[SuctionCycle, SuctionCycle, str, str, str]]:
    """Return list of (first, retry, recovery_type, confidence, notes) within a box group."""
    out = []
    for i in range(len(group) - 1):
        c0, c1 = group[i], group[i + 1]
        d_left = pose_dist(np.array(c0.pre_pose_left), np.array(c1.pre_pose_left))
        d_right = pose_dist(np.array(c0.pre_pose_right), np.array(c1.pre_pose_right))
        gap = c1.on - c0.off
        mode = c0.mode
        closes0 = [f for f in c0.grip_closes if c0.on <= f <= c0.off]
        opens0 = [o for o in c0.grip_opens if o > c0.on]

        if d_left <= SAME_BOX_POSE_THR:
            conf = "HIGH"
        elif d_left <= LIKELY_SAME_BOX_POSE_THR:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        if not c0.completed:
            med_dur = templates.get(mode, {}).get("duration", {}).get("median", np.nan)
            mad = templates.get(mode, {}).get("duration", {}).get("mad", 0.0)
            if np.isfinite(med_dur) and c0.duration > med_dur + 3 * mad + 30:
                rtype = "STALL_THEN_RETRY"
                notes = "Long stall on first attempt then retry pattern"
            elif mode == "HANDOVER" and closes0 and opens0:
                rtype = "SUBTASK_ROLLBACK"
                notes = "Reached gripper phase then returned to suction pick for same box"
            elif mode == "HANDOVER" and closes0:
                rtype = "HANDOVER_RETRY"
                notes = "Incomplete handover then repeated suction cycle for same box"
            else:
                rtype = "PICK_RETRY"
                notes = "Suction cycle ended without completion then re-pick for same box"
            if conf == "LOW":
                conf = "MEDIUM"  # incomplete+retry is still meaningful
            out.append((c0, c1, rtype, conf, notes + f" | Ldist={d_left:.3f} Rdist={d_right:.3f} gap={gap}"))
        else:
            # Completed cycle then another cycle forced onto same box (surplus evidence)
            if mode == "LEFT_PLACE":
                rtype = "PLACEMENT_RETRY"
                notes = "LEFT_PLACE completed then another suction cycle assigned to same box"
            else:
                # Return to pick after an apparently finished handover/place
                rtype = "SUBTASK_ROLLBACK"
                notes = "Completed HANDOVER cycle then returned to pick-like suction for same box"
            # If pose match is weak, keep as LOW (possible mis-grouping)
            out.append((c0, c1, rtype, conf, notes + f" | Ldist={d_left:.3f} Rdist={d_right:.3f} gap={gap}"))
    return out


def detect_recoveries(
    date: str,
    source: str,
    ep_idx: int,
    task_text: str,
    states: np.ndarray,
    events: list[Event],
    cycles: list[SuctionCycle],
    templates: dict[str, dict],
) -> tuple[list[RecoveryCandidate], list[dict]]:
    """Return recovery candidates and rejected anomalies."""
    cands: list[RecoveryCandidate] = []
    rejected: list[dict] = []
    n = len(states)

    def add(
        rtype: str,
        mode: str,
        box_id: Any,
        start: int,
        end: int,
        first: int,
        retry: int,
        conf: str,
        notes: str,
        arm: str = "",
        signal: str = "",
    ):
        s_pat, g_pat = event_pattern(events, start, end)
        cands.append(
            RecoveryCandidate(
                date=date,
                source_dataset=source,
                episode_index=ep_idx,
                task_text=task_text,
                box_id=box_id,
                mode=mode,
                recovery_type=rtype,
                start_frame=start,
                end_frame=end,
                first_attempt_frame=first,
                retry_frame=retry,
                suction_event_pattern=s_pat,
                gripper_event_pattern=g_pat,
                confidence=conf,
                visual_support="PENDING",
                notes=notes,
                arm_support=arm,
                signal_evidence=signal,
                events=[asdict(e) for e in events if start <= e.frame <= end],
            )
        )

    by_box: dict[Any, list[SuctionCycle]] = {}
    for c in cycles:
        by_box.setdefault(c.box_id, []).append(c)

    for box_id, group in by_box.items():
        for c0, c1, rtype, conf, notes in classify_same_box_retry(group, templates):
            d_left = pose_dist(np.array(c0.pre_pose_left), np.array(c1.pre_pose_left))
            d_right = pose_dist(np.array(c0.pre_pose_right), np.array(c1.pre_pose_right))
            add(
                rtype,
                c0.mode,
                box_id,
                c0.on,
                min(n - 1, c1.off + 5),
                c0.on,
                c1.on,
                conf,
                notes,
                arm=f"pre_pose_dist L={d_left:.3f} R={d_right:.3f}",
                signal=f"same_box_cycles={len(group)}; gap={c1.on - c0.off}; c0_completed={c0.completed}",
            )

        # Within-cycle gripper CLOSE→OPEN→CLOSE while suction ON: true HANDOVER_RETRY
        for c in group:
            if c.mode != "HANDOVER":
                continue
            closes = sorted(c.grip_closes)
            opens = sorted(c.grip_opens)
            closes_during = [f for f in closes if c.on <= f <= c.off]
            if len(closes_during) >= 2:
                mid_open = [o for o in opens if closes_during[0] < o < closes_during[1]]
                if mid_open:
                    add(
                        "HANDOVER_RETRY",
                        c.mode,
                        box_id,
                        c.on,
                        c.off,
                        closes_during[0],
                        closes_during[1],
                        "HIGH",
                        "Gripper CLOSE→OPEN→CLOSE while suction still ON (re-handover)",
                        signal=f"closes_during_on={closes_during}",
                    )
                else:
                    rejected.append(
                        {
                            "date": date,
                            "source_dataset": source,
                            "episode_index": ep_idx,
                            "reason": "multiple grip closes without reopen during suction; likely noise/overlap",
                            "frames": closes_during,
                        }
                    )

            # Post-place extra grip without suction re-ON: known harmless variation (NOT recovery)
            if closes_during:
                first_open_after = [o for o in opens if o > closes_during[0]]
                if first_open_after:
                    place_open = first_open_after[0]
                    later_closes = [f for f in closes if f > place_open]
                    later_opens = [o for o in opens if o > place_open]
                    if later_closes and later_opens:
                        seg = states[place_open : later_closes[0] + 1, 7:14]
                        mot = joint_path_length(seg)
                        rejected.append(
                            {
                                "date": date,
                                "source_dataset": source,
                                "episode_index": ep_idx,
                                "reason": "post-place extra gripper open/close without suction re-ON (harmless adjustment / known variation)",
                                "frames": [place_open, later_closes[0], later_opens[0]],
                                "right_motion": mot,
                                "box_id": box_id,
                            }
                        )

    # Surplus cycles not grouped (should be rare after pose assignment)
    expected_n = EXPECTED_BOXES_DEFAULT
    n_unique_boxes = len(by_box)
    if len(cycles) > expected_n and not cands:
        add(
            "OTHER",
            cycles[-1].mode,
            cycles[-1].box_id,
            cycles[0].on,
            cycles[-1].off,
            cycles[0].on,
            cycles[-1].on,
            "LOW",
            f"Extra suction cycles ({len(cycles)}>{expected_n}) but pose grouping yielded no retry pair",
            signal=f"n_cycles={len(cycles)}; n_unique_boxes={n_unique_boxes}",
        )

    uniq: list[RecoveryCandidate] = []
    for c in cands:
        dup = False
        for u in uniq:
            if c.recovery_type == u.recovery_type and c.box_id == u.box_id and abs(c.retry_frame - u.retry_frame) <= 5:
                # keep higher confidence
                rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
                if rank.get(c.confidence, 0) > rank.get(u.confidence, 0):
                    uniq.remove(u)
                    uniq.append(c)
                dup = True
                break
        if not dup:
            uniq.append(c)
    return uniq, rejected


def build_templates(normal_cycles: list[tuple[str, SuctionCycle]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for mode in ["HANDOVER", "LEFT_PLACE"]:
        cs = [c for m, c in normal_cycles if m == mode]
        if not cs:
            out[mode] = {}
            continue
        out[mode] = {
            "n_cycles": len(cs),
            "suction_on_events_per_box": robust_stats([1] * len(cs)),
            "suction_off_events_per_box": robust_stats([1] * len(cs)),
            "gripper_close_events": robust_stats([len(c.grip_closes) for c in cs]),
            "gripper_open_events": robust_stats([len(c.grip_opens) for c in cs]),
            "duration": robust_stats([c.duration for c in cs]),
            "left_motion": robust_stats([c.left_motion for c in cs]),
            "right_motion": robust_stats([c.right_motion for c in cs]),
            "on_to_first_close": robust_stats(
                [(min(c.grip_closes) - c.on) for c in cs if c.grip_closes]
            )
            if mode == "HANDOVER"
            else {},
            "close_to_suction_off": robust_stats(
                [
                    (c.off - min(f for f in c.grip_closes if f <= c.off))
                    for c in cs
                    if any(f <= c.off for f in c.grip_closes)
                ]
            )
            if mode == "HANDOVER"
            else {},
        }
    return out


def sample_video_frames(
    root: Path,
    ep_row: pd.Series,
    frames: list[tuple[str, int]],
    out_dir: Path,
    cam: str = "observation.images.camera_head",
) -> list[Path]:
    """Sample frames by episode-local frame index using video timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = int(ep_row[f"videos/{cam}/chunk_index"])
    file_index = int(ep_row[f"videos/{cam}/file_index"])
    from_ts = float(ep_row[f"videos/{cam}/from_timestamp"])
    video_path = root / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    if not video_path.exists():
        return []
    saved = []
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        for tag, fidx in frames:
            t = from_ts + (fidx / FPS)
            # Seek in stream time_base units (AV1 files need stream-aware seek).
            container.seek(int(t / stream.time_base), any_frame=False, backward=True, stream=stream)
            best = None
            best_dt = 1e9
            for i, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    continue
                pts_t = float(frame.pts * stream.time_base)
                dt = abs(pts_t - t)
                if dt < best_dt:
                    best_dt = dt
                    best = frame
                if pts_t >= t or i > 60:
                    break
            if best is None:
                continue
            img = best.to_ndarray(format="rgb24")
            path = out_dir / f"{tag}_f{fidx}_{cam.split('.')[-1]}.jpg"
            Image.fromarray(img).save(path, quality=85)
            saved.append(path)
        container.close()
    except Exception as e:
        (out_dir / "video_error.txt").write_text(f"{video_path}: {e}")
    return saved


def visual_score_from_samples(saved: list[Path], cand: RecoveryCandidate) -> tuple[str, str]:
    n_saved = len(saved)
    if n_saved == 0:
        return "LOW", "video sample unavailable/failed"
    # Check that sampled frames actually differ (seek health)
    arrs = []
    for p in saved[:4]:
        try:
            arrs.append(np.asarray(Image.open(p), dtype=np.float32).mean())
        except Exception:
            pass
    distinct = len(arrs) >= 2 and (max(arrs) - min(arrs) > 0.5 or np.std(arrs) > 0.2)
    if not distinct and n_saved >= 2:
        return "LOW", "sampled frames nearly identical (seek/view inconclusive)"
    if n_saved >= 8 and cand.confidence == "HIGH" and distinct:
        return "HIGH", "multi-camera frames differ across attempt/retry; consistent with retry"
    if n_saved >= 3 and distinct:
        if cand.confidence == "HIGH":
            return "MEDIUM", "head-camera frames differ across attempt/retry; actuator+arm consistent"
        return "MEDIUM", "representative frames extracted; temporally distinct"
    if n_saved > 0:
        return "LOW", "partial/ambiguous frame extraction"
    return "LOW", "video sample unavailable/failed"


def task_text_from_ep(ep_row: pd.Series) -> str:
    t = ep_row.get("tasks")
    if isinstance(t, (list, np.ndarray)):
        return str(t[0]) if len(t) else ""
    return str(t)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    inventory = {
        "raw_0806": str(Path("/media/jushen/kaka-liu/datasets/raw/0806")),
        "raw_0806_exists": Path("/media/jushen/kaka-liu/datasets/raw/0806").exists(),
        "raw_0806_02": str(Path("/media/jushen/kaka-liu/datasets/raw/0806_02")),
        "raw_0806_02_exists": Path("/media/jushen/kaka-liu/datasets/raw/0806_02").exists(),
        "raw_0807": str(Path("/media/jushen/kaka-liu/datasets/raw/0807")),
        "raw_0807_exists": Path("/media/jushen/kaka-liu/datasets/raw/0807").exists(),
        "raw_0806_02_recover_dir": str(Path("/media/jushen/kaka-liu/datasets/raw/0806_02/recover")),
        "note": "Task2 datasets under lerobot_style/task2_* excluded. Explicit recover/ folder noted but not mined as unlabeled recovery.",
        "datasets_used": DATASETS,
    }

    all_candidates: list[RecoveryCandidate] = []
    all_rejected: list[dict] = []
    episode_rows: list[dict] = []
    detailed_segments: list[dict] = []
    normal_for_template: list[tuple[str, SuctionCycle]] = []
    actuator_verify: dict[str, Any] = {}

    # Pass 1: templates from 0806 pairtask completed cycles that look normal (exactly 1 suction per box)
    pair = DATASETS[0]
    root = pair["root"]
    ep_table = load_episode_table(root)
    state_df = load_states(root)
    for ep_idx, g in state_df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        states = np.stack(g["_state"].to_numpy())
        _, _, events, cycles = process_episode_signals(states)
        ep_row = ep_table.loc[ep_table.episode_index == ep_idx].iloc[0]
        task = task_text_from_ep(ep_row)
        boxes = parse_boxes(task)
        cycles = assign_boxes(cycles, boxes, "0806", True)
        # collect normal if 2 cycles, each completed, each one suction
        if len(cycles) == 2 and all(c.completed for c in cycles):
            for c in cycles:
                # exclude cycles with >1 close-open after place as "less normal" for duration stats still ok
                normal_for_template.append((c.mode, c))
    templates = build_templates(normal_for_template)
    # Also compute templates per mode using only first cycle of each mode for cleaner stats
    (OUT_DIR / "normal_templates.json").write_text(json.dumps(templates, indent=2))

    # Pass 2: full detection on all datasets
    for ds in DATASETS:
        root = ds["root"]
        date = ds["date"]
        source = ds["source_dataset"]
        has_box = ds["has_box_ids"]
        ep_table = load_episode_table(root)
        state_df = load_states(root)

        # actuator verification
        s = state_df.suction.to_numpy()
        gr = state_df.gripper.to_numpy()
        actuator_verify[source] = {
            "suction_min_p50_p90_max": [float(s.min()), float(np.median(s)), float(np.percentile(s, 90)), float(s.max())],
            "gripper_min_p50_p90_max": [float(gr.min()), float(np.median(gr)), float(np.percentile(gr, 90)), float(gr.max())],
            "suction_thr_ok": float(np.percentile(s[s > SUCTION_THR], 10)) > 0.5 if np.any(s > SUCTION_THR) else False,
            "gripper_levels": {
                "open_med": float(np.median(gr[gr < GRIPPER_THR])) if np.any(gr < GRIPPER_THR) else None,
                "closed_med": float(np.median(gr[gr > GRIPPER_THR])) if np.any(gr > GRIPPER_THR) else None,
            },
        }

        mode_counts = {"HANDOVER": 0, "LEFT_PLACE": 0}
        for ep_idx, g in state_df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            states = np.stack(g["_state"].to_numpy())
            suction_bin, gripper_bin, events, cycles = process_episode_signals(states)
            ep_row = ep_table.loc[ep_table.episode_index == int(ep_idx)].iloc[0]
            task = task_text_from_ep(ep_row)
            boxes = parse_boxes(task) if has_box else []
            cycles = assign_boxes(cycles, boxes, date, has_box)

            for c in cycles:
                mode_counts[c.mode] = mode_counts.get(c.mode, 0) + 1

            cands, rejected = detect_recoveries(
                date, source, int(ep_idx), task, states, events, cycles, templates
            )
            all_rejected.extend(rejected)

            # Visual validation for HIGH/MEDIUM
            for cand in cands:
                if cand.confidence in {"HIGH", "MEDIUM"}:
                    sample_frames = [
                        ("before_first", max(0, cand.first_attempt_frame - 10)),
                        ("first_attempt", cand.first_attempt_frame),
                        ("retry", cand.retry_frame),
                        ("end", min(len(states) - 1, cand.end_frame)),
                    ]
                    out_sub = FRAME_DIR / source / f"ep{cand.episode_index}_{cand.recovery_type}_{cand.start_frame}"
                    cams = ["observation.images.camera_head"]
                    if cand.confidence == "HIGH":
                        cams += [
                            "observation.images.camera_left",
                            "observation.images.camera_right",
                        ]
                    saved = []
                    for cam in cams:
                        saved.extend(sample_video_frames(root, ep_row, sample_frames, out_sub, cam=cam))
                    vs, desc = visual_score_from_samples(saved, cand)
                    cand.visual_support = vs
                    cand.notes = (cand.notes + f" | visual: {desc}").strip()
                else:
                    cand.visual_support = "LOW"

                if cand.confidence == "HIGH" and cand.visual_support == "LOW":
                    cand.confidence = "MEDIUM"
                    cand.notes += " | confidence lowered: weak visual support"

            all_candidates.extend(cands)
            useful = [c for c in cands if c.confidence in {"HIGH", "MEDIUM"}]
            episode_rows.append(
                {
                    "date": date,
                    "source_dataset": source,
                    "episode_index": int(ep_idx),
                    "task_text": task,
                    "num_expected_boxes": len(boxes) if boxes else 2,
                    "num_suction_cycles": len(cycles),
                    "num_gripper_close_events": sum(1 for e in events if e.kind == "GRIPPER_CLOSE"),
                    "num_gripper_open_events": sum(1 for e in events if e.kind == "GRIPPER_OPEN"),
                    "num_recovery_candidates": len(useful),
                    "candidate_types": "|".join(sorted({c.recovery_type for c in useful})) or "",
                    "overall_confidence": (
                        "HIGH"
                        if any(c.confidence == "HIGH" for c in useful)
                        else ("MEDIUM" if useful else ("LOW" if cands else "NONE"))
                    ),
                    "inferred_modes": "|".join(c.mode for c in cycles),
                    "box_ids": "|".join(str(c.box_id) for c in cycles),
                }
            )
            detailed_segments.append(
                {
                    "date": date,
                    "source_dataset": source,
                    "episode_index": int(ep_idx),
                    "task_text": task,
                    "events": [asdict(e) for e in events],
                    "cycles": [asdict(c) for c in cycles],
                    "candidates": [asdict(c) for c in cands],
                }
            )

        actuator_verify[source]["mode_cycle_counts"] = mode_counts

    # Write outputs
    cand_df = pd.DataFrame([asdict(c) for c in all_candidates])
    if len(cand_df):
        # drop heavy events column from csv
        csv_df = cand_df.drop(columns=["events"], errors="ignore")
    else:
        csv_df = pd.DataFrame(
            columns=[
                "date",
                "source_dataset",
                "episode_index",
                "task_text",
                "box_id",
                "mode",
                "recovery_type",
                "start_frame",
                "end_frame",
                "first_attempt_frame",
                "retry_frame",
                "suction_event_pattern",
                "gripper_event_pattern",
                "confidence",
                "visual_support",
                "notes",
            ]
        )
    ep_df = pd.DataFrame(episode_rows)
    for df in (csv_df, ep_df):
        if len(df) and "date" in df.columns:
            df["date"] = df["date"].astype(str).map(lambda x: x.zfill(4) if str(x).isdigit() else x)
    # Keep leading zeros in date
    csv_df.to_csv(OUT_DIR / "recovery_candidates.csv", index=False)
    ep_df.to_csv(OUT_DIR / "episode_summary.csv", index=False)
    (OUT_DIR / "recovery_segments.json").write_text(json.dumps(detailed_segments, indent=2))
    (OUT_DIR / "rejected_anomalies.json").write_text(json.dumps(all_rejected, indent=2))
    (OUT_DIR / "data_inventory.json").write_text(json.dumps(inventory, indent=2, default=str))
    (OUT_DIR / "actuator_verification.json").write_text(json.dumps(actuator_verify, indent=2))

    # Summary stats
    useful = csv_df[csv_df["confidence"].isin(["HIGH", "MEDIUM"])] if len(csv_df) else csv_df
    summary = {
        "n_candidates_all": int(len(csv_df)),
        "n_high": int((csv_df["confidence"] == "HIGH").sum()) if len(csv_df) else 0,
        "n_medium": int((csv_df["confidence"] == "MEDIUM").sum()) if len(csv_df) else 0,
        "n_low": int((csv_df["confidence"] == "LOW").sum()) if len(csv_df) else 0,
        "type_counts_useful": useful["recovery_type"].value_counts().to_dict() if len(useful) else {},
        "type_counts_all": csv_df["recovery_type"].value_counts().to_dict() if len(csv_df) else {},
        "by_date_useful": useful.groupby("date").size().to_dict() if len(useful) else {},
        "n_rejected_anomalies": len(all_rejected),
        "templates": templates,
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

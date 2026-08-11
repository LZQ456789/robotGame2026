"""Offline Subtask labeling for Task1 Subtask-only V1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mycode.task1_subtask_v1.common import (
    EXPECTED_CYCLES,
    expected_mode,
    falling_edges,
    make_baseline_prompt,
    make_subtask_prompt,
    process_actuator_signals,
    rising_edges,
)


@dataclass
class CycleLabel:
    on: int
    off: int
    box_id: int
    mode: str
    grip_closes: list[int] = field(default_factory=list)
    grip_opens: list[int] = field(default_factory=list)
    place_end: int | None = None
    subtasks: list[dict] = field(default_factory=list)
    confidence: str = "HIGH"
    notes: str = ""
    valid: bool = True


@dataclass
class EpisodeLabelResult:
    valid: bool
    reason: str
    pair_lo: int | None
    pair_hi: int | None
    cycles: list[CycleLabel]
    # Per-frame arrays length = ep_len
    box_id: np.ndarray
    mode: np.ndarray  # object
    subtask_id: np.ndarray  # object
    confidence: np.ndarray  # object
    is_terminal_idle: np.ndarray
    is_post_place_variation: np.ndarray
    is_valid_operation_frame: np.ndarray
    prompt_baseline: np.ndarray  # object
    prompt_subtask: np.ndarray  # object


def _detect_cycles(suction: np.ndarray, gripper: np.ndarray) -> list[dict]:
    on_edges = rising_edges(suction)
    off_edges = falling_edges(suction)
    g_close = rising_edges(gripper)
    g_open = falling_edges(gripper)
    cycles: list[dict] = []
    off_i = 0
    for idx, on in enumerate(on_edges):
        while off_i < len(off_edges) and off_edges[off_i] <= on:
            off_i += 1
        if off_i >= len(off_edges):
            # incomplete trailing suction — keep for counting but mark incomplete later
            off = len(suction) - 1
            incomplete = True
        else:
            off = off_edges[off_i]
            off_i += 1
            incomplete = False
        next_on = on_edges[idx + 1] if idx + 1 < len(on_edges) else len(suction)
        closes_during = [f for f in g_close if on <= f <= off]
        # Keep all gripper edges after this suction-on for place-end search.
        # Do NOT clip opens to next_on: some demos keep the right gripper closed
        # into the next left-suction cycle (S2 overlaps next box start).
        closes = [f for f in g_close if f >= on]
        opens = [f for f in g_open if f >= on]
        if closes_during:
            mode = "HANDOVER"
            completed = (not incomplete) and off > closes_during[0]
        else:
            mode = "LEFT_PLACE"
            completed = not incomplete
        cycles.append(
            {
                "on": on,
                "off": off,
                "mode_inferred": mode,
                "completed": completed,
                "closes": closes,
                "opens": opens,
                "closes_during": closes_during,
                "next_on": next_on,
            }
        )
    return cycles


def _segment_cycle(
    c: CycleLabel, s0_start: int, ep_len: int, next_on: int
) -> tuple[list[dict], int, str]:
    """Return subtasks, semantic_place_end, notes.

    Frame assignment for S2 is clipped to ``next_on`` when the first gripper
    OPEN occurs after the next box cycle starts (known dual-arm overlap).
    ``semantic_place_end`` remains the true first OPEN for post-place carving.
    """
    notes = ""
    subs: list[dict] = []
    s0_end = c.on
    if s0_end < s0_start:
        # Previous S2 overlapped this cycle start; approach frames already consumed.
        s0_end = s0_start
        notes = "S0_skipped_due_to_overlap"
    if s0_end > s0_start:
        subs.append({"name": "S0_APPROACH_PICK", "start": s0_start, "end": s0_end})
    elif s0_end == s0_start:
        # Zero-length S0 is allowed at episode start or after overlap.
        subs.append({"name": "S0_APPROACH_PICK", "start": s0_start, "end": s0_end})

    if c.mode == "HANDOVER":
        closes_during = [f for f in c.grip_closes if c.on <= f <= c.off]
        if not closes_during:
            return subs, c.off, "missing_gripper_close_during_suction_on"
        if c.off <= closes_during[0]:
            return subs, c.off, "suction_off_before_gripper_close"
        # Transfer may start after a delayed S0 cursor under overlap.
        transfer_start = max(c.on, s0_start)
        if c.off <= transfer_start:
            return subs, c.off, "transfer_empty_after_overlap"
        subs.append({"name": "S1_TRANSFER_TO_GRIPPER", "start": transfer_start, "end": c.off})
        opens_after = [f for f in c.grip_opens if f > c.off]
        if not opens_after:
            opens_after = [f for f in c.grip_opens if f > closes_during[0]]
        if not opens_after:
            return subs, c.off, "missing_first_gripper_open_after_handover"
        semantic_place_end = opens_after[0]
        # Label span clipped so next box owns frames from its suction-on onward.
        label_place_end = min(semantic_place_end, next_on, ep_len)
        if semantic_place_end > next_on:
            notes = (notes + "; S2_overlaps_next_suction_clipped").strip("; ")
        subs.append({"name": "S2_GRIPPER_PLACEMENT", "start": c.off, "end": label_place_end})
        c.place_end = semantic_place_end
        return subs, semantic_place_end, notes

    # LEFT_PLACE
    place_start = max(c.on, s0_start)
    if c.off < place_start:
        return subs, c.off, "left_place_empty_after_overlap"
    subs.append({"name": "S1_LEFT_DIRECT_PLACE", "start": place_start, "end": c.off})
    c.place_end = c.off
    return subs, c.off, notes


def label_episode(
    states: np.ndarray,
    original_task: str,
    pair_lo: int,
    pair_hi: int,
) -> EpisodeLabelResult:
    ep_len = len(states)
    empty = lambda fill="": np.array([fill] * ep_len, dtype=object)  # noqa: E731
    box_arr = np.full(ep_len, -1, dtype=np.int16)
    mode_arr = empty("")
    sub_arr = empty("")
    conf_arr = empty("")
    term = np.zeros(ep_len, dtype=bool)
    post = np.zeros(ep_len, dtype=bool)
    op = np.zeros(ep_len, dtype=bool)
    p_base = empty("")
    p_sub = empty("")

    def fail(reason: str, cycles: list[CycleLabel] | None = None) -> EpisodeLabelResult:
        return EpisodeLabelResult(
            valid=False,
            reason=reason,
            pair_lo=pair_lo,
            pair_hi=pair_hi,
            cycles=cycles or [],
            box_id=box_arr,
            mode=mode_arr,
            subtask_id=sub_arr,
            confidence=conf_arr,
            is_terminal_idle=term,
            is_post_place_variation=post,
            is_valid_operation_frame=op,
            prompt_baseline=p_base,
            prompt_subtask=p_sub,
        )

    suction, gripper = process_actuator_signals(states)
    raw_cycles = _detect_cycles(suction, gripper)

    if len(raw_cycles) != EXPECTED_CYCLES:
        return fail(f"expected_{EXPECTED_CYCLES}_cycles_got_{len(raw_cycles)}")

    expected_boxes = [pair_lo, pair_hi]
    cycles: list[CycleLabel] = []
    for i, raw in enumerate(raw_cycles):
        box_id = expected_boxes[i]
        mode = expected_mode(box_id)
        if raw["mode_inferred"] != mode:
            return fail(
                f"mode_mismatch_box{box_id}_expected_{mode}_inferred_{raw['mode_inferred']}",
            )
        if not raw["completed"]:
            return fail(f"incomplete_cycle_box{box_id}")
        cycles.append(
            CycleLabel(
                on=raw["on"],
                off=raw["off"],
                box_id=box_id,
                mode=mode,
                grip_closes=raw["closes"],
                grip_opens=raw["opens"],
            )
        )

    # Segment with post-place carving
    cursor = 0
    for i, c in enumerate(cycles):
        next_on = cycles[i + 1].on if i + 1 < len(cycles) else ep_len
        subs, semantic_place_end, notes = _segment_cycle(c, cursor, ep_len, next_on)
        fatal_prefixes = (
            "missing",
            "suction",
            "transfer_empty",
            "left_place_empty",
        )
        if any(notes.startswith(p) for p in fatal_prefixes):
            c.valid = False
            c.notes = notes
            return fail(f"box{c.box_id}_{notes}", cycles)
        c.notes = notes
        label_place_end = next(
            (int(s["end"]) for s in subs if s["name"] in {"S2_GRIPPER_PLACEMENT", "S1_LEFT_DIRECT_PLACE"}),
            c.off,
        )
        # Post-place variation only in the gap before the next suction-on.
        # When S2 was clipped due to overlap, semantic_place_end may be > next_on;
        # those overlapped frames belong to the next box cycle, not post-place.
        post_start = label_place_end
        post_end = label_place_end
        later_grip = sorted(
            f for f in (c.grip_closes + c.grip_opens) if label_place_end < f < next_on
        )
        if later_grip:
            post_end = min(next_on, later_grip[-1] + 1)
            c.notes = (c.notes + "; post-place gripper adjustment").strip("; ")
        c.subtasks = subs
        c.place_end = semantic_place_end

        # Write per-frame labels for subtasks [start, end)
        for s in subs:
            a, b = int(s["start"]), int(s["end"])
            if b < a:
                return fail(f"box{c.box_id}_negative_span_{s['name']}", cycles)
            # Zero-length S0 is recorded but contributes no frames.
            b = min(b, ep_len)
            for f in range(a, b):
                box_arr[f] = c.box_id
                mode_arr[f] = c.mode
                sub_arr[f] = s["name"]
                conf_arr[f] = "MEDIUM" if "overlaps_next_suction" in (c.notes or "") and s["name"] == "S2_GRIPPER_PLACEMENT" else "HIGH"
                op[f] = True
                p_base[f] = make_baseline_prompt(pair_lo, pair_hi)
                p_sub[f] = make_subtask_prompt(pair_lo, pair_hi, c.box_id, s["name"])

        if post_end > post_start:
            for f in range(post_start, min(post_end, next_on)):
                if not op[f]:
                    box_arr[f] = c.box_id
                    mode_arr[f] = c.mode
                    sub_arr[f] = ""
                    conf_arr[f] = "POST_PLACE"
                    post[f] = True
                    p_base[f] = make_baseline_prompt(pair_lo, pair_hi)
                    p_sub[f] = make_baseline_prompt(pair_lo, pair_hi)

        # Next S0 starts after labeled place/post-place, but never after next_on
        # (overlapped semantic place belongs to the next cycle's timeline).
        cursor = min(next_on, max(label_place_end, post_end))

    # Terminal idle: after last cycle cursor to end
    if cursor < ep_len:
        term[cursor:] = True
        for f in range(cursor, ep_len):
            conf_arr[f] = "TERMINAL_IDLE"
            p_base[f] = make_baseline_prompt(pair_lo, pair_hi)
            p_sub[f] = make_baseline_prompt(pair_lo, pair_hi)

    # Sanity: every operation frame must have subtask
    if np.any(op & (sub_arr == "")):
        return fail("operation_frame_missing_subtask", cycles)

    return EpisodeLabelResult(
        valid=True,
        reason="OK",
        pair_lo=pair_lo,
        pair_hi=pair_hi,
        cycles=cycles,
        box_id=box_arr,
        mode=mode_arr,
        subtask_id=sub_arr,
        confidence=conf_arr,
        is_terminal_idle=term,
        is_post_place_variation=post,
        is_valid_operation_frame=op,
        prompt_baseline=p_base,
        prompt_subtask=p_sub,
    )


def apply_action_chunk_safety(
    is_valid_operation_frame: np.ndarray,
    box_id: np.ndarray,
    subtask_id: np.ndarray,
    is_terminal_idle: np.ndarray,
    is_post_place_variation: np.ndarray,
    action_horizon: int,
) -> tuple[np.ndarray, dict]:
    """A frame is a valid train anchor iff future action targets stay in same (box, subtask).

    For π0.5, action_delta_indices = range(chunk_size), so targets are frames
    t .. t+action_horizon-1 inclusive (action_horizon frames).
    """
    n = len(is_valid_operation_frame)
    anchors = np.zeros(n, dtype=bool)
    # Candidates: valid operation, not terminal idle, not post-place
    candidates = is_valid_operation_frame & (~is_terminal_idle) & (~is_post_place_variation)
    violations_if_unfiltered = 0
    short_subtask_blocks = 0

    for t in np.flatnonzero(candidates):
        end = t + action_horizon
        if end > n:
            violations_if_unfiltered += 1
            continue
        window_box = box_id[t:end]
        window_sub = subtask_id[t:end]
        if not np.all(window_box == box_id[t]):
            violations_if_unfiltered += 1
            continue
        if not np.all(window_sub == subtask_id[t]):
            violations_if_unfiltered += 1
            continue
        if not np.all(is_valid_operation_frame[t:end]):
            violations_if_unfiltered += 1
            continue
        anchors[t] = True

    # Per-subtask usability
    subtask_stats: dict[str, dict] = {}
    for sid in sorted(set(subtask_id[candidates].tolist())):
        if not sid:
            continue
        mask = candidates & (subtask_id == sid)
        op_n = int(mask.sum())
        anc_n = int((anchors & (subtask_id == sid)).sum())
        # durations of contiguous runs
        durations = []
        idx = np.flatnonzero(mask)
        if len(idx):
            breaks = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks + 1, len(idx)]
            for a, b in zip(starts, ends, strict=True):
                durations.append(int(idx[b - 1] - idx[a] + 1))
        max_dur = max(durations) if durations else 0
        if max_dur < action_horizon:
            short_subtask_blocks += 1
        subtask_stats[sid] = {
            "operation_frames": op_n,
            "valid_train_anchors": anc_n,
            "max_run_duration": max_dur,
            "shorter_than_horizon": bool(max_dur < action_horizon),
        }

    stats = {
        "candidate_operation_frames": int(candidates.sum()),
        "valid_train_anchors": int(anchors.sum()),
        "violations_if_unfiltered": int(violations_if_unfiltered),
        "subtask_stats": subtask_stats,
        "short_subtask_types": short_subtask_blocks,
    }
    return anchors, stats

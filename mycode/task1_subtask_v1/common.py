"""Shared constants and helpers for Task1 Subtask-only V1."""

from __future__ import annotations

import re
from typing import Final

import numpy as np

SUCTION_THR: Final = 0.35
GRIPPER_THR: Final = 0.3
FPS: Final = 26
MEDIAN_K: Final = 5
# Spec: persistence/debounce = 3 frames; keep min ON duration = 15.
DEBOUNCE_FRAMES: Final = 3
MIN_ON_FRAMES: Final = 15
MIN_OFF_FRAMES: Final = 5
EXPECTED_CYCLES: Final = 2
ACTION_HORIZON_DEFAULT: Final = 50  # π0.5 chunk_size

HANDOVER_BOXES: Final = frozenset({1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14})
LEFT_PLACE_BOXES: Final = frozenset({7, 8, 9, 15, 16, 17, 18})
VALID_PAIRS: Final = frozenset((i, i + 1) for i in range(1, 18, 2))

SUBTASK_TEXT: Final = {
    "S0_APPROACH_PICK": "Approach and pick up the current box with the left suction cup.",
    "S1_TRANSFER_TO_GRIPPER": "Transfer the current box from the left suction cup to the right gripper.",
    "S2_GRIPPER_PLACEMENT": "Place the current box into its target carton position with the right gripper.",
    "S1_LEFT_DIRECT_PLACE": "Place the current box directly into its target carton position with the left suction cup.",
}

DATASETS: Final = [
    {
        "date": "0804",
        "source": "0806_from_raw_0804",
        "root": "/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0804",
        "expected_episodes": 70,
        "expected_frames": 70811,
    },
    {
        "date": "0805",
        "source": "0806_from_raw_0805",
        "root": "/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0805",
        "expected_episodes": 140,
        "expected_frames": 124561,
    },
    {
        "date": "0806",
        "source": "0806_from_raw_0806_progress_pairtask",
        "root": "/media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0806_progress_pairtask",
        "expected_episodes": 27,
        "expected_frames": 29415,
    },
]

DEFAULT_META_DIR: Final = "/media/jushen/kaka-liu/datasets/analysis/task1_subtask_v1"


def expected_mode(box_id: int) -> str:
    if box_id in HANDOVER_BOXES:
        return "HANDOVER"
    if box_id in LEFT_PLACE_BOXES:
        return "LEFT_PLACE"
    raise ValueError(f"box_id={box_id} is outside Task1 range 1–18")


def parse_pair(task: str) -> tuple[int, int]:
    """Parse pair_lo/pair_hi from Chinese or English task text. Fail loudly."""
    text = str(task).strip()
    m = re.search(r"boxes?\s+(\d+)\s+and\s+(\d+)", text, flags=re.I)
    if m is None:
        m = re.search(r"第\s*(\d+)\s*个.*?第\s*(\d+)\s*个", text)
    if m is None:
        m = re.search(r"(?<!\d)(\d{1,2})\s*[-–—]\s*(\d{1,2})(?!\d)", text)
    if m is None:
        raise ValueError(f"Cannot parse Task1 pair from task text: {text!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    if (lo, hi) not in VALID_PAIRS:
        raise ValueError(f"Parsed pair {lo}-{hi} is not a supported Task1 pair from task={text!r}")
    return lo, hi


def english_pair_task(lo: int, hi: int) -> str:
    return f"Pack medicine boxes {lo} and {hi} into the carton."


def make_baseline_prompt(lo: int, hi: int) -> str:
    return english_pair_task(lo, hi)


def make_subtask_prompt(lo: int, hi: int, box_id: int, subtask_id: str, total_boxes: int = 18) -> str:
    if subtask_id not in SUBTASK_TEXT:
        raise KeyError(f"Unknown subtask_id={subtask_id}")
    return (
        f"{english_pair_task(lo, hi)}\n"
        f"Current box: {box_id} of {total_boxes}.\n"
        f"Current subtask: {SUBTASK_TEXT[subtask_id]}"
    )


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


def process_actuator_signals(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return debounced suction/gripper binary signals from observation.state."""
    if states.ndim != 2 or states.shape[1] < 16:
        raise ValueError(f"Expected state shape (T, >=16), got {states.shape}")
    # Spec debounce=3; also enforce min valid ON duration=15 via a second pass floor.
    suction_raw = median_filter(states[:, 14]) > SUCTION_THR
    gripper_raw = median_filter(states[:, 15]) > GRIPPER_THR
    suction = debounce_binary(suction_raw, DEBOUNCE_FRAMES, DEBOUNCE_FRAMES)
    gripper = debounce_binary(gripper_raw, DEBOUNCE_FRAMES, DEBOUNCE_FRAMES)
    # Enforce minimum ON duration for suction events (validated preprocessing).
    suction = debounce_binary(suction, MIN_ON_FRAMES, MIN_OFF_FRAMES)
    gripper = debounce_binary(gripper, MIN_ON_FRAMES, MIN_OFF_FRAMES)
    return suction, gripper

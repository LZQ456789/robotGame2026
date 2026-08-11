#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

"""Task-text helpers for medicine-box packing and error-recovery episodes.

Normal episode example:
    Pack medicine boxes 9 and 10 into the carton

Error / recovery episode example (placed count is required):
    Recover after 8 boxes placed: Pack medicine boxes 9 and 10 into the carton

The progress head and inference controller both rely on these parsers.
Error-recovery prompting is implemented but intended to stay disabled until
recovery data and runtime wiring are ready.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Explicit "already placed N boxes when the error happened".
_PLACED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"recover\s+after\s+(\d+)\s+boxes?\s+placed", re.IGNORECASE),
    re.compile(r"after\s+(\d+)\s+boxes?\s+placed", re.IGNORECASE),
    re.compile(r"placed[_\s-]*(\d+)\s+boxes?", re.IGNORECASE),
    re.compile(r"(\d+)\s+boxes?\s+already\s+placed", re.IGNORECASE),
    re.compile(r"已放置\s*(\d+)\s*个"),
    re.compile(r"已放\s*(\d+)\s*盒"),
    re.compile(r"error[_\s-]*after[_\s-]*(\d+)", re.IGNORECASE),
)

# Pair range near the end of the sentence / filename.
_PAIR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"boxes?\s+(\d+)\s+and\s+(\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"抓取第\s*(\d+)\s*个和第\s*(\d+)\s*个",
    ),
    re.compile(
        r"(\d+)\s*[-–—]\s*(\d+)(?:\s*_trajectory)?",
    ),
)


@dataclass(frozen=True)
class ParsedTaskText:
    """Structured fields extracted from a dataset / runtime task string."""

    raw: str
    first_box: int | None = None
    second_box: int | None = None
    # Boxes already correctly placed when this episode starts / when error hit.
    placed_before: int | None = None
    is_recovery: bool = False

    @property
    def has_pair(self) -> bool:
        return self.first_box is not None and self.second_box is not None

    @property
    def completed_before(self) -> float | None:
        """Progress lower bound in box-count units (not normalized)."""
        if self.placed_before is not None:
            return float(self.placed_before)
        if self.first_box is not None:
            return float(self.first_box - 1)
        return None

    @property
    def completed_after(self) -> float | None:
        """Progress upper bound in box-count units (not normalized)."""
        if self.second_box is not None:
            return float(self.second_box)
        if self.placed_before is not None:
            # Recovery with only a placed count: keep a zero-width bound so the
            # label stays anchored at the failure point until pair text exists.
            return float(self.placed_before)
        return None


def parse_task_text(task_text: str) -> ParsedTaskText:
    """Parse normal or recovery task text.

    Recovery texts must encode how many boxes were already placed when the
    error occurred. Pair numbers are optional but preferred so recovery
    episodes can still advance local progress.
    """
    text = str(task_text).strip()
    placed_before: int | None = None
    matched_placed_pattern = False
    for pattern in _PLACED_PATTERNS:
        match = pattern.search(text)
        if match:
            placed_before = int(match.group(1))
            matched_placed_pattern = True
            break

    first_box: int | None = None
    second_box: int | None = None
    for pattern in _PAIR_PATTERNS:
        match = pattern.search(text)
        if match:
            first_box = int(match.group(1))
            second_box = int(match.group(2))
            break

    if first_box is None or second_box is None:
        numbers = [int(value) for value in re.findall(r"\d+", text)]
        # Prefer the last two integers as the box pair. Recovery strings also
        # contain the placed count, so pair patterns / trailing ints are used.
        if len(numbers) >= 2:
            first_box, second_box = numbers[-2:]

    if first_box is not None and second_box is not None and second_box < first_box:
        first_box, second_box = None, None

    if placed_before is not None and placed_before < 0:
        placed_before = None
        matched_placed_pattern = False

    is_recovery = matched_placed_pattern and placed_before is not None

    return ParsedTaskText(
        raw=text,
        first_box=first_box,
        second_box=second_box,
        placed_before=placed_before,
        is_recovery=is_recovery,
    )


def format_normal_task(first_box: int, second_box: int, *, language: str = "en") -> str:
    """Build the default two-box packing instruction."""
    if language == "zh":
        return f"抓取第{first_box}个和第{second_box}个药盒"
    return f"Pack medicine boxes {first_box} and {second_box} into the carton"


def format_recovery_task(
    placed_before: int,
    first_box: int,
    second_box: int,
    *,
    language: str = "en",
) -> str:
    """Build recovery text that records boxes already placed at failure time."""
    if placed_before < 0:
        raise ValueError("placed_before must be >= 0")
    if first_box <= 0 or second_box < first_box:
        raise ValueError("box pair must satisfy 1 <= first_box <= second_box")

    if language == "zh":
        return (
            f"已放置{placed_before}个药盒后纠错："
            f"抓取第{first_box}个和第{second_box}个药盒"
        )
    return (
        f"Recover after {placed_before} boxes placed: "
        f"Pack medicine boxes {first_box} and {second_box} into the carton"
    )


def pair_bounds_from_parsed(parsed: ParsedTaskText) -> tuple[float, float] | None:
    """Return ``[completed_before, completed_after]`` box-count bounds."""
    completed_before = parsed.completed_before
    completed_after = parsed.completed_after
    if completed_before is None or completed_after is None:
        return None
    if completed_after < completed_before:
        return None
    return float(completed_before), float(completed_after)

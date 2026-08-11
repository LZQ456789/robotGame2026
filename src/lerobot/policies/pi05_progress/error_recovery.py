#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

"""Inference-time progress / error-recovery orchestration for packing tasks.

This module is complete but **disabled by default**. Wire it into your robot
loop and set ``enable_error_recovery=True`` only after recovery demonstrations
are in the training mix.

Error data contract
-------------------
Recovery episodes / runtime recovery prompts must encode how many boxes were
already placed when the failure happened, e.g.:

    Recover after 8 boxes placed: Pack medicine boxes 9 and 10 into the carton

The controller keeps ``boxes_completed`` as the trusted placed count. Progress
predictions are used to advance normal segments. Entering recovery requires an
explicit external signal (or a future verifier); progress alone is not treated
as proof of success or failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .task_text import format_normal_task, format_recovery_task, parse_task_text


class EpisodePhase(str, Enum):
    NORMAL = "normal"
    RECOVERY = "recovery"
    DONE = "done"


@dataclass
class ErrorRecoveryConfig:
    """Runtime switches for the packing controller.

    Keep ``enable_error_recovery=False`` until recovery data is ready.
    """

    total_task_units: int = 18
    boxes_per_segment: int = 2
    language: Literal["en", "zh"] = "en"
    # Master switch: when False, recovery APIs are no-ops / raise clearly.
    enable_error_recovery: bool = False
    # Advance to the next pair when predicted progress reaches this fraction
    # of the current segment's upper bound (in normalized [0, 1] progress).
    segment_advance_margin: float = 0.02
    # Normalized progress considered "task finished".
    done_progress: float = 1.0
    # Optional stall heuristic (still does not auto-enter recovery unless
    # enable_error_recovery and auto_recover_on_stall are both True).
    stall_progress_epsilon: float = 0.01
    stall_steps: int = 40
    auto_recover_on_stall: bool = False


@dataclass
class ControllerState:
    phase: EpisodePhase = EpisodePhase.NORMAL
    boxes_completed: int = 0
    first_box: int = 1
    second_box: int = 2
    task_text: str = ""
    steps_in_segment: int = 0
    last_progress: float = 0.0
    peak_progress_in_segment: float = 0.0
    stall_counter: int = 0
    recovery_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ControllerDecision:
    """One control step for the outer inference loop."""

    phase: EpisodePhase
    task_text: str
    boxes_completed: int
    first_box: int
    second_box: int
    progress: float
    segment_low: float
    segment_high: float
    advanced_segment: bool = False
    entered_recovery: bool = False
    finished: bool = False
    message: str = ""


class MedicineBoxProgressController:
    """Progress-guided segment switching with optional error recovery.

    Typical loop::

        ctrl = MedicineBoxProgressController(ErrorRecoveryConfig(...))
        ctrl.reset()
        while not done:
            obs["task"] = ctrl.current_task_text
            action = policy.select_action(obs)
            progress = policy.predict_progress(obs).item()
            decision = ctrl.step(progress)
            # Optional, from a future verifier / human interrupt:
            # decision = ctrl.signal_error(placed_boxes=ctrl.state.boxes_completed)
            robot.send(action)
            if decision.finished:
                break
    """

    def __init__(self, config: ErrorRecoveryConfig | None = None):
        self.config = config or ErrorRecoveryConfig()
        if self.config.total_task_units <= 0:
            raise ValueError("total_task_units must be positive")
        if self.config.boxes_per_segment <= 0:
            raise ValueError("boxes_per_segment must be positive")
        self.state = ControllerState()
        self.reset()

    def reset(self, boxes_completed: int = 0) -> ControllerDecision:
        """Start (or restart) from a known placed-box count."""
        boxes_completed = int(max(0, min(boxes_completed, self.config.total_task_units)))
        self.state = ControllerState(boxes_completed=boxes_completed)
        if boxes_completed >= self.config.total_task_units:
            self.state.phase = EpisodePhase.DONE
            self.state.task_text = ""
            self.state.first_box = self.config.total_task_units
            self.state.second_box = self.config.total_task_units
        else:
            self.state.phase = EpisodePhase.NORMAL
            self._set_segment_from_completed(boxes_completed)
            self.state.task_text = format_normal_task(
                self.state.first_box,
                self.state.second_box,
                language=self.config.language,
            )
        return self._decision(progress=boxes_completed / self.config.total_task_units, message="reset")

    @property
    def current_task_text(self) -> str:
        return self.state.task_text

    @property
    def finished(self) -> bool:
        return self.state.phase == EpisodePhase.DONE

    def segment_bounds(self) -> tuple[float, float]:
        """Normalized [low, high] progress for the active segment."""
        low = self.state.boxes_completed / self.config.total_task_units
        high = min(self.state.second_box, self.config.total_task_units) / self.config.total_task_units
        return low, high

    def step(self, progress: float, *, error_signal: bool = False) -> ControllerDecision:
        """Update controller from a predicted progress in ``[0, 1]``."""
        progress = float(max(0.0, min(1.0, progress)))
        if self.state.phase == EpisodePhase.DONE:
            return self._decision(progress=progress, finished=True, message="already done")

        self.state.steps_in_segment += 1
        self.state.last_progress = progress
        self.state.peak_progress_in_segment = max(self.state.peak_progress_in_segment, progress)

        entered_recovery = False
        if error_signal:
            decision = self.signal_error(placed_boxes=self.state.boxes_completed)
            return decision

        if (
            self.config.enable_error_recovery
            and self.config.auto_recover_on_stall
            and self.state.phase == EpisodePhase.NORMAL
            and self._is_stalled(progress)
        ):
            entered_recovery = True
            self._enter_recovery(self.state.boxes_completed)
            return self._decision(
                progress=progress,
                entered_recovery=True,
                message="auto recovery from progress stall",
            )

        advanced = False
        low, high = self.segment_bounds()
        advance_threshold = max(0.0, high - self.config.segment_advance_margin)
        if progress >= advance_threshold or progress >= self.config.done_progress:
            advanced = self._advance_segment()
            low, high = self.segment_bounds()

        finished = self.state.phase == EpisodePhase.DONE
        return self._decision(
            progress=progress,
            advanced_segment=advanced,
            entered_recovery=entered_recovery,
            finished=finished,
            message="advanced segment" if advanced else "continue",
        )

    def signal_error(self, placed_boxes: int | None = None) -> ControllerDecision:
        """Enter recovery using the placed-box count at failure time.

        This is the primary recovery entrypoint. It is a no-op that returns a
        clear message when ``enable_error_recovery`` is False.
        """
        if not self.config.enable_error_recovery:
            return self._decision(
                progress=self.state.last_progress,
                message=(
                    "error recovery disabled "
                    "(set ErrorRecoveryConfig.enable_error_recovery=True to activate)"
                ),
            )

        if self.state.phase == EpisodePhase.DONE:
            return self._decision(progress=1.0, finished=True, message="already done")

        placed = self.state.boxes_completed if placed_boxes is None else int(placed_boxes)
        placed = max(0, min(placed, self.config.total_task_units))
        self._enter_recovery(placed)
        return self._decision(
            progress=placed / self.config.total_task_units,
            entered_recovery=True,
            message=f"recovery after {placed} boxes placed",
        )

    def mark_boxes_completed(self, boxes_completed: int) -> ControllerDecision:
        """Externally correct the trusted placed count (e.g. from a verifier)."""
        return self.reset(boxes_completed=boxes_completed)

    def format_recovery_task_text(self, placed_before: int, first_box: int, second_box: int) -> str:
        """Helper for dataset conversion of recovery demos."""
        return format_recovery_task(
            placed_before,
            first_box,
            second_box,
            language=self.config.language,
        )

    def describe(self) -> dict:
        low, high = self.segment_bounds()
        return {
            "enable_error_recovery": self.config.enable_error_recovery,
            "phase": self.state.phase.value,
            "boxes_completed": self.state.boxes_completed,
            "first_box": self.state.first_box,
            "second_box": self.state.second_box,
            "task_text": self.state.task_text,
            "segment_low": low,
            "segment_high": high,
            "total_task_units": self.config.total_task_units,
            "recovery_count": self.state.recovery_count,
        }

    def _set_segment_from_completed(self, boxes_completed: int) -> None:
        first = boxes_completed + 1
        second = min(boxes_completed + self.config.boxes_per_segment, self.config.total_task_units)
        self.state.boxes_completed = boxes_completed
        self.state.first_box = first
        self.state.second_box = second
        self.state.steps_in_segment = 0
        self.state.peak_progress_in_segment = boxes_completed / self.config.total_task_units
        self.state.stall_counter = 0

    def _enter_recovery(self, placed_boxes: int) -> None:
        if placed_boxes >= self.config.total_task_units:
            self.state.phase = EpisodePhase.DONE
            self.state.boxes_completed = self.config.total_task_units
            self.state.task_text = ""
            return

        self._set_segment_from_completed(placed_boxes)
        self.state.phase = EpisodePhase.RECOVERY
        self.state.recovery_count += 1
        self.state.task_text = format_recovery_task(
            placed_boxes,
            self.state.first_box,
            self.state.second_box,
            language=self.config.language,
        )
        self.state.notes.append(
            f"recovery#{self.state.recovery_count}: placed_before={placed_boxes}, "
            f"retry={self.state.first_box}-{self.state.second_box}"
        )

    def _advance_segment(self) -> bool:
        completed = min(self.state.second_box, self.config.total_task_units)
        if completed >= self.config.total_task_units:
            self.state.phase = EpisodePhase.DONE
            self.state.boxes_completed = self.config.total_task_units
            self.state.task_text = ""
            return True

        self.state.phase = EpisodePhase.NORMAL
        self._set_segment_from_completed(completed)
        self.state.task_text = format_normal_task(
            self.state.first_box,
            self.state.second_box,
            language=self.config.language,
        )
        return True

    def _is_stalled(self, progress: float) -> bool:
        if progress + self.config.stall_progress_epsilon >= self.state.peak_progress_in_segment:
            self.state.stall_counter = 0
            return False
        self.state.stall_counter += 1
        return self.state.stall_counter >= self.config.stall_steps

    def _decision(
        self,
        *,
        progress: float,
        advanced_segment: bool = False,
        entered_recovery: bool = False,
        finished: bool = False,
        message: str = "",
    ) -> ControllerDecision:
        low, high = self.segment_bounds()
        return ControllerDecision(
            phase=self.state.phase,
            task_text=self.state.task_text,
            boxes_completed=self.state.boxes_completed,
            first_box=self.state.first_box,
            second_box=self.state.second_box,
            progress=progress,
            segment_low=low,
            segment_high=high,
            advanced_segment=advanced_segment,
            entered_recovery=entered_recovery,
            finished=finished or self.state.phase == EpisodePhase.DONE,
            message=message,
        )


def recovery_progress_anchor(task_text: str, total_task_units: int = 18) -> float | None:
    """Normalized progress anchor from recovery task text (placed count / total)."""
    parsed = parse_task_text(task_text)
    if parsed.placed_before is None:
        return None
    return max(0.0, min(1.0, parsed.placed_before / float(total_task_units)))

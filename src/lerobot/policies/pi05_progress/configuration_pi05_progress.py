#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from typing import Literal

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_progress")
@dataclass
class PI05ProgressConfig(PI05Config):
    """PI0.5 with an auxiliary visual-language task-progress prediction head.

    ``task_fraction`` requires no extra annotation for the medicine-box
    dataset. It parses the two box numbers from each task and interpolates
    between them using the current episode fraction.

    Recovery / error episodes should encode how many boxes were already
    placed when the failure happened, e.g.
    ``Recover after 8 boxes placed: Pack medicine boxes 9 and 10 into the carton``.
    Those texts are parsed for progress bounds even while runtime recovery
    orchestration stays disabled by default.

    ``episode_fraction`` requires no extra annotation and uses
    ``frame_index / (episode_length - 1)`` as weak supervision. It represents
    progress within the current two-box episode, not verified task success.

    ``dataset`` consumes an explicit scalar feature in ``progress_key``. This
    mode should be preferred when semantic or success-aware labels become
    available.
    """

    progress_loss_weight: float = 0.1
    progress_hidden_dim: int = 256
    progress_target_mode: Literal["task_fraction", "episode_fraction", "dataset"] = "task_fraction"
    # Full packing task is capped at 18 boxes for the current data plan.
    total_task_units: int = 18
    # The observation prefix ensures the standard processor preserves this
    # optional feature instead of dropping it as unknown complementary data.
    progress_key: str = "observation.task_progress"
    # Runtime recovery controller switch. Keep False until recovery demos and
    # outer-loop wiring are ready. Training-time parsing of recovery task text
    # does not require enabling this flag.
    enable_error_recovery: bool = False
    boxes_per_segment: int = 2
    task_language: Literal["en", "zh"] = "en"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.progress_loss_weight < 0:
            raise ValueError("progress_loss_weight must be non-negative")
        if self.progress_hidden_dim <= 0:
            raise ValueError("progress_hidden_dim must be positive")
        if self.total_task_units <= 0:
            raise ValueError("total_task_units must be positive")
        if self.boxes_per_segment <= 0:
            raise ValueError("boxes_per_segment must be positive")

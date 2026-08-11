#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

from typing import Any

from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from .configuration_pi05_progress import PI05ProgressConfig


def make_pi05_progress_pre_post_processors(
    config: PI05ProgressConfig,
    dataset_stats: dict | None = None,
    dataset_meta: Any | None = None,
):
    """Reuse PI0.5 preprocessing; progress targets remain batch metadata."""
    del dataset_meta
    return make_pi05_pre_post_processors(config, dataset_stats)

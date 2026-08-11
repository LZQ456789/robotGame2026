"""Task1 Subtask-only V1 dataset: multi-source concat + dynamic task prompts.

Does not modify original LeRobot datasets. Videos remain in source roots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mycode.task1_subtask_v1.common import DATASETS, DEFAULT_META_DIR


class _MetaProxy:
    """Expose combined episode table while delegating feature/stats to first source."""

    def __init__(self, base_meta, episodes_df: pd.DataFrame, tasks: pd.DataFrame):
        self._base = base_meta
        self.episodes = episodes_df
        self.tasks = tasks
        self._stats = base_meta.stats

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, value):
        self._stats = value


class Task1SubtaskCombinedDataset(torch.utils.data.Dataset):
    """Concatenate 0804+0805+0806 and override ``task`` from sidecar prompts.

    Index space is contiguous across sources (same order as DATASETS).
    Pair-balanced sampling should only feed ``valid_train_anchor`` indices.
    """

    def __init__(
        self,
        meta_dir: str | Path = DEFAULT_META_DIR,
        *,
        enable_subtask_prompt: bool = True,
        delta_timestamps: dict[str, list[float]] | None = None,
        video_backend: str = "pyav",
        image_transforms=None,
        tolerance_s: float = 1e-4,
    ):
        self.meta_dir = Path(meta_dir)
        self.enable_subtask_prompt = bool(enable_subtask_prompt)
        frames_path = self.meta_dir / "task1_subtask_v1_frames.parquet"
        summary_path = self.meta_dir / "task1_subtask_v1_summary.json"
        if not frames_path.exists():
            raise FileNotFoundError(
                f"Missing sidecar {frames_path}. Run mycode/preprocessing/build_task1_subtask_v1.py first."
            )
        self.summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        self.frames = pd.read_parquet(frames_path)
        # Keep only valid episodes' rows for mapping; invalid rows stay for audit but are not sampled.
        self.frames = self.frames.sort_values(
            ["source_date", "source_episode_index", "frame_index"]
        ).reset_index(drop=True)

        self.datasets: list[LeRobotDataset] = []
        self.source_offsets: list[int] = []  # global start index per source
        self.source_dates: list[str] = []
        offset = 0
        for ds_info in DATASETS:
            root = Path(ds_info["root"])
            ds = LeRobotDataset(
                repo_id=root.name,
                root=str(root),
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                video_backend=video_backend,
                return_uint8=True,
                tolerance_s=tolerance_s,
            )
            self.datasets.append(ds)
            self.source_offsets.append(offset)
            self.source_dates.append(ds_info["date"])
            offset += ds.num_frames
        self._num_frames = offset

        # Fast sidecar lookup: (date, global_index_in_source) -> row position
        self._lookup = {
            (str(r.source_date), int(r.global_index_in_source)): i
            for i, r in enumerate(self.frames.itertuples(index=False))
        }

        # Combined episode table for sampler compatibility / logging
        ep_rows = []
        for src_i, ds in enumerate(self.datasets):
            date = self.source_dates[src_i]
            off = self.source_offsets[src_i]
            for ep_idx in range(ds.num_episodes):
                ep = ds.meta.episodes[ep_idx]
                ep_rows.append(
                    {
                        "episode_index": off  # not used as key
                        + int(ep_idx),  # placeholder unique id
                        "dataset_from_index": off + int(ep["dataset_from_index"]),
                        "dataset_to_index": off + int(ep["dataset_to_index"]),
                        "source_date": date,
                        "source_episode_index": int(ep_idx),
                    }
                )
        # Fix unique episode_index sequentially
        for i, row in enumerate(ep_rows):
            row["episode_index"] = i
        self._episodes_df = pd.DataFrame(ep_rows)

        # Aggregate stats across sources for normalization
        stats_list = [ds.meta.stats for ds in self.datasets]
        aggregated = aggregate_stats(stats_list)

        # Dummy tasks table (actual prompt comes from sidecar)
        tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index(["Task1 Subtask V1"], name="task"))
        self.meta = _MetaProxy(self.datasets[0].meta, self._episodes_df, tasks)
        self.meta.stats = aggregated

        self.repo_id = "task1_subtask_v1"
        self.root = str(self.meta_dir)
        self.episodes = None  # all
        self.absolute_to_relative_idx = None
        self.num_episodes = len(self._episodes_df)
        self.num_frames = self._num_frames

        # Valid-anchor index maps for sampler
        anchors = self.frames[self.frames["is_valid_train_anchor"] & self.frames["episode_valid"]]
        self.pair_to_global_indices: dict[str, list[int]] = {}
        for (lo, hi), g in anchors.groupby(["pair_lo", "pair_hi"]):
            pair = f"{int(lo)}-{int(hi)}"
            idxs = []
            for r in g.itertuples():
                src_i = self.source_dates.index(r.source_date)
                local = int(r.global_index_in_source)
                idxs.append(self.source_offsets[src_i] + local)
            self.pair_to_global_indices[pair] = idxs

        if len(self.pair_to_global_indices) != 9:
            raise RuntimeError(
                f"Expected 9 pairs with anchors, got {sorted(self.pair_to_global_indices)} "
                f"counts={[ (k,len(v)) for k,v in sorted(self.pair_to_global_indices.items()) ]}"
            )

    def set_subtask_enabled(self, enabled: bool) -> None:
        self.enable_subtask_prompt = bool(enabled)

    def _locate(self, idx: int) -> tuple[int, int, dict[str, Any]]:
        if idx < 0 or idx >= self._num_frames:
            raise IndexError(idx)
        src_i = 0
        for i, off in enumerate(self.source_offsets):
            end = off + self.datasets[i].num_frames
            if off <= idx < end:
                src_i = i
                break
        local = idx - self.source_offsets[src_i]
        date = self.source_dates[src_i]
        row_i = self._lookup.get((date, local))
        if row_i is None:
            raise KeyError(f"Sidecar miss for date={date} local_index={local}")
        meta_row = self.frames.iloc[row_i].to_dict()
        return src_i, local, meta_row

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> dict:
        src_i, local, meta_row = self._locate(idx)
        item = self.datasets[src_i][local]
        if self.enable_subtask_prompt:
            if meta_row.get("is_valid_operation_frame") and meta_row.get("subtask_id"):
                prompt = meta_row["prompt_subtask"]
            else:
                prompt = meta_row["prompt_baseline"]
        else:
            prompt = meta_row["prompt_baseline"]
        item["task"] = str(prompt)
        return item


def make_task1_subtask_dataset(
    cfg,
    *,
    enable_subtask_prompt: bool,
) -> Task1SubtaskCombinedDataset:
    """Factory used by make_dataset when task1_subtask_meta_dir is set."""
    from lerobot.transforms import ImageTransforms

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    # Resolve delta timestamps from the first source meta.
    first = Path(DATASETS[0]["root"])
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(repo_id=first.name, root=str(first))
    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
    meta_dir = cfg.dataset.task1_subtask_meta_dir or DEFAULT_META_DIR
    return Task1SubtaskCombinedDataset(
        meta_dir=meta_dir,
        enable_subtask_prompt=enable_subtask_prompt,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.dataset.video_backend,
        image_transforms=image_transforms,
        tolerance_s=cfg.tolerance_s,
    )

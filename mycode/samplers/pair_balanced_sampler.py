"""Pair-balanced distributed-safe sampler for Task1 Subtask-only V1.

All ranks independently generate the same epoch order from (seed, epoch).
Accelerate then shards via BatchSamplerShard (same pattern as EpisodeAwareSampler).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch


class PairBalancedSampler:
    """Sample with equal pair probability, then uniform within-pair anchors.

    Each yielded index is a dataset-relative index into the wrapped combined dataset.
    """

    def __init__(
        self,
        pair_to_indices: dict[str, Sequence[int]],
        *,
        shuffle: bool = True,
        seed: int = 0,
        samples_per_epoch: int | None = None,
    ):
        if not pair_to_indices:
            raise ValueError("pair_to_indices must be non-empty")
        self.pairs = sorted(pair_to_indices.keys())
        self._pair_arrays = {
            p: np.asarray(pair_to_indices[p], dtype=np.int64) for p in self.pairs
        }
        for p, arr in self._pair_arrays.items():
            if len(arr) == 0:
                raise ValueError(f"Pair {p} has zero valid train anchors")
        self.num_pairs = len(self.pairs)
        self.shuffle = shuffle
        self.seed = int(seed)
        self._epoch = 0
        self._start_index = 0
        total_anchors = int(sum(len(a) for a in self._pair_arrays.values()))
        # Default epoch length = total valid anchors (same scale as naive epoch).
        self._num_samples = int(samples_per_epoch) if samples_per_epoch is not None else total_anchors
        if self._num_samples <= 0:
            raise ValueError("samples_per_epoch must be > 0")

    @property
    def indices(self) -> list[int]:
        """Materialize one unshuffled round-robin epoch (introspection only)."""
        return list(self._iter_epoch(epoch=0, start=0, shuffle=False))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = int(state["epoch"])
        self._start_index = int(state["start_index"])

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start, shuffle=self.shuffle)

    def _iter_epoch(self, epoch: int, start: int, shuffle: bool) -> Iterator[int]:
        gen = self._epoch_generator(epoch)
        # Pre-draw pair choices and within-pair slots deterministically.
        if shuffle:
            pair_choices = torch.randint(
                0, self.num_pairs, (self._num_samples,), generator=gen
            ).tolist()
            # Independent within-pair uniforms; use a second derived generator stream.
            within = {}
            for p_i, p in enumerate(self.pairs):
                n = len(self._pair_arrays[p])
                # Count how many times this pair is chosen
                count = sum(1 for c in pair_choices if c == p_i)
                if count == 0:
                    within[p_i] = []
                    continue
                within[p_i] = torch.randint(0, n, (count,), generator=gen).tolist()
            within_pos = {p_i: 0 for p_i in range(self.num_pairs)}
            for k in range(start, self._num_samples):
                p_i = pair_choices[k]
                j = within[p_i][within_pos[p_i]]
                within_pos[p_i] += 1
                yield int(self._pair_arrays[self.pairs[p_i]][j])
        else:
            # Round-robin pairs, sequential within pair (wrap).
            for k in range(start, self._num_samples):
                p_i = k % self.num_pairs
                p = self.pairs[p_i]
                arr = self._pair_arrays[p]
                yield int(arr[(k // self.num_pairs) % len(arr)])


def compute_pair_balanced_sampler_state(
    step: int, num_samples: int, batch_size: int, num_processes: int
) -> dict:
    """Same resume math as EpisodeAwareSampler / compute_sampler_state."""
    batches_per_epoch = math.ceil(math.ceil(num_samples / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_samples)
    return {"epoch": epoch, "start_index": start_index}

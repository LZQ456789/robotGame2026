#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package

from .configuration_pi05_progress import PI05ProgressConfig
from .error_recovery import ErrorRecoveryConfig, MedicineBoxProgressController
from .task_text import pair_bounds_from_parsed, parse_task_text


class PI05ProgressPytorch(PI05Pytorch):
    """PI0.5 core that predicts action flow and scalar task progress together."""

    def __init__(self, config: PI05ProgressConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        vlm_width = get_gemma_config(config.paligemma_variant).width
        self.progress_head = nn.Sequential(
            nn.Linear(vlm_width, config.progress_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.progress_hidden_dim, 1),
        )

    def _progress_logits(self, prefix_output: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        # Images precede language tokens, so the final valid prefix token is
        # the final non-padding language token and carries the fused context.
        last_valid_indices = prefix_pad_masks.long().sum(dim=1).sub(1).clamp_min(0)
        batch_indices = torch.arange(prefix_output.shape[0], device=prefix_output.device)
        pooled = prefix_output[batch_indices, last_valid_indices]
        return self.progress_head(pooled.float()).squeeze(-1)

    def _encode_prefix(self, images, img_masks, tokens, masks) -> tuple[Tensor, Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_output, _), _ = self.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(att_2d_masks),
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        return prefix_output, prefix_pad_masks

    def predict_progress(self, images, img_masks, tokens, masks) -> Tensor:
        """Return progress in ``[0, 1]`` for the current observation."""
        prefix_output, prefix_pad_masks = self._encode_prefix(images, img_masks, tokens, masks)
        return torch.sigmoid(self._progress_logits(prefix_output, prefix_pad_masks))

    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> tuple[Tensor, Tensor]:
        """Compute per-dimension action loss and progress logits in one VLM pass."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_output, suffix_output), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_output, suffix_output

        prefix_output, suffix_output = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_output = suffix_output[:, -self.config.chunk_size :].to(dtype=torch.float32)
        predicted_flow = self._apply_checkpoint(self.action_out_proj, suffix_output)
        action_losses = F.mse_loss(u_t, predicted_flow, reduction="none")
        progress_logits = self._progress_logits(prefix_output, prefix_pad_masks)
        return action_losses, progress_logits


class PI05ProgressPolicy(PI05Policy):
    """PI0.5 policy with a jointly trained task-progress regression head."""

    config_class = PI05ProgressConfig
    name = "pi05_progress"

    def __init__(
        self,
        config: PI05ProgressConfig,
        dataset_meta: Any | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs,
    ):
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05ProgressPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)

        episode_lengths, episode_starts = self._read_episode_metadata(dataset_meta)
        self.register_buffer("_episode_lengths", episode_lengths, persistent=False)
        self.register_buffer("_episode_starts", episode_starts, persistent=False)
        self.register_buffer(
            "_task_progress_bounds",
            self._read_task_progress_bounds(dataset_meta),
            persistent=False,
        )
        progress_stats = (dataset_stats or {}).get(config.progress_key, {})
        self.register_buffer(
            "_progress_q01",
            torch.as_tensor(progress_stats.get("q01", [])).float().reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "_progress_q99",
            torch.as_tensor(progress_stats.get("q99", [])).float().reshape(-1),
            persistent=False,
        )
        self.reset()

    @staticmethod
    def _read_episode_metadata(dataset_meta: Any | None) -> tuple[Tensor, Tensor]:
        empty = torch.empty(0, dtype=torch.long)
        if dataset_meta is None or getattr(dataset_meta, "episodes", None) is None:
            return empty, empty

        episodes = dataset_meta.episodes
        columns = set(getattr(episodes, "column_names", []))
        if not columns and hasattr(episodes, "columns"):
            columns = set(episodes.columns)
        if "episode_index" not in columns:
            return empty, empty

        episode_indices = [int(v) for v in episodes["episode_index"]]
        if "length" in columns:
            lengths = [int(v) for v in episodes["length"]]
        elif {"dataset_from_index", "dataset_to_index"} <= columns:
            lengths = [
                int(end) - int(start)
                for start, end in zip(
                    episodes["dataset_from_index"],
                    episodes["dataset_to_index"],
                    strict=True,
                )
            ]
        else:
            return empty, empty

        if "dataset_from_index" in columns:
            starts = [int(v) for v in episodes["dataset_from_index"]]
        else:
            starts = []
            next_start = 0
            for length in lengths:
                starts.append(next_start)
                next_start += length

        length_table = torch.zeros(max(episode_indices) + 1, dtype=torch.long)
        start_table = torch.zeros_like(length_table)
        for episode_index, length, start in zip(episode_indices, lengths, starts, strict=True):
            length_table[episode_index] = length
            start_table[episode_index] = start
        return length_table, start_table

    @staticmethod
    def _read_task_progress_bounds(dataset_meta: Any | None) -> Tensor:
        """Map task_index to [completed_before, completed_after] in box units.

        Supports normal packing text and recovery text that records how many
        boxes were already placed when the error occurred, e.g.
        ``Recover after 8 boxes placed: Pack medicine boxes 9 and 10 into the carton``.
        For recovery episodes, ``completed_before`` prefers the explicit placed
        count so progress is anchored at the failure point.
        """
        if dataset_meta is None or getattr(dataset_meta, "tasks", None) is None:
            return torch.empty((0, 2), dtype=torch.float32)

        parsed_rows: list[tuple[int, float, float]] = []
        for task_text, row in dataset_meta.tasks.iterrows():
            bounds = pair_bounds_from_parsed(parse_task_text(str(task_text)))
            if bounds is None:
                continue
            completed_before, completed_after = bounds
            task_index = int(row["task_index"])
            parsed_rows.append((task_index, completed_before, completed_after))

        if not parsed_rows:
            return torch.empty((0, 2), dtype=torch.float32)
        table = torch.full((max(item[0] for item in parsed_rows) + 1, 2), -1.0)
        for task_index, completed_before, completed_after in parsed_rows:
            table[task_index] = torch.tensor([completed_before, completed_after])
        return table

    def build_progress_controller(
        self,
        *,
        enable_error_recovery: bool | None = None,
        language: str | None = None,
    ) -> MedicineBoxProgressController:
        """Create the inference orchestrator (recovery off unless explicitly enabled)."""
        enabled = (
            self.config.enable_error_recovery if enable_error_recovery is None else enable_error_recovery
        )
        task_language = self.config.task_language if language is None else language
        return MedicineBoxProgressController(
            ErrorRecoveryConfig(
                total_task_units=self.config.total_task_units,
                boxes_per_segment=self.config.boxes_per_segment,
                language=task_language,  # type: ignore[arg-type]
                enable_error_recovery=enabled,
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, strict: bool = False, **kwargs):
        # The PI0.5 checkpoint has no progress-head weights, so non-strict loading
        # is required on the first fine-tuning run.
        return super().from_pretrained(
            pretrained_name_or_path,
            strict=strict,
            **kwargs,
        )

    def _get_progress_targets(self, batch: dict[str, Tensor], batch_size: int) -> Tensor:
        if self.config.progress_target_mode == "dataset":
            if self.config.progress_key not in batch:
                raise KeyError(
                    f"Progress target '{self.config.progress_key}' is missing from the batch. "
                    "Add it as a scalar dataset feature or use "
                    "--policy.progress_target_mode=episode_fraction."
                )
            targets = batch[self.config.progress_key].float().reshape(batch_size, -1)[:, 0]
            # The reused PI0.5 preprocessor treats observation.* scalars as
            # STATE and applies QUANTILES normalization. Restore the original
            # [0, 1] labels before computing the auxiliary loss.
            if self._progress_q01.numel() and self._progress_q99.numel():
                q01 = self._progress_q01[0].to(targets.device)
                q99 = self._progress_q99[0].to(targets.device)
                targets = (targets + 1.0) * (q99 - q01) / 2.0 + q01
            return targets.clamp(0, 1)

        if "episode_index" not in batch or ("frame_index" not in batch and "index" not in batch):
            raise KeyError(
                "episode_fraction progress requires episode_index and either frame_index or index."
            )
        if self._episode_lengths.numel() == 0:
            raise RuntimeError(
                "Episode lengths are unavailable. Pass dataset metadata or use dataset progress targets."
            )

        episode_indices = batch["episode_index"].long().reshape(-1)
        if episode_indices.max() >= self._episode_lengths.numel():
            raise IndexError("Batch contains an episode index missing from dataset metadata.")
        reference = batch["frame_index"] if "frame_index" in batch else batch["index"]
        lengths = self._episode_lengths[episode_indices].to(reference.device)
        if (lengths <= 1).any():
            raise ValueError("All episodes must contain at least two frames for episode_fraction progress.")
        if "frame_index" in batch:
            frame_indices = batch["frame_index"].float().reshape(-1)
        else:
            starts = self._episode_starts[episode_indices].to(reference.device)
            frame_indices = batch["index"].float().reshape(-1) - starts
        episode_fraction = (frame_indices / (lengths.float() - 1)).clamp(0, 1)
        if self.config.progress_target_mode == "episode_fraction":
            return episode_fraction

        if "task_index" not in batch:
            raise KeyError("task_fraction progress requires task_index in the training batch.")
        if self._task_progress_bounds.numel() == 0:
            raise RuntimeError(
                "Could not parse box-number ranges from dataset tasks. "
                "Use episode_fraction or provide dataset progress targets."
            )
        task_indices = batch["task_index"].long().reshape(-1)
        if task_indices.max() >= self._task_progress_bounds.shape[0]:
            raise IndexError("Batch contains a task index missing from task metadata.")
        bounds = self._task_progress_bounds[task_indices].to(reference.device)
        if (bounds < 0).any():
            raise ValueError(
                "At least one task does not contain a parseable positive box-number range."
            )
        completed_before, completed_after = bounds.unbind(dim=1)
        completed = completed_before + episode_fraction * (completed_after - completed_before)
        return (completed / self.config.total_task_units).clamp(0, 1)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        action_losses, progress_logits = self.model(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        action_losses = action_losses[:, :, :original_action_dim]
        action_loss_per_sample = action_losses.mean(dim=(1, 2))

        progress_targets = self._get_progress_targets(batch, actions.shape[0]).to(progress_logits.device)
        progress_predictions = torch.sigmoid(progress_logits)
        progress_loss_per_sample = F.smooth_l1_loss(
            progress_predictions,
            progress_targets,
            reduction="none",
        )
        total_loss_per_sample = (
            action_loss_per_sample + self.config.progress_loss_weight * progress_loss_per_sample
        )

        loss_dict = {
            "loss": total_loss_per_sample.mean().item(),
            "action_loss": action_loss_per_sample.mean().item(),
            "progress_loss": progress_loss_per_sample.mean().item(),
            "progress_pred_mean": progress_predictions.mean().item(),
            "progress_target_mean": progress_targets.mean().item(),
            "loss_per_dim": action_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        if reduction == "none":
            return total_loss_per_sample, loss_dict
        return total_loss_per_sample.mean(), loss_dict

    @torch.no_grad()
    def predict_progress(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict current within-episode progress without generating actions."""
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        return self.model.predict_progress(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
        )

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        targets["modules_to_save"] = ["progress_head"]
        return targets

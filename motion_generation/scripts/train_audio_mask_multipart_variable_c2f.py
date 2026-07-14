#!/usr/bin/env python3
"""Train variable-gap coarse-to-fine multipart motion infilling.

The temporal gap is variable, while RVQ levels are generated in q0 -> qN
order. Training can transition from ground-truth quantizer prefixes to detached
self-generated prefixes and uses greedy residual targets after a generated
prefix changes the residual being quantized.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_pt_utils import get_length_grouped_indices


THIS_DIR = Path(__file__).resolve().parent
MOTION_GENERATION_DIR = THIS_DIR.parent
PROJECT_DIR = MOTION_GENERATION_DIR.parent
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer  # noqa: E402
from scripts.train_audio_mask_multipart import (  # noqa: E402
    build_token_layout,
    configure_wandb,
    count_parameters,
    discover_names,
    format_fps_for_dir,
    load_manifest,
    load_sequences,
    parse_part_order,
    read_split_file,
)
from utils.msd.multipart_adapter import MultipartCodebookSet  # noqa: E402
from utils.multipart_motion import PART_ORDER  # noqa: E402


IGNORE_INDEX = -100


@dataclass
class VariableGapMaskExample:
    name: str
    left_idx: int
    right_idx: int
    gap_frames: int
    motion_tokens: list[list[int]]
    audio_features: torch.Tensor


class VariableGapMaskDataset(Dataset):
    """Clip-balanced variable-gap windows that can be resampled each epoch."""

    def __init__(
        self,
        sequences: Sequence[Dict[str, Any]],
        *,
        min_gap_frames: int,
        max_gap_frames: int,
        windows_per_sequence: int,
        gap_bucket_weights: Sequence[float] = (0.35, 0.35, 0.30),
        seed: int = 42,
    ) -> None:
        if min_gap_frames < 1:
            raise ValueError("min_gap_frames must be >= 1")
        if max_gap_frames < min_gap_frames:
            raise ValueError("max_gap_frames must be >= min_gap_frames")
        if windows_per_sequence < 1:
            raise ValueError("windows_per_sequence must be >= 1")
        if len(gap_bucket_weights) != 3 or sum(gap_bucket_weights) <= 0:
            raise ValueError("gap_bucket_weights must contain three positive-total values")

        self.sequences = list(sequences)
        self.min_gap_frames = int(min_gap_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.windows_per_sequence = int(windows_per_sequence)
        self.gap_bucket_weights = tuple(float(value) for value in gap_bucket_weights)
        self.seed = int(seed)
        self.epoch = -1
        self.eligible: list[tuple[int, int, list[int], int]] = []
        self.windows: list[tuple[int, int, int]] = []

        for sequence_idx, item in enumerate(self.sequences):
            usable = self._usable_motion_frames(item)
            valid_gaps = list(
                range(self.min_gap_frames, min(self.max_gap_frames, usable - 2) + 1)
            )
            if not valid_gaps:
                continue
            candidate_count = sum(usable - gap - 1 for gap in valid_gaps)
            sample_count = min(self.windows_per_sequence, candidate_count)
            self.eligible.append((sequence_idx, usable, valid_gaps, sample_count))
        self.resample(0)

    def _bucket_index(self, gap: int) -> int:
        span = self.max_gap_frames - self.min_gap_frames + 1
        short_end = self.min_gap_frames + max(1, math.ceil(span * 0.20)) - 1
        medium_end = self.min_gap_frames + max(2, math.ceil(span * 0.47)) - 1
        if gap <= short_end:
            return 0
        if gap <= medium_end:
            return 1
        return 2

    def _length_weights(self, valid_gaps: Sequence[int]) -> list[float]:
        counts = [0, 0, 0]
        for gap in valid_gaps:
            counts[self._bucket_index(gap)] += 1
        return [
            self.gap_bucket_weights[bucket] / max(1, counts[bucket])
            for gap in valid_gaps
            for bucket in [self._bucket_index(gap)]
        ]

    def resample(self, epoch: int) -> None:
        """Draw new unique windows for every eligible clip for ``epoch``."""
        epoch = int(epoch)
        if epoch == self.epoch:
            return
        rng = random.Random(self.seed + epoch * 1_000_003)
        windows: list[tuple[int, int, int]] = []
        for sequence_idx, usable, valid_gaps, sample_count in self.eligible:
            gap_weights = self._length_weights(valid_gaps)
            selected: set[tuple[int, int]] = set()
            attempts = 0
            max_attempts = max(100, sample_count * 50)
            while len(selected) < sample_count and attempts < max_attempts:
                gap = rng.choices(valid_gaps, weights=gap_weights, k=1)[0]
                left_idx = rng.randint(0, usable - gap - 2)
                selected.add((left_idx, gap))
                attempts += 1

            # The weighted rejection loop can be unlucky on very short clips.
            # Fill deterministically from remaining candidates without changing
            # the requested number of windows.
            if len(selected) < sample_count:
                candidates = [
                    (left_idx, gap)
                    for gap in valid_gaps
                    for left_idx in range(usable - gap - 1)
                    if (left_idx, gap) not in selected
                ]
                rng.shuffle(candidates)
                selected.update(candidates[: sample_count - len(selected)])
            windows.extend(
                (sequence_idx, left_idx, gap)
                for left_idx, gap in sorted(selected)
            )
        self.windows = windows
        self.epoch = epoch

    @staticmethod
    def _usable_motion_frames(item: Dict[str, Any]) -> int:
        token_frames = len(item["motion_tokens"])
        audio_frames = int(item["audio_features"].shape[0])
        audio_fps = float(item["audio_fps"])
        token_fps = float(item["motion_token_fps"])
        if audio_frames <= 0 or audio_fps <= 0 or token_fps <= 0:
            return 0
        audio_usable = int(math.floor((audio_frames - 1) * token_fps / audio_fps)) + 1
        return max(0, min(token_frames, audio_usable))

    @staticmethod
    def _audio_feature_for_frame(item: Dict[str, Any], token_idx: int) -> torch.Tensor:
        audio: np.ndarray = item["audio_features"]
        audio_idx = int(
            round(token_idx * float(item["audio_fps"]) / float(item["motion_token_fps"]))
        )
        audio_idx = max(0, min(audio_idx, len(audio) - 1))
        return torch.as_tensor(audio[audio_idx], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def sequence_lengths(self) -> list[int]:
        return [gap + 2 for _, _, gap in self.windows]

    def __getitem__(self, index: int) -> VariableGapMaskExample:
        sequence_idx, left_idx, gap = self.windows[index]
        item = self.sequences[sequence_idx]
        right_idx = left_idx + gap + 1
        frame_indices = list(range(left_idx, right_idx + 1))
        return VariableGapMaskExample(
            name=str(item.get("name", sequence_idx)),
            left_idx=left_idx,
            right_idx=right_idx,
            gap_frames=gap,
            motion_tokens=[list(item["motion_tokens"][frame]) for frame in frame_indices],
            audio_features=torch.stack(
                [self._audio_feature_for_frame(item, frame) for frame in frame_indices]
            ),
        )


class EpochResamplingLengthGroupedSampler(Sampler[int]):
    """Resample dataset windows, then group similar lengths for each epoch."""

    def __init__(
        self,
        dataset: VariableGapMaskDataset,
        *,
        batch_size: int,
        tokens_per_frame: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.tokens_per_frame = int(tokens_per_frame)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.resample(self.epoch)

    def __iter__(self):
        self.dataset.resample(self.epoch)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        lengths = [
            frames * self.tokens_per_frame
            for frames in self.dataset.sequence_lengths
        ]
        return iter(
            get_length_grouped_indices(
                lengths,
                batch_size=self.batch_size,
                generator=generator,
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)


class VariableGapC2FCollator:
    """Pad variable windows by complete token frames."""

    def __init__(self, config: AudioMotionConfig) -> None:
        self.config = config
        self.mask_token_id = int(getattr(config, "mask_token_id", config.vocab_size - 1))
        self.tokens_per_frame = int(config.num_tokens_per_frame)
        self.codebook_size = int(config.codebook_size)
        self.audio_feat_dim = int(config.audio_feat_dim)

    def _global_frame(self, raw_frame: Sequence[int]) -> list[int]:
        if len(raw_frame) != self.tokens_per_frame:
            raise ValueError(
                f"Expected {self.tokens_per_frame} raw IDs, got {len(raw_frame)}"
            )
        return [
            int(raw_id) + slot * self.codebook_size
            for slot, raw_id in enumerate(raw_frame)
        ]

    def __call__(self, examples: Sequence[VariableGapMaskExample]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_frames = max(len(example.motion_tokens) for example in examples)
        batch_input: list[list[int]] = []
        batch_gt: list[list[int]] = []
        batch_attention: list[list[bool]] = []
        batch_middle: list[list[bool]] = []
        batch_audio: list[torch.Tensor] = []

        for example in examples:
            frames = len(example.motion_tokens)
            if frames != example.gap_frames + 2:
                raise ValueError(
                    f"{example.name}: {frames} frames do not match gap {example.gap_frames}"
                )
            input_ids: list[int] = []
            gt_ids: list[int] = []
            attention: list[bool] = []
            middle: list[bool] = []
            for frame_idx, raw_frame in enumerate(example.motion_tokens):
                global_frame = self._global_frame(raw_frame)
                is_middle = 0 < frame_idx < frames - 1
                gt_ids.extend(global_frame)
                input_ids.extend(
                    [self.mask_token_id] * self.tokens_per_frame
                    if is_middle
                    else global_frame
                )
                attention.extend([True] * self.tokens_per_frame)
                middle.extend([is_middle] * self.tokens_per_frame)

            pad_frames = max_frames - frames
            pad_tokens = pad_frames * self.tokens_per_frame
            input_ids.extend([self.mask_token_id] * pad_tokens)
            gt_ids.extend([self.mask_token_id] * pad_tokens)
            attention.extend([False] * pad_tokens)
            middle.extend([False] * pad_tokens)
            audio = F.pad(example.audio_features, (0, 0, 0, pad_frames))

            batch_input.append(input_ids)
            batch_gt.append(gt_ids)
            batch_attention.append(attention)
            batch_middle.append(middle)
            batch_audio.append(audio)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "gt_ids": torch.tensor(batch_gt, dtype=torch.long),
            "audio_features": torch.stack(batch_audio).float(),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.bool),
            "middle_mask": torch.tensor(batch_middle, dtype=torch.bool),
            "gap_lengths": torch.tensor(
                [example.gap_frames for example in examples], dtype=torch.long
            ),
        }


class ResidualTargetBuilder:
    """Construct original or generated-prefix residual targets from codebooks."""

    def __init__(
        self,
        codebooks: Mapping[str, torch.Tensor],
        part_order: Sequence[str],
        codebook_size: int,
        num_quantizers: int,
    ) -> None:
        self.part_order = tuple(str(part) for part in part_order)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.codebooks = {part: codebooks[part].detach().float().cpu() for part in self.part_order}
        self._device: Optional[torch.device] = None

    def _ensure_device(self, device: torch.device) -> None:
        if self._device == device:
            return
        self.codebooks = {part: value.to(device) for part, value in self.codebooks.items()}
        self._device = device

    @staticmethod
    def _nearest(residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        distances = (
            residual.square().sum(dim=-1, keepdim=True)
            + codebook.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * residual @ codebook.transpose(0, 1)
        )
        return distances.argmin(dim=-1)

    def raw_part_ids(self, global_ids: torch.Tensor, part_idx: int) -> torch.Tensor:
        q = self.num_quantizers
        start = part_idx * q
        offsets = (
            torch.arange(start, start + q, device=global_ids.device) * self.codebook_size
        )
        # Padded frames reuse the mask token and therefore do not map to a
        # valid local code ID. They are removed by the frame-valid mask later;
        # clamping here only keeps intermediate codebook gathers well-defined.
        return (global_ids[..., start : start + q] - offsets).clamp(
            0, self.codebook_size - 1
        )

    def target_latent(self, gt_view: torch.Tensor, part_idx: int) -> torch.Tensor:
        part = self.part_order[part_idx]
        books = self.codebooks[part]
        raw = self.raw_part_ids(gt_view, part_idx).long()
        return sum(books[q].index_select(0, raw[..., q].reshape(-1)).reshape(*raw.shape[:-1], -1)
                   for q in range(self.num_quantizers))

    @torch.no_grad()
    def build_targets(
        self,
        gt_ids: torch.Tensor,
        current_ids: torch.Tensor,
        middle_mask: torch.Tensor,
        *,
        stage: int,
        adaptive: bool,
    ) -> torch.Tensor:
        self._ensure_device(gt_ids.device)
        q = self.num_quantizers
        ntpf = len(self.part_order) * q
        if gt_ids.shape[1] % ntpf != 0:
            raise ValueError("Sequence length is not divisible by tokens_per_frame")
        frames = gt_ids.shape[1] // ntpf
        gt_view = gt_ids.reshape(gt_ids.shape[0], frames, ntpf)
        current_view = current_ids.reshape_as(gt_view)
        middle_view = middle_mask.reshape_as(gt_view)
        frame_valid = middle_view[..., 0]
        targets = torch.full_like(gt_ids, IGNORE_INDEX)
        target_view = targets.reshape_as(gt_view)

        for part_idx, part in enumerate(self.part_order):
            slot = part_idx * q + stage
            if adaptive and stage > 0:
                target = self.target_latent(gt_view, part_idx)
                current_raw = self.raw_part_ids(current_view, part_idx).long()
                cumulative = sum(
                    self.codebooks[part][prefix].index_select(
                        0, current_raw[..., prefix].reshape(-1)
                    ).reshape(*current_raw.shape[:-1], -1)
                    for prefix in range(stage)
                )
                residual = (target - cumulative)[frame_valid]
                raw_target = self._nearest(residual, self.codebooks[part][stage])
            else:
                raw_target = self.raw_part_ids(gt_view, part_idx)[..., stage][frame_valid].long()
            target_view[..., slot][frame_valid] = raw_target + slot * self.codebook_size
        return targets


class VariableGapC2FTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_builder: ResidualTargetBuilder,
        stage_weights: Sequence[float],
        self_forcing_warmup_ratio: float,
        self_forcing_ramp_ratio: float,
        self_forcing_max_prob: float,
        embedding_loss_weight: float,
        final_latent_loss_weight: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(stage_weights) != target_builder.num_quantizers or sum(stage_weights) <= 0:
            raise ValueError("stage_weights must match the number of quantizers")
        self.target_builder = target_builder
        self.stage_weights = torch.tensor(stage_weights, dtype=torch.float64)
        self.self_forcing_warmup_ratio = float(self_forcing_warmup_ratio)
        self.self_forcing_ramp_ratio = float(self_forcing_ramp_ratio)
        self.self_forcing_max_prob = float(self_forcing_max_prob)
        self.embedding_loss_weight = float(embedding_loss_weight)
        self.final_latent_loss_weight = float(final_latent_loss_weight)
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, float] = {}

    @property
    def num_quantizers(self) -> int:
        return self.target_builder.num_quantizers

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not isinstance(dataset, VariableGapMaskDataset):
            return super()._get_train_sampler(train_dataset)
        ntpf = int(self.model.config.num_tokens_per_frame)
        return EpochResamplingLengthGroupedSampler(
            dataset,
            batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
            tokens_per_frame=ntpf,
            seed=int(self.args.seed),
        )

    def _stage_mask(self, middle_mask: torch.Tensor, stage: int) -> torch.Tensor:
        ntpf = int(self.model.config.num_tokens_per_frame)
        slots = torch.arange(middle_mask.shape[1], device=middle_mask.device).remainder(ntpf)
        quantizer = slots.remainder(self.num_quantizers).view(1, -1)
        return middle_mask & quantizer.eq(stage)

    def _record(self, key: str, value: torch.Tensor | float, count: float = 1.0) -> None:
        number = float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)
        self._metric_sums[key] = self._metric_sums.get(key, 0.0) + number * count
        self._metric_counts[key] = self._metric_counts.get(key, 0.0) + count

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        logs = dict(logs)
        for key, total in self._metric_sums.items():
            count = self._metric_counts.get(key, 0.0)
            if count > 0:
                logs[key] = total / count
        self._metric_sums.clear()
        self._metric_counts.clear()
        super().log(logs, *args, **kwargs)

    def _self_forcing_probability(self) -> float:
        max_steps = max(1, int(getattr(self.state, "max_steps", 1)))
        progress = float(getattr(self.state, "global_step", 0)) / max_steps
        if progress <= self.self_forcing_warmup_ratio:
            return 0.0
        if self.self_forcing_ramp_ratio <= 0:
            return self.self_forcing_max_prob
        ramp_progress = (progress - self.self_forcing_warmup_ratio) / self.self_forcing_ramp_ratio
        return self.self_forcing_max_prob * min(1.0, max(0.0, ramp_progress))

    def _sample_stage(self) -> int:
        generator = torch.Generator().manual_seed(
            int(self.args.seed) + int(getattr(self.state, "global_step", 0))
        )
        return int(torch.multinomial(self.stage_weights, 1, generator=generator).item())

    def _use_self_forcing(self, stage: int) -> tuple[bool, float]:
        probability = self._self_forcing_probability()
        if stage == 0 or probability <= 0:
            return False, probability
        generator = torch.Generator().manual_seed(
            int(self.args.seed) + 104729 + int(getattr(self.state, "global_step", 0))
        )
        return bool(torch.rand((), generator=generator).item() < probability), probability

    def _fill_prefix(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        gt_ids: torch.Tensor,
        audio: torch.Tensor,
        attention_mask: torch.Tensor,
        middle_mask: torch.Tensor,
        stage: int,
        self_forced: bool,
    ) -> torch.Tensor:
        current = input_ids.clone()
        if stage == 0:
            return current
        if not self_forced:
            for prefix in range(stage):
                fill = self._stage_mask(middle_mask, prefix)
                current[fill] = gt_ids[fill]
            return current

        with torch.no_grad():
            for prefix in range(stage):
                logits = model(
                    input_ids=current,
                    audio_features=audio,
                    attention_mask=attention_mask,
                )
                predictions = logits.argmax(dim=-1)
                fill = self._stage_mask(middle_mask, prefix)
                current[fill] = predictions[fill]
        return current

    def _embedding_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gt_ids: torch.Tensor,
        current_ids: torch.Tensor,
        middle_mask: torch.Tensor,
        stage: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.target_builder._ensure_device(logits.device)
        q = self.num_quantizers
        ntpf = int(self.model.config.num_tokens_per_frame)
        frames = logits.shape[1] // ntpf
        target_view = targets.reshape(targets.shape[0], frames, ntpf)
        gt_view = gt_ids.reshape_as(target_view)
        current_view = current_ids.reshape_as(target_view)
        middle_view = middle_mask.reshape_as(target_view)
        embedding_losses = []
        final_losses = []

        for part_idx, part in enumerate(self.target_builder.part_order):
            slot = part_idx * q + stage
            valid = middle_view[..., slot]
            if not bool(valid.any()):
                continue
            flat_logits = logits.reshape(logits.shape[0], frames, ntpf, -1)[..., slot, :][valid]
            start = slot * self.target_builder.codebook_size
            local_logits = flat_logits[:, start : start + self.target_builder.codebook_size]
            probabilities = local_logits.float().softmax(dim=-1)
            book = self.target_builder.codebooks[part][stage]
            expected = probabilities @ book
            raw_target = (target_view[..., slot][valid] - start).long()
            embedding_losses.append(F.l1_loss(expected, book.index_select(0, raw_target)))

            if stage == q - 1:
                target_latent = self.target_builder.target_latent(gt_view, part_idx)[valid]
                current_raw = self.target_builder.raw_part_ids(current_view, part_idx).long()
                prefix_latent = sum(
                    self.target_builder.codebooks[part][prefix].index_select(
                        0, current_raw[..., prefix][valid]
                    )
                    for prefix in range(stage)
                )
                final_losses.append(F.l1_loss(prefix_latent + expected, target_latent))

        zero = logits.sum() * 0.0
        embed_loss = torch.stack(embedding_losses).mean() if embedding_losses else zero
        final_loss = torch.stack(final_losses).mean() if final_losses else zero
        return embed_loss, final_loss

    def _record_stage_metrics(
        self,
        prefix: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gt_ids: torch.Tensor,
        middle_mask: torch.Tensor,
        stage: int,
        ce_loss: torch.Tensor,
        embed_loss: torch.Tensor,
        final_loss: torch.Tensor,
    ) -> None:
        predictions = logits.argmax(dim=-1)
        valid = targets.ne(IGNORE_INDEX)
        count = int(valid.sum().item())
        if count == 0:
            return
        self._record(f"{prefix}/q{stage}_ce", ce_loss)
        self._record(f"{prefix}/q{stage}_embed", embed_loss)
        if stage == self.num_quantizers - 1:
            self._record(f"{prefix}/q{stage}_final_latent", final_loss)
        self._record(
            f"{prefix}/q{stage}_adaptive_acc",
            predictions[valid].eq(targets[valid]).float().mean(),
        )
        original_valid = self._stage_mask(middle_mask, stage)
        self._record(
            f"{prefix}/q{stage}_original_acc",
            predictions[original_valid].eq(gt_ids[original_valid]).float().mean(),
        )
        ntpf = int(self.model.config.num_tokens_per_frame)
        slots = torch.arange(targets.shape[1], device=targets.device).remainder(ntpf)
        for part_idx, part in enumerate(self.target_builder.part_order):
            slot = part_idx * self.num_quantizers + stage
            part_valid = valid & slots.view(1, -1).eq(slot)
            part_count = int(part_valid.sum().item())
            if part_count:
                self._record(
                    f"{prefix}/{part}_q{stage}_adaptive_acc",
                    predictions[part_valid].eq(targets[part_valid]).float().mean(),
                )

    def _stage_loss(
        self,
        model: torch.nn.Module,
        current_ids: torch.Tensor,
        gt_ids: torch.Tensor,
        audio: torch.Tensor,
        attention_mask: torch.Tensor,
        middle_mask: torch.Tensor,
        *,
        stage: int,
        adaptive: bool,
        metric_prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self.target_builder.build_targets(
            gt_ids,
            current_ids,
            middle_mask,
            stage=stage,
            adaptive=adaptive,
        )
        logits = model(
            input_ids=current_ids,
            audio_features=audio,
            attention_mask=attention_mask,
        )
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        embed_loss, final_loss = self._embedding_losses(
            logits, targets, gt_ids, current_ids, middle_mask, stage
        )
        loss = (
            ce_loss
            + self.embedding_loss_weight * embed_loss
            + self.final_latent_loss_weight * final_loss
        )
        self._record_stage_metrics(
            metric_prefix,
            logits,
            targets,
            gt_ids,
            middle_mask,
            stage,
            ce_loss,
            embed_loss,
            final_loss,
        )
        return loss, logits

    def _evaluation_rollout(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        current = inputs["input_ids"].clone()
        losses = []
        for stage in range(self.num_quantizers):
            loss, logits = self._stage_loss(
                model,
                current,
                inputs["gt_ids"],
                inputs["audio_features"],
                inputs["attention_mask"],
                inputs["middle_mask"],
                stage=stage,
                adaptive=stage > 0,
                metric_prefix="eval_c2f",
            )
            losses.append(loss)
            predictions = logits.argmax(dim=-1)
            fill = self._stage_mask(inputs["middle_mask"], stage)
            current[fill] = predictions[fill]

        valid = inputs["middle_mask"]
        self._record(
            "eval_c2f/final_original_acc",
            current[valid].eq(inputs["gt_ids"][valid]).float().mean(),
        )
        for gap in inputs["gap_lengths"].unique().tolist():
            rows = inputs["gap_lengths"].eq(int(gap))
            gap_valid = valid & rows.unsqueeze(-1)
            if bool(gap_valid.any()):
                self._record(
                    f"eval_c2f/gap_{int(gap)}_original_acc",
                    current[gap_valid].eq(inputs["gt_ids"][gap_valid]).float().mean(),
                )
        return torch.stack(losses).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        if not model.training:
            loss = self._evaluation_rollout(model, inputs)
            return (loss, {"loss": loss}) if return_outputs else loss

        stage = self._sample_stage()
        self_forced, probability = self._use_self_forcing(stage)
        current = self._fill_prefix(
            model,
            inputs["input_ids"],
            inputs["gt_ids"],
            inputs["audio_features"],
            inputs["attention_mask"],
            inputs["middle_mask"],
            stage,
            self_forced,
        )
        loss, _ = self._stage_loss(
            model,
            current,
            inputs["gt_ids"],
            inputs["audio_features"],
            inputs["attention_mask"],
            inputs["middle_mask"],
            stage=stage,
            adaptive=self_forced and stage > 0,
            metric_prefix="train_c2f",
        )
        self._record("train_c2f/self_forcing_probability", probability)
        self._record("train_c2f/self_forced_batch", float(self_forced))
        self._record("train_c2f/gap_mean", inputs["gap_lengths"].float().mean())
        return (loss, {"loss": loss}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def parse_float_list(text: str, expected: int, name: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values")
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(f"{name} values must be non-negative with a positive sum")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument(
        "--data_dir",
        default=str(PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs"),
    )
    parser.add_argument("--motion_token_dir", default=None)
    parser.add_argument("--audio_feat_dir", default=None)
    parser.add_argument("--train_split_file", default=None)
    parser.add_argument("--eval_split_file", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_eval_clips", type=int, default=None)
    parser.add_argument("--train_windows_per_sequence", type=int, default=1)
    parser.add_argument("--eval_windows_per_sequence", type=int, default=4)

    parser.add_argument("--min_gap_frames", type=int, default=1)
    parser.add_argument("--max_gap_frames", type=int, default=15)
    parser.add_argument("--gap_bucket_weights", default="0.35,0.35,0.30")
    parser.add_argument("--audio_fps", type=float, default=10.0)
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--motion_token_fps", type=float, default=None)
    parser.add_argument("--motion_token_unit_length", type=float, default=None)

    parser.add_argument("--part_order", default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--num_quantizers_per_part", type=int, default=None)
    parser.add_argument("--num_tokens_per_frame", type=int, default=None)
    codec_root = PROJECT_DIR / "checkpoints" / "multipart_rvqvae"
    for part in PART_ORDER:
        parser.add_argument(
            f"--{part}_ckpt",
            type=Path,
            default=codec_root / f"rvq_{part}_512x4_bs256_cosine" / "model" / "best.pth",
        )

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--stage_weights", default="0.35,0.25,0.20,0.20")
    parser.add_argument("--self_forcing_warmup_ratio", type=float, default=0.10)
    parser.add_argument("--self_forcing_ramp_ratio", type=float, default=0.30)
    parser.add_argument("--self_forcing_max_prob", type=float, default=1.0)
    parser.add_argument("--embedding_loss_weight", type=float, default=0.1)
    parser.add_argument("--final_latent_loss_weight", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=100.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run_batches", type=int, default=0)

    parser.add_argument("--report_to", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_tags", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not 0 <= args.self_forcing_max_prob <= 1:
        raise ValueError("self_forcing_max_prob must be in [0,1]")
    if args.min_gap_frames < 1 or args.max_gap_frames < args.min_gap_frames:
        raise ValueError("Invalid gap range")

    data_dir = Path(args.data_dir)
    token_dir = Path(
        args.motion_token_dir or data_dir / "motion_token_data_multipart_512x4"
    )
    audio_dir = Path(
        args.audio_feat_dir
        or data_dir / f"audio_features_hubert_layer9_fps{format_fps_for_dir(args.audio_fps)}"
    )
    manifest = load_manifest(token_dir)
    part_order = parse_part_order(args.part_order, manifest)
    num_quantizers = int(
        args.num_quantizers_per_part
        or (manifest.get("num_quantizers") if manifest else None)
        or 4
    )
    ntpf = int(
        args.num_tokens_per_frame
        or (manifest.get("tokens_per_frame") if manifest else None)
        or len(part_order) * num_quantizers
    )
    if ntpf != len(part_order) * num_quantizers:
        raise ValueError("tokens_per_frame must equal parts * quantizers")
    codebook_size = int(
        args.codebook_size or (manifest.get("codebook_size") if manifest else None) or 512
    )
    vocab_size = codebook_size * ntpf + 1
    max_seq_len = (args.max_gap_frames + 2) * ntpf
    if max_seq_len > args.max_position_embeddings:
        raise ValueError(
            f"Maximum sequence has {max_seq_len} tokens but max_position_embeddings="
            f"{args.max_position_embeddings}"
        )

    train_split = read_split_file(Path(args.train_split_file) if args.train_split_file else None)
    eval_split = read_split_file(Path(args.eval_split_file) if args.eval_split_file else None)
    train_names = discover_names(token_dir, audio_dir, train_split)
    if eval_split is not None:
        eval_names = discover_names(token_dir, audio_dir, eval_split)
    elif args.eval_ratio > 0 and len(train_names) > 1:
        shuffled = train_names[:]
        random.Random(args.seed).shuffle(shuffled)
        eval_size = max(1, int(len(shuffled) * args.eval_ratio))
        eval_names, train_names = sorted(shuffled[:eval_size]), sorted(shuffled[eval_size:])
    else:
        eval_names = []

    load_kwargs = dict(
        codebook_size=codebook_size,
        num_tokens_per_frame=ntpf,
        audio_fps=args.audio_fps,
        source_motion_fps_fallback=args.motion_fps,
        motion_token_fps_override=args.motion_token_fps,
        motion_token_unit_length_override=args.motion_token_unit_length,
    )
    train_sequences, train_stats = load_sequences(
        train_names, token_dir, audio_dir, max_sequences=args.max_train_clips, **load_kwargs
    )
    eval_sequences, eval_stats = load_sequences(
        eval_names, token_dir, audio_dir, max_sequences=args.max_eval_clips, **load_kwargs
    )
    gap_weights = parse_float_list(args.gap_bucket_weights, 3, "gap_bucket_weights")
    train_dataset = VariableGapMaskDataset(
        train_sequences,
        min_gap_frames=args.min_gap_frames,
        max_gap_frames=args.max_gap_frames,
        windows_per_sequence=args.train_windows_per_sequence,
        gap_bucket_weights=gap_weights,
        seed=args.seed,
    )
    eval_dataset = (
        VariableGapMaskDataset(
            eval_sequences,
            min_gap_frames=args.min_gap_frames,
            max_gap_frames=args.max_gap_frames,
            windows_per_sequence=args.eval_windows_per_sequence,
            gap_bucket_weights=gap_weights,
            seed=args.seed + 1,
        )
        if eval_sequences
        else None
    )
    if not train_dataset:
        raise ValueError("No variable-gap training windows were created")

    token_layout = (
        manifest.get("token_layout")
        if manifest and manifest.get("token_layout")
        else build_token_layout(part_order, num_quantizers)
    )
    if args.model_name_or_path:
        model = AudioMotionTransformer.from_pretrained(
            args.model_name_or_path, local_files_only=True
        )
        if model.config.vocab_size != vocab_size or model.config.num_tokens_per_frame != ntpf:
            raise ValueError("Checkpoint vocabulary/token layout does not match multipart data")
        if model.config.max_position_embeddings < max_seq_len:
            raise ValueError("Checkpoint position table is too short for max_gap_frames")
    else:
        config = AudioMotionConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_position_embeddings,
            vocab_size=vocab_size,
            codebook_size=codebook_size,
            audio_feat_dim=args.audio_feat_dim,
            num_tokens_per_frame=ntpf,
            num_frames=args.max_gap_frames + 2,
            dropout=args.dropout,
            constrain_token_logits=True,
        )
        model = AudioMotionTransformer(config)
    model.config.part_order = list(part_order)
    model.config.num_quantizers_per_part = num_quantizers
    model.config.token_layout = token_layout
    model.config.mask_token_id = vocab_size - 1
    model.config.min_gap_frames = args.min_gap_frames
    model.config.max_gap_frames = args.max_gap_frames
    model.config.variable_gap = True
    model.config.generation_order = "quantizer_coarse_to_fine"
    model.config.constrain_token_logits = True

    checkpoint_paths = {part: Path(getattr(args, f"{part}_ckpt")) for part in part_order}
    codebooks = MultipartCodebookSet.from_checkpoints(
        checkpoint_paths, torch.device("cpu"), part_order
    )
    if codebooks.codebook_size != codebook_size or codebooks.num_quantizers != num_quantizers:
        raise ValueError("Codec checkpoints do not match token manifest")
    model.config.part_checkpoints = {part: str(path) for part, path in checkpoint_paths.items()}
    target_builder = ResidualTargetBuilder(
        codebooks.codebooks, part_order, codebook_size, num_quantizers
    )
    collator = VariableGapC2FCollator(model.config)
    stage_weights = parse_float_list(args.stage_weights, num_quantizers, "stage_weights")

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = (
        args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    default_run_name = (
        f"mask_multipart_variable_c2f_gap{args.min_gap_frames}-{args.max_gap_frames}_"
        f"bs{effective_batch}"
    )
    configure_wandb(args, default_run_name)
    total_params, trainable_params = count_parameters(model)
    print("=" * 76)
    print("Variable-gap coarse-to-fine multipart infilling")
    print(f"Output:           {args.output_dir}")
    print(f"Initialization:   {args.model_name_or_path or 'from scratch'}")
    print(f"Train clips:      {len(train_sequences)} / {train_stats['requested']}")
    print(f"Train windows:    {len(train_dataset)}")
    print(
        "Train sampling:   "
        f"up to {args.train_windows_per_sequence} unique windows/eligible clip, "
        "resampled every epoch"
    )
    if eval_dataset is not None:
        print(f"Eval clips:       {len(eval_sequences)} / {eval_stats['requested']}")
        print(f"Eval windows:     {len(eval_dataset)}")
        print("Eval sampling:    fixed across evaluations")
    print(f"Gap range:        {args.min_gap_frames}-{args.max_gap_frames} token frames")
    print(f"Max sequence:     {max_seq_len} tokens ({args.max_gap_frames + 2} frames)")
    print(f"Gap weights:      {gap_weights}")
    print(f"Stage weights:    {stage_weights}")
    print(
        "Self-forcing:     "
        f"warmup={args.self_forcing_warmup_ratio}, ramp={args.self_forcing_ramp_ratio}, "
        f"max={args.self_forcing_max_prob}"
    )
    print(
        f"Loss weights:     embed={args.embedding_loss_weight}, "
        f"final_latent={args.final_latent_loss_weight}"
    )
    print(f"Parameters:       {total_params:,} / {trainable_params:,} trainable")
    print(f"Effective batch:  {effective_batch}")
    print(f"Steps/epoch:      {steps_per_epoch}")
    print(f"Total steps:      {total_steps}")
    print("=" * 76)

    training_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_grad_norm": args.max_grad_norm,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "report_to": args.report_to,
        "dataloader_num_workers": args.dataloader_num_workers,
        "run_name": args.wandb_run_name or default_run_name,
    }
    if eval_dataset is not None and len(eval_dataset) > 0:
        training_kwargs.update({"eval_strategy": "steps", "eval_steps": args.eval_steps})
    else:
        training_kwargs["eval_strategy"] = "no"
    training_args = TrainingArguments(**training_kwargs)
    trainer = VariableGapC2FTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        target_builder=target_builder,
        stage_weights=stage_weights,
        self_forcing_warmup_ratio=args.self_forcing_warmup_ratio,
        self_forcing_ramp_ratio=args.self_forcing_ramp_ratio,
        self_forcing_max_prob=args.self_forcing_max_prob,
        embedding_loss_weight=args.embedding_loss_weight,
        final_latent_loss_weight=args.final_latent_loss_weight,
    )

    if args.dry_run_batches > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).train()
        loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx > args.dry_run_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = trainer.compute_loss(model, batch)
            loss.backward()
            print(
                f"dry_run {batch_idx}/{args.dry_run_batches} loss={float(loss):.5f} "
                f"shape={tuple(batch['input_ids'].shape)} gaps={batch['gap_lengths'].tolist()}"
            )
            model.zero_grad(set_to_none=True)
        return

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()

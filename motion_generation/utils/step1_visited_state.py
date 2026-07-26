"""Local DAgger-like visited-state rollouts for Step 1 endpoint training.

Unlike the full-history self-forcing curriculum, this module rolls only the
one or two anchors immediately preceding the interval selected for frozen
Step 2 guidance.  The resulting previous anchor is a state actually visited by
the current planner.  Unselected rows remain exact GT-history replay.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from models.step1_mimi_planner import IGNORE_INDEX, MimiQwenPlanner
from utils.adaptive_anchor_tokens import BODY_SLOT_COUNT
from utils.step1_self_forcing import generate_history_batch


@dataclass
class VisitedStateBatchStats:
    clips: int = 0
    anchors: int = 0
    tokens: int = 0
    correct: int = 0
    depth1_clips: int = 0
    depth2_clips: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.tokens)


def deterministic_visited_indices(
    names: Sequence[str],
    probability: float,
    *,
    seed: int,
    epoch: int,
    batch_index: int,
) -> list[int]:
    """Select an exact resume-deterministic replay/visited-state split."""

    count = min(len(names), max(0, round(len(names) * float(probability))))
    if count == 0:
        return []
    scored = []
    for index, name in enumerate(names):
        key = f"visited|{seed}|{epoch}|{batch_index}|{name}".encode("utf-8")
        score = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        scored.append((score, index))
    return sorted(index for _, index in sorted(scored)[:count])


def local_rollout_target_slots(
    *,
    target_slots: torch.Tensor,
    target_anchor_ids: torch.Tensor,
    selected_anchor_groups: torch.Tensor,
    names: Sequence[str],
    rollout_depths: Sequence[int],
    seed: int,
    epoch: int,
    batch_index: int,
) -> tuple[torch.Tensor, list[int]]:
    """Mask targets to the one/two anchors immediately before each endpoint."""

    if target_slots.shape != target_anchor_ids.shape:
        raise ValueError("target_slots and target_anchor_ids must share [B,L]")
    batch = target_slots.shape[0]
    if tuple(selected_anchor_groups.shape) != (batch,):
        raise ValueError("selected_anchor_groups must have shape [B]")
    if len(names) != batch:
        raise ValueError("names must match the rollout batch")
    requested_depths = sorted({int(value) for value in rollout_depths})
    if not requested_depths or any(value not in {1, 2} for value in requested_depths):
        raise ValueError("rollout_depths must contain one or both of [1,2]")

    result = torch.full_like(target_slots, -1)
    selected_depths: list[int] = []
    for row, name in enumerate(names):
        right_group = int(selected_anchor_groups[row])
        valid_depths = [
            value for value in requested_depths if value <= right_group
        ]
        if not valid_depths:
            raise ValueError(
                f"{name}: selected group {right_group} has no previous "
                "anchor available for a visited-state rollout"
            )
        key = (
            f"visited-depth|{seed}|{epoch}|{batch_index}|{name}|{right_group}"
        ).encode("utf-8")
        depth = valid_depths[
            int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
            % len(valid_depths)
        ]
        first_group = right_group - depth
        mask = (
            target_anchor_ids[row].ge(first_group)
            & target_anchor_ids[row].lt(right_group)
            & target_slots[row].ge(0)
        )
        if int(mask.sum()) != depth * BODY_SLOT_COUNT:
            raise ValueError(
                f"{name}: depth-{depth} local rollout selected "
                f"{int(mask.sum())} tokens, expected {depth * BODY_SLOT_COUNT}"
            )
        result[row, mask] = target_slots[row, mask]
        selected_depths.append(depth)
    return result, selected_depths


def apply_visited_state_history(
    model: MimiQwenPlanner,
    batch: Mapping[str, torch.Tensor],
    names: Sequence[str],
    selected_indices: Sequence[int],
    *,
    rollout_depths: Sequence[int],
    seed: int,
    epoch: int,
    batch_index: int,
    microbatch_size: int,
    use_bf16: bool,
) -> tuple[torch.Tensor, VisitedStateBatchStats]:
    """Roll local preceding anchors and return a mixed visited/GT input batch."""

    generated_input_ids = batch["input_ids"].clone()
    stats = VisitedStateBatchStats()
    if not selected_indices:
        return generated_input_ids, stats
    if microbatch_size <= 0:
        raise ValueError("visited-state microbatch_size must be positive")

    for start in range(0, len(selected_indices), microbatch_size):
        raw_indices = selected_indices[start : start + microbatch_size]
        indices = torch.as_tensor(
            raw_indices,
            dtype=torch.long,
            device=batch["input_ids"].device,
        )
        chunk_names = [str(names[index]) for index in raw_indices]
        rollout_slots, depths = local_rollout_target_slots(
            target_slots=batch["target_slots"].index_select(0, indices),
            target_anchor_ids=batch["target_anchor_ids"].index_select(
                0, indices
            ),
            selected_anchor_groups=batch["selected_anchor_groups"].index_select(
                0, indices
            ),
            names=chunk_names,
            rollout_depths=rollout_depths,
            seed=seed,
            epoch=epoch,
            batch_index=batch_index,
        )
        result = generate_history_batch(
            model,
            input_ids=batch["input_ids"].index_select(0, indices),
            attention_mask=batch["attention_mask"].index_select(0, indices),
            audio_codes=batch["audio_codes"].index_select(0, indices),
            target_slots=rollout_slots,
            use_bf16=use_bf16,
        )
        generated_input_ids.index_copy_(0, indices, result.input_ids)
        labels = batch["motion_local_labels"].index_select(0, indices)
        target_mask = result.predicted_local_ids.ge(0)
        if bool(labels[target_mask].eq(IGNORE_INDEX).any()):
            raise ValueError("Visited-state rollout selected an ignored label")
        stats.clips += len(raw_indices)
        stats.anchors += sum(depths)
        stats.tokens += int(target_mask.sum())
        stats.correct += int(
            result.predicted_local_ids[target_mask]
            .eq(labels[target_mask])
            .sum()
        )
        stats.depth1_clips += sum(value == 1 for value in depths)
        stats.depth2_clips += sum(value == 2 for value in depths)
    return generated_input_ids, stats

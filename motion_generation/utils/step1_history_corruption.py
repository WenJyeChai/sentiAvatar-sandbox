"""Deterministic on-the-fly corruption of sparse Step 1 motion history."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from models.step1_mimi_planner import IGNORE_INDEX
from utils.adaptive_anchor_tokens import BODY_CODEBOOK_SIZE, BODY_SLOT_COUNT


@dataclass(frozen=True)
class HistoryCorruptionCurriculumState:
    phase_index: int
    example_probability: float
    anchor_probability: float


@dataclass
class HistoryCorruptionStats:
    clips: int = 0
    anchors: int = 0
    donor_anchors: int = 0
    previous_copy_anchors: int = 0


def history_corruption_curriculum_state(
    epoch_number: int,
    phases: Sequence[Mapping[str, Any]],
) -> HistoryCorruptionCurriculumState:
    """Return the explicit corruption phase for a one-based epoch."""

    if epoch_number <= 0:
        raise ValueError("epoch_number must be one-based and positive")
    for phase_index, phase in enumerate(phases):
        if int(phase["start_epoch"]) <= epoch_number <= int(
            phase["end_epoch"]
        ):
            return HistoryCorruptionCurriculumState(
                phase_index=phase_index,
                example_probability=float(phase["example_probability"]),
                anchor_probability=float(phase["anchor_probability"]),
            )
    raise ValueError(
        f"No history-corruption curriculum phase covers epoch {epoch_number}"
    )


def _hash_score(*values: object) -> int:
    encoded = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def deterministic_corruption_indices(
    names: Sequence[str],
    probability: float,
    *,
    seed: int,
    epoch: int,
    batch_index: int,
) -> list[int]:
    """Select an exact, resume-stable number of examples on each rank."""

    if not 0 <= probability <= 1:
        raise ValueError("history-corruption probability must be in [0,1]")
    count = min(
        len(names),
        max(0, math.floor(len(names) * probability + 0.5)),
    )
    scored = [
        (
            _hash_score(
                "history-example", seed, epoch, batch_index, name
            ),
            index,
        )
        for index, name in enumerate(names)
    ]
    return sorted(index for _, index in sorted(scored)[:count])


def _anchor_positions(
    target_slots: torch.Tensor,
    motion_local_labels: torch.Tensor,
) -> list[torch.Tensor]:
    positions = target_slots.ge(0).nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() % BODY_SLOT_COUNT:
        raise ValueError(
            "History corruption requires complete 16-slot anchors"
        )
    anchor_count = int(positions.numel() // BODY_SLOT_COUNT)
    if anchor_count == 0:
        return []
    groups = list(positions.view(anchor_count, BODY_SLOT_COUNT).unbind(0))
    expected = torch.arange(
        BODY_SLOT_COUNT, device=target_slots.device
    )
    for group in groups:
        if not torch.equal(target_slots.index_select(0, group), expected):
            raise ValueError(
                "History-corruption slots must repeat in 0..15 order"
            )
        if bool(
            motion_local_labels.index_select(0, group).eq(IGNORE_INDEX).any()
        ):
            raise ValueError(
                "A history-corruption anchor is missing GT local labels"
            )
    return groups


def apply_history_corruption(
    batch: Mapping[str, torch.Tensor],
    names: Sequence[str],
    selected_indices: Sequence[int],
    *,
    motion_token_ids: torch.Tensor,
    anchor_probability: float,
    donor_probability: float,
    seed: int,
    epoch: int,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, HistoryCorruptionStats]:
    """Replace previous anchors while preserving every clean CE target.

    The final supervised anchor is never corrupted because it is not temporal
    history for a later anchor. A donor operation copies one complete valid
    16-slot GT anchor, preferably from another example in the same batch.
    Previous-copy corruption duplicates the immediately preceding predicted
    anchor input and therefore can propagate an already-corrupted state.
    """

    input_ids = batch["input_ids"]
    target_slots = batch["target_slots"]
    labels = batch["motion_local_labels"]
    if not (
        input_ids.shape == target_slots.shape == labels.shape
        and input_ids.ndim == 2
    ):
        raise ValueError(
            "input_ids, target_slots and motion_local_labels must share [B,L]"
        )
    if len(names) != input_ids.shape[0]:
        raise ValueError("names must match the history-corruption batch")
    if not 0 <= anchor_probability <= 1:
        raise ValueError("anchor_probability must be in [0,1]")
    if not 0 <= donor_probability <= 1:
        raise ValueError("donor_probability must be in [0,1]")
    expected_table_shape = (BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE)
    if tuple(motion_token_ids.shape) != expected_table_shape:
        raise ValueError(
            f"motion_token_ids must have shape {expected_table_shape}"
        )

    groups_by_row = [
        _anchor_positions(target_slots[row], labels[row])
        for row in range(input_ids.shape[0])
    ]
    corrupted = input_ids.clone()
    corrupted_rows = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )
    stats = HistoryCorruptionStats()

    for row in selected_indices:
        row = int(row)
        if not 0 <= row < input_ids.shape[0]:
            raise IndexError(f"Selected corruption row is invalid: {row}")
        groups = groups_by_row[row]
        # Exclude the last anchor: no subsequent anchor uses it as history.
        eligible_groups = list(range(max(0, len(groups) - 1)))
        if not eligible_groups or anchor_probability <= 0:
            continue
        corrupt_count = min(
            len(eligible_groups),
            max(
                1,
                math.floor(
                    len(eligible_groups) * anchor_probability + 0.5
                ),
            ),
        )
        scored_groups = sorted(
            (
                _hash_score(
                    "history-anchor",
                    seed,
                    epoch,
                    batch_index,
                    names[row],
                    group_index,
                ),
                group_index,
            )
            for group_index in eligible_groups
        )
        selected_groups = sorted(
            group_index
            for _, group_index in scored_groups[:corrupt_count]
        )

        row_corruptions = 0
        for group_index in selected_groups:
            operation_score = _hash_score(
                "history-operation",
                seed,
                epoch,
                batch_index,
                names[row],
                group_index,
            )
            use_donor = (
                operation_score / float(2**64) < donor_probability
                or group_index == 0
            )
            target_positions = groups[group_index]

            if not use_donor:
                previous_positions = groups[group_index - 1]
                corrupted[row, target_positions] = corrupted[
                    row, previous_positions
                ]
                stats.previous_copy_anchors += 1
            else:
                donor_rows = [
                    candidate
                    for candidate, candidate_groups in enumerate(groups_by_row)
                    if candidate != row and candidate_groups
                ]
                if donor_rows:
                    donor_row = donor_rows[
                        _hash_score(
                            "history-donor-row",
                            seed,
                            epoch,
                            batch_index,
                            names[row],
                            group_index,
                        )
                        % len(donor_rows)
                    ]
                    donor_groups = groups_by_row[donor_row]
                else:
                    donor_row = row
                    donor_groups = [
                        group
                        for candidate_index, group in enumerate(groups)
                        if candidate_index != group_index
                    ]
                if not donor_groups:
                    continue
                donor_group = donor_groups[
                    _hash_score(
                        "history-donor-anchor",
                        seed,
                        epoch,
                        batch_index,
                        names[row],
                        group_index,
                    )
                    % len(donor_groups)
                ]
                donor_local_ids = labels[donor_row].index_select(
                    0, donor_group
                )
                slots = torch.arange(
                    BODY_SLOT_COUNT, device=input_ids.device
                )
                donor_token_ids = motion_token_ids[
                    slots, donor_local_ids
                ]
                corrupted[row, target_positions] = donor_token_ids
                stats.donor_anchors += 1

            row_corruptions += 1

        if row_corruptions:
            corrupted_rows[row] = True
            stats.clips += 1
            stats.anchors += row_corruptions

    return corrupted, corrupted_rows, stats

from __future__ import annotations

import sys
from pathlib import Path

import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.step1_mimi_planner import IGNORE_INDEX  # noqa: E402
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    body_global_id,
)
from utils.step1_history_corruption import (  # noqa: E402
    apply_history_corruption,
    deterministic_corruption_indices,
    history_corruption_curriculum_state,
)


def synthetic_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch_size = 3
    anchor_count = 4
    length = 2 + anchor_count * BODY_SLOT_COUNT
    input_ids = torch.ones((batch_size, length), dtype=torch.long)
    target_slots = torch.full_like(input_ids, -1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    table = torch.tensor(
        [
            [
                body_global_id(slot, local_id)
                for local_id in range(BODY_CODEBOOK_SIZE)
            ]
            for slot in range(BODY_SLOT_COUNT)
        ],
        dtype=torch.long,
    )
    for row in range(batch_size):
        for anchor_index in range(anchor_count):
            start = 2 + anchor_index * BODY_SLOT_COUNT
            positions = slice(start, start + BODY_SLOT_COUNT)
            target_slots[row, positions] = torch.arange(BODY_SLOT_COUNT)
            local_ids = torch.tensor(
                [
                    (
                        row * 101
                        + anchor_index * 47
                        + slot * 13
                    )
                    % BODY_CODEBOOK_SIZE
                    for slot in range(BODY_SLOT_COUNT)
                ]
            )
            labels[row, positions] = local_ids
            input_ids[row, positions] = table[
                torch.arange(BODY_SLOT_COUNT), local_ids
            ]
    return {
        "input_ids": input_ids,
        "target_slots": target_slots,
        "motion_local_labels": labels,
    }, table


def test_curriculum_and_example_selection_are_deterministic() -> None:
    phases = [
        {
            "start_epoch": 1,
            "end_epoch": 4,
            "example_probability": 0.2,
            "anchor_probability": 0.2,
        },
        {
            "start_epoch": 5,
            "end_epoch": 10,
            "example_probability": 0.4,
            "anchor_probability": 0.3,
        },
        {
            "start_epoch": 11,
            "end_epoch": 20,
            "example_probability": 0.6,
            "anchor_probability": 0.4,
        },
    ]
    state = history_corruption_curriculum_state(12, phases)
    assert state.phase_index == 2
    assert state.example_probability == 0.6
    assert state.anchor_probability == 0.4
    names = [f"clip-{index}" for index in range(10)]
    first = deterministic_corruption_indices(
        names, 0.4, seed=42, epoch=4, batch_index=3
    )
    second = deterministic_corruption_indices(
        names, 0.4, seed=42, epoch=4, batch_index=3
    )
    assert first == second
    assert len(first) == 4


def test_valid_donor_corruption_preserves_targets_and_final_anchor() -> None:
    batch, table = synthetic_batch()
    original_ids = batch["input_ids"].clone()
    original_labels = batch["motion_local_labels"].clone()
    corrupted, mask, stats = apply_history_corruption(
        batch,
        ["a", "b", "c"],
        [0],
        motion_token_ids=table,
        anchor_probability=1.0,
        donor_probability=1.0,
        seed=42,
        epoch=0,
        batch_index=0,
    )
    assert torch.equal(batch["input_ids"], original_ids)
    assert torch.equal(batch["motion_local_labels"], original_labels)
    assert mask.tolist() == [True, False, False]
    assert stats.clips == 1
    assert stats.anchors == 3
    assert stats.donor_anchors == 3
    assert stats.previous_copy_anchors == 0
    assert torch.equal(corrupted[1:], original_ids[1:])
    final_start = 2 + 3 * BODY_SLOT_COUNT
    assert torch.equal(
        corrupted[0, final_start : final_start + BODY_SLOT_COUNT],
        original_ids[0, final_start : final_start + BODY_SLOT_COUNT],
    )
    assert not torch.equal(
        corrupted[0, 2 : 2 + 3 * BODY_SLOT_COUNT],
        original_ids[0, 2 : 2 + 3 * BODY_SLOT_COUNT],
    )


def test_previous_copy_can_propagate_a_corrupted_history_state() -> None:
    batch, table = synthetic_batch()
    corrupted, _, stats = apply_history_corruption(
        batch,
        ["a", "b", "c"],
        [0],
        motion_token_ids=table,
        anchor_probability=1.0,
        donor_probability=0.0,
        seed=7,
        epoch=2,
        batch_index=1,
    )
    # The first anchor must use a donor because the observed seed is kept clean.
    assert stats.donor_anchors == 1
    assert stats.previous_copy_anchors == 2
    first = corrupted[0, 2 : 2 + BODY_SLOT_COUNT]
    second = corrupted[
        0, 2 + BODY_SLOT_COUNT : 2 + 2 * BODY_SLOT_COUNT
    ]
    third = corrupted[
        0, 2 + 2 * BODY_SLOT_COUNT : 2 + 3 * BODY_SLOT_COUNT
    ]
    assert torch.equal(first, second)
    assert torch.equal(second, third)

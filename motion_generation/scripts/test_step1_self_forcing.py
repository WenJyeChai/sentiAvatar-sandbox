from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen2Config


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    IGNORE_INDEX,
    MimiQwenPlanner,
    MimiQwenPlannerConfig,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    body_global_id,
)
from utils.step1_self_forcing import (  # noqa: E402
    apply_generated_history,
    deterministic_generated_indices,
    generate_history_batch,
    generated_history_curriculum_state,
    generated_history_probability,
    generated_history_rollout_target_slots,
)
from utils.step1_visited_state import (  # noqa: E402
    apply_visited_state_history,
    deterministic_visited_indices,
    local_rollout_target_slots,
)


def tiny_q0q3_planner() -> MimiQwenPlanner:
    vocabulary = BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE + 8
    language_model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=vocabulary,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=True,
            use_cache=True,
        )
    )
    table = [
        [body_global_id(slot, local_id) for local_id in range(BODY_CODEBOOK_SIZE)]
        for slot in range(BODY_SLOT_COUNT)
    ]
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=language_model.config.to_dict(),
            audio_placeholder_id=BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE,
            motion_token_ids=table,
            mimi_codebooks_used=[0, 1, 2, 3],
        ),
        language_model=language_model,
    )
    planner.eval()
    return planner


def tiny_ordinary_q0q3_planner(
    planner_attention_mode: str = "causal",
) -> MimiQwenPlanner:
    audio_cardinality = 1024
    audio_start = BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE
    audio_token_ids = [
        [
            audio_start + stream * audio_cardinality + local_id
            for local_id in range(audio_cardinality)
        ]
        for stream in range(4)
    ]
    vocabulary = audio_start + 4 * audio_cardinality + 8
    language_model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=vocabulary,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=True,
            use_cache=True,
        )
    )
    motion_table = [
        [
            body_global_id(slot, local_id)
            for local_id in range(BODY_CODEBOOK_SIZE)
        ]
        for slot in range(BODY_SLOT_COUNT)
    ]
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=language_model.config.to_dict(),
            audio_placeholder_id=vocabulary - 1,
            motion_token_ids=motion_table,
            audio_codec="moss_audio_tokenizer_nano",
            audio_cardinality=audio_cardinality,
            audio_codebooks_stored=16,
            audio_codebooks_used=[0, 1, 2, 3],
            audio_input_representation="ordinary_tokens",
            audio_token_ids=audio_token_ids,
            planner_attention_mode=planner_attention_mode,
        ),
        language_model=language_model,
    )
    planner.eval()
    return planner


def synthetic_batch(planner: MimiQwenPlanner) -> dict[str, torch.Tensor]:
    length = 24
    input_ids = torch.ones((2, length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 20:] = 0
    input_ids[0, 20:] = 0
    audio_codes = torch.full((2, length, 4), -1, dtype=torch.long)
    input_ids[:, 1] = planner.config.audio_placeholder_id
    audio_codes[0, 1] = torch.tensor([10, 20, 30, 40])
    audio_codes[1, 1] = torch.tensor([11, 21, 31, 41])
    target_slots = torch.full_like(input_ids, -1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for row, start in ((0, 4), (1, 6)):
        target_slots[row, start : start + BODY_SLOT_COUNT] = torch.arange(BODY_SLOT_COUNT)
        local_ids = torch.tensor(
            [(row * 37 + slot * 17 + 5) % BODY_CODEBOOK_SIZE for slot in range(BODY_SLOT_COUNT)]
        )
        labels[row, start : start + BODY_SLOT_COUNT] = local_ids
        for slot, local_id in enumerate(local_ids.tolist()):
            input_ids[row, start + slot] = planner.motion_token_ids[slot, local_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_codes": audio_codes,
        "target_slots": target_slots,
        "motion_local_labels": labels,
    }


def synthetic_visited_batch(
    planner: MimiQwenPlanner,
) -> dict[str, torch.Tensor]:
    length = 60
    input_ids = torch.ones((2, length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    audio_codes = torch.full((2, length, 4), -1, dtype=torch.long)
    target_slots = torch.full_like(input_ids, -1)
    target_anchor_ids = torch.full_like(input_ids, -1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    starts = (4, 24, 44)
    for row in range(2):
        for group, start in enumerate(starts):
            target_slots[row, start : start + BODY_SLOT_COUNT] = torch.arange(
                BODY_SLOT_COUNT
            )
            target_anchor_ids[row, start : start + BODY_SLOT_COUNT] = group
            local_ids = torch.tensor(
                [
                    (row * 71 + group * 43 + slot * 13 + 5)
                    % BODY_CODEBOOK_SIZE
                    for slot in range(BODY_SLOT_COUNT)
                ]
            )
            labels[row, start : start + BODY_SLOT_COUNT] = local_ids
            for slot, local_id in enumerate(local_ids.tolist()):
                input_ids[row, start + slot] = planner.motion_token_ids[
                    slot, local_id
                ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_codes": audio_codes,
        "target_slots": target_slots,
        "target_anchor_ids": target_anchor_ids,
        "motion_local_labels": labels,
        "selected_anchor_groups": torch.tensor([1, 2]),
    }


def test_curriculum_probability_and_exact_selection() -> None:
    assert generated_history_probability(
        10.0, activation_epoch=None, ramp_epochs=10, max_probability=0.5
    ) == 0.0
    assert generated_history_probability(
        5.0, activation_epoch=5, ramp_epochs=10, max_probability=0.5
    ) == 0.0
    assert abs(
        generated_history_probability(
            10.0, activation_epoch=5, ramp_epochs=10, max_probability=0.5
        )
        - 0.25
    ) < 1e-8
    assert generated_history_probability(
        15.0, activation_epoch=5, ramp_epochs=10, max_probability=0.5
    ) == 0.5

    names = [f"clip-{index}" for index in range(32)]
    first = deterministic_generated_indices(names, 0.5, seed=42, epoch=9, batch_index=3)
    second = deterministic_generated_indices(names, 0.5, seed=42, epoch=9, batch_index=3)
    assert first == second
    assert len(first) == 16


def test_explicit_suffix_to_full_curriculum() -> None:
    phases = [
        {
            "start_epoch": 1,
            "end_epoch": 3,
            "probability": 0.2,
            "min_previous_anchors": 1,
            "max_previous_anchors": 2,
            "full_prefix_probability": 0.0,
        },
        {
            "start_epoch": 4,
            "end_epoch": 7,
            "probability": 0.4,
            "min_previous_anchors": 1,
            "max_previous_anchors": 4,
            "full_prefix_probability": 0.0,
        },
        {
            "start_epoch": 8,
            "end_epoch": 12,
            "probability": 0.6,
            "min_previous_anchors": 1,
            "max_previous_anchors": 4,
            "full_prefix_probability": 0.5,
        },
        {
            "start_epoch": 13,
            "end_epoch": 20,
            "probability": 0.6,
            "min_previous_anchors": 1,
            "max_previous_anchors": 4,
            "full_prefix_probability": 1.0,
        },
    ]
    first = generated_history_curriculum_state(1, phases)
    assert first.probability == 0.2
    assert first.max_previous_anchors == 2
    final = generated_history_curriculum_state(20, phases)
    assert final.probability == 0.6
    assert final.full_prefix_probability == 1.0

    slots = torch.arange(BODY_SLOT_COUNT).repeat(2).unsqueeze(0)
    local, depths, full = generated_history_rollout_target_slots(
        slots,
        ["clip"],
        min_previous_anchors=1,
        max_previous_anchors=1,
        full_prefix_probability=0.0,
        seed=42,
        epoch=0,
        batch_index=0,
    )
    assert int(local.ge(0).sum()) == 2 * BODY_SLOT_COUNT
    assert int(depths[0]) == 1
    assert not bool(full[0])

    full_slots, depths, full = generated_history_rollout_target_slots(
        slots,
        ["clip"],
        min_previous_anchors=1,
        max_previous_anchors=1,
        full_prefix_probability=1.0,
        seed=42,
        epoch=12,
        batch_index=0,
    )
    assert torch.equal(full_slots, slots)
    assert int(depths[0]) == 1
    assert bool(full[0])


def test_ordinary_audio_kv_slices_are_validated_as_complete_sequences() -> None:
    planner = tiny_ordinary_q0q3_planner()
    length = 28
    input_ids = torch.ones((2, length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    audio_codes = torch.full((2, length, 4), -1, dtype=torch.long)
    target_slots = torch.full_like(input_ids, -1)

    for row, audio_start, codes in (
        (0, 1, (10, 20, 30, 40)),
        # Row 0's first target at position 6 cuts through this row's
        # q0--q3 block at positions 5--8.
        (1, 5, (11, 21, 31, 41)),
    ):
        audio_codes[row, audio_start] = torch.tensor(codes)
        for stream, code in enumerate(codes):
            input_ids[row, audio_start + stream] = planner.audio_token_ids[
                stream, code
            ]

    for row, target_start in ((0, 6), (1, 10)):
        target_slots[
            row, target_start : target_start + BODY_SLOT_COUNT
        ] = torch.arange(BODY_SLOT_COUNT)
        for slot in range(BODY_SLOT_COUNT):
            input_ids[row, target_start + slot] = planner.motion_token_ids[
                slot, (row * 31 + slot * 7) % BODY_CODEBOOK_SIZE
            ]

    result = generate_history_batch(
        planner,
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_codes=audio_codes,
        target_slots=target_slots,
        use_bf16=False,
    )
    assert result.generated_anchors == 2

    corrupted = input_ids.clone()
    corrupted[1, 6] = planner.audio_token_ids[1, 22]
    try:
        generate_history_batch(
            planner,
            input_ids=corrupted,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            target_slots=target_slots,
            use_bf16=False,
        )
    except ValueError as error:
        assert "Ordinary audio IDs do not match" in str(error)
    else:
        raise AssertionError("Corrupted ordinary-audio block was not rejected")


def test_prefix_lm_ordinary_audio_generated_suffix_runs_one_row() -> None:
    planner = tiny_ordinary_q0q3_planner("prefix_lm")
    length = 28
    input_ids = torch.ones((1, length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    audio_codes = torch.full((1, length, 4), -1, dtype=torch.long)
    audio_codes[0, 1] = torch.tensor([10, 20, 30, 40])
    for stream, code in enumerate((10, 20, 30, 40)):
        input_ids[0, 1 + stream] = planner.audio_token_ids[stream, code]
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 6 : 6 + BODY_SLOT_COUNT] = torch.arange(
        BODY_SLOT_COUNT
    )
    for slot in range(BODY_SLOT_COUNT):
        input_ids[0, 6 + slot] = planner.motion_token_ids[
            slot, (slot * 11 + 3) % BODY_CODEBOOK_SIZE
        ]
    prefix_mask = torch.zeros_like(input_ids)
    prefix_mask[:, :6] = 1
    result = generate_history_batch(
        planner,
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_codes=audio_codes,
        target_slots=target_slots,
        bidirectional_prefix_mask=prefix_mask,
        use_bf16=False,
    )
    assert result.generated_anchors == 1
    assert int(result.predicted_local_ids.ge(0).sum()) == BODY_SLOT_COUNT


def test_batched_cached_rollout_replaces_all_and_only_targets() -> None:
    planner = tiny_q0q3_planner()
    batch = synthetic_batch(planner)
    original = batch["input_ids"].clone()
    result = generate_history_batch(
        planner,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        audio_codes=batch["audio_codes"],
        target_slots=batch["target_slots"],
        use_bf16=False,
    )
    target_mask = batch["target_slots"].ge(0)
    assert torch.equal(batch["input_ids"], original)
    assert torch.equal(result.predicted_local_ids.ge(0), target_mask)
    assert torch.equal(result.input_ids[~target_mask], original[~target_mask])
    for row, position in target_mask.nonzero(as_tuple=False).tolist():
        slot = int(batch["target_slots"][row, position])
        local_id = int(result.predicted_local_ids[row, position])
        assert int(result.input_ids[row, position]) == int(planner.motion_token_ids[slot, local_id])
    assert result.generated_tokens == 2 * BODY_SLOT_COUNT
    assert result.generated_anchors == 2
    assert all(parameter.grad is None for parameter in planner.parameters())

    # The first row's cached decisions must match full-prefix recomputation
    # while using all four synchronous Mimi streams.
    prefix_ids: list[int] = []
    row_slots = batch["target_slots"][0]
    last_target = int(row_slots.ge(0).nonzero(as_tuple=False)[-1].item())
    for position in range(last_target + 1):
        slot = int(row_slots[position].item())
        if slot >= 0:
            ids = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0)
            logits = planner.next_slot_logits(
                ids,
                torch.ones_like(ids),
                batch["audio_codes"][0, :position].unsqueeze(0),
                slot=slot,
            )
            local_id = int(logits.argmax(dim=-1).item())
            assert local_id == int(result.predicted_local_ids[0, position])
            prefix_ids.append(int(planner.motion_token_ids[slot, local_id]))
        else:
            prefix_ids.append(int(batch["input_ids"][0, position]))


def test_apply_generated_history_leaves_unselected_rows_teacher_forced() -> None:
    planner = tiny_q0q3_planner()
    batch = synthetic_batch(planner)
    generated, stats = apply_generated_history(
        planner,
        batch,
        [1],
        microbatch_size=1,
        use_bf16=False,
    )
    assert torch.equal(generated[0], batch["input_ids"][0])
    target_mask = batch["target_slots"][1].ge(0)
    assert stats.clips == 1
    assert stats.anchors == 1
    assert stats.tokens == BODY_SLOT_COUNT
    assert not torch.equal(generated[1, target_mask], batch["input_ids"][1, target_mask]) or stats.correct > 0


def test_generated_history_can_feed_gradient_enabled_training_forward() -> None:
    planner = tiny_q0q3_planner()
    batch = synthetic_batch(planner)
    generated, _ = apply_generated_history(
        planner,
        batch,
        [0, 1],
        microbatch_size=2,
        use_bf16=False,
    )

    # Regression guard: inference tensors cannot be saved by embedding
    # backward, even though input IDs themselves never require gradients.
    assert not torch.is_inference(generated)
    planner.train()
    output = planner(
        input_ids=generated,
        attention_mask=batch["attention_mask"],
        audio_codes=batch["audio_codes"],
        target_slots=batch["target_slots"],
        motion_local_labels=batch["motion_local_labels"],
    )
    assert output.loss is not None
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in planner.parameters())


def test_local_visited_state_rolls_only_preceding_anchor_groups() -> None:
    planner = tiny_q0q3_planner()
    batch = synthetic_visited_batch(planner)
    rollout_slots, depths = local_rollout_target_slots(
        target_slots=batch["target_slots"],
        target_anchor_ids=batch["target_anchor_ids"],
        selected_anchor_groups=batch["selected_anchor_groups"],
        names=["clip-0", "clip-1"],
        rollout_depths=[1, 2],
        seed=42,
        epoch=0,
        batch_index=0,
    )
    # Group 1 can only roll one predecessor; group 2 may roll one or two.
    assert depths[0] == 1
    assert depths[1] in {1, 2}
    assert int(rollout_slots[0].ge(0).sum()) == BODY_SLOT_COUNT
    assert int(rollout_slots[1].ge(0).sum()) == depths[1] * BODY_SLOT_COUNT
    assert not bool(
        (
            rollout_slots.ge(0)
            & batch["target_anchor_ids"].ge(
                batch["selected_anchor_groups"].view(-1, 1)
            )
        ).any()
    )


def test_visited_state_keeps_gt_replay_rows_and_does_not_roll_the_right_anchor() -> None:
    planner = tiny_q0q3_planner()
    batch = synthetic_visited_batch(planner)
    original = batch["input_ids"].clone()
    generated, stats = apply_visited_state_history(
        planner,
        batch,
        ["clip-0", "clip-1"],
        [1],
        rollout_depths=[2],
        seed=42,
        epoch=0,
        batch_index=0,
        microbatch_size=1,
        use_bf16=False,
    )

    assert torch.equal(generated[0], original[0])
    selected_groups = batch["target_anchor_ids"][1]
    rolled = torch.isin(selected_groups, torch.tensor([0, 1]))
    right = selected_groups.eq(2)
    assert torch.equal(generated[1, ~rolled], original[1, ~rolled])
    assert torch.equal(generated[1, right], original[1, right])
    assert stats.clips == 1
    assert stats.anchors == 2
    assert stats.tokens == 2 * BODY_SLOT_COUNT
    assert stats.depth2_clips == 1

    selected = deterministic_visited_indices(
        [f"clip-{index}" for index in range(32)],
        0.5,
        seed=42,
        epoch=0,
        batch_index=0,
    )
    assert len(selected) == 16

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
    generated_history_probability,
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

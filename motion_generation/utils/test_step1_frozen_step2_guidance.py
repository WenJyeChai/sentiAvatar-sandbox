from __future__ import annotations

import sys
from pathlib import Path

import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer
from utils.step1_frozen_step2_guidance import (
    FrozenStep2AnchorGuidance,
    planner_history_left_boundaries,
)


def tiny_step2() -> AudioMotionTransformer:
    config = AudioMotionConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        vocab_size=16 * 512 + 1,
        codebook_size=512,
        audio_feat_dim=8,
        num_tokens_per_frame=16,
        num_frames=8,
        dropout=0.0,
        cond_drop_prob=0.0,
        constrain_token_logits=True,
        num_parts=4,
        num_quantizers_per_part=4,
        max_gap_frames=15,
        audio_fusion_mode="legacy_additive",
    )
    return AudioMotionTransformer(config)


def guidance_batch(batch: int = 2, gap: int = 3) -> dict[str, torch.Tensor]:
    frames = gap + 2
    generator = torch.Generator().manual_seed(7)
    return {
        "motion_tokens": torch.randint(
            0,
            512,
            (batch, frames, 16),
            generator=generator,
        ),
        "audio_features": torch.randn(
            batch,
            frames,
            8,
            generator=generator,
        ),
        "frame_mask": torch.ones(batch, frames, dtype=torch.bool),
        "gap_lengths": torch.full((batch,), gap, dtype=torch.long),
    }


def test_hard_st_step2_guidance_reaches_anchor_logits_only() -> None:
    torch.manual_seed(3)
    step2 = tiny_step2()
    guidance = FrozenStep2AnchorGuidance(
        step2,
        stage_weights=[0.35, 0.25, 0.20, 0.20],
        temperature=1.0,
    )
    selected_logits = torch.randn(2, 16, 512, requires_grad=True)
    output = guidance(selected_logits, guidance_batch())

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert tuple(output.stage_ce.shape) == (4,)
    assert tuple(output.hard_local_ids.shape) == (2, 16)
    assert torch.equal(output.hard_local_ids, selected_logits.argmax(dim=-1))
    assert int(output.count) == 2 * 3 * 16
    assert int(output.q0_count) == 2 * 3 * 4

    output.loss.backward()
    assert selected_logits.grad is not None
    assert torch.isfinite(selected_logits.grad).all()
    assert float(selected_logits.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in step2.parameters())
    assert all(not parameter.requires_grad for parameter in step2.parameters())


def test_st_boundary_forward_value_is_exact_hard_embedding() -> None:
    torch.manual_seed(11)
    guidance = FrozenStep2AnchorGuidance(
        tiny_step2(),
        stage_weights=[1, 1, 1, 1],
        temperature=0.7,
    )
    logits = torch.randn(1, 16, 512, requires_grad=True)
    hard_ids, embeddings = guidance._straight_through_boundary(logits)
    weight = guidance.model.embed_tokens.weight[: 16 * 512].reshape(16, 512, -1)
    expected = torch.stack(
        [weight[slot, hard_ids[0, slot]] for slot in range(16)],
        dim=0,
    ).unsqueeze(0)
    assert torch.allclose(embeddings, expected)

    embeddings.float().square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


def test_planner_history_extracts_the_previous_anchor_or_seed_fallback() -> None:
    motion_token_ids = torch.arange(16 * 512).reshape(16, 512)
    target_slots = torch.arange(16).repeat(2).unsqueeze(0).repeat(3, 1)
    target_anchor_ids = torch.cat(
        [torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)]
    ).unsqueeze(0).repeat(3, 1)
    input_ids = torch.empty(3, 32, dtype=torch.long)
    local_by_row_group = torch.empty(3, 2, 16, dtype=torch.long)
    for row in range(3):
        for group in range(2):
            for slot in range(16):
                local = (row * 101 + group * 37 + slot * 3) % 512
                local_by_row_group[row, group, slot] = local
                input_ids[row, group * 16 + slot] = motion_token_ids[slot, local]
    fallback = torch.full((3, 16), 499, dtype=torch.long)

    result = planner_history_left_boundaries(
        input_ids=input_ids,
        target_slots=target_slots,
        target_anchor_ids=target_anchor_ids,
        selected_anchor_groups=torch.tensor([0, 1, 2]),
        motion_token_ids=motion_token_ids,
        fallback_left_local_ids=fallback,
    )

    assert torch.equal(result[0], fallback[0])
    assert torch.equal(result[1], local_by_row_group[1, 0])
    assert torch.equal(result[2], local_by_row_group[2, 1])


def test_planner_history_guidance_uses_detached_left_boundary() -> None:
    guidance = FrozenStep2AnchorGuidance(
        tiny_step2(),
        stage_weights=[0.35, 0.25, 0.20, 0.20],
        temperature=1.0,
        left_boundary_mode="planner_history",
    )
    logits = torch.randn(2, 16, 512, requires_grad=True)
    batch = guidance_batch()
    batch["left_boundary_local_ids"] = torch.stack(
        [
            torch.arange(16) + 100,
            torch.arange(16) + 200,
        ]
    )
    hard_ids, boundary_embeddings = guidance._straight_through_boundary(
        logits
    )
    prepared = guidance._prepare_step2_inputs(
        hard_local_ids=hard_ids,
        boundary_embeddings=boundary_embeddings,
        motion_tokens=batch["motion_tokens"],
        audio_features=batch["audio_features"],
        frame_mask=batch["frame_mask"],
        gap_lengths=batch["gap_lengths"],
        left_boundary_local_ids=batch["left_boundary_local_ids"],
    )
    offsets = torch.arange(16) * 512
    prepared_left = prepared["input_ids"].reshape(2, 5, 16)[:, 0] - offsets
    assert torch.equal(prepared_left, batch["left_boundary_local_ids"])

    output = guidance(logits, batch)

    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0
    assert batch["left_boundary_local_ids"].grad is None

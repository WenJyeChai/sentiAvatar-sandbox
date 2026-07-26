from __future__ import annotations

import sys
from pathlib import Path

import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer
from utils.step1_frozen_step2_guidance import FrozenStep2AnchorGuidance


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

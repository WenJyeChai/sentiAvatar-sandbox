from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.variable_c2f_evaluation import (
    EvalWindowRecord,
    InfillModelSpec,
    VariableGapMaskExample,
    apply_audio_input_mode,
    apply_model_inference_ablation,
)


def make_record(*, sequence_idx: int = 3, left_idx: int = 5, gap: int = 4):
    frame_count = gap + 2
    return EvalWindowRecord(
        sequence_idx=sequence_idx,
        name="example",
        left_idx=left_idx,
        gap_frames=gap,
        example=VariableGapMaskExample(
            name="example",
            left_idx=left_idx,
            right_idx=left_idx + gap + 1,
            gap_frames=gap,
            motion_tokens=[[0] for _ in range(frame_count)],
            audio_features=torch.zeros(frame_count, 2),
        ),
    )


def test_correct_audio_input_is_unchanged():
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    result = apply_audio_input_mode(
        audio, [make_record()], mode="correct", seed=42
    )
    assert result is audio


def test_zero_audio_input_preserves_shape_and_dtype():
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    result = apply_audio_input_mode(audio, [make_record()], mode="zero", seed=42)
    assert result.shape == audio.shape
    assert result.dtype == audio.dtype
    assert torch.count_nonzero(result) == 0


def test_temporal_shuffle_is_deterministic_and_preserves_frames():
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    record = make_record()
    first = apply_audio_input_mode(
        audio, [record], mode="temporal_shuffle", seed=42
    )
    second = apply_audio_input_mode(
        audio, [record], mode="temporal_shuffle", seed=42
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, audio)
    assert torch.equal(first[0, :, 0].sort().values, audio[0, :, 0].sort().values)


def test_unknown_audio_input_mode_is_rejected():
    audio = torch.zeros(1, 6, 2)
    try:
        apply_audio_input_mode(audio, [make_record()], mode="invalid", seed=42)
    except ValueError as error:
        assert "audio_input_mode" in str(error)
    else:
        raise AssertionError("Expected invalid audio mode to raise ValueError")


def test_posterior_disabled_ablation_keeps_additive_audio_and_is_idempotent():
    model = SimpleNamespace(
        config=SimpleNamespace(audio_conditioning_mode="additive_residual_posterior"),
        audio_residual_posterior=object(),
    )
    spec = InfillModelSpec(
        name="posterior_disabled",
        checkpoint=Path("unused"),
        decoder="c2f",
        allowed_gaps=(3,),
        audio_posterior_enabled=False,
    )
    apply_model_inference_ablation(model, spec)
    apply_model_inference_ablation(model, spec)
    assert model.config.audio_conditioning_mode == "additive"

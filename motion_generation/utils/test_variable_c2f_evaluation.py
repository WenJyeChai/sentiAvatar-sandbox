from __future__ import annotations

import sys
from pathlib import Path

import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.variable_c2f_evaluation import (
    EvalWindowRecord,
    VariableGapMaskExample,
    apply_audio_input_mode,
)


def make_record(sequence_idx: int, offset: float, gap: int = 4) -> EvalWindowRecord:
    frames = gap + 2
    audio = torch.arange(frames * 2, dtype=torch.float32).reshape(frames, 2) + offset
    example = VariableGapMaskExample(
        name=f"clip-{sequence_idx}",
        left_idx=sequence_idx,
        right_idx=sequence_idx + gap + 1,
        gap_frames=gap,
        motion_tokens=[[0] for _ in range(frames)],
        audio_features=audio,
    )
    return EvalWindowRecord(
        sequence_idx=sequence_idx,
        name=example.name,
        left_idx=example.left_idx,
        gap_frames=gap,
        example=example,
    )


def apply(mode: str, records):
    audio = torch.stack([record.example.audio_features for record in records])
    return audio, apply_audio_input_mode(
        audio,
        records,
        records,
        start_index=0,
        mode=mode,
        seed=42,
    )


def test_correct_and_zero_audio_controls():
    records = [make_record(0, 0.0), make_record(1, 100.0)]
    audio, correct = apply("correct", records)
    _, zero = apply("zero", records)
    assert correct is audio
    assert torch.count_nonzero(zero) == 0


def test_temporal_shuffle_is_deterministic_and_preserves_frame_values():
    records = [make_record(0, 0.0), make_record(1, 100.0)]
    audio, first = apply("temporal_shuffle", records)
    _, second = apply("temporal_shuffle", records)
    assert torch.equal(first, second)
    assert not torch.equal(first, audio)
    assert torch.equal(first[0, :, 0].sort().values, audio[0, :, 0].sort().values)


def test_shift_mean_and_cross_clip_have_distinct_semantics():
    records = [make_record(0, 0.0), make_record(1, 100.0)]
    audio, shifted = apply("temporal_shift", records)
    _, mean = apply("temporal_mean", records)
    _, crossed = apply("cross_clip", records)
    assert torch.equal(shifted[0], audio[0].roll(3, dims=0))
    assert torch.allclose(mean[0], audio[0].mean(dim=0, keepdim=True).expand_as(audio[0]))
    assert torch.equal(crossed[0], audio[1])
    assert torch.equal(crossed[1], audio[0])


def test_unknown_audio_control_is_rejected():
    records = [make_record(0, 0.0)]
    try:
        apply("unknown", records)
    except ValueError as error:
        assert "audio_input_mode" in str(error)
    else:
        raise AssertionError("Expected an invalid audio mode to raise ValueError")

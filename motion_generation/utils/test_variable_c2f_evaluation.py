from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.variable_c2f_evaluation import (
    EvalWindowRecord,
    InfillModelSpec,
    VariableGapMaskExample,
    apply_audio_input_mode,
    decoded_face_group_metrics,
    generated_gap_dynamics_metrics,
    infer_c2f_window_records_with_metrics,
    infer_window_records,
    paired_bootstrap_window_differences,
    seam_discontinuity_metrics,
)
from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer


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


def test_generated_gap_dynamics_scale_with_motion_amplitude():
    time = np.arange(48, dtype=np.float64)
    body = np.stack([time, time**2, np.sin(time), np.cos(time)], axis=1)
    base = generated_gap_dynamics_metrics(
        body, gap=3, codec_unit_length=2, feature_start=0
    )
    doubled = generated_gap_dynamics_metrics(
        2.0 * body, gap=3, codec_unit_length=2, feature_start=0
    )
    for metric in (
        "gap_velocity_l2_rms",
        "gap_acceleration_l2_rms",
        "gap_jerk_l2_rms",
    ):
        assert np.isclose(doubled[metric], 2.0 * base[metric])


def test_face_metrics_report_lip_and_non_lip_groups():
    gt = np.zeros((5, 51), dtype=np.float32)
    pred = gt.copy()
    pred[:, 24] = 1.0
    pred[:, 0] = 0.5
    metrics = decoded_face_group_metrics(gt, pred)
    assert metrics["face_rmse"] > 0
    assert metrics["face_lip_rmse"] > 0
    assert metrics["face_non_lip_rmse"] > 0
    assert metrics["face_velocity_rmse"] == 0


def test_seam_metrics_report_an_interior_normalized_excess():
    time = np.arange(60, dtype=np.float64)
    body = np.stack([np.sin(time / 4), np.cos(time / 5)], axis=1)
    body[20:] += 2.0
    metrics = seam_discontinuity_metrics(body, boundary_stride=20)
    assert metrics["seam_count"] == 2
    assert np.isfinite(metrics["seam_accel_excess_ratio"])
    assert np.isfinite(metrics["seam_jerk_excess_ratio"])
    assert metrics["interior_accel_l2_mean"] > 0


def test_paired_bootstrap_uses_clip_level_differences():
    frame = pd.DataFrame(
        [
            {"model": "base", "sequence_idx": 0, "left_idx": 1, "gap": 3, "body_rmse": 1.0},
            {"model": "base", "sequence_idx": 1, "left_idx": 2, "gap": 3, "body_rmse": 2.0},
            {"model": "candidate", "sequence_idx": 0, "left_idx": 1, "gap": 3, "body_rmse": 2.0},
            {"model": "candidate", "sequence_idx": 1, "left_idx": 2, "gap": 3, "body_rmse": 4.0},
        ]
    )
    result = paired_bootstrap_window_differences(
        frame,
        reference_model="base",
        candidate_models=["candidate"],
        metrics=["body_rmse"],
        iterations=200,
        seed=7,
    )
    assert len(result) == 1
    assert result.iloc[0]["mean_difference"] == 1.5
    assert result.iloc[0]["ci_low"] <= 1.5 <= result.iloc[0]["ci_high"]


def test_c2f_metrics_preserve_hard_rollout_and_count_each_stage(tmp_path):
    torch.manual_seed(7)
    config = AudioMotionConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        max_position_embeddings=128,
        vocab_size=512 * 16 + 1,
        codebook_size=512,
        audio_feat_dim=2,
        num_tokens_per_frame=16,
        num_parts=4,
        num_quantizers_per_part=4,
        dropout=0.0,
        cond_drop_prob=0.0,
    )
    model = AudioMotionTransformer(config).eval()
    records = []
    for sequence_idx, gap in enumerate((2, 3)):
        frames = gap + 2
        motion_tokens = [
            [int((frame * 17 + slot * 11) % 512) for slot in range(16)]
            for frame in range(frames)
        ]
        example = VariableGapMaskExample(
            name=f"metric-{sequence_idx}",
            left_idx=sequence_idx * 10,
            right_idx=sequence_idx * 10 + gap + 1,
            gap_frames=gap,
            motion_tokens=motion_tokens,
            audio_features=torch.arange(
                frames * 2, dtype=torch.float32
            ).reshape(frames, 2),
        )
        records.append(
            EvalWindowRecord(
                sequence_idx=sequence_idx,
                name=example.name,
                left_idx=example.left_idx,
                gap_frames=gap,
                example=example,
            )
        )
    spec = InfillModelSpec(
        name="tiny",
        checkpoint=tmp_path,
        decoder="c2f",
        allowed_gaps=(1, 2, 3),
    )

    expected = infer_window_records(
        model,
        spec,
        records,
        batch_size=2,
        device=torch.device("cpu"),
    )
    actual, metrics = infer_c2f_window_records_with_metrics(
        model,
        spec,
        records,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert len(actual) == len(expected) == len(metrics) == 2
    for prediction, reference, record, metric in zip(
        actual, expected, records, metrics
    ):
        assert np.array_equal(prediction, reference)
        gt = np.asarray(record.example.motion_tokens[1:-1], dtype=np.int64)
        assert metric["token_count"] == record.gap_frames * 16
        assert metric["correct"] == int(np.equal(prediction, gt).sum())
        assert np.isclose(
            metric["accuracy"],
            np.equal(prediction, gt).mean(),
        )
        assert np.isfinite(metric["cross_entropy"])
        for stage in range(4):
            stage_prediction = prediction[:, stage::4]
            stage_gt = gt[:, stage::4]
            assert metric[f"q{stage}_token_count"] == record.gap_frames * 4
            assert metric[f"q{stage}_correct"] == int(
                np.equal(stage_prediction, stage_gt).sum()
            )

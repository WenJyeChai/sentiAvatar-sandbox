from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.causal_codec_evaluation import (  # noqa: E402
    geodesic_degrees,
    reconstruction_metrics,
    update_usage,
    usage_rows,
)


def identity_sixd(frames: int, joints: int) -> torch.Tensor:
    value = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return value.repeat(frames, joints)


def test_identical_rotations_have_zero_geodesic_error() -> None:
    rotations = identity_sixd(frames=4, joints=3)
    angles = geodesic_degrees("upper", rotations, rotations)
    assert torch.allclose(angles, torch.zeros_like(angles))


def test_lower_metrics_include_root_trajectory() -> None:
    rotations = identity_sixd(frames=4, joints=2)
    target = torch.cat((torch.zeros(4, 3), rotations), dim=-1)
    prediction = target.clone()
    prediction[:, 0] = 0.5
    metrics = reconstruction_metrics(
        part="lower",
        target_raw=target,
        prediction_raw=prediction,
        target_normalized=target,
        prediction_normalized=prediction,
    )
    assert metrics["root_velocity_rmse"] > 0
    assert metrics["root_trajectory_final_drift"] == 2.0
    assert metrics["geodesic_deg_mean"] == 0.0


def test_usage_statistics_report_dead_codes_and_entropy() -> None:
    histogram = np.zeros((4, 8), dtype=np.int64)
    codes = np.asarray([[0, 1, 2, 3], [0, 1, 2, 4], [1, 1, 2, 4]])
    update_usage(histogram, codes)
    usage = {part: histogram.copy() for part in ("upper", "lower", "feet", "hands")}
    rows = usage_rows(usage)
    upper_q0 = next(
        row for row in rows if row["part"] == "upper" and row["quantizer"] == 0
    )
    assert upper_q0["tokens"] == 3
    assert upper_q0["used_codes"] == 2
    assert upper_q0["utilization"] == 0.25
    assert upper_q0["effective_codes"] > 1.0

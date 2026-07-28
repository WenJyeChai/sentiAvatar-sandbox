from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from scripts.evaluate_step1_adaptive_motion import (  # noqa: E402
    annotate_gap_bins,
    endpoint_penalty_by_gap_bin,
    summarize_decoded_intervals_by_gap_bin,
    summarize_step2_c2f_by_gap_bin,
)
from scripts.evaluate_step1_stage1_supplied_gap import (  # noqa: E402
    summarize_by_gap_bin,
)
from utils.step1_gap_bins import supplied_gap_bin  # noqa: E402


def step2_row(
    condition: str,
    name: str,
    gap: int,
    *,
    cross_entropy: float,
    accuracy: float,
) -> dict[str, float | int | str]:
    token_count = gap * 16
    row: dict[str, float | int | str] = {
        "condition": condition,
        "name": name,
        "gap": gap,
        "token_count": token_count,
        "ce_sum": cross_entropy * token_count,
        "correct": round(accuracy * token_count),
    }
    for stage in range(4):
        stage_count = gap * 4
        row[f"q{stage}_token_count"] = stage_count
        row[f"q{stage}_ce_sum"] = cross_entropy * stage_count
        row[f"q{stage}_correct"] = round(accuracy * stage_count)
    return row


def test_gap_bin_boundaries_and_stage1_summary() -> None:
    assert supplied_gap_bin(0).label == "eos_tail_0_2"
    assert supplied_gap_bin(3).label == "small_3_6"
    assert supplied_gap_bin(6).label == "small_3_6"
    assert supplied_gap_bin(7).label == "medium_7_10"
    assert supplied_gap_bin(10).label == "medium_7_10"
    assert supplied_gap_bin(11).label == "large_11_15"
    assert supplied_gap_bin(15).label == "large_11_15"

    gaps = np.asarray([1, 3, 6, 8, 12, 15])
    targets = np.zeros((len(gaps), 16), dtype=np.int64)
    predictions = targets.copy()
    predictions[2, 0] = 1
    rows = summarize_by_gap_bin(
        label="checkpoint",
        gaps=gaps,
        targets=targets,
        predictions=predictions,
    )
    by_label = {row["gap_bin"]: row for row in rows}
    assert by_label["eos_tail_0_2"]["main_gap_bin"] is False
    assert by_label["small_3_6"]["anchors"] == 2
    assert by_label["medium_7_10"]["anchors"] == 1
    assert by_label["large_11_15"]["anchors"] == 2
    assert by_label["small_3_6"]["accuracy"] < 1.0


def test_step2_gap_bin_summary_and_generated_endpoint_penalty() -> None:
    gt = "supplied_gap3_15_gt_anchors"
    generated = "teacher__supplied_gap3_15_generated_history"
    frame = pd.DataFrame(
        [
            step2_row(gt, "a", 3, cross_entropy=3.0, accuracy=0.5),
            step2_row(gt, "b", 6, cross_entropy=4.0, accuracy=0.25),
            step2_row(gt, "c", 8, cross_entropy=5.0, accuracy=0.125),
            step2_row(gt, "d", 12, cross_entropy=6.0, accuracy=0.0625),
            step2_row(
                generated,
                "a",
                3,
                cross_entropy=5.0,
                accuracy=0.25,
            ),
            step2_row(
                generated,
                "b",
                6,
                cross_entropy=6.0,
                accuracy=0.125,
            ),
            step2_row(
                generated,
                "c",
                8,
                cross_entropy=8.0,
                accuracy=0.0625,
            ),
            step2_row(
                generated,
                "d",
                12,
                cross_entropy=10.0,
                accuracy=0.0,
            ),
        ]
    )
    annotated = annotate_gap_bins(frame)
    summary = summarize_step2_c2f_by_gap_bin(annotated)
    assert set(summary["gap_bin"]) == {
        "small_3_6",
        "medium_7_10",
        "large_11_15",
    }
    small_gt = summary[
        (summary["condition"] == gt)
        & (summary["gap_bin"] == "small_3_6")
    ].iloc[0]
    expected_small_ce = (3.0 * 3 + 4.0 * 6) / 9
    assert abs(small_gt["cross_entropy"] - expected_small_ce) < 1e-8

    penalty = endpoint_penalty_by_gap_bin(summary)
    large = penalty[penalty["gap_bin"] == "large_11_15"].iloc[0]
    assert large["condition"] == generated
    assert abs(large["delta_cross_entropy"] - 4.0) < 1e-8
    assert large["delta_accuracy"] < 0


def test_decoded_interval_summary_is_gap_grouped() -> None:
    rows = []
    for gap, rmse in ((3, 0.1), (6, 0.2), (8, 0.3), (12, 0.4)):
        specification = supplied_gap_bin(gap)
        row = {
            "condition": "generated",
            "name": f"clip-{gap}",
            "gap": gap,
            "gap_bin_order": specification.order,
            "gap_bin": specification.label,
            "main_gap_bin": specification.main_bin,
            "motion_frames": 2 * gap,
            "missing_token_count": 16 * gap,
            "missing_token_correct": 4 * gap,
            "missing_q0_count": 4 * gap,
            "missing_q0_correct": gap,
        }
        for prefix in ("codec_relative", "raw_gt"):
            row[f"{prefix}_mae"] = rmse / 2
            row[f"{prefix}_rmse"] = rmse
            row[f"{prefix}_velocity_rmse"] = rmse
            row[f"{prefix}_acceleration_rmse"] = rmse
            row[f"{prefix}_jerk_rmse"] = rmse
        rows.append(row)
    summary = summarize_decoded_intervals_by_gap_bin(
        pd.DataFrame(rows)
    )
    assert len(summary) == 3
    assert np.allclose(summary["missing_token_accuracy"], 0.25)
    assert np.allclose(summary["missing_q0_accuracy"], 0.25)

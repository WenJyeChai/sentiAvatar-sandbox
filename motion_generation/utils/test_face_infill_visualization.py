from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.face_infill_visualization import (
    FaceInfillVisualComparison,
    plot_face_infill_summary,
    save_face_infill_animation,
    select_representative_clip_row,
    select_representative_window_rows,
)


def test_representative_selection_uses_per_gap_median():
    frame = pd.DataFrame(
        [
            {"model": "stage2", "gap": 3, "sequence_idx": 0, "left_idx": 2, "name": "a", "face_rmse": 0.1},
            {"model": "stage2", "gap": 3, "sequence_idx": 1, "left_idx": 4, "name": "b", "face_rmse": 0.2},
            {"model": "stage2", "gap": 3, "sequence_idx": 2, "left_idx": 6, "name": "c", "face_rmse": 1.0},
            {"model": "stage2", "gap": 7, "sequence_idx": 3, "left_idx": 8, "name": "d", "face_rmse": 0.3},
        ]
    )
    selected = select_representative_window_rows(
        frame, model_name="stage2", gaps=[3, 7]
    )
    assert selected.loc[selected["gap"].eq(3), "name"].item() == "b"
    assert selected.loc[selected["gap"].eq(7), "name"].item() == "d"


def test_representative_clip_requires_every_requested_gap():
    rows = []
    for sequence_idx, name, values in (
        (0, "a", (0.1, 0.2, 0.3)),
        (1, "b", (0.4, 0.5, 0.6)),
        (2, "c", (1.0, 1.1, 1.2)),
    ):
        rows.extend(
            {
                "model": "stage2",
                "gap": gap,
                "sequence_idx": sequence_idx,
                "name": name,
                "face_rmse": value,
            }
            for gap, value in zip((3, 7, 15), values)
        )
    rows.pop()  # Clip c is incomplete and must not be selected.
    selected = select_representative_clip_row(
        pd.DataFrame(rows), model_name="stage2", gaps=[3, 7, 15]
    )
    assert selected.iloc[0]["name"] in {"a", "b"}
    assert selected.iloc[0]["observed_gaps"] == 3


def _synthetic_comparison() -> FaceInfillVisualComparison:
    frames = 8
    variants = ("Raw GT", "Codec GT", "Stage 1", "Stage 2")
    base = np.zeros((frames, 63, 3), dtype=np.float32)
    base[:, :, 0] = np.arange(63, dtype=np.float32)[None] * 0.01
    base[:, :, 1] = np.arange(63, dtype=np.float32)[None] * 0.02
    base[:, :, 2] = np.arange(frames, dtype=np.float32)[:, None] * 0.01
    positions = {
        label: base + idx * 0.005 for idx, label in enumerate(variants)
    }
    faces = {
        label: np.linspace(0, 1, frames * 51, dtype=np.float32).reshape(frames, 51)
        + idx * 0.01
        for idx, label in enumerate(variants)
    }
    bodies = {
        label: np.zeros((frames, 153), dtype=np.float32) for label in variants
    }
    metrics = {
        label: {"body_rmse": float(idx) / 100, "face_rmse": float(idx) / 50}
        for idx, label in enumerate(variants)
    }
    return FaceInfillVisualComparison(
        name="session/example",
        gap=3,
        left_idx=5,
        raw_frame_start=20,
        infill_start=2,
        infill_end=6,
        fps=20,
        positions=positions,
        faces=faces,
        body_features=bodies,
        metrics=metrics,
    )


def test_static_and_animated_visualizations_write_files(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    comparison = _synthetic_comparison()
    png_path = tmp_path / "comparison.png"
    gif_path = tmp_path / "comparison.gif"
    figure = plot_face_infill_summary(comparison, output_path=png_path)
    assert figure is not None
    assert png_path.exists() and png_path.stat().st_size > 0
    save_face_infill_animation(comparison, gif_path, frame_step=2)
    assert gif_path.exists() and gif_path.stat().st_size > 0

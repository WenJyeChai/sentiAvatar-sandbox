from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.cache_step2_interval_costs import (
    all_examples,
    load_existing_clip,
    save_clip,
)
from scripts.train_audio_mask_multipart_variable_c2f import VariableGapMaskExample


def _item(frames: int = 10) -> dict:
    return {
        "name": "clip",
        "motion_tokens": [[0] * 16 for _ in range(frames)],
        "audio_features": np.zeros((8, 4), dtype=np.float32),
        "audio_fps": 12.5,
        "motion_token_fps": 10.0,
    }


def test_repair_examples_only_reach_missing_suffix() -> None:
    examples = all_examples(_item(), num_frames=10, minimum_right_idx=8)
    assert examples
    assert min(example.right_idx for example in examples) == 8
    assert max(example.right_idx for example in examples) == 9
    assert all(example.right_idx >= 8 for example in examples)


def test_save_clip_extends_and_preserves_existing_costs(tmp_path: Path) -> None:
    old_ce = np.full((8, 16), np.inf, dtype=np.float32)
    old_latent = np.full_like(old_ce, np.inf)
    old_ce[:, 0] = 0
    old_latent[:, 0] = 0
    old_ce[0, 3] = 1.25
    old_latent[0, 3] = 0.5
    example = VariableGapMaskExample(
        name="clip",
        left_idx=6,
        right_idx=9,
        gap_frames=2,
        motion_tokens=[[0] * 16 for _ in range(4)],
        audio_features=torch.zeros(4, 4),
    )
    path = tmp_path / "clip.npz"
    save_clip(
        path,
        num_frames=10,
        examples=[example],
        ce_values=np.asarray([2.0], dtype=np.float32),
        latent_values=np.asarray([0.75], dtype=np.float32),
        existing_ce=old_ce,
        existing_latent=old_latent,
    )
    ce, latent = load_existing_clip(path)
    assert ce.shape == (10, 16)
    assert ce[0, 3] == 1.25
    assert latent[0, 3] == 0.5
    assert ce[6, 2] == 2.0
    assert latent[6, 2] == 0.75
    assert np.array_equal(ce[:, 0], np.zeros(10, dtype=np.float32))

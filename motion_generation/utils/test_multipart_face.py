from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.multipart_motion import (
    FACE_PART,
    PartNormalizer,
    compute_selected_part_normalizer,
)


def _write_face_fixture(root: Path) -> tuple[str, np.ndarray]:
    name = "session/example"
    value = np.linspace(0.0, 1.0, 10 * 51, dtype=np.float32).reshape(10, 51)
    face_path = root / "arkit_data" / "session" / "example.npy"
    face_path.parent.mkdir(parents=True)
    np.save(face_path, value)
    split_dir = root / "split"
    split_dir.mkdir()
    (split_dir / "train_file_list.txt").write_text(f"{name}\n", encoding="utf-8")
    return name, value


def test_face_normalizer_round_trip(tmp_path):
    _, value = _write_face_fixture(tmp_path)
    normalizer, metadata = compute_selected_part_normalizer(
        tmp_path, ["session/example"], [FACE_PART]
    )
    assert metadata["part_order"] == [FACE_PART]
    assert metadata["frame_counts"][FACE_PART] == 10

    path = tmp_path / "normalizer.npz"
    normalizer.save(path, metadata=metadata)
    loaded = PartNormalizer.load(path)
    assert set(loaded.mean) == {FACE_PART}
    reconstructed = loaded.normalize(FACE_PART, value) * loaded.std[FACE_PART]
    reconstructed += loaded.mean[FACE_PART]
    np.testing.assert_allclose(reconstructed, value, atol=1e-6)

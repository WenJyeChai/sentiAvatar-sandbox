from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.precompute_moss_nano_audio_tokens import (  # noqa: E402
    AudioItem,
    NANO_CARDINALITY,
    NANO_CODEBOOKS,
    NANO_FRAME_SIZE,
    canonical_name_path,
    resample_audio,
    validate_existing_token_file,
    write_token_file,
)


def test_nano_resampling_has_deterministic_physical_length():
    source = np.linspace(-1.0, 1.0, num=16_000, dtype=np.float32)
    output = resample_audio(source, 16_000)
    assert output.dtype == np.float32
    assert output.shape == (48_000,)
    assert np.isfinite(output).all()


def test_nano_uses_only_complete_causal_frames():
    item = AudioItem(
        name="partial",
        source_path=Path("partial.wav"),
        source_sample_rate=48_000,
        source_num_samples=223_200,
        audio_48k=np.zeros(223_200, dtype=np.float32),
    )
    assert item.target_frames == 58
    assert 223_200 // NANO_FRAME_SIZE == 58


def test_nano_token_file_stores_all_16_codebooks(tmp_path: Path):
    name = "session/clip"
    item = AudioItem(
        name=name,
        source_path=Path("source.wav"),
        source_sample_rate=16_000,
        source_num_samples=16_000,
        audio_48k=np.zeros(NANO_FRAME_SIZE * 3, dtype=np.float32),
    )
    codes = np.arange(NANO_CODEBOOKS * 3, dtype=np.uint16).reshape(
        NANO_CODEBOOKS, 3
    ) % NANO_CARDINALITY
    output = canonical_name_path(tmp_path, name, ".npz")
    write_token_file(output, item, codes, model_dir=Path("nano"))
    assert validate_existing_token_file(output, name)
    with np.load(output, allow_pickle=False) as payload:
        assert payload["codes"].shape == (16, 3)
        assert int(payload["sample_rate"]) == 48_000
        assert int(payload["frame_size"]) == 3_840
        assert str(payload["frame_count_rule"]) == "floor_complete_frames"
        assert int(payload["cardinality"]) == 1_024
        assert str(payload["codec"]) == "moss_audio_tokenizer_nano"

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from scripts.export_moss_nano_all16_features import (  # noqa: E402
    FEATURE_DIM,
    NanoCodeItem,
    decode_all16_batch,
    validate_feature_file,
    write_feature_file,
)
from scripts.precompute_moss_nano_audio_tokens import (  # noqa: E402
    NANO_CODEBOOKS,
)


class FakeAll16Quantizer(torch.nn.Module):
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        summed = codes.to(torch.float32).sum(dim=0)
        return summed.unsqueeze(1).expand(-1, FEATURE_DIM, -1).contiguous()


def make_item(name: str, frames: int, value: int) -> NanoCodeItem:
    codes = np.full((NANO_CODEBOOKS, frames), value, dtype=np.int64)
    return NanoCodeItem(name=name, path=Path(f"{name}.npz"), codes=codes)


def test_all16_decoder_uses_every_codebook_and_slices_padding():
    short = make_item("short", 2, 1)
    long = make_item("long", 4, 2)
    outputs = decode_all16_batch(
        FakeAll16Quantizer(),
        [short, long],
        torch.device("cpu"),
    )
    assert outputs[0].shape == (2, FEATURE_DIM)
    assert outputs[1].shape == (4, FEATURE_DIM)
    assert np.all(outputs[0] == 16.0)
    assert np.all(outputs[1] == 32.0)


def test_feature_writer_roundtrips_float16_atomically(tmp_path: Path):
    path = tmp_path / "session" / "clip.npy"
    feature = np.linspace(
        -1.0,
        1.0,
        num=3 * FEATURE_DIM,
        dtype=np.float32,
    ).reshape(3, FEATURE_DIM)
    write_feature_file(path, feature, feature_dtype="float16")
    assert validate_feature_file(path, 3)
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float16
    assert stored.shape == (3, FEATURE_DIM)

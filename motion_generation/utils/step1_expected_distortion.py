"""Codec-codebook geometry for the Step 1 expected-distortion objective."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping

import torch

from scripts.export_multipart_motion_tokens import load_part_codec
from utils.adaptive_anchor_tokens import (
    BODY_CODEBOOK_SIZE,
    BODY_PART_ORDER,
    BODY_SLOT_COUNT,
    QUANTIZERS_PER_PART,
)


def normalized_part_codebook_distance_table(codebooks: torch.Tensor, part: str) -> torch.Tensor:
    """Return normalized squared code distances for one part, ``[4, 512, 512]``."""

    codebooks = torch.as_tensor(codebooks, dtype=torch.float32, device="cpu")
    expected_shape = (QUANTIZERS_PER_PART, BODY_CODEBOOK_SIZE)
    if tuple(codebooks.shape[:2]) != expected_shape or codebooks.ndim != 3:
        raise ValueError(
            f"{part} codebooks must have shape [{expected_shape[0]}, "
            f"{expected_shape[1]}, D], got {tuple(codebooks.shape)}"
        )
    tables: list[torch.Tensor] = []
    for quantizer in range(QUANTIZERS_PER_PART):
        entries = codebooks[quantizer]
        centered = entries - entries.mean(dim=0, keepdim=True)
        # E[||X-Y||^2] = 2 * sum_d Var(X_d). The per-dimension mean
        # matches the distance definition below and avoids constructing a
        # second matrix just to obtain the scale.
        scale = 2.0 * centered.square().mean()
        if not torch.isfinite(scale) or float(scale) <= 0:
            raise ValueError(f"Degenerate codebook geometry for {part}_q{quantizer}")
        distances = torch.cdist(entries, entries, p=2).square() / entries.shape[-1]
        distances = distances / scale
        distances.fill_diagonal_(0.0)
        tables.append(distances)
    return torch.stack(tables, dim=0).contiguous()


def normalized_codebook_distance_table(codebooks_by_part: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return normalized squared code distances with shape ``[16, 512, 512]``.

    Each RVQ codebook is normalized by its mean all-pairs squared distance.
    Consequently a random prediction has expected cost close to one, an exact
    prediction has zero cost, and one loss weight is meaningful across parts
    and residual levels.
    """

    part_tables: list[torch.Tensor] = []
    for part in BODY_PART_ORDER:
        if part not in codebooks_by_part:
            raise KeyError(f"Missing codebooks for body part {part!r}")
        part_tables.append(
            normalized_part_codebook_distance_table(codebooks_by_part[part], part)
        )
    result = torch.cat(part_tables, dim=0).contiguous()
    if tuple(result.shape) != (BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE, BODY_CODEBOOK_SIZE):
        raise AssertionError(f"Unexpected distance-table shape: {tuple(result.shape)}")
    return result


def load_normalized_codebook_distance_table(
    checkpoint_by_part: Mapping[str, str | Path],
) -> torch.Tensor:
    """Load frozen causal codec codebooks and build the Step 1 cost table.

    Codecs are loaded one at a time on CPU so the trainer does not retain four
    RVQ-VAE instances after extracting their codebooks.
    """

    codebooks_by_part: dict[str, torch.Tensor] = {}
    for part in BODY_PART_ORDER:
        if part not in checkpoint_by_part:
            raise KeyError(f"Missing codec checkpoint for body part {part!r}")
        loaded = load_part_codec(Path(checkpoint_by_part[part]), torch.device("cpu"))
        if loaded.part != part:
            raise ValueError(
                f"Configured {part} checkpoint contains the {loaded.part} codec: "
                f"{loaded.checkpoint_path}"
            )
        if not loaded.causal:
            raise ValueError(f"Expected a causal codec for {part}: {loaded.checkpoint_path}")
        codebooks_by_part[part] = (
            loaded.model.quantizers[part].codebooks.detach().float().cpu().clone()
        )
        del loaded
        gc.collect()
    return normalized_codebook_distance_table(codebooks_by_part)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
sys.path.insert(0, str(MODULE_DIR))

from scripts.export_multipart_motion_tokens import (  # noqa: E402
    LoadedPartCodec,
    configure_strict_inference_math,
    load_part_codec,
)
from utils.multipart_motion import (  # noqa: E402
    PART_ORDER,
    load_motion_dict,
    load_name_list,
    motion_path_for_name,
    split_motion_parts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit trained body codecs for strict encoder/decoder causality, exact "
            "prefix behavior, streaming equivalence, and length alignment."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs",
    )
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--upper_ckpt", type=Path, required=True)
    parser.add_argument("--lower_ckpt", type=Path, required=True)
    parser.add_argument("--feet_ckpt", type=Path, required=True)
    parser.add_argument("--hands_ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_clips", type=int, default=8)
    parser.add_argument(
        "--max_source_frames",
        type=int,
        default=128,
        help="Even prefix length used per audit clip; bounds the quadratic streaming check.",
    )
    parser.add_argument("--root_abs_threshold", type=float, default=10.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        max_error = float((actual - expected).abs().max().item())
        raise AssertionError(f"{name}: max absolute error {max_error:.8g} exceeds {atol}")


@torch.no_grad()
def audit_part_clip(
    loaded: LoadedPartCodec,
    value: np.ndarray,
    device: torch.device,
    max_source_frames: int,
    atol: float,
) -> Dict[str, int | float]:
    part = loaded.part
    unit = loaded.unit_length
    source_frames = min(int(value.shape[0]), int(max_source_frames))
    source_frames -= source_frames % unit
    if source_frames < unit * 2:
        raise ValueError(f"Need at least {unit * 2} frames for the causality audit")

    normalized = loaded.normalizer.normalize(part, value[:source_frames])
    x = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
    model = loaded.model

    full_latent = model.encoders[part](model.preprocess(x))
    full_codes = model.encode({part: x})[part]
    expected_tokens = source_frames // unit
    if full_codes.shape[1] != expected_tokens:
        raise AssertionError(
            f"{part}: expected {expected_tokens} tokens from {source_frames} frames, "
            f"got {full_codes.shape[1]}"
        )

    stream_codes = []
    for token_count in range(1, expected_tokens + 1):
        end = token_count * unit
        prefix_x = x[:, :end]
        prefix_latent = model.encoders[part](model.preprocess(prefix_x))
        prefix_codes = model.encode({part: prefix_x})[part]
        assert_close(
            f"{part}/encoder_prefix_latent/{token_count}",
            prefix_latent,
            full_latent[..., :token_count],
            atol,
        )
        if not torch.equal(prefix_codes, full_codes[:, :token_count]):
            raise AssertionError(f"{part}: encoder token prefix changed at {token_count}")
        stream_codes.append(prefix_codes[:, -1:])
    if not torch.equal(torch.cat(stream_codes, dim=1), full_codes):
        raise AssertionError(f"{part}: streaming encoder codes differ from one-shot codes")

    boundary_tokens = max(1, expected_tokens // 2)
    boundary_source = boundary_tokens * unit
    perturbed_x = x.clone()
    generator = torch.Generator(device=device).manual_seed(1701)
    perturbation = torch.randn(
        perturbed_x[:, boundary_source:].shape,
        generator=generator,
        device=device,
        dtype=perturbed_x.dtype,
    )
    perturbed_x[:, boundary_source:] = perturbation * 10.0
    perturbed_latent = model.encoders[part](model.preprocess(perturbed_x))
    perturbed_codes = model.encode({part: perturbed_x})[part]
    assert_close(
        f"{part}/future_input_latent",
        perturbed_latent[..., :boundary_tokens],
        full_latent[..., :boundary_tokens],
        atol,
    )
    if not torch.equal(perturbed_codes[:, :boundary_tokens], full_codes[:, :boundary_tokens]):
        raise AssertionError(f"{part}: future motion changed earlier token IDs")

    full_decoded = model.decode({part: full_codes})[part]
    if full_decoded.shape[1] != source_frames:
        raise AssertionError(
            f"{part}: decoded {full_decoded.shape[1]} frames from {expected_tokens} tokens; "
            f"expected {source_frames}"
        )
    stream_motion = []
    for token_count in range(1, expected_tokens + 1):
        end = token_count * unit
        prefix_decoded = model.decode({part: full_codes[:, :token_count]})[part]
        assert_close(
            f"{part}/decoder_prefix/{token_count}",
            prefix_decoded,
            full_decoded[:, :end],
            atol,
        )
        stream_motion.append(prefix_decoded[:, -unit:])
    assert_close(
        f"{part}/streaming_decoder",
        torch.cat(stream_motion, dim=1),
        full_decoded,
        atol,
    )

    future_codes = full_codes.clone()
    future_codes[:, boundary_tokens:] = (future_codes[:, boundary_tokens:] + 1) % loaded.codebook_size
    future_decoded = model.decode({part: future_codes})[part]
    assert_close(
        f"{part}/future_token_decoder",
        future_decoded[:, :boundary_source],
        full_decoded[:, :boundary_source],
        atol,
    )

    odd_x = torch.cat([x, x[:, -1:]], dim=1)
    odd_codes = model.encode({part: odd_x})[part]
    if odd_codes.shape[1] != expected_tokens:
        raise AssertionError(f"{part}: an incomplete source tail emitted an extra token")

    return {
        "source_frames": source_frames,
        "token_frames": expected_tokens,
        "decoded_frames": int(full_decoded.shape[1]),
        "unit_length": unit,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    math_mode = configure_strict_inference_math(device)
    print("Strict inference math:", math_mode)
    data_root = args.data_root.resolve()
    split_file = (args.split_file or data_root / "split" / "val_file_list.txt").resolve()
    motion_dir = data_root / "motion_data"
    checkpoint_by_part: Mapping[str, Path] = {
        "upper": args.upper_ckpt,
        "lower": args.lower_ckpt,
        "feet": args.feet_ckpt,
        "hands": args.hands_ckpt,
    }
    codecs = {
        part: load_part_codec(checkpoint_by_part[part], device)
        for part in PART_ORDER
    }
    noncausal = [part for part, loaded in codecs.items() if not loaded.causal]
    if noncausal:
        raise ValueError(f"Refusing causal audit for noncausal checkpoint(s): {noncausal}")

    names = load_name_list(split_file)
    results: Dict[str, object] = {
        "status": "passed",
        "split_file": str(split_file),
        "device": str(device),
        "math_mode": math_mode,
        "parts": {},
    }
    audited = 0
    for name in names:
        path = motion_path_for_name(motion_dir, name)
        if not path.exists():
            continue
        parts, _ = split_motion_parts(
            load_motion_dict(path),
            abs_threshold=args.root_abs_threshold,
        )
        if min(value.shape[0] for value in parts.values()) < 4:
            continue
        for part in PART_ORDER:
            part_results = results["parts"].setdefault(part, [])
            part_results.append(
                {
                    "name": name,
                    **audit_part_clip(
                        codecs[part],
                        parts[part],
                        device,
                        max_source_frames=args.max_source_frames,
                        atol=args.atol,
                    ),
                }
            )
        audited += 1
        print(f"[passed] {name}")
        if audited >= args.max_clips:
            break

    if audited == 0:
        raise RuntimeError(f"No usable clips found in {split_file}")
    results["audited_clips"] = audited
    output_path = args.output_json
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Report: {output_path}")
    print(f"Causal codec audit passed for {audited} clips x {len(PART_ORDER)} parts")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
sys.path.insert(0, str(MODULE_DIR))

from models.multipart_rvqvae import MultiPartRVQVAE  # noqa: E402
from utils.multipart_motion import (  # noqa: E402
    PART_DIMS,
    PART_ORDER,
    PartNormalizer,
    load_motion_dict,
    load_name_list,
    motion_path_for_name,
    split_motion_parts,
)


def torch_load_trusted(path: Path, map_location=None):
    # Training checkpoints may contain pathlib.PosixPath values. Python cannot
    # instantiate that class on Windows, so map it to the native path class for
    # the duration of this explicitly trusted pickle load.
    saved_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    finally:
        pathlib.PosixPath = saved_posix_path


@dataclass
class LoadedPartCodec:
    part: str
    model: MultiPartRVQVAE
    normalizer: PartNormalizer
    checkpoint_path: Path
    codebook_size: int
    num_quantizers: int
    unit_length: int


def infer_part_from_path(path: Path) -> Optional[str]:
    text = str(path).lower()
    for part in PART_ORDER:
        if part in text:
            return part
    return None


def load_part_codec(path: Path, device: torch.device) -> LoadedPartCodec:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Part checkpoint not found: {path}")
    checkpoint = torch_load_trusted(path, map_location=device)
    args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    part_order = model_config.get("part_order") or args.get("parts") or [infer_part_from_path(path)]
    if isinstance(part_order, str):
        part_order = [part_order]
    part_order = [part for part in part_order if part is not None]
    if len(part_order) != 1:
        raise ValueError(f"Expected one part in {path}, got part_order={part_order}")
    part = str(part_order[0])

    codebook_size = int(args.get("codebook_size", model_config.get("nb_code", 512)))
    code_dim = int(args.get("code_dim", model_config.get("code_dim", 512)))
    num_quantizers = int(args.get("num_quantizers", model_config.get("num_quantizers", 4)))
    down_t = int(args.get("down_t", 1))
    stride_t = int(args.get("stride_t", 2))

    model = MultiPartRVQVAE(
        part_dims=PART_DIMS,
        part_order=[part],
        nb_code=codebook_size,
        code_dim=code_dim,
        num_quantizers=num_quantizers,
        down_t=down_t,
        stride_t=stride_t,
        width=int(args.get("width", 512)),
        depth=int(args.get("depth", 3)),
        dilation_growth_rate=int(args.get("dilation_growth_rate", 3)),
        activation=str(args.get("activation", "relu")),
        norm=args.get("norm", None),
        vq_cnn_depth=int(args.get("vq_cnn_depth", 3)),
        shared_codebook=bool(args.get("shared_codebook", False)),
        quantize_dropout_prob=float(args.get("quantize_dropout_prob", 0.0)),
        quantize_dropout_cutoff_index=int(args.get("quantize_dropout_cutoff_index", 1)),
        mu=float(args.get("mu", 0.99)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    normalizer_path = Path(str(checkpoint.get("normalizer_path", ""))).expanduser()
    if not normalizer_path.exists():
        normalizer_path = path.parent.parent / "meta" / "normalizer.npz"
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer not found for {part}: {normalizer_path}")
    normalizer = PartNormalizer.load(normalizer_path)

    return LoadedPartCodec(
        part=part,
        model=model,
        normalizer=normalizer,
        checkpoint_path=path,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        unit_length=down_t * stride_t,
    )


def output_json_path(output_dir: Path, name: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return output_dir / Path(*parts).with_suffix(".json")


def build_token_layout(part_order, num_quantizers: int):
    layout = []
    slot = 0
    for part in part_order:
        for quantizer in range(num_quantizers):
            layout.append({"slot": slot, "part": part, "quantizer": quantizer})
            slot += 1
    return layout


@torch.no_grad()
def encode_motion_parts(
    motion: Mapping[str, np.ndarray],
    codecs: Mapping[str, LoadedPartCodec],
    device: torch.device,
    root_abs_threshold: float,
) -> tuple[list[list[int]], Dict[str, object]]:
    parts, meta = split_motion_parts(motion, abs_threshold=root_abs_threshold)
    code_by_part: Dict[str, np.ndarray] = {}
    for part in PART_ORDER:
        loaded = codecs[part]
        x_np = loaded.normalizer.normalize(part, parts[part])
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        code_idx = loaded.model.encode({part: x})[part]
        code_by_part[part] = code_idx.squeeze(0).detach().cpu().numpy().astype(np.int64)

    token_frames = min(value.shape[0] for value in code_by_part.values())
    tokens: list[list[int]] = []
    for frame_idx in range(token_frames):
        frame = []
        for part in PART_ORDER:
            frame.extend(int(v) for v in code_by_part[part][frame_idx].tolist())
        tokens.append(frame)

    return tokens, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 16-token multipart RVQ-VAE motion token JSONs.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs",
    )
    parser.add_argument("--motion_dir", type=Path, default=None)
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--upper_ckpt", type=Path, required=True)
    parser.add_argument("--lower_ckpt", type=Path, required=True)
    parser.add_argument("--feet_ckpt", type=Path, required=True)
    parser.add_argument("--hands_ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--root_abs_threshold", type=float, default=10.0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    data_dir = args.data_dir.resolve()
    motion_dir = (args.motion_dir or data_dir / "motion_data").resolve()
    split_file = (args.split_file or data_dir / "split" / "all_file_list.txt").resolve()
    output_dir = (args.output_dir or data_dir / "motion_token_data_multipart_512x4").resolve()

    codecs = {
        "upper": load_part_codec(args.upper_ckpt, device),
        "lower": load_part_codec(args.lower_ckpt, device),
        "feet": load_part_codec(args.feet_ckpt, device),
        "hands": load_part_codec(args.hands_ckpt, device),
    }
    for expected, loaded in codecs.items():
        if loaded.part != expected:
            raise ValueError(f"{expected}_ckpt loaded part '{loaded.part}', expected '{expected}'")

    codebook_sizes = {loaded.codebook_size for loaded in codecs.values()}
    quantizer_counts = {loaded.num_quantizers for loaded in codecs.values()}
    unit_lengths = {loaded.unit_length for loaded in codecs.values()}
    if len(codebook_sizes) != 1 or len(quantizer_counts) != 1 or len(unit_lengths) != 1:
        raise ValueError(
            "All part codecs must share codebook_size, num_quantizers, and unit_length. "
            f"Got codebook_sizes={codebook_sizes}, quantizers={quantizer_counts}, unit_lengths={unit_lengths}"
        )

    codebook_size = next(iter(codebook_sizes))
    num_quantizers = next(iter(quantizer_counts))
    unit_length = next(iter(unit_lengths))
    token_layout = build_token_layout(PART_ORDER, num_quantizers)
    token_fps = float(args.motion_fps) / float(unit_length)

    names = load_name_list(split_file)
    if args.max_clips is not None:
        names = names[: args.max_clips]
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"exported": 0, "skipped_existing": 0, "missing": 0, "failed": 0}
    for index, name in enumerate(names, start=1):
        motion_path = motion_path_for_name(motion_dir, name)
        out_path = output_json_path(output_dir, name)
        if out_path.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
            continue
        if not motion_path.exists():
            counts["missing"] += 1
            continue
        try:
            motion = load_motion_dict(motion_path)
            tokens, meta = encode_motion_parts(
                motion,
                codecs,
                device,
                root_abs_threshold=args.root_abs_threshold,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "name": name,
                "tokens": tokens,
                "fps": float(args.motion_fps),
                "motion_token_fps": token_fps,
                "motion_token_unit_length": unit_length,
                "codebook_size": codebook_size,
                "num_quantizers": num_quantizers,
                "part_order": list(PART_ORDER),
                "tokens_per_frame": len(PART_ORDER) * num_quantizers,
                "token_layout": token_layout,
                "root_schema": meta.get("root_schema"),
                "root_mean_norm": meta.get("root_mean_norm"),
                "source_motion_path": str(motion_path),
                "part_checkpoints": {part: str(loaded.checkpoint_path) for part, loaded in codecs.items()},
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            counts["exported"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"[failed] {name}: {exc}")

        if index % 100 == 0:
            print(f"{index}/{len(names)} {counts}")

    with open(output_dir / "export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_file": str(split_file),
                "motion_dir": str(motion_dir),
                "output_dir": str(output_dir),
                "counts": counts,
                "motion_fps": float(args.motion_fps),
                "motion_token_fps": token_fps,
                "motion_token_unit_length": unit_length,
                "codebook_size": codebook_size,
                "num_quantizers": num_quantizers,
                "part_order": list(PART_ORDER),
                "token_layout": token_layout,
            },
            f,
            indent=2,
        )
    print("Done:", counts)
    print("Output:", output_dir)


if __name__ == "__main__":
    main()

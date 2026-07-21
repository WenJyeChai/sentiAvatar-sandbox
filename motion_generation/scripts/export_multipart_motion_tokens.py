#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
FORMAT_VERSION = 2

from models.multipart_rvqvae import MultiPartRVQVAE  # noqa: E402
from utils.multipart_motion import (  # noqa: E402
    FACE_PART,
    MULTIMODAL_PART_ORDER,
    PART_DIMS,
    PART_ORDER,
    PartNormalizer,
    load_face_coefficients,
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
    normalizer_path: Path
    checkpoint_path: Path
    codebook_size: int
    num_quantizers: int
    unit_length: int
    causal: bool


def infer_part_from_path(path: Path) -> Optional[str]:
    text = str(path).lower()
    for part in MULTIMODAL_PART_ORDER:
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
    causal = bool(model_config.get("causal", args.get("causal", False)))

    checkpoint_part_dims = model_config.get("part_dims") or {}
    model = MultiPartRVQVAE(
        part_dims={**PART_DIMS, **checkpoint_part_dims},
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
        causal=causal,
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
        normalizer_path=normalizer_path.resolve(),
        checkpoint_path=path,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        unit_length=int(
            model_config.get(
                "unit_length",
                stride_t ** down_t if causal else down_t * stride_t,
            )
        ),
        causal=causal,
    )


def output_json_path(output_dir: Path, name: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return output_dir / Path(*parts).with_suffix(".json")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).replace("\\", "/").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assigned_shard_names(names, num_shards: int, shard_id: int):
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_id < num_shards")
    if len(names) != len(set(names)):
        raise ValueError("Split file contains duplicate clip names")
    return [name for index, name in enumerate(names) if index % num_shards == shard_id]


def shard_manifest_name(num_shards: int, shard_id: int) -> str:
    return f"export_manifest_shard_{shard_id:05d}_of_{num_shards:05d}.json"


def atomic_write_json(path: Path, payload, *, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_signature(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def existing_output_matches(path: Path, name: str, signature: str) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            isinstance(payload, dict)
            and payload.get("format_version") == FORMAT_VERSION
            and payload.get("name") == name
            and payload.get("export_signature") == signature
        )
    except (OSError, ValueError, TypeError):
        return False


def configure_strict_inference_math(device: torch.device) -> Dict[str, object]:
    """Disable TF32 so full-prefix and streaming RVQ decisions stay stable."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
    return {
        "device_type": device.type,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


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
    face: Optional[np.ndarray] = None,
    part_order=PART_ORDER,
) -> tuple[list[list[int]], Dict[str, object]]:
    parts, meta = split_motion_parts(motion, abs_threshold=root_abs_threshold)
    if FACE_PART in part_order:
        if face is None:
            raise ValueError("Face coefficients are required when face is in part_order")
        parts[FACE_PART] = np.asarray(face, dtype=np.float32)
    source_lengths = {part: int(parts[part].shape[0]) for part in part_order}
    source_frames = min(source_lengths.values())
    parts = {part: parts[part][:source_frames] for part in part_order}
    meta["source_part_frames"] = source_lengths
    meta["aligned_source_frames"] = source_frames
    code_by_part: Dict[str, np.ndarray] = {}
    for part in part_order:
        loaded = codecs[part]
        x_np = loaded.normalizer.normalize(part, parts[part])
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        code_idx = loaded.model.encode({part: x})[part]
        code_by_part[part] = code_idx.squeeze(0).detach().cpu().numpy().astype(np.int64)

    token_frames = min(value.shape[0] for value in code_by_part.values())
    tokens: list[list[int]] = []
    for frame_idx in range(token_frames):
        frame = []
        for part in part_order:
            frame.extend(int(v) for v in code_by_part[part][frame_idx].tolist())
        tokens.append(frame)

    return tokens, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export multipart RVQ-VAE body tokens with an optional ARKit face stream.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs",
    )
    parser.add_argument("--motion_dir", type=Path, default=None)
    parser.add_argument("--face_dir", type=Path, default=None)
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--upper_ckpt", type=Path, required=True)
    parser.add_argument("--lower_ckpt", type=Path, required=True)
    parser.add_argument("--feet_ckpt", type=Path, required=True)
    parser.add_argument("--hands_ckpt", type=Path, required=True)
    parser.add_argument(
        "--face_ckpt",
        type=Path,
        default=None,
        help="Optional face RVQ checkpoint. Adds four face slots after the 16 body slots.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--root_abs_threshold", type=float, default=10.0)
    parser.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Optional number of assigned clips to export in this shard (debug only).",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of disjoint modulo shards sharing the output directory.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Zero-based shard index in [0, num_shards).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("max_clips must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_id < num_shards")
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    math_mode = configure_strict_inference_math(device)
    data_dir = args.data_dir.resolve()
    motion_dir = (args.motion_dir or data_dir / "motion_data").resolve()
    face_dir = (args.face_dir or data_dir / "arkit_data").resolve()
    split_file = (args.split_file or data_dir / "split" / "all_file_list.txt").resolve()
    default_output_name = (
        "motion_face_token_data_multipart_512x4"
        if args.face_ckpt is not None
        else "motion_token_data_multipart_512x4"
    )
    output_dir = (args.output_dir or data_dir / default_output_name).resolve()

    codecs = {
        "upper": load_part_codec(args.upper_ckpt, device),
        "lower": load_part_codec(args.lower_ckpt, device),
        "feet": load_part_codec(args.feet_ckpt, device),
        "hands": load_part_codec(args.hands_ckpt, device),
    }
    if args.face_ckpt is not None:
        codecs[FACE_PART] = load_part_codec(args.face_ckpt, device)
    part_order = MULTIMODAL_PART_ORDER if args.face_ckpt is not None else PART_ORDER
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
    token_layout = build_token_layout(part_order, num_quantizers)
    token_fps = float(args.motion_fps) / float(unit_length)

    all_names = load_name_list(split_file)
    names = assigned_shard_names(all_names, args.num_shards, args.shard_id)
    if args.max_clips is not None:
        names = names[: args.max_clips]
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_fingerprints = {
        part: {
            "path": str(loaded.checkpoint_path),
            "sha256": sha256_file(loaded.checkpoint_path),
            "normalizer_path": str(loaded.normalizer_path),
            "normalizer_sha256": sha256_file(loaded.normalizer_path),
        }
        for part, loaded in codecs.items()
    }
    signature_payload = {
        "format_version": FORMAT_VERSION,
        "motion_fps": float(args.motion_fps),
        "motion_token_fps": token_fps,
        "motion_token_unit_length": unit_length,
        "codebook_size": codebook_size,
        "num_quantizers": num_quantizers,
        "part_order": list(part_order),
        "token_layout": token_layout,
        "causal_by_part": {part: loaded.causal for part, loaded in codecs.items()},
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "math_mode": math_mode,
    }
    signature = export_signature(signature_payload)
    print(
        f"Shard {args.shard_id}/{args.num_shards}: {len(names)} assigned "
        f"from {len(all_names)} split clips"
    )
    print("Strict inference math:", math_mode)

    counts = {
        "exported": 0,
        "skipped_existing": 0,
        "invalid_existing_rewritten": 0,
        "missing": 0,
        "failed": 0,
    }
    for index, name in enumerate(names, start=1):
        motion_path = motion_path_for_name(motion_dir, name)
        face_path = motion_path_for_name(face_dir, name)
        out_path = output_json_path(output_dir, name)
        if out_path.exists() and not args.overwrite:
            if existing_output_matches(out_path, name, signature):
                counts["skipped_existing"] += 1
                continue
            counts["invalid_existing_rewritten"] += 1
        if not motion_path.exists() or (FACE_PART in part_order and not face_path.exists()):
            counts["missing"] += 1
            continue
        try:
            motion = load_motion_dict(motion_path)
            face = load_face_coefficients(face_path) if FACE_PART in part_order else None
            tokens, meta = encode_motion_parts(
                motion,
                codecs,
                device,
                root_abs_threshold=args.root_abs_threshold,
                face=face,
                part_order=part_order,
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
                "part_order": list(part_order),
                "tokens_per_frame": len(part_order) * num_quantizers,
                "token_layout": token_layout,
                "root_schema": meta.get("root_schema"),
                "root_mean_norm": meta.get("root_mean_norm"),
                "source_part_frames": meta.get("source_part_frames"),
                "aligned_source_frames": meta.get("aligned_source_frames"),
                "source_motion_path": str(motion_path),
                "source_face_path": str(face_path) if FACE_PART in part_order else None,
                "part_checkpoints": {part: str(loaded.checkpoint_path) for part, loaded in codecs.items()},
                "causal_by_part": {part: loaded.causal for part, loaded in codecs.items()},
                "body_causal": all(codecs[part].causal for part in PART_ORDER),
            }
            payload["format_version"] = FORMAT_VERSION
            payload["export_signature"] = signature
            atomic_write_json(out_path, payload)
            counts["exported"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"[failed] {name}: {exc}")

        if index % 100 == 0:
            print(f"{index}/{len(names)} {counts}")

    manifest = {
        "format_version": FORMAT_VERSION,
        "split_file": str(split_file),
        "motion_dir": str(motion_dir),
        "output_dir": str(output_dir),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "total_split_clips": len(all_names),
        "assigned_clips": len(names),
        "split_names_sha256": names_sha256(all_names),
        "assigned_names_sha256": names_sha256(names),
        "max_clips": args.max_clips,
        "counts": counts,
        "motion_fps": float(args.motion_fps),
        "motion_token_fps": token_fps,
        "motion_token_unit_length": unit_length,
        "codebook_size": codebook_size,
        "num_quantizers": num_quantizers,
        "part_order": list(part_order),
        "tokens_per_frame": len(part_order) * num_quantizers,
        "token_layout": token_layout,
        "causal_by_part": {part: loaded.causal for part, loaded in codecs.items()},
        "body_causal": all(codecs[part].causal for part in PART_ORDER),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "math_mode": math_mode,
        "export_signature": signature,
    }
    manifest_path = output_dir / shard_manifest_name(args.num_shards, args.shard_id)
    atomic_write_json(manifest_path, manifest, indent=2)
    if args.num_shards == 1:
        # Preserve the historical one-process output until the verifier replaces
        # it with the consolidated manifest.
        atomic_write_json(output_dir / "export_manifest.json", manifest, indent=2)
    print("Done:", counts)
    print("Output:", output_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

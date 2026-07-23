#!/usr/bin/env python3
"""Decode stored MOSS Nano q0-q15 codes into Step 2 conditioning features.

The MOSS Nano quantizer reconstructs its 768-D quantized latent by summing the
decoded contribution from every supplied residual codebook and applying the
frozen quantizer output projection. This exporter supplies all 16 stored RVQ
layers, writes one native-rate ``[time, 768]`` feature array per clip, and
leaves the existing Step 2 audio projection trainable.

No waveform decoder is used. Step 2 aligns the native 12.5 Hz feature frames
to its 10 Hz motion-token frames by nearest physical time.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch


from precompute_moss_nano_audio_tokens import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    NANO_CARDINALITY,
    NANO_CODEBOOKS,
    NANO_FRAME_RATE,
    NANO_SAMPLE_RATE,
    canonical_name_path,
    load_model,
    read_names,
    sha256_file,
)


FEATURE_DIM = 768
FEATURE_FORMAT_VERSION = 1
DEFAULT_TOKEN_DIR_NAME = "audio_tokens_moss_nano_48k_12p5hz_16cb"
DEFAULT_FEATURE_DIR_NAME = "audio_features_moss_nano_all16_12p5hz_768d"


@dataclass(frozen=True)
class NanoCodeItem:
    name: str
    path: Path
    codes: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.codes.shape[1])


def load_code_item(name: str, token_dir: Path) -> NanoCodeItem:
    path = canonical_name_path(token_dir, name, ".npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        codes = np.asarray(payload["codes"])
        codec = str(payload["codec"].item())
        stored_name = str(payload["name"].item())
        sample_rate = int(payload["sample_rate"].item())
        frame_rate = float(payload["frame_rate"].item())
        num_codebooks = int(payload["num_codebooks"].item())
        cardinality = int(payload["cardinality"].item())

    if codec != "moss_audio_tokenizer_nano" or stored_name != name:
        raise ValueError(f"{path}: token identity/codec mismatch")
    if sample_rate != NANO_SAMPLE_RATE or not np.isclose(
        frame_rate, NANO_FRAME_RATE
    ):
        raise ValueError(f"{path}: unexpected Nano rate metadata")
    if num_codebooks != NANO_CODEBOOKS or cardinality != NANO_CARDINALITY:
        raise ValueError(f"{path}: unexpected Nano quantizer metadata")
    if codes.ndim != 2 or codes.shape[0] != NANO_CODEBOOKS or codes.shape[1] < 1:
        raise ValueError(
            f"{path}: expected [{NANO_CODEBOOKS}, time] codes, got {codes.shape}"
        )
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"{path}: codes must have integer dtype, got {codes.dtype}")
    if int(codes.min()) < 0 or int(codes.max()) >= NANO_CARDINALITY:
        raise ValueError(f"{path}: code outside [0, {NANO_CARDINALITY - 1}]")
    return NanoCodeItem(
        name=name,
        path=path,
        codes=np.ascontiguousarray(codes, dtype=np.int64),
    )


def iter_code_batches(
    names: Sequence[str],
    token_dir: Path,
    *,
    batch_size: int,
    max_padded_batch_frames: int,
) -> Iterator[list[NanoCodeItem]]:
    if batch_size < 1 or max_padded_batch_frames < 1:
        raise ValueError("Batch limits must be positive")
    batch: list[NanoCodeItem] = []
    longest = 0
    for name in names:
        item = load_code_item(name, token_dir)
        proposed_longest = max(longest, item.frames)
        proposed_cost = proposed_longest * (len(batch) + 1)
        if batch and (
            len(batch) >= batch_size
            or proposed_cost > max_padded_batch_frames
        ):
            yield batch
            batch = []
            longest = 0
        batch.append(item)
        longest = max(longest, item.frames)
    if batch:
        yield batch


@torch.inference_mode()
def decode_all16_batch(
    quantizer: torch.nn.Module,
    items: Sequence[NanoCodeItem],
    device: torch.device,
) -> list[np.ndarray]:
    if not items:
        return []
    max_frames = max(item.frames for item in items)
    padded = np.zeros(
        (NANO_CODEBOOKS, len(items), max_frames),
        dtype=np.int64,
    )
    for batch_index, item in enumerate(items):
        padded[:, batch_index, : item.frames] = item.codes

    codes = torch.from_numpy(padded).to(device=device)
    decoded = quantizer.decode_codes(codes)
    expected_shape = (len(items), FEATURE_DIM, max_frames)
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected decoded latent shape {tuple(decoded.shape)}; "
            f"expected {expected_shape}"
        )
    decoded = decoded.detach().to(device="cpu", dtype=torch.float32)

    outputs: list[np.ndarray] = []
    for batch_index, item in enumerate(items):
        feature = (
            decoded[batch_index, :, : item.frames]
            .transpose(0, 1)
            .contiguous()
            .numpy()
        )
        if feature.shape != (item.frames, FEATURE_DIM):
            raise RuntimeError(f"{item.name}: invalid feature shape {feature.shape}")
        if not np.isfinite(feature).all():
            raise RuntimeError(f"{item.name}: decoded feature contains NaN or Inf")
        outputs.append(feature)
    return outputs


def write_feature_file(
    path: Path,
    feature: np.ndarray,
    *,
    feature_dtype: str,
) -> None:
    if feature.ndim != 2 or feature.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [time, {FEATURE_DIM}] feature, got {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError("Feature contains NaN or Inf")
    dtype = np.float16 if feature_dtype == "float16" else np.float32
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npy", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, feature.astype(dtype, copy=False), allow_pickle=False)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def validate_feature_file(path: Path, expected_frames: int) -> bool:
    try:
        feature = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            feature.shape == (expected_frames, FEATURE_DIM)
            and feature.dtype in (np.dtype(np.float16), np.dtype(np.float32))
            and np.isfinite(feature).all()
        )
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen MOSS Nano q0-q15 quantized latents for Step 2"
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--token_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_padded_batch_frames", type=int, default=32_768)
    parser.add_argument(
        "--feature_dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--compute_dtype",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument(
        "--attention_implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_id < num_shards")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("--max_clips must be positive")

    requested_device = torch.device(args.device)
    device = (
        requested_device
        if requested_device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    data_dir = args.data_dir.resolve()
    token_dir = (
        args.token_dir or data_dir / DEFAULT_TOKEN_DIR_NAME
    ).resolve()
    output_dir = (
        args.output_dir or data_dir / DEFAULT_FEATURE_DIR_NAME
    ).resolve()
    split_file = (
        args.split_file or data_dir / "split" / "all_file_list.txt"
    ).resolve()
    args.model_dir = args.model_dir.resolve()
    if not token_dir.is_dir():
        raise FileNotFoundError(f"Nano token directory not found: {token_dir}")

    all_names = read_names(split_file)
    names = [
        name
        for index, name in enumerate(all_names)
        if index % args.num_shards == args.shard_id
    ]
    if args.max_clips is not None:
        names = names[: args.max_clips]
    output_dir.mkdir(parents=True, exist_ok=True)

    pending: list[str] = []
    counts = Counter()
    for name in names:
        output_path = canonical_name_path(output_dir, name, ".npy")
        if output_path.exists() and not args.overwrite:
            if not args.verify_existing:
                counts["skipped_existing"] += 1
                continue
            item = load_code_item(name, token_dir)
            if validate_feature_file(output_path, item.frames):
                counts["skipped_existing"] += 1
                continue
            counts["invalid_existing"] += 1
        pending.append(name)

    print("=" * 76)
    print("MOSS Nano all-16-RVQ feature export for Step 2")
    print(f"Model:          {args.model_dir}")
    print(f"Token root:     {token_dir}")
    print(f"Feature root:   {output_dir}")
    print("Representation: q0-q15 summed frozen quantized latent")
    print(f"Feature shape:  native {NANO_FRAME_RATE:g} Hz x {FEATURE_DIM} dims")
    print(f"Storage dtype:  {args.feature_dtype}")
    print(f"Device:         {device}")
    print(
        f"Shard:          {args.shard_id}/{args.num_shards} "
        f"({len(names)} assigned, {len(pending)} pending)"
    )
    print("=" * 76)

    started = time.perf_counter()
    model = load_model(args, device) if pending else None
    quantizer = None if model is None else model.quantizer.eval()
    if quantizer is not None:
        if (
            int(getattr(quantizer, "num_quantizers", -1)) != NANO_CODEBOOKS
            or int(getattr(quantizer, "output_dim", -1)) != FEATURE_DIM
        ):
            raise RuntimeError("Loaded Nano quantizer does not match all16/768-D contract")

    processed = 0
    total_frames = 0
    for batch in iter_code_batches(
        pending,
        token_dir,
        batch_size=args.batch_size,
        max_padded_batch_frames=args.max_padded_batch_frames,
    ):
        assert quantizer is not None
        features = decode_all16_batch(quantizer, batch, device)
        for item, feature in zip(batch, features):
            output_path = canonical_name_path(output_dir, item.name, ".npy")
            write_feature_file(
                output_path,
                feature,
                feature_dtype=args.feature_dtype,
            )
            counts["exported"] += 1
            processed += 1
            total_frames += item.frames
        if processed and processed % 100 < len(batch):
            elapsed = time.perf_counter() - started
            print(
                f"{processed}/{len(pending)} exported | "
                f"{total_frames} frames | {elapsed:.1f}s"
            )

    config_path = args.model_dir / "config.json"
    weight_files = sorted(args.model_dir.glob("*.safetensors"))
    manifest = {
        "format_version": FEATURE_FORMAT_VERSION,
        "representation": "moss_nano_quantized_latent_q0_q15",
        "codec": "moss_audio_tokenizer_nano",
        "causal_codec": True,
        "all_rvq_layers": True,
        "stored_codebooks": list(range(NANO_CODEBOOKS)),
        "num_codebooks": NANO_CODEBOOKS,
        "codebook_size": NANO_CARDINALITY,
        "aggregation": (
            "sum decoded residual-codebook contributions, then apply the "
            "frozen quantizer output projection"
        ),
        "feature_dim": FEATURE_DIM,
        "feature_dtype": args.feature_dtype,
        "sample_rate": NANO_SAMPLE_RATE,
        "feature_fps": NANO_FRAME_RATE,
        "step2_motion_alignment": "nearest physical-time frame; no interpolation",
        "token_dir": str(token_dir),
        "output_dir": str(output_dir),
        "split_file": str(split_file),
        "model_dir": str(args.model_dir),
        "model_config_sha256": sha256_file(config_path),
        "model_weights": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_files
        ],
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_clips": len(names),
        "counts": dict(counts),
        "exported_frames": total_frames,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = (
        output_dir
        / f"manifest_shard_{args.shard_id:05d}_of_{args.num_shards:05d}.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("Done:", dict(counts))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

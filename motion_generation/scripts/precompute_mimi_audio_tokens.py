#!/usr/bin/env python3
"""Precompute causal Mimi audio tokens for the Phase 1 planner.

All eight Moshi codebooks are stored, while the fixed Phase 1 model initially
reads q0 only.  Offline Mimi encoding is safe here because the checkpoint is
causal and the repository preflight verifies exact offline/streaming equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs"
DEFAULT_MIMI_WEIGHT = (
    PROJECT_DIR
    / "checkpoints"
    / "mimi"
    / "tokenizer-e351c8d8-checkpoint125.safetensors"
)
MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_RATE = 12.5
MIMI_FRAME_SIZE = 1_920
MIMI_CODEBOOKS = 8
MIMI_CARDINALITY = 2_048
FORMAT_VERSION = 1


@dataclass
class AudioItem:
    name: str
    source_path: Path
    source_sample_rate: int
    source_num_samples: int
    audio_24k: np.ndarray

    @property
    def target_frames(self) -> int:
        return math.ceil(len(self.audio_24k) / MIMI_FRAME_SIZE)


def canonical_name_path(root: Path, name: str, suffix: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return root / Path(*parts).with_suffix(suffix)


def read_names(split_file: Path) -> list[str]:
    names = [line.strip().replace("\\", "/") for line in split_file.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        raise ValueError(f"Split file contains duplicate clip names: {split_file}")
    return names


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    elif np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    else:
        raise TypeError(f"Unsupported WAV dtype: {audio.dtype}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono/stereo WAV, got shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("Waveform contains NaN or Inf")
    return np.ascontiguousarray(audio, dtype=np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = MIMI_SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate:
        return np.ascontiguousarray(audio)
    factor = math.gcd(int(source_rate), int(target_rate))
    output = resample_poly(audio, target_rate // factor, source_rate // factor)
    expected = round(len(audio) * target_rate / source_rate)
    if len(output) > expected:
        output = output[:expected]
    elif len(output) < expected:
        output = np.pad(output, (0, expected - len(output)))
    return np.ascontiguousarray(output, dtype=np.float32)


def load_audio_item(name: str, wav_root: Path) -> AudioItem:
    source_path = canonical_name_path(wav_root, name, ".wav")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_rate, source_audio = wavfile.read(source_path)
    source_num_samples = int(source_audio.shape[0])
    audio = pcm_to_float32(source_audio)
    if len(audio) == 0:
        raise ValueError(f"Empty waveform: {source_path}")
    audio_24k = resample_audio(audio, int(source_rate), MIMI_SAMPLE_RATE)
    return AudioItem(
        name=name,
        source_path=source_path,
        source_sample_rate=int(source_rate),
        source_num_samples=source_num_samples,
        audio_24k=audio_24k,
    )


def iter_audio_batches(
    names: Sequence[str],
    wav_root: Path,
    *,
    batch_size: int,
    max_padded_batch_seconds: float,
) -> Iterator[list[AudioItem]]:
    if batch_size < 1 or max_padded_batch_seconds <= 0:
        raise ValueError("batch_size and max_padded_batch_seconds must be positive")
    batch: list[AudioItem] = []
    longest = 0
    for name in names:
        item = load_audio_item(name, wav_root)
        proposed_longest = max(longest, len(item.audio_24k))
        proposed_cost = proposed_longest * (len(batch) + 1) / MIMI_SAMPLE_RATE
        if batch and (len(batch) >= batch_size or proposed_cost > max_padded_batch_seconds):
            yield batch
            batch = []
            longest = 0
        batch.append(item)
        longest = max(longest, len(item.audio_24k))
    if batch:
        yield batch


@torch.inference_mode()
def encode_batch(mimi: torch.nn.Module, items: Sequence[AudioItem], device: torch.device) -> list[np.ndarray]:
    if not items:
        return []
    max_samples = max(item.target_frames * MIMI_FRAME_SIZE for item in items)
    waveform = torch.zeros((len(items), 1, max_samples), dtype=torch.float32, device=device)
    for row, item in enumerate(items):
        source = torch.from_numpy(item.audio_24k).to(device=device)
        waveform[row, 0, : source.numel()] = source
    codes = mimi.encode(waveform)
    if tuple(codes.shape[:2]) != (len(items), MIMI_CODEBOOKS):
        raise RuntimeError(f"Unexpected Mimi output shape: {tuple(codes.shape)}")
    outputs = []
    for row, item in enumerate(items):
        item_codes = codes[row, :, : item.target_frames].detach().cpu().numpy()
        if item_codes.min() < 0 or item_codes.max() >= MIMI_CARDINALITY:
            raise RuntimeError(f"Mimi code outside [0, {MIMI_CARDINALITY - 1}] for {item.name}")
        outputs.append(item_codes.astype(np.uint16, copy=False))
    return outputs


def write_token_file(path: Path, item: AudioItem, codes: np.ndarray) -> None:
    if codes.shape != (MIMI_CODEBOOKS, item.target_frames):
        raise ValueError(f"Invalid codes shape {codes.shape} for {item.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(
                handle,
                codes=codes,
                format_version=np.asarray(FORMAT_VERSION, dtype=np.int32),
                name=np.asarray(item.name),
                source_path=np.asarray(str(item.source_path)),
                source_sample_rate=np.asarray(item.source_sample_rate, dtype=np.int32),
                source_num_samples=np.asarray(item.source_num_samples, dtype=np.int64),
                sample_rate=np.asarray(MIMI_SAMPLE_RATE, dtype=np.int32),
                num_samples=np.asarray(len(item.audio_24k), dtype=np.int64),
                frame_rate=np.asarray(MIMI_FRAME_RATE, dtype=np.float32),
                frame_size=np.asarray(MIMI_FRAME_SIZE, dtype=np.int32),
                num_codebooks=np.asarray(MIMI_CODEBOOKS, dtype=np.int32),
                cardinality=np.asarray(MIMI_CARDINALITY, dtype=np.int32),
            )
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def validate_existing_token_file(path: Path, expected_name: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            codes = payload["codes"]
            return (
                str(payload["name"].item()) == expected_name
                and int(payload["format_version"].item()) == FORMAT_VERSION
                and codes.ndim == 2
                and codes.shape[0] == MIMI_CODEBOOKS
                and codes.dtype == np.uint16
                and int(payload["sample_rate"].item()) == MIMI_SAMPLE_RATE
                and float(payload["frame_rate"].item()) == MIMI_FRAME_RATE
                and int(payload["cardinality"].item()) == MIMI_CARDINALITY
            )
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mimi(args: argparse.Namespace, device: torch.device):
    moshi_package = args.moshi_repo.resolve() / "moshi"
    if not moshi_package.is_dir():
        raise FileNotFoundError(
            f"Moshi Python package not found at {moshi_package}. "
            "Pass --moshi_repo /path/to/moshi."
        )
    sys.path.insert(0, str(moshi_package))
    from moshi.models import loaders  # pylint: disable=import-outside-toplevel

    mimi = loaders.get_mimi(
        filename=args.mimi_weight.resolve(),
        device=device,
        num_codebooks=MIMI_CODEBOOKS,
    )
    mimi.eval()
    contract = (
        mimi.sample_rate == MIMI_SAMPLE_RATE
        and mimi.frame_rate == MIMI_FRAME_RATE
        and mimi.frame_size == MIMI_FRAME_SIZE
        and mimi.num_codebooks == MIMI_CODEBOOKS
        and mimi.cardinality == MIMI_CARDINALITY
        and mimi.quantizer.n_q_semantic == 1
    )
    if not contract:
        raise RuntimeError("Loaded Mimi model does not match the Phase 1 contract")
    return mimi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute causal Mimi q0-q7 tokens for Phase 1")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--wav_dir", type=Path, default=None)
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--moshi_repo", type=Path, default=PROJECT_DIR.parent / "moshi")
    parser.add_argument("--mimi_weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_padded_batch_seconds", type=float, default=120.0)
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
    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    data_dir = args.data_dir.resolve()
    wav_dir = (args.wav_dir or data_dir / "wav_data").resolve()
    split_file = (args.split_file or data_dir / "split" / "all_file_list.txt").resolve()
    output_dir = (args.output_dir or data_dir / "audio_tokens_mimi_12p5hz_8cb").resolve()
    if not args.mimi_weight.is_file():
        raise FileNotFoundError(f"Mimi checkpoint not found: {args.mimi_weight}")

    all_names = read_names(split_file)
    names = [name for index, name in enumerate(all_names) if index % args.num_shards == args.shard_id]
    if args.max_clips is not None:
        names = names[: args.max_clips]
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    counts = Counter()
    for name in names:
        output_path = canonical_name_path(output_dir, name, ".npz")
        if output_path.exists() and not args.overwrite:
            if not args.verify_existing or validate_existing_token_file(output_path, name):
                counts["skipped_existing"] += 1
                continue
            counts["invalid_existing"] += 1
        pending.append(name)

    print("=" * 72)
    print("Phase 1 Mimi token preprocessing")
    print(f"Moshi source: {args.moshi_repo.resolve()}")
    print(f"Mimi weight: {args.mimi_weight.resolve()}")
    print(f"Audio root:   {wav_dir}")
    print(f"Output root:  {output_dir}")
    print(f"Device:       {device}")
    print(f"Shard:        {args.shard_id}/{args.num_shards} ({len(names)} assigned, {len(pending)} pending)")
    print("=" * 72)

    start = time.perf_counter()
    mimi = load_mimi(args, device) if pending else None
    processed = 0
    source_rates = Counter()
    for batch in iter_audio_batches(
        pending,
        wav_dir,
        batch_size=args.batch_size,
        max_padded_batch_seconds=args.max_padded_batch_seconds,
    ):
        assert mimi is not None
        batch_codes = encode_batch(mimi, batch, device)
        for item, codes in zip(batch, batch_codes):
            output_path = canonical_name_path(output_dir, item.name, ".npz")
            write_token_file(output_path, item, codes)
            counts["exported"] += 1
            source_rates[item.source_sample_rate] += 1
            processed += 1
        if processed and (processed % 100 < len(batch)):
            elapsed = time.perf_counter() - start
            print(f"{processed}/{len(pending)} exported | {elapsed:.1f}s | source_rates={dict(source_rates)}")

    manifest = {
        "format_version": FORMAT_VERSION,
        "split_file": str(split_file),
        "wav_dir": str(wav_dir),
        "output_dir": str(output_dir),
        "mimi_weight": str(args.mimi_weight.resolve()),
        "mimi_sha256": sha256_file(args.mimi_weight.resolve()),
        "moshi_repo": str(args.moshi_repo.resolve()),
        "sample_rate": MIMI_SAMPLE_RATE,
        "frame_rate": MIMI_FRAME_RATE,
        "frame_size": MIMI_FRAME_SIZE,
        "num_codebooks": MIMI_CODEBOOKS,
        "cardinality": MIMI_CARDINALITY,
        "planner_initial_codebooks": [0],
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_clips": len(names),
        "counts": dict(counts),
        "source_sample_rates_exported": dict(source_rates),
        "elapsed_seconds": time.perf_counter() - start,
    }
    manifest_path = output_dir / f"manifest_shard_{args.shard_id:05d}_of_{args.num_shards:05d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done:", dict(counts))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

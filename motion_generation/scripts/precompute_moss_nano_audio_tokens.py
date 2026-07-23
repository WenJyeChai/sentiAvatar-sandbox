#!/usr/bin/env python3
"""Precompute causal MOSS-Audio-Tokenizer-Nano tokens for Step 1.

The exporter resamples dataset audio to 48 kHz, duplicates mono audio into the
stereo interface expected by Nano, and stores all 16 RVQ codebooks. Step 1 can
then select q0-q3 without having to re-encode the dataset if a later experiment
uses a different RVQ prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
DEFAULT_MODEL_DIR = PROJECT_DIR / "checkpoints" / "moss_audio_tokenizer_nano"
NANO_SAMPLE_RATE = 48_000
NANO_FRAME_RATE = 12.5
NANO_FRAME_SIZE = 3_840
NANO_CHANNELS = 2
NANO_CODEBOOKS = 16
NANO_CARDINALITY = 1_024
FORMAT_VERSION = 2


@dataclass
class AudioItem:
    name: str
    source_path: Path
    source_sample_rate: int
    source_num_samples: int
    audio_48k: np.ndarray

    @property
    def target_frames(self) -> int:
        return math.ceil(len(self.audio_48k) / NANO_FRAME_SIZE)


def canonical_name_path(root: Path, name: str, suffix: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return root / Path(*parts).with_suffix(suffix)


def read_names(split_file: Path) -> list[str]:
    names = [line.strip().replace("\\", "/") for line in split_file.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        raise ValueError(f"Split file contains duplicate clip names: {split_file}")
    return names


def pcm_to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / float(max(abs(info.min), info.max))
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


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = NANO_SAMPLE_RATE) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    audio = np.asarray(audio, dtype=np.float32)
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
    mono = pcm_to_mono_float32(source_audio)
    if len(mono) == 0:
        raise ValueError(f"Empty waveform: {source_path}")
    return AudioItem(
        name=name,
        source_path=source_path,
        source_sample_rate=int(source_rate),
        source_num_samples=source_num_samples,
        audio_48k=resample_audio(mono, int(source_rate)),
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
        proposed_longest = max(longest, len(item.audio_48k))
        proposed_cost = proposed_longest * (len(batch) + 1) / NANO_SAMPLE_RATE
        if batch and (len(batch) >= batch_size or proposed_cost > max_padded_batch_seconds):
            yield batch
            batch = []
            longest = 0
        batch.append(item)
        longest = max(longest, len(item.audio_48k))
    if batch:
        yield batch


@torch.inference_mode()
def encode_batch(
    model: torch.nn.Module,
    items: Sequence[AudioItem],
    device: torch.device,
    *,
    chunk_duration: float | None,
) -> list[np.ndarray]:
    stereo = [
        torch.from_numpy(item.audio_48k)
        .to(device=device)
        .unsqueeze(0)
        .expand(NANO_CHANNELS, -1)
        .contiguous()
        for item in items
    ]
    encoded = model.batch_encode(
        stereo,
        num_quantizers=NANO_CODEBOOKS,
        chunk_duration=chunk_duration,
    )
    codes = encoded.audio_codes
    lengths = encoded.audio_codes_lengths
    if codes is None or lengths is None or tuple(codes.shape[:2]) != (NANO_CODEBOOKS, len(items)):
        shape = None if codes is None else tuple(codes.shape)
        raise RuntimeError(f"Unexpected Nano output shape: {shape}")
    outputs: list[np.ndarray] = []
    for row, item in enumerate(items):
        length = int(lengths[row].item())
        if length != item.target_frames:
            raise RuntimeError(
                f"Nano returned {length} frames for {item.name}; expected {item.target_frames} "
                f"from {len(item.audio_48k)} samples"
            )
        item_codes = codes[:, row, :length].detach().cpu().numpy()
        if item_codes.size and (item_codes.min() < 0 or item_codes.max() >= NANO_CARDINALITY):
            raise RuntimeError(f"Nano code outside [0, {NANO_CARDINALITY - 1}] for {item.name}")
        outputs.append(item_codes.astype(np.uint16, copy=False))
    return outputs


def write_token_file(path: Path, item: AudioItem, codes: np.ndarray, *, model_dir: Path) -> None:
    if codes.shape != (NANO_CODEBOOKS, item.target_frames):
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
                codec=np.asarray("moss_audio_tokenizer_nano"),
                causal=np.asarray(True),
                name=np.asarray(item.name),
                source_path=np.asarray(str(item.source_path)),
                source_sample_rate=np.asarray(item.source_sample_rate, dtype=np.int32),
                source_num_samples=np.asarray(item.source_num_samples, dtype=np.int64),
                sample_rate=np.asarray(NANO_SAMPLE_RATE, dtype=np.int32),
                num_samples=np.asarray(len(item.audio_48k), dtype=np.int64),
                frame_rate=np.asarray(NANO_FRAME_RATE, dtype=np.float32),
                frame_size=np.asarray(NANO_FRAME_SIZE, dtype=np.int32),
                channels=np.asarray(NANO_CHANNELS, dtype=np.int32),
                mono_to_stereo=np.asarray(True),
                num_codebooks=np.asarray(NANO_CODEBOOKS, dtype=np.int32),
                cardinality=np.asarray(NANO_CARDINALITY, dtype=np.int32),
                model_dir=np.asarray(str(model_dir)),
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
                and str(payload["codec"].item()) == "moss_audio_tokenizer_nano"
                and int(payload["format_version"].item()) == FORMAT_VERSION
                and codes.ndim == 2
                and codes.shape[0] == NANO_CODEBOOKS
                and codes.dtype == np.uint16
                and int(payload["sample_rate"].item()) == NANO_SAMPLE_RATE
                and float(payload["frame_rate"].item()) == NANO_FRAME_RATE
                and int(payload["frame_size"].item()) == NANO_FRAME_SIZE
                and int(payload["cardinality"].item()) == NANO_CARDINALITY
            )
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(args: argparse.Namespace, device: torch.device):
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Nano model directory not found: {args.model_dir}")

    # Some MOSS revisions use the older spelling while newer Transformers
    # exposes PretrainedConfig. Keep this local compatibility alias isolated.
    import transformers.configuration_utils as configuration_utils

    if not hasattr(configuration_utils, "PreTrainedConfig") and hasattr(configuration_utils, "PretrainedConfig"):
        configuration_utils.PreTrainedConfig = configuration_utils.PretrainedConfig

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_dir.resolve(),
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    if hasattr(model, "set_attention_implementation"):
        model.set_attention_implementation(args.attention_implementation)
    if hasattr(model, "set_compute_dtype"):
        model.set_compute_dtype(args.compute_dtype)

    quantizer_kwargs = getattr(model.config, "quantizer_kwargs", {})
    contract = (
        int(getattr(model, "sampling_rate", -1)) == NANO_SAMPLE_RATE
        and int(getattr(model, "downsample_rate", -1)) == NANO_FRAME_SIZE
        and int(getattr(model.config, "number_channels", -1)) == NANO_CHANNELS
        and int(quantizer_kwargs.get("num_quantizers", -1)) == NANO_CODEBOOKS
        and int(quantizer_kwargs.get("codebook_size", -1)) == NANO_CARDINALITY
    )
    if not contract:
        raise RuntimeError("Loaded model does not match the MOSS Nano 48 kHz / 16-codebook contract")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute causal MOSS Nano q0-q15 tokens")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--wav_dir", type=Path, default=None)
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_padded_batch_seconds", type=float, default=80.0)
    parser.add_argument("--chunk_duration", type=float, default=2.0)
    parser.add_argument("--compute_dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--attention_implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
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
    if args.chunk_duration <= 0:
        raise ValueError("--chunk_duration must be positive")
    chunk_samples = round(args.chunk_duration * NANO_SAMPLE_RATE)
    if chunk_samples % NANO_FRAME_SIZE:
        raise ValueError("--chunk_duration * 48000 must be divisible by 3840")

    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    data_dir = args.data_dir.resolve()
    wav_dir = (args.wav_dir or data_dir / "wav_data").resolve()
    split_file = (args.split_file or data_dir / "split" / "all_file_list.txt").resolve()
    output_dir = (args.output_dir or data_dir / "audio_tokens_moss_nano_48k_12p5hz_16cb").resolve()
    args.model_dir = args.model_dir.resolve()

    all_names = read_names(split_file)
    names = [name for index, name in enumerate(all_names) if index % args.num_shards == args.shard_id]
    if args.max_clips is not None:
        names = names[: args.max_clips]
    output_dir.mkdir(parents=True, exist_ok=True)

    pending: list[str] = []
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
    print("Step 1 MOSS Nano token preprocessing")
    print(f"Model:        {args.model_dir}")
    print(f"Audio root:   {wav_dir}")
    print(f"Output root:  {output_dir}")
    print(f"Device:       {device}")
    print(f"Codebooks:    q0-q15 stored; Step 1 will read q0-q3")
    print(f"Streaming:    {args.chunk_duration:.3f}s chunks")
    print(f"Shard:        {args.shard_id}/{args.num_shards} ({len(names)} assigned, {len(pending)} pending)")
    print("=" * 72)

    start = time.perf_counter()
    model = load_model(args, device) if pending else None
    processed = 0
    source_rates = Counter()
    for batch in iter_audio_batches(
        pending,
        wav_dir,
        batch_size=args.batch_size,
        max_padded_batch_seconds=args.max_padded_batch_seconds,
    ):
        assert model is not None
        batch_codes = encode_batch(model, batch, device, chunk_duration=args.chunk_duration)
        for item, codes in zip(batch, batch_codes):
            output_path = canonical_name_path(output_dir, item.name, ".npz")
            write_token_file(output_path, item, codes, model_dir=args.model_dir)
            counts["exported"] += 1
            source_rates[item.source_sample_rate] += 1
            processed += 1
        if processed and processed % 100 < len(batch):
            elapsed = time.perf_counter() - start
            print(f"{processed}/{len(pending)} exported | {elapsed:.1f}s | source_rates={dict(source_rates)}")

    config_path = args.model_dir / "config.json"
    weight_files = sorted(args.model_dir.glob("*.safetensors"))
    manifest = {
        "format_version": FORMAT_VERSION,
        "codec": "moss_audio_tokenizer_nano",
        "causal": True,
        "split_file": str(split_file),
        "wav_dir": str(wav_dir),
        "output_dir": str(output_dir),
        "model_dir": str(args.model_dir),
        "model_config_sha256": sha256_file(config_path),
        "model_weights": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in weight_files
        ],
        "sample_rate": NANO_SAMPLE_RATE,
        "frame_rate": NANO_FRAME_RATE,
        "frame_size": NANO_FRAME_SIZE,
        "channels": NANO_CHANNELS,
        "mono_to_stereo": True,
        "num_codebooks": NANO_CODEBOOKS,
        "cardinality": NANO_CARDINALITY,
        "stored_codebooks": list(range(NANO_CODEBOOKS)),
        "planner_codebooks": [0, 1, 2, 3],
        "chunk_duration": args.chunk_duration,
        "compute_dtype": args.compute_dtype,
        "attention_implementation": args.attention_implementation,
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

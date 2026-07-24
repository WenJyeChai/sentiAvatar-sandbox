#!/usr/bin/env python3
"""Cache detached frozen-Step-2 costs for every feasible anchor interval.

Run independent modulo shards on separate GPUs.  Each cache file stores
per-missing-frame canonical token CE and final hard RVQ-latent L1.  The later
DP calibration converts these means to total interval risk by multiplying by
the number of missing frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.audio_motion_model import AudioMotionTransformer  # noqa: E402
from scripts.train_audio_mask_multipart import load_sequences, read_split_file  # noqa: E402
from scripts.train_audio_mask_multipart_variable_c2f import (  # noqa: E402
    VariableGapC2FCollator,
    VariableGapMaskDataset,
    VariableGapMaskExample,
)
from utils.msd.multipart_adapter import MultipartCodebookSet  # noqa: E402
from utils.step1_adaptive_schedule import MAX_GAP, cache_path  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR
        / "motion_generation/configs/"
        "audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split_file", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> str:
    files = [path / "config.json"]
    files.extend(sorted(path.glob("*.safetensors")))
    files.extend(sorted(path.glob("pytorch_model*.bin")))
    if len(files) == 1:
        raise FileNotFoundError(f"No model weights found in Step 2 checkpoint: {path}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key} must be a mapping")
    return dict(value)


def all_examples(item: Mapping[str, Any]) -> list[VariableGapMaskExample]:
    usable = VariableGapMaskDataset._usable_motion_frames(dict(item))
    result = []
    for gap in range(1, min(MAX_GAP, usable - 2) + 1):
        for left in range(usable - gap - 1):
            right = left + gap + 1
            frames = range(left, right + 1)
            result.append(
                VariableGapMaskExample(
                    name=str(item["name"]),
                    left_idx=left,
                    right_idx=right,
                    gap_frames=gap,
                    motion_tokens=[
                        list(item["motion_tokens"][frame]) for frame in frames
                    ],
                    audio_features=torch.stack(
                        [
                            VariableGapMaskDataset._audio_feature_for_frame(
                                dict(item), frame
                            )
                            for frame in frames
                        ]
                    ),
                )
            )
    return result


def stage_mask(
    middle_mask: torch.Tensor,
    *,
    stage: int,
    tokens_per_frame: int,
    num_quantizers: int,
) -> torch.Tensor:
    slots = torch.arange(
        middle_mask.shape[1], device=middle_mask.device
    ).remainder(tokens_per_frame)
    return middle_mask & slots.remainder(num_quantizers).view(1, -1).eq(stage)


def hard_latent_l1_per_example(
    predicted_ids: torch.Tensor,
    gt_ids: torch.Tensor,
    middle_mask: torch.Tensor,
    codebooks: MultipartCodebookSet,
) -> torch.Tensor:
    batch, tokens = predicted_ids.shape
    ntpf = codebooks.tokens_per_frame
    frames = tokens // ntpf
    predicted = predicted_ids.reshape(batch, frames, ntpf)
    target = gt_ids.reshape_as(predicted)
    frame_valid = middle_mask.reshape_as(predicted)[..., 0]
    totals = torch.zeros(batch, device=predicted_ids.device, dtype=torch.float32)
    counts = torch.zeros_like(totals)
    for part_index, part in enumerate(codebooks.part_order):
        predicted_latent = 0.0
        target_latent = 0.0
        for quantizer in range(codebooks.num_quantizers):
            slot = part_index * codebooks.num_quantizers + quantizer
            offset = slot * codebooks.codebook_size
            predicted_local = (predicted[..., slot] - offset).clamp(
                0, codebooks.codebook_size - 1
            )
            target_local = (target[..., slot] - offset).clamp(
                0, codebooks.codebook_size - 1
            )
            book = codebooks.codebooks[part][quantizer]
            predicted_latent = predicted_latent + book[predicted_local]
            target_latent = target_latent + book[target_local]
        error = (predicted_latent - target_latent).abs().mean(dim=-1)
        totals += (error * frame_valid).sum(dim=-1)
        counts += frame_valid.sum(dim=-1)
    return totals / counts.clamp_min(1)


@torch.inference_mode()
def score_batch(
    model: AudioMotionTransformer,
    batch: Mapping[str, torch.Tensor],
    *,
    codebooks: MultipartCodebookSet,
    device: torch.device,
    use_bf16: bool,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }
    current = tensors["input_ids"].clone()
    gt_ids = tensors["gt_ids"]
    middle = tensors["middle_mask"]
    ntpf = int(model.config.num_tokens_per_frame)
    quantizers = int(model.config.num_quantizers_per_part)
    ce_sum = torch.zeros(current.shape[0], device=device, dtype=torch.float32)
    ce_count = torch.zeros_like(ce_sum)
    autocast = use_bf16 and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
        encoded_audio = model.audio_encoder(tensors["audio_features"])
        for stage in range(quantizers):
            logits = model(
                input_ids=current,
                audio_features=None,
                encoded_audio=encoded_audio,
                attention_mask=tensors["attention_mask"],
                middle_mask=middle,
                gap_lengths=tensors["gap_lengths"],
                c2f_stage=stage,
            )
            valid = stage_mask(
                middle,
                stage=stage,
                tokens_per_frame=ntpf,
                num_quantizers=quantizers,
            )
            positions = valid.nonzero(as_tuple=False)
            selected_loss = F.cross_entropy(
                logits[valid].float(),
                gt_ids[valid],
                reduction="none",
            )
            ce_sum.scatter_add_(0, positions[:, 0], selected_loss)
            ce_count.scatter_add_(
                0,
                positions[:, 0],
                torch.ones_like(selected_loss),
            )
            predictions = logits.argmax(dim=-1)
            current[valid] = predictions[valid]
    latent = hard_latent_l1_per_example(current, gt_ids, middle, codebooks)
    ce = ce_sum / ce_count.clamp_min(1)
    return ce.cpu().numpy(), latent.cpu().numpy()


def save_clip(
    path: Path,
    *,
    num_frames: int,
    examples: Sequence[VariableGapMaskExample],
    ce_values: np.ndarray,
    latent_values: np.ndarray,
) -> None:
    ce = np.full((num_frames, MAX_GAP + 1), np.inf, dtype=np.float32)
    latent = np.full_like(ce, np.inf)
    ce[:, 0] = 0.0
    latent[:, 0] = 0.0
    for example, ce_value, latent_value in zip(examples, ce_values, latent_values):
        ce[example.left_idx, example.gap_frames] = float(ce_value)
        latent[example.left_idx, example.gap_frames] = float(latent_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            ce=ce,
            hard_latent_l1=latent,
            num_frames=np.int64(num_frames),
        )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment = section(config, "experiment")
    data = section(config, "data")
    audio = section(config, "audio_conditioning")
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else project_path(experiment["output_dir"])
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Frozen Step 2 checkpoint not found: {checkpoint}")
    split_paths = [path.resolve() for path in args.split_file]
    names: list[str] = []
    for path in split_paths:
        names.extend(read_split_file(path))
    names = list(dict.fromkeys(str(name).replace("\\", "/") for name in names))
    assigned = names[args.shard_id :: args.num_shards]
    if args.max_clips is not None:
        assigned = assigned[: args.max_clips]

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = AudioMotionTransformer.from_pretrained(
        checkpoint, local_files_only=True
    ).to(device).eval()
    part_order = tuple(str(value) for value in data["part_order"].split(","))
    codec_paths = {
        part: project_path(data[f"{part}_ckpt"]) for part in part_order
    }
    codebooks = MultipartCodebookSet.from_checkpoints(
        codec_paths, device=device, part_order=part_order
    )
    if codebooks.tokens_per_frame != int(model.config.num_tokens_per_frame):
        raise ValueError("Step 2 and causal-codec token layouts differ")
    collator = VariableGapC2FCollator(model.config)
    motion_dir = project_path(data["motion_token_dir"])
    audio_dir = project_path(data["audio_feat_dir"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = skipped = missing = 0
    start = time.perf_counter()

    for clip_index, name in enumerate(assigned):
        destination = cache_path(output_dir, name)
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        sequences, stats = load_sequences(
            [name],
            motion_dir,
            audio_dir,
            codebook_size=int(model.config.codebook_size),
            num_tokens_per_frame=int(model.config.num_tokens_per_frame),
            audio_fps=float(audio["audio_fps"]),
            source_motion_fps_fallback=20.0,
            motion_token_fps_override=None,
            motion_token_unit_length_override=None,
        )
        if not sequences:
            print(f"[skip] {name}: {stats}")
            missing += 1
            continue
        item = sequences[0]
        usable = VariableGapMaskDataset._usable_motion_frames(item)
        examples = all_examples(item)
        ce_parts: list[np.ndarray] = []
        latent_parts: list[np.ndarray] = []
        for start_index in range(0, len(examples), args.batch_size):
            batch_examples = examples[start_index : start_index + args.batch_size]
            ce, latent = score_batch(
                model,
                collator(batch_examples),
                codebooks=codebooks,
                device=device,
                use_bf16=bool(args.bf16),
            )
            ce_parts.append(ce)
            latent_parts.append(latent)
        save_clip(
            destination,
            num_frames=usable,
            examples=examples,
            ce_values=np.concatenate(ce_parts) if ce_parts else np.empty(0),
            latent_values=np.concatenate(latent_parts) if latent_parts else np.empty(0),
        )
        completed += 1
        if completed % args.log_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"shard={args.shard_id} completed={completed}/{len(assigned)} "
                f"windows={len(examples)} elapsed={elapsed:.1f}s"
            )

    manifest = {
        "schema": "sentiavatar.step2_interval_costs.v1",
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint),
        "split_files": [str(path) for path in split_paths],
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "assigned": len(assigned),
        "completed": completed,
        "existing_skipped": skipped,
        "missing_or_bad": missing,
        "max_gap": MAX_GAP,
        "cost_arrays": {
            "ce": "canonical token CE per missing token after q0->q3 rollout",
            "hard_latent_l1": "hard decoded RVQ latent L1 per missing frame/part",
        },
        "boundary_content": "ground_truth",
        "audio_representation": audio.get("audio_representation"),
    }
    manifest_path = output_dir / f"manifest_shard_{args.shard_id:02d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()

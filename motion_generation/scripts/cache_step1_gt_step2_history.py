#!/usr/bin/env python3
"""Cache fixed-schedule GT-boundary Step 2 histories for Stage 1.

Each clip receives one deterministic feasible uniform-gap schedule. Frozen
Step 2 greedily generates every missing frame from GT left/right boundaries.
The cache stores the complete missing sequence plus the GT right endpoint for
every interval; Stage 1 later samples recent 1--15-frame suffixes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.cache_step2_interval_costs import (  # noqa: E402
    checkpoint_fingerprint,
    sha256_file,
)
from scripts.train_audio_mask_multipart import (  # noqa: E402
    load_sequences,
    read_split_file,
)
from utils.step1_adaptive_schedule import random_uniform_schedule  # noqa: E402
from utils.step1_step2_history import (  # noqa: E402
    STEP2_HISTORY_CACHE_SCHEMA,
    cache_path,
    save_step2_history_cache,
)
from utils.variable_c2f_evaluation import (  # noqa: E402
    EvalWindowRecord,
    InfillModelSpec,
    VariableGapMaskExample,
    audio_feature_for_token_frame,
    infer_window_records,
    load_audio_motion_transformer,
)


DEFAULT_STEP2_CONFIG = (
    PROJECT_DIR
    / "motion_generation/configs/"
    "audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml"
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key!r} must be a mapping")
    return dict(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2_config", type=Path, default=DEFAULT_STEP2_CONFIG)
    parser.add_argument("--step2_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--split_file",
        type=Path,
        action="append",
        required=True,
        help="Repeat for train and validation splits.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip_batch_size", type=int, default=128)
    parser.add_argument("--schedule_seed", type=int, default=42)
    parser.add_argument("--min_gap", type=int, default=3)
    parser.add_argument("--max_gap", type=int, default=15)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def build_records(
    sequences: list[Mapping[str, Any]],
    *,
    schedule_seed: int,
    min_gap: int,
    max_gap: int,
) -> tuple[
    list[EvalWindowRecord],
    list[tuple[tuple[int, ...], tuple[int, ...]]],
    list[tuple[int, int]],
]:
    records: list[EvalWindowRecord] = []
    schedules: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    prediction_index: list[tuple[int, int]] = []
    for sequence_idx, item in enumerate(sequences):
        dense = np.asarray(item["motion_tokens"], dtype=np.int64)
        name = str(item["name"]).replace("\\", "/")
        schedule = random_uniform_schedule(
            len(dense),
            min_gap=min_gap,
            max_gap=max_gap,
            seed=schedule_seed,
            epoch=0,
            name=name,
        )
        anchors = tuple(int(value) for value in schedule.anchor_times)
        gaps = tuple(
            int(right - left - 1)
            for left, right in zip(anchors[:-1], anchors[1:])
        )
        schedules.append((anchors, gaps))
        for interval_idx, (left, right, gap) in enumerate(
            zip(anchors[:-1], anchors[1:], gaps)
        ):
            if gap == 0:
                continue
            frames = dense[left : right + 1]
            example = VariableGapMaskExample(
                name=name,
                left_idx=int(left),
                right_idx=int(right),
                gap_frames=int(gap),
                motion_tokens=frames.tolist(),
                audio_features=torch.stack(
                    [
                        audio_feature_for_token_frame(item, frame)
                        for frame in range(left, right + 1)
                    ]
                ),
            )
            records.append(
                EvalWindowRecord(
                    sequence_idx=sequence_idx,
                    name=name,
                    left_idx=int(left),
                    gap_frames=int(gap),
                    example=example,
                )
            )
            prediction_index.append((sequence_idx, interval_idx))
    return records, schedules, prediction_index


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.clip_batch_size <= 0:
        raise ValueError("batch_size and clip_batch_size must be positive")
    if not 3 <= args.min_gap <= args.max_gap <= 15:
        raise ValueError("Require 3 <= min_gap <= max_gap <= 15")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")

    config_path = args.step2_config.resolve()
    config = load_yaml(config_path)
    experiment = section(config, "experiment")
    data = section(config, "data")
    audio = section(config, "audio_conditioning")
    checkpoint = (
        args.step2_checkpoint.resolve()
        if args.step2_checkpoint is not None
        else project_path(experiment["output_dir"])
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Frozen Step 2 checkpoint not found: {checkpoint}")
    configured_checkpoint = project_path(experiment["output_dir"])
    if args.step2_checkpoint is None and configured_checkpoint != checkpoint:
        raise ValueError("Step 2 source config and checkpoint disagree")

    split_paths = [path.resolve() for path in args.split_file]
    names: list[str] = []
    for split_path in split_paths:
        names.extend(read_split_file(split_path))
    names = list(
        dict.fromkeys(str(name).replace("\\", "/") for name in names)
    )
    assigned = names[args.shard_id :: args.num_shards]
    if args.max_clips is not None:
        assigned = assigned[: args.max_clips]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        name
        for name in assigned
        if args.overwrite or not cache_path(output_dir, name).is_file()
    ]
    skipped = len(assigned) - len(pending)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_audio_motion_transformer(checkpoint, device)
    if int(model.config.num_tokens_per_frame) != 16:
        raise ValueError("GT-boundary history requires 16 body tokens per frame")
    if int(model.config.codebook_size) != 512:
        raise ValueError("GT-boundary history requires 512-entry body codebooks")
    spec = InfillModelSpec(
        name="gt_boundary_step2_history",
        checkpoint=checkpoint,
        decoder="c2f",
        allowed_gaps=tuple(range(1, 16)),
    )
    fingerprint = checkpoint_fingerprint(checkpoint)
    motion_dir = project_path(data["motion_token_dir"])
    audio_dir = project_path(data["audio_feat_dir"])

    completed = missing = 0
    started = time.perf_counter()
    for chunk_start in range(0, len(pending), args.clip_batch_size):
        chunk_names = pending[
            chunk_start : chunk_start + args.clip_batch_size
        ]
        sequences, stats = load_sequences(
            chunk_names,
            motion_dir,
            audio_dir,
            codebook_size=512,
            num_tokens_per_frame=16,
            audio_fps=float(audio["audio_fps"]),
            source_motion_fps_fallback=20.0,
            motion_token_fps_override=10.0,
            motion_token_unit_length_override=2.0,
        )
        loaded_by_name = {
            str(item["name"]).replace("\\", "/"): item for item in sequences
        }
        missing_names = [
            name for name in chunk_names if name not in loaded_by_name
        ]
        if missing_names:
            missing += len(missing_names)
            for name in missing_names[:10]:
                print(f"[skip] {name}: could not load aligned Step 2 inputs")
            print("load stats:", stats)
        ordered = [
            loaded_by_name[name]
            for name in chunk_names
            if name in loaded_by_name
        ]
        records, schedules, prediction_index = build_records(
            ordered,
            schedule_seed=args.schedule_seed,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
        )
        predictions = infer_window_records(
            model,
            spec,
            records,
            batch_size=args.batch_size,
            device=device,
        )
        predicted = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in zip(prediction_index, predictions)
        }
        for sequence_idx, (item, (anchors, gaps)) in enumerate(
            zip(ordered, schedules)
        ):
            dense = np.asarray(item["motion_tokens"], dtype=np.int64)
            interval_frames = []
            for interval_idx, (right, gap) in enumerate(
                zip(anchors[1:], gaps)
            ):
                middle = (
                    predicted[(sequence_idx, interval_idx)]
                    if gap
                    else np.empty((0, 16), dtype=np.int64)
                )
                if middle.shape != (gap, 16):
                    raise ValueError(
                        f"{item['name']}: Step 2 returned {middle.shape} "
                        f"for gap={gap}"
                    )
                interval_frames.append(
                    np.concatenate([middle, dense[[right]]], axis=0)
                )
            destination = cache_path(output_dir, str(item["name"]))
            save_step2_history_cache(
                destination,
                name=str(item["name"]),
                token_frames=len(dense),
                anchor_times=anchors,
                interval_frames=interval_frames,
                schedule_seed=args.schedule_seed,
                step2_checkpoint_fingerprint=fingerprint,
            )
            completed += 1
            if completed % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"shard={args.shard_id} completed={completed} "
                    f"covered={completed + skipped}/{len(assigned)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    manifest = {
        "schema": STEP2_HISTORY_CACHE_SCHEMA,
        "step2_config": str(config_path),
        "step2_config_sha256": sha256_file(config_path),
        "step2_checkpoint": str(checkpoint),
        "step2_checkpoint_fingerprint": fingerprint,
        "split_files": [str(path) for path in split_paths],
        "motion_token_dir": str(motion_dir),
        "audio_feature_dir": str(audio_dir),
        "audio_representation": audio.get("audio_representation"),
        "schedule": {
            "distribution": "deterministic_feasible_uniform",
            "seed": int(args.schedule_seed),
            "min_gap": int(args.min_gap),
            "max_gap": int(args.max_gap),
            "epoch": 0,
        },
        "boundary_content": "ground_truth",
        "missing_content": "frozen_step2_greedy_q0_to_q3",
        "stored_interval": "all_missing_frames_plus_gt_right_endpoint",
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "assigned": len(assigned),
        "completed": completed,
        "existing_skipped": skipped,
        "missing_or_bad": missing,
    }
    manifest_path = output_dir / f"manifest_shard_{args.shard_id:02d}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote: {manifest_path}")
    if missing:
        raise SystemExit(
            f"NO-GO: {missing} assigned clips could not be cached"
        )


if __name__ == "__main__":
    main()

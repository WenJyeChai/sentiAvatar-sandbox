#!/usr/bin/env python3
"""Calibrate anchor penalties and materialize frozen-Step-2 DP schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import read_split_names  # noqa: E402
from utils.step1_adaptive_schedule import (  # noqa: E402
    calibrate_anchor_penalty,
    load_edge_costs,
    parse_curriculum,
    solve_step2_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cost_dir", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, action="append", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--calibration_max_clips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--mean_gap_tolerance", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_incomplete_manifests",
        action="store_true",
        help="Allow smoke-test caches without a complete, consistent shard manifest set.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name} must be a mapping")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_schedule_file(
    path: Path,
    *,
    names: list[str],
    schedules: list[Any],
) -> None:
    offsets = [0]
    anchors: list[int] = []
    probabilities: list[tuple[float, ...]] = []
    masks: list[bool] = []
    normal_gap_sum = 0
    normal_gap_count = 0
    tail_count = 0
    for schedule in schedules:
        for anchor in schedule.anchor_times:
            anchors.append(int(anchor))
            target = schedule.soft_targets_by_left.get(int(anchor))
            probabilities.append(
                tuple(float(value) for value in target)
                if target is not None
                else (0.0,) * 16
            )
            masks.append(target is not None)
        offsets.append(len(anchors))
        normal_gap_sum += sum(schedule.normal_gaps)
        normal_gap_count += len(schedule.normal_gaps)
        tail_count += int(schedule.tail_gap is not None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            names=np.asarray(names),
            offsets=np.asarray(offsets, dtype=np.int64),
            anchors=np.asarray(anchors, dtype=np.int32),
            gap_target_probs=np.asarray(probabilities, dtype=np.float32),
            gap_target_mask=np.asarray(masks, dtype=np.bool_),
            normal_gap_sum=np.int64(normal_gap_sum),
            normal_gap_count=np.int64(normal_gap_count),
            tail_count=np.int64(tail_count),
        )


def validate_manifests(cost_dir: Path, *, allow_incomplete: bool) -> list[dict[str, Any]]:
    paths = sorted(cost_dir.glob("manifest_shard_*.json"))
    if not paths:
        if allow_incomplete:
            return []
        raise FileNotFoundError(f"No cache shard manifests found in {cost_dir}")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_shards = {int(value["num_shards"]) for value in manifests}
    checkpoint_hashes = {
        str(value.get("checkpoint_fingerprint", value["checkpoint_config_sha256"]))
        for value in manifests
    }
    shard_ids = {int(value["shard_id"]) for value in manifests}
    errors = []
    if len(expected_shards) != 1:
        errors.append(f"inconsistent num_shards={sorted(expected_shards)}")
    if len(checkpoint_hashes) != 1:
        errors.append("shards were produced from different Step 2 checkpoints")
    if len(expected_shards) == 1:
        expected = next(iter(expected_shards))
        if shard_ids != set(range(expected)):
            errors.append(
                f"shard ids are {sorted(shard_ids)}, expected 0..{expected - 1}"
            )
    for value in manifests:
        if int(value.get("missing_or_bad", 0)):
            errors.append(
                f"shard {value['shard_id']} has missing_or_bad={value['missing_or_bad']}"
            )
        covered = int(value.get("completed", 0)) + int(
            value.get("existing_skipped", 0)
        )
        if covered != int(value.get("assigned", -1)):
            errors.append(
                f"shard {value['shard_id']} covered {covered}/"
                f"{value.get('assigned')} assigned clips"
            )
    if errors and not allow_incomplete:
        raise ValueError("Invalid interval-cost cache manifests: " + "; ".join(errors))
    for error in errors:
        print(f"[manifest warning] {error}")
    return manifests


def main() -> None:
    args = parse_args()
    if args.ce_weight < 0 or args.latent_weight < 0:
        raise ValueError("Cost weights must be non-negative")
    if args.ce_weight + args.latent_weight <= 0:
        raise ValueError("At least one cost weight must be positive")
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    training = section(config, "training")
    adaptive = section(config, "adaptive_gap")
    if not bool(adaptive.get("enabled", False)):
        raise ValueError("The Step 1 config does not enable adaptive_gap")
    phases = parse_curriculum(
        adaptive.get("phases", []),
        num_epochs=int(training["num_train_epochs"]),
    )
    names: list[str] = []
    split_paths = [path.resolve() for path in args.split_file]
    for path in split_paths:
        names.extend(read_split_names(path))
    names = list(dict.fromkeys(names))
    cost_dir = args.cost_dir.resolve()
    manifests = validate_manifests(
        cost_dir, allow_incomplete=bool(args.allow_incomplete_manifests)
    )
    output_json = args.output_json.resolve()
    if output_json.exists() and not args.overwrite:
        raise FileExistsError(f"Calibration already exists: {output_json}")

    # Stable random calibration subset; schedule materialization still covers
    # every requested train/eval clip.
    rng = np.random.default_rng(args.seed)
    calibration_indices = np.arange(len(names))
    rng.shuffle(calibration_indices)
    if args.calibration_max_clips > 0:
        calibration_indices = calibration_indices[: args.calibration_max_clips]
    calibration_names = [names[int(index)] for index in calibration_indices]
    calibration_costs = [
        load_edge_costs(
            cost_dir,
            name,
            ce_weight=args.ce_weight,
            latent_weight=args.latent_weight,
        )
        for name in calibration_names
    ]
    phase_records = []
    for phase_index, phase in enumerate(phases):
        if phase.mode != "step2_dp":
            continue
        assert phase.target_mean_gap is not None
        penalty, calibration_mean = calibrate_anchor_penalty(
            calibration_costs,
            min_gap=phase.min_gap,
            max_gap=phase.max_gap,
            target_mean_gap=phase.target_mean_gap,
            tolerance=args.mean_gap_tolerance,
        )
        schedules = []
        for clip_index, name in enumerate(names, start=1):
            costs = load_edge_costs(
                cost_dir,
                name,
                ce_weight=args.ce_weight,
                latent_weight=args.latent_weight,
            )
            schedules.append(
                solve_step2_schedule(
                    costs,
                    min_gap=phase.min_gap,
                    max_gap=phase.max_gap,
                    anchor_penalty=penalty,
                    temperature=phase.temperature,
                )
            )
            if clip_index % 1000 == 0:
                print(
                    f"phase={phase_index} materialized={clip_index}/{len(names)}"
                )
        schedule_path = output_json.parent / (
            f"{output_json.stem}_phase_{phase_index:02d}.npz"
        )
        save_schedule_file(schedule_path, names=names, schedules=schedules)
        all_gaps = [
            gap for schedule in schedules for gap in schedule.normal_gaps
        ]
        record = {
            "phase_index": phase_index,
            "start_epoch": phase.start_epoch,
            "end_epoch": phase.end_epoch,
            "min_gap": phase.min_gap,
            "max_gap": phase.max_gap,
            "target_mean_gap": phase.target_mean_gap,
            "calibrated_anchor_penalty": penalty,
            "calibration_mean_gap": calibration_mean,
            "materialized_mean_gap": float(np.mean(all_gaps)),
            "temperature": phase.temperature,
            "schedule_loss_weight_start": phase.schedule_loss_weight_start,
            "schedule_loss_weight_end": phase.schedule_loss_weight_end,
            "schedule_file": schedule_path.name,
            "schedule_file_sha256": sha256_file(schedule_path),
        }
        phase_records.append(record)
        print(json.dumps(record, indent=2))

    payload = {
        "schema": "sentiavatar.step1_adaptive_gap_calibration.v1",
        "config": str(config_path),
        "cost_dir": str(cost_dir),
        "cost_manifests": [
            {
                "shard_id": int(value["shard_id"]),
                "checkpoint": str(value["checkpoint"]),
                "checkpoint_config_sha256": str(
                    value["checkpoint_config_sha256"]
                ),
                "checkpoint_fingerprint": str(
                    value.get(
                        "checkpoint_fingerprint",
                        value["checkpoint_config_sha256"],
                    )
                ),
            }
            for value in manifests
        ],
        "split_files": [str(path) for path in split_paths],
        "clips": len(names),
        "calibration_clips": len(calibration_names),
        "seed": args.seed,
        "cost": {
            "ce_weight": args.ce_weight,
            "hard_latent_l1_weight": args.latent_weight,
            "reduction": "gap * (ce_weight * mean_ce + latent_weight * mean_l1)",
            "boundary_content": "ground_truth",
        },
        "tail_policy": "gaps 0--2 only when landing exactly on EOS; excluded from target mean",
        "phases": phase_records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()

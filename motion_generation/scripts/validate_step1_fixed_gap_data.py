#!/usr/bin/env python3
"""Validate every aligned Phase 1 training record before launching DDP."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    build_dataset,
    data_config_from_config,
    load_config,
    load_neutral_seed,
    resolve_data_paths,
    section,
    validate_adaptive_gap_config,
    validate_paths,
)
from models.step1_mimi_planner import load_text_map, read_split_names  # noqa: E402
from utils.adaptive_anchor_tokens import ensure_step1_special_tokens, gap_from_anchor_times  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 1 audio/motion/text alignment")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "motion_generation" / "configs" / "step1_multipart_fixed_gap3.yaml",
    )
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_eval_clips", type=int, default=None)
    parser.add_argument("--max_reported_errors", type=int, default=20)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def percentile_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {
        "count": len(values),
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def validate_split(dataset, split_name: str, max_reported_errors: int) -> dict:
    sequence_lengths: list[int] = []
    anchor_counts: list[int] = []
    audio_counts: list[int] = []
    target_counts: list[int] = []
    gap_counts = Counter()
    annotation_patterns = Counter()
    errors = []
    for index, name in enumerate(dataset.names):
        try:
            item = dataset[index]
            sequence_lengths.append(len(item["input_ids"]))
            anchor_counts.append(len(item["anchor_times"]))
            audio_counts.append(item["audio_boundaries"][-1])
            target_counts.append(sum(slot >= 0 for slot in item["target_slots"]))
            annotation_patterns[item.get("annotation_pattern", "unknown")] += 1
            for left, right in zip(item["anchor_times"], item["anchor_times"][1:]):
                gap_counts[gap_from_anchor_times(left, right)] += 1
        except Exception as exc:  # collect multiple data failures in one audit
            if len(errors) < max_reported_errors:
                errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        if (index + 1) % 1_000 == 0:
            print(f"{split_name}: {index + 1}/{len(dataset)} checked, errors={len(errors)}")
    return {
        "split": split_name,
        "assigned_clips": len(dataset),
        "valid_clips": len(sequence_lengths),
        "error_count_at_least": len(dataset) - len(sequence_lengths),
        "reported_errors": errors,
        "sequence_lengths": percentile_summary(sequence_lengths),
        "anchor_counts": percentile_summary(anchor_counts),
        "audio_frame_counts": percentile_summary(audio_counts),
        "mimi_frame_counts": percentile_summary(audio_counts),
        "annotation_patterns": dict(sorted(annotation_patterns.items())),
        "supervised_token_counts": percentile_summary(target_counts),
        "gap_counts": {str(key): value for key, value in sorted(gap_counts.items())},
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    paths = resolve_data_paths(config)
    validate_paths(paths, resume=None)
    data_config = data_config_from_config(config)
    training = section(config, "training")
    adaptive_gap = validate_adaptive_gap_config(
        config,
        num_epochs=int(training.get("num_train_epochs", 10)),
    )
    tokenizer = AutoTokenizer.from_pretrained(paths["base_model"], local_files_only=True)
    added = ensure_step1_special_tokens(
        tokenizer,
        include_structured_text=data_config.get("text_serialization") == "structured_fields",
    )
    print(f"Tokenizer controls added in-memory: {len(added)}")
    text_map = load_text_map(paths["text_json"])
    neutral_seed = load_neutral_seed(data_config.get("neutral_seed_json"))
    train_names = read_split_names(paths["train_split"])
    eval_names = read_split_names(paths["eval_split"])
    if args.max_train_clips is not None:
        train_names = train_names[: args.max_train_clips]
    if args.max_eval_clips is not None:
        eval_names = eval_names[: args.max_eval_clips]

    train_dataset = build_dataset(
        train_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=neutral_seed,
        training=True,
        adaptive_gap=adaptive_gap,
    )
    eval_dataset = build_dataset(
        eval_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=neutral_seed,
        training=False,
        adaptive_gap=adaptive_gap,
    )
    report = {"config": str(args.config.resolve())}
    if adaptive_gap["enabled"]:
        report["adaptive_phases"] = []
        for phase_index, phase in enumerate(adaptive_gap["phases"]):
            epoch = phase.start_epoch - 1
            train_dataset.set_epoch(epoch)
            eval_dataset.set_epoch(epoch)
            phase_report = {
                "phase_index": phase_index,
                "epoch": phase.start_epoch,
                "mode": phase.mode,
                "gap_range": [phase.min_gap, phase.max_gap],
                "target_mean_gap": phase.target_mean_gap,
                "train": validate_split(
                    train_dataset,
                    f"train_phase_{phase_index}",
                    args.max_reported_errors,
                ),
                "eval": validate_split(
                    eval_dataset,
                    f"eval_phase_{phase_index}",
                    args.max_reported_errors,
                ),
            }
            report["adaptive_phases"].append(phase_report)
    else:
        report["train"] = validate_split(
            train_dataset, "train", args.max_reported_errors
        )
        report["eval"] = validate_split(
            eval_dataset, "eval", args.max_reported_errors
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        output = args.output_json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote:", output)
    if adaptive_gap["enabled"]:
        failures = sum(
            phase["train"]["error_count_at_least"]
            + phase["eval"]["error_count_at_least"]
            for phase in report["adaptive_phases"]
        )
    else:
        failures = (
            report["train"]["error_count_at_least"]
            + report["eval"]["error_count_at_least"]
        )
    if failures:
        raise SystemExit(f"NO-GO: {failures} Phase 1 data records failed validation")
    print("GO: every selected Phase 1 record passed serialization and alignment checks")


if __name__ == "__main__":
    main()

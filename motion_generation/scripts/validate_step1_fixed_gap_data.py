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
    audio_contract_from_config,
    build_dataset,
    data_config_from_config,
    load_config,
    load_neutral_seed,
    resolve_data_paths,
    section,
    validate_adaptive_gap_config,
    validate_frozen_step2_guidance_config,
    validate_paths,
    validate_provided_gap_config,
    validate_step2_history_config,
)
from models.step1_mimi_planner import load_text_map, read_split_names  # noqa: E402
from utils.adaptive_anchor_tokens import (  # noqa: E402
    ensure_nano_audio_tokens,
    ensure_step1_special_tokens,
    gap_from_anchor_times,
)


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
    audio_metadata_frame_counts: list[int] = []
    audio_vocabulary_token_position_counts: list[int] = []
    target_counts: list[int] = []
    prefix_counts: list[int] = []
    gap_counts = Counter()
    annotation_patterns = Counter()
    step2_guidance_gaps = Counter()
    step2_history_interval_counts: list[int] = []
    step2_history_frame_counts: list[int] = []
    step2_history_available_frame_counts: list[int] = []
    step2_history_corrupted_token_counts: list[int] = []
    interval_audio_frame_counts: list[int] = []
    supervised_gap_counts: list[int] = []
    errors = []
    audio_input_representation = str(
        getattr(dataset, "audio_input_representation", "fused_frame")
    )
    audio_token_ids = {
        int(token_id)
        for row in getattr(dataset, "audio_token_ids", ())
        for token_id in row
    }
    for index, name in enumerate(dataset.names):
        try:
            item = dataset[index]
            item_audio_codes = np.asarray(item["audio_codes"], dtype=np.int64)
            if item_audio_codes.ndim != 2:
                raise ValueError(
                    f"Serialized audio_codes must be [L,K], got {item_audio_codes.shape}"
                )
            metadata_audio_frames = int(
                np.any(item_audio_codes >= 0, axis=-1).sum()
            )
            vocabulary_audio_positions = sum(
                int(token_id) in audio_token_ids for token_id in item["input_ids"]
            )
            expected_audio_frames = int(item["audio_boundaries"][-1])
            if metadata_audio_frames != expected_audio_frames:
                raise ValueError(
                    "Serialized audio metadata frame count does not match the "
                    f"aligned audio length: {metadata_audio_frames} != "
                    f"{expected_audio_frames}"
                )
            if audio_input_representation == "ordinary_tokens":
                expected_audio_positions = (
                    expected_audio_frames * len(dataset.audio_codebooks_used)
                )
                if vocabulary_audio_positions != expected_audio_positions:
                    raise ValueError(
                        "Ordinary audio-token position count does not equal "
                        "frames times configured codebooks: "
                        f"{vocabulary_audio_positions} != {expected_audio_positions}"
                    )
            elif vocabulary_audio_positions:
                raise ValueError(
                    "Fused-frame serialization unexpectedly contains ordinary "
                    f"audio vocabulary tokens: {vocabulary_audio_positions}"
                )
            if getattr(dataset, "sequence_layout", "") == "interval_audio_isolated":
                if any(bool(value) for value in item["gap_target_mask"]):
                    raise ValueError(
                        "Stage 1 interval layout must not supervise supplied gaps"
                    )
                if sum(
                    bool(value)
                    for value in item["planner_gap_context_mask"]
                ) != len(item["anchor_times"]) - 1:
                    raise ValueError(
                        "Every interval must retain exactly one supplied gap "
                        "as future context"
                    )
                for anchor_group, (left, right) in enumerate(
                    zip(
                        item["audio_boundaries"][:-1],
                        item["audio_boundaries"][1:],
                    )
                ):
                    observed = sum(
                        int(group) == anchor_group
                        and bool(np.any(codes >= 0))
                        for group, codes in zip(
                            item["audio_anchor_ids"],
                            item_audio_codes,
                        )
                    )
                    expected = int(right) - int(left)
                    if observed != expected:
                        raise ValueError(
                            f"Interval {anchor_group} contains {observed} "
                            f"audio frames, expected {expected}"
                        )
                    interval_audio_frame_counts.append(observed)
                    target_segments = {
                        int(segment)
                        for segment, target_group in zip(
                            item["planner_segment_ids"],
                            item["target_anchor_ids"],
                        )
                        if int(target_group) == anchor_group
                    }
                    if target_segments != {anchor_group + 1}:
                        raise ValueError(
                            f"Anchor group {anchor_group} uses segment IDs "
                            f"{sorted(target_segments)}"
                        )
            item_gaps = [
                gap_from_anchor_times(left, right)
                for left, right in zip(
                    item["anchor_times"],
                    item["anchor_times"][1:],
                )
            ]
            supervised_tokens = sum(
                slot >= 0 for slot in item["target_slots"]
            )
            expected_supervised_tokens = (
                max(0, len(item["anchor_times"]) - 1) * 16
            )
            if supervised_tokens != expected_supervised_tokens:
                raise ValueError(
                    "Only the planned 16-ID anchors may receive CE: "
                    f"{supervised_tokens} supervised positions != "
                    f"{expected_supervised_tokens}"
                )
            guidance_gap = None
            if "step2_guidance_gap" in item:
                guidance_gap = int(item["step2_guidance_gap"])
                expected_frames = guidance_gap + 2
                if len(item["step2_guidance_motion_tokens"]) != expected_frames:
                    raise ValueError("Step 2 guidance motion window is not gap+2")
                if item["step2_guidance_audio_features"].shape[0] != expected_frames:
                    raise ValueError("Step 2 guidance audio window is not gap+2")
            # Commit statistics only after every field, including the optional
            # online Step 2 window, has passed validation.
            sequence_lengths.append(len(item["input_ids"]))
            anchor_counts.append(len(item["anchor_times"]))
            audio_counts.append(item["audio_boundaries"][-1])
            audio_metadata_frame_counts.append(metadata_audio_frames)
            audio_vocabulary_token_position_counts.append(
                vocabulary_audio_positions
            )
            target_counts.append(supervised_tokens)
            prefix_counts.append(
                sum(bool(value) for value in item["bidirectional_prefix_mask"])
            )
            annotation_patterns[item.get("annotation_pattern", "unknown")] += 1
            gap_counts.update(item_gaps)
            if guidance_gap is not None:
                step2_guidance_gaps[guidance_gap] += 1
            step2_history_interval_counts.append(
                int(item.get("step2_history_intervals", 0))
            )
            step2_history_frame_counts.append(
                int(item.get("step2_history_frames", 0))
            )
            step2_history_available_frame_counts.append(
                int(item.get("step2_history_available_frames", 0))
            )
            step2_history_corrupted_token_counts.append(
                int(item.get("step2_history_corrupted_tokens", 0))
            )
            supervised_gap_counts.append(
                sum(bool(value) for value in item["gap_target_mask"])
            )
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
        "audio_metadata_frame_counts": percentile_summary(
            audio_metadata_frame_counts
        ),
        "audio_vocabulary_token_position_counts": percentile_summary(
            audio_vocabulary_token_position_counts
        ),
        "annotation_patterns": dict(sorted(annotation_patterns.items())),
        "supervised_token_counts": percentile_summary(target_counts),
        "bidirectional_prefix_token_counts": percentile_summary(prefix_counts),
        "gap_counts": {str(key): value for key, value in sorted(gap_counts.items())},
        "step2_guidance_gap_counts": {
            str(key): value for key, value in sorted(step2_guidance_gaps.items())
        },
        "step2_history_intervals": percentile_summary(
            step2_history_interval_counts
        ),
        "step2_history_frames": percentile_summary(
            step2_history_frame_counts
        ),
        "step2_history_available_frames": percentile_summary(
            step2_history_available_frame_counts
        ),
        "step2_history_corrupted_tokens": percentile_summary(
            step2_history_corrupted_token_counts
        ),
        "target_interval_audio_frames": percentile_summary(
            interval_audio_frame_counts
        ),
        "supervised_gap_tokens": percentile_summary(
            supervised_gap_counts
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    paths = resolve_data_paths(config)
    validate_paths(paths, resume=None)
    data_config = data_config_from_config(config)
    audio_contract = audio_contract_from_config(config)
    training = section(config, "training")
    adaptive_gap = validate_adaptive_gap_config(
        config,
        num_epochs=int(training.get("num_train_epochs", 10)),
    )
    provided_gap = validate_provided_gap_config(config)
    step2_history = validate_step2_history_config(config)
    if adaptive_gap["enabled"] and provided_gap["enabled"]:
        raise ValueError(
            "adaptive_gap and provided_gap_training are mutually exclusive"
        )
    frozen_step2_guidance = validate_frozen_step2_guidance_config(config)
    tokenizer = AutoTokenizer.from_pretrained(paths["base_model"], local_files_only=True)
    added = ensure_step1_special_tokens(
        tokenizer,
        include_structured_text=data_config.get("text_serialization") == "structured_fields",
        include_step2_history=bool(step2_history["register_tokens"]),
    )
    print(f"Tokenizer controls added in-memory: {len(added)}")
    added_audio = []
    if audio_contract["input_representation"] == "ordinary_tokens":
        added_audio = ensure_nano_audio_tokens(tokenizer)
        print(f"Ordinary Nano audio tokens added in-memory: {len(added_audio)}")
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
        provided_gap=provided_gap,
        frozen_step2_guidance=frozen_step2_guidance,
        step2_history=step2_history,
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
        provided_gap=provided_gap,
        frozen_step2_guidance=frozen_step2_guidance,
        step2_history=step2_history,
    )
    report = {
        "config": str(args.config.resolve()),
        "audio_input_representation": audio_contract["input_representation"],
        "audio_vocabulary_token_count": (
            len(audio_contract["codebooks_used"])
            * int(audio_contract["cardinality"])
            if audio_contract["input_representation"] == "ordinary_tokens"
            else 0
        ),
        "audio_vocabulary_tokens_added_in_memory": len(added_audio),
        "provided_gap_training": json.loads(
            json.dumps(provided_gap, default=str)
        ),
        "step2_history": json.loads(
            json.dumps(step2_history, default=str)
        ),
    }
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

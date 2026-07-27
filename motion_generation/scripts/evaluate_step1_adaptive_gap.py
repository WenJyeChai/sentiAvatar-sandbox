#!/usr/bin/env python3
"""Evaluate learned Step 1 gaps, generated anchors, and fixed-rate controls.

The protocol has three distinct layers:

1. teacher-forced final-phase oracle likelihood;
2. free adaptive scheduling with GT or generated anchor history;
3. fixed-gap and frozen-Step-2-DP controls evaluated at the same token rate.

This runner writes sparse rollout caches for downstream anchor-substitution and
full frozen-Step-2 motion evaluation.  Rollouts use validation token length only
for exact EOS stopping; EOS-clipped decisions are reported explicitly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    Step1PlannerCollator,
    canonical_data_path,
    load_motion_tokens,
    load_text_map,
    read_split_names,
)
from scripts.evaluate_step1_multipart_comparison import (  # noqa: E402
    deterministic_subset,
    load_planner,
    load_source_config,
)
from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    build_dataset,
    data_config_from_config,
    evaluate,
    load_neutral_seed,
    resolve_data_paths,
    section,
    validate_adaptive_gap_config,
)
from utils.adaptive_anchor_tokens import MOTION_START_TOKEN  # noqa: E402
from utils.inference_math import configure_strict_inference_math  # noqa: E402
from utils.step1_adaptive_evaluation import (  # noqa: E402
    AdaptiveRolloutExample,
    AdaptiveRolloutResult,
    initial_prefix_from_serialized_item,
    make_ground_truth_result,
    rollout_policy_batch,
    write_adaptive_rollout_cache,
)
from utils.step1_adaptive_schedule import load_edge_costs  # noqa: E402
from utils.step1_planner_evaluation import summarize_slot_metrics  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_int_list(value: str, *, label: str) -> tuple[int, ...]:
    result = tuple(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate values")
    if any(not 3 <= item <= 15 for item in result):
        raise ValueError(f"{label} values must lie in [3,15]")
    return result


def parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
        label, raw_path = (part.strip() for part in value.split("=", 1))
        if not label or not raw_path:
            raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
        if label in result:
            raise ValueError(f"Duplicate checkpoint label {label!r}")
        path = project_path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing checkpoint {label}: {path}")
        result[label] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR
        / "motion_generation"
        / "outputs"
        / "step1_adaptive_gap_evaluation",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--teacher_max_clips",
        type=int,
        default=0,
        help="0 evaluates the complete validation split.",
    )
    parser.add_argument(
        "--rollout_max_clips",
        type=int,
        default=128,
        help="0 evaluates the complete validation split.",
    )
    parser.add_argument("--teacher_batch_size", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument(
        "--fixed_gaps",
        default="3,5,7,9,11,15",
        help="GT-anchor fixed schedules used for the quality-rate frontier.",
    )
    parser.add_argument(
        "--generated_fixed_gaps",
        default="7",
        help="Fixed schedules that also generate anchor content.",
    )
    parser.add_argument(
        "--cost_dir",
        type=Path,
        default=None,
        help="Override the interval-cost directory recorded by calibration.",
    )
    parser.add_argument("--skip_teacher_forced", action="store_true")
    parser.add_argument("--skip_adaptive_gt_history", action="store_true")
    parser.add_argument("--skip_generated_history", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def final_phase_contract(
    source_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    training = section(source_config, "training")
    adaptive = validate_adaptive_gap_config(
        source_config, num_epochs=int(training["num_train_epochs"])
    )
    if not adaptive["enabled"]:
        raise ValueError("Checkpoint source config does not enable adaptive_gap")
    calibration_path = Path(adaptive["calibration_json"]).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    phase_index = len(adaptive["phases"]) - 1
    phase = next(
        (
            dict(value)
            for value in calibration["phases"]
            if int(value["phase_index"]) == phase_index
        ),
        None,
    )
    if phase is None:
        raise ValueError(
            f"Calibration does not contain final curriculum phase {phase_index}"
        )
    if int(phase["min_gap"]) != 3 or int(phase["max_gap"]) != 15:
        raise ValueError("Final evaluation phase must cover normal gaps 3--15")
    return adaptive, calibration, calibration_path


def build_final_phase_dataset(
    checkpoint: Path,
    source_config: Mapping[str, Any],
    names: Sequence[str],
):
    paths = resolve_data_paths(source_config)
    training = section(source_config, "training")
    data_config = data_config_from_config(source_config)
    data_config["random_seed"] = int(training.get("seed", 42))
    adaptive, calibration, calibration_path = final_phase_contract(source_config)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(
        list(names),
        tokenizer=tokenizer,
        paths=paths,
        text_map=load_text_map(paths["text_json"]),
        data_config=data_config,
        neutral_seed=load_neutral_seed(data_config.get("neutral_seed_json")),
        training=False,
        adaptive_gap=adaptive,
    )
    # Dataset epochs are zero-based; the final completed epoch is N-1.
    dataset.set_epoch(int(training["num_train_epochs"]) - 1)
    return dataset, tokenizer, paths, adaptive, calibration, calibration_path


def build_rollout_examples(
    dataset,
    tokenizer,
    paths: Mapping[str, Path],
) -> list[AdaptiveRolloutExample]:
    motion_start_id = int(tokenizer.convert_tokens_to_ids(MOTION_START_TOKEN))
    examples = []
    for index, name in enumerate(dataset.names):
        item = dataset[index]
        prefix, audio_codes = initial_prefix_from_serialized_item(
            item, motion_start_id=motion_start_id
        )
        prefix_length = len(prefix)
        serialized_audio = np.asarray(item["audio_codes"], dtype=np.int64)
        if serialized_audio.ndim == 1:
            serialized_audio = serialized_audio[:, None]
        prefix_audio_codes = serialized_audio[:prefix_length].copy()
        bidirectional_prefix_mask = np.asarray(
            item.get(
                "bidirectional_prefix_mask",
                [0] * len(item["input_ids"]),
            ),
            dtype=bool,
        )[:prefix_length].copy()
        dense, _ = load_motion_tokens(
            canonical_data_path(paths["motion_token_dir"], name, ".json"),
            require_causal=True,
        )
        examples.append(
            AdaptiveRolloutExample(
                name=name,
                initial_input_ids=prefix,
                audio_codes=audio_codes,
                dense_motion_tokens=np.asarray(dense, dtype=np.int64),
                oracle_anchor_times=tuple(int(value) for value in item["anchor_times"]),
                initial_audio_codes=prefix_audio_codes,
                bidirectional_prefix_mask=bidirectional_prefix_mask,
                audio_fps=float(dataset.audio_frame_rate),
                motion_fps=10.0,
            )
        )
    return examples


def phase_cost_contract(
    calibration: Mapping[str, Any],
    *,
    cost_dir_override: Path | None,
) -> tuple[Path, float, float, float]:
    cost_dir = (
        project_path(cost_dir_override)
        if cost_dir_override is not None
        else Path(str(calibration["cost_dir"])).resolve()
    )
    if not cost_dir.is_dir():
        raise FileNotFoundError(f"Missing Step 2 interval-cost directory: {cost_dir}")
    cost = calibration["cost"]
    final_phase = max(
        calibration["phases"], key=lambda value: int(value["phase_index"])
    )
    return (
        cost_dir,
        float(cost["ce_weight"]),
        float(cost["hard_latent_l1_weight"]),
        float(final_phase["calibrated_anchor_penalty"]),
    )


def schedule_objective(
    result: AdaptiveRolloutResult,
    edge_costs: np.ndarray,
    *,
    anchor_penalty: float,
) -> tuple[float, float]:
    risk = 0.0
    for left, gap in zip(result.anchor_times[:-1], result.executed_gaps):
        value = float(edge_costs[int(left), int(gap)])
        if not math.isfinite(value) and int(gap) == 0:
            value = 0.0
        if not math.isfinite(value):
            raise ValueError(
                f"{result.name}: frozen Step 2 has no cost for left={left}, gap={gap}"
            )
        risk += value
    objective = risk + len(result.executed_gaps) * float(anchor_penalty)
    return risk, objective


def summarize_condition(
    *,
    condition: str,
    checkpoint: str,
    results: Sequence[AdaptiveRolloutResult],
    edge_costs_by_name: Mapping[str, np.ndarray],
    oracle_objective_by_name: Mapping[str, float],
    oracle_times_by_name: Mapping[str, tuple[int, ...]],
    anchor_penalty: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not results:
        raise ValueError(f"{condition}: no results")
    per_clip = []
    histogram: dict[tuple[str, int], int] = {}
    normal_gaps: list[int] = []
    all_gaps: list[int] = []
    confidences: list[float] = []
    entropies: list[float] = []
    for result in results:
        costs = edge_costs_by_name[result.name]
        risk, objective = schedule_objective(
            result, costs, anchor_penalty=anchor_penalty
        )
        oracle_objective = oracle_objective_by_name[result.name]
        selected = set(result.anchor_times[1:-1])
        oracle = set(oracle_times_by_name[result.name][1:-1])
        precision = len(selected & oracle) / max(1, len(selected))
        recall = len(selected & oracle) / max(1, len(oracle))
        gaps = [int(value) for value in result.executed_gaps]
        normals = [value for value in gaps if value >= 3]
        normal_gaps.extend(normals)
        all_gaps.extend(gaps)
        for gap in gaps:
            histogram[("executed", gap)] = histogram.get(("executed", gap), 0) + 1
        for gap in result.predicted_gap_decisions:
            histogram[("predicted", int(gap))] = (
                histogram.get(("predicted", int(gap)), 0) + 1
            )
        finite_confidence = result.gap_confidence[
            np.isfinite(result.gap_confidence)
        ]
        finite_entropy = result.gap_entropy[np.isfinite(result.gap_entropy)]
        confidences.extend(finite_confidence.tolist())
        entropies.extend(finite_entropy.tolist())
        per_clip.append(
            {
                "condition": condition,
                "checkpoint": checkpoint,
                "name": result.name,
                "token_frames": result.token_length,
                "intervals": len(gaps),
                "normal_intervals": len(normals),
                "mean_normal_gap": (
                    float(np.mean(normals)) if normals else math.nan
                ),
                "anchor_fraction_including_seed": (
                    len(result.anchor_times) / result.token_length
                ),
                "eos_clipped_decisions": result.eos_clipped_decisions,
                "eos_clipped_fraction": (
                    result.eos_clipped_decisions
                    / max(1, len(result.predicted_gap_decisions))
                ),
                "cached_gt_boundary_step2_risk": risk,
                "cached_gt_boundary_step2_risk_per_frame": (
                    risk / result.token_length
                ),
                "rate_regularized_objective": objective,
                "objective_per_frame": objective / result.token_length,
                "oracle_objective": oracle_objective,
                "oracle_regret": objective - oracle_objective,
                "oracle_regret_per_frame": (
                    objective - oracle_objective
                )
                / result.token_length,
                "exact_internal_anchor_precision_vs_oracle": precision,
                "exact_internal_anchor_recall_vs_oracle": recall,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
    per_clip_frame = pd.DataFrame(per_clip)
    intervals = int(per_clip_frame["intervals"].sum())
    clipped = int(per_clip_frame["eos_clipped_decisions"].sum())
    summary = {
        "condition": condition,
        "checkpoint": checkpoint,
        "policy": results[0].policy,
        "anchor_history": results[0].anchor_history,
        "clips": len(results),
        "intervals": intervals,
        "mean_anchors_including_seed": float(
            np.mean([len(result.anchor_times) for result in results])
        ),
        "mean_anchor_fraction_including_seed": float(
            per_clip_frame["anchor_fraction_including_seed"].mean()
        ),
        "mean_normal_gap": float(np.mean(normal_gaps)) if normal_gaps else math.nan,
        "median_normal_gap": (
            float(np.median(normal_gaps)) if normal_gaps else math.nan
        ),
        "p95_normal_gap": (
            float(np.percentile(normal_gaps, 95)) if normal_gaps else math.nan
        ),
        "tail_interval_fraction": (
            sum(value <= 2 for value in all_gaps) / max(1, len(all_gaps))
        ),
        "eos_clipped_decisions": clipped,
        "eos_clipped_fraction": clipped
        / max(
            1,
            sum(len(result.predicted_gap_decisions) for result in results),
        ),
        "gap_confidence": (
            float(np.mean(confidences)) if confidences else math.nan
        ),
        "gap_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "cached_gt_boundary_step2_risk_per_frame": float(
            per_clip_frame["cached_gt_boundary_step2_risk"].sum()
            / per_clip_frame["token_frames"].sum()
        ),
        "objective_per_frame": float(
            per_clip_frame["rate_regularized_objective"].sum()
            / per_clip_frame["token_frames"].sum()
        ),
        "oracle_regret_per_frame": float(
            per_clip_frame["oracle_regret"].sum()
            / per_clip_frame["token_frames"].sum()
        ),
        "exact_internal_anchor_precision_vs_oracle": float(
            per_clip_frame["exact_internal_anchor_precision_vs_oracle"].mean()
        ),
        "exact_internal_anchor_recall_vs_oracle": float(
            per_clip_frame["exact_internal_anchor_recall_vs_oracle"].mean()
        ),
        "elapsed_seconds": float(
            sum(result.elapsed_seconds for result in results)
        ),
    }
    histogram_rows = [
        {
            "condition": condition,
            "checkpoint": checkpoint,
            "kind": kind,
            "gap": gap,
            "count": count,
            "fraction": count
            / max(
                1,
                sum(
                    candidate
                    for (candidate_kind, _), candidate in histogram.items()
                    if candidate_kind == kind
                ),
            ),
        }
        for (kind, gap), count in sorted(histogram.items())
    ]
    return summary, per_clip, histogram_rows


def summarize_anchor_tokens(
    condition: str,
    checkpoint: str,
    results: Sequence[AdaptiveRolloutResult],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.concatenate([result.target_anchors for result in results], axis=0)
    predictions = np.concatenate([result.anchors for result in results], axis=0)
    summary, slots = summarize_slot_metrics(labels, predictions)
    confidence = np.concatenate(
        [result.anchor_confidence for result in results], axis=0
    )
    entropy = np.concatenate(
        [result.anchor_entropy for result in results], axis=0
    )
    summary.update(
        {
            "condition": condition,
            "checkpoint": checkpoint,
            "clips": len(results),
            "mean_confidence": (
                float(np.nanmean(confidence))
                if bool(np.isfinite(confidence).any())
                else math.nan
            ),
            "mean_entropy": (
                float(np.nanmean(entropy))
                if bool(np.isfinite(entropy).any())
                else math.nan
            ),
        }
    )
    return summary, [
        {"condition": condition, "checkpoint": checkpoint, **row}
        for row in slots
    ]


def rollout_in_batches(
    model,
    tokenizer,
    examples: Sequence[AdaptiveRolloutExample],
    *,
    policy: str,
    anchor_history: str,
    fixed_gap: int | None,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
    progress_label: str,
) -> list[AdaptiveRolloutResult]:
    results: list[AdaptiveRolloutResult] = []
    for start in range(0, len(examples), batch_size):
        results.extend(
            rollout_policy_batch(
                model,
                tokenizer,
                examples[start : start + batch_size],
                policy=policy,
                anchor_history=anchor_history,
                fixed_gap=fixed_gap,
                device=device,
                use_bf16=use_bf16,
            )
        )
        if len(results) % max(batch_size, 80) == 0 or len(results) == len(examples):
            print(
                f"{progress_label}: {len(results)}/{len(examples)} clips",
                flush=True,
            )
    return results


def main() -> None:
    args = parse_args()
    if args.teacher_max_clips < 0 or args.rollout_max_clips < 0:
        raise ValueError("clip limits must be non-negative")
    if args.teacher_batch_size < 1 or args.rollout_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    checkpoints = parse_checkpoints(args.checkpoint)
    fixed_gaps = parse_int_list(args.fixed_gaps, label="fixed_gaps")
    generated_fixed_gaps = parse_int_list(
        args.generated_fixed_gaps, label="generated_fixed_gaps"
    )
    if not set(generated_fixed_gaps).issubset(fixed_gaps):
        raise ValueError("generated_fixed_gaps must be a subset of fixed_gaps")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "rollout_cache"
    device = torch.device(args.device)
    use_bf16 = bool(not args.no_bf16 and device.type == "cuda")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a CUDA device without bf16 support")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    math_mode = configure_strict_inference_math(device)

    source_configs = {
        label: load_source_config(path) for label, path in checkpoints.items()
    }
    first_label = next(iter(checkpoints))
    first_config = source_configs[first_label]
    first_paths = resolve_data_paths(first_config)
    validation_names = read_split_names(first_paths["eval_split"])
    for label, config in source_configs.items():
        candidate_paths = resolve_data_paths(config)
        if read_split_names(candidate_paths["eval_split"]) != validation_names:
            raise ValueError(f"{label} uses a different validation split")
        if candidate_paths["motion_token_dir"] != first_paths["motion_token_dir"]:
            raise ValueError(f"{label} uses a different motion-token export")
    teacher_names = deterministic_subset(
        validation_names, args.teacher_max_clips, args.subset_seed
    )
    rollout_names = deterministic_subset(
        validation_names, args.rollout_max_clips, args.subset_seed
    )

    # One reference dataset supplies final-phase oracle schedules, GT motion,
    # and exactly the same deterministic seed serialization used in training.
    (
        reference_dataset,
        reference_tokenizer,
        reference_paths,
        _adaptive,
        calibration,
        calibration_path,
    ) = build_final_phase_dataset(
        checkpoints[first_label], first_config, rollout_names
    )
    examples = build_rollout_examples(
        reference_dataset, reference_tokenizer, reference_paths
    )
    (
        cost_dir,
        ce_weight,
        latent_weight,
        anchor_penalty,
    ) = phase_cost_contract(calibration, cost_dir_override=args.cost_dir)
    edge_costs_by_name = {
        example.name: load_edge_costs(
            cost_dir,
            example.name,
            ce_weight=ce_weight,
            latent_weight=latent_weight,
        )
        for example in examples
    }

    oracle_results = [
        make_ground_truth_result(example, policy="oracle")
        for example in examples
    ]
    oracle_objective_by_name = {
        result.name: schedule_objective(
            result,
            edge_costs_by_name[result.name],
            anchor_penalty=anchor_penalty,
        )[1]
        for result in oracle_results
    }
    oracle_times_by_name = {
        result.name: result.anchor_times for result in oracle_results
    }

    teacher_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    per_clip_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    anchor_slot_rows: list[dict[str, Any]] = []
    cache_contract: dict[str, str] = {}

    def register_condition(
        condition: str,
        checkpoint_label: str,
        results: Sequence[AdaptiveRolloutResult],
        *,
        summarize_tokens: bool,
    ) -> None:
        summary, clips, histogram = summarize_condition(
            condition=condition,
            checkpoint=checkpoint_label,
            results=results,
            edge_costs_by_name=edge_costs_by_name,
            oracle_objective_by_name=oracle_objective_by_name,
            oracle_times_by_name=oracle_times_by_name,
            anchor_penalty=anchor_penalty,
        )
        schedule_rows.append(summary)
        per_clip_rows.extend(clips)
        histogram_rows.extend(histogram)
        if summarize_tokens:
            anchor_summary, slot_summary = summarize_anchor_tokens(
                condition, checkpoint_label, results
            )
            anchor_rows.append(anchor_summary)
            anchor_slot_rows.extend(slot_summary)
        cache_dir = cache_root / condition
        write_adaptive_rollout_cache(results, cache_dir)
        cache_contract[condition] = str(cache_dir)

    register_condition(
        "step2_dp_oracle_gt_anchors",
        "control",
        oracle_results,
        summarize_tokens=False,
    )
    for gap in fixed_gaps:
        results = [
            make_ground_truth_result(example, policy="fixed", fixed_gap=gap)
            for example in examples
        ]
        register_condition(
            f"fixed_gap{gap}_gt_anchors",
            "control",
            results,
            summarize_tokens=False,
        )

    for checkpoint_label, checkpoint in checkpoints.items():
        print(f"\n=== {checkpoint_label}: {checkpoint} ===", flush=True)
        config = source_configs[checkpoint_label]
        model = load_planner(checkpoint, dtype=dtype, device=device)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not args.skip_teacher_forced:
            (
                teacher_dataset,
                _teacher_tokenizer,
                _teacher_paths,
                _teacher_adaptive,
                _teacher_calibration,
                _teacher_calibration_path,
            ) = build_final_phase_dataset(checkpoint, config, teacher_names)
            teacher_loader = DataLoader(
                teacher_dataset,
                batch_size=args.teacher_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=bool(args.num_workers > 0),
                collate_fn=Step1PlannerCollator(
                    tokenizer.pad_token_id, pad_to_multiple_of=8
                ),
            )
            metrics = evaluate(
                model,
                teacher_loader,
                device=device,
                distributed=False,
                use_bf16=use_bf16,
            )
            teacher_rows.append(
                {
                    "checkpoint": checkpoint_label,
                    "checkpoint_path": str(checkpoint),
                    "clips": len(teacher_names),
                    **metrics,
                }
            )
            del teacher_loader, teacher_dataset
            model.eval()

        # Rebuild examples with this checkpoint tokenizer. Token IDs should be
        # identical, but this prevents accidental cross-tokenizer evaluation.
        (
            checkpoint_dataset,
            checkpoint_tokenizer,
            checkpoint_paths,
            _checkpoint_adaptive,
            _checkpoint_calibration,
            _checkpoint_calibration_path,
        ) = build_final_phase_dataset(checkpoint, config, rollout_names)
        checkpoint_examples = build_rollout_examples(
            checkpoint_dataset, checkpoint_tokenizer, checkpoint_paths
        )

        if not args.skip_adaptive_gt_history:
            condition = f"{checkpoint_label}__adaptive_gt_history"
            results = rollout_in_batches(
                model,
                checkpoint_tokenizer,
                checkpoint_examples,
                policy="adaptive",
                anchor_history="ground_truth",
                fixed_gap=None,
                batch_size=args.rollout_batch_size,
                device=device,
                use_bf16=use_bf16,
                progress_label=condition,
            )
            register_condition(
                condition,
                checkpoint_label,
                results,
                summarize_tokens=False,
            )

        if not args.skip_generated_history:
            condition = f"{checkpoint_label}__adaptive_generated_history"
            results = rollout_in_batches(
                model,
                checkpoint_tokenizer,
                checkpoint_examples,
                policy="adaptive",
                anchor_history="generated",
                fixed_gap=None,
                batch_size=args.rollout_batch_size,
                device=device,
                use_bf16=use_bf16,
                progress_label=condition,
            )
            register_condition(
                condition,
                checkpoint_label,
                results,
                summarize_tokens=True,
            )
            for gap in generated_fixed_gaps:
                condition = (
                    f"{checkpoint_label}__fixed_gap{gap}_generated_history"
                )
                results = rollout_in_batches(
                    model,
                    checkpoint_tokenizer,
                    checkpoint_examples,
                    policy="fixed",
                    anchor_history="generated",
                    fixed_gap=gap,
                    batch_size=args.rollout_batch_size,
                    device=device,
                    use_bf16=use_bf16,
                    progress_label=condition,
                )
                register_condition(
                    condition,
                    checkpoint_label,
                    results,
                    summarize_tokens=True,
                )

        del model, tokenizer, checkpoint_dataset, checkpoint_examples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    teacher_df = pd.DataFrame(teacher_rows)
    schedule_df = pd.DataFrame(schedule_rows)
    per_clip_df = pd.DataFrame(per_clip_rows)
    histogram_df = pd.DataFrame(histogram_rows)
    anchor_df = pd.DataFrame(anchor_rows)
    anchor_slots_df = pd.DataFrame(anchor_slot_rows)
    teacher_df.to_csv(output_dir / "teacher_forced_final_phase.csv", index=False)
    schedule_df.to_csv(output_dir / "schedule_summary.csv", index=False)
    per_clip_df.to_csv(output_dir / "schedule_per_clip.csv", index=False)
    histogram_df.to_csv(output_dir / "gap_histogram.csv", index=False)
    anchor_df.to_csv(output_dir / "generated_anchor_summary.csv", index=False)
    anchor_slots_df.to_csv(
        output_dir / "generated_anchor_per_slot.csv", index=False
    )
    (output_dir / "selected_names.json").write_text(
        json.dumps(rollout_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "protocol": {
            "name": "adaptive_gap_closed_loop_offline_known_duration",
            "validation_split": str(first_paths["eval_split"]),
            "teacher_clips": len(teacher_names),
            "rollout_clips": len(rollout_names),
            "subset_seed": args.subset_seed,
            "normal_gap_range": [3, 15],
            "eos_tail_gap_range": [0, 2],
            "duration_contract": (
                "Validation token length is used only for exact EOS stopping and "
                "final-decision clipping. This is not strict unknown-duration inference."
            ),
            "fixed_gaps": list(fixed_gaps),
            "generated_fixed_gaps": list(generated_fixed_gaps),
            "calibration_json": str(calibration_path),
            "cost_dir": str(cost_dir),
            "cost_weights": {
                "ce": ce_weight,
                "hard_latent_l1": latent_weight,
            },
            "cost_boundary_content": calibration["cost"].get(
                "boundary_content", "ground_truth"
            ),
            "anchor_penalty": anchor_penalty,
            "frozen_step2_cost_manifests": calibration.get(
                "cost_manifests", []
            ),
            "math_mode": math_mode,
            "test_split_used": False,
        },
        "checkpoints": {
            label: str(path) for label, path in checkpoints.items()
        },
        "rollout_caches": cache_contract,
        "teacher_forced": records(teacher_df),
        "schedule_summary": records(schedule_df),
        "generated_anchor_summary": records(anchor_df),
    }
    (output_dir / "adaptive_evaluation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )

    if not teacher_df.empty:
        print("\nTeacher-forced final-phase oracle evaluation")
        print(teacher_df.to_string(index=False))
    print("\nSchedule and frozen-Step-2 objective comparison")
    print(
        schedule_df.sort_values("objective_per_frame").to_string(index=False)
    )
    if not anchor_df.empty:
        print("\nGenerated-anchor token comparison")
        print(anchor_df.sort_values("accuracy", ascending=False).to_string(index=False))
    print(f"\nWrote adaptive evaluation outputs: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Stage 1 supplied-gap anchor predictors and export Step 2 inputs.

This runner is deliberately different from
``evaluate_step1_multipart_comparison.py``:

* it accepts one or more arbitrary ``LABEL=CHECKPOINT`` arguments;
* it evaluates the deterministic epoch-0 supplied gap schedule used by
  ``Step1ProvidedGapDataset`` rather than assuming one fixed gap;
* it writes generated-history and matched GT-anchor caches in the adaptive
  rollout schema consumed by ``evaluate_step1_adaptive_motion.py``.

The one-checkpoint mode is important when the full-audio teacher and causal
student live on different machines.  Identical clip selection and schedule
hashes make the resulting bundles comparable after they are copied together.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    canonical_data_path,
    load_motion_tokens,
    read_split_names,
)
from scripts.cache_step2_interval_costs import checkpoint_fingerprint  # noqa: E402
from scripts.evaluate_step1_multipart_comparison import (  # noqa: E402
    deterministic_subset,
    load_planner,
    load_source_config,
    make_dataset_and_loader,
)
from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    audio_contract_from_config,
    planner_context_from_config,
    resolve_data_paths,
    section,
    validate_provided_gap_config,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    BODY_SLOTS,
    gap_from_anchor_times,
)
from utils.inference_math import configure_strict_inference_math  # noqa: E402
from utils.step1_planner_evaluation import (  # noqa: E402
    evaluate_rollouts,
    greedy_rollout_batch,
    teacher_forced_metrics,
)


DEFAULT_STEP2_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15"
)
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Step 1 checkpoint to evaluate; repeat to evaluate co-located models.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Portable Stage 1 metrics and rollout-cache bundle.",
    )
    parser.add_argument(
        "--step2_checkpoint",
        type=Path,
        default=DEFAULT_STEP2_CHECKPOINT,
        help="Exact frozen Step 2 checkpoint used by downstream motion evaluation.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=128,
        help="0 selects the complete validation split.",
    )
    parser.add_argument("--teacher_batch_size", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not SAFE_LABEL.fullmatch(label):
            raise ValueError(
                f"Checkpoint label must match {SAFE_LABEL.pattern}, got {label!r}"
            )
        if label in result:
            raise ValueError(f"Duplicate checkpoint label: {label}")
        path = project_path(raw_path.strip())
        if not path.is_dir():
            raise FileNotFoundError(f"{label} checkpoint missing: {path}")
        result[label] = path
    if not result:
        raise ValueError("At least one checkpoint is required")
    return result


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any] | Sequence[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slot_name(slot: int) -> str:
    spec = BODY_SLOTS[int(slot)]
    return f"{spec.part}_q{spec.quantizer}"


def adaptive_cache_payload(
    *,
    name: str,
    token_length: int,
    anchor_times: Sequence[int],
    anchors: np.ndarray,
    min_gap: int,
    max_gap: int,
    anchor_history: str,
) -> dict[str, Any]:
    times = tuple(int(value) for value in anchor_times)
    values = np.asarray(anchors, dtype=np.int64)
    expected = (len(times) - 1, BODY_SLOT_COUNT)
    if len(times) < 2 or times[0] != 0 or times[-1] != int(token_length) - 1:
        raise ValueError(
            f"{name}: supplied schedule must span [0, {int(token_length) - 1}]"
        )
    if any(right <= left for left, right in zip(times[:-1], times[1:])):
        raise ValueError(f"{name}: anchor times must be strictly increasing")
    if values.shape != expected:
        raise ValueError(f"{name}: anchor array {values.shape} != {expected}")
    if np.any((values < 0) | (values >= BODY_CODEBOOK_SIZE)):
        raise ValueError(f"{name}: anchor IDs leave the multipart codebooks")
    gaps = tuple(
        gap_from_anchor_times(left, right)
        for left, right in zip(times[:-1], times[1:])
    )
    for gap_index, gap in enumerate(gaps):
        is_tail = gap_index == len(gaps) - 1 and gap <= 2
        if not is_tail and not int(min_gap) <= gap <= int(max_gap):
            raise ValueError(
                f"{name}: non-tail supplied gap {gap} is outside "
                f"[{min_gap}, {max_gap}]"
            )
    return {
        "schema": "sentiavatar.step1_adaptive_rollout.v1",
        "name": name,
        "policy": f"supplied_uniform_gap_{min_gap}_{max_gap}",
        "anchor_history": anchor_history,
        "duration_contract": "offline_known_motion_token_length",
        "token_length": int(token_length),
        "eos_clipped_decisions": 0,
        "predicted_gap_decisions": [int(value) for value in gaps],
        "executed_gaps": [int(value) for value in gaps],
        "anchors": [
            {
                "time": int(time_value),
                "tokens": [int(value) for value in anchor],
            }
            for time_value, anchor in zip(times[1:], values)
        ],
    }


def cache_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return root / Path(*parts).with_suffix(".json")


def summarize_by_gap(
    *,
    label: str,
    gaps: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    negative_log_likelihood: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    gaps = np.asarray(gaps, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape or targets.shape[1:] != (BODY_SLOT_COUNT,):
        raise ValueError("Gap summary targets/predictions must share shape [N, 16]")
    if len(gaps) != len(targets):
        raise ValueError("Gap count does not match anchor count")
    if negative_log_likelihood is not None:
        negative_log_likelihood = np.asarray(
            negative_log_likelihood, dtype=np.float64
        )
        if negative_log_likelihood.shape != targets.shape:
            raise ValueError("Negative log likelihood must match anchor targets")
    rows: list[dict[str, Any]] = []
    for gap in sorted(set(int(value) for value in gaps)):
        selected = gaps == gap
        correct = predictions[selected] == targets[selected]
        row: dict[str, Any] = {
            "checkpoint": label,
            "gap": int(gap),
            "anchors": int(selected.sum()),
            "tokens": int(correct.size),
            "accuracy": float(correct.mean()),
        }
        for quantizer in range(4):
            slots = [
                slot
                for slot, spec in enumerate(BODY_SLOTS)
                if spec.quantizer == quantizer
            ]
            row[f"q{quantizer}_accuracy"] = float(correct[:, slots].mean())
        if negative_log_likelihood is not None:
            row["cross_entropy"] = float(
                negative_log_likelihood[selected].mean()
            )
            row["perplexity"] = float(np.exp(min(50.0, row["cross_entropy"])))
        rows.append(row)
    return rows


def per_clip_rollout_rows(
    label: str,
    results,
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        correct = result.predicted_anchors == result.target_anchors
        gaps = [
            gap_from_anchor_times(left, right)
            for left, right in zip(
                result.anchor_times[:-1], result.anchor_times[1:]
            )
        ]
        rows.append(
            {
                "checkpoint": label,
                "name": result.name,
                "anchors": len(result.predicted_anchors),
                "mean_gap": float(np.mean(gaps)),
                "accuracy": float(correct.mean()),
                "q0_accuracy": float(correct[:, [0, 4, 8, 12]].mean()),
                "mean_confidence": float(result.confidence.mean()),
                "mean_entropy": float(result.entropy.mean()),
                "elapsed_seconds": float(result.elapsed_seconds),
            }
        )
    return rows


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.teacher_batch_size < 1 or args.rollout_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_clips < 0:
        raise ValueError("max_clips must be >= 0")
    checkpoints = parse_checkpoints(args.checkpoint)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step2_checkpoint = project_path(args.step2_checkpoint)
    if not step2_checkpoint.is_dir():
        raise FileNotFoundError(f"Frozen Step 2 checkpoint missing: {step2_checkpoint}")

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
    validation_all = read_split_names(first_paths["eval_split"])
    selected_names = deterministic_subset(
        validation_all, args.max_clips, args.subset_seed
    )
    first_training = section(first_config, "training")
    first_supplied = validate_provided_gap_config(first_config)
    if not first_supplied["enabled"]:
        raise ValueError(
            f"{first_label} is not a provided_gap_training Stage 1 checkpoint"
        )
    min_gap = int(first_supplied["min_gap"])
    max_gap = int(first_supplied["max_gap"])
    first_audio = audio_contract_from_config(first_config)

    contracts: list[dict[str, Any]] = []
    for label, config in source_configs.items():
        paths = resolve_data_paths(config)
        supplied = validate_provided_gap_config(config)
        training = section(config, "training")
        audio = audio_contract_from_config(config)
        if not supplied["enabled"]:
            raise ValueError(f"{label} does not enable provided_gap_training")
        if (
            int(supplied["min_gap"]),
            int(supplied["max_gap"]),
            str(supplied["distribution"]),
        ) != (min_gap, max_gap, str(first_supplied["distribution"])):
            raise ValueError(f"{label} uses a different supplied-gap contract")
        if int(training.get("seed", 42)) != int(first_training.get("seed", 42)):
            raise ValueError(f"{label} uses a different schedule seed")
        if read_split_names(paths["eval_split"]) != validation_all:
            raise ValueError(f"{label} uses a different ordered validation split")
        if paths["motion_token_dir"].resolve() != first_paths[
            "motion_token_dir"
        ].resolve():
            raise ValueError(f"{label} uses a different motion-token export")
        audio_keys = (
            "codec",
            "frame_rate",
            "stored_codebooks",
            "cardinality",
            "codebooks_used",
            "input_representation",
        )
        if any(audio[key] != first_audio[key] for key in audio_keys):
            raise ValueError(f"{label} uses a different audio-token contract")
        context = planner_context_from_config(config)
        contracts.append(
            {
                "checkpoint": label,
                "checkpoint_path": str(checkpoints[label]),
                "attention_mode": context["attention_mode"],
                "sequence_layout": context["sequence_layout"],
                "configured_epochs": int(training.get("num_train_epochs", 0)),
                "audio_codec": audio["codec"],
                "audio_codebooks_used": json.dumps(audio["codebooks_used"]),
                "audio_input_representation": audio["input_representation"],
                "supplied_min_gap": min_gap,
                "supplied_max_gap": max_gap,
                "schedule_seed": int(training.get("seed", 42)),
            }
        )

    step2_fingerprint = checkpoint_fingerprint(step2_checkpoint)
    teacher_rows: list[dict[str, Any]] = []
    teacher_slot_rows: list[dict[str, Any]] = []
    teacher_gap_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    rollout_slot_rows: list[dict[str, Any]] = []
    rollout_gap_rows: list[dict[str, Any]] = []
    rollout_horizon_rows: list[dict[str, Any]] = []
    rollout_clip_rows: list[dict[str, Any]] = []
    schedule_manifest: list[dict[str, Any]] | None = None
    common_targets: np.ndarray | None = None
    condition_paths: dict[str, str] = {}
    gt_condition = f"supplied_gap{min_gap}_{max_gap}_gt_anchors"
    gt_cache_root = output_dir / "rollout_cache" / gt_condition
    condition_paths[gt_condition] = str(gt_cache_root.resolve())

    for label, checkpoint in checkpoints.items():
        print(f"\n=== {label}: {checkpoint} ===", flush=True)
        config = source_configs[label]
        _, teacher_loader, _, _, _ = make_dataset_and_loader(
            checkpoint=checkpoint,
            source_config=config,
            names=selected_names,
            batch_size=args.teacher_batch_size,
            workers=args.num_workers,
            preserve_times=False,
        )
        model = load_planner(checkpoint, dtype=dtype, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        teacher = teacher_forced_metrics(
            model,
            teacher_loader,
            device=device,
            use_bf16=use_bf16,
        )
        teacher_rows.append(
            {
                "checkpoint": label,
                "checkpoint_path": str(checkpoint),
                **teacher["summary"],
                "peak_gpu_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
            }
        )
        teacher_slot_rows.extend(
            {"checkpoint": label, **row} for row in teacher["slot_rows"]
        )
        del teacher_loader

        _, rollout_loader, _, _, _ = make_dataset_and_loader(
            checkpoint=checkpoint,
            source_config=config,
            names=selected_names,
            batch_size=args.rollout_batch_size,
            workers=args.num_workers,
            preserve_times=True,
        )
        rollout_results = []
        for batch_index, batch in enumerate(rollout_loader, start=1):
            rollout_results.extend(
                greedy_rollout_batch(
                    model,
                    batch,
                    device=device,
                    use_bf16=use_bf16,
                )
            )
            if batch_index % 10 == 0:
                print(
                    f"{label}: rollout {len(rollout_results)}/"
                    f"{len(selected_names)} clips",
                    flush=True,
                )
        if [result.name for result in rollout_results] != selected_names:
            raise ValueError(f"{label}: rollout order differs from selected names")

        current_manifest: list[dict[str, Any]] = []
        current_gaps: list[int] = []
        previous_anchors: list[np.ndarray] = []
        generated_condition = (
            f"{label}__supplied_gap{min_gap}_{max_gap}_generated_history"
        )
        generated_root = output_dir / "rollout_cache" / generated_condition
        condition_paths[generated_condition] = str(generated_root.resolve())
        for result in rollout_results:
            dense, _ = load_motion_tokens(
                canonical_data_path(
                    first_paths["motion_token_dir"], result.name, ".json"
                ),
                require_causal=True,
            )
            dense_array = np.asarray(dense, dtype=np.int64)
            times = tuple(int(value) for value in result.anchor_times)
            expected_targets = dense_array[list(times[1:])]
            if not np.array_equal(result.target_anchors, expected_targets):
                raise ValueError(
                    f"{result.name}: serialized targets differ from dense tokens"
                )
            gaps = [
                gap_from_anchor_times(left, right)
                for left, right in zip(times[:-1], times[1:])
            ]
            current_gaps.extend(gaps)
            previous_anchors.extend(dense_array[list(times[:-1])])
            current_manifest.append(
                {
                    "name": result.name,
                    "token_length": len(dense_array),
                    "anchor_times": list(times),
                    "gaps": gaps,
                }
            )
            if schedule_manifest is None:
                gt_payload = adaptive_cache_payload(
                    name=result.name,
                    token_length=len(dense_array),
                    anchor_times=times,
                    anchors=expected_targets,
                    min_gap=min_gap,
                    max_gap=max_gap,
                    anchor_history="ground_truth",
                )
                atomic_write_json(
                    cache_path(gt_cache_root, result.name), gt_payload
                )
            generated_payload = adaptive_cache_payload(
                name=result.name,
                token_length=len(dense_array),
                anchor_times=times,
                anchors=result.predicted_anchors,
                min_gap=min_gap,
                max_gap=max_gap,
                anchor_history="generated",
            )
            atomic_write_json(
                cache_path(generated_root, result.name), generated_payload
            )

        if schedule_manifest is None:
            schedule_manifest = current_manifest
        elif current_manifest != schedule_manifest:
            raise AssertionError(
                f"{label}: deterministic supplied schedules differ from "
                f"{first_label}"
            )

        measured = evaluate_rollouts(rollout_results)
        if common_targets is None:
            common_targets = measured["labels"]
        elif not np.array_equal(common_targets, measured["labels"]):
            raise AssertionError(f"{label}: rollout targets differ from {first_label}")
        gaps_array = np.asarray(current_gaps, dtype=np.int64)
        if len(gaps_array) != len(measured["labels"]):
            raise AssertionError("Gap vector does not match rollout anchors")
        if not np.array_equal(teacher["labels"], measured["labels"]):
            raise AssertionError(
                f"{label}: teacher-forced and rollout targets differ"
            )

        teacher_gap_rows.extend(
            summarize_by_gap(
                label=label,
                gaps=gaps_array,
                targets=teacher["labels"],
                predictions=teacher["predictions"],
                negative_log_likelihood=teacher["negative_log_likelihood"],
            )
        )
        rollout_gap_rows.extend(
            summarize_by_gap(
                label=label,
                gaps=gaps_array,
                targets=measured["labels"],
                predictions=measured["predictions"],
            )
        )
        previous_array = np.asarray(previous_anchors, dtype=np.int64)
        copy_accuracy = float(
            (previous_array == measured["labels"]).mean()
        )
        rollout_rows.append(
            {
                "checkpoint": label,
                "checkpoint_path": str(checkpoint),
                **measured["summary"],
                "teacher_forced_accuracy_same_subset": teacher["summary"][
                    "accuracy"
                ],
                "teacher_forced_ce_same_subset": teacher["summary"][
                    "cross_entropy"
                ],
                "previous_gt_anchor_copy_accuracy": copy_accuracy,
                "accuracy_drop_from_teacher_forcing": measured["summary"][
                    "accuracy"
                ]
                - teacher["summary"]["accuracy"],
                "accuracy_margin_over_previous_copy": measured["summary"][
                    "accuracy"
                ]
                - copy_accuracy,
            }
        )
        rollout_slot_rows.extend(
            {"checkpoint": label, **row} for row in measured["slot_rows"]
        )
        rollout_horizon_rows.extend(
            {"checkpoint": label, **row} for row in measured["horizon_rows"]
        )
        rollout_clip_rows.extend(per_clip_rollout_rows(label, rollout_results))

        del model, teacher, rollout_loader, rollout_results, measured
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert schedule_manifest is not None
    schedule_hash = stable_json_hash(schedule_manifest)
    names_hash = stable_json_hash(selected_names)
    atomic_write_json(output_dir / "selected_names.json", selected_names)
    atomic_write_json(output_dir / "schedule_manifest.json", schedule_manifest)

    contracts_df = pd.DataFrame(contracts)
    teacher_df = pd.DataFrame(teacher_rows).sort_values("cross_entropy")
    teacher_slots_df = pd.DataFrame(teacher_slot_rows)
    teacher_gaps_df = pd.DataFrame(teacher_gap_rows)
    rollout_df = pd.DataFrame(rollout_rows).sort_values(
        "accuracy", ascending=False
    )
    rollout_slots_df = pd.DataFrame(rollout_slot_rows)
    rollout_gaps_df = pd.DataFrame(rollout_gap_rows)
    rollout_horizon_df = pd.DataFrame(rollout_horizon_rows)
    rollout_clips_df = pd.DataFrame(rollout_clip_rows)
    for filename, frame in (
        ("stage1_contracts.csv", contracts_df),
        ("stage1_teacher_forced.csv", teacher_df),
        ("stage1_teacher_forced_per_slot.csv", teacher_slots_df),
        ("stage1_teacher_forced_by_gap.csv", teacher_gaps_df),
        ("stage1_generated_rollout.csv", rollout_df),
        ("stage1_generated_rollout_per_slot.csv", rollout_slots_df),
        ("stage1_generated_rollout_by_gap.csv", rollout_gaps_df),
        ("stage1_generated_rollout_horizon.csv", rollout_horizon_df),
        ("stage1_generated_rollout_per_clip.csv", rollout_clips_df),
    ):
        write_dataframe(frame, output_dir / filename)

    protocol = {
        "name": "stage1_supplied_gap_anchor_evaluation",
        "validation_split": str(first_paths["eval_split"]),
        "selected_clips": len(selected_names),
        "rollout_clips": len(selected_names),
        "subset_seed": int(args.subset_seed),
        "selected_names_sha256": names_hash,
        "schedule_sha256": schedule_hash,
        "schedule_seed": int(first_training.get("seed", 42)),
        "supplied_gap_distribution": str(first_supplied["distribution"]),
        "supplied_gap_range": [min_gap, max_gap],
        "duration_contract": "offline_known_motion_token_length",
        "frozen_step2_cost_manifests": [
            {
                "checkpoint": str(step2_checkpoint),
                "checkpoint_fingerprint": step2_fingerprint,
            }
        ],
        "math_mode": math_mode,
        "test_split_used": False,
    }
    adaptive_report = {
        "protocol": protocol,
        "checkpoints": {
            label: str(path) for label, path in checkpoints.items()
        },
        "rollout_caches": condition_paths,
    }
    atomic_write_json(
        output_dir / "adaptive_evaluation_report.json", adaptive_report
    )
    report = {
        "schema": "sentiavatar.step1_stage1_supplied_gap_eval.v1",
        "protocol": protocol,
        "contracts": records(contracts_df),
        "teacher_forced": records(teacher_df),
        "generated_rollout": records(rollout_df),
        "rollout_caches": condition_paths,
    }
    atomic_write_json(output_dir / "stage1_evaluation_report.json", report)

    print("\nTeacher-forced comparison")
    print(teacher_df.to_string(index=False))
    print("\nGenerated-history rollout comparison")
    print(rollout_df.to_string(index=False))
    print(f"\nSchedule SHA-256: {schedule_hash}")
    print(f"Step 2 SHA-256:    {step2_fingerprint}")
    print(f"Wrote portable Stage 1 bundle: {output_dir}")


if __name__ == "__main__":
    main()

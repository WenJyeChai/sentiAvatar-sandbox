#!/usr/bin/env python3
"""Compare multipart Step 1 checkpoints on one held-out validation protocol.

The runner is intentionally representation-specific. Every checkpoint here
predicts the same 16 causal multipart IDs per fixed-gap anchor, so token CE and
accuracy are directly comparable. The released SentiAvatar planner uses a
different legacy RVQ representation and is evaluated by a separate runner.
"""

from __future__ import annotations

import argparse
import gc
import json
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
    MimiQwenPlanner,
    Step1PlannerCollator,
    load_text_map,
    read_split_names,
)
from scripts.export_multipart_motion_tokens import configure_strict_inference_math  # noqa: E402
from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    build_dataset,
    load_neutral_seed,
    resolve_data_paths,
    section,
)
from utils.step1_planner_evaluation import (  # noqa: E402
    Step1EvaluationCollator,
    collect_fixed_gap_targets,
    evaluate_reference_baselines,
    evaluate_rollouts,
    greedy_rollout_batch,
    teacher_forced_metrics,
    write_rollout_cache,
)


DEFAULT_CHECKPOINTS = (
    (
        "self_forcing_best",
        "checkpoints/step1_multipart_fixed_gap3_self_forcing_q0q3_full/best",
    ),
    (
        "self_forcing_best_rollout",
        "checkpoints/step1_multipart_fixed_gap3_self_forcing_q0q3_full/best_rollout",
    ),
    ("q0_6k_final", "checkpoints/step1_multipart_fixed_gap3_main6000/final"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare causal multipart Step 1 planner checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeat to override the three default checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR
        / "motion_generation"
        / "outputs"
        / "step1_baseline_comparison",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
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
    parser.add_argument("--write_rollout_cache", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def parse_checkpoints(values: Sequence[str] | None) -> dict[str, Path]:
    pairs = DEFAULT_CHECKPOINTS if values is None else []
    if values is not None:
        parsed: list[tuple[str, str]] = []
        for value in values:
            if "=" not in value:
                raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
            label, path = value.split("=", 1)
            label = label.strip()
            if not label or not path.strip():
                raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
            parsed.append((label, path.strip()))
        pairs = parsed
    result = {label: project_path(path).resolve() for label, path in pairs}
    if len(result) != len(pairs):
        raise ValueError("Checkpoint labels must be unique")
    if len(result) < 2:
        raise ValueError("At least two checkpoints are required for comparison")
    return result


def deterministic_subset(names: Sequence[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(names):
        return list(names)
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(names), size=limit, replace=False))
    return [str(names[int(index)]) for index in indices]


def load_source_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "phase1_source_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint source configuration missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return payload


def load_planner(path: Path, *, dtype: torch.dtype, device: torch.device) -> MimiQwenPlanner:
    try:
        model = MimiQwenPlanner.from_pretrained(
            path, dtype=dtype, local_files_only=True
        )
    except TypeError:
        model = MimiQwenPlanner.from_pretrained(
            path, torch_dtype=dtype, local_files_only=True
        )
    model.gradient_checkpointing_disable()
    model.language_model.config.use_cache = True
    return model.to(device).eval()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def make_dataset_and_loader(
    *,
    checkpoint: Path,
    source_config: Mapping[str, Any],
    names: Sequence[str],
    batch_size: int,
    workers: int,
    preserve_times: bool,
):
    paths = resolve_data_paths(source_config)
    data_config = section(source_config, "data")
    training_config = section(source_config, "training")
    model_audio = section(source_config, "audio")
    if "mimi_codebooks_used" not in data_config:
        data_config["mimi_codebooks_used"] = model_audio.get(
            "mimi_codebooks_used", [0]
        )
    data_config["random_seed"] = int(training_config.get("seed", 42))
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
    )
    base_collator = Step1PlannerCollator(tokenizer.pad_token_id, pad_to_multiple_of=8)
    collator = Step1EvaluationCollator(base_collator) if preserve_times else base_collator
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers > 0),
        collate_fn=collator,
    )
    return dataset, loader, paths, data_config, training_config


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    for label, checkpoint in checkpoints.items():
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"{label} checkpoint missing: {checkpoint}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = bool(not args.no_bf16 and device.type == "cuda")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a CUDA device without bf16 support")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    math_mode = configure_strict_inference_math(device)

    source_configs = {
        label: load_source_config(checkpoint)
        for label, checkpoint in checkpoints.items()
    }
    first_label = next(iter(checkpoints))
    reference_paths = resolve_data_paths(source_configs[first_label])
    validation_all = read_split_names(reference_paths["eval_split"])
    for label, config in source_configs.items():
        candidate = read_split_names(resolve_data_paths(config)["eval_split"])
        if candidate != validation_all:
            raise ValueError(
                f"{label} does not use the same ordered validation split as {first_label}"
            )
    teacher_names = deterministic_subset(
        validation_all, args.teacher_max_clips, args.subset_seed
    )
    rollout_names = deterministic_subset(
        validation_all, args.rollout_max_clips, args.subset_seed
    )

    # Fit every reference on one common training split: the first checkpoint's
    # source split. With the defaults this is the complete 19,019-clip split.
    reference_train_names = read_split_names(reference_paths["train_split"])
    reference_data = section(source_configs[first_label], "data")
    fixed_gap = int(reference_data.get("fixed_gap", 3))
    reference_train_targets, _, _ = collect_fixed_gap_targets(
        names=reference_train_names,
        motion_token_dir=reference_paths["motion_token_dir"],
        fixed_gap=fixed_gap,
        require_causal=True,
    )
    validation_targets, validation_previous, _ = collect_fixed_gap_targets(
        names=teacher_names,
        motion_token_dir=reference_paths["motion_token_dir"],
        fixed_gap=fixed_gap,
        require_causal=True,
    )
    baseline_summary, baseline_slots = evaluate_reference_baselines(
        train_targets=reference_train_targets,
        validation_targets=validation_targets,
        validation_previous=validation_previous,
    )
    baseline_df = pd.DataFrame(baseline_summary)
    baseline_slots_df = pd.DataFrame(baseline_slots)
    baseline_df.to_csv(output_dir / "multipart_reference_baselines.csv", index=False)
    baseline_slots_df.to_csv(
        output_dir / "multipart_reference_baselines_per_slot.csv", index=False
    )

    contracts: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    teacher_slot_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    rollout_slot_rows: list[dict[str, Any]] = []
    rollout_horizon_rows: list[dict[str, Any]] = []
    common_rollout_labels: np.ndarray | None = None

    for label, checkpoint in checkpoints.items():
        print(f"\n=== {label}: {checkpoint} ===", flush=True)
        config = source_configs[label]
        candidate_paths = resolve_data_paths(config)
        candidate_data = section(config, "data")
        candidate_training = section(config, "training")
        candidate_audio = section(config, "audio")
        candidate_generated = section(config, "generated_history")
        train_names = read_split_names(candidate_paths["train_split"])
        if int(candidate_data.get("fixed_gap", 3)) != fixed_gap:
            raise ValueError(f"{label} uses a different fixed gap")
        if candidate_paths["motion_token_dir"].resolve() != reference_paths[
            "motion_token_dir"
        ].resolve():
            raise ValueError(f"{label} uses a different multipart token export")
        codebooks = candidate_data.get(
            "mimi_codebooks_used", candidate_audio.get("mimi_codebooks_used", [0])
        )
        contracts.append(
            {
                "checkpoint": label,
                "checkpoint_path": str(checkpoint),
                "training_clips": len(train_names),
                "configured_epochs": int(candidate_training.get("num_train_epochs", 0)),
                "mimi_codebooks_used": json.dumps(list(codebooks)),
                "generated_history_enabled": bool(candidate_generated.get("enabled", False)),
                "generated_history_max_probability": float(
                    candidate_generated.get("max_probability", 0.0)
                ),
                "seed_mode": candidate_data.get("seed_mode"),
                "fixed_gap": fixed_gap,
            }
        )

        _, teacher_loader, _, _, _ = make_dataset_and_loader(
            checkpoint=checkpoint,
            source_config=config,
            names=teacher_names,
            batch_size=args.teacher_batch_size,
            workers=args.num_workers,
            preserve_times=False,
        )
        model = load_planner(checkpoint, dtype=dtype, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        teacher = teacher_forced_metrics(
            model, teacher_loader, device=device, use_bf16=use_bf16
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
        del teacher, teacher_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _, rollout_loader, _, _, _ = make_dataset_and_loader(
            checkpoint=checkpoint,
            source_config=config,
            names=rollout_names,
            batch_size=args.rollout_batch_size,
            workers=args.num_workers,
            preserve_times=True,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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
                    f"{label}: rollout {len(rollout_results)}/{len(rollout_names)} clips",
                    flush=True,
                )
        measured = evaluate_rollouts(rollout_results)
        if common_rollout_labels is None:
            common_rollout_labels = measured["labels"]
        elif not np.array_equal(common_rollout_labels, measured["labels"]):
            raise AssertionError(f"{label} rollout targets differ from the first checkpoint")
        rollout_targets, rollout_previous, _ = collect_fixed_gap_targets(
            names=rollout_names,
            motion_token_dir=reference_paths["motion_token_dir"],
            fixed_gap=fixed_gap,
            require_causal=True,
        )
        if not np.array_equal(measured["labels"], rollout_targets):
            raise AssertionError(f"{label} rollout target order differs from baseline order")
        rollout_baselines, _ = evaluate_reference_baselines(
            train_targets=reference_train_targets,
            validation_targets=rollout_targets,
            validation_previous=rollout_previous,
        )
        copy_accuracy = float(
            next(
                row["accuracy"]
                for row in rollout_baselines
                if row["baseline"] == "previous_gt_anchor_copy"
            )
        )
        teacher_same_subset_dataset, teacher_same_subset_loader, _, _, _ = (
            make_dataset_and_loader(
                checkpoint=checkpoint,
                source_config=config,
                names=rollout_names,
                batch_size=args.teacher_batch_size,
                workers=args.num_workers,
                preserve_times=False,
            )
        )
        del teacher_same_subset_dataset
        teacher_same = teacher_forced_metrics(
            model,
            teacher_same_subset_loader,
            device=device,
            use_bf16=use_bf16,
        )["summary"]
        summary = {
            "checkpoint": label,
            "checkpoint_path": str(checkpoint),
            **measured["summary"],
            "teacher_forced_accuracy_same_subset": teacher_same["accuracy"],
            "teacher_forced_ce_same_subset": teacher_same["cross_entropy"],
            "previous_copy_accuracy_same_subset": copy_accuracy,
            "accuracy_drop_from_teacher_forcing": measured["summary"]["accuracy"]
            - teacher_same["accuracy"],
            "accuracy_margin_over_previous_copy": measured["summary"]["accuracy"]
            - copy_accuracy,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }
        rollout_rows.append(summary)
        rollout_slot_rows.extend(
            {"checkpoint": label, **row} for row in measured["slot_rows"]
        )
        rollout_horizon_rows.extend(
            {"checkpoint": label, **row} for row in measured["horizon_rows"]
        )
        if args.write_rollout_cache:
            write_rollout_cache(
                rollout_results, output_dir / "rollout_cache" / label
            )
        del (
            model,
            rollout_loader,
            rollout_results,
            measured,
            teacher_same_subset_loader,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contracts_df = pd.DataFrame(contracts)
    teacher_df = pd.DataFrame(teacher_rows).sort_values("cross_entropy")
    teacher_slots_df = pd.DataFrame(teacher_slot_rows)
    rollout_df = pd.DataFrame(rollout_rows).sort_values("accuracy", ascending=False)
    rollout_slots_df = pd.DataFrame(rollout_slot_rows)
    rollout_horizon_df = pd.DataFrame(rollout_horizon_rows)
    contracts_df.to_csv(output_dir / "multipart_contracts.csv", index=False)
    teacher_df.to_csv(output_dir / "multipart_teacher_forced.csv", index=False)
    teacher_slots_df.to_csv(
        output_dir / "multipart_teacher_forced_per_slot.csv", index=False
    )
    rollout_df.to_csv(output_dir / "multipart_generated_rollout.csv", index=False)
    rollout_slots_df.to_csv(
        output_dir / "multipart_generated_rollout_per_slot.csv", index=False
    )
    rollout_horizon_df.to_csv(
        output_dir / "multipart_generated_rollout_horizon.csv", index=False
    )

    report = {
        "protocol": {
            "validation_split": str(reference_paths["eval_split"]),
            "teacher_clips": len(teacher_names),
            "rollout_clips": len(rollout_names),
            "subset_seed": args.subset_seed,
            "fixed_gap": fixed_gap,
            "common_unigram_training_split": str(reference_paths["train_split"]),
            "common_unigram_training_clips": len(reference_train_names),
            "test_split_used": False,
            "math_mode": math_mode,
        },
        "contracts": records(contracts_df),
        "reference_baselines": records(baseline_df),
        "teacher_forced": records(teacher_df),
        "generated_rollout": records(rollout_df),
    }
    (output_dir / "multipart_comparison_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    print("\nTeacher-forced comparison")
    print(teacher_df.to_string(index=False))
    print("\nGenerated-rollout comparison")
    print(rollout_df.to_string(index=False))
    print(f"\nWrote comparison outputs: {output_dir}")


if __name__ == "__main__":
    main()

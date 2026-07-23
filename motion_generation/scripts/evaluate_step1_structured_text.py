#!/usr/bin/env python3
"""Evaluate a structured-text Step 1 checkpoint and its tag contribution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
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
    parse_structured_text,
    read_split_names,
)
from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    build_dataset,
    data_config_from_config,
    load_neutral_seed,
    resolve_data_paths,
)
from utils.inference_math import configure_strict_inference_math  # noqa: E402
from utils.step1_planner_evaluation import (  # noqa: E402
    Step1EvaluationCollator,
    evaluate_rollouts,
    greedy_rollout_batch,
    teacher_forced_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_max_clips", type=int, default=0)
    parser.add_argument("--rollout_max_clips", type=int, default=128)
    parser.add_argument("--teacher_batch_size", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def project_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (PROJECT_DIR / value).resolve()


def source_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "phase1_source_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def deterministic_subset(names: Sequence[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(names):
        return list(names)
    generator = np.random.default_rng(int(seed))
    indices = np.sort(generator.choice(len(names), size=limit, replace=False))
    return [str(names[int(index)]) for index in indices]


def json_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": dict(result["summary"]),
        "slot_rows": list(result["slot_rows"]),
    }


def build_loader(
    names: Sequence[str],
    *,
    tokenizer,
    paths: Mapping[str, Path],
    text_map: Mapping[str, str],
    data_config: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
    preserve_times: bool,
):
    dataset = build_dataset(
        list(names),
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=load_neutral_seed(data_config.get("neutral_seed_json")),
        training=False,
    )
    base = Step1PlannerCollator(tokenizer.pad_token_id, pad_to_multiple_of=8)
    collator = Step1EvaluationCollator(base) if preserve_times else base
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        collate_fn=collator,
    )
    return dataset, loader


def main() -> None:
    args = parse_args()
    checkpoint = project_path(args.checkpoint)
    output_json = project_path(args.output_json)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config = source_config(checkpoint)
    data_config = data_config_from_config(config)
    if data_config.get("text_serialization") != "structured_fields":
        raise ValueError("This evaluator requires a structured_fields checkpoint")
    paths = resolve_data_paths(config)
    text_map = load_text_map(paths["text_json"])
    all_names = read_split_names(paths["eval_split"])
    teacher_names = deterministic_subset(
        all_names, args.teacher_max_clips, args.subset_seed
    )
    rollout_names = deterministic_subset(
        all_names, args.rollout_max_clips, args.subset_seed
    )

    device = torch.device(args.device)
    use_bf16 = not args.no_bf16 and device.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    configure_strict_inference_math(device)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = MimiQwenPlanner.from_pretrained(
            checkpoint, dtype=dtype, local_files_only=True
        )
    except TypeError:
        model = MimiQwenPlanner.from_pretrained(
            checkpoint, torch_dtype=dtype, local_files_only=True
        )
    model.gradient_checkpointing_disable()
    model.language_model.config.use_cache = True
    model.to(device).eval()

    _, teacher_loader = build_loader(
        teacher_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        batch_size=args.teacher_batch_size,
        num_workers=args.num_workers,
        preserve_times=False,
    )
    teacher = teacher_forced_metrics(
        model, teacher_loader, device=device, use_bf16=use_bf16
    )

    grouped_names: dict[str, list[str]] = defaultdict(list)
    for name in teacher_names:
        grouped_names[parse_structured_text(text_map[name]).annotation_pattern].append(name)
    subgroup_results: dict[str, Any] = {}
    for pattern in ("expression+action", "action-only", "expression-only", "no-tags"):
        names = grouped_names.get(pattern, [])
        if not names:
            continue
        _, loader = build_loader(
            names,
            tokenizer=tokenizer,
            paths=paths,
            text_map=text_map,
            data_config=data_config,
            batch_size=args.teacher_batch_size,
            num_workers=args.num_workers,
            preserve_times=False,
        )
        subgroup_results[pattern] = {
            "clips": len(names),
            **json_metrics(
                teacher_forced_metrics(
                    model, loader, device=device, use_bf16=use_bf16
                )
            ),
        }

    tags_removed_config = dict(data_config)
    tags_removed_config["drop_structured_tags"] = True
    _, tags_removed_loader = build_loader(
        teacher_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=tags_removed_config,
        batch_size=args.teacher_batch_size,
        num_workers=args.num_workers,
        preserve_times=False,
    )
    tags_removed = teacher_forced_metrics(
        model, tags_removed_loader, device=device, use_bf16=use_bf16
    )

    _, rollout_loader = build_loader(
        rollout_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        batch_size=args.rollout_batch_size,
        num_workers=args.num_workers,
        preserve_times=True,
    )
    rollouts = []
    for batch in rollout_loader:
        rollouts.extend(
            greedy_rollout_batch(
                model,
                batch,
                device=device,
                use_bf16=use_bf16,
            )
        )
    rollout = evaluate_rollouts(rollouts)

    full_summary = dict(teacher["summary"])
    removed_summary = dict(tags_removed["summary"])
    payload = {
        "checkpoint": str(checkpoint),
        "audio_contract": {
            "codec": model.config.audio_codec,
            "sample_rate": model.config.audio_sample_rate,
            "frame_rate": model.config.audio_frame_rate,
            "frame_size": model.config.audio_frame_size,
            "cardinality": model.config.audio_cardinality,
            "stored_codebooks": model.config.audio_codebooks_stored,
            "codebooks_used": model.config.audio_codebooks_used,
        },
        "teacher_forced": json_metrics(teacher),
        "annotation_subgroups": subgroup_results,
        "tags_removed_teacher_forced": {
            **json_metrics(tags_removed),
            "delta_cross_entropy_vs_full": (
                float(removed_summary["cross_entropy"])
                - float(full_summary["cross_entropy"])
            ),
            "delta_accuracy_vs_full": (
                float(removed_summary["accuracy"])
                - float(full_summary["accuracy"])
            ),
        },
        "generated_rollout": {
            "summary": dict(rollout["summary"]),
            "slot_rows": list(rollout["slot_rows"]),
            "horizon_rows": list(rollout["horizon_rows"]),
        },
        "protocol": {
            "validation_split": str(paths["eval_split"]),
            "teacher_clips": len(teacher_names),
            "rollout_clips": len(rollout_names),
            "subset_seed": args.subset_seed,
            "tag_removal": (
                "Expression/action fields are rebuilt with missing markers; "
                "transcript and audio remain unchanged."
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload["teacher_forced"]["summary"], indent=2))
    print(json.dumps(payload["tags_removed_teacher_forced"], indent=2))
    print(json.dumps(payload["generated_rollout"]["summary"], indent=2))
    print("Wrote:", output_json)


if __name__ == "__main__":
    main()

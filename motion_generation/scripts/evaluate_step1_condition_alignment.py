#!/usr/bin/env python3
"""Evaluate correct-vs-corrupted condition likelihood under fixed histories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Subset
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
from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    build_condition_corruption,
    build_dataset,
    corrupted_model_inputs,
    load_neutral_seed,
    mimi_codebooks_from_config,
    move_batch,
    move_condition_metadata,
    resolve_data_paths,
    section,
    validate_condition_alignment_config,
)
from utils.step1_condition_alignment import (  # noqa: E402
    counterfactual_likelihood_loss,
)
from utils.step1_self_forcing import (  # noqa: E402
    GeneratedHistoryBatchStats,
    apply_generated_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_clips", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--skip_generated_prefix", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def source_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "phase1_source_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a config mapping in {path}")
    return payload


def deterministic_subset(names: list[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(names):
        return names
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(names), generator=generator)[:limit].sort().values
    return [names[int(index)] for index in indices]


@torch.inference_mode()
def evaluate_history_condition(
    model: MimiQwenPlanner,
    loader: DataLoader,
    *,
    device: torch.device,
    use_bf16: bool,
    alignment: Mapping[str, Any],
    generated_prefix: bool,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    token_loss_sum = 0.0
    token_correct = 0
    token_count = 0
    gap_sums = {"audio": 0.0, "text": 0.0}
    gap_counts = {"audio": 0, "text": 0}
    rollout = GeneratedHistoryBatchStats()
    autocast_enabled = use_bf16 and device.type == "cuda"
    for batch_index, batch in enumerate(loader):
        inputs = move_batch(batch, device)
        metadata = move_condition_metadata(batch, device)
        if generated_prefix:
            generated_ids, stats = apply_generated_history(
                model,
                inputs,
                list(range(inputs["input_ids"].shape[0])),
                microbatch_size=inputs["input_ids"].shape[0],
                use_bf16=use_bf16,
            )
            inputs["input_ids"] = generated_ids
            rollout.update(stats)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            positive = model(**inputs, return_token_losses=True)
        token_loss_sum += float(positive.ce_loss) * int(positive.count)
        token_correct += int(positive.correct)
        token_count += int(positive.count)
        selected = list(range(inputs["input_ids"].shape[0]))
        for modality in ("audio", "text"):
            corruption = build_condition_corruption(
                modality,
                inputs=inputs,
                metadata=metadata,
                names=batch["names"],
                selected_indices=selected,
                alignment=alignment,
                seed=seed,
                epoch=0,
                batch_index=batch_index,
            )
            if not corruption.selected_indices.numel():
                continue
            negative_inputs = corrupted_model_inputs(inputs, corruption)
            mask = corruption.target_mask.index_select(0, corruption.selected_indices)
            positive_loss = positive.per_token_loss.index_select(
                0, corruption.selected_indices
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                negative = model(**negative_inputs, return_token_losses=True)
                _, gap = counterfactual_likelihood_loss(
                    positive_token_loss=positive_loss,
                    negative_token_loss=negative.per_token_loss,
                    target_mask=mask,
                    margin_nats=float(alignment["margin_nats"]),
                )
            gap_sums[modality] += float(gap.sum())
            gap_counts[modality] += int(gap.numel())
    result = {
        "history": "fixed_generated_prefix" if generated_prefix else "teacher_forced",
        "tokens": token_count,
        "cross_entropy": token_loss_sum / max(1, token_count),
        "accuracy": token_correct / max(1, token_count),
        "condition_gap_nats_per_token": {
            modality: gap_sums[modality] / max(1, gap_counts[modality])
            for modality in ("audio", "text")
        },
        "condition_examples": gap_counts,
    }
    if generated_prefix:
        result["rollout"] = {
            "clips": rollout.clips,
            "anchors": rollout.anchors,
            "tokens": rollout.tokens,
            "accuracy": rollout.accuracy,
            "q0_accuracy": rollout.q0_accuracy,
            "mean_confidence": rollout.mean_confidence,
            "mean_entropy": rollout.mean_entropy,
        }
    return result


def main() -> None:
    args = parse_args()
    checkpoint = project_path(args.checkpoint)
    output_json = project_path(args.output_json)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config = source_config(checkpoint)
    alignment = validate_condition_alignment_config(config)
    # Evaluate both modalities even if this checkpoint trained only one.
    alignment["evaluate"] = True
    alignment["eval_modalities"] = ["audio", "text"]
    paths = resolve_data_paths(config)
    data_config = section(config, "data")
    data_config["mimi_codebooks_used"] = mimi_codebooks_from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    dtype = torch.float32 if args.no_bf16 else torch.bfloat16
    model = MimiQwenPlanner.from_pretrained(
        checkpoint, torch_dtype=dtype, local_files_only=True
    )
    model.gradient_checkpointing_disable()
    model.language_model.config.use_cache = True
    device = torch.device(args.device)
    model.to(device).eval()
    text_map = load_text_map(paths["text_json"])
    names = deterministic_subset(
        read_split_names(paths["eval_split"]), args.max_clips, args.subset_seed
    )
    dataset = build_dataset(
        names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=load_neutral_seed(data_config.get("neutral_seed_json")),
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=Step1PlannerCollator(tokenizer.pad_token_id, pad_to_multiple_of=8),
    )
    results = [
        evaluate_history_condition(
            model,
            loader,
            device=device,
            use_bf16=not args.no_bf16,
            alignment=alignment,
            generated_prefix=False,
            seed=args.subset_seed,
        )
    ]
    if not args.skip_generated_prefix:
        results.append(
            evaluate_history_condition(
                model,
                loader,
                device=device,
                use_bf16=not args.no_bf16,
                alignment=alignment,
                generated_prefix=True,
                seed=args.subset_seed,
            )
        )
    payload = {
        "checkpoint": str(checkpoint),
        "clips": len(dataset),
        "audio_corruption": {
            "type": "same_clip_causal_past_shift",
            "shift_anchors": int(alignment["audio_past_shift_anchors"]),
        },
        "text_corruption": "different in-batch transcript resampled to identical token length",
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()


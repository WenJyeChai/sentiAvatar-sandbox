#!/usr/bin/env python3
"""Evaluate the released SentiAvatar Qwen Step 1 planner in its native space."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import read_split_names  # noqa: E402
from utils.inference_math import configure_strict_inference_math  # noqa: E402
from utils.legacy_step1_evaluation import (  # noqa: E402
    LegacyGenerationResult,
    build_legacy_example,
    evaluate_legacy_generations,
    parse_legacy_generated_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the released SentiAvatar Step 1 planner"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "llm",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs",
        help="Root containing legacy motion_token_data and HuBERT token exports.",
    )
    parser.add_argument("--motion_token_dir", type=Path, default=None)
    parser.add_argument("--audio_token_dir", type=Path, default=None)
    parser.add_argument("--text_json", type=Path, default=None)
    parser.add_argument("--validation_split", type=Path, default=None)
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
    parser.add_argument("--max_clips", type=int, default=128, help="0 uses all validation clips")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def resolve_optional(value: Path | None, default: Path) -> Path:
    path = value if value is not None else default
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def deterministic_subset(names: Sequence[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(names):
        return list(names)
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(names), size=limit, replace=False))
    return [str(names[int(index)]) for index in indices]


def load_text_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected name-to-text mapping in {path}")
    return {str(name): str(text) for name, text in payload.items()}


def batched(values: Sequence[Any], size: int):
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    motion_token_dir = resolve_optional(
        args.motion_token_dir, data_dir / "motion_token_data"
    )
    audio_token_dir = resolve_optional(
        args.audio_token_dir, data_dir / "audio_tokens_hubert_layer9_fps10"
    )
    text_json = resolve_optional(args.text_json, data_dir / "text_data" / "motion2text.json")
    validation_split = resolve_optional(
        args.validation_split, data_dir / "split" / "val_file_list.txt"
    )
    required = {
        "released SentiAvatar checkpoint": checkpoint,
        "legacy motion-token directory": motion_token_dir,
        "legacy HuBERT-token directory": audio_token_dir,
        "motion text": text_json,
        "validation split": validation_split,
    }
    missing = {label: str(path) for label, path in required.items() if not path.exists()}
    if missing:
        details = json.dumps(missing, indent=2)
        raise FileNotFoundError(
            "Released SentiAvatar evaluation prerequisites are missing:\n"
            f"{details}\n"
            "Generate the two legacy token exports with "
            "motion_generation/scripts/preprocess_data.py before running this comparison."
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = deterministic_subset(
        read_split_names(validation_split), args.max_clips, args.subset_seed
    )
    text_map = load_text_map(text_json)
    missing_text = [name for name in names if name not in text_map]
    if missing_text:
        raise KeyError(f"Text is missing for {len(missing_text)} clips; first={missing_text[0]}")

    examples = []
    for index, name in enumerate(names, start=1):
        examples.append(
            build_legacy_example(
                name=name,
                text=text_map[name],
                motion_token_dir=motion_token_dir,
                audio_token_dir=audio_token_dir,
                step=args.step,
            )
        )
        if index % 100 == 0:
            print(f"serialized {index}/{len(names)} legacy examples", flush=True)

    device = torch.device(args.device)
    use_bf16 = bool(not args.no_bf16 and device.type == "cuda")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a CUDA device without bf16 support")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    math_mode = configure_strict_inference_math(device)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    model = model.to(device).eval()
    model.config.use_cache = True
    stop_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if stop_id < 0:
        raise ValueError("Released tokenizer does not contain <|im_end|>")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: list[LegacyGenerationResult] = []
    raw_rows: list[dict[str, Any]] = []
    processed = 0
    with torch.inference_mode():
        for batch_examples in batched(examples, args.batch_size):
            encoded = tokenizer(
                [example.prompt for example in batch_examples],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            longest_expected = max(len(example.target_anchors) for example in batch_examples)
            # Each complete anchor uses one [frame_N] plus four [res_Q_ID]
            # tokens. Reserve additional control/EOS space as [STEP_4] is not
            # necessarily represented by one tokenizer ID.
            max_new_tokens = min(args.max_new_tokens, 5 * longest_expected + 32)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_per_clip = (time.perf_counter() - started) / len(batch_examples)
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            for example, token_ids in zip(batch_examples, continuation):
                token_list = token_ids.detach().cpu().tolist()
                predicted_times, predicted = parse_legacy_generated_plan(tokenizer, token_list)
                results.append(
                    LegacyGenerationResult(
                        example=example,
                        predicted_times=predicted_times,
                        predicted_anchors=predicted,
                        elapsed_seconds=elapsed_per_clip,
                    )
                )
                raw_rows.append(
                    {
                        "name": example.name,
                        "sampled_times": list(example.sampled_times),
                        "expected_anchors": int(len(example.target_anchors)),
                        "predicted_times": list(predicted_times),
                        "predicted_anchors": predicted.tolist(),
                        "raw_output": tokenizer.decode(token_list, skip_special_tokens=False),
                    }
                )
            processed += len(batch_examples)
            print(f"legacy generation {processed}/{len(examples)} clips", flush=True)

    measured = evaluate_legacy_generations(results)
    summary = dict(measured["summary"])
    summary.update(
        {
            "checkpoint_path": str(checkpoint),
            "validation_split": str(validation_split),
            "step": args.step,
            "subset_seed": args.subset_seed,
            "seed_protocol": "no motion seed; released interval-end frame labels",
            "decoding": "greedy",
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "math_mode": math_mode,
            "test_split_used": False,
        }
    )
    summary_df = pd.DataFrame([summary])
    clips_df = pd.DataFrame(measured["clip_rows"])
    horizon_df = pd.DataFrame(measured["horizon_rows"])
    summary_df.to_csv(output_dir / "legacy_sentiavatar_rollout.csv", index=False)
    clips_df.to_csv(output_dir / "legacy_sentiavatar_rollout_per_clip.csv", index=False)
    horizon_df.to_csv(output_dir / "legacy_sentiavatar_rollout_horizon.csv", index=False)
    with (output_dir / "legacy_sentiavatar_raw_generations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "legacy_sentiavatar_report.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "checkpoint": str(checkpoint),
                    "motion_token_dir": str(motion_token_dir),
                    "audio_token_dir": str(audio_token_dir),
                    "validation_split": str(validation_split),
                    "clips": len(examples),
                    "step": args.step,
                    "subset_seed": args.subset_seed,
                    "decoding": "greedy",
                    "test_split_used": False,
                },
                "summary": summary,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print("\nReleased SentiAvatar Step 1 generated-rollout evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nWrote legacy comparison outputs: {output_dir}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

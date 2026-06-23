#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Train Step 2 autoregressive motion infill LLM.

This trainer fine-tunes a separate checkpoint from the Step 1 planner LLM:

    input:
        left motion anchor + right motion anchor + middle audio tokens

    target:
        missing middle motion tokens

After training, the output checkpoint can be served with vllm_server.py on a
second port and used as the infill model.

Example:
    python scripts/train_vllm_infill.py \
        --base_model_path ../checkpoints/llm \
        --output_dir ../checkpoints/llm_infill \
        --data_dir ../data \
        --use_lora \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed


# Allow running from either repo root or motion_generation/.
THIS_DIR = Path(__file__).resolve().parent
MOTION_GENERATION_DIR = THIS_DIR.parent
PROJECT_DIR = MOTION_GENERATION_DIR.parent
sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.vllm_infill_model import (  # noqa: E402
    ensure_infill_special_tokens,
    INFILL_ROLE_TOKENS,
    MotionInfillCollator,
    MotionInfillSFTDataset,
    load_tokenizer_for_infill,
    maybe_enable_lora,
)


def format_fps_for_dir(fps: float) -> str:
    """Match preprocess_data.py folder names such as fps10 or fps50."""

    if float(fps).is_integer():
        return str(int(fps))
    return str(fps).replace(".", "p")


@contextmanager
def timed_stage(name: str, enabled: bool = True):
    """Print wall-clock timing for a coarse training setup stage."""

    if not enabled:
        yield
        return

    start = time.perf_counter()
    print(f"[Timing] {name} ...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[Timing] {name}: {elapsed:.3f}s")


def read_split_file(path: Optional[str]) -> Optional[List[str]]:
    """Read a split file containing one sample name per line."""

    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    # Preserve nested relative paths from the dataset split, e.g.
    #   fbx_to_json_data_susu_chonglu/20260115/Human_4_73_02_A
    #
    # Do not use Path(name).stem here: it would drop the parent folders and turn
    # the example above into only "Human_4_73_02_A", which cannot match nested
    # token files under motion_token_data/ or audio_tokens_*/.
    normalized = []
    for name in names:
        name = name.replace("\\", "/").strip().strip("/")
        suffix = Path(name).suffix
        if suffix in {".wav", ".npy", ".json"}:
            name = name[: -len(suffix)]
        normalized.append(name)

    return normalized


def load_token_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load token JSON files from existing preprocessing outputs.

    Expected format:
        {"tokens": ..., "fps": ...}

    A plain list is also accepted for convenience while experimenting.
    """

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"tokens": data}

    raise ValueError(f"Unsupported token JSON format: {path}")


def extract_action_text(raw_text: str) -> Optional[str]:
    """
    Extract the action/expression tag from motion2text.json.

    The dataset text often looks like:
        【表情：认真聆听】【动作：缓慢点头】...

    For infilling, action_text is optional context. If parsing fails, returning
    None is fine; the model can still train from anchors and audio.
    """

    tags = re.findall(r"【(.+?)】", raw_text)
    if not tags:
        return None

    last_tag = tags[-1]

    # If motion says "no action", use an expression tag when available. This
    # follows the same idea as pipeline_infer.py.
    if last_tag == "动作：无动作":
        for tag in tags:
            if tag.startswith("表情：") and tag != "表情：无表情":
                expression = tag.replace("表情：", "")
                return expression if "动作" in expression else f"动作：{expression}"

    return last_tag


def load_action_text_map(path: Optional[str]) -> Dict[str, str]:
    """Load optional name -> action_text metadata."""

    if path is None or not Path(path).exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        motion2text = json.load(f)

    result: Dict[str, str] = {}
    for name, raw_text in motion2text.items():
        action_text = extract_action_text(raw_text)
        if action_text:
            normalized = name.replace("\\", "/").strip().strip("/")
            suffix = Path(normalized).suffix
            if suffix in {".wav", ".npy", ".json"}:
                normalized = normalized[: -len(suffix)]
            result[normalized] = action_text

    return result


def discover_names(
    motion_token_dir: Path,
    audio_token_dir: Path,
    split_names: Optional[Sequence[str]],
) -> List[str]:
    """
    Find sample names that have both motion-token and audio-token files.

    If split_names is provided, preserve that order and filter missing files.
    Otherwise, use the intersection of both directories.
    """

    if split_names is not None:
        names = list(split_names)
    else:
        motion_names = {
            path.relative_to(motion_token_dir).with_suffix("").as_posix()
            for path in motion_token_dir.rglob("*.json")
        }
        audio_names = {
            path.relative_to(audio_token_dir).with_suffix("").as_posix()
            for path in audio_token_dir.rglob("*.json")
        }
        names = sorted(motion_names & audio_names)

    available = []
    for name in names:
        if (motion_token_dir / f"{name}.json").exists() and (
            audio_token_dir / f"{name}.json"
        ).exists():
            available.append(name)

    return available


def load_sequences(
    names: Sequence[str],
    motion_token_dir: Path,
    audio_token_dir: Path,
    action_text_map: Dict[str, str],
    *,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load raw sequences before converting them into infill windows.

    Each returned item is consumed by MotionInfillSFTDataset.
    """

    sequences: List[Dict[str, Any]] = []

    for name in names:
        if max_samples is not None and len(sequences) >= max_samples:
            break

        motion_payload = load_token_json(motion_token_dir / f"{name}.json")
        audio_payload = load_token_json(audio_token_dir / f"{name}.json")

        if not motion_payload or not audio_payload:
            continue

        motion_tokens = motion_payload.get("tokens")
        audio_tokens = audio_payload.get("tokens")

        if not motion_tokens or not audio_tokens:
            continue

        sequences.append(
            {
                "name": name,
                "motion_tokens": motion_tokens,
                "audio_tokens": audio_tokens,
                "motion_fps": motion_payload.get("fps"),
                "audio_fps": audio_payload.get("fps"),
                "action_text": action_text_map.get(name),
            }
        )

    return sequences


def split_train_eval(
    sequences: List[Dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Create an eval split when the user did not provide one."""

    if eval_ratio <= 0 or len(sequences) < 2:
        return sequences, None

    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)

    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_sequences = shuffled[:eval_size]
    train_sequences = shuffled[eval_size:]
    return train_sequences, eval_sequences


def save_lora_adapter_preflight(model, output_dir: str) -> None:
    """
    Verify PEFT adapter checkpointing before spending time training.

    This catches tied-weight/safetensors issues at setup time. It intentionally
    exercises the same model.save_pretrained path Trainer uses for PEFT
    checkpointing, without mutating the model.
    """

    preflight_dir = Path(output_dir) / "_lora_save_preflight"
    if preflight_dir.exists():
        shutil.rmtree(preflight_dir)

    try:
        model.save_pretrained(preflight_dir, safe_serialization=True)
    finally:
        if preflight_dir.exists():
            shutil.rmtree(preflight_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Step 2 autoregressive vLLM-compatible infill LLM"
    )

    # Model paths.
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/llm"),
        help="Step 1 planner LLM checkpoint used to initialize Step 2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/llm_infill"),
        help="Where to save the separately fine-tuned Step 2 infill checkpoint",
    )

    # Data paths.
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data"),
        help="Dataset root containing motion/audio token folders",
    )
    parser.add_argument("--motion_token_dir", type=str, default=None)
    parser.add_argument("--audio_token_dir", type=str, default=None)
    parser.add_argument("--motion2text_json", type=str, default=None)
    parser.add_argument("--train_split_file", type=str, default=None)
    parser.add_argument("--eval_split_file", type=str, default=None)
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="Used only when --eval_split_file is not provided",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_windows_per_sequence",
        type=int,
        default=None,
        help="Limit windows per clip for quick debugging",
    )

    # Task format.
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Sparse keyframe spacing; step=4 predicts 3 middle frames",
    )
    parser.add_argument(
        "--min_history_frames",
        type=int,
        default=0,
        help="Minimum number of GT history frames in each rolling-context prompt",
    )
    parser.add_argument(
        "--max_history_frames",
        type=int,
        default=8,
        help="Maximum number of GT history frames in each rolling-context prompt",
    )
    parser.add_argument(
        "--history_source",
        type=str,
        default="gt",
        choices=["gt"],
        help="History source for baseline training. Later: mixed/generated.",
    )
    parser.add_argument(
        "--audio_fps",
        type=float,
        default=None,
        help="Override audio token fps if JSON metadata is missing",
    )
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=None,
        help="Override motion token fps if JSON metadata is missing",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--debug_tokenization_examples",
        type=int,
        default=0,
        help="Print token-count breakdowns for the first N collated examples.",
    )
    parser.add_argument(
        "--profile_startup",
        action="store_true",
        help="Print wall-clock timings for data/model/trainer setup stages.",
    )
    parser.add_argument(
        "--profile_collator_batches",
        type=int,
        default=0,
        help="Print tokenization/padding timing for the first N collator batches.",
    )

    # Training.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Train all weights. By default, use LoRA if --use_lora is set.",
    )

    # LoRA.
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--save_adapter_only",
        action="store_true",
        help=(
            "With --use_lora, save only the PEFT adapter. By default LoRA is "
            "merged into the base model so output_dir can be served directly "
            "by vllm_server.py."
        ),
    )
    parser.add_argument(
        "--skip_lora_save_preflight",
        action="store_true",
        help="Skip the small PEFT adapter save check before LoRA training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start = time.perf_counter()

    data_dir = Path(args.data_dir)
    motion_token_dir = Path(args.motion_token_dir or data_dir / "motion_token_data")
    default_audio_fps = args.audio_fps if args.audio_fps is not None else 10.0
    audio_fps_tag = format_fps_for_dir(default_audio_fps)
    audio_token_dir = Path(
        args.audio_token_dir or data_dir / f"audio_tokens_hubert_layer9_fps{audio_fps_tag}"
    )
    motion2text_json = args.motion2text_json or str(
        data_dir / "text_data/motion2text.json"
    )

    with timed_stage("load action text map", args.profile_startup):
        action_text_map = load_action_text_map(motion2text_json)

    with timed_stage("read/discover train split", args.profile_startup):
        train_split_names = read_split_file(args.train_split_file)
        train_names = discover_names(
            motion_token_dir, audio_token_dir, train_split_names
        )

    with timed_stage("load train token JSON sequences", args.profile_startup):
        train_sequences = load_sequences(
            train_names,
            motion_token_dir,
            audio_token_dir,
            action_text_map,
            max_samples=args.max_samples,
        )

    eval_sequences = None
    with timed_stage("read/load eval split", args.profile_startup):
        eval_split_names = read_split_file(args.eval_split_file)
        if eval_split_names is not None:
            eval_names = discover_names(
                motion_token_dir, audio_token_dir, eval_split_names
            )
            eval_sequences = load_sequences(
                eval_names,
                motion_token_dir,
                audio_token_dir,
                action_text_map,
                max_samples=args.max_samples,
            )
        else:
            train_sequences, eval_sequences = split_train_eval(
                train_sequences,
                eval_ratio=args.eval_ratio,
                seed=args.seed,
            )

    with timed_stage("build train infill windows", args.profile_startup):
        train_dataset = MotionInfillSFTDataset(
            train_sequences,
            step=args.step,
            audio_fps=args.audio_fps,
            motion_fps=args.motion_fps,
            min_history_frames=args.min_history_frames,
            max_history_frames=args.max_history_frames,
            history_source=args.history_source,
            max_windows_per_sequence=args.max_windows_per_sequence,
            seed=args.seed,
        )
    eval_dataset = None
    if eval_sequences:
        with timed_stage("build eval infill windows", args.profile_startup):
            eval_dataset = MotionInfillSFTDataset(
                eval_sequences,
                step=args.step,
                audio_fps=args.audio_fps,
                motion_fps=args.motion_fps,
                min_history_frames=args.min_history_frames,
                max_history_frames=args.max_history_frames,
                history_source=args.history_source,
                max_windows_per_sequence=args.max_windows_per_sequence,
                seed=args.seed + 1,
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No training windows were built. Check data paths/splits.")

    print("=" * 70)
    print("Step 2 vLLM-compatible infill training")
    print(f"Base model:       {args.base_model_path}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Motion tokens:    {motion_token_dir}")
    print(f"Audio tokens:     {audio_token_dir}")
    print(f"Train sequences:  {len(train_sequences)}")
    print(f"Train windows:    {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval windows:     {len(eval_dataset)}")
    print(f"History frames:   {args.min_history_frames}-{args.max_history_frames} ({args.history_source})")
    print(f"LoRA:             {args.use_lora}")
    print("=" * 70)

    with timed_stage("load tokenizer", args.profile_startup):
        tokenizer = load_tokenizer_for_infill(args.base_model_path)

    collator = MotionInfillCollator(
        tokenizer,
        max_length=args.max_length,
        debug_examples=args.debug_tokenization_examples,
        profile_batches=args.profile_collator_batches,
    )

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    with timed_stage("load base model", args.profile_startup):
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    with timed_stage("ensure infill role tokens", args.profile_startup):
        added_role_tokens = ensure_infill_special_tokens(tokenizer, model)
    if added_role_tokens:
        print(f"Added Step 2 role tokens to tokenizer: {added_role_tokens}")

    if args.use_lora:
        with timed_stage("configure LoRA", args.profile_startup):
            infill_role_token_ids = [
                tokenizer.convert_tokens_to_ids(token)
                for token in INFILL_ROLE_TOKENS
            ]
            if any(
                token_id is None or token_id < 0
                for token_id in infill_role_token_ids
            ):
                raise RuntimeError(
                    "Failed to resolve one or more Step 2 role-token ids after "
                    "adding them to the tokenizer."
                )

            model = maybe_enable_lora(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                trainable_token_indices=infill_role_token_ids,
            )
        # Print PEFT trainable parameter count when available.
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        if not args.skip_lora_save_preflight:
            with timed_stage("PEFT adapter save preflight", args.profile_startup):
                save_lora_adapter_preflight(model, args.output_dir)
            print("PEFT adapter save preflight passed.")
    elif not args.full_finetune:
        print(
            "[WARN] Neither --use_lora nor --full_finetune was set. "
            "Proceeding with full fine-tuning because no adapter was requested."
        )

    training_args_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": "none",
    }

    if eval_dataset is not None and len(eval_dataset) > 0:
        training_args_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": args.eval_steps,
            }
        )

    with timed_stage("create TrainingArguments", args.profile_startup):
        training_args = TrainingArguments(**training_args_kwargs)

    with timed_stage("create Trainer", args.profile_startup):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )

    with timed_stage("trainer.train", True):
        trainer.train()

    # Save model and tokenizer together so vllm_server.py can load this path.
    with timed_stage("save model/tokenizer", True):
        if args.use_lora and not args.save_adapter_only:
            trained_model = trainer.model
            if not hasattr(trained_model, "merge_and_unload"):
                raise RuntimeError(
                    "LoRA model does not expose merge_and_unload(); cannot "
                    "create a standalone vLLM-ready checkpoint. Re-run with "
                    "--save_adapter_only to save the adapter instead."
                )
            merged_lm = trained_model.merge_and_unload()
            if hasattr(merged_lm, "tie_weights"):
                merged_lm.tie_weights()
            merged_lm.save_pretrained(args.output_dir, safe_serialization=True)
            print("Merged LoRA adapter into the base model for vLLM serving.")
        else:
            trainer.save_model(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Step 2 infill checkpoint to: {args.output_dir}")
    print(f"[Timing] total script runtime: {time.perf_counter() - run_start:.3f}s")


if __name__ == "__main__":
    main()

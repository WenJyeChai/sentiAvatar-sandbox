#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Train the compact Step 2 audio-aware causal FIM transformer.

This is the small-model sibling of train_vllm_infill.py. It does not use the
Qwen tokenizer or vLLM checkpoint. It trains AudioFIMCausalLM from scratch with:

    motion tokens: compact RVQ ids, old Step 2 style
    audio:         continuous HuBERT layer9 features
    objective:     causal FIM, predict the middle motion frames only

First target:
    classic gap with step=4:
        left frame t, right frame t+4, predict frames t+1, t+2, t+3.

The explicit [LEN_N] path and --step argument are kept so variable gaps can be
enabled later without changing the checkpoint format.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, set_seed


THIS_DIR = Path(__file__).resolve().parent
MOTION_GENERATION_DIR = THIS_DIR.parent
PROJECT_DIR = MOTION_GENERATION_DIR.parent
sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_fim_causal_model import (  # noqa: E402
    AudioFIMCausalCollator,
    AudioFIMCausalConfig,
    AudioFIMCausalDataset,
    AudioFIMCausalLM,
    AudioFIMTokenMapper,
)


def format_fps_for_dir(fps: float) -> str:
    if float(fps).is_integer():
        return str(int(fps))
    return str(fps).replace(".", "p")


@contextmanager
def timed_stage(name: str, enabled: bool = True):
    if not enabled:
        yield
        return

    start = time.perf_counter()
    print(f"[Timing] {name} ...")
    try:
        yield
    finally:
        print(f"[Timing] {name}: {time.perf_counter() - start:.3f}s")


def read_split_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    normalized = []
    for name in names:
        name = name.replace("\\", "/").strip().strip("/")
        suffix = Path(name).suffix
        if suffix in {".wav", ".npy", ".json"}:
            name = name[: -len(suffix)]
        normalized.append(name)
    return normalized


def load_token_json(path: Path) -> Optional[Dict[str, Any]]:
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
    tags = re.findall(r"ã€(.+?)ã€‘", raw_text)
    if not tags:
        return None

    last_tag = tags[-1]
    if last_tag == "åŠ¨ä½œï¼šæ— åŠ¨ä½œ":
        for tag in tags:
            if tag.startswith("è¡¨æƒ…ï¼š") and tag != "è¡¨æƒ…ï¼šæ— è¡¨æƒ…":
                expression = tag.replace("è¡¨æƒ…ï¼š", "")
                return expression if "åŠ¨ä½œ" in expression else f"åŠ¨ä½œï¼š{expression}"
    return last_tag


def load_action_text_map(path: Optional[str]) -> Dict[str, str]:
    # This compact model does not consume text yet. We still parse the map so
    # sequence metadata stays parallel with train_vllm_infill.py for later use.
    if path is None or not Path(path).exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        motion2text = json.load(f)

    result: Dict[str, str] = {}
    for name, raw_text in motion2text.items():
        action_text = extract_action_text(raw_text)
        if not action_text:
            continue
        normalized = name.replace("\\", "/").strip().strip("/")
        suffix = Path(normalized).suffix
        if suffix in {".wav", ".npy", ".json"}:
            normalized = normalized[: -len(suffix)]
        result[normalized] = action_text
    return result


def discover_names(
    motion_token_dir: Path,
    audio_feat_dir: Path,
    split_names: Optional[Sequence[str]],
) -> List[str]:
    if split_names is not None:
        names = list(split_names)
    else:
        motion_names = {
            path.relative_to(motion_token_dir).with_suffix("").as_posix()
            for path in motion_token_dir.rglob("*.json")
        }
        audio_names = {
            path.relative_to(audio_feat_dir).with_suffix("").as_posix()
            for path in audio_feat_dir.rglob("*.npy")
        }
        names = sorted(motion_names & audio_names)

    available = []
    for name in names:
        if (motion_token_dir / f"{name}.json").exists() and (
            audio_feat_dir / f"{name}.npy"
        ).exists():
            available.append(name)
    return available


def load_sequences(
    names: Sequence[str],
    motion_token_dir: Path,
    audio_feat_dir: Path,
    action_text_map: Dict[str, str],
    *,
    max_samples: Optional[int] = None,
    audio_fps: float = 10.0,
    motion_fps: float = 20.0,
) -> List[Dict[str, Any]]:
    sequences: List[Dict[str, Any]] = []

    for name in names:
        if max_samples is not None and len(sequences) >= max_samples:
            break

        motion_payload = load_token_json(motion_token_dir / f"{name}.json")
        if not motion_payload:
            continue

        motion_tokens = motion_payload.get("tokens")
        if not motion_tokens:
            continue

        audio_path = audio_feat_dir / f"{name}.npy"
        if not audio_path.exists():
            continue

        audio_features = np.load(audio_path).astype(np.float32)
        if audio_features.ndim != 2 or audio_features.shape[0] == 0:
            continue

        sequences.append(
            {
                "name": name,
                "motion_tokens": motion_tokens,
                "audio_features": audio_features,
                "motion_fps": motion_payload.get("fps") or motion_fps,
                "audio_fps": audio_fps,
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
    if eval_ratio <= 0 or len(sequences) < 2:
        return sequences, None

    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[eval_size:], shuffled[:eval_size]


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def run_loss_sanity_check(
    model: AudioFIMCausalLM,
    dataset: AudioFIMCausalDataset,
    collator: AudioFIMCausalCollator,
    *,
    num_examples: int,
    use_bf16: bool,
    use_fp16: bool,
) -> None:
    """
    Print a first-batch CE/logit sanity check before Trainer takes over.

    For vocab_size=2075, all-zero logits should give CE ~= log(2075)=7.637.
    If zero-logit loss is normal but model loss is huge, inspect logit scale.
    If zero-logit loss is huge, labels/shift/vocab are wrong.
    """

    was_training = model.training
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda" and use_bf16:
        dtype = torch.bfloat16
    elif device.type == "cuda" and use_fp16:
        dtype = torch.float16

    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)

    examples = [dataset[i] for i in range(min(num_examples, len(dataset)))]
    batch = collator(examples)
    batch = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }

    outputs = model(**batch)
    logits = outputs.logits.detach().float()
    labels = batch["labels"]
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    supervised = int(valid.sum().item())

    zero_logits = torch.zeros(
        logits[:, :-1, :].shape,
        dtype=torch.float32,
        device=device,
    )
    zero_loss = torch.nn.functional.cross_entropy(
        zero_logits.view(-1, zero_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    target_ids = shift_labels[valid]
    shared_embed_head = (
        model.embed_tokens.weight.data_ptr() == model.out_head.weight.data_ptr()
    )

    print("=" * 70)
    print("[AudioFIM loss sanity]")
    print(f"examples:                    {len(examples)}")
    print(f"seq_len:                     {batch['input_ids'].shape[1]}")
    print(f"audio_bank_len:              {batch['audio_features'].shape[1]}")
    print(f"supervised labels:           {supervised}")
    print(f"target id min/max:           {int(target_ids.min())}/{int(target_ids.max())}")
    print(f"vocab_size:                  {model.config.vocab_size}")
    print(f"expected uniform CE:         {math.log(model.config.vocab_size):.4f}")
    print(f"zero-logit CE:               {float(zero_loss.detach().cpu()):.4f}")
    print(f"model CE:                    {float(outputs.loss.detach().float().cpu()):.4f}")
    print(
        "logits mean/std/min/max:     "
        f"{float(logits.mean().cpu()):.4f}/"
        f"{float(logits.std().cpu()):.4f}/"
        f"{float(logits.min().cpu()):.4f}/"
        f"{float(logits.max().cpu()):.4f}"
    )
    print(f"tie_word_embeddings config:  {model.config.tie_word_embeddings}")
    print(f"embed/out_head share memory: {shared_embed_head}")
    print("=" * 70)

    if was_training:
        model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train compact Step 2 audio-aware causal FIM transformer"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/audio_fim_causal"),
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data"),
        help="Dataset root containing motion_token_data and audio feature dirs",
    )
    parser.add_argument("--motion_token_dir", type=str, default=None)
    parser.add_argument("--audio_feat_dir", type=str, default=None)
    parser.add_argument("--motion2text_json", type=str, default=None)
    parser.add_argument("--train_split_file", type=str, default=None)
    parser.add_argument("--eval_split_file", type=str, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_windows_per_sequence", type=int, default=None)

    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Classic gap uses step=4, predicting 3 middle frames.",
    )
    parser.add_argument("--audio_fps", type=float, default=10.0)
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--min_history_frames", type=int, default=0)
    parser.add_argument("--max_history_frames", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--debug_examples", type=int, default=0)
    parser.add_argument(
        "--debug_loss_sanity",
        type=int,
        default=0,
        help=(
            "Run a first-batch CE/logit sanity check on this many examples "
            "before training. Use this when loss scale looks suspicious."
        ),
    )
    parser.add_argument("--profile_startup", action="store_true")
    parser.add_argument("--profile_collator_batches", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument(
        "--max_gap_frames",
        type=int,
        default=16,
        help="Reserve [LEN_1]..[LEN_N] tokens for future variable-gap training.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start = time.perf_counter()

    if args.step - 1 > args.max_gap_frames:
        raise ValueError("--max_gap_frames must be >= --step - 1")

    data_dir = Path(args.data_dir)
    motion_token_dir = Path(args.motion_token_dir or data_dir / "motion_token_data")
    audio_fps_tag = format_fps_for_dir(args.audio_fps)
    audio_feat_dir = Path(
        args.audio_feat_dir
        or data_dir / f"audio_features_hubert_layer9_fps{audio_fps_tag}"
    )
    motion2text_json = args.motion2text_json or str(
        data_dir / "text_data/motion2text.json"
    )

    with timed_stage("load action text map", args.profile_startup):
        action_text_map = load_action_text_map(motion2text_json)

    with timed_stage("read/discover train split", args.profile_startup):
        train_split_names = read_split_file(args.train_split_file)
        train_names = discover_names(
            motion_token_dir,
            audio_feat_dir,
            train_split_names,
        )

    with timed_stage("load train sequences", args.profile_startup):
        train_sequences = load_sequences(
            train_names,
            motion_token_dir,
            audio_feat_dir,
            action_text_map,
            max_samples=args.max_samples,
            audio_fps=args.audio_fps,
            motion_fps=args.motion_fps,
        )

    eval_sequences = None
    with timed_stage("read/load eval split", args.profile_startup):
        eval_split_names = read_split_file(args.eval_split_file)
        if eval_split_names is not None:
            eval_names = discover_names(
                motion_token_dir,
                audio_feat_dir,
                eval_split_names,
            )
            eval_sequences = load_sequences(
                eval_names,
                motion_token_dir,
                audio_feat_dir,
                action_text_map,
                max_samples=args.max_samples,
                audio_fps=args.audio_fps,
                motion_fps=args.motion_fps,
            )
        else:
            train_sequences, eval_sequences = split_train_eval(
                train_sequences,
                eval_ratio=args.eval_ratio,
                seed=args.seed,
            )

    with timed_stage("build train FIM windows", args.profile_startup):
        train_dataset = AudioFIMCausalDataset(
            train_sequences,
            step=args.step,
            audio_fps=args.audio_fps,
            motion_fps=args.motion_fps,
            min_history_frames=args.min_history_frames,
            max_history_frames=args.max_history_frames,
            max_windows_per_sequence=args.max_windows_per_sequence,
            seed=args.seed,
        )

    eval_dataset = None
    if eval_sequences:
        with timed_stage("build eval FIM windows", args.profile_startup):
            eval_dataset = AudioFIMCausalDataset(
                eval_sequences,
                step=args.step,
                audio_fps=args.audio_fps,
                motion_fps=args.motion_fps,
                min_history_frames=args.min_history_frames,
                max_history_frames=args.max_history_frames,
                max_windows_per_sequence=args.max_windows_per_sequence,
                seed=args.seed + 1,
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No training windows were built. Check data paths/splits.")

    config = AudioFIMCausalConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        audio_feat_dim=args.audio_feat_dim,
        max_gap_frames=args.max_gap_frames,
        dropout=args.dropout,
    )
    model = AudioFIMCausalLM(config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params, trainable_params = count_parameters(model)

    print("=" * 70)
    print("Step 2 compact AudioFIM causal training")
    print(f"Output dir:       {args.output_dir}")
    print(f"Motion tokens:    {motion_token_dir}")
    print(f"Audio features:   {audio_feat_dir}")
    print(f"Train sequences:  {len(train_sequences)}")
    print(f"Train windows:    {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval windows:     {len(eval_dataset)}")
    print(f"Gap setup:        step={args.step}, predict={args.step - 1} frames")
    print(f"History frames:   {args.min_history_frames}-{args.max_history_frames}")
    print(
        "Architecture:     "
        f"L={config.num_layers}, H={config.hidden_size}, "
        f"heads={config.num_heads}, ffn={config.intermediate_size}, "
        f"vocab={config.vocab_size}"
    )
    print(f"Parameters:       {total_params:,} total / {trainable_params:,} trainable")
    print(f"Tie embeddings:   {config.tie_word_embeddings}")
    print(
        "Shared emb/head:  "
        f"{model.embed_tokens.weight.data_ptr() == model.out_head.weight.data_ptr()}"
    )
    print("=" * 70)

    collator = AudioFIMCausalCollator(
        config,
        max_length=args.max_length,
        debug_examples=args.debug_examples,
        profile_batches=args.profile_collator_batches,
    )

    if args.debug_loss_sanity > 0:
        run_loss_sanity_check(
            model,
            train_dataset,
            collator,
            num_examples=args.debug_loss_sanity,
            use_bf16=args.bf16,
            use_fp16=args.fp16,
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
        )

    with timed_stage("trainer.train", True):
        trainer.train()

    with timed_stage("save model/token map", True):
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero():
            mapper = AudioFIMTokenMapper(config)
            mapper.save_json(Path(args.output_dir) / "compact_token_map.json")

    print(f"Saved compact AudioFIM checkpoint to: {args.output_dir}")
    print(f"[Timing] total script runtime: {time.perf_counter() - run_start:.3f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the Phase 1 causal Mimi -> fixed-gap multipart anchor planner."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    MimiQwenPlanner,
    MimiQwenPlannerConfig,
    Step1FixedGapDataset,
    Step1PlannerCollator,
    load_text_map,
    read_split_names,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_SLOTS,
    MIMI_FRAME_TOKEN,
    ensure_step1_special_tokens,
    motion_token_id_table,
    validate_anchor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed [gap_3] multipart Step 1 planner")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "motion_generation" / "configs" / "step1_multipart_fixed_gap3.yaml",
    )
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_eval_clips", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return dict(value)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Phase 1 training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return distributed, rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_neutral_seed(path_value: Optional[str]) -> Optional[tuple[int, ...]]:
    if not path_value:
        return None
    path = project_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tokens", payload.get("anchor_tokens"))
    if not isinstance(payload, list):
        raise ValueError(f"Neutral seed JSON must contain a 16-id list: {path}")
    tokens = tuple(int(value) for value in payload)
    validate_anchor(tokens)
    return tokens


def resolve_data_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = section(config, "paths")
    data_dir = project_path(paths.get("data_dir", "SuSuInterActs/SuSuInterActs"))
    return {
        "base_model": project_path(paths.get("base_model", "checkpoints/llm")),
        "output_dir": project_path(paths.get("output_dir", "checkpoints/step1_multipart_fixed_gap3")),
        "data_dir": data_dir,
        "motion_token_dir": project_path(
            paths.get("motion_token_dir", data_dir / "motion_token_data_multipart_causal_512x4")
        ),
        "mimi_token_dir": project_path(
            paths.get("mimi_token_dir", data_dir / "audio_tokens_mimi_12p5hz_8cb")
        ),
        "text_json": project_path(paths.get("text_json", data_dir / "text_data/motion2text.json")),
        "train_split": project_path(paths.get("train_split", data_dir / "split/train_file_list.txt")),
        "eval_split": project_path(paths.get("eval_split", data_dir / "split/val_file_list.txt")),
    }


def validate_paths(paths: Mapping[str, Path], resume: Optional[Path]) -> None:
    files = ("text_json", "train_split", "eval_split")
    directories = ("motion_token_dir", "mimi_token_dir")
    if resume is None and not paths["base_model"].is_dir():
        raise FileNotFoundError(f"Base Qwen checkpoint not found: {paths['base_model']}")
    for key in files:
        if not paths[key].is_file():
            raise FileNotFoundError(f"Required {key} not found: {paths[key]}")
    for key in directories:
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Required {key} not found: {paths[key]}")


def build_model_and_tokenizer(
    *,
    base_model: Path,
    resume: Optional[Path],
    dtype: torch.dtype,
) -> tuple[MimiQwenPlanner, Any, list[str]]:
    if resume is not None:
        resume = resume.resolve()
        tokenizer = AutoTokenizer.from_pretrained(resume, local_files_only=True)
        model = MimiQwenPlanner.from_pretrained(
            resume,
            torch_dtype=dtype,
            local_files_only=True,
        )
        added = ensure_step1_special_tokens(tokenizer)
        if added:
            raise RuntimeError(f"Resume checkpoint is missing Step 1 controls: {added}")
        expected_table = torch.tensor(motion_token_id_table(tokenizer), dtype=torch.long)
        if not torch.equal(model.motion_token_ids.cpu(), expected_table):
            raise RuntimeError("Resume checkpoint motion-token classifier table does not match tokenizer")
        return model, tokenizer, []

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    language_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
    )
    added = ensure_step1_special_tokens(tokenizer, language_model)
    language_model.config.use_cache = False
    table = motion_token_id_table(tokenizer)
    audio_placeholder_id = int(tokenizer.convert_tokens_to_ids(MIMI_FRAME_TOKEN))
    planner_config = MimiQwenPlannerConfig(
        language_model_config=language_model.config.to_dict(),
        audio_placeholder_id=audio_placeholder_id,
        motion_token_ids=table,
        mimi_cardinality=2_048,
        mimi_codebooks_stored=8,
        mimi_codebooks_used=[0],
    )
    model = MimiQwenPlanner(planner_config, language_model=language_model)
    model.tie_weights()
    return model, tokenizer, added


def build_dataset(
    names: list[str],
    *,
    tokenizer: Any,
    paths: Mapping[str, Path],
    text_map: Mapping[str, str],
    data_config: Mapping[str, Any],
    neutral_seed: Optional[tuple[int, ...]],
    training: bool,
) -> Step1FixedGapDataset:
    generated_anchor_dir_value = data_config.get("generated_anchor_dir") if training else None
    generated_anchor_dir = project_path(generated_anchor_dir_value) if generated_anchor_dir_value else None
    return Step1FixedGapDataset(
        names,
        tokenizer=tokenizer,
        motion_token_dir=paths["motion_token_dir"],
        mimi_token_dir=paths["mimi_token_dir"],
        text_map=text_map,
        fixed_gap=int(data_config.get("fixed_gap", 3)),
        max_length=int(data_config.get("max_length", 2_048)),
        seed_mode=str(data_config.get("seed_mode", "observed")),
        neutral_seed_tokens=neutral_seed,
        neutral_seed_probability=float(data_config.get("neutral_seed_probability", 0.5)),
        previous_seed_probability=float(data_config.get("previous_seed_probability", 0.5)),
        generated_anchor_dir=generated_anchor_dir,
        generated_prefix_probability=(
            float(data_config.get("generated_prefix_probability", 0.0)) if training else 0.0
        ),
        random_seed=int(data_config.get("random_seed", 42)),
        require_causal_motion=bool(data_config.get("require_causal_motion", True)),
        max_duration_mismatch_seconds=float(data_config.get("max_duration_mismatch_seconds", 0.12)),
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "attention_mask", "audio_codes", "target_slots", "motion_local_labels")
    return {key: batch[key].to(device=device, non_blocking=True) for key in keys}


def reduce_sums(values: torch.Tensor, distributed: bool) -> torch.Tensor:
    values = values.detach()
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    distributed: bool,
    use_bf16: bool,
) -> dict[str, Any]:
    model.eval()
    totals = torch.zeros(3 + 2 * len(BODY_SLOTS), dtype=torch.float64, device=device)
    autocast_enabled = use_bf16 and device.type == "cuda"
    for batch in loader:
        inputs = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model(**inputs)
        count = output.count.to(torch.float64)
        totals[0] += output.loss.detach().to(torch.float64) * count
        totals[1] += output.correct.to(torch.float64)
        totals[2] += count
        totals[3 : 3 + len(BODY_SLOTS)] += output.per_slot_correct.to(torch.float64)
        totals[3 + len(BODY_SLOTS) :] += output.per_slot_count.to(torch.float64)
    totals = reduce_sums(totals, distributed)
    model.train()
    denominator = max(1.0, float(totals[2]))
    slot_correct = totals[3 : 3 + len(BODY_SLOTS)]
    slot_count = totals[3 + len(BODY_SLOTS) :]
    per_slot = {}
    for slot, spec in enumerate(BODY_SLOTS):
        per_slot[f"{spec.part}_q{spec.quantizer}"] = (
            float(slot_correct[slot]) / max(1.0, float(slot_count[slot]))
        )
    return {
        "loss": float(totals[0]) / denominator,
        "slot_accuracy": float(totals[1]) / denominator,
        "per_slot_accuracy": per_slot,
    }


def optimizer_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    output_dir: Path,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    source_config: Mapping[str, Any],
    distributed: bool,
    rank: int,
    final: bool = False,
    checkpoint_name: Optional[str] = None,
    best_eval_loss: float = math.inf,
    epochs_without_improvement: int = 0,
) -> Path:
    if final and checkpoint_name is not None:
        raise ValueError("final and checkpoint_name cannot both be set")
    checkpoint_dir = output_dir / (
        checkpoint_name or ("final" if final else f"checkpoint-{global_step}")
    )
    if is_main(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "best_eval_loss": float(best_eval_loss),
                "epochs_without_improvement": int(epochs_without_improvement),
            },
            checkpoint_dir / "training_state.pt",
        )
        (checkpoint_dir / "phase1_source_config.json").write_text(
            json.dumps(source_config, indent=2), encoding="utf-8"
        )
        (output_dir / "latest_checkpoint.txt").write_text(str(checkpoint_dir), encoding="utf-8")
        print(f"Saved checkpoint: {checkpoint_dir}")
    barrier(distributed)
    return checkpoint_dir


def load_training_state(
    checkpoint: Optional[Path],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> tuple[int, int, int, float, int]:
    if checkpoint is None:
        return 0, 0, 0, math.inf, 0
    state_path = checkpoint.resolve() / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume training state not found: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(state["scheduler"])
    return (
        int(state["global_step"]),
        int(state["epoch"]),
        int(state["batch_in_epoch"]),
        float(state.get("best_eval_loss", math.inf)),
        int(state.get("epochs_without_improvement", 0)),
    )


def initialize_wandb(
    config: Mapping[str, Any], *, rank: int, output_dir: Path
):
    monitoring = section(config, "monitoring")
    if not is_main(rank) or str(monitoring.get("report_to", "none")) != "wandb":
        return None
    import wandb  # pylint: disable=import-outside-toplevel

    mode = str(monitoring.get("wandb_mode", "online"))
    return wandb.init(
        project=str(monitoring.get("wandb_project", "sentiavatar-step1")),
        entity=monitoring.get("wandb_entity"),
        name=str(monitoring.get("wandb_run_name", "step1_multipart_fixed_gap3")),
        tags=list(monitoring.get("wandb_tags", ["phase1", "step1", "mimi", "fixed-gap3"])),
        mode=mode,
        dir=str(output_dir),
        config=dict(config),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    paths = resolve_data_paths(config)
    distributed, rank, local_rank, world_size, device = setup_distributed()
    training = section(config, "training")
    data_config = section(config, "data")
    seed = int(training.get("seed", 42))
    seed_everything(seed, rank)
    resume = args.resume_from_checkpoint.resolve() if args.resume_from_checkpoint else None
    validate_paths(paths, resume)

    use_bf16 = bool(training.get("bf16", True))
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but this CUDA device does not support it")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    if is_main(rank):
        print(f"Loading tokenizer/model on rank {rank}, device={device}, dtype={dtype}")
    model, tokenizer, added_tokens = build_model_and_tokenizer(
        base_model=paths["base_model"],
        resume=resume,
        dtype=dtype,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.language_model.config.use_cache = False
    model.to(device)

    text_map = load_text_map(paths["text_json"])
    train_names = read_split_names(paths["train_split"])
    eval_names = read_split_names(paths["eval_split"])
    if args.max_train_clips is not None:
        train_names = train_names[: args.max_train_clips]
    if args.max_eval_clips is not None:
        eval_names = eval_names[: args.max_eval_clips]
    neutral_seed = load_neutral_seed(data_config.get("neutral_seed_json"))
    data_config["random_seed"] = seed
    train_dataset = build_dataset(
        train_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=neutral_seed,
        training=True,
    )
    eval_dataset = build_dataset(
        eval_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=neutral_seed,
        training=False,
    )
    # Fail before DDP training if the serialization contract is broken.
    if len(train_dataset):
        first = train_dataset[0]
        if is_main(rank):
            print(
                "First serialized train example:",
                {
                    "name": first["name"],
                    "sequence_length": len(first["input_ids"]),
                    "anchor_times": first["anchor_times"],
                    "audio_boundaries": first["audio_boundaries"],
                    "supervised_tokens": sum(slot >= 0 for slot in first["target_slots"]),
                },
            )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    ) if distributed else None
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None
    collator = Step1PlannerCollator(tokenizer.pad_token_id, pad_to_multiple_of=8)
    workers = int(data_config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("per_device_train_batch_size", 2)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training.get("per_device_eval_batch_size", 2)),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collator,
    )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    accumulation = int(training.get("gradient_accumulation_steps", 8))
    epochs = int(training.get("num_train_epochs", 10))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * epochs
    warmup_steps = round(total_steps * float(training.get("warmup_ratio", 0.03)))
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(training.get("weight_decay", 0.01))),
        lr=float(training.get("learning_rate", 2e-5)),
        betas=tuple(float(v) for v in training.get("adam_betas", [0.9, 0.95])),
        eps=float(training.get("adam_epsilon", 1e-8)),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    (
        global_step,
        start_epoch,
        resume_batch,
        best_eval_loss,
        epochs_without_improvement,
    ) = load_training_state(resume, optimizer, scheduler, device)

    if is_main(rank):
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print("=" * 76)
        print("Phase 1 fixed-gap multipart planner")
        print(f"Base/resume:       {resume or paths['base_model']}")
        print(f"Output:            {paths['output_dir']}")
        print(f"Motion tokens:     {paths['motion_token_dir']}")
        print(f"Mimi tokens:       {paths['mimi_token_dir']}")
        print(f"Train/eval clips:  {len(train_dataset)}/{len(eval_dataset)}")
        print(f"World size:        {world_size}")
        print(f"Added controls:    {added_tokens}")
        print(f"Parameters:        {trainable:,} trainable / {total:,} total")
        print(f"Updates:           {total_steps} ({updates_per_epoch}/epoch)")
        print("=" * 76)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(config, rank=rank, output_dir=paths["output_dir"])
    log_steps = int(training.get("logging_steps", 20))
    eval_steps = int(training.get("eval_steps", 500))
    save_steps = int(training.get("save_steps", 500))
    save_best = bool(training.get("save_best", True))
    early_stopping_patience = int(training.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0))
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    autocast_enabled = use_bf16 and device.type == "cuda"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = torch.zeros(3 + 2 * len(BODY_SLOTS), dtype=torch.float64, device=device)
    run_start = time.perf_counter()

    completed_epochs = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_batch_start = resume_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < epoch_batch_start:
                continue
            group_start = (batch_index // accumulation) * accumulation
            group_end = min(group_start + accumulation, len(train_loader))
            group_size = group_end - group_start
            should_step = batch_index + 1 == group_end
            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_step
                else contextlib.nullcontext()
            )
            inputs = move_batch(batch, device)
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    output = model(**inputs)
                    scaled_loss = output.loss / group_size
                scaled_loss.backward()
            count = output.count.detach().to(torch.float64)
            running[0] += output.loss.detach().to(torch.float64) * count
            running[1] += output.correct.detach().to(torch.float64)
            running[2] += count
            running[3 : 3 + len(BODY_SLOTS)] += output.per_slot_correct.detach().to(torch.float64)
            running[3 + len(BODY_SLOTS) :] += output.per_slot_count.detach().to(torch.float64)

            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_steps == 0:
                totals = reduce_sums(running.clone(), distributed)
                denominator = max(1.0, float(totals[2]))
                if is_main(rank):
                    train_metrics = {
                        "train/loss": float(totals[0]) / denominator,
                        "train/slot_accuracy": float(totals[1]) / denominator,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (batch_index + 1) / max(1, len(train_loader)),
                    }
                    print(
                        f"step={global_step}/{total_steps} epoch={epoch + 1}/{epochs} "
                        f"loss={train_metrics['train/loss']:.5f} "
                        f"slot_acc={train_metrics['train/slot_accuracy']:.4%} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"elapsed={time.perf_counter() - run_start:.1f}s"
                    )
                    if wandb_run is not None:
                        wandb_run.log(train_metrics, step=global_step)
                running.zero_()

            if eval_steps > 0 and global_step % eval_steps == 0:
                eval_metrics = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    distributed=distributed,
                    use_bf16=use_bf16,
                )
                if is_main(rank):
                    print(
                        f"[eval] step={global_step} loss={eval_metrics['loss']:.5f} "
                        f"slot_acc={eval_metrics['slot_accuracy']:.4%}"
                    )
                    print("[eval slots]", json.dumps(eval_metrics["per_slot_accuracy"], sort_keys=True))
                    if wandb_run is not None:
                        wandb_payload = {
                            "eval/loss": eval_metrics["loss"],
                            "eval/slot_accuracy": eval_metrics["slot_accuracy"],
                        }
                        wandb_payload.update(
                            {
                                f"eval_slot/{name}": value
                                for name, value in eval_metrics["per_slot_accuracy"].items()
                            }
                        )
                        wandb_run.log(wandb_payload, step=global_step)

            if save_steps > 0 and global_step % save_steps == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    output_dir=paths["output_dir"],
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_index + 1,
                    source_config=config,
                    distributed=distributed,
                    rank=rank,
                    best_eval_loss=best_eval_loss,
                    epochs_without_improvement=epochs_without_improvement,
                )
        resume_batch = 0

        eval_metrics = evaluate(
            model,
            eval_loader,
            device=device,
            distributed=distributed,
            use_bf16=use_bf16,
        )
        if is_main(rank):
            print(
                f"[epoch eval] epoch={epoch + 1} loss={eval_metrics['loss']:.5f} "
                f"slot_acc={eval_metrics['slot_accuracy']:.4%}"
            )
            print("[epoch eval slots]", json.dumps(eval_metrics["per_slot_accuracy"], sort_keys=True))
            if wandb_run is not None:
                wandb_payload = {
                    "eval/loss": eval_metrics["loss"],
                    "eval/slot_accuracy": eval_metrics["slot_accuracy"],
                    "eval/completed_epoch": epoch + 1,
                }
                wandb_payload.update(
                    {
                        f"eval_slot/{name}": value
                        for name, value in eval_metrics["per_slot_accuracy"].items()
                    }
                )
                wandb_run.log(wandb_payload, step=global_step)

        completed_epochs = epoch + 1
        improved = eval_metrics["loss"] < best_eval_loss - early_stopping_min_delta
        if improved:
            best_eval_loss = float(eval_metrics["loss"])
            epochs_without_improvement = 0
            if save_best:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    output_dir=paths["output_dir"],
                    global_step=global_step,
                    epoch=completed_epochs,
                    batch_in_epoch=0,
                    source_config=config,
                    distributed=distributed,
                    rank=rank,
                    checkpoint_name="best",
                    best_eval_loss=best_eval_loss,
                    epochs_without_improvement=epochs_without_improvement,
                )
        else:
            epochs_without_improvement += 1
        if is_main(rank):
            print(
                f"[early stopping] best_loss={best_eval_loss:.5f} "
                f"epochs_without_improvement={epochs_without_improvement}/"
                f"{early_stopping_patience or 'disabled'}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/best_loss": best_eval_loss,
                        "eval/epochs_without_improvement": epochs_without_improvement,
                    },
                    step=global_step,
                )
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            if is_main(rank):
                print(f"Early stopping after {completed_epochs} completed epochs")
            break

    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        output_dir=paths["output_dir"],
        global_step=global_step,
        epoch=completed_epochs,
        batch_in_epoch=0,
        source_config=config,
        distributed=distributed,
        rank=rank,
        final=True,
        best_eval_loss=best_eval_loss,
        epochs_without_improvement=epochs_without_improvement,
    )
    if is_main(rank):
        suffix = " (early stopped)" if stopped_early else ""
        print(f"Training complete{suffix} in {time.perf_counter() - run_start:.1f}s")
        if wandb_run is not None:
            wandb_run.finish()
    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

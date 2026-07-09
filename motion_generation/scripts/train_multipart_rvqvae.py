#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected but keep CLI robust.
    tqdm = None


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
sys.path.insert(0, str(MODULE_DIR))

from models.multipart_rvqvae import MultiPartRVQVAE  # noqa: E402
from utils.multipart_motion import (  # noqa: E402
    PART_DIMS,
    PART_ORDER,
    PartNormalizer,
    compute_part_normalizer,
    crop_or_pad_parts,
    load_motion_dict,
    load_name_list,
    motion_path_for_name,
    split_motion_parts,
)


class MultipartMotionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        split_file: Optional[Path],
        normalizer: PartNormalizer,
        window_size: int,
        root_abs_threshold: float,
        part_order,
        train: bool,
        max_clips: Optional[int] = None,
    ) -> None:
        self.data_root = data_root
        self.motion_dir = data_root / "motion_data"
        self.split_file = split_file or data_root / "split" / f"{split}_file_list.txt"
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        names = load_name_list(self.split_file)
        existing = [name for name in names if motion_path_for_name(self.motion_dir, name).exists()]
        if max_clips is not None:
            existing = existing[:max_clips]
        if not existing:
            raise RuntimeError(f"No usable motion files found for split '{split}'")

        self.names = existing
        self.normalizer = normalizer
        self.window_size = int(window_size)
        self.root_abs_threshold = float(root_abs_threshold)
        self.part_order = tuple(part_order)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        name = self.names[idx]
        motion = load_motion_dict(motion_path_for_name(self.motion_dir, name))
        parts, meta = split_motion_parts(motion, abs_threshold=self.root_abs_threshold)
        frames = min(value.shape[0] for value in parts.values())

        if self.train and frames > self.window_size:
            start = random.randint(0, frames - self.window_size)
        else:
            start = max(0, (frames - self.window_size) // 2)

        cropped = crop_or_pad_parts(parts, self.window_size, start=start)
        tensors = {
            part: torch.from_numpy(self.normalizer.normalize(part, cropped[part]))
            for part in self.part_order
        }
        return {
            "parts": tensors,
            "name": name,
            "root_schema": meta["root_schema"],
        }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args: argparse.Namespace) -> torch.device:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1

    requested_device = torch.device(args.device)
    if args.distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return device

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return requested_device


def cleanup_distributed(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    return int(getattr(args, "rank", 0)) == 0


def barrier(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.barrier()


def average_totals(
    totals: Mapping[str, float],
    count: int,
    device: torch.device,
    distributed: bool,
) -> Dict[str, float]:
    keys = sorted(totals.keys())
    values = [float(count)] + [float(totals[key]) for key in keys]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(float(tensor[0].item()), 1.0)
    return {key: float(tensor[i + 1].item() / total_count) for i, key in enumerate(keys)}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def move_parts(parts: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {part: value.to(device, non_blocking=True) for part, value in parts.items()}


def part_velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def compute_losses(
    inputs: Mapping[str, torch.Tensor],
    output: Mapping[str, object],
    rec_weight: float,
    vel_weight: float,
    commit_weight: float,
    part_order,
) -> Dict[str, torch.Tensor]:
    rec = output["rec"]
    commit = output["commit_loss"]
    perplexity = output["perplexity"]
    rec_terms = []
    vel_terms = []
    commit_terms = []
    metrics: Dict[str, torch.Tensor] = {}

    for part in part_order:
        pred = rec[part]
        target = inputs[part]
        frames = min(pred.shape[1], target.shape[1])
        pred = pred[:, :frames]
        target = target[:, :frames]
        part_rec = F.smooth_l1_loss(pred, target)
        part_vel = part_velocity_loss(pred, target)
        part_commit = commit[part]
        part_total = rec_weight * part_rec + vel_weight * part_vel + commit_weight * part_commit
        rec_terms.append(part_rec)
        vel_terms.append(part_vel)
        commit_terms.append(part_commit)
        metrics[f"{part}/loss"] = part_total.detach()
        metrics[f"{part}/rec"] = part_rec.detach()
        metrics[f"{part}/vel"] = part_vel.detach()
        metrics[f"{part}/commit"] = part_commit.detach()
        metrics[f"{part}/perplexity"] = perplexity[part].detach()

    rec_loss = torch.stack(rec_terms).mean()
    vel_loss = torch.stack(vel_terms).mean()
    commit_loss = torch.stack(commit_terms).mean()
    total = rec_weight * rec_loss + vel_weight * vel_loss + commit_weight * commit_loss
    metrics.update({
        "total": total,
        "loss": total.detach(),
        "rec": rec_loss.detach(),
        "vel": vel_loss.detach(),
        "commit": commit_loss.detach(),
    })
    return metrics


def run_epoch(
    model: MultiPartRVQVAE,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    epoch: int = 0,
    split: str = "train",
    wandb_run=None,
    state: Optional[Dict[str, int]] = None,
    on_step_end=None,
    progress=None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    count = 0
    start_time = time.time()
    model_core = unwrap_model(model)
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    for step, batch in enumerate(loader, start=1):
        inputs = move_parts(batch["parts"], device)
        with torch.set_grad_enabled(is_train):
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with amp_context:
                output = model(inputs)
                losses = compute_losses(
                    inputs,
                    output,
                    rec_weight=args.rec_weight,
                    vel_weight=args.vel_weight,
                    commit_weight=args.commit_weight,
                    part_order=model_core.part_order,
                )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if state is not None:
                    state["global_step"] = int(state.get("global_step", 0)) + 1

        batch_size = next(iter(inputs.values())).shape[0]
        count += batch_size
        for key in losses:
            totals.setdefault(key, 0.0)
            totals[key] += float(losses[key].item()) * batch_size

        global_step = int(state.get("global_step", step) if state is not None else step)
        if is_train and progress is not None:
            progress.update(1)
            progress.set_postfix(
                {
                    "gstep": global_step,
                    "epoch": f"{epoch}/{args.epochs}",
                    "loss": f"{float(losses['loss'].item()):.4f}",
                    "rec": f"{float(losses['rec'].item()):.4f}",
                    "vel": f"{float(losses['vel'].item()):.4f}",
                }
            )
        should_log = is_train and args.log_every > 0 and global_step % args.log_every == 0
        if should_log:
            elapsed = time.time() - start_time
            avg = average_totals(
                totals,
                count,
                device=device,
                distributed=getattr(args, "distributed", False),
            )
            if is_main_process(args):
                message = (
                    f"global_step {global_step:07d} epoch_step {step:05d}/{len(loader):05d} "
                    f"loss={avg['total']:.5f} rec={avg['rec']:.5f} "
                    f"vel={avg['vel']:.5f} commit={avg['commit']:.5f} "
                    f"time={elapsed:.1f}s"
                )
                if progress is not None:
                    progress.write(message)
                else:
                    print(message)
            if wandb_run is not None and is_main_process(args):
                payload = {f"{split}/{key}": value for key, value in avg.items()}
                payload["epoch"] = epoch
                payload[f"{split}/examples"] = count
                if is_train:
                    payload["train/lr"] = optimizer.param_groups[0]["lr"]
                wandb_run.log(payload, step=global_step)

        if is_train and on_step_end is not None:
            on_step_end(global_step)
            model.train(True)

        if args.dry_run_batches and step >= args.dry_run_batches:
            break

    return average_totals(
        totals,
        count,
        device=device,
        distributed=getattr(args, "distributed", False),
    )


def save_checkpoint(
    path: Path,
    model: MultiPartRVQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    epoch: int,
    args: argparse.Namespace,
    normalizer_path: Path,
    best_val: Optional[float],
    global_step: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_core = unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_core.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_core.config_dict(),
            "args": vars(args),
            "normalizer_path": str(normalizer_path),
            "best_val": best_val,
            "global_step": global_step,
        },
        path,
    )


def prune_step_checkpoints(model_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints = sorted(
        model_dir.glob("step_*.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    excess = len(checkpoints) - save_total_limit
    for path in checkpoints[: max(0, excess)]:
        path.unlink(missing_ok=True)


def build_model(args: argparse.Namespace) -> MultiPartRVQVAE:
    return MultiPartRVQVAE(
        part_dims=PART_DIMS,
        part_order=args.parts,
        nb_code=args.codebook_size,
        code_dim=args.code_dim,
        num_quantizers=args.num_quantizers,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.activation,
        norm=args.norm,
        vq_cnn_depth=args.vq_cnn_depth,
        shared_codebook=args.shared_codebook,
        quantize_dropout_prob=args.quantize_dropout_prob,
        quantize_dropout_cutoff_index=args.quantize_dropout_cutoff_index,
        mu=args.mu,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_training_steps: int,
) -> Optional[LambdaLR]:
    if args.lr_scheduler_type == "constant":
        return None
    if total_training_steps <= 0:
        return None

    warmup_steps = int(args.warmup_steps)
    if args.warmup_ratio > 0:
        warmup_steps = max(warmup_steps, int(total_training_steps * args.warmup_ratio))
    warmup_steps = min(max(warmup_steps, 0), total_training_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if args.lr_scheduler_type == "cosine":
            min_factor = float(args.min_lr_ratio)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine
        raise ValueError(f"Unsupported lr_scheduler_type: {args.lr_scheduler_type}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _wandb_config(args: argparse.Namespace) -> Dict[str, object]:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    config["part_order"] = list(args.parts)
    config["part_dims"] = {part: PART_DIMS[part] for part in args.parts}
    return config


def init_wandb(args: argparse.Namespace, run_dir: Path):
    report_to = set(args.report_to or [])
    if not (args.wandb or "wandb" in report_to):
        return None
    if not is_main_process(args):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in this Python environment. Install it or run without --wandb."
        ) from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name or args.experiment_name,
        "dir": str(run_dir),
        "config": _wandb_config(args),
    }
    if args.wandb_tags:
        init_kwargs["tags"] = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def prepare_normalizer(args: argparse.Namespace, run_dir: Path) -> PartNormalizer:
    data_root = args.data_root.resolve()
    motion_dir = data_root / "motion_data"
    split_file = (
        args.train_split_file.resolve()
        if args.train_split_file is not None
        else data_root / "split" / f"{args.train_split}_file_list.txt"
    )
    names = load_name_list(split_file)

    if args.normalizer_path is not None:
        return PartNormalizer.load(args.normalizer_path)

    meta_dir = run_dir / "meta"
    normalizer_path = meta_dir / "normalizer.npz"
    if getattr(args, "distributed", False) and not is_main_process(args):
        barrier(args)
        return PartNormalizer.load(normalizer_path)

    if normalizer_path.exists() and not args.recompute_stats:
        normalizer = PartNormalizer.load(normalizer_path)
        barrier(args)
        return normalizer

    print("Computing part normalizer from training split...")
    normalizer, metadata = compute_part_normalizer(
        motion_dir=motion_dir,
        names=names,
        abs_threshold=args.root_abs_threshold,
        max_clips=args.max_stat_clips,
    )
    metadata.update(
        {
            "source_data_root": str(data_root),
            "train_split": args.train_split,
            "note": "Root channel canonicalized to per-frame delta before stats.",
        }
    )
    normalizer.save(normalizer_path, metadata=metadata)
    print(f"Saved normalizer: {normalizer_path}")
    print(f"Root schema counts: {metadata['schema_counts']}")
    barrier(args)
    return normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-part RVQ-VAE for SentiAvatar motion.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs",
        help="Dataset root containing motion_data/ and split/.",
    )
    parser.add_argument("--data_dir", type=Path, default=None, help="Alias for --data_root.")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--train_split_file", type=Path, default=None)
    parser.add_argument("--eval_split_file", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "multipart_rvqvae",
    )
    parser.add_argument("--experiment_name", type=str, default="parts_512x4")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=list(PART_ORDER),
        help=(
            "Motion parts to train. Use one or more of: upper lower feet hands, "
            "or use 'all'. Examples: --parts hands  |  --parts upper lower"
        ),
    )
    parser.add_argument("--normalizer_path", type=Path, default=None)
    parser.add_argument("--recompute_stats", action="store_true")
    parser.add_argument("--root_abs_threshold", type=float, default=10.0)

    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--max_stat_clips", type=int, default=None)

    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--code_dim", type=int, default=512)
    parser.add_argument("--down_t", type=int, default=1)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--vq_cnn_depth", type=int, default=3)
    parser.add_argument("--shared_codebook", action="store_true")
    parser.add_argument("--quantize_dropout_prob", type=float, default=0.0)
    parser.add_argument("--quantize_dropout_cutoff_index", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.99)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--vel_weight", type=float, default=1.0)
    parser.add_argument("--commit_weight", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Accepted for CLI compatibility. The CNN RVQ-VAE does not currently use checkpointed blocks.",
    )
    parser.add_argument("--dry_run_batches", type=int, default=0)
    parser.add_argument("--report_to", nargs="*", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="sentiavatar-rvqvae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Pass through to wandb.init(mode=...). Use offline on machines without network.",
    )
    parser.add_argument(
        "--wandb_watch",
        action="store_true",
        help="Ask W&B to watch gradients/parameters. Useful but can slow training.",
    )
    return parser.parse_args()


def normalize_parts(parts) -> list[str]:
    if not parts:
        return list(PART_ORDER)
    lowered = [str(part).lower() for part in parts]
    if "all" in lowered:
        return list(PART_ORDER)
    valid = set(PART_ORDER)
    unknown = [part for part in lowered if part not in valid]
    if unknown:
        raise ValueError(f"Unknown --parts value(s): {unknown}. Valid values: {list(PART_ORDER)} or all")
    deduped = []
    for part in lowered:
        if part not in deduped:
            deduped.append(part)
    return deduped


def reconcile_cli_aliases(args: argparse.Namespace) -> argparse.Namespace:
    if args.data_dir is not None:
        args.data_root = args.data_dir
    if args.num_train_epochs is not None:
        args.epochs = args.num_train_epochs
    if args.learning_rate is not None:
        args.lr = args.learning_rate
    if args.logging_steps is not None:
        args.log_every = args.logging_steps
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = args.batch_size
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps != 1:
        raise ValueError("gradient_accumulation_steps other than 1 is not implemented for this RVQ trainer yet")
    report_to = []
    for item in args.report_to or []:
        report_to.extend(part.strip().lower() for part in str(item).split(",") if part.strip())
    args.report_to = report_to
    if "wandb" in args.report_to:
        args.wandb = True
    return args


def main() -> None:
    args = parse_args()
    args = reconcile_cli_aliases(args)
    args.parts = normalize_parts(args.parts)
    seed_all(args.seed)
    device = setup_distributed(args)

    run_dir = args.output_dir.resolve() / args.experiment_name
    model_dir = run_dir / "model"
    if is_main_process(args):
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier(args)
    wandb_run = init_wandb(args, run_dir)
    if is_main_process(args):
        with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)
    barrier(args)

    normalizer = prepare_normalizer(args, run_dir)
    normalizer_path = (
        args.normalizer_path.resolve()
        if args.normalizer_path is not None
        else run_dir / "meta" / "normalizer.npz"
    )

    train_dataset = MultipartMotionDataset(
        data_root=args.data_root.resolve(),
        split=args.train_split,
        split_file=args.train_split_file.resolve() if args.train_split_file is not None else None,
        normalizer=normalizer,
        window_size=args.window_size,
        root_abs_threshold=args.root_abs_threshold,
        part_order=args.parts,
        train=True,
        max_clips=args.max_train_clips,
    )
    val_dataset = None
    val_split_file = (
        args.eval_split_file.resolve()
        if args.eval_split_file is not None
        else args.data_root.resolve() / "split" / f"{args.val_split}_file_list.txt"
    )
    if args.val_split and val_split_file.exists():
        val_dataset = MultipartMotionDataset(
            data_root=args.data_root.resolve(),
            split=args.val_split,
            split_file=val_split_file,
            normalizer=normalizer,
            window_size=args.window_size,
            root_abs_threshold=args.root_abs_threshold,
            part_order=args.parts,
            train=False,
            max_clips=args.max_val_clips,
        )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if getattr(args, "distributed", False)
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False)
        if getattr(args, "distributed", False) and val_dataset is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = build_model(args).to(device)
    start_epoch = 1
    best_val: Optional[float] = None
    optimizer_state = None
    scheduler_state = None
    state = {"global_step": 0}

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = checkpoint.get("best_val")
        state["global_step"] = int(checkpoint.get("global_step", 0))
        if is_main_process(args):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if getattr(args, "distributed", False):
        model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    total_training_steps_for_scheduler = len(train_loader) * max(0, args.epochs - start_epoch + 1)
    scheduler = build_scheduler(optimizer, args, total_training_steps_for_scheduler)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if wandb_run is not None and args.wandb_watch:
        wandb_run.watch(model, log="gradients", log_freq=max(args.log_every, 1))

    if is_main_process(args):
        train_samples_per_epoch = (
            len(train_sampler) * args.world_size
            if train_sampler is not None
            else len(train_dataset)
        )
        val_samples_per_eval = (
            len(val_sampler) * args.world_size
            if val_sampler is not None
            else (len(val_dataset) if val_dataset is not None else 0)
        )
        steps_per_epoch = len(train_loader)
        total_optimizer_steps = steps_per_epoch * max(0, args.epochs - start_epoch + 1)
        eval_loader_steps = len(val_loader) if val_loader is not None else 0
        print(f"Run dir: {run_dir}")
        print(f"Device: {device}")
        print(f"World size: {args.world_size}")
        print(f"Train examples: {len(train_dataset)} unique")
        print(f"Train samples/epoch: {train_samples_per_epoch} including DDP padding")
        print(f"Train steps/epoch: {steps_per_epoch} optimizer steps per rank")
        print(f"Total optimizer steps: {total_optimizer_steps} remaining ({steps_per_epoch} x {args.epochs - start_epoch + 1} epochs)")
        if val_dataset is not None:
            print(f"Val examples: {len(val_dataset)} unique")
            print(f"Val samples/eval: {val_samples_per_eval} including DDP padding")
            print(f"Val steps/eval: {eval_loader_steps} per rank")
        selected_part_dims = {part: PART_DIMS[part] for part in args.parts}
        print(f"Parts: {selected_part_dims}")
        print(
            f"Codebooks: {args.codebook_size} codes x {args.num_quantizers} quantizers "
            f"x {len(args.parts)} parts"
        )
        print(
            f"Batch: per_device_train={args.per_device_train_batch_size}, "
            f"effective_train={args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps}"
        )
        print(
            f"Cadence: logging_steps={args.log_every}, "
            f"eval_steps={args.eval_steps or 'epoch'}, "
            f"save_steps={args.save_steps or 'epoch'}, "
            f"save_total_limit={args.save_total_limit}"
        )
        print(
            f"LR schedule: {args.lr_scheduler_type}, "
            f"warmup_steps={args.warmup_steps}, warmup_ratio={args.warmup_ratio}, "
            f"min_lr_ratio={args.min_lr_ratio}"
        )

    training_start_time = time.time()
    def step_eval_and_save(global_step: int, epoch: int) -> None:
        nonlocal best_val
        if global_step <= 0:
            return
        val_metrics = None
        if args.eval_steps > 0 and val_loader is not None and global_step % args.eval_steps == 0:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    args,
                    optimizer=None,
                    epoch=epoch,
                    split="val",
                    wandb_run=wandb_run,
                )
            current_val = val_metrics["total"]
            if best_val is None or current_val < best_val:
                best_val = current_val
                if is_main_process(args):
                    save_checkpoint(
                        model_dir / "best.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        args,
                        normalizer_path,
                        best_val,
                        global_step=global_step,
                    )
            if is_main_process(args):
                print(f"eval global_step {global_step:07d} val_loss={current_val:.5f}")
            if wandb_run is not None and is_main_process(args):
                payload = {f"val/{key}": value for key, value in val_metrics.items()}
                payload.update({"epoch": epoch, "best/loss": best_val})
                wandb_run.log(payload, step=global_step)

        if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main_process(args):
            save_checkpoint(
                model_dir / f"step_{global_step}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                normalizer_path,
                best_val,
                global_step=global_step,
            )
            save_checkpoint(
                model_dir / "latest.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                normalizer_path,
                best_val,
                global_step=global_step,
            )
            prune_step_checkpoints(model_dir, args.save_total_limit)

    total_remaining_steps = len(train_loader) * max(0, args.epochs - start_epoch + 1)
    global_progress = None
    if is_main_process(args) and not args.disable_tqdm and tqdm is not None:
        global_progress = tqdm(
            total=total_remaining_steps,
            initial=0,
            desc="train total",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_start = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                args,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                split="train",
                wandb_run=wandb_run,
                state=state,
                on_step_end=lambda global_step, current_epoch=epoch: step_eval_and_save(
                    global_step,
                    current_epoch,
                ),
                progress=global_progress,
            )
            val_metrics = None
            if val_loader is not None and args.eval_steps <= 0:
                with torch.no_grad():
                    val_metrics = run_epoch(
                        model,
                        val_loader,
                        device,
                        args,
                        optimizer=None,
                        epoch=epoch,
                        split="val",
                        wandb_run=wandb_run,
                    )

            elapsed = time.time() - epoch_start
            epochs_completed = max(1, epoch - start_epoch + 1)
            avg_epoch_sec = (time.time() - training_start_time) / epochs_completed
            remaining_epochs = max(0, args.epochs - epoch)
            eta_text = format_duration(avg_epoch_sec * remaining_epochs)
            line = (
                f"epoch {epoch:04d}/{args.epochs:04d} "
                f"train_loss={train_metrics['total']:.5f} "
                f"rec={train_metrics['rec']:.5f} vel={train_metrics['vel']:.5f} "
                f"commit={train_metrics['commit']:.5f}"
            )
            if val_metrics is not None:
                line += f" val_loss={val_metrics['total']:.5f}"
            line += f" time={elapsed:.1f}s eta={eta_text}"
            if is_main_process(args):
                if global_progress is not None:
                    global_progress.write(line)
                else:
                    print(line)

            current_val = val_metrics["total"] if val_metrics is not None else train_metrics["total"]
            if best_val is None or current_val < best_val:
                best_val = current_val
                if is_main_process(args):
                    save_checkpoint(
                        model_dir / "best.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        args,
                        normalizer_path,
                        best_val,
                        global_step=state["global_step"],
                    )

            if wandb_run is not None and is_main_process(args):
                payload = {
                    "epoch": epoch,
                    "global_step": state["global_step"],
                    "time/epoch_sec": elapsed,
                    "best/loss": best_val,
                }
                payload.update({f"train/{key}_epoch": value for key, value in train_metrics.items()})
                if val_metrics is not None:
                    payload.update({f"val/{key}_epoch": value for key, value in val_metrics.items()})
                wandb_run.log(payload, step=state["global_step"])

            if is_main_process(args):
                save_checkpoint(
                    model_dir / "latest.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    normalizer_path,
                    best_val,
                    global_step=state["global_step"],
                )
                if args.save_every > 0 and epoch % args.save_every == 0:
                    save_checkpoint(
                        model_dir / f"epoch_{epoch}.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        args,
                        normalizer_path,
                        best_val,
                        global_step=state["global_step"],
                    )
    finally:
        if global_progress is not None:
            global_progress.close()

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(args)


if __name__ == "__main__":
    main()

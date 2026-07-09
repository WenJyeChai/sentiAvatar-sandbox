#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
        normalizer: PartNormalizer,
        window_size: int,
        root_abs_threshold: float,
        train: bool,
        max_clips: Optional[int] = None,
    ) -> None:
        self.data_root = data_root
        self.motion_dir = data_root / "motion_data"
        self.split_file = data_root / "split" / f"{split}_file_list.txt"
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
            for part in PART_ORDER
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
) -> Dict[str, torch.Tensor]:
    rec = output["rec"]
    commit = output["commit_loss"]
    perplexity = output["perplexity"]
    rec_terms = []
    vel_terms = []
    commit_terms = []
    metrics: Dict[str, torch.Tensor] = {}

    for part in PART_ORDER:
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
    epoch: int = 0,
    split: str = "train",
    wandb_run=None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    count = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        inputs = move_parts(batch["parts"], device)
        with torch.set_grad_enabled(is_train):
            output = model(inputs)
            losses = compute_losses(
                inputs,
                output,
                rec_weight=args.rec_weight,
                vel_weight=args.vel_weight,
                commit_weight=args.commit_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_size = next(iter(inputs.values())).shape[0]
        count += batch_size
        for key in losses:
            totals.setdefault(key, 0.0)
            totals[key] += float(losses[key].item()) * batch_size

        if is_train and args.log_every > 0 and step % args.log_every == 0:
            elapsed = time.time() - start_time
            avg = {key: totals[key] / max(count, 1) for key in totals}
            print(
                f"step {step:05d}/{len(loader):05d} "
                f"loss={avg['total']:.5f} rec={avg['rec']:.5f} "
                f"vel={avg['vel']:.5f} commit={avg['commit']:.5f} "
                f"time={elapsed:.1f}s"
            )
            if wandb_run is not None:
                payload = {f"{split}/{key}": value for key, value in avg.items()}
                payload["epoch"] = epoch
                payload[f"{split}/examples"] = count
                if is_train:
                    payload["train/lr"] = optimizer.param_groups[0]["lr"]
                wandb_run.log(payload)

        if args.dry_run_batches and step >= args.dry_run_batches:
            break

    return {key: totals[key] / max(count, 1) for key in totals}


def save_checkpoint(
    path: Path,
    model: MultiPartRVQVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    normalizer_path: Path,
    best_val: Optional[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config_dict(),
            "args": vars(args),
            "normalizer_path": str(normalizer_path),
            "best_val": best_val,
        },
        path,
    )


def build_model(args: argparse.Namespace) -> MultiPartRVQVAE:
    return MultiPartRVQVAE(
        part_dims=PART_DIMS,
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


def _wandb_config(args: argparse.Namespace) -> Dict[str, object]:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    config["part_order"] = list(PART_ORDER)
    config["part_dims"] = dict(PART_DIMS)
    return config


def init_wandb(args: argparse.Namespace, run_dir: Path):
    if not args.wandb:
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
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def prepare_normalizer(args: argparse.Namespace, run_dir: Path) -> PartNormalizer:
    data_root = args.data_root.resolve()
    motion_dir = data_root / "motion_data"
    split_file = data_root / "split" / f"{args.train_split}_file_list.txt"
    names = load_name_list(split_file)

    if args.normalizer_path is not None:
        return PartNormalizer.load(args.normalizer_path)

    meta_dir = run_dir / "meta"
    normalizer_path = meta_dir / "normalizer.npz"
    if normalizer_path.exists() and not args.recompute_stats:
        return PartNormalizer.load(normalizer_path)

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
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "multipart_rvqvae",
    )
    parser.add_argument("--experiment_name", type=str, default="parts_512x4")
    parser.add_argument("--normalizer_path", type=Path, default=None)
    parser.add_argument("--recompute_stats", action="store_true")
    parser.add_argument("--root_abs_threshold", type=float, default=10.0)

    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
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
    parser.add_argument("--lr", type=float, default=2e-4)
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
    parser.add_argument("--dry_run_batches", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="sentiavatar-rvqvae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
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


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device

    run_dir = args.output_dir.resolve() / args.experiment_name
    model_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, run_dir)
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    normalizer = prepare_normalizer(args, run_dir)
    normalizer_path = (
        args.normalizer_path.resolve()
        if args.normalizer_path is not None
        else run_dir / "meta" / "normalizer.npz"
    )

    train_dataset = MultipartMotionDataset(
        data_root=args.data_root.resolve(),
        split=args.train_split,
        normalizer=normalizer,
        window_size=args.window_size,
        root_abs_threshold=args.root_abs_threshold,
        train=True,
        max_clips=args.max_train_clips,
    )
    val_dataset = None
    val_split_file = args.data_root.resolve() / "split" / f"{args.val_split}_file_list.txt"
    if args.val_split and val_split_file.exists():
        val_dataset = MultipartMotionDataset(
            data_root=args.data_root.resolve(),
            split=args.val_split,
            normalizer=normalizer,
            window_size=args.window_size,
            root_abs_threshold=args.root_abs_threshold,
            train=False,
            max_clips=args.max_val_clips,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if wandb_run is not None and args.wandb_watch:
        wandb_run.watch(model, log="gradients", log_freq=max(args.log_every, 1))
    start_epoch = 1
    best_val: Optional[float] = None

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = checkpoint.get("best_val")
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Train clips: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val clips: {len(val_dataset)}")
    print(f"Parts: {PART_DIMS}")
    print(
        f"Codebooks: {args.codebook_size} codes x {args.num_quantizers} quantizers "
        f"x {len(PART_ORDER)} parts"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
            epoch=epoch,
            split="train",
            wandb_run=wandb_run,
        )
        val_metrics = None
        if val_loader is not None:
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
        line = (
            f"epoch {epoch:04d}/{args.epochs:04d} "
            f"train_loss={train_metrics['total']:.5f} "
            f"rec={train_metrics['rec']:.5f} vel={train_metrics['vel']:.5f} "
            f"commit={train_metrics['commit']:.5f}"
        )
        if val_metrics is not None:
            line += f" val_loss={val_metrics['total']:.5f}"
        line += f" time={elapsed:.1f}s"
        print(line)

        current_val = val_metrics["total"] if val_metrics is not None else train_metrics["total"]
        if best_val is None or current_val < best_val:
            best_val = current_val
            save_checkpoint(
                model_dir / "best.pth",
                model,
                optimizer,
                epoch,
                args,
                normalizer_path,
                best_val,
            )

        if wandb_run is not None:
            payload = {
                "epoch": epoch,
                "time/epoch_sec": elapsed,
                "best/loss": best_val,
            }
            payload.update({f"train/{key}_epoch": value for key, value in train_metrics.items()})
            if val_metrics is not None:
                payload.update({f"val/{key}_epoch": value for key, value in val_metrics.items()})
            wandb_run.log(payload)

        save_checkpoint(
            model_dir / "latest.pth",
            model,
            optimizer,
            epoch,
            args,
            normalizer_path,
            best_val,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                model_dir / f"epoch_{epoch}.pth",
                model,
                optimizer,
                epoch,
                args,
                normalizer_path,
                best_val,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

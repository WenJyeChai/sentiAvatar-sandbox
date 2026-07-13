"""Residual recoverability audit for multipart RVQ infilling.

The audit fixes the infiller's predicted q0 and asks how much of that error can
be repaired by q1..qN.  Ground-truth codec tokens define the target quantized
latent, which isolates infilling error from the codec reconstruction floor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = MOTION_GENERATION_DIR.parent
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.multipart_motion import PART_ORDER  # noqa: E402


VARIANT_ORDER = (
    "codec_gt",
    "one_shot",
    "pred_q0_gt_tail",
    "pred_q0_greedy_tail",
    "pred_q0_beam_tail",
)


@dataclass
class RecoveryTables:
    samples: pd.DataFrame
    tokens: pd.DataFrame
    summary: pd.DataFrame
    recovery: pd.DataFrame
    failures: pd.DataFrame
    metadata: dict[str, Any]


def _validate_search_inputs(
    target: torch.Tensor,
    fixed_q0: torch.Tensor,
    codebooks: torch.Tensor,
) -> None:
    if target.ndim != 2:
        raise ValueError(f"target must have shape (N,D), got {tuple(target.shape)}")
    if fixed_q0.ndim != 1 or fixed_q0.shape[0] != target.shape[0]:
        raise ValueError(
            f"fixed_q0 must have shape ({target.shape[0]},), got {tuple(fixed_q0.shape)}"
        )
    if codebooks.ndim != 3 or codebooks.shape[0] < 2:
        raise ValueError(f"codebooks must have shape (Q,C,D) with Q>=2, got {tuple(codebooks.shape)}")
    if codebooks.shape[-1] != target.shape[-1]:
        raise ValueError("target and codebook embedding dimensions differ")
    if fixed_q0.numel() and (
        int(fixed_q0.min()) < 0 or int(fixed_q0.max()) >= codebooks.shape[1]
    ):
        raise ValueError("fixed_q0 contains an out-of-range code")


def _nearest_codes(residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return nearest code indices without materializing (N,C,D) differences."""
    residual = residual.float()
    codebook = codebook.float()
    distances = (
        residual.square().sum(dim=-1, keepdim=True)
        + codebook.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * residual @ codebook.transpose(0, 1)
    )
    return distances.argmin(dim=-1)


@torch.no_grad()
def greedy_residual_tail(
    target: torch.Tensor,
    fixed_q0: torch.Tensor,
    codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily requantize q1..qN after fixing q0.

    Returns full code tuples ``(N,Q)`` and their cumulative embeddings ``(N,D)``.
    """
    _validate_search_inputs(target, fixed_q0, codebooks)
    target = target.float()
    codebooks = codebooks.float()
    codes = [fixed_q0.long()]
    cumulative = codebooks[0].index_select(0, fixed_q0.long())
    for quantizer in range(1, codebooks.shape[0]):
        code = _nearest_codes(target - cumulative, codebooks[quantizer])
        codes.append(code)
        cumulative = cumulative + codebooks[quantizer].index_select(0, code)
    return torch.stack(codes, dim=-1), cumulative


@torch.no_grad()
def beam_residual_tail(
    target: torch.Tensor,
    fixed_q0: torch.Tensor,
    codebooks: torch.Tensor,
    beam_width: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Beam-search q1..qN while keeping q0 fixed.

    This is an approximate upper bound on tail recoverability. Distances are
    computed algebraically so the search never creates an ``(N,B,C,D)`` tensor.
    """
    _validate_search_inputs(target, fixed_q0, codebooks)
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    target = target.float()
    codebooks = codebooks.float()
    batch = target.shape[0]
    codebook_size = codebooks.shape[1]
    cumulative = codebooks[0].index_select(0, fixed_q0.long()).unsqueeze(1)
    paths = fixed_q0.long().view(batch, 1, 1)

    for quantizer in range(1, codebooks.shape[0]):
        book = codebooks[quantizer]
        residual = target.unsqueeze(1) - cumulative
        distances = (
            residual.square().sum(dim=-1, keepdim=True)
            + book.square().sum(dim=-1).view(1, 1, -1)
            - 2.0 * torch.matmul(residual, book.transpose(0, 1))
        )
        candidate_count = distances.shape[1] * codebook_size
        keep = min(int(beam_width), int(candidate_count))
        _, flat_indices = torch.topk(
            distances.reshape(batch, candidate_count),
            k=keep,
            dim=-1,
            largest=False,
            sorted=True,
        )
        parent = torch.div(flat_indices, codebook_size, rounding_mode="floor")
        code = flat_indices.remainder(codebook_size)
        gather_cumulative = parent.unsqueeze(-1).expand(-1, -1, cumulative.shape[-1])
        cumulative = cumulative.gather(1, gather_cumulative) + book[code]
        gather_paths = parent.unsqueeze(-1).expand(-1, -1, paths.shape[-1])
        paths = torch.cat([paths.gather(1, gather_paths), code.unsqueeze(-1)], dim=-1)

    final_error = (target.unsqueeze(1) - cumulative).square().sum(dim=-1)
    best = final_error.argmin(dim=-1)
    row = torch.arange(batch, device=target.device)
    return paths[row, best], cumulative[row, best]


def tokens_to_part_latent(tokens: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Map ``(...,Q)`` raw code IDs to cumulative RVQ embeddings."""
    if tokens.shape[-1] != codebooks.shape[0]:
        raise ValueError(
            f"Expected {codebooks.shape[0]} quantizer IDs, got {tokens.shape[-1]}"
        )
    flat = tokens.reshape(-1, tokens.shape[-1]).long()
    latent = sum(
        codebooks[q].index_select(0, flat[:, q])
        for q in range(codebooks.shape[0])
    )
    return latent.reshape(*tokens.shape[:-1], codebooks.shape[-1])


@torch.no_grad()
def build_recovery_variants(
    gt_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    codebooks: Mapping[str, torch.Tensor],
    part_order: Sequence[str] = PART_ORDER,
    beam_width: int = 32,
) -> dict[str, torch.Tensor]:
    """Construct one-shot, fixed-q0, greedy-tail, and beam-tail sequences."""
    if gt_tokens.shape != predicted_tokens.shape or gt_tokens.ndim != 3:
        raise ValueError("gt_tokens and predicted_tokens must share shape (B,T,P*Q)")
    order = tuple(str(part) for part in part_order)
    quantizers = int(next(iter(codebooks.values())).shape[0])
    expected_slots = len(order) * quantizers
    if gt_tokens.shape[-1] != expected_slots:
        raise ValueError(f"Expected {expected_slots} token slots, got {gt_tokens.shape[-1]}")

    variants = {
        "codec_gt": gt_tokens.clone(),
        "one_shot": predicted_tokens.clone(),
        "pred_q0_gt_tail": gt_tokens.clone(),
        "pred_q0_greedy_tail": gt_tokens.clone(),
    }
    if beam_width > 0:
        variants["pred_q0_beam_tail"] = gt_tokens.clone()

    # Anchors are fixed ground truth for every comparison.
    for value in variants.values():
        value[:, 0] = gt_tokens[:, 0]
        value[:, -1] = gt_tokens[:, -1]

    for part_idx, part in enumerate(order):
        start = part_idx * quantizers
        end = start + quantizers
        books = codebooks[part].to(gt_tokens.device).float()
        gt_part = gt_tokens[:, 1:-1, start:end]
        pred_q0 = predicted_tokens[:, 1:-1, start].reshape(-1).long()
        target = tokens_to_part_latent(gt_part, books).reshape(-1, books.shape[-1])
        greedy_codes, _ = greedy_residual_tail(target, pred_q0, books)
        middle_shape = (*gt_part.shape[:-1], quantizers)

        variants["pred_q0_gt_tail"][:, 1:-1, start] = pred_q0.reshape(gt_part.shape[:-1])
        variants["pred_q0_greedy_tail"][:, 1:-1, start:end] = greedy_codes.reshape(
            middle_shape
        )
        if beam_width > 0:
            beam_codes, _ = beam_residual_tail(target, pred_q0, books, beam_width)
            variants["pred_q0_beam_tail"][:, 1:-1, start:end] = beam_codes.reshape(
                middle_shape
            )
    return variants


@torch.no_grad()
def _decode_variant_parts(
    decoder: Any,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Batch-decode raw multipart IDs to denormalized part features."""
    decoded: dict[str, torch.Tensor] = {}
    quantizers = decoder.num_quantizers
    for part_idx, part in enumerate(decoder.part_order):
        start = part_idx * quantizers
        loaded = decoder.codecs[part]
        value = loaded.model.decode({part: tokens[..., start : start + quantizers]})[part]
        decoded[part] = loaded.normalizer.denormalize_tensor(part, value).float()
    return decoded


def _latent_metrics(value: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = value - target
    cosine = torch.nn.functional.cosine_similarity(value, target, dim=-1, eps=1e-8)
    return {
        "latent_l2": torch.linalg.vector_norm(diff, dim=-1),
        "latent_rmse": diff.square().mean(dim=-1).sqrt(),
        "latent_cosine": cosine,
    }


def _decoded_metrics(value: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = value - target
    velocity = torch.diff(value, dim=1, prepend=value[:, :1])
    target_velocity = torch.diff(target, dim=1, prepend=target[:, :1])
    velocity_diff = velocity - target_velocity
    return {
        "decoded_mae": diff.abs().mean(dim=-1),
        "decoded_rmse": diff.square().mean(dim=-1).sqrt(),
        "decoded_velocity_rmse": velocity_diff.square().mean(dim=-1).sqrt(),
    }


def _raw_ids_from_global(
    global_ids: torch.Tensor,
    tokens_per_frame: int,
    codebook_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = global_ids.shape[1] // tokens_per_frame
    shaped = global_ids.reshape(global_ids.shape[0], frames, tokens_per_frame)
    offsets = (
        torch.arange(tokens_per_frame, device=global_ids.device).view(1, 1, -1)
        * codebook_size
    )
    raw = shaped - offsets
    invalid = raw.lt(0) | raw.ge(codebook_size)
    return raw.clamp(0, codebook_size - 1).long(), invalid


def _build_summary(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    return (
        samples.groupby(["variant", "part", "q0_correct"], as_index=False, observed=True)
        .agg(
            samples=("name", "size"),
            clips=("name", "nunique"),
            latent_l2=("latent_l2", "mean"),
            latent_rmse=("latent_rmse", "mean"),
            latent_cosine=("latent_cosine", "mean"),
            decoded_mae=("decoded_mae", "mean"),
            decoded_rmse=("decoded_rmse", "mean"),
            decoded_velocity_rmse=("decoded_velocity_rmse", "mean"),
        )
    )


def _build_recovery_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Create paired recovery statistics against fixed-q0 baselines."""
    if samples.empty:
        return pd.DataFrame()
    index = [
        "dataset_idx",
        "name",
        "left_idx",
        "relative_frame",
        "token_frame",
        "part",
        "q0_correct",
    ]
    metrics = ["latent_rmse", "decoded_rmse", "decoded_velocity_rmse"]
    wide = samples.pivot(index=index, columns="variant", values=metrics)
    rows: list[dict[str, Any]] = []
    candidates = ["pred_q0_greedy_tail", "pred_q0_beam_tail"]
    for candidate in candidates:
        if (metrics[0], candidate) not in wide.columns:
            continue
        for reference in ("pred_q0_gt_tail", "one_shot"):
            if (metrics[0], reference) not in wide.columns:
                continue
            for metric in metrics:
                baseline = wide[(metric, reference)]
                recovered = wide[(metric, candidate)]
                gain = baseline - recovered
                fraction = gain / baseline.where(baseline.abs() > 1e-12)
                payload = pd.DataFrame(
                    {
                        "baseline": baseline,
                        "recovered": recovered,
                        "gain": gain,
                        "recovery_fraction": fraction,
                        "improved": recovered < baseline,
                    }
                ).reset_index()
                payload["candidate"] = candidate
                payload["reference"] = reference
                payload["metric"] = metric
                rows.extend(payload.to_dict("records"))
    return pd.DataFrame(rows)


@torch.no_grad()
def run_residual_recoverability_audit(
    context: Any,
    *,
    beam_width: int = 32,
) -> RecoveryTables:
    """Run the paired recoverability audit over an existing audit context."""
    if context.decoder is None:
        raise ValueError(
            "Residual recoverability requires full codecs; set config.include_decoded_errors=True"
        )
    config = context.config
    decoder = context.decoder
    codebooks = {
        part: value.to(context.device).float()
        for part, value in decoder.codebooks.items()
    }
    ntpf = decoder.tokens_per_frame
    quantizers = decoder.num_quantizers
    codebook_size = decoder.codebook_size
    unit_length = decoder.unit_length
    sample_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    progress = tqdm(total=len(context.selected_indices), desc="residual recoverability")
    for batch_start in range(0, len(context.selected_indices), config.batch_size):
        indices = context.selected_indices[batch_start : batch_start + config.batch_size]
        examples = [context.dataset[index] for index in indices]
        batch = context.collator(examples)
        input_ids = batch["input_ids"].to(context.device)
        audio = batch["audio_features"].to(context.device)
        generated_global = context.model.generate_sbs(
            input_ids,
            audio,
            generate_steps=config.generate_steps,
        )
        predicted_tokens, invalid = _raw_ids_from_global(
            generated_global, ntpf, codebook_size
        )
        gt_tokens = torch.as_tensor(
            np.asarray([example.motion_tokens for example in examples]),
            dtype=torch.long,
            device=context.device,
        )
        variants = build_recovery_variants(
            gt_tokens,
            predicted_tokens,
            codebooks,
            decoder.part_order,
            beam_width=beam_width,
        )
        decoded = {
            variant: _decode_variant_parts(decoder, tokens)
            for variant, tokens in variants.items()
        }

        for local_idx, (dataset_idx, example) in enumerate(zip(indices, examples)):
            name = str(example.name)
            if bool(invalid[local_idx].any()):
                failure_rows.append(
                    {
                        "dataset_idx": int(dataset_idx),
                        "name": name,
                        "stage": "global_to_raw_ids",
                        "error": f"clamped {int(invalid[local_idx].sum())} invalid token IDs",
                    }
                )

            for part_idx, part in enumerate(decoder.part_order):
                start = part_idx * quantizers
                end = start + quantizers
                books = codebooks[part]
                part_latents = {
                    variant: tokens_to_part_latent(tokens[local_idx, :, start:end], books)
                    for variant, tokens in variants.items()
                }
                latent_target = part_latents["codec_gt"]
                decoded_target = decoded["codec_gt"][part][local_idx]
                decoded_by_variant = {
                    variant: values[part][local_idx]
                    for variant, values in decoded.items()
                }
                decoded_metric_by_variant = {
                    variant: _decoded_metrics(value.unsqueeze(0), decoded_target.unsqueeze(0))
                    for variant, value in decoded_by_variant.items()
                }

                for relative_frame in range(1, gt_tokens.shape[1] - 1):
                    token_frame = int(example.left_idx + relative_frame)
                    q0_gt = int(gt_tokens[local_idx, relative_frame, start])
                    q0_pred = int(predicted_tokens[local_idx, relative_frame, start])
                    q0_correct = q0_gt == q0_pred
                    raw_start = relative_frame * unit_length
                    raw_end = min(raw_start + unit_length, decoded_target.shape[0])

                    token_row: dict[str, Any] = {
                        "dataset_idx": int(dataset_idx),
                        "name": name,
                        "left_idx": int(example.left_idx),
                        "relative_frame": relative_frame,
                        "token_frame": token_frame,
                        "part": part,
                        "q0_correct": q0_correct,
                    }
                    for q in range(quantizers):
                        token_row[f"gt_q{q}"] = int(
                            variants["codec_gt"][local_idx, relative_frame, start + q]
                        )
                        token_row[f"one_shot_q{q}"] = int(
                            variants["one_shot"][local_idx, relative_frame, start + q]
                        )
                        token_row[f"greedy_q{q}"] = int(
                            variants["pred_q0_greedy_tail"][local_idx, relative_frame, start + q]
                        )
                        if "pred_q0_beam_tail" in variants:
                            token_row[f"beam_q{q}"] = int(
                                variants["pred_q0_beam_tail"][local_idx, relative_frame, start + q]
                            )
                    token_rows.append(token_row)

                    for variant in variants:
                        latent_stats = _latent_metrics(
                            part_latents[variant][relative_frame],
                            latent_target[relative_frame],
                        )
                        decoded_stats = decoded_metric_by_variant[variant]
                        sample_rows.append(
                            {
                                "dataset_idx": int(dataset_idx),
                                "name": name,
                                "left_idx": int(example.left_idx),
                                "right_idx": int(example.right_idx),
                                "relative_frame": relative_frame,
                                "token_frame": token_frame,
                                "part": part,
                                "q0_gt": q0_gt,
                                "q0_pred": q0_pred,
                                "q0_correct": q0_correct,
                                "variant": variant,
                                "latent_l2": float(latent_stats["latent_l2"]),
                                "latent_rmse": float(latent_stats["latent_rmse"]),
                                "latent_cosine": float(latent_stats["latent_cosine"]),
                                "decoded_mae": float(
                                    decoded_stats["decoded_mae"][0, raw_start:raw_end].mean()
                                ),
                                "decoded_rmse": float(
                                    decoded_stats["decoded_rmse"][0, raw_start:raw_end].mean()
                                ),
                                "decoded_velocity_rmse": float(
                                    decoded_stats["decoded_velocity_rmse"][0, raw_start:raw_end].mean()
                                ),
                            }
                        )
        progress.update(len(indices))
    progress.close()

    samples = pd.DataFrame(sample_rows)
    tokens = pd.DataFrame(token_rows)
    summary = _build_summary(samples)
    recovery = _build_recovery_table(samples)
    failures = pd.DataFrame(
        failure_rows,
        columns=["dataset_idx", "name", "stage", "error"],
    )
    metadata = {
        "experiment": "multipart_residual_recoverability",
        "target": "ground-truth cumulative quantized latent",
        "variants": [variant for variant in VARIANT_ORDER if variant in variants],
        "beam_width": int(beam_width),
        "num_windows": int(len(context.selected_indices)),
        "num_samples": int(len(samples)),
        "num_clips": int(samples["name"].nunique()) if not samples.empty else 0,
        "part_order": list(decoder.part_order),
        "num_quantizers": int(quantizers),
        "codebook_size": int(codebook_size),
        "unit_length": int(unit_length),
        "mask_checkpoint": str(config.mask_checkpoint),
        "part_checkpoints": {
            part: str(path) for part, path in config.part_checkpoints.items()
        },
    }
    return RecoveryTables(samples, tokens, summary, recovery, failures, metadata)


def save_recovery_tables(tables: RecoveryTables, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables.samples.to_csv(output_dir / "recoverability_samples.csv", index=False)
    tables.tokens.to_csv(output_dir / "recoverability_tokens.csv", index=False)
    tables.summary.to_csv(output_dir / "recoverability_summary.csv", index=False)
    tables.recovery.to_csv(output_dir / "recoverability_paired.csv", index=False)
    tables.failures.to_csv(output_dir / "recoverability_failures.csv", index=False)
    with open(output_dir / "recoverability_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(tables.metadata, handle, indent=2)


def paired_recovery_summary(
    recovery: pd.DataFrame,
    *,
    q0_errors_only: bool = True,
) -> pd.DataFrame:
    data = recovery
    if q0_errors_only:
        data = data.loc[~data["q0_correct"]]
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby(["candidate", "reference", "metric", "part"], as_index=False)
        .agg(
            samples=("name", "size"),
            baseline=("baseline", "mean"),
            recovered=("recovered", "mean"),
            mean_gain=("gain", "mean"),
            median_recovery_fraction=("recovery_fraction", "median"),
            improved_fraction=("improved", "mean"),
        )
    )


def plot_recovery_by_part(
    recovery: pd.DataFrame,
    *,
    metric: str = "latent_rmse",
    reference: str = "pred_q0_gt_tail",
    q0_errors_only: bool = True,
):
    import matplotlib.pyplot as plt

    data = recovery.loc[
        recovery["metric"].eq(metric) & recovery["reference"].eq(reference)
    ].copy()
    if q0_errors_only:
        data = data.loc[~data["q0_correct"]]
    if data.empty:
        raise ValueError("No recovery rows matched the requested plot filters")
    grouped = data.groupby(["part", "candidate"], observed=True)[
        ["baseline", "recovered"]
    ].mean()
    recovered = grouped["recovered"].unstack("candidate")
    baseline = grouped["baseline"].groupby(level="part").first().rename("baseline")
    plot_data = pd.concat([baseline, recovered], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data.plot(kind="bar", ax=ax)
    ax.set(
        title=f"Residual recovery by part: {metric}",
        xlabel="Part",
        ylabel=metric,
    )
    ax.legend(title="Token construction")
    fig.tight_layout()
    return fig


def plot_recovery_fraction(
    recovery: pd.DataFrame,
    *,
    metric: str = "latent_rmse",
    reference: str = "pred_q0_gt_tail",
    candidate: str = "pred_q0_beam_tail",
):
    import matplotlib.pyplot as plt

    data = recovery.loc[
        recovery["metric"].eq(metric)
        & recovery["reference"].eq(reference)
        & recovery["candidate"].eq(candidate)
        & ~recovery["q0_correct"]
    ].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["recovery_fraction"])
    if data.empty:
        raise ValueError("No finite recovery fractions matched the requested plot filters")
    parts = [part for part in PART_ORDER if part in set(data["part"])]
    values = [data.loc[data["part"].eq(part), "recovery_fraction"].to_numpy() for part in parts]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=parts, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(
        title=f"{candidate} recovery fraction on incorrect q0",
        xlabel="Part",
        ylabel=f"Fraction of {metric} removed",
    )
    fig.tight_layout()
    return fig


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--mask_checkpoint", type=Path, default=None)
    for part in PART_ORDER:
        parser.add_argument(f"--{part}_ckpt", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_clips", type=int, default=64)
    parser.add_argument("--max_windows", type=int, default=1024)
    parser.add_argument("--beam_width", type=int, default=32)
    parser.add_argument("--generate_steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    from utils.msd.complexity_error_audit import (
        default_audit_config,
        load_audit_context,
    )

    args = parse_args()
    config = default_audit_config(args.project_dir)
    if args.mask_checkpoint is not None:
        config.mask_checkpoint = args.mask_checkpoint.resolve()
    for part in PART_ORDER:
        value = getattr(args, f"{part}_ckpt")
        if value is not None:
            config.part_checkpoints[part] = value.resolve()
    config.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.project_dir.resolve()
        / "motion_generation"
        / "outputs"
        / "residual_recoverability_audit"
    )
    config.device = args.device
    config.batch_size = args.batch_size
    config.max_clips = args.max_clips
    config.max_windows = args.max_windows
    config.generate_steps = args.generate_steps
    config.include_decoded_errors = True

    context = load_audit_context(config)
    tables = run_residual_recoverability_audit(context, beam_width=args.beam_width)
    save_recovery_tables(tables, config.output_dir)
    print(paired_recovery_summary(tables.recovery).to_string(index=False))
    print(f"Saved audit to {config.output_dir}")


if __name__ == "__main__":
    main()

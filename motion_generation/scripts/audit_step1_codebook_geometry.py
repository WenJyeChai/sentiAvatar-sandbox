#!/usr/bin/env python3
"""Audit whether RVQ embedding distance predicts decoded motion damage.

The expected-distortion loss is justified only if its frozen codebook metric
is aligned with the causal codec decoder. This script samples local windows
from the balanced 6K training split, replaces one RVQ ID at a time with codes
spanning near-to-far embedding distances, and measures decoded damage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import canonical_data_path, load_motion_tokens  # noqa: E402
from scripts.export_multipart_motion_tokens import load_part_codec  # noqa: E402
from utils.adaptive_anchor_tokens import BODY_PART_ORDER, BODY_CODEBOOK_SIZE  # noqa: E402
from utils.causal_codec_evaluation import geodesic_degrees  # noqa: E402
from utils.step1_expected_distortion import (  # noqa: E402
    normalized_part_codebook_distance_table,
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_split",
        type=Path,
        default=Path("motion_generation/data_splits/step1_balanced_seed42/train_step1_main_6000.txt"),
    )
    parser.add_argument(
        "--motion_token_dir",
        type=Path,
        default=Path("SuSuInterActs/SuSuInterActs/motion_token_data_multipart_causal_512x4"),
    )
    for part in BODY_PART_ORDER:
        parser.add_argument(
            f"--{part}_ckpt",
            type=Path,
            default=Path(
                f"checkpoints/causal_multipart_rvqvae/"
                f"causal_rvq_{part}_512x4_scratch/model/best.pth"
            ),
        )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples_per_slot", type=int, default=32)
    parser.add_argument("--alternatives_per_sample", type=int, default=8)
    parser.add_argument("--left_context", type=int, default=8)
    parser.add_argument("--right_context", type=int, default=16)
    parser.add_argument("--decode_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum_slot_rho", type=float, default=0.20)
    parser.add_argument("--minimum_median_rho", type=float, default=0.50)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("checkpoints/step1_codebook_geometry_audit_6k.json"),
    )
    return parser.parse_args()


def read_names(path: Path) -> list[str]:
    names = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def sample_windows(
    *,
    names: list[str],
    token_dir: Path,
    count: int,
    left: int,
    right: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    windows: list[np.ndarray] = []
    metadata: list[dict[str, int | str]] = []
    required = left + right + 1
    for name_index in order:
        name = names[int(name_index)]
        path = canonical_data_path(token_dir, name, ".json")
        try:
            dense, _ = load_motion_tokens(path, require_causal=True)
        except (FileNotFoundError, ValueError):
            continue
        dense_array = np.asarray(dense, dtype=np.int64)
        if dense_array.shape[0] < required:
            continue
        center = int(rng.integers(left, dense_array.shape[0] - right))
        windows.append(dense_array[center - left : center + right + 1])
        metadata.append({"name": name, "token_time": center})
        if len(windows) == count:
            break
    if len(windows) != count:
        raise RuntimeError(f"Found only {len(windows)} valid windows; requested {count}")
    return np.stack(windows), metadata


def alternative_ids(distances: torch.Tensor, target: int, count: int) -> torch.Tensor:
    if not 1 <= count < BODY_CODEBOOK_SIZE:
        raise ValueError("alternatives_per_sample must be in [1, 511]")
    order = torch.argsort(distances[int(target)])
    order = order[order.ne(int(target))]
    positions = torch.linspace(0, len(order) - 1, steps=count).round().long()
    return order.index_select(0, positions)


@torch.inference_mode()
def audit_slot(
    *,
    loaded,
    part_index: int,
    quantizer: int,
    windows: np.ndarray,
    center: int,
    distances: torch.Tensor,
    alternatives: int,
    decode_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int]]]:
    part = loaded.part
    start = part_index * 4
    reference_codes = torch.as_tensor(
        windows[:, :, start : start + 4], dtype=torch.long
    )
    sequences: list[torch.Tensor] = []
    embedding_costs: list[float] = []
    sample_indices: list[int] = []
    candidate_ids: list[int] = []
    for sample_index, codes in enumerate(reference_codes):
        target = int(codes[center, quantizer])
        choices = alternative_ids(distances[quantizer], target, alternatives)
        sequences.append(codes)
        for candidate in choices.tolist():
            changed = codes.clone()
            changed[center, quantizer] = int(candidate)
            sequences.append(changed)
            embedding_costs.append(float(distances[quantizer, target, candidate]))
            sample_indices.append(sample_index)
            candidate_ids.append(int(candidate))

    all_codes = torch.stack(sequences)
    decoded_chunks: list[torch.Tensor] = []
    for batch_start in range(0, len(all_codes), decode_batch_size):
        batch = all_codes[batch_start : batch_start + decode_batch_size].to(device)
        decoded_chunks.append(loaded.model.decode({part: batch})[part].detach().cpu())
    decoded = torch.cat(decoded_chunks)
    frames = decoded.shape[1]
    features = decoded.shape[2]
    decoded = decoded.reshape(len(windows), alternatives + 1, frames, features)
    reference = decoded[:, :1]
    changed = decoded[:, 1:]
    normalized_rmse = torch.sqrt((changed - reference).square().mean(dim=(-1, -2)))

    flat_reference = reference.expand_as(changed).reshape(-1, frames, features)
    flat_changed = changed.reshape(-1, frames, features)
    reference_raw = loaded.normalizer.denormalize_tensor(part, flat_reference)
    changed_raw = loaded.normalizer.denormalize_tensor(part, flat_changed)
    geodesic = geodesic_degrees(part, reference_raw, changed_raw).mean(dim=(-1, -2))

    rmse_values = normalized_rmse.reshape(-1).numpy()
    geodesic_values = geodesic.cpu().numpy()
    cost_values = np.asarray(embedding_costs, dtype=np.float64)
    rho_rmse = float(spearmanr(cost_values, rmse_values).statistic)
    rho_geodesic = float(spearmanr(cost_values, geodesic_values).statistic)
    rows = [
        {
            "sample_index": int(sample_index),
            "candidate_id": int(candidate),
            "embedding_distance": float(cost),
            "normalized_feature_rmse": float(rmse),
            "geodesic_degrees": float(angle),
        }
        for sample_index, candidate, cost, rmse, angle in zip(
            sample_indices,
            candidate_ids,
            cost_values,
            rmse_values,
            geodesic_values,
        )
    ]
    summary = {
        "slot": f"{part}_q{quantizer}",
        "part": part,
        "quantizer": quantizer,
        "comparisons": len(rows),
        "spearman_embedding_vs_normalized_rmse": rho_rmse,
        "spearman_embedding_vs_geodesic": rho_geodesic,
        "mean_embedding_distance": float(cost_values.mean()),
        "mean_normalized_feature_rmse": float(rmse_values.mean()),
        "mean_geodesic_degrees": float(geodesic_values.mean()),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    if args.samples_per_slot <= 0 or args.decode_batch_size <= 0:
        raise ValueError("samples_per_slot and decode_batch_size must be positive")
    train_split = project_path(args.train_split)
    token_dir = project_path(args.motion_token_dir)
    output_json = project_path(args.output_json)
    names = read_names(train_split)
    windows, sample_metadata = sample_windows(
        names=names,
        token_dir=token_dir,
        count=args.samples_per_slot,
        left=args.left_context,
        right=args.right_context,
        seed=args.seed,
    )
    device = torch.device(args.device)
    summaries: list[dict[str, float | int | str]] = []
    detail: dict[str, list[dict[str, float | int]]] = {}
    for part_index, part in enumerate(BODY_PART_ORDER):
        checkpoint = project_path(getattr(args, f"{part}_ckpt"))
        print(f"Loading {part}: {checkpoint}")
        loaded = load_part_codec(checkpoint, device)
        if not loaded.causal:
            raise ValueError(f"Geometry audit requires a causal codec: {checkpoint}")
        codebooks = loaded.model.quantizers[part].codebooks.detach().float().cpu()
        distances = normalized_part_codebook_distance_table(codebooks, part)
        for quantizer in range(4):
            summary, rows = audit_slot(
                loaded=loaded,
                part_index=part_index,
                quantizer=quantizer,
                windows=windows,
                center=args.left_context,
                distances=distances,
                alternatives=args.alternatives_per_sample,
                decode_batch_size=args.decode_batch_size,
                device=device,
            )
            summaries.append(summary)
            detail[str(summary["slot"])] = rows
            print(
                f"{summary['slot']}: rho_rmse="
                f"{summary['spearman_embedding_vs_normalized_rmse']:.3f}, "
                f"rho_geodesic={summary['spearman_embedding_vs_geodesic']:.3f}"
            )
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rmse_rhos = np.asarray(
        [row["spearman_embedding_vs_normalized_rmse"] for row in summaries],
        dtype=np.float64,
    )
    passed = bool(
        np.isfinite(rmse_rhos).all()
        and float(rmse_rhos.min()) >= args.minimum_slot_rho
        and float(np.median(rmse_rhos)) >= args.minimum_median_rho
    )
    payload = {
        "status": "GO" if passed else "NO_GO",
        "interpretation": (
            "Codec embedding distance is sufficiently monotonic with local decoded damage."
            if passed
            else "At least one codec slot failed the configured geometry-alignment gate."
        ),
        "config": {
            "train_split": str(train_split),
            "split_clips": len(names),
            "motion_token_dir": str(token_dir),
            "samples_per_slot": args.samples_per_slot,
            "alternatives_per_sample": args.alternatives_per_sample,
            "left_context": args.left_context,
            "right_context": args.right_context,
            "minimum_slot_rho": args.minimum_slot_rho,
            "minimum_median_rho": args.minimum_median_rho,
            "seed": args.seed,
        },
        "aggregate": {
            "minimum_rmse_spearman": float(rmse_rhos.min()),
            "median_rmse_spearman": float(np.median(rmse_rhos)),
        },
        "samples": sample_metadata,
        "slots": summaries,
        "comparisons": detail,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["aggregate"]}, indent=2))
    print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()

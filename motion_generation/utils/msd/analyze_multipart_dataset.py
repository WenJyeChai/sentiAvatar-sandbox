#!/usr/bin/env python3
"""Build a descriptive MSD atlas for exported multipart motion tokens.

This script is intentionally model-error agnostic.  It describes what the
multipart Motion Spectral Descriptor measures in a dataset before MSD is used
as a difficulty predictor, loss weight, curriculum signal, or keyframe policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from .msd import (
        MSDConfig,
        compute_msd_components,
        concatenate_part_embeddings,
        multipart_tokens_to_embeddings,
    )
    from .multipart_adapter import MultipartCodebookSet, MultipartTokenDataset, resolve_device
except ImportError:
    from msd import (  # type: ignore
        MSDConfig,
        compute_msd_components,
        concatenate_part_embeddings,
        multipart_tokens_to_embeddings,
    )
    from multipart_adapter import (  # type: ignore
        MultipartCodebookSet,
        MultipartTokenDataset,
        resolve_device,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "SuSuInterActs" / "SuSuInterActs"
DEFAULT_CODEC_ROOT = PROJECT_ROOT / "checkpoints" / "multipart_rvqvae"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "multipart_atlas"
PART_ORDER = ("upper", "lower", "feet", "hands")
TAG_PATTERN = re.compile(r"【([^】]+)】")


def default_checkpoint(part: str) -> Path:
    return DEFAULT_CODEC_ROOT / f"rvq_{part}_512x4_bs256_cosine" / "model" / "best.pth"


def load_text_map(data_dir: Path) -> dict[str, str]:
    path = data_dir / "text_data" / "motion2text.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key).replace("\\", "/"): str(value) for key, value in payload.items()}


def extract_tags(text: str) -> tuple[str, str]:
    action = ""
    expression = ""
    for tag in TAG_PATTERN.findall(text or ""):
        if tag.startswith("动作："):
            action = tag.removeprefix("动作：").strip()
        elif tag.startswith("表情："):
            expression = tag.removeprefix("表情：").strip()
    return action or "未标注", expression or "未标注"


def read_token_metadata(dataset: MultipartTokenDataset, name: str) -> dict[str, object]:
    path = dataset.token_path(name)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return {
        "root_schema": str(payload.get("root_schema") or "unknown"),
        "root_mean_norm": float(payload.get("root_mean_norm") or 0.0),
    }


def spectral_shape(phi: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return spectral centroid and high-band share for L2-normalized bands."""
    bands = torch.arange(phi.shape[-1], dtype=phi.dtype, device=phi.device)
    mass = phi.sum(dim=-1).clamp_min(torch.finfo(phi.dtype).eps)
    denominator = max(1, int(phi.shape[-1]) - 1)
    centroid = (phi * bands).sum(dim=-1) / mass / denominator
    high_start = int(math.ceil(phi.shape[-1] / 2))
    high_share = phi[..., high_start:].sum(dim=-1) / mass
    return (
        centroid.detach().cpu().numpy().astype(np.float32),
        high_share.detach().cpu().numpy().astype(np.float32),
    )


def longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def quantile(value: pd.Series, q: float) -> float:
    return float(value.quantile(q))


def histogram(values: pd.Series, bins: int = 36, *, log: bool = False) -> dict[str, list[float]]:
    array = values.to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if log:
        array = np.log1p(np.clip(array, 0.0, None))
    counts, edges = np.histogram(array, bins=bins)
    return {
        "edges": [round(float(value), 7) for value in edges],
        "counts": [int(value) for value in counts],
    }


def correlation_rows(frame_metrics: pd.DataFrame, clip_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level, table in (("frame", frame_metrics), ("clip", clip_metrics)):
        for metric in ("omega", "log_energy"):
            suffix = "omega" if metric == "omega" else "energy"
            if level == "clip":
                suffix += "_mean"
            columns = [f"{part}_{suffix}" for part in PART_ORDER]
            values = table[columns].copy()
            if metric == "log_energy":
                values = np.log1p(values.clip(lower=0.0))
            corr = values.corr(method="spearman")
            for row_part in PART_ORDER:
                for col_part in PART_ORDER:
                    rows.append(
                        {
                            "level": level,
                            "metric": metric,
                            "part_a": row_part,
                            "part_b": col_part,
                            "spearman": float(corr.loc[columns[PART_ORDER.index(row_part)], columns[PART_ORDER.index(col_part)]]),
                        }
                    )
    return pd.DataFrame(rows)


def build_clip_metrics(frame_metrics: pd.DataFrame, token_fps: float) -> pd.DataFrame:
    energy_columns = ["combined_energy", *[f"{part}_energy" for part in PART_ORDER]]
    thresholds = {column: quantile(frame_metrics[column], 0.75) for column in energy_columns}
    rows: list[dict[str, object]] = []
    for name, clip in frame_metrics.groupby("name", sort=False):
        first = clip.iloc[0]
        row: dict[str, object] = {
            "name": name,
            "token_frames": int(len(clip)),
            "duration_s": float(len(clip) / token_fps),
            "root_schema": first["root_schema"],
            "root_mean_norm": float(first["root_mean_norm"]),
            "action": first["action"],
            "expression": first["expression"],
        }
        for prefix in ("combined", *PART_ORDER):
            for metric in ("omega", "energy", "centroid", "high_share"):
                column = f"{prefix}_{metric}"
                row[f"{column}_mean"] = float(clip[column].mean())
                row[f"{column}_median"] = float(clip[column].median())
                row[f"{column}_p90"] = quantile(clip[column], 0.90)
            if prefix != "combined":
                contribution = f"{prefix}_contribution"
                row[f"{contribution}_mean"] = float(clip[contribution].mean())
                row[f"{contribution}_p90"] = quantile(clip[contribution], 0.90)
            energy_column = f"{prefix}_energy"
            active = clip[energy_column].to_numpy() >= thresholds[energy_column]
            row[f"{prefix}_active_fraction"] = float(active.mean())
            row[f"{prefix}_longest_burst_s"] = float(longest_true_run(active) / token_fps)
        rows.append(row)
    return pd.DataFrame(rows)


def build_action_summary(clip_metrics: pd.DataFrame, minimum_clips: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for action, group in clip_metrics.groupby("action"):
        if len(group) < minimum_clips:
            continue
        contribution_means = {
            part: float(group[f"{part}_contribution_mean"].mean()) for part in PART_ORDER
        }
        rows.append(
            {
                "action": action,
                "clips": int(len(group)),
                "duration_s": float(group["duration_s"].sum()),
                "combined_energy_median": float(group["combined_energy_mean"].median()),
                "combined_omega_median": float(group["combined_omega_mean"].median()),
                "combined_centroid_median": float(group["combined_centroid_mean"].median()),
                "dominant_part": max(contribution_means, key=contribution_means.get),
                **{f"{part}_contribution": value for part, value in contribution_means.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["clips", "action"], ascending=[False, True])


def representative_rows(clip_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    targets = (("lowest", 0.02), ("low", 0.25), ("median", 0.50), ("high", 0.75), ("highest", 0.98))
    for prefix in ("combined", *PART_ORDER):
        column = f"{prefix}_energy_mean"
        for tier, percentile in targets:
            target = quantile(clip_metrics[column], percentile)
            idx = (clip_metrics[column] - target).abs().idxmin()
            row = clip_metrics.loc[idx].copy()
            row["descriptor"] = prefix
            row["tier"] = tier
            row["target_percentile"] = percentile
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe multipart MSD across a dataset split")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion-fps", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=0.8)
    parser.add_argument("--minimum-action-clips", type=int, default=5)
    parser.add_argument("--max-clips", type=int, default=None)
    for part in PART_ORDER:
        parser.add_argument(f"--{part}-ckpt", type=Path, default=default_checkpoint(part))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoints = {part: Path(getattr(args, f"{part}_ckpt")) for part in PART_ORDER}
    codec = MultipartCodebookSet.from_checkpoints(checkpoints, device, PART_ORDER)
    dataset = MultipartTokenDataset(args.data_dir, args.token_dir)
    dataset.validate_layout(codec)
    token_fps = float(args.motion_fps) / float(codec.unit_length)
    window = max(2, int(round(float(args.window_seconds) * token_fps)))
    text_map = load_text_map(Path(args.data_dir))
    frame_tables: list[pd.DataFrame] = []
    phi_sums = {prefix: np.zeros(window, dtype=np.float64) for prefix in ("combined", *PART_ORDER)}
    phi_counts = {prefix: 0 for prefix in phi_sums}

    for clip_index, (name, tokens) in enumerate(
        dataset.iter_tokens(
            args.split,
            expected_slots=codec.tokens_per_frame,
            device=device,
            limit=args.max_clips,
        ),
        start=1,
    ):
        embeddings = multipart_tokens_to_embeddings(tokens, codec.codebooks, codec.part_order)
        combined = concatenate_part_embeddings(embeddings, codec.part_order)
        components = {"combined": compute_msd_components(combined, MSDConfig(W=window))}
        components.update(
            {part: compute_msd_components(embeddings[part], MSDConfig(W=window)) for part in PART_ORDER}
        )
        metadata = read_token_metadata(dataset, name)
        action, expression = extract_tags(text_map.get(name, ""))
        frames = int(tokens.shape[0])
        table: dict[str, object] = {
            "name": np.repeat(name, frames),
            "token_frame": np.arange(frames, dtype=np.int32),
            "time_s": np.arange(frames, dtype=np.float32) / token_fps,
            "root_schema": np.repeat(metadata.get("root_schema", "unknown"), frames),
            "root_mean_norm": np.repeat(metadata.get("root_mean_norm", 0.0), frames),
            "action": np.repeat(action, frames),
            "expression": np.repeat(expression, frames),
        }
        for prefix, value in components.items():
            phi = value.phi.detach().cpu().numpy().astype(np.float32)
            centroid, high_share = spectral_shape(value.phi)
            table[f"{prefix}_omega"] = value.omega.detach().cpu().numpy().astype(np.float32)
            table[f"{prefix}_energy"] = value.energy.detach().cpu().numpy().astype(np.float32)
            table[f"{prefix}_centroid"] = centroid
            table[f"{prefix}_high_share"] = high_share
            phi_sums[prefix] += phi.sum(axis=0)
            phi_counts[prefix] += frames

        contribution_denominator = sum(
            np.square(np.asarray(table[f"{part}_energy"], dtype=np.float64)) for part in PART_ORDER
        )
        contribution_denominator = np.maximum(contribution_denominator, np.finfo(np.float64).eps)
        for part in PART_ORDER:
            table[f"{part}_contribution"] = (
                np.square(np.asarray(table[f"{part}_energy"], dtype=np.float64))
                / contribution_denominator
            ).astype(np.float32)
        frame_tables.append(pd.DataFrame(table))
        if clip_index % 100 == 0:
            print(f"Processed {clip_index} clips")

    if not frame_tables:
        raise RuntimeError(f"No token clips found for split '{args.split}'")
    frame_metrics = pd.concat(frame_tables, ignore_index=True)
    clip_metrics = build_clip_metrics(frame_metrics, token_fps)
    action_summary = build_action_summary(clip_metrics, args.minimum_action_clips)
    correlations = correlation_rows(frame_metrics, clip_metrics)
    representatives = representative_rows(clip_metrics)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_metrics.to_csv(output_dir / "frame_metrics.csv.gz", index=False, compression="gzip")
    clip_metrics.to_csv(output_dir / "clip_metrics.csv", index=False)
    action_summary.to_csv(output_dir / "action_summary.csv", index=False)
    correlations.to_csv(output_dir / "part_correlations.csv", index=False)
    representatives.to_csv(output_dir / "representative_clips.csv", index=False)

    prefixes = ("combined", *PART_ORDER)
    summary: dict[str, object] = {
        "metadata": {
            **codec.metadata(),
            "data_dir": str(Path(args.data_dir).resolve()),
            "token_dir": str(Path(args.token_dir).resolve()),
            "split": args.split,
            "clips": int(len(clip_metrics)),
            "token_frames": int(len(frame_metrics)),
            "duration_hours": float(len(frame_metrics) / token_fps / 3600.0),
            "token_fps": token_fps,
            "window_seconds": float(args.window_seconds),
            "window_token_frames": window,
            "omega_theoretical_max": float(1.0 / math.sqrt(window)),
        },
        "descriptors": {},
        "mean_phi": {
            prefix: [round(float(value), 7) for value in phi_sums[prefix] / phi_counts[prefix]]
            for prefix in prefixes
        },
        "root_schemas": {
            str(key): int(value) for key, value in clip_metrics["root_schema"].value_counts().items()
        },
        "dominant_part_fraction": {
            part: float(
                frame_metrics[[f"{p}_contribution" for p in PART_ORDER]]
                .idxmax(axis=1)
                .eq(f"{part}_contribution")
                .mean()
            )
            for part in PART_ORDER
        },
    }
    descriptors = summary["descriptors"]
    assert isinstance(descriptors, dict)
    for prefix in prefixes:
        omega = frame_metrics[f"{prefix}_omega"]
        energy = frame_metrics[f"{prefix}_energy"]
        descriptors[prefix] = {
            "omega": {
                "mean": float(omega.mean()),
                "median": float(omega.median()),
                "p10": quantile(omega, 0.10),
                "p90": quantile(omega, 0.90),
                "histogram": histogram(omega),
            },
            "energy": {
                "mean": float(energy.mean()),
                "median": float(energy.median()),
                "p10": quantile(energy, 0.10),
                "p90": quantile(energy, 0.90),
                "histogram_log1p": histogram(energy, log=True),
            },
            "centroid": {
                "median": float(frame_metrics[f"{prefix}_centroid"].median()),
                "p90": quantile(frame_metrics[f"{prefix}_centroid"], 0.90),
            },
            "high_share": {
                "median": float(frame_metrics[f"{prefix}_high_share"].median()),
                "p90": quantile(frame_metrics[f"{prefix}_high_share"], 0.90),
            },
        }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        f"Atlas complete: clips={len(clip_metrics)}, token_frames={len(frame_metrics)}, "
        f"duration_hours={summary['metadata']['duration_hours']:.3f}, output={output_dir}"
    )


if __name__ == "__main__":
    main()

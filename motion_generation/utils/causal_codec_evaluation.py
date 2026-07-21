"""Evaluation helpers for the four causal multipart RVQ-VAE body codecs."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.export_multipart_motion_tokens import LoadedPartCodec
from utils.multipart_motion import (
    PART_ORDER,
    load_motion_dict,
    motion_path_for_name,
    split_motion_parts,
)
from utils.rotation_utils import sixd_to_matrix


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean((actual - expected) ** 2)).item())


def mae(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(actual - expected)).item())


def smooth_l1(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float(F.smooth_l1_loss(actual, expected).item())


def rotation_features(part: str, value: torch.Tensor) -> torch.Tensor:
    rotation = value[..., 3:] if part == "lower" else value
    if rotation.shape[-1] % 6:
        raise ValueError(f"{part} rotation dimensions are not divisible by 6")
    return rotation.reshape(*rotation.shape[:-1], -1, 6)


def geodesic_degrees(
    part: str, target: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    target_matrix = sixd_to_matrix(rotation_features(part, target))
    prediction_matrix = sixd_to_matrix(rotation_features(part, prediction))
    relative = target_matrix.transpose(-1, -2) @ prediction_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    return torch.rad2deg(torch.acos(cosine))


def reconstruction_metrics(
    *,
    part: str,
    target_raw: torch.Tensor,
    prediction_raw: torch.Tensor,
    target_normalized: torch.Tensor,
    prediction_normalized: torch.Tensor,
) -> dict[str, float]:
    if target_raw.shape != prediction_raw.shape:
        raise ValueError(
            f"{part} reconstruction shape mismatch: "
            f"{tuple(prediction_raw.shape)} != {tuple(target_raw.shape)}"
        )
    metrics = {
        "feature_mae": mae(prediction_raw, target_raw),
        "feature_rmse": rmse(prediction_raw, target_raw),
        "normalized_smooth_l1": smooth_l1(prediction_normalized, target_normalized),
    }
    if target_raw.shape[0] > 1:
        target_velocity = target_raw[1:] - target_raw[:-1]
        prediction_velocity = prediction_raw[1:] - prediction_raw[:-1]
        target_velocity_normalized = target_normalized[1:] - target_normalized[:-1]
        prediction_velocity_normalized = (
            prediction_normalized[1:] - prediction_normalized[:-1]
        )
        metrics.update(
            velocity_mae=mae(prediction_velocity, target_velocity),
            velocity_rmse=rmse(prediction_velocity, target_velocity),
            normalized_velocity_smooth_l1=smooth_l1(
                prediction_velocity_normalized, target_velocity_normalized
            ),
        )
    else:
        metrics.update(
            velocity_mae=0.0,
            velocity_rmse=0.0,
            normalized_velocity_smooth_l1=0.0,
        )
    if target_raw.shape[0] > 2:
        target_acceleration = target_raw[2:] - 2 * target_raw[1:-1] + target_raw[:-2]
        prediction_acceleration = (
            prediction_raw[2:] - 2 * prediction_raw[1:-1] + prediction_raw[:-2]
        )
        metrics["acceleration_rmse"] = rmse(
            prediction_acceleration, target_acceleration
        )
    else:
        metrics["acceleration_rmse"] = 0.0

    angles = geodesic_degrees(part, target_raw, prediction_raw).reshape(-1)
    metrics["geodesic_deg_mean"] = float(angles.mean().item())
    metrics["geodesic_deg_p90"] = float(torch.quantile(angles, 0.90).item())
    if part == "lower":
        target_root_velocity = target_raw[:, :3]
        prediction_root_velocity = prediction_raw[:, :3]
        target_trajectory = torch.cumsum(target_root_velocity, dim=0)
        prediction_trajectory = torch.cumsum(prediction_root_velocity, dim=0)
        trajectory_distance = torch.linalg.vector_norm(
            prediction_trajectory - target_trajectory, dim=-1
        )
        metrics.update(
            root_velocity_rmse=rmse(prediction_root_velocity, target_root_velocity),
            root_trajectory_error_mean=float(trajectory_distance.mean().item()),
            root_trajectory_final_drift=float(trajectory_distance[-1].item()),
        )
    else:
        metrics.update(
            root_velocity_rmse=math.nan,
            root_trajectory_error_mean=math.nan,
            root_trajectory_final_drift=math.nan,
        )
    return metrics


@torch.no_grad()
def evaluate_part_clip(
    loaded: LoadedPartCodec,
    value: np.ndarray,
    device: torch.device,
    rvq_levels: Sequence[int],
) -> tuple[list[dict[str, float | int | str]], np.ndarray, dict[int, np.ndarray]]:
    part = loaded.part
    complete_frames = (int(value.shape[0]) // loaded.unit_length) * loaded.unit_length
    if complete_frames < loaded.unit_length:
        raise ValueError(f"{part} has fewer than {loaded.unit_length} complete frames")
    target_np = np.asarray(value[:complete_frames], dtype=np.float32)
    normalized_np = loaded.normalizer.normalize(part, target_np)
    target_normalized = torch.as_tensor(normalized_np, device=device)
    x = target_normalized.unsqueeze(0)

    synchronize(device)
    started = time.perf_counter()
    codes = loaded.model.encode({part: x})[part]
    synchronize(device)
    encode_seconds = time.perf_counter() - started
    expected_tokens = complete_frames // loaded.unit_length
    if tuple(codes.shape) != (1, expected_tokens, loaded.num_quantizers):
        raise ValueError(
            f"{part} code shape {tuple(codes.shape)} != "
            f"(1, {expected_tokens}, {loaded.num_quantizers})"
        )

    rows: list[dict[str, float | int | str]] = []
    predictions: dict[int, np.ndarray] = {}
    duration_seconds = complete_frames / 20.0
    for levels in sorted(set(int(level) for level in rvq_levels)):
        if not 1 <= levels <= loaded.num_quantizers:
            raise ValueError(
                f"Requested {levels} RVQ levels for {part}; "
                f"available={loaded.num_quantizers}"
            )
        synchronize(device)
        started = time.perf_counter()
        decoded_normalized = loaded.model.decode({part: codes[..., :levels]})[part]
        synchronize(device)
        decode_seconds = time.perf_counter() - started
        decoded_normalized = decoded_normalized.squeeze(0)
        prediction_raw = loaded.normalizer.denormalize_tensor(
            part, decoded_normalized.unsqueeze(0)
        ).squeeze(0)
        target_raw = torch.as_tensor(target_np, device=device)
        metrics = reconstruction_metrics(
            part=part,
            target_raw=target_raw,
            prediction_raw=prediction_raw,
            target_normalized=target_normalized,
            prediction_normalized=decoded_normalized,
        )
        rows.append(
            {
                "part": part,
                "rvq_levels": levels,
                "source_frames": complete_frames,
                "token_frames": expected_tokens,
                "duration_seconds": duration_seconds,
                "encode_ms": encode_seconds * 1_000.0,
                "decode_ms": decode_seconds * 1_000.0,
                "encode_rtf": encode_seconds / duration_seconds,
                "decode_rtf": decode_seconds / duration_seconds,
                **metrics,
            }
        )
        predictions[levels] = prediction_raw.detach().cpu().numpy()
    return rows, codes.squeeze(0).detach().cpu().numpy(), predictions


def initialize_usage(codecs: Mapping[str, LoadedPartCodec]) -> dict[str, np.ndarray]:
    return {
        part: np.zeros(
            (loaded.num_quantizers, loaded.codebook_size), dtype=np.int64
        )
        for part, loaded in codecs.items()
    }


def update_usage(counts: np.ndarray, codes: np.ndarray) -> None:
    if codes.ndim != 2 or codes.shape[1] != counts.shape[0]:
        raise ValueError(f"Unexpected code array shape {codes.shape}")
    for quantizer in range(counts.shape[0]):
        counts[quantizer] += np.bincount(
            codes[:, quantizer], minlength=counts.shape[1]
        )


def usage_rows(usage: Mapping[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    rows = []
    for part in PART_ORDER:
        counts = usage[part]
        for quantizer, histogram in enumerate(counts):
            total = int(histogram.sum())
            probabilities = histogram.astype(np.float64) / max(1, total)
            positive = probabilities > 0
            entropy_nats = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
            rows.append(
                {
                    "part": part,
                    "quantizer": quantizer,
                    "tokens": total,
                    "used_codes": int(np.count_nonzero(histogram)),
                    "utilization": float(np.count_nonzero(histogram) / len(histogram)),
                    "entropy_bits": entropy_nats / math.log(2.0),
                    "effective_codes": math.exp(entropy_nats),
                    "top_code_share": float(probabilities.max()) if total else 0.0,
                }
            )
    return rows


@torch.no_grad()
def evaluate_dataset(
    *,
    codecs: Mapping[str, LoadedPartCodec],
    data_dir: Path,
    names: Sequence[str],
    device: torch.device,
    rvq_levels: Sequence[int] = (4,),
    max_clips: int | None = None,
    max_examples: int = 3,
    root_abs_threshold: float = 10.0,
) -> dict[str, object]:
    selected = list(names[:max_clips] if max_clips is not None else names)
    motion_dir = Path(data_dir) / "motion_data"
    rows: list[dict[str, float | int | str]] = []
    errors = []
    examples: Dict[str, Dict[str, object]] = {}
    usage = initialize_usage(codecs)
    root_schemas = Counter()
    evaluated = 0
    for index, name in enumerate(selected, start=1):
        path = motion_path_for_name(motion_dir, name)
        try:
            parts, metadata = split_motion_parts(
                load_motion_dict(path), abs_threshold=root_abs_threshold
            )
            root_schemas[str(metadata["root_schema"])] += 1
            example = {"target": {}, "prediction": {}}
            for part in PART_ORDER:
                part_rows, codes, predictions = evaluate_part_clip(
                    codecs[part], parts[part], device, rvq_levels
                )
                for row in part_rows:
                    row["name"] = name
                    row["root_schema"] = str(metadata["root_schema"])
                rows.extend(part_rows)
                update_usage(usage[part], codes)
                if len(examples) < max_examples:
                    complete_frames = part_rows[0]["source_frames"]
                    example["target"][part] = np.asarray(
                        parts[part][:complete_frames], dtype=np.float32
                    )
                    example["prediction"][part] = predictions[max(predictions)]
            if len(examples) < max_examples:
                examples[name] = example
            evaluated += 1
        except Exception as exc:
            errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        if index % 50 == 0:
            print(
                f"Evaluated {index}/{len(selected)} clips; "
                f"valid={evaluated}, errors={len(errors)}",
                flush=True,
            )
    return {
        "rows": rows,
        "usage_rows": usage_rows(usage),
        "examples": examples,
        "errors": errors,
        "requested_clips": len(selected),
        "evaluated_clips": evaluated,
        "root_schemas": dict(root_schemas),
    }

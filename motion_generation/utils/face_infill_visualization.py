"""Paired body and ARKit visualizations for face-enabled infill models."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from utils.constants import (
    BODY_JOINTS_ID,
    KINEMATIC_CHAIN,
    LEFT_HAND_JOINTS_ID,
    RIGHT_HAND_JOINTS_ID,
)
from utils.fk_model import WorldPosFromQuat
from utils.multipart_motion import (
    ARKIT_BLENDSHAPE_NAMES,
    FACE_PART,
    MULTIMODAL_PART_ORDER,
    canonicalize_body_root,
    load_face_coefficients,
    load_motion_dict,
    merge_parts_to_legacy_motion,
    motion_path_for_name,
)
from utils.rotation_utils import sixd_to_quaternion
from utils.variable_c2f_evaluation import (
    EvalWindowRecord,
    InfillModelSpec,
    decode_multipart_part_batch,
    assemble_tiled_token_sequences,
    infer_window_records,
    tiled_gap_windows,
    usable_token_frames,
)


DISPLAY_FACE_CHANNELS: tuple[str, ...] = (
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthSmileLeft",
    "mouthSmileRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "browInnerUp",
)


@dataclass(frozen=True)
class FaceInfillVisualComparison:
    name: str
    gap: int
    left_idx: int
    raw_frame_start: int
    infill_start: int
    infill_end: int
    fps: int
    positions: Mapping[str, np.ndarray]
    faces: Mapping[str, np.ndarray]
    body_features: Mapping[str, np.ndarray]
    metrics: Mapping[str, Mapping[str, float]]

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(self.positions.keys())

    @property
    def frames(self) -> int:
        return min(len(value) for value in self.positions.values())


@dataclass(frozen=True)
class FullClipFaceInfillComparison:
    name: str
    gap: int
    fps: int
    source_token_frames: int
    positions: Mapping[str, np.ndarray]
    faces: Mapping[str, np.ndarray]
    body_features: Mapping[str, np.ndarray]
    metrics: Mapping[str, Mapping[str, float]]
    generated_mask: np.ndarray

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(self.positions.keys())

    @property
    def frames(self) -> int:
        return min(len(value) for value in self.positions.values())


def select_representative_window_rows(
    frame: pd.DataFrame,
    *,
    model_name: str,
    gaps: Sequence[int],
    metric: str = "face_rmse",
) -> pd.DataFrame:
    """Select the row nearest the model's per-gap median error."""
    required = {"model", "gap", "sequence_idx", "left_idx", "name", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Window metric table is missing columns: {missing}")

    selected = []
    for gap in gaps:
        group = frame[
            frame["model"].eq(model_name) & frame["gap"].eq(int(gap))
        ].dropna(subset=[metric])
        if group.empty:
            raise ValueError(f"No {model_name} rows available for gap={gap}")
        median = float(group[metric].median())
        row = group.loc[(group[metric] - median).abs().idxmin()].copy()
        row["selection_median"] = median
        row["selection_distance"] = abs(float(row[metric]) - median)
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def select_representative_clip_row(
    frame: pd.DataFrame,
    *,
    model_name: str,
    gaps: Sequence[int],
    metric: str = "face_rmse",
) -> pd.DataFrame:
    """Select one clip nearest the median aggregate error across all gaps."""
    required = {"model", "gap", "sequence_idx", "name", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Window metric table is missing columns: {missing}")
    requested = {int(gap) for gap in gaps}
    values = frame[
        frame["model"].eq(model_name) & frame["gap"].isin(requested)
    ].dropna(subset=[metric])
    grouped = (
        values.groupby(["sequence_idx", "name"], as_index=False)
        .agg(**{metric: (metric, "mean")}, observed_gaps=("gap", "nunique"))
    )
    grouped = grouped[grouped["observed_gaps"].eq(len(requested))]
    if grouped.empty:
        raise ValueError(
            f"No {model_name} clip has metrics for every requested gap {sorted(requested)}"
        )
    median = float(grouped[metric].median())
    selected = grouped.loc[[(grouped[metric] - median).abs().idxmin()]].copy()
    selected["selection_median"] = median
    selected["selection_distance"] = (selected[metric] - median).abs()
    return selected.reset_index(drop=True)


def _legacy_motion_to_quat(
    motion: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    body = torch.as_tensor(motion["body"], dtype=torch.float32, device=device)
    left = torch.as_tensor(motion["left"], dtype=torch.float32, device=device)
    right = torch.as_tensor(motion["right"], dtype=torch.float32, device=device)
    frames = min(len(body), len(left), len(right))
    body, left, right = body[:frames], left[:frames], right[:frames]

    root = torch.cumsum(body[:, :3], dim=0)
    root = root + torch.tensor([0.0, 0.0, 102.0], device=device)
    body_quat = sixd_to_quaternion(body[:, 3:].reshape(-1, 6)).reshape(frames, 25, 4)
    left_quat = sixd_to_quaternion(left.reshape(-1, 6)).reshape(frames, 20, 4)
    right_quat = sixd_to_quaternion(right.reshape(-1, 6)).reshape(frames, 20, 4)

    quat = torch.zeros(frames, 63, 4, dtype=torch.float32, device=device)
    quat[:, BODY_JOINTS_ID] = body_quat
    quat[:, LEFT_HAND_JOINTS_ID[1:]] = left_quat[:, 1:]
    quat[:, RIGHT_HAND_JOINTS_ID[1:]] = right_quat[:, 1:]
    return quat, root


@torch.no_grad()
def _motion_positions(
    motions: Mapping[str, Mapping[str, np.ndarray]],
    *,
    template_bvh: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    labels = list(motions)
    values = [_legacy_motion_to_quat(motions[label], device) for label in labels]
    frames = min(value[0].shape[0] for value in values)
    quats = torch.stack([value[0][:frames] for value in values])
    roots = torch.stack([value[1][:frames] for value in values])
    fk = WorldPosFromQuat(template_bvh_path=str(template_bvh)).to(device).eval()
    positions = fk(quats, roots).detach().cpu().numpy().astype(np.float32)
    return {label: positions[idx] for idx, label in enumerate(labels)}


def _rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    length = min(len(reference), len(candidate))
    delta = np.asarray(candidate[:length], dtype=np.float64) - np.asarray(
        reference[:length], dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


@torch.no_grad()
def prepare_face_infill_visual_comparison(
    record: EvalWindowRecord,
    models: Mapping[str, tuple[Any, InfillModelSpec]],
    codecs: Mapping[str, Any],
    *,
    device: torch.device,
    template_bvh: Path,
    motion_dir: Path,
    face_dir: Path,
    fps: int = 20,
    part_order: Sequence[str] = MULTIMODAL_PART_ORDER,
) -> FaceInfillVisualComparison:
    """Infer one shared window and decode raw GT, codec GT, and every model."""
    if FACE_PART not in part_order:
        raise ValueError("Face visualization requires the face RVQ part")
    gt_tokens = np.asarray(record.example.motion_tokens, dtype=np.int64)
    token_variants: Dict[str, np.ndarray] = {"Codec GT": gt_tokens}
    for label, (model, spec) in models.items():
        if not spec.supports_gap(record.gap_frames):
            raise ValueError(f"{spec.name} does not support gap={record.gap_frames}")
        prediction = infer_window_records(
            model,
            spec,
            [record],
            batch_size=1,
            device=device,
            slot_constrained=False,
        )[0]
        complete = gt_tokens.copy()
        complete[1:-1] = prediction
        token_variants[label] = complete

    labels = list(token_variants)
    decoded_parts = decode_multipart_part_batch(
        np.stack([token_variants[label] for label in labels]),
        codecs,
        device,
        part_order=part_order,
    )
    decoded_motions = {
        label: merge_parts_to_legacy_motion(
            {
                part: decoded_parts[part][idx]
                for part in part_order
                if part != FACE_PART
            }
        )
        for idx, label in enumerate(labels)
    }
    decoded_faces = {
        label: np.asarray(decoded_parts[FACE_PART][idx], dtype=np.float32)
        for idx, label in enumerate(labels)
    }

    unit_length = int(next(iter(codecs.values())).unit_length)
    raw_start = int(record.left_idx) * unit_length
    raw_motion = load_motion_dict(
        motion_path_for_name(Path(motion_dir), record.name)
    )
    canonical_body, _, _ = canonicalize_body_root(raw_motion["body"])
    raw_face = load_face_coefficients(
        motion_path_for_name(Path(face_dir), record.name)
    )
    decoded_frames = min(
        min(len(value["body"]) for value in decoded_motions.values()),
        min(len(value) for value in decoded_faces.values()),
    )
    raw_available = min(
        len(canonical_body),
        len(raw_motion["left"]),
        len(raw_motion["right"]),
        len(raw_face),
    ) - raw_start
    frames = min(decoded_frames, raw_available)
    if frames < 2:
        raise ValueError(
            f"Not enough aligned frames for {record.name} at raw frame {raw_start}"
        )

    raw_gt_motion = {
        "body": np.asarray(canonical_body[raw_start : raw_start + frames], dtype=np.float32),
        "left": np.asarray(raw_motion["left"][raw_start : raw_start + frames], dtype=np.float32),
        "right": np.asarray(raw_motion["right"][raw_start : raw_start + frames], dtype=np.float32),
    }
    body_features: Dict[str, np.ndarray] = {"Raw GT": raw_gt_motion["body"]}
    faces: Dict[str, np.ndarray] = {
        "Raw GT": np.asarray(raw_face[raw_start : raw_start + frames], dtype=np.float32)
    }
    motions: Dict[str, Mapping[str, np.ndarray]] = {"Raw GT": raw_gt_motion}
    for label in labels:
        motions[label] = {
            key: np.asarray(value[:frames], dtype=np.float32)
            for key, value in decoded_motions[label].items()
        }
        body_features[label] = motions[label]["body"]
        faces[label] = decoded_faces[label][:frames]

    positions = _motion_positions(
        motions,
        template_bvh=Path(template_bvh),
        device=device,
    )
    raw_body = body_features["Raw GT"]
    raw_face_value = faces["Raw GT"]
    metrics = {
        label: {
            "body_rmse": 0.0 if label == "Raw GT" else _rmse(raw_body, body_features[label]),
            "face_rmse": 0.0 if label == "Raw GT" else _rmse(raw_face_value, faces[label]),
        }
        for label in positions
    }

    frames_per_token = frames / max(1, len(gt_tokens))
    infill_start = int(round(frames_per_token))
    infill_end = int(round((record.gap_frames + 1) * frames_per_token))
    infill_start = max(0, min(infill_start, frames - 1))
    infill_end = max(infill_start + 1, min(infill_end, frames))
    return FaceInfillVisualComparison(
        name=record.name,
        gap=int(record.gap_frames),
        left_idx=int(record.left_idx),
        raw_frame_start=raw_start,
        infill_start=infill_start,
        infill_end=infill_end,
        fps=int(fps),
        positions=positions,
        faces=faces,
        body_features=body_features,
        metrics=metrics,
    )


@torch.no_grad()
def prepare_tiled_full_clip_comparison(
    item: Mapping[str, Any],
    models: Mapping[str, tuple[Any, InfillModelSpec]],
    codecs: Mapping[str, Any],
    *,
    gap: int,
    device: torch.device,
    template_bvh: Path,
    motion_dir: Path,
    face_dir: Path | None = None,
    batch_size: int = 32,
    fps: int = 20,
    part_order: Sequence[str] = MULTIMODAL_PART_ORDER,
) -> FullClipFaceInfillComparison:
    """Generate tiled body or body+face infills while retaining anchors and clip tail."""
    face_enabled = FACE_PART in part_order
    if face_enabled and face_dir is None:
        raise ValueError("face_dir is required when face is present in part_order")
    sequence = [item]
    records = tiled_gap_windows(sequence, int(gap))
    if not records:
        raise ValueError(f"{item.get('name', '<unknown>')} is too short for gap={gap}")

    source = np.asarray(item["motion_tokens"], dtype=np.int64)
    usable = usable_token_frames(item)
    token_variants: Dict[str, np.ndarray] = {}
    codec_gt: np.ndarray | None = None
    covered_tokens = 0
    for label, (model, spec) in models.items():
        if not spec.supports_gap(gap):
            raise ValueError(f"{spec.name} does not support gap={gap}")
        predictions = infer_window_records(
            model,
            spec,
            records,
            batch_size=batch_size,
            device=device,
            slot_constrained=False,
        )
        gt_full, pred_full = assemble_tiled_token_sequences(
            sequence, records, predictions
        )[0]
        gt_value = np.asarray(gt_full, dtype=np.int64)
        pred_value = np.asarray(pred_full, dtype=np.int64)
        covered_tokens = len(gt_value)
        if covered_tokens < usable:
            tail = source[covered_tokens:usable]
            gt_value = np.concatenate([gt_value, tail], axis=0)
            pred_value = np.concatenate([pred_value, tail], axis=0)
        if codec_gt is None:
            codec_gt = gt_value
        elif not np.array_equal(codec_gt, gt_value):
            raise RuntimeError("Models did not receive identical tiled ground-truth tokens")
        token_variants[label] = pred_value
    if codec_gt is None:
        raise ValueError("At least one infill model is required")

    all_token_variants: Dict[str, np.ndarray] = {"Codec GT": codec_gt, **token_variants}
    labels = list(all_token_variants)
    decoded_parts = decode_multipart_part_batch(
        np.stack([all_token_variants[label] for label in labels]),
        codecs,
        device,
        part_order=part_order,
    )
    decoded_motions = {
        label: merge_parts_to_legacy_motion(
            {
                part: decoded_parts[part][idx]
                for part in part_order
                if part != FACE_PART
            }
        )
        for idx, label in enumerate(labels)
    }
    decoded_faces = (
        {
            label: np.asarray(decoded_parts[FACE_PART][idx], dtype=np.float32)
            for idx, label in enumerate(labels)
        }
        if face_enabled
        else {}
    )

    name = str(item["name"])
    raw_motion = load_motion_dict(motion_path_for_name(Path(motion_dir), name))
    canonical_body, _, _ = canonicalize_body_root(raw_motion["body"])
    frame_limits = [
        min(len(value["body"]) for value in decoded_motions.values()),
        len(canonical_body),
        len(raw_motion["left"]),
        len(raw_motion["right"]),
    ]
    raw_face: np.ndarray | None = None
    if face_enabled:
        raw_face = load_face_coefficients(motion_path_for_name(Path(face_dir), name))
        frame_limits.extend(
            [min(len(value) for value in decoded_faces.values()), len(raw_face)]
        )
    frames = min(frame_limits)
    if frames < 2:
        raise ValueError(f"Not enough aligned full-clip frames for {name}")

    raw_gt_motion = {
        "body": np.asarray(canonical_body[:frames], dtype=np.float32),
        "left": np.asarray(raw_motion["left"][:frames], dtype=np.float32),
        "right": np.asarray(raw_motion["right"][:frames], dtype=np.float32),
    }
    motions: Dict[str, Mapping[str, np.ndarray]] = {"Raw GT": raw_gt_motion}
    body_features: Dict[str, np.ndarray] = {"Raw GT": raw_gt_motion["body"]}
    faces: Dict[str, np.ndarray] = {}
    if raw_face is not None:
        faces["Raw GT"] = np.asarray(raw_face[:frames], dtype=np.float32)
    for label in labels:
        motions[label] = {
            key: np.asarray(value[:frames], dtype=np.float32)
            for key, value in decoded_motions[label].items()
        }
        body_features[label] = motions[label]["body"]
        if face_enabled:
            faces[label] = decoded_faces[label][:frames]

    positions = _motion_positions(
        motions,
        template_bvh=Path(template_bvh),
        device=device,
    )
    raw_body_value = body_features["Raw GT"]
    metrics: Dict[str, Dict[str, float]] = {}
    for label in positions:
        value = {
            "body_rmse": 0.0
            if label == "Raw GT"
            else _rmse(raw_body_value, body_features[label])
        }
        if face_enabled:
            value["face_rmse"] = (
                0.0 if label == "Raw GT" else _rmse(faces["Raw GT"], faces[label])
            )
        metrics[label] = value

    token_generated = np.zeros(len(codec_gt), dtype=bool)
    for record in records:
        token_generated[
            record.left_idx + 1 : record.left_idx + record.gap_frames + 1
        ] = True
    frame_to_token = np.minimum(
        np.floor(np.arange(frames) * len(token_generated) / frames).astype(np.int64),
        len(token_generated) - 1,
    )
    generated_mask = token_generated[frame_to_token]
    return FullClipFaceInfillComparison(
        name=name,
        gap=int(gap),
        fps=int(fps),
        source_token_frames=len(codec_gt),
        positions=positions,
        faces=faces,
        body_features=body_features,
        metrics=metrics,
        generated_mask=generated_mask,
    )


def _skeleton_edges() -> tuple[tuple[int, int], ...]:
    edges = []
    seen = set()
    for chain in KINEMATIC_CHAIN:
        for parent, child in zip(chain[:-1], chain[1:]):
            edge = (int(parent), int(child))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return tuple(edges)


def _project_positions(positions: np.ndarray) -> np.ndarray:
    value = np.asarray(positions, dtype=np.float32)
    value = value - value[:1, :1]
    x = value[..., 0] + 0.25 * value[..., 2]
    y = -value[..., 1] + 0.12 * value[..., 2]
    return np.stack([x, y], axis=-1)


def _face_limits(faces: Mapping[str, np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(value).reshape(-1) for value in faces.values()])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(finite, [0.01, 0.99])
    low = min(0.0, float(low))
    high = max(0.1, float(high))
    if high <= low:
        high = low + 1.0
    return low, high


def plot_face_infill_summary(
    comparison: FaceInfillVisualComparison,
    *,
    keyframes: int = 5,
    output_path: Path | None = None,
):
    """Plot full-body keyframes and exact 51D ARKit heatmaps."""
    import matplotlib.pyplot as plt

    variants = comparison.variants
    keyframes = max(3, int(keyframes))
    inner = np.linspace(
        comparison.infill_start,
        max(comparison.infill_start, comparison.infill_end - 1),
        keyframes - 2,
    ).round().astype(int)
    frame_indices = np.concatenate(
        ([max(0, comparison.infill_start - 1)], inner, [min(comparison.frames - 1, comparison.infill_end)])
    )
    projected = {label: _project_positions(comparison.positions[label]) for label in variants}
    all_points = np.concatenate([value.reshape(-1, 2) for value in projected.values()])
    finite = all_points[np.isfinite(all_points).all(axis=1)]
    low = finite.min(axis=0) if len(finite) else np.array([-1.0, -1.0])
    high = finite.max(axis=0) if len(finite) else np.array([1.0, 1.0])
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) * 0.55, 0.5)
    face_low, face_high = _face_limits(comparison.faces)
    edges = _skeleton_edges()

    fig = plt.figure(figsize=(4.0 * keyframes + 7.0, 3.2 * len(variants)))
    grid = fig.add_gridspec(
        len(variants),
        keyframes + 1,
        width_ratios=[1.0] * keyframes + [2.2],
        hspace=0.25,
        wspace=0.12,
    )
    heatmap = None
    for row, label in enumerate(variants):
        body_rmse = comparison.metrics[label]["body_rmse"]
        face_rmse = comparison.metrics[label]["face_rmse"]
        for col, frame_idx in enumerate(frame_indices):
            axis = fig.add_subplot(grid[row, col])
            points = projected[label][min(frame_idx, len(projected[label]) - 1)]
            for parent, child in edges:
                axis.plot(
                    points[[parent, child], 0],
                    points[[parent, child], 1],
                    color="#245b78" if label == "Raw GT" else "#b24b35",
                    linewidth=1.4,
                )
            axis.scatter(points[:, 0], points[:, 1], s=3, color="#202020")
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] + radius, center[1] - radius)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            in_gap = comparison.infill_start <= frame_idx < comparison.infill_end
            axis.set_facecolor("#fff0ed" if in_gap else "#f5f5f3")
            if row == 0:
                region = "infill" if in_gap else "anchor"
                axis.set_title(f"frame {frame_idx} ({region})", fontsize=9)
            if col == 0:
                axis.set_ylabel(
                    f"{label}\nbody RMSE {body_rmse:.4f}\nface RMSE {face_rmse:.4f}",
                    fontsize=9,
                )

        axis = fig.add_subplot(grid[row, -1])
        heatmap = axis.imshow(
            comparison.faces[label].T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=face_low,
            vmax=face_high,
        )
        axis.axvspan(
            comparison.infill_start - 0.5,
            comparison.infill_end - 0.5,
            color="#4db6ac",
            alpha=0.16,
        )
        selected = [ARKIT_BLENDSHAPE_NAMES.index(name) for name in DISPLAY_FACE_CHANNELS]
        axis.set_yticks(selected)
        axis.set_yticklabels(DISPLAY_FACE_CHANNELS, fontsize=7)
        axis.set_xlabel("20 FPS frame")
        if row == 0:
            axis.set_title("51D ARKit coefficients (selected labels shown)")

    if heatmap is not None:
        fig.colorbar(heatmap, ax=fig.axes, fraction=0.012, pad=0.01, label="coefficient")
    fig.suptitle(
        f"{comparison.name} | gap={comparison.gap} token frames | "
        f"raw start={comparison.raw_frame_start} | shaded region=infill",
        fontsize=13,
        y=0.995,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def save_face_infill_animation(
    comparison: FaceInfillVisualComparison,
    output_path: Path,
    *,
    frame_step: int = 2,
) -> Path:
    """Save a synchronized body skeleton and facial-control GIF."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    variants = comparison.variants
    projected = {label: _project_positions(comparison.positions[label]) for label in variants}
    all_points = np.concatenate([value.reshape(-1, 2) for value in projected.values()])
    finite = all_points[np.isfinite(all_points).all(axis=1)]
    low = finite.min(axis=0) if len(finite) else np.array([-1.0, -1.0])
    high = finite.max(axis=0) if len(finite) else np.array([1.0, 1.0])
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) * 0.55, 0.5)
    edges = _skeleton_edges()

    fig, axes = plt.subplots(2, len(variants), figsize=(4.0 * len(variants), 7.0))
    if len(variants) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    body_lines: Dict[str, list[Any]] = {}
    face_bars: Dict[str, Any] = {}
    for col, label in enumerate(variants):
        body_axis = axes[0, col]
        body_axis.set_xlim(center[0] - radius, center[0] + radius)
        body_axis.set_ylim(center[1] + radius, center[1] - radius)
        body_axis.set_aspect("equal")
        body_axis.set_xticks([])
        body_axis.set_yticks([])
        body_axis.set_title(label)
        body_lines[label] = [
            body_axis.plot([], [], color="#245b78" if label == "Raw GT" else "#b24b35", linewidth=1.5)[0]
            for _ in edges
        ]

        face_axis = axes[1, col]
        face_bars[label] = face_axis.barh(
            np.arange(len(selected)),
            np.zeros(len(selected)),
            color=bar_colors,
        )
        face_axis.set_xlim(face_low, face_high)
        face_axis.set_yticks(np.arange(len(selected)))
        face_axis.set_yticklabels(DISPLAY_FACE_CHANNELS, fontsize=7)
        face_axis.invert_yaxis()
        face_axis.grid(axis="x", alpha=0.2)
        face_axis.set_xlabel("ARKit coefficient")

    title = fig.suptitle("")
    frame_step = max(1, int(frame_step))
    frames = list(range(0, comparison.frames, frame_step))
    if frames[-1] != comparison.frames - 1:
        frames.append(comparison.frames - 1)

    def update(frame_idx: int):
        in_gap = comparison.infill_start <= frame_idx < comparison.infill_end
        region = "INFILL" if in_gap else "context anchor"
        title.set_text(
            f"{comparison.name} | gap={comparison.gap} | frame {frame_idx + 1}/{comparison.frames} | {region}"
        )
        artists = [title]
        for col, label in enumerate(variants):
            points = projected[label][frame_idx]
            for line, (parent, child) in zip(body_lines[label], edges):
                line.set_data(points[[parent, child], 0], points[[parent, child], 1])
                artists.append(line)
            values = comparison.faces[label][frame_idx, selected]
            for bar, value in zip(face_bars[label], values):
                bar.set_width(float(value))
                artists.append(bar)
            color = "#fff0ed" if in_gap else "#f5f5f3"
            axes[0, col].set_facecolor(color)
            axes[1, col].set_facecolor(color)
        return artists

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 * frame_step / comparison.fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output_path,
        writer=PillowWriter(fps=max(1, round(comparison.fps / frame_step))),
        dpi=85,
    )
    plt.close(fig)
    return output_path


def _project_root_centered_positions(positions: np.ndarray) -> np.ndarray:
    value = np.asarray(positions, dtype=np.float32)
    value = value - value[:, :1]
    x = value[..., 0] + 0.25 * value[..., 2]
    y = -value[..., 1] + 0.12 * value[..., 2]
    return np.stack([x, y], axis=-1)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    value = np.asarray(mask, dtype=bool)
    padded = np.pad(value.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _audio_rms_envelope(
    audio_path: Path | None,
    *,
    frames: int,
    fps: int,
) -> np.ndarray:
    if audio_path is None or not Path(audio_path).exists():
        return np.zeros(frames, dtype=np.float32)
    try:
        import soundfile as sf
    except ImportError:
        return np.zeros(frames, dtype=np.float32)
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim > 1:
        value = value.mean(axis=1)
    envelope = np.zeros(frames, dtype=np.float32)
    for frame_idx in range(frames):
        start = int(round(frame_idx * sample_rate / fps))
        end = int(round((frame_idx + 1) * sample_rate / fps))
        chunk = value[start:min(end, len(value))]
        if len(chunk):
            envelope[frame_idx] = float(np.sqrt(np.mean(np.square(chunk))))
    scale = float(np.quantile(envelope, 0.95)) if np.any(envelope) else 0.0
    if scale > 0:
        envelope = np.clip(envelope / scale, 0.0, 1.0)
    return envelope


def plot_tiled_full_clip_summary(
    comparison: FullClipFaceInfillComparison,
    *,
    keyframes: int = 7,
    output_path: Path | None = None,
):
    """Plot body keyframes and, when available, full-clip ARKit heatmaps."""
    import matplotlib.pyplot as plt

    variants = comparison.variants
    keyframes = max(3, int(keyframes))
    frame_indices = np.linspace(0, comparison.frames - 1, keyframes).round().astype(int)
    projected = {
        label: _project_root_centered_positions(comparison.positions[label])
        for label in variants
    }
    all_points = np.concatenate([value.reshape(-1, 2) for value in projected.values()])
    finite = all_points[np.isfinite(all_points).all(axis=1)]
    low = finite.min(axis=0) if len(finite) else np.array([-1.0, -1.0])
    high = finite.max(axis=0) if len(finite) else np.array([1.0, 1.0])
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) * 0.55, 0.5)
    has_face = bool(comparison.faces)
    face_low, face_high = _face_limits(comparison.faces) if has_face else (0.0, 1.0)
    edges = _skeleton_edges()
    generated_runs = _true_runs(comparison.generated_mask)

    extra_columns = 1 if has_face else 0
    fig = plt.figure(
        figsize=(3.5 * keyframes + (7.0 if has_face else 0.0), 3.0 * len(variants))
    )
    grid = fig.add_gridspec(
        len(variants),
        keyframes + extra_columns,
        width_ratios=[1.0] * keyframes + ([2.5] if has_face else []),
        hspace=0.28,
        wspace=0.12,
    )
    heatmap = None
    selected = (
        [ARKIT_BLENDSHAPE_NAMES.index(name) for name in DISPLAY_FACE_CHANNELS]
        if has_face
        else []
    )
    for row, label in enumerate(variants):
        for col, frame_idx in enumerate(frame_indices):
            axis = fig.add_subplot(grid[row, col])
            points = projected[label][frame_idx]
            for parent, child in edges:
                axis.plot(
                    points[[parent, child], 0],
                    points[[parent, child], 1],
                    color="#245b78" if label == "Raw GT" else "#b24b35",
                    linewidth=1.3,
                )
            axis.scatter(points[:, 0], points[:, 1], s=2.5, color="#202020")
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] + radius, center[1] - radius)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            generated = bool(comparison.generated_mask[frame_idx])
            axis.set_facecolor("#fff0ed" if generated else "#f5f5f3")
            if row == 0:
                axis.set_title(
                    f"{frame_idx / comparison.fps:.1f}s\n"
                    f"{'generated' if generated else 'anchor'}",
                    fontsize=8,
                )
            if col == 0:
                metrics = comparison.metrics[label]
                metric_lines = [f"body {metrics['body_rmse']:.4f}"]
                if "face_rmse" in metrics:
                    metric_lines.append(f"face {metrics['face_rmse']:.4f}")
                axis.set_ylabel(
                    f"{label}\n" + "\n".join(metric_lines),
                    fontsize=9,
                )

        if has_face:
            axis = fig.add_subplot(grid[row, -1])
            heatmap = axis.imshow(
                comparison.faces[label].T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="magma",
                vmin=face_low,
                vmax=face_high,
            )
            for start, end in generated_runs:
                axis.axvspan(start - 0.5, end - 0.5, color="#4db6ac", alpha=0.12)
            axis.set_yticks(selected)
            axis.set_yticklabels(DISPLAY_FACE_CHANNELS, fontsize=7)
            axis.set_xlabel("20 FPS frame")
            if row == 0:
                axis.set_title("Full-clip 51D ARKit coefficients")

    if heatmap is not None:
        fig.colorbar(heatmap, ax=fig.axes, fraction=0.012, pad=0.01, label="coefficient")
    generated_fraction = float(np.mean(comparison.generated_mask))
    fig.suptitle(
        f"{comparison.name} | tiled gap={comparison.gap} token frames | "
        f"{comparison.frames / comparison.fps:.1f}s | generated={generated_fraction:.1%}",
        fontsize=13,
        y=0.995,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


class _ArkitFaceGlyph:
    """Lightweight 2D diagnostic driven directly by 51 ARKit coefficients."""

    def __init__(self, axis: Any) -> None:
        from matplotlib.patches import Ellipse, Polygon

        self.axis = axis
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.15, 1.15)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("Schematic ARKit face", fontsize=9)

        skin = "#f1c7a5"
        self.head = Ellipse((0.0, 0.0), 1.62, 2.08, facecolor=skin, edgecolor="#3a302b", linewidth=1.8)
        self.left_cheek = Ellipse((-0.56, -0.05), 0.30, 0.22, facecolor="#e98f88", edgecolor="none", alpha=0.0)
        self.right_cheek = Ellipse((0.56, -0.05), 0.30, 0.22, facecolor="#e98f88", edgecolor="none", alpha=0.0)
        self.left_eye = Ellipse((-0.35, 0.30), 0.34, 0.13, facecolor="white", edgecolor="#292929", linewidth=1.4)
        self.right_eye = Ellipse((0.35, 0.30), 0.34, 0.13, facecolor="white", edgecolor="#292929", linewidth=1.4)
        self.left_pupil = Ellipse((-0.35, 0.30), 0.075, 0.075, facecolor="#202020", edgecolor="none")
        self.right_pupil = Ellipse((0.35, 0.30), 0.075, 0.075, facecolor="#202020", edgecolor="none")
        self.mouth = Polygon(
            [(-0.32, -0.43), (0.0, -0.38), (0.32, -0.43), (0.0, -0.48)],
            closed=True,
            facecolor="#4a1518",
            edgecolor="#9b3f48",
            linewidth=2.0,
        )
        for patch in (
            self.head,
            self.left_cheek,
            self.right_cheek,
            self.left_eye,
            self.right_eye,
            self.left_pupil,
            self.right_pupil,
            self.mouth,
        ):
            axis.add_patch(patch)

        self.left_brow = axis.plot([], [], color="#4a342a", linewidth=3.0, solid_capstyle="round")[0]
        self.right_brow = axis.plot([], [], color="#4a342a", linewidth=3.0, solid_capstyle="round")[0]
        self.nose = axis.plot([0.0, -0.04, 0.0, 0.06], [0.18, -0.02, -0.12, -0.14], color="#8c6652", linewidth=1.4)[0]
        self.mouth_midline = axis.plot([], [], color="#e7a0a4", linewidth=1.0)[0]
        self.status = axis.text(
            0.0,
            -1.06,
            "",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )
        self.artists = [
            self.head,
            self.left_cheek,
            self.right_cheek,
            self.left_eye,
            self.right_eye,
            self.left_pupil,
            self.right_pupil,
            self.mouth,
            self.left_brow,
            self.right_brow,
            self.nose,
            self.mouth_midline,
            self.status,
        ]

    @staticmethod
    def _values(coefficients: np.ndarray) -> Dict[str, float]:
        value = np.asarray(coefficients, dtype=np.float32)
        return {
            name: max(0.0, float(value[idx]))
            for idx, name in enumerate(ARKIT_BLENDSHAPE_NAMES)
        }

    def update(self, coefficients: np.ndarray) -> list[Any]:
        values = self._values(coefficients)

        puff = min(1.5, values["cheekPuff"])
        self.head.set_width(1.62 + 0.08 * puff)
        self.left_cheek.set_alpha(min(0.55, 0.08 + 0.40 * puff + 0.20 * values["cheekSquintLeft"]))
        self.right_cheek.set_alpha(min(0.55, 0.08 + 0.40 * puff + 0.20 * values["cheekSquintRight"]))

        eye_specs = (
            (
                self.left_eye,
                self.left_pupil,
                -0.35,
                values["eyeBlinkLeft"],
                values["eyeWideLeft"],
                values["eyeSquintLeft"],
                values["eyeLookOutLeft"] - values["eyeLookInLeft"],
                values["eyeLookUpLeft"] - values["eyeLookDownLeft"],
            ),
            (
                self.right_eye,
                self.right_pupil,
                0.35,
                values["eyeBlinkRight"],
                values["eyeWideRight"],
                values["eyeSquintRight"],
                values["eyeLookInRight"] - values["eyeLookOutRight"],
                values["eyeLookUpRight"] - values["eyeLookDownRight"],
            ),
        )
        for eye, pupil, base_x, blink, wide, squint, gaze_x, gaze_y in eye_specs:
            aperture = np.clip(0.13 * (1.0 - blink) + 0.08 * wide - 0.05 * squint, 0.012, 0.24)
            eye.set_center((base_x, 0.30))
            eye.set_height(float(aperture))
            pupil.set_center((base_x + 0.055 * np.clip(gaze_x, -1, 1), 0.30 + 0.045 * np.clip(gaze_y, -1, 1)))
            pupil.set_height(float(min(0.075, aperture * 0.75)))
            pupil.set_alpha(float(np.clip((aperture - 0.01) / 0.07, 0.0, 1.0)))

        brow_inner = values["browInnerUp"]
        left_outer_y = 0.57 + 0.15 * values["browOuterUpLeft"] - 0.13 * values["browDownLeft"]
        right_outer_y = 0.57 + 0.15 * values["browOuterUpRight"] - 0.13 * values["browDownRight"]
        left_inner_y = 0.56 + 0.17 * brow_inner - 0.10 * values["browDownLeft"]
        right_inner_y = 0.56 + 0.17 * brow_inner - 0.10 * values["browDownRight"]
        self.left_brow.set_data([-0.60, -0.14], [left_outer_y, left_inner_y])
        self.right_brow.set_data([0.14, 0.60], [right_inner_y, right_outer_y])

        jaw_open = values["jawOpen"]
        mouth_close = values["mouthClose"]
        funnel = values["mouthFunnel"]
        pucker = values["mouthPucker"]
        stretch = 0.5 * (values["mouthStretchLeft"] + values["mouthStretchRight"])
        smile_left = values["mouthSmileLeft"]
        smile_right = values["mouthSmileRight"]
        frown_left = values["mouthFrownLeft"]
        frown_right = values["mouthFrownRight"]
        mouth_shift = 0.10 * (values["mouthRight"] - values["mouthLeft"])
        mouth_shift += 0.08 * (values["jawRight"] - values["jawLeft"])
        width = np.clip(
            0.62 + 0.25 * stretch + 0.08 * (smile_left + smile_right) - 0.27 * pucker - 0.20 * funnel,
            0.22,
            0.92,
        )
        height = np.clip(
            0.045 + 0.34 * jaw_open + 0.14 * funnel - 0.10 * mouth_close,
            0.018,
            0.48,
        )
        center_y = -0.42 - 0.08 * jaw_open
        left_y = center_y + 0.14 * smile_left - 0.13 * frown_left
        right_y = center_y + 0.14 * smile_right - 0.13 * frown_right
        upper_y = center_y + height / 2 + 0.06 * (
            values["mouthUpperUpLeft"] + values["mouthUpperUpRight"]
        )
        lower_y = center_y - height / 2 - 0.06 * (
            values["mouthLowerDownLeft"] + values["mouthLowerDownRight"]
        )
        left_x, right_x = mouth_shift - width / 2, mouth_shift + width / 2
        self.mouth.set_xy(
            np.asarray(
                [
                    [left_x, left_y],
                    [mouth_shift, upper_y],
                    [right_x, right_y],
                    [mouth_shift, lower_y],
                ]
            )
        )
        self.mouth_midline.set_data(
            [left_x + 0.05 * width, mouth_shift, right_x - 0.05 * width],
            [left_y, center_y, right_y],
        )

        sneer = values["noseSneerLeft"] - values["noseSneerRight"]
        self.nose.set_data(
            [0.0, -0.04, 0.0, 0.06],
            [0.18, -0.02, -0.12 + 0.03 * sneer, -0.14 - 0.03 * sneer],
        )
        ranked = np.argsort(np.asarray(coefficients, dtype=np.float32))[-3:][::-1]
        active = " | ".join(
            f"{ARKIT_BLENDSHAPE_NAMES[idx]} {float(coefficients[idx]):.2f}"
            for idx in ranked
        )
        self.status.set_text(active)
        return self.artists


def save_tiled_full_clip_video(
    comparison: FullClipFaceInfillComparison,
    output_path: Path,
    *,
    audio_path: Path | None = None,
    frame_step: int = 2,
) -> Path:
    """Render a body or body+face full-clip MP4 and mux original audio."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for full-clip MP4 rendering")
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg

    variants = comparison.variants
    projected = {
        label: _project_root_centered_positions(comparison.positions[label])
        for label in variants
    }
    all_points = np.concatenate([value.reshape(-1, 2) for value in projected.values()])
    finite = all_points[np.isfinite(all_points).all(axis=1)]
    low = finite.min(axis=0) if len(finite) else np.array([-1.0, -1.0])
    high = finite.max(axis=0) if len(finite) else np.array([1.0, 1.0])
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) * 0.55, 0.5)
    has_face = bool(comparison.faces)
    edges = _skeleton_edges()

    fig = plt.figure(figsize=(4.0 * len(variants), 7.6 if has_face else 5.2))
    if has_face:
        grid = fig.add_gridspec(3, len(variants), height_ratios=[3.3, 2.5, 0.65])
        timeline_row = 2
    else:
        grid = fig.add_gridspec(2, len(variants), height_ratios=[3.3, 0.65])
        timeline_row = 1
    body_axes = [fig.add_subplot(grid[0, col]) for col in range(len(variants))]
    face_axes = (
        [fig.add_subplot(grid[1, col]) for col in range(len(variants))]
        if has_face
        else []
    )
    timeline_axis = fig.add_subplot(grid[timeline_row, :])
    body_lines: Dict[str, list[Any]] = {}
    face_glyphs: Dict[str, _ArkitFaceGlyph] = {}
    for col, label in enumerate(variants):
        body_axis = body_axes[col]
        body_axis.set_xlim(center[0] - radius, center[0] + radius)
        body_axis.set_ylim(center[1] + radius, center[1] - radius)
        body_axis.set_aspect("equal")
        body_axis.set_xticks([])
        body_axis.set_yticks([])
        metrics = comparison.metrics[label]
        metric_text = f"body RMSE {metrics['body_rmse']:.4f}"
        if "face_rmse" in metrics:
            metric_text += f" | face RMSE {metrics['face_rmse']:.4f}"
        body_axis.set_title(
            f"{label}\n{metric_text}",
            fontsize=9,
        )
        body_lines[label] = [
            body_axis.plot(
                [], [],
                color="#245b78" if label == "Raw GT" else "#b24b35",
                linewidth=1.5,
            )[0]
            for _ in edges
        ]

        if has_face:
            face_axis = face_axes[col]
            face_glyphs[label] = _ArkitFaceGlyph(face_axis)

    x = np.arange(comparison.frames)
    envelope = _audio_rms_envelope(
        Path(audio_path) if audio_path is not None else None,
        frames=comparison.frames,
        fps=comparison.fps,
    )
    timeline_axis.fill_between(
        x,
        0,
        1,
        where=comparison.generated_mask,
        color="#ef9a9a",
        alpha=0.35,
        step="mid",
        label="generated",
    )
    timeline_axis.plot(x, envelope, color="#2f6f8f", linewidth=1.0, label="audio RMS")
    current_line = timeline_axis.axvline(0, color="#202020", linewidth=1.5)
    timeline_axis.set_xlim(0, max(1, comparison.frames - 1))
    timeline_axis.set_ylim(0, 1.05)
    timeline_axis.set_xlabel("20 FPS frame")
    timeline_axis.set_yticks([])
    timeline_axis.legend(loc="upper right", ncol=2, fontsize=8)
    title = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    frame_step = max(1, int(frame_step))
    render_frames = list(range(0, comparison.frames, frame_step))
    if render_frames[-1] != comparison.frames - 1:
        render_frames.append(comparison.frames - 1)

    def update(frame_idx: int):
        generated = bool(comparison.generated_mask[frame_idx])
        region = "GENERATED" if generated else "anchor/context"
        title.set_text(
            f"{comparison.name} | tiled gap={comparison.gap} | "
            f"{frame_idx / comparison.fps:.2f}s | {region}"
        )
        artists = [title, current_line]
        current_line.set_xdata([frame_idx, frame_idx])
        for col, label in enumerate(variants):
            points = projected[label][frame_idx]
            for line, (parent, child) in zip(body_lines[label], edges):
                line.set_data(points[[parent, child], 0], points[[parent, child], 1])
                artists.append(line)
            if has_face:
                artists.extend(face_glyphs[label].update(comparison.faces[label][frame_idx]))
            background = "#fff0ed" if generated else "#f5f5f3"
            body_axes[col].set_facecolor(background)
            if has_face:
                face_axes[col].set_facecolor(background)
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=render_frames,
        interval=1000 * frame_step / comparison.fps,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_only = output_path.with_name(f".{output_path.stem}.video_only.mp4")
    writer = FFMpegWriter(
        fps=max(1, round(comparison.fps / frame_step)),
        codec="libx264",
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    animation.save(video_only, writer=writer, dpi=85)
    plt.close(fig)

    audio_value = Path(audio_path) if audio_path is not None else None
    if audio_value is not None and audio_value.exists():
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_only),
                "-i",
                str(audio_value),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio mux failed: {result.stderr[-1000:]}")
        video_only.unlink(missing_ok=True)
    else:
        output_path.unlink(missing_ok=True)
        video_only.replace(output_path)
    return output_path

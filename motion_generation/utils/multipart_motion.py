from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.constants import LEFT_HAND_JOINTS_ID, RIGHT_HAND_JOINTS_ID


PART_ORDER: Tuple[str, ...] = ("upper", "lower", "feet", "hands")
FACE_PART = "face"
MULTIMODAL_PART_ORDER: Tuple[str, ...] = (*PART_ORDER, FACE_PART)
ARKIT_BLENDSHAPE_NAMES: Tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
ARKIT_LIP_INDICES: Tuple[int, ...] = (24, 26, 31, 33, 34, 37, 39, 40, 41, 42, 47, 48)

# Body feature layout is root(3) + 25 body joints * 6D rotation.
LOWER_BODY_JOINTS: Tuple[int, ...] = (0, 1, 2, 5, 6)
FEET_JOINTS: Tuple[int, ...] = (3, 4, 7, 8)
UPPER_BODY_JOINTS: Tuple[int, ...] = (
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
)

PART_DIMS: Dict[str, int] = {
    "upper": len(UPPER_BODY_JOINTS) * 6,
    "lower": 3 + len(LOWER_BODY_JOINTS) * 6,
    "feet": len(FEET_JOINTS) * 6,
    "hands": (len(LEFT_HAND_JOINTS_ID) - 1 + len(RIGHT_HAND_JOINTS_ID) - 1) * 6,
    FACE_PART: 51,
}


def motion_path_for_name(motion_dir: Path, name: str) -> Path:
    """Resolve split names that may contain POSIX-style subfolders."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return motion_dir / Path(*parts).with_suffix(".npy")


def load_name_list(split_file: Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def load_motion_dict(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict-like .npy motion file: {path}")
    for key in ("body", "left", "right"):
        if key not in data:
            raise KeyError(f"Motion file missing key '{key}': {path}")
    return data


def load_face_coefficients(path: Path) -> np.ndarray:
    """Load one 51D ARKit coefficient sequence."""
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != PART_DIMS[FACE_PART]:
        raise ValueError(f"Expected face shape (T, 51), got {value.shape}: {path}")
    if value.shape[0] == 0:
        raise ValueError(f"Face sequence is empty: {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"Face sequence contains NaN or Inf: {path}")
    return value


def classify_root_channel(
    body: np.ndarray,
    abs_threshold: float = 10.0,
) -> Tuple[str, float]:
    """Classify root channel as absolute position or already-frame-delta."""
    root = np.asarray(body[:, :3], dtype=np.float32)
    mean_norm = float(np.linalg.norm(root, axis=1).mean()) if len(root) else 0.0
    schema = "absolute" if mean_norm >= abs_threshold else "delta"
    return schema, mean_norm


def canonicalize_body_root(
    body: np.ndarray,
    abs_threshold: float = 10.0,
    force_schema: Optional[str] = None,
) -> Tuple[np.ndarray, str, float]:
    """Return body features whose first 3 dims are frame deltas.

    The released data has two root-channel schemas. Older clips already store
    per-frame displacement/velocity; newer clips store absolute pelvis/root
    position around 100 cm height. The codec contract is always frame delta.
    """
    out = np.asarray(body, dtype=np.float32).copy()
    schema, mean_norm = classify_root_channel(out, abs_threshold=abs_threshold)
    if force_schema is not None:
        if force_schema not in {"absolute", "delta"}:
            raise ValueError("force_schema must be one of: absolute, delta")
        schema = force_schema

    root = out[:, :3].copy()
    delta = np.zeros_like(root)
    if schema == "absolute":
        if len(root) > 1:
            delta[1:] = root[1:] - root[:-1]
    else:
        delta[:] = root
        if len(delta):
            delta[0] = 0.0
    out[:, :3] = delta
    return out, schema, mean_norm


def _body_joint_features(body: np.ndarray, joints: Sequence[int]) -> np.ndarray:
    chunks = [body[:, 3 + joint * 6 : 3 + (joint + 1) * 6] for joint in joints]
    return np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)


def split_motion_parts(
    motion: Mapping[str, np.ndarray],
    abs_threshold: float = 10.0,
    force_root_schema: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float | str]]:
    """Split a legacy SentiAvatar motion dict into RVQ training parts."""
    body, schema, mean_norm = canonicalize_body_root(
        np.asarray(motion["body"], dtype=np.float32),
        abs_threshold=abs_threshold,
        force_schema=force_root_schema,
    )
    left = np.asarray(motion["left"], dtype=np.float32)
    right = np.asarray(motion["right"], dtype=np.float32)

    if body.ndim != 2 or body.shape[1] != 153:
        raise ValueError(f"Expected body shape (T, 153), got {body.shape}")
    if left.ndim != 2 or left.shape[1] != 120:
        raise ValueError(f"Expected left shape (T, 120), got {left.shape}")
    if right.ndim != 2 or right.shape[1] != 120:
        raise ValueError(f"Expected right shape (T, 120), got {right.shape}")

    frames = min(body.shape[0], left.shape[0], right.shape[0])
    body = body[:frames]
    left = left[:frames]
    right = right[:frames]

    parts = {
        "upper": _body_joint_features(body, UPPER_BODY_JOINTS),
        "lower": np.concatenate(
            [body[:, :3], _body_joint_features(body, LOWER_BODY_JOINTS)],
            axis=-1,
        ).astype(np.float32, copy=False),
        "feet": _body_joint_features(body, FEET_JOINTS),
        "hands": np.concatenate([left[:, 6:], right[:, 6:]], axis=-1).astype(
            np.float32,
            copy=False,
        ),
    }
    meta: Dict[str, float | str] = {
        "root_schema": schema,
        "root_mean_norm": mean_norm,
        "frames": float(frames),
    }
    return parts, meta


def crop_or_pad_parts(
    parts: Mapping[str, np.ndarray],
    window_size: int,
    start: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    frames = min(value.shape[0] for value in parts.values())
    if frames <= 0:
        raise ValueError("Cannot crop an empty motion sequence")
    if start is None:
        start = 0
    end = min(start + window_size, frames)

    out: Dict[str, np.ndarray] = {}
    for part, value in parts.items():
        clip = np.asarray(value[start:end], dtype=np.float32)
        if clip.shape[0] < window_size:
            pad = window_size - clip.shape[0]
            tail = clip[-1:] if clip.shape[0] else np.zeros((1, value.shape[1]), dtype=np.float32)
            clip = np.concatenate([clip, np.repeat(tail, pad, axis=0)], axis=0)
        out[part] = clip
    return out


def merge_parts_to_legacy_motion(parts: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Merge generated part features back into legacy body/left/right arrays."""
    frames = min(np.asarray(parts[p]).shape[0] for p in PART_ORDER)
    body = np.zeros((frames, 153), dtype=np.float32)
    left = np.zeros((frames, 120), dtype=np.float32)
    right = np.zeros((frames, 120), dtype=np.float32)

    lower = np.asarray(parts["lower"], dtype=np.float32)[:frames]
    feet = np.asarray(parts["feet"], dtype=np.float32)[:frames]
    upper = np.asarray(parts["upper"], dtype=np.float32)[:frames]
    hands = np.asarray(parts["hands"], dtype=np.float32)[:frames]

    body[:, :3] = lower[:, :3]
    _scatter_body_joint_features(body, LOWER_BODY_JOINTS, lower[:, 3:])
    _scatter_body_joint_features(body, FEET_JOINTS, feet)
    _scatter_body_joint_features(body, UPPER_BODY_JOINTS, upper)

    hand_l_body = body[:, 3 + 23 * 6 : 3 + 24 * 6]
    hand_r_body = body[:, 3 + 24 * 6 : 3 + 25 * 6]
    left[:, :6] = hand_l_body
    right[:, :6] = hand_r_body
    left[:, 6:] = hands[:, :114]
    right[:, 6:] = hands[:, 114:]
    return {"body": body, "left": left, "right": right}


def _scatter_body_joint_features(body: np.ndarray, joints: Sequence[int], values: np.ndarray) -> None:
    for offset, joint in enumerate(joints):
        body[:, 3 + joint * 6 : 3 + (joint + 1) * 6] = values[
            :, offset * 6 : (offset + 1) * 6
        ]


@dataclass
class PartNormalizer:
    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]

    def normalize(self, part: str, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean[part]) / self.std[part]).astype(np.float32, copy=False)

    def denormalize_tensor(self, part: str, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean[part], dtype=value.dtype, device=value.device)
        std = torch.as_tensor(self.std[part], dtype=value.dtype, device=value.device)
        return value * std.view(1, 1, -1) + mean.view(1, 1, -1)

    def save(self, path: Path, metadata: Optional[Mapping[str, object]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        if set(self.mean) != set(self.std):
            raise ValueError("Normalizer mean/std keys do not match")
        for part in self.mean:
            payload[f"{part}_mean"] = self.mean[part].astype(np.float32)
            payload[f"{part}_std"] = self.std[part].astype(np.float32)
        np.savez(path, **payload)
        if metadata is not None:
            with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "PartNormalizer":
        data = np.load(path)
        parts = [key[: -len("_mean")] for key in data.files if key.endswith("_mean")]
        if not parts:
            raise ValueError(f"No normalizer statistics found in {path}")
        missing_std = [part for part in parts if f"{part}_std" not in data.files]
        if missing_std:
            raise ValueError(f"Missing std statistics for {missing_std} in {path}")
        mean = {part: data[f"{part}_mean"].astype(np.float32) for part in parts}
        std = {part: data[f"{part}_std"].astype(np.float32) for part in parts}
        return cls(mean=mean, std=std)


def compute_selected_part_normalizer(
    data_root: Path,
    names: Sequence[str],
    part_order: Sequence[str],
    abs_threshold: float = 10.0,
    max_clips: Optional[int] = None,
) -> Tuple[PartNormalizer, Dict[str, object]]:
    """Compute statistics for any body-part subset and/or the ARKit face stream."""
    order = tuple(str(part) for part in part_order)
    unknown = [part for part in order if part not in PART_DIMS]
    if unknown:
        raise ValueError(f"Unknown part(s): {unknown}")
    needs_motion = any(part in PART_ORDER for part in order)
    needs_face = FACE_PART in order
    motion_dir = Path(data_root) / "motion_data"
    face_dir = Path(data_root) / "arkit_data"
    sums = {part: np.zeros(PART_DIMS[part], dtype=np.float64) for part in order}
    sums_sq = {part: np.zeros(PART_DIMS[part], dtype=np.float64) for part in order}
    counts = {part: 0 for part in order}
    status: MutableMapping[str, int] = {
        "absolute": 0,
        "delta": 0,
        "missing": 0,
        "failed": 0,
        "length_mismatch": 0,
    }

    selected = list(names[:max_clips] if max_clips is not None else names)
    for name in selected:
        motion_path = motion_path_for_name(motion_dir, name)
        face_path = motion_path_for_name(face_dir, name)
        if (needs_motion and not motion_path.exists()) or (needs_face and not face_path.exists()):
            status["missing"] += 1
            continue
        try:
            values: Dict[str, np.ndarray] = {}
            if needs_motion:
                motion_parts, meta = split_motion_parts(
                    load_motion_dict(motion_path), abs_threshold=abs_threshold
                )
                status[str(meta["root_schema"])] += 1
                values.update({part: motion_parts[part] for part in order if part in PART_ORDER})
            if needs_face:
                values[FACE_PART] = load_face_coefficients(face_path)
            frames = min(value.shape[0] for value in values.values())
            if any(value.shape[0] != frames for value in values.values()):
                status["length_mismatch"] += 1
        except Exception:
            status["failed"] += 1
            continue
        for part in order:
            value64 = np.asarray(values[part][:frames], dtype=np.float64)
            sums[part] += value64.sum(axis=0)
            sums_sq[part] += np.square(value64).sum(axis=0)
            counts[part] += value64.shape[0]

    mean: Dict[str, np.ndarray] = {}
    std: Dict[str, np.ndarray] = {}
    for part in order:
        if counts[part] == 0:
            raise RuntimeError(f"No frames available while computing stats for part '{part}'")
        part_mean = sums[part] / counts[part]
        variance = np.maximum(sums_sq[part] / counts[part] - np.square(part_mean), 1e-8)
        mean[part] = part_mean.astype(np.float32)
        std[part] = np.maximum(np.sqrt(variance), 1e-4).astype(np.float32)

    metadata: Dict[str, object] = {
        "part_order": list(order),
        "part_dims": {part: PART_DIMS[part] for part in order},
        "root_abs_threshold": abs_threshold,
        "schema_counts": dict(status),
        "num_requested_clips": len(selected),
        "frame_counts": counts,
    }
    return PartNormalizer(mean=mean, std=std), metadata


def compute_part_normalizer(
    motion_dir: Path,
    names: Sequence[str],
    abs_threshold: float = 10.0,
    max_clips: Optional[int] = None,
) -> Tuple[PartNormalizer, Dict[str, object]]:
    sums = {part: np.zeros(PART_DIMS[part], dtype=np.float64) for part in PART_ORDER}
    sums_sq = {part: np.zeros(PART_DIMS[part], dtype=np.float64) for part in PART_ORDER}
    counts = {part: 0 for part in PART_ORDER}
    schema_counts: MutableMapping[str, int] = {"absolute": 0, "delta": 0, "missing": 0, "failed": 0}

    selected = list(names[:max_clips] if max_clips is not None else names)
    for name in selected:
        path = motion_path_for_name(motion_dir, name)
        if not path.exists():
            schema_counts["missing"] += 1
            continue
        try:
            motion = load_motion_dict(path)
            parts, meta = split_motion_parts(motion, abs_threshold=abs_threshold)
        except Exception:
            schema_counts["failed"] += 1
            continue
        schema_counts[str(meta["root_schema"])] += 1
        for part, value in parts.items():
            value64 = np.asarray(value, dtype=np.float64)
            sums[part] += value64.sum(axis=0)
            sums_sq[part] += np.square(value64).sum(axis=0)
            counts[part] += value64.shape[0]

    mean: Dict[str, np.ndarray] = {}
    std: Dict[str, np.ndarray] = {}
    for part in PART_ORDER:
        if counts[part] == 0:
            raise RuntimeError(f"No frames available while computing stats for part '{part}'")
        part_mean = sums[part] / counts[part]
        variance = np.maximum(sums_sq[part] / counts[part] - np.square(part_mean), 1e-8)
        mean[part] = part_mean.astype(np.float32)
        std[part] = np.sqrt(variance).astype(np.float32)
        std[part] = np.maximum(std[part], 1e-4).astype(np.float32)

    metadata: Dict[str, object] = {
        "part_order": list(PART_ORDER),
        "part_dims": PART_DIMS,
        "root_abs_threshold": abs_threshold,
        "schema_counts": dict(schema_counts),
        "num_requested_clips": len(selected),
        "frame_counts": counts,
    }
    return PartNormalizer(mean=mean, std=std), metadata

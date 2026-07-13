"""Reusable workflow for the multipart MSD complexity-error audit.

The notebook in ``motion_generation/notebooks`` intentionally contains only
configuration and workflow cells. Data loading, model inference, feature
extraction, statistics, caching, and plotting live here so the experiment can
also be tested or run from a regular Python process.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = MOTION_GENERATION_DIR.parent
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.audio_motion_model import AudioMotionTransformer  # noqa: E402
from scripts.train_audio_mask_multipart import (  # noqa: E402
    AudioMotionMaskCollator,
    AudioMotionMaskDataset,
    discover_names,
    evenly_spaced_indices,
    load_manifest,
    load_sequences,
    read_split_file,
)
from utils.constants import BODY_JOINTS_ID  # noqa: E402
from utils.multipart_motion import (  # noqa: E402
    PART_ORDER,
    canonicalize_body_root,
    load_motion_dict,
    motion_path_for_name,
)
from utils.msd.msd import (  # noqa: E402
    MSDConfig,
    concatenate_part_embeddings,
    compute_msd_components,
    multipart_tokens_to_embeddings,
)
from utils.msd.multipart_adapter import (  # noqa: E402
    MultipartCodebookSet,
    MultipartCodecAdapter,
)


TAU_NOISE = {
    "old_maya": 0.0010287028271704912,
    "new_maya": 0.00174132885877043,
    "chonglu": 0.00033553200773894787,
}
TAU_ACTIVE_PROVISIONAL = {"chonglu": 0.0008195838308893144}


@dataclass
class AuditConfig:
    project_dir: Path
    data_dir: Path
    motion_token_dir: Path
    audio_feature_dir: Path
    split_file: Path
    mask_checkpoint: Path
    part_checkpoints: dict[str, Path]
    output_dir: Path
    device: str = "cuda:0"
    step: int = 4
    window_stride: Optional[int] = None
    audio_fps: float = 10.0
    motion_fps: float = 20.0
    batch_size: int = 128
    generate_steps: int = 1
    msd_window: int = 8
    max_clips: Optional[int] = 64
    max_windows: Optional[int] = None
    include_fk_speed: bool = True
    include_decoded_errors: bool = False
    seed: int = 42


@dataclass
class AuditContext:
    config: AuditConfig
    device: torch.device
    model: AudioMotionTransformer
    codebooks: MultipartCodebookSet | MultipartCodecAdapter
    decoder: Optional[MultipartCodecAdapter]
    sequences: list[dict[str, Any]]
    dataset: AudioMotionMaskDataset
    collator: AudioMotionMaskCollator
    selected_indices: list[int]
    sequence_by_name: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    load_stats: dict[str, int]
    fk_extractor: Optional["FKSpeedExtractor"]


@dataclass
class AuditTables:
    frames: pd.DataFrame
    tokens: pd.DataFrame
    clips: pd.DataFrame
    failures: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class ClipComplexity:
    phi: np.ndarray
    omega: np.ndarray
    energy: np.ndarray
    part_phi: dict[str, np.ndarray]
    part_omega: dict[str, np.ndarray]
    part_energy: dict[str, np.ndarray]


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the repository root from a notebook or script working directory."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "motion_generation").is_dir() and (
            candidate / "SuSuInterActs"
        ).exists():
            return candidate
    raise RuntimeError("Could not find a repository root containing motion_generation/")


def default_audit_config(project_dir: Optional[Path] = None) -> AuditConfig:
    root = Path(project_dir or find_project_root()).resolve()
    data = root / "SuSuInterActs" / "SuSuInterActs"
    codec_root = root / "checkpoints" / "multipart_rvqvae"
    return AuditConfig(
        project_dir=root,
        data_dir=data,
        motion_token_dir=data / "motion_token_data_multipart_512x4",
        audio_feature_dir=data / "audio_features_hubert_layer9_fps10",
        split_file=data / "split" / "val_file_list.txt",
        mask_checkpoint=root / "checkpoints" / "mask_multipart",
        part_checkpoints={
            part: codec_root
            / f"rvq_{part}_512x4_bs256_cosine"
            / "model"
            / "best.pth"
            for part in PART_ORDER
        },
        output_dir=root
        / "motion_generation"
        / "outputs"
        / "multipart_complexity_error_audit",
    )


def path_status(config: AuditConfig) -> pd.DataFrame:
    rows = [
        ("data_dir", config.data_dir),
        ("motion_token_dir", config.motion_token_dir),
        ("audio_feature_dir", config.audio_feature_dir),
        ("split_file", config.split_file),
        ("mask_checkpoint", config.mask_checkpoint),
    ]
    rows.extend((f"codec_{part}", path) for part, path in config.part_checkpoints.items())
    return pd.DataFrame(
        [{"item": label, "path": str(path), "exists": Path(path).exists()} for label, path in rows]
    )


def validate_paths(config: AuditConfig) -> None:
    missing = path_status(config)
    missing = missing.loc[~missing["exists"]]
    if not missing.empty:
        details = "\n".join(f"  {row.item}: {row.path}" for row in missing.itertuples())
        raise FileNotFoundError(f"Missing audit inputs:\n{details}")


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; using CPU instead of {requested}")
        return torch.device("cpu")
    return torch.device(requested)


def _config_to_jsonable(config: AuditConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["part_checkpoints"] = {
        part: str(path) for part, path in config.part_checkpoints.items()
    }
    return payload


def _load_text_map(data_dir: Path) -> dict[str, str]:
    candidates = [
        data_dir / "text_data" / "motion2text.json",
        data_dir / "text_data" / "train.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return {str(key).replace("\\", "/"): str(value or "") for key, value in payload.items()}
    return {}


def motion_label_kind(text: str) -> str:
    """Match the existing Task 2 default/annotated action-label rule."""
    import re

    tags = re.findall(r"【([^】]+)】", text or "")
    action_tags = [tag.strip() for tag in tags if tag.strip().startswith("动作：")]
    if not action_tags or all(tag == "动作：无动作" for tag in action_tags):
        return "default"
    return "annotated"


def classify_schema(name: str, motion: Mapping[str, np.ndarray]) -> str:
    prefix = name.replace("\\", "/").split("/", 1)[0]
    body = np.asarray(motion["body"])
    root_mean = float(np.linalg.norm(body[:, :3], axis=-1).mean()) if len(body) else 0.0
    if prefix == "fbx_to_json_data_susu_chonglu":
        return "chonglu"
    if prefix == "fbx_to_json_data_susu_retarget_maya":
        if "positions" not in motion and root_mean < 1.0:
            return "old_maya"
        if "positions" in motion and root_mean > 10.0:
            return "new_maya"
    return "outlier"


class FKSpeedExtractor:
    """Compute pelvis-relative mean-joint FK speed and pool it to token rate."""

    def __init__(self, data_dir: Path, device: torch.device, motion_fps: float = 20.0):
        from utils.fk_model import WorldPosFromQuat

        template = MOTION_GENERATION_DIR / "meta" / "template_susu_retarget_63nodes.bvh"
        self.data_dir = Path(data_dir)
        self.motion_dir = self.data_dir / "motion_data"
        self.device = device
        self.motion_fps = float(motion_fps)
        self.fk = WorldPosFromQuat(template_bvh_path=str(template)).to(device).eval()
        self.cache: dict[tuple[str, int, int], np.ndarray] = {}

    @torch.no_grad()
    def token_speed(self, name: str, token_frames: int, unit_length: int) -> np.ndarray:
        key = (name, int(token_frames), int(unit_length))
        if key in self.cache:
            return self.cache[key]

        from scripts.train_audio_fim_causal import body_features_to_quat_motion

        path = motion_path_for_name(self.motion_dir, name)
        motion = load_motion_dict(path)
        body, _schema, _root_mean = canonicalize_body_root(np.asarray(motion["body"]))
        canonical_motion = dict(motion)
        canonical_motion["body"] = body
        quat_motion = body_features_to_quat_motion(
            torch.as_tensor(body, dtype=torch.float32, device=self.device),
            canonical_motion,
            self.device,
            src_fps=self.motion_fps,
            tgt_fps=self.motion_fps,
        )
        quat = torch.as_tensor(quat_motion["quat"], dtype=torch.float32, device=self.device).unsqueeze(0)
        offset = torch.as_tensor(quat_motion["offset"], dtype=torch.float32, device=self.device).unsqueeze(0)
        positions = self.fk(quat, offset)[0][:, BODY_JOINTS_ID]
        relative = positions - positions[:, :1]
        if len(relative) < 2:
            result = np.full(token_frames, np.nan, dtype=np.float32)
        else:
            speed = torch.linalg.norm(relative[1:] - relative[:-1], dim=-1).mean(dim=-1)
            speed = torch.cat([speed[:1], speed], dim=0)
            usable = min(int(speed.shape[0]), int(token_frames * unit_length))
            usable -= usable % unit_length
            pooled = speed[:usable].reshape(-1, unit_length).mean(dim=-1)
            result = np.full(token_frames, np.nan, dtype=np.float32)
            result[: min(token_frames, len(pooled))] = pooled[:token_frames].cpu().numpy()
        self.cache[key] = result
        return result


def load_audit_context(config: AuditConfig) -> AuditContext:
    """Load the frozen infiller, multipart codebooks/codecs, and eval windows."""
    validate_paths(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    model = AudioMotionTransformer.from_pretrained(
        str(config.mask_checkpoint),
        local_files_only=True,
    ).to(device).eval()

    part_order = tuple(str(part) for part in getattr(model.config, "part_order", PART_ORDER))
    if part_order != tuple(PART_ORDER):
        raise ValueError(f"Expected part order {PART_ORDER}, got {part_order}")

    decoder: Optional[MultipartCodecAdapter]
    if config.include_decoded_errors:
        decoder = MultipartCodecAdapter.from_checkpoints(config.part_checkpoints, device, part_order)
        codebooks: MultipartCodebookSet | MultipartCodecAdapter = decoder
    else:
        decoder = None
        codebooks = MultipartCodebookSet.from_checkpoints(config.part_checkpoints, device, part_order)

    manifest = load_manifest(config.motion_token_dir) or {}
    expected = {
        "codebook_size": codebooks.codebook_size,
        "num_quantizers": codebooks.num_quantizers,
        "tokens_per_frame": codebooks.tokens_per_frame,
    }
    for key, value in expected.items():
        if key in manifest and int(manifest[key]) != int(value):
            raise ValueError(f"Token manifest {key}={manifest[key]} but codecs use {value}")
    if int(model.config.codebook_size) != codebooks.codebook_size:
        raise ValueError("Mask model and multipart codecs use different codebook sizes")
    if int(model.config.num_tokens_per_frame) != codebooks.tokens_per_frame:
        raise ValueError("Mask model and multipart codecs use different token-frame widths")
    if int(model.config.num_frames) != config.step + 1:
        raise ValueError(
            f"Mask model expects {model.config.num_frames} frames, but step={config.step} gives {config.step + 1}"
        )

    names = discover_names(
        config.motion_token_dir,
        config.audio_feature_dir,
        read_split_file(config.split_file),
    )
    sequences, load_stats = load_sequences(
        names,
        config.motion_token_dir,
        config.audio_feature_dir,
        codebook_size=codebooks.codebook_size,
        num_tokens_per_frame=codebooks.tokens_per_frame,
        audio_fps=config.audio_fps,
        source_motion_fps_fallback=config.motion_fps,
        motion_token_fps_override=None,
        motion_token_unit_length_override=None,
        max_sequences=config.max_clips,
    )
    dataset = AudioMotionMaskDataset(
        sequences,
        step=config.step,
        window_stride=config.window_stride,
        seed=config.seed,
    )
    collator = AudioMotionMaskCollator(model.config)
    count = len(dataset) if config.max_windows is None else min(config.max_windows, len(dataset))
    selected_indices = evenly_spaced_indices(len(dataset), count)
    fk_extractor = (
        FKSpeedExtractor(config.data_dir, device, config.motion_fps)
        if config.include_fk_speed
        else None
    )
    return AuditContext(
        config=config,
        device=device,
        model=model,
        codebooks=codebooks,
        decoder=decoder,
        sequences=sequences,
        dataset=dataset,
        collator=collator,
        selected_indices=selected_indices,
        sequence_by_name={str(item["name"]): item for item in sequences},
        manifest=manifest,
        load_stats=load_stats,
        fk_extractor=fk_extractor,
    )


def context_summary(context: AuditContext) -> pd.DataFrame:
    config = context.config
    rows = [
        ("device", str(context.device)),
        ("clips", len(context.sequences)),
        ("all_windows", len(context.dataset)),
        ("selected_windows", len(context.selected_indices)),
        ("batch_size", config.batch_size),
        ("step", config.step),
        ("window_stride", config.window_stride or config.step),
        ("tokens_per_frame", context.codebooks.tokens_per_frame),
        ("codebook_size", context.codebooks.codebook_size),
        ("quantizers_per_part", context.codebooks.num_quantizers),
        ("msd_window", config.msd_window),
        ("fk_speed", config.include_fk_speed),
        ("decoded_errors", config.include_decoded_errors),
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


@torch.no_grad()
def compute_clip_complexity(
    tokens: torch.Tensor,
    codebooks: Mapping[str, torch.Tensor],
    part_order: Sequence[str] = PART_ORDER,
    window: int = 8,
) -> ClipComplexity:
    embeddings = multipart_tokens_to_embeddings(tokens, codebooks, part_order)
    combined = concatenate_part_embeddings(embeddings, part_order)
    combined_components = compute_msd_components(combined, MSDConfig(W=window))
    part_components = {
        part: compute_msd_components(embeddings[part], MSDConfig(W=window))
        for part in part_order
    }
    return ClipComplexity(
        phi=combined_components.phi.cpu().numpy().astype(np.float32),
        omega=combined_components.omega.cpu().numpy().astype(np.float32),
        energy=combined_components.energy.cpu().numpy().astype(np.float32),
        part_phi={part: value.phi.cpu().numpy().astype(np.float32) for part, value in part_components.items()},
        part_omega={part: value.omega.cpu().numpy().astype(np.float32) for part, value in part_components.items()},
        part_energy={part: value.energy.cpu().numpy().astype(np.float32) for part, value in part_components.items()},
    )


def _sixd_geodesic_degrees(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.size == 0:
        return np.empty((gt.shape[0], 0), dtype=np.float32)
    gt_t = torch.as_tensor(gt, dtype=torch.float32).reshape(gt.shape[0], -1, 6)
    pred_t = torch.as_tensor(pred, dtype=torch.float32).reshape(pred.shape[0], -1, 6)

    def to_matrix(value: torch.Tensor) -> torch.Tensor:
        first = F.normalize(value[..., :3], dim=-1)
        second_raw = value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first
        second = F.normalize(second_raw, dim=-1)
        third = torch.cross(first, second, dim=-1)
        return torch.stack([first, second, third], dim=-1)

    relative = to_matrix(gt_t).transpose(-1, -2) @ to_matrix(pred_t)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)).numpy().astype(np.float32)


def _pool_raw_curve(values: np.ndarray, token_frames: int, unit_length: int) -> np.ndarray:
    result = np.full(token_frames, np.nan, dtype=np.float32)
    for token_idx in range(token_frames):
        start = token_idx * unit_length
        end = min(start + unit_length, len(values))
        if end > start:
            result[token_idx] = float(np.nanmean(values[start:end]))
    return result


def decoded_error_curves(
    gt_parts: Mapping[str, torch.Tensor],
    pred_parts: Mapping[str, torch.Tensor],
    token_frames: int,
    unit_length: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return one decoded error curve per part and token frame."""
    curves: dict[str, dict[str, np.ndarray]] = {}
    combined_gt: list[np.ndarray] = []
    combined_pred: list[np.ndarray] = []
    combined_rot_gt: list[np.ndarray] = []
    combined_rot_pred: list[np.ndarray] = []

    for part in PART_ORDER:
        gt = gt_parts[part].detach().cpu().numpy().astype(np.float32)
        pred = pred_parts[part].detach().cpu().numpy().astype(np.float32)
        frames = min(len(gt), len(pred))
        gt, pred = gt[:frames], pred[:frames]
        diff = pred - gt
        velocity_diff = np.diff(pred, axis=0, prepend=pred[:1]) - np.diff(gt, axis=0, prepend=gt[:1])
        accel_diff = np.diff(velocity_diff, axis=0, prepend=velocity_diff[:1])
        rot_gt = gt[:, 3:] if part == "lower" else gt
        rot_pred = pred[:, 3:] if part == "lower" else pred
        geodesic = _sixd_geodesic_degrees(rot_gt, rot_pred).mean(axis=-1)
        curves[part] = {
            "mae": _pool_raw_curve(np.abs(diff).mean(axis=-1), token_frames, unit_length),
            "rmse": _pool_raw_curve(np.sqrt(np.square(diff).mean(axis=-1)), token_frames, unit_length),
            "velocity_rmse": _pool_raw_curve(
                np.sqrt(np.square(velocity_diff).mean(axis=-1)), token_frames, unit_length
            ),
            "acceleration_rmse": _pool_raw_curve(
                np.sqrt(np.square(accel_diff).mean(axis=-1)), token_frames, unit_length
            ),
            "geodesic_deg": _pool_raw_curve(geodesic, token_frames, unit_length),
        }
        combined_gt.append(gt)
        combined_pred.append(pred)
        combined_rot_gt.append(rot_gt)
        combined_rot_pred.append(rot_pred)

    frames = min(value.shape[0] for value in combined_gt)
    gt_all = np.concatenate([value[:frames] for value in combined_gt], axis=-1)
    pred_all = np.concatenate([value[:frames] for value in combined_pred], axis=-1)
    diff = pred_all - gt_all
    velocity_diff = np.diff(pred_all, axis=0, prepend=pred_all[:1]) - np.diff(
        gt_all, axis=0, prepend=gt_all[:1]
    )
    accel_diff = np.diff(velocity_diff, axis=0, prepend=velocity_diff[:1])
    geo = _sixd_geodesic_degrees(
        np.concatenate([value[:frames] for value in combined_rot_gt], axis=-1),
        np.concatenate([value[:frames] for value in combined_rot_pred], axis=-1),
    ).mean(axis=-1)
    curves["combined"] = {
        "mae": _pool_raw_curve(np.abs(diff).mean(axis=-1), token_frames, unit_length),
        "rmse": _pool_raw_curve(np.sqrt(np.square(diff).mean(axis=-1)), token_frames, unit_length),
        "velocity_rmse": _pool_raw_curve(
            np.sqrt(np.square(velocity_diff).mean(axis=-1)), token_frames, unit_length
        ),
        "acceleration_rmse": _pool_raw_curve(
            np.sqrt(np.square(accel_diff).mean(axis=-1)), token_frames, unit_length
        ),
        "geodesic_deg": _pool_raw_curve(geo, token_frames, unit_length),
    }
    return curves


def _raw_ids(global_ids: np.ndarray, codebook_size: int, tokens_per_frame: int) -> np.ndarray:
    slots = np.arange(global_ids.shape[-1], dtype=np.int64) % tokens_per_frame
    return global_ids - slots[None, :] * codebook_size


def _safe_value(values: np.ndarray, index: int) -> float:
    if index < 0 or index >= len(values):
        return float("nan")
    return float(values[index])


def _clip_metadata(
    name: str,
    config: AuditConfig,
    text_map: Mapping[str, str],
) -> tuple[str, str]:
    path = motion_path_for_name(config.data_dir / "motion_data", name)
    if not path.exists():
        return "missing", motion_label_kind(text_map.get(name, ""))
    motion = load_motion_dict(path)
    return classify_schema(name, motion), motion_label_kind(text_map.get(name, ""))


@torch.no_grad()
def run_audit(context: AuditContext) -> AuditTables:
    """Run inference and return unstratified per-frame/per-token audit tables."""
    config = context.config
    model = context.model
    ntpf = context.codebooks.tokens_per_frame
    quantizers = context.codebooks.num_quantizers
    codebook_size = context.codebooks.codebook_size
    unit_length = context.codebooks.unit_length
    complexity_cache: dict[str, ClipComplexity] = {}
    speed_cache: dict[str, np.ndarray] = {}
    metadata_cache: dict[str, tuple[str, str]] = {}
    text_map = _load_text_map(config.data_dir)
    frame_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    progress = tqdm(total=len(context.selected_indices), desc="complexity-error audit")
    for batch_start in range(0, len(context.selected_indices), config.batch_size):
        indices = context.selected_indices[batch_start : batch_start + config.batch_size]
        examples = [context.dataset[index] for index in indices]
        batch = context.collator(examples)
        input_ids = batch["input_ids"].to(context.device)
        labels = batch["labels"].to(context.device)
        audio = batch["audio_features"].to(context.device)
        logits = model(input_ids=input_ids, audio_features=audio)
        log_probs = logits.float().log_softmax(dim=-1)
        safe_labels = labels.masked_fill(labels.eq(-100), 0)
        nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        teacher = logits.argmax(dim=-1)
        generated = model.generate_sbs(input_ids, audio, generate_steps=config.generate_steps)

        labels_np = labels.cpu().numpy()
        nll_np = nll.cpu().numpy()
        teacher_np = teacher.cpu().numpy()
        generated_np = generated.cpu().numpy()

        for local_idx, (dataset_idx, example) in enumerate(zip(indices, examples)):
            name = str(example.name)
            item = context.sequence_by_name[name]
            if name not in complexity_cache:
                full_tokens = torch.as_tensor(
                    item["motion_tokens"], dtype=torch.long, device=context.device
                )
                complexity_cache[name] = compute_clip_complexity(
                    full_tokens,
                    context.codebooks.codebooks,
                    context.codebooks.part_order,
                    config.msd_window,
                )
                metadata_cache[name] = _clip_metadata(name, config, text_map)
                if context.fk_extractor is not None:
                    try:
                        speed_cache[name] = context.fk_extractor.token_speed(
                            name, len(item["motion_tokens"]), unit_length
                        )
                    except Exception as exc:  # noqa: BLE001
                        speed_cache[name] = np.full(len(item["motion_tokens"]), np.nan)
                        failure_rows.append(
                            {"name": name, "stage": "fk_speed", "error": f"{type(exc).__name__}: {exc}"}
                        )

            complexity = complexity_cache[name]
            schema, label_kind = metadata_cache[name]
            speed = speed_cache.get(name, np.full(len(item["motion_tokens"]), np.nan))
            gt_window = np.asarray(example.motion_tokens, dtype=np.int64)
            pred_global = generated_np[local_idx].reshape(config.step + 1, ntpf)
            pred_window = _raw_ids(pred_global, codebook_size, ntpf)
            invalid_fraction = float(((pred_window < 0) | (pred_window >= codebook_size)).mean())
            pred_window = np.clip(pred_window, 0, codebook_size - 1)

            decoded_curves: dict[str, dict[str, np.ndarray]] = {}
            if context.decoder is not None:
                gt_parts = context.decoder.decode_parts(
                    torch.as_tensor(gt_window, dtype=torch.long, device=context.device)
                )
                pred_parts = context.decoder.decode_parts(
                    torch.as_tensor(pred_window, dtype=torch.long, device=context.device)
                )
                decoded_curves = decoded_error_curves(
                    gt_parts, pred_parts, config.step + 1, unit_length
                )

            for relative_frame in range(1, config.step):
                absolute_frame = int(example.left_idx + relative_frame)
                start = relative_frame * ntpf
                end = start + ntpf
                target_global = labels_np[local_idx, start:end]
                teacher_global = teacher_np[local_idx, start:end]
                generated_global = generated_np[local_idx, start:end]
                valid = target_global != -100
                teacher_match = teacher_global == target_global
                generated_match = generated_global == target_global
                frame_row: dict[str, Any] = {
                    "dataset_idx": int(dataset_idx),
                    "name": name,
                    "schema": schema,
                    "label_kind": label_kind,
                    "left_idx": int(example.left_idx),
                    "right_idx": int(example.right_idx),
                    "relative_frame": relative_frame,
                    "token_frame": absolute_frame,
                    "time_s": absolute_frame / float(item["motion_token_fps"]),
                    "fk_speed": _safe_value(speed, absolute_frame),
                    "omega": _safe_value(complexity.omega, absolute_frame),
                    "energy": _safe_value(complexity.energy, absolute_frame),
                    "teacher_nll": float(nll_np[local_idx, start:end][valid].mean()),
                    "teacher_acc": float(teacher_match[valid].mean()),
                    "generated_acc": float(generated_match[valid].mean()),
                    "generated_exact_frame": float(generated_match[valid].all()),
                    "invalid_token_fraction": invalid_fraction,
                }
                for band in range(config.msd_window):
                    frame_row[f"phi_{band}"] = float(complexity.phi[absolute_frame, band])
                for part in PART_ORDER:
                    frame_row[f"omega_{part}"] = _safe_value(
                        complexity.part_omega[part], absolute_frame
                    )
                    frame_row[f"energy_{part}"] = _safe_value(
                        complexity.part_energy[part], absolute_frame
                    )
                    for band in range(config.msd_window):
                        frame_row[f"phi_{part}_{band}"] = float(
                            complexity.part_phi[part][absolute_frame, band]
                        )
                for decoded_part, metrics in decoded_curves.items():
                    for metric, curve in metrics.items():
                        frame_row[f"decoded_{decoded_part}_{metric}"] = _safe_value(
                            curve, relative_frame
                        )
                frame_rows.append(frame_row)

                for slot in range(ntpf):
                    part_idx = slot // quantizers
                    quantizer = slot % quantizers
                    part = str(context.codebooks.part_order[part_idx])
                    position = start + slot
                    token_rows.append(
                        {
                            "dataset_idx": int(dataset_idx),
                            "name": name,
                            "schema": schema,
                            "label_kind": label_kind,
                            "token_frame": absolute_frame,
                            "relative_frame": relative_frame,
                            "slot": slot,
                            "part": part,
                            "quantizer": quantizer,
                            "target_id": int(target_global[slot] - slot * codebook_size),
                            "teacher_id": int(teacher_global[slot] - slot * codebook_size),
                            "generated_id": int(generated_global[slot] - slot * codebook_size),
                            "teacher_correct": bool(teacher_match[slot]),
                            "generated_correct": bool(generated_match[slot]),
                            "teacher_nll": float(nll_np[local_idx, position]),
                            "target_probability": float(math.exp(-nll_np[local_idx, position])),
                            "fk_speed": _safe_value(speed, absolute_frame),
                            "omega": _safe_value(complexity.omega, absolute_frame),
                            "energy": _safe_value(complexity.energy, absolute_frame),
                            "part_omega": _safe_value(complexity.part_omega[part], absolute_frame),
                            "part_energy": _safe_value(complexity.part_energy[part], absolute_frame),
                        }
                    )
            progress.update(1)
    progress.close()

    frames = pd.DataFrame(frame_rows)
    tokens = pd.DataFrame(token_rows)
    failures = pd.DataFrame(failure_rows, columns=["name", "stage", "error"])
    clips = build_clip_table(frames, tokens)
    metadata = {
        "config": _config_to_jsonable(config),
        "load_stats": context.load_stats,
        "num_frames": int(len(frames)),
        "num_tokens": int(len(tokens)),
        "num_clips": int(frames["name"].nunique()) if not frames.empty else 0,
        "tau_noise": TAU_NOISE,
        "tau_active_provisional": TAU_ACTIVE_PROVISIONAL,
    }
    return AuditTables(frames=frames, tokens=tokens, clips=clips, failures=failures, metadata=metadata)


def build_clip_table(frames: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame()
    frame_metrics = [
        column
        for column in frames.columns
        if (
            column in {"fk_speed", "omega", "energy", "teacher_nll", "teacher_acc", "generated_acc"}
            or column.startswith("omega_")
            or column.startswith("energy_")
            or column.startswith("decoded_")
        )
        and pd.api.types.is_numeric_dtype(frames[column])
    ]
    clips = frames.groupby(["name", "schema", "label_kind"], as_index=False)[frame_metrics].mean()
    if not tokens.empty:
        part = (
            tokens.groupby(["name", "part"], as_index=False)
            .agg(
                part_teacher_nll=("teacher_nll", "mean"),
                part_teacher_acc=("teacher_correct", "mean"),
                part_generated_acc=("generated_correct", "mean"),
            )
            .pivot(index="name", columns="part")
        )
        part.columns = [f"{metric}_{part_name}" for metric, part_name in part.columns]
        clips = clips.merge(part.reset_index(), on="name", how="left")
    return clips


def _quantile_codes(values: pd.Series, bins: int = 4, prefix: str = "Q") -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = values.dropna()
    if valid.empty:
        return result
    percentile = valid.rank(method="average", pct=True)
    codes = np.ceil(percentile * bins).clip(1, bins).astype(int)
    result.loc[valid.index] = [f"{prefix}{code}" for code in codes]
    return result


def add_empirical_strata(tables: AuditTables) -> AuditTables:
    """Add schema-aware speed regimes and empirical MSD quartiles."""
    frames = tables.frames.copy()
    tokens = tables.tokens.copy()
    if frames.empty:
        return tables

    frame_strata = [
        "tau_noise",
        "speed_regime",
        "active_regime",
        "speed_quartile",
        "omega_quartile",
    ]
    token_strata = [*frame_strata, "part_omega_quartile"]
    frames = frames.drop(columns=[column for column in frame_strata if column in frames], errors="ignore")
    tokens = tokens.drop(columns=[column for column in token_strata if column in tokens], errors="ignore")

    frames["tau_noise"] = frames["schema"].map(TAU_NOISE)
    frames["speed_regime"] = np.where(
        frames["fk_speed"].isna() | frames["tau_noise"].isna(),
        "unknown",
        np.where(frames["fk_speed"] <= frames["tau_noise"], "still_noise", "moving"),
    )
    frames["active_regime"] = "unavailable"
    chonglu = frames["schema"].eq("chonglu") & frames["fk_speed"].notna()
    frames.loc[chonglu, "active_regime"] = np.where(
        frames.loc[chonglu, "fk_speed"] >= TAU_ACTIVE_PROVISIONAL["chonglu"],
        "active_provisional",
        np.where(
            frames.loc[chonglu, "fk_speed"] <= TAU_NOISE["chonglu"],
            "still_noise",
            "transition_provisional",
        ),
    )
    frames["speed_quartile"] = pd.NA
    frames["omega_quartile"] = pd.NA
    moving = frames["speed_regime"].eq("moving")
    for _schema, indices in frames.loc[moving].groupby("schema").groups.items():
        frames.loc[indices, "speed_quartile"] = _quantile_codes(frames.loc[indices, "fk_speed"])
        frames.loc[indices, "omega_quartile"] = _quantile_codes(frames.loc[indices, "omega"])
    frames.loc[frames["speed_regime"].eq("still_noise"), ["speed_quartile", "omega_quartile"]] = "idle"

    keys = frames[
        [
            "dataset_idx",
            "token_frame",
            "tau_noise",
            "speed_regime",
            "active_regime",
            "speed_quartile",
            "omega_quartile",
        ]
    ]
    tokens = tokens.merge(keys, on=["dataset_idx", "token_frame"], how="left")
    tokens["part_omega_quartile"] = pd.NA
    moving_tokens = tokens["speed_regime"].eq("moving")
    for (_schema, _part), indices in tokens.loc[moving_tokens].groupby(["schema", "part"]).groups.items():
        tokens.loc[indices, "part_omega_quartile"] = _quantile_codes(
            tokens.loc[indices, "part_omega"]
        )
    tokens.loc[tokens["speed_regime"].eq("still_noise"), "part_omega_quartile"] = "idle"
    clips = build_clip_table(frames, tokens)
    return AuditTables(frames, tokens, clips, tables.failures.copy(), dict(tables.metadata))


def save_audit_tables(tables: AuditTables, output_dir: Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables.frames.to_csv(output / "frames.csv.gz", index=False, compression="gzip")
    tables.tokens.to_csv(output / "tokens.csv.gz", index=False, compression="gzip")
    tables.clips.to_csv(output / "clips.csv", index=False)
    tables.failures.to_csv(output / "failures.csv", index=False)
    with open(output / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(tables.metadata, handle, indent=2, ensure_ascii=False)


def load_audit_tables(output_dir: Path) -> AuditTables:
    output = Path(output_dir)
    with open(output / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return AuditTables(
        frames=pd.read_csv(output / "frames.csv.gz"),
        tokens=pd.read_csv(output / "tokens.csv.gz"),
        clips=pd.read_csv(output / "clips.csv"),
        failures=pd.read_csv(output / "failures.csv"),
        metadata=metadata,
    )


def run_or_load_audit(
    config: AuditConfig,
    *,
    run_extraction: bool,
    overwrite: bool = False,
) -> tuple[AuditTables, Optional[AuditContext]]:
    cache = config.output_dir / "metadata.json"
    if cache.exists() and not overwrite:
        tables = load_audit_tables(config.output_dir)
        saved = tables.metadata.get("config", {})
        current = _config_to_jsonable(config)
        critical = [
            "motion_token_dir",
            "audio_feature_dir",
            "split_file",
            "mask_checkpoint",
            "step",
            "window_stride",
            "audio_fps",
            "motion_fps",
            "generate_steps",
            "msd_window",
            "max_clips",
            "max_windows",
            "include_fk_speed",
            "include_decoded_errors",
            "seed",
        ]
        mismatches = {
            key: (saved.get(key), current.get(key))
            for key in critical
            if saved.get(key) != current.get(key)
        }
        if saved.get("part_checkpoints") != current.get("part_checkpoints"):
            mismatches["part_checkpoints"] = (
                saved.get("part_checkpoints"),
                current.get("part_checkpoints"),
            )
        if mismatches:
            details = "\n".join(
                f"  {key}: cached={old!r}, requested={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(
                "Cached audit configuration does not match this run. "
                "Use a new output_dir or set overwrite=True.\n" + details
            )
        return add_empirical_strata(tables), None
    if not run_extraction:
        raise FileNotFoundError(
            f"No cached audit at {config.output_dir}. Set RUN_EXTRACTION=True for the first run."
        )
    context = load_audit_context(config)
    tables = add_empirical_strata(run_audit(context))
    save_audit_tables(tables, config.output_dir)
    return tables, context


def distribution_summary(frames: pd.DataFrame) -> pd.DataFrame:
    variables = ["fk_speed", "omega", "energy"] + [
        f"{prefix}_{part}" for prefix in ("omega", "energy") for part in PART_ORDER
    ]
    rows: list[dict[str, Any]] = []
    for schema, group in frames.groupby("schema", dropna=False):
        for variable in variables:
            values = pd.to_numeric(group[variable], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "schema": schema,
                    "variable": variable,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "p05": values.quantile(0.05),
                    "p25": values.quantile(0.25),
                    "p50": values.quantile(0.50),
                    "p75": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def regime_summary(frames: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frames.groupby(["schema", "speed_regime"], dropna=False)
        .size()
        .rename("frames")
        .reset_index()
    )
    counts["fraction"] = counts["frames"] / counts.groupby("schema")["frames"].transform("sum")
    return counts


def error_by_complexity(tokens: pd.DataFrame) -> pd.DataFrame:
    order = ["idle", "Q1", "Q2", "Q3", "Q4"]
    result = (
        tokens.groupby(["part", "quantizer", "part_omega_quartile"], dropna=False)
        .agg(
            tokens=("teacher_nll", "size"),
            teacher_nll=("teacher_nll", "mean"),
            teacher_acc=("teacher_correct", "mean"),
            generated_acc=("generated_correct", "mean"),
        )
        .reset_index()
    )
    result["part_omega_quartile"] = pd.Categorical(
        result["part_omega_quartile"], categories=order, ordered=True
    )
    return result.sort_values(["part", "quantizer", "part_omega_quartile"])


def decoded_error_by_complexity(frames: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in frames.columns
        if column.startswith("decoded_")
    ]
    if not metrics:
        return pd.DataFrame()
    long = frames.melt(
        id_vars=["name", "schema", "omega_quartile", "speed_regime"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    return (
        long.groupby(["metric", "omega_quartile"], dropna=False)
        .agg(frames=("value", "size"), mean=("value", "mean"), median=("value", "median"))
        .reset_index()
    )


def _spearman(x: pd.Series, y: pd.Series) -> float:
    values = pd.concat([x, y], axis=1).dropna()
    if len(values) < 3 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(values.iloc[:, 0].corr(values.iloc[:, 1], method="spearman"))


def cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    cluster: str = "name",
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    clean = frame[[cluster, x, y]].dropna()
    observed = _spearman(clean[x], clean[y])
    clusters = clean[cluster].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return {"rho": observed, "ci_low": np.nan, "ci_high": np.nan, "clusters": len(clusters)}
    grouped = {key: group for key, group in clean.groupby(cluster)}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        draw = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        estimates.append(_spearman(draw[x], draw[y]))
    finite = np.asarray(estimates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "rho": observed,
        "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
        "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
        "clusters": int(len(clusters)),
    }


def correlation_summary(tokens: pd.DataFrame, iterations: int = 500, seed: int = 42) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("part_omega", "teacher_nll"),
        ("part_energy", "teacher_nll"),
        ("fk_speed", "teacher_nll"),
        ("part_omega", "generated_correct"),
    ]
    for part, group in tokens.groupby("part"):
        for pair_idx, (x, y) in enumerate(pairs):
            stats = cluster_bootstrap_spearman(
                group,
                x,
                y,
                iterations=iterations,
                seed=seed + pair_idx,
            )
            rows.append({"part": part, "x": x, "y": y, **stats})
    return pd.DataFrame(rows)


def cliff_delta(first: Iterable[float], second: Iterable[float], max_values: int = 20000) -> float:
    """Cliff's delta, with deterministic thinning for very large frame sets."""
    a = np.asarray(list(first), dtype=np.float64)
    b = np.asarray(list(second), dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) > max_values:
        a = a[np.linspace(0, len(a) - 1, max_values, dtype=int)]
    if len(b) > max_values:
        b = b[np.linspace(0, len(b) - 1, max_values, dtype=int)]
    greater = sum(np.count_nonzero(value > b) for value in a)
    lower = sum(np.count_nonzero(value < b) for value in a)
    return float((greater - lower) / (len(a) * len(b)))


def quartile_effect_summary(tokens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (part, quantizer), group in tokens.groupby(["part", "quantizer"]):
        q1 = group.loc[group["part_omega_quartile"].eq("Q1"), "teacher_nll"]
        q4 = group.loc[group["part_omega_quartile"].eq("Q4"), "teacher_nll"]
        rows.append(
            {
                "part": part,
                "quantizer": quantizer,
                "q1_n": len(q1),
                "q4_n": len(q4),
                "q1_nll": q1.mean(),
                "q4_nll": q4.mean(),
                "q4_minus_q1": q4.mean() - q1.mean(),
                "cliff_delta_q4_vs_q1": cliff_delta(q4, q1),
            }
        )
    return pd.DataFrame(rows)


def kruskal_summary(tokens: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import kruskal
    except ImportError:
        return pd.DataFrame([{"status": "scipy unavailable"}])
    rows = []
    for part, group in tokens.groupby("part"):
        samples = [
            group.loc[group["part_omega_quartile"].eq(label), "teacher_nll"].dropna().to_numpy()
            for label in ("Q1", "Q2", "Q3", "Q4")
        ]
        if any(len(sample) == 0 for sample in samples):
            continue
        statistic, pvalue = kruskal(*samples)
        rows.append({"part": part, "H": statistic, "pvalue": pvalue, "frames": sum(map(len, samples))})
    return pd.DataFrame(rows)


def _sample_rows(frame: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(max_points, random_state=seed)


def plot_speed_distributions(frames: pd.DataFrame) -> plt.Figure:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"old_maya": "#2978A0", "new_maya": "#D1495B", "chonglu": "#2E8B57", "outlier": "#777777"}
    for schema, group in frames.groupby("schema"):
        values = np.sort(group["fk_speed"].dropna().to_numpy())
        if len(values) == 0:
            continue
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=schema, color=colors.get(schema))
        axes[1].hist(values, bins=100, density=True, histtype="step", linewidth=1.7, label=schema, color=colors.get(schema))
        tau = TAU_NOISE.get(schema)
        if tau is not None:
            axes[1].axvline(tau, color=colors.get(schema), linestyle="--", alpha=0.8)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set(title="FK Speed ECDF", xlabel="Mean joint speed (m/frame)", ylabel="Cumulative fraction")
    axes[1].set(title="FK Speed Density", xlabel="Mean joint speed (m/frame)", ylabel="Density")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_complexity_distributions(frames: pd.DataFrame) -> plt.Figure:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"combined": "#111111", "upper": "#2978A0", "lower": "#D1495B", "feet": "#6A4C93", "hands": "#2E8B57"}
    for part in ("combined", *PART_ORDER):
        omega_col = "omega" if part == "combined" else f"omega_{part}"
        energy_col = "energy" if part == "combined" else f"energy_{part}"
        axes[0].hist(frames[omega_col].dropna(), bins=80, density=True, histtype="step", linewidth=1.7, label=part, color=colors[part])
        positive = frames.loc[frames[energy_col] > 0, energy_col].dropna()
        axes[1].hist(positive, bins=np.geomspace(positive.min(), positive.max(), 80) if len(positive) else 20, density=True, histtype="step", linewidth=1.7, label=part, color=colors[part])
    axes[1].set_xscale("log")
    axes[0].set(title="MSD Spectral Spread", xlabel="Omega", ylabel="Density")
    axes[1].set(title="MSD Spectral Energy", xlabel="Energy", ylabel="Density")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_speed_complexity_joint(frames: pd.DataFrame, max_points: int = 100000) -> plt.Figure:
    import matplotlib.pyplot as plt

    data = _sample_rows(frames.dropna(subset=["fk_speed", "omega", "energy"]), max_points)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    first = axes[0].hexbin(data["fk_speed"], data["omega"], gridsize=55, xscale="log", mincnt=1, cmap="viridis")
    second = axes[1].hexbin(data["energy"], data["omega"], gridsize=55, xscale="log", mincnt=1, cmap="magma")
    axes[0].set(title="Physical Speed vs MSD", xlabel="FK speed (m/frame)", ylabel="Omega")
    axes[1].set(title="Spectral Energy vs MSD", xlabel="Spectral energy", ylabel="Omega")
    fig.colorbar(first, ax=axes[0], label="Frames")
    fig.colorbar(second, ax=axes[1], label="Frames")
    fig.tight_layout()
    return fig


def plot_error_quartiles(tokens: pd.DataFrame, metric: str = "teacher_nll") -> plt.Figure:
    import matplotlib.pyplot as plt

    summary = (
        tokens.dropna(subset=["part_omega_quartile"])
        .groupby(["part", "part_omega_quartile"], observed=True)[metric]
        .mean()
        .unstack(0)
        .reindex(["idle", "Q1", "Q2", "Q3", "Q4"])
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    summary.plot(marker="o", ax=ax)
    ax.set(title=f"{metric} by Part-specific MSD Quartile", xlabel="Complexity stratum", ylabel=metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_part_omega_correlation(frames: pd.DataFrame) -> plt.Figure:
    import matplotlib.pyplot as plt

    columns = [f"omega_{part}" for part in PART_ORDER]
    corr = frames[columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [part.replace("omega_", "") for part in columns]
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center")
    ax.set_title("Per-part Omega Spearman Correlation")
    fig.colorbar(image, ax=ax, label="rho")
    fig.tight_layout()
    return fig


def select_example_clips(frames: pd.DataFrame, count_per_tier: int = 2) -> pd.DataFrame:
    clip_scores = (
        frames.groupby(["name", "schema"], as_index=False)
        .agg(omega=("omega", "mean"), energy=("energy", "mean"), generated_acc=("generated_acc", "mean"))
        .sort_values("omega")
    )
    if clip_scores.empty:
        return clip_scores
    selected = []
    for tier, quantile in (("low", 0.05), ("median", 0.50), ("high", 0.95)):
        target = clip_scores["omega"].quantile(quantile)
        rows = clip_scores.assign(distance=(clip_scores["omega"] - target).abs()).nsmallest(count_per_tier, "distance")
        rows = rows.assign(tier=tier)
        selected.append(rows.drop(columns="distance"))
    return pd.concat(selected, ignore_index=True)


def plot_clip_timeline(frames: pd.DataFrame, name: str) -> plt.Figure:
    import matplotlib.pyplot as plt

    clip = frames.loc[frames["name"].eq(name)].sort_values("token_frame")
    if clip.empty:
        raise KeyError(f"Clip not present in audit tables: {name}")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(clip["time_s"], clip["fk_speed"], color="#111111")
    tau = TAU_NOISE.get(str(clip["schema"].iloc[0]))
    if tau is not None:
        axes[0].axhline(tau, linestyle="--", color="#D1495B", label="tau_noise")
        axes[0].legend()
    for part in PART_ORDER:
        axes[1].plot(clip["time_s"], clip[f"omega_{part}"], label=part)
    axes[2].plot(clip["time_s"], clip["teacher_nll"], label="NLL", color="#6A4C93")
    axes[2].plot(clip["time_s"], 1.0 - clip["generated_acc"], label="1 - generated accuracy", color="#2E8B57")
    axes[0].set_ylabel("FK speed")
    axes[1].set_ylabel("Omega")
    axes[2].set_ylabel("Error")
    axes[2].set_xlabel("Time (s)")
    axes[1].legend(ncol=4)
    axes[2].legend()
    fig.suptitle(name)
    fig.tight_layout()
    return fig

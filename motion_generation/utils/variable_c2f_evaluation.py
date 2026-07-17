"""Evaluation helpers for variable-gap multipart motion infillers."""

from __future__ import annotations

import importlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm.auto import tqdm

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer
from utils.multipart_motion import (
    PART_ORDER,
    canonicalize_body_root,
    merge_parts_to_legacy_motion,
    motion_path_for_name,
)

if TYPE_CHECKING:
    from scripts.export_multipart_motion_tokens import LoadedPartCodec
else:
    LoadedPartCodec = Any


@dataclass(frozen=True)
class InfillModelSpec:
    name: str
    checkpoint: Path
    decoder: str
    allowed_gaps: tuple[int, ...]
    generate_steps: int = 1
    audio_input_mode: str = "correct"
    audio_ablation_seed: int = 42
    disable_audio_adapters: bool = False

    def supports_gap(self, gap: int) -> bool:
        return int(gap) in self.allowed_gaps


@dataclass(frozen=True)
class EvalWindowRecord:
    sequence_idx: int
    name: str
    left_idx: int
    gap_frames: int
    example: "VariableGapMaskExample"


@dataclass
class VariableGapMaskExample:
    name: str
    left_idx: int
    right_idx: int
    gap_frames: int
    motion_tokens: list[list[int]]
    audio_features: torch.Tensor


class VariableGapC2FCollator:
    """Pad variable windows by complete multipart token frames."""

    def __init__(self, config: AudioMotionConfig) -> None:
        self.mask_token_id = int(getattr(config, "mask_token_id", config.vocab_size - 1))
        self.tokens_per_frame = int(config.num_tokens_per_frame)
        self.codebook_size = int(config.codebook_size)

    def __call__(self, examples: Sequence[VariableGapMaskExample]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_frames = max(len(example.motion_tokens) for example in examples)
        batch_input, batch_gt = [], []
        batch_attention, batch_middle, batch_audio = [], [], []
        for example in examples:
            frames = len(example.motion_tokens)
            input_ids, gt_ids = [], []
            attention, middle = [], []
            for frame_idx, raw_frame in enumerate(example.motion_tokens):
                if len(raw_frame) != self.tokens_per_frame:
                    raise ValueError(
                        f"Expected {self.tokens_per_frame} tokens, got {len(raw_frame)}"
                    )
                global_frame = [
                    int(raw_id) + slot * self.codebook_size
                    for slot, raw_id in enumerate(raw_frame)
                ]
                is_middle = 0 < frame_idx < frames - 1
                gt_ids.extend(global_frame)
                input_ids.extend(
                    [self.mask_token_id] * self.tokens_per_frame
                    if is_middle
                    else global_frame
                )
                attention.extend([True] * self.tokens_per_frame)
                middle.extend([is_middle] * self.tokens_per_frame)
            pad_frames = max_frames - frames
            pad_tokens = pad_frames * self.tokens_per_frame
            input_ids.extend([self.mask_token_id] * pad_tokens)
            gt_ids.extend([self.mask_token_id] * pad_tokens)
            attention.extend([False] * pad_tokens)
            middle.extend([False] * pad_tokens)
            batch_input.append(input_ids)
            batch_gt.append(gt_ids)
            batch_attention.append(attention)
            batch_middle.append(middle)
            batch_audio.append(F.pad(example.audio_features, (0, 0, 0, pad_frames)))
        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "gt_ids": torch.tensor(batch_gt, dtype=torch.long),
            "audio_features": torch.stack(batch_audio).float(),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.bool),
            "middle_mask": torch.tensor(batch_middle, dtype=torch.bool),
            "gap_lengths": torch.tensor(
                [example.gap_frames for example in examples], dtype=torch.long
            ),
        }


def require_path(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_audio_motion_transformer(
    checkpoint: Path,
    device: torch.device,
) -> AudioMotionTransformer:
    """Load a local Trainer checkpoint without attempting a Hub download."""
    checkpoint = require_path(Path(checkpoint), "infill checkpoint")
    config = AudioMotionConfig.from_pretrained(str(checkpoint), local_files_only=True)
    model = AudioMotionTransformer(config)
    safetensors_path = checkpoint / "model.safetensors"
    bin_path = checkpoint / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = {}
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
    elif bin_path.exists():
        try:
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {checkpoint}"
        )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_part_codecs(
    checkpoint_by_part: Mapping[str, Path],
    device: torch.device,
    part_order: Sequence[str] = PART_ORDER,
) -> Dict[str, LoadedPartCodec]:
    from scripts.export_multipart_motion_tokens import load_part_codec

    codecs = {
        part: load_part_codec(require_path(Path(checkpoint_by_part[part]), f"{part} codec"), device)
        for part in part_order
    }
    for expected, loaded in codecs.items():
        if loaded.part != expected:
            raise ValueError(f"{expected} checkpoint contains part '{loaded.part}'")
    if len({codec.codebook_size for codec in codecs.values()}) != 1:
        raise ValueError("All multipart codecs must use the same codebook size")
    if len({codec.num_quantizers for codec in codecs.values()}) != 1:
        raise ValueError("All multipart codecs must use the same quantizer count")
    if len({codec.unit_length for codec in codecs.values()}) != 1:
        raise ValueError("All multipart codecs must use the same temporal unit length")
    return codecs


def usable_token_frames(item: Mapping[str, Any]) -> int:
    token_frames = len(item["motion_tokens"])
    audio_frames = int(item["audio_features"].shape[0])
    audio_fps = float(item["audio_fps"])
    token_fps = float(item["motion_token_fps"])
    if audio_frames <= 0 or audio_fps <= 0 or token_fps <= 0:
        return 0
    audio_usable = int(np.floor((audio_frames - 1) * token_fps / audio_fps)) + 1
    return max(0, min(token_frames, audio_usable))


def audio_feature_for_token_frame(
    item: Mapping[str, Any],
    token_idx: int,
) -> torch.Tensor:
    audio = item["audio_features"]
    audio_idx = int(
        round(token_idx * float(item["audio_fps"]) / float(item["motion_token_fps"]))
    )
    audio_idx = max(0, min(audio_idx, len(audio) - 1))
    feature = np.asarray(audio[audio_idx], dtype=np.float32)
    if not np.isfinite(feature).all():
        raise ValueError(
            f"Non-finite audio feature in {item.get('name', '<unknown>')} at index {audio_idx}"
        )
    return torch.from_numpy(feature)


def make_window_record(
    sequences: Sequence[Mapping[str, Any]],
    sequence_idx: int,
    left_idx: int,
    gap_frames: int,
) -> EvalWindowRecord:
    item = sequences[sequence_idx]
    right_idx = left_idx + gap_frames + 1
    frame_indices = list(range(left_idx, right_idx + 1))
    example = VariableGapMaskExample(
        name=str(item.get("name", sequence_idx)),
        left_idx=left_idx,
        right_idx=right_idx,
        gap_frames=gap_frames,
        motion_tokens=[list(item["motion_tokens"][idx]) for idx in frame_indices],
        audio_features=torch.stack(
            [audio_feature_for_token_frame(item, idx) for idx in frame_indices]
        ),
    )
    return EvalWindowRecord(
        sequence_idx=sequence_idx,
        name=example.name,
        left_idx=left_idx,
        gap_frames=gap_frames,
        example=example,
    )


def deterministic_gap_windows(
    sequences: Sequence[Mapping[str, Any]],
    gap_frames: int,
    *,
    windows_per_clip: int = 1,
    seed: int = 42,
) -> list[EvalWindowRecord]:
    """Sample the same unique windows for every model at one fixed gap."""
    if gap_frames < 1 or windows_per_clip < 1:
        raise ValueError("gap_frames and windows_per_clip must be positive")
    records: list[EvalWindowRecord] = []
    for sequence_idx, item in enumerate(sequences):
        usable = usable_token_frames(item)
        candidate_count = usable - gap_frames - 1
        if candidate_count <= 0:
            continue
        count = min(windows_per_clip, candidate_count)
        rng = random.Random(seed + sequence_idx * 1_000_003 + gap_frames * 10_007)
        left_indices = sorted(rng.sample(range(candidate_count), count))
        records.extend(
            make_window_record(sequences, sequence_idx, left_idx, gap_frames)
            for left_idx in left_indices
        )
    return records


def tiled_gap_windows(
    sequences: Sequence[Mapping[str, Any]],
    gap_frames: int,
) -> list[EvalWindowRecord]:
    """Tile each clip with contiguous anchor-to-anchor infill windows."""
    records: list[EvalWindowRecord] = []
    stride = gap_frames + 1
    for sequence_idx, item in enumerate(sequences):
        usable = usable_token_frames(item)
        for left_idx in range(0, usable - gap_frames - 1, stride):
            records.append(
                make_window_record(sequences, sequence_idx, left_idx, gap_frames)
            )
    return records


def _slot_constrained_c2f(
    model: AudioMotionTransformer,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    output = batch["input_ids"].clone()
    middle = batch["middle_mask"] & batch["attention_mask"]
    ntpf = int(model.config.num_tokens_per_frame)
    num_quantizers = int(model.config.num_quantizers_per_part)
    codebook_size = int(model.config.codebook_size)
    slots = torch.arange(output.shape[1], device=output.device).remainder(ntpf)
    for quantizer in range(num_quantizers):
        logits = model(
            output,
            audio_features=batch["audio_features"],
            attention_mask=batch["attention_mask"],
            middle_mask=batch["middle_mask"],
            gap_lengths=batch["gap_lengths"],
            c2f_stage=quantizer,
        )
        for slot in range(quantizer, ntpf, num_quantizers):
            fill = middle & slots.view(1, -1).eq(slot)
            start = slot * codebook_size
            local = logits[..., start : start + codebook_size].argmax(dim=-1) + start
            output[fill] = local[fill]
    return output


def apply_audio_input_mode(
    audio_features: torch.Tensor,
    chunk: Sequence[EvalWindowRecord],
    all_records: Sequence[EvalWindowRecord],
    *,
    start_index: int,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Apply deterministic inference-only audio controls to a window batch."""
    mode = str(mode).lower()
    valid_modes = {
        "correct",
        "temporal_shift",
        "temporal_shuffle",
        "cross_clip",
        "temporal_mean",
        "zero",
    }
    if mode not in valid_modes:
        raise ValueError(f"audio_input_mode must be one of {sorted(valid_modes)}")
    if mode == "correct":
        return audio_features
    if mode == "zero":
        return torch.zeros_like(audio_features)

    transformed = audio_features.clone()
    if mode == "cross_clip":
        if len(all_records) < 2:
            raise ValueError("cross_clip audio requires at least two evaluation records")
        for row, record in enumerate(chunk):
            global_index = start_index + row
            donor = None
            for distance in range(1, len(all_records)):
                candidate = all_records[(global_index + distance) % len(all_records)]
                if candidate.sequence_idx != record.sequence_idx:
                    donor = candidate
                    break
            if donor is None:
                donor = all_records[(global_index + 1) % len(all_records)]
            donor_audio = donor.example.audio_features.to(
                device=audio_features.device, dtype=audio_features.dtype
            )
            if donor_audio.shape != transformed[row].shape:
                raise ValueError("cross_clip donor does not match the window audio shape")
            transformed[row] = donor_audio
        return transformed

    for row, record in enumerate(chunk):
        frame_count = min(record.gap_frames + 2, transformed.shape[1])
        source = audio_features[row, :frame_count]
        if mode == "temporal_mean":
            transformed[row, :frame_count] = source.mean(dim=0, keepdim=True)
        elif mode == "temporal_shift":
            transformed[row, :frame_count] = source.roll(
                shifts=max(1, frame_count // 2), dims=0
            )
        else:
            local_seed = (
                int(seed)
                + int(record.sequence_idx) * 1_000_003
                + int(record.left_idx) * 10_007
                + int(record.gap_frames) * 1_009
            ) % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(local_seed)
            permutation = torch.randperm(frame_count, generator=generator).to(
                audio_features.device
            )
            transformed[row, :frame_count] = source[permutation]
    return transformed


@torch.no_grad()
def infer_window_records(
    model: AudioMotionTransformer,
    spec: InfillModelSpec,
    records: Sequence[EvalWindowRecord],
    *,
    batch_size: int,
    device: torch.device,
    slot_constrained: bool = False,
) -> list[np.ndarray]:
    if not records:
        return []
    if spec.disable_audio_adapters:
        if not model.audio_adapters:
            raise ValueError(
                f"{spec.name} requests disabled adapters but the model has none"
            )
        with torch.no_grad():
            for adapter in model.audio_adapters.values():
                adapter.residual_scale.zero_()
    collator = VariableGapC2FCollator(model.config)
    ntpf = int(model.config.num_tokens_per_frame)
    codebook_size = int(model.config.codebook_size)
    offsets = np.arange(ntpf, dtype=np.int64) * codebook_size
    predictions: list[np.ndarray] = []
    model.eval()
    for start_idx in tqdm(
        range(0, len(records), batch_size),
        desc=f"infer {spec.name} gap={records[0].gap_frames}",
        leave=False,
    ):
        chunk = records[start_idx : start_idx + batch_size]
        batch = collator([record.example for record in chunk])
        batch = {key: value.to(device) for key, value in batch.items()}
        batch["audio_features"] = apply_audio_input_mode(
            batch["audio_features"],
            chunk,
            records,
            start_index=start_idx,
            mode=spec.audio_input_mode,
            seed=spec.audio_ablation_seed,
        )
        if spec.decoder == "c2f":
            if slot_constrained:
                output = _slot_constrained_c2f(model, batch)
            else:
                output = model.generate_quantizer_coarse_to_fine(
                    batch["input_ids"],
                    batch["audio_features"],
                    middle_mask=batch["middle_mask"],
                    attention_mask=batch["attention_mask"],
                    gap_lengths=batch["gap_lengths"],
                )
        elif spec.decoder == "fixed_sbs":
            if slot_constrained:
                raise ValueError("slot_constrained is currently implemented only for C2F")
            output = model.generate_sbs(
                batch["input_ids"],
                batch["audio_features"],
                generate_steps=spec.generate_steps,
                attention_mask=batch["attention_mask"],
            )
        else:
            raise ValueError(f"Unknown decoder: {spec.decoder}")

        global_frames = output.reshape(len(chunk), -1, ntpf).detach().cpu().numpy()
        for row, record in zip(global_frames, chunk):
            frames = record.gap_frames + 2
            predictions.append((row[1 : frames - 1] - offsets).astype(np.int64))
    return predictions


@torch.no_grad()
def decode_multipart_token_batch(
    frame_batch: Sequence[Sequence[Sequence[int]]] | np.ndarray,
    codecs: Mapping[str, LoadedPartCodec],
    device: torch.device,
    *,
    part_order: Sequence[str] = PART_ORDER,
    clip_invalid: bool = True,
) -> list[np.ndarray]:
    arr = np.asarray(frame_batch, dtype=np.int64)
    if arr.ndim != 3:
        raise ValueError(f"Expected token frames shaped (B, T, slots), got {arr.shape}")
    codebook_size = int(next(iter(codecs.values())).codebook_size)
    num_quantizers = int(next(iter(codecs.values())).num_quantizers)
    expected_slots = len(part_order) * num_quantizers
    if arr.shape[2] != expected_slots:
        raise ValueError(f"Expected {expected_slots} token slots, got {arr.shape[2]}")
    if clip_invalid:
        arr = np.clip(arr, 0, codebook_size - 1)
    elif np.any((arr < 0) | (arr >= codebook_size)):
        raise ValueError("Cannot decode invalid local token IDs")

    part_features: Dict[str, np.ndarray] = {}
    for part_idx, part in enumerate(part_order):
        cols = arr[..., part_idx * num_quantizers : (part_idx + 1) * num_quantizers]
        indices = torch.as_tensor(cols, dtype=torch.long, device=device)
        loaded = codecs[part]
        reconstructed = loaded.model.decode({part: indices})[part]
        reconstructed = loaded.normalizer.denormalize_tensor(part, reconstructed)
        part_features[part] = reconstructed.float().cpu().numpy()
    return [
        merge_parts_to_legacy_motion(
            {part: values[batch_idx] for part, values in part_features.items()}
        )["body"].astype(np.float32)
        for batch_idx in range(arr.shape[0])
    ]


@torch.no_grad()
def decode_multipart_tokens(
    frames: Sequence[Sequence[int]] | np.ndarray,
    codecs: Mapping[str, LoadedPartCodec],
    device: torch.device,
    *,
    part_order: Sequence[str] = PART_ORDER,
    clip_invalid: bool = True,
) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected token frames shaped (T, slots), got {arr.shape}")
    return decode_multipart_token_batch(
        arr[None],
        codecs,
        device,
        part_order=part_order,
        clip_invalid=clip_invalid,
    )[0]


def invalid_token_fraction(frames: np.ndarray, codebook_size: int) -> float:
    arr = np.asarray(frames, dtype=np.int64)
    return float(np.mean((arr < 0) | (arr >= codebook_size)))


def decoded_feature_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt64 = np.asarray(gt, dtype=np.float64)
    pred64 = np.asarray(pred, dtype=np.float64)
    length = min(len(gt64), len(pred64))
    gt64, pred64 = gt64[:length], pred64[:length]
    diff = pred64 - gt64
    result = {
        "body_mae": float(np.mean(np.abs(diff))),
        "body_rmse": float(np.sqrt(np.mean(np.square(diff)))),
    }
    for order, label in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        if length > order:
            delta = np.diff(pred64, n=order, axis=0) - np.diff(gt64, n=order, axis=0)
            result[f"body_{label}_rmse"] = float(np.sqrt(np.mean(np.square(delta))))
        else:
            result[f"body_{label}_rmse"] = np.nan
    return result


def window_metric_row(
    record: EvalWindowRecord,
    predicted: np.ndarray,
    codecs: Mapping[str, LoadedPartCodec],
    device: torch.device,
    *,
    model_name: str,
    part_order: Sequence[str] = PART_ORDER,
    decoded_gt: Optional[np.ndarray] = None,
    decoded_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    gt = np.asarray(record.example.motion_tokens[1:-1], dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    codebook_size = int(next(iter(codecs.values())).codebook_size)
    num_quantizers = int(next(iter(codecs.values())).num_quantizers)
    matches = predicted == gt
    row: Dict[str, Any] = {
        "model": model_name,
        "name": record.name,
        "sequence_idx": record.sequence_idx,
        "left_idx": record.left_idx,
        "gap": record.gap_frames,
        "token_acc": float(matches.mean()),
        "exact_frame_acc": float(matches.all(axis=1).mean()),
        "exact_gap_acc": float(matches.all()),
        "invalid_token_frac": invalid_token_fraction(predicted, codebook_size),
    }
    for part_idx, part in enumerate(part_order):
        cols = slice(part_idx * num_quantizers, (part_idx + 1) * num_quantizers)
        row[f"part_{part}_acc"] = float(matches[:, cols].mean())
    for quantizer in range(num_quantizers):
        row[f"q{quantizer}_acc"] = float(matches[:, quantizer::num_quantizers].mean())
    if decoded_gt is None:
        decoded_gt = decode_multipart_tokens(gt, codecs, device, part_order=part_order)
    if decoded_pred is None:
        decoded_pred = decode_multipart_tokens(
            predicted, codecs, device, part_order=part_order
        )
    row.update(decoded_feature_metrics(decoded_gt, decoded_pred))
    return row


def evaluate_model_windows(
    model: AudioMotionTransformer,
    spec: InfillModelSpec,
    sequences: Sequence[Mapping[str, Any]],
    codecs: Mapping[str, LoadedPartCodec],
    gaps: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    windows_per_clip: int = 1,
    seed: int = 42,
    slot_constrained: bool = False,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for gap in gaps:
        if not spec.supports_gap(gap):
            continue
        records = deterministic_gap_windows(
            sequences,
            gap,
            windows_per_clip=windows_per_clip,
            seed=seed,
        )
        predictions = infer_window_records(
            model,
            spec,
            records,
            batch_size=batch_size,
            device=device,
            slot_constrained=slot_constrained,
        )
        for start_idx in tqdm(
            range(0, len(records), batch_size),
            desc=f"metrics {spec.name} gap={gap}",
            leave=False,
        ):
            record_batch = records[start_idx : start_idx + batch_size]
            prediction_batch = predictions[start_idx : start_idx + batch_size]
            gt_batch = np.stack(
                [np.asarray(record.example.motion_tokens[1:-1]) for record in record_batch]
            )
            pred_batch = np.stack(prediction_batch)
            decoded = decode_multipart_token_batch(
                np.concatenate([gt_batch, pred_batch], axis=0), codecs, device
            )
            split = len(record_batch)
            rows.extend(
                window_metric_row(
                    record,
                    prediction,
                    codecs,
                    device,
                    model_name=spec.name,
                    decoded_gt=decoded[row_idx],
                    decoded_pred=decoded[split + row_idx],
                )
                for row_idx, (record, prediction) in enumerate(
                    zip(record_batch, prediction_batch)
                )
            )
    return pd.DataFrame(rows)


def add_gap_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gap_bucket"] = pd.cut(
        frame["gap"],
        bins=[0, 3, 7, 15],
        labels=["short_1_3", "medium_4_7", "long_8_15"],
    )
    return frame


def summarize_window_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        column
        for column in frame.columns
        if column.endswith("_acc")
        or column.endswith("_rmse")
        or column in {"token_acc", "invalid_token_frac"}
    ]
    by_gap = frame.groupby(["model", "gap"], observed=True)[metrics].mean()
    # Macro averaging gives every gap equal weight instead of weighting long clips more.
    macro = by_gap.groupby(level="model").mean()
    return by_gap, macro


def paired_bootstrap_window_differences(
    frame: pd.DataFrame,
    *,
    reference_model: str,
    candidate_models: Sequence[str],
    metrics: Sequence[str],
    iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap paired per-clip metric differences against one reference model."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    required = {"model", "sequence_idx", "left_idx", "gap", *metrics}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing window metric columns: {', '.join(missing)}")

    keys = ["sequence_idx", "left_idx", "gap"]
    reference = frame.loc[frame["model"] == reference_model, keys + list(metrics)]
    if reference.empty:
        raise ValueError(f"Reference model not found: {reference_model}")
    reference = reference.rename(columns={metric: f"{metric}_reference" for metric in metrics})
    rows: list[Dict[str, Any]] = []
    for candidate_name in candidate_models:
        candidate = frame.loc[frame["model"] == candidate_name, keys + list(metrics)]
        paired = candidate.merge(reference, on=keys, how="inner", validate="one_to_one")
        if paired.empty:
            continue
        for gap, gap_frame in paired.groupby("gap", observed=True):
            clip_ids = np.asarray(sorted(gap_frame["sequence_idx"].unique()))
            if clip_ids.size == 0:
                continue
            rng = np.random.default_rng(
                int(seed) + int(gap) * 1009 + sum(ord(char) for char in candidate_name)
            )
            for metric in metrics:
                differences = (
                    gap_frame.assign(
                        difference=gap_frame[metric]
                        - gap_frame[f"{metric}_reference"]
                    )
                    .groupby("sequence_idx", observed=True)["difference"]
                    .mean()
                    .reindex(clip_ids)
                    .to_numpy(dtype=np.float64)
                )
                draws = rng.choice(
                    differences,
                    size=(int(iterations), len(differences)),
                    replace=True,
                ).mean(axis=1)
                rows.append(
                    {
                        "reference": reference_model,
                        "candidate": candidate_name,
                        "gap": int(gap),
                        "metric": metric,
                        "clips": len(differences),
                        "mean_difference": float(differences.mean()),
                        "ci_low": float(np.quantile(draws, 0.025)),
                        "ci_high": float(np.quantile(draws, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def assemble_tiled_token_sequences(
    sequences: Sequence[Mapping[str, Any]],
    records: Sequence[EvalWindowRecord],
    predictions: Sequence[np.ndarray],
) -> Dict[int, tuple[list[list[int]], list[list[int]]]]:
    grouped: Dict[int, list[tuple[EvalWindowRecord, np.ndarray]]] = {}
    for record, prediction in zip(records, predictions):
        grouped.setdefault(record.sequence_idx, []).append((record, prediction))

    outputs: Dict[int, tuple[list[list[int]], list[list[int]]]] = {}
    for sequence_idx, values in grouped.items():
        values.sort(key=lambda value: value[0].left_idx)
        source = sequences[sequence_idx]["motion_tokens"]
        gt_full: list[list[int]] = []
        pred_full: list[list[int]] = []
        for window_idx, (record, predicted_middle) in enumerate(values):
            left = record.left_idx
            right = left + record.gap_frames + 1
            if window_idx == 0:
                gt_full.append(list(source[left]))
                pred_full.append(list(source[left]))
            gt_full.extend(list(source[idx]) for idx in range(left + 1, right + 1))
            pred_full.extend(np.asarray(predicted_middle, dtype=np.int64).tolist())
            pred_full.append(list(source[right]))
        outputs[sequence_idx] = (gt_full, pred_full)
    return outputs


def load_raw_gt_body(
    motion_dir: Path,
    name: str,
    target_len: int,
    *,
    canonicalize_root: bool,
) -> Optional[np.ndarray]:
    path = motion_path_for_name(Path(motion_dir), name)
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    if not isinstance(raw, dict) or "body" not in raw:
        return None
    body = np.asarray(raw["body"], dtype=np.float32)
    if canonicalize_root:
        body, _, _ = canonicalize_body_root(body)
    return body[:target_len]


def save_evaluator_motion(path: Path, name: str, body: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = np.asarray(body, dtype=np.float32)
    zeros = np.zeros((len(body), 120), dtype=np.float32)
    np.save(path, {"name": name, "body": body, "left": zeros, "right": zeros})


def clean_output_files(path: Path, suffix: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob(f"*_{suffix}.npy"):
        old.unlink()


def export_full_clip_gap(
    model: AudioMotionTransformer,
    spec: InfillModelSpec,
    sequences: Sequence[Mapping[str, Any]],
    codecs: Mapping[str, LoadedPartCodec],
    *,
    gap: int,
    batch_size: int,
    device: torch.device,
    output_root: Path,
    motion_dir: Path,
    gt_source: str = "raw_motion_data",
    canonicalize_raw_root: bool = False,
    slot_constrained: bool = False,
    clean: bool = True,
) -> pd.DataFrame:
    if not spec.supports_gap(gap):
        raise ValueError(f"{spec.name} does not support gap={gap}")
    gap_root = Path(output_root) / f"gap{gap:02d}"
    gt_dir = gap_root / "gt"
    codec_dir = gap_root / "mp_codec_gt"
    pred_dir = gap_root / spec.name
    if clean:
        clean_output_files(gt_dir, "gt")
        clean_output_files(codec_dir, "pred")
        clean_output_files(pred_dir, "pred")

    records = tiled_gap_windows(sequences, gap)
    predictions = infer_window_records(
        model,
        spec,
        records,
        batch_size=batch_size,
        device=device,
        slot_constrained=slot_constrained,
    )
    assembled = assemble_tiled_token_sequences(sequences, records, predictions)
    rows = []
    for sequence_idx, (gt_tokens, pred_tokens) in tqdm(
        assembled.items(), desc=f"decode/export {spec.name} gap={gap}", leave=False
    ):
        item = sequences[sequence_idx]
        decoded_gt = decode_multipart_tokens(gt_tokens, codecs, device)
        decoded_pred = decode_multipart_tokens(pred_tokens, codecs, device)
        target_len = min(len(decoded_gt), len(decoded_pred))
        raw_gt = load_raw_gt_body(
            motion_dir,
            str(item["name"]),
            target_len,
            canonicalize_root=canonicalize_raw_root,
        )
        if gt_source == "raw_motion_data":
            gt_body = raw_gt if raw_gt is not None and len(raw_gt) else decoded_gt
        elif gt_source == "decoded_tokens":
            gt_body = decoded_gt
        else:
            raise ValueError("gt_source must be raw_motion_data or decoded_tokens")
        target_len = min(len(gt_body), len(decoded_gt), len(decoded_pred))
        if target_len < 2:
            continue
        stem = f"{sequence_idx:06d}"
        save_evaluator_motion(gt_dir / f"{stem}_gt.npy", str(item["name"]), gt_body[:target_len])
        save_evaluator_motion(
            codec_dir / f"{stem}_pred.npy", str(item["name"]), decoded_gt[:target_len]
        )
        save_evaluator_motion(
            pred_dir / f"{stem}_pred.npy", str(item["name"]), decoded_pred[:target_len]
        )
        rows.append(
            {
                "model": spec.name,
                "gap": gap,
                "file_stem": stem,
                "name": item["name"],
                "token_frames": len(gt_tokens),
                "frames_20fps": target_len,
                "gt_source": gt_source,
                "canonicalize_raw_root": canonicalize_raw_root,
            }
        )
    manifest = pd.DataFrame(rows)
    gap_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(gap_root / f"manifest_{spec.name}.csv", index=False)
    return manifest


def load_yaml_namespace(path: Path) -> SimpleNamespace:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    def convert(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: convert(item) for key, item in value.items()})
        return value

    return convert(data)


def torch_load_trusted(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_official_evaluator_helpers(evaluation_dir: Path) -> Dict[str, Any]:
    names = [
        "ChronTMR",
        "PredMotionTextDataset",
        "calculate_alignment_single",
        "calculate_esd",
        "compute_128_sample_metrics",
        "compute_fid_diversity_metrics",
        "extract_audio_beats",
        "extract_motion_beats",
        "extract_motion_beats_for_esd",
        "velocity_onset_correlation",
    ]
    saved_path = list(sys.path)
    prefixes = ("models", "datasets", "evaluate_pred_motion_v2")
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in prefixes or name.startswith("models.") or name.startswith("datasets.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(evaluation_dir))
    try:
        module = importlib.import_module("evaluate_pred_motion_v2")
        return {name: getattr(module, name) for name in names}
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name in prefixes or name.startswith("models.") or name.startswith("datasets."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def evaluator_length_to_mask(
    lengths: torch.Tensor,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(max_length, device=device).unsqueeze(0)
    return positions < lengths.to(device=device, dtype=torch.long).unsqueeze(1)


def load_evaluator_motion_encoder(
    evaluator_checkpoint: Path,
    evaluator_config: Path,
    device: torch.device,
):
    from evaluation.models.actor import ACTORStyleEncoder

    cfg = load_yaml_namespace(require_path(evaluator_config, "evaluator config"))
    encoder = ACTORStyleEncoder(
        cfg.motion_encoder.nfeats,
        cfg.motion_encoder.vae,
        cfg.motion_encoder.latent_dim,
        cfg.motion_encoder.ff_size,
        cfg.motion_encoder.num_layers,
        cfg.motion_encoder.num_heads,
        cfg.motion_encoder.dropout,
        cfg.motion_encoder.activation,
    )
    checkpoint = torch_load_trusted(
        require_path(evaluator_checkpoint, "evaluator checkpoint"), map_location="cpu"
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    state = {
        key.replace("motion_encoder.", "", 1): value
        for key, value in checkpoint.items()
        if key.startswith("motion_encoder.")
    }
    encoder.load_state_dict(state, strict=False)
    return cfg, encoder.to(device).eval()


def load_evaluator_body(path: Path) -> tuple[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        return str(raw.get("name", path.stem)), np.asarray(raw["body"], dtype=np.float32)
    return path.stem, np.asarray(raw, dtype=np.float32)


@torch.no_grad()
def encode_evaluator_latents(
    motion_dir: Path,
    suffix: str,
    cfg,
    encoder,
    evaluator_stats_dir: Path,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> tuple[list[str], np.ndarray]:
    files = sorted(Path(motion_dir).glob(f"*_{suffix}.npy"))
    if not files:
        raise FileNotFoundError(f"No *_{suffix}.npy files found in {motion_dir}")
    mean = torch_load_trusted(Path(evaluator_stats_dir) / "mean.pt", map_location="cpu").float()
    std = torch_load_trusted(Path(evaluator_stats_dir) / "std.pt", map_location="cpu").float()
    max_len = int(cfg.dataset.max_motion_length)
    names: list[str] = []
    latents: list[torch.Tensor] = []
    motions: list[torch.Tensor] = []
    lengths: list[int] = []

    def flush() -> None:
        if not motions:
            return
        x = torch.stack(motions).to(device)
        lens = torch.tensor(lengths, dtype=torch.long, device=device)
        mask = evaluator_length_to_mask(lens, max_len, device)
        latents.append(encoder({"x": x, "mask": mask})[:, 0].detach().cpu())
        motions.clear()
        lengths.clear()

    for path in tqdm(files, desc=f"encode {Path(motion_dir).name}", leave=False):
        name, body = load_evaluator_body(path)
        length = min(len(body), max_len)
        body_tensor = torch.as_tensor(body[:length], dtype=torch.float32)
        body_dim = body_tensor.shape[1]
        body_tensor = (body_tensor - mean[:body_dim]) / (std[:body_dim] + 1e-12)
        if length < max_len:
            body_tensor = torch.cat(
                [body_tensor, torch.zeros(max_len - length, body_dim)], dim=0
            )
        names.append(name)
        motions.append(body_tensor)
        lengths.append(length)
        if len(motions) >= batch_size:
            flush()
    flush()
    return names, torch.cat(latents).numpy()


def compute_fid_table(
    gap_root: Path,
    model_names: Sequence[str],
    *,
    evaluation_dir: Path,
    evaluator_checkpoint: Path,
    evaluator_config: Path,
    evaluator_stats_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    diversity_times: int = 300,
) -> pd.DataFrame:
    helpers = load_official_evaluator_helpers(evaluation_dir)
    cfg, encoder = load_evaluator_motion_encoder(
        evaluator_checkpoint, evaluator_config, device
    )
    _, gt_latents = encode_evaluator_latents(
        Path(gap_root) / "gt",
        "gt",
        cfg,
        encoder,
        evaluator_stats_dir,
        device,
        batch_size=batch_size,
    )
    rows = []
    for model_name in ["mp_codec_gt", *model_names]:
        _, pred_latents = encode_evaluator_latents(
            Path(gap_root) / model_name,
            "pred",
            cfg,
            encoder,
            evaluator_stats_dir,
            device,
            batch_size=batch_size,
        )
        metrics = helpers["compute_fid_diversity_metrics"](
            gt_latents, pred_latents, diversity_times=diversity_times
        )
        rows.append({"model": model_name, "num_clips": len(pred_latents), **metrics})
    return pd.DataFrame(rows).set_index("model")


def load_chrontmr_retrieval_model(
    evaluator_checkpoint: Path,
    evaluator_config: Path,
    text_model_path: str,
    helpers: Mapping[str, Any],
    device: torch.device,
):
    import transformers
    from transformers import AutoTokenizer

    cfg = load_yaml_namespace(evaluator_config)
    resolved_text_path = str(Path(text_model_path)) if Path(text_model_path).exists() else text_model_path
    cfg.model.text_model_name = resolved_text_path
    original = transformers.AutoModel.from_pretrained

    def redirected(model_name, *args, **kwargs):
        name = str(model_name)
        if "bert-base-chinese" in name or name.startswith("/data/home/jinch/"):
            return original(resolved_text_path, *args, **kwargs)
        return original(model_name, *args, **kwargs)

    transformers.AutoModel.from_pretrained = redirected
    try:
        model = helpers["ChronTMR"](cfg, vae=False)
    finally:
        transformers.AutoModel.from_pretrained = original
    checkpoint = torch_load_trusted(evaluator_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(resolved_text_path)
    return cfg, model, tokenizer


def compute_retrieval_table(
    gap_root: Path,
    model_names: Sequence[str],
    *,
    evaluation_dir: Path,
    evaluator_checkpoint: Path,
    evaluator_config: Path,
    evaluator_stats_dir: Path,
    motion2text_json: Path,
    text_model_path: str,
    device: torch.device,
) -> pd.DataFrame:
    helpers = load_official_evaluator_helpers(evaluation_dir)
    cfg, model, tokenizer = load_chrontmr_retrieval_model(
        evaluator_checkpoint,
        evaluator_config,
        text_model_path,
        helpers,
        device,
    )
    targets = [("mp_real_gt", Path(gap_root) / "gt", "gt")]
    targets.extend(
        (name, Path(gap_root) / name, "pred")
        for name in ["mp_codec_gt", *model_names]
    )
    rows = []
    for name, motion_dir, suffix in targets:
        dataset = helpers["PredMotionTextDataset"](
            cfg=cfg,
            motion_dir=str(motion_dir),
            motion2text_path=str(motion2text_json),
            stats_dir=str(evaluator_stats_dir),
            motion_type=suffix,
        )
        if not len(dataset):
            continue
        seed = int(cfg.train.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        metrics = helpers["compute_128_sample_metrics"](
            cfg, model, dataset, device, tokenizer
        )
        rows.append({"model": name, "num_clips": len(dataset), **metrics})
    return pd.DataFrame(rows).set_index("model")


def seam_discontinuity_metrics(
    body: np.ndarray,
    *,
    boundary_stride: int,
) -> Dict[str, float]:
    body = np.asarray(body, dtype=np.float64)
    acceleration: list[float] = []
    jerk: list[float] = []
    interior_acceleration: list[float] = []
    interior_jerk: list[float] = []
    boundaries = set(range(boundary_stride, len(body), boundary_stride))

    def acceleration_discontinuity(center: int) -> float:
        left = body[center] - 2 * body[center - 1] + body[center - 2]
        right = body[center + 2] - 2 * body[center + 1] + body[center]
        return float(np.linalg.norm(right - left))

    def jerk_discontinuity(center: int) -> float:
        left = (
            body[center]
            - 3 * body[center - 1]
            + 3 * body[center - 2]
            - body[center - 3]
        )
        right = (
            body[center + 3]
            - 3 * body[center + 2]
            + 3 * body[center + 1]
            - body[center]
        )
        return float(np.linalg.norm(right - left))

    for boundary in range(boundary_stride, len(body), boundary_stride):
        if 2 <= boundary and boundary + 2 < len(body):
            acceleration.append(acceleration_discontinuity(boundary))
        if 3 <= boundary and boundary + 3 < len(body):
            jerk.append(jerk_discontinuity(boundary))
    for center in range(2, max(2, len(body) - 2)):
        if center not in boundaries:
            interior_acceleration.append(acceleration_discontinuity(center))
    for center in range(3, max(3, len(body) - 3)):
        if center not in boundaries:
            interior_jerk.append(jerk_discontinuity(center))

    acceleration_mean = float(np.mean(acceleration)) if acceleration else np.nan
    jerk_mean = float(np.mean(jerk)) if jerk else np.nan
    interior_acceleration_mean = (
        float(np.mean(interior_acceleration)) if interior_acceleration else np.nan
    )
    interior_jerk_mean = float(np.mean(interior_jerk)) if interior_jerk else np.nan
    return {
        "seam_accel_l2_mean": acceleration_mean,
        "seam_accel_l2_p95": float(np.percentile(acceleration, 95)) if acceleration else np.nan,
        "seam_jerk_l2_mean": jerk_mean,
        "seam_jerk_l2_p95": float(np.percentile(jerk, 95)) if jerk else np.nan,
        "interior_accel_l2_mean": interior_acceleration_mean,
        "interior_jerk_l2_mean": interior_jerk_mean,
        "seam_accel_excess_ratio": acceleration_mean / max(interior_acceleration_mean, 1e-12),
        "seam_jerk_excess_ratio": jerk_mean / max(interior_jerk_mean, 1e-12),
        "seam_count": max(len(acceleration), len(jerk)),
    }


def generated_gap_dynamics_metrics(
    body: np.ndarray,
    *,
    gap: int,
    codec_unit_length: int,
    feature_start: int = 3,
) -> Dict[str, float]:
    """Measure dynamics strictly inside generated token blocks, excluding anchors."""
    body = np.asarray(body, dtype=np.float64)
    if body.ndim != 2:
        raise ValueError(f"Expected body shaped (frames, features), got {body.shape}")
    if gap < 1 or codec_unit_length < 1:
        raise ValueError("gap and codec_unit_length must be >= 1")
    features = body[:, int(feature_start) :]
    token_ids = np.arange(len(features)) // int(codec_unit_length)
    generated = token_ids % (int(gap) + 1) != 0

    def derivative_rms(order: int) -> tuple[float, int]:
        if len(features) <= order:
            return np.nan, 0
        derivative = np.diff(features, n=order, axis=0)
        valid = np.ones(len(derivative), dtype=bool)
        for offset in range(order + 1):
            valid &= generated[offset : offset + len(derivative)]
        if not bool(valid.any()):
            return np.nan, 0
        squared_l2 = np.square(derivative[valid]).sum(axis=1)
        return float(np.sqrt(squared_l2.mean())), int(valid.sum())

    velocity_rms, velocity_frames = derivative_rms(1)
    acceleration_rms, acceleration_frames = derivative_rms(2)
    jerk_rms, jerk_frames = derivative_rms(3)
    return {
        "gap_velocity_l2_rms": velocity_rms,
        "gap_acceleration_l2_rms": acceleration_rms,
        "gap_jerk_l2_rms": jerk_rms,
        "gap_velocity_frames": velocity_frames,
        "gap_acceleration_frames": acceleration_frames,
        "gap_jerk_frames": jerk_frames,
    }


def compute_gap_dynamics_table(
    gap_root: Path,
    model_names: Sequence[str],
    *,
    gap: int,
    codec_unit_length: int,
    feature_start: int = 3,
) -> pd.DataFrame:
    targets = [("mp_real_gt", Path(gap_root) / "gt", "gt")]
    targets.extend(
        (name, Path(gap_root) / name, "pred")
        for name in ["mp_codec_gt", *model_names]
    )
    rows = []
    for name, directory, suffix in targets:
        metrics = [
            generated_gap_dynamics_metrics(
                load_evaluator_body(path)[1],
                gap=gap,
                codec_unit_length=codec_unit_length,
                feature_start=feature_start,
            )
            for path in sorted(directory.glob(f"*_{suffix}.npy"))
        ]
        frame = pd.DataFrame(metrics)
        if frame.empty:
            continue
        rows.append(
            {
                "model": name,
                "num_clips": len(frame),
                **{
                    column: float(frame[column].mean())
                    for column in frame.columns
                },
            }
        )
    result = pd.DataFrame(rows).set_index("model")
    if "mp_codec_gt" in result.index:
        for metric in (
            "gap_velocity_l2_rms",
            "gap_acceleration_l2_rms",
            "gap_jerk_l2_rms",
        ):
            denominator = float(result.loc["mp_codec_gt", metric])
            result[f"{metric}_ratio_to_codec"] = result[metric] / max(
                denominator, 1e-12
            )
    return result


def compute_seam_table(
    gap_root: Path,
    model_names: Sequence[str],
    *,
    gap: int,
    codec_unit_length: int,
) -> pd.DataFrame:
    boundary_stride = (gap + 1) * codec_unit_length
    targets = [("mp_real_gt", Path(gap_root) / "gt", "gt")]
    targets.extend(
        (name, Path(gap_root) / name, "pred")
        for name in ["mp_codec_gt", *model_names]
    )
    rows = []
    for name, directory, suffix in targets:
        metrics = [
            seam_discontinuity_metrics(
                load_evaluator_body(path)[1], boundary_stride=boundary_stride
            )
            for path in sorted(directory.glob(f"*_{suffix}.npy"))
        ]
        frame = pd.DataFrame(metrics)
        if frame.empty:
            continue
        rows.append(
            {
                "model": name,
                "num_clips": len(frame),
                "seam_count": int(frame["seam_count"].sum()),
                **{
                    column: float(frame[column].mean())
                    for column in frame.columns
                    if column != "seam_count"
                },
            }
        )
    return pd.DataFrame(rows).set_index("model")


def compute_beat_table(
    gap_root: Path,
    model_names: Sequence[str],
    *,
    evaluation_dir: Path,
    wav_dir: Path,
    fps: int = 20,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    helpers = load_official_evaluator_helpers(evaluation_dir)
    targets = [("mp_real_gt", Path(gap_root) / "gt", "gt")]
    targets.extend(
        (name, Path(gap_root) / name, "pred")
        for name in ["mp_codec_gt", *model_names]
    )
    rows = []
    for name, directory, suffix in targets:
        bas_values: list[float] = []
        bhr_values: list[float] = []
        esd_values: list[float] = []
        audio_to_motion_values: list[float] = []
        motion_to_audio_values: list[float] = []
        audio_beat_counts: list[int] = []
        motion_beat_counts: list[int] = []
        esd_motion_beat_counts: list[int] = []
        voc_values: list[float] = []
        used = 0
        for path in tqdm(sorted(directory.glob(f"*_{suffix}.npy")), desc=f"beat {name}", leave=False):
            clip_name, body = load_evaluator_body(path)
            wav_path = Path(wav_dir) / f"{clip_name}.wav"
            if not wav_path.exists():
                continue
            audio_beats = helpers["extract_audio_beats"](str(wav_path))
            motion_beats = helpers["extract_motion_beats"](body, fps=fps)
            bas, bhr = helpers["calculate_alignment_single"](
                motion_beats, audio_beats, tolerance=tolerance
            )
            esd_motion_beats = helpers["extract_motion_beats_for_esd"](
                body, fps=fps
            )
            esd = helpers["calculate_esd"](audio_beats, esd_motion_beats)
            if len(audio_beats) and len(esd_motion_beats):
                distances = np.abs(
                    np.asarray(audio_beats)[:, None]
                    - np.asarray(esd_motion_beats)[None, :]
                )
                audio_to_motion_values.append(float(distances.min(axis=1).mean()))
                motion_to_audio_values.append(float(distances.min(axis=0).mean()))
            elif len(audio_beats) == 0 and len(esd_motion_beats) == 0:
                audio_to_motion_values.append(0.0)
                motion_to_audio_values.append(0.0)
            else:
                audio_to_motion_values.append(2.0)
                motion_to_audio_values.append(2.0)
            audio_beat_counts.append(len(audio_beats))
            motion_beat_counts.append(len(motion_beats))
            esd_motion_beat_counts.append(len(esd_motion_beats))
            try:
                voc = helpers["velocity_onset_correlation"](
                    body, str(wav_path), fps=fps
                )
                if voc is not None and np.isfinite(voc):
                    voc_values.append(float(voc))
            except Exception:
                pass
            if bas is not None and not np.isnan(bas):
                bas_values.append(float(bas))
            if bhr is not None:
                bhr_values.append(float(bhr))
            esd_values.append(float(esd))
            used += 1
        rows.append(
            {
                "model": name,
                "num_clips": used,
                "BAS_distance_lower_better": np.mean(bas_values) if bas_values else np.nan,
                "BHR_higher_better": np.mean(bhr_values) if bhr_values else np.nan,
                "ESD_lower_better": np.mean(esd_values) if esd_values else np.nan,
                "ESD_audio_to_motion_lower_better": (
                    np.mean(audio_to_motion_values)
                    if audio_to_motion_values
                    else np.nan
                ),
                "ESD_motion_to_audio_lower_better": (
                    np.mean(motion_to_audio_values)
                    if motion_to_audio_values
                    else np.nan
                ),
                "audio_beats_per_clip": (
                    np.mean(audio_beat_counts) if audio_beat_counts else np.nan
                ),
                "motion_beats_per_clip": (
                    np.mean(motion_beat_counts) if motion_beat_counts else np.nan
                ),
                "esd_motion_beats_per_clip": (
                    np.mean(esd_motion_beat_counts)
                    if esd_motion_beat_counts
                    else np.nan
                ),
                "motion_to_audio_beat_count_ratio": (
                    np.sum(motion_beat_counts) / max(np.sum(audio_beat_counts), 1)
                    if motion_beat_counts
                    else np.nan
                ),
                "VOC_higher_better": np.mean(voc_values) if voc_values else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

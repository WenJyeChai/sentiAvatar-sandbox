#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed


THIS_DIR = Path(__file__).resolve().parent
MOTION_GENERATION_DIR = THIS_DIR.parent
PROJECT_DIR = MOTION_GENERATION_DIR.parent
sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer  # noqa: E402


IGNORE_INDEX = -100
DEFAULT_PART_ORDER = ["upper", "lower", "feet", "hands"]


@contextmanager
def timed_stage(name: str, enabled: bool = True):
    if not enabled:
        yield
        return
    start = time.perf_counter()
    print(f"[Timing] {name} ...")
    try:
        yield
    finally:
        print(f"[Timing] {name}: {time.perf_counter() - start:.3f}s")


def format_fps_for_dir(fps: float) -> str:
    if float(fps).is_integer():
        return str(int(fps))
    return str(fps).replace(".", "p")


def normalize_name(name: str) -> str:
    name = name.replace("\\", "/").strip().strip("/")
    suffix = Path(name).suffix.lower()
    if suffix in {".wav", ".npy", ".json", ".npz"}:
        name = name[: -len(suffix)]
    return name


def read_split_file(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [normalize_name(line) for line in f if line.strip()]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_token_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = load_json(path)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"tokens": data}
    raise ValueError(f"Unsupported token JSON format: {path}")


def discover_names(
    motion_token_dir: Path,
    audio_feat_dir: Path,
    split_names: Optional[Sequence[str]],
) -> List[str]:
    if split_names is not None:
        names = list(split_names)
    else:
        motion_names = {
            path.relative_to(motion_token_dir).with_suffix("").as_posix()
            for path in motion_token_dir.rglob("*.json")
            if path.name != "export_manifest.json"
        }
        audio_names = {
            path.relative_to(audio_feat_dir).with_suffix("").as_posix()
            for path in audio_feat_dir.rglob("*.npy")
        }
        names = sorted(motion_names & audio_names)

    available = []
    for name in names:
        name = normalize_name(name)
        if (motion_token_dir / f"{name}.json").exists() and (
            audio_feat_dir / f"{name}.npy"
        ).exists():
            available.append(name)
    return available


def parse_part_order(text: Optional[str], manifest: Optional[Dict[str, Any]]) -> List[str]:
    if text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
    elif manifest and manifest.get("part_order"):
        parts = [str(part) for part in manifest["part_order"]]
    else:
        parts = list(DEFAULT_PART_ORDER)
    if not parts:
        raise ValueError("part_order is empty")
    return parts


def build_token_layout(part_order: Sequence[str], num_quantizers_per_part: int) -> List[Dict[str, Any]]:
    layout: List[Dict[str, Any]] = []
    slot = 0
    for part in part_order:
        for quantizer in range(num_quantizers_per_part):
            layout.append({"slot": slot, "part": part, "quantizer": quantizer})
            slot += 1
    return layout


def load_manifest(motion_token_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = motion_token_dir / "export_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Bad manifest format: {manifest_path}")
    return manifest


def validate_motion_tokens(
    tokens: Any,
    *,
    name: str,
    codebook_size: int,
    num_tokens_per_frame: int,
) -> Optional[List[List[int]]]:
    if not isinstance(tokens, list) or not tokens:
        return None

    result: List[List[int]] = []
    for frame_idx, frame in enumerate(tokens):
        if not isinstance(frame, list) or len(frame) != num_tokens_per_frame:
            raise ValueError(
                f"{name}: frame {frame_idx} has {len(frame) if isinstance(frame, list) else 'non-list'} "
                f"tokens, expected {num_tokens_per_frame}"
            )
        values = [int(value) for value in frame]
        bad = [value for value in values if value < 0 or value >= codebook_size]
        if bad:
            raise ValueError(
                f"{name}: frame {frame_idx} has token outside [0,{codebook_size - 1}]: {bad[:5]}"
            )
        result.append(values)
    return result


def load_sequences(
    names: Sequence[str],
    motion_token_dir: Path,
    audio_feat_dir: Path,
    *,
    codebook_size: int,
    num_tokens_per_frame: int,
    audio_fps: float,
    source_motion_fps_fallback: float,
    motion_token_fps_override: Optional[float],
    motion_token_unit_length_override: Optional[float],
    max_sequences: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    sequences: List[Dict[str, Any]] = []
    stats = {
        "requested": len(names),
        "loaded": 0,
        "missing": 0,
        "bad_tokens": 0,
        "bad_audio": 0,
    }

    for name in names:
        if max_sequences is not None and len(sequences) >= max_sequences:
            break

        token_path = motion_token_dir / f"{name}.json"
        audio_path = audio_feat_dir / f"{name}.npy"
        payload = load_token_json(token_path)
        if payload is None or not audio_path.exists():
            stats["missing"] += 1
            continue

        try:
            motion_tokens = validate_motion_tokens(
                payload.get("tokens"),
                name=name,
                codebook_size=codebook_size,
                num_tokens_per_frame=num_tokens_per_frame,
            )
        except Exception as exc:
            print(f"[skip bad tokens] {name}: {exc}")
            stats["bad_tokens"] += 1
            continue
        if not motion_tokens:
            stats["bad_tokens"] += 1
            continue

        audio_features = np.load(audio_path).astype(np.float32)
        if audio_features.ndim != 2 or audio_features.shape[0] == 0:
            stats["bad_audio"] += 1
            continue

        source_motion_fps = float(payload.get("fps") or source_motion_fps_fallback)
        unit_length = float(
            motion_token_unit_length_override
            if motion_token_unit_length_override is not None
            else payload.get("motion_token_unit_length") or 2.0
        )
        if unit_length <= 0:
            stats["bad_tokens"] += 1
            continue

        motion_token_fps = float(
            motion_token_fps_override
            if motion_token_fps_override is not None
            else payload.get("motion_token_fps") or source_motion_fps / unit_length
        )
        if motion_token_fps <= 0:
            stats["bad_tokens"] += 1
            continue

        sequences.append(
            {
                "name": name,
                "motion_tokens": motion_tokens,
                "audio_features": audio_features,
                "source_motion_fps": source_motion_fps,
                "motion_token_unit_length": unit_length,
                "motion_token_fps": motion_token_fps,
                "audio_fps": audio_fps,
            }
        )
        stats["loaded"] += 1

    return sequences, stats


@dataclass
class AudioMotionMaskExample:
    name: str
    left_idx: int
    right_idx: int
    motion_tokens: List[List[int]]
    audio_features: torch.Tensor


class AudioMotionMaskDataset(Dataset):
    """Classic bidirectional Step 2 windows for AudioMotionTransformer."""

    def __init__(
        self,
        sequences: Sequence[Dict[str, Any]],
        *,
        step: int = 4,
        window_stride: Optional[int] = None,
        max_windows_per_sequence: Optional[int] = None,
        seed: int = 42,
    ):
        if step < 2:
            raise ValueError("step must be >= 2")
        stride = int(window_stride or step)
        if stride < 1:
            raise ValueError("window_stride must be >= 1")

        self.sequences = list(sequences)
        self.step = int(step)
        self.window_stride = stride
        self.windows: List[tuple[int, int]] = []
        rng = random.Random(seed)

        for seq_idx, item in enumerate(self.sequences):
            usable_frames = self._usable_motion_frames(item)
            starts = list(range(0, usable_frames - self.step, self.window_stride))
            if max_windows_per_sequence is not None and len(starts) > max_windows_per_sequence:
                rng.shuffle(starts)
                starts = sorted(starts[:max_windows_per_sequence])
            self.windows.extend((seq_idx, left_idx) for left_idx in starts)

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _usable_motion_frames(item: Dict[str, Any]) -> int:
        token_frames = len(item["motion_tokens"])
        audio_frames = int(item["audio_features"].shape[0])
        audio_fps = float(item["audio_fps"])
        motion_token_fps = float(item["motion_token_fps"])
        if audio_frames <= 0 or audio_fps <= 0 or motion_token_fps <= 0:
            return 0
        audio_usable = int(math.floor((audio_frames - 1) * motion_token_fps / audio_fps)) + 1
        return max(0, min(token_frames, audio_usable))

    @staticmethod
    def _audio_feature_for_frame(item: Dict[str, Any], motion_token_idx: int) -> torch.Tensor:
        audio_features: np.ndarray = item["audio_features"]
        audio_idx = int(
            round(
                float(motion_token_idx)
                * float(item["audio_fps"])
                / float(item["motion_token_fps"])
            )
        )
        audio_idx = max(0, min(audio_idx, len(audio_features) - 1))
        return torch.tensor(audio_features[audio_idx], dtype=torch.float32)

    def __getitem__(self, idx: int) -> AudioMotionMaskExample:
        seq_idx, left_idx = self.windows[idx]
        item = self.sequences[seq_idx]
        right_idx = left_idx + self.step
        frame_indices = list(range(left_idx, right_idx + 1))

        return AudioMotionMaskExample(
            name=str(item.get("name", seq_idx)),
            left_idx=left_idx,
            right_idx=right_idx,
            motion_tokens=[list(item["motion_tokens"][i]) for i in frame_indices],
            audio_features=torch.stack(
                [self._audio_feature_for_frame(item, i) for i in frame_indices],
                dim=0,
            ),
        )


class AudioMotionMaskCollator:
    def __init__(self, config: AudioMotionConfig):
        self.config = config
        self.mask_token_id = int(config.vocab_size) - 1
        self.num_frames = int(config.num_frames)
        self.num_tokens_per_frame = int(config.num_tokens_per_frame)
        self.codebook_size = int(config.codebook_size)

    def _motion_frame_to_ids(self, frame: Sequence[int]) -> List[int]:
        return [
            int(raw_id) + slot * self.codebook_size
            for slot, raw_id in enumerate(frame)
        ]

    def encode(self, example: AudioMotionMaskExample) -> tuple[List[int], List[int]]:
        if len(example.motion_tokens) != self.num_frames:
            raise ValueError(
                f"{example.name}: got {len(example.motion_tokens)} frames, expected {self.num_frames}"
            )

        input_ids: List[int] = []
        labels: List[int] = []
        last_frame = self.num_frames - 1
        for frame_idx, frame in enumerate(example.motion_tokens):
            token_ids = self._motion_frame_to_ids(frame)
            is_middle = 0 < frame_idx < last_frame
            for token_id in token_ids:
                if is_middle:
                    input_ids.append(self.mask_token_id)
                    labels.append(token_id)
                else:
                    input_ids.append(token_id)
                    labels.append(IGNORE_INDEX)
        return input_ids, labels

    def __call__(self, examples: Sequence[AudioMotionMaskExample]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        audio_features = []
        for example in examples:
            encoded_input_ids, encoded_labels = self.encode(example)
            input_ids.append(encoded_input_ids)
            labels.append(encoded_labels)
            audio_features.append(example.audio_features.to(dtype=torch.float32))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "audio_features": torch.stack(audio_features, dim=0),
        }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evenly_spaced_indices(length: int, count: int) -> List[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, count, dtype=np.int64).tolist()))


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def slot_stat_masks(
    labels: torch.Tensor,
    *,
    num_tokens_per_frame: int,
    num_quantizers_per_part: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = labels.ne(IGNORE_INDEX)
    positions = torch.arange(labels.size(1), device=labels.device).view(1, -1)
    slots = positions.remainder(num_tokens_per_frame).expand_as(labels)
    part_ids = slots.div(num_quantizers_per_part, rounding_mode="floor")
    quantizer_ids = slots.remainder(num_quantizers_per_part)
    return valid, part_ids, quantizer_ids


class AudioMotionMaskMetricsCallback(TrainerCallback):
    def __init__(
        self,
        *,
        eval_dataset: AudioMotionMaskDataset,
        collator: AudioMotionMaskCollator,
        part_order: Sequence[str],
        num_quantizers_per_part: int,
        teacher_forced_examples: int = 0,
        teacher_forced_batch_size: int = 256,
        teacher_forced_every_n_evals: int = 1,
        generation_examples: int = 0,
        generation_batch_size: int = 128,
        generation_every_n_evals: int = 1,
        generate_steps: int = 1,
    ):
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.part_order = list(part_order)
        self.num_quantizers_per_part = int(num_quantizers_per_part)
        self.teacher_forced_indices = evenly_spaced_indices(
            len(eval_dataset), teacher_forced_examples
        )
        self.teacher_forced_batch_size = int(teacher_forced_batch_size)
        self.teacher_forced_every_n_evals = max(1, int(teacher_forced_every_n_evals))
        self.generation_indices = evenly_spaced_indices(len(eval_dataset), generation_examples)
        self.generation_batch_size = int(generation_batch_size)
        self.generation_every_n_evals = max(1, int(generation_every_n_evals))
        self.generate_steps = max(1, int(generate_steps))
        self.eval_count = 0

    def _accumulate_accuracy(
        self,
        payload: Dict[str, float],
        prefix: str,
        preds: torch.Tensor,
        labels: torch.Tensor,
        *,
        logits: Optional[torch.Tensor] = None,
        topk: Optional[torch.Tensor] = None,
    ) -> None:
        valid, part_ids, quantizer_ids = slot_stat_masks(
            labels,
            num_tokens_per_frame=self.collator.num_tokens_per_frame,
            num_quantizers_per_part=self.num_quantizers_per_part,
        )
        correct = preds.eq(labels) & valid
        total = int(valid.sum().item())
        if total == 0:
            return

        def add_mean_stat(key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
            stat_total = int(mask.sum().item())
            if stat_total == 0:
                return
            payload[f"{key}_sum"] = payload.get(f"{key}_sum", 0.0) + float(
                values[mask].sum().item()
            )
            payload[f"{key}_count"] = payload.get(f"{key}_count", 0.0) + stat_total

        payload[f"{prefix}/token_acc"] = payload.get(f"{prefix}/token_acc", 0.0) + int(
            correct.sum().item()
        )
        payload[f"{prefix}/token_total"] = payload.get(f"{prefix}/token_total", 0.0) + total

        nll = pred_conf = target_prob = None
        if logits is not None:
            logits_f = logits.float()
            safe_labels = labels.masked_fill(~valid, 0).unsqueeze(-1)
            target_logits = logits_f.gather(-1, safe_labels).squeeze(-1)
            logsumexp = torch.logsumexp(logits_f, dim=-1)
            nll = logsumexp - target_logits
            pred_conf = (logits_f.max(dim=-1).values - logsumexp).exp()
            target_prob = (-nll).exp()
            add_mean_stat(f"{prefix}/nll", nll, valid)
            add_mean_stat(f"{prefix}/pred_conf", pred_conf, valid)
            add_mean_stat(f"{prefix}/target_prob", target_prob, valid)

        for part_idx, part in enumerate(self.part_order):
            mask = valid & part_ids.eq(part_idx)
            part_total = int(mask.sum().item())
            if part_total:
                payload[f"{prefix}/{part}_acc"] = payload.get(f"{prefix}/{part}_acc", 0.0) + int(
                    (correct & mask).sum().item()
                )
                payload[f"{prefix}/{part}_total"] = payload.get(
                    f"{prefix}/{part}_total", 0.0
                ) + part_total
                if nll is not None and pred_conf is not None and target_prob is not None:
                    add_mean_stat(f"{prefix}/{part}_nll", nll, mask)
                    add_mean_stat(f"{prefix}/{part}_pred_conf", pred_conf, mask)
                    add_mean_stat(f"{prefix}/{part}_target_prob", target_prob, mask)

        for q_idx in range(self.num_quantizers_per_part):
            mask = valid & quantizer_ids.eq(q_idx)
            q_total = int(mask.sum().item())
            if q_total:
                payload[f"{prefix}/q{q_idx}_acc"] = payload.get(
                    f"{prefix}/q{q_idx}_acc", 0.0
                ) + int((correct & mask).sum().item())
                payload[f"{prefix}/q{q_idx}_total"] = payload.get(
                    f"{prefix}/q{q_idx}_total", 0.0
                ) + q_total
                if nll is not None and pred_conf is not None and target_prob is not None:
                    add_mean_stat(f"{prefix}/q{q_idx}_nll", nll, mask)
                    add_mean_stat(f"{prefix}/q{q_idx}_pred_conf", pred_conf, mask)
                    add_mean_stat(f"{prefix}/q{q_idx}_target_prob", target_prob, mask)

        if topk is not None:
            labels_expanded = labels.unsqueeze(-1)
            for k in (5, 10):
                k = min(k, topk.size(-1))
                hit = topk[..., :k].eq(labels_expanded).any(dim=-1) & valid
                payload[f"{prefix}/top{k}_acc"] = payload.get(
                    f"{prefix}/top{k}_acc", 0.0
                ) + int(hit.sum().item())
                payload[f"{prefix}/top{k}_total"] = payload.get(
                    f"{prefix}/top{k}_total", 0.0
                ) + total

    @staticmethod
    def _finalize_payload(payload: Dict[str, float]) -> Dict[str, float]:
        final: Dict[str, float] = {}
        for key, value in payload.items():
            if key.endswith("_acc"):
                total_key = key[:-4] + "_total"
                total = payload.get(total_key, 0.0)
                if total > 0:
                    final[key] = float(value) / float(total)
            elif key.endswith("_sum"):
                metric_key = key[:-4]
                count = payload.get(f"{metric_key}_count", 0.0)
                if count > 0:
                    final[metric_key] = float(value) / float(count)
        return final

    @staticmethod
    def _log(payload: Dict[str, float], step: int) -> None:
        if not payload:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(payload, step=step)
        except Exception:
            pass

        preview_items = list(payload.items())[:8]
        preview = ", ".join(f"{key}={value:.4f}" for key, value in preview_items)
        print(f"[mask eval metrics] step={step} {preview}")

    def _teacher_forced_metrics(self, raw_model: AudioMotionTransformer, device: torch.device) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if not self.teacher_forced_indices:
            return payload

        for start in range(0, len(self.teacher_forced_indices), self.teacher_forced_batch_size):
            indices = self.teacher_forced_indices[start : start + self.teacher_forced_batch_size]
            examples = [self.eval_dataset[i] for i in indices]
            batch = move_batch_to_device(self.collator(examples), device)
            logits = raw_model(
                input_ids=batch["input_ids"],
                audio_features=batch["audio_features"],
            )
            preds = logits.argmax(dim=-1)
            topk = logits.topk(k=min(10, logits.size(-1)), dim=-1).indices
            self._accumulate_accuracy(
                payload,
                "eval_teacher",
                preds,
                batch["labels"],
                logits=logits,
                topk=topk,
            )
        return self._finalize_payload(payload)

    def _generation_metrics(self, raw_model: AudioMotionTransformer, device: torch.device) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if not self.generation_indices:
            return payload

        for start in range(0, len(self.generation_indices), self.generation_batch_size):
            indices = self.generation_indices[start : start + self.generation_batch_size]
            examples = [self.eval_dataset[i] for i in indices]
            batch = move_batch_to_device(self.collator(examples), device)
            output = raw_model.generate_sbs(
                batch["input_ids"],
                batch["audio_features"],
                generate_steps=self.generate_steps,
            )
            self._accumulate_accuracy(payload, "eval_gen", output, batch["labels"])
        return self._finalize_payload(payload)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        del control, kwargs
        is_world_process_zero = (
            not hasattr(state, "is_world_process_zero") or state.is_world_process_zero
        )
        if model is None or not is_world_process_zero:
            return

        self.eval_count += 1
        raw_model = model.module if hasattr(model, "module") else model
        device = next(raw_model.parameters()).device
        was_training = raw_model.training
        raw_model.eval()

        with torch.no_grad():
            payload: Dict[str, float] = {}
            if (
                self.teacher_forced_indices
                and self.eval_count % self.teacher_forced_every_n_evals == 0
            ):
                payload.update(self._teacher_forced_metrics(raw_model, device))
            if (
                self.generation_indices
                and self.eval_count % self.generation_every_n_evals == 0
            ):
                payload.update(self._generation_metrics(raw_model, device))

        if was_training:
            raw_model.train()
        self._log(payload, int(state.global_step))


class AudioMotionMaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        outputs = model(**inputs)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def configure_wandb(args: argparse.Namespace, default_run_name: str) -> Optional[str]:
    if args.report_to != "wandb":
        return None

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested with --report_to wandb, but wandb is not installed."
        ) from exc

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    return args.wandb_run_name or default_run_name


def run_dry_run(
    *,
    model: AudioMotionTransformer,
    dataset: AudioMotionMaskDataset,
    collator: AudioMotionMaskCollator,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.to(device).train()
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > batches:
            break
        batch = move_batch_to_device(batch, device)
        loss, _, acc = model(**batch)
        print(
            f"dry_run batch {batch_idx}/{batches} "
            f"loss={float(loss.item()):.5f} acc={float(acc.item()):.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original bidirectional Step 2 mask transformer on multipart RVQ tokens."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs"),
    )
    parser.add_argument("--motion_token_dir", type=str, default=None)
    parser.add_argument("--audio_feat_dir", type=str, default=None)
    parser.add_argument("--train_split_file", type=str, default=None)
    parser.add_argument("--eval_split_file", type=str, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_eval_clips", type=int, default=None)
    parser.add_argument("--max_windows_per_sequence", type=int, default=None)

    parser.add_argument("--step", type=int, default=4)
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Stride over token frames. Omit to match the old non-overlap step stride.",
    )
    parser.add_argument("--audio_fps", type=float, default=10.0)
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--motion_token_fps", type=float, default=None)
    parser.add_argument("--motion_token_unit_length", type=float, default=None)

    parser.add_argument("--part_order", type=str, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--num_quantizers_per_part", type=int, default=None)
    parser.add_argument("--num_tokens_per_frame", type=int, default=None)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cond_drop_prob", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=100.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--dry_run_batches", type=int, default=0)

    parser.add_argument(
        "--eval_metric_examples",
        type=int,
        default=1024,
        help="Teacher-forced eval windows for token/top-k/part accuracy.",
    )
    parser.add_argument("--eval_metric_batch_size", type=int, default=256)
    parser.add_argument("--eval_metric_every_n_evals", type=int, default=1)
    parser.add_argument(
        "--eval_gen_metric_examples",
        type=int,
        default=0,
        help="Greedy generate_sbs eval windows. More expensive than teacher-forced.",
    )
    parser.add_argument("--eval_gen_metric_batch_size", type=int, default=128)
    parser.add_argument("--eval_gen_metric_every_n_evals", type=int, default=1)
    parser.add_argument("--generate_steps", type=int, default=1)

    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--profile_startup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    set_seed(args.seed)

    if args.step < 2:
        raise ValueError("--step must be >= 2")
    if args.audio_fps <= 0 or args.motion_fps <= 0:
        raise ValueError("--audio_fps and --motion_fps must be > 0")

    data_dir = Path(args.data_dir)
    motion_token_dir = Path(
        args.motion_token_dir or data_dir / "motion_token_data_multipart_512x4"
    )
    audio_feat_dir = Path(
        args.audio_feat_dir
        or data_dir / f"audio_features_hubert_layer9_fps{format_fps_for_dir(args.audio_fps)}"
    )
    manifest = load_manifest(motion_token_dir)
    part_order = parse_part_order(args.part_order, manifest)
    num_quantizers_per_part = int(
        args.num_quantizers_per_part
        or (manifest.get("num_quantizers") if manifest else None)
        or 4
    )
    num_tokens_per_frame = int(
        args.num_tokens_per_frame
        or (manifest.get("tokens_per_frame") if manifest else None)
        or (len(part_order) * num_quantizers_per_part)
    )
    expected_tokens_per_frame = len(part_order) * num_quantizers_per_part
    if num_tokens_per_frame != expected_tokens_per_frame:
        raise ValueError(
            "--num_tokens_per_frame must equal len(part_order) * --num_quantizers_per_part. "
            f"Got {num_tokens_per_frame} vs {expected_tokens_per_frame}."
        )
    codebook_size = int(
        args.codebook_size or (manifest.get("codebook_size") if manifest else None) or 512
    )
    vocab_size = codebook_size * num_tokens_per_frame + 1
    mask_token_id = vocab_size - 1
    num_frames = args.step + 1
    seq_len = num_frames * num_tokens_per_frame
    if seq_len > args.max_position_embeddings:
        raise ValueError(
            f"Sequence length {seq_len} exceeds --max_position_embeddings {args.max_position_embeddings}"
        )

    train_split_names = read_split_file(Path(args.train_split_file) if args.train_split_file else None)
    eval_split_names = read_split_file(Path(args.eval_split_file) if args.eval_split_file else None)

    train_names = discover_names(motion_token_dir, audio_feat_dir, train_split_names)
    if eval_split_names is not None:
        eval_names = discover_names(motion_token_dir, audio_feat_dir, eval_split_names)
    elif args.eval_ratio > 0 and len(train_names) > 1:
        rng = random.Random(args.seed)
        shuffled = train_names[:]
        rng.shuffle(shuffled)
        eval_size = max(1, int(len(shuffled) * args.eval_ratio))
        eval_names = sorted(shuffled[:eval_size])
        train_names = sorted(shuffled[eval_size:])
    else:
        eval_names = []

    train_sequences, train_stats = load_sequences(
        train_names,
        motion_token_dir,
        audio_feat_dir,
        codebook_size=codebook_size,
        num_tokens_per_frame=num_tokens_per_frame,
        audio_fps=args.audio_fps,
        source_motion_fps_fallback=args.motion_fps,
        motion_token_fps_override=args.motion_token_fps,
        motion_token_unit_length_override=args.motion_token_unit_length,
        max_sequences=args.max_train_clips,
    )
    eval_sequences, eval_stats = load_sequences(
        eval_names,
        motion_token_dir,
        audio_feat_dir,
        codebook_size=codebook_size,
        num_tokens_per_frame=num_tokens_per_frame,
        audio_fps=args.audio_fps,
        source_motion_fps_fallback=args.motion_fps,
        motion_token_fps_override=args.motion_token_fps,
        motion_token_unit_length_override=args.motion_token_unit_length,
        max_sequences=args.max_eval_clips,
    )

    train_dataset = AudioMotionMaskDataset(
        train_sequences,
        step=args.step,
        window_stride=args.window_stride,
        max_windows_per_sequence=args.max_windows_per_sequence,
        seed=args.seed,
    )
    eval_dataset = None
    if eval_sequences:
        eval_dataset = AudioMotionMaskDataset(
            eval_sequences,
            step=args.step,
            window_stride=args.window_stride,
            max_windows_per_sequence=args.max_windows_per_sequence,
            seed=args.seed + 1,
        )

    if len(train_dataset) == 0:
        raise ValueError("No train windows found. Check token/audio dirs and split files.")

    token_layout = (
        manifest.get("token_layout")
        if manifest and manifest.get("token_layout")
        else build_token_layout(part_order, num_quantizers_per_part)
    )

    config = AudioMotionConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        vocab_size=vocab_size,
        codebook_size=codebook_size,
        audio_feat_dim=args.audio_feat_dim,
        num_tokens_per_frame=num_tokens_per_frame,
        num_frames=num_frames,
        dropout=args.dropout,
        cond_drop_prob=args.cond_drop_prob,
        constrain_token_logits=True,
    )
    config.part_order = part_order
    config.num_quantizers_per_part = num_quantizers_per_part
    config.token_layout = token_layout
    config.mask_token_id = mask_token_id
    config.step = args.step
    model = AudioMotionTransformer(config)

    total_params, trainable_params = count_parameters(model)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_train_batch = (
        int(args.per_device_train_batch_size)
        * max(1, world_size)
        * int(args.gradient_accumulation_steps)
    )
    steps_per_epoch = math.ceil(len(train_dataset) / max(1, effective_train_batch))
    total_optimizer_steps = math.ceil(float(args.num_train_epochs) * steps_per_epoch)
    window_stride = int(args.window_stride or args.step)
    default_run_name = (
        f"mask_multipart_step{args.step}_"
        f"{codebook_size}x{num_quantizers_per_part}x{len(part_order)}_"
        f"bs{effective_train_batch}"
    )
    wandb_run_name = configure_wandb(args, default_run_name)

    print("=" * 72)
    print("Step 2 bidirectional AudioMotion mask training")
    print(f"Output dir:       {args.output_dir}")
    print(f"Motion tokens:    {motion_token_dir}")
    print(f"Audio features:   {audio_feat_dir}")
    print(f"Train clips:      {len(train_sequences)} loaded / {train_stats['requested']} requested")
    print(f"Train windows:    {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval clips:       {len(eval_sequences)} loaded / {eval_stats['requested']} requested")
        print(f"Eval windows:     {len(eval_dataset)}")
    print(f"Step/window:      step={args.step}, frames={num_frames}, stride={window_stride}")
    print(f"Token layout:     parts={part_order}, q/part={num_quantizers_per_part}")
    print(f"Tokens/frame:     {num_tokens_per_frame}, seq_len={seq_len}")
    print(f"Vocab:            {vocab_size} ({codebook_size} codes x {num_tokens_per_frame} slots + mask {mask_token_id})")
    print(f"Slot constraint:  enabled")
    print(f"Architecture:     L={args.num_layers}, H={args.hidden_size}, heads={args.num_heads}, ffn={args.intermediate_size}")
    print(f"Parameters:       {total_params:,} total / {trainable_params:,} trainable")
    print(f"Batch:            per_device={args.per_device_train_batch_size}, world={world_size}, grad_accum={args.gradient_accumulation_steps}, effective={effective_train_batch}")
    print(f"Steps/epoch:      {steps_per_epoch}")
    print(f"Total steps:      {total_optimizer_steps}")
    print(f"LR scheduler:     {args.lr_scheduler_type}, lr={args.learning_rate:g}, warmup_ratio={args.warmup_ratio:g}")
    print(f"Gradient ckpt:    {bool(args.gradient_checkpointing)}")
    print(f"Report to:        {args.report_to}")
    if wandb_run_name:
        print(f"W&B run:          {wandb_run_name}")
    if eval_dataset is not None:
        print(
            "Eval metrics:     "
            f"teacher={args.eval_metric_examples}, gen={args.eval_gen_metric_examples}, "
            f"every {args.eval_metric_every_n_evals}/{args.eval_gen_metric_every_n_evals} eval(s)"
        )
    print("=" * 72)

    collator = AudioMotionMaskCollator(config)
    if args.dry_run_batches > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_dry_run(
            model=model,
            dataset=train_dataset,
            collator=collator,
            batches=args.dry_run_batches,
            batch_size=args.per_device_train_batch_size,
            device=device,
        )
        return

    training_args_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_grad_norm": args.max_grad_norm,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "report_to": args.report_to,
        "dataloader_num_workers": args.dataloader_num_workers,
    }
    if wandb_run_name:
        training_args_kwargs["run_name"] = wandb_run_name
    if eval_dataset is not None and len(eval_dataset) > 0:
        training_args_kwargs.update({"eval_strategy": "steps", "eval_steps": args.eval_steps})
    else:
        training_args_kwargs["eval_strategy"] = "no"

    with timed_stage("create TrainingArguments", args.profile_startup):
        training_args = TrainingArguments(**training_args_kwargs)

    callbacks: List[TrainerCallback] = []
    if eval_dataset is not None and len(eval_dataset) > 0 and (
        args.eval_metric_examples > 0 or args.eval_gen_metric_examples > 0
    ):
        callbacks.append(
            AudioMotionMaskMetricsCallback(
                eval_dataset=eval_dataset,
                collator=collator,
                part_order=part_order,
                num_quantizers_per_part=num_quantizers_per_part,
                teacher_forced_examples=args.eval_metric_examples,
                teacher_forced_batch_size=args.eval_metric_batch_size,
                teacher_forced_every_n_evals=args.eval_metric_every_n_evals,
                generation_examples=args.eval_gen_metric_examples,
                generation_batch_size=args.eval_gen_metric_batch_size,
                generation_every_n_evals=args.eval_gen_metric_every_n_evals,
                generate_steps=args.generate_steps,
            )
        )

    with timed_stage("create Trainer", args.profile_startup):
        trainer = AudioMotionMaskTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )

    with timed_stage("trainer.train", True):
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    with timed_stage("save model/layout", True):
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero():
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "multipart_token_layout.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "codebook_size": codebook_size,
                        "num_tokens_per_frame": num_tokens_per_frame,
                        "num_quantizers_per_part": num_quantizers_per_part,
                        "part_order": part_order,
                        "token_layout": token_layout,
                        "mask_token_id": mask_token_id,
                    },
                    f,
                    indent=2,
                )
            with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_clips": len(train_sequences),
                        "train_windows": len(train_dataset),
                        "eval_clips": len(eval_sequences),
                        "eval_windows": len(eval_dataset) if eval_dataset is not None else 0,
                        "effective_train_batch": effective_train_batch,
                        "steps_per_epoch": steps_per_epoch,
                        "total_optimizer_steps": total_optimizer_steps,
                        "window_stride": window_stride,
                        "lr_scheduler_type": args.lr_scheduler_type,
                    },
                    f,
                    indent=2,
                )

    print(f"Saved bidirectional multipart Step 2 checkpoint to: {args.output_dir}")
    print(f"[Timing] total script runtime: {time.perf_counter() - run_start:.3f}s")


if __name__ == "__main__":
    main()

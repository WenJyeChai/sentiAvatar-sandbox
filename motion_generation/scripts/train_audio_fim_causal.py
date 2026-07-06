#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Train the compact Step 2 audio-aware causal FIM transformer.

This is the small-model sibling of train_vllm_infill.py. It does not use the
Qwen tokenizer or vLLM checkpoint. It trains AudioFIMCausalLM from scratch with:

    motion tokens: compact RVQ ids, old Step 2 style
    audio:         continuous HuBERT layer9 features
    objective:     causal FIM, predict the middle motion frames only

First target:
    classic gap with step=4:
        left frame t, right frame t+4, predict frames t+1, t+2, t+3.

The explicit [LEN_N] path and --step argument are kept so variable gaps can be
enabled later without changing the checkpoint format.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed


THIS_DIR = Path(__file__).resolve().parent
MOTION_GENERATION_DIR = THIS_DIR.parent
PROJECT_DIR = MOTION_GENERATION_DIR.parent
sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_fim_causal_model import (  # noqa: E402
    AudioFIMCausalCollator,
    AudioFIMCausalConfig,
    AudioFIMCausalDataset,
    AudioFIMCausalLM,
    AudioFIMTokenMapper,
    IGNORE_INDEX,
)
from configs.default_config import Config  # noqa: E402
from models.rvqvae import RVQVAE  # noqa: E402
from utils.constants import (  # noqa: E402
    BODY_JOINTS_ID,
    LEFT_HAND_JOINTS_ID,
    RIGHT_HAND_JOINTS_ID,
)
from utils.rotation_utils import sixd_to_quaternion  # noqa: E402


def format_fps_for_dir(fps: float) -> str:
    if float(fps).is_integer():
        return str(int(fps))
    return str(fps).replace(".", "p")


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


def read_split_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    normalized = []
    for name in names:
        name = name.replace("\\", "/").strip().strip("/")
        suffix = Path(name).suffix
        if suffix in {".wav", ".npy", ".json"}:
            name = name[: -len(suffix)]
        normalized.append(name)
    return normalized


def load_token_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"tokens": data}
    raise ValueError(f"Unsupported token JSON format: {path}")


def extract_action_text(raw_text: str) -> Optional[str]:
    tags = re.findall(r"ã€(.+?)ã€‘", raw_text)
    if not tags:
        return None

    last_tag = tags[-1]
    if last_tag == "åŠ¨ä½œï¼šæ— åŠ¨ä½œ":
        for tag in tags:
            if tag.startswith("è¡¨æƒ…ï¼š") and tag != "è¡¨æƒ…ï¼šæ— è¡¨æƒ…":
                expression = tag.replace("è¡¨æƒ…ï¼š", "")
                return expression if "åŠ¨ä½œ" in expression else f"åŠ¨ä½œï¼š{expression}"
    return last_tag


def load_action_text_map(path: Optional[str]) -> Dict[str, str]:
    # This compact model does not consume text yet. We still parse the map so
    # sequence metadata stays parallel with train_vllm_infill.py for later use.
    if path is None or not Path(path).exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        motion2text = json.load(f)

    result: Dict[str, str] = {}
    for name, raw_text in motion2text.items():
        action_text = extract_action_text(raw_text)
        if not action_text:
            continue
        normalized = name.replace("\\", "/").strip().strip("/")
        suffix = Path(normalized).suffix
        if suffix in {".wav", ".npy", ".json"}:
            normalized = normalized[: -len(suffix)]
        result[normalized] = action_text
    return result


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
        }
        audio_names = {
            path.relative_to(audio_feat_dir).with_suffix("").as_posix()
            for path in audio_feat_dir.rglob("*.npy")
        }
        names = sorted(motion_names & audio_names)

    available = []
    for name in names:
        if (motion_token_dir / f"{name}.json").exists() and (
            audio_feat_dir / f"{name}.npy"
        ).exists():
            available.append(name)
    return available


def load_sequences(
    names: Sequence[str],
    motion_token_dir: Path,
    audio_feat_dir: Path,
    action_text_map: Dict[str, str],
    *,
    max_samples: Optional[int] = None,
    audio_fps: float = 10.0,
    motion_fps: float = 20.0,
    motion_token_fps: Optional[float] = None,
    motion_token_unit_length: float = 2.0,
) -> List[Dict[str, Any]]:
    sequences: List[Dict[str, Any]] = []
    if motion_token_unit_length <= 0:
        raise ValueError("motion_token_unit_length must be > 0")
    if motion_token_fps is not None and motion_token_fps <= 0:
        raise ValueError("motion_token_fps must be > 0")

    for name in names:
        if max_samples is not None and len(sequences) >= max_samples:
            break

        motion_payload = load_token_json(motion_token_dir / f"{name}.json")
        if not motion_payload:
            continue

        motion_tokens = motion_payload.get("tokens")
        if not motion_tokens:
            continue

        audio_path = audio_feat_dir / f"{name}.npy"
        if not audio_path.exists():
            continue

        audio_features = np.load(audio_path).astype(np.float32)
        if audio_features.ndim != 2 or audio_features.shape[0] == 0:
            continue

        source_motion_fps = float(motion_payload.get("fps") or motion_fps)
        if source_motion_fps <= 0:
            continue

        token_motion_fps = (
            float(motion_token_fps)
            if motion_token_fps is not None
            else source_motion_fps / float(motion_token_unit_length)
        )

        sequences.append(
            {
                "name": name,
                "motion_tokens": motion_tokens,
                "audio_features": audio_features,
                "source_motion_fps": source_motion_fps,
                "motion_token_unit_length": float(motion_token_unit_length),
                "motion_token_fps": token_motion_fps,
                "motion_fps": token_motion_fps,
                "audio_fps": audio_fps,
                "action_text": action_text_map.get(name),
            }
        )

    return sequences


def split_train_eval(
    sequences: List[Dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if eval_ratio <= 0 or len(sequences) < 2:
        return sequences, None

    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[eval_size:], shuffled[:eval_size]


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


ARCHITECTURE_PRESETS: Dict[str, Dict[str, int]] = {
    "base": {
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 16,
        "intermediate_size": 1536,
    },
    # Cleanest half-size ablation: keep width/head dimension identical and
    # halve only the transformer stack depth.
    "half_depth": {
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 16,
        "intermediate_size": 1536,
    },
    # Roughly half-ish parameters while preserving the 8-layer depth.
    "half_params": {
        "hidden_size": 384,
        "num_layers": 8,
        "num_heads": 12,
        "intermediate_size": 1152,
    },
    # Half the hidden width. This is a more aggressive small-model ablation.
    "half_width": {
        "hidden_size": 256,
        "num_layers": 8,
        "num_heads": 8,
        "intermediate_size": 768,
    },
    # Bigger-model ablations.
    "double_depth": {
        "hidden_size": 512,
        "num_layers": 16,
        "num_heads": 16,
        "intermediate_size": 1536,
    },
    # Closest to 2x parameters for this causal implementation.
    "double_params": {
        "hidden_size": 736,
        "num_layers": 8,
        "num_heads": 16,
        "intermediate_size": 2208,
    },
    # Rounder wide setting, slightly above 2x.
    "double_width": {
        "hidden_size": 768,
        "num_layers": 8,
        "num_heads": 16,
        "intermediate_size": 2304,
    },
}


def apply_architecture_preset(args: argparse.Namespace) -> Optional[str]:
    preset_name = getattr(args, "architecture_preset", "custom")
    if preset_name == "custom":
        return None

    preset = ARCHITECTURE_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(args, key, int(value))
    return preset_name


def configure_wandb(args: argparse.Namespace, default_run_name: str) -> Optional[str]:
    """Configure optional Weights & Biases logging for HuggingFace Trainer."""

    if args.report_to != "wandb":
        return None

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested with --report_to wandb, but wandb is not "
            "installed. Install wandb or run with --report_to none."
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


@dataclass(frozen=True)
class SelfForcingPhase:
    start_value: float
    start_is_percent: bool
    self_force_prob: float
    prefix_choices: Tuple[int, ...]


def parse_self_forcing_schedule(schedule: str) -> List[SelfForcingPhase]:
    """
    Parse a step-based self-forcing schedule.

    Format:
        start_step:self_force_prob:prefix_choices;...

    Example:
        0:0.0:;5000:0.25:0,4;15000:0.5:0,4,8
        0%:0.0:;20%:0.25:0,4;50%:0.5:0,4,8

    `prefix_choices` are generated-prefix lengths measured in compact RVQ
    motion tokens. With 4 RVQ tokens per frame, prefix 4 means one generated
    middle frame is used as context before the supervised suffix.
    """

    if schedule is None:
        schedule = ""
    schedule = schedule.strip()
    if schedule.lower() in {"", "off", "none", "teacher"}:
        return [SelfForcingPhase(0.0, False, 0.0, ())]

    phases: List[SelfForcingPhase] = []
    for raw_phase in schedule.split(";"):
        raw_phase = raw_phase.strip()
        if not raw_phase:
            continue
        parts = raw_phase.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(
                "Invalid --self_forcing_schedule phase "
                f"{raw_phase!r}. Expected start:prob:prefixes."
            )

        raw_start = parts[0].strip()
        if raw_start.endswith("%"):
            start_is_percent = True
            start_value = float(raw_start[:-1].strip()) / 100.0
            if not 0.0 <= start_value <= 1.0:
                raise ValueError(
                    "Self-forcing percentage phase starts must be in [0%, 100%]"
                )
        else:
            start_is_percent = False
            start_value = float(int(raw_start))
            if start_value < 0:
                raise ValueError("Self-forcing phase start_step must be >= 0")

        prob = float(parts[1])
        if not 0.0 <= prob <= 1.0:
            raise ValueError("Self-forcing probability must be in [0, 1]")

        choices: Tuple[int, ...] = ()
        if len(parts) == 3 and parts[2].strip():
            parsed_choices = []
            for value in parts[2].split(","):
                value = value.strip()
                if not value:
                    continue
                prefix_len = int(value)
                if prefix_len < 0:
                    raise ValueError("Self-forcing prefix choices must be >= 0")
                parsed_choices.append(prefix_len)
            choices = tuple(sorted(set(parsed_choices)))

        phases.append(SelfForcingPhase(start_value, start_is_percent, prob, choices))

    if not phases:
        return [SelfForcingPhase(0.0, False, 0.0, ())]

    seen_starts = set()
    for phase in phases:
        start_key = (phase.start_is_percent, phase.start_value)
        if start_key in seen_starts:
            raise ValueError(
                "Duplicate self-forcing phase start "
                f"{format_self_forcing_phase_start(phase)}"
            )
        seen_starts.add(start_key)

    has_zero_start = any(
        (not phase.start_is_percent and phase.start_value == 0)
        or (phase.start_is_percent and phase.start_value == 0.0)
        for phase in phases
    )
    if not has_zero_start:
        phases.insert(0, SelfForcingPhase(0.0, False, 0.0, ()))

    return phases


def format_self_forcing_phase_start(phase: SelfForcingPhase) -> str:
    if phase.start_is_percent:
        percent = phase.start_value * 100.0
        if float(percent).is_integer():
            return f"{int(percent)}%"
        return f"{percent:g}%"
    return str(int(phase.start_value))


def format_self_forcing_schedule(phases: Sequence[SelfForcingPhase]) -> str:
    chunks = []
    for phase in phases:
        choices = ",".join(str(choice) for choice in phase.prefix_choices)
        chunks.append(
            f"{format_self_forcing_phase_start(phase)}:"
            f"{phase.self_force_prob:g}:{choices}"
        )
    return ";".join(chunks)


@dataclass(frozen=True)
class RVQCurriculumPhase:
    start_value: float
    start_is_percent: bool
    active_quantizers: int


def parse_rvq_curriculum_schedule(schedule: str) -> List[RVQCurriculumPhase]:
    """
    Parse an RVQ coarse-to-fine schedule.

    Format:
        start_step:active_quantizers;...

    Example:
        0%:1;20%:2;45%:3;70%:4

    `active_quantizers` means how many RVQ levels contribute to the training
    loss. For the default 4-level RVQ, 1 supervises only res_1 and 4 restores
    the normal full loss.
    """

    if schedule is None:
        schedule = ""
    schedule = schedule.strip()
    if schedule.lower() in {"", "off", "none", "full"}:
        return []

    phases: List[RVQCurriculumPhase] = []
    for raw_phase in schedule.split(";"):
        raw_phase = raw_phase.strip()
        if not raw_phase:
            continue
        parts = raw_phase.split(":")
        if len(parts) != 2:
            raise ValueError(
                "Invalid --rvq_curriculum_schedule phase "
                f"{raw_phase!r}. Expected start:active_quantizers."
            )

        raw_start = parts[0].strip()
        if raw_start.endswith("%"):
            start_is_percent = True
            start_value = float(raw_start[:-1].strip()) / 100.0
            if not 0.0 <= start_value <= 1.0:
                raise ValueError(
                    "RVQ curriculum percentage phase starts must be in [0%, 100%]"
                )
        else:
            start_is_percent = False
            start_value = float(int(raw_start))
            if start_value < 0:
                raise ValueError("RVQ curriculum phase start_step must be >= 0")

        active_quantizers = int(parts[1].strip())
        if active_quantizers < 1:
            raise ValueError("RVQ curriculum active_quantizers must be >= 1")

        phases.append(
            RVQCurriculumPhase(start_value, start_is_percent, active_quantizers)
        )

    if not phases:
        return []

    seen_starts = set()
    for phase in phases:
        start_key = (phase.start_is_percent, phase.start_value)
        if start_key in seen_starts:
            raise ValueError(
                "Duplicate RVQ curriculum phase start "
                f"{format_rvq_curriculum_phase_start(phase)}"
            )
        seen_starts.add(start_key)

    has_zero_start = any(
        (not phase.start_is_percent and phase.start_value == 0)
        or (phase.start_is_percent and phase.start_value == 0.0)
        for phase in phases
    )
    if not has_zero_start:
        first = phases[0]
        phases.insert(
            0,
            RVQCurriculumPhase(0.0, False, first.active_quantizers),
        )

    return phases


def format_rvq_curriculum_phase_start(phase: RVQCurriculumPhase) -> str:
    if phase.start_is_percent:
        percent = phase.start_value * 100.0
        if float(percent).is_integer():
            return f"{int(percent)}%"
        return f"{percent:g}%"
    return str(int(phase.start_value))


def format_rvq_curriculum_schedule(phases: Sequence[RVQCurriculumPhase]) -> str:
    if not phases:
        return "off/full"

    chunks = []
    for phase in phases:
        chunks.append(
            f"{format_rvq_curriculum_phase_start(phase)}:"
            f"{phase.active_quantizers}"
        )
    return ";".join(chunks)


class AudioFIMSelfForcingTrainer(Trainer):
    """Trainer with optional scheduled self-forcing and RVQ loss curriculum."""

    def __init__(
        self,
        *args,
        self_forcing_schedule: Optional[Sequence[SelfForcingPhase]] = None,
        rvq_curriculum_schedule: Optional[Sequence[RVQCurriculumPhase]] = None,
        rvq_curriculum_mask_inactive_inputs: bool = False,
        audio_condition_dropout_prob: float = 0.0,
        motion_history_dropout_prob: float = 0.0,
        motion_anchor_token_dropout_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_forcing_schedule = list(
            self_forcing_schedule or [SelfForcingPhase(0.0, False, 0.0, ())]
        )
        self.rvq_curriculum_schedule = list(rvq_curriculum_schedule or [])
        self.rvq_curriculum_mask_inactive_inputs = bool(
            rvq_curriculum_mask_inactive_inputs
        )
        self.audio_condition_dropout_prob = float(audio_condition_dropout_prob)
        self.motion_history_dropout_prob = float(motion_history_dropout_prob)
        self.motion_anchor_token_dropout_prob = float(motion_anchor_token_dropout_prob)
        self._last_forcing_state: Dict[str, float] = {
            "active": 0.0,
            "prefix_len": -1.0,
            "phase_prob": 0.0,
            "phase_max_prefix": -1.0,
        }
        self._last_rvq_curriculum_state: Dict[str, float] = {
            "active_quantizers": -1.0,
            "masked_labels": 0.0,
        }
        self._last_conditioning_state: Dict[str, float] = {
            "audio_dropped_examples": 0.0,
            "history_masked_tokens": 0.0,
            "anchor_masked_tokens": 0.0,
        }

    def _phase_for_step(self, step: int) -> SelfForcingPhase:
        current = self.self_forcing_schedule[0]
        current_start = -1
        for phase in self.self_forcing_schedule:
            start_step = self._phase_start_step(phase)
            if int(step) >= start_step and start_step >= current_start:
                current = phase
                current_start = start_step
        return current

    def _total_training_steps(self) -> int:
        total_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if total_steps <= 0:
            total_steps = int(getattr(self.args, "max_steps", 0) or 0)
        return max(0, total_steps)

    def _phase_start_step(self, phase: SelfForcingPhase) -> int:
        if not phase.start_is_percent:
            return int(phase.start_value)

        total_steps = self._total_training_steps()
        if total_steps <= 0:
            return 0 if phase.start_value <= 0 else 10**12
        return int(math.floor(float(total_steps) * float(phase.start_value)))

    def _rvq_phase_for_step(self, step: int) -> Optional[RVQCurriculumPhase]:
        if not self.rvq_curriculum_schedule:
            return None

        current = self.rvq_curriculum_schedule[0]
        current_start = -1
        for phase in self.rvq_curriculum_schedule:
            start_step = self._rvq_phase_start_step(phase)
            if int(step) >= start_step and start_step >= current_start:
                current = phase
                current_start = start_step
        return current

    def _rvq_phase_start_step(self, phase: RVQCurriculumPhase) -> int:
        if not phase.start_is_percent:
            return int(phase.start_value)

        total_steps = self._total_training_steps()
        if total_steps <= 0:
            return 0 if phase.start_value <= 0 else 10**12
        return int(math.floor(float(total_steps) * float(phase.start_value)))

    def _unwrap_config(self, model) -> AudioFIMCausalConfig:
        try:
            base_model = self.accelerator.unwrap_model(model)
        except Exception:
            base_model = getattr(model, "module", model)
        return base_model.config

    def _active_quantizers_for_step(
        self,
        step: int,
        config: AudioFIMCausalConfig,
    ) -> int:
        phase = self._rvq_phase_for_step(step)
        if phase is None:
            return int(config.num_quantizers)

        active_quantizers = int(phase.active_quantizers)
        if active_quantizers < 1 or active_quantizers > int(config.num_quantizers):
            raise ValueError(
                "RVQ curriculum active_quantizers must be in "
                f"1..{config.num_quantizers}, got {active_quantizers}"
            )
        return active_quantizers

    def _apply_rvq_curriculum_to_tensors(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        config: AudioFIMCausalConfig,
        active_quantizers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if int(active_quantizers) >= int(config.num_quantizers):
            return input_ids, labels, 0

        motion_label_mask = (labels >= 0) & (labels < int(config.motion_vocab_size))
        if not bool(motion_label_mask.any().item()):
            return input_ids, labels, 0

        quantizer_ids = torch.div(
            labels.clamp_min(0),
            int(config.codebook_size),
            rounding_mode="floor",
        )
        inactive_mask = motion_label_mask & (
            quantizer_ids >= int(active_quantizers)
        )
        masked_count = int(inactive_mask.sum().item())
        if masked_count == 0:
            return input_ids, labels, 0

        labels = labels.clone()
        labels[inactive_mask] = IGNORE_INDEX

        if self.rvq_curriculum_mask_inactive_inputs:
            input_ids = input_ids.clone()
            input_ids[inactive_mask] = int(config.mask_token_id)

        return input_ids, labels, masked_count

    def _apply_rvq_curriculum_to_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        config: AudioFIMCausalConfig,
        active_quantizers: int,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        input_ids, labels, masked_count = self._apply_rvq_curriculum_to_tensors(
            inputs["input_ids"],
            inputs["labels"],
            config=config,
            active_quantizers=active_quantizers,
        )
        if masked_count == 0:
            return inputs, 0

        masked_inputs = dict(inputs)
        masked_inputs["input_ids"] = input_ids
        masked_inputs["labels"] = labels
        return masked_inputs, masked_count

    def _apply_conditioning_dropout_to_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        config: AudioFIMCausalConfig,
        training: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        stats = {
            "audio_dropped_examples": 0.0,
            "history_masked_tokens": 0.0,
            "anchor_masked_tokens": 0.0,
        }
        if not training:
            return inputs, stats

        output = inputs
        batch_size = int(inputs["input_ids"].shape[0])
        device = inputs["input_ids"].device

        if self.audio_condition_dropout_prob > 0.0:
            drop_mask = (
                torch.rand(batch_size, device=device)
                < float(self.audio_condition_dropout_prob)
            )
            dropped = int(drop_mask.sum().item())
            if dropped:
                if output is inputs:
                    output = dict(inputs)
                audio_features = output["audio_features"].clone()
                audio_features[drop_mask] = 0.0
                output["audio_features"] = audio_features
                stats["audio_dropped_examples"] = float(dropped)

        if (
            self.motion_history_dropout_prob <= 0.0
            and self.motion_anchor_token_dropout_prob <= 0.0
        ):
            return output, stats

        input_ids = output["input_ids"]
        masked_ids = input_ids.clone()
        history_masked = 0
        anchor_masked = 0

        for batch_idx in range(batch_size):
            ids = input_ids[batch_idx]
            real_len = int(inputs["attention_mask"][batch_idx].sum().item())
            real_ids = ids[:real_len]

            history_positions = (
                real_ids == int(config.history_token_id)
            ).nonzero(as_tuple=True)[0]
            mask_positions = (
                real_ids == int(config.mask_token_id)
            ).nonzero(as_tuple=True)[0]
            right_positions = (
                real_ids == int(config.right_anchor_token_id)
            ).nonzero(as_tuple=True)[0]
            if (
                history_positions.numel() == 0
                or mask_positions.numel() == 0
                or right_positions.numel() == 0
            ):
                continue

            first_mask = int(mask_positions[0].item())
            right_anchor_marker = int(right_positions[0].item())
            left_anchor_start = first_mask - int(config.num_quantizers)
            left_anchor_end = first_mask
            right_anchor_start = right_anchor_marker + 1
            right_anchor_end = right_anchor_start + int(config.num_quantizers)

            if self.motion_history_dropout_prob > 0.0:
                if random.random() < float(self.motion_history_dropout_prob):
                    history_start = int(history_positions[0].item()) + 1
                    history_end = max(history_start, left_anchor_start)
                    if history_end > history_start:
                        token_slice = ids[history_start:history_end]
                        motion_mask = (
                            (token_slice >= 0)
                            & (token_slice < int(config.motion_vocab_size))
                        )
                        if bool(motion_mask.any().item()):
                            target = masked_ids[batch_idx, history_start:history_end]
                            target[motion_mask] = int(config.mask_token_id)
                            history_masked += int(motion_mask.sum().item())

            if self.motion_anchor_token_dropout_prob > 0.0:
                for start, end in (
                    (left_anchor_start, left_anchor_end),
                    (right_anchor_start, right_anchor_end),
                ):
                    start = max(0, start)
                    end = min(real_len, end)
                    if end <= start:
                        continue
                    token_slice = ids[start:end]
                    motion_mask = (
                        (token_slice >= 0)
                        & (token_slice < int(config.motion_vocab_size))
                    )
                    if not bool(motion_mask.any().item()):
                        continue
                    drop_mask = (
                        torch.rand(end - start, device=device)
                        < float(self.motion_anchor_token_dropout_prob)
                    )
                    final_mask = motion_mask & drop_mask
                    if bool(final_mask.any().item()):
                        target = masked_ids[batch_idx, start:end]
                        target[final_mask] = int(config.mask_token_id)
                        anchor_masked += int(final_mask.sum().item())

        if history_masked or anchor_masked:
            if output is inputs:
                output = dict(inputs)
            output["input_ids"] = masked_ids
            stats["history_masked_tokens"] = float(history_masked)
            stats["anchor_masked_tokens"] = float(anchor_masked)

        return output, stats

    def _pad_token_sequences(
        self,
        sequences: Sequence[torch.Tensor],
        *,
        pad_value: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(seq.numel()) for seq in sequences)
        batch = torch.full(
            (len(sequences), max_len),
            int(pad_value),
            dtype=dtype,
            device=device,
        )
        attention = torch.zeros(
            (len(sequences), max_len),
            dtype=torch.long,
            device=device,
        )
        for idx, seq in enumerate(sequences):
            length = int(seq.numel())
            batch[idx, :length] = seq.to(device=device, dtype=dtype)
            attention[idx, :length] = 1
        return batch, attention

    def _select_next_ids(
        self,
        logits: torch.Tensor,
        *,
        token_idx: int,
        config: AudioFIMCausalConfig,
    ) -> torch.Tensor:
        quantizer_idx = int(token_idx) % config.num_quantizers
        allowed_start = quantizer_idx * config.codebook_size
        allowed_end = allowed_start + config.codebook_size
        invalid = torch.ones_like(logits, dtype=torch.bool)
        invalid[:, allowed_start:allowed_end] = False
        logits = logits.masked_fill(invalid, float("-inf"))
        return logits.argmax(dim=-1)

    def _compute_self_forcing_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        requested_prefix_len: int,
        active_quantizers: int,
    ):
        config = self._unwrap_config(model)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        audio_frame_ids = inputs["audio_frame_ids"]
        audio_features = inputs["audio_features"]

        device = input_ids.device
        batch_size = int(input_ids.shape[0])
        real_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

        prompt_ids: List[torch.Tensor] = []
        prompt_audio_ids: List[torch.Tensor] = []
        target_ids: List[torch.Tensor] = []
        target_audio_ids: List[torch.Tensor] = []
        motion_target_counts: List[int] = []

        for batch_idx in range(batch_size):
            real_len = int(real_lengths[batch_idx].item())
            item_labels = labels[batch_idx, :real_len]
            supervised = (item_labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if supervised.numel() == 0:
                raise ValueError("Self-forcing batch contains no supervised labels")

            target_start = int(supervised[0].item())
            item_target_ids = item_labels[target_start:real_len]
            item_target_audio_ids = audio_frame_ids[batch_idx, target_start:real_len]

            prompt_ids.append(input_ids[batch_idx, :target_start])
            prompt_audio_ids.append(audio_frame_ids[batch_idx, :target_start])
            target_ids.append(item_target_ids)
            target_audio_ids.append(item_target_audio_ids)
            motion_count = int(
                (
                    (item_target_ids >= 0)
                    & (item_target_ids < config.motion_vocab_size)
                )
                .sum()
                .item()
            )
            motion_target_counts.append(motion_count)

        prefix_len = min(
            int(requested_prefix_len),
            min(motion_target_counts) if motion_target_counts else 0,
        )
        prefix_len = max(0, prefix_len)

        contexts = [tokens.clone() for tokens in prompt_ids]
        context_audio_ids = [tokens.clone() for tokens in prompt_audio_ids]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for token_idx in range(prefix_len):
                    gen_input_ids, gen_attention_mask = self._pad_token_sequences(
                        contexts,
                        pad_value=config.pad_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    gen_audio_frame_ids, _ = self._pad_token_sequences(
                        context_audio_ids,
                        pad_value=-1,
                        dtype=torch.long,
                        device=device,
                    )
                    outputs = model(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        audio_features=audio_features,
                        audio_frame_ids=gen_audio_frame_ids,
                    )
                    lengths = gen_attention_mask.sum(dim=1).to(dtype=torch.long)
                    batch_index = torch.arange(batch_size, device=device)
                    next_logits = outputs.logits[
                        batch_index,
                        lengths - 1,
                        :,
                    ].clone()
                    next_ids = self._select_next_ids(
                        next_logits,
                        token_idx=token_idx,
                        config=config,
                    )

                    for batch_idx in range(batch_size):
                        contexts[batch_idx] = torch.cat(
                            [contexts[batch_idx], next_ids[batch_idx : batch_idx + 1]]
                        )
                        context_audio_ids[batch_idx] = torch.cat(
                            [
                                context_audio_ids[batch_idx],
                                target_audio_ids[batch_idx][
                                    token_idx : token_idx + 1
                                ],
                            ]
                        )
        finally:
            if was_training:
                model.train()

        new_input_ids: List[torch.Tensor] = []
        new_audio_frame_ids: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        for batch_idx in range(batch_size):
            suffix_ids = target_ids[batch_idx][prefix_len:]
            suffix_audio_ids = target_audio_ids[batch_idx][prefix_len:]
            item_input_ids = torch.cat([contexts[batch_idx], suffix_ids])
            item_audio_ids = torch.cat(
                [context_audio_ids[batch_idx], suffix_audio_ids]
            )
            ignored = torch.full(
                (int(contexts[batch_idx].numel()),),
                IGNORE_INDEX,
                dtype=torch.long,
                device=device,
            )
            item_labels = torch.cat([ignored, suffix_ids])
            new_input_ids.append(item_input_ids)
            new_audio_frame_ids.append(item_audio_ids)
            new_labels.append(item_labels)

        train_input_ids, train_attention_mask = self._pad_token_sequences(
            new_input_ids,
            pad_value=config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        train_audio_frame_ids, _ = self._pad_token_sequences(
            new_audio_frame_ids,
            pad_value=-1,
            dtype=torch.long,
            device=device,
        )
        train_labels, _ = self._pad_token_sequences(
            new_labels,
            pad_value=IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )
        train_input_ids, train_labels, masked_labels = (
            self._apply_rvq_curriculum_to_tensors(
                train_input_ids,
                train_labels,
                config=config,
                active_quantizers=active_quantizers,
            )
        )
        outputs = model(
            input_ids=train_input_ids,
            attention_mask=train_attention_mask,
            labels=train_labels,
            audio_features=audio_features,
            audio_frame_ids=train_audio_frame_ids,
        )
        return outputs.loss, outputs, prefix_len, masked_labels

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
        **kwargs,
    ):
        del num_items_in_batch
        del kwargs
        global_step = int(self.state.global_step)
        config = self._unwrap_config(model)
        loss_inputs, conditioning_stats = self._apply_conditioning_dropout_to_inputs(
            inputs,
            config=config,
            training=bool(model.training),
        )
        active_quantizers = self._active_quantizers_for_step(global_step, config)
        phase = self._phase_for_step(global_step)
        max_prefix = max(phase.prefix_choices) if phase.prefix_choices else -1

        use_self_forcing = (
            bool(model.training)
            and phase.self_force_prob > 0.0
            and bool(phase.prefix_choices)
            and random.random() < phase.self_force_prob
        )
        if not use_self_forcing:
            active_for_loss = (
                active_quantizers if bool(model.training) else int(config.num_quantizers)
            )
            loss_inputs, masked_labels = self._apply_rvq_curriculum_to_inputs(
                loss_inputs,
                config=config,
                active_quantizers=active_for_loss,
            )
            outputs = model(**loss_inputs)
            self._last_forcing_state = {
                "active": 0.0,
                "prefix_len": -1.0,
                "phase_prob": float(phase.self_force_prob),
                "phase_max_prefix": float(max_prefix),
            }
            self._last_rvq_curriculum_state = {
                "active_quantizers": float(active_for_loss),
                "masked_labels": float(masked_labels),
            }
            self._last_conditioning_state = conditioning_stats
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        requested_prefix_len = random.choice(list(phase.prefix_choices))
        loss, outputs, actual_prefix_len, masked_labels = self._compute_self_forcing_loss(
            model,
            loss_inputs,
            requested_prefix_len=requested_prefix_len,
            active_quantizers=active_quantizers,
        )
        self._last_forcing_state = {
            "active": 1.0,
            "prefix_len": float(actual_prefix_len),
            "phase_prob": float(phase.self_force_prob),
            "phase_max_prefix": float(max_prefix),
        }
        self._last_rvq_curriculum_state = {
            "active_quantizers": float(active_quantizers),
            "masked_labels": float(masked_labels),
        }
        self._last_conditioning_state = conditioning_stats
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        logs = dict(logs)
        phase = self._phase_for_step(int(self.state.global_step))
        max_prefix = max(phase.prefix_choices) if phase.prefix_choices else -1
        phase_start_step = self._phase_start_step(phase)
        config = self._unwrap_config(self.model)
        rvq_phase = self._rvq_phase_for_step(int(self.state.global_step))
        if rvq_phase is None:
            rvq_phase_start_step = 0
            active_quantizers = int(config.num_quantizers)
        else:
            rvq_phase_start_step = self._rvq_phase_start_step(rvq_phase)
            active_quantizers = int(rvq_phase.active_quantizers)
        logs.setdefault("forcing/self_prob", float(phase.self_force_prob))
        logs.setdefault("forcing/max_prefix", float(max_prefix))
        logs.setdefault("forcing/phase_start_step", float(phase_start_step))
        logs.setdefault("forcing/total_steps", float(self._total_training_steps()))
        logs.setdefault(
            "forcing/active",
            float(self._last_forcing_state.get("active", 0.0)),
        )
        logs.setdefault(
            "forcing/prefix_len",
            float(self._last_forcing_state.get("prefix_len", -1.0)),
        )
        logs.setdefault(
            "rvq_curriculum/active_quantizers",
            float(active_quantizers),
        )
        logs.setdefault(
            "rvq_curriculum/phase_start_step",
            float(rvq_phase_start_step),
        )
        logs.setdefault(
            "rvq_curriculum/masked_labels",
            float(self._last_rvq_curriculum_state.get("masked_labels", 0.0)),
        )
        logs.setdefault(
            "rvq_curriculum/mask_inactive_inputs",
            float(self.rvq_curriculum_mask_inactive_inputs),
        )
        logs.setdefault(
            "conditioning/audio_dropout_prob",
            float(self.audio_condition_dropout_prob),
        )
        logs.setdefault(
            "conditioning/audio_dropped_examples",
            float(self._last_conditioning_state.get("audio_dropped_examples", 0.0)),
        )
        logs.setdefault(
            "conditioning/history_dropout_prob",
            float(self.motion_history_dropout_prob),
        )
        logs.setdefault(
            "conditioning/history_masked_tokens",
            float(self._last_conditioning_state.get("history_masked_tokens", 0.0)),
        )
        logs.setdefault(
            "conditioning/anchor_token_dropout_prob",
            float(self.motion_anchor_token_dropout_prob),
        )
        logs.setdefault(
            "conditioning/anchor_masked_tokens",
            float(self._last_conditioning_state.get("anchor_masked_tokens", 0.0)),
        )
        try:
            gate_model = self.accelerator.unwrap_model(self.model)
        except Exception:
            gate_model = getattr(self.model, "module", self.model)
        audio_layer_gates = getattr(gate_model, "audio_layer_gates", None)
        if audio_layer_gates is not None:
            gate_values = audio_layer_gates.detach().float()
            logs.setdefault(
                "audio_fusion/gate_abs_mean",
                float(gate_values.abs().mean().cpu()),
            )
            logs.setdefault(
                "audio_fusion/gate_abs_max",
                float(gate_values.abs().max().cpu()),
            )
        return super().log(logs, *args, **kwargs)


def _token_color(value: int) -> tuple[int, int, int]:
    """Stable color for compact 0..511 RVQ token values."""

    value = int(value) % 512
    return (
        40 + (value * 37) % 176,
        40 + (value * 67) % 176,
        40 + (value * 97) % 176,
    )


def render_token_comparison_video(
    *,
    gt_motion: Sequence[Sequence[int]],
    pred_motion: Sequence[Sequence[int]],
    title: str,
    fps: int = 4,
    repeat_per_frame: int = 4,
) -> np.ndarray:
    """
    Build a small GT-vs-pred RVQ token comparison video for W&B.

    This is intentionally token-space visualization. It is cheap enough to run
    during evaluation and catches whether the infill model is generating the
    right middle frames before we add heavier RVQVAE/BVH rendering.

    Returns:
        uint8 video array shaped (T, C, H, W), as expected by wandb.Video.
    """

    del fps
    gt = [list(frame) for frame in gt_motion]
    pred = [list(frame) for frame in pred_motion]
    frames = max(len(gt), len(pred), 1)
    quantizers = max(
        max((len(frame) for frame in gt), default=4),
        max((len(frame) for frame in pred), default=4),
    )

    width, height = 860, 420
    cell_w, cell_h = 88, 48
    left_x, right_x = 110, 500
    top_y = 112
    frame_images: List[np.ndarray] = []

    token_matches = 0
    token_total = 0
    for frame_idx in range(frames):
        gt_frame = gt[frame_idx] if frame_idx < len(gt) else []
        pred_frame = pred[frame_idx] if frame_idx < len(pred) else []
        for q_idx in range(min(len(gt_frame), len(pred_frame))):
            token_total += 1
            token_matches += int(int(gt_frame[q_idx]) == int(pred_frame[q_idx]))
    token_acc = token_matches / max(1, token_total)

    for current in range(frames):
        image = Image.new("RGB", (width, height), (248, 248, 246))
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), title[:92], fill=(20, 20, 20))
        draw.text(
            (24, 44),
            f"middle frame {current + 1}/{frames} | token acc {token_acc:.3f}",
            fill=(70, 70, 70),
        )
        draw.text((left_x, 84), "Ground truth", fill=(20, 20, 20))
        draw.text((right_x, 84), "Prediction", fill=(20, 20, 20))

        for row in range(frames):
            y0 = top_y + row * (cell_h + 18)
            row_outline = (30, 110, 220) if row == current else (185, 185, 185)
            draw.text((24, y0 + 14), f"F{row + 1}", fill=(30, 30, 30))
            gt_frame = gt[row] if row < len(gt) else []
            pred_frame = pred[row] if row < len(pred) else []

            for q_idx in range(quantizers):
                gt_val = int(gt_frame[q_idx]) if q_idx < len(gt_frame) else None
                pred_val = int(pred_frame[q_idx]) if q_idx < len(pred_frame) else None

                for panel_x, value in (
                    (left_x, gt_val),
                    (right_x, pred_val),
                ):
                    x0 = panel_x + q_idx * (cell_w + 6)
                    x1 = x0 + cell_w
                    y1 = y0 + cell_h
                    fill = (230, 230, 230) if value is None else _token_color(value)
                    draw.rectangle(
                        [x0, y0, x1, y1],
                        fill=fill,
                        outline=row_outline,
                        width=3 if row == current else 1,
                    )
                    text = "--" if value is None else str(value)
                    draw.text((x0 + 8, y0 + 15), text, fill=(0, 0, 0))

                if gt_val is not None and pred_val is not None:
                    match = gt_val == pred_val
                    status = "OK" if match else "ERR"
                    fill = (30, 125, 50) if match else (170, 40, 40)
                    draw.text(
                        (right_x + quantizers * (cell_w + 6) + 14, y0 + 15),
                        f"q{q_idx + 1} {status}",
                        fill=fill,
                    )

        for _ in range(max(1, repeat_per_frame)):
            frame_images.append(np.asarray(image, dtype=np.uint8))

    video = np.stack(frame_images, axis=0)
    return video.transpose(0, 3, 1, 2)


def _parse_rvqvae_opt_value(value_str: str) -> Any:
    value_str = value_str.strip()
    if value_str == "True":
        return True
    if value_str == "False":
        return False
    if value_str == "None":
        return None
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            return ast.literal_eval(value_str)
        except (SyntaxError, ValueError):
            return value_str
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        return value_str


def _parse_rvqvae_opt_txt(opt_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    with open(opt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = _parse_rvqvae_opt_value(value)
    return result


def load_rvqvae_config_from_checkpoint(checkpoint_path: Path) -> Config:
    opt_path = checkpoint_path.parent.parent / "opt.txt"
    if not opt_path.exists():
        raise FileNotFoundError(f"RVQVAE opt.txt not found: {opt_path}")

    opt = _parse_rvqvae_opt_txt(opt_path)
    config = Config(
        name=opt.get("name", "VQVAE_v2"),
        dataset_name=opt.get("dataset_name", "quat63nodes_v2"),
        checkpoints_dir=opt.get("checkpoints_dir", "./checkpoints"),
        log_dir=opt.get("log_dir", "./log/vq"),
        gpu_id=opt.get("gpu_id", 0),
        local_rank=opt.get("local_rank", 0),
        seed=opt.get("seed", 3407),
        debug=opt.get("debug", False),
    )

    config.data.data_root = opt.get("data_root", "")
    config.data.body_parts = opt.get("body_parts", ["body", "left", "right", "positions"])
    config.data.body_joints_num = opt.get("body_joints_num", 24)
    config.data.left_joints_num = opt.get("left_joints_num", 20)
    config.data.right_joints_num = opt.get("right_joints_num", 20)
    config.data.total_joints_num = opt.get("total_joints_num", 63)
    config.data.body_dim = opt.get("body_dim", 153)
    config.data.left_dim = opt.get("left_dim", 120)
    config.data.right_dim = opt.get("right_dim", 120)
    config.data.whole_dim = opt.get("whole_dim", 393)
    config.data.window_size = opt.get("window_size", 64)
    config.data.batch_size = opt.get("batch_size", 128)
    config.data.num_workers = opt.get("num_workers", 4)
    config.data.fps = opt.get("fps", 20)

    config.model.nb_code = opt.get("nb_code", 512)
    config.model.code_dim = opt.get("code_dim", 512)
    config.model.down_t = opt.get("down_t", 1)
    config.model.stride_t = opt.get("stride_t", 2)
    config.model.width = opt.get("width", 512)
    config.model.depth = opt.get("depth", 3)
    config.model.dilation_growth_rate = opt.get("dilation_growth_rate", 3)
    config.model.vq_act = opt.get("vq_act", "relu")
    config.model.vq_norm = opt.get("vq_norm", None)
    config.model.vq_cnn_depth = opt.get("vq_cnn_depth", 3)
    config.model.num_quantizers = opt.get("num_quantizers", 4)
    config.model.shared_codebook = opt.get("shared_codebook", False)
    config.model.quantize_dropout_prob = opt.get("quantize_dropout_prob", 0.8)
    config.model.quantize_dropout_cutoff_index = opt.get(
        "quantize_dropout_cutoff_index",
        1,
    )
    config.model.use_whole_encoder = opt.get("use_whole_encoder", False)
    config.model.mu = opt.get("mu", 0.99)
    config.unit_length = config.model.down_t * 2
    config.save_root = os.path.join(config.checkpoints_dir, config.dataset_name, config.name)
    config.model_dir = os.path.join(config.save_root, "model")
    config.meta_dir = os.path.join(config.save_root, "meta")
    config.eval_dir = os.path.join(config.save_root, "animation")
    config.log_path = os.path.join(config.log_dir, config.dataset_name, config.name)
    return config


def resolve_motion_token_unit_length(
    requested_unit_length: Optional[float],
    rvqvae_ckpt: Path,
) -> tuple[float, str]:
    if requested_unit_length is not None:
        if requested_unit_length <= 0:
            raise ValueError("--motion_token_unit_length must be > 0")
        return float(requested_unit_length), "cli"

    try:
        rvq_config = load_rvqvae_config_from_checkpoint(rvqvae_ckpt)
        unit_length = float(getattr(rvq_config, "unit_length", 0) or 0)
        if unit_length <= 0:
            unit_length = float(rvq_config.model.down_t * 2)
        if unit_length <= 0:
            raise ValueError(f"invalid RVQVAE unit_length={unit_length}")
        return unit_length, "rvqvae_config"
    except Exception as exc:
        print(
            "[Motion token timing] Could not read RVQVAE unit length from "
            f"{rvqvae_ckpt}: {exc}. Falling back to 2."
        )
        return 2.0, "fallback"


def load_rvqvae_model_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RVQVAE, Config]:
    rvq_config = load_rvqvae_config_from_checkpoint(checkpoint_path)
    rvqvae = RVQVAE(
        config=rvq_config,
        input_dim=rvq_config.data.whole_dim,
        nb_code=rvq_config.model.nb_code,
        code_dim=rvq_config.model.code_dim,
        output_dim=rvq_config.model.code_dim,
        down_t=rvq_config.model.down_t,
        stride_t=rvq_config.model.stride_t,
        width=rvq_config.model.width,
        depth=rvq_config.model.depth,
        dilation_growth_rate=rvq_config.model.dilation_growth_rate,
        activation=rvq_config.model.vq_act,
        norm=rvq_config.model.vq_norm,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    rvqvae.load_state_dict(state_dict)
    return rvqvae.to(device).eval(), rvq_config


def load_motion_dict(path: Path) -> Dict[str, np.ndarray]:
    motion = np.load(path, allow_pickle=True)
    if isinstance(motion, np.ndarray) and motion.dtype == object:
        motion = motion.item()
    if not isinstance(motion, dict):
        raise ValueError(f"Unsupported motion_data format: {path}")
    return motion


def slice_motion_dict(
    motion_dict: Dict[str, np.ndarray],
    start_idx: int,
    frames: int,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for key, value in motion_dict.items():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            result[key] = arr[start_idx : start_idx + frames]
        else:
            result[key] = arr
    return result


def resample_motion_tensor(
    x: torch.Tensor,
    *,
    src_fps: float,
    tgt_fps: float,
) -> torch.Tensor:
    if abs(float(src_fps) - float(tgt_fps)) < 1e-6 or x.shape[1] <= 1:
        return x

    new_frames = max(1, int(round(x.shape[1] * float(tgt_fps) / float(src_fps))))
    batch, _, joints, dims = x.shape
    flat = x.permute(0, 2, 3, 1).contiguous().view(batch, joints * dims, x.shape[1])
    flat = F.interpolate(flat, size=new_frames, mode="linear", align_corners=True)
    return flat.view(batch, joints, dims, new_frames).permute(0, 3, 1, 2).contiguous()


def pad_or_trim_hand_motion(hand: torch.Tensor, frames: int) -> torch.Tensor:
    if hand.shape[1] == 0:
        return torch.zeros(
            hand.shape[0],
            frames,
            hand.shape[2],
            dtype=hand.dtype,
            device=hand.device,
        )
    if hand.shape[1] == frames:
        return hand
    if hand.shape[1] > frames:
        return hand[:, :frames]

    pad = frames - hand.shape[1]
    return F.pad(hand.permute(0, 2, 1), (0, pad), mode="replicate").permute(0, 2, 1)


@torch.no_grad()
def decode_body_tokens_to_features(
    rvqvae: RVQVAE,
    body_tokens: Sequence[Sequence[int]],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tokens = np.asarray(body_tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError(f"Expected body tokens shaped (frames, quantizers), got {tokens.shape}")

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    decoded = rvqvae.forward_decoder({"body": token_tensor})
    decoded = decoded[0] if decoded.ndim == 3 else decoded.sum(0)
    return decoded.float() * std + mean


def body_features_to_quat_motion(
    body_features: torch.Tensor,
    motion_dict: Dict[str, np.ndarray],
    device: torch.device,
    *,
    src_fps: float,
    tgt_fps: float,
) -> Dict[str, np.ndarray]:
    body_features = body_features.to(device=device, dtype=torch.float32)
    frames = int(body_features.shape[0])

    offset_frame0 = torch.tensor([0.0, 0.0, 102.0], device=device)
    offset_vel = body_features[:, :3].clone()
    for frame_idx in range(1, offset_vel.shape[0]):
        offset_vel[frame_idx] = offset_vel[frame_idx] + offset_vel[frame_idx - 1]
    offset = (offset_vel + offset_frame0).reshape(1, frames, 1, 3)

    body_6d = body_features[:, 3:].reshape(1, frames, 25, 6)

    left_raw = motion_dict.get("left")
    right_raw = motion_dict.get("right")
    if left_raw is None:
        left_raw = np.zeros((frames, 20 * 6), dtype=np.float32)
    if right_raw is None:
        right_raw = np.zeros((frames, 20 * 6), dtype=np.float32)

    left = torch.tensor(left_raw, dtype=torch.float32, device=device).unsqueeze(0)
    right = torch.tensor(right_raw, dtype=torch.float32, device=device).unsqueeze(0)
    left = pad_or_trim_hand_motion(left, frames).reshape(1, frames, 20, 6)
    right = pad_or_trim_hand_motion(right, frames).reshape(1, frames, 20, 6)

    body_6d = resample_motion_tensor(body_6d, src_fps=src_fps, tgt_fps=tgt_fps)
    left = resample_motion_tensor(left, src_fps=src_fps, tgt_fps=tgt_fps)
    right = resample_motion_tensor(right, src_fps=src_fps, tgt_fps=tgt_fps)
    offset = resample_motion_tensor(offset, src_fps=src_fps, tgt_fps=tgt_fps)

    out_frames = int(body_6d.shape[1])
    body_quat = sixd_to_quaternion(body_6d.reshape(-1, 6)).reshape(
        1,
        out_frames,
        25,
        4,
    )
    left_quat = sixd_to_quaternion(left.reshape(-1, 6)).reshape(1, out_frames, 20, 4)
    right_quat = sixd_to_quaternion(right.reshape(-1, 6)).reshape(
        1,
        out_frames,
        20,
        4,
    )

    merged = torch.zeros(out_frames, 63, 4, device=device)
    merged[:, BODY_JOINTS_ID] = body_quat[0]
    merged[:, LEFT_HAND_JOINTS_ID[1:]] = left_quat[0, :, 1:]
    merged[:, RIGHT_HAND_JOINTS_ID[1:]] = right_quat[0, :, 1:]

    return {
        "offset": offset.reshape(out_frames, 3).detach().cpu().numpy(),
        "quat": merged.detach().cpu().numpy(),
    }


def quat_motion_to_joint_positions(
    motion: Dict[str, np.ndarray],
    postprocesser: Any,
    device: torch.device,
) -> np.ndarray:
    from actions.postprocess import process_batch_data
    from utils.visualization_torch.Animation import positions_global
    from utils.visualization_torch.Quaternions import Quaternions

    input_quat = torch.as_tensor(motion["quat"], dtype=torch.float32, device=device)
    input_offset = torch.as_tensor(motion["offset"], dtype=torch.float32, device=device)
    if input_quat.ndim == 3:
        input_quat = input_quat.unsqueeze(0)
    if input_offset.ndim == 2:
        input_offset = input_offset.unsqueeze(0)

    final_quats, final_root_pos = process_batch_data(
        input_quat,
        input_offset,
        postprocesser.anim,
        postprocesser.skel.src_joint_dict,
        shape="wxyz",
    )
    final_quats = final_quats.detach().cpu()
    final_root_pos = final_root_pos.detach().cpu() if final_root_pos is not None else None

    num_frames = int(final_quats.shape[1])
    base_pos = postprocesser.anim.positions[0].detach().cpu().clone()
    current_pos = base_pos.unsqueeze(0).repeat(num_frames, 1, 1)
    if final_root_pos is not None:
        current_pos[:, 0, :] = final_root_pos[0]

    postprocesser.anim.rotations = Quaternions(final_quats[0])
    postprocesser.anim.positions = current_pos
    positions = positions_global(postprocesser.anim)
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    return np.asarray(positions, dtype=np.float32)


def render_decoded_motion_comparison_video(
    *,
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    skeleton_edges: Sequence[tuple[int, int]],
    joint_names: Sequence[str],
    title: str,
    fps: int,
    source_frames: int,
    middle_start: int,
    middle_end: int,
) -> np.ndarray:
    width, height = 1280, 720
    panel_w, panel_h = 580, 560
    left_origin = (40, 110)
    right_origin = (660, 110)
    frame_count = max(len(gt_positions), len(pred_positions), 1)

    def project(points: np.ndarray) -> np.ndarray:
        x = points[..., 0] + 0.25 * points[..., 2]
        y = -points[..., 1] + 0.12 * points[..., 2]
        return np.stack([x, y], axis=-1)

    gt_2d = project(gt_positions)
    pred_2d = project(pred_positions)
    combined = np.concatenate([gt_2d.reshape(-1, 2), pred_2d.reshape(-1, 2)], axis=0)
    finite = np.isfinite(combined).all(axis=1)
    combined = combined[finite]
    if combined.size == 0:
        combined = np.zeros((1, 2), dtype=np.float32)
    min_xy = combined.min(axis=0)
    max_xy = combined.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    center = (min_xy + max_xy) * 0.5
    scale = min((panel_w - 80) / span[0], (panel_h - 100) / span[1])

    def to_screen(points_2d: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        x0, y0 = origin
        out = np.empty_like(points_2d)
        out[..., 0] = x0 + panel_w * 0.5 + (points_2d[..., 0] - center[0]) * scale
        out[..., 1] = y0 + panel_h * 0.5 + (points_2d[..., 1] - center[1]) * scale
        return out

    gt_screen = to_screen(gt_2d, left_origin)
    pred_screen = to_screen(pred_2d, right_origin)
    frame_images: List[np.ndarray] = []
    joint_count = max(
        int(gt_screen.shape[1]) if gt_screen.ndim >= 2 else 0,
        int(pred_screen.shape[1]) if pred_screen.ndim >= 2 else 0,
        1,
    )
    valid_edges = [
        (int(parent), int(child))
        for parent, child in skeleton_edges
        if 0 <= int(parent) < joint_count and 0 <= int(child) < joint_count
    ]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    marker_indices = [
        name_to_idx[name]
        for name in ("pelvis", "head", "hand_l", "hand_r", "foot_l", "foot_r")
        if name in name_to_idx and name_to_idx[name] < joint_count
    ]
    if not marker_indices:
        marker_indices = [0]

    def frame_at(seq: np.ndarray, idx: int) -> np.ndarray:
        if len(seq) == 0:
            return np.zeros((joint_count, 2), dtype=np.float32)
        return seq[min(idx, len(seq) - 1)]

    def draw_skeleton(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int]) -> None:
        for parent, child in valid_edges:
            if parent >= len(points) or child >= len(points):
                continue
            ax, ay = points[parent]
            bx, by = points[child]
            draw.line([(float(ax), float(ay)), (float(bx), float(by))], fill=color, width=4)
        for joint_idx in marker_indices:
            if joint_idx >= len(points):
                continue
            x, y = points[joint_idx]
            r = 5 if joint_idx == 0 else 4
            draw.ellipse(
                [float(x - r), float(y - r), float(x + r), float(y + r)],
                fill=color,
                outline=(15, 15, 15),
            )

    def draw_panel(
        draw: ImageDraw.ImageDraw,
        origin: tuple[int, int],
        label: str,
        points: np.ndarray,
        trajectory: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        x0, y0 = origin
        draw.rectangle(
            [x0, y0, x0 + panel_w, y0 + panel_h],
            fill=(248, 248, 246),
            outline=(180, 180, 180),
            width=2,
        )
        draw.text((x0 + 18, y0 + 16), label, fill=(20, 20, 20))
        root_path = trajectory[: min(current + 1, len(trajectory)), 0]
        if len(root_path) >= 2:
            draw.line(
                [(float(x), float(y)) for x, y in root_path],
                fill=(160, 160, 160),
                width=2,
            )
        draw_skeleton(draw, points, color)

    for current in range(frame_count):
        image = Image.new("RGB", (width, height), (236, 236, 232))
        draw = ImageDraw.Draw(image)
        source_pos = (current + 0.5) * max(1, source_frames) / max(1, frame_count)
        in_middle = middle_start <= source_pos < middle_end
        region = "INFILL" if in_middle else "context"
        accent = (200, 62, 42) if in_middle else (70, 90, 110)

        draw.text((34, 26), title[:110], fill=(20, 20, 20))
        draw.text(
            (34, 54),
            f"frame {current + 1}/{frame_count} | {fps} fps | region: {region}",
            fill=accent,
        )
        draw.rectangle([34, 82, width - 34, 92], fill=(205, 205, 200))
        progress_x = 34 + int((width - 68) * (current + 1) / frame_count)
        draw.rectangle([34, 82, progress_x, 92], fill=accent)

        draw_panel(
            draw,
            left_origin,
            "Ground truth",
            frame_at(gt_screen, current),
            gt_screen,
            (42, 104, 185),
        )
        draw_panel(
            draw,
            right_origin,
            "Prediction",
            frame_at(pred_screen, current),
            pred_screen,
            (188, 76, 50),
        )
        frame_images.append(np.asarray(image, dtype=np.uint8))

    video = np.stack(frame_images, axis=0)
    return video.transpose(0, 3, 1, 2)


def motion_token_accuracy(
    gt_motion: Sequence[Sequence[int]],
    pred_motion: Sequence[Sequence[int]],
) -> float:
    matches = 0
    total = 0
    for gt_frame, pred_frame in zip(gt_motion, pred_motion):
        for gt_token, pred_token in zip(gt_frame, pred_frame):
            matches += int(int(gt_token) == int(pred_token))
            total += 1
    return matches / max(1, total)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evenly_spaced_indices(length: int, count: int) -> List[int]:
    length = int(length)
    count = max(0, int(count))
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


class AudioFIMEvalMetricsCallback(TrainerCallback):
    """Log stronger Step 2 eval metrics without materializing full eval logits."""

    def __init__(
        self,
        *,
        eval_dataset: AudioFIMCausalDataset,
        collator: AudioFIMCausalCollator,
        config: AudioFIMCausalConfig,
        teacher_forced_examples: int,
        teacher_forced_batch_size: int,
        teacher_forced_every_n_evals: int,
        generation_examples: int,
        generation_every_n_evals: int,
        audio_cfg_scale: float = 1.0,
    ):
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.config = config
        self.teacher_forced_indices = evenly_spaced_indices(
            len(eval_dataset),
            teacher_forced_examples,
        )
        self.teacher_forced_batch_size = max(1, int(teacher_forced_batch_size))
        self.teacher_forced_every_n_evals = max(1, int(teacher_forced_every_n_evals))
        self.generation_indices = evenly_spaced_indices(
            len(eval_dataset),
            generation_examples,
        )
        self.generation_every_n_evals = max(1, int(generation_every_n_evals))
        self.audio_cfg_scale = float(audio_cfg_scale)
        self._eval_calls = 0

    def _world_process_zero(self, state) -> bool:
        return not hasattr(state, "is_world_process_zero") or state.is_world_process_zero

    def _log(self, payload: Dict[str, float], step: int) -> None:
        if not payload:
            return

        try:
            import wandb
        except ImportError:
            wandb = None
        if wandb is not None and wandb.run is not None:
            wandb.log(payload, step=step)

        preview_keys = [
            "eval_metrics/teacher_ce",
            "eval_metrics/teacher_zero_audio_ce",
            "eval_metrics/teacher_audio_ce_delta",
            "eval_metrics/teacher_token_acc",
            "eval_metrics/teacher_top5_acc",
            "eval_gen/token_acc_mean",
            "eval_gen/exact_frame_acc",
        ]
        preview = ", ".join(
            f"{key}={payload[key]:.4f}" for key in preview_keys if key in payload
        )
        if preview:
            print(f"[AudioFIM eval metrics] step={step} {preview}")

    @torch.no_grad()
    def _teacher_forced_metrics(self, forward_model, device: torch.device) -> Dict[str, float]:
        totals: Dict[str, int] = {
            "tokens": 0,
            "correct": 0,
            "top5": 0,
            "top10": 0,
            "motion_tokens": 0,
            "motion_correct": 0,
            "zero_audio_correct": 0,
            "zero_audio_motion_correct": 0,
            "eos": 0,
            "eos_correct": 0,
        }
        loss_sums: Dict[str, float] = {
            "ce": 0.0,
            "motion_ce": 0.0,
            "zero_audio_ce": 0.0,
        }
        for q_idx in range(self.config.num_quantizers):
            totals[f"q{q_idx + 1}"] = 0
            totals[f"q{q_idx + 1}_correct"] = 0
            loss_sums[f"q{q_idx + 1}_ce"] = 0.0

        for start in range(0, len(self.teacher_forced_indices), self.teacher_forced_batch_size):
            batch_indices = self.teacher_forced_indices[
                start : start + self.teacher_forced_batch_size
            ]
            examples = [self.eval_dataset[idx] for idx in batch_indices]
            batch = self.collator(examples)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }

            outputs = forward_model(**batch)
            logits = outputs.logits.detach()
            labels = batch["labels"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels != IGNORE_INDEX
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                continue

            token_losses = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(shift_labels)
            loss_sums["ce"] += float(token_losses[valid].sum().item())

            pred_ids = shift_logits.argmax(dim=-1)
            totals["tokens"] += valid_count
            totals["correct"] += int((pred_ids[valid] == shift_labels[valid]).sum().item())

            valid_logits = shift_logits[valid]
            valid_labels = shift_labels[valid]
            top_k = min(10, int(valid_logits.shape[-1]))
            top_ids = valid_logits.topk(k=top_k, dim=-1).indices
            totals["top5"] += int(
                (top_ids[:, : min(5, top_k)] == valid_labels.unsqueeze(-1))
                .any(dim=-1)
                .sum()
                .item()
            )
            totals["top10"] += int(
                (top_ids == valid_labels.unsqueeze(-1)).any(dim=-1).sum().item()
            )

            motion_mask = valid & (
                (shift_labels >= 0) & (shift_labels < self.config.motion_vocab_size)
            )
            motion_total = int(motion_mask.sum().item())
            totals["motion_tokens"] += motion_total
            if motion_total:
                totals["motion_correct"] += int(
                    (pred_ids[motion_mask] == shift_labels[motion_mask]).sum().item()
                )
                loss_sums["motion_ce"] += float(
                    token_losses[motion_mask].sum().item()
                )

            eos_mask = valid & (shift_labels == self.config.eos_token_id)
            eos_total = int(eos_mask.sum().item())
            totals["eos"] += eos_total
            if eos_total:
                totals["eos_correct"] += int(
                    (pred_ids[eos_mask] == shift_labels[eos_mask]).sum().item()
                )

            for q_idx in range(self.config.num_quantizers):
                start_id = q_idx * self.config.codebook_size
                end_id = start_id + self.config.codebook_size
                q_mask = valid & ((shift_labels >= start_id) & (shift_labels < end_id))
                q_total = int(q_mask.sum().item())
                totals[f"q{q_idx + 1}"] += q_total
                if q_total:
                    totals[f"q{q_idx + 1}_correct"] += int(
                        (pred_ids[q_mask] == shift_labels[q_mask]).sum().item()
                    )
                    loss_sums[f"q{q_idx + 1}_ce"] += float(
                        token_losses[q_mask].sum().item()
                    )

            zero_audio_batch = dict(batch)
            zero_audio_batch["audio_features"] = torch.zeros_like(
                batch["audio_features"]
            )
            zero_outputs = forward_model(**zero_audio_batch)
            zero_shift_logits = zero_outputs.logits.detach()[:, :-1, :].contiguous()
            zero_token_losses = F.cross_entropy(
                zero_shift_logits.float().view(-1, zero_shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(shift_labels)
            loss_sums["zero_audio_ce"] += float(
                zero_token_losses[valid].sum().item()
            )

            zero_pred_ids = zero_shift_logits.argmax(dim=-1)
            totals["zero_audio_correct"] += int(
                (zero_pred_ids[valid] == shift_labels[valid]).sum().item()
            )
            if motion_total:
                totals["zero_audio_motion_correct"] += int(
                    (zero_pred_ids[motion_mask] == shift_labels[motion_mask])
                    .sum()
                    .item()
                )

        teacher_ce = safe_div(loss_sums["ce"], totals["tokens"])
        zero_audio_ce = safe_div(loss_sums["zero_audio_ce"], totals["tokens"])
        teacher_acc = safe_div(totals["correct"], totals["tokens"])
        zero_audio_acc = safe_div(totals["zero_audio_correct"], totals["tokens"])

        metrics = {
            "eval_metrics/teacher_examples": float(len(self.teacher_forced_indices)),
            "eval_metrics/teacher_ce": teacher_ce,
            "eval_metrics/teacher_motion_ce": safe_div(
                loss_sums["motion_ce"],
                totals["motion_tokens"],
            ),
            "eval_metrics/teacher_zero_audio_ce": zero_audio_ce,
            "eval_metrics/teacher_audio_ce_delta": zero_audio_ce - teacher_ce,
            "eval_metrics/teacher_token_acc": teacher_acc,
            "eval_metrics/teacher_zero_audio_token_acc": zero_audio_acc,
            "eval_metrics/teacher_audio_acc_delta": teacher_acc - zero_audio_acc,
            "eval_metrics/teacher_motion_token_acc": safe_div(
                totals["motion_correct"],
                totals["motion_tokens"],
            ),
            "eval_metrics/teacher_zero_audio_motion_token_acc": safe_div(
                totals["zero_audio_motion_correct"],
                totals["motion_tokens"],
            ),
            "eval_metrics/teacher_top5_acc": safe_div(totals["top5"], totals["tokens"]),
            "eval_metrics/teacher_top10_acc": safe_div(totals["top10"], totals["tokens"]),
            "eval_metrics/teacher_eos_acc": safe_div(totals["eos_correct"], totals["eos"]),
        }
        for q_idx in range(self.config.num_quantizers):
            metrics[f"eval_metrics/teacher_q{q_idx + 1}_acc"] = safe_div(
                totals[f"q{q_idx + 1}_correct"],
                totals[f"q{q_idx + 1}"],
            )
            metrics[f"eval_metrics/teacher_q{q_idx + 1}_ce"] = safe_div(
                loss_sums[f"q{q_idx + 1}_ce"],
                totals[f"q{q_idx + 1}"],
            )
        return metrics

    @torch.no_grad()
    def _generation_metrics(self, base_model) -> Dict[str, float]:
        totals: Dict[str, int] = {
            "tokens": 0,
            "correct": 0,
            "frames": 0,
            "exact_frames": 0,
            "gaps": 0,
            "exact_gaps": 0,
        }
        for q_idx in range(self.config.num_quantizers):
            totals[f"q{q_idx + 1}"] = 0
            totals[f"q{q_idx + 1}_correct"] = 0

        for dataset_idx in self.generation_indices:
            example = self.eval_dataset[dataset_idx]
            pred_motion = base_model.generate_infill(
                history_motion=example.history_motion,
                left_anchor=example.left_anchor,
                right_anchor=example.right_anchor,
                middle_audio_features=example.middle_audio_features,
                left_audio_feature=example.left_audio_feature,
                right_audio_feature=example.right_audio_feature,
                history_audio_features=example.history_audio_features,
                temperature=0.0,
                audio_cfg_scale=self.audio_cfg_scale,
            )
            gap_exact = True
            for frame_idx, gt_frame in enumerate(example.middle_motion):
                pred_frame = pred_motion[frame_idx] if frame_idx < len(pred_motion) else []
                frame_exact = len(gt_frame) == len(pred_frame)
                for q_idx, gt_token in enumerate(gt_frame):
                    if q_idx >= len(pred_frame):
                        totals["tokens"] += 1
                        totals[f"q{q_idx + 1}"] += 1
                        frame_exact = False
                        gap_exact = False
                        continue
                    match = int(gt_token) == int(pred_frame[q_idx])
                    totals["tokens"] += 1
                    totals["correct"] += int(match)
                    totals[f"q{q_idx + 1}"] += 1
                    totals[f"q{q_idx + 1}_correct"] += int(match)
                    frame_exact = frame_exact and match
                    gap_exact = gap_exact and match
                totals["frames"] += 1
                totals["exact_frames"] += int(frame_exact)
            totals["gaps"] += 1
            totals["exact_gaps"] += int(gap_exact)

        metrics = {
            "eval_gen/examples": float(len(self.generation_indices)),
            "eval_gen/audio_cfg_scale": float(self.audio_cfg_scale),
            "eval_gen/token_acc_mean": safe_div(totals["correct"], totals["tokens"]),
            "eval_gen/exact_frame_acc": safe_div(
                totals["exact_frames"],
                totals["frames"],
            ),
            "eval_gen/exact_gap_acc": safe_div(totals["exact_gaps"], totals["gaps"]),
        }
        for q_idx in range(self.config.num_quantizers):
            metrics[f"eval_gen/q{q_idx + 1}_acc"] = safe_div(
                totals[f"q{q_idx + 1}_correct"],
                totals[f"q{q_idx + 1}"],
            )
        return metrics

    def on_evaluate(self, args, state, control, **kwargs):
        del args, control

        if not self._world_process_zero(state):
            return
        if not self.teacher_forced_indices and not self.generation_indices:
            return

        self._eval_calls += 1
        model = kwargs.get("model")
        if model is None:
            return

        base_model = model.module if hasattr(model, "module") else model
        device = next(base_model.parameters()).device
        was_training = model.training
        model.eval()
        payload: Dict[str, float] = {}

        if (
            self.teacher_forced_indices
            and self._eval_calls % self.teacher_forced_every_n_evals == 0
        ):
            payload.update(self._teacher_forced_metrics(base_model, device))

        if (
            self.generation_indices
            and self._eval_calls % self.generation_every_n_evals == 0
            and hasattr(base_model, "generate_infill")
        ):
            payload.update(self._generation_metrics(base_model))

        self._log(payload, step=state.global_step)
        if was_training:
            model.train()


class WandbEvalVideoCallback(TrainerCallback):
    """Log cheap GT-vs-pred token videos for fixed eval examples."""

    def __init__(
        self,
        *,
        eval_dataset: AudioFIMCausalDataset,
        num_examples: int,
        every_n_evals: int,
        fps: int,
        audio_cfg_scale: float = 1.0,
    ):
        self.eval_dataset = eval_dataset
        self.num_examples = max(0, int(num_examples))
        self.every_n_evals = max(1, int(every_n_evals))
        self.fps = max(1, int(fps))
        self.audio_cfg_scale = float(audio_cfg_scale)
        self._eval_calls = 0

    def on_evaluate(self, args, state, control, **kwargs):
        del args, control

        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if self.num_examples <= 0 or self.eval_dataset is None:
            return

        self._eval_calls += 1
        if self._eval_calls % self.every_n_evals != 0:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        model = kwargs.get("model")
        if model is None:
            return
        base_model = model.module if hasattr(model, "module") else model
        if not hasattr(base_model, "generate_infill"):
            return

        was_training = base_model.training
        log_payload: Dict[str, Any] = {}

        for sample_idx in range(min(self.num_examples, len(self.eval_dataset))):
            example = self.eval_dataset[sample_idx]
            pred_motion = base_model.generate_infill(
                history_motion=example.history_motion,
                left_anchor=example.left_anchor,
                right_anchor=example.right_anchor,
                middle_audio_features=example.middle_audio_features,
                left_audio_feature=example.left_audio_feature,
                right_audio_feature=example.right_audio_feature,
                history_audio_features=example.history_audio_features,
                temperature=0.0,
                audio_cfg_scale=self.audio_cfg_scale,
            )
            video = render_token_comparison_video(
                gt_motion=example.middle_motion,
                pred_motion=pred_motion,
                title=(
                    f"{example.name} | step {state.global_step} | "
                    f"cfg {self.audio_cfg_scale:g}"
                ),
                fps=self.fps,
            )
            log_payload[f"eval_video/token_compare_{sample_idx}"] = wandb.Video(
                video,
                fps=self.fps,
                format="mp4",
            )

        if log_payload:
            wandb.log(log_payload, step=state.global_step)

        if was_training:
            base_model.train()


class WandbEvalMotionVideoCallback(TrainerCallback):
    """Log decoded GT-vs-pred motion videos for fixed eval examples."""

    def __init__(
        self,
        *,
        eval_dataset: AudioFIMCausalDataset,
        motion_data_dir: Path,
        rvqvae_ckpt: Path,
        mean_path: Path,
        std_path: Path,
        num_examples: int,
        every_n_evals: int,
        fps: int,
        audio_cfg_scale: float = 1.0,
        scan_examples: int = 200,
    ):
        self.eval_dataset = eval_dataset
        self.motion_data_dir = motion_data_dir
        self.rvqvae_ckpt = rvqvae_ckpt
        self.mean_path = mean_path
        self.std_path = std_path
        self.num_examples = max(0, int(num_examples))
        self.every_n_evals = max(1, int(every_n_evals))
        self.fps = max(1, int(fps))
        self.audio_cfg_scale = float(audio_cfg_scale)
        self._eval_calls = 0
        self._rvqvae: Optional[RVQVAE] = None
        self._rvq_config: Optional[Config] = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._postprocesser: Any = None
        self._missing_warned: set[str] = set()
        self.sample_indices = self._select_sample_indices(scan_examples)

    def _select_sample_indices(self, scan_examples: int) -> List[int]:
        scored: List[tuple[int, int]] = []
        max_scan = min(len(self.eval_dataset), max(self.num_examples, int(scan_examples)))
        for idx in range(max_scan):
            try:
                example = self.eval_dataset[idx]
            except Exception:
                continue
            scored.append((len(example.history_motion), idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored[: self.num_examples]]

    def _ensure_decode_stack(self, device: torch.device) -> None:
        if self._rvqvae is None:
            self._rvqvae, self._rvq_config = load_rvqvae_model_for_eval(
                self.rvqvae_ckpt,
                device,
            )
            self._mean = torch.tensor(
                np.load(self.mean_path),
                dtype=torch.float32,
                device=device,
            )
            self._std = torch.tensor(
                np.load(self.std_path),
                dtype=torch.float32,
                device=device,
            )
            from actions.postprocess import MotionPostprocesser

            self._postprocesser = MotionPostprocesser()
            print(
                "[W&B motion video] loaded RVQVAE decoder: "
                f"{self.rvqvae_ckpt}"
            )
            return

        if next(self._rvqvae.parameters()).device != device:
            self._rvqvae = self._rvqvae.to(device)
            if self._mean is not None:
                self._mean = self._mean.to(device)
            if self._std is not None:
                self._std = self._std.to(device)

    def on_evaluate(self, args, state, control, **kwargs):
        del args, control

        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if self.num_examples <= 0 or self.eval_dataset is None:
            return

        self._eval_calls += 1
        if self._eval_calls % self.every_n_evals != 0:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        model = kwargs.get("model")
        if model is None:
            return
        base_model = model.module if hasattr(model, "module") else model
        if not hasattr(base_model, "generate_infill"):
            return

        device = next(base_model.parameters()).device
        self._ensure_decode_stack(device)
        if (
            self._rvqvae is None
            or self._rvq_config is None
            or self._mean is None
            or self._std is None
            or self._postprocesser is None
        ):
            return

        was_training = base_model.training
        log_payload: Dict[str, Any] = {}

        for sample_slot, dataset_idx in enumerate(self.sample_indices):
            example = self.eval_dataset[dataset_idx]
            motion_path = self.motion_data_dir / f"{example.name}.npy"
            if not motion_path.exists():
                key = str(motion_path)
                if key not in self._missing_warned:
                    print(f"[W&B motion video] missing motion_data file, skip: {motion_path}")
                    self._missing_warned.add(key)
                continue

            try:
                pred_middle = base_model.generate_infill(
                    history_motion=example.history_motion,
                    left_anchor=example.left_anchor,
                    right_anchor=example.right_anchor,
                    middle_audio_features=example.middle_audio_features,
                    left_audio_feature=example.left_audio_feature,
                    right_audio_feature=example.right_audio_feature,
                    history_audio_features=example.history_audio_features,
                    temperature=0.0,
                    audio_cfg_scale=self.audio_cfg_scale,
                )

                prefix = [list(frame) for frame in example.history_motion]
                prefix.append(list(example.left_anchor))
                gt_clip = prefix + [list(frame) for frame in example.middle_motion]
                gt_clip.append(list(example.right_anchor))
                pred_clip = prefix + [list(frame) for frame in pred_middle]
                pred_clip.append(list(example.right_anchor))

                src_fps = float(self._rvq_config.data.fps)
                gt_features = decode_body_tokens_to_features(
                    self._rvqvae,
                    gt_clip,
                    self._mean,
                    self._std,
                    device,
                )
                pred_features = decode_body_tokens_to_features(
                    self._rvqvae,
                    pred_clip,
                    self._mean,
                    self._std,
                    device,
                )
                clip_start_idx = max(0, int(example.left_idx) - len(example.history_motion))
                unit_length = int(getattr(self._rvq_config, "unit_length", 2))
                motion_dict = slice_motion_dict(
                    load_motion_dict(motion_path),
                    clip_start_idx * unit_length,
                    int(gt_features.shape[0]),
                )
                gt_motion = body_features_to_quat_motion(
                    gt_features,
                    motion_dict,
                    device,
                    src_fps=src_fps,
                    tgt_fps=float(self.fps),
                )
                pred_motion = body_features_to_quat_motion(
                    pred_features,
                    motion_dict,
                    device,
                    src_fps=src_fps,
                    tgt_fps=float(self.fps),
                )
                gt_positions = quat_motion_to_joint_positions(
                    gt_motion,
                    self._postprocesser,
                    device,
                )
                pred_positions = quat_motion_to_joint_positions(
                    pred_motion,
                    self._postprocesser,
                    device,
                )
                token_acc = motion_token_accuracy(example.middle_motion, pred_middle)
                middle_start = len(example.history_motion) + 1
                middle_end = middle_start + len(example.middle_motion)
                joint_names = list(getattr(self._postprocesser.anim, "names", []))
                skeleton_edges = [
                    (int(parent), int(child))
                    for child, parent in enumerate(self._postprocesser.anim.parents)
                    if int(parent) >= 0
                ]
                video = render_decoded_motion_comparison_video(
                    gt_positions=gt_positions,
                    pred_positions=pred_positions,
                    skeleton_edges=skeleton_edges,
                    joint_names=joint_names,
                    title=(
                        f"{example.name} | step {state.global_step} | "
                        f"middle token acc {token_acc:.3f} | "
                        f"cfg {self.audio_cfg_scale:g}"
                    ),
                    fps=self.fps,
                    source_frames=len(gt_clip),
                    middle_start=middle_start,
                    middle_end=middle_end,
                )
                log_payload[f"eval_video/motion_compare_{sample_slot}"] = wandb.Video(
                    video,
                    fps=self.fps,
                    format="mp4",
                )
                log_payload[f"eval_motion/token_acc_{sample_slot}"] = token_acc
            except Exception as exc:
                print(
                    "[W&B motion video] failed to render "
                    f"sample={example.name}: {exc}"
                )

        if log_payload:
            wandb.log(log_payload, step=state.global_step)

        if was_training:
            base_model.train()


@torch.no_grad()
def run_loss_sanity_check(
    model: AudioFIMCausalLM,
    dataset: AudioFIMCausalDataset,
    collator: AudioFIMCausalCollator,
    *,
    num_examples: int,
    use_bf16: bool,
    use_fp16: bool,
) -> None:
    """
    Print a first-batch CE/logit sanity check before Trainer takes over.

    For vocab_size=2075, all-zero logits should give CE ~= log(2075)=7.637.
    If zero-logit loss is normal but model loss is huge, inspect logit scale.
    If zero-logit loss is huge, labels/shift/vocab are wrong.
    """

    was_training = model.training
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda" and use_bf16:
        dtype = torch.bfloat16
    elif device.type == "cuda" and use_fp16:
        dtype = torch.float16

    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)

    examples = [dataset[i] for i in range(min(num_examples, len(dataset)))]
    batch = collator(examples)
    batch = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }

    outputs = model(**batch)
    logits = outputs.logits.detach().float()
    labels = batch["labels"]
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    supervised = int(valid.sum().item())

    zero_logits = torch.zeros(
        logits[:, :-1, :].shape,
        dtype=torch.float32,
        device=device,
    )
    zero_loss = torch.nn.functional.cross_entropy(
        zero_logits.view(-1, zero_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    target_ids = shift_labels[valid]
    shared_embed_head = (
        model.embed_tokens.weight.data_ptr() == model.out_head.weight.data_ptr()
    )

    print("=" * 70)
    print("[AudioFIM loss sanity]")
    print(f"examples:                    {len(examples)}")
    print(f"seq_len:                     {batch['input_ids'].shape[1]}")
    print(f"audio_bank_len:              {batch['audio_features'].shape[1]}")
    print(f"supervised labels:           {supervised}")
    print(f"target id min/max:           {int(target_ids.min())}/{int(target_ids.max())}")
    print(f"vocab_size:                  {model.config.vocab_size}")
    print(f"expected uniform CE:         {math.log(model.config.vocab_size):.4f}")
    print(f"zero-logit CE:               {float(zero_loss.detach().cpu()):.4f}")
    print(f"model CE:                    {float(outputs.loss.detach().float().cpu()):.4f}")
    print(
        "logits mean/std/min/max:     "
        f"{float(logits.mean().cpu()):.4f}/"
        f"{float(logits.std().cpu()):.4f}/"
        f"{float(logits.min().cpu()):.4f}/"
        f"{float(logits.max().cpu()):.4f}"
    )
    print(f"tie_word_embeddings config:  {model.config.tie_word_embeddings}")
    print(f"embed/out_head share memory: {shared_embed_head}")
    print("=" * 70)

    if was_training:
        model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train compact Step 2 audio-aware causal FIM transformer"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/audio_fim_causal"),
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data"),
        help="Dataset root containing motion_token_data and audio feature dirs",
    )
    parser.add_argument("--motion_token_dir", type=str, default=None)
    parser.add_argument("--audio_feat_dir", type=str, default=None)
    parser.add_argument("--motion2text_json", type=str, default=None)
    parser.add_argument("--train_split_file", type=str, default=None)
    parser.add_argument("--eval_split_file", type=str, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_windows_per_sequence", type=int, default=None)

    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Classic gap uses step=4, predicting 3 middle frames.",
    )
    parser.add_argument("--audio_fps", type=float, default=10.0)
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=20.0,
        help=(
            "Source/raw motion FPS fallback when motion token JSON has no fps. "
            "Do not use this as codec-token FPS."
        ),
    )
    parser.add_argument(
        "--motion_token_fps",
        type=float,
        default=None,
        help=(
            "Explicit FPS of RVQVAE motion-token timesteps. If omitted, derive "
            "from source motion FPS divided by --motion_token_unit_length."
        ),
    )
    parser.add_argument(
        "--motion_token_unit_length",
        type=float,
        default=None,
        help=(
            "Raw motion frames represented by one RVQVAE token timestep. If "
            "omitted, read from the RVQVAE config when available, otherwise use 2."
        ),
    )
    parser.add_argument("--min_history_frames", type=int, default=0)
    parser.add_argument("--max_history_frames", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--debug_examples", type=int, default=0)
    parser.add_argument(
        "--debug_loss_sanity",
        type=int,
        default=0,
        help=(
            "Run a first-batch CE/logit sanity check on this many examples "
            "before training. Use this when loss scale looks suspicious."
        ),
    )
    parser.add_argument("--profile_startup", action="store_true")
    parser.add_argument("--profile_collator_batches", type=int, default=0)

    parser.add_argument(
        "--architecture_preset",
        type=str,
        default="custom",
        choices=[
            "custom",
            "base",
            "half_depth",
            "half_params",
            "half_width",
            "double_depth",
            "double_params",
            "double_width",
        ],
        help=(
            "Optional architecture preset. custom leaves the explicit "
            "--hidden_size/--num_layers/--num_heads/--intermediate_size args "
            "unchanged. Presets override those four architecture args."
        ),
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument(
        "--audio_fusion",
        type=str,
        default="input_add",
        choices=["input_add", "gated_layers"],
        help=(
            "How HuBERT audio enters the transformer. input_add is the original "
            "embedding-level addition. gated_layers injects audio before every "
            "transformer block through zero-init learned gates."
        ),
    )
    parser.add_argument(
        "--audio_gate_init",
        type=float,
        default=0.0,
        help="Initial value for gated_layers audio gates.",
    )
    parser.add_argument(
        "--max_gap_frames",
        type=int,
        default=16,
        help="Reserve [LEN_1]..[LEN_N] tokens for future variable-gap training.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--audio_condition_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "Per-example probability of zeroing all audio features during "
            "training. Enables audio CFG at inference."
        ),
    )
    parser.add_argument(
        "--motion_history_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "Per-example probability of masking all history motion context "
            "tokens during training."
        ),
    )
    parser.add_argument(
        "--motion_anchor_token_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "Per-token probability of masking left/right anchor RVQ tokens "
            "during training."
        ),
    )
    parser.add_argument(
        "--self_forcing_schedule",
        type=str,
        default="0:0.0:",
        help=(
            "Step-based schedule start:prob:prefixes;... for scheduled "
            "self-forcing. Phase starts can be optimizer steps or percentages "
            "of total optimizer steps, such as 5000 or 20%. Prefix lengths "
            "count compact RVQ motion tokens. Example: "
            "'0%:0.0:;20%:0.25:0,4;50%:0.5:0,4,8;"
            "80%:1.0:0,1,2,3,4,5,6,7,8'."
        ),
    )
    parser.add_argument(
        "--rvq_curriculum_schedule",
        type=str,
        default="off",
        help=(
            "Optional coarse-to-fine RVQ loss schedule start:active_quantizers;... "
            "Phase starts can be optimizer steps or percentages of total optimizer "
            "steps. active_quantizers=1 trains only res_1, while 4 trains all "
            "RVQ levels. Example: '0%:1;20%:2;45%:3;70%:4'. Use off/full to "
            "disable."
        ),
    )
    parser.add_argument(
        "--rvq_curriculum_mask_inactive_inputs",
        action="store_true",
        help=(
            "When RVQ curriculum is active, replace inactive supervised target "
            "tokens in input_ids with [MASK] in addition to masking their loss. "
            "Leave off for the conservative first experiment."
        ),
    )
    parser.add_argument(
        "--eval_metric_examples",
        type=int,
        default=0,
        help=(
            "Number of evenly-spaced eval windows for batched teacher-forced "
            "token/top-k accuracy metrics. Set >0 to enable."
        ),
    )
    parser.add_argument("--eval_metric_batch_size", type=int, default=256)
    parser.add_argument(
        "--eval_metric_every_n_evals",
        type=int,
        default=1,
        help="Run teacher-forced eval metrics every N evaluation calls.",
    )
    parser.add_argument(
        "--eval_gen_metric_examples",
        type=int,
        default=0,
        help=(
            "Number of evenly-spaced eval windows for greedy free-running "
            "generation accuracy metrics. Set >0 to enable."
        ),
    )
    parser.add_argument(
        "--eval_gen_metric_every_n_evals",
        type=int,
        default=1,
        help="Run free-running generation metrics every N evaluation calls.",
    )
    parser.add_argument(
        "--eval_audio_cfg_scale",
        type=float,
        default=1.0,
        help=(
            "Audio classifier-free guidance scale used by free-running eval "
            "metrics and W&B videos. 1.0 is normal conditional generation."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="Trainer reporting backend. Use 'wandb' to enable W&B logging.",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags, e.g. audio_fim,debug,step2.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Use offline when the server has no network.",
    )
    parser.add_argument(
        "--wandb_log_eval_videos",
        action="store_true",
        help="Log cheap GT-vs-pred token comparison videos during evaluation.",
    )
    parser.add_argument("--wandb_video_examples", type=int, default=2)
    parser.add_argument(
        "--wandb_video_every_n_evals",
        type=int,
        default=1,
        help="Log eval videos every N evaluation calls.",
    )
    parser.add_argument("--wandb_video_fps", type=int, default=4)
    parser.add_argument(
        "--wandb_log_eval_motion_videos",
        action="store_true",
        help="Decode eval predictions through RVQVAE and log GT-vs-pred motion videos.",
    )
    parser.add_argument("--wandb_motion_video_examples", type=int, default=2)
    parser.add_argument(
        "--wandb_motion_video_every_n_evals",
        type=int,
        default=1,
        help="Log decoded motion videos every N evaluation calls.",
    )
    parser.add_argument("--wandb_motion_video_fps", type=int, default=20)
    parser.add_argument(
        "--rvqvae_ckpt",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/rvqvae/model/epoch_30.pth"),
        help="Step 1 RVQVAE checkpoint used to decode motion tokens for eval videos.",
    )
    parser.add_argument(
        "--rvqvae_mean_path",
        type=str,
        default=str(MOTION_GENERATION_DIR / "meta/mta_gen_demo/mean.npy"),
    )
    parser.add_argument(
        "--rvqvae_std_path",
        type=str,
        default=str(MOTION_GENERATION_DIR / "meta/mta_gen_demo/std.npy"),
    )
    parser.add_argument(
        "--motion_data_dir",
        type=str,
        default=None,
        help="Original motion_data directory used for hand placeholders in decoded eval videos.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start = time.perf_counter()
    active_architecture_preset = apply_architecture_preset(args)

    if args.step - 1 > args.max_gap_frames:
        raise ValueError("--max_gap_frames must be >= --step - 1")
    if args.audio_fps <= 0:
        raise ValueError("--audio_fps must be > 0")
    if args.motion_fps <= 0:
        raise ValueError("--motion_fps must be > 0")
    for name in (
        "audio_condition_dropout_prob",
        "motion_history_dropout_prob",
        "motion_anchor_token_dropout_prob",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1]")
    if args.eval_audio_cfg_scale < 0:
        raise ValueError("--eval_audio_cfg_scale must be >= 0")
    self_forcing_schedule = parse_self_forcing_schedule(
        args.self_forcing_schedule
    )
    rvq_curriculum_schedule = parse_rvq_curriculum_schedule(
        args.rvq_curriculum_schedule
    )

    data_dir = Path(args.data_dir)
    motion_token_dir = Path(args.motion_token_dir or data_dir / "motion_token_data")
    motion_data_dir = Path(args.motion_data_dir or data_dir / "motion_data")
    audio_fps_tag = format_fps_for_dir(args.audio_fps)
    audio_feat_dir = Path(
        args.audio_feat_dir
        or data_dir / f"audio_features_hubert_layer9_fps{audio_fps_tag}"
    )
    motion_token_unit_length, motion_token_unit_length_source = (
        resolve_motion_token_unit_length(
            args.motion_token_unit_length,
            Path(args.rvqvae_ckpt),
        )
    )
    if args.motion_token_fps is not None and args.motion_token_fps <= 0:
        raise ValueError("--motion_token_fps must be > 0")
    default_motion_token_fps = (
        float(args.motion_token_fps)
        if args.motion_token_fps is not None
        else float(args.motion_fps) / float(motion_token_unit_length)
    )
    if default_motion_token_fps <= 0:
        raise ValueError("Derived motion token FPS must be > 0")
    motion2text_json = args.motion2text_json or str(
        data_dir / "text_data/motion2text.json"
    )

    if args.report_to == "wandb" and args.wandb_log_eval_motion_videos:
        required_paths = {
            "rvqvae checkpoint": Path(args.rvqvae_ckpt),
            "rvqvae mean": Path(args.rvqvae_mean_path),
            "rvqvae std": Path(args.rvqvae_std_path),
            "motion_data dir": motion_data_dir,
        }
        missing = [
            f"{name}: {path}"
            for name, path in required_paths.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Decoded W&B motion videos need these files/dirs:\n"
                + "\n".join(f"  - {item}" for item in missing)
            )

    with timed_stage("load action text map", args.profile_startup):
        action_text_map = load_action_text_map(motion2text_json)

    with timed_stage("read/discover train split", args.profile_startup):
        train_split_names = read_split_file(args.train_split_file)
        train_names = discover_names(
            motion_token_dir,
            audio_feat_dir,
            train_split_names,
        )

    with timed_stage("load train sequences", args.profile_startup):
        train_sequences = load_sequences(
            train_names,
            motion_token_dir,
            audio_feat_dir,
            action_text_map,
            max_samples=args.max_samples,
            audio_fps=args.audio_fps,
            motion_fps=args.motion_fps,
            motion_token_fps=args.motion_token_fps,
            motion_token_unit_length=motion_token_unit_length,
        )

    eval_sequences = None
    with timed_stage("read/load eval split", args.profile_startup):
        eval_split_names = read_split_file(args.eval_split_file)
        if eval_split_names is not None:
            eval_names = discover_names(
                motion_token_dir,
                audio_feat_dir,
                eval_split_names,
            )
            eval_sequences = load_sequences(
                eval_names,
                motion_token_dir,
                audio_feat_dir,
                action_text_map,
                max_samples=args.max_samples,
                audio_fps=args.audio_fps,
                motion_fps=args.motion_fps,
                motion_token_fps=args.motion_token_fps,
                motion_token_unit_length=motion_token_unit_length,
            )
        else:
            train_sequences, eval_sequences = split_train_eval(
                train_sequences,
                eval_ratio=args.eval_ratio,
                seed=args.seed,
            )

    with timed_stage("build train FIM windows", args.profile_startup):
        train_dataset = AudioFIMCausalDataset(
            train_sequences,
            step=args.step,
            audio_fps=args.audio_fps,
            motion_fps=default_motion_token_fps,
            min_history_frames=args.min_history_frames,
            max_history_frames=args.max_history_frames,
            max_windows_per_sequence=args.max_windows_per_sequence,
            seed=args.seed,
        )

    eval_dataset = None
    if eval_sequences:
        with timed_stage("build eval FIM windows", args.profile_startup):
            eval_dataset = AudioFIMCausalDataset(
                eval_sequences,
                step=args.step,
                audio_fps=args.audio_fps,
                motion_fps=default_motion_token_fps,
                min_history_frames=args.min_history_frames,
                max_history_frames=args.max_history_frames,
                max_windows_per_sequence=args.max_windows_per_sequence,
                seed=args.seed + 1,
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No training windows were built. Check data paths/splits.")

    config = AudioFIMCausalConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        audio_feat_dim=args.audio_feat_dim,
        audio_fusion=args.audio_fusion,
        audio_gate_init=args.audio_gate_init,
        max_gap_frames=args.max_gap_frames,
        dropout=args.dropout,
    )
    model = AudioFIMCausalLM(config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params, trainable_params = count_parameters(model)
    default_run_name = (
        f"audio_fim_causal_step{args.step}_"
        f"afps{format_fps_for_dir(args.audio_fps)}_"
        f"mtokfps{format_fps_for_dir(default_motion_token_fps)}"
    )
    wandb_run_name = configure_wandb(args, default_run_name)

    print("=" * 70)
    print("Step 2 compact AudioFIM causal training")
    print(f"Output dir:       {args.output_dir}")
    print(f"Motion tokens:    {motion_token_dir}")
    if args.report_to == "wandb" and args.wandb_log_eval_motion_videos:
        print(f"Motion data:      {motion_data_dir}")
        print(f"RVQVAE decode:    {args.rvqvae_ckpt}")
    print(f"Audio features:   {audio_feat_dir}")
    print(f"Audio FPS:        {args.audio_fps:g}")
    print(f"Source motion FPS fallback: {args.motion_fps:g}")
    print(
        "Motion token unit length: "
        f"{motion_token_unit_length:g} ({motion_token_unit_length_source})"
    )
    if args.motion_token_fps is None:
        print(
            "Motion token FPS: "
            f"derived per sequence from json/source fps / {motion_token_unit_length:g} "
            f"(fallback {default_motion_token_fps:g})"
        )
    else:
        print(f"Motion token FPS: {default_motion_token_fps:g} (cli override)")
    print(
        "Audio/token ratio fallback: "
        f"{float(args.audio_fps) / float(default_motion_token_fps):.4g}"
    )
    print(f"Train sequences:  {len(train_sequences)}")
    print(f"Train windows:    {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval windows:     {len(eval_dataset)}")
    print(f"Gap setup:        step={args.step}, predict={args.step - 1} frames")
    print(f"History frames:   {args.min_history_frames}-{args.max_history_frames}")
    print(
        "Self-forcing:    "
        f"{format_self_forcing_schedule(self_forcing_schedule)}"
    )
    print(
        "RVQ curriculum:  "
        f"{format_rvq_curriculum_schedule(rvq_curriculum_schedule)}"
    )
    if args.rvq_curriculum_mask_inactive_inputs:
        print("RVQ curr inputs: mask inactive supervised targets")
    print(
        "Conditioning:     "
        f"audio_drop={args.audio_condition_dropout_prob:g}, "
        f"history_drop={args.motion_history_dropout_prob:g}, "
        f"anchor_token_drop={args.motion_anchor_token_dropout_prob:g}, "
        f"eval_cfg={args.eval_audio_cfg_scale:g}"
    )
    print(
        "Architecture:     "
        f"L={config.num_layers}, H={config.hidden_size}, "
        f"heads={config.num_heads}, ffn={config.intermediate_size}, "
        f"vocab={config.vocab_size}"
    )
    print(f"Arch preset:      {active_architecture_preset or 'custom'}")
    print(f"Audio fusion:     {config.audio_fusion}")
    if config.audio_fusion == "gated_layers":
        print(f"Audio gate init:  {config.audio_gate_init:g}")
    print(f"Parameters:       {total_params:,} total / {trainable_params:,} trainable")
    print(f"Tie embeddings:   {config.tie_word_embeddings}")
    print(
        "Shared emb/head:  "
        f"{model.embed_tokens.weight.data_ptr() == model.out_head.weight.data_ptr()}"
    )
    print(f"Report to:        {args.report_to}")
    if wandb_run_name:
        print(f"W&B run:          {wandb_run_name}")
    if args.report_to == "wandb" and args.wandb_log_eval_videos:
            print(f"W&B token video:  every {args.wandb_video_every_n_evals} eval(s)")
    if args.report_to == "wandb" and args.wandb_log_eval_motion_videos:
        print(
            "W&B motion video: "
            f"{args.wandb_motion_video_examples} sample(s), "
            f"{args.wandb_motion_video_fps} fps, "
            f"every {args.wandb_motion_video_every_n_evals} eval(s)"
        )
    if eval_dataset is not None and (
        args.eval_metric_examples > 0 or args.eval_gen_metric_examples > 0
    ):
        print(
            "Eval metrics:     "
            f"teacher={args.eval_metric_examples} window(s), "
            f"gen={args.eval_gen_metric_examples} window(s), "
            f"cfg={args.eval_audio_cfg_scale:g}"
        )
    print("=" * 70)

    collator = AudioFIMCausalCollator(
        config,
        max_length=args.max_length,
        debug_examples=args.debug_examples,
        profile_batches=args.profile_collator_batches,
    )

    if args.debug_loss_sanity > 0:
        run_loss_sanity_check(
            model,
            train_dataset,
            collator,
            num_examples=args.debug_loss_sanity,
            use_bf16=args.bf16,
            use_fp16=args.fp16,
        )

    training_args_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": args.report_to,
    }
    if wandb_run_name:
        training_args_kwargs["run_name"] = wandb_run_name

    if eval_dataset is not None and len(eval_dataset) > 0:
        training_args_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": args.eval_steps,
            }
        )

    with timed_stage("create TrainingArguments", args.profile_startup):
        training_args = TrainingArguments(**training_args_kwargs)

    callbacks = []
    if eval_dataset is not None and len(eval_dataset) > 0 and (
        args.eval_metric_examples > 0 or args.eval_gen_metric_examples > 0
    ):
        callbacks.append(
            AudioFIMEvalMetricsCallback(
                eval_dataset=eval_dataset,
                collator=collator,
                config=config,
                teacher_forced_examples=args.eval_metric_examples,
                teacher_forced_batch_size=args.eval_metric_batch_size,
                teacher_forced_every_n_evals=args.eval_metric_every_n_evals,
                generation_examples=args.eval_gen_metric_examples,
                generation_every_n_evals=args.eval_gen_metric_every_n_evals,
                audio_cfg_scale=args.eval_audio_cfg_scale,
            )
        )
    if (
        args.report_to == "wandb"
        and args.wandb_log_eval_videos
        and eval_dataset is not None
        and len(eval_dataset) > 0
    ):
        callbacks.append(
            WandbEvalVideoCallback(
                eval_dataset=eval_dataset,
                num_examples=args.wandb_video_examples,
                every_n_evals=args.wandb_video_every_n_evals,
                fps=args.wandb_video_fps,
                audio_cfg_scale=args.eval_audio_cfg_scale,
            )
        )
    if (
        args.report_to == "wandb"
        and args.wandb_log_eval_motion_videos
        and eval_dataset is not None
        and len(eval_dataset) > 0
    ):
        callbacks.append(
            WandbEvalMotionVideoCallback(
                eval_dataset=eval_dataset,
                motion_data_dir=motion_data_dir,
                rvqvae_ckpt=Path(args.rvqvae_ckpt),
                mean_path=Path(args.rvqvae_mean_path),
                std_path=Path(args.rvqvae_std_path),
                num_examples=args.wandb_motion_video_examples,
                every_n_evals=args.wandb_motion_video_every_n_evals,
                fps=args.wandb_motion_video_fps,
                audio_cfg_scale=args.eval_audio_cfg_scale,
            )
        )

    with timed_stage("create Trainer", args.profile_startup):
        trainer = AudioFIMSelfForcingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
            self_forcing_schedule=self_forcing_schedule,
            rvq_curriculum_schedule=rvq_curriculum_schedule,
            rvq_curriculum_mask_inactive_inputs=(
                args.rvq_curriculum_mask_inactive_inputs
            ),
            audio_condition_dropout_prob=args.audio_condition_dropout_prob,
            motion_history_dropout_prob=args.motion_history_dropout_prob,
            motion_anchor_token_dropout_prob=args.motion_anchor_token_dropout_prob,
        )

    with timed_stage("trainer.train", True):
        trainer.train()

    with timed_stage("save model/token map", True):
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero():
            mapper = AudioFIMTokenMapper(config)
            mapper.save_json(Path(args.output_dir) / "compact_token_map.json")

    print(f"Saved compact AudioFIM checkpoint to: {args.output_dir}")
    print(f"[Timing] total script runtime: {time.perf_counter() - run_start:.3f}s")


if __name__ == "__main__":
    main()

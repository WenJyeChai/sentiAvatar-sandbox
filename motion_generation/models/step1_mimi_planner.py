"""Causal audio-token-conditioned Qwen planner and fixed-gap Step 1 dataset.

The historical class names retain ``Mimi`` for checkpoint compatibility, while
the runtime audio contract now supports both Mimi and MOSS Nano tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from utils.adaptive_anchor_tokens import (
    ACTION_MISSING_TOKEN,
    ACTION_TOKEN,
    ANCHOR_TOKEN,
    AUDIO_END_TOKEN,
    BODY_CODEBOOK_SIZE,
    BODY_PART_ORDER,
    BODY_SLOT_COUNT,
    EXPRESSION_MISSING_TOKEN,
    EXPRESSION_TOKEN,
    GAP_TOKENS,
    MIMI_FRAME_TOKEN,
    MOTION_END_TOKEN,
    MOTION_START_TOKEN,
    SEED_TOKEN_BY_MODE,
    STEP1_ROLE_TOKEN,
    TRANSCRIPT_TOKEN,
    body_token,
    causal_audio_boundaries,
    fixed_anchor_times,
    gap_from_anchor_times,
    validate_anchor,
    validate_motion_payload,
)
from utils.step1_adaptive_schedule import (
    GapCurriculumPhase,
    GapSchedule,
    load_calibration,
    phase_for_epoch,
    random_curriculum_schedule,
)


IGNORE_INDEX = -100
MIMI_FRAME_RATE = 12.5
MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_SIZE = 1_920
MIMI_CARDINALITY = 2_048
MIMI_STORED_CODEBOOKS = 8
MOTION_TOKEN_FPS = 10.0
STRUCTURED_TAG_PATTERN = re.compile(r"【\s*(表情|动作)\s*[:：]\s*(.*?)\s*】")


def canonical_data_path(root: Path, name: str, suffix: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return root / Path(*parts).with_suffix(suffix)


def read_split_names(path: Path) -> list[str]:
    names = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        raise ValueError(f"Split contains duplicate names: {path}")
    return names


def load_text_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(name).replace("\\", "/"): str(text) for name, text in payload.items()}


def load_audio_tokens(
    path: Path,
    *,
    codec: str,
    sample_rate: int,
    frame_rate: float,
    frame_size: int,
    stored_codebooks: int,
    cardinality: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        codes = np.asarray(payload["codes"])
        result = {key: payload[key].item() for key in payload.files if key != "codes"}
    if codes.ndim != 2 or codes.shape[0] != int(stored_codebooks):
        raise ValueError(
            f"Expected {codec} codes [{stored_codebooks}, T], got {codes.shape} in {path}"
        )
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"{codec} codes must be integer, got {codes.dtype} in {path}")
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= int(cardinality)):
        raise ValueError(f"{codec} code outside [0, {cardinality - 1}] in {path}")
    expected = {
        "sample_rate": int(sample_rate),
        "frame_size": int(frame_size),
        "num_codebooks": int(stored_codebooks),
        "cardinality": int(cardinality),
    }
    for key, value in expected.items():
        if int(result.get(key, -1)) != value:
            raise ValueError(
                f"{codec} metadata {key}={result.get(key)}; expected {value} in {path}"
            )
    if not math.isclose(float(result.get("frame_rate", -1)), float(frame_rate)):
        raise ValueError(
            f"{codec} frame rate {result.get('frame_rate')}; expected {frame_rate} in {path}"
        )
    result["codes"] = codes.astype(np.int64, copy=False)
    return result


def load_mimi_tokens(path: Path) -> dict[str, Any]:
    """Backward-compatible loader for existing Mimi checkpoints and tests."""

    return load_audio_tokens(
        path,
        codec="mimi",
        sample_rate=MIMI_SAMPLE_RATE,
        frame_rate=MIMI_FRAME_RATE,
        frame_size=MIMI_FRAME_SIZE,
        stored_codebooks=MIMI_STORED_CODEBOOKS,
        cardinality=MIMI_CARDINALITY,
    )


@dataclass(frozen=True)
class StructuredText:
    expression: Optional[str]
    action: Optional[str]
    transcript: str

    @property
    def annotation_pattern(self) -> str:
        if self.expression and self.action:
            return "expression+action"
        if self.expression:
            return "expression-only"
        if self.action:
            return "action-only"
        return "no-tags"


def parse_structured_text(raw_text: str) -> StructuredText:
    """Extract optional Chinese expression/action tags without conflating absence.

    Explicit values such as ``【动作：无动作】`` remain present action
    annotations. Repeated annotations are retained in source order and joined
    with a Chinese semicolon.
    """

    values: dict[str, list[str]] = {"表情": [], "动作": []}
    for match in STRUCTURED_TAG_PATTERN.finditer(str(raw_text)):
        kind, value = match.groups()
        value = " ".join(value.split()).strip()
        if value and value not in values[kind]:
            values[kind].append(value)
    transcript = STRUCTURED_TAG_PATTERN.sub("", str(raw_text))
    transcript = " ".join(transcript.split()).strip()
    return StructuredText(
        expression="；".join(values["表情"]) or None,
        action="；".join(values["动作"]) or None,
        transcript=transcript,
    )


def load_motion_tokens(path: Path, require_causal: bool = True) -> tuple[list[list[int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    tokens = validate_motion_payload(payload, require_causal=require_causal)
    return tokens, payload


def load_generated_anchors(path: Path) -> dict[int, tuple[int, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("anchors"), list):
        pairs = ((entry["time"], entry["tokens"]) for entry in payload["anchors"])
    elif isinstance(payload, dict) and isinstance(payload.get("anchor_tokens_by_time"), dict):
        pairs = payload["anchor_tokens_by_time"].items()
    else:
        raise ValueError(
            f"Generated-prefix file must contain 'anchors' or 'anchor_tokens_by_time': {path}"
        )
    result: dict[int, tuple[int, ...]] = {}
    for raw_time, raw_anchor in pairs:
        anchor = tuple(int(value) for value in raw_anchor)
        validate_anchor(anchor)
        result[int(raw_time)] = anchor
    return result


def deterministic_choice(probability: float, *, seed: int, epoch: int, name: str, time: int) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    key = f"{seed}|{epoch}|{name}|{time}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return value < probability


@dataclass
class Step1Sequence:
    name: str
    input_ids: list[int]
    audio_codes: list[list[int]]
    target_slots: list[int]
    motion_local_labels: list[int]
    text_mask: list[int]
    expression_mask: list[int]
    action_mask: list[int]
    transcript_mask: list[int]
    audio_anchor_ids: list[int]
    target_anchor_ids: list[int]
    anchor_times: tuple[int, ...]
    audio_boundaries: tuple[int, ...]
    generated_prefix_anchors: int
    annotation_pattern: str
    gap_target_probs: list[list[float]]
    gap_target_mask: list[int]
    gap_loss_weight: float
    normal_gaps: tuple[int, ...]
    tail_gap: Optional[int]


class Step1FixedGapDataset(Dataset):
    """One causal, interleaved planner sequence per utterance.

    Text is fully visible at the beginning.  A runtime gap control is followed
    by only the new causal audio-token frames for that interval, then a 16-slot anchor.
    Only anchor slots are supervised in Phase 1.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        tokenizer: Any,
        motion_token_dir: Path,
        mimi_token_dir: Path,
        text_map: Mapping[str, str],
        fixed_gap: int = 3,
        max_length: int = 2_048,
        seed_mode: str = "observed",
        neutral_seed_tokens: Optional[Sequence[int]] = None,
        neutral_seed_probability: float = 0.5,
        previous_seed_probability: float = 0.5,
        generated_anchor_dir: Optional[Path] = None,
        generated_prefix_probability: float = 0.0,
        random_seed: int = 42,
        require_causal_motion: bool = True,
        max_duration_mismatch_seconds: float = 0.12,
        mimi_codebooks_used: Optional[Sequence[int]] = None,
        audio_codec: str = "mimi",
        audio_sample_rate: int = MIMI_SAMPLE_RATE,
        audio_frame_rate: float = MIMI_FRAME_RATE,
        audio_frame_size: int = MIMI_FRAME_SIZE,
        audio_codebooks_stored: int = MIMI_STORED_CODEBOOKS,
        audio_cardinality: int = MIMI_CARDINALITY,
        audio_codebooks_used: Optional[Sequence[int]] = None,
        text_serialization: str = "raw",
        drop_structured_tags: bool = False,
    ) -> None:
        if seed_mode not in {"observed", "previous", "neutral", "mixed_known", "mixed_all"}:
            raise ValueError(
                "Training seed_mode must be observed, previous, neutral, mixed_known, or mixed_all"
            )
        if seed_mode in {"neutral", "mixed_all"}:
            if neutral_seed_tokens is None:
                raise ValueError(f"seed_mode={seed_mode} requires neutral_seed_tokens")
            validate_anchor(neutral_seed_tokens)
        if not 0 <= neutral_seed_probability <= 1:
            raise ValueError("neutral_seed_probability must be in [0, 1]")
        if not 0 <= previous_seed_probability <= 1:
            raise ValueError("previous_seed_probability must be in [0, 1]")
        if not 0 <= generated_prefix_probability <= 1:
            raise ValueError("generated_prefix_probability must be in [0, 1]")
        self.names = [str(name).replace("\\", "/") for name in names]
        self.tokenizer = tokenizer
        self.motion_token_dir = Path(motion_token_dir)
        self.mimi_token_dir = Path(mimi_token_dir)
        self.text_map = dict(text_map)
        self.fixed_gap = int(fixed_gap)
        self.max_length = int(max_length)
        self.seed_mode = seed_mode
        self.neutral_seed_tokens = tuple(int(v) for v in neutral_seed_tokens) if neutral_seed_tokens else None
        self.neutral_seed_probability = float(neutral_seed_probability)
        self.previous_seed_probability = float(previous_seed_probability)
        self.generated_anchor_dir = Path(generated_anchor_dir) if generated_anchor_dir else None
        self.generated_prefix_probability = float(generated_prefix_probability)
        self.random_seed = int(random_seed)
        self.require_causal_motion = bool(require_causal_motion)
        self.max_duration_mismatch_seconds = float(max_duration_mismatch_seconds)
        self.audio_codec = str(audio_codec)
        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_frame_rate = float(audio_frame_rate)
        self.audio_frame_size = int(audio_frame_size)
        self.audio_codebooks_stored = int(audio_codebooks_stored)
        self.audio_cardinality = int(audio_cardinality)
        selected_codebooks = audio_codebooks_used
        if selected_codebooks is None:
            selected_codebooks = mimi_codebooks_used or [0]
        self.audio_codebooks_used = tuple(int(value) for value in selected_codebooks)
        # Legacy alias used by rollout/evaluation helpers.
        self.mimi_codebooks_used = self.audio_codebooks_used
        if not self.audio_codebooks_used:
            raise ValueError("At least one audio codebook must be used")
        if len(set(self.audio_codebooks_used)) != len(self.audio_codebooks_used):
            raise ValueError("Audio codebook indices must be unique")
        if any(not 0 <= value < self.audio_codebooks_stored for value in self.audio_codebooks_used):
            raise ValueError(
                f"Audio codebook indices must be in [0, {self.audio_codebooks_stored - 1}]"
            )
        if self.audio_sample_rate <= 0 or self.audio_frame_rate <= 0 or self.audio_frame_size <= 0:
            raise ValueError("Audio sample rate, frame rate, and frame size must be positive")
        if self.audio_codebooks_stored <= 0 or self.audio_cardinality <= 1:
            raise ValueError("Audio codebook count/cardinality is invalid")
        if text_serialization not in {"raw", "structured_fields"}:
            raise ValueError("text_serialization must be raw or structured_fields")
        self.text_serialization = str(text_serialization)
        self.drop_structured_tags = bool(drop_structured_tags)
        self.epoch = 0
        self._single_token_ids = self._build_single_token_ids()

    def _build_single_token_ids(self) -> dict[str, int]:
        tokens = {
            STEP1_ROLE_TOKEN,
            MOTION_START_TOKEN,
            MOTION_END_TOKEN,
            ANCHOR_TOKEN,
            MIMI_FRAME_TOKEN,
            AUDIO_END_TOKEN,
            *GAP_TOKENS,
            *SEED_TOKEN_BY_MODE.values(),
        }
        if self.text_serialization == "structured_fields":
            tokens.update(
                {
                    EXPRESSION_TOKEN,
                    EXPRESSION_MISSING_TOKEN,
                    ACTION_TOKEN,
                    ACTION_MISSING_TOKEN,
                    TRANSCRIPT_TOKEN,
                }
            )
        result = {}
        for token in tokens:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Step 1 control {token} is not one tokenizer id: {encoded}")
            result[token] = int(encoded[0])
        return result

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _anchor_schedule(
        self,
        name: str,
        motion_tokens: Sequence[Sequence[int]],
    ) -> GapSchedule:
        anchor_times = fixed_anchor_times(len(motion_tokens), gap=self.fixed_gap)
        gaps = tuple(
            gap_from_anchor_times(left, right)
            for left, right in zip(anchor_times[:-1], anchor_times[1:])
        )
        normal = tuple(gap for gap in gaps if gap >= 3)
        tail = gaps[-1] if gaps and gaps[-1] <= 2 else None
        return GapSchedule(anchor_times, {}, normal, tail, 0.0)

    def __len__(self) -> int:
        return len(self.names)

    def _select_seed(self, name: str, observed: Sequence[int]) -> tuple[str, tuple[int, ...]]:
        if self.seed_mode == "observed":
            return "observed", tuple(int(v) for v in observed)
        if self.seed_mode == "previous":
            return "previous", tuple(int(v) for v in observed)
        if self.seed_mode == "neutral":
            assert self.neutral_seed_tokens is not None
            return "neutral", self.neutral_seed_tokens
        if self.seed_mode == "mixed_all":
            use_neutral = deterministic_choice(
                self.neutral_seed_probability,
                seed=self.random_seed,
                epoch=self.epoch,
                name=name,
                time=-1,
            )
            if use_neutral:
                assert self.neutral_seed_tokens is not None
                return "neutral", self.neutral_seed_tokens
        use_previous = deterministic_choice(
            self.previous_seed_probability,
            seed=self.random_seed,
            epoch=self.epoch,
            name=name,
            time=-2,
        )
        mode = "previous" if use_previous else "observed"
        return mode, tuple(int(v) for v in observed)

    def _generated_anchors(self, name: str) -> dict[int, tuple[int, ...]]:
        if self.generated_anchor_dir is None or self.generated_prefix_probability <= 0:
            return {}
        path = canonical_data_path(self.generated_anchor_dir, name, ".json")
        if not path.is_file():
            raise FileNotFoundError(
                f"Generated-prefix probability is positive but cache is missing: {path}"
            )
        return load_generated_anchors(path)

    def _build_text_prefix(
        self,
        raw_text: str,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], str]:
        parsed = parse_structured_text(raw_text)
        if self.text_serialization == "raw":
            prompt = f"Human: {STEP1_ROLE_TOKEN}{raw_text}<|im_end|>\nAssistant:"
            input_ids = [int(value) for value in self.tokenizer.encode(prompt, add_special_tokens=False)]
            role_id = self._single_token_ids[STEP1_ROLE_TOKEN]
            im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if len(im_end_ids) != 1:
                raise ValueError("<|im_end|> must be a single token")
            try:
                role_position = input_ids.index(role_id)
                text_end = input_ids.index(int(im_end_ids[0]), role_position + 1)
            except ValueError as error:
                raise ValueError("Could not locate the transcript span in the Step 1 prompt") from error
            text_mask = [
                int(role_position < position < text_end)
                for position in range(len(input_ids))
            ]
            zeros = [0] * len(input_ids)
            return (
                input_ids,
                text_mask,
                zeros.copy(),
                zeros.copy(),
                text_mask.copy(),
                parsed.annotation_pattern,
            )

        input_ids: list[int] = []
        expression_mask: list[int] = []
        action_mask: list[int] = []
        transcript_mask: list[int] = []

        def append_ids(ids: Sequence[int], field: Optional[str] = None) -> None:
            values = [int(value) for value in ids]
            input_ids.extend(values)
            expression_mask.extend([int(field == "expression")] * len(values))
            action_mask.extend([int(field == "action")] * len(values))
            transcript_mask.extend([int(field == "transcript")] * len(values))

        def append_text(text: str, field: Optional[str] = None) -> None:
            append_ids(self.tokenizer.encode(text, add_special_tokens=False), field)

        def append_control(token: str) -> None:
            append_ids([self._single_token_ids[token]])

        expression = None if self.drop_structured_tags else parsed.expression
        action = None if self.drop_structured_tags else parsed.action
        append_text("Human: ")
        append_control(STEP1_ROLE_TOKEN)
        append_text("\n")
        append_control(EXPRESSION_TOKEN if expression else EXPRESSION_MISSING_TOKEN)
        if expression:
            append_text(f" {expression}", field="expression")
        append_text("\n")
        append_control(ACTION_TOKEN if action else ACTION_MISSING_TOKEN)
        if action:
            append_text(f" {action}", field="action")
        append_text("\n")
        append_control(TRANSCRIPT_TOKEN)
        if parsed.transcript:
            append_text(f" {parsed.transcript}", field="transcript")
        append_text("<|im_end|>\nAssistant:")
        text_mask = [
            int(bool(expression_value or action_value or transcript_value))
            for expression_value, action_value, transcript_value in zip(
                expression_mask,
                action_mask,
                transcript_mask,
            )
        ]
        return (
            input_ids,
            text_mask,
            expression_mask,
            action_mask,
            transcript_mask,
            parsed.annotation_pattern,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence = self.build_sequence(self.names[index])
        return {
            "name": sequence.name,
            "input_ids": sequence.input_ids,
            "audio_codes": sequence.audio_codes,
            "target_slots": sequence.target_slots,
            "motion_local_labels": sequence.motion_local_labels,
            "text_mask": sequence.text_mask,
            "expression_mask": sequence.expression_mask,
            "action_mask": sequence.action_mask,
            "transcript_mask": sequence.transcript_mask,
            "audio_anchor_ids": sequence.audio_anchor_ids,
            "target_anchor_ids": sequence.target_anchor_ids,
            "anchor_times": sequence.anchor_times,
            "audio_boundaries": sequence.audio_boundaries,
            "generated_prefix_anchors": sequence.generated_prefix_anchors,
            "annotation_pattern": sequence.annotation_pattern,
            "gap_target_probs": sequence.gap_target_probs,
            "gap_target_mask": sequence.gap_target_mask,
            "gap_loss_weight": sequence.gap_loss_weight,
            "normal_gaps": sequence.normal_gaps,
            "tail_gap": sequence.tail_gap,
        }

    def build_sequence(self, name: str) -> Step1Sequence:
        motion_path = canonical_data_path(self.motion_token_dir, name, ".json")
        audio_path = canonical_data_path(self.mimi_token_dir, name, ".npz")
        motion_tokens, _ = load_motion_tokens(motion_path, require_causal=self.require_causal_motion)
        audio_payload = load_audio_tokens(
            audio_path,
            codec=self.audio_codec,
            sample_rate=self.audio_sample_rate,
            frame_rate=self.audio_frame_rate,
            frame_size=self.audio_frame_size,
            stored_codebooks=self.audio_codebooks_stored,
            cardinality=self.audio_cardinality,
        )
        audio_token_codes = np.asarray(audio_payload["codes"])[list(self.audio_codebooks_used)]
        if name not in self.text_map:
            raise KeyError(f"Missing text annotation for {name}")

        audio_duration = int(audio_payload["num_samples"]) / self.audio_sample_rate
        motion_duration = len(motion_tokens) / MOTION_TOKEN_FPS
        mismatch = abs(audio_duration - motion_duration)
        if mismatch > self.max_duration_mismatch_seconds:
            raise ValueError(
                f"Audio/motion duration mismatch for {name}: audio={audio_duration:.4f}s, "
                f"motion={motion_duration:.4f}s, error={mismatch:.4f}s"
            )

        schedule = self._anchor_schedule(name, motion_tokens)
        anchor_times = schedule.anchor_times
        audio_boundaries = causal_audio_boundaries(
            anchor_times,
            audio_frames=audio_token_codes.shape[1],
            audio_fps=self.audio_frame_rate,
            motion_fps=MOTION_TOKEN_FPS,
        )
        if len(anchor_times) != len(audio_boundaries):
            raise AssertionError("Anchor/audio boundary count mismatch")
        generated = self._generated_anchors(name)

        (
            input_ids,
            text_mask,
            expression_mask,
            action_mask,
            transcript_mask,
            annotation_pattern,
        ) = self._build_text_prefix(self.text_map[name])
        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(im_end_ids) != 1:
            raise ValueError("<|im_end|> must be a single token")
        empty_audio = [-1] * len(self.audio_codebooks_used)
        audio_codes = [empty_audio.copy() for _ in input_ids]
        target_slots = [-1] * len(input_ids)
        motion_local_labels = [IGNORE_INDEX] * len(input_ids)
        audio_anchor_ids = [-1] * len(input_ids)
        target_anchor_ids = [-1] * len(input_ids)
        gap_target_probs = [[0.0] * len(GAP_TOKENS) for _ in input_ids]
        gap_target_mask = [0] * len(input_ids)

        def append_control(token: str) -> None:
            input_ids.append(self._single_token_ids[token])
            audio_codes.append(empty_audio.copy())
            target_slots.append(-1)
            motion_local_labels.append(IGNORE_INDEX)
            text_mask.append(0)
            expression_mask.append(0)
            action_mask.append(0)
            transcript_mask.append(0)
            audio_anchor_ids.append(-1)
            target_anchor_ids.append(-1)
            gap_target_probs.append([0.0] * len(GAP_TOKENS))
            gap_target_mask.append(0)

        def append_anchor(
            input_anchor: Sequence[int],
            target_anchor: Optional[Sequence[int]],
            anchor_group: int = -1,
        ) -> None:
            validate_anchor(input_anchor)
            if target_anchor is not None:
                validate_anchor(target_anchor)
            append_control(ANCHOR_TOKEN)
            for slot, input_local_id in enumerate(input_anchor):
                token_id = self.tokenizer.convert_tokens_to_ids(body_token(slot, int(input_local_id)))
                if token_id is None:
                    raise ValueError(f"Tokenizer is missing body token for slot {slot}, id {input_local_id}")
                input_ids.append(int(token_id))
                audio_codes.append(empty_audio.copy())
                text_mask.append(0)
                expression_mask.append(0)
                action_mask.append(0)
                transcript_mask.append(0)
                audio_anchor_ids.append(-1)
                gap_target_probs.append([0.0] * len(GAP_TOKENS))
                gap_target_mask.append(0)
                if target_anchor is None:
                    target_slots.append(-1)
                    motion_local_labels.append(IGNORE_INDEX)
                    target_anchor_ids.append(-1)
                else:
                    target_slots.append(slot)
                    motion_local_labels.append(int(target_anchor[slot]))
                    target_anchor_ids.append(int(anchor_group))

        append_control(MOTION_START_TOKEN)
        selected_seed_mode, seed_anchor = self._select_seed(name, motion_tokens[0])
        append_control(SEED_TOKEN_BY_MODE[selected_seed_mode])
        append_anchor(seed_anchor, target_anchor=None)

        audio_cursor = 0
        generated_prefix_anchors = 0
        for anchor_index in range(1, len(anchor_times)):
            left_time = anchor_times[anchor_index - 1]
            target_time = anchor_times[anchor_index]
            gap = gap_from_anchor_times(left_time, target_time)
            append_control(GAP_TOKENS[gap])
            soft_target = schedule.soft_targets_by_left.get(left_time)
            if soft_target is not None:
                if len(soft_target) != len(GAP_TOKENS):
                    raise ValueError(
                        f"{name}: gap target at {left_time} has {len(soft_target)} "
                        f"entries, expected {len(GAP_TOKENS)}"
                    )
                gap_target_probs[-1] = [float(value) for value in soft_target]
                gap_target_mask[-1] = 1
            next_audio_boundary = audio_boundaries[anchor_index]
            for frame_codes in audio_token_codes[:, audio_cursor:next_audio_boundary].T:
                append_control(MIMI_FRAME_TOKEN)
                audio_codes[-1] = [int(code) for code in frame_codes]
                audio_anchor_ids[-1] = anchor_index - 1
            audio_cursor = next_audio_boundary

            gt_anchor = tuple(int(v) for v in motion_tokens[target_time])
            input_anchor = gt_anchor
            if target_time in generated and deterministic_choice(
                self.generated_prefix_probability,
                seed=self.random_seed,
                epoch=self.epoch,
                name=name,
                time=target_time,
            ):
                input_anchor = generated[target_time]
                generated_prefix_anchors += 1
            append_anchor(
                input_anchor,
                target_anchor=gt_anchor,
                anchor_group=anchor_index - 1,
            )

        if audio_cursor != audio_token_codes.shape[1]:
            raise AssertionError(
                f"Did not consume all {self.audio_codec} frames for {name}: "
                f"{audio_cursor}/{audio_token_codes.shape[1]}"
            )
        append_control(AUDIO_END_TOKEN)
        append_control(MOTION_END_TOKEN)
        input_ids.append(int(im_end_ids[0]))
        audio_codes.append(empty_audio.copy())
        target_slots.append(-1)
        motion_local_labels.append(IGNORE_INDEX)
        text_mask.append(0)
        expression_mask.append(0)
        action_mask.append(0)
        transcript_mask.append(0)
        audio_anchor_ids.append(-1)
        target_anchor_ids.append(-1)
        gap_target_probs.append([0.0] * len(GAP_TOKENS))
        gap_target_mask.append(0)

        lengths = {
            len(input_ids),
            len(audio_codes),
            len(target_slots),
            len(motion_local_labels),
            len(text_mask),
            len(expression_mask),
            len(action_mask),
            len(transcript_mask),
            len(audio_anchor_ids),
            len(target_anchor_ids),
            len(gap_target_probs),
            len(gap_target_mask),
        }
        if len(lengths) != 1:
            raise AssertionError(f"Serialized field lengths differ for {name}: {lengths}")
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Serialized sequence for {name} has {len(input_ids)} tokens, exceeding "
                f"max_length={self.max_length}. Do not silently truncate anchor supervision."
            )
        expected_targets = (len(anchor_times) - 1) * BODY_SLOT_COUNT
        actual_targets = sum(slot >= 0 for slot in target_slots)
        if actual_targets != expected_targets:
            raise AssertionError(f"Expected {expected_targets} anchor targets, got {actual_targets}")
        return Step1Sequence(
            name=name,
            input_ids=input_ids,
            audio_codes=audio_codes,
            target_slots=target_slots,
            motion_local_labels=motion_local_labels,
            text_mask=text_mask,
            expression_mask=expression_mask,
            action_mask=action_mask,
            transcript_mask=transcript_mask,
            audio_anchor_ids=audio_anchor_ids,
            target_anchor_ids=target_anchor_ids,
            anchor_times=anchor_times,
            audio_boundaries=audio_boundaries,
            generated_prefix_anchors=generated_prefix_anchors,
            annotation_pattern=annotation_pattern,
            gap_target_probs=gap_target_probs,
            gap_target_mask=gap_target_mask,
            gap_loss_weight=float(getattr(schedule, "schedule_loss_weight", 0.0)),
            normal_gaps=schedule.normal_gaps,
            tail_gap=schedule.tail_gap,
        )


class Step1AdaptiveGapDataset(Step1FixedGapDataset):
    """Step 1 dataset driven by pre-materialized frozen-Step-2 DP schedules."""

    def __init__(
        self,
        *args: Any,
        curriculum_phases: Sequence[GapCurriculumPhase],
        calibration_json: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.curriculum_phases = tuple(curriculum_phases)
        calibration = load_calibration(Path(calibration_json))
        raw_phases = calibration["phases"]
        self._phase_schedules: dict[int, dict[str, tuple[tuple[int, ...], dict[int, tuple[float, ...]]]]] = {}
        for entry in raw_phases:
            if not isinstance(entry, Mapping):
                raise ValueError("Calibration phase entries must be mappings")
            phase_index = int(entry["phase_index"])
            if not 0 <= phase_index < len(self.curriculum_phases):
                raise ValueError(f"Calibration has unknown phase index {phase_index}")
            configured_phase = self.curriculum_phases[phase_index]
            expected_contract = (
                configured_phase.min_gap,
                configured_phase.max_gap,
                configured_phase.target_mean_gap,
                configured_phase.temperature,
            )
            actual_contract = (
                int(entry["min_gap"]),
                int(entry["max_gap"]),
                float(entry["target_mean_gap"]),
                float(entry["temperature"]),
            )
            if actual_contract != expected_contract:
                raise ValueError(
                    f"Calibration phase {phase_index} contract {actual_contract} "
                    f"does not match config {expected_contract}"
                )
            schedule_file = Path(str(entry["schedule_file"]))
            if not schedule_file.is_absolute():
                schedule_file = Path(calibration_json).resolve().parent / schedule_file
            expected_hash = entry.get("schedule_file_sha256")
            if expected_hash:
                digest = hashlib.sha256()
                with schedule_file.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != str(expected_hash):
                    raise ValueError(
                        f"Adaptive schedule hash mismatch for {schedule_file}"
                    )
            self._phase_schedules[phase_index] = self._load_schedule_file(schedule_file)
        expected_dp_phases = {
            index
            for index, phase in enumerate(self.curriculum_phases)
            if phase.mode == "step2_dp"
        }
        if set(self._phase_schedules) != expected_dp_phases:
            raise ValueError(
                "Calibration DP phases do not match config: "
                f"calibration={sorted(self._phase_schedules)}, "
                f"config={sorted(expected_dp_phases)}"
            )

    @staticmethod
    def _load_schedule_file(
        path: Path,
    ) -> dict[str, tuple[tuple[int, ...], dict[int, tuple[float, ...]]]]:
        if not path.is_file():
            raise FileNotFoundError(f"Adaptive schedule file not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            names = [str(value) for value in payload["names"].tolist()]
            offsets = np.asarray(payload["offsets"], dtype=np.int64)
            anchors = np.asarray(payload["anchors"], dtype=np.int64)
            probabilities = np.asarray(payload["gap_target_probs"], dtype=np.float32)
            target_mask = np.asarray(payload["gap_target_mask"], dtype=np.bool_)
        if offsets.shape != (len(names) + 1,):
            raise ValueError(f"{path}: offsets do not match names")
        if probabilities.shape != (len(anchors), len(GAP_TOKENS)):
            raise ValueError(f"{path}: malformed gap_target_probs {probabilities.shape}")
        if target_mask.shape != (len(anchors),):
            raise ValueError(f"{path}: malformed gap_target_mask")
        result = {}
        for index, name in enumerate(names):
            start, end = int(offsets[index]), int(offsets[index + 1])
            clip_anchors = tuple(int(value) for value in anchors[start:end])
            targets = {
                int(clip_anchors[row - start]): tuple(
                    float(value) for value in probabilities[row]
                )
                for row in range(start, end)
                if bool(target_mask[row])
            }
            result[name.replace("\\", "/")] = (clip_anchors, targets)
        return result

    def _anchor_schedule(
        self,
        name: str,
        motion_tokens: Sequence[Sequence[int]],
    ) -> GapSchedule:
        phase_index, phase = phase_for_epoch(self.curriculum_phases, self.epoch)
        if phase.mode == "random":
            base = random_curriculum_schedule(
                len(motion_tokens),
                min_gap=phase.min_gap,
                max_gap=phase.max_gap,
                seed=self.random_seed,
                epoch=self.epoch,
                name=name,
            )
        else:
            if phase_index not in self._phase_schedules:
                raise KeyError(
                    f"Calibration does not contain DP schedules for phase {phase_index}"
                )
            try:
                anchor_times, soft_targets = self._phase_schedules[phase_index][name]
            except KeyError as error:
                raise KeyError(
                    f"Calibration phase {phase_index} has no schedule for {name}"
                ) from error
            if not anchor_times or anchor_times[0] != 0 or anchor_times[-1] != len(motion_tokens) - 1:
                raise ValueError(
                    f"{name}: cached schedule endpoints do not match T={len(motion_tokens)}"
                )
            gaps = tuple(
                gap_from_anchor_times(left, right)
                for left, right in zip(anchor_times[:-1], anchor_times[1:])
            )
            base = GapSchedule(
                anchor_times=anchor_times,
                soft_targets_by_left=soft_targets,
                normal_gaps=tuple(gap for gap in gaps if gap >= 3),
                tail_gap=gaps[-1] if gaps and gaps[-1] <= 2 else None,
                total_cost=0.0,
            )
        # GapSchedule is frozen and intentionally minimal. Attach the phase
        # weight on a tiny proxy object used only during serialization.
        return _WeightedGapSchedule(base, phase.loss_weight(self.epoch))


class _WeightedGapSchedule:
    def __init__(self, schedule: GapSchedule, schedule_loss_weight: float) -> None:
        self.anchor_times = schedule.anchor_times
        self.soft_targets_by_left = schedule.soft_targets_by_left
        self.normal_gaps = schedule.normal_gaps
        self.tail_gap = schedule.tail_gap
        self.total_cost = schedule.total_cost
        self.schedule_loss_weight = float(schedule_loss_weight)


class Step1PlannerCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        longest = max(len(example["input_ids"]) for example in examples)
        if self.pad_to_multiple_of > 1:
            longest = math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of

        def padded(key: str, fill: int) -> torch.Tensor:
            rows = []
            for example in examples:
                values = list(example[key])
                rows.append(values + [fill] * (longest - len(values)))
            return torch.tensor(rows, dtype=torch.long)

        def padded_optional(key: str, fill: int) -> torch.Tensor:
            rows = []
            for example in examples:
                values = list(example.get(key, [fill] * len(example["input_ids"])))
                rows.append(values + [fill] * (longest - len(values)))
            return torch.tensor(rows, dtype=torch.long)

        input_ids = padded("input_ids", self.pad_token_id)
        codebook_counts = {
            len(frame_codes)
            for example in examples
            for frame_codes in example["audio_codes"]
        }
        if len(codebook_counts) != 1:
            raise ValueError(f"Examples use inconsistent audio codebook counts: {codebook_counts}")
        codebook_count = codebook_counts.pop()
        padded_audio = []
        for example in examples:
            values = [list(map(int, frame_codes)) for frame_codes in example["audio_codes"]]
            values.extend([[-1] * codebook_count for _ in range(longest - len(values))])
            padded_audio.append(values)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
            "audio_codes": torch.tensor(padded_audio, dtype=torch.long),
            "target_slots": padded("target_slots", -1),
            "motion_local_labels": padded("motion_local_labels", IGNORE_INDEX),
            "text_mask": padded_optional("text_mask", 0).bool(),
            "expression_mask": padded_optional("expression_mask", 0).bool(),
            "action_mask": padded_optional("action_mask", 0).bool(),
            "transcript_mask": padded_optional("transcript_mask", 0).bool(),
            "audio_anchor_ids": padded_optional("audio_anchor_ids", -1),
            "target_anchor_ids": padded_optional("target_anchor_ids", -1),
            "names": [str(example["name"]) for example in examples],
            "annotation_patterns": [
                str(example.get("annotation_pattern", "unknown")) for example in examples
            ],
            "generated_prefix_anchors": torch.tensor(
                [int(example["generated_prefix_anchors"]) for example in examples], dtype=torch.long
            ),
            "gap_target_probs": torch.tensor(
                [
                    list(example.get("gap_target_probs", [[0.0] * len(GAP_TOKENS)] * len(example["input_ids"])))
                    + [[0.0] * len(GAP_TOKENS) for _ in range(longest - len(example["input_ids"]))]
                    for example in examples
                ],
                dtype=torch.float32,
            ),
            "gap_target_mask": padded_optional("gap_target_mask", 0).bool(),
            "gap_loss_weights": torch.tensor(
                [float(example.get("gap_loss_weight", 0.0)) for example in examples],
                dtype=torch.float32,
            ),
            "normal_gap_sum": torch.tensor(
                [sum(int(value) for value in example.get("normal_gaps", ())) for example in examples],
                dtype=torch.long,
            ),
            "normal_gap_count": torch.tensor(
                [len(example.get("normal_gaps", ())) for example in examples],
                dtype=torch.long,
            ),
            "tail_gap": torch.tensor(
                [
                    -1 if example.get("tail_gap") is None else int(example["tail_gap"])
                    for example in examples
                ],
                dtype=torch.long,
            ),
        }


class MimiQwenPlannerConfig(PretrainedConfig):
    model_type = "mimi_qwen_step1_planner"

    def __init__(
        self,
        *,
        language_model_config: Optional[dict[str, Any]] = None,
        audio_placeholder_id: int = -1,
        motion_token_ids: Optional[Sequence[Sequence[int]]] = None,
        mimi_cardinality: int = MIMI_CARDINALITY,
        mimi_codebooks_stored: int = MIMI_STORED_CODEBOOKS,
        mimi_codebooks_used: Optional[Sequence[int]] = None,
        audio_codec: str = "mimi",
        audio_sample_rate: int = MIMI_SAMPLE_RATE,
        audio_frame_rate: float = MIMI_FRAME_RATE,
        audio_frame_size: int = MIMI_FRAME_SIZE,
        audio_cardinality: Optional[int] = None,
        audio_codebooks_stored: Optional[int] = None,
        audio_codebooks_used: Optional[Sequence[int]] = None,
        gap_token_ids: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.language_model_config = language_model_config or {}
        self.audio_placeholder_id = int(audio_placeholder_id)
        self.motion_token_ids = [list(map(int, row)) for row in (motion_token_ids or [])]
        self.audio_codec = str(audio_codec)
        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_frame_rate = float(audio_frame_rate)
        self.audio_frame_size = int(audio_frame_size)
        self.audio_cardinality = int(
            mimi_cardinality if audio_cardinality is None else audio_cardinality
        )
        self.audio_codebooks_stored = int(
            mimi_codebooks_stored if audio_codebooks_stored is None else audio_codebooks_stored
        )
        selected_codebooks = audio_codebooks_used
        if selected_codebooks is None:
            selected_codebooks = mimi_codebooks_used or [0]
        self.audio_codebooks_used = list(map(int, selected_codebooks))
        # Preserve legacy config fields so existing rollout/evaluation code and
        # old Mimi checkpoints remain loadable.
        self.mimi_cardinality = self.audio_cardinality
        self.mimi_codebooks_stored = self.audio_codebooks_stored
        self.mimi_codebooks_used = list(self.audio_codebooks_used)
        self.gap_token_ids = list(map(int, gap_token_ids or []))


@dataclass
class MimiQwenPlannerOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    expected_distortion_loss: Optional[torch.Tensor] = None
    expected_distortion_count: Optional[torch.Tensor] = None
    correct: Optional[torch.Tensor] = None
    count: Optional[torch.Tensor] = None
    per_slot_correct: Optional[torch.Tensor] = None
    per_slot_count: Optional[torch.Tensor] = None
    per_example_loss_sum: Optional[torch.Tensor] = None
    per_example_correct: Optional[torch.Tensor] = None
    per_example_count: Optional[torch.Tensor] = None
    per_token_loss: Optional[torch.Tensor] = None
    gap_loss: Optional[torch.Tensor] = None
    gap_count: Optional[torch.Tensor] = None
    gap_correct: Optional[torch.Tensor] = None


class MimiQwenPlanner(PreTrainedModel):
    """Qwen with codebook-specific causal audio embeddings and slot CE."""

    config_class = MimiQwenPlannerConfig
    base_model_prefix = "language_model"

    def __init__(
        self,
        config: MimiQwenPlannerConfig,
        language_model: Optional[PreTrainedModel] = None,
    ) -> None:
        super().__init__(config)
        if language_model is None:
            language_config_dict = dict(config.language_model_config)
            model_type = language_config_dict.pop("model_type")
            language_config = AutoConfig.for_model(model_type, **language_config_dict)
            language_model = AutoModelForCausalLM.from_config(language_config)
        self.language_model = language_model
        hidden_size = int(language_model.config.hidden_size)
        embedding_dtype = language_model.get_input_embeddings().weight.dtype
        if not config.audio_codebooks_used:
            raise ValueError("At least one audio codebook must be configured")
        if config.audio_codebooks_used[0] != 0:
            raise ValueError("The first configured audio codebook must be q0")
        if len(set(config.audio_codebooks_used)) != len(config.audio_codebooks_used):
            raise ValueError("Configured audio codebooks must be unique")
        if any(
            not 0 <= value < config.audio_codebooks_stored
            for value in config.audio_codebooks_used
        ):
            raise ValueError("Configured audio codebook index is outside the stored streams")

        # Preserve the historical q0 parameter name so q0-only checkpoints remain loadable.
        self.audio_embedding = nn.Embedding(
            config.audio_cardinality, hidden_size, dtype=embedding_dtype
        )
        self.additional_audio_embeddings = nn.ModuleList(
            nn.Embedding(config.audio_cardinality, hidden_size, dtype=embedding_dtype)
            for _ in config.audio_codebooks_used[1:]
        )
        codebook_count = len(config.audio_codebooks_used)
        self.audio_fusion = (
            nn.Linear(
                codebook_count * hidden_size,
                hidden_size,
                bias=False,
                dtype=embedding_dtype,
            )
            if codebook_count > 1
            else nn.Identity()
        )
        initializer_range = float(getattr(language_model.config, "initializer_range", 0.02))
        nn.init.normal_(self.audio_embedding.weight, mean=0.0, std=initializer_range)
        for embedding in self.additional_audio_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=initializer_range)
        if isinstance(self.audio_fusion, nn.Linear):
            # Start as a variance-preserving average while leaving the fusion learnable.
            with torch.no_grad():
                self.audio_fusion.weight.zero_()
                scale = codebook_count ** -0.5
                identity = torch.eye(hidden_size, dtype=self.audio_fusion.weight.dtype)
                for index in range(codebook_count):
                    left = index * hidden_size
                    self.audio_fusion.weight[:, left : left + hidden_size].copy_(identity * scale)

        motion_token_ids = torch.tensor(config.motion_token_ids, dtype=torch.long)
        if tuple(motion_token_ids.shape) != (BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE):
            raise ValueError(
                f"motion_token_ids must be [{BODY_SLOT_COUNT}, {BODY_CODEBOOK_SIZE}], "
                f"got {tuple(motion_token_ids.shape)}"
            )
        self.register_buffer("motion_token_ids", motion_token_ids, persistent=True)
        gap_token_ids = torch.tensor(config.gap_token_ids, dtype=torch.long)
        if gap_token_ids.numel() not in {0, len(GAP_TOKENS)}:
            raise ValueError(
                f"gap_token_ids must be empty or contain {len(GAP_TOKENS)} ids"
            )
        self.register_buffer("gap_token_ids", gap_token_ids, persistent=False)
        # This table is derived from frozen codec checkpoints at training time.
        # It is deliberately excluded from planner checkpoints (~16 MiB fp32).
        self.register_buffer(
            "motion_codebook_distances",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        if not 0 <= config.audio_placeholder_id < language_model.config.vocab_size:
            raise ValueError("audio_placeholder_id is outside the Qwen vocabulary")
        self.post_init()

    def set_motion_codebook_distances(self, distances: torch.Tensor) -> None:
        expected = (BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE, BODY_CODEBOOK_SIZE)
        distances = torch.as_tensor(distances, dtype=torch.float32)
        if tuple(distances.shape) != expected:
            raise ValueError(f"motion codebook distances must have shape {expected}")
        if not torch.isfinite(distances).all() or bool((distances < 0).any()):
            raise ValueError("motion codebook distances must be finite and non-negative")
        diagonal = distances.diagonal(dim1=-2, dim2=-1)
        if not torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6):
            raise ValueError("motion codebook distance diagonals must be zero")
        self.motion_codebook_distances = distances.contiguous()

    def set_gap_token_ids(self, token_ids: Sequence[int]) -> None:
        values = torch.as_tensor(token_ids, dtype=torch.long)
        if tuple(values.shape) != (len(GAP_TOKENS),):
            raise ValueError(f"Expected {len(GAP_TOKENS)} gap token ids")
        if int(values.min()) < 0 or int(values.max()) >= self.language_model.config.vocab_size:
            raise ValueError("Gap token id is outside the language-model vocabulary")
        self.gap_token_ids = values.to(self.motion_token_ids.device)
        self.config.gap_token_ids = [int(value) for value in values.tolist()]

    @classmethod
    def from_qwen_pretrained(
        cls,
        model_path: str | Path,
        *,
        audio_placeholder_id: int,
        motion_token_ids: Sequence[Sequence[int]],
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
        mimi_codebooks_used: Optional[Sequence[int]] = None,
        audio_codec: str = "mimi",
        audio_sample_rate: int = MIMI_SAMPLE_RATE,
        audio_frame_rate: float = MIMI_FRAME_RATE,
        audio_frame_size: int = MIMI_FRAME_SIZE,
        audio_cardinality: int = MIMI_CARDINALITY,
        audio_codebooks_stored: int = MIMI_STORED_CODEBOOKS,
        audio_codebooks_used: Optional[Sequence[int]] = None,
        gap_token_ids: Optional[Sequence[int]] = None,
    ) -> "MimiQwenPlanner":
        language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        config = MimiQwenPlannerConfig(
            language_model_config=language_model.config.to_dict(),
            audio_placeholder_id=audio_placeholder_id,
            motion_token_ids=motion_token_ids,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_frame_rate=audio_frame_rate,
            audio_frame_size=audio_frame_size,
            audio_cardinality=audio_cardinality,
            audio_codebooks_stored=audio_codebooks_stored,
            audio_codebooks_used=audio_codebooks_used or mimi_codebooks_used or [0],
            mimi_cardinality=audio_cardinality,
            mimi_codebooks_stored=audio_codebooks_stored,
            mimi_codebooks_used=mimi_codebooks_used or [0],
            gap_token_ids=gap_token_ids,
        )
        return cls(config, language_model=language_model)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.language_model.gradient_checkpointing_disable()

    def prepare_input_embeddings(self, input_ids: torch.Tensor, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.ndim == input_ids.ndim:
            audio_codes = audio_codes.unsqueeze(-1)
        expected_audio_shape = (*input_ids.shape, len(self.config.audio_codebooks_used))
        if tuple(audio_codes.shape) != expected_audio_shape:
            raise ValueError(
                f"audio_codes must have shape {expected_audio_shape}, got {tuple(audio_codes.shape)}"
            )
        placeholder_mask = input_ids.eq(self.config.audio_placeholder_id)
        code_mask = audio_codes.ge(0)
        complete_code_mask = code_mask.all(dim=-1)
        if not torch.equal(code_mask.any(dim=-1), complete_code_mask):
            raise ValueError("Every audio frame must provide all configured codebooks")
        if not torch.equal(placeholder_mask, complete_code_mask):
            raise ValueError("audio_codes must be set exactly at [mimi_frame] placeholder positions")
        if complete_code_mask.any():
            selected = audio_codes[complete_code_mask]
            if int(selected.min()) < 0 or int(selected.max()) >= self.config.audio_cardinality:
                raise ValueError("Audio input code is outside the configured cardinality")
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        if not bool(complete_code_mask.any()):
            return text_embeddings

        # Embed only real audio positions; materializing four [B,L,H] tensors is
        # unnecessarily expensive for 2,560-token sequences on 24 GB GPUs.
        selected_codes = audio_codes[complete_code_mask]
        codebook_embeddings = [self.audio_embedding(selected_codes[:, 0])]
        codebook_embeddings.extend(
            embedding(selected_codes[:, index + 1])
            for index, embedding in enumerate(self.additional_audio_embeddings)
        )
        fused_audio = self.audio_fusion(torch.cat(codebook_embeddings, dim=-1))
        flat_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-1])
        flat_positions = complete_code_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        return flat_embeddings.index_copy(0, flat_positions, fused_audio).view_as(text_embeddings)

    def _base_model_forward(self, **kwargs: Any):
        base_prefix = getattr(self.language_model, "base_model_prefix", "model")
        base_model = getattr(self.language_model, base_prefix)
        return base_model(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        target_slots: torch.Tensor,
        motion_local_labels: torch.Tensor,
        expected_distortion_weight: float = 0.0,
        expected_distortion_example_mask: Optional[torch.Tensor] = None,
        gap_target_probs: Optional[torch.Tensor] = None,
        gap_target_mask: Optional[torch.Tensor] = None,
        gap_loss_weights: Optional[torch.Tensor] = None,
        return_token_losses: bool = False,
        **kwargs: Any,
    ) -> MimiQwenPlannerOutput:
        del kwargs
        if not (
            input_ids.shape
            == attention_mask.shape
            == target_slots.shape
            == motion_local_labels.shape
        ):
            raise ValueError("All non-audio planner input tensors must have the same [B, L] shape")
        if tuple(audio_codes.shape[:2]) != tuple(input_ids.shape):
            raise ValueError("audio_codes must begin with the same [B, L] shape as input_ids")
        target_mask = target_slots.ge(0)
        label_mask = motion_local_labels.ne(IGNORE_INDEX)
        if not torch.equal(target_mask, label_mask):
            raise ValueError("target_slots and motion_local_labels masks must match")
        if target_mask[:, 0].any():
            raise ValueError("The first sequence position cannot be an autoregressive target")
        expected_distortion_weight = float(expected_distortion_weight)
        if expected_distortion_weight < 0:
            raise ValueError("expected_distortion_weight must be non-negative")
        if expected_distortion_example_mask is not None:
            if tuple(expected_distortion_example_mask.shape) != (input_ids.shape[0],):
                raise ValueError("expected_distortion_example_mask must have shape [B]")
            expected_distortion_example_mask = expected_distortion_example_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        if expected_distortion_weight > 0 and not self.motion_codebook_distances.numel():
            raise RuntimeError(
                "Expected-distortion loss is enabled but codec distance tables were not loaded"
            )

        inputs_embeds = self.prepare_input_embeddings(input_ids, audio_codes)
        base_outputs = self._base_model_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = base_outputs.last_hidden_state[:, :-1]
        shifted_slots = target_slots[:, 1:]
        shifted_labels = motion_local_labels[:, 1:]
        output_weight = self.language_model.get_output_embeddings().weight

        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        correct = torch.zeros((), device=hidden.device, dtype=torch.long)
        count = torch.zeros((), device=hidden.device, dtype=torch.long)
        per_slot_correct = torch.zeros(BODY_SLOT_COUNT, device=hidden.device, dtype=torch.long)
        per_slot_count = torch.zeros(BODY_SLOT_COUNT, device=hidden.device, dtype=torch.long)
        per_example_loss_sum = torch.zeros(input_ids.shape[0], device=hidden.device, dtype=torch.float32)
        per_example_correct = torch.zeros(input_ids.shape[0], device=hidden.device, dtype=torch.long)
        per_example_count = torch.zeros(input_ids.shape[0], device=hidden.device, dtype=torch.long)
        distortion_sum = hidden.new_zeros((), dtype=torch.float32)
        distortion_count = torch.zeros((), device=hidden.device, dtype=torch.long)
        per_token_loss = (
            torch.zeros_like(input_ids, dtype=torch.float32)
            if bool(return_token_losses)
            else None
        )
        for slot in range(BODY_SLOT_COUNT):
            mask = shifted_slots.eq(slot)
            slot_count = mask.sum()
            if not bool(slot_count):
                continue
            slot_hidden = hidden[mask]
            allowed_ids = self.motion_token_ids[slot]
            classifier_weight = output_weight.index_select(0, allowed_ids)
            logits = F.linear(slot_hidden, classifier_weight).float()
            labels = shifted_labels[mask]
            token_losses = F.cross_entropy(logits, labels, reduction="none")
            token_correct = logits.argmax(dim=-1).eq(labels)
            loss_sum = loss_sum + token_losses.sum()
            slot_correct = token_correct.sum()
            correct = correct + slot_correct
            count = count + slot_count
            per_slot_correct[slot] = slot_correct
            per_slot_count[slot] = slot_count
            example_indices = mask.nonzero(as_tuple=False)[:, 0]
            if expected_distortion_weight > 0:
                auxiliary_mask = (
                    torch.ones_like(example_indices, dtype=torch.bool)
                    if expected_distortion_example_mask is None
                    else expected_distortion_example_mask.index_select(0, example_indices)
                )
                if bool(auxiliary_mask.any()):
                    auxiliary_logits = logits[auxiliary_mask]
                    auxiliary_labels = labels[auxiliary_mask]
                    costs = self.motion_codebook_distances[slot].index_select(
                        0, auxiliary_labels
                    )
                    token_distortion = (
                        F.softmax(auxiliary_logits, dim=-1) * costs
                    ).sum(dim=-1)
                    distortion_sum = distortion_sum + token_distortion.sum()
                    distortion_count = distortion_count + auxiliary_mask.sum()
            per_example_loss_sum.scatter_add_(0, example_indices, token_losses)
            per_example_correct.scatter_add_(0, example_indices, token_correct.to(torch.long))
            per_example_count.scatter_add_(
                0, example_indices, torch.ones_like(example_indices, dtype=torch.long)
            )
            if per_token_loss is not None:
                positions = mask.nonzero(as_tuple=False)
                flat_indices = positions[:, 0] * input_ids.shape[1] + positions[:, 1] + 1
                per_token_loss = per_token_loss.reshape(-1).scatter(
                    0, flat_indices, token_losses
                ).view_as(input_ids)
        if not bool(count):
            raise ValueError("Batch contains no supervised anchor tokens")
        ce_loss = loss_sum / count
        if expected_distortion_weight > 0 and not bool(distortion_count):
            raise ValueError("No supervised tokens were selected for expected-distortion loss")
        expected_distortion_loss = (
            distortion_sum / distortion_count
            if bool(distortion_count)
            else distortion_sum
        )
        gap_loss = hidden.new_zeros((), dtype=torch.float32)
        gap_count = torch.zeros((), device=hidden.device, dtype=torch.long)
        gap_correct = torch.zeros((), device=hidden.device, dtype=torch.long)
        if gap_target_probs is not None or gap_target_mask is not None:
            if gap_target_probs is None or gap_target_mask is None:
                raise ValueError("gap_target_probs and gap_target_mask must be provided together")
            if tuple(gap_target_probs.shape) != (*input_ids.shape, len(GAP_TOKENS)):
                raise ValueError(
                    f"gap_target_probs must be [B,L,{len(GAP_TOKENS)}]"
                )
            if tuple(gap_target_mask.shape) != tuple(input_ids.shape):
                raise ValueError("gap_target_mask must have shape [B,L]")
            if gap_target_mask[:, 0].any():
                raise ValueError("The first sequence position cannot be a gap target")
            shifted_gap_mask = gap_target_mask[:, 1:].bool()
            gap_count = shifted_gap_mask.sum()
            if bool(gap_count):
                if self.gap_token_ids.numel() != len(GAP_TOKENS):
                    raise RuntimeError("Gap supervision is present but gap_token_ids are unset")
                targets = gap_target_probs[:, 1:][shifted_gap_mask].float()
                if not torch.allclose(
                    targets.sum(dim=-1),
                    torch.ones(targets.shape[0], device=targets.device),
                    atol=1e-4,
                ):
                    raise ValueError("Every supervised soft gap target must sum to one")
                classifier_weight = output_weight.index_select(0, self.gap_token_ids)
                gap_logits = F.linear(hidden[shifted_gap_mask], classifier_weight).float()
                token_gap_losses = -(
                    targets * F.log_softmax(gap_logits, dim=-1)
                ).sum(dim=-1)
                gap_correct = gap_logits.argmax(dim=-1).eq(targets.argmax(dim=-1)).sum()
                if gap_loss_weights is None:
                    weights = torch.ones_like(token_gap_losses)
                else:
                    if tuple(gap_loss_weights.shape) != (input_ids.shape[0],):
                        raise ValueError("gap_loss_weights must have shape [B]")
                    example_indices = shifted_gap_mask.nonzero(as_tuple=False)[:, 0]
                    weights = gap_loss_weights.float().index_select(0, example_indices)
                gap_loss = (token_gap_losses * weights).sum() / gap_count
        loss = (
            ce_loss
            + expected_distortion_weight * expected_distortion_loss
            + gap_loss
        )
        return MimiQwenPlannerOutput(
            loss=loss,
            ce_loss=ce_loss,
            expected_distortion_loss=expected_distortion_loss,
            expected_distortion_count=distortion_count,
            correct=correct,
            count=count,
            per_slot_correct=per_slot_correct,
            per_slot_count=per_slot_count,
            per_example_loss_sum=per_example_loss_sum,
            per_example_correct=per_example_correct,
            per_example_count=per_example_count,
            per_token_loss=per_token_loss,
            gap_loss=gap_loss,
            gap_count=gap_count,
            gap_correct=gap_correct,
        )

    @torch.inference_mode()
    def next_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        """Return the restricted 512-way logits for the next anchor slot."""

        if not 0 <= int(slot) < BODY_SLOT_COUNT:
            raise ValueError(f"slot must be in [0, {BODY_SLOT_COUNT - 1}]")
        inputs_embeds = self.prepare_input_embeddings(input_ids, audio_codes)
        outputs = self._base_model_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -1]
        allowed_ids = self.motion_token_ids[int(slot)]
        classifier_weight = self.language_model.get_output_embeddings().weight.index_select(0, allowed_ids)
        return F.linear(hidden, classifier_weight).float()

    @torch.inference_mode()
    def next_gap_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return the restricted 16-way logits for the next causal gap decision."""

        if self.gap_token_ids.numel() != len(GAP_TOKENS):
            raise RuntimeError("This checkpoint does not define the 16 gap-token ids")
        inputs_embeds = self.prepare_input_embeddings(input_ids, audio_codes)
        outputs = self._base_model_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -1]
        classifier_weight = self.language_model.get_output_embeddings().weight.index_select(
            0, self.gap_token_ids
        )
        return F.linear(hidden, classifier_weight).float()

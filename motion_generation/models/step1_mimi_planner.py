"""Causal Mimi-conditioned Qwen planner and fixed-gap Phase 1 dataset."""

from __future__ import annotations

import hashlib
import json
import math
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
    ANCHOR_TOKEN,
    AUDIO_END_TOKEN,
    BODY_CODEBOOK_SIZE,
    BODY_PART_ORDER,
    BODY_SLOT_COUNT,
    GAP_TOKENS,
    MIMI_FRAME_TOKEN,
    MOTION_END_TOKEN,
    MOTION_START_TOKEN,
    SEED_TOKEN_BY_MODE,
    STEP1_ROLE_TOKEN,
    body_token,
    causal_audio_boundaries,
    fixed_anchor_times,
    gap_from_anchor_times,
    validate_anchor,
    validate_motion_payload,
)


IGNORE_INDEX = -100
MIMI_FRAME_RATE = 12.5
MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_SIZE = 1_920
MIMI_CARDINALITY = 2_048
MIMI_STORED_CODEBOOKS = 8
MOTION_TOKEN_FPS = 10.0


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


def load_mimi_tokens(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        codes = np.asarray(payload["codes"])
        result = {key: payload[key].item() for key in payload.files if key != "codes"}
    if codes.ndim != 2 or codes.shape[0] != MIMI_STORED_CODEBOOKS:
        raise ValueError(f"Expected Mimi codes [8, T], got {codes.shape} in {path}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"Mimi codes must be integer, got {codes.dtype} in {path}")
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= MIMI_CARDINALITY):
        raise ValueError(f"Mimi code outside [0, {MIMI_CARDINALITY - 1}] in {path}")
    expected = {
        "sample_rate": MIMI_SAMPLE_RATE,
        "frame_size": MIMI_FRAME_SIZE,
        "num_codebooks": MIMI_STORED_CODEBOOKS,
        "cardinality": MIMI_CARDINALITY,
    }
    for key, value in expected.items():
        if int(result.get(key, -1)) != value:
            raise ValueError(f"Mimi metadata {key}={result.get(key)}; expected {value} in {path}")
    if not math.isclose(float(result.get("frame_rate", -1)), MIMI_FRAME_RATE):
        raise ValueError(f"Mimi frame rate mismatch in {path}")
    result["codes"] = codes.astype(np.int64, copy=False)
    return result


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
    audio_codes: list[int]
    target_slots: list[int]
    motion_local_labels: list[int]
    anchor_times: tuple[int, ...]
    audio_boundaries: tuple[int, ...]
    generated_prefix_anchors: int


class Step1FixedGapDataset(Dataset):
    """One causal, interleaved planner sequence per utterance.

    Text is fully visible at the beginning.  A runtime gap control is followed
    by only the new causal Mimi frames for that interval, then a 16-slot anchor.
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
        result = {}
        for token in tokens:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Step 1 control {token} is not one tokenizer id: {encoded}")
            result[token] = int(encoded[0])
        return result

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence = self.build_sequence(self.names[index])
        return {
            "name": sequence.name,
            "input_ids": sequence.input_ids,
            "audio_codes": sequence.audio_codes,
            "target_slots": sequence.target_slots,
            "motion_local_labels": sequence.motion_local_labels,
            "anchor_times": sequence.anchor_times,
            "audio_boundaries": sequence.audio_boundaries,
            "generated_prefix_anchors": sequence.generated_prefix_anchors,
        }

    def build_sequence(self, name: str) -> Step1Sequence:
        motion_path = canonical_data_path(self.motion_token_dir, name, ".json")
        audio_path = canonical_data_path(self.mimi_token_dir, name, ".npz")
        motion_tokens, _ = load_motion_tokens(motion_path, require_causal=self.require_causal_motion)
        mimi_payload = load_mimi_tokens(audio_path)
        q0 = np.asarray(mimi_payload["codes"])[0]
        if name not in self.text_map:
            raise KeyError(f"Missing text annotation for {name}")

        audio_duration = int(mimi_payload["num_samples"]) / MIMI_SAMPLE_RATE
        motion_duration = len(motion_tokens) / MOTION_TOKEN_FPS
        mismatch = abs(audio_duration - motion_duration)
        if mismatch > self.max_duration_mismatch_seconds:
            raise ValueError(
                f"Audio/motion duration mismatch for {name}: audio={audio_duration:.4f}s, "
                f"motion={motion_duration:.4f}s, error={mismatch:.4f}s"
            )

        anchor_times = fixed_anchor_times(len(motion_tokens), gap=self.fixed_gap)
        audio_boundaries = causal_audio_boundaries(
            anchor_times,
            audio_frames=len(q0),
            audio_fps=MIMI_FRAME_RATE,
            motion_fps=MOTION_TOKEN_FPS,
        )
        if len(anchor_times) != len(audio_boundaries):
            raise AssertionError("Anchor/audio boundary count mismatch")
        generated = self._generated_anchors(name)

        prompt = f"Human: {STEP1_ROLE_TOKEN}{self.text_map[name]}<|im_end|>\nAssistant:"
        input_ids = [int(value) for value in self.tokenizer.encode(prompt, add_special_tokens=False)]
        audio_codes = [-1] * len(input_ids)
        target_slots = [-1] * len(input_ids)
        motion_local_labels = [IGNORE_INDEX] * len(input_ids)

        def append_control(token: str) -> None:
            input_ids.append(self._single_token_ids[token])
            audio_codes.append(-1)
            target_slots.append(-1)
            motion_local_labels.append(IGNORE_INDEX)

        def append_anchor(input_anchor: Sequence[int], target_anchor: Optional[Sequence[int]]) -> None:
            validate_anchor(input_anchor)
            if target_anchor is not None:
                validate_anchor(target_anchor)
            append_control(ANCHOR_TOKEN)
            for slot, input_local_id in enumerate(input_anchor):
                token_id = self.tokenizer.convert_tokens_to_ids(body_token(slot, int(input_local_id)))
                if token_id is None:
                    raise ValueError(f"Tokenizer is missing body token for slot {slot}, id {input_local_id}")
                input_ids.append(int(token_id))
                audio_codes.append(-1)
                if target_anchor is None:
                    target_slots.append(-1)
                    motion_local_labels.append(IGNORE_INDEX)
                else:
                    target_slots.append(slot)
                    motion_local_labels.append(int(target_anchor[slot]))

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
            next_audio_boundary = audio_boundaries[anchor_index]
            for code in q0[audio_cursor:next_audio_boundary]:
                append_control(MIMI_FRAME_TOKEN)
                audio_codes[-1] = int(code)
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
            append_anchor(input_anchor, target_anchor=gt_anchor)

        if audio_cursor != len(q0):
            raise AssertionError(f"Did not consume all Mimi frames for {name}: {audio_cursor}/{len(q0)}")
        append_control(AUDIO_END_TOKEN)
        append_control(MOTION_END_TOKEN)
        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(im_end_ids) != 1:
            raise ValueError("<|im_end|> must be a single token")
        input_ids.append(int(im_end_ids[0]))
        audio_codes.append(-1)
        target_slots.append(-1)
        motion_local_labels.append(IGNORE_INDEX)

        lengths = {len(input_ids), len(audio_codes), len(target_slots), len(motion_local_labels)}
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
            anchor_times=anchor_times,
            audio_boundaries=audio_boundaries,
            generated_prefix_anchors=generated_prefix_anchors,
        )


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

        input_ids = padded("input_ids", self.pad_token_id)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
            "audio_codes": padded("audio_codes", -1),
            "target_slots": padded("target_slots", -1),
            "motion_local_labels": padded("motion_local_labels", IGNORE_INDEX),
            "names": [str(example["name"]) for example in examples],
            "generated_prefix_anchors": torch.tensor(
                [int(example["generated_prefix_anchors"]) for example in examples], dtype=torch.long
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
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.language_model_config = language_model_config or {}
        self.audio_placeholder_id = int(audio_placeholder_id)
        self.motion_token_ids = [list(map(int, row)) for row in (motion_token_ids or [])]
        self.mimi_cardinality = int(mimi_cardinality)
        self.mimi_codebooks_stored = int(mimi_codebooks_stored)
        self.mimi_codebooks_used = list(map(int, mimi_codebooks_used or [0]))


@dataclass
class MimiQwenPlannerOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    correct: Optional[torch.Tensor] = None
    count: Optional[torch.Tensor] = None
    per_slot_correct: Optional[torch.Tensor] = None
    per_slot_count: Optional[torch.Tensor] = None


class MimiQwenPlanner(PreTrainedModel):
    """Qwen with a separate 2,048-entry Mimi q0 embedding and slot CE."""

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
        self.audio_embedding = nn.Embedding(config.mimi_cardinality, hidden_size)
        initializer_range = float(getattr(language_model.config, "initializer_range", 0.02))
        nn.init.normal_(self.audio_embedding.weight, mean=0.0, std=initializer_range)

        motion_token_ids = torch.tensor(config.motion_token_ids, dtype=torch.long)
        if tuple(motion_token_ids.shape) != (BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE):
            raise ValueError(
                f"motion_token_ids must be [{BODY_SLOT_COUNT}, {BODY_CODEBOOK_SIZE}], "
                f"got {tuple(motion_token_ids.shape)}"
            )
        self.register_buffer("motion_token_ids", motion_token_ids, persistent=True)
        if not 0 <= config.audio_placeholder_id < language_model.config.vocab_size:
            raise ValueError("audio_placeholder_id is outside the Qwen vocabulary")
        self.post_init()

    @classmethod
    def from_qwen_pretrained(
        cls,
        model_path: str | Path,
        *,
        audio_placeholder_id: int,
        motion_token_ids: Sequence[Sequence[int]],
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
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
            mimi_cardinality=MIMI_CARDINALITY,
            mimi_codebooks_stored=MIMI_STORED_CODEBOOKS,
            mimi_codebooks_used=[0],
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
        if input_ids.shape != audio_codes.shape:
            raise ValueError("input_ids and audio_codes must have the same shape")
        placeholder_mask = input_ids.eq(self.config.audio_placeholder_id)
        code_mask = audio_codes.ge(0)
        if not torch.equal(placeholder_mask, code_mask):
            raise ValueError("audio_codes must be set exactly at [mimi_frame] placeholder positions")
        if code_mask.any():
            selected = audio_codes[code_mask]
            if int(selected.min()) < 0 or int(selected.max()) >= self.config.mimi_cardinality:
                raise ValueError("Mimi input code is outside the configured cardinality")
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        safe_codes = audio_codes.clamp(min=0)
        audio_embeddings = self.audio_embedding(safe_codes)
        return torch.where(placeholder_mask.unsqueeze(-1), audio_embeddings, text_embeddings)

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
        **kwargs: Any,
    ) -> MimiQwenPlannerOutput:
        del kwargs
        if not (
            input_ids.shape
            == attention_mask.shape
            == audio_codes.shape
            == target_slots.shape
            == motion_local_labels.shape
        ):
            raise ValueError("All planner input tensors must have the same [B, L] shape")
        target_mask = target_slots.ge(0)
        label_mask = motion_local_labels.ne(IGNORE_INDEX)
        if not torch.equal(target_mask, label_mask):
            raise ValueError("target_slots and motion_local_labels masks must match")
        if target_mask[:, 0].any():
            raise ValueError("The first sequence position cannot be an autoregressive target")

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
            loss_sum = loss_sum + F.cross_entropy(logits, labels, reduction="sum")
            slot_correct = logits.argmax(dim=-1).eq(labels).sum()
            correct = correct + slot_correct
            count = count + slot_count
            per_slot_correct[slot] = slot_correct
            per_slot_count[slot] = slot_count
        if not bool(count):
            raise ValueError("Batch contains no supervised anchor tokens")
        loss = loss_sum / count
        return MimiQwenPlannerOutput(
            loss=loss,
            correct=correct,
            count=count,
            per_slot_correct=per_slot_correct,
            per_slot_count=per_slot_count,
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

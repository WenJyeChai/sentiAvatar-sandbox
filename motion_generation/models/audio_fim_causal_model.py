#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Small audio-aware causal FIM transformer for Step 2 motion infill.

This model is intentionally separate from:
    - audio_motion_model.py: the original BERT-style mask transformer.
    - vllm_infill_model.py: the Qwen/vLLM-compatible infill fine-tuning path.

Design target:
    Keep the original Step 2 transformer capacity roughly intact, but change
    the training/inference direction to causal FIM. The prompt uses masked
    middle motion placeholders as audio carriers:

        history + left + masked middle slots + right -> middle motion

Motion tokens use the old compact integer mapping. HuBERT audio remains a
continuous feature stream and is projected into the transformer hidden space.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel


IGNORE_INDEX = -100

FrameTokens = List[int]
MotionSequence = List[FrameTokens]


@dataclass(frozen=True)
class AudioFIMSpecialTokenNames:
    mask: str = "[MASK]"
    pad: str = "[PAD]"
    bos: str = "[BOS]"
    eos: str = "[EOS]"
    infill: str = "[INFILL]"
    history: str = "[HISTORY]"
    left_anchor: str = "[LEFT_ANCHOR]"
    right_anchor: str = "[RIGHT_ANCHOR]"
    middle_audio: str = "[MIDDLE_AUDIO]"
    audio_frame: str = "[AUDIO_FRAME]"
    middle_motion: str = "[MIDDLE_MOTION]"


class AudioFIMCausalConfig(PretrainedConfig):
    """
    Config for the compact causal Step 2 FIM model.

    The default core transformer matches the original mask transformer:
        hidden_size=512, num_layers=8, num_heads=16, intermediate_size=1536.

    The total parameter count is not forced to be bit-identical because this
    model reserves extra special/length tokens for causal FIM prompts. Some
    special tokens are retained for compatibility with earlier prompt layouts.
    """

    model_type = "audio_fim_causal"

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 16,
        intermediate_size: int = 1536,
        max_position_embeddings: int = 512,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        audio_feat_dim: int = 768,
        max_gap_frames: int = 16,
        dropout: float = 0.2,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        vocab_size: Optional[int] = None,
        **kwargs,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be >= 1")

        motion_vocab_size = int(codebook_size) * int(num_quantizers)
        cursor = motion_vocab_size
        mask_token_id = cursor
        cursor += 1
        pad_token_id = cursor
        cursor += 1
        bos_token_id = cursor
        cursor += 1
        eos_token_id = cursor
        cursor += 1
        infill_token_id = cursor
        cursor += 1
        history_token_id = cursor
        cursor += 1
        left_anchor_token_id = cursor
        cursor += 1
        right_anchor_token_id = cursor
        cursor += 1
        middle_audio_token_id = cursor
        cursor += 1
        audio_frame_token_id = cursor
        cursor += 1
        middle_motion_token_id = cursor
        cursor += 1
        length_token_start_id = cursor
        computed_vocab_size = cursor + int(max_gap_frames)

        # Configs loaded by from_pretrained include these in kwargs. We compute
        # them from codebook/special layout, so avoid duplicate super() args.
        kwargs.pop("pad_token_id", None)
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("tie_word_embeddings", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.motion_vocab_size = int(motion_vocab_size)
        self.audio_feat_dim = int(audio_feat_dim)
        self.max_gap_frames = int(max_gap_frames)
        self.dropout = float(dropout)
        self.rms_norm_eps = float(rms_norm_eps)
        self.hidden_act = hidden_act
        self.initializer_range = float(initializer_range)

        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.infill_token_id = int(infill_token_id)
        self.history_token_id = int(history_token_id)
        self.left_anchor_token_id = int(left_anchor_token_id)
        self.right_anchor_token_id = int(right_anchor_token_id)
        self.middle_audio_token_id = int(middle_audio_token_id)
        self.audio_frame_token_id = int(audio_frame_token_id)
        self.middle_motion_token_id = int(middle_motion_token_id)
        self.length_token_start_id = int(length_token_start_id)
        self.vocab_size = int(vocab_size or computed_vocab_size)

        if self.vocab_size < computed_vocab_size:
            raise ValueError(
                f"vocab_size={self.vocab_size} is too small for computed "
                f"compact FIM vocabulary size {computed_vocab_size}"
            )


class AudioFIMTokenMapper:
    """Small integer ID mapper replacing a text/BPE tokenizer for this model."""

    names = AudioFIMSpecialTokenNames()

    def __init__(self, config: AudioFIMCausalConfig):
        self.config = config

    @property
    def pad_token_id(self) -> int:
        return self.config.pad_token_id

    def motion_token_id(self, value: int, quantizer_idx: int) -> int:
        if quantizer_idx < 0 or quantizer_idx >= self.config.num_quantizers:
            raise ValueError(f"Invalid quantizer index: {quantizer_idx}")
        value = int(value)
        if value < 0 or value >= self.config.codebook_size:
            raise ValueError(
                f"Motion token {value} outside 0..{self.config.codebook_size - 1}"
            )
        return quantizer_idx * self.config.codebook_size + value

    def motion_frame_to_ids(self, frame: Sequence[int]) -> List[int]:
        if len(frame) != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} tokens per frame, "
                f"got {len(frame)}"
            )
        return [
            self.motion_token_id(value, quantizer_idx)
            for quantizer_idx, value in enumerate(frame)
        ]

    def motion_ids_to_frame(self, token_ids: Sequence[int]) -> List[int]:
        if len(token_ids) != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} token ids, got {len(token_ids)}"
            )
        frame = []
        for quantizer_idx, token_id in enumerate(token_ids):
            expected_start = quantizer_idx * self.config.codebook_size
            expected_end = expected_start + self.config.codebook_size
            token_id = int(token_id)
            if not (expected_start <= token_id < expected_end):
                raise ValueError(
                    f"Token id {token_id} is not valid for quantizer {quantizer_idx}"
                )
            frame.append(token_id - expected_start)
        return frame

    def length_token_id(self, gap_frames: int) -> int:
        gap_frames = int(gap_frames)
        if gap_frames < 1 or gap_frames > self.config.max_gap_frames:
            raise ValueError(
                f"gap_frames must be in 1..{self.config.max_gap_frames}, got {gap_frames}"
            )
        return self.config.length_token_start_id + gap_frames - 1

    def token_name(self, token_id: int) -> str:
        token_id = int(token_id)
        cfg = self.config
        if 0 <= token_id < cfg.motion_vocab_size:
            quantizer_idx = token_id // cfg.codebook_size
            value = token_id % cfg.codebook_size
            return f"[res_{quantizer_idx + 1}_{value}]"

        special = {
            cfg.mask_token_id: self.names.mask,
            cfg.pad_token_id: self.names.pad,
            cfg.bos_token_id: self.names.bos,
            cfg.eos_token_id: self.names.eos,
            cfg.infill_token_id: self.names.infill,
            cfg.history_token_id: self.names.history,
            cfg.left_anchor_token_id: self.names.left_anchor,
            cfg.right_anchor_token_id: self.names.right_anchor,
            cfg.middle_audio_token_id: self.names.middle_audio,
            cfg.audio_frame_token_id: self.names.audio_frame,
            cfg.middle_motion_token_id: self.names.middle_motion,
        }
        if token_id in special:
            return special[token_id]

        length_offset = token_id - cfg.length_token_start_id
        if 0 <= length_offset < cfg.max_gap_frames:
            return f"[LEN_{length_offset + 1}]"

        return f"[UNK_{token_id}]"

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "vocab_size": self.config.vocab_size,
            "motion_vocab_size": self.config.motion_vocab_size,
            "codebook_size": self.config.codebook_size,
            "num_quantizers": self.config.num_quantizers,
            "ids": {
                "mask": self.config.mask_token_id,
                "pad": self.config.pad_token_id,
                "bos": self.config.bos_token_id,
                "eos": self.config.eos_token_id,
                "infill": self.config.infill_token_id,
                "history": self.config.history_token_id,
                "left_anchor": self.config.left_anchor_token_id,
                "right_anchor": self.config.right_anchor_token_id,
                "middle_audio": self.config.middle_audio_token_id,
                "audio_frame": self.config.audio_frame_token_id,
                "middle_motion": self.config.middle_motion_token_id,
                "length_token_start": self.config.length_token_start_id,
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class AudioFIMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AudioFIMAudioEncoder(nn.Module):
    """Project HuBERT features into the transformer hidden space."""

    def __init__(self, audio_feat_dim: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.proj(audio_features)


class AudioFIMPositionEmbedding(nn.Module):
    """Trainable position embedding initialized with sinusoidal values."""

    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def init_sinusoidal(self) -> None:
        max_pos, hidden_size = self.position_embeddings.weight.shape
        position = torch.arange(max_pos).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        embeddings = torch.zeros(max_pos, hidden_size)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        with torch.no_grad():
            self.position_embeddings.weight.copy_(embeddings)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        return self.position_embeddings(position_ids)


class AudioFIMCausalLM(PreTrainedModel):
    """
    Compact causal decoder for Step 2 audio-aware FIM.

    Forward inputs:
        input_ids:       (B, T)
        attention_mask:  (B, T), 1 real, 0 padding
        audio_features:  (B, A, 768)
        audio_frame_ids: (B, T), local audio index per token, -1 for none
        labels:          (B, T), prompt/pad = -100, targets = token IDs
    """

    config_class = AudioFIMCausalConfig
    base_model_prefix = "audio_fim_causal"
    supports_gradient_checkpointing = True

    def __init__(self, config: AudioFIMCausalConfig):
        super().__init__(config)
        self.config = config
        self.mapper = AudioFIMTokenMapper(config)
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.audio_encoder = AudioFIMAudioEncoder(
            config.audio_feat_dim, config.hidden_size, config.dropout
        )
        self.position_emb = AudioFIMPositionEmbedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.input_norm = AudioFIMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation=ACT2FN[config.hidden_act],
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = AudioFIMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        self.position_emb.init_sinusoidal()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.out_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.out_head = new_embeddings

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _fuse_audio(
        self,
        hidden_states: torch.Tensor,
        audio_features: Optional[torch.Tensor],
        audio_frame_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if audio_features is None or audio_frame_ids is None:
            return hidden_states
        if audio_features.numel() == 0:
            return hidden_states

        audio_features = audio_features.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        encoded_audio = self.audio_encoder(audio_features)
        valid = audio_frame_ids >= 0
        safe_ids = audio_frame_ids.clamp(min=0)
        gather_index = safe_ids.unsqueeze(-1).expand(
            -1, -1, encoded_audio.size(-1)
        )
        audio_emb = torch.gather(encoded_audio, dim=1, index=gather_index)
        audio_emb = audio_emb * valid.unsqueeze(-1).to(audio_emb.dtype)
        return hidden_states + audio_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_frame_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        del kwargs

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states + self.position_emb(
            seq_len, input_ids.device, position_ids=position_ids
        )
        hidden_states = self._fuse_audio(
            hidden_states,
            audio_features=audio_features,
            audio_frame_ids=audio_frame_ids,
        )
        hidden_states = self.dropout(self.input_norm(hidden_states))

        causal_mask = self._build_causal_mask(seq_len, input_ids.device)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda x, layer=layer: layer(
                        x,
                        src_mask=causal_mask,
                        src_key_padding_mask=key_padding_mask,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    src_mask=causal_mask,
                    src_key_padding_mask=key_padding_mask,
                )

        hidden_states = self.final_norm(hidden_states)
        logits = self.out_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_infill(
        self,
        *,
        history_motion: MotionSequence,
        left_anchor: FrameTokens,
        right_anchor: FrameTokens,
        middle_audio_features: torch.Tensor,
        left_audio_feature: Optional[torch.Tensor] = None,
        right_audio_feature: Optional[torch.Tensor] = None,
        history_audio_features: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> MotionSequence:
        """
        Generate a fixed-gap middle motion sequence.

        This simple helper favors correctness over maximum speed. It rebuilds
        the full context each generated token. A KV-cache path can be added once
        the training behavior is confirmed.
        """

        self.eval()
        device = next(self.parameters()).device
        builder = AudioFIMSequenceBuilder(self.config)
        example = AudioFIMCausalExample(
            name="inference",
            left_idx=0,
            right_idx=int(middle_audio_features.shape[0]) + 1,
            history_motion=[list(frame) for frame in history_motion],
            left_anchor=list(left_anchor),
            right_anchor=list(right_anchor),
            middle_motion=[],
            history_audio_features=history_audio_features,
            left_audio_feature=left_audio_feature,
            right_audio_feature=right_audio_feature,
            middle_audio_features=middle_audio_features,
        )
        encoded = builder.encode_prompt(example)
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
        audio_frame_ids = torch.tensor(
            [encoded.audio_frame_ids], dtype=torch.long, device=device
        )
        audio_features = encoded.audio_features.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        generated_ids: List[int] = []
        total_motion_tokens = (
            int(middle_audio_features.shape[0]) * self.config.num_quantizers
        )

        for token_idx in range(total_motion_tokens):
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                audio_frame_ids=audio_frame_ids,
            )
            next_logits = outputs.logits[:, -1, :].clone()

            quantizer_idx = token_idx % self.config.num_quantizers
            allowed_start = quantizer_idx * self.config.codebook_size
            allowed_end = allowed_start + self.config.codebook_size
            invalid = torch.ones_like(next_logits, dtype=torch.bool)
            invalid[:, allowed_start:allowed_end] = False
            next_logits = next_logits.masked_fill(invalid, float("-inf"))

            if temperature and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_ids.append(int(next_id.item()))
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.ones_like(input_ids)

            middle_frame_idx = token_idx // self.config.num_quantizers
            next_audio_id = encoded.middle_audio_local_ids[middle_frame_idx]
            next_audio = torch.tensor(
                [[next_audio_id]], dtype=torch.long, device=device
            )
            audio_frame_ids = torch.cat([audio_frame_ids, next_audio], dim=1)

        frames = []
        for start in range(0, len(generated_ids), self.config.num_quantizers):
            frames.append(
                self.mapper.motion_ids_to_frame(
                    generated_ids[start : start + self.config.num_quantizers]
                )
            )
        return frames


@dataclass
class AudioFIMCausalExample:
    name: str
    left_idx: int
    right_idx: int
    history_motion: MotionSequence
    left_anchor: FrameTokens
    right_anchor: FrameTokens
    middle_motion: MotionSequence
    history_audio_features: Optional[torch.Tensor]
    left_audio_feature: Optional[torch.Tensor]
    right_audio_feature: Optional[torch.Tensor]
    middle_audio_features: torch.Tensor


@dataclass
class EncodedAudioFIMExample:
    input_ids: List[int]
    labels: List[int]
    audio_frame_ids: List[int]
    audio_features: torch.Tensor
    section_lengths: Dict[str, int]
    middle_audio_local_ids: List[int]


class AudioFIMSequenceBuilder:
    """Serialize one example into compact causal FIM tensors."""

    def __init__(self, config: AudioFIMCausalConfig):
        self.config = config
        self.mapper = AudioFIMTokenMapper(config)

    def _add_audio(
        self,
        audio_bank: List[torch.Tensor],
        feature: Optional[torch.Tensor],
    ) -> int:
        if feature is None:
            return -1
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)
        audio_bank.append(feature.to(dtype=torch.float32).view(-1))
        return len(audio_bank) - 1

    def encode_prompt(self, example: AudioFIMCausalExample) -> EncodedAudioFIMExample:
        return self.encode(example, include_targets=False)

    def encode(
        self,
        example: AudioFIMCausalExample,
        *,
        include_targets: bool = True,
    ) -> EncodedAudioFIMExample:
        input_ids: List[int] = []
        labels: List[int] = []
        audio_frame_ids: List[int] = []
        audio_bank: List[torch.Tensor] = []
        section_lengths: Dict[str, int] = {}

        def append(token_id: int, *, label: int = IGNORE_INDEX, audio_id: int = -1):
            input_ids.append(int(token_id))
            labels.append(int(label))
            audio_frame_ids.append(int(audio_id))

        def section(name: str, fn) -> None:
            before = len(input_ids)
            fn()
            section_lengths[name] = len(input_ids) - before

        gap_frames = len(example.middle_motion)
        if not include_targets:
            gap_frames = int(example.middle_audio_features.shape[0])

        middle_audio_local_ids = [
            self._add_audio(audio_bank, example.middle_audio_features[i])
            for i in range(int(example.middle_audio_features.shape[0]))
        ]
        history_audio_local_ids: List[int] = []
        if example.history_audio_features is not None:
            history_audio_local_ids = [
                self._add_audio(audio_bank, example.history_audio_features[i])
                for i in range(int(example.history_audio_features.shape[0]))
            ]
        left_audio_id = self._add_audio(audio_bank, example.left_audio_feature)
        right_audio_id = self._add_audio(audio_bank, example.right_audio_feature)

        section(
            "prefix",
            lambda: (
                append(self.config.bos_token_id),
                append(self.config.infill_token_id),
                append(self.mapper.length_token_id(gap_frames)),
            ),
        )

        def add_history_left_context():
            append(self.config.history_token_id)
            for frame_idx, frame in enumerate(example.history_motion):
                audio_id = (
                    history_audio_local_ids[frame_idx]
                    if frame_idx < len(history_audio_local_ids)
                    else -1
                )
                for token_id in self.mapper.motion_frame_to_ids(frame):
                    append(token_id, audio_id=audio_id)

            for token_id in self.mapper.motion_frame_to_ids(example.left_anchor):
                append(token_id, audio_id=left_audio_id)

        section("history_left_context", add_history_left_context)

        def add_middle_mask_placeholders():
            for audio_id in middle_audio_local_ids:
                for _ in range(self.config.num_quantizers):
                    append(self.config.mask_token_id, audio_id=audio_id)

        section("middle_mask_placeholders", add_middle_mask_placeholders)

        def add_right_anchor():
            append(self.config.right_anchor_token_id)
            for token_id in self.mapper.motion_frame_to_ids(example.right_anchor):
                append(token_id, audio_id=right_audio_id)

        section("right_anchor", add_right_anchor)

        section("middle_motion_marker", lambda: append(self.config.middle_motion_token_id))

        if include_targets:
            def add_targets():
                for frame_idx, frame in enumerate(example.middle_motion):
                    audio_id = (
                        middle_audio_local_ids[frame_idx]
                        if frame_idx < len(middle_audio_local_ids)
                        else -1
                    )
                    for token_id in self.mapper.motion_frame_to_ids(frame):
                        append(token_id, label=token_id, audio_id=audio_id)
                append(self.config.eos_token_id, label=self.config.eos_token_id)

            section("target_motion", add_targets)

        if audio_bank:
            audio_features = torch.stack(audio_bank, dim=0)
        else:
            audio_features = torch.zeros(0, self.config.audio_feat_dim)

        return EncodedAudioFIMExample(
            input_ids=input_ids,
            labels=labels,
            audio_frame_ids=audio_frame_ids,
            audio_features=audio_features,
            section_lengths=section_lengths,
            middle_audio_local_ids=middle_audio_local_ids,
        )


class AudioFIMCausalDataset(Dataset):
    """
    Build classic Step 2 infill windows from dense motion tokens and HuBERT
    continuous features.

    First version targets the classic gap:
        left frame t, right frame t+4, predict t+1..t+3.

    The `step` and `[LEN_N]` path are kept explicit so variable gaps can be
    enabled later without changing the model interface.
    """

    def __init__(
        self,
        sequences: Sequence[Dict[str, Any]],
        *,
        step: int = 4,
        audio_fps: float = 10.0,
        motion_fps: float = 20.0,
        min_history_frames: int = 0,
        max_history_frames: int = 8,
        max_windows_per_sequence: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if step < 2:
            raise ValueError("step must be >= 2")
        if min_history_frames < 0 or max_history_frames < min_history_frames:
            raise ValueError(
                "history window must satisfy 0 <= min_history_frames <= max_history_frames"
            )

        self.sequences = list(sequences)
        self.step = int(step)
        self.default_audio_fps = float(audio_fps)
        self.default_motion_fps = float(motion_fps)
        self.min_history_frames = int(min_history_frames)
        self.max_history_frames = int(max_history_frames)
        self.rng = random.Random(seed)
        self.windows: List[Tuple[int, int, int]] = []

        for seq_idx, item in enumerate(self.sequences):
            motion_tokens = item["motion_tokens"]
            num_motion_frames = len(motion_tokens)
            made = 0
            for left_idx in range(0, num_motion_frames - self.step, self.step):
                available = left_idx
                low = min(self.min_history_frames, available)
                high = min(self.max_history_frames, available)
                history_len = self.rng.randint(low, high) if high > 0 else 0
                self.windows.append((seq_idx, left_idx, history_len))
                made += 1
                if (
                    max_windows_per_sequence is not None
                    and made >= max_windows_per_sequence
                ):
                    break

    def __len__(self) -> int:
        return len(self.windows)

    def _audio_feature_for_frame(
        self,
        audio_features: np.ndarray,
        motion_frame_idx: int,
        *,
        audio_fps: float,
        motion_fps: float,
    ) -> torch.Tensor:
        audio_idx = int(round(float(motion_frame_idx) * audio_fps / motion_fps))
        audio_idx = max(0, min(audio_idx, len(audio_features) - 1))
        return torch.tensor(audio_features[audio_idx], dtype=torch.float32)

    def __getitem__(self, idx: int) -> AudioFIMCausalExample:
        seq_idx, left_idx, history_len = self.windows[idx]
        item = self.sequences[seq_idx]
        motion_tokens: MotionSequence = item["motion_tokens"]
        audio_features: np.ndarray = item["audio_features"]
        audio_fps = float(item.get("audio_fps") or self.default_audio_fps)
        motion_fps = float(item.get("motion_fps") or self.default_motion_fps)

        right_idx = left_idx + self.step
        middle_indices = list(range(left_idx + 1, right_idx))
        history_start = left_idx - history_len
        history_indices = list(range(history_start, left_idx))

        history_motion = [list(motion_tokens[i]) for i in history_indices]
        middle_motion = [list(motion_tokens[i]) for i in middle_indices]

        history_audio_features = None
        if history_indices:
            history_audio_features = torch.stack(
                [
                    self._audio_feature_for_frame(
                        audio_features,
                        i,
                        audio_fps=audio_fps,
                        motion_fps=motion_fps,
                    )
                    for i in history_indices
                ],
                dim=0,
            )

        middle_audio_features = torch.stack(
            [
                self._audio_feature_for_frame(
                    audio_features,
                    i,
                    audio_fps=audio_fps,
                    motion_fps=motion_fps,
                )
                for i in middle_indices
            ],
            dim=0,
        )

        return AudioFIMCausalExample(
            name=item.get("name", str(seq_idx)),
            left_idx=left_idx,
            right_idx=right_idx,
            history_motion=history_motion,
            left_anchor=list(motion_tokens[left_idx]),
            right_anchor=list(motion_tokens[right_idx]),
            middle_motion=middle_motion,
            history_audio_features=history_audio_features,
            left_audio_feature=self._audio_feature_for_frame(
                audio_features,
                left_idx,
                audio_fps=audio_fps,
                motion_fps=motion_fps,
            ),
            right_audio_feature=self._audio_feature_for_frame(
                audio_features,
                right_idx,
                audio_fps=audio_fps,
                motion_fps=motion_fps,
            ),
            middle_audio_features=middle_audio_features,
        )


class AudioFIMCausalCollator:
    """Pad compact FIM examples for HuggingFace Trainer."""

    def __init__(
        self,
        config: AudioFIMCausalConfig,
        *,
        max_length: int = 512,
        pad_to_multiple_of: Optional[int] = 8,
        debug_examples: int = 0,
        profile_batches: int = 0,
    ):
        self.config = config
        self.builder = AudioFIMSequenceBuilder(config)
        self.mapper = AudioFIMTokenMapper(config)
        self.max_length = int(max_length)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.debug_examples = int(debug_examples)
        self._debug_printed = 0
        self.profile_batches = int(profile_batches)
        self._profile_seen = 0

    def _debug(self, example: AudioFIMCausalExample, encoded: EncodedAudioFIMExample):
        if self._debug_printed >= self.debug_examples:
            return

        supervised = sum(label != IGNORE_INDEX for label in encoded.labels)
        print("=" * 70)
        print(f"[AudioFIM tokenization debug #{self._debug_printed + 1}]")
        print(f"name:                 {example.name}")
        print(f"left/right idx:       {example.left_idx}/{example.right_idx}")
        print(f"history frames:       {len(example.history_motion)}")
        print(f"middle audio frames:  {example.middle_audio_features.shape[0]}")
        print(f"middle motion frames: {len(example.middle_motion)}")
        print(f"input tokens:         {len(encoded.input_ids)}")
        print(f"supervised labels:    {supervised}")
        print(f"local audio frames:   {encoded.audio_features.shape[0]}")
        print("section token counts:")
        for name, length in encoded.section_lengths.items():
            print(f"  {name:22s} {length:4d}")
        print("sequence preview:")
        print(" ".join(self.mapper.token_name(t) for t in encoded.input_ids[:128]))
        print("=" * 70)
        self._debug_printed += 1

    def __call__(self, examples: Sequence[AudioFIMCausalExample]) -> Dict[str, torch.Tensor]:
        profile = self.profile_batches > 0 and self._profile_seen < self.profile_batches
        t0 = time.perf_counter() if profile else None
        encoded = [self.builder.encode(example) for example in examples]
        t_encoded = time.perf_counter() if profile else None

        for example, item in zip(examples, encoded):
            self._debug(example, item)

        max_len = max(len(item.input_ids) for item in encoded)
        if max_len > self.max_length:
            raise ValueError(
                f"Batch contains seq_len={max_len}, larger than max_length={self.max_length}. "
                "Reduce history length or increase --max_length."
            )

        if self.pad_to_multiple_of is not None:
            multiple = int(self.pad_to_multiple_of)
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        max_audio = max(item.audio_features.shape[0] for item in encoded)
        max_audio = max(1, max_audio)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_audio_frame_ids = []
        batch_audio_features = []

        for item in encoded:
            pad_len = max_len - len(item.input_ids)
            batch_input_ids.append(
                item.input_ids + [self.config.pad_token_id] * pad_len
            )
            batch_attention_mask.append([1] * len(item.input_ids) + [0] * pad_len)
            batch_labels.append(item.labels + [IGNORE_INDEX] * pad_len)
            batch_audio_frame_ids.append(item.audio_frame_ids + [-1] * pad_len)

            audio = item.audio_features
            if audio.shape[0] < max_audio:
                audio_pad = torch.zeros(
                    max_audio - audio.shape[0],
                    self.config.audio_feat_dim,
                    dtype=audio.dtype,
                )
                audio = torch.cat([audio, audio_pad], dim=0)
            batch_audio_features.append(audio)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "audio_frame_ids": torch.tensor(batch_audio_frame_ids, dtype=torch.long),
            "audio_features": torch.stack(batch_audio_features, dim=0),
        }

        if profile:
            t_done = time.perf_counter()
            lengths = [len(item.input_ids) for item in encoded]
            supervised = [
                sum(label != IGNORE_INDEX for label in item.labels)
                for item in encoded
            ]
            print(
                "[AudioFIM collator profile] "
                f"batch={self._profile_seen + 1} "
                f"examples={len(examples)} "
                f"seq_len={max_len} "
                f"audio_len={max_audio} "
                f"raw_len_min/avg/max={min(lengths)}/"
                f"{sum(lengths) / len(lengths):.1f}/{max(lengths)} "
                f"labels_avg={sum(supervised) / len(supervised):.1f} "
                f"encode_s={t_encoded - t0:.4f} "
                f"pad_tensor_s={t_done - t_encoded:.4f} "
                f"total_s={t_done - t0:.4f}"
            )
            self._profile_seen += 1

        return batch

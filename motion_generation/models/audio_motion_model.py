"""
Audio-Motion Mask Transformer Model (Hubert Features Version)

输入: motion_1, motion_5 (各4个motion token) + 5帧的hubert audio特征
输出: 预测被mask的motion token (motion_2, motion_3, motion_4)

Audio特征(hubert layer9, 768dim)通过线性层投影后，以加法方式融合到对应帧的motion token embedding上。
Motion token: codebook_size=512, 4 quantizers, offset=512
vocab_size = 512 * 4 + 1 = 2049 (最后一个是mask token)

@File    :   audio_motion_model.py
@Time    :   2025/07/16
"""

import random
import math
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations import use_kernel_forward_from_hub
from transformers.activations import ACT2FN
from transformers import PretrainedConfig


class AudioMotionConfig(PretrainedConfig):
    model_type = "audio_motion_transformer"

    def __init__(
        self,
        hidden_size=512,
        num_layers=8,
        num_heads=16,
        intermediate_size=1536,
        max_position_embeddings=512,
        vocab_size=2049,        # 512 * 4 + 1 (mask token)
        codebook_size=512,      # 每个quantizer的codebook大小
        audio_feat_dim=768,     # hubert layer9 特征维度
        num_tokens_per_frame=4, # 每帧motion token数量 (4 quantizers)
        num_frames=5,           # 窗口帧数
        dropout=0.2,
        cond_drop_prob=0.2,
        constrain_token_logits=False,
        audio_conditioning_mode="additive",
        audio_residual_part_order=None,
        audio_residual_num_quantizers=0,
        audio_residual_latent_dims=None,
        audio_residual_prior_weight=1.0,
        audio_residual_gate_init=-4.0,
        audio_residual_log_sigma_min=-3.0,
        audio_residual_log_sigma_max=3.0,
        rms_norm_eps=1e-6,
        hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.audio_feat_dim = audio_feat_dim
        self.num_tokens_per_frame = num_tokens_per_frame
        self.num_frames = num_frames
        self.dropout = dropout
        self.cond_drop_prob = cond_drop_prob
        self.constrain_token_logits = constrain_token_logits
        self.audio_conditioning_mode = audio_conditioning_mode
        self.audio_residual_part_order = list(audio_residual_part_order or [])
        self.audio_residual_num_quantizers = int(audio_residual_num_quantizers)
        self.audio_residual_latent_dims = list(audio_residual_latent_dims or [])
        self.audio_residual_prior_weight = float(audio_residual_prior_weight)
        self.audio_residual_gate_init = float(audio_residual_gate_init)
        self.audio_residual_log_sigma_min = float(audio_residual_log_sigma_min)
        self.audio_residual_log_sigma_max = float(audio_residual_log_sigma_max)
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act


@use_kernel_forward_from_hub("RMSNorm")
class AudioMotionRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AudioEncoder(nn.Module):
    """将hubert音频原始特征投影到hidden_size空间"""

    def __init__(self, audio_feat_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, audio_features):
        """
        Args:
            audio_features: (batch_size, num_frames, audio_feat_dim)
        Returns:
            (batch_size, num_frames, hidden_size)
        """
        return self.proj(audio_features)


class AudioResidualSlotHead(nn.Module):
    """Predict one RVQ stage's residual Gaussian and confidence gate."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        gate_init: float,
        log_sigma_min: float,
        log_sigma_max: float,
    ) -> None:
        super().__init__()
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
        self.prefix_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_sigma = nn.Linear(hidden_size, 1)
        self.gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.zeros_(self.log_sigma.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init))

    def forward(
        self,
        motion_states: torch.Tensor,
        audio_states: torch.Tensor,
        prefix_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prefix_states = self.prefix_proj(prefix_latent)
        fused = self.trunk(
            torch.cat([motion_states, audio_states, prefix_states], dim=-1)
        )
        log_sigma = self.log_sigma(fused).clamp(
            self.log_sigma_min, self.log_sigma_max
        )
        return {
            "mu": self.mu(fused),
            "log_sigma": log_sigma,
            "gate": torch.sigmoid(self.gate(fused)),
        }


class AudioResidualPosterior(nn.Module):
    """Map audio and motion context to distributions over frozen RVQ codes."""

    def __init__(
        self,
        hidden_size: int,
        codebooks: Sequence[torch.Tensor],
        part_order: Sequence[str],
        num_quantizers: int,
        codebook_size: int,
        prior_weight: float,
        gate_init: float,
        log_sigma_min: float,
        log_sigma_max: float,
    ) -> None:
        super().__init__()
        self.part_order = tuple(str(part) for part in part_order)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.prior_weight = float(prior_weight)
        if len(codebooks) != len(self.part_order):
            raise ValueError("One stacked codebook tensor is required per body part")

        self.audio_temporal = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )
        self.audio_norm = nn.LayerNorm(hidden_size)
        self.heads = nn.ModuleDict()
        self._codebook_names: list[str] = []
        for part_idx, (part, codebook) in enumerate(zip(self.part_order, codebooks)):
            codebook = torch.as_tensor(codebook).detach().float()
            if codebook.dim() != 3:
                raise ValueError(
                    f"{part} codebooks must have shape (Q,K,D), got {tuple(codebook.shape)}"
                )
            if codebook.shape[:2] != (self.num_quantizers, self.codebook_size):
                raise ValueError(
                    f"{part} codebook layout {tuple(codebook.shape[:2])} does not match "
                    f"({self.num_quantizers},{self.codebook_size})"
                )
            buffer_name = f"codebook_{part_idx}"
            self.register_buffer(buffer_name, codebook, persistent=True)
            self._codebook_names.append(buffer_name)
            for stage in range(self.num_quantizers):
                self.heads[self._head_key(part_idx, stage)] = AudioResidualSlotHead(
                    hidden_size=hidden_size,
                    latent_dim=int(codebook.shape[-1]),
                    gate_init=gate_init,
                    log_sigma_min=log_sigma_min,
                    log_sigma_max=log_sigma_max,
                )

    @staticmethod
    def _head_key(part_idx: int, stage: int) -> str:
        return f"part{part_idx}_q{stage}"

    def codebook(self, part_idx: int) -> torch.Tensor:
        return getattr(self, self._codebook_names[part_idx])

    def set_codebooks(self, codebooks: Sequence[torch.Tensor]) -> None:
        if len(codebooks) != len(self.part_order):
            raise ValueError("Codebook part count changed")
        for part_idx, value in enumerate(codebooks):
            target = self.codebook(part_idx)
            value = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if value.shape != target.shape:
                raise ValueError(
                    f"Codebook shape changed for {self.part_order[part_idx]}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.copy_(value)

    def encode_temporal_audio(
        self,
        encoded_audio: torch.Tensor,
        frame_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        temporal = self.audio_temporal(encoded_audio.transpose(1, 2)).transpose(1, 2)
        temporal = self.audio_norm(encoded_audio + temporal)
        if frame_mask is not None:
            temporal = temporal * frame_mask.unsqueeze(-1).to(temporal.dtype)
        return temporal

    def _raw_part_ids(
        self,
        input_view: torch.Tensor,
        part_idx: int,
    ) -> torch.Tensor:
        start_slot = part_idx * self.num_quantizers
        offsets = (
            torch.arange(
                start_slot,
                start_slot + self.num_quantizers,
                device=input_view.device,
            )
            * self.codebook_size
        )
        return (input_view[..., start_slot : start_slot + self.num_quantizers] - offsets).clamp(
            0, self.codebook_size - 1
        )

    def _prefix_latent(
        self,
        input_view: torch.Tensor,
        part_idx: int,
        stage: int,
    ) -> torch.Tensor:
        book = self.codebook(part_idx)
        latent_shape = (*input_view.shape[:2], int(book.shape[-1]))
        if stage == 0:
            return book.new_zeros(latent_shape)
        raw_ids = self._raw_part_ids(input_view, part_idx).long()
        return sum(
            book[prefix].index_select(0, raw_ids[..., prefix].reshape(-1)).reshape(
                *raw_ids.shape[:-1], -1
            )
            for prefix in range(stage)
        )

    @staticmethod
    def _code_logits(
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        mean_squared_distance = (
            mu.unsqueeze(-2)
            - codebook.view(1, 1, codebook.shape[0], codebook.shape[1])
        ).square().mean(dim=-1)
        variance = torch.exp(2.0 * log_sigma.float()).clamp_min(1e-8)
        return -0.5 * mean_squared_distance.float() / variance

    def forward(
        self,
        logits: torch.Tensor,
        motion_states: torch.Tensor,
        encoded_audio: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        frame_mask: Optional[torch.Tensor] = None,
        stage: Optional[int] = None,
        negative_encoded_audio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[tuple[int, int], dict[str, torch.Tensor]]]:
        ntpf = len(self.part_order) * self.num_quantizers
        if input_ids.shape[1] % ntpf != 0:
            raise ValueError("Input sequence is not divisible by the multipart token layout")
        frames = input_ids.shape[1] // ntpf
        input_view = input_ids.reshape(input_ids.shape[0], frames, ntpf)
        hidden_view = motion_states.reshape(
            motion_states.shape[0], frames, ntpf, motion_states.shape[-1]
        )
        logits_view = logits.reshape(logits.shape[0], frames, ntpf, logits.shape[-1])
        positive_audio = self.encode_temporal_audio(encoded_audio, frame_mask)
        negative_audio = (
            self.encode_temporal_audio(negative_encoded_audio, frame_mask)
            if negative_encoded_audio is not None
            else None
        )
        stages = range(self.num_quantizers) if stage is None else (int(stage),)
        details: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        adjusted = logits_view.clone()

        for part_idx, part in enumerate(self.part_order):
            part_book = self.codebook(part_idx)
            for quantizer in stages:
                if quantizer < 0 or quantizer >= self.num_quantizers:
                    raise ValueError(f"Invalid quantizer stage {quantizer}")
                slot = part_idx * self.num_quantizers + quantizer
                prefix = self._prefix_latent(input_view, part_idx, quantizer)
                head = self.heads[self._head_key(part_idx, quantizer)]
                parameters = head(hidden_view[..., slot, :], positive_audio, prefix)
                prior_logits = self._code_logits(
                    parameters["mu"],
                    parameters["log_sigma"],
                    part_book[quantizer],
                )
                start = slot * self.codebook_size
                end = start + self.codebook_size
                adjusted[..., slot, start:end] = (
                    logits_view[..., slot, start:end]
                    + self.prior_weight
                    * parameters["gate"]
                    * prior_logits.to(logits.dtype)
                )
                slot_details = {
                    **parameters,
                    "prior_logits": prior_logits,
                    "part": part,
                }
                if negative_audio is not None:
                    negative_parameters = head(
                        hidden_view[..., slot, :], negative_audio, prefix
                    )
                    slot_details.update(
                        {
                            "negative_mu": negative_parameters["mu"],
                            "negative_log_sigma": negative_parameters["log_sigma"],
                            "negative_prior_logits": self._code_logits(
                                negative_parameters["mu"],
                                negative_parameters["log_sigma"],
                                part_book[quantizer],
                            ),
                        }
                    )
                details[(part_idx, quantizer)] = slot_details
        return adjusted.reshape_as(logits), details


class PositionEmbedding(nn.Module):
    """可学习的位置嵌入，使用正弦初始化"""

    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self._init_embeddings(max_position_embeddings, hidden_size)

    def _init_embeddings(self, max_pos, hidden_size):
        position = torch.arange(max_pos).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        embeddings = torch.zeros(max_pos, hidden_size)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        self.position_embeddings.weight.data.copy_(embeddings)

    def forward(self, seq_len, device):
        """
        Args:
            seq_len: int
            device: torch.device
        Returns:
            (1, seq_len, hidden_size)
        """
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        return self.position_embeddings(position_ids)


class AudioMotionTransformer(PreTrainedModel):
    config_class = AudioMotionConfig

    def __init__(self, config: AudioMotionConfig):
        super().__init__(config)
        self.config = config
        self._validate_audio_conditioning_mode(config.audio_conditioning_mode)

        # Motion token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Audio encoder: 768 -> hidden_size
        self.audio_encoder = AudioEncoder(
            config.audio_feat_dim, config.hidden_size, config.dropout
        )
        self.audio_residual_posterior: Optional[AudioResidualPosterior] = None
        if (
            config.audio_conditioning_mode
            in {"residual_posterior", "additive_residual_posterior"}
            and config.audio_residual_part_order
            and config.audio_residual_num_quantizers > 0
            and config.audio_residual_latent_dims
        ):
            placeholder_books = [
                torch.zeros(
                    config.audio_residual_num_quantizers,
                    config.codebook_size,
                    int(latent_dim),
                )
                for latent_dim in config.audio_residual_latent_dims
            ]
            self.audio_residual_posterior = self._build_audio_residual_posterior(
                placeholder_books,
                config.audio_residual_part_order,
                config.audio_residual_num_quantizers,
            )

        # Position embedding
        self.position_emb = PositionEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation=ACT2FN[config.hidden_act],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
        )

        # Output head
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Normalization
        self.norm = AudioMotionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()
        if self.audio_residual_posterior is not None:
            for head in self.audio_residual_posterior.heads.values():
                nn.init.zeros_(head.gate.weight)
                nn.init.constant_(
                    head.gate.bias, float(config.audio_residual_gate_init)
                )

    @staticmethod
    def _validate_audio_conditioning_mode(mode: str) -> None:
        if mode not in {
            "additive",
            "residual_posterior",
            "additive_residual_posterior",
        }:
            raise ValueError(
                "audio_conditioning_mode must be additive, residual_posterior, "
                "or additive_residual_posterior"
            )

    def _build_audio_residual_posterior(
        self,
        codebooks: Sequence[torch.Tensor],
        part_order: Sequence[str],
        num_quantizers: int,
    ) -> AudioResidualPosterior:
        return AudioResidualPosterior(
            hidden_size=int(self.config.hidden_size),
            codebooks=codebooks,
            part_order=part_order,
            num_quantizers=num_quantizers,
            codebook_size=int(self.config.codebook_size),
            prior_weight=float(self.config.audio_residual_prior_weight),
            gate_init=float(self.config.audio_residual_gate_init),
            log_sigma_min=float(self.config.audio_residual_log_sigma_min),
            log_sigma_max=float(self.config.audio_residual_log_sigma_max),
        )

    def configure_audio_residual_posterior(
        self,
        codebooks: Mapping[str, torch.Tensor],
        part_order: Sequence[str],
        num_quantizers: int,
        *,
        mode: str,
        prior_weight: float = 1.0,
        gate_init: float = -4.0,
        log_sigma_min: float = -3.0,
        log_sigma_max: float = 3.0,
    ) -> None:
        """Attach or refresh the frozen multipart codebooks used by the prior."""
        self._validate_audio_conditioning_mode(mode)
        self.config.audio_conditioning_mode = mode
        if mode == "additive":
            return

        order = tuple(str(part) for part in part_order)
        stacked = [torch.as_tensor(codebooks[part]).detach().float() for part in order]
        latent_dims = [int(value.shape[-1]) for value in stacked]
        self.config.audio_residual_part_order = list(order)
        self.config.audio_residual_num_quantizers = int(num_quantizers)
        self.config.audio_residual_latent_dims = latent_dims
        self.config.audio_residual_prior_weight = float(prior_weight)
        self.config.audio_residual_gate_init = float(gate_init)
        self.config.audio_residual_log_sigma_min = float(log_sigma_min)
        self.config.audio_residual_log_sigma_max = float(log_sigma_max)

        compatible = (
            self.audio_residual_posterior is not None
            and self.audio_residual_posterior.part_order == order
            and self.audio_residual_posterior.num_quantizers == int(num_quantizers)
            and all(
                self.audio_residual_posterior.codebook(index).shape == value.shape
                for index, value in enumerate(stacked)
            )
        )
        if compatible:
            self.audio_residual_posterior.prior_weight = float(prior_weight)
            self.audio_residual_posterior.set_codebooks(stacked)
            for head in self.audio_residual_posterior.heads.values():
                head.log_sigma_min = float(log_sigma_min)
                head.log_sigma_max = float(log_sigma_max)
            return

        posterior = self._build_audio_residual_posterior(stacked, order, num_quantizers)
        self.audio_residual_posterior = posterior.to(next(self.parameters()).device)

    @property
    def uses_audio_residual_posterior(self) -> bool:
        return (
            self.config.audio_conditioning_mode
            in {"residual_posterior", "additive_residual_posterior"}
            and self.audio_residual_posterior is not None
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _run_encoder(self, x, attention_mask=None):
        encoder_mask = None
        key_padding_mask = None
        if attention_mask is not None:
            if (
                attention_mask.dim() == 2
                and attention_mask.shape[0] == x.shape[0]
                and attention_mask.shape[1] == x.shape[1]
            ):
                key_padding_mask = ~attention_mask.to(dtype=torch.bool)
            else:
                # Preserve support for callers that provide a square/causal
                # transformer mask rather than a batch padding mask.
                encoder_mask = attention_mask

        if self.gradient_checkpointing and self.training:
            hidden_states = x
            for layer in self.encoder.layers:
                hidden_states = checkpoint(
                    lambda states, layer=layer: layer(
                        states,
                        src_mask=encoder_mask,
                        src_key_padding_mask=key_padding_mask,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            if self.encoder.norm is not None:
                hidden_states = self.encoder.norm(hidden_states)
            return hidden_states

        return self.encoder(
            x,
            mask=encoder_mask,
            src_key_padding_mask=key_padding_mask,
        )

    def _slot_valid_mask(self, seq_len, device):
        """Return valid vocab ids for each motion-token slot.

        Slot j is trained with the id range
        [j * codebook_size, (j + 1) * codebook_size). This optional mask keeps
        inference from producing a token from the wrong residual/part slot.
        """
        ntpf = int(self.config.num_tokens_per_frame)
        codebook_size = int(self.config.codebook_size)
        vocab = int(self.config.vocab_size)
        slot_ids = torch.arange(seq_len, device=device) % ntpf
        vocab_ids = torch.arange(vocab, device=device)
        starts = slot_ids[:, None] * codebook_size
        ends = starts + codebook_size
        return (vocab_ids[None, :] >= starts) & (vocab_ids[None, :] < ends)

    def constrain_logits_by_slot(self, logits):
        if not getattr(self.config, "constrain_token_logits", False):
            return logits
        valid = self._slot_valid_mask(logits.size(1), logits.device).unsqueeze(0)
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def _fuse_encoded_audio(self, token_embeddings, audio_emb):
        """
        将音频特征融合到motion token embedding上。
        每帧的audio特征投影后加到该帧的所有4个motion token embedding上。

        Args:
            token_embeddings: (batch_size, seq_len, hidden_size)
                seq_len = num_frames * num_tokens_per_frame
            audio_features: (batch_size, num_frames, audio_feat_dim)
        Returns:
            (batch_size, seq_len, hidden_size) - 融合后的embedding
        """
        num_tokens_per_frame = self.config.num_tokens_per_frame

        # 投影音频特征: (B, num_frames, hidden_size)
        # 扩展到每帧的每个token: (B, num_frames, 1, hidden_size) -> (B, num_frames, 4, hidden_size)
        audio_emb = audio_emb.unsqueeze(2).expand(
            -1, -1, num_tokens_per_frame, -1
        )
        # reshape: (B, num_frames * 4, hidden_size) = (B, seq_len, hidden_size)
        audio_emb = audio_emb.reshape(
            audio_emb.size(0), -1, audio_emb.size(-1)
        )

        if audio_emb.shape[:2] != token_embeddings.shape[:2]:
            raise ValueError(
                "Audio/token length mismatch: "
                f"audio expands to {audio_emb.shape[1]} tokens but input has "
                f"{token_embeddings.shape[1]}"
            )

        return token_embeddings + audio_emb

    def _fuse_audio(self, token_embeddings, audio_features):
        return self._fuse_encoded_audio(
            token_embeddings, self.audio_encoder(audio_features)
        )

    def forward(
        self,
        input_ids,
        labels=None,
        audio_features=None,
        attention_mask=None,
        audio_prior_stage=None,
        negative_audio_features=None,
        return_audio_prior_details=False,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len) - Motion token IDs (with offset)
                seq_len = num_frames * num_tokens_per_frame = 5 * 4 = 20
            labels: (batch_size, seq_len) - Target token IDs, -100 for ignore
            audio_features: (batch_size, num_frames, audio_feat_dim)
                num_frames = 5, audio_feat_dim = 768
        Returns:
            If labels provided: (loss, predictions, accuracy)
            Else: logits
        """
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        # Embed motion tokens: (B, seq_len, hidden_size)
        x = self.embed_tokens(input_ids)

        # Add position embeddings
        x = x + self.position_emb(x.size(1), x.device)

        encoded_audio = None
        if audio_features is not None:
            encoded_audio = self.audio_encoder(audio_features)
            if self.config.audio_conditioning_mode in {
                "additive",
                "additive_residual_posterior",
            }:
                x = self._fuse_encoded_audio(x, encoded_audio)

        # Normalize and encode
        x = self.norm(x)
        output = self._run_encoder(x, attention_mask)  # (B, seq_len, hidden_size)

        # Get logits
        logits = self.out_head(output)  # (B, seq_len, vocab_size)
        audio_prior_details = {}
        if self.uses_audio_residual_posterior and encoded_audio is not None:
            ntpf = int(self.config.num_tokens_per_frame)
            frame_mask = None
            if (
                attention_mask is not None
                and attention_mask.dim() == 2
                and attention_mask.shape == input_ids.shape
            ):
                frame_mask = attention_mask.reshape(
                    attention_mask.shape[0], -1, ntpf
                )[..., 0].to(dtype=torch.bool)
            negative_encoded_audio = (
                self.audio_encoder(negative_audio_features)
                if negative_audio_features is not None
                else None
            )
            logits, audio_prior_details = self.audio_residual_posterior(
                logits,
                output,
                encoded_audio,
                input_ids,
                frame_mask=frame_mask,
                stage=audio_prior_stage,
                negative_encoded_audio=negative_encoded_audio,
            )
        logits = self.constrain_logits_by_slot(logits)

        if return_audio_prior_details:
            if labels is not None:
                raise ValueError(
                    "return_audio_prior_details is only supported when labels are omitted"
                )
            return {"logits": logits, "audio_prior": audio_prior_details}

        if labels is None:
            return logits

        # Calculate loss (only on masked positions, labels=-100 for non-masked)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            # Only compute accuracy on non-ignored positions
            valid_mask = labels != -100
            if valid_mask.sum() > 0:
                acc = (preds[valid_mask] == labels[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0)

        return loss, preds, acc

    @torch.no_grad()
    def generate_quantizer_coarse_to_fine(
        self,
        input_ids,
        audio_features,
        middle_mask=None,
        attention_mask=None,
    ):
        """Fill multipart RVQ tokens in q0 -> qN order.

        Each pass fills the selected quantizer for every missing temporal frame
        and every body part. Padding positions are excluded by ``attention_mask``.
        """
        device = next(self.parameters()).device
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        audio_features = torch.as_tensor(
            audio_features, dtype=torch.float32, device=device
        )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = torch.as_tensor(
                attention_mask, dtype=torch.bool, device=device
            )
        mask_token_id = int(
            getattr(self.config, "mask_token_id", self.config.vocab_size - 1)
        )
        if middle_mask is None:
            middle_mask = input_ids.eq(mask_token_id) & attention_mask
        else:
            middle_mask = torch.as_tensor(
                middle_mask, dtype=torch.bool, device=device
            ) & attention_mask
        if middle_mask.shape != input_ids.shape:
            raise ValueError(
                f"middle_mask shape {tuple(middle_mask.shape)} does not match "
                f"input_ids {tuple(input_ids.shape)}"
            )

        ntpf = int(self.config.num_tokens_per_frame)
        quantizers = int(getattr(self.config, "num_quantizers_per_part", 1))
        if ntpf % quantizers != 0:
            raise ValueError(
                f"num_tokens_per_frame={ntpf} is not divisible by "
                f"num_quantizers_per_part={quantizers}"
            )
        slots = torch.arange(input_ids.size(1), device=device).remainder(ntpf)
        quantizer_slots = slots.remainder(quantizers).view(1, -1)
        output_ids = input_ids.clone()

        for quantizer in range(quantizers):
            fill = (
                middle_mask
                & quantizer_slots.eq(quantizer)
                & output_ids.eq(mask_token_id)
            )
            if not bool(fill.any()):
                continue
            forward_kwargs = {
                "audio_features": audio_features,
                "attention_mask": attention_mask,
            }
            if self.uses_audio_residual_posterior:
                forward_kwargs["audio_prior_stage"] = quantizer
            logits = self.forward(output_ids, **forward_kwargs)
            predictions = logits.argmax(dim=-1)
            output_ids[fill] = predictions[fill]
        return output_ids

    def generate_sbs(self, input_ids, audio_features, generate_steps=1, attention_mask=None):
        """
        Step-by-step generation: 逐步填充mask token。
        按照预测概率从高到低依次填充。

        Args:
            input_ids: (batch_size, seq_len) - 包含mask token的输入
            audio_features: (batch_size, num_frames, audio_feat_dim)
            generate_steps: int - 生成步数
        Returns:
            output_ids: (batch_size, seq_len) - 填充后的token序列
        """
        mask_token_id = self.config.vocab_size - 1  # 2048
        num_tokens_per_frame = self.config.num_tokens_per_frame

        device = next(self.parameters()).device
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=device)
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features, dtype=torch.float32, device=device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        if self.uses_audio_residual_posterior:
            return self.generate_quantizer_coarse_to_fine(
                input_ids,
                audio_features,
                attention_mask=attention_mask,
            )

        step_out = input_ids.clone()
        batch_size = step_out.size(0)

        # 找到所有mask位置
        mask_positions = [
            (step_out[b] == mask_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(batch_size)
        ]

        total_masks = max(len(pos) for pos in mask_positions) if mask_positions else 0
        if total_masks == 0:
            return step_out

        masks_per_step = max(1, total_masks // generate_steps)

        for step in range(generate_steps):
            logits = self.forward(
                step_out,
                audio_features=audio_features,
                attention_mask=attention_mask,
            )
            # logits: (B, seq_len, vocab_size)
            preds_probs, preds_index = logits.max(dim=-1)

            for b in range(batch_size):
                remaining = mask_positions[b]
                if not remaining:
                    continue

                # 按概率排序，选择概率最高的位置填充
                probs_at_mask = [(pos, preds_probs[b, pos].item()) for pos in remaining]
                probs_at_mask.sort(key=lambda x: x[1], reverse=True)

                num_to_fill = min(masks_per_step, len(probs_at_mask))
                if step == generate_steps - 1:
                    num_to_fill = len(probs_at_mask)  # 最后一步填充所有剩余

                for i in range(num_to_fill):
                    pos = probs_at_mask[i][0]
                    step_out[b, pos] = preds_index[b, pos]
                    mask_positions[b].remove(pos)

        return step_out

    def interpolate(self, motion_frame1, motion_frame5, audio_features_5frames):
        """
        插帧推理接口：给定第1帧和第5帧的motion token，以及5帧的audio特征，
        预测中间3帧的motion token。

        Args:
            motion_frame1: list of 4 ints - 第1帧的4个motion token (raw, 0~511)
            motion_frame5: list of 4 ints - 第5帧的4个motion token (raw, 0~511)
            audio_features_5frames: (5, 768) numpy array or tensor - 5帧的hubert特征
        Returns:
            list of lists - 5帧的motion tokens [[frame1], [frame2], [frame3], [frame4], [frame5]]
                每个frame是4个raw token (0~511)
        """
        mask_token_id = self.config.vocab_size - 1  # 2048
        ntpf = self.config.num_tokens_per_frame
        codebook_size = self.config.codebook_size

        # offset: [0, 512, 1024, 1536]
        offsets = [codebook_size * i for i in range(ntpf)]

        input_tokens = []
        # Frame 1 (known)
        for j in range(ntpf):
            input_tokens.append(motion_frame1[j] + offsets[j])
        # Frames 2, 3, 4 (masked)
        for _ in range(3 * ntpf):
            input_tokens.append(mask_token_id)
        # Frame 5 (known)
        for j in range(ntpf):
            input_tokens.append(motion_frame5[j] + offsets[j])

        device = next(self.parameters()).device
        input_ids = torch.tensor([input_tokens], device=device)

        if not isinstance(audio_features_5frames, torch.Tensor):
            audio_features_5frames = torch.tensor(
                audio_features_5frames, dtype=torch.float32
            )
        audio_feat = audio_features_5frames.unsqueeze(0).to(device)

        # Generate
        output = self.generate_sbs(input_ids, audio_feat, generate_steps=1)
        output = output[0].cpu().tolist()

        # Parse output into frames
        frames = []
        for f in range(self.config.num_frames):
            frame_tokens = []
            for j in range(ntpf):
                token_id = output[f * ntpf + j]
                frame_tokens.append(token_id - offsets[j])
            frames.append(frame_tokens)

        return frames

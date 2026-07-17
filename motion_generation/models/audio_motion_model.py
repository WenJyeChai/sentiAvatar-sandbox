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
        rms_norm_eps=1e-6,
        hidden_act="gelu",
        num_parts=None,
        num_quantizers_per_part=4,
        max_gap_frames=15,
        audio_fusion_mode="legacy_additive",
        audio_adapter_layers=None,
        audio_adapter_dim=256,
        audio_adapter_heads=8,
        audio_router_dim=64,
        audio_router_embedding_dim=16,
        audio_additive_gate_scale=0.5,
        audio_relative_bias_max_distance=16,
        audio_adapter_target_only=True,
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
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.num_parts = int(
            num_parts
            if num_parts is not None
            else num_tokens_per_frame // max(1, int(num_quantizers_per_part))
        )
        self.num_quantizers_per_part = int(num_quantizers_per_part)
        self.max_gap_frames = int(max_gap_frames)
        self.audio_fusion_mode = str(audio_fusion_mode)
        self.audio_adapter_layers = list(audio_adapter_layers or [])
        self.audio_adapter_dim = int(audio_adapter_dim)
        self.audio_adapter_heads = int(audio_adapter_heads)
        self.audio_router_dim = int(audio_router_dim)
        self.audio_router_embedding_dim = int(audio_router_embedding_dim)
        self.audio_additive_gate_scale = float(audio_additive_gate_scale)
        self.audio_relative_bias_max_distance = int(
            audio_relative_bias_max_distance
        )
        self.audio_adapter_target_only = bool(audio_adapter_target_only)


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


class AudioConditioningRouter(nn.Module):
    """Produce small part/RVQ/stage-aware gates for shared audio paths."""

    def __init__(
        self,
        num_parts,
        num_quantizers,
        embedding_dim,
        hidden_dim,
        num_adapter_gates,
    ):
        super().__init__()
        self.num_quantizers = int(num_quantizers)
        self.part_embedding = nn.Embedding(num_parts, embedding_dim)
        self.quantizer_embedding = nn.Embedding(num_quantizers, embedding_dim)
        self.stage_embedding = nn.Embedding(num_quantizers + 1, embedding_dim)
        self.uncertainty_sentinel = nn.Parameter(torch.zeros(1))
        self.uncertainty_projection = nn.Linear(1, embedding_dim)
        geometry_dim = 9
        input_dim = 4 * embedding_dim + geometry_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, 1 + num_adapter_gates)

    def forward(
        self,
        part_ids,
        quantizer_ids,
        stage_ids,
        frame_ids,
        gap_lengths,
        max_gap_frames,
        uncertainty=None,
    ):
        batch = gap_lengths.shape[0]
        sequence = part_ids.shape[0]
        part = self.part_embedding(part_ids).unsqueeze(0).expand(batch, -1, -1)
        quantizer = self.quantizer_embedding(quantizer_ids).unsqueeze(0).expand(
            batch, -1, -1
        )
        stage = self.stage_embedding(stage_ids).unsqueeze(1).expand(-1, sequence, -1)

        gap = gap_lengths.float().clamp_min(1.0)
        gap_norm = (gap / max(1.0, float(max_gap_frames))).clamp(0.0, 1.0)
        anchor_distance = gap + 1.0
        distance_left = (
            frame_ids.float().unsqueeze(0) / anchor_distance.unsqueeze(1)
        ).clamp(0.0, 1.0)
        distance_right = (
            (anchor_distance.unsqueeze(1) - frame_ids.float().unsqueeze(0))
            / anchor_distance.unsqueeze(1)
        ).clamp(0.0, 1.0)
        gap_feature = gap_norm.unsqueeze(1).expand(-1, sequence)
        if uncertainty is None:
            uncertainty = self.uncertainty_sentinel.view(1, 1).expand(
                batch, sequence
            )
        else:
            uncertainty = torch.as_tensor(
                uncertainty, dtype=gap_feature.dtype, device=gap_feature.device
            )
            if uncertainty.shape != (batch, sequence):
                raise ValueError(
                    "audio uncertainty must have shape (batch, token_sequence)"
                )
        uncertainty_feature = self.uncertainty_projection(uncertainty.unsqueeze(-1))
        geometry = torch.stack(
            [
                gap_feature,
                torch.sin(math.pi * gap_feature),
                torch.cos(math.pi * gap_feature),
                distance_left,
                torch.sin(math.pi * distance_left),
                torch.cos(math.pi * distance_left),
                distance_right,
                torch.sin(math.pi * distance_right),
                torch.cos(math.pi * distance_right),
            ],
            dim=-1,
        )
        return self.output(
            self.trunk(
                torch.cat(
                    [part, quantizer, stage, uncertainty_feature, geometry], dim=-1
                )
            )
        )


class LowRankTemporalAudioAdapter(nn.Module):
    """Low-rank multi-head cross-attention over temporally aligned audio memory."""

    def __init__(
        self,
        hidden_size,
        bottleneck_dim,
        num_heads,
        relative_bias_max_distance,
    ):
        super().__init__()
        if bottleneck_dim % num_heads != 0:
            raise ValueError("audio_adapter_dim must be divisible by audio_adapter_heads")
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.bottleneck_dim // self.num_heads
        self.max_distance = int(relative_bias_max_distance)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, bottleneck_dim)
        self.key = nn.Linear(hidden_size, bottleneck_dim)
        self.value = nn.Linear(hidden_size, bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, hidden_size)
        self.relative_bias = nn.Embedding(2 * self.max_distance + 1, num_heads)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def project_memory(self, memory):
        normalized = self.memory_norm(memory)
        batch, frames, _ = normalized.shape
        key = self.key(normalized).reshape(
            batch, frames, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized).reshape(
            batch, frames, self.num_heads, self.head_dim
        ).transpose(1, 2)
        return key, value

    def forward(
        self,
        hidden_states,
        audio_memory,
        token_frame_ids,
        audio_frame_mask,
        gate,
        query_mask,
        projected_memory=None,
    ):
        batch, sequence, _ = hidden_states.shape
        query = self.query(self.query_norm(hidden_states)).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key, value = (
            projected_memory
            if projected_memory is not None
            else self.project_memory(audio_memory)
        )
        audio_frames = key.shape[2]
        audio_ids = torch.arange(audio_frames, device=hidden_states.device)
        offsets = token_frame_ids[:, None] - audio_ids[None, :]
        offsets = offsets.clamp(-self.max_distance, self.max_distance)
        bias = self.relative_bias(offsets + self.max_distance).permute(2, 0, 1)
        attention_bias = bias.unsqueeze(0).expand(batch, -1, -1, -1).to(query.dtype)
        attention_bias = attention_bias.masked_fill(
            ~audio_frame_mask[:, None, None, :],
            torch.finfo(query.dtype).min,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, sequence, self.bottleneck_dim
        )
        residual = self.output(attended)
        residual = residual * (2.0 * torch.sigmoid(gate)).unsqueeze(-1)
        residual = residual * query_mask.unsqueeze(-1).to(residual.dtype)
        return hidden_states + self.residual_scale.to(residual.dtype) * residual


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
        self._validate_audio_config()

        # Motion token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Audio encoder: 768 -> hidden_size
        self.audio_encoder = AudioEncoder(
            config.audio_feat_dim, config.hidden_size, config.dropout
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

        self.part_embedding = None
        self.quantizer_embedding = None
        self.stage_embedding = None
        self.audio_router = None
        self.audio_adapters = nn.ModuleDict()
        if config.audio_fusion_mode != "legacy_additive":
            self.part_embedding = nn.Embedding(config.num_parts, config.hidden_size)
            self.quantizer_embedding = nn.Embedding(
                config.num_quantizers_per_part, config.hidden_size
            )
            self.stage_embedding = nn.Embedding(
                config.num_quantizers_per_part + 1, config.hidden_size
            )
            adapter_layers = list(config.audio_adapter_layers)
            self.audio_router = AudioConditioningRouter(
                config.num_parts,
                config.num_quantizers_per_part,
                config.audio_router_embedding_dim,
                config.audio_router_dim,
                len(adapter_layers),
            )
            if config.audio_fusion_mode == "routed_cross_attention":
                for layer_number in adapter_layers:
                    self.audio_adapters[str(layer_number)] = LowRankTemporalAudioAdapter(
                        config.hidden_size,
                        config.audio_adapter_dim,
                        config.audio_adapter_heads,
                        config.audio_relative_bias_max_distance,
                    )

        # Output head
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Normalization
        self.norm = AudioMotionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.last_audio_conditioning_stats = {}

        # Initialize weights
        self.post_init()
        self._initialize_audio_identity()

    def _validate_audio_config(self):
        modes = {
            "legacy_additive",
            "routed_additive",
            "routed_cross_attention",
        }
        if self.config.audio_fusion_mode not in modes:
            raise ValueError(f"audio_fusion_mode must be one of {sorted(modes)}")
        expected = (
            int(self.config.num_parts)
            * int(self.config.num_quantizers_per_part)
        )
        if int(self.config.num_tokens_per_frame) != expected:
            raise ValueError(
                "num_tokens_per_frame must equal num_parts * "
                "num_quantizers_per_part"
            )
        layers = [int(value) for value in self.config.audio_adapter_layers]
        if len(set(layers)) != len(layers):
            raise ValueError("audio_adapter_layers must not contain duplicates")
        if any(value < 1 or value > int(self.config.num_layers) for value in layers):
            raise ValueError("audio_adapter_layers use one-based encoder layer numbers")
        if self.config.audio_fusion_mode != "routed_cross_attention" and layers:
            raise ValueError(
                "audio_adapter_layers require audio_fusion_mode=routed_cross_attention"
            )

    def _initialize_audio_identity(self):
        if self.part_embedding is not None:
            nn.init.zeros_(self.part_embedding.weight)
            nn.init.zeros_(self.quantizer_embedding.weight)
            nn.init.zeros_(self.stage_embedding.weight)
            nn.init.zeros_(self.audio_router.output.weight)
            nn.init.zeros_(self.audio_router.output.bias)
        for adapter in self.audio_adapters.values():
            nn.init.zeros_(adapter.relative_bias.weight)
            nn.init.zeros_(adapter.residual_scale)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _run_encoder(
        self,
        x,
        attention_mask=None,
        audio_memory=None,
        audio_frame_mask=None,
        adapter_gates=None,
        adapter_query_mask=None,
        adapter_memory_cache=None,
    ):
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

        use_manual_layers = bool(self.audio_adapters)
        if (self.gradient_checkpointing and self.training) or use_manual_layers:
            hidden_states = x
            adapter_index = {
                int(layer_number): index
                for index, layer_number in enumerate(self.config.audio_adapter_layers)
            }
            for layer_number, layer in enumerate(self.encoder.layers, start=1):
                if self.gradient_checkpointing and self.training:
                    hidden_states = checkpoint(
                        lambda states, layer=layer: layer(
                            states,
                            src_mask=encoder_mask,
                            src_key_padding_mask=key_padding_mask,
                        ),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = layer(
                        hidden_states,
                        src_mask=encoder_mask,
                        src_key_padding_mask=key_padding_mask,
                    )
                key = str(layer_number)
                if key in self.audio_adapters:
                    if audio_memory is None or adapter_gates is None:
                        raise ValueError("Audio adapters require encoded audio and gates")
                    hidden_states = self.audio_adapters[key](
                        hidden_states,
                        audio_memory,
                        self._token_frame_ids(hidden_states.shape[1], hidden_states.device),
                        audio_frame_mask,
                        adapter_gates[..., adapter_index[layer_number]],
                        adapter_query_mask,
                        projected_memory=(adapter_memory_cache or {}).get(key),
                    )
            if self.encoder.norm is not None:
                hidden_states = self.encoder.norm(hidden_states)
            return hidden_states

        return self.encoder(
            x,
            mask=encoder_mask,
            src_key_padding_mask=key_padding_mask,
        )

    def _token_frame_ids(self, seq_len, device):
        ntpf = int(self.config.num_tokens_per_frame)
        return torch.arange(seq_len, device=device).div(ntpf, rounding_mode="floor")

    def _conditioning_ids(self, input_ids, c2f_stage):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        ntpf = int(self.config.num_tokens_per_frame)
        quantizers = int(self.config.num_quantizers_per_part)
        slots = torch.arange(seq_len, device=device).remainder(ntpf)
        quantizer_ids = slots.remainder(quantizers)
        part_ids = slots.div(quantizers, rounding_mode="floor")
        if c2f_stage is None:
            stage_ids = torch.full(
                (batch,), quantizers, dtype=torch.long, device=device
            )
        else:
            stage_ids = torch.as_tensor(c2f_stage, dtype=torch.long, device=device)
            if stage_ids.dim() == 0:
                stage_ids = stage_ids.expand(batch)
            if stage_ids.shape != (batch,):
                raise ValueError("c2f_stage must be a scalar or one value per batch row")
            if bool(((stage_ids < 0) | (stage_ids >= quantizers)).any()):
                raise ValueError("c2f_stage is outside the configured RVQ range")
        return part_ids, quantizer_ids, stage_ids

    def _frame_masks_and_gaps(self, input_ids, attention_mask, gap_lengths):
        batch, seq_len = input_ids.shape
        ntpf = int(self.config.num_tokens_per_frame)
        if seq_len % ntpf:
            raise ValueError("Motion token sequence is not divisible by tokens_per_frame")
        frames = seq_len // ntpf
        if attention_mask is None:
            frame_mask = torch.ones((batch, frames), dtype=torch.bool, device=input_ids.device)
        else:
            frame_mask = attention_mask.to(dtype=torch.bool).reshape(batch, frames, ntpf).any(-1)
        if gap_lengths is None:
            gaps = frame_mask.sum(dim=-1).sub(2).clamp_min(1)
        else:
            gaps = torch.as_tensor(gap_lengths, dtype=torch.long, device=input_ids.device)
            if gaps.dim() == 0:
                gaps = gaps.expand(batch)
            if gaps.shape != (batch,):
                raise ValueError("gap_lengths must be a scalar or one value per batch row")
        return frame_mask, gaps

    def _expand_audio_memory(self, audio_memory, seq_len):
        ntpf = int(self.config.num_tokens_per_frame)
        expanded = audio_memory.unsqueeze(2).expand(-1, -1, ntpf, -1)
        expanded = expanded.reshape(audio_memory.shape[0], -1, audio_memory.shape[-1])
        if expanded.shape[1] != seq_len:
            raise ValueError(
                f"Audio expands to {expanded.shape[1]} tokens but input has {seq_len}"
            )
        return expanded

    def _adapter_query_mask(
        self,
        attention_mask,
        middle_mask,
        quantizer_ids,
        stage_ids,
        input_ids,
    ):
        valid = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(dtype=torch.bool)
        )
        if not self.config.audio_adapter_target_only:
            return valid
        if middle_mask is None:
            return valid
        target_quantizer = quantizer_ids.unsqueeze(0).eq(stage_ids.unsqueeze(1))
        return valid & middle_mask.to(dtype=torch.bool) & target_quantizer

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

    def _fuse_audio(self, token_embeddings, audio_features):
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
        audio_emb = self.audio_encoder(audio_features)

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

    def forward(
        self,
        input_ids,
        labels=None,
        audio_features=None,
        attention_mask=None,
        middle_mask=None,
        gap_lengths=None,
        c2f_stage=None,
        audio_uncertainty=None,
        encoded_audio=None,
        adapter_memory_cache=None,
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

        self.last_audio_conditioning_stats = {}
        x = self.embed_tokens(input_ids)
        x = x + self.position_emb(x.size(1), x.device)

        audio_memory = encoded_audio
        if self.config.audio_fusion_mode == "legacy_additive":
            if audio_memory is not None:
                x = x + self._expand_audio_memory(audio_memory, x.shape[1])
            elif audio_features is not None:
                x = self._fuse_audio(x, audio_features)
        else:
            if audio_memory is None and audio_features is not None:
                audio_memory = self.audio_encoder(audio_features)
            part_ids, quantizer_ids, stage_ids = self._conditioning_ids(
                input_ids, c2f_stage
            )
            x = (
                x
                + self.part_embedding(part_ids).unsqueeze(0)
                + self.quantizer_embedding(quantizer_ids).unsqueeze(0)
                + self.stage_embedding(stage_ids).unsqueeze(1)
            )
            audio_frame_mask, gaps = self._frame_masks_and_gaps(
                input_ids, attention_mask, gap_lengths
            )
            frame_ids = self._token_frame_ids(input_ids.shape[1], input_ids.device)
            gate_logits = self.audio_router(
                part_ids,
                quantizer_ids,
                stage_ids,
                frame_ids,
                gaps,
                self.config.max_gap_frames,
                uncertainty=audio_uncertainty,
            )
            additive_gate = 1.0 + self.config.audio_additive_gate_scale * torch.tanh(
                gate_logits[..., 0]
            )
            if audio_memory is not None:
                x = x + additive_gate.unsqueeze(-1) * self._expand_audio_memory(
                    audio_memory, x.shape[1]
                )
            adapter_gates = gate_logits[..., 1:] if self.audio_adapters else None
            adapter_query_mask = (
                self._adapter_query_mask(
                    attention_mask,
                    middle_mask,
                    quantizer_ids,
                    stage_ids,
                    input_ids,
                )
                if self.audio_adapters
                else None
            )
            with torch.no_grad():
                self.last_audio_conditioning_stats["additive_gate_mean"] = (
                    additive_gate.detach().float().mean()
                )
                for index, layer_number in enumerate(self.config.audio_adapter_layers):
                    gate = 2.0 * torch.sigmoid(adapter_gates[..., index])
                    self.last_audio_conditioning_stats[
                        f"adapter_layer{layer_number}_gate_mean"
                    ] = gate.detach().float().mean()
                    self.last_audio_conditioning_stats[
                        f"adapter_layer{layer_number}_scale"
                    ] = self.audio_adapters[str(layer_number)].residual_scale.detach().float()

        # Normalize and encode
        x = self.norm(x)
        output = self._run_encoder(
            x,
            attention_mask,
            audio_memory=audio_memory,
            audio_frame_mask=(audio_frame_mask if self.audio_adapters else None),
            adapter_gates=(adapter_gates if self.audio_adapters else None),
            adapter_query_mask=(adapter_query_mask if self.audio_adapters else None),
            adapter_memory_cache=adapter_memory_cache,
        )

        # Get logits
        logits = self.out_head(output)  # (B, seq_len, vocab_size)
        logits = self.constrain_logits_by_slot(logits)

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
        gap_lengths=None,
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

        frames = input_ids.shape[1] // int(self.config.num_tokens_per_frame)
        if gap_lengths is None:
            gap_lengths = middle_mask.reshape(
                input_ids.shape[0], frames, int(self.config.num_tokens_per_frame)
            ).any(-1).sum(-1)
        else:
            gap_lengths = torch.as_tensor(
                gap_lengths, dtype=torch.long, device=device
            )
        encoded_audio = self.audio_encoder(audio_features)
        adapter_memory_cache = {
            key: adapter.project_memory(encoded_audio)
            for key, adapter in self.audio_adapters.items()
        }

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
            logits = self.forward(
                output_ids,
                attention_mask=attention_mask,
                middle_mask=middle_mask,
                gap_lengths=gap_lengths,
                c2f_stage=quantizer,
                encoded_audio=encoded_audio,
                adapter_memory_cache=adapter_memory_cache,
            )
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

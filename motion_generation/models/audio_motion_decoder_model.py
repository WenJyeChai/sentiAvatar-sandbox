
class RotaryEmbedding1D(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        """
        Args:
            position_ids: (B, T)

        Returns:
            cos: (B, T, 1, head_dim)
            sin: (B, T, 1, head_dim)
        """
        freqs = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2)
        sin = emb.sin().unsqueeze(2)
        return cos, sin

    def rotate_half(x):
        """
        Args:
            x: (..., head_dim)

        Returns:
            rotated x
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(q, k, cos, sin):
        """
        Args:
            q: (B, T, num_heads, head_dim)
            k: (B, T, num_heads, head_dim)
            cos/sin: (B, T, 1, head_dim)

        Returns:
            q_rot, k_rot
        """
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

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

class AudioMotionCausalLM(PreTrainedModel):
    config_class = AudioMotionConfig

    def __init__(self, config: AudioMotionConfig):
        super().__init__(config)
        self.config = config

        # Motion token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Audio encoder: 768 -> hidden_size
        self.audio_encoder = AudioEncoder(
            config.audio_feat_dim, config.hidden_size, config.dropout
        )
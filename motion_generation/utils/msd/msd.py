"""
Motion Spectral Descriptor (MSD) module — WS1 of infill-v0.

Implements DynMask-style MSD (Zhou & Zhang et al., 2026) adapted to a
4-level residual-VQ token stream (SentiAvatar codec: 4 codebooks x 512,
2x temporal downsample => token rate = 10 Hz for 20 fps motion).

Pipeline (per clip):
    tokens (T,4) --sum of frozen level embeddings--> x (T,D)
    velocity v_t = x_t - x_{t-1}  (v_1 := v_2)
    sliding window W (replicate-padded), Type-II DCT along time per dim
    per-band energy f_k = ||F_{k,:}||_2  -> phi_t = f/(||f||+eps) in R^W
    Omega(t) = mean(phi_t)              # spectral SPREAD, range ~[0, 1/sqrt(W)]

Key decisions (see walkthrough / spec):
  * Token-rate default W = 8 (= 0.8 s @ 10 Hz). Decoded-motion variant
    uses W = 16 (= 0.8 s @ 20 fps) to match the time span.
  * Optional energy floor to kill the "normalized jitter looks broadband"
    failure mode (OFF for tokens, ON for decoded motion).
  * Loss weights: PER-CLIP z-score -> w = clip(1 + slope*z, lo, hi).
    Per-clip keeps mean weight ~1 per clip => idle-heavy clips are not
    globally defunded (persona guard). Corpus mode available via stats arg.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Codec embedding access (THE integration point — adapt key names to your ckpt)
# --------------------------------------------------------------------------

def load_rvq_codebooks(ckpt_path: str,
                       key_fmt: str = "quantizer.layers.{}.codebook.weight",
                       num_levels: int = 4) -> torch.Tensor:
    """Return frozen codebooks as a (num_levels, V, D) tensor.

    ADAPT `key_fmt` to your R-VQVAE state dict. Common alternatives:
        "quantizer.vq_layers.{}.embedding.weight"
        "rvq.codebooks.{}"
    Verify: shape per level == (512, D) and D matches the codec latent dim.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    books = []
    for lv in range(num_levels):
        key = key_fmt.format(lv)
        if key not in sd:
            raise KeyError(
                f"Codebook key '{key}' not found. Available candidate keys:\n"
                + "\n".join(k for k in sd if "cod" in k.lower() or "emb" in k.lower())
            )
        books.append(sd[key].float())
    return torch.stack(books, dim=0)  # (L, V, D)


def tokens_to_embedding(tokens: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """tokens: (T, L) int64 with PER-LEVEL indices in [0, V). codebooks: (L, V, D).
    If your pipeline stores offset-unified ids (level k adds k*512), subtract
    offsets BEFORE calling. Returns x: (T, D) = sum over levels (residual algebra).
    """
    T, L = tokens.shape
    assert L == codebooks.shape[0], "level count mismatch"
    x = torch.zeros(T, codebooks.shape[-1], dtype=codebooks.dtype, device=tokens.device)
    for lv in range(L):
        x += F.embedding(tokens[:, lv], codebooks[lv])
    return x


# --------------------------------------------------------------------------
# Core MSD
# --------------------------------------------------------------------------

def dct2_matrix(W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Type-II DCT basis, rows = bands k, cols = time n (no orthonormal scaling;
    matches DynMask Eq. 5)."""
    n = torch.arange(W, dtype=dtype, device=device)
    k = torch.arange(W, dtype=dtype, device=device)
    return torch.cos(torch.pi / W * (n[None, :] + 0.5) * k[:, None])  # (W, W)


@dataclass
class MSDConfig:
    W: int = 8                 # window length in FRAMES OF THE INPUT SEQUENCE
    eps: float = 1e-8
    energy_floor: float = 0.0  # absolute floor on ||f||_2; 0 disables.
                               # For decoded motion: set from corpus percentile
                               # (see scripts/precompute_msd.py --calibrate-floor)


@torch.no_grad()
def compute_msd(x: torch.Tensor, cfg: MSDConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (T, D) continuous per-frame features (token-embedding sums OR decoded
    motion features). Returns (phi (T, W), omega (T,)).
    """
    T, D = x.shape
    W = cfg.W
    # velocity with DynMask boundary convention v_1 = v_2
    v = x[1:] - x[:-1]                      # (T-1, D)
    v = torch.cat([v[:1], v], dim=0)        # (T, D)

    # centered sliding windows via replicate padding + unfold
    pad_l, pad_r = (W - 1) // 2, W // 2
    vp = F.pad(v.t().unsqueeze(0), (pad_l, pad_r), mode="replicate")[0].t()  # (T+W-1, D)
    windows = vp.unfold(0, W, 1)            # (T, D, W)  [unfold puts window last]
    windows = windows.permute(0, 2, 1)      # (T, W, D)

    C = dct2_matrix(W, device=x.device, dtype=x.dtype)          # (W, W)
    Fkd = torch.einsum("kw,twd->tkd", C, windows)               # (T, W, D)
    f = Fkd.pow(2).sum(dim=-1).clamp_min(0).sqrt()              # (T, W) band energies

    norm = f.norm(dim=-1, keepdim=True)                         # (T, 1)
    phi = f / (norm + cfg.eps)
    if cfg.energy_floor > 0:
        phi = torch.where(norm > cfg.energy_floor, phi, torch.zeros_like(phi))
    omega = phi.mean(dim=-1)                                    # (T,)
    return phi, omega


@torch.no_grad()
def msd_from_tokens(tokens: torch.Tensor, codebooks: torch.Tensor,
                    cfg: MSDConfig = MSDConfig(W=8)) -> tuple[torch.Tensor, torch.Tensor]:
    """Variant 1.1 — MSD on summed RVQ embeddings (token rate, 10 Hz)."""
    return compute_msd(tokens_to_embedding(tokens, codebooks), cfg)


@torch.no_grad()
def msd_from_motion(motion: torch.Tensor,
                    cfg: MSDConfig = MSDConfig(W=16)) -> tuple[torch.Tensor, torch.Tensor]:
    """Variant 1.2 — MSD on decoded continuous motion features (20 fps).
    `motion`: (L, 393) rotation features (or FK positions if you prefer).
    W=16 matches the 0.8 s span of W=8 at token rate. Consider energy_floor>0.
    """
    return compute_msd(motion, cfg)


def pool_motion_omega_to_token_rate(omega_motion: torch.Tensor,
                                    T_tok: int) -> torch.Tensor:
    """Average-pool 20 fps Omega onto the 10 Hz token grid for the 1.3
    agreement study. Handles odd lengths by trimming the tail frame."""
    L = (omega_motion.shape[0] // 2) * 2
    pooled = omega_motion[:L].view(-1, 2).mean(dim=-1)
    return pooled[:T_tok]


# --------------------------------------------------------------------------
# Loss weights (Mode A supervision reallocation)
# --------------------------------------------------------------------------

def omega_to_weights(omega: torch.Tensor,
                     lo: float = 0.5, hi: float = 2.0, slope: float = 0.5,
                     stats: tuple[float, float] | None = None) -> torch.Tensor:
    """w_t = clip(1 + slope * z_t, lo, hi).

    stats=None      -> PER-CLIP z-score (default; mean weight ~1 per clip,
                       idle clips not globally defunded — persona guard).
    stats=(mu,sig)  -> corpus-level z-score (explicit opt-in only).
    Degenerate clips (std ~ 0, e.g., pure idle) get uniform weight 1.
    """
    if stats is None:
        mu, sig = omega.mean(), omega.std()
    else:
        mu, sig = stats
    if float(sig) < 1e-6:
        return torch.ones_like(omega)
    z = (omega - mu) / sig
    return (1.0 + slope * z).clamp(lo, hi)

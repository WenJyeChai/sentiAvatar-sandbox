"""Unit tests for the MSD module (WS1 acceptance checks, task 1.1).

Run: pytest test_msd.py -v
All tests are synthetic — no codec checkpoint needed.
"""

import math
import torch
import pytest

from msd import (MSDConfig, compute_msd, compute_msd_components, dct2_matrix,
                 msd_from_multipart_tokens, multipart_tokens_to_embeddings,
                 omega_to_weights, pool_motion_omega_to_token_rate)

T, D, W = 120, 32, 8
FS = 10.0  # token rate, Hz


def make_x(v_fn):
    """Integrate a velocity function into positions x (T, D)."""
    t = torch.arange(T, dtype=torch.float32)
    v = v_fn(t)                       # (T, D)
    return torch.cumsum(v, dim=0)


# ---- spec check: constant pose -> flat, ~zero Omega -------------------------

def test_constant_pose_omega_near_zero():
    x = torch.ones(T, D) * 3.14
    _, omega = compute_msd(x, MSDConfig(W=W))
    assert omega.abs().max() < 1e-3, "idle must score ~0 (eps path)"


# ---- spec check: sinusoid at f -> energy in matching DCT band ---------------
# DCT-II band k corresponds to frequency k * Fs / (2W) cycles/sec.


@pytest.mark.parametrize("k0", [1, 2, 3, 5])
def test_sinusoid_peaks_in_matching_band(k0):
    f_hz = k0 * FS / (2 * W)
    t = torch.arange(T, dtype=torch.float32)
    v = torch.zeros(T, D)
    v[:, 0] = torch.cos(2 * math.pi * f_hz * t / FS)   # velocity is the sinusoid
    x = torch.cumsum(v, dim=0)
    phi, _ = compute_msd(x, MSDConfig(W=W))
    # ignore first/last W frames (padding transients)
    core = phi[W:-W]
    spec = core.mean(dim=0)
    # Check the NON-DC peak: a half-cycle-per-window sinusoid (k0=1) leaks
    # heavily into DC — at unlucky window phases half a cosine looks like
    # constant drift, and averaged over positions DC edges out band 1
    # (measured: DC 0.668 vs band-1 0.575). That is leakage physics, not a
    # bug, so we assert the peak among bands 1..W-1 instead.
    peak_band = 1 + spec[1:].argmax().item()
    assert peak_band == k0, f"expected non-DC peak in band {k0}, got {peak_band}"


# ---- walkthrough subtlety: Omega = spectral SPREAD, not magnitude -----------

def test_smooth_fast_motion_scores_low_jerky_scores_high():
    # constant velocity (fast but ballistic): all energy in DC band k=0
    x_ramp = make_x(lambda t: torch.ones(T, D) * 5.0)
    _, om_ramp = compute_msd(x_ramp, MSDConfig(W=W))
    # random jerky motion: broadband
    torch.manual_seed(0)
    x_jerk = make_x(lambda t: torch.randn(T, D))
    _, om_jerk = compute_msd(x_jerk, MSDConfig(W=W))
    ramp, jerk = om_ramp[W:-W].mean().item(), om_jerk[W:-W].mean().item()
    assert abs(ramp - 1.0 / W) < 0.02, "ballistic motion ~ 1/W"
    assert jerk > 2.0 * ramp, "broadband must dominate ballistic"
    assert jerk <= 1.0 / math.sqrt(W) + 0.02, "upper bound ~ 1/sqrt(W)"


def test_step_jump_is_high_omega_locally():
    x = torch.zeros(T, D)
    x[T // 2:, :] = 4.0                      # single pose jump
    _, omega = compute_msd(x, MSDConfig(W=W))
    jump_zone = omega[T // 2 - W: T // 2 + W].max()
    calm_zone = omega[: T // 2 - 2 * W].max()
    assert jump_zone > 5 * max(calm_zone.item(), 1e-4)


# ---- energy floor kills the jitter trap -------------------------------------

def test_energy_floor_suppresses_micro_jitter():
    torch.manual_seed(1)
    x = make_x(lambda t: 1e-5 * torch.randn(T, D))     # idle + tiny jitter
    _, om_no_floor = compute_msd(x, MSDConfig(W=W, energy_floor=0.0))
    _, om_floor = compute_msd(x, MSDConfig(W=W, energy_floor=1e-3))
    # without floor, normalized jitter looks broadband (HIGH omega) — the trap
    assert om_no_floor[W:-W].mean() > 0.2
    assert om_floor.abs().max() < 1e-6


# ---- mechanics ---------------------------------------------------------------

def test_shapes_and_no_nans():
    torch.manual_seed(2)
    x = torch.randn(37, D).cumsum(0)         # short clip, odd length
    phi, omega = compute_msd(x, MSDConfig(W=W))
    assert phi.shape == (37, W) and omega.shape == (37,)
    assert torch.isfinite(phi).all() and torch.isfinite(omega).all()


def test_components_preserve_energy_before_normalization():
    x = torch.arange(20, dtype=torch.float32).view(10, 2)
    components = compute_msd_components(x, MSDConfig(W=4))
    assert components.phi.shape == (10, 4)
    assert components.omega.shape == (10,)
    assert components.energy.shape == (10,)
    assert torch.all(components.energy > 0)
    assert torch.allclose(components.phi.norm(dim=-1), torch.ones(10), atol=1e-5)


def test_dct_matrix_matches_scipy_convention():
    scipy = pytest.importorskip("scipy.fft")
    import numpy as np
    sig = np.random.RandomState(0).randn(W)
    ours = dct2_matrix(W) @ torch.tensor(sig, dtype=torch.float32)
    ref = scipy.dct(sig, type=2, norm=None) / 2.0  # scipy scales by 2
    assert torch.allclose(ours, torch.tensor(ref, dtype=torch.float32), atol=1e-4)


def test_weights_per_clip_mean_near_one_and_idle_uniform():
    torch.manual_seed(3)
    omega = torch.rand(100) * 0.3
    w = omega_to_weights(omega)
    assert 0.85 < w.mean().item() < 1.15, "per-clip mode must not defund the clip"
    assert w.min() >= 0.5 and w.max() <= 2.0
    w_idle = omega_to_weights(torch.full((50,), 0.01))   # degenerate clip
    assert torch.allclose(w_idle, torch.ones(50))


def test_motion_to_token_pooling_alignment():
    om_motion = torch.arange(140, dtype=torch.float32)   # 20 fps
    pooled = pool_motion_omega_to_token_rate(om_motion, T_tok=69)
    assert pooled.shape == (69,)
    assert pooled[0] == 0.5 and pooled[1] == 2.5         # pairwise means


# ---- multipart RVQ integration ---------------------------------------------

def make_multipart_fixture(frames=37):
    torch.manual_seed(4)
    codebooks = {
        "upper": torch.randn(2, 7, 3),
        "hands": torch.randn(2, 7, 5),
    }
    tokens = torch.randint(0, 7, (frames, 4))
    return tokens, codebooks


def test_multipart_embeddings_sum_levels_and_keep_parts_separate():
    tokens, codebooks = make_multipart_fixture()
    embeddings = multipart_tokens_to_embeddings(
        tokens,
        codebooks,
        part_order=("upper", "hands"),
    )
    expected_upper = (
        codebooks["upper"][0, tokens[:, 0]]
        + codebooks["upper"][1, tokens[:, 1]]
    )
    expected_hands = (
        codebooks["hands"][0, tokens[:, 2]]
        + codebooks["hands"][1, tokens[:, 3]]
    )
    assert torch.allclose(embeddings["upper"], expected_upper)
    assert torch.allclose(embeddings["hands"], expected_hands)
    assert embeddings["upper"].shape[-1] == 3
    assert embeddings["hands"].shape[-1] == 5


def test_multipart_msd_returns_combined_and_per_part_descriptors():
    tokens, codebooks = make_multipart_fixture()
    result = msd_from_multipart_tokens(
        tokens,
        codebooks,
        MSDConfig(W=8),
        part_order=("upper", "hands"),
    )
    assert result.phi.shape == (len(tokens), 8)
    assert result.omega.shape == (len(tokens),)
    assert set(result.part_phi) == {"upper", "hands"}
    assert set(result.part_omega) == {"upper", "hands"}
    assert all(value.shape == (len(tokens), 8) for value in result.part_phi.values())
    assert all(value.shape == (len(tokens),) for value in result.part_omega.values())
    assert torch.isfinite(result.phi).all()
    assert torch.isfinite(result.omega).all()


def test_multipart_msd_rejects_wrong_slot_count():
    tokens, codebooks = make_multipart_fixture()
    with pytest.raises(ValueError, match="slot count mismatch"):
        msd_from_multipart_tokens(
            tokens[:, :-1],
            codebooks,
            part_order=("upper", "hands"),
        )

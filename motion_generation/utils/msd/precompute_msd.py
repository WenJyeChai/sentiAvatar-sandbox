#!/usr/bin/env python3
"""Run MSD verification and cache experiments with SentiAvatar multipart RVQ."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

try:
    from .msd import (
        MSDConfig,
        compute_msd,
        msd_from_multipart_tokens,
        omega_to_weights,
        pool_motion_omega_to_token_rate,
    )
    from .multipart_adapter import (
        MultipartCodecAdapter,
        MultipartCodebookSet,
        MultipartTokenDataset,
        apply_speed_variant,
        cache_key,
        resolve_device,
    )
except ImportError:
    from msd import (  # type: ignore
        MSDConfig,
        compute_msd,
        msd_from_multipart_tokens,
        omega_to_weights,
        pool_motion_omega_to_token_rate,
    )
    from multipart_adapter import (  # type: ignore
        MultipartCodecAdapter,
        MultipartCodebookSet,
        MultipartTokenDataset,
        apply_speed_variant,
        cache_key,
        resolve_device,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "SuSuInterActs" / "SuSuInterActs"
DEFAULT_CODEC_ROOT = PROJECT_ROOT / "checkpoints" / "multipart_rvqvae"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "multipart"
DEFAULT_PART_ORDER = ("upper", "lower", "feet", "hands")


def default_checkpoint(part: str) -> Path:
    return (
        DEFAULT_CODEC_ROOT
        / f"rvq_{part}_512x4_bs256_cosine"
        / "model"
        / "best.pth"
    )


def parse_part_weights(text: str | None) -> dict[str, float]:
    weights = {part: 1.0 for part in DEFAULT_PART_ORDER}
    if not text:
        return weights
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "--part-weights must look like upper=1,lower=1,feet=1,hands=1"
            )
        part, raw_value = (piece.strip() for piece in item.split("=", 1))
        if part not in weights:
            raise ValueError(f"Unknown part in --part-weights: {part}")
        value = float(raw_value)
        if value < 0:
            raise ValueError(f"Part weights must be non-negative, got {part}={value}")
        weights[part] = value
    if not any(value > 0 for value in weights.values()):
        raise ValueError("At least one part weight must be positive")
    return weights


def load_context(args: argparse.Namespace, *, require_decoder: bool = False):
    device = resolve_device(args.device)
    checkpoints = {
        part: Path(getattr(args, f"{part}_ckpt")).expanduser().resolve()
        for part in DEFAULT_PART_ORDER
    }
    codec = (
        MultipartCodecAdapter.from_checkpoints(checkpoints, device, DEFAULT_PART_ORDER)
        if require_decoder
        else MultipartCodebookSet.from_checkpoints(checkpoints, device, DEFAULT_PART_ORDER)
    )
    dataset = MultipartTokenDataset(
        Path(args.data_dir),
        Path(args.token_dir) if args.token_dir is not None else None,
    )
    dataset.validate_layout(codec)
    codebooks = codec.codebooks
    part_weights = parse_part_weights(args.part_weights)

    token_fps = float(args.motion_fps) / float(codec.unit_length)
    w_token = max(2, int(round(float(args.window_seconds) * token_fps)))
    w_motion = max(2, int(round(float(args.window_seconds) * float(args.motion_fps))))
    print(
        "Multipart MSD: "
        f"parts={list(codec.part_order)}, q={codec.num_quantizers}, "
        f"codes={codec.codebook_size}, tokens/frame={codec.tokens_per_frame}"
    )
    print(
        f"Timing: motion={args.motion_fps:g} fps, token={token_fps:g} fps, "
        f"unit_length={codec.unit_length}, W_token={w_token}, W_motion={w_motion}"
    )
    print(f"Part weights: {part_weights}")
    return device, codec, dataset, codebooks, part_weights, w_token, w_motion


def safe_output_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")[-180:]


def finite_spearman(a: torch.Tensor, b: torch.Tensor) -> float | None:
    from scipy.stats import spearmanr

    length = min(int(a.shape[0]), int(b.shape[0]))
    if length < 2:
        return None
    a_np = a[:length].detach().float().cpu().numpy()
    b_np = b[:length].detach().float().cpu().numpy()
    if float(a_np.std()) < 1e-6 or float(b_np.std()) < 1e-6:
        return None
    rho = float(spearmanr(a_np, b_np).correlation)
    return rho if math.isfinite(rho) else None


def descriptor(codec, codebooks, tokens, cfg, part_weights):
    return msd_from_multipart_tokens(
        tokens,
        codebooks,
        cfg,
        part_order=codec.part_order,
        part_weights=part_weights,
    )


def concatenate_decoded_parts(
    decoded_parts: Mapping[str, torch.Tensor],
    part_order,
    part_weights: Mapping[str, float],
) -> torch.Tensor:
    return torch.cat(
        [
            decoded_parts[part] * float(part_weights.get(part, 1.0)) ** 0.5
            for part in part_order
        ],
        dim=-1,
    )


def cmd_inspect(args: argparse.Namespace) -> None:
    _, codec, dataset, codebooks, part_weights, w_token, _ = load_context(
        args,
        require_decoder=True,
    )
    count = 0
    for name, tokens in dataset.iter_tokens(
        args.split,
        expected_slots=codec.tokens_per_frame,
        device=codec.device,
        limit=args.n,
    ):
        result = descriptor(codec, codebooks, tokens, MSDConfig(W=w_token), part_weights)
        decoded = codec.decode_parts(tokens)
        print(
            f"{name}: tokens={tuple(tokens.shape)}, phi={tuple(result.phi.shape)}, "
            f"omega=[{float(result.omega.min()):.4f},{float(result.omega.max()):.4f}], "
            f"decoded={{{', '.join(f'{p}:{tuple(x.shape)}' for p, x in decoded.items())}}}"
        )
        count += 1
    if count == 0:
        raise RuntimeError(f"No multipart token clips found for split '{args.split}'")


def cmd_agreement(args: argparse.Namespace) -> None:
    _, codec, dataset, codebooks, part_weights, w_token, w_motion = load_context(
        args,
        require_decoder=True,
    )
    rows: list[dict[str, object]] = []
    skipped = 0
    for name, tokens in dataset.iter_tokens(
        args.split,
        expected_slots=codec.tokens_per_frame,
        device=codec.device,
        limit=args.n,
    ):
        token_result = descriptor(
            codec,
            codebooks,
            tokens,
            MSDConfig(W=w_token),
            part_weights,
        )
        decoded_parts = codec.decode_parts(tokens)
        decoded_combined = concatenate_decoded_parts(
            decoded_parts,
            codec.part_order,
            part_weights,
        )
        _, omega_motion = compute_msd(
            decoded_combined,
            MSDConfig(W=w_motion, energy_floor=args.energy_floor),
        )
        omega_motion = pool_motion_omega_to_token_rate(
            omega_motion,
            token_result.omega.shape[0],
            unit_length=codec.unit_length,
        )
        combined_rho = finite_spearman(token_result.omega, omega_motion)
        if combined_rho is None:
            skipped += 1
            continue

        row: dict[str, object] = {
            "clip_id": name,
            "token_frames": int(tokens.shape[0]),
            "spearman": combined_rho,
        }
        for part in codec.part_order:
            _, part_motion_omega = compute_msd(
                decoded_parts[part],
                MSDConfig(W=w_motion, energy_floor=args.energy_floor),
            )
            part_motion_omega = pool_motion_omega_to_token_rate(
                part_motion_omega,
                token_result.part_omega[part].shape[0],
                unit_length=codec.unit_length,
            )
            row[f"spearman_{part}"] = finite_spearman(
                token_result.part_omega[part],
                part_motion_omega,
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("Agreement study produced no scorable clips")

    metric_names = ["spearman", *[f"spearman_{part}" for part in codec.part_order]]
    medians: dict[str, float] = {}
    print(f"Agreement clips scored={len(rows)}, skipped_constant={skipped}")
    for metric in metric_names:
        values = np.asarray(
            [float(row[metric]) for row in rows if row.get(metric) is not None],
            dtype=np.float64,
        )
        if not len(values):
            continue
        medians[metric] = float(np.median(values))
        print(
            f"  {metric:16s} median={np.median(values):.3f} "
            f"p10={np.percentile(values, 10):.3f} "
            f"frac>{args.gate_threshold:g}={(values > args.gate_threshold).mean():.2%}"
        )

    combined_pass = medians.get("spearman", float("-inf")) > args.gate_threshold
    part_pass = all(
        medians.get(f"spearman_{part}", float("-inf")) > args.gate_threshold
        for part in codec.part_order
    )
    print(
        "GATE:",
        "PASS: combined and every part support token-level MSD"
        if combined_pass and part_pass
        else "REVIEW: use decoded MSD or investigate the failing part descriptors",
    )

    out = Path(args.out or DEFAULT_OUTPUT_ROOT / "agreement.csv").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "token_frames", *metric_names]
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Agreement rows: {out}")


def cmd_heatmaps(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, codec, dataset, codebooks, part_weights, w_token, _ = load_context(args)
    requested = set(args.clips or [])
    outdir = Path(args.out or DEFAULT_OUTPUT_ROOT / "heatmaps").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    scan_limit = None if requested else args.n
    for name, tokens in dataset.iter_tokens(
        args.split,
        expected_slots=codec.tokens_per_frame,
        device=codec.device,
        limit=scan_limit,
    ):
        if requested and name not in requested:
            continue
        result = descriptor(codec, codebooks, tokens, MSDConfig(W=w_token), part_weights)
        fig, (heat_ax, curve_ax) = plt.subplots(
            2,
            1,
            figsize=(13, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        heat_ax.imshow(
            result.phi.detach().cpu().numpy().T,
            aspect="auto",
            origin="lower",
            cmap="magma",
        )
        heat_ax.set_ylabel("DCT band")
        curve_ax.plot(result.omega.detach().cpu().numpy(), label="combined", linewidth=2)
        for part in codec.part_order:
            curve_ax.plot(
                result.part_omega[part].detach().cpu().numpy(),
                label=part,
                alpha=0.8,
            )
        curve_ax.set_ylabel("Omega")
        curve_ax.set_xlabel("token frame")
        curve_ax.legend(ncol=len(codec.part_order) + 1, fontsize=8)
        fig.suptitle(name)
        fig.tight_layout()
        path = outdir / f"{safe_output_name(name)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")
        written += 1
        if not requested and written >= args.n:
            break
        if requested and written >= len(requested):
            break
    if written == 0:
        raise RuntimeError("No requested clips were found in the token dataset")


def cache_arrays(result, weight_args, part_order) -> dict[str, np.ndarray]:
    lo, hi, slope = weight_args
    arrays: dict[str, np.ndarray] = {
        "phi": result.phi.detach().cpu().numpy().astype(np.float16),
        "omega": result.omega.detach().cpu().numpy().astype(np.float32),
        "weight": omega_to_weights(result.omega, lo=lo, hi=hi, slope=slope)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }
    for part in part_order:
        arrays[f"phi_{part}"] = (
            result.part_phi[part].detach().cpu().numpy().astype(np.float16)
        )
        arrays[f"omega_{part}"] = (
            result.part_omega[part].detach().cpu().numpy().astype(np.float32)
        )
        arrays[f"weight_{part}"] = (
            omega_to_weights(result.part_omega[part], lo=lo, hi=hi, slope=slope)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return arrays


def cmd_cache(args: argparse.Namespace) -> None:
    _, codec, dataset, codebooks, part_weights, w_token, w_motion = load_context(args)
    outdir = Path(args.out or DEFAULT_OUTPUT_ROOT / "cache_token_msd").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    written = cached = 0

    for split in args.splits:
        split_clips = 0
        for name, base_tokens in dataset.iter_tokens(
            split,
            expected_slots=codec.tokens_per_frame,
            device=codec.device,
            limit=args.n,
        ):
            split_clips += 1
            for speed in args.speeds:
                from_tokens = base_tokens
                if abs(float(speed) - 1.0) >= 1e-6:
                    from_tokens = apply_speed_variant(base_tokens, speed)

                key = cache_key(name, speed)
                path = outdir / f"{key}.npz"
                was_cached = path.exists() and not args.overwrite
                if not was_cached:
                    result = descriptor(
                        codec,
                        codebooks,
                        from_tokens,
                        MSDConfig(W=w_token),
                        part_weights,
                    )
                    arrays = cache_arrays(
                        result,
                        (args.weight_lo, args.weight_hi, args.weight_slope),
                        codec.part_order,
                    )
                    np.savez_compressed(path, **arrays)
                    written += 1
                else:
                    cached += 1
                entries.append(
                    {
                        "key": key,
                        "clip": name,
                        "split": split,
                        "speed": float(speed),
                        "T": int(from_tokens.shape[0]),
                        "path": path.name,
                        "cached": was_cached,
                    }
                )
        print(f"Cache split={split}: processed {split_clips} clips")

    manifest = {
        "metadata": {
            **codec.metadata(),
            "data_dir": str(dataset.data_dir),
            "token_dir": str(dataset.token_dir),
            "window_seconds": float(args.window_seconds),
            "W_token": w_token,
            "W_motion": w_motion,
            "part_weights": part_weights,
            "weight_lo": args.weight_lo,
            "weight_hi": args.weight_hi,
            "weight_slope": args.weight_slope,
        },
        "entries": entries,
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Cache complete: entries={len(entries)}, written={written}, reused={cached}, "
        f"manifest={manifest_path}"
    )


def cmd_calibrate_floor(args: argparse.Namespace) -> None:
    _, codec, dataset, _, _, _, w_motion = load_context(args, require_decoder=True)
    energies = []
    clips = 0
    for _, tokens in dataset.iter_tokens(
        args.split,
        expected_slots=codec.tokens_per_frame,
        device=codec.device,
        limit=args.n,
    ):
        motion = codec.decode_combined_features(tokens)
        velocity = motion[1:] - motion[:-1]
        if velocity.shape[0] < w_motion:
            continue
        energy = torch.sqrt(
            torch.nn.functional.avg_pool1d(
                (velocity**2).sum(-1, keepdim=True).t().unsqueeze(0),
                w_motion,
                1,
            )[0, 0]
            * w_motion
        )
        energies.append(energy.detach().cpu())
        clips += 1
    if not energies:
        raise RuntimeError("No clips were long enough to calibrate the decoded-motion floor")
    values = torch.cat(energies).numpy()
    print(f"Decoded clips: {clips}, windows: {len(values)}")
    print(
        f"Suggested energy_floor={np.percentile(values, args.floor_percentile):.6e} "
        f"(p{args.floor_percentile:g}; p50={np.percentile(values, 50):.6e}, "
        f"p95={np.percentile(values, 95):.6e})"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--token-dir", type=Path, default=None)
    for part in DEFAULT_PART_ORDER:
        parser.add_argument(
            f"--{part}-ckpt",
            type=Path,
            default=default_checkpoint(part),
        )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion-fps", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=0.8)
    parser.add_argument("--part-weights", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multipart SentiAvatar Motion Spectral Descriptor experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Validate codecs/tokens and decode a few clips")
    add_common_args(inspect_parser)

    agreement = subparsers.add_parser("agreement", help="Token-vs-decoded MSD agreement gate")
    add_common_args(agreement)
    agreement.add_argument("--energy-floor", type=float, default=0.0)
    agreement.add_argument("--gate-threshold", type=float, default=0.9)

    heatmaps = subparsers.add_parser("heatmaps", help="Combined and per-part MSD plots")
    add_common_args(heatmaps)
    heatmaps.add_argument("--clips", nargs="*", default=[])

    cache = subparsers.add_parser("cache", help="Precompute combined and per-part MSD caches")
    add_common_args(cache)
    cache.set_defaults(n=None)
    cache.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    cache.add_argument("--speeds", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--weight-lo", type=float, default=0.5)
    cache.add_argument("--weight-hi", type=float, default=2.0)
    cache.add_argument("--weight-slope", type=float, default=0.5)

    floor = subparsers.add_parser("calibrate-floor", help="Calibrate decoded-motion energy floor")
    add_common_args(floor)
    floor.set_defaults(split="train", n=200)
    floor.add_argument("--floor-percentile", type=float, default=5.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n is not None and args.n <= 0:
        raise ValueError("--n must be positive or omitted")
    if args.motion_fps <= 0 or args.window_seconds <= 0:
        raise ValueError("--motion-fps and --window-seconds must be positive")
    commands = {
        "inspect": cmd_inspect,
        "agreement": cmd_agreement,
        "heatmaps": cmd_heatmaps,
        "cache": cmd_cache,
        "calibrate-floor": cmd_calibrate_floor,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

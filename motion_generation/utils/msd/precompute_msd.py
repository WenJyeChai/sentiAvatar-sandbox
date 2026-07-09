"""WS1 tasks 1.3–1.5: agreement study, visual sanity, corpus cache.

You must adapt exactly TWO functions to your codebase (marked ADAPT):
    iter_clips()      -> yields (clip_id, tokens (T,4) int64) per speed variant
    decode_tokens()   -> tokens -> continuous motion features (L, 393) @ 20 fps

Usage:
    python precompute_msd.py agreement  --ckpt codec.pt --n 500
    python precompute_msd.py heatmaps   --ckpt codec.pt --clips id1 id2 ...
    python precompute_msd.py cache      --ckpt codec.pt --out msd_cache/
    python precompute_msd.py calibrate-floor --ckpt codec.pt --n 200
"""

from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import torch

from msd import (MSDConfig, load_rvq_codebooks, msd_from_tokens,
                 msd_from_motion, pool_motion_omega_to_token_rate,
                 omega_to_weights)

W_TOKEN, W_MOTION = 8, 16   # matched 0.8 s spans @ 10 Hz / 20 fps


# ----------------------------- ADAPT THESE -----------------------------------

def iter_clips(split="train", speed=1.0, limit=None):
    """ADAPT: yield (clip_id: str, tokens: LongTensor (T, 4) per-level indices).
    Subtract level offsets here if your storage uses the unified 2048 vocab."""
    raise NotImplementedError("wire to your tokenized-dataset reader")


def decode_tokens(tokens, codec):
    """ADAPT: run your frozen R-VQVAE decoder. Returns (L, 393) float @ 20 fps."""
    raise NotImplementedError("wire to your codec decoder")


def load_codec_for_decoding(ckpt_path):
    """ADAPT: load the full codec (only needed for `agreement`/`calibrate-floor`)."""
    raise NotImplementedError

# ------------------------------------------------------------------------------


def cmd_agreement(args):
    """Task 1.3 — GATE: median per-clip Spearman(Omega_token, Omega_motion) > 0.9."""
    from scipy.stats import spearmanr
    books = load_rvq_codebooks(args.ckpt)
    codec = load_codec_for_decoding(args.ckpt)
    floor = args.energy_floor
    rows = []
    for clip_id, tokens in iter_clips("train", 1.0, limit=args.n):
        _, om_tok = msd_from_tokens(tokens, books, MSDConfig(W=W_TOKEN))
        motion = decode_tokens(tokens, codec)
        _, om_mot = msd_from_motion(motion, MSDConfig(W=W_MOTION, energy_floor=floor))
        om_mot = pool_motion_omega_to_token_rate(om_mot, om_tok.shape[0])
        if om_tok.std() < 1e-6 or om_mot.std() < 1e-6:
            continue  # pure-idle clip: correlation undefined, skip (report count)
        rho, _ = spearmanr(om_tok.numpy(), om_mot.numpy())
        rows.append((clip_id, float(rho)))
    rhos = np.array([r for _, r in rows])
    print(f"clips scored: {len(rows)}   median rho: {np.median(rhos):.3f}   "
          f"p10: {np.percentile(rhos,10):.3f}   frac>0.9: {(rhos>0.9).mean():.2%}")
    verdict = "PASS -> use token-level MSD (1.1) everywhere" \
              if np.median(rhos) > 0.9 else \
              "FAIL -> precompute decoded-motion MSD (1.2) instead"
    print("GATE:", verdict)
    out = pathlib.Path(args.out or "agreement_study.csv")
    out.write_text("clip_id,spearman\n" + "\n".join(f"{c},{r:.4f}" for c, r in rows))
    print("per-clip results ->", out)


def cmd_heatmaps(args):
    """Task 1.4 — Omega curves + phi heatmaps for hand-picked clips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    books = load_rvq_codebooks(args.ckpt)
    ids = set(args.clips)
    for clip_id, tokens in iter_clips("train", 1.0):
        if clip_id not in ids:
            continue
        phi, om = msd_from_tokens(tokens, books, MSDConfig(W=W_TOKEN))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
        a1.imshow(phi.T.numpy(), aspect="auto", origin="lower", cmap="magma")
        a1.set_ylabel("DCT band k")
        a2.plot(om.numpy()); a2.set_ylabel("Omega"); a2.set_xlabel("token frame (10 Hz)")
        a2.axhline(1 / W_TOKEN, ls="--", lw=0.8, label="ballistic 1/W")
        a2.legend(loc="upper right", fontsize=8)
        fig.suptitle(clip_id)
        fig.savefig(f"heatmap_{clip_id}.png", dpi=140, bbox_inches="tight")
        print("wrote", f"heatmap_{clip_id}.png")
        # Eyeball check: high-Omega must align with gesture strokes; fast
        # smooth arcs SHOULD sit near 1/W — that is correct, not a bug.


def cmd_cache(args):
    """Task 1.5 — precompute phi/omega/weights for all splits & speed variants."""
    books = load_rvq_codebooks(args.ckpt)
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split in ("train", "val", "test"):
        for speed in (0.9, 1.0, 1.1):
            for clip_id, tokens in iter_clips(split, speed):
                phi, om = msd_from_tokens(tokens, books, MSDConfig(W=W_TOKEN))
                w = omega_to_weights(om)              # per-clip mode (default)
                key = f"{clip_id}@{speed:.1f}"
                np.savez_compressed(outdir / f"{key}.npz",
                                    phi=phi.numpy().astype(np.float16),
                                    omega=om.numpy().astype(np.float32),
                                    weight=w.numpy().astype(np.float32))
                manifest.append({"key": key, "split": split, "T": int(om.shape[0])})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"cached {len(manifest)} entries -> {outdir}")
    # Dataloader contract: load once per worker, index by key; assert cache hit
    # (spec 1.5 check: NO on-the-fly DCT during training).


def cmd_calibrate_floor(args):
    """Helper for the decoded-motion energy floor: 5th percentile of window
    energies over active (non-idle) frames — below it, phi is zeroed."""
    codec = load_codec_for_decoding(args.ckpt)
    energies = []
    for _, tokens in iter_clips("train", 1.0, limit=args.n):
        motion = decode_tokens(tokens, codec)
        v = motion[1:] - motion[:-1]
        # window energy proxy: rolling L2 over W_MOTION frames
        e = torch.sqrt(torch.nn.functional.avg_pool1d(
            (v ** 2).sum(-1, keepdim=True).t()[None], W_MOTION, 1)[0, 0] * W_MOTION)
        energies.append(e)
    e = torch.cat(energies).numpy()
    print(f"suggested energy_floor = {np.percentile(e, 5):.4e} "
          f"(5th pct; distribution p50={np.percentile(e,50):.3e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("agreement", "heatmaps", "cache", "calibrate-floor"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True)
        s.add_argument("--n", type=int, default=500)
        s.add_argument("--out", default=None)
        s.add_argument("--energy-floor", type=float, default=0.0)
        s.add_argument("--clips", nargs="*", default=[])
    a = p.parse_args()
    dispatch = {"agreement": cmd_agreement, "heatmaps": cmd_heatmaps,
                "cache": cmd_cache, "calibrate-floor": cmd_calibrate_floor}
    dispatch[a.cmd](a)

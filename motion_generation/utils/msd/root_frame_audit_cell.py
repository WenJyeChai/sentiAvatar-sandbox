# ============================================================================
# ROOT-FRAME AUDIT
#
# Paste/run after the notebook diagnostic helper cells. This uses the notebook
# names: valid_names, get_tokens, decode_tokens_to_body, load_motion_dict,
# decoded_body_to_body_joint_positions, body_features_to_quat_motion,
# get_fk_model, BODY_JOINTS_ID, DEVICE, RAW_MOTION_FPS, W_MOTION.
# ============================================================================
import numpy as np
import torch
from scipy.stats import spearmanr

N_AUDIT = 20
audit_names = valid_names(SPLIT_FOR_QUICK_CHECKS, limit=N_AUDIT)
PELVIS = 0  # pelvis within BODY_JOINTS_ID ordering


def root_relative(P):
    return P - P[:, PELVIS:PELVIS + 1]


def omega_spearman(a, b):
    n = min(a.shape[0], b.shape[0])
    if n == 0 or a[:n].std() <= 1e-6 or b[:n].std() <= 1e-6:
        return None
    return float(spearmanr(a[:n].numpy(), b[:n].numpy()).correlation)


def best_scale_rmse(src, dst):
    src_flat = src.reshape(-1).float()
    dst_flat = dst.reshape(-1).float()
    scale = float((src_flat @ dst_flat) / ((src_flat @ src_flat) + 1e-12))
    rmse_best = float(torch.sqrt(((src * scale - dst) ** 2).mean()))
    rmse_x100 = float(torch.sqrt(((src * 100.0 - dst) ** 2).mean()))
    return scale, rmse_x100, rmse_best


report = []
for name in tqdm(audit_names, desc="root audit"):
    tokens, _ = get_tokens(name)
    body_decoded = decode_tokens_to_body(tokens)

    # FK path actually used by the decoded-motion MSD study.
    pos_dec_fk = decoded_body_to_body_joint_positions(name, body_decoded)  # (L, 3J)
    L = pos_dec_fk.shape[0]
    P_dec = pos_dec_fk.view(L, -1, 3)
    root_traj = P_dec[:, PELVIS]
    root_net = float((root_traj[-1] - root_traj[0]).norm())
    root_std = float(root_traj.std(dim=0).norm())

    # Max offset is not the MSD-relevant quantity; velocity effect is.
    P_dec_rr = root_relative(P_dec)
    sub_effect = float((P_dec - P_dec_rr).abs().max())
    if L > 1:
        vel = P_dec[1:] - P_dec[:-1]
        vel_rr = P_dec_rr[1:] - P_dec_rr[:-1]
        sub_velocity_effect = float((vel - vel_rr).abs().max())
    else:
        sub_velocity_effect = 0.0

    motion = load_motion_dict(name)
    has_pos = "positions" in motion
    body_raw = np.asarray(motion["body"], dtype=np.float32)
    root_channel_mean = float(np.linalg.norm(body_raw[:, :3], axis=1).mean())

    row = dict(
        name=name.split("/")[-1],
        root_net=root_net,
        root_std=root_std,
        sub_effect=sub_effect,
        sub_velocity_effect=sub_velocity_effect,
        has_positions=has_pos,
        root_channel_mean=root_channel_mean,
    )

    if has_pos:
        gtp_all = torch.tensor(np.asarray(motion["positions"], dtype=np.float32))
        gtp_all = gtp_all.reshape(gtp_all.shape[0], -1, 3)
        gtp = gtp_all[:, BODY_JOINTS_ID]
        row["gt_root_net"] = float((gtp[-1, PELVIS] - gtp[0, PELVIS]).norm())

        # Old comparison: GT positions vs codec-reconstructed FK positions.
        # Useful as a codec agreement probe, but not a C-test.
        n_dec = min(gtp.shape[0], P_dec_rr.shape[0])
        gtp_dec_rr = root_relative(gtp[:n_dec])
        _, om_gt = msd_from_motion(gtp_dec_rr.reshape(n_dec, -1), MSDConfig(W=W_MOTION))
        _, om_dec_fk = msd_from_motion(P_dec_rr[:n_dec].reshape(n_dec, -1), MSDConfig(W=W_MOTION))
        rho_dec = omega_spearman(om_gt, om_dec_fk)
        if rho_dec is not None:
            row["rho_gtpos_vs_decoded_fk"] = rho_dec

        # Corrected C-test: raw GT body features -> FK vs raw GT positions.
        body_raw_t = torch.tensor(body_raw, dtype=torch.float32, device=DEVICE)
        quat_motion = body_features_to_quat_motion(
            body_raw_t,
            motion,
            DEVICE,
            src_fps=RAW_MOTION_FPS,
            tgt_fps=RAW_MOTION_FPS,
        )
        quat = torch.tensor(quat_motion["quat"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        offset = torch.tensor(quat_motion["offset"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        pos_gt_fk = get_fk_model()(quat, offset)[0][:, BODY_JOINTS_ID].cpu()
        n_gt = min(gtp.shape[0], pos_gt_fk.shape[0])
        gtp_rr = root_relative(gtp[:n_gt])
        gt_fk_rr = root_relative(pos_gt_fk[:n_gt])
        _, om_gt_pos = msd_from_motion(gtp_rr.reshape(n_gt, -1), MSDConfig(W=W_MOTION))
        _, om_gt_fk = msd_from_motion(gt_fk_rr.reshape(n_gt, -1), MSDConfig(W=W_MOTION))
        rho_gt_fk = omega_spearman(om_gt_pos, om_gt_fk)
        if rho_gt_fk is not None:
            row["rho_gtpos_vs_gt_fk"] = rho_gt_fk
        scale, rmse_x100, rmse_best = best_scale_rmse(gt_fk_rr, gtp_rr)
        row["gt_fk_scale_best"] = scale
        row["gt_fk_rmse_x100"] = rmse_x100
        row["gt_fk_rmse_best"] = rmse_best

    report.append(row)


# ---- summary ----------------------------------------------------------------
import statistics as st

nets = [r["root_net"] for r in report]
subs = [r["sub_effect"] for r in report]
sub_vels = [r["sub_velocity_effect"] for r in report]
root_ch = [r["root_channel_mean"] for r in report]

print(f"FK decoded-root net displacement: median={st.median(nets):.4f}  max={max(nets):.4f}")
print(f"root_relative() max offset effect: median={st.median(subs):.4f}  max={max(subs):.4f}")
print(f"root_relative() max velocity effect: median={st.median(sub_vels):.4f}  max={max(sub_vels):.4f}")
print(f"raw npys with 'positions' key: {sum(r['has_positions'] for r in report)}/{len(report)}")
print(f"body[:, :3] mean norm: median={st.median(root_ch):.5f}")

if any("rho_gtpos_vs_decoded_fk" in r for r in report):
    rhos = [r["rho_gtpos_vs_decoded_fk"] for r in report if "rho_gtpos_vs_decoded_fk" in r]
    print(f"Omega GT positions vs decoded-token FK (codec path): median={st.median(rhos):.3f}")

if any("rho_gtpos_vs_gt_fk" in r for r in report):
    rhos = [r["rho_gtpos_vs_gt_fk"] for r in report if "rho_gtpos_vs_gt_fk" in r]
    rmses = [r["gt_fk_rmse_x100"] for r in report if "gt_fk_rmse_x100" in r]
    print(f"Corrected C-test Omega GT positions vs GT-feature FK: median={st.median(rhos):.3f}")
    print(f"Corrected C-test RMSE after FK*100 scaling: median={st.median(rmses):.3f}")

print("""
VERDICT GUIDE
A) sub_velocity_effect ~ 0 on non-locomotion clips -> root subtraction is an
   MSD velocity no-op for those clips; keep it as a cheap guard and for walkers.
B) sub_velocity_effect large with root_net large -> root subtraction is
   load-bearing for locomotion; keep it.
C) positions exists everywhere and corrected GT-position vs GT-feature-FK rho is
   high -> raw positions minus pelvis can replace FK in the GT pipeline.
D) positions absent or schema split locally -> reconcile local copy vs release;
   use the FK route for GT speed until the shortcut is proven corpus-wide.
""")

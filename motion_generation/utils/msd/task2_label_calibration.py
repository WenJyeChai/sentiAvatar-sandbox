"""Task 2 label-based FK-speed calibration for MSD.

This script calibrates FK-frame motion-speed thresholds by comparing clips with
default motion labels against clips with annotated action labels, separately for
old retarget-maya, new retarget-maya, and chonglu schemas.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
MG = ROOT / "motion_generation"
DATA = ROOT / "SuSuInterActs" / "SuSuInterActs"
OUT = MG / "utils" / "msd" / "outputs"
SPOT = OUT / "task2_spot_checks"

FPS = 20.0
N_PER_GROUP = 500
ACTION_PREFIX = "\u52a8\u4f5c\uff1a"
NO_ACTION = "\u52a8\u4f5c\uff1a\u65e0\u52a8\u4f5c"
TAG_RE = re.compile(r"\u3010([^\u3011]+)\u3011")


def _setup_imports() -> None:
    sys.path.insert(0, str(MG))


def motion_label_kind(text_map: dict[str, str], name: str) -> str:
    """Return default/annotated for motion-action calibration.

    Untagged and expression-only clips are default for motion, because they do
    not carry an explicit non-idle action label.
    """

    text = text_map.get(name, "") or ""
    tags = TAG_RE.findall(text)
    action_tags = [t.strip() for t in tags if t.strip().startswith(ACTION_PREFIX)]
    if not action_tags:
        return "default"
    if all(t == NO_ACTION for t in action_tags):
        return "default"
    return "annotated"


def motion_schema_info(sample_path_fn, data_root: Path, name: str) -> tuple[str, bool, float]:
    motion = np.load(sample_path_fn(data_root, "motion_data", name, ".npy"), allow_pickle=True).item()
    body = np.asarray(motion["body"])
    has_positions = "positions" in motion
    root_mean = float(np.linalg.norm(body[:, :3], axis=-1).mean()) if body.size else float("nan")
    prefix = name.split("/")[0]

    if prefix == "fbx_to_json_data_susu_chonglu":
        schema = "chonglu"
    elif (
        prefix == "fbx_to_json_data_susu_retarget_maya"
        and not has_positions
        and root_mean < 1.0
    ):
        schema = "old_maya"
    elif (
        prefix == "fbx_to_json_data_susu_retarget_maya"
        and has_positions
        and root_mean > 10.0
    ):
        schema = "new_maya"
    else:
        schema = "outlier"
    return schema, has_positions, root_mean


def concat(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [x for x in arrays if len(x)]
    return np.concatenate(arrays) if arrays else np.empty((0,), dtype=np.float64)


def percentile_dict(arr: np.ndarray, qs: list[int]) -> dict[str, float]:
    if len(arr) == 0:
        return {f"p{q}": float("nan") for q in qs}
    vals = np.percentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, vals)}


def rolling_means(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return np.empty((0,), dtype=np.float64)
    window = max(1, min(int(window), len(arr)))
    cumsum = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def choose_tau_noise_from_still_windows(
    default_clips: list[np.ndarray],
    *,
    fps: float,
    window_seconds: float = 1.0,
    bottom_fraction: float = 0.05,
    frame_percentile: float = 95.0,
) -> dict[str, float | int]:
    """Measure a noise floor from the stillest contiguous default-label windows."""

    window = max(1, int(round(float(fps) * float(window_seconds))))
    window_records: list[tuple[float, int, int, int]] = []
    for clip_idx, speed in enumerate(default_clips):
        if len(speed) == 0:
            continue
        means = rolling_means(speed, window)
        for start, mean_speed in enumerate(means):
            window_records.append((float(mean_speed), clip_idx, start, min(window, len(speed))))

    if not window_records:
        raise ValueError("Cannot compute tau_noise: no default windows were available")

    window_records.sort(key=lambda item: item[0])
    selected_count = max(1, int(np.ceil(len(window_records) * float(bottom_fraction))))
    selected = window_records[:selected_count]

    selected_frames = []
    for _mean_speed, clip_idx, start, width in selected:
        speed = default_clips[clip_idx]
        selected_frames.append(speed[start : start + width])
    selected_frame_values = np.concatenate(selected_frames)

    return {
        "tau": float(np.percentile(selected_frame_values, frame_percentile)),
        "window_frames": int(window),
        "window_seconds": float(window_seconds),
        "bottom_fraction": float(bottom_fraction),
        "frame_percentile": float(frame_percentile),
        "candidate_windows": int(len(window_records)),
        "selected_windows": int(selected_count),
        "selected_frames": int(len(selected_frame_values)),
        "selected_window_mean_p50": float(np.percentile([x[0] for x in selected], 50)),
        "selected_window_mean_p95": float(np.percentile([x[0] for x in selected], 95)),
    }


def choose_separation(
    default_arr: np.ndarray,
    annotated_arr: np.ndarray,
    min_tau: float,
) -> dict[str, float]:
    """Youden-J threshold for active-vs-default, constrained above tau_noise."""

    combined = np.concatenate([default_arr, annotated_arr])
    grid = np.unique(np.percentile(combined, np.linspace(0.5, 99.5, 397)))
    grid = grid[grid >= min_tau]

    best: tuple[float, float, float, float, float] | None = None
    for threshold in grid:
        true_positive = float(np.mean(annotated_arr >= threshold))
        false_positive = float(np.mean(default_arr >= threshold))
        youden_j = true_positive - false_positive
        balanced_accuracy = 0.5 * (true_positive + (1.0 - false_positive))
        record = (
            youden_j,
            balanced_accuracy,
            float(threshold),
            true_positive,
            false_positive,
        )
        if best is None or record > best:
            best = record

    assert best is not None
    return {
        "threshold": best[2],
        "annotated_above": best[3],
        "default_above": best[4],
        "youden_j": best[0],
        "balanced_accuracy": best[1],
    }


def gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    x_axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x_axis**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def analyze_log_bimodality(speed: np.ndarray, *, bins: int = 128) -> dict[str, object]:
    """Find a low/high-speed valley in the unlabeled log-speed histogram, if clear."""

    positive = speed[speed > 0]
    if len(positive) < 100:
        return {"status": "insufficient_frames", "threshold": None, "peak_count": 0}

    log_speed = np.log10(positive)
    lo, hi = np.percentile(log_speed, [0.2, 99.8])
    counts, edges = np.histogram(log_speed, bins=bins, range=(float(lo), float(hi)))
    smooth = np.convolve(counts.astype(np.float64), gaussian_kernel(sigma=2.0, radius=8), mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])

    max_height = float(np.max(smooth))
    min_peak_height = max_height * 0.05
    peaks: list[int] = []
    for idx in range(1, len(smooth) - 1):
        if smooth[idx] > min_peak_height and smooth[idx] >= smooth[idx - 1] and smooth[idx] > smooth[idx + 1]:
            if peaks and idx - peaks[-1] < 6:
                if smooth[idx] > smooth[peaks[-1]]:
                    peaks[-1] = idx
            else:
                peaks.append(idx)

    candidates = []
    for left, right in zip(peaks, peaks[1:]):
        if centers[right] - centers[left] < 0.25:
            continue
        valley_rel = left + int(np.argmin(smooth[left : right + 1]))
        valley_height = float(smooth[valley_rel])
        lower_peak = float(min(smooth[left], smooth[right]))
        ratio = valley_height / lower_peak if lower_peak > 0 else 1.0
        candidates.append(
            {
                "left_peak_log10": float(centers[left]),
                "right_peak_log10": float(centers[right]),
                "valley_log10": float(centers[valley_rel]),
                "threshold": float(10.0 ** centers[valley_rel]),
                "valley_ratio": float(ratio),
                "left_peak_height": float(smooth[left]),
                "right_peak_height": float(smooth[right]),
                "valley_height": valley_height,
            }
        )

    if not candidates:
        return {
            "status": "unimodal_or_no_clear_valley",
            "threshold": None,
            "peak_count": int(len(peaks)),
            "hist_edges": edges.tolist(),
            "hist_counts": counts.tolist(),
            "smooth_counts": smooth.tolist(),
        }

    best = min(candidates, key=lambda item: float(item["valley_ratio"]))
    status = "clear_valley" if float(best["valley_ratio"]) <= 0.70 else "valley_too_shallow"
    return {
        "status": status,
        "threshold": best["threshold"] if status == "clear_valley" else None,
        "peak_count": int(len(peaks)),
        "best_candidate": best,
        "hist_edges": edges.tolist(),
        "hist_counts": counts.tolist(),
        "smooth_counts": smooth.tolist(),
    }


def root_speed_from_body(body: np.ndarray, schema: str) -> np.ndarray:
    """Schema-aware root speed in FK-frame meters/frame from body[:, :3]."""

    root = np.asarray(body[:, :3], dtype=np.float64)
    if len(root) < 2:
        return np.empty((0,), dtype=np.float64)
    if schema == "old_maya":
        # Old maya stores per-frame root displacement in centimeters.
        delta = root[1:]
    else:
        # New maya/chonglu store absolute root position in centimeters.
        delta = np.diff(root, axis=0)
    return np.linalg.norm(delta, axis=-1) / 100.0


def sample_gold_frames(
    clip_rows: list[dict[str, object]],
    schema: str,
    *,
    frames_per_schema: int = 200,
) -> list[dict[str, object]]:
    """Sample frames across speed deciles for human idle/active labeling."""

    frame_rows: list[tuple[float, str, int, str]] = []
    for row in clip_rows:
        if row["schema"] != schema:
            continue
        speed = row["speed"]
        assert isinstance(speed, np.ndarray)
        for idx, value in enumerate(speed):
            frame_rows.append((float(value), str(row["name"]), idx + 1, str(row["label"])))

    if not frame_rows:
        return []

    speeds = np.array([row[0] for row in frame_rows], dtype=np.float64)
    decile_edges = np.percentile(speeds, np.arange(0, 101, 10))
    rng = random.Random(f"gold:{schema}:20260708")
    per_decile = max(1, frames_per_schema // 10)
    selected: list[dict[str, object]] = []

    for decile in range(10):
        lo = decile_edges[decile]
        hi = decile_edges[decile + 1]
        if decile == 9:
            pool = [row for row in frame_rows if lo <= row[0] <= hi]
        else:
            pool = [row for row in frame_rows if lo <= row[0] < hi]
        if len(pool) > per_decile:
            pool = rng.sample(pool, per_decile)
        for speed, name, frame_idx, label in sorted(pool):
            selected.append(
                {
                    "schema": schema,
                    "clip": name,
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx / FPS),
                    "speed": speed,
                    "speed_decile": int(decile + 1),
                    "source_label": label,
                    "human_idle_active": "",
                }
            )
    return selected


def safe_file(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-140:]


def main() -> None:
    _setup_imports()

    from scripts.test_motion_codec_reconstruction import (  # noqa: WPS433
        load_name_list,
        load_text_map,
        sample_path,
    )
    from scripts.train_audio_fim_causal import (  # noqa: WPS433
        body_features_to_quat_motion,
        load_motion_dict,
    )
    from utils.constants import BODY_JOINTS_ID  # noqa: WPS433
    from utils.fk_model import WorldPosFromQuat  # noqa: WPS433

    random.seed(20260708)
    np.random.seed(20260708)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")
    SPOT.mkdir(parents=True, exist_ok=True)

    names = load_name_list(DATA / "split" / "all_file_list.txt")
    text_map = load_text_map(DATA / "text_data" / "motion2text.json")

    print(f"Census over {len(names)} clips...")
    counts: Counter[tuple[str, str]] = Counter()
    records: list[dict[str, object]] = []
    for idx, name in enumerate(names, 1):
        schema, has_positions, root_mean = motion_schema_info(sample_path, DATA, name)
        label = motion_label_kind(text_map, name)
        counts[(schema, label)] += 1
        records.append(
            {
                "name": name,
                "schema": schema,
                "label": label,
                "has_positions": has_positions,
                "root_mean": root_mean,
            }
        )
        if idx % 5000 == 0:
            print(f"  census {idx}/{len(names)}")

    print("Label/schema counts:")
    for (schema, label), count in sorted(counts.items()):
        print(f"  {schema:10s} {label:10s} {count}")

    grouped: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        schema = str(record["schema"])
        if schema in {"old_maya", "new_maya", "chonglu"}:
            grouped[(schema, str(record["label"]))].append(str(record["name"]))

    samples: dict[tuple[str, str], list[str]] = {}
    for schema in ["old_maya", "new_maya", "chonglu"]:
        for label in ["default", "annotated"]:
            pool = sorted(grouped[(schema, label)])
            if len(pool) > N_PER_GROUP:
                rng = random.Random(f"{schema}:{label}:20260708")
                pool = sorted(rng.sample(pool, N_PER_GROUP))
            samples[(schema, label)] = pool
            print(f"Sample {schema:9s} {label:10s}: {len(pool)} clips")

    template = MG / "meta" / "template_susu_retarget_63nodes.bvh"
    fk = WorldPosFromQuat(template_bvh_path=str(template)).to(device).eval()

    @torch.no_grad()
    def fk_speeds(name: str, schema: str) -> tuple[np.ndarray, np.ndarray]:
        motion = load_motion_dict(sample_path(DATA, "motion_data", name, ".npy"))
        body_np = np.asarray(motion["body"], dtype=np.float32)
        body_raw = torch.tensor(body_np, device=device)
        quat_motion = body_features_to_quat_motion(
            body_raw,
            motion,
            device,
            src_fps=FPS,
            tgt_fps=FPS,
        )
        quat = torch.as_tensor(quat_motion["quat"], dtype=torch.float32, device=device).unsqueeze(0)
        offset = torch.as_tensor(quat_motion["offset"], dtype=torch.float32, device=device).unsqueeze(0)
        pos = fk(quat, offset)[0][:, BODY_JOINTS_ID]
        if pos.shape[0] < 2:
            empty = np.empty((0,), dtype=np.float64)
            return empty, empty

        rel = pos - pos[:, 0:1]
        speed = torch.linalg.norm(rel[1:] - rel[:-1], dim=-1).mean(dim=-1)
        return (
            speed.cpu().numpy().astype(np.float64),
            root_speed_from_body(body_np, schema).astype(np.float64),
        )

    all_sample_names = [
        (schema, label, name)
        for (schema, label), pool in samples.items()
        for name in pool
    ]

    frame_speed: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    root_speed: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    clip_rows: list[dict[str, object]] = []

    print(f"Computing FK speeds for {len(all_sample_names)} sampled clips...")
    for idx, (schema, label, name) in enumerate(all_sample_names, 1):
        try:
            speed, root = fk_speeds(name, schema)
            if len(speed) == 0:
                continue
            frame_speed[(schema, label)].append(speed)
            root_speed[(schema, label)].append(root)
            clip_rows.append(
                {
                    "name": name,
                    "schema": schema,
                    "label": label,
                    "speed": speed,
                    "root": root,
                    "mean_speed": float(np.mean(speed)),
                    "p90_speed": float(np.percentile(speed, 90)),
                    "max_speed": float(np.max(speed)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FK error {schema}/{label} {name}: {type(exc).__name__}: {exc}")

        if idx % 50 == 0 or idx == len(all_sample_names):
            print(f"  FK {idx}/{len(all_sample_names)}")

    speed_arrays = {key: concat(value) for key, value in frame_speed.items()}
    root_arrays = {key: concat(value) for key, value in root_speed.items()}

    thresholds: dict[str, dict[str, object]] = {}
    gesture_tiers: dict[str, dict[str, float]] = {}
    bimodality: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    for schema in ["old_maya", "new_maya", "chonglu"]:
        default = speed_arrays[(schema, "default")]
        annotated = speed_arrays[(schema, "annotated")]
        default_clips = frame_speed[(schema, "default")]
        root_default = root_arrays[(schema, "default")]
        root_annotated = root_arrays[(schema, "annotated")]
        all_speed = np.concatenate([default, annotated])
        gesture_tiers[schema] = percentile_dict(all_speed, [75, 90, 95, 99])

        tau_noise_record = choose_tau_noise_from_still_windows(default_clips, fps=FPS)
        tau_noise = float(tau_noise_record["tau"])
        separation = choose_separation(default, annotated, min_tau=tau_noise)
        log_bimodal = analyze_log_bimodality(all_speed)
        bimodality[schema] = log_bimodal
        thr_root = float(np.percentile(root_default, 95))

        label_premise_pass = bool(float(separation["youden_j"]) >= 0.10)
        if schema == "chonglu" and label_premise_pass:
            tau_active: float | None = float(separation["threshold"])
            tau_active_status = "provisional_label_youden"
        elif log_bimodal["status"] == "clear_valley":
            tau_active = float(log_bimodal["threshold"])
            tau_active_status = "unlabeled_log_speed_valley"
        else:
            tau_active = None
            tau_active_status = "blocked_gold_set_required"

        if tau_active is None:
            idle_at_tau_active = None
            active_default = None
            active_annotated = None
            idle_gate_at_tau_active = "not_applicable"
        else:
            idle_at_tau_active = float(np.mean(all_speed <= tau_active))
            active_default = float(np.mean(default >= tau_active))
            active_annotated = float(np.mean(annotated >= tau_active))
            idle_gate_at_tau_active = "pass" if 0.30 <= idle_at_tau_active <= 0.50 else "fail"

        thresholds[schema] = {
            "tau_noise": tau_noise,
            "tau_noise_method": "p95_frames_inside_bottom_5pct_rolling_1s_default_windows",
            "tau_noise_window_frames": int(tau_noise_record["window_frames"]),
            "tau_noise_selected_windows": int(tau_noise_record["selected_windows"]),
            "tau_noise_selected_frames": int(tau_noise_record["selected_frames"]),
            "tau_noise_default_percentile": float(np.mean(default <= tau_noise)),
            "tau_noise_idle_all_diagnostic": float(np.mean(all_speed <= tau_noise)),
            "tau_active": tau_active,
            "tau_active_status": tau_active_status,
            "tau_active_source": tau_active_status,
            "idle_at_tau_active": idle_at_tau_active,
            "idle_gate_at_tau_active": idle_gate_at_tau_active,
            "thr_root": thr_root,
            "root_speed_source": "schema_aware_body_root_channel_old_velocity_new_position_diff",
            "label_youden_threshold": float(separation["threshold"]),
            "label_youden_j": float(separation["youden_j"]),
            "label_balanced_accuracy": float(separation["balanced_accuracy"]),
            "label_premise_pass": label_premise_pass,
            "active_default": active_default,
            "active_annotated": active_annotated,
            "sample_default_clips": len(samples[(schema, "default")]),
            "sample_annotated_clips": len(samples[(schema, "annotated")]),
            "default_frames": int(len(default)),
            "annotated_frames": int(len(annotated)),
            "root_default_p95": thr_root,
            "root_annotated_p95": float(np.percentile(root_annotated, 95)),
        }

        for label, arr in [("default", default), ("annotated", annotated)]:
            row: dict[str, object] = {
                "schema": schema,
                "label": label,
                "frames": int(len(arr)),
            }
            row.update(percentile_dict(arr, [10, 25, 40, 50, 75, 90, 95, 99]))
            summaries.append(row)

    for schema in ["old_maya", "new_maya", "chonglu"]:
        default = speed_arrays[(schema, "default")]
        annotated = speed_arrays[(schema, "annotated")]
        threshold = thresholds[schema]
        cap = float(np.percentile(np.concatenate([default, annotated]), 99.5))
        bins = np.linspace(0.0, cap, 80)

        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
        ax.hist(
            default,
            bins=bins,
            density=True,
            alpha=0.48,
            label="default motion label",
            color="#4C78A8",
        )
        ax.hist(
            annotated,
            bins=bins,
            density=True,
            alpha=0.48,
            label="annotated action label",
            color="#F58518",
        )
        ax.axvline(
            threshold["tau_noise"],
            color="#1B7837",
            linewidth=2,
            label=f"tau_noise={threshold['tau_noise']:.5f}",
        )
        if threshold["tau_active"] is not None:
            ax.axvline(
                threshold["tau_active"],
                color="#B2182B",
                linewidth=2,
                label=f"tau_active={float(threshold['tau_active']):.5f}",
            )
        else:
            ax.text(
                0.98,
                0.78,
                "tau_active blocked:\ngold set required",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#B2182B",
            )
        ax.set_title(f"Task 2 FK frame-speed calibration: {schema}")
        ax.set_xlabel("mean joint FK speed after pelvis subtraction (m/frame)")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(SPOT / f"hist_{schema}.png")
        plt.close(fig)

        log_info = bimodality[schema]
        if "hist_edges" in log_info:
            edges = np.asarray(log_info["hist_edges"], dtype=np.float64)
            hist_counts = np.asarray(log_info["hist_counts"], dtype=np.float64)
            smooth = np.asarray(log_info["smooth_counts"], dtype=np.float64)
            centers = 0.5 * (edges[:-1] + edges[1:])
            fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
            ax.bar(centers, hist_counts, width=float(edges[1] - edges[0]), color="#A6CEE3", alpha=0.55, label="log-speed histogram")
            ax.plot(centers, smooth, color="#1F78B4", linewidth=2, label="smoothed")
            if log_info.get("best_candidate"):
                best = log_info["best_candidate"]
                assert isinstance(best, dict)
                ax.axvline(float(best["left_peak_log10"]), color="#4C78A8", linestyle="--", linewidth=1.2)
                ax.axvline(float(best["right_peak_log10"]), color="#F58518", linestyle="--", linewidth=1.2)
                ax.axvline(float(best["valley_log10"]), color="#B2182B", linewidth=2, label="candidate valley")
            ax.set_title(f"Unlabeled log-speed bimodality check: {schema} ({log_info['status']})")
            ax.set_xlabel("log10 mean joint FK speed")
            ax.set_ylabel("frame count")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.22)
            fig.tight_layout()
            fig.savefig(SPOT / f"log_hist_{schema}.png")
            plt.close(fig)

    for row in clip_rows:
        threshold = thresholds[str(row["schema"])]
        speed = row["speed"]
        assert isinstance(speed, np.ndarray)
        tau_noise = float(threshold["tau_noise"])
        tau_active_obj = threshold["tau_active"]
        if tau_active_obj is None:
            row["ramp_frac"] = 0.0
            row["near_tau_active_frac"] = 0.0
            row["above_active_frac"] = 0.0
            continue
        tau_active = float(tau_active_obj)
        row["ramp_frac"] = float(np.mean((speed > tau_noise) & (speed < tau_active))) if tau_active > tau_noise else 0.0
        row["near_tau_active_frac"] = float(np.mean(np.abs(speed - tau_active) <= max(tau_active * 0.12, 1e-8)))
        row["above_active_frac"] = float(np.mean(speed >= tau_active))

    spot_specs = [
        ("chonglu", "default"),
        ("chonglu", "annotated"),
    ]
    used: set[str] = set()
    spots: list[dict[str, object]] = []
    for schema, label in spot_specs:
        pool = [
            row
            for row in clip_rows
            if row["schema"] == schema and row["label"] == label and row["name"] not in used
        ]
        if not pool:
            continue
        pool.sort(
            key=lambda row: (
                float(row["ramp_frac"]),
                float(row["near_tau_active_frac"]),
                -abs(float(row["above_active_frac"]) - 0.35),
            ),
            reverse=True,
        )
        chosen = pool[0]
        used.add(str(chosen["name"]))
        spots.append(chosen)

    spot_manifest: list[dict[str, object]] = []
    for idx, row in enumerate(spots, 1):
        schema = str(row["schema"])
        label = str(row["label"])
        name = str(row["name"])
        threshold = thresholds[schema]
        speed = row["speed"]
        root = row["root"]
        assert isinstance(speed, np.ndarray)
        assert isinstance(root, np.ndarray)

        x_axis = np.arange(1, len(speed) + 1) / FPS
        tau_noise = float(threshold["tau_noise"])
        if threshold["tau_active"] is None:
            continue
        tau_active = float(threshold["tau_active"])
        thr_root = float(threshold["thr_root"])

        fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
        ax.plot(x_axis, speed, color="#243B53", linewidth=1.2, label="FK joint speed")
        ax.fill_between(
            x_axis,
            0,
            speed,
            where=speed <= tau_noise,
            color="#1B7837",
            alpha=0.20,
            label="idle <= tau_noise",
        )
        ax.fill_between(
            x_axis,
            0,
            speed,
            where=(speed > tau_noise) & (speed < tau_active),
            color="#F2C94C",
            alpha=0.28,
            label="ramp zone",
        )
        ax.fill_between(
            x_axis,
            0,
            speed,
            where=speed >= tau_active,
            color="#B2182B",
            alpha=0.18,
            label="active >= tau_active",
        )
        ax.axhline(tau_noise, color="#1B7837", linewidth=1.8)
        ax.axhline(tau_active, color="#B2182B", linewidth=1.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mean joint speed (m/frame)")
        ax.grid(True, alpha=0.22)

        ax2 = ax.twinx()
        ax2.plot(x_axis, root, color="#7B3294", linewidth=0.9, alpha=0.65, label="global pelvis speed")
        ax2.axhline(thr_root, color="#7B3294", linewidth=1.2, alpha=0.7, linestyle="--")
        ax2.set_ylabel("pelvis speed (m/frame)")

        title = f"{schema} / {label}: {name}"
        if len(title) > 118:
            title = title[:115] + "..."
        ax.set_title(title)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=7)
        fig.tight_layout()

        outfile = SPOT / f"spot_{idx}_{schema}_{label}_{safe_file(name)}.png"
        fig.savefig(outfile)
        plt.close(fig)
        spot_manifest.append(
            {
                "schema": schema,
                "label": label,
                "name": name,
                "file": str(outfile.relative_to(ROOT)),
                "ramp_frac": float(row["ramp_frac"]),
                "above_active_frac": float(row["above_active_frac"]),
            }
        )

    gold_frames: list[dict[str, object]] = []
    for schema in ["old_maya", "new_maya"]:
        if thresholds[schema]["tau_active_status"] == "blocked_gold_set_required":
            gold_frames.extend(sample_gold_frames(clip_rows, schema, frames_per_schema=200))

    gold_manifest_path = OUT / "task2_maya_goldset_frame_manifest.csv"
    if gold_frames:
        with gold_manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "schema",
                    "clip",
                    "frame_idx",
                    "time_s",
                    "speed",
                    "speed_decile",
                    "source_label",
                    "human_idle_active",
                ],
            )
            writer.writeheader()
            writer.writerows(gold_frames)

    result = {
        "config": {
            "fps": FPS,
            "n_per_group": N_PER_GROUP,
            "label_rule": (
                "default = no action tag, expression-only, untagged, or explicit no-action; "
                "annotated = any non-default action tag"
            ),
        },
        "counts": [
            {"schema": schema, "label": label, "clips": count}
            for (schema, label), count in sorted(counts.items())
        ],
        "thresholds": thresholds,
        "bimodality": bimodality,
        "gesture_intensity_tiers": gesture_tiers,
        "speed_summaries": summaries,
        "spot_checks": spot_manifest,
        "gold_set_manifest": str(gold_manifest_path.relative_to(ROOT)) if gold_frames else None,
        "histograms": [
            str((SPOT / f"hist_{schema}.png").relative_to(ROOT))
            for schema in ["old_maya", "new_maya", "chonglu"]
        ],
        "log_histograms": [
            str((SPOT / f"log_hist_{schema}.png").relative_to(ROOT))
            for schema in ["old_maya", "new_maya", "chonglu"]
        ],
    }

    result_path = OUT / "task2_label_calibration_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nTHRESHOLDS_JSON")
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))
    print("\nSPEED_SUMMARY_JSON")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print("\nGESTURE_INTENSITY_TIERS_JSON")
    print(json.dumps(gesture_tiers, indent=2, ensure_ascii=False))
    print("\nSPOT_MANIFEST_JSON")
    print(json.dumps(spot_manifest, indent=2, ensure_ascii=False))
    if gold_frames:
        print(f"\nGold-set frame manifest: {gold_manifest_path.relative_to(ROOT)} ({len(gold_frames)} rows)")
    print(f"\nWrote {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Motion-level evaluation for adaptive Step 1 sparse-anchor plans.

Two deliberately separate protocols are exported:

* ``anchor_substitution`` replaces only selected GT anchor token frames and
  retains every non-anchor GT token.  It isolates anchor-content damage.
* ``step2_infilled`` sends every sparse interval through the frozen Step 2 C2F
  infiller with the selected GT or generated endpoints.  It is the relevant
  end-to-end quality test for adaptive scheduling.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    canonical_data_path,
    load_motion_tokens,
)
from scripts.evaluate_step1_anchor_fid import (  # noqa: E402
    DEFAULT_CODEC_PATHS,
    compute_anchor_fid,
    require_file,
)
from scripts.evaluate_step1_multipart_comparison import (  # noqa: E402
    load_source_config,
)
from scripts.cache_step2_interval_costs import (  # noqa: E402
    checkpoint_fingerprint,
    sha256_file,
)
from scripts.train_audio_mask_multipart import load_sequences  # noqa: E402
from utils.inference_math import configure_strict_inference_math  # noqa: E402
from utils.multipart_motion import (  # noqa: E402
    PART_ORDER,
    canonicalize_body_root,
    load_motion_dict,
    merge_parts_to_legacy_motion,
    motion_path_for_name,
)
from utils.step1_adaptive_evaluation import (  # noqa: E402
    AdaptiveRolloutResult,
    load_adaptive_rollout_cache,
)
from utils.variable_c2f_evaluation import (  # noqa: E402
    EvalWindowRecord,
    InfillModelSpec,
    VariableGapMaskExample,
    audio_feature_for_token_frame,
    clean_output_files,
    decode_multipart_part_batch,
    decoded_feature_metrics,
    infer_c2f_window_records_with_metrics,
    load_audio_motion_transformer,
    load_part_codecs,
    save_evaluator_motion,
)


DEFAULT_STEP2_CONFIG = (
    PROJECT_DIR
    / "motion_generation"
    / "configs"
    / "audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml"
)
DEFAULT_STEP2_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15"
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive_output_dir",
        "--rollout_output_dir",
        dest="adaptive_output_dir",
        type=Path,
        required=True,
        help=(
            "Output from evaluate_step1_adaptive_gap.py or "
            "prepare_step1_fixed_gap_motion_evaluation.py."
        ),
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help=(
            "Repeat to select rollout-cache conditions. Defaults to the DP oracle, "
            "fixed gaps 3/7/15, and every adaptive/fixed7 generated condition."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR
        / "motion_generation"
        / "outputs"
        / "step1_adaptive_motion_evaluation",
    )
    parser.add_argument("--step2_config", type=Path, default=DEFAULT_STEP2_CONFIG)
    parser.add_argument(
        "--step2_checkpoint", type=Path, default=DEFAULT_STEP2_CHECKPOINT
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--step2_batch_size", type=int, default=256)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--diversity_times", type=int, default=300)
    parser.add_argument("--metric_seed", type=int, default=42)
    parser.add_argument(
        "--evaluation_dir", type=Path, default=PROJECT_DIR / "evaluation"
    )
    parser.add_argument(
        "--evaluator_checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "eval_model" / "best_model.pt",
    )
    parser.add_argument(
        "--evaluator_config",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "config" / "train_bert_orig.yaml",
    )
    parser.add_argument(
        "--evaluator_stats_dir",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "stats" / "humanml3d" / "guoh3dfeats",
    )
    parser.add_argument("--no_canonicalize_raw_root", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export_only", action="store_true")
    mode.add_argument("--fid_only", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return dict(value)


def default_conditions(available: Sequence[str]) -> list[str]:
    exact = {
        "step2_dp_oracle_gt_anchors",
        "fixed_gap3_gt_anchors",
        "fixed_gap7_gt_anchors",
        "fixed_gap15_gt_anchors",
    }
    suffixes = (
        "__adaptive_gt_history",
        "__adaptive_generated_history",
        "__fixed_gap7_generated_history",
    )
    return [
        value
        for value in available
        if value in exact or value.endswith(suffixes)
    ]


def load_rollouts(
    *,
    names: Sequence[str],
    conditions: Sequence[str],
    cache_by_condition: Mapping[str, str],
    motion_token_dir: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, AdaptiveRolloutResult]],
]:
    dense_by_name: dict[str, np.ndarray] = {}
    rollout_by_condition: dict[str, dict[str, AdaptiveRolloutResult]] = {
        condition: {} for condition in conditions
    }
    for name in names:
        dense, _ = load_motion_tokens(
            canonical_data_path(motion_token_dir, name, ".json"),
            require_causal=True,
        )
        dense_array = np.asarray(dense, dtype=np.int64)
        dense_by_name[name] = dense_array
        for condition in conditions:
            cache_root = Path(cache_by_condition[condition])
            cache_path = canonical_data_path(cache_root, name, ".json")
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing {condition} rollout cache for {name}: {cache_path}"
                )
            result = load_adaptive_rollout_cache(
                cache_path, dense_motion_tokens=dense_array
            )
            if result.name != name:
                raise ValueError(
                    f"{cache_path}: payload name {result.name!r} != {name!r}"
                )
            rollout_by_condition[condition][name] = result
    return dense_by_name, rollout_by_condition


def build_step2_records(
    sequences: Sequence[Mapping[str, Any]],
    *,
    rollouts: Mapping[str, AdaptiveRolloutResult],
) -> tuple[list[EvalWindowRecord], list[tuple[int, int]]]:
    records: list[EvalWindowRecord] = []
    index: list[tuple[int, int]] = []
    for sequence_idx, item in enumerate(sequences):
        name = str(item["name"])
        result = rollouts[name]
        dense = np.asarray(item["motion_tokens"], dtype=np.int64)
        if len(dense) != result.token_length:
            raise ValueError(
                f"{name}: Step 2 sequence length {len(dense)} != "
                f"rollout length {result.token_length}"
            )
        endpoints = np.concatenate(
            [dense[[0]], np.asarray(result.anchors, dtype=np.int64)], axis=0
        )
        for interval_index, (left, right) in enumerate(
            zip(result.anchor_times[:-1], result.anchor_times[1:])
        ):
            gap = int(right - left - 1)
            if gap == 0:
                continue
            frames = dense[left : right + 1].copy()
            frames[0] = endpoints[interval_index]
            frames[-1] = endpoints[interval_index + 1]
            example = VariableGapMaskExample(
                name=name,
                left_idx=int(left),
                right_idx=int(right),
                gap_frames=gap,
                motion_tokens=frames.tolist(),
                audio_features=torch.stack(
                    [
                        audio_feature_for_token_frame(item, token_index)
                        for token_index in range(left, right + 1)
                    ]
                ),
            )
            records.append(
                EvalWindowRecord(
                    sequence_idx=sequence_idx,
                    name=name,
                    left_idx=int(left),
                    gap_frames=gap,
                    example=example,
                )
            )
            index.append((sequence_idx, interval_index))
    return records, index


def assemble_step2_tokens(
    sequences: Sequence[Mapping[str, Any]],
    *,
    rollouts: Mapping[str, AdaptiveRolloutResult],
    predictions: Sequence[np.ndarray],
    prediction_index: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    predicted_by_interval = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in zip(prediction_index, predictions)
    }
    output_by_name = {}
    for sequence_idx, item in enumerate(sequences):
        name = str(item["name"])
        result = rollouts[name]
        dense = np.asarray(item["motion_tokens"], dtype=np.int64)
        output = np.full_like(dense, -1)
        output[0] = dense[0]
        for interval_index, (left, right) in enumerate(
            zip(result.anchor_times[:-1], result.anchor_times[1:])
        ):
            gap = int(right - left - 1)
            output[left] = dense[0] if interval_index == 0 else result.anchors[
                interval_index - 1
            ]
            if gap:
                middle = predicted_by_interval[(sequence_idx, interval_index)]
                if middle.shape != (gap, dense.shape[1]):
                    raise ValueError(
                        f"{name}: Step 2 returned {middle.shape} for gap={gap}"
                    )
                output[left + 1 : right] = middle
            output[right] = result.anchors[interval_index]
        if np.any(output < 0):
            missing = np.flatnonzero(np.any(output < 0, axis=1))
            raise ValueError(f"{name}: Step 2 assembly left frames {missing[:8]}")
        output_by_name[name] = output
    return output_by_name


def anchor_substitution_tokens(
    dense: np.ndarray,
    result: AdaptiveRolloutResult,
) -> np.ndarray:
    output = np.asarray(dense, dtype=np.int64).copy()
    output[list(result.anchor_times[1:])] = result.anchors
    return output


def token_metrics(
    dense: np.ndarray,
    predicted: np.ndarray,
    result: AdaptiveRolloutResult,
) -> dict[str, Any]:
    missing = np.ones(len(dense), dtype=bool)
    missing[list(result.anchor_times)] = False
    if not bool(missing.any()):
        return {
            "missing_frames": 0,
            "missing_token_accuracy": float("nan"),
            "missing_q0_accuracy": float("nan"),
        }
    correct = predicted[missing] == dense[missing]
    return {
        "missing_frames": int(missing.sum()),
        "missing_token_accuracy": float(correct.mean()),
        "missing_q0_accuracy": float(correct[:, [0, 4, 8, 12]].mean()),
    }


def summarize_step2_c2f_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Token-weighted hard-rollout Step 2 likelihood by endpoint condition."""

    if frame.empty:
        return pd.DataFrame()
    rows = []
    for condition, values in frame.groupby("condition", sort=False):
        token_count = int(values["token_count"].sum())
        row: dict[str, Any] = {
            "condition": str(condition),
            "clips": int(values["name"].nunique()),
            "intervals": len(values),
            "tokens": token_count,
            "cross_entropy": (
                float(values["ce_sum"].sum()) / token_count
                if token_count
                else float("nan")
            ),
            "accuracy": (
                float(values["correct"].sum()) / token_count
                if token_count
                else float("nan")
            ),
        }
        row["perplexity"] = float(np.exp(row["cross_entropy"]))
        for stage in range(4):
            stage_count = int(values[f"q{stage}_token_count"].sum())
            row[f"q{stage}_cross_entropy"] = (
                float(values[f"q{stage}_ce_sum"].sum()) / stage_count
                if stage_count
                else float("nan")
            )
            row[f"q{stage}_accuracy"] = (
                float(values[f"q{stage}_correct"].sum()) / stage_count
                if stage_count
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_motion_root(root: Path, conditions: Sequence[str]) -> None:
    clean_output_files(root / "motions" / "raw_gt", "gt")
    for condition in ("causal_codec_reconstruction", *conditions):
        clean_output_files(root / "motions" / condition, "pred")


def clean_visual_motion_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.npy"):
        old.unlink()


def save_visual_motion(
    path: Path,
    name: str,
    motion: Mapping[str, np.ndarray],
    *,
    frames: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        path,
        {
            "name": str(name),
            **{
                key: np.asarray(motion[key][:frames], dtype=np.float32)
                for key in ("body", "left", "right")
            },
        },
    )


def export_motion_conditions(
    *,
    names: Sequence[str],
    conditions: Sequence[str],
    dense_by_name: Mapping[str, np.ndarray],
    rollout_by_condition: Mapping[
        str, Mapping[str, AdaptiveRolloutResult]
    ],
    step2_tokens_by_condition: Mapping[str, Mapping[str, np.ndarray]],
    codecs,
    motion_dir: Path,
    device: torch.device,
    output_dir: Path,
    canonicalize_raw_root: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_root = output_dir / "anchor_substitution"
    step2_root = output_dir / "step2_infilled"
    visual_root = output_dir / "visual_motion"
    prepare_motion_root(anchor_root, conditions)
    prepare_motion_root(step2_root, conditions)
    visual_dirs = [
        visual_root / "raw_gt",
        visual_root / "causal_codec_reconstruction",
    ]
    visual_dirs.extend(
        visual_root / protocol / condition
        for protocol in ("anchor_substitution", "step2_infilled")
        for condition in conditions
    )
    for directory in visual_dirs:
        clean_visual_motion_dir(directory)
    manifest_rows = []
    metric_rows = []
    for clip_index, name in enumerate(names):
        dense = dense_by_name[name]
        anchor_tokens = {
            condition: anchor_substitution_tokens(
                dense, rollout_by_condition[condition][name]
            )
            for condition in conditions
        }
        step2_tokens = {
            condition: step2_tokens_by_condition[condition][name]
            for condition in conditions
        }
        ordered = [
            dense,
            *(anchor_tokens[value] for value in conditions),
            *(step2_tokens[value] for value in conditions),
        ]
        decoded_parts = decode_multipart_part_batch(
            np.stack(ordered, axis=0),
            codecs,
            device,
            part_order=PART_ORDER,
            clip_invalid=False,
        )
        decoded = [
            merge_parts_to_legacy_motion(
                {
                    part: decoded_parts[part][batch_index]
                    for part in PART_ORDER
                }
            )
            for batch_index in range(len(ordered))
        ]
        decoded_codec = decoded[0]
        anchor_decoded = dict(
            zip(conditions, decoded[1 : 1 + len(conditions)])
        )
        step2_decoded = dict(
            zip(conditions, decoded[1 + len(conditions) :])
        )
        raw_path = motion_path_for_name(motion_dir, name)
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw GT motion for {name}")
        source_motion = load_motion_dict(raw_path)
        raw_body = np.asarray(source_motion["body"], dtype=np.float32)
        if canonicalize_raw_root:
            raw_body, _, _ = canonicalize_body_root(raw_body)
        raw_gt = {
            "body": raw_body,
            "left": np.asarray(source_motion["left"], dtype=np.float32),
            "right": np.asarray(source_motion["right"], dtype=np.float32),
        }
        target_len = min(
            *(len(raw_gt[key]) for key in ("body", "left", "right")),
            *(len(decoded_codec[key]) for key in ("body", "left", "right")),
            *(
                len(value[key])
                for value in anchor_decoded.values()
                for key in ("body", "left", "right")
            ),
            *(
                len(value[key])
                for value in step2_decoded.values()
                for key in ("body", "left", "right")
            ),
        )
        if target_len < 2:
            raise ValueError(f"{name}: fewer than two aligned decoded frames")
        stem = f"{clip_index:06d}"
        save_visual_motion(
            visual_root / "raw_gt" / f"{stem}.npy",
            name,
            raw_gt,
            frames=target_len,
        )
        save_visual_motion(
            visual_root / "causal_codec_reconstruction" / f"{stem}.npy",
            name,
            decoded_codec,
            frames=target_len,
        )
        for root in (anchor_root, step2_root):
            save_evaluator_motion(
                root / "motions" / "raw_gt" / f"{stem}_gt.npy",
                name,
                raw_gt["body"][:target_len],
            )
            save_evaluator_motion(
                root
                / "motions"
                / "causal_codec_reconstruction"
                / f"{stem}_pred.npy",
                name,
                decoded_codec["body"][:target_len],
            )
        for protocol, values, token_values in (
            ("anchor_substitution", anchor_decoded, anchor_tokens),
            ("step2_infilled", step2_decoded, step2_tokens),
        ):
            root = anchor_root if protocol == "anchor_substitution" else step2_root
            for condition, motion in values.items():
                save_evaluator_motion(
                    root
                    / "motions"
                    / condition
                    / f"{stem}_pred.npy",
                    name,
                    motion["body"][:target_len],
                )
                save_visual_motion(
                    visual_root / protocol / condition / f"{stem}.npy",
                    name,
                    motion,
                    frames=target_len,
                )
                row = {
                    "clip_index": clip_index,
                    "name": name,
                    "protocol": protocol,
                    "condition": condition,
                    **decoded_feature_metrics(
                        decoded_codec["body"][:target_len],
                        motion["body"][:target_len],
                        prefix="codec_relative",
                    ),
                    **decoded_feature_metrics(
                        raw_gt["body"][:target_len],
                        motion["body"][:target_len],
                        prefix="raw_gt",
                    ),
                }
                if protocol == "step2_infilled":
                    row.update(
                        token_metrics(
                            dense,
                            token_values[condition],
                            rollout_by_condition[condition][name],
                        )
                    )
                metric_rows.append(row)
        manifest_rows.append(
            {
                "clip_index": clip_index,
                "name": name,
                "token_frames": len(dense),
                "motion_frames": target_len,
                "canonicalize_raw_root": canonicalize_raw_root,
                "visual_motion_stem": stem,
            }
        )
        if (clip_index + 1) % 25 == 0:
            print(f"decoded/exported {clip_index + 1}/{len(names)} clips", flush=True)
    return pd.DataFrame(manifest_rows), pd.DataFrame(metric_rows)


def compute_protocol_fid(
    *,
    root: Path,
    conditions: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    canonicalize_raw_root: bool,
) -> pd.DataFrame:
    return compute_anchor_fid(
        output_dir=root,
        conditions=["causal_codec_reconstruction", *conditions],
        evaluation_dir=project_path(args.evaluation_dir),
        evaluator_checkpoint=require_file(
            args.evaluator_checkpoint, "motion evaluator checkpoint"
        ),
        evaluator_config=require_file(
            args.evaluator_config, "motion evaluator config"
        ),
        evaluator_stats_dir=project_path(args.evaluator_stats_dir),
        device=device,
        batch_size=args.fid_batch_size,
        diversity_times=args.diversity_times,
        metric_seed=args.metric_seed,
        canonicalize_raw_root=canonicalize_raw_root,
    )


def main() -> None:
    args = parse_args()
    if args.step2_batch_size < 1 or args.fid_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    adaptive_output = project_path(args.adaptive_output_dir)
    report_path = adaptive_output / "adaptive_evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing adaptive report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache_by_condition = {
        str(key): str(value)
        for key, value in report["rollout_caches"].items()
    }
    conditions = (
        [str(value) for value in args.condition]
        if args.condition
        else default_conditions(list(cache_by_condition))
    )
    if not conditions:
        raise ValueError("No motion-evaluation conditions were selected")
    missing = sorted(set(conditions).difference(cache_by_condition))
    if missing:
        raise KeyError(f"Adaptive report has no rollout caches for {missing}")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    math_mode = configure_strict_inference_math(device)
    canonicalize_raw_root = not args.no_canonicalize_raw_root
    names = json.loads(
        (adaptive_output / "selected_names.json").read_text(encoding="utf-8")
    )
    names = [str(value) for value in names]
    first_checkpoint = project_path(
        next(iter(report["checkpoints"].values()))
    )
    source_config = load_source_config(first_checkpoint)
    source_paths = section(source_config, "paths")
    data_dir = project_path(source_paths["data_dir"])
    motion_token_dir = project_path(source_paths["motion_token_dir"])

    contract_path = output_dir / "adaptive_motion_contract.json"
    if not args.fid_only:
        dense_by_name, rollout_by_condition = load_rollouts(
            names=names,
            conditions=conditions,
            cache_by_condition=cache_by_condition,
            motion_token_dir=motion_token_dir,
        )
        step2_config_path = project_path(args.step2_config)
        step2_config = load_yaml(step2_config_path)
        step2_data = section(step2_config, "data")
        step2_audio = section(step2_config, "audio_conditioning")
        audio_feat_dir = project_path(step2_data["audio_feat_dir"])
        sequences, load_stats = load_sequences(
            names,
            motion_token_dir,
            audio_feat_dir,
            codebook_size=512,
            num_tokens_per_frame=16,
            audio_fps=float(step2_audio["audio_fps"]),
            source_motion_fps_fallback=20.0,
            motion_token_fps_override=10.0,
            motion_token_unit_length_override=2.0,
        )
        if int(load_stats["loaded"]) != len(names):
            raise ValueError(
                f"Loaded {load_stats['loaded']}/{len(names)} Step 2 sequences: "
                f"{load_stats}"
            )
        if [str(item["name"]) for item in sequences] != names:
            raise ValueError("Step 2 sequence order differs from selected names")

        step2_checkpoint = project_path(args.step2_checkpoint)
        manifest_fingerprints = {
            str(value["checkpoint_fingerprint"])
            for value in report["protocol"].get(
                "frozen_step2_cost_manifests", []
            )
            if value.get("checkpoint_fingerprint")
        }
        if len(manifest_fingerprints) > 1:
            raise ValueError(
                "Adaptive schedule costs came from mixed Step 2 checkpoints"
            )
        selected_fingerprint = checkpoint_fingerprint(step2_checkpoint)
        if manifest_fingerprints:
            expected_fingerprint = next(iter(manifest_fingerprints))
            if selected_fingerprint != expected_fingerprint:
                raise ValueError(
                    "Motion evaluation Step 2 weights do not match the frozen "
                    "checkpoint that produced the adaptive training costs: "
                    f"selected={selected_fingerprint}, expected={expected_fingerprint}"
                )
        model = load_audio_motion_transformer(step2_checkpoint, device)
        spec = InfillModelSpec(
            name="frozen_step2_reference",
            checkpoint=step2_checkpoint,
            decoder="c2f",
            allowed_gaps=tuple(range(1, 16)),
        )
        step2_tokens_by_condition: dict[str, dict[str, np.ndarray]] = {}
        step2_metric_rows: list[dict[str, Any]] = []
        for condition in conditions:
            print(f"\n=== frozen Step 2: {condition} ===", flush=True)
            window_records, prediction_index = build_step2_records(
                sequences,
                rollouts=rollout_by_condition[condition],
            )
            predictions, interval_metrics = infer_c2f_window_records_with_metrics(
                model,
                spec,
                window_records,
                batch_size=args.step2_batch_size,
                device=device,
            )
            step2_metric_rows.extend(
                {"condition": condition, **value}
                for value in interval_metrics
            )
            step2_tokens_by_condition[condition] = assemble_step2_tokens(
                sequences,
                rollouts=rollout_by_condition[condition],
                predictions=predictions,
                prediction_index=prediction_index,
            )
        step2_metrics = pd.DataFrame(step2_metric_rows)
        step2_summary = summarize_step2_c2f_metrics(step2_metrics)
        step2_metrics.to_csv(
            output_dir / "step2_c2f_metrics_per_interval.csv",
            index=False,
        )
        step2_summary.to_csv(
            output_dir / "step2_c2f_summary.csv",
            index=False,
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        codec_paths = {
            part: require_file(
                project_path(step2_data.get(f"{part}_ckpt", DEFAULT_CODEC_PATHS[part])),
                f"{part} causal codec",
            )
            for part in PART_ORDER
        }
        codecs = load_part_codecs(codec_paths, device, part_order=PART_ORDER)
        manifest, decoded_metrics = export_motion_conditions(
            names=names,
            conditions=conditions,
            dense_by_name=dense_by_name,
            rollout_by_condition=rollout_by_condition,
            step2_tokens_by_condition=step2_tokens_by_condition,
            codecs=codecs,
            motion_dir=data_dir / "motion_data",
            device=device,
            output_dir=output_dir,
            canonicalize_raw_root=canonicalize_raw_root,
        )
        manifest.to_csv(output_dir / "motion_manifest.csv", index=False)
        decoded_metrics.to_csv(
            output_dir / "decoded_metrics_per_clip.csv", index=False
        )
        decoded_summary = (
            decoded_metrics.groupby(["protocol", "condition"], as_index=False)
            .mean(numeric_only=True)
            .drop(columns=["clip_index"], errors="ignore")
        )
        decoded_summary.to_csv(
            output_dir / "decoded_metrics_summary.csv", index=False
        )
        contract = {
            "schema": "sentiavatar.step1_adaptive_motion_eval.v1",
            "adaptive_output_dir": str(adaptive_output),
            "conditions": conditions,
            "selected_clips": len(names),
            "step2_config": str(step2_config_path),
            "step2_checkpoint": str(step2_checkpoint),
            "step2_checkpoint_fingerprint": selected_fingerprint,
            "audio_feature_dir": str(audio_feat_dir),
            "causal_codecs": {
                key: str(value) for key, value in codec_paths.items()
            },
            "causal_codec_fingerprints": {
                key: sha256_file(value) for key, value in codec_paths.items()
            },
            "canonicalize_raw_root": canonicalize_raw_root,
            "math_mode": math_mode,
            "visual_motion_root": str(
                (output_dir / "visual_motion").resolve()
            ),
            "protocols": {
                "anchor_substitution": (
                    "GT non-anchor tokens retained; isolates sparse-anchor damage."
                ),
                "step2_infilled": (
                    "Every non-adjacent sparse interval generated by the frozen "
                    "Step 2 reference using the selected endpoints."
                ),
                "step2_c2f_likelihood": (
                    "Canonical missing-token likelihood with the model's own "
                    "hard q0-to-q3 prefixes."
                ),
            },
        }
        contract_path.write_text(
            json.dumps(contract, indent=2), encoding="utf-8"
        )
        del codecs, step2_tokens_by_condition
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"FID-only mode requires an export contract: {contract_path}"
            )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        conditions = [str(value) for value in contract["conditions"]]
        canonicalize_raw_root = bool(contract["canonicalize_raw_root"])
        step2_summary_path = output_dir / "step2_c2f_summary.csv"
        step2_summary = (
            pd.read_csv(step2_summary_path)
            if step2_summary_path.is_file()
            else pd.DataFrame()
        )

    if args.export_only:
        if not step2_summary.empty:
            print("\nFrozen Step 2 C2F likelihood")
            print(step2_summary.to_string(index=False))
        print(f"Motion export complete: {output_dir}")
        return
    for stat_name in ("mean.pt", "std.pt"):
        require_file(
            project_path(args.evaluator_stats_dir) / stat_name,
            f"evaluator {stat_name}",
        )
    fid_frames = []
    for protocol in ("anchor_substitution", "step2_infilled"):
        frame = compute_protocol_fid(
            root=output_dir / protocol,
            conditions=conditions,
            args=args,
            device=device,
            canonicalize_raw_root=canonicalize_raw_root,
        )
        frame.insert(0, "protocol", protocol)
        fid_frames.append(frame)
    fid = pd.concat(fid_frames, ignore_index=True)
    fid.to_csv(output_dir / "adaptive_motion_fid.csv", index=False)
    report_payload = {
        "protocol": {
            "selected_clips": len(names),
            "canonicalize_raw_root": canonicalize_raw_root,
            "metric_seed": args.metric_seed,
            "warning": (
                "Anchor-substitution and frozen-Step-2 infilling FID answer "
                "different questions and must not be merged."
            ),
        },
        "conditions": conditions,
        "step2_c2f": json.loads(
            step2_summary.to_json(orient="records")
        ),
        "fid": json.loads(fid.to_json(orient="records")),
    }
    (output_dir / "adaptive_motion_report.json").write_text(
        json.dumps(report_payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if not step2_summary.empty:
        print("\nFrozen Step 2 C2F likelihood")
        print(step2_summary.to_string(index=False))
    print("\nAdaptive motion FID")
    print(fid.to_string(index=False))
    print(f"\nWrote adaptive motion outputs: {output_dir}")


if __name__ == "__main__":
    main()

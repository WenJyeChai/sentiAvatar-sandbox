#!/usr/bin/env python3
"""Evaluate Step 1 anchors with oracle-gap decoded-motion FID.

The planner is rolled out causally. Its predicted anchors replace the matching
GT anchors in the complete causal-codec token sequence, while every non-anchor
token remains GT. The resulting dense tokens are decoded by the causal body
codecs and evaluated against both raw motion and the causal-codec
reconstruction. This measures anchor damage; it is not Step 2 evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    canonical_data_path,
    load_motion_tokens,
    read_split_names,
)
from scripts.evaluate_step1_multipart_comparison import (  # noqa: E402
    deterministic_subset,
    load_planner,
    load_source_config,
    make_dataset_and_loader,
)
from scripts.export_multipart_motion_tokens import (  # noqa: E402
    configure_strict_inference_math,
)
from utils.multipart_motion import PART_ORDER  # noqa: E402
from utils.step1_planner_evaluation import (  # noqa: E402
    RolloutResult,
    build_anchor_substitution_token_variants,
    evaluate_rollouts,
    greedy_rollout_batch,
)
from utils.variable_c2f_evaluation import (  # noqa: E402
    clean_output_files,
    decode_multipart_token_batch,
    decoded_feature_metrics,
    encode_evaluator_latents,
    load_evaluator_motion_encoder,
    load_official_evaluator_helpers,
    load_part_codecs,
    load_raw_gt_body,
    save_evaluator_motion,
)


DEFAULT_CHECKPOINTS = (
    (
        "self_forcing_best_rollout",
        "checkpoints/step1_multipart_fixed_gap3_self_forcing_q0q3_full/best_rollout",
    ),
    ("q0_6k_final", "checkpoints/step1_multipart_fixed_gap3_main6000/final"),
)
DEFAULT_CODEC_PATHS = {
    part: (
        f"checkpoints/causal_multipart_rvqvae/"
        f"causal_rvq_{part}_512x4_scratch/model/best.pth"
    )
    for part in PART_ORDER
}
RESERVED_CONDITIONS = {
    "raw_gt",
    "causal_codec_reconstruction",
    "oracle_previous_gt_anchor_copy",
    "seed_hold",
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_labeled_checkpoints(values: Sequence[str] | None) -> dict[str, Path]:
    pairs: Sequence[tuple[str, str]]
    if values is None:
        pairs = DEFAULT_CHECKPOINTS
    else:
        parsed: list[tuple[str, str]] = []
        for value in values:
            if "=" not in value:
                raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
            label, raw_path = (part.strip() for part in value.split("=", 1))
            if not label or not raw_path:
                raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
            parsed.append((label, raw_path))
        pairs = parsed
    result: dict[str, Path] = {}
    for label, raw_path in pairs:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(f"Unsafe checkpoint label {label!r}")
        if label in RESERVED_CONDITIONS:
            raise ValueError(f"Checkpoint label {label!r} is reserved")
        if label in result:
            raise ValueError(f"Duplicate checkpoint label {label!r}")
        result[label] = project_path(raw_path)
    if not result:
        raise ValueError("At least one Step 1 checkpoint is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle-gap anchor-substitution FID for the causal Step 1 planner"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeat to override the default Step 1 checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR / "motion_generation" / "outputs" / "step1_anchor_fid",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max_clips",
        type=int,
        default=128,
        help="Deterministic validation subset; 0 uses all validation clips.",
    )
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--diversity_times", type=int, default=300)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument(
        "--no_canonicalize_raw_root",
        action="store_true",
        help=(
            "Use the historical non-canonicalized raw-root protocol. The default "
            "canonicalizes raw roots to the causal codec's delta-root contract."
        ),
    )
    for part in PART_ORDER:
        parser.add_argument(
            f"--{part}_ckpt",
            type=Path,
            default=project_path(DEFAULT_CODEC_PATHS[part]),
        )
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export_only", action="store_true")
    mode.add_argument("--fid_only", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def generate_rollouts(
    checkpoints: Mapping[str, Path],
    *,
    names: Sequence[str],
    device: torch.device,
    use_bf16: bool,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, list[RolloutResult]], dict[str, Any], Path, int, pd.DataFrame]:
    rollout_by_label: dict[str, list[RolloutResult]] = {}
    source_configs: dict[str, dict[str, Any]] = {}
    reference_token_dir: Path | None = None
    reference_split: list[str] | None = None
    fixed_gap: int | None = None
    summary_rows: list[dict[str, Any]] = []

    for label, checkpoint in checkpoints.items():
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing {label} checkpoint: {checkpoint}")
        config = load_source_config(checkpoint)
        source_configs[label] = config
        _, loader, paths, data_config, _ = make_dataset_and_loader(
            checkpoint=checkpoint,
            source_config=config,
            names=names,
            batch_size=batch_size,
            workers=num_workers,
            preserve_times=True,
        )
        candidate_split = read_split_names(paths["eval_split"])
        candidate_gap = int(data_config.get("fixed_gap", 3))
        candidate_token_dir = paths["motion_token_dir"].resolve()
        if reference_split is None:
            reference_split = candidate_split
            reference_token_dir = candidate_token_dir
            fixed_gap = candidate_gap
        else:
            if candidate_split != reference_split:
                raise ValueError(f"{label} uses a different validation split")
            if candidate_token_dir != reference_token_dir:
                raise ValueError(f"{label} uses a different causal token export")
            if candidate_gap != fixed_gap:
                raise ValueError(f"{label} uses fixed_gap={candidate_gap}, expected {fixed_gap}")

        print(f"\n=== rollout {label}: {checkpoint} ===", flush=True)
        model = load_planner(
            checkpoint,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            device=device,
        )
        results: list[RolloutResult] = []
        for batch_index, batch in enumerate(loader, start=1):
            results.extend(
                greedy_rollout_batch(
                    model,
                    batch,
                    device=device,
                    use_bf16=use_bf16,
                )
            )
            if batch_index % 10 == 0:
                print(f"{label}: {len(results)}/{len(names)} clips", flush=True)
        if [result.name for result in results] != list(names):
            raise AssertionError(f"{label} rollout order differs from the selected split")
        measured = evaluate_rollouts(results)["summary"]
        summary_rows.append(
            {"checkpoint": label, "checkpoint_path": str(checkpoint), **measured}
        )
        rollout_by_label[label] = results
        del model, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert reference_token_dir is not None and fixed_gap is not None
    first_config = source_configs[next(iter(checkpoints))]
    return (
        rollout_by_label,
        first_config,
        reference_token_dir,
        fixed_gap,
        pd.DataFrame(summary_rows),
    )


def clean_motion_outputs(motion_root: Path, conditions: Sequence[str]) -> None:
    clean_output_files(motion_root / "raw_gt", "gt")
    for condition in conditions:
        clean_output_files(motion_root / condition, "pred")


def export_anchor_substitution_motions(
    *,
    names: Sequence[str],
    rollout_by_label: Mapping[str, Sequence[RolloutResult]],
    motion_token_dir: Path,
    motion_dir: Path,
    codecs,
    device: torch.device,
    output_dir: Path,
    canonicalize_raw_root: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    model_conditions = {
        label: f"{label}_rollout_anchor_substitution" for label in rollout_by_label
    }
    conditions = [
        "causal_codec_reconstruction",
        "oracle_previous_gt_anchor_copy",
        "seed_hold",
        *model_conditions.values(),
    ]
    motion_root = output_dir / "motions"
    clean_motion_outputs(motion_root, conditions)
    results_by_label = {
        label: {result.name: result for result in results}
        for label, results in rollout_by_label.items()
    }
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for clip_index, name in enumerate(names):
        dense_path = canonical_data_path(motion_token_dir, name, ".json")
        dense_tokens, _ = load_motion_tokens(dense_path, require_causal=True)
        first_label = next(iter(rollout_by_label))
        reference_result = results_by_label[first_label][name]
        base_variants = build_anchor_substitution_token_variants(
            dense_tokens=dense_tokens,
            anchor_times=reference_result.anchor_times,
            predicted_anchors=reference_result.predicted_anchors,
            target_anchors=reference_result.target_anchors,
        )
        tokens_by_condition: dict[str, np.ndarray] = {
            "causal_codec_reconstruction": base_variants[
                "causal_codec_reconstruction"
            ],
            "oracle_previous_gt_anchor_copy": base_variants[
                "previous_gt_anchor_copy"
            ],
            "seed_hold": base_variants["seed_hold"],
        }
        for label, condition in model_conditions.items():
            result = results_by_label[label][name]
            if result.anchor_times != reference_result.anchor_times:
                raise ValueError(f"{name}: {label} uses different anchor times")
            variants = build_anchor_substitution_token_variants(
                dense_tokens=dense_tokens,
                anchor_times=result.anchor_times,
                predicted_anchors=result.predicted_anchors,
                target_anchors=result.target_anchors,
            )
            tokens_by_condition[condition] = variants["rollout_anchor_substitution"]

        ordered_conditions = list(tokens_by_condition)
        token_batch = np.stack(
            [tokens_by_condition[condition] for condition in ordered_conditions], axis=0
        )
        decoded_batch = decode_multipart_token_batch(
            token_batch,
            codecs,
            device,
            part_order=PART_ORDER,
            clip_invalid=False,
        )
        decoded_by_condition = dict(zip(ordered_conditions, decoded_batch))
        decoded_reference = decoded_by_condition["causal_codec_reconstruction"]
        raw_gt = load_raw_gt_body(
            motion_dir,
            name,
            len(decoded_reference),
            canonicalize_root=canonicalize_raw_root,
        )
        if raw_gt is None or not len(raw_gt):
            raise FileNotFoundError(f"Missing raw GT motion for {name} under {motion_dir}")
        target_len = min(
            len(raw_gt), *(len(value) for value in decoded_by_condition.values())
        )
        if target_len < 2:
            raise ValueError(f"{name}: fewer than two aligned decoded frames")
        stem = f"{clip_index:06d}"
        save_evaluator_motion(
            motion_root / "raw_gt" / f"{stem}_gt.npy",
            name,
            raw_gt[:target_len],
        )
        for condition, decoded in decoded_by_condition.items():
            prediction = decoded[:target_len]
            save_evaluator_motion(
                motion_root / condition / f"{stem}_pred.npy",
                name,
                prediction,
            )
            metric_rows.append(
                {
                    "clip_index": clip_index,
                    "name": name,
                    "condition": condition,
                    **decoded_feature_metrics(
                        decoded_reference[:target_len],
                        prediction,
                        prefix="codec_relative",
                    ),
                    **decoded_feature_metrics(
                        raw_gt[:target_len],
                        prediction,
                        prefix="raw_gt",
                    ),
                }
            )
        manifest_rows.append(
            {
                "clip_index": clip_index,
                "name": name,
                "token_frames": len(dense_tokens),
                "motion_frames": target_len,
                "anchor_count_including_seed": len(reference_result.anchor_times),
                "predicted_anchor_count": len(reference_result.predicted_anchors),
                "predicted_anchor_fraction": (
                    len(reference_result.predicted_anchors) / len(dense_tokens)
                ),
                "canonicalize_raw_root": canonicalize_raw_root,
            }
        )
        if (clip_index + 1) % 25 == 0:
            print(f"decoded/exported {clip_index + 1}/{len(names)} clips", flush=True)

    return pd.DataFrame(manifest_rows), pd.DataFrame(metric_rows), conditions


def assert_same_names(
    reference_names: Sequence[str], candidate_names: Sequence[str], label: str
) -> None:
    if list(reference_names) != list(candidate_names):
        raise ValueError(f"{label} evaluator file order/names differ from the reference")


def compute_anchor_fid(
    *,
    output_dir: Path,
    conditions: Sequence[str],
    evaluation_dir: Path,
    evaluator_checkpoint: Path,
    evaluator_config: Path,
    evaluator_stats_dir: Path,
    device: torch.device,
    batch_size: int,
    diversity_times: int,
    canonicalize_raw_root: bool,
) -> pd.DataFrame:
    helpers = load_official_evaluator_helpers(evaluation_dir)
    cfg, encoder = load_evaluator_motion_encoder(
        evaluator_checkpoint, evaluator_config, device
    )
    motion_root = output_dir / "motions"
    raw_names, raw_latents = encode_evaluator_latents(
        motion_root / "raw_gt",
        "gt",
        cfg,
        encoder,
        evaluator_stats_dir,
        device,
        batch_size=batch_size,
    )
    names_by_condition: dict[str, list[str]] = {}
    latents_by_condition: dict[str, np.ndarray] = {}
    for condition in conditions:
        condition_names, condition_latents = encode_evaluator_latents(
            motion_root / condition,
            "pred",
            cfg,
            encoder,
            evaluator_stats_dir,
            device,
            batch_size=batch_size,
        )
        assert_same_names(raw_names, condition_names, condition)
        names_by_condition[condition] = condition_names
        latents_by_condition[condition] = condition_latents

    del names_by_condition
    references = {
        (
            "raw_motion_canonicalized"
            if canonicalize_raw_root
            else "raw_motion_historical_root"
        ): raw_latents,
        "causal_codec_reconstruction": latents_by_condition[
            "causal_codec_reconstruction"
        ],
    }
    rows: list[dict[str, Any]] = []
    metric_fn = helpers["compute_fid_diversity_metrics"]
    for reference, reference_latents in references.items():
        for condition in conditions:
            metrics = metric_fn(
                reference_latents,
                latents_by_condition[condition],
                diversity_times=diversity_times,
            )
            rows.append(
                {
                    "reference": reference,
                    "condition": condition,
                    "num_clips": len(reference_latents),
                    **metrics,
                    "diversity_gap": metrics["Diversity_Gen"]
                    - metrics["Diversity_GT"],
                }
            )
    frame = pd.DataFrame(rows)
    for reference in frame["reference"].unique():
        mask = frame["reference"].eq(reference)
        floor = float(
            frame.loc[
                mask & frame["condition"].eq("causal_codec_reconstruction"),
                "FID_norm_by_GT",
            ].iloc[0]
        )
        frame.loc[mask, "delta_fid_norm_vs_codec_floor"] = (
            frame.loc[mask, "FID_norm_by_GT"] - floor
        )
    return frame


def load_export_contract(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "anchor_fid_contract.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"FID-only mode requires a completed export contract: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("conditions"):
        raise ValueError(f"Malformed export contract: {path}")
    return payload


def main() -> None:
    args = parse_args()
    if args.max_clips < 0:
        raise ValueError("max_clips must be non-negative")
    if args.rollout_batch_size < 1 or args.fid_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = bool(not args.no_bf16 and device.type == "cuda")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a CUDA device without bf16 support")
    math_mode = configure_strict_inference_math(device)
    canonicalize_raw_root = not args.no_canonicalize_raw_root

    conditions: list[str]
    checkpoint_contract: dict[str, str] = {}
    fixed_gap: int | None = None
    selected_names: list[str] = []
    if not args.fid_only:
        checkpoints = parse_labeled_checkpoints(args.checkpoint)
        first_config = load_source_config(next(iter(checkpoints.values())))
        from scripts.train_step1_multipart_fixed_gap3 import resolve_data_paths

        first_paths = resolve_data_paths(first_config)
        validation_names = read_split_names(first_paths["eval_split"])
        selected_names = deterministic_subset(
            validation_names, args.max_clips, args.subset_seed
        )
        rollout_by_label, _, motion_token_dir, fixed_gap, rollout_summary = (
            generate_rollouts(
                checkpoints,
                names=selected_names,
                device=device,
                use_bf16=use_bf16,
                batch_size=args.rollout_batch_size,
                num_workers=args.num_workers,
            )
        )
        codec_paths = {
            part: require_file(getattr(args, f"{part}_ckpt"), f"{part} causal codec")
            for part in PART_ORDER
        }
        codecs = load_part_codecs(codec_paths, device, part_order=PART_ORDER)
        noncausal = [part for part, loaded in codecs.items() if not loaded.causal]
        if noncausal:
            raise ValueError(f"Anchor FID requires causal body codecs, got {noncausal}")
        if any(loaded.codebook_size != 512 for loaded in codecs.values()):
            raise ValueError("Every causal body codec must use a 512-entry codebook")
        if any(loaded.num_quantizers != 4 for loaded in codecs.values()):
            raise ValueError("Every causal body codec must use four RVQ levels")

        manifest, decoded_metrics, conditions = export_anchor_substitution_motions(
            names=selected_names,
            rollout_by_label=rollout_by_label,
            motion_token_dir=motion_token_dir,
            motion_dir=first_paths["data_dir"] / "motion_data",
            codecs=codecs,
            device=device,
            output_dir=output_dir,
            canonicalize_raw_root=canonicalize_raw_root,
        )
        manifest.to_csv(output_dir / "anchor_fid_manifest.csv", index=False)
        rollout_summary.to_csv(output_dir / "rollout_token_summary.csv", index=False)
        decoded_metrics.to_csv(output_dir / "decoded_metrics_per_clip.csv", index=False)
        decoded_summary = (
            decoded_metrics.groupby("condition", as_index=False)
            .mean(numeric_only=True)
            .drop(columns=["clip_index"], errors="ignore")
        )
        decoded_summary.to_csv(output_dir / "decoded_metrics_summary.csv", index=False)
        checkpoint_contract = {
            label: str(path) for label, path in checkpoints.items()
        }
        contract = {
            "protocol": "oracle_gap_anchor_substitution",
            "interpretation": (
                "GT non-anchor causal tokens are retained; this evaluates Step 1 "
                "anchor damage and is not Step 2 infilling FID."
            ),
            "conditions": conditions,
            "checkpoints": checkpoint_contract,
            "causal_codecs": {part: str(path) for part, path in codec_paths.items()},
            "validation_split": str(first_paths["eval_split"]),
            "selected_clips": len(selected_names),
            "subset_seed": args.subset_seed,
            "fixed_gap": fixed_gap,
            "canonicalize_raw_root": canonicalize_raw_root,
            "math_mode": math_mode,
        }
        (output_dir / "anchor_fid_contract.json").write_text(
            json.dumps(contract, indent=2), encoding="utf-8"
        )
        del codecs, rollout_by_label
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        contract = load_export_contract(output_dir)
        conditions = [str(value) for value in contract["conditions"]]
        checkpoint_contract = {
            str(key): str(value)
            for key, value in contract.get("checkpoints", {}).items()
        }
        fixed_gap = int(contract.get("fixed_gap", 3))
        canonicalize_raw_root = bool(contract.get("canonicalize_raw_root", True))
        selected_names = [
            str(value)
            for value in pd.read_csv(output_dir / "anchor_fid_manifest.csv")["name"]
        ]

    if args.export_only:
        print(f"Export complete: {output_dir}")
        return

    evaluation_dir = project_path(args.evaluation_dir)
    evaluator_checkpoint = require_file(
        args.evaluator_checkpoint, "motion evaluator checkpoint"
    )
    evaluator_config = require_file(args.evaluator_config, "motion evaluator config")
    evaluator_stats_dir = project_path(args.evaluator_stats_dir)
    for stat_name in ("mean.pt", "std.pt"):
        require_file(evaluator_stats_dir / stat_name, f"evaluator {stat_name}")
    fid = compute_anchor_fid(
        output_dir=output_dir,
        conditions=conditions,
        evaluation_dir=evaluation_dir,
        evaluator_checkpoint=evaluator_checkpoint,
        evaluator_config=evaluator_config,
        evaluator_stats_dir=evaluator_stats_dir,
        device=device,
        batch_size=args.fid_batch_size,
        diversity_times=args.diversity_times,
        canonicalize_raw_root=canonicalize_raw_root,
    )
    fid.to_csv(output_dir / "anchor_fid_metrics.csv", index=False)
    report = {
        "protocol": {
            "name": "oracle_gap_anchor_substitution_fid",
            "selected_clips": len(selected_names),
            "fixed_gap": fixed_gap,
            "canonicalize_raw_root": canonicalize_raw_root,
            "codec_relative_reference": "causal_codec_reconstruction",
            "warning": (
                "Non-anchor tokens are GT. These scores isolate anchor damage and "
                "must not be reported as end-to-end Step 1 plus Step 2 FID."
            ),
        },
        "checkpoints": checkpoint_contract,
        "fid": records(fid),
    }
    (output_dir / "anchor_fid_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    print("\nAnchor-substitution FID")
    print(fid.to_string(index=False))
    print(f"\nWrote anchor FID outputs: {output_dir}")


if __name__ == "__main__":
    main()

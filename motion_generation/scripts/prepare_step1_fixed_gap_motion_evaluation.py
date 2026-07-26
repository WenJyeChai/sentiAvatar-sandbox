#!/usr/bin/env python3
"""Convert fixed-gap Step 1 rollout caches into the motion-evaluation contract.

``evaluate_step1_multipart_comparison.py --write_rollout_cache`` predates the
adaptive sparse-plan cache schema consumed by the frozen-Step-2 motion
evaluator.  This adapter performs a strict, lossless conversion without
regenerating anchors.  It also materializes a matched GT-anchor fixed-schedule
control and fingerprints the exact Step 2 checkpoint used during training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    canonical_data_path,
    load_motion_tokens,
    read_split_names,
)
from scripts.cache_step2_interval_costs import checkpoint_fingerprint  # noqa: E402
from scripts.evaluate_step1_multipart_comparison import (  # noqa: E402
    deterministic_subset,
    load_source_config,
)
from scripts.train_step1_multipart_fixed_gap3 import (  # noqa: E402
    data_config_from_config,
    resolve_data_paths,
    section,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_SLOT_COUNT,
    fixed_anchor_times,
    gap_from_anchor_times,
)


SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison_output_dir",
        type=Path,
        required=True,
        help=(
            "Output from evaluate_step1_multipart_comparison.py run with "
            "--write_rollout_cache."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "motion_generation"
            / "outputs"
            / "step1_fixed_gap_motion_input"
        ),
    )
    parser.add_argument(
        "--checkpoint_label",
        action="append",
        default=None,
        help="Optional checkpoint label to include; repeat for multiple labels.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint_contract(
    report: Mapping[str, Any],
    selected_labels: Sequence[str] | None,
) -> dict[str, Path]:
    rows = report.get("contracts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Comparison report contains no checkpoint contracts")
    requested = None if selected_labels is None else set(selected_labels)
    result: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Invalid comparison checkpoint contract")
        label = str(row["checkpoint"])
        if requested is not None and label not in requested:
            continue
        if not SAFE_LABEL.fullmatch(label):
            raise ValueError(f"Unsafe checkpoint label: {label!r}")
        path = project_path(str(row["checkpoint_path"]))
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
        if label in result:
            raise ValueError(f"Duplicate checkpoint label: {label}")
        result[label] = path
    if requested is not None:
        missing = sorted(requested.difference(result))
        if missing:
            raise KeyError(f"Comparison report has no checkpoints for {missing}")
    if not result:
        raise ValueError("No checkpoints were selected")
    return result


def validate_source_contracts(
    checkpoints: Mapping[str, Path],
) -> tuple[dict[str, Any], Path, list[str], int, Path]:
    configs = {
        label: load_source_config(checkpoint)
        for label, checkpoint in checkpoints.items()
    }
    first_label = next(iter(checkpoints))
    first_config = configs[first_label]
    first_paths = resolve_data_paths(first_config)
    first_data = data_config_from_config(first_config)
    fixed_gap = int(first_data.get("fixed_gap", -1))
    if fixed_gap < 0:
        raise ValueError("Fixed-gap checkpoint source config has no valid data.fixed_gap")
    validation_names = read_split_names(first_paths["eval_split"])
    motion_token_dir = first_paths["motion_token_dir"].resolve()

    step2_checkpoints: dict[str, Path] = {}
    for label, config in configs.items():
        paths = resolve_data_paths(config)
        data = data_config_from_config(config)
        if read_split_names(paths["eval_split"]) != validation_names:
            raise ValueError(f"{label} uses a different validation split")
        if paths["motion_token_dir"].resolve() != motion_token_dir:
            raise ValueError(f"{label} uses a different motion-token export")
        if int(data.get("fixed_gap", -1)) != fixed_gap:
            raise ValueError(f"{label} uses a different fixed gap")
        guidance = section(config, "frozen_step2_guidance")
        if not bool(guidance.get("enabled", False)):
            raise ValueError(f"{label} did not enable frozen Step 2 guidance")
        step2_path = project_path(str(guidance["checkpoint"]))
        if not step2_path.is_dir():
            raise FileNotFoundError(
                f"{label} frozen Step 2 checkpoint is missing: {step2_path}"
            )
        step2_checkpoints[label] = step2_path
    unique_step2 = {value.resolve() for value in step2_checkpoints.values()}
    if len(unique_step2) != 1:
        raise ValueError(
            "Selected checkpoints were trained against different frozen Step 2 weights: "
            f"{step2_checkpoints}"
        )
    return (
        first_config,
        motion_token_dir,
        validation_names,
        fixed_gap,
        next(iter(unique_step2)),
    )


def adaptive_cache_payload(
    *,
    name: str,
    token_length: int,
    anchor_times: Sequence[int],
    anchors: np.ndarray,
    fixed_gap: int,
    anchor_history: str,
) -> dict[str, Any]:
    times = tuple(int(value) for value in anchor_times)
    values = np.asarray(anchors, dtype=np.int64)
    expected = (len(times) - 1, BODY_SLOT_COUNT)
    if values.shape != expected:
        raise ValueError(f"{name}: anchor array {values.shape} != {expected}")
    if np.any(values < 0) or np.any(values >= 512):
        raise ValueError(f"{name}: anchor IDs leave the 512-entry codebooks")
    gaps = tuple(
        gap_from_anchor_times(left, right)
        for left, right in zip(times[:-1], times[1:])
    )
    return {
        "schema": "sentiavatar.step1_adaptive_rollout.v1",
        "name": name,
        "policy": f"fixed_gap_{fixed_gap}",
        "anchor_history": anchor_history,
        "duration_contract": "offline_known_motion_token_length",
        "token_length": int(token_length),
        "eos_clipped_decisions": 0,
        "predicted_gap_decisions": list(gaps),
        "executed_gaps": list(gaps),
        "anchors": [
            {
                "time": int(time_value),
                "tokens": [int(value) for value in anchor],
            }
            for time_value, anchor in zip(times[1:], values)
        ],
    }


def convert_legacy_payload(
    payload: Mapping[str, Any],
    *,
    name: str,
    dense_tokens: np.ndarray,
    expected_times: Sequence[int],
    fixed_gap: int,
) -> dict[str, Any]:
    if str(payload.get("name")) != name:
        raise ValueError(
            f"Legacy cache name {payload.get('name')!r} does not match {name!r}"
        )
    if str(payload.get("decoder")) != "greedy":
        raise ValueError(f"{name}: expected a greedy fixed-gap rollout cache")
    records = payload.get("anchors")
    if not isinstance(records, list):
        raise ValueError(f"{name}: legacy cache anchors must be a list")
    times = (0, *(int(value["time"]) for value in records))
    if tuple(times) != tuple(int(value) for value in expected_times):
        raise ValueError(
            f"{name}: cached anchor times {times} do not match fixed-gap "
            f"schedule {tuple(expected_times)}"
        )
    anchors = np.asarray(
        [value["tokens"] for value in records],
        dtype=np.int64,
    )
    return adaptive_cache_payload(
        name=name,
        token_length=len(dense_tokens),
        anchor_times=times,
        anchors=anchors,
        fixed_gap=fixed_gap,
        anchor_history="generated",
    )


def prepare_fixed_gap_motion_input(
    *,
    comparison_output_dir: Path,
    output_dir: Path,
    selected_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    comparison_output_dir = Path(comparison_output_dir).resolve()
    output_dir = Path(output_dir).resolve()
    report_path = comparison_output_dir / "multipart_comparison_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing multipart comparison report: {report_path}")
    comparison_report = load_json_mapping(report_path)
    checkpoints = checkpoint_contract(comparison_report, selected_labels)
    (
        _source_config,
        motion_token_dir,
        validation_names,
        fixed_gap,
        step2_checkpoint,
    ) = validate_source_contracts(checkpoints)

    protocol = comparison_report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Comparison report has no protocol mapping")
    report_gap = int(protocol.get("fixed_gap", -1))
    if report_gap != fixed_gap:
        raise ValueError(
            f"Comparison fixed_gap={report_gap} differs from checkpoints={fixed_gap}"
        )
    rollout_count = int(protocol.get("rollout_clips", 0))
    subset_seed = int(protocol.get("subset_seed", 42))
    if rollout_count < 1:
        raise ValueError("Comparison report has no rollout clips")
    names = deterministic_subset(validation_names, rollout_count, subset_seed)
    if len(names) != rollout_count:
        raise AssertionError("Selected rollout name count differs from the report")

    gt_condition = f"fixed_gap{fixed_gap}_gt_anchors"
    condition_paths: dict[str, str] = {
        gt_condition: str((output_dir / "rollout_cache" / gt_condition).resolve())
    }
    generated_conditions = {
        label: f"{label}__fixed_gap{fixed_gap}_generated_history"
        for label in checkpoints
    }
    for condition in generated_conditions.values():
        condition_paths[condition] = str(
            (output_dir / "rollout_cache" / condition).resolve()
        )

    converted_counts = {condition: 0 for condition in condition_paths}
    for name in names:
        dense, _ = load_motion_tokens(
            canonical_data_path(motion_token_dir, name, ".json"),
            require_causal=True,
        )
        dense_array = np.asarray(dense, dtype=np.int64)
        expected_times = fixed_anchor_times(len(dense_array), gap=fixed_gap)
        gt_payload = adaptive_cache_payload(
            name=name,
            token_length=len(dense_array),
            anchor_times=expected_times,
            anchors=dense_array[list(expected_times[1:])],
            fixed_gap=fixed_gap,
            anchor_history="ground_truth",
        )
        gt_destination = canonical_data_path(
            Path(condition_paths[gt_condition]),
            name,
            ".json",
        )
        atomic_write_json(gt_destination, gt_payload)
        converted_counts[gt_condition] += 1

        for label, condition in generated_conditions.items():
            source = canonical_data_path(
                comparison_output_dir / "rollout_cache" / label,
                name,
                ".json",
            )
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing --write_rollout_cache output for {label}/{name}: {source}"
                )
            converted = convert_legacy_payload(
                load_json_mapping(source),
                name=name,
                dense_tokens=dense_array,
                expected_times=expected_times,
                fixed_gap=fixed_gap,
            )
            destination = canonical_data_path(
                Path(condition_paths[condition]),
                name,
                ".json",
            )
            atomic_write_json(destination, converted)
            converted_counts[condition] += 1

    step2_fingerprint = checkpoint_fingerprint(step2_checkpoint)
    adapted_report = {
        "protocol": {
            "name": "fixed_gap_generated_anchor_evaluation",
            "validation_split": str(protocol["validation_split"]),
            "rollout_clips": len(names),
            "subset_seed": subset_seed,
            "fixed_gaps": [fixed_gap],
            "generated_fixed_gaps": [fixed_gap],
            "duration_contract": "offline_known_motion_token_length",
            "frozen_step2_cost_manifests": [
                {
                    "checkpoint": str(step2_checkpoint),
                    "checkpoint_fingerprint": step2_fingerprint,
                }
            ],
            "source_comparison_report": str(report_path),
            "test_split_used": False,
        },
        "checkpoints": {
            label: str(path) for label, path in checkpoints.items()
        },
        "rollout_caches": condition_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "selected_names.json", names)
    atomic_write_json(
        output_dir / "adaptive_evaluation_report.json",
        adapted_report,
    )
    summary = {
        "status": "passed",
        "comparison_output_dir": str(comparison_output_dir),
        "output_dir": str(output_dir),
        "clips": len(names),
        "fixed_gap": fixed_gap,
        "conditions": list(condition_paths),
        "converted_counts": converted_counts,
        "step2_checkpoint": str(step2_checkpoint),
        "step2_checkpoint_fingerprint": step2_fingerprint,
    }
    atomic_write_json(output_dir / "fixed_gap_adapter_report.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_fixed_gap_motion_input(
        comparison_output_dir=args.comparison_output_dir,
        output_dir=args.output_dir,
        selected_labels=args.checkpoint_label,
    )
    print(json.dumps(summary, indent=2))
    print(
        "PASS: fixed-gap caches are ready for frozen-Step-2 motion evaluation"
    )


if __name__ == "__main__":
    main()

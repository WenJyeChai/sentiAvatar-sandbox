from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.step1_mimi_planner import canonical_data_path
from scripts.prepare_step1_fixed_gap_motion_evaluation import (
    prepare_fixed_gap_motion_input,
)
from utils.step1_adaptive_evaluation import load_adaptive_rollout_cache


LABELS = ("control_final", "guided_final")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    data_root = tmp_path / "dataset"
    motion_dir = data_root / "motion_tokens"
    split_path = data_root / "eval.txt"
    names = ["group/clip-a", "group/clip-b"]
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    for clip_index, name in enumerate(names):
        tokens = [
            [
                int((clip_index * 37 + frame * 13 + slot * 7) % 512)
                for slot in range(16)
            ]
            for frame in range(18)
        ]
        write_json(
            canonical_data_path(motion_dir, name, ".json"),
            {
                "name": name,
                "tokens": tokens,
                "codebook_size": 512,
                "num_quantizers": 4,
                "part_order": ["upper", "lower", "feet", "hands"],
                "tokens_per_frame": 16,
                "motion_token_fps": 10.0,
                "body_causal": True,
            },
        )

    step2_checkpoint = tmp_path / "step2"
    write_json(step2_checkpoint / "config.json", {"model_type": "test"})
    (step2_checkpoint / "model.safetensors").write_bytes(b"fixed-step2-test")

    comparison_dir = tmp_path / "comparison"
    contracts = []
    for label_index, label in enumerate(LABELS):
        checkpoint = tmp_path / label
        source_config = {
            "paths": {
                "data_root": str(data_root),
                "motion_token_dir": str(motion_dir),
                "train_split": str(split_path),
                "eval_split": str(split_path),
            },
            "data": {"fixed_gap": 7},
            "frozen_step2_guidance": {
                "enabled": True,
                "checkpoint": str(step2_checkpoint),
            },
        }
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "phase1_source_config.json").write_text(
            json.dumps(source_config),
            encoding="utf-8",
        )
        contracts.append(
            {"checkpoint": label, "checkpoint_path": str(checkpoint)}
        )
        for name in names:
            dense = json.loads(
                canonical_data_path(motion_dir, name, ".json").read_text(
                    encoding="utf-8"
                )
            )["tokens"]
            anchors = []
            for time_value in (8, 16, 17):
                values = np.asarray(dense[time_value], dtype=np.int64)
                values = (values + label_index + 1) % 512
                anchors.append(
                    {
                        "time": time_value,
                        "tokens": values.tolist(),
                    }
                )
            write_json(
                canonical_data_path(
                    comparison_dir / "rollout_cache" / label,
                    name,
                    ".json",
                ),
                {
                    "name": name,
                    "decoder": "greedy",
                    "anchors": anchors,
                },
            )

    write_json(
        comparison_dir / "multipart_comparison_report.json",
        {
            "protocol": {
                "validation_split": str(split_path),
                "rollout_clips": len(names),
                "subset_seed": 42,
                "fixed_gap": 7,
            },
            "contracts": contracts,
        },
    )
    return comparison_dir, motion_dir, names


def test_prepare_fixed_gap_motion_input_converts_and_validates_caches(tmp_path):
    comparison_dir, motion_dir, names = make_fixture(tmp_path)
    output_dir = tmp_path / "adapted"

    summary = prepare_fixed_gap_motion_input(
        comparison_output_dir=comparison_dir,
        output_dir=output_dir,
        selected_labels=LABELS,
    )

    assert summary["status"] == "passed"
    assert summary["clips"] == 2
    assert summary["fixed_gap"] == 7
    expected_conditions = {
        "fixed_gap7_gt_anchors",
        "control_final__fixed_gap7_generated_history",
        "guided_final__fixed_gap7_generated_history",
    }
    assert set(summary["conditions"]) == expected_conditions
    report = json.loads(
        (output_dir / "adaptive_evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(report["rollout_caches"]) == expected_conditions

    for name in names:
        dense = np.asarray(
            json.loads(
                canonical_data_path(motion_dir, name, ".json").read_text(
                    encoding="utf-8"
                )
            )["tokens"],
            dtype=np.int64,
        )
        for condition in expected_conditions:
            cache = canonical_data_path(
                Path(report["rollout_caches"][condition]),
                name,
                ".json",
            )
            rollout = load_adaptive_rollout_cache(
                cache,
                dense_motion_tokens=dense,
            )
            assert rollout.anchor_times == (0, 8, 16, 17)
            assert rollout.executed_gaps == (7, 7, 0)
            if condition == "fixed_gap7_gt_anchors":
                assert np.array_equal(
                    rollout.anchors,
                    dense[[8, 16, 17]],
                )


def test_prepare_fixed_gap_motion_input_rejects_stale_anchor_times(tmp_path):
    comparison_dir, _motion_dir, names = make_fixture(tmp_path)
    stale_path = canonical_data_path(
        comparison_dir / "rollout_cache" / "guided_final",
        names[0],
        ".json",
    )
    stale = json.loads(stale_path.read_text(encoding="utf-8"))
    stale["anchors"][0]["time"] = 7
    write_json(stale_path, stale)

    with pytest.raises(ValueError, match="do not match fixed-gap"):
        prepare_fixed_gap_motion_input(
            comparison_output_dir=comparison_dir,
            output_dir=tmp_path / "adapted",
            selected_labels=LABELS,
        )

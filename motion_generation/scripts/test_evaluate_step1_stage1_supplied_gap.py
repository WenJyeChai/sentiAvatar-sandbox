from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_step1_stage1_supplied_gap import (
    adaptive_cache_payload,
    stable_json_hash,
    summarize_by_gap,
)


def test_adaptive_cache_payload_accepts_normal_gaps_and_short_eos_tail() -> None:
    payload = adaptive_cache_payload(
        name="session/clip",
        token_length=20,
        anchor_times=(0, 16, 19),
        anchors=np.zeros((2, 16), dtype=np.int64),
        min_gap=3,
        max_gap=15,
        anchor_history="generated",
    )
    assert payload["executed_gaps"] == [15, 2]
    assert payload["anchor_history"] == "generated"
    assert [value["time"] for value in payload["anchors"]] == [16, 19]


def test_adaptive_cache_payload_rejects_non_tail_short_gap() -> None:
    with pytest.raises(ValueError, match="non-tail supplied gap"):
        adaptive_cache_payload(
            name="session/clip",
            token_length=10,
            anchor_times=(0, 3, 9),
            anchors=np.zeros((2, 16), dtype=np.int64),
            min_gap=3,
            max_gap=15,
            anchor_history="generated",
        )


def test_stable_json_hash_does_not_depend_on_mapping_order() -> None:
    assert stable_json_hash({"b": 2, "a": 1}) == stable_json_hash(
        {"a": 1, "b": 2}
    )


def test_summarize_by_gap_reports_separate_anchor_groups() -> None:
    targets = np.zeros((3, 16), dtype=np.int64)
    predictions = targets.copy()
    predictions[1, 0] = 1
    rows = summarize_by_gap(
        label="candidate",
        gaps=np.asarray([3, 7, 3]),
        targets=targets,
        predictions=predictions,
        negative_log_likelihood=np.ones_like(targets, dtype=np.float64),
    )
    by_gap = {int(row["gap"]): row for row in rows}
    assert by_gap[3]["accuracy"] == 1.0
    assert by_gap[7]["accuracy"] == 15 / 16
    assert by_gap[3]["cross_entropy"] == 1.0

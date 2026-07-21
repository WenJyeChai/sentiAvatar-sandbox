from __future__ import annotations

from collections import Counter

import pytest

from build_step1_training_subsets import (
    ClipMetadata,
    balanced_priority,
    enrich_metadata,
    extract_annotation,
    parse_subsets,
    validate_official_splits,
)


def make_rows(count: int = 120) -> list[ClipMetadata]:
    rows = []
    for index in range(count):
        source = "minority" if index % 10 == 0 else "majority"
        rows.append(
            ClipMetadata(
                name=f"{source}/session_{index % 8:02d}/clip_{index:04d}",
                source=source,
                session=f"{source}/session_{index % 8:02d}",
                duration_seconds=1.0 + index % 20,
                text_characters=2 + index % 31,
                annotation_pattern=("expression+action" if index % 3 else "action-only"),
                expression=f"expression_{index % 12}" if index % 3 else "<unannotated>",
                action=f"action_{index % 24}",
                joint_speed_mean=index / count,
                joint_speed_p90=(index * 7 % count) / count,
                body_rotation_speed=(index * 11 % count) / count,
                body_rotation_p90=(index * 17 % count) / count,
                hand_rotation_speed=(index * 13 % count) / count,
                hand_rotation_p90=(index * 19 % count) / count,
            )
        )
    return enrich_metadata(rows, bins=5)


def test_extract_annotation_handles_both_single_and_missing_tags() -> None:
    expression, action, pattern, length = extract_annotation(
        "【表情：微笑】【动作：点头】你好"
    )
    assert (expression, action, pattern, length) == ("微笑", "点头", "expression+action", 2)
    assert extract_annotation("【动作：挥手】再见")[:3] == (
        "<unannotated>",
        "挥手",
        "action-only",
    )
    assert extract_annotation("普通文本")[:3] == (
        "<unannotated>",
        "<unannotated>",
        "no-tags",
    )


def test_balanced_priority_is_deterministic_unique_and_nested() -> None:
    rows = make_rows()
    first = balanced_priority(rows, 60, seed=42)
    second = balanced_priority(rows, 60, seed=42)
    prefix = balanced_priority(rows, 20, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 60
    assert first[:20] == prefix

    selected_sources = Counter(rows[index].source for index in first[:20])
    assert set(selected_sources) == {"majority", "minority"}
    assert len({rows[index].session for index in first[:20]}) >= 6
    complexity_counts = Counter(rows[index].complexity_bin for index in first)
    assert set(complexity_counts) == {row.complexity_bin for row in rows}
    assert max(complexity_counts.values()) - min(complexity_counts.values()) <= 5


def test_split_contract_and_subset_parser() -> None:
    validate_official_splits(["a", "b"], ["c"], ["d"])
    with pytest.raises(ValueError, match="overlap"):
        validate_official_splits(["a", "b"], ["b"], ["d"])
    assert parse_subsets(["tiny=2", "large=4"], train_size=5) == [
        ("tiny", 2),
        ("large", 4),
    ]
    with pytest.raises(ValueError, match="unique"):
        parse_subsets(["a=2", "b=2"], train_size=5)

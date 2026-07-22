from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from utils.legacy_step1_evaluation import (  # noqa: E402
    LegacyGenerationResult,
    LegacyStep1Example,
    build_legacy_prompt,
    evaluate_legacy_generations,
    extract_legacy_action_text,
    parse_legacy_generated_ids,
    parse_legacy_generated_plan,
)


class FakeTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def convert_ids_to_tokens(self, token_ids):
        return [self.tokens[int(index)] for index in token_ids]


def test_legacy_prompt_and_token_parser() -> None:
    assert extract_legacy_action_text("【表情：开心】【动作：挥手】你好") == "动作：挥手"
    prompt, times = build_legacy_prompt("动作：挥手", list(range(10)), step=4)
    assert times == (0, 4, 8)
    assert prompt == "Human: 动作：挥手[audio_0][audio_4][audio_8]<|im_end|>\nAssistant:"
    tokenizer = FakeTokenizer(
        [
            "[step_4]",
            "[res_1_3]",
            "[res_2_4]",
            "[res_3_5]",
            "[res_4_6]",
            "[frame_3]",
            "[res_1_3]",
            "[res_2_4]",
            "[res_3_5]",
            "[res_4_6]",
            "<|im_end|>",
        ]
    )
    times, parsed = parse_legacy_generated_plan(tokenizer, list(range(11)))
    assert times == (3,)
    assert np.array_equal(parsed, np.asarray([[3, 4, 5, 6]]))
    assert np.array_equal(parse_legacy_generated_ids(tokenizer, list(range(11))), parsed)


def test_strict_legacy_metrics_penalize_missing_anchors() -> None:
    target = np.asarray(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        dtype=np.int64,
    )
    example = LegacyStep1Example(
        name="clip/a",
        prompt="prompt",
        sampled_times=(3, 7, 11),
        target_anchors=target,
        previous_anchors=np.asarray(
            [[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64
        ),
    )
    # The first two interval-end frames are correct; the last is missing and
    # must count as four incorrect IDs.
    predicted = target[:2].copy()
    measured = evaluate_legacy_generations(
        [LegacyGenerationResult(example, (3, 7), predicted, elapsed_seconds=0.5)]
    )
    summary = measured["summary"]
    assert summary["coverage"] == 2 / 3
    assert summary["accuracy"] == 2 / 3
    assert summary["matched_only_accuracy"] == 1.0
    assert summary["exact_length_rate"] == 0.0

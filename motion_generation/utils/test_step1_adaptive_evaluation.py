from types import SimpleNamespace

import numpy as np
import torch

from utils.adaptive_anchor_tokens import BODY_SLOT_COUNT
from utils.step1_adaptive_evaluation import (
    AdaptiveRolloutExample,
    make_ground_truth_result,
    rollout_policy_batch,
)
from scripts.evaluate_step1_adaptive_motion import (
    assemble_step2_tokens,
    build_step2_records,
)


class _Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        if token.startswith("[gap_"):
            return 10 + int(token[5:-1])
        if token == "[anchor]":
            return 30
        if token == "[mimi_frame]":
            return 31
        if token == "[motion_start]":
            return 32
        if token.startswith("[body_"):
            return 1000 + int(token[6:-1])
        raise KeyError(token)


class _LanguageModel:
    def __init__(self):
        self.output = torch.nn.Embedding(10_000, 1)
        with torch.no_grad():
            self.output.weight.zero_()
            # Learned-gap controller always prefers gap 7.
            self.output.weight[17] = 2.0
            # Every slot classifier prefers local ID zero.
            for slot in range(BODY_SLOT_COUNT):
                self.output.weight[1000 + slot * 512] = 2.0

    def get_output_embeddings(self):
        return self.output


class _FakePlanner:
    def __init__(self):
        self.language_model = _LanguageModel()
        self.motion_token_ids = torch.stack(
            [
                torch.arange(
                    1000 + slot * 512,
                    1000 + (slot + 1) * 512,
                    dtype=torch.long,
                )
                for slot in range(BODY_SLOT_COUNT)
            ]
        )
        self.gap_token_ids = torch.arange(10, 26, dtype=torch.long)

    def eval(self):
        return self

    def prepare_input_embeddings(self, input_ids, audio_codes):
        del audio_codes
        return torch.ones(
            (*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device
        )

    def prepare_planner_attention_mask(
        self, attention_mask, bidirectional_prefix_mask, *, dtype
    ):
        del bidirectional_prefix_mask, dtype
        return attention_mask

    def _base_model_forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(
            last_hidden_state=torch.ones_like(inputs_embeds),
            past_key_values=(),
        )


def _example(name, token_length, prefix_length, oracle_times):
    dense = np.zeros((token_length, BODY_SLOT_COUNT), dtype=np.int64)
    audio_frames = int(np.ceil(token_length * 1.25))
    return AdaptiveRolloutExample(
        name=name,
        initial_input_ids=tuple(range(100, 100 + prefix_length)),
        audio_codes=np.zeros((audio_frames, 4), dtype=np.int64),
        dense_motion_tokens=dense,
        oracle_anchor_times=tuple(oracle_times),
    )


def test_adaptive_generated_rollout_handles_variable_prefixes_and_eos_clip():
    examples = [
        _example("short", 10, 5, (0, 5, 9)),
        _example("long", 17, 8, (0, 8, 16)),
    ]
    results = rollout_policy_batch(
        _FakePlanner(),
        _Tokenizer(),
        examples,
        policy="adaptive",
        anchor_history="generated",
        device=torch.device("cpu"),
        use_bf16=False,
    )
    assert results[0].anchor_times == (0, 8, 9)
    assert results[0].predicted_gap_decisions == (7, 7)
    assert results[0].executed_gaps == (7, 0)
    assert results[0].eos_clipped_decisions == 1
    assert results[1].anchor_times == (0, 8, 16)
    assert results[1].executed_gaps == (7, 7)
    assert results[1].eos_clipped_decisions == 0
    assert np.array_equal(results[0].anchors, results[0].target_anchors)
    assert np.array_equal(results[1].anchors, results[1].target_anchors)


def test_control_results_cover_exact_final_frame():
    example = _example("control", 18, 5, (0, 8, 17))
    fixed = make_ground_truth_result(example, policy="fixed", fixed_gap=7)
    oracle = make_ground_truth_result(example, policy="oracle")
    assert fixed.anchor_times == (0, 8, 16, 17)
    assert fixed.executed_gaps == (7, 7, 0)
    assert oracle.anchor_times == (0, 8, 17)
    assert oracle.executed_gaps == (7, 8)


def test_full_audio_teacher_rollout_marks_audio_as_preloaded():
    example = _example("teacher", 17, 30, (0, 8, 16))
    prefix_audio = np.full((30, 4), -1, dtype=np.int64)
    prefix_audio[2 : 2 + len(example.audio_codes)] = 0
    example = AdaptiveRolloutExample(
        name=example.name,
        initial_input_ids=example.initial_input_ids,
        audio_codes=example.audio_codes,
        dense_motion_tokens=example.dense_motion_tokens,
        oracle_anchor_times=example.oracle_anchor_times,
        initial_audio_codes=prefix_audio,
        bidirectional_prefix_mask=np.ones(30, dtype=bool),
    )
    result = rollout_policy_batch(
        _FakePlanner(),
        _Tokenizer(),
        [example],
        policy="adaptive",
        anchor_history="generated",
        device=torch.device("cpu"),
        use_bf16=False,
    )[0]
    assert result.anchor_times == (0, 8, 16)
    assert result.executed_gaps == (7, 7)


def test_step2_assembly_skips_adjacent_eos_interval():
    example = _example("adjacent_tail", 10, 5, (0, 5, 9))
    result = make_ground_truth_result(example, policy="fixed", fixed_gap=7)
    sequence = {
        "name": example.name,
        "motion_tokens": example.dense_motion_tokens.tolist(),
        "audio_features": np.zeros((13, 3), dtype=np.float32),
        "audio_fps": 12.5,
        "motion_token_fps": 10.0,
    }
    records, prediction_index = build_step2_records(
        [sequence],
        rollouts={example.name: result},
    )
    assert len(records) == 1
    assert records[0].gap_frames == 7
    predicted_middle = np.ones((7, BODY_SLOT_COUNT), dtype=np.int64)
    assembled = assemble_step2_tokens(
        [sequence],
        rollouts={example.name: result},
        predictions=[predicted_middle],
        prediction_index=prediction_index,
    )[example.name]
    assert np.array_equal(assembled[1:8], predicted_middle)
    assert np.array_equal(assembled[8:], np.zeros((2, BODY_SLOT_COUNT)))

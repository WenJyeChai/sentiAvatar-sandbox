from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config


MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    IGNORE_INDEX,
    MimiQwenPlanner,
    MimiQwenPlannerConfig,
    Step1AdaptiveGapDataset,
    Step1FixedGapDataset,
    Step1PlannerCollator,
    Step1ProvidedGapDataset,
    build_prefix_lm_attention_mask,
    parse_structured_text,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    ACTION_MISSING_TOKEN,
    ACTION_TOKEN,
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    EXPRESSION_MISSING_TOKEN,
    EXPRESSION_TOKEN,
    GAP_TOKENS,
    MIMI_FRAME_TOKEN,
    MOTION_HISTORY_FRAME_TOKEN,
    STEP2_HISTORY_END_TOKEN,
    STEP2_HISTORY_MASK_TOKENS,
    STEP2_HISTORY_START_TOKEN,
    TRANSCRIPT_TOKEN,
    body_global_id,
    body_token,
    causal_audio_boundaries,
    audio_token_id_table,
    ensure_nano_audio_tokens,
    ensure_step1_special_tokens,
    fixed_anchor_times,
    gap_from_anchor_times,
    motion_token_id_table,
    nano_audio_token,
    parse_body_token,
    split_body_global_id,
)
from utils.step1_expected_distortion import (  # noqa: E402
    normalized_codebook_distance_table,
)
from utils.step1_condition_alignment import (  # noqa: E402
    corrupt_audio_with_causal_past,
    corrupt_text_condition,
    counterfactual_likelihood_loss,
)
from utils.step1_adaptive_schedule import parse_curriculum  # noqa: E402
from utils.step1_step2_history import (  # noqa: E402
    cache_path as step2_history_cache_path,
    load_step2_history_cache,
    save_step2_history_cache,
)


@pytest.fixture(scope="module")
def step1_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_DIR / "checkpoints" / "llm", local_files_only=True)
    ensure_step1_special_tokens(tokenizer, include_structured_text=True)
    return tokenizer


def _write_synthetic_clip(root: Path, name: str, *, token_frames: int = 36, audio_frames: int = 45):
    motion_dir = root / "motion"
    audio_dir = root / "audio"
    motion_path = motion_dir / f"{name}.json"
    audio_path = audio_dir / f"{name}.npz"
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = [
        [(frame * 17 + slot * 3) % BODY_CODEBOOK_SIZE for slot in range(BODY_SLOT_COUNT)]
        for frame in range(token_frames)
    ]
    motion_payload = {
        "name": name,
        "tokens": tokens,
        "fps": 20.0,
        "motion_token_fps": 10.0,
        "motion_token_unit_length": 2,
        "codebook_size": 512,
        "num_quantizers": 4,
        "part_order": ["upper", "lower", "feet", "hands"],
        "tokens_per_frame": 16,
        "body_causal": True,
    }
    motion_path.write_text(json.dumps(motion_payload), encoding="utf-8")
    codes = np.stack(
        [np.arange(audio_frames, dtype=np.uint16) + codebook * 100 for codebook in range(8)]
    )
    np.savez_compressed(
        audio_path,
        codes=codes,
        format_version=np.asarray(1, dtype=np.int32),
        name=np.asarray(name),
        sample_rate=np.asarray(24_000, dtype=np.int32),
        num_samples=np.asarray(token_frames * 2_400, dtype=np.int64),
        frame_rate=np.asarray(12.5, dtype=np.float32),
        frame_size=np.asarray(1_920, dtype=np.int32),
        num_codebooks=np.asarray(8, dtype=np.int32),
        cardinality=np.asarray(2_048, dtype=np.int32),
    )
    return motion_dir, audio_dir, tokens, codes


def test_body_slot_mapping_round_trips_all_ids():
    seen = set()
    for slot in range(BODY_SLOT_COUNT):
        for local_id in range(BODY_CODEBOOK_SIZE):
            global_id = body_global_id(slot, local_id)
            assert split_body_global_id(global_id) == (slot, local_id)
            assert parse_body_token(body_token(slot, local_id)) == (slot, local_id)
            seen.add(global_id)
    assert seen == set(range(BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE))


def test_fixed_gap3_schedule_and_audio_alignment():
    times = fixed_anchor_times(36, gap=3)
    assert times == (0, 4, 8, 12, 16, 20, 24, 28, 32, 35)
    assert [gap_from_anchor_times(a, b) for a, b in zip(times, times[1:])] == [3] * 8 + [2]
    assert causal_audio_boundaries(times, audio_frames=45) == (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)


def test_dataset_serializes_causal_audio_before_each_anchor(tmp_path: Path, step1_tokenizer):
    name = "session/clip"
    motion_dir, audio_dir, tokens, codes = _write_synthetic_clip(tmp_path, name)
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "【表情：认真】【动作：点头】测试文本。"},
        fixed_gap=3,
        seed_mode="observed",
    )
    item = dataset[0]
    assert item["anchor_times"] == (0, 4, 8, 12, 16, 20, 24, 28, 32, 35)
    assert item["audio_boundaries"] == (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
    assert sum(frame_codes[0] >= 0 for frame_codes in item["audio_codes"]) == 45
    assert [
        frame_codes[0] for frame_codes in item["audio_codes"] if frame_codes[0] >= 0
    ] == codes[0].astype(int).tolist()
    assert sum(slot >= 0 for slot in item["target_slots"]) == 9 * BODY_SLOT_COUNT
    assert [slot for slot in item["target_slots"] if slot >= 0] == list(range(16)) * 9

    gap3_id = step1_tokenizer.convert_tokens_to_ids(GAP_TOKENS[3])
    gap2_id = step1_tokenizer.convert_tokens_to_ids(GAP_TOKENS[2])
    mimi_id = step1_tokenizer.convert_tokens_to_ids(MIMI_FRAME_TOKEN)
    assert item["input_ids"].count(gap3_id) == 8
    assert item["input_ids"].count(gap2_id) == 1
    assert item["input_ids"].count(mimi_id) == 45

    first_target_position = next(i for i, slot in enumerate(item["target_slots"]) if slot == 0)
    first_gap_position = item["input_ids"].index(gap3_id)
    first_audio_positions = [
        i for i in range(first_gap_position + 1, first_target_position) if item["input_ids"][i] == mimi_id
    ]
    assert len(first_audio_positions) == 5
    assert item["motion_local_labels"][first_target_position : first_target_position + 16] == tokens[4]
    assert sum(item["text_mask"]) > 0
    assert set(value for value in item["audio_anchor_ids"] if value >= 0) == set(range(9))
    assert item["target_anchor_ids"][first_target_position : first_target_position + 16] == [0] * 16


def test_dataset_serializes_full_audio_and_seed_as_one_teacher_prefix(
    tmp_path: Path, step1_tokenizer
):
    name = "session/full_audio_teacher"
    motion_dir, audio_dir, _, codes = _write_synthetic_clip(tmp_path, name)
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "full audio teacher"},
        fixed_gap=7,
        seed_mode="observed",
        sequence_layout="full_audio_prefix",
    )
    item = dataset[0]
    prefix = np.asarray(item["bidirectional_prefix_mask"], dtype=bool)
    prefix_length = int(prefix.sum())
    assert np.array_equal(
        prefix,
        np.arange(len(prefix)) < prefix_length,
    )
    assert not (np.asarray(item["target_slots"])[prefix] >= 0).any()

    audio_positions = np.flatnonzero(
        np.asarray(item["audio_codes"], dtype=np.int64)[:, 0] >= 0
    )
    assert len(audio_positions) == codes.shape[1]
    assert bool(prefix[audio_positions].all())
    assert np.array_equal(
        np.asarray(item["audio_codes"], dtype=np.int64)[audio_positions, 0],
        codes[0].astype(np.int64),
    )
    first_gap_position = next(
        index
        for index, token_id in enumerate(item["input_ids"])
        if token_id
        in {
            step1_tokenizer.convert_tokens_to_ids(token)
            for token in GAP_TOKENS
        }
    )
    assert first_gap_position == prefix_length
    assert not prefix[first_gap_position]


def test_prefix_lm_mask_has_bidirectional_prefix_and_causal_plan():
    attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
    prefix = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    mask = build_prefix_lm_attention_mask(
        attention,
        prefix,
        dtype=torch.float32,
    )[0, 0]
    allowed = mask.eq(0)
    assert allowed[0].tolist() == [True, True, True, False, False, False]
    assert allowed[2].tolist() == [True, True, True, False, False, False]
    assert allowed[3].tolist() == [True, True, True, True, False, False]
    assert allowed[4].tolist() == [True, True, True, True, True, False]
    # Padding queries keep valid causal keys to avoid an all-masked SDPA row.
    assert allowed[5].tolist() == [True, True, True, True, True, False]


def test_gt_boundary_step2_history_is_dense_causal_context_only(
    tmp_path: Path,
):
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_DIR / "checkpoints" / "llm",
        local_files_only=True,
    )
    ensure_step1_special_tokens(
        tokenizer,
        include_structured_text=True,
        include_step2_history=True,
    )
    name = "session/step2_history"
    motion_dir, audio_dir, dense_tokens, _ = _write_synthetic_clip(
        tmp_path,
        name,
    )
    anchors = (0, 8, 16, 24, 32, 35)
    interval_frames = []
    for interval_index, (left, right) in enumerate(
        zip(anchors[:-1], anchors[1:])
    ):
        gap = gap_from_anchor_times(left, right)
        predicted = np.asarray(
            [
                [
                    (300 + 19 * interval_index + 7 * frame + slot) % 512
                    for slot in range(16)
                ]
                for frame in range(gap)
            ],
            dtype=np.int64,
        )
        interval_frames.append(
            np.concatenate(
                [
                    predicted,
                    np.asarray(dense_tokens[right], dtype=np.int64)[None],
                ],
                axis=0,
            )
        )
    cache_root = tmp_path / "history_cache"
    history_path = step2_history_cache_path(cache_root, name)
    save_step2_history_cache(
        history_path,
        name=name,
        token_frames=len(dense_tokens),
        anchor_times=anchors,
        interval_frames=interval_frames,
        schedule_seed=42,
        step2_checkpoint_fingerprint="unit-test",
    )
    loaded = load_step2_history_cache(
        history_path,
        dense_motion_tokens=dense_tokens,
    )
    assert loaded.anchor_times == anchors

    dataset = Step1ProvidedGapDataset(
        [name],
        tokenizer=tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "dense history"},
        min_gap=3,
        max_gap=15,
        resample_each_epoch=False,
        seed_mode="observed",
        sequence_layout="full_audio_prefix",
        provided_schedule_cache_dir=cache_root,
        step2_history_cache_dir=cache_root,
        step2_history_min_frames=1,
        step2_history_max_frames=15,
    )
    item = dataset[0]
    assert item["anchor_times"] == anchors
    assert item["step2_history_intervals"] == len(anchors) - 2
    assert 1 * (len(anchors) - 2) <= item["step2_history_frames"]
    assert item["step2_history_frames"] <= 15 * (len(anchors) - 2)
    assert sum(slot >= 0 for slot in item["target_slots"]) == (
        len(anchors) - 1
    ) * BODY_SLOT_COUNT

    start_id = tokenizer.convert_tokens_to_ids(STEP2_HISTORY_START_TOKEN)
    end_id = tokenizer.convert_tokens_to_ids(STEP2_HISTORY_END_TOKEN)
    frame_id = tokenizer.convert_tokens_to_ids(MOTION_HISTORY_FRAME_TOKEN)
    start_positions = [
        index
        for index, token_id in enumerate(item["input_ids"])
        if token_id == start_id
    ]
    end_positions = [
        index
        for index, token_id in enumerate(item["input_ids"])
        if token_id == end_id
    ]
    assert len(start_positions) == len(end_positions) == len(anchors) - 2
    assert item["input_ids"].count(frame_id) == item["step2_history_frames"]
    for interval_index, (start, end) in enumerate(
        zip(start_positions, end_positions)
    ):
        assert all(
            slot == -1 for slot in item["target_slots"][start : end + 1]
        )
        assert all(
            label == IGNORE_INDEX
            for label in item["motion_local_labels"][start : end + 1]
        )
        last_frame_marker = max(
            position
            for position in range(start, end)
            if item["input_ids"][position] == frame_id
        )
        endpoint = [
            parse_body_token(tokenizer.convert_ids_to_tokens(token_id))[1]
            for token_id in item["input_ids"][
                last_frame_marker + 1 : last_frame_marker + 1 + BODY_SLOT_COUNT
            ]
        ]
        assert endpoint == dense_tokens[anchors[interval_index + 1]]

    control = Step1ProvidedGapDataset(
        [name],
        tokenizer=tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "dense history"},
        min_gap=3,
        max_gap=15,
        resample_each_epoch=False,
        seed_mode="observed",
        sequence_layout="full_audio_prefix",
        provided_schedule_cache_dir=cache_root,
    )[0]
    assert control["anchor_times"] == item["anchor_times"]
    assert start_id not in control["input_ids"]
    assert sum(slot >= 0 for slot in control["target_slots"]) == sum(
        slot >= 0 for slot in item["target_slots"]
    )


def test_step2_history_corruption_preserves_gt_endpoint(tmp_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_DIR / "checkpoints" / "llm",
        local_files_only=True,
    )
    ensure_step1_special_tokens(
        tokenizer,
        include_structured_text=True,
        include_step2_history=True,
    )
    name = "session/corrupt_step2_history"
    motion_dir, audio_dir, dense_tokens, _ = _write_synthetic_clip(
        tmp_path,
        name,
    )
    anchors = (0, 8, 16, 24, 32, 35)
    interval_frames = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        gap = gap_from_anchor_times(left, right)
        missing = np.full((gap, BODY_SLOT_COUNT), 123, dtype=np.int64)
        interval_frames.append(
            np.concatenate(
                [missing, np.asarray(dense_tokens[right])[None]],
                axis=0,
            )
        )
    cache_root = tmp_path / "history_cache"
    save_step2_history_cache(
        step2_history_cache_path(cache_root, name),
        name=name,
        token_frames=len(dense_tokens),
        anchor_times=anchors,
        interval_frames=interval_frames,
        schedule_seed=42,
        step2_checkpoint_fingerprint="unit-test",
    )
    item = Step1ProvidedGapDataset(
        [name],
        tokenizer=tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "corrupt history"},
        min_gap=3,
        max_gap=15,
        resample_each_epoch=False,
        seed_mode="observed",
        sequence_layout="full_audio_prefix",
        provided_schedule_cache_dir=cache_root,
        step2_history_cache_dir=cache_root,
        step2_history_min_frames=15,
        step2_history_max_frames=15,
        step2_history_corruption_probability=1.0,
        step2_history_corruption_rate_min=1.0,
        step2_history_corruption_rate_max=1.0,
        step2_history_mask_weight=1.0,
        step2_history_same_slot_replace_weight=0.0,
        step2_history_previous_hold_weight=0.0,
        step2_history_preserve_current_endpoint=True,
    )[0]
    mask_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in STEP2_HISTORY_MASK_TOKENS
    }
    assert item["step2_history_corrupted_tokens"] > 0
    assert any(token_id in mask_ids for token_id in item["input_ids"])
    end_id = tokenizer.convert_tokens_to_ids(STEP2_HISTORY_END_TOKEN)
    frame_id = tokenizer.convert_tokens_to_ids(MOTION_HISTORY_FRAME_TOKEN)
    for interval_index, end in enumerate(
        index
        for index, token_id in enumerate(item["input_ids"])
        if token_id == end_id
    ):
        marker = max(
            position
            for position in range(end)
            if item["input_ids"][position] == frame_id
        )
        endpoint_ids = item["input_ids"][
            marker + 1 : marker + 1 + BODY_SLOT_COUNT
        ]
        assert not any(token_id in mask_ids for token_id in endpoint_ids)
        endpoint = [
            parse_body_token(tokenizer.convert_ids_to_tokens(token_id))[1]
            for token_id in endpoint_ids
        ]
        assert endpoint == dense_tokens[anchors[interval_index + 1]]


def test_dataset_and_collator_build_one_step2_guidance_window(
    tmp_path: Path,
    step1_tokenizer,
):
    name = "session/guidance"
    motion_dir, audio_dir, tokens, _ = _write_synthetic_clip(tmp_path, name)
    feature_dir = tmp_path / "step2_audio"
    feature_path = feature_dir / f"{name}.npy"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.arange(45 * 8, dtype=np.float32).reshape(45, 8)
    np.save(feature_path, features)
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "guidance"},
        fixed_gap=7,
        seed_mode="observed",
        step2_guidance_audio_feature_dir=feature_dir,
        step2_guidance_audio_fps=12.5,
        step2_guidance_audio_feat_dim=8,
        step2_guidance_resample=True,
        step2_guidance_required_gap=7,
    )
    item = dataset[0]
    assert item["step2_guidance_gap"] == 7
    assert len(item["step2_guidance_motion_tokens"]) == 9
    assert item["step2_guidance_audio_features"].shape == (9, 8)
    group = item["step2_guidance_anchor_group"]
    left_time = item["anchor_times"][group]
    right_time = item["anchor_times"][group + 1]
    assert item["step2_guidance_motion_tokens"] == tokens[left_time : right_time + 1]

    batch = Step1PlannerCollator(step1_tokenizer.pad_token_id)([item, item])
    guidance = batch["step2_guidance"]
    assert tuple(guidance["motion_tokens"].shape) == (2, 9, 16)
    assert tuple(guidance["audio_features"].shape) == (2, 9, 8)
    assert guidance["frame_mask"].all()
    assert torch.equal(guidance["gap_lengths"], torch.tensor([7, 7]))
    assert torch.equal(guidance["anchor_groups"], torch.tensor([group, group]))


def test_adaptive_dataset_loads_materialized_dp_schedule(
    tmp_path: Path, step1_tokenizer
):
    name = "session/adaptive"
    motion_dir, audio_dir, _, _ = _write_synthetic_clip(tmp_path, name)
    anchors = np.asarray([0, 8, 16, 24, 32, 35], dtype=np.int32)
    probabilities = np.zeros((len(anchors), len(GAP_TOKENS)), dtype=np.float32)
    probabilities[:4, 7] = 1.0
    target_mask = np.asarray([True, True, True, True, False, False])
    schedule_path = tmp_path / "phase_00.npz"
    np.savez_compressed(
        schedule_path,
        names=np.asarray([name]),
        offsets=np.asarray([0, len(anchors)], dtype=np.int64),
        anchors=anchors,
        gap_target_probs=probabilities,
        gap_target_mask=target_mask,
    )
    digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "phases": [
                    {
                        "phase_index": 0,
                        "min_gap": 3,
                        "max_gap": 15,
                        "target_mean_gap": 7.0,
                        "temperature": 0.35,
                        "schedule_file": schedule_path.name,
                        "schedule_file_sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    phases = parse_curriculum(
        [
            {
                "start_epoch": 1,
                "end_epoch": 1,
                "mode": "step2_dp",
                "min_gap": 3,
                "max_gap": 15,
                "target_mean_gap": 7.0,
                "temperature": 0.35,
                "schedule_loss_weight": 1.0,
            }
        ],
        num_epochs=1,
    )
    dataset = Step1AdaptiveGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "测试。"},
        seed_mode="observed",
        curriculum_phases=phases,
        calibration_json=calibration_path,
    )
    item = dataset[0]
    assert item["anchor_times"] == tuple(anchors.tolist())
    assert item["normal_gaps"] == (7, 7, 7, 7)
    assert item["tail_gap"] == 2
    assert sum(item["gap_target_mask"]) == 4
    assert item["gap_loss_weight"] == 1.0


def test_provided_gap_dataset_resamples_uniform_inputs_without_gap_targets(
    tmp_path: Path,
    step1_tokenizer,
):
    name = "session/provided_gap"
    motion_dir, audio_dir, _, _ = _write_synthetic_clip(
        tmp_path,
        name,
        token_frames=180,
        audio_frames=225,
    )
    dataset = Step1ProvidedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "provided gap"},
        seed_mode="observed",
        min_gap=3,
        max_gap=15,
        resample_each_epoch=True,
    )
    dataset.set_epoch(0)
    first = dataset[0]
    repeated = dataset[0]
    assert first["anchor_times"] == repeated["anchor_times"]
    assert all(3 <= gap <= 15 for gap in first["normal_gaps"])
    assert first["tail_gap"] is None or 0 <= first["tail_gap"] <= 2
    assert not any(first["gap_target_mask"])
    assert first["gap_loss_weight"] == 0.0

    dataset.set_epoch(1)
    second = dataset[0]
    assert first["anchor_times"] != second["anchor_times"]
    assert not any(second["gap_target_mask"])

    fixed_eval = Step1ProvidedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "provided gap"},
        seed_mode="observed",
        min_gap=3,
        max_gap=15,
        resample_each_epoch=False,
    )
    fixed_eval.set_epoch(0)
    eval_times = fixed_eval[0]["anchor_times"]
    fixed_eval.set_epoch(99)
    assert fixed_eval[0]["anchor_times"] == eval_times


def test_dataset_serializes_synchronous_q0_q3_frames(tmp_path: Path, step1_tokenizer):
    name = "session/q0q3"
    motion_dir, audio_dir, _, codes = _write_synthetic_clip(tmp_path, name)
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "q0 q3"},
        mimi_codebooks_used=[0, 1, 2, 3],
    )
    item = dataset[0]
    observed = np.asarray(
        [frame for frame in item["audio_codes"] if all(code >= 0 for code in frame)],
        dtype=np.int64,
    )
    assert observed.shape == (45, 4)
    assert np.array_equal(observed, codes[:4].T)


def test_structured_text_parser_distinguishes_missing_from_explicit_no_action():
    both = parse_structured_text(
        "\u3010\u8868\u60c5\uff1a\u773c\u795e\u8ba4\u771f\u3011"
        "\u3010\u52a8\u4f5c\uff1a\u65e0\u52a8\u4f5c\u3011"
        "\u8bf7\u8bf4\u5b9e\u8bdd\u3002"
    )
    assert both.expression == "\u773c\u795e\u8ba4\u771f"
    assert both.action == "\u65e0\u52a8\u4f5c"
    assert both.transcript == "\u8bf7\u8bf4\u5b9e\u8bdd\u3002"
    assert both.annotation_pattern == "expression+action"
    missing = parse_structured_text("\u53ea\u6709\u5bf9\u8bdd\u3002")
    assert missing.expression is None
    assert missing.action is None
    assert missing.annotation_pattern == "no-tags"


def test_structured_serialization_has_field_specific_markers_and_masks(
    tmp_path: Path, step1_tokenizer
):
    names = ["both", "transcript_only"]
    text_map = {
        "both": (
            "\u3010\u8868\u60c5\uff1a\u773c\u795e\u8ba4\u771f\u3011"
            "\u3010\u52a8\u4f5c\uff1a\u65e0\u52a8\u4f5c\u3011"
            "\u5b89\u5b89\uff0c\u8bf4\u5b9e\u8bdd\u3002"
        ),
        "transcript_only": "\u4e3a\u4ec0\u4e48\u8fd9\u6837\u60f3\uff1f",
    }
    items = []
    for name in names:
        motion_dir, audio_dir, _, _ = _write_synthetic_clip(tmp_path, name)
        dataset = Step1FixedGapDataset(
            [name],
            tokenizer=step1_tokenizer,
            motion_token_dir=motion_dir,
            mimi_token_dir=audio_dir,
            text_map=text_map,
            text_serialization="structured_fields",
        )
        items.append(dataset[0])
    both, transcript_only = items
    assert step1_tokenizer.convert_tokens_to_ids(EXPRESSION_TOKEN) in both["input_ids"]
    assert step1_tokenizer.convert_tokens_to_ids(ACTION_TOKEN) in both["input_ids"]
    assert step1_tokenizer.convert_tokens_to_ids(TRANSCRIPT_TOKEN) in both["input_ids"]
    assert sum(both["expression_mask"]) > 0
    assert sum(both["action_mask"]) > 0
    assert sum(both["transcript_mask"]) > 0
    assert both["annotation_pattern"] == "expression+action"
    assert step1_tokenizer.convert_tokens_to_ids(EXPRESSION_MISSING_TOKEN) in transcript_only["input_ids"]
    assert step1_tokenizer.convert_tokens_to_ids(ACTION_MISSING_TOKEN) in transcript_only["input_ids"]
    assert sum(transcript_only["expression_mask"]) == 0
    assert sum(transcript_only["action_mask"]) == 0
    assert sum(transcript_only["transcript_mask"]) > 0
    assert transcript_only["annotation_pattern"] == "no-tags"


def test_nano_q0_q3_contract_reads_all_16_stored_codebooks(
    tmp_path: Path, step1_tokenizer
):
    name = "nano/clip"
    motion_dir, audio_dir, tokens, _ = _write_synthetic_clip(tmp_path, name)
    audio_path = audio_dir / f"{name}.npz"
    nano_frames = 45
    nano_codes = np.stack(
        [
            (np.arange(nano_frames, dtype=np.uint16) + codebook * 50) % 1024
            for codebook in range(16)
        ]
    )
    np.savez_compressed(
        audio_path,
        codes=nano_codes,
        format_version=np.asarray(2, dtype=np.int32),
        codec=np.asarray("moss_audio_tokenizer_nano"),
        name=np.asarray(name),
        sample_rate=np.asarray(48_000, dtype=np.int32),
        num_samples=np.asarray(len(tokens) * 4_800, dtype=np.int64),
        frame_rate=np.asarray(12.5, dtype=np.float32),
        frame_size=np.asarray(3_840, dtype=np.int32),
        num_codebooks=np.asarray(16, dtype=np.int32),
        cardinality=np.asarray(1_024, dtype=np.int32),
    )
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "\u6d4b\u8bd5"},
        audio_codec="moss_audio_tokenizer_nano",
        audio_sample_rate=48_000,
        audio_frame_rate=12.5,
        audio_frame_size=3_840,
        audio_codebooks_stored=16,
        audio_cardinality=1_024,
        audio_codebooks_used=[0, 1, 2, 3],
        text_serialization="structured_fields",
    )
    item = dataset[0]
    observed = np.asarray(
        [frame for frame in item["audio_codes"] if all(code >= 0 for code in frame)]
    )
    assert observed.shape == (nano_frames, 4)
    assert np.array_equal(observed, nano_codes[:4].T)


def test_nano_q0_q3_ordinary_tokens_are_time_major_and_keep_q0_metadata(
    tmp_path: Path,
):
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_DIR / "checkpoints" / "llm",
        local_files_only=True,
    )
    ensure_step1_special_tokens(tokenizer, include_structured_text=True)
    added = ensure_nano_audio_tokens(tokenizer)
    assert len(added) == 4 * 1_024
    token_table = audio_token_id_table(tokenizer)

    name = "nano/ordinary"
    motion_dir, audio_dir, tokens, _ = _write_synthetic_clip(tmp_path, name)
    nano_frames = 45
    nano_codes = np.stack(
        [
            (np.arange(nano_frames, dtype=np.uint16) + codebook * 50) % 1_024
            for codebook in range(16)
        ]
    )
    np.savez_compressed(
        audio_dir / f"{name}.npz",
        codes=nano_codes,
        sample_rate=np.asarray(48_000, dtype=np.int32),
        num_samples=np.asarray(len(tokens) * 4_800, dtype=np.int64),
        frame_rate=np.asarray(12.5, dtype=np.float32),
        frame_size=np.asarray(3_840, dtype=np.int32),
        num_codebooks=np.asarray(16, dtype=np.int32),
        cardinality=np.asarray(1_024, dtype=np.int32),
    )
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "ordinary Nano"},
        audio_codec="moss_audio_tokenizer_nano",
        audio_sample_rate=48_000,
        audio_frame_rate=12.5,
        audio_frame_size=3_840,
        audio_codebooks_stored=16,
        audio_cardinality=1_024,
        audio_codebooks_used=[0, 1, 2, 3],
        audio_input_representation="ordinary_tokens",
        sequence_layout="full_audio_prefix",
    )
    item = dataset[0]
    metadata = np.asarray(item["audio_codes"], dtype=np.int64)
    starts = np.flatnonzero(np.all(metadata >= 0, axis=-1))
    assert len(starts) == nano_frames
    assert np.array_equal(metadata[starts], nano_codes[:4].T)
    for frame, position in enumerate(starts):
        expected = [
            token_table[codebook][int(nano_codes[codebook, frame])]
            for codebook in range(4)
        ]
        assert item["input_ids"][position : position + 4] == expected
        assert np.all(metadata[position + 1 : position + 4] == -1)
        assert all(item["bidirectional_prefix_mask"][position : position + 4])
    assert tokenizer.encode(
        nano_audio_token(3, 1_023),
        add_special_tokens=False,
    ) == [token_table[3][1_023]]
    assert tokenizer.convert_tokens_to_ids(MIMI_FRAME_TOKEN) not in item["input_ids"]


def test_generated_prefix_changes_inputs_but_keeps_gt_labels(tmp_path: Path, step1_tokenizer):
    name = "session/clip"
    motion_dir, audio_dir, tokens, _ = _write_synthetic_clip(tmp_path, name)
    generated_dir = tmp_path / "generated"
    generated_path = generated_dir / f"{name}.json"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_anchor = [(value + 1) % 512 for value in tokens[4]]
    generated_path.write_text(
        json.dumps({"anchors": [{"time": 4, "tokens": generated_anchor}]}), encoding="utf-8"
    )
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "测试"},
        generated_anchor_dir=generated_dir,
        generated_prefix_probability=1.0,
    )
    item = dataset[0]
    first_target = next(i for i, slot in enumerate(item["target_slots"]) if slot == 0)
    assert item["input_ids"][first_target] == step1_tokenizer.convert_tokens_to_ids(
        body_token(0, generated_anchor[0])
    )
    assert item["motion_local_labels"][first_target] == tokens[4][0]
    assert item["generated_prefix_anchors"] == 1


def test_collator_masks_padding_and_preserves_modal_fields(tmp_path: Path, step1_tokenizer):
    name = "clip"
    motion_dir, audio_dir, _, _ = _write_synthetic_clip(tmp_path, name, token_frames=9, audio_frames=12)
    dataset = Step1FixedGapDataset(
        [name],
        tokenizer=step1_tokenizer,
        motion_token_dir=motion_dir,
        mimi_token_dir=audio_dir,
        text_map={name: "测试"},
        max_duration_mismatch_seconds=0.2,
    )
    item = dataset[0]
    collator = Step1PlannerCollator(step1_tokenizer.pad_token_id, pad_to_multiple_of=8)
    batch = collator([item, item])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert torch.equal(batch["audio_codes"].ge(0).all(dim=-1), batch["input_ids"].eq(
        step1_tokenizer.convert_tokens_to_ids(MIMI_FRAME_TOKEN)
    ))
    assert torch.equal(batch["target_slots"].ge(0), batch["motion_local_labels"].ne(IGNORE_INDEX))
    assert batch["text_mask"].dtype == torch.bool
    assert torch.equal(batch["target_slots"].ge(0), batch["target_anchor_ids"].ge(0))


def test_condition_corruptions_preserve_targets_history_and_sequence_length(
    tmp_path: Path, step1_tokenizer
):
    examples = []
    for name, text in (("a/clip", "first transcript"), ("b/clip", "different words here")):
        motion_dir, audio_dir, _, _ = _write_synthetic_clip(tmp_path, name)
        dataset = Step1FixedGapDataset(
            [name],
            tokenizer=step1_tokenizer,
            motion_token_dir=motion_dir,
            mimi_token_dir=audio_dir,
            text_map={name: text},
            fixed_gap=3,
            seed_mode="observed",
        )
        examples.append(dataset[0])
    batch = Step1PlannerCollator(step1_tokenizer.pad_token_id)(examples)
    text_corruption = corrupt_text_condition(
        input_ids=batch["input_ids"],
        audio_codes=batch["audio_codes"],
        text_mask=batch["text_mask"],
        target_anchor_ids=batch["target_anchor_ids"],
        names=batch["names"],
        selected_indices=[0],
        seed=42,
        epoch=1,
        batch_index=2,
    )
    assert text_corruption.selected_indices.tolist() == [0]
    outside_text = ~batch["text_mask"][0]
    assert torch.equal(
        text_corruption.input_ids[0, outside_text], batch["input_ids"][0, outside_text]
    )
    assert not torch.equal(
        text_corruption.input_ids[0, batch["text_mask"][0]],
        batch["input_ids"][0, batch["text_mask"][0]],
    )
    assert torch.equal(text_corruption.audio_codes, batch["audio_codes"])
    assert int(text_corruption.target_mask[0].sum()) == int(batch["target_slots"][0].ge(0).sum())

    audio_corruption = corrupt_audio_with_causal_past(
        input_ids=batch["input_ids"],
        audio_codes=batch["audio_codes"],
        audio_anchor_ids=batch["audio_anchor_ids"],
        target_anchor_ids=batch["target_anchor_ids"],
        selected_indices=[0],
        shift_anchors=2,
    )
    assert audio_corruption.selected_indices.tolist() == [0]
    assert torch.equal(audio_corruption.input_ids, batch["input_ids"])
    assert int(audio_corruption.target_mask[0].sum()) == 7 * 16
    destination = batch["audio_anchor_ids"][0].eq(2)
    source = batch["audio_anchor_ids"][0].eq(0)
    assert torch.equal(
        audio_corruption.audio_codes[0, destination], batch["audio_codes"][0, source]
    )


def test_counterfactual_loss_rewards_higher_wrong_condition_nll():
    positive = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    negative = torch.tensor([[1.8, 0.0], [2.6, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, False]])
    loss, gap = counterfactual_likelihood_loss(
        positive_token_loss=positive,
        negative_token_loss=negative,
        target_mask=mask,
        margin_nats=0.05,
    )
    assert torch.allclose(gap, torch.tensor([0.8, 0.6]))
    loss.backward()
    assert bool((negative.grad[mask] < 0).all())


def _tiny_planner(
    planner_attention_mode: str = "causal",
) -> MimiQwenPlanner:
    vocabulary = BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE + 8
    qwen_config = Qwen2Config(
        vocab_size=vocabulary,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    language_model = AutoModelForCausalLM.from_config(qwen_config)
    table = [
        [body_global_id(slot, local_id) for local_id in range(BODY_CODEBOOK_SIZE)]
        for slot in range(BODY_SLOT_COUNT)
    ]
    config = MimiQwenPlannerConfig(
        language_model_config=language_model.config.to_dict(),
        audio_placeholder_id=BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE,
        motion_token_ids=table,
        planner_attention_mode=planner_attention_mode,
    )
    planner = MimiQwenPlanner(config, language_model=language_model)
    planner.tie_weights()
    return planner


def test_tiny_prefix_lm_planner_backpropagates_without_plan_leakage():
    planner = _tiny_planner("prefix_lm")
    labels = torch.tensor(
        [(slot * 13 + 5) % BODY_CODEBOOK_SIZE for slot in range(BODY_SLOT_COUNT)]
    )
    input_ids = torch.zeros((1, 22), dtype=torch.long)
    for slot in range(BODY_SLOT_COUNT):
        input_ids[0, 5 + slot] = planner.motion_token_ids[slot, labels[slot]]
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 5:21] = torch.arange(BODY_SLOT_COUNT)
    motion_labels = torch.full_like(input_ids, IGNORE_INDEX)
    motion_labels[0, 5:21] = labels
    prefix = torch.zeros_like(input_ids, dtype=torch.bool)
    prefix[:, :4] = True
    output = planner(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_codes=torch.full_like(input_ids, -1),
        target_slots=target_slots,
        motion_local_labels=motion_labels,
        bidirectional_prefix_mask=prefix,
    )
    assert torch.isfinite(output.loss)
    assert int(output.gap_count) == 0
    assert torch.allclose(output.loss, output.ce_loss)
    output.loss.backward()
    assert (
        planner.language_model.model.layers[0].self_attn.q_proj.weight.grad
        is not None
    )


def test_ordinary_audio_tokens_use_qwen_embeddings_without_custom_fusion():
    body_vocabulary = BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE
    audio_ids = torch.arange(
        body_vocabulary,
        body_vocabulary + 4 * 1_024,
        dtype=torch.long,
    ).reshape(4, 1_024)
    qwen_config = Qwen2Config(
        vocab_size=body_vocabulary + 4 * 1_024 + 2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    language_model = AutoModelForCausalLM.from_config(qwen_config)
    table = [
        [body_global_id(slot, local_id) for local_id in range(BODY_CODEBOOK_SIZE)]
        for slot in range(BODY_SLOT_COUNT)
    ]
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=language_model.config.to_dict(),
            audio_placeholder_id=qwen_config.vocab_size - 1,
            motion_token_ids=table,
            audio_codec="moss_audio_tokenizer_nano",
            audio_sample_rate=48_000,
            audio_frame_rate=12.5,
            audio_frame_size=3_840,
            audio_cardinality=1_024,
            audio_codebooks_stored=16,
            audio_codebooks_used=[0, 1, 2, 3],
            audio_input_representation="ordinary_tokens",
            audio_token_ids=audio_ids.tolist(),
        ),
        language_model=language_model,
    )
    codes = torch.tensor([[[7, 11, 13, 17]]])
    input_ids = audio_ids[:, codes[0, 0]].diagonal().reshape(1, 4)
    metadata = torch.full((1, 4, 4), -1, dtype=torch.long)
    metadata[0, 0] = codes[0, 0]
    expected = planner.language_model.get_input_embeddings()(input_ids)
    actual = planner.prepare_input_embeddings(input_ids, metadata)
    assert torch.equal(actual, expected)
    assert planner.audio_embedding is None
    assert len(planner.additional_audio_embeddings) == 0


def test_wrapping_does_not_reinitialize_language_model():
    qwen_config = Qwen2Config(
        vocab_size=BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE + 8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    language_model = AutoModelForCausalLM.from_config(qwen_config)
    before = language_model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    table = [
        [body_global_id(slot, local_id) for local_id in range(BODY_CODEBOOK_SIZE)]
        for slot in range(BODY_SLOT_COUNT)
    ]
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=language_model.config.to_dict(),
            audio_placeholder_id=BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE,
            motion_token_ids=table,
        ),
        language_model=language_model,
    )
    after = planner.language_model.model.layers[0].self_attn.q_proj.weight.detach()
    assert torch.equal(before, after)


def test_tiny_planner_slot_loss_backprop_and_save_load(tmp_path: Path):
    planner = _tiny_planner()
    labels = torch.tensor([(slot * 19 + 7) % 512 for slot in range(16)], dtype=torch.long)
    input_ids = torch.zeros((1, 20), dtype=torch.long)
    audio_placeholder = planner.config.audio_placeholder_id
    input_ids[0, 2] = audio_placeholder
    for slot in range(16):
        input_ids[0, 4 + slot] = planner.motion_token_ids[slot, labels[slot]]
    attention = torch.ones_like(input_ids)
    audio_codes = torch.full_like(input_ids, -1)
    audio_codes[0, 2] = 123
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 4:] = torch.arange(16)
    motion_labels = torch.full_like(input_ids, IGNORE_INDEX)
    motion_labels[0, 4:] = labels

    output = planner(
        input_ids=input_ids,
        attention_mask=attention,
        audio_codes=audio_codes,
        target_slots=target_slots,
        motion_local_labels=motion_labels,
    )
    assert torch.isfinite(output.loss)
    assert int(output.count) == 16
    output.loss.backward()
    assert planner.audio_embedding.weight.grad is not None
    assert float(planner.audio_embedding.weight.grad[123].abs().sum()) > 0

    save_dir = tmp_path / "planner"
    planner.save_pretrained(save_dir, safe_serialization=True)
    reloaded = MimiQwenPlanner.from_pretrained(save_dir, local_files_only=True)
    assert reloaded.config.audio_placeholder_id == audio_placeholder
    assert torch.equal(reloaded.motion_token_ids, planner.motion_token_ids)
    with torch.no_grad():
        reloaded_output = reloaded(
            input_ids=input_ids,
            attention_mask=attention,
            audio_codes=audio_codes,
            target_slots=target_slots,
            motion_local_labels=motion_labels,
        )
    assert torch.isfinite(reloaded_output.loss)


def test_tiny_planner_soft_gap_loss_uses_next_token_alignment():
    planner = _tiny_planner()
    planner.set_gap_token_ids(list(range(len(GAP_TOKENS))))
    labels = torch.tensor([(slot * 19 + 7) % 512 for slot in range(16)], dtype=torch.long)
    input_ids = torch.zeros((1, 20), dtype=torch.long)
    input_ids[0, 3] = 5
    for slot in range(16):
        input_ids[0, 4 + slot] = planner.motion_token_ids[slot, labels[slot]]
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 4:] = torch.arange(16)
    motion_labels = torch.full_like(input_ids, IGNORE_INDEX)
    motion_labels[0, 4:] = labels
    gap_probabilities = torch.zeros((1, 20, len(GAP_TOKENS)))
    gap_probabilities[0, 3, 5] = 0.75
    gap_probabilities[0, 3, 6] = 0.25
    gap_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    gap_mask[0, 3] = True
    output = planner(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_codes=torch.full_like(input_ids, -1),
        target_slots=target_slots,
        motion_local_labels=motion_labels,
        gap_target_probs=gap_probabilities,
        gap_target_mask=gap_mask,
        gap_loss_weights=torch.tensor([0.5]),
    )
    assert torch.isfinite(output.gap_loss)
    assert int(output.gap_count) == 1
    assert torch.allclose(output.loss, output.ce_loss + output.gap_loss)
    output.loss.backward()


def test_tiny_planner_returns_one_selected_logit_table_per_slot():
    planner = _tiny_planner()
    labels = torch.tensor([(slot * 23 + 3) % 512 for slot in range(16)])
    input_ids = torch.zeros((2, 20), dtype=torch.long)
    target_slots = torch.full_like(input_ids, -1)
    motion_labels = torch.full_like(input_ids, IGNORE_INDEX)
    target_anchor_ids = torch.full_like(input_ids, -1)
    for row in range(2):
        for slot in range(16):
            input_ids[row, 4 + slot] = planner.motion_token_ids[slot, labels[slot]]
        target_slots[row, 4:] = torch.arange(16)
        motion_labels[row, 4:] = labels
        target_anchor_ids[row, 4:] = row

    output = planner(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_codes=torch.full_like(input_ids, -1),
        target_slots=target_slots,
        motion_local_labels=motion_labels,
        target_anchor_ids=target_anchor_ids,
        selected_anchor_groups=torch.tensor([0, 1]),
    )
    assert tuple(output.selected_anchor_logits.shape) == (2, 16, 512)
    output.selected_anchor_logits.square().mean().backward()
    assert planner.language_model.get_output_embeddings().weight.grad is not None


def test_normalized_codebook_distances_are_symmetric_and_unit_scaled():
    generator = torch.Generator().manual_seed(9)
    codebooks = {
        part: torch.randn(4, 512, 7, generator=generator)
        for part in ("upper", "lower", "feet", "hands")
    }
    distances = normalized_codebook_distance_table(codebooks)
    assert distances.shape == (16, 512, 512)
    assert torch.allclose(distances, distances.transpose(-1, -2), atol=1e-6)
    assert torch.count_nonzero(distances.diagonal(dim1=-2, dim2=-1)) == 0
    # Includes zero self-pairs, matching the normalization identity exactly.
    assert torch.allclose(
        distances.mean(dim=(-1, -2)), torch.ones(16), rtol=2e-4, atol=2e-4
    )


def test_expected_distortion_adds_to_ce_and_can_select_examples():
    planner = _tiny_planner()
    distances = torch.ones(16, 512, 512)
    distances.diagonal(dim1=-2, dim2=-1).zero_()
    planner.set_motion_codebook_distances(distances)
    labels = torch.tensor([(slot * 13 + 5) % 512 for slot in range(16)], dtype=torch.long)
    input_ids = torch.zeros((1, 18), dtype=torch.long)
    for slot in range(16):
        input_ids[0, 2 + slot] = planner.motion_token_ids[slot, labels[slot]]
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 2:] = torch.arange(16)
    motion_labels = torch.full_like(input_ids, IGNORE_INDEX)
    motion_labels[0, 2:] = labels
    output = planner(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_codes=torch.full_like(input_ids, -1),
        target_slots=target_slots,
        motion_local_labels=motion_labels,
        expected_distortion_weight=0.25,
        expected_distortion_example_mask=torch.tensor([True]),
    )
    assert int(output.expected_distortion_count) == 16
    assert float(output.expected_distortion_loss.detach()) > 0
    assert torch.allclose(
        output.loss,
        output.ce_loss + 0.25 * output.expected_distortion_loss,
    )
    output.loss.backward()
    assert planner.language_model.get_output_embeddings().weight.grad is not None


def test_expected_distortion_requires_loaded_codec_geometry():
    planner = _tiny_planner()
    input_ids = torch.zeros((1, 3), dtype=torch.long)
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 2] = 0
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, 2] = 1
    with pytest.raises(RuntimeError, match="distance tables"):
        planner(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            audio_codes=torch.full_like(input_ids, -1),
            target_slots=target_slots,
            motion_local_labels=labels,
            expected_distortion_weight=0.1,
        )


def test_returned_per_token_losses_preserve_gradient_and_match_ce():
    planner = _tiny_planner()
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 2] = 0
    target_slots[0, 3] = 1
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, 2:] = torch.tensor([7, 11])
    output = planner(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_codes=torch.full_like(input_ids, -1),
        target_slots=target_slots,
        motion_local_labels=labels,
        return_token_losses=True,
    )
    assert output.per_token_loss.shape == input_ids.shape
    assert torch.allclose(output.per_token_loss.sum(), output.ce_loss * output.count)
    output.per_token_loss[:, 2:].mean().backward()
    assert planner.language_model.get_output_embeddings().weight.grad is not None


def test_audio_code_must_match_placeholder_positions():
    planner = _tiny_planner()
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    audio_codes = torch.full_like(input_ids, -1)
    audio_codes[0, 1] = 12
    with pytest.raises(ValueError, match="exactly"):
        planner.prepare_input_embeddings(input_ids, audio_codes)


def test_q0_q3_sparse_audio_fusion_backpropagates_all_codebooks(tmp_path: Path):
    planner = _tiny_planner()
    qwen = planner.language_model
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=qwen.config.to_dict(),
            audio_placeholder_id=planner.config.audio_placeholder_id,
            motion_token_ids=planner.motion_token_ids.tolist(),
            mimi_codebooks_used=[0, 1, 2, 3],
        ),
        language_model=qwen,
    )
    labels = torch.tensor([(slot * 11 + 3) % 512 for slot in range(16)], dtype=torch.long)
    input_ids = torch.zeros((1, 20), dtype=torch.long)
    input_ids[0, 2] = planner.config.audio_placeholder_id
    for slot in range(16):
        input_ids[0, 4 + slot] = planner.motion_token_ids[slot, labels[slot]]
    audio_codes = torch.full((1, 20, 4), -1, dtype=torch.long)
    audio_codes[0, 2] = torch.tensor([101, 202, 303, 404])
    target_slots = torch.full_like(input_ids, -1)
    target_slots[0, 4:] = torch.arange(16)
    motion_labels = torch.full_like(input_ids, IGNORE_INDEX)
    motion_labels[0, 4:] = labels
    output = planner(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_codes=audio_codes,
        target_slots=target_slots,
        motion_local_labels=motion_labels,
    )
    output.loss.backward()
    assert float(planner.audio_embedding.weight.grad[101].abs().sum()) > 0
    for embedding, code in zip(planner.additional_audio_embeddings, [202, 303, 404]):
        assert float(embedding.weight.grad[code].abs().sum()) > 0
    assert planner.audio_fusion.weight.grad is not None

    save_dir = tmp_path / "q0q3-planner"
    planner.save_pretrained(save_dir, safe_serialization=True)
    reloaded = MimiQwenPlanner.from_pretrained(save_dir, local_files_only=True)
    assert reloaded.config.mimi_codebooks_used == [0, 1, 2, 3]
    assert len(reloaded.additional_audio_embeddings) == 3
    assert torch.equal(reloaded.audio_fusion.weight, planner.audio_fusion.weight)


def test_nano_planner_uses_1024_way_audio_embeddings():
    base = _tiny_planner()
    qwen = base.language_model
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=qwen.config.to_dict(),
            audio_placeholder_id=base.config.audio_placeholder_id,
            motion_token_ids=base.motion_token_ids.tolist(),
            audio_codec="moss_audio_tokenizer_nano",
            audio_sample_rate=48_000,
            audio_frame_rate=12.5,
            audio_frame_size=3_840,
            audio_cardinality=1024,
            audio_codebooks_stored=16,
            audio_codebooks_used=[0, 1, 2, 3],
        ),
        language_model=qwen,
    )
    assert planner.config.audio_codec == "moss_audio_tokenizer_nano"
    assert planner.config.audio_sample_rate == 48_000
    assert planner.config.audio_frame_size == 3_840
    assert planner.config.audio_cardinality == 1024
    assert planner.config.audio_codebooks_stored == 16
    assert planner.config.audio_codebooks_used == [0, 1, 2, 3]
    assert planner.config.mimi_cardinality == 1024
    assert planner.audio_embedding.num_embeddings == 1024
    assert all(
        embedding.num_embeddings == 1024
        for embedding in planner.additional_audio_embeddings
    )


def test_q0_q3_rejects_partial_audio_frames():
    planner = _tiny_planner()
    qwen = planner.language_model
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=qwen.config.to_dict(),
            audio_placeholder_id=planner.config.audio_placeholder_id,
            motion_token_ids=planner.motion_token_ids.tolist(),
            mimi_codebooks_used=[0, 1, 2, 3],
        ),
        language_model=qwen,
    )
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    input_ids[0, 1] = planner.config.audio_placeholder_id
    audio_codes = torch.full((1, 4, 4), -1, dtype=torch.long)
    audio_codes[0, 1, :2] = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="all configured"):
        planner.prepare_input_embeddings(input_ids, audio_codes)


def test_q0_q3_fusion_matches_bf16_language_embedding_dtype():
    base = _tiny_planner()
    qwen = base.language_model.to(dtype=torch.bfloat16)
    planner = MimiQwenPlanner(
        MimiQwenPlannerConfig(
            language_model_config=qwen.config.to_dict(),
            audio_placeholder_id=base.config.audio_placeholder_id,
            motion_token_ids=base.motion_token_ids.tolist(),
            mimi_codebooks_used=[0, 1, 2, 3],
        ),
        language_model=qwen,
    )
    input_ids = torch.zeros((1, 3), dtype=torch.long)
    input_ids[0, 1] = planner.config.audio_placeholder_id
    audio_codes = torch.full((1, 3, 4), -1, dtype=torch.long)
    audio_codes[0, 1] = torch.tensor([1, 2, 3, 4])
    embeddings = planner.prepare_input_embeddings(input_ids, audio_codes)
    assert embeddings.dtype == torch.bfloat16
    assert planner.audio_embedding.weight.dtype == torch.bfloat16
    assert planner.audio_fusion.weight.dtype == torch.bfloat16

from __future__ import annotations

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
    Step1FixedGapDataset,
    Step1PlannerCollator,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    GAP_TOKENS,
    MIMI_FRAME_TOKEN,
    body_global_id,
    body_token,
    causal_audio_boundaries,
    ensure_step1_special_tokens,
    fixed_anchor_times,
    gap_from_anchor_times,
    motion_token_id_table,
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


@pytest.fixture(scope="module")
def step1_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_DIR / "checkpoints" / "llm", local_files_only=True)
    ensure_step1_special_tokens(tokenizer)
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


def _tiny_planner() -> MimiQwenPlanner:
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
    )
    planner = MimiQwenPlanner(config, language_model=language_model)
    planner.tie_weights()
    return planner


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

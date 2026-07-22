from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Qwen2Config


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    MimiQwenPlanner,
    MimiQwenPlannerConfig,
    Step1PlannerCollator,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    body_global_id,
)
from utils.step1_planner_evaluation import (  # noqa: E402
    evaluate_reference_baselines,
    evaluate_rollouts,
    fit_unigram_prior,
    greedy_rollout_batch,
    greedy_rollout_item,
    Step1EvaluationCollator,
    summarize_slot_metrics,
)


def tiny_planner() -> MimiQwenPlanner:
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
        use_cache=True,
    )
    language_model = AutoModelForCausalLM.from_config(qwen_config)
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
    planner.eval()
    return planner


def test_summary_and_unigram_reference() -> None:
    labels = np.tile(np.arange(BODY_SLOT_COUNT), (12, 1)) % BODY_CODEBOOK_SIZE
    predictions = labels.copy()
    summary, rows = summarize_slot_metrics(labels, predictions)
    assert summary["accuracy"] == 1.0
    assert len(rows) == BODY_SLOT_COUNT
    prior = fit_unigram_prior(labels)
    assert prior.shape == (BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE)
    assert np.allclose(prior.sum(axis=1), 1.0)

    baseline_rows, _ = evaluate_reference_baselines(
        train_targets=labels,
        validation_targets=labels,
        validation_previous=labels,
    )
    uniform = next(row for row in baseline_rows if row["baseline"] == "uniform_reference")
    assert uniform["accuracy"] == 1.0 / BODY_CODEBOOK_SIZE
    copied = next(row for row in baseline_rows if row["baseline"] == "previous_gt_anchor_copy")
    assert copied["accuracy"] == 1.0


def test_cached_greedy_rollout_generates_all_slots() -> None:
    planner = tiny_planner()
    audio_placeholder = planner.config.audio_placeholder_id
    target = np.asarray([(slot * 13 + 5) % BODY_CODEBOOK_SIZE for slot in range(BODY_SLOT_COUNT)])
    target_token_ids = [
        int(planner.motion_token_ids[slot, target[slot]]) for slot in range(BODY_SLOT_COUNT)
    ]
    second_target = (target + 7) % BODY_CODEBOOK_SIZE
    second_target_token_ids = [
        int(planner.motion_token_ids[slot, second_target[slot]]) for slot in range(BODY_SLOT_COUNT)
    ]
    between = [3, audio_placeholder, 2]
    input_ids = [1, audio_placeholder, 2, *target_token_ids, *between, *second_target_token_ids]
    audio_codes = [
        -1, 17, -1, *([-1] * BODY_SLOT_COUNT), -1, 29, -1, *([-1] * BODY_SLOT_COUNT)
    ]
    target_slots = [
        -1, -1, -1, *range(BODY_SLOT_COUNT), -1, -1, -1, *range(BODY_SLOT_COUNT)
    ]
    labels = [
        -100, -100, -100, *target.tolist(), -100, -100, -100, *second_target.tolist()
    ]
    item = {
        "name": "synthetic/clip",
        "input_ids": input_ids,
        "audio_codes": audio_codes,
        "target_slots": target_slots,
        "motion_local_labels": labels,
        "anchor_times": (0, 4, 8),
    }
    result = greedy_rollout_item(
        planner,
        item,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    assert result.predicted_anchors.shape == (2, BODY_SLOT_COUNT)
    assert result.target_anchors.shape == (2, BODY_SLOT_COUNT)
    assert np.array_equal(result.target_anchors[0], target)
    assert np.array_equal(result.target_anchors[1], second_target)
    assert np.all((0 <= result.predicted_anchors) & (result.predicted_anchors < BODY_CODEBOOK_SIZE))

    # Compare cached decoding against full-prefix recomputation for both anchors.
    generated_prefix_ids = input_ids[:3]
    generated_prefix_audio = audio_codes[:3]
    recomputed = []
    for anchor_index in range(2):
        predicted = []
        for slot in range(BODY_SLOT_COUNT):
            ids = torch.tensor(generated_prefix_ids, dtype=torch.long).unsqueeze(0)
            audio = torch.tensor(generated_prefix_audio, dtype=torch.long).unsqueeze(0)
            logits = planner.next_slot_logits(
                ids, torch.ones_like(ids), audio, slot=slot
            )
            local_id = int(logits.argmax(dim=-1).item())
            predicted.append(local_id)
            generated_prefix_ids.append(int(planner.motion_token_ids[slot, local_id]))
            generated_prefix_audio.append(-1)
        recomputed.append(predicted)
        if anchor_index == 0:
            generated_prefix_ids.extend(between)
            generated_prefix_audio.extend([-1, 29, -1])
    assert np.array_equal(result.predicted_anchors, np.asarray(recomputed))

    summary = evaluate_rollouts([result])
    assert summary["summary"]["anchors"] == 2
    assert len(summary["slot_rows"]) == BODY_SLOT_COUNT
    assert result.cache_payload()["anchors"][0]["time"] == 4


def test_batched_rollout_matches_individual_rollout() -> None:
    planner = tiny_planner()
    audio_placeholder = planner.config.audio_placeholder_id

    def make_item(name: str, offset: int, anchors: int) -> dict[str, object]:
        input_ids = [1, audio_placeholder, 2]
        audio_codes = [-1, 11 + offset, -1]
        target_slots = [-1, -1, -1]
        labels = [-100, -100, -100]
        times = [0]
        for anchor_index in range(anchors):
            local_ids = [
                (offset + 19 * anchor_index + 7 * slot) % BODY_CODEBOOK_SIZE
                for slot in range(BODY_SLOT_COUNT)
            ]
            input_ids.extend(
                int(planner.motion_token_ids[slot, local_id])
                for slot, local_id in enumerate(local_ids)
            )
            audio_codes.extend([-1] * BODY_SLOT_COUNT)
            target_slots.extend(range(BODY_SLOT_COUNT))
            labels.extend(local_ids)
            times.append((anchor_index + 1) * 4)
            if anchor_index + 1 < anchors:
                input_ids.extend([3, audio_placeholder, 2])
                audio_codes.extend([-1, 21 + offset + anchor_index, -1])
                target_slots.extend([-1, -1, -1])
                labels.extend([-100, -100, -100])
        return {
            "name": name,
            "input_ids": input_ids,
            "audio_codes": [[value] for value in audio_codes],
            "target_slots": target_slots,
            "motion_local_labels": labels,
            "anchor_times": tuple(times),
            "generated_prefix_anchors": 0,
        }

    items = [make_item("clip/a", 3, 1), make_item("clip/b", 17, 2)]
    collator = Step1EvaluationCollator(
        Step1PlannerCollator(pad_token_id=0, pad_to_multiple_of=1)
    )
    batched = greedy_rollout_batch(
        planner,
        collator(items),
        device=torch.device("cpu"),
        use_bf16=False,
    )
    individual = [
        greedy_rollout_item(
            planner,
            item,
            device=torch.device("cpu"),
            use_bf16=False,
        )
        for item in items
    ]
    assert [result.name for result in batched] == [result.name for result in individual]
    for batch_result, item_result in zip(batched, individual):
        assert np.array_equal(batch_result.predicted_anchors, item_result.predicted_anchors)
        assert np.array_equal(batch_result.target_anchors, item_result.target_anchors)
        assert np.allclose(batch_result.confidence, item_result.confidence, atol=1e-6)

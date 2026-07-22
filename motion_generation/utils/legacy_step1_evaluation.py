"""Native evaluation helpers for the released SentiAvatar Step 1 planner.

The released planner predicts four legacy whole-body RVQ IDs per sparse frame
from an action label and HuBERT tokens. Those IDs are not interchangeable with
the newer 16 multipart IDs, so this module reports performance relative to
uniform and persistence references inside the legacy representation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from models.step1_mimi_planner import canonical_data_path


LEGACY_CODEBOOK_SIZE = 512
LEGACY_QUANTIZERS = 4
LEGACY_TOKEN_PATTERN = re.compile(r"\[res_([1-4])_(\d+)\]", re.IGNORECASE)
LEGACY_FRAME_PATTERN = re.compile(r"\[frame_(\d+)\]", re.IGNORECASE)


def load_legacy_tokens(path: Path) -> list[Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("tokens") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"Expected a token list in {path}")
    return values


def extract_legacy_action_text(text: str) -> str:
    """Match the released pipeline's last-tag action-label convention."""

    text = str(text)
    tags = re.findall(r"【(.+?)】", text)
    if not tags:
        return text
    action = tags[-1]
    if action == "动作：无动作":
        for tag in tags:
            if tag.startswith("表情：") and tag != "表情：无表情":
                expression = tag.removeprefix("表情：")
                return expression if "动作" in expression else f"动作：{expression}"
    return action


def _audio_value(value: Any) -> int:
    if isinstance(value, list):
        if not value:
            raise ValueError("Encountered an empty legacy audio-token frame")
        value = value[0]
    return int(value)


def build_legacy_prompt(
    action_text: str,
    audio_tokens: Sequence[Any],
    *,
    step: int = 4,
    offset: int = 0,
) -> tuple[str, tuple[int, ...]]:
    if step <= 0:
        raise ValueError("step must be positive")
    sampled = tuple(range(int(offset), len(audio_tokens), int(step)))
    audio = "".join(f"[audio_{_audio_value(audio_tokens[index])}]" for index in sampled)
    return f"Human: {action_text}{audio}<|im_end|>\nAssistant:", sampled


def parse_legacy_generated_plan(
    tokenizer, token_ids: Sequence[int]
) -> tuple[tuple[int, ...], np.ndarray]:
    """Parse complete frame-labelled legacy anchors.

    The released checkpoint emits ``[frame_3]``, ``[frame_7]``, ... followed
    by one ``[res_Q_ID]`` token for each of the four RVQ levels. Incomplete
    frames are rejected rather than silently pairing unrelated residual lists.
    """

    parsed: list[tuple[int, list[int]]] = []
    active_time: int | None = None
    active_codes: dict[int, int] = {}

    def flush() -> None:
        nonlocal active_time, active_codes
        if active_time is not None and set(active_codes) == set(range(LEGACY_QUANTIZERS)):
            parsed.append(
                (active_time, [active_codes[index] for index in range(LEGACY_QUANTIZERS)])
            )
        active_time = None
        active_codes = {}

    for token in tokenizer.convert_ids_to_tokens([int(value) for value in token_ids]):
        value = str(token)
        frame_match = LEGACY_FRAME_PATTERN.fullmatch(value)
        if frame_match is not None:
            flush()
            active_time = int(frame_match.group(1))
            continue
        residual_match = LEGACY_TOKEN_PATTERN.fullmatch(value)
        if residual_match is None or active_time is None:
            continue
        quantizer = int(residual_match.group(1)) - 1
        local_id = int(residual_match.group(2))
        if 0 <= local_id < LEGACY_CODEBOOK_SIZE and quantizer not in active_codes:
            active_codes[quantizer] = local_id
    flush()
    if not parsed:
        return (), np.empty((0, LEGACY_QUANTIZERS), dtype=np.int64)
    return (
        tuple(time for time, _ in parsed),
        np.asarray([codes for _, codes in parsed], dtype=np.int64),
    )


def parse_legacy_generated_ids(tokenizer, token_ids: Sequence[int]) -> np.ndarray:
    """Compatibility wrapper returning only complete frame-labelled anchors."""

    return parse_legacy_generated_plan(tokenizer, token_ids)[1]


@dataclass(frozen=True)
class LegacyStep1Example:
    name: str
    prompt: str
    sampled_times: tuple[int, ...]
    target_anchors: np.ndarray
    previous_anchors: np.ndarray


def build_legacy_example(
    *,
    name: str,
    text: str,
    motion_token_dir: Path,
    audio_token_dir: Path,
    step: int = 4,
) -> LegacyStep1Example:
    motion_path = canonical_data_path(Path(motion_token_dir), name, ".json")
    audio_path = canonical_data_path(Path(audio_token_dir), name, ".json")
    motion = np.asarray(load_legacy_tokens(motion_path), dtype=np.int64)
    if motion.ndim != 2 or motion.shape[1] != LEGACY_QUANTIZERS:
        raise ValueError(f"{name}: expected legacy motion tokens [T, 4], got {motion.shape}")
    if motion.size and (motion.min() < 0 or motion.max() >= LEGACY_CODEBOOK_SIZE):
        raise ValueError(f"{name}: legacy motion IDs must lie in [0, 511]")
    audio = load_legacy_tokens(audio_path)
    prompt, sampled = build_legacy_prompt(
        extract_legacy_action_text(text), audio, step=step
    )
    # The released checkpoint declares interval-end anchors: frame 3, 7, 11,
    # ... for step four. The audio prompt is sampled at 0, 4, 8, ... . Respect
    # those explicit frame labels instead of shifting them onto our newer
    # seed-at-zero schedule.
    valid_times = tuple(index + step - 1 for index in sampled if index + step - 1 < len(motion))
    if not valid_times:
        raise ValueError(f"{name}: no aligned legacy interval-end anchors")
    previous_times = (0, *valid_times[:-1])
    return LegacyStep1Example(
        name=str(name),
        prompt=prompt,
        sampled_times=valid_times,
        target_anchors=motion[np.asarray(valid_times, dtype=np.int64)],
        previous_anchors=motion[np.asarray(previous_times, dtype=np.int64)],
    )


@dataclass(frozen=True)
class LegacyGenerationResult:
    example: LegacyStep1Example
    predicted_times: tuple[int, ...]
    predicted_anchors: np.ndarray
    elapsed_seconds: float


def evaluate_legacy_generations(
    results: Sequence[LegacyGenerationResult],
) -> dict[str, Any]:
    """Score generated plans strictly, counting missing IDs as incorrect.

    Generated anchors are aligned by their explicit ``[frame_N]`` labels.
    Missing expected frames count as incorrect; extra or duplicate labels do
    not increase coverage.
    """

    if not results:
        raise ValueError("At least one legacy generation result is required")
    target_rows: list[np.ndarray] = []
    strict_rows: list[np.ndarray] = []
    previous_rows: list[np.ndarray] = []
    matched_correct: list[np.ndarray] = []
    clip_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    total_expected = 0
    total_covered = 0

    for result in results:
        target = np.asarray(result.example.target_anchors, dtype=np.int64)
        previous = np.asarray(result.example.previous_anchors, dtype=np.int64)
        predicted_all = np.asarray(result.predicted_anchors, dtype=np.int64)
        strict = np.full(target.shape, -1, dtype=np.int64)
        by_time: dict[int, np.ndarray] = {}
        for time_value, anchor in zip(result.predicted_times, predicted_all):
            by_time.setdefault(int(time_value), anchor)
        covered_indices = [
            index
            for index, time_value in enumerate(result.example.sampled_times)
            if int(time_value) in by_time
        ]
        for index in covered_indices:
            strict[index] = by_time[int(result.example.sampled_times[index])]
        covered = len(covered_indices)
        if covered:
            indices = np.asarray(covered_indices, dtype=np.int64)
            matched_correct.append(strict[indices] == target[indices])
        target_rows.append(target)
        strict_rows.append(strict)
        previous_rows.append(previous)
        total_expected += len(target)
        total_covered += covered
        clip_rows.append(
            {
                "name": result.example.name,
                "expected_total_anchors": int(len(result.example.target_anchors)),
                "predicted_total_anchors": int(len(predicted_all)),
                "evaluated_anchors": int(len(target)),
                "covered_anchors": int(covered),
                "coverage": float(covered / max(1, len(target))),
                "length_error": int(len(predicted_all) - len(result.example.target_anchors)),
                "exact_length": bool(len(predicted_all) == len(result.example.target_anchors)),
                "strict_accuracy": float((strict == target).mean()),
                "previous_copy_accuracy": float((previous == target).mean()),
                "elapsed_seconds": float(result.elapsed_seconds),
            }
        )
        denominator = max(1, len(target) - 1)
        for anchor_index in range(len(target)):
            horizon_rows.append(
                {
                    "name": result.example.name,
                    "anchor_index": anchor_index + 1,
                    "target_time": int(result.example.sampled_times[anchor_index]),
                    "relative_horizon": float(anchor_index / denominator),
                    "covered": bool(anchor_index in covered_indices),
                    "accuracy": float((strict[anchor_index] == target[anchor_index]).mean()),
                    "q0_accuracy": float(strict[anchor_index, 0] == target[anchor_index, 0]),
                }
            )

    targets = np.concatenate(target_rows, axis=0)
    strict_predictions = np.concatenate(strict_rows, axis=0)
    previous = np.concatenate(previous_rows, axis=0)
    correct = strict_predictions == targets
    copy_correct = previous == targets
    matched = (
        np.concatenate(matched_correct, axis=0)
        if matched_correct
        else np.empty((0, LEGACY_QUANTIZERS), dtype=bool)
    )
    summary: dict[str, Any] = {
        "system": "released_sentiavatar_step1",
        "representation": "legacy_whole_body_rvq_4x512",
        "clips": len(results),
        "anchors": int(len(targets)),
        "tokens": int(targets.size),
        "accuracy": float(correct.mean()),
        "matched_only_accuracy": float(matched.mean()) if matched.size else 0.0,
        "coverage": float(total_covered / max(1, total_expected)),
        "previous_copy_accuracy": float(copy_correct.mean()),
        "accuracy_margin_over_previous_copy": float(correct.mean() - copy_correct.mean()),
        "uniform_accuracy": 1.0 / LEGACY_CODEBOOK_SIZE,
        "exact_length_rate": float(np.mean([row["exact_length"] for row in clip_rows])),
        "mean_absolute_length_error": float(
            np.mean([abs(row["length_error"]) for row in clip_rows])
        ),
        "elapsed_seconds": float(sum(result.elapsed_seconds for result in results)),
    }
    for quantizer in range(LEGACY_QUANTIZERS):
        summary[f"q{quantizer}_accuracy"] = float(correct[:, quantizer].mean())
        summary[f"previous_copy_q{quantizer}_accuracy"] = float(
            copy_correct[:, quantizer].mean()
        )
    summary["persistence_ratio"] = float(
        summary["accuracy"] / max(1e-12, summary["previous_copy_accuracy"])
    )
    summary["normalized_accuracy_above_uniform"] = float(
        (summary["accuracy"] - summary["uniform_accuracy"])
        / (1.0 - summary["uniform_accuracy"])
    )
    return {
        "summary": summary,
        "clip_rows": clip_rows,
        "horizon_rows": horizon_rows,
        "targets": targets,
        "predictions": strict_predictions,
    }

"""Evaluation helpers for learned-gap causal Step 1 planners.

The fixed-gap evaluator serializes the complete schedule before generation and
therefore cannot evaluate a planner whose next anchor time depends on its own
generated history.  This module implements the missing closed-loop controller.

Evaluation is offline-known-duration: the validation motion-token length is
used only to stop at the exact final frame and to clip a final overshooting
decision.  Every such clip is reported.  Normal decisions are restricted to
gaps 3--15; an executed gap 0--2 is legal only at the final frame.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.step1_mimi_planner import MimiQwenPlanner
from utils.adaptive_anchor_tokens import (
    ANCHOR_TOKEN,
    BODY_SLOT_COUNT,
    GAP_TOKENS,
    MIMI_FRAME_TOKEN,
    body_token,
    fixed_anchor_times,
    gap_from_anchor_times,
)


NORMAL_GAPS = tuple(range(3, 16))


@dataclass(frozen=True)
class AdaptiveRolloutExample:
    """One clip prepared for closed-loop Step 1 evaluation."""

    name: str
    initial_input_ids: tuple[int, ...]
    audio_codes: np.ndarray
    dense_motion_tokens: np.ndarray
    oracle_anchor_times: tuple[int, ...]
    audio_fps: float = 12.5
    motion_fps: float = 10.0

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio_codes)
        dense = np.asarray(self.dense_motion_tokens)
        if not self.initial_input_ids:
            raise ValueError(f"{self.name}: initial prefix is empty")
        if audio.ndim != 2 or audio.shape[1] < 1:
            raise ValueError(f"{self.name}: audio_codes must be [A,C]")
        if dense.ndim != 2 or dense.shape[1] != BODY_SLOT_COUNT:
            raise ValueError(
                f"{self.name}: dense_motion_tokens must be [T,{BODY_SLOT_COUNT}]"
            )
        if len(dense) < 1:
            raise ValueError(f"{self.name}: dense motion is empty")
        if np.any((dense < 0) | (dense >= 512)):
            raise ValueError(f"{self.name}: motion token is outside [0,511]")
        if self.oracle_anchor_times:
            if self.oracle_anchor_times[0] != 0:
                raise ValueError(f"{self.name}: oracle schedule must begin at zero")
            if self.oracle_anchor_times[-1] != len(dense) - 1:
                raise ValueError(f"{self.name}: oracle schedule must end at T-1")


@dataclass
class AdaptiveRolloutResult:
    """One generated or controlled sparse anchor plan."""

    name: str
    policy: str
    anchor_history: str
    token_length: int
    anchor_times: tuple[int, ...]
    anchors: np.ndarray
    target_anchors: np.ndarray
    predicted_gap_decisions: tuple[int, ...]
    executed_gaps: tuple[int, ...]
    gap_confidence: np.ndarray
    gap_entropy: np.ndarray
    anchor_confidence: np.ndarray
    anchor_entropy: np.ndarray
    eos_clipped_decisions: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        count = len(self.anchor_times) - 1
        if len(self.anchor_times) < 1 or self.anchor_times[0] != 0:
            raise ValueError(f"{self.name}: anchor times must start at zero")
        if self.anchor_times[-1] != self.token_length - 1:
            raise ValueError(f"{self.name}: anchor times must end at token_length-1")
        if len(self.executed_gaps) != count:
            raise ValueError(f"{self.name}: gap and anchor counts differ")
        expected = (count, BODY_SLOT_COUNT)
        if np.asarray(self.anchors).shape != expected:
            raise ValueError(f"{self.name}: anchors must have shape {expected}")
        if np.asarray(self.target_anchors).shape != expected:
            raise ValueError(f"{self.name}: target anchors must have shape {expected}")

    @property
    def normal_gaps(self) -> tuple[int, ...]:
        return tuple(int(gap) for gap in self.executed_gaps if int(gap) >= 3)

    @property
    def tail_gap(self) -> Optional[int]:
        if self.executed_gaps and self.executed_gaps[-1] <= 2:
            return int(self.executed_gaps[-1])
        return None

    def cache_payload(self) -> dict[str, Any]:
        return {
            "schema": "sentiavatar.step1_adaptive_rollout.v1",
            "name": self.name,
            "policy": self.policy,
            "anchor_history": self.anchor_history,
            "duration_contract": "offline_known_motion_token_length",
            "token_length": self.token_length,
            "eos_clipped_decisions": self.eos_clipped_decisions,
            "predicted_gap_decisions": list(self.predicted_gap_decisions),
            "executed_gaps": list(self.executed_gaps),
            "anchors": [
                {
                    "time": int(anchor_time),
                    "tokens": [int(value) for value in anchor],
                }
                for anchor_time, anchor in zip(self.anchor_times[1:], self.anchors)
            ],
        }


def initial_prefix_from_serialized_item(
    item: Mapping[str, Any],
    *,
    motion_start_id: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Recover text/seed prefix and chronological raw audio from a dataset item."""

    input_ids = np.asarray(item["input_ids"], dtype=np.int64)
    matches = np.flatnonzero(input_ids == int(motion_start_id))
    if len(matches) != 1:
        raise ValueError(
            f"{item.get('name', '<unknown>')}: expected one [motion_start], got {len(matches)}"
        )
    # [motion_start], seed-mode token, [anchor], then 16 seed body tokens.
    end = int(matches[0]) + 3 + BODY_SLOT_COUNT
    if end > len(input_ids):
        raise ValueError(f"{item.get('name', '<unknown>')}: truncated seed prefix")
    raw_audio = np.asarray(item["audio_codes"], dtype=np.int64)
    if raw_audio.ndim == 1:
        raw_audio = raw_audio[:, None]
    valid = np.all(raw_audio >= 0, axis=1)
    return tuple(int(value) for value in input_ids[:end]), raw_audio[valid].copy()


def _position_ids(attention_mask: torch.Tensor, width: Optional[int] = None) -> torch.Tensor:
    values = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
    return values if width is None else values[:, -int(width) :]


def _selected_logits(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    allowed_ids: torch.Tensor,
) -> torch.Tensor:
    if allowed_ids.ndim == 1:
        return F.linear(hidden, output_weight.index_select(0, allowed_ids)).float()
    classifier_weight = output_weight[allowed_ids]
    return torch.einsum("nh,nvh->nv", hidden, classifier_weight).float()


def _confidence_entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(dim=-1)
    selected = probabilities.max(dim=-1).values
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    return selected, entropy


def _audio_boundary(
    target_time: int,
    *,
    final_time: int,
    audio_frames: int,
    audio_fps: float,
    motion_fps: float,
) -> int:
    if target_time >= final_time:
        return int(audio_frames)
    boundary = int(math.ceil(int(target_time) * float(audio_fps) / float(motion_fps)))
    return min(int(audio_frames), max(0, boundary))


def controlled_anchor_times(
    token_length: int,
    *,
    policy: str,
    fixed_gap: Optional[int] = None,
    oracle_anchor_times: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Return a non-learned control schedule."""

    if policy == "fixed":
        if fixed_gap is None:
            raise ValueError("fixed policy requires fixed_gap")
        return fixed_anchor_times(token_length, gap=int(fixed_gap))
    if policy == "oracle":
        values = tuple(int(value) for value in (oracle_anchor_times or ()))
        if not values or values[0] != 0 or values[-1] != int(token_length) - 1:
            raise ValueError("oracle schedule endpoints do not match token length")
        return values
    raise ValueError(f"controlled schedule does not support policy={policy!r}")


def make_ground_truth_result(
    example: AdaptiveRolloutExample,
    *,
    policy: str,
    fixed_gap: Optional[int] = None,
) -> AdaptiveRolloutResult:
    """Create a placement-only condition without running the planner."""

    times = controlled_anchor_times(
        len(example.dense_motion_tokens),
        policy=policy,
        fixed_gap=fixed_gap,
        oracle_anchor_times=example.oracle_anchor_times,
    )
    gaps = tuple(
        gap_from_anchor_times(left, right)
        for left, right in zip(times[:-1], times[1:])
    )
    targets = np.asarray(example.dense_motion_tokens)[list(times[1:])].astype(
        np.int64, copy=True
    )
    count = len(gaps)
    return AdaptiveRolloutResult(
        name=example.name,
        policy=(
            f"fixed_gap_{int(fixed_gap)}" if policy == "fixed" else "step2_dp_oracle"
        ),
        anchor_history="ground_truth",
        token_length=len(example.dense_motion_tokens),
        anchor_times=times,
        anchors=targets.copy(),
        target_anchors=targets,
        predicted_gap_decisions=(),
        executed_gaps=gaps,
        gap_confidence=np.full(count, np.nan, dtype=np.float64),
        gap_entropy=np.full(count, np.nan, dtype=np.float64),
        anchor_confidence=np.full((count, BODY_SLOT_COUNT), np.nan, dtype=np.float64),
        anchor_entropy=np.full((count, BODY_SLOT_COUNT), np.nan, dtype=np.float64),
        eos_clipped_decisions=0,
        elapsed_seconds=0.0,
    )


@torch.inference_mode()
def rollout_policy_batch(
    model: MimiQwenPlanner,
    tokenizer: Any,
    examples: Sequence[AdaptiveRolloutExample],
    *,
    policy: str,
    anchor_history: str,
    device: torch.device,
    use_bf16: bool,
    fixed_gap: Optional[int] = None,
) -> list[AdaptiveRolloutResult]:
    """Run a learned/fixed/oracle schedule with GT or generated anchor history.

    Padding is allowed between dynamic chunks.  It is masked in attention and
    explicit cumulative ``position_ids`` ensure that real tokens retain their
    causal positions.  This permits variable interval audio lengths in one KV
    cache batch without recomputing complete prefixes.
    """

    if not examples:
        return []
    if policy not in {"adaptive", "fixed", "oracle"}:
        raise ValueError("policy must be adaptive, fixed, or oracle")
    if anchor_history not in {"ground_truth", "generated"}:
        raise ValueError("anchor_history must be ground_truth or generated")
    if policy == "fixed" and fixed_gap not in NORMAL_GAPS:
        raise ValueError("fixed_gap must lie in [3,15]")

    model.eval()
    batch_size = len(examples)
    audio_streams = [
        np.asarray(example.audio_codes, dtype=np.int64) for example in examples
    ]
    dense_streams = [
        np.asarray(example.dense_motion_tokens, dtype=np.int64)
        for example in examples
    ]
    final_times = torch.tensor(
        [len(values) - 1 for values in dense_streams],
        dtype=torch.long,
        device=device,
    )
    current_times = torch.zeros(batch_size, dtype=torch.long, device=device)
    audio_cursors = [0] * batch_size
    schedule_indices = [0] * batch_size
    anchor_times: list[list[int]] = [[0] for _ in examples]
    anchors: list[list[list[int]]] = [[] for _ in examples]
    targets: list[list[list[int]]] = [[] for _ in examples]
    predicted_gaps: list[list[int]] = [[] for _ in examples]
    executed_gaps: list[list[int]] = [[] for _ in examples]
    gap_confidence: list[list[float]] = [[] for _ in examples]
    gap_entropy: list[list[float]] = [[] for _ in examples]
    anchor_confidence: list[list[list[float]]] = [[] for _ in examples]
    anchor_entropy: list[list[list[float]]] = [[] for _ in examples]
    eos_clipped = [0] * batch_size

    pad_id = int(tokenizer.pad_token_id)
    audio_channels = int(np.asarray(examples[0].audio_codes).shape[1])
    if any(stream.shape[1] != audio_channels for stream in audio_streams):
        raise ValueError("All rollout examples must use the same audio codebooks")
    prefix_width = max(len(example.initial_input_ids) for example in examples)
    input_ids = torch.full(
        (batch_size, prefix_width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    audio_codes = torch.full(
        (batch_size, prefix_width, audio_channels),
        -1,
        dtype=torch.long,
        device=device,
    )
    prefix_lengths = []
    for row, example in enumerate(examples):
        length = len(example.initial_input_ids)
        prefix_lengths.append(length)
        input_ids[row, :length] = torch.tensor(
            example.initial_input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1

    autocast_enabled = bool(use_bf16 and device.type == "cuda")
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        initial = model._base_model_forward(  # pylint: disable=protected-access
            inputs_embeds=model.prepare_input_embeddings(input_ids, audio_codes),
            attention_mask=attention_mask,
            position_ids=_position_ids(attention_mask),
            use_cache=True,
            return_dict=True,
        )
    row_ids = torch.arange(batch_size, device=device)
    last_positions = torch.tensor(prefix_lengths, device=device) - 1
    hidden = initial.last_hidden_state[row_ids, last_positions]
    past_key_values: Any = initial.past_key_values
    full_attention = attention_mask
    output_weight = model.language_model.get_output_embeddings().weight
    gap_ids = model.gap_token_ids.index_select(
        0, torch.tensor(NORMAL_GAPS, dtype=torch.long, device=device)
    )
    gap_token_ids = [
        int(tokenizer.convert_tokens_to_ids(token)) for token in GAP_TOKENS
    ]
    anchor_token_id = int(tokenizer.convert_tokens_to_ids(ANCHOR_TOKEN))
    audio_token_id = int(tokenizer.convert_tokens_to_ids(MIMI_FRAME_TOKEN))

    def cached_forward(
        chunk_ids: torch.Tensor,
        chunk_audio: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal past_key_values, full_attention
        full_attention = torch.cat([full_attention, chunk_mask], dim=1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model._base_model_forward(  # pylint: disable=protected-access
                inputs_embeds=model.prepare_input_embeddings(chunk_ids, chunk_audio),
                attention_mask=full_attention,
                position_ids=_position_ids(full_attention, chunk_ids.shape[1]),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = output.past_key_values
        return output.last_hidden_state

    started = time.perf_counter()
    while bool(current_times.lt(final_times).any()):
        active = current_times.lt(final_times)
        active_rows = active.nonzero(as_tuple=False).squeeze(-1)
        proposed = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device
        )
        round_gap_conf = torch.full(
            (batch_size,), float("nan"), dtype=torch.float32, device=device
        )
        round_gap_entropy = torch.full_like(round_gap_conf, float("nan"))
        if policy == "adaptive":
            logits = _selected_logits(
                hidden.index_select(0, active_rows),
                output_weight,
                gap_ids,
            )
            local = logits.argmax(dim=-1)
            proposed[active_rows] = local + NORMAL_GAPS[0]
            confidence, entropy = _confidence_entropy(logits)
            round_gap_conf[active_rows] = confidence
            round_gap_entropy[active_rows] = entropy
        elif policy == "fixed":
            proposed[active_rows] = int(fixed_gap)
        else:
            for row in active_rows.tolist():
                schedule = examples[row].oracle_anchor_times
                index = schedule_indices[row]
                if schedule[index] != int(current_times[row]):
                    raise ValueError(
                        f"{examples[row].name}: oracle cursor left its schedule"
                    )
                next_time = schedule[index + 1]
                proposed[row] = int(next_time - schedule[index] - 1)

        target_times = current_times.clone()
        for row in active_rows.tolist():
            raw_gap = int(proposed[row])
            if policy == "oracle":
                target_time = examples[row].oracle_anchor_times[
                    schedule_indices[row] + 1
                ]
            else:
                raw_target = int(current_times[row]) + raw_gap + 1
                target_time = min(raw_target, int(final_times[row]))
                if raw_target > int(final_times[row]):
                    eos_clipped[row] += 1
            executed = target_time - int(current_times[row]) - 1
            if not 0 <= executed <= 15:
                raise ValueError(
                    f"{examples[row].name}: executed illegal gap {executed}"
                )
            if executed <= 2 and target_time != int(final_times[row]):
                raise ValueError(
                    f"{examples[row].name}: tail gap {executed} away from EOS"
                )
            target_times[row] = target_time
            if policy == "adaptive":
                predicted_gaps[row].append(raw_gap)
            executed_gaps[row].append(executed)
            gap_confidence[row].append(float(round_gap_conf[row]))
            gap_entropy[row].append(float(round_gap_entropy[row]))

        next_boundaries = list(audio_cursors)
        audio_counts = [0] * batch_size
        for row in active_rows.tolist():
            boundary = _audio_boundary(
                int(target_times[row]),
                final_time=int(final_times[row]),
                audio_frames=len(audio_streams[row]),
                audio_fps=examples[row].audio_fps,
                motion_fps=examples[row].motion_fps,
            )
            if boundary < audio_cursors[row]:
                raise ValueError(
                    f"{examples[row].name}: audio boundary moved backwards"
                )
            next_boundaries[row] = boundary
            audio_counts[row] = boundary - audio_cursors[row]
        max_audio = max(audio_counts)
        known_width = max_audio + 2
        known_ids = torch.full(
            (batch_size, known_width),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        known_mask = torch.zeros_like(known_ids)
        known_audio = torch.full(
            (batch_size, known_width, audio_channels),
            -1,
            dtype=torch.long,
            device=device,
        )
        for row in active_rows.tolist():
            known_ids[row, 0] = gap_token_ids[executed_gaps[row][-1]]
            known_mask[row, 0] = 1
            count = audio_counts[row]
            if count:
                known_ids[row, 1 : 1 + count] = audio_token_id
                known_audio[row, 1 : 1 + count] = torch.as_tensor(
                    audio_streams[row][audio_cursors[row] : next_boundaries[row]],
                    dtype=torch.long,
                    device=device,
                )
                known_mask[row, 1 : 1 + count] = 1
            # Common physical column; padding before it is attention-masked.
            known_ids[row, -1] = anchor_token_id
            known_mask[row, -1] = 1
        hidden = cached_forward(known_ids, known_audio, known_mask)[:, -1]

        round_anchors = np.zeros(
            (batch_size, BODY_SLOT_COUNT), dtype=np.int64
        )
        round_confidence = np.full(
            (batch_size, BODY_SLOT_COUNT), np.nan, dtype=np.float64
        )
        round_entropy = np.full_like(round_confidence, np.nan)
        if anchor_history == "generated":
            for slot in range(BODY_SLOT_COUNT):
                allowed = model.motion_token_ids[slot]
                logits = _selected_logits(
                    hidden.index_select(0, active_rows),
                    output_weight,
                    allowed,
                )
                local = logits.argmax(dim=-1)
                confidence, entropy = _confidence_entropy(logits)
                token_ids = allowed.index_select(0, local)
                append_ids = torch.full(
                    (batch_size, 1),
                    pad_id,
                    dtype=torch.long,
                    device=device,
                )
                append_mask = torch.zeros_like(append_ids)
                append_audio = torch.full(
                    (batch_size, 1, audio_channels),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                append_ids[active_rows, 0] = token_ids
                append_mask[active_rows, 0] = 1
                hidden = cached_forward(
                    append_ids, append_audio, append_mask
                )[:, -1]
                for index, row in enumerate(active_rows.tolist()):
                    round_anchors[row, slot] = int(local[index])
                    round_confidence[row, slot] = float(confidence[index])
                    round_entropy[row, slot] = float(entropy[index])
        else:
            gt_ids = torch.full(
                (batch_size, BODY_SLOT_COUNT),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            gt_mask = torch.zeros_like(gt_ids)
            gt_audio = torch.full(
                (batch_size, BODY_SLOT_COUNT, audio_channels),
                -1,
                dtype=torch.long,
                device=device,
            )
            for row in active_rows.tolist():
                target = dense_streams[row][int(target_times[row])]
                round_anchors[row] = target
                gt_ids[row] = torch.tensor(
                    [
                        int(tokenizer.convert_tokens_to_ids(body_token(slot, int(value))))
                        for slot, value in enumerate(target)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                gt_mask[row] = 1
            hidden = cached_forward(gt_ids, gt_audio, gt_mask)[:, -1]

        for row in active_rows.tolist():
            target_time = int(target_times[row])
            target = dense_streams[row][target_time]
            anchor_times[row].append(target_time)
            anchors[row].append(round_anchors[row].tolist())
            targets[row].append(target.astype(np.int64).tolist())
            anchor_confidence[row].append(round_confidence[row].tolist())
            anchor_entropy[row].append(round_entropy[row].tolist())
            current_times[row] = target_time
            audio_cursors[row] = next_boundaries[row]
            if policy == "oracle":
                schedule_indices[row] += 1

    elapsed = (time.perf_counter() - started) / batch_size
    del past_key_values
    results = []
    policy_name = (
        f"fixed_gap_{int(fixed_gap)}" if policy == "fixed" else policy
    )
    for row, example in enumerate(examples):
        if audio_cursors[row] != len(audio_streams[row]):
            raise ValueError(
                f"{example.name}: consumed {audio_cursors[row]}/"
                f"{len(audio_streams[row])} audio frames"
            )
        count = len(anchors[row])
        results.append(
            AdaptiveRolloutResult(
                name=example.name,
                policy=policy_name,
                anchor_history=anchor_history,
                token_length=len(dense_streams[row]),
                anchor_times=tuple(anchor_times[row]),
                anchors=np.asarray(anchors[row], dtype=np.int64).reshape(
                    count, BODY_SLOT_COUNT
                ),
                target_anchors=np.asarray(targets[row], dtype=np.int64).reshape(
                    count, BODY_SLOT_COUNT
                ),
                predicted_gap_decisions=tuple(predicted_gaps[row]),
                executed_gaps=tuple(executed_gaps[row]),
                gap_confidence=np.asarray(gap_confidence[row], dtype=np.float64),
                gap_entropy=np.asarray(gap_entropy[row], dtype=np.float64),
                anchor_confidence=np.asarray(
                    anchor_confidence[row], dtype=np.float64
                ).reshape(count, BODY_SLOT_COUNT),
                anchor_entropy=np.asarray(
                    anchor_entropy[row], dtype=np.float64
                ).reshape(count, BODY_SLOT_COUNT),
                eos_clipped_decisions=eos_clipped[row],
                elapsed_seconds=elapsed,
            )
        )
    return results


def write_adaptive_rollout_cache(
    results: Sequence[AdaptiveRolloutResult],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    for result in results:
        parts = PurePosixPath(result.name.replace("\\", "/")).parts
        path = output_dir / Path(*parts).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result.cache_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)


def load_adaptive_rollout_cache(
    path: Path,
    *,
    dense_motion_tokens: np.ndarray,
) -> AdaptiveRolloutResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "sentiavatar.step1_adaptive_rollout.v1":
        raise ValueError(f"Unsupported adaptive rollout cache: {path}")
    dense = np.asarray(dense_motion_tokens, dtype=np.int64)
    records = payload.get("anchors", [])
    times = (0, *(int(value["time"]) for value in records))
    anchors = np.asarray(
        [value["tokens"] for value in records], dtype=np.int64
    ).reshape(len(records), BODY_SLOT_COUNT)
    targets = dense[list(times[1:])]
    gaps = tuple(
        gap_from_anchor_times(left, right)
        for left, right in zip(times[:-1], times[1:])
    )
    if tuple(int(value) for value in payload.get("executed_gaps", [])) != gaps:
        raise ValueError(f"{path}: cached gaps do not match anchor times")
    count = len(records)
    return AdaptiveRolloutResult(
        name=str(payload["name"]),
        policy=str(payload["policy"]),
        anchor_history=str(payload["anchor_history"]),
        token_length=int(payload["token_length"]),
        anchor_times=times,
        anchors=anchors,
        target_anchors=targets,
        predicted_gap_decisions=tuple(
            int(value) for value in payload.get("predicted_gap_decisions", [])
        ),
        executed_gaps=gaps,
        gap_confidence=np.full(count, np.nan),
        gap_entropy=np.full(count, np.nan),
        anchor_confidence=np.full((count, BODY_SLOT_COUNT), np.nan),
        anchor_entropy=np.full((count, BODY_SLOT_COUNT), np.nan),
        eos_clipped_decisions=int(payload.get("eos_clipped_decisions", 0)),
        elapsed_seconds=0.0,
    )

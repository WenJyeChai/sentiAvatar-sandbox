"""On-policy generated-history rollout for the causal Step 1 planner.

The rollout is intentionally inference-only: generated discrete motion IDs are
detached, fed back through an append-only Qwen KV cache, and later used as the
input history for a separate gradient-bearing CE pass.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from models.step1_mimi_planner import IGNORE_INDEX, MimiQwenPlanner
from utils.adaptive_anchor_tokens import BODY_SLOT_COUNT


@dataclass
class GeneratedHistoryResult:
    input_ids: torch.Tensor
    predicted_local_ids: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    generated_tokens: int
    generated_anchors: int


@dataclass
class GeneratedHistoryBatchStats:
    clips: int = 0
    anchors: int = 0
    tokens: int = 0
    correct: int = 0
    q0_correct: int = 0
    q0_tokens: int = 0
    confidence_sum: float = 0.0
    entropy_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.tokens)

    @property
    def q0_accuracy(self) -> float:
        return self.q0_correct / max(1, self.q0_tokens)

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / max(1, self.tokens)

    @property
    def mean_entropy(self) -> float:
        return self.entropy_sum / max(1, self.tokens)

    def update(self, other: "GeneratedHistoryBatchStats") -> None:
        for name in (
            "clips",
            "anchors",
            "tokens",
            "correct",
            "q0_correct",
            "q0_tokens",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.confidence_sum += other.confidence_sum
        self.entropy_sum += other.entropy_sum


def generated_history_probability(
    epoch_progress: float,
    *,
    activation_epoch: int | None,
    ramp_epochs: int,
    max_probability: float,
) -> float:
    """Cosine ramp from zero at activation to the configured maximum."""

    if activation_epoch is None or epoch_progress <= activation_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_probability)
    progress = min(1.0, max(0.0, (epoch_progress - activation_epoch) / ramp_epochs))
    return float(max_probability) * 0.5 * (1.0 - math.cos(math.pi * progress))


def deterministic_generated_indices(
    names: Sequence[str],
    probability: float,
    *,
    seed: int,
    epoch: int,
    batch_index: int,
) -> list[int]:
    """Select an exact per-rank clip count while remaining resume deterministic."""

    count = min(len(names), max(0, round(len(names) * float(probability))))
    if count == 0:
        return []
    scored = []
    for index, name in enumerate(names):
        key = f"{seed}|{epoch}|{batch_index}|{name}".encode("utf-8")
        score = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        scored.append((score, index))
    return sorted(index for _, index in sorted(scored)[:count])


def _validate_rollout_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_codes: torch.Tensor,
    target_slots: torch.Tensor,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, L]")
    if not (input_ids.shape == attention_mask.shape == target_slots.shape):
        raise ValueError("input_ids, attention_mask, and target_slots must share [B, L]")
    if tuple(audio_codes.shape[:2]) != tuple(input_ids.shape) or audio_codes.ndim != 3:
        raise ValueError("audio_codes must be [B, L, C]")
    target_mask = target_slots.ge(0)
    if bool((target_mask & attention_mask.eq(0)).any()):
        raise ValueError("A padded position cannot be a rollout target")
    if bool(target_mask[:, 0].any()):
        raise ValueError("The first sequence position cannot be a rollout target")
    for row in range(input_ids.shape[0]):
        slots = target_slots[row][target_mask[row]]
        if slots.numel() % BODY_SLOT_COUNT:
            raise ValueError("Rollout targets must form complete 16-slot anchors")
        expected = torch.arange(BODY_SLOT_COUNT, device=slots.device).repeat(
            slots.numel() // BODY_SLOT_COUNT
        )
        if not torch.equal(slots, expected):
            raise ValueError("Rollout target slots must repeat in 0..15 order")


@torch.inference_mode()
def generate_history_batch(
    model: MimiQwenPlanner,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_codes: torch.Tensor,
    target_slots: torch.Tensor,
    use_bf16: bool,
) -> GeneratedHistoryResult:
    """Replace every supervised anchor token with an autoregressive prediction.

    Rows may have different target positions. They advance through one common
    padded token stream; known spans are prefetched as chunks and every target
    position is predicted before that position is committed to the KV cache.
    """

    _validate_rollout_tensors(input_ids, attention_mask, audio_codes, target_slots)
    generated_ids = input_ids.clone()
    predicted = torch.full_like(target_slots, -1)
    confidence = torch.zeros_like(target_slots, dtype=torch.float32)
    entropy = torch.zeros_like(target_slots, dtype=torch.float32)
    target_mask = target_slots.ge(0)
    target_positions = target_mask.any(dim=0).nonzero(as_tuple=False).squeeze(-1).tolist()
    if not target_positions:
        raise ValueError("A generated-history batch must contain rollout targets")

    past_key_values: Any = None
    hidden: torch.Tensor | None = None
    cursor = 0
    output_weight = model.language_model.get_output_embeddings().weight
    autocast_enabled = bool(use_bf16 and input_ids.device.type == "cuda")

    def cached_forward(left: int, right: int) -> torch.Tensor:
        nonlocal past_key_values
        embeddings = model.prepare_input_embeddings(
            generated_ids[:, left:right], audio_codes[:, left:right]
        )
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            outputs = model._base_model_forward(  # pylint: disable=protected-access
                inputs_embeds=embeddings,
                attention_mask=attention_mask[:, :right],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        return outputs.last_hidden_state[:, -1]

    for position in target_positions:
        if cursor < position:
            hidden = cached_forward(cursor, position)
        if hidden is None:
            raise AssertionError("A target was reached without a preceding hidden state")

        rows = target_mask[:, position].nonzero(as_tuple=False).squeeze(-1)
        slots = target_slots[rows, position]
        allowed_ids = model.motion_token_ids.index_select(0, slots)
        classifier_weight = output_weight[allowed_ids]
        logits = torch.einsum("nh,nvh->nv", hidden.index_select(0, rows), classifier_weight).float()
        probabilities = logits.softmax(dim=-1)
        local_ids = logits.argmax(dim=-1)
        token_ids = allowed_ids.gather(1, local_ids[:, None]).squeeze(1)
        generated_ids[rows, position] = token_ids
        predicted[rows, position] = local_ids
        selected_probability = probabilities.gather(1, local_ids[:, None]).squeeze(1)
        confidence[rows, position] = selected_probability
        entropy[rows, position] = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)

        hidden = cached_forward(position, position + 1)
        cursor = position + 1

    generated_tokens = int(target_mask.sum().item())
    del past_key_values
    return GeneratedHistoryResult(
        input_ids=generated_ids,
        predicted_local_ids=predicted,
        confidence=confidence,
        entropy=entropy,
        generated_tokens=generated_tokens,
        generated_anchors=generated_tokens // BODY_SLOT_COUNT,
    )


def apply_generated_history(
    model: MimiQwenPlanner,
    batch: Mapping[str, torch.Tensor],
    selected_indices: Sequence[int],
    *,
    microbatch_size: int,
    use_bf16: bool,
) -> tuple[torch.Tensor, GeneratedHistoryBatchStats]:
    """Return a normal cloned input tensor with selected rows fully self-forced.

    ``generate_history_batch`` owns the inference-mode boundary for the model
    rollout.  This wrapper must stay outside that boundary because its returned
    token IDs are consumed by the subsequent gradient-enabled training forward.
    """

    generated_input_ids = batch["input_ids"].clone()
    stats = GeneratedHistoryBatchStats()
    if not selected_indices:
        return generated_input_ids, stats
    if microbatch_size <= 0:
        raise ValueError("rollout microbatch_size must be positive")

    for start in range(0, len(selected_indices), microbatch_size):
        indices = torch.as_tensor(
            selected_indices[start : start + microbatch_size],
            dtype=torch.long,
            device=batch["input_ids"].device,
        )
        result = generate_history_batch(
            model,
            input_ids=batch["input_ids"].index_select(0, indices),
            attention_mask=batch["attention_mask"].index_select(0, indices),
            audio_codes=batch["audio_codes"].index_select(0, indices),
            target_slots=batch["target_slots"].index_select(0, indices),
            use_bf16=use_bf16,
        )
        generated_input_ids.index_copy_(0, indices, result.input_ids)
        labels = batch["motion_local_labels"].index_select(0, indices)
        target_mask = result.predicted_local_ids.ge(0)
        q0_mask = target_mask & batch["target_slots"].index_select(0, indices).remainder(4).eq(0)
        stats.update(
            GeneratedHistoryBatchStats(
                clips=len(indices),
                anchors=result.generated_anchors,
                tokens=result.generated_tokens,
                correct=int(
                    result.predicted_local_ids[target_mask].eq(labels[target_mask]).sum().item()
                ),
                q0_correct=int(
                    result.predicted_local_ids[q0_mask].eq(labels[q0_mask]).sum().item()
                ),
                q0_tokens=int(q0_mask.sum().item()),
                confidence_sum=float(result.confidence[target_mask].sum().item()),
                entropy_sum=float(result.entropy[target_mask].sum().item()),
            )
        )
    return generated_input_ids, stats


@torch.inference_mode()
def rollout_quality_metrics(
    model: MimiQwenPlanner,
    loader,
    *,
    device: torch.device,
    use_bf16: bool,
) -> GeneratedHistoryBatchStats:
    """Evaluate fully generated histories on a small deterministic loader."""

    totals = GeneratedHistoryBatchStats()
    for batch in loader:
        tensor_batch = {
            key: value.to(device=device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
            and key
            in {"input_ids", "attention_mask", "audio_codes", "target_slots", "motion_local_labels"}
        }
        selected = list(range(tensor_batch["input_ids"].shape[0]))
        _, stats = apply_generated_history(
            model,
            tensor_batch,
            selected,
            microbatch_size=len(selected),
            use_bf16=use_bf16,
        )
        totals.update(stats)
    return totals


def validate_generated_labels(batch: Mapping[str, torch.Tensor]) -> None:
    labels = batch["motion_local_labels"]
    slots = batch["target_slots"]
    if not torch.equal(labels.ne(IGNORE_INDEX), slots.ge(0)):
        raise ValueError("Generated-history labels and target-slot masks do not match")

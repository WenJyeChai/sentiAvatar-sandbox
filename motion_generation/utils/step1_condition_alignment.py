"""Loss-only condition interventions for the causal Step 1 planner.

The helpers preserve sequence length, targets, and motion history. Only the
selected conditioning modality changes, so the resulting likelihood gap is an
outcome-level measure of text/audio reliance rather than an attention proxy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class ConditionCorruption:
    input_ids: torch.Tensor
    audio_codes: torch.Tensor
    selected_indices: torch.Tensor
    target_mask: torch.Tensor
    modality: str


def deterministic_condition_indices(
    names: Sequence[str],
    fraction: float,
    *,
    seed: int,
    epoch: int,
    batch_index: int,
) -> list[int]:
    if not 0 <= float(fraction) <= 1:
        raise ValueError("condition-alignment fraction must be in [0, 1]")
    count = min(len(names), max(0, round(len(names) * float(fraction))))
    scored = []
    for index, name in enumerate(names):
        key = f"cf|{seed}|{epoch}|{batch_index}|{name}".encode("utf-8")
        scored.append((int.from_bytes(hashlib.sha256(key).digest()[:8], "big"), index))
    return sorted(index for _, index in sorted(scored)[:count])


def _resample_indices(source_count: int, target_count: int, device: torch.device) -> torch.Tensor:
    if source_count <= 0 or target_count <= 0:
        raise ValueError("Cannot resample an empty condition span")
    return torch.linspace(
        0, source_count - 1, steps=target_count, device=device
    ).round().long()


def corrupt_text_condition(
    *,
    input_ids: torch.Tensor,
    audio_codes: torch.Tensor,
    text_mask: torch.Tensor,
    target_anchor_ids: torch.Tensor,
    names: Sequence[str],
    selected_indices: Sequence[int],
    seed: int,
    epoch: int,
    batch_index: int,
) -> ConditionCorruption:
    """Replace transcript tokens while preserving every sequence position."""

    if input_ids.ndim != 2 or text_mask.shape != input_ids.shape:
        raise ValueError("input_ids and text_mask must share shape [B, L]")
    if tuple(audio_codes.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("audio_codes must begin with the input [B, L] shape")
    if target_anchor_ids.shape != input_ids.shape:
        raise ValueError("target_anchor_ids must share shape [B, L]")
    batch_size = input_ids.shape[0]
    if len(names) != batch_size:
        raise ValueError("names must match the batch size")
    original = input_ids
    corrupted = input_ids.clone()
    valid_rows: list[int] = []
    target_mask = torch.zeros_like(text_mask, dtype=torch.bool)
    transcript_tokens = [
        tuple(int(value) for value in original[row, text_mask[row]].tolist())
        for row in range(batch_size)
    ]
    for row in selected_indices:
        row = int(row)
        candidates = [
            donor
            for donor in range(batch_size)
            if donor != row
            and transcript_tokens[donor]
            and transcript_tokens[donor] != transcript_tokens[row]
        ]
        if not candidates or not transcript_tokens[row]:
            continue
        candidates.sort(
            key=lambda donor: hashlib.sha256(
                f"text|{seed}|{epoch}|{batch_index}|{names[row]}|{names[donor]}".encode(
                    "utf-8"
                )
            ).digest()
        )
        donor = candidates[0]
        destination_positions = text_mask[row].nonzero(as_tuple=False).squeeze(-1)
        donor_positions = text_mask[donor].nonzero(as_tuple=False).squeeze(-1)
        donor_indices = _resample_indices(
            int(donor_positions.numel()), int(destination_positions.numel()), input_ids.device
        )
        replacement = original[donor, donor_positions.index_select(0, donor_indices)]
        if torch.equal(replacement, original[row, destination_positions]):
            continue
        corrupted[row, destination_positions] = replacement
        target_mask[row] = target_anchor_ids[row].ge(0)
        if bool(target_mask[row].any()):
            valid_rows.append(row)
    indices = torch.as_tensor(valid_rows, dtype=torch.long, device=input_ids.device)
    return ConditionCorruption(
        input_ids=corrupted,
        audio_codes=audio_codes.clone(),
        selected_indices=indices,
        target_mask=target_mask,
        modality="text",
    )


def corrupt_audio_with_causal_past(
    *,
    input_ids: torch.Tensor,
    audio_codes: torch.Tensor,
    audio_anchor_ids: torch.Tensor,
    target_anchor_ids: torch.Tensor,
    selected_indices: Sequence[int],
    shift_anchors: int,
) -> ConditionCorruption:
    """Replace each eligible audio interval with a strictly earlier interval."""

    if shift_anchors <= 0:
        raise ValueError("audio shift_anchors must be positive")
    if input_ids.ndim != 2 or tuple(audio_codes.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("input_ids/audio_codes must have compatible [B, L] shapes")
    if audio_anchor_ids.shape != input_ids.shape or target_anchor_ids.shape != input_ids.shape:
        raise ValueError("anchor-id tensors must share the input [B, L] shape")
    original = audio_codes
    corrupted = audio_codes.clone()
    target_mask = torch.zeros_like(target_anchor_ids, dtype=torch.bool)
    valid_rows: list[int] = []
    for row_value in selected_indices:
        row = int(row_value)
        changed_groups: list[int] = []
        groups = torch.unique(audio_anchor_ids[row][audio_anchor_ids[row].ge(0)]).tolist()
        for group_value in groups:
            group = int(group_value)
            source_group = group - int(shift_anchors)
            if source_group < 0:
                continue
            destination_positions = audio_anchor_ids[row].eq(group).nonzero(
                as_tuple=False
            ).squeeze(-1)
            source_positions = audio_anchor_ids[row].eq(source_group).nonzero(
                as_tuple=False
            ).squeeze(-1)
            if not destination_positions.numel() or not source_positions.numel():
                continue
            source_indices = _resample_indices(
                int(source_positions.numel()),
                int(destination_positions.numel()),
                audio_codes.device,
            )
            replacement = original[
                row, source_positions.index_select(0, source_indices)
            ]
            corrupted[row, destination_positions] = replacement
            changed_groups.append(group)
        for group in changed_groups:
            target_mask[row] |= target_anchor_ids[row].eq(group)
        if bool(target_mask[row].any()):
            valid_rows.append(row)
    indices = torch.as_tensor(valid_rows, dtype=torch.long, device=input_ids.device)
    return ConditionCorruption(
        input_ids=input_ids.clone(),
        audio_codes=corrupted,
        selected_indices=indices,
        target_mask=target_mask,
        modality="audio",
    )


def masked_per_example_mean(
    values: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    if values.shape != target_mask.shape or values.ndim != 2:
        raise ValueError("values and target_mask must share shape [B, L]")
    counts = target_mask.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise ValueError("Every selected counterfactual example needs supervised targets")
    return (values * target_mask.to(values.dtype)).sum(dim=1) / counts


def counterfactual_likelihood_loss(
    *,
    positive_token_loss: torch.Tensor,
    negative_token_loss: torch.Tensor,
    target_mask: torch.Tensor,
    margin_nats: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank correct conditions above corrupted ones by mean target log-likelihood."""

    positive_nll = masked_per_example_mean(positive_token_loss, target_mask).detach()
    negative_nll = masked_per_example_mean(negative_token_loss, target_mask)
    # log p(correct) - log p(wrong) == NLL(wrong) - NLL(correct)
    condition_gap = negative_nll - positive_nll
    loss = F.softplus(float(margin_nats) - condition_gap).mean()
    return loss, condition_gap.detach()


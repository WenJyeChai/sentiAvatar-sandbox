"""Evaluation helpers for the causal fixed-gap Step 1 anchor planner.

The helpers deliberately separate teacher-forced token prediction, generated-
prefix rollout, simple data baselines, and codec-space anchor substitution.
Decoding a sequence with GT non-anchor tokens is an oracle-gap diagnostic; it
is not a replacement for a Step 2 infilling rollout.
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
from torch.utils.data import Dataset

from models.step1_mimi_planner import (
    IGNORE_INDEX,
    MimiQwenPlanner,
    canonical_data_path,
    load_mimi_tokens,
    load_motion_tokens,
)
from scripts.export_multipart_motion_tokens import LoadedPartCodec
from utils.adaptive_anchor_tokens import (
    BODY_CODEBOOK_SIZE,
    BODY_PART_ORDER,
    BODY_SLOT_COUNT,
    BODY_SLOTS,
    fixed_anchor_times,
)
from utils.causal_codec_evaluation import reconstruction_metrics


def slot_name(slot: int) -> str:
    spec = BODY_SLOTS[int(slot)]
    return f"{spec.part}_q{spec.quantizer}"


def summarize_slot_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    negative_log_likelihood: Optional[np.ndarray] = None,
    top5_correct: Optional[np.ndarray] = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("labels and predictions must share shape [N, 16]")
    if labels.shape[1] != BODY_SLOT_COUNT:
        raise ValueError(f"Expected {BODY_SLOT_COUNT} slots, got {labels.shape}")
    if negative_log_likelihood is not None:
        negative_log_likelihood = np.asarray(negative_log_likelihood, dtype=np.float64)
        if negative_log_likelihood.shape != labels.shape:
            raise ValueError("negative_log_likelihood must match labels")
    if top5_correct is not None:
        top5_correct = np.asarray(top5_correct, dtype=bool)
        if top5_correct.shape != labels.shape:
            raise ValueError("top5_correct must match labels")

    correct = predictions == labels
    rows: list[dict[str, float | int | str]] = []
    for slot, spec in enumerate(BODY_SLOTS):
        row: dict[str, float | int | str] = {
            "slot": slot,
            "part": spec.part,
            "quantizer": spec.quantizer,
            "slot_name": slot_name(slot),
            "count": int(labels.shape[0]),
            "accuracy": float(correct[:, slot].mean()) if len(labels) else math.nan,
        }
        if negative_log_likelihood is not None:
            row["cross_entropy"] = float(negative_log_likelihood[:, slot].mean())
            row["perplexity"] = float(math.exp(min(50.0, row["cross_entropy"])))
        if top5_correct is not None:
            row["top5_accuracy"] = float(top5_correct[:, slot].mean())
        rows.append(row)

    summary: dict[str, float | int] = {
        "anchors": int(labels.shape[0]),
        "tokens": int(labels.size),
        "accuracy": float(correct.mean()) if labels.size else math.nan,
    }
    if negative_log_likelihood is not None:
        summary["cross_entropy"] = float(negative_log_likelihood.mean())
        summary["perplexity"] = float(math.exp(min(50.0, summary["cross_entropy"])))
    if top5_correct is not None:
        summary["top5_accuracy"] = float(top5_correct.mean())
    for quantizer in range(4):
        indices = [slot for slot, spec in enumerate(BODY_SLOTS) if spec.quantizer == quantizer]
        summary[f"q{quantizer}_accuracy"] = float(correct[:, indices].mean())
    return summary, rows


def collect_fixed_gap_targets(
    *,
    names: Sequence[str],
    motion_token_dir: Path,
    fixed_gap: int,
    require_causal: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    """Collect GT target anchors and the preceding GT anchor for baselines."""

    targets: list[tuple[int, ...]] = []
    previous: list[tuple[int, ...]] = []
    index_rows: list[dict[str, int | str]] = []
    for name in names:
        path = canonical_data_path(Path(motion_token_dir), name, ".json")
        dense, _ = load_motion_tokens(path, require_causal=require_causal)
        times = fixed_anchor_times(len(dense), gap=fixed_gap)
        for anchor_index, (left, right) in enumerate(zip(times, times[1:]), start=1):
            previous.append(tuple(int(value) for value in dense[left]))
            targets.append(tuple(int(value) for value in dense[right]))
            index_rows.append(
                {
                    "name": str(name),
                    "anchor_index": anchor_index,
                    "previous_time": int(left),
                    "target_time": int(right),
                }
            )
    if not targets:
        empty = np.empty((0, BODY_SLOT_COUNT), dtype=np.int64)
        return empty, empty.copy(), index_rows
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(previous, dtype=np.int64),
        index_rows,
    )


def fit_unigram_prior(targets: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 2 or targets.shape[1] != BODY_SLOT_COUNT:
        raise ValueError("targets must have shape [N, 16]")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    counts = np.full((BODY_SLOT_COUNT, BODY_CODEBOOK_SIZE), float(alpha), dtype=np.float64)
    for slot in range(BODY_SLOT_COUNT):
        counts[slot] += np.bincount(targets[:, slot], minlength=BODY_CODEBOOK_SIZE)
    return counts / counts.sum(axis=1, keepdims=True)


def evaluate_reference_baselines(
    *,
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    validation_previous: np.ndarray,
    alpha: float = 1.0,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate uniform, train-unigram, and previous-anchor-copy references."""

    prior = fit_unigram_prior(train_targets, alpha=alpha)
    majority = prior.argmax(axis=1)
    majority_predictions = np.broadcast_to(majority, validation_targets.shape)
    prior_nll = -np.log(
        prior[np.arange(BODY_SLOT_COUNT)[None, :], validation_targets]
    )

    uniform_predictions = np.zeros_like(validation_targets)
    uniform_nll = np.full(validation_targets.shape, math.log(BODY_CODEBOOK_SIZE))
    results = []
    slot_rows = []
    for baseline, predictions, nll in (
        ("uniform_reference", uniform_predictions, uniform_nll),
        ("train_unigram_majority", majority_predictions, prior_nll),
        ("previous_gt_anchor_copy", validation_previous, None),
    ):
        summary, rows = summarize_slot_metrics(
            validation_targets,
            predictions,
            negative_log_likelihood=nll,
        )
        if baseline == "uniform_reference":
            expected_accuracy = 1.0 / BODY_CODEBOOK_SIZE
            summary["accuracy"] = expected_accuracy
            for quantizer in range(4):
                summary[f"q{quantizer}_accuracy"] = expected_accuracy
            for row in rows:
                row["accuracy"] = expected_accuracy
        summary["baseline"] = baseline
        results.append(summary)
        for row in rows:
            row["baseline"] = baseline
            slot_rows.append(row)
    return results, slot_rows


def _move_tensor_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if torch.is_tensor(value)
        and key in {"input_ids", "attention_mask", "audio_codes", "target_slots", "motion_local_labels"}
    }


@torch.inference_mode()
def teacher_forced_metrics(
    model: MimiQwenPlanner,
    loader,
    *,
    device: torch.device,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Return CE, top-1, and top-5 metrics without materializing full-vocab logits."""

    model.eval()
    nll_by_slot: list[list[np.ndarray]] = [[] for _ in range(BODY_SLOT_COUNT)]
    labels_by_slot: list[list[np.ndarray]] = [[] for _ in range(BODY_SLOT_COUNT)]
    predictions_by_slot: list[list[np.ndarray]] = [[] for _ in range(BODY_SLOT_COUNT)]
    top5_by_slot: list[list[np.ndarray]] = [[] for _ in range(BODY_SLOT_COUNT)]
    started = time.perf_counter()
    for batch in loader:
        values = _move_tensor_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_bf16 and device.type == "cuda"),
        ):
            embeddings = model.prepare_input_embeddings(
                values["input_ids"], values["audio_codes"]
            )
            outputs = model._base_model_forward(  # pylint: disable=protected-access
                inputs_embeds=embeddings,
                attention_mask=values["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state[:, :-1]
        shifted_slots = values["target_slots"][:, 1:]
        shifted_labels = values["motion_local_labels"][:, 1:]
        output_weight = model.language_model.get_output_embeddings().weight
        for slot in range(BODY_SLOT_COUNT):
            mask = shifted_slots.eq(slot)
            if not bool(mask.any()):
                continue
            logits = F.linear(
                hidden[mask],
                output_weight.index_select(0, model.motion_token_ids[slot]),
            ).float()
            labels = shifted_labels[mask]
            nll = F.cross_entropy(logits, labels, reduction="none")
            top5 = logits.topk(k=5, dim=-1).indices.eq(labels[:, None]).any(dim=-1)
            nll_by_slot[slot].append(nll.cpu().numpy())
            labels_by_slot[slot].append(labels.cpu().numpy())
            predictions_by_slot[slot].append(logits.argmax(dim=-1).cpu().numpy())
            top5_by_slot[slot].append(top5.cpu().numpy())

    counts = {sum(len(chunk) for chunk in values) for values in labels_by_slot}
    if len(counts) != 1:
        raise ValueError(f"Teacher-forced slot counts differ: {sorted(counts)}")
    labels = np.stack([np.concatenate(values) for values in labels_by_slot], axis=1)
    predictions = np.stack(
        [np.concatenate(values) for values in predictions_by_slot], axis=1
    )
    nll = np.stack([np.concatenate(values) for values in nll_by_slot], axis=1)
    top5 = np.stack([np.concatenate(values) for values in top5_by_slot], axis=1)
    summary, slot_rows = summarize_slot_metrics(
        labels,
        predictions,
        negative_log_likelihood=nll,
        top5_correct=top5,
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    return {
        "summary": summary,
        "slot_rows": slot_rows,
        "labels": labels,
        "predictions": predictions,
        "negative_log_likelihood": nll,
        "top5_correct": top5,
    }


@dataclass
class RolloutResult:
    name: str
    anchor_times: tuple[int, ...]
    target_anchors: np.ndarray
    predicted_anchors: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    elapsed_seconds: float

    def cache_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decoder": "greedy",
            "anchors": [
                {"time": int(time_value), "tokens": [int(value) for value in anchor]}
                for time_value, anchor in zip(
                    self.anchor_times[1:], self.predicted_anchors
                )
            ],
        }


def _target_groups(item: Mapping[str, Any]) -> list[np.ndarray]:
    target_slots = np.asarray(item["target_slots"], dtype=np.int64)
    positions = np.flatnonzero(target_slots >= 0)
    if len(positions) % BODY_SLOT_COUNT:
        raise ValueError("Target positions are not divisible into 16-slot anchors")
    groups = [positions[start : start + BODY_SLOT_COUNT] for start in range(0, len(positions), BODY_SLOT_COUNT)]
    for group in groups:
        if not np.array_equal(group, np.arange(group[0], group[0] + BODY_SLOT_COUNT)):
            raise ValueError("Anchor target positions must be contiguous")
        if not np.array_equal(target_slots[group], np.arange(BODY_SLOT_COUNT)):
            raise ValueError("Anchor target slot order is not 0..15")
    return groups


@torch.inference_mode()
def _cached_chunk(
    model: MimiQwenPlanner,
    *,
    input_ids: Sequence[int],
    audio_codes: Sequence[int],
    device: torch.device,
    past_key_values: Any,
    use_bf16: bool,
) -> tuple[torch.Tensor, Any]:
    if len(input_ids) != len(audio_codes) or not input_ids:
        raise ValueError("A cached chunk must contain matching non-empty ids and audio codes")
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    audio = torch.as_tensor(audio_codes, dtype=torch.long, device=device).unsqueeze(0)
    embeddings = model.prepare_input_embeddings(ids, audio)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(use_bf16 and device.type == "cuda"),
    ):
        outputs = model._base_model_forward(  # pylint: disable=protected-access
            inputs_embeds=embeddings,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    return outputs.last_hidden_state[:, -1], outputs.past_key_values


@torch.inference_mode()
def greedy_rollout_item(
    model: MimiQwenPlanner,
    item: Mapping[str, Any],
    *,
    device: torch.device,
    use_bf16: bool = True,
) -> RolloutResult:
    """Generate all target anchors while retaining only the known seed anchor."""

    model.eval()
    input_ids = [int(value) for value in item["input_ids"]]
    audio_codes = [int(value) for value in item["audio_codes"]]
    labels = np.asarray(item["motion_local_labels"], dtype=np.int64)
    groups = _target_groups(item)
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    confidences: list[list[float]] = []
    entropies: list[list[float]] = []
    output_weight = model.language_model.get_output_embeddings().weight
    cursor = 0
    past = None
    started = time.perf_counter()

    for group in groups:
        first = int(group[0])
        hidden, past = _cached_chunk(
            model,
            input_ids=input_ids[cursor:first],
            audio_codes=audio_codes[cursor:first],
            device=device,
            past_key_values=past,
            use_bf16=use_bf16,
        )
        predicted_anchor: list[int] = []
        anchor_confidence: list[float] = []
        anchor_entropy: list[float] = []
        for slot in range(BODY_SLOT_COUNT):
            logits = F.linear(
                hidden,
                output_weight.index_select(0, model.motion_token_ids[slot]),
            ).float().squeeze(0)
            probabilities = logits.softmax(dim=-1)
            local_id = int(logits.argmax().item())
            predicted_anchor.append(local_id)
            anchor_confidence.append(float(probabilities[local_id].item()))
            anchor_entropy.append(
                float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
            )
            token_id = int(model.motion_token_ids[slot, local_id].item())
            hidden, past = _cached_chunk(
                model,
                input_ids=[token_id],
                audio_codes=[-1],
                device=device,
                past_key_values=past,
                use_bf16=use_bf16,
            )
        predictions.append(predicted_anchor)
        targets.append([int(labels[position]) for position in group])
        confidences.append(anchor_confidence)
        entropies.append(anchor_entropy)
        cursor = int(group[-1]) + 1

    return RolloutResult(
        name=str(item["name"]),
        anchor_times=tuple(int(value) for value in item["anchor_times"]),
        target_anchors=np.asarray(targets, dtype=np.int64),
        predicted_anchors=np.asarray(predictions, dtype=np.int64),
        confidence=np.asarray(confidences, dtype=np.float64),
        entropy=np.asarray(entropies, dtype=np.float64),
        elapsed_seconds=time.perf_counter() - started,
    )


def evaluate_rollouts(results: Sequence[RolloutResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one rollout is required")
    labels = np.concatenate([result.target_anchors for result in results], axis=0)
    predictions = np.concatenate([result.predicted_anchors for result in results], axis=0)
    confidence = np.concatenate([result.confidence for result in results], axis=0)
    entropy = np.concatenate([result.entropy for result in results], axis=0)
    summary, slot_rows = summarize_slot_metrics(labels, predictions)
    summary.update(
        clips=len(results),
        mean_confidence=float(confidence.mean()),
        mean_entropy=float(entropy.mean()),
        elapsed_seconds=float(sum(result.elapsed_seconds for result in results)),
    )

    horizon_rows = []
    for result in results:
        correct = result.predicted_anchors == result.target_anchors
        denominator = max(1, len(correct) - 1)
        for anchor_index, anchor_correct in enumerate(correct, start=1):
            horizon_rows.append(
                {
                    "name": result.name,
                    "anchor_index": anchor_index,
                    "target_time": int(result.anchor_times[anchor_index]),
                    "relative_horizon": float((anchor_index - 1) / denominator),
                    "accuracy": float(anchor_correct.mean()),
                    "q0_accuracy": float(anchor_correct[[0, 4, 8, 12]].mean()),
                    "confidence": float(result.confidence[anchor_index - 1].mean()),
                    "entropy": float(result.entropy[anchor_index - 1].mean()),
                }
            )
    return {
        "summary": summary,
        "slot_rows": slot_rows,
        "horizon_rows": horizon_rows,
        "labels": labels,
        "predictions": predictions,
    }


def write_rollout_cache(results: Sequence[RolloutResult], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    for result in results:
        parts = PurePosixPath(result.name.replace("\\", "/")).parts
        path = output_dir / Path(*parts).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result.cache_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)


class ShuffledAudioDataset(Dataset):
    """Keep the target/template fixed while replacing q0 with another clip's q0."""

    def __init__(
        self,
        base_dataset,
        *,
        donor_names: Sequence[str],
        mimi_token_dir: Path,
    ) -> None:
        if len(base_dataset) != len(donor_names):
            raise ValueError("base_dataset and donor_names must have the same length")
        self.base_dataset = base_dataset
        self.donor_names = [str(name) for name in donor_names]
        self.mimi_token_dir = Path(mimi_token_dir)
        self.names = list(base_dataset.names)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        donor_path = canonical_data_path(
            self.mimi_token_dir, self.donor_names[index], ".npz"
        )
        donor = np.asarray(load_mimi_tokens(donor_path)["codes"])[0]
        positions = np.flatnonzero(np.asarray(item["audio_codes"]) >= 0)
        if not len(positions) or not len(donor):
            return item
        donor_indices = np.rint(
            np.linspace(0, len(donor) - 1, num=len(positions))
        ).astype(np.int64)
        audio_codes = np.asarray(item["audio_codes"], dtype=np.int64)
        audio_codes[positions] = donor[donor_indices]
        item["audio_codes"] = audio_codes.tolist()
        return item


def _part_codes(dense_tokens: np.ndarray, part_index: int) -> torch.Tensor:
    start = int(part_index) * 4
    return torch.as_tensor(
        dense_tokens[:, start : start + 4], dtype=torch.long
    ).unsqueeze(0)


@torch.inference_mode()
def codec_anchor_diagnostics(
    *,
    codecs: Mapping[str, LoadedPartCodec],
    dense_tokens: Sequence[Sequence[int]],
    anchor_times: Sequence[int],
    predicted_anchors: np.ndarray,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    """Evaluate predicted anchors with GT gaps retained as an explicit oracle.

    The result isolates damage caused by replacing anchor token IDs. It does not
    measure Step 2 and must be labelled ``oracle_gap_anchor_substitution``.
    """

    dense = np.asarray(dense_tokens, dtype=np.int64)
    predicted_anchors = np.asarray(predicted_anchors, dtype=np.int64)
    times = np.asarray(anchor_times[1:], dtype=np.int64)
    if predicted_anchors.shape != (len(times), BODY_SLOT_COUNT):
        raise ValueError("Predicted anchors do not match non-seed anchor times")
    predicted_dense = dense.copy()
    copy_dense = dense.copy()
    predicted_dense[times] = predicted_anchors
    previous_times = np.asarray(anchor_times[:-1], dtype=np.int64)
    copy_dense[times] = dense[previous_times]

    rows: list[dict[str, float | int | str]] = []
    for part_index, part in enumerate(BODY_PART_ORDER):
        loaded = codecs[part]
        gt_codes = _part_codes(dense, part_index).to(device)
        predicted_codes = _part_codes(predicted_dense, part_index).to(device)
        copy_codes = _part_codes(copy_dense, part_index).to(device)
        quantizer = loaded.model.quantizers[part]
        gt_anchor_codes = _part_codes(dense[times], part_index).to(device)
        predicted_anchor_codes = _part_codes(predicted_anchors, part_index).to(device)
        gt_latent = quantizer.get_codebook_entry(gt_anchor_codes)
        predicted_latent = quantizer.get_codebook_entry(predicted_anchor_codes)
        latent_rmse = float(torch.sqrt(torch.mean((predicted_latent - gt_latent) ** 2)).item())

        decoded = {}
        decoded_normalized = {}
        for variant, codes in (
            ("gt_dense_codec_reference", gt_codes),
            ("oracle_gap_anchor_substitution", predicted_codes),
            ("oracle_gap_previous_anchor_copy", copy_codes),
        ):
            normalized = loaded.model.decode({part: codes})[part].squeeze(0)
            raw = loaded.normalizer.denormalize_tensor(
                part, normalized.unsqueeze(0)
            ).squeeze(0)
            decoded_normalized[variant] = normalized
            decoded[variant] = raw

        target_raw = decoded["gt_dense_codec_reference"]
        target_normalized = decoded_normalized["gt_dense_codec_reference"]
        for variant in (
            "oracle_gap_anchor_substitution",
            "oracle_gap_previous_anchor_copy",
        ):
            metrics = reconstruction_metrics(
                part=part,
                target_raw=target_raw,
                prediction_raw=decoded[variant],
                target_normalized=target_normalized,
                prediction_normalized=decoded_normalized[variant],
            )
            rows.append(
                {
                    "part": part,
                    "variant": variant,
                    "token_frames": int(len(dense)),
                    "anchor_count": int(len(times)),
                    "predicted_anchor_latent_rmse": latent_rmse,
                    **metrics,
                }
            )
    return rows

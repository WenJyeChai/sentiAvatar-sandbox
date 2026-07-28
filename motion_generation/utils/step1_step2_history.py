"""Offline GT-boundary Step 2 history cache contracts for Step 1.

The cache owns the fixed supplied-gap schedule and the frozen Step 2 greedy
missing-frame predictions for each completed interval.  Training samples a
recent contiguous suffix at serialization time; the cache itself never stores
truncated or corrupted histories.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np

from utils.adaptive_anchor_tokens import (
    BODY_CODEBOOK_SIZE,
    BODY_SLOT_COUNT,
    gap_from_anchor_times,
    validate_anchor,
)


STEP2_HISTORY_CACHE_SCHEMA = "sentiavatar.step1_gt_boundary_step2_history.v1"


def cache_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(str(name).replace("\\", "/")).parts
    return Path(root) / Path(*parts).with_suffix(".npz")


def deterministic_uint64(*values: object) -> int:
    encoded = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def deterministic_unit_interval(*values: object) -> float:
    return deterministic_uint64(*values) / float(2**64)


@dataclass(frozen=True)
class Step2HistoryCache:
    name: str
    token_frames: int
    anchor_times: tuple[int, ...]
    gaps: tuple[int, ...]
    interval_offsets: tuple[int, ...]
    history_frames: np.ndarray
    schedule_seed: int
    step2_checkpoint_fingerprint: str

    @property
    def num_intervals(self) -> int:
        return len(self.gaps)

    def interval(self, interval_index: int) -> np.ndarray:
        interval_index = int(interval_index)
        if not 0 <= interval_index < self.num_intervals:
            raise IndexError(
                f"History interval must lie in [0,{self.num_intervals - 1}]"
            )
        start = self.interval_offsets[interval_index]
        end = self.interval_offsets[interval_index + 1]
        return self.history_frames[start:end]


def validate_cache(
    cache: Step2HistoryCache,
    *,
    dense_motion_tokens: Sequence[Sequence[int]] | None = None,
) -> None:
    if cache.token_frames < 1:
        raise ValueError(f"{cache.name}: token_frames must be positive")
    if (
        not cache.anchor_times
        or cache.anchor_times[0] != 0
        or cache.anchor_times[-1] != cache.token_frames - 1
    ):
        raise ValueError(
            f"{cache.name}: cached anchors must span [0,{cache.token_frames - 1}]"
        )
    if len(cache.gaps) != len(cache.anchor_times) - 1:
        raise ValueError(f"{cache.name}: cached gaps do not match anchor count")
    expected_gaps = tuple(
        gap_from_anchor_times(left, right)
        for left, right in zip(cache.anchor_times[:-1], cache.anchor_times[1:])
    )
    if cache.gaps != expected_gaps:
        raise ValueError(
            f"{cache.name}: cached gaps {cache.gaps} != schedule {expected_gaps}"
        )
    if len(cache.interval_offsets) != len(cache.gaps) + 1:
        raise ValueError(f"{cache.name}: interval offsets do not match gaps")
    if (
        cache.interval_offsets[0] != 0
        or cache.interval_offsets[-1] != len(cache.history_frames)
        or any(
            right < left
            for left, right in zip(
                cache.interval_offsets[:-1], cache.interval_offsets[1:]
            )
        )
    ):
        raise ValueError(f"{cache.name}: malformed interval offsets")
    if cache.history_frames.ndim != 2 or cache.history_frames.shape[1] != BODY_SLOT_COUNT:
        raise ValueError(
            f"{cache.name}: history frames must be [N,{BODY_SLOT_COUNT}], "
            f"got {cache.history_frames.shape}"
        )
    if cache.history_frames.size and (
        int(cache.history_frames.min()) < 0
        or int(cache.history_frames.max()) >= BODY_CODEBOOK_SIZE
    ):
        raise ValueError(
            f"{cache.name}: history token lies outside [0,{BODY_CODEBOOK_SIZE - 1}]"
        )
    for interval_index, gap in enumerate(cache.gaps):
        frames = cache.interval(interval_index)
        expected_frames = int(gap) + 1
        if len(frames) != expected_frames:
            raise ValueError(
                f"{cache.name}: interval {interval_index} stores {len(frames)} "
                f"frames; expected gap+right_endpoint={expected_frames}"
            )
        validate_anchor(frames[-1].tolist())

    if dense_motion_tokens is not None:
        dense = np.asarray(dense_motion_tokens, dtype=np.int64)
        if dense.shape != (cache.token_frames, BODY_SLOT_COUNT):
            raise ValueError(
                f"{cache.name}: dense tokens {dense.shape} do not match cache "
                f"({cache.token_frames},{BODY_SLOT_COUNT})"
            )
        for interval_index, right_time in enumerate(cache.anchor_times[1:]):
            if not np.array_equal(
                cache.interval(interval_index)[-1],
                dense[int(right_time)],
            ):
                raise ValueError(
                    f"{cache.name}: interval {interval_index} does not end at "
                    f"the GT right endpoint t={right_time}"
                )


def load_step2_history_cache(
    path: Path,
    *,
    dense_motion_tokens: Sequence[Sequence[int]] | None = None,
) -> Step2HistoryCache:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing GT-boundary Step 2 history cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        if schema != STEP2_HISTORY_CACHE_SCHEMA:
            raise ValueError(
                f"{path}: history schema {schema!r} != "
                f"{STEP2_HISTORY_CACHE_SCHEMA!r}"
            )
        cache = Step2HistoryCache(
            name=str(payload["name"].item()).replace("\\", "/"),
            token_frames=int(payload["token_frames"].item()),
            anchor_times=tuple(
                int(value)
                for value in np.asarray(payload["anchor_times"], dtype=np.int64)
            ),
            gaps=tuple(
                int(value)
                for value in np.asarray(payload["gaps"], dtype=np.int64)
            ),
            interval_offsets=tuple(
                int(value)
                for value in np.asarray(payload["interval_offsets"], dtype=np.int64)
            ),
            history_frames=np.asarray(
                payload["history_frames"], dtype=np.int64
            ),
            schedule_seed=int(payload["schedule_seed"].item()),
            step2_checkpoint_fingerprint=str(
                payload["step2_checkpoint_fingerprint"].item()
            ),
        )
    validate_cache(cache, dense_motion_tokens=dense_motion_tokens)
    return cache


def save_step2_history_cache(
    path: Path,
    *,
    name: str,
    token_frames: int,
    anchor_times: Sequence[int],
    interval_frames: Sequence[np.ndarray],
    schedule_seed: int,
    step2_checkpoint_fingerprint: str,
) -> None:
    normalized_name = str(name).replace("\\", "/")
    anchors = tuple(int(value) for value in anchor_times)
    gaps = tuple(
        gap_from_anchor_times(left, right)
        for left, right in zip(anchors[:-1], anchors[1:])
    )
    arrays: list[np.ndarray] = []
    offsets = [0]
    for interval_index, (gap, raw_frames) in enumerate(
        zip(gaps, interval_frames)
    ):
        frames = np.asarray(raw_frames, dtype=np.int64)
        if frames.shape != (gap + 1, BODY_SLOT_COUNT):
            raise ValueError(
                f"{normalized_name}: interval {interval_index} has shape "
                f"{frames.shape}; expected {(gap + 1, BODY_SLOT_COUNT)}"
            )
        if frames.size and (
            int(frames.min()) < 0
            or int(frames.max()) >= BODY_CODEBOOK_SIZE
        ):
            raise ValueError(
                f"{normalized_name}: interval {interval_index} contains an "
                "out-of-range body ID"
            )
        arrays.append(frames.astype(np.uint16, copy=False))
        offsets.append(offsets[-1] + len(frames))
    history_frames = (
        np.concatenate(arrays, axis=0)
        if arrays
        else np.empty((0, BODY_SLOT_COUNT), dtype=np.uint16)
    )
    cache = Step2HistoryCache(
        name=normalized_name,
        token_frames=int(token_frames),
        anchor_times=anchors,
        gaps=gaps,
        interval_offsets=tuple(offsets),
        history_frames=history_frames.astype(np.int64),
        schedule_seed=int(schedule_seed),
        step2_checkpoint_fingerprint=str(step2_checkpoint_fingerprint),
    )
    validate_cache(cache)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(STEP2_HISTORY_CACHE_SCHEMA),
            name=np.asarray(normalized_name),
            token_frames=np.asarray(int(token_frames), dtype=np.int64),
            anchor_times=np.asarray(anchors, dtype=np.int32),
            gaps=np.asarray(gaps, dtype=np.int8),
            interval_offsets=np.asarray(offsets, dtype=np.int64),
            history_frames=history_frames,
            schedule_seed=np.asarray(int(schedule_seed), dtype=np.int64),
            step2_checkpoint_fingerprint=np.asarray(
                str(step2_checkpoint_fingerprint)
            ),
        )
    temporary.replace(path)


def load_history_manifests(root: Path) -> list[dict]:
    manifests = []
    for path in sorted(Path(root).glob("manifest_shard_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != STEP2_HISTORY_CACHE_SCHEMA:
            raise ValueError(
                f"{path}: manifest schema {payload.get('schema')!r} is invalid"
            )
        manifests.append(payload)
    if not manifests:
        raise FileNotFoundError(
            f"No manifest_shard_*.json files found in history cache {root}"
        )
    shard_counts = {int(payload.get("num_shards", -1)) for payload in manifests}
    if len(shard_counts) != 1 or next(iter(shard_counts)) < 1:
        raise ValueError("GT-boundary Step 2 history manifests disagree on shard count")
    num_shards = next(iter(shard_counts))
    shard_ids = [int(payload.get("shard_id", -1)) for payload in manifests]
    if sorted(shard_ids) != list(range(num_shards)):
        raise ValueError(
            "GT-boundary Step 2 history cache is missing/duplicating shards: "
            f"found {sorted(shard_ids)}, expected {list(range(num_shards))}"
        )
    failures = sum(
        int(payload.get("missing_or_bad", 0)) for payload in manifests
    )
    if failures:
        raise ValueError(
            f"GT-boundary Step 2 history manifests report {failures} bad clips"
        )
    incomplete = [
        int(payload.get("shard_id", -1))
        for payload in manifests
        if int(payload.get("completed", 0))
        + int(payload.get("existing_skipped", 0))
        != int(payload.get("assigned", -1))
    ]
    if incomplete:
        raise ValueError(
            "GT-boundary Step 2 history shards are incomplete: "
            f"{incomplete}"
        )
    fingerprints = {
        str(payload.get("step2_checkpoint_fingerprint", ""))
        for payload in manifests
    }
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        raise ValueError(
            "GT-boundary Step 2 history manifests use mixed/missing checkpoints"
        )
    schedules = {
        json.dumps(payload.get("schedule", {}), sort_keys=True)
        for payload in manifests
    }
    if len(schedules) != 1:
        raise ValueError(
            "GT-boundary Step 2 history manifests use different schedules"
        )
    return manifests

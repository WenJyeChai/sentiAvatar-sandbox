"""Frozen-Step-2 interval costs and dynamic-programming targets for Step 1.

The cost matrix convention used here is ``edge_costs[left, gap]``, where the
right anchor is ``left + gap + 1``.  Gaps 3--15 are normal planner decisions;
gaps 0--2 are permitted only when they land exactly on the final motion frame.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


MAX_GAP = 15
TAIL_MAX_GAP = 2


@dataclass(frozen=True)
class GapCurriculumPhase:
    start_epoch: int
    end_epoch: int
    min_gap: int
    max_gap: int
    target_mean_gap: Optional[float]
    schedule_loss_weight_start: float
    schedule_loss_weight_end: float
    temperature: float
    mode: str = "step2_dp"

    def __post_init__(self) -> None:
        if self.start_epoch < 1 or self.end_epoch < self.start_epoch:
            raise ValueError("Curriculum epochs must be positive and ordered")
        if not 3 <= self.min_gap <= self.max_gap <= MAX_GAP:
            raise ValueError("Normal curriculum gaps must lie in [3, 15]")
        if self.mode not in {"random", "step2_dp"}:
            raise ValueError("Curriculum mode must be random or step2_dp")
        if self.mode == "step2_dp" and self.target_mean_gap is None:
            raise ValueError("step2_dp phases require target_mean_gap")
        if self.schedule_loss_weight_start < 0 or self.schedule_loss_weight_end < 0:
            raise ValueError("schedule loss weights must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def contains(self, completed_epoch_zero_based: int) -> bool:
        epoch = int(completed_epoch_zero_based) + 1
        return self.start_epoch <= epoch <= self.end_epoch

    def loss_weight(self, completed_epoch_zero_based: int) -> float:
        epoch = int(completed_epoch_zero_based) + 1
        if not self.start_epoch <= epoch <= self.end_epoch:
            raise ValueError(f"Epoch {epoch} is outside this curriculum phase")
        span = self.end_epoch - self.start_epoch
        progress = 1.0 if span == 0 else (epoch - self.start_epoch) / span
        return float(
            self.schedule_loss_weight_start
            + progress
            * (self.schedule_loss_weight_end - self.schedule_loss_weight_start)
        )


@dataclass(frozen=True)
class GapSchedule:
    anchor_times: tuple[int, ...]
    soft_targets_by_left: Mapping[int, tuple[float, ...]]
    normal_gaps: tuple[int, ...]
    tail_gap: Optional[int]
    total_cost: float


def parse_curriculum(
    values: Sequence[Mapping[str, object]],
    *,
    num_epochs: int,
) -> tuple[GapCurriculumPhase, ...]:
    phases = tuple(
        GapCurriculumPhase(
            start_epoch=int(value["start_epoch"]),
            end_epoch=int(value["end_epoch"]),
            min_gap=int(value.get("min_gap", 3)),
            max_gap=int(value["max_gap"]),
            target_mean_gap=(
                None
                if value.get("target_mean_gap") is None
                else float(value["target_mean_gap"])
            ),
            schedule_loss_weight_start=float(
                value.get(
                    "schedule_loss_weight_start",
                    value.get("schedule_loss_weight", 1.0),
                )
            ),
            schedule_loss_weight_end=float(
                value.get(
                    "schedule_loss_weight_end",
                    value.get("schedule_loss_weight", 1.0),
                )
            ),
            temperature=float(value.get("temperature", 0.35)),
            mode=str(value.get("mode", "step2_dp")),
        )
        for value in values
    )
    if not phases:
        raise ValueError("At least one adaptive-gap curriculum phase is required")
    expected_start = 1
    for phase in phases:
        if phase.start_epoch != expected_start:
            raise ValueError(
                f"Curriculum must be contiguous: expected epoch {expected_start}, "
                f"got {phase.start_epoch}"
            )
        expected_start = phase.end_epoch + 1
    if expected_start - 1 != int(num_epochs):
        raise ValueError(
            f"Curriculum ends at epoch {expected_start - 1}, expected {num_epochs}"
        )
    return phases


def phase_for_epoch(
    phases: Sequence[GapCurriculumPhase],
    epoch_zero_based: int,
) -> tuple[int, GapCurriculumPhase]:
    for index, phase in enumerate(phases):
        if phase.contains(epoch_zero_based):
            return index, phase
    raise ValueError(f"No curriculum phase covers epoch {epoch_zero_based + 1}")


def cache_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(str(name).replace("\\", "/")).parts
    return Path(root) / Path(*parts).with_suffix(".npz")


def load_edge_costs(
    root: Path,
    name: str,
    *,
    ce_weight: float,
    latent_weight: float,
) -> np.ndarray:
    path = cache_path(root, name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen-Step-2 interval cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        ce = np.asarray(payload["ce"], dtype=np.float64)
        latent = np.asarray(payload["hard_latent_l1"], dtype=np.float64)
    if ce.shape != latent.shape or ce.ndim != 2 or ce.shape[1] != MAX_GAP + 1:
        raise ValueError(
            f"{path}: expected matching [T, {MAX_GAP + 1}] cost arrays, "
            f"got ce={ce.shape}, latent={latent.shape}"
        )
    per_frame = float(ce_weight) * ce + float(latent_weight) * latent
    # Step 2 reports mean error over the missing interval. Dynamic programming
    # needs total interval risk; otherwise long gaps are artificially cheap.
    costs = per_frame * np.arange(MAX_GAP + 1, dtype=np.float64)[None, :]
    if bool(np.isneginf(costs).any()):
        raise ValueError(f"{path}: interval costs contain -inf")
    return costs


def _candidate_edges(
    edge_costs: np.ndarray,
    left: int,
    *,
    min_gap: int,
    max_gap: int,
) -> list[tuple[int, int, float, bool]]:
    final = edge_costs.shape[0] - 1
    candidates: list[tuple[int, int, float, bool]] = []
    for gap in range(min_gap, max_gap + 1):
        right = left + gap + 1
        if right > final:
            continue
        cost = float(edge_costs[left, gap])
        if math.isfinite(cost):
            candidates.append((gap, right, cost, False))
    tail_gap = final - left - 1
    if 0 <= tail_gap <= TAIL_MAX_GAP:
        cost = float(edge_costs[left, tail_gap])
        if not math.isfinite(cost) and tail_gap == 0:
            cost = 0.0
        if math.isfinite(cost):
            candidates.append((tail_gap, final, cost, True))
    return candidates


def solve_step2_schedule(
    edge_costs: np.ndarray,
    *,
    min_gap: int,
    max_gap: int,
    anchor_penalty: float,
    temperature: float,
) -> GapSchedule:
    """Solve the globally optimal anchor path and soft next-gap oracle."""

    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    if edge_costs.ndim != 2 or edge_costs.shape[1] != MAX_GAP + 1:
        raise ValueError(f"edge_costs must be [T, {MAX_GAP + 1}]")
    if edge_costs.shape[0] < 1:
        raise ValueError("A motion sequence must contain at least one frame")
    if not 3 <= min_gap <= max_gap <= MAX_GAP:
        raise ValueError("Normal gaps must lie in [3, 15]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    final = edge_costs.shape[0] - 1
    if final == 0:
        return GapSchedule((0,), {}, (), None, 0.0)

    value = np.full(final + 1, np.inf, dtype=np.float64)
    next_right = np.full(final + 1, -1, dtype=np.int64)
    next_gap = np.full(final + 1, -1, dtype=np.int64)
    value[final] = 0.0
    for left in range(final - 1, -1, -1):
        for gap, right, cost, _is_tail in _candidate_edges(
            edge_costs, left, min_gap=min_gap, max_gap=max_gap
        ):
            candidate = cost + float(anchor_penalty) + value[right]
            if candidate < value[left]:
                value[left] = candidate
                next_right[left] = right
                next_gap[left] = gap
    if not math.isfinite(float(value[0])):
        raise ValueError(
            f"No valid path reaches final frame {final} with normal gaps "
            f"{min_gap}--{max_gap} and EOS tails 0--2"
        )

    anchors = [0]
    normal_gaps: list[int] = []
    tail_gap: Optional[int] = None
    left = 0
    while left != final:
        right = int(next_right[left])
        gap = int(next_gap[left])
        if right <= left:
            raise RuntimeError(f"Broken DP predecessor at frame {left}")
        if gap <= TAIL_MAX_GAP:
            if right != final:
                raise RuntimeError("A 0--2 gap was selected away from EOS")
            tail_gap = gap
        else:
            normal_gaps.append(gap)
        anchors.append(right)
        left = right

    soft_targets: dict[int, tuple[float, ...]] = {}
    for left, right in zip(anchors[:-1], anchors[1:]):
        selected_gap = right - left - 1
        if selected_gap <= TAIL_MAX_GAP:
            continue
        logits = np.full(MAX_GAP + 1, -np.inf, dtype=np.float64)
        for gap, candidate_right, cost, is_tail in _candidate_edges(
            edge_costs, left, min_gap=min_gap, max_gap=max_gap
        ):
            if is_tail:
                continue
            score = cost + float(anchor_penalty) + value[candidate_right]
            if math.isfinite(score):
                logits[gap] = -score / float(temperature)
        finite = np.isfinite(logits)
        if not bool(finite.any()):
            raise RuntimeError(f"No soft normal-gap target at selected frame {left}")
        maximum = float(logits[finite].max())
        probabilities = np.zeros(MAX_GAP + 1, dtype=np.float64)
        probabilities[finite] = np.exp(logits[finite] - maximum)
        probabilities /= probabilities.sum()
        soft_targets[left] = tuple(float(value) for value in probabilities)

    return GapSchedule(
        anchor_times=tuple(anchors),
        soft_targets_by_left=soft_targets,
        normal_gaps=tuple(normal_gaps),
        tail_gap=tail_gap,
        total_cost=float(value[0]),
    )


def random_curriculum_schedule(
    num_frames: int,
    *,
    min_gap: int,
    max_gap: int,
    seed: int,
    epoch: int,
    name: str,
) -> GapSchedule:
    """Deterministic content warm-up path with legal normal gaps and EOS tail."""

    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return GapSchedule((0,), {}, (), None, 0.0)
    digest = hashlib.sha256(f"{seed}|{epoch}|{name}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    final = num_frames - 1
    anchors = [0]
    normal: list[int] = []
    tail: Optional[int] = None
    while anchors[-1] < final:
        left = anchors[-1]
        remaining = final - left - 1
        if remaining <= TAIL_MAX_GAP:
            tail = remaining
            anchors.append(final)
            break
        valid: list[int] = []
        for gap in range(min_gap, max_gap + 1):
            right = left + gap + 1
            if right > final:
                continue
            after = final - right - 1
            if after <= TAIL_MAX_GAP or after >= min_gap:
                valid.append(gap)
        if not valid:
            raise ValueError(f"Could not construct a legal warm-up schedule for T={num_frames}")
        gap = int(rng.choice(valid))
        normal.append(gap)
        anchors.append(left + gap + 1)
    return GapSchedule(tuple(anchors), {}, tuple(normal), tail, 0.0)


def schedule_mean_gap(schedules: Iterable[GapSchedule]) -> float:
    gaps = [gap for schedule in schedules for gap in schedule.normal_gaps]
    return float(np.mean(gaps)) if gaps else float("nan")


def calibrate_anchor_penalty(
    edge_costs_by_clip: Sequence[np.ndarray],
    *,
    min_gap: int,
    max_gap: int,
    target_mean_gap: float,
    tolerance: float = 0.05,
    max_iterations: int = 40,
) -> tuple[float, float]:
    """Bisection-calibrate the anchor penalty to an interval-weighted mean gap."""

    if not edge_costs_by_clip:
        raise ValueError("Calibration requires at least one clip")

    def observed(penalty: float) -> float:
        schedules = (
            solve_step2_schedule(
                costs,
                min_gap=min_gap,
                max_gap=max_gap,
                anchor_penalty=penalty,
                temperature=1.0,
            )
            for costs in edge_costs_by_clip
        )
        return schedule_mean_gap(schedules)

    low, high = -1.0, 1.0
    low_mean, high_mean = observed(low), observed(high)
    for _ in range(30):
        if low_mean <= target_mean_gap <= high_mean:
            break
        if target_mean_gap < low_mean:
            high, high_mean = low, low_mean
            low *= 2.0
            low_mean = observed(low)
        else:
            low, low_mean = high, high_mean
            high *= 2.0
            high_mean = observed(high)
    else:
        raise ValueError(
            f"Could not bracket target mean gap {target_mean_gap}: "
            f"[{low_mean}, {high_mean}]"
        )

    best_penalty = low
    best_mean = low_mean
    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        middle_mean = observed(middle)
        if abs(middle_mean - target_mean_gap) < abs(best_mean - target_mean_gap):
            best_penalty, best_mean = middle, middle_mean
        if abs(middle_mean - target_mean_gap) <= tolerance:
            break
        if middle_mean < target_mean_gap:
            low = middle
        else:
            high = middle
    return float(best_penalty), float(best_mean)


def load_calibration(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("phases"), list):
        raise ValueError(f"Invalid adaptive-gap calibration: {path}")
    return payload

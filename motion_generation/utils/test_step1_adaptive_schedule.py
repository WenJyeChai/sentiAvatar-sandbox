from __future__ import annotations

import numpy as np

from .step1_adaptive_schedule import (
    MAX_GAP,
    calibrate_anchor_penalty,
    parse_curriculum,
    random_curriculum_schedule,
    solve_step2_schedule,
)


def _constant_costs(frames: int) -> np.ndarray:
    costs = np.full((frames, MAX_GAP + 1), np.inf, dtype=np.float64)
    costs[:, 0] = 0.0
    for left in range(frames):
        for gap in range(1, MAX_GAP + 1):
            if left + gap + 1 < frames:
                costs[left, gap] = float(gap)
    return costs


def test_dp_never_uses_short_gap_away_from_eos() -> None:
    schedule = solve_step2_schedule(
        _constant_costs(37),
        min_gap=3,
        max_gap=15,
        anchor_penalty=1.0,
        temperature=0.5,
    )
    gaps = [
        right - left - 1
        for left, right in zip(schedule.anchor_times[:-1], schedule.anchor_times[1:])
    ]
    assert schedule.anchor_times[0] == 0
    assert schedule.anchor_times[-1] == 36
    assert all(gap >= 3 for gap in gaps[:-1])
    assert all(gap >= 3 for gap in gaps) or gaps[-1] in {0, 1, 2}
    assert all(
        abs(sum(target) - 1.0) < 1e-6
        for target in schedule.soft_targets_by_left.values()
    )
    assert all(
        all(probability == 0.0 for probability in target[:3])
        for target in schedule.soft_targets_by_left.values()
    )


def test_larger_anchor_penalty_increases_mean_gap() -> None:
    costs = [_constant_costs(frames) for frames in (60, 73, 91)]
    low = [
        solve_step2_schedule(
            value,
            min_gap=3,
            max_gap=15,
            anchor_penalty=-5.0,
            temperature=1.0,
        )
        for value in costs
    ]
    high = [
        solve_step2_schedule(
            value,
            min_gap=3,
            max_gap=15,
            anchor_penalty=20.0,
            temperature=1.0,
        )
        for value in costs
    ]
    assert np.mean([gap for item in high for gap in item.normal_gaps]) >= np.mean(
        [gap for item in low for gap in item.normal_gaps]
    )


def test_penalty_calibration_reaches_attainable_target() -> None:
    costs = []
    for frames in range(60, 90, 3):
        value = _constant_costs(frames)
        for gap in range(1, MAX_GAP + 1):
            finite = np.isfinite(value[:, gap])
            value[finite, gap] = 0.05 * gap * (gap - 7) ** 2
        costs.append(value)
    penalty, observed = calibrate_anchor_penalty(
        costs,
        min_gap=3,
        max_gap=15,
        target_mean_gap=7.0,
        tolerance=0.5,
    )
    assert np.isfinite(penalty)
    assert abs(observed - 7.0) <= 1.0


def test_random_warmup_is_deterministic_and_legal() -> None:
    first = random_curriculum_schedule(
        80, min_gap=3, max_gap=7, seed=42, epoch=2, name="clip/a"
    )
    second = random_curriculum_schedule(
        80, min_gap=3, max_gap=7, seed=42, epoch=2, name="clip/a"
    )
    assert first == second
    assert all(3 <= gap <= 7 for gap in first.normal_gaps)
    assert first.tail_gap is None or 0 <= first.tail_gap <= 2


def test_50_epoch_curriculum_parses_contiguously() -> None:
    phases = parse_curriculum(
        [
            {
                "start_epoch": 1,
                "end_epoch": 5,
                "mode": "random",
                "min_gap": 3,
                "max_gap": 7,
                "target_mean_gap": None,
                "schedule_loss_weight": 0,
            },
            {
                "start_epoch": 6,
                "end_epoch": 12,
                "min_gap": 3,
                "max_gap": 7,
                "target_mean_gap": 4.5,
                "schedule_loss_weight_start": 0,
                "schedule_loss_weight_end": 1,
            },
            {
                "start_epoch": 13,
                "end_epoch": 25,
                "min_gap": 3,
                "max_gap": 11,
                "target_mean_gap": 6,
            },
            {
                "start_epoch": 26,
                "end_epoch": 50,
                "min_gap": 3,
                "max_gap": 15,
                "target_mean_gap": 7,
            },
        ],
        num_epochs=50,
    )
    assert len(phases) == 4
    assert phases[1].loss_weight(5) == 0.0
    assert phases[1].loss_weight(11) == 1.0

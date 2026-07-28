"""Shared reporting bins for supplied Step 1 gap evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GapBin:
    label: str
    order: int
    minimum: int
    maximum: int
    main_bin: bool


GAP_BINS = (
    GapBin("eos_tail_0_2", 0, 0, 2, False),
    GapBin("small_3_6", 1, 3, 6, True),
    GapBin("medium_7_10", 2, 7, 10, True),
    GapBin("large_11_15", 3, 11, 15, True),
)


def supplied_gap_bin(gap: int) -> GapBin:
    value = int(gap)
    for specification in GAP_BINS:
        if specification.minimum <= value <= specification.maximum:
            return specification
    raise ValueError(
        f"Gap {value} is outside the supplied evaluation range 0--15"
    )

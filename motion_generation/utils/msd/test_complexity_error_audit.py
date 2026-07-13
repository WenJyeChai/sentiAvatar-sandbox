from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from complexity_error_audit import (
    AuditTables,
    add_empirical_strata,
    compute_clip_complexity,
    correlation_summary,
    distribution_summary,
    error_by_complexity,
    quartile_effect_summary,
    regime_summary,
)


def make_tables() -> AuditTables:
    frame_rows = []
    token_rows = []
    for frame_idx in range(16):
        schema = "old_maya" if frame_idx < 8 else "chonglu"
        name = f"{schema}/clip_{frame_idx // 4}"
        speed = 0.0001 + frame_idx * 0.0004
        row = {
            "dataset_idx": frame_idx,
            "name": name,
            "schema": schema,
            "label_kind": "default",
            "token_frame": frame_idx,
            "fk_speed": speed,
            "omega": 0.13 + frame_idx * 0.01,
            "energy": 1.0 + frame_idx,
            "teacher_nll": 0.2 + frame_idx * 0.02,
            "teacher_acc": 0.9 - frame_idx * 0.01,
            "generated_acc": 0.8 - frame_idx * 0.01,
        }
        for part_idx, part in enumerate(("upper", "lower", "feet", "hands")):
            row[f"omega_{part}"] = row["omega"] + part_idx * 0.001
            row[f"energy_{part}"] = row["energy"] + part_idx
        frame_rows.append(row)
        for part_idx, part in enumerate(("upper", "lower", "feet", "hands")):
            for quantizer in range(4):
                token_rows.append(
                    {
                        "dataset_idx": frame_idx,
                        "name": name,
                        "schema": schema,
                        "label_kind": "default",
                        "token_frame": frame_idx,
                        "part": part,
                        "quantizer": quantizer,
                        "teacher_nll": row["teacher_nll"] + quantizer * 0.01,
                        "teacher_correct": frame_idx < 12,
                        "generated_correct": frame_idx < 10,
                        "fk_speed": speed,
                        "omega": row["omega"],
                        "energy": row["energy"],
                        "part_omega": row[f"omega_{part}"],
                        "part_energy": row[f"energy_{part}"],
                    }
                )
    return AuditTables(
        frames=pd.DataFrame(frame_rows),
        tokens=pd.DataFrame(token_rows),
        clips=pd.DataFrame(),
        failures=pd.DataFrame(columns=["name", "stage", "error"]),
        metadata={},
    )


def test_clip_complexity_shapes():
    torch.manual_seed(7)
    tokens = torch.randint(0, 8, (12, 16))
    codebooks = {
        part: torch.randn(4, 8, 6)
        for part in ("upper", "lower", "feet", "hands")
    }
    result = compute_clip_complexity(tokens, codebooks, window=4)
    assert result.phi.shape == (12, 4)
    assert result.energy.shape == (12,)
    assert set(result.part_phi) == {"upper", "lower", "feet", "hands"}
    assert all(value.shape == (12,) for value in result.part_energy.values())


def test_strata_are_idempotent_and_summaries_run():
    once = add_empirical_strata(make_tables())
    twice = add_empirical_strata(once)
    assert not any(column.endswith(("_x", "_y")) for column in twice.tokens.columns)
    assert set(twice.frames["speed_regime"]) == {"still_noise", "moving"}
    assert twice.tokens["part_omega_quartile"].notna().all()
    assert not distribution_summary(twice.frames).empty
    assert np.isclose(regime_summary(twice.frames).groupby("schema")["fraction"].sum(), 1.0).all()
    assert not error_by_complexity(twice.tokens).empty
    assert not quartile_effect_summary(twice.tokens).empty
    assert not correlation_summary(twice.tokens, iterations=5).empty


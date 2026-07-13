from __future__ import annotations

import torch

from residual_recoverability_audit import (
    beam_residual_tail,
    build_recovery_variants,
    greedy_residual_tail,
    tokens_to_part_latent,
)


def test_greedy_tail_can_cancel_a_fixed_q0_error():
    codebooks = torch.tensor(
        [
            [[0.0], [2.0]],
            [[0.0], [-1.0]],
            [[0.0], [-1.0]],
        ]
    )
    target = torch.tensor([[0.0]])
    fixed_q0 = torch.tensor([1])

    codes, latent = greedy_residual_tail(target, fixed_q0, codebooks)

    assert codes.tolist() == [[1, 1, 1]]
    assert torch.allclose(latent, target)


def test_beam_tail_can_outperform_greedy_tail():
    codebooks = torch.tensor(
        [
            [[0.0], [5.0]],
            [[0.1], [1.0]],
            [[-1.0], [10.0]],
        ]
    )
    target = torch.tensor([[0.0]])
    fixed_q0 = torch.tensor([0])

    greedy_codes, greedy_latent = greedy_residual_tail(target, fixed_q0, codebooks)
    beam_codes, beam_latent = beam_residual_tail(target, fixed_q0, codebooks, beam_width=2)

    assert greedy_codes.tolist() == [[0, 0, 0]]
    assert beam_codes.tolist() == [[0, 1, 0]]
    assert torch.linalg.vector_norm(beam_latent - target) < torch.linalg.vector_norm(
        greedy_latent - target
    )


def test_build_variants_only_changes_middle_frames_and_preserves_predicted_q0():
    parts = ("upper", "lower")
    codebooks = {
        part: torch.tensor(
            [
                [[0.0], [2.0]],
                [[0.0], [-1.0]],
                [[0.0], [-1.0]],
            ]
        )
        for part in parts
    }
    gt = torch.zeros((1, 3, 6), dtype=torch.long)
    predicted = gt.clone()
    predicted[:, 1, 0] = 1
    predicted[:, 1, 3] = 1
    predicted[:, 1, 1:] = 1

    variants = build_recovery_variants(
        gt,
        predicted,
        codebooks,
        part_order=parts,
        beam_width=2,
    )

    for tokens in variants.values():
        assert torch.equal(tokens[:, 0], gt[:, 0])
        assert torch.equal(tokens[:, -1], gt[:, -1])
    for start in (0, 3):
        assert int(variants["pred_q0_gt_tail"][0, 1, start]) == 1
        assert int(variants["pred_q0_greedy_tail"][0, 1, start]) == 1
        assert int(variants["pred_q0_beam_tail"][0, 1, start]) == 1
        recovered = tokens_to_part_latent(
            variants["pred_q0_greedy_tail"][0, 1, start : start + 3],
            codebooks[parts[start // 3]],
        )
        assert torch.allclose(recovered, torch.zeros_like(recovered))

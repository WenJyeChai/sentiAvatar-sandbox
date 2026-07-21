from __future__ import annotations

import math

import torch

from train_step1_multipart_fixed_gap3 import load_training_state


def test_training_state_restores_early_stopping_fields(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": 12,
            "epoch": 3,
            "batch_in_epoch": 0,
            "best_eval_loss": 4.25,
            "epochs_without_improvement": 2,
        },
        checkpoint / "training_state.pt",
    )

    state = load_training_state(checkpoint, optimizer, scheduler, torch.device("cpu"))
    assert state == (12, 3, 0, 4.25, 2)


def test_old_training_state_is_backward_compatible(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "old-checkpoint"
    checkpoint.mkdir()
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": 7,
            "epoch": 1,
            "batch_in_epoch": 4,
        },
        checkpoint / "training_state.pt",
    )

    global_step, epoch, batch, best_loss, stale_epochs = load_training_state(
        checkpoint, optimizer, scheduler, torch.device("cpu")
    )
    assert (global_step, epoch, batch, stale_epochs) == (7, 1, 4, 0)
    assert math.isinf(best_loss)

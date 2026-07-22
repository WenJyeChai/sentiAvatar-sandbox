from __future__ import annotations

import math

import torch

from train_step1_multipart_fixed_gap3 import (
    auxiliary_weight_at_epoch,
    load_training_state,
    validate_auxiliary_loss_config,
)


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
            "curriculum_activation_epoch": 5,
            "best_rollout_accuracy": 0.031,
        },
        checkpoint / "training_state.pt",
    )

    state = load_training_state(checkpoint, optimizer, scheduler, torch.device("cpu"))
    assert state == (12, 3, 0, 4.25, 2, 5, 0.031)


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

    (
        global_step,
        epoch,
        batch,
        best_loss,
        stale_epochs,
        activation_epoch,
        best_rollout,
    ) = load_training_state(
        checkpoint, optimizer, scheduler, torch.device("cpu")
    )
    assert (global_step, epoch, batch, stale_epochs) == (7, 1, 4, 0)
    assert math.isinf(best_loss)
    assert activation_epoch is None
    assert math.isinf(best_rollout) and best_rollout < 0


def test_auxiliary_weight_ramps_without_changing_control() -> None:
    assert auxiliary_weight_at_epoch(0.5, maximum=0.1, warmup_epochs=1.0) == 0.05
    assert auxiliary_weight_at_epoch(2.0, maximum=0.1, warmup_epochs=1.0) == 0.1
    assert auxiliary_weight_at_epoch(2.0, maximum=0.0, warmup_epochs=1.0) == 0.0


def test_auxiliary_control_defaults_to_disabled() -> None:
    config = validate_auxiliary_loss_config({})
    assert config["type"] == "none"
    assert config["weight"] == 0.0

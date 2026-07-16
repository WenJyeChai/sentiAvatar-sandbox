#!/usr/bin/env python3
"""Create a curated W&B workspace for variable-gap C2F experiments."""

from __future__ import annotations

import argparse


PARTS = ("upper", "lower", "feet", "hands")
QUANTIZERS = range(4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True, help="W&B user or team name")
    parser.add_argument("--project", default="sentiavatar-variable-c2f")
    parser.add_argument("--name", default="Variable C2F Monitoring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise ImportError(
            "This utility requires wandb-workspaces. Install it with "
            "`pip install wandb-workspaces`."
        ) from exc

    def line(title: str, metrics: list[str]):
        return wr.LinePlot(
            title=title,
            x="Step",
            y=metrics,
            smoothing_type="exponentialTimeWeighted",
            smoothing_factor=0.2,
            max_runs_to_show=20,
        )

    train = "train/train_c2f"
    evaluation = "eval/c2f"
    sections = [
        ws.Section(
            name="01 Optimization",
            panels=[
                line("Train and evaluation loss", ["train/loss", "eval/loss"]),
                line("Learning rate", ["train/learning_rate"]),
                line("Gradient norm", ["train/grad_norm"]),
                line(
                    "Self-forcing and recovery exposure",
                    [
                        f"{train}/self_forcing_probability",
                        f"{train}/self_forced_batch",
                        f"{train}/soft_recovery_batch",
                        f"{train}/adaptive_target_batch",
                    ],
                ),
                line("Sampled gap length", [f"{train}/gap_mean"]),
            ],
            is_open=True,
            pinned=True,
        ),
        ws.Section(
            name="02 C2F Stages",
            panels=[
                line(
                    "Train CE by quantizer",
                    [f"{train}/q{q}_ce" for q in QUANTIZERS],
                ),
                line(
                    "Evaluation CE by quantizer",
                    [f"{evaluation}/q{q}_ce" for q in QUANTIZERS],
                ),
                line(
                    "Train original-token accuracy",
                    [f"{train}/q{q}_original_acc" for q in QUANTIZERS],
                ),
                line(
                    "Evaluation original-token accuracy",
                    [f"{evaluation}/q{q}_original_acc" for q in QUANTIZERS],
                ),
                line(
                    "Train embedding loss",
                    [f"{train}/q{q}_embed" for q in QUANTIZERS],
                ),
                line(
                    "Evaluation embedding loss",
                    [f"{evaluation}/q{q}_embed" for q in QUANTIZERS],
                ),
                line(
                    "Soft expected final latent",
                    [f"{train}/q3_final_latent", f"{evaluation}/q3_final_latent"],
                ),
                line(
                    "Final rollout token accuracy",
                    [f"{evaluation}/final_original_acc"],
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="03 Soft Recovery",
            panels=[
                line(
                    "Train soft-recovery loss",
                    [f"{train}/q{q}_soft_recovery" for q in range(1, 4)],
                ),
                line(
                    "Evaluation soft-recovery loss",
                    [f"{evaluation}/q{q}_soft_recovery" for q in range(1, 4)],
                ),
                line(
                    "Recovery-pool entropy",
                    [f"{train}/q{q}_recovery_pool_entropy" for q in range(1, 4)],
                ),
                line(
                    "Nearest recovery equals original",
                    [
                        f"{train}/q{q}_recovery_top1_original_rate"
                        for q in range(1, 4)
                    ],
                ),
                line(
                    "Recovery samples per logged batch",
                    [f"{train}/q{q}_recovery_samples" for q in range(1, 4)],
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="04 Hard Latent Quality",
            panels=[
                line("Hard latent RMSE", [f"{evaluation}/hard_latent_rmse"]),
                line(
                    "Hard latent RMSE by q0 outcome",
                    [
                        f"{evaluation}/hard_latent_rmse_q0_correct",
                        f"{evaluation}/hard_latent_rmse_q0_wrong",
                    ],
                ),
                line(
                    "Hard latent RMSE by gap bucket",
                    [
                        f"{evaluation}/hard_latent_rmse_gap_1_3",
                        f"{evaluation}/hard_latent_rmse_gap_4_7",
                        f"{evaluation}/hard_latent_rmse_gap_8_15",
                    ],
                ),
                line(
                    "Hard latent RMSE by body part",
                    [f"{evaluation}/{part}_hard_latent_rmse" for part in PARTS],
                ),
            ],
            is_open=True,
            pinned=True,
        ),
        ws.Section(
            name="05 Gap Accuracy",
            panels=[
                line(
                    "Short gaps",
                    [f"{evaluation}/gap_{gap}_original_acc" for gap in range(1, 4)],
                ),
                line(
                    "Medium gaps",
                    [f"{evaluation}/gap_{gap}_original_acc" for gap in range(4, 8)],
                ),
                line(
                    "Long gaps",
                    [f"{evaluation}/gap_{gap}_original_acc" for gap in range(8, 16)],
                ),
            ],
            is_open=False,
        ),
        ws.Section(
            name="06 Part Accuracy",
            panels=[
                line(
                    f"Evaluation q{q} accuracy by part",
                    [f"{evaluation}/{part}_q{q}_adaptive_acc" for part in PARTS],
                )
                for q in QUANTIZERS
            ],
            is_open=False,
        ),
    ]

    workspace = ws.Workspace(
        name=args.name,
        entity=args.entity,
        project=args.project,
        sections=sections,
        settings=ws.WorkspaceSettings(
            x_axis="Step",
            smoothing_type="exponentialTimeWeighted",
            smoothing_weight=2,
            group_by_prefix="first",
            sort_panels_alphabetically=False,
            max_runs=20,
        ),
        auto_generate_panels=False,
    )
    workspace.save()
    print(f"Saved W&B workspace: {workspace.url}")


if __name__ == "__main__":
    main()

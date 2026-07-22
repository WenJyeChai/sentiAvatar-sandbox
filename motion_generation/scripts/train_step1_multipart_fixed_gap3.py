#!/usr/bin/env python3
"""Train the Phase 1 causal Mimi -> fixed-gap multipart anchor planner."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# The tokenizer is constructed in the parent process and then inherited by
# DataLoader workers. Explicitly disabling its internal thread pool avoids the
# Hugging Face post-fork warning; DataLoader workers still provide parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.step1_mimi_planner import (  # noqa: E402
    MimiQwenPlanner,
    MimiQwenPlannerConfig,
    Step1FixedGapDataset,
    Step1PlannerCollator,
    load_text_map,
    read_split_names,
)
from utils.adaptive_anchor_tokens import (  # noqa: E402
    BODY_SLOTS,
    MIMI_FRAME_TOKEN,
    ensure_step1_special_tokens,
    motion_token_id_table,
    validate_anchor,
)
from utils.step1_self_forcing import (  # noqa: E402
    GeneratedHistoryBatchStats,
    apply_generated_history,
    deterministic_generated_indices,
    generated_history_probability,
    rollout_quality_metrics,
    validate_generated_labels,
)
from utils.step1_condition_alignment import (  # noqa: E402
    ConditionCorruption,
    corrupt_audio_with_causal_past,
    corrupt_text_condition,
    counterfactual_likelihood_loss,
    deterministic_condition_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed [gap_3] multipart Step 1 planner")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "motion_generation" / "configs" / "step1_multipart_fixed_gap3.yaml",
    )
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init_from_checkpoint",
        type=Path,
        default=None,
        help="Load planner/tokenizer weights but start a fresh optimizer, schedule, and epoch count.",
    )
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_eval_clips", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return dict(value)


def mimi_codebooks_from_config(config: Mapping[str, Any]) -> list[int]:
    audio = section(config, "audio")
    data = section(config, "data")
    values = audio.get("mimi_codebooks_used", data.get("mimi_codebooks_used", [0]))
    if not isinstance(values, list) or not values:
        raise ValueError("audio.mimi_codebooks_used must be a non-empty list")
    codebooks = [int(value) for value in values]
    if codebooks[0] != 0:
        raise ValueError("audio.mimi_codebooks_used must begin with q0")
    if len(set(codebooks)) != len(codebooks) or any(not 0 <= value < 8 for value in codebooks):
        raise ValueError("audio.mimi_codebooks_used must contain unique indices in [0, 7]")
    if "mimi_codebooks_used" in audio and "mimi_codebooks_used" in data:
        data_codebooks = [int(value) for value in data["mimi_codebooks_used"]]
        if data_codebooks != codebooks:
            raise ValueError("audio and data Mimi codebook configurations disagree")
    if len(codebooks) > 1:
        if str(audio.get("fusion", "concat_linear")) != "concat_linear":
            raise ValueError("q0-q3 currently supports only concat_linear audio fusion")
        if not bool(audio.get("sparse_audio_embedding", True)):
            raise ValueError("q0-q3 requires sparse_audio_embedding on 24 GB GPUs")
    return codebooks


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Phase 1 training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return distributed, rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_neutral_seed(path_value: Optional[str]) -> Optional[tuple[int, ...]]:
    if not path_value:
        return None
    path = project_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tokens", payload.get("anchor_tokens"))
    if not isinstance(payload, list):
        raise ValueError(f"Neutral seed JSON must contain a 16-id list: {path}")
    tokens = tuple(int(value) for value in payload)
    validate_anchor(tokens)
    return tokens


def resolve_data_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = section(config, "paths")
    data_dir = project_path(paths.get("data_dir", "SuSuInterActs/SuSuInterActs"))
    return {
        "base_model": project_path(paths.get("base_model", "checkpoints/llm")),
        "output_dir": project_path(paths.get("output_dir", "checkpoints/step1_multipart_fixed_gap3")),
        "data_dir": data_dir,
        "motion_token_dir": project_path(
            paths.get("motion_token_dir", data_dir / "motion_token_data_multipart_causal_512x4")
        ),
        "mimi_token_dir": project_path(
            paths.get("mimi_token_dir", data_dir / "audio_tokens_mimi_12p5hz_8cb")
        ),
        "text_json": project_path(paths.get("text_json", data_dir / "text_data/motion2text.json")),
        "train_split": project_path(paths.get("train_split", data_dir / "split/train_file_list.txt")),
        "eval_split": project_path(paths.get("eval_split", data_dir / "split/val_file_list.txt")),
    }


def validate_paths(paths: Mapping[str, Path], resume: Optional[Path]) -> None:
    files = ("text_json", "train_split", "eval_split")
    directories = ("motion_token_dir", "mimi_token_dir")
    if resume is None and not paths["base_model"].is_dir():
        raise FileNotFoundError(f"Base Qwen checkpoint not found: {paths['base_model']}")
    for key in files:
        if not paths[key].is_file():
            raise FileNotFoundError(f"Required {key} not found: {paths[key]}")
    for key in directories:
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Required {key} not found: {paths[key]}")


def build_model_and_tokenizer(
    *,
    base_model: Path,
    resume: Optional[Path],
    dtype: torch.dtype,
    mimi_codebooks_used: list[int],
) -> tuple[MimiQwenPlanner, Any, list[str]]:
    if resume is not None:
        resume = resume.resolve()
        tokenizer = AutoTokenizer.from_pretrained(resume, local_files_only=True)
        model = MimiQwenPlanner.from_pretrained(
            resume,
            torch_dtype=dtype,
            local_files_only=True,
        )
        added = ensure_step1_special_tokens(tokenizer)
        if added:
            raise RuntimeError(f"Resume checkpoint is missing Step 1 controls: {added}")
        expected_table = torch.tensor(motion_token_id_table(tokenizer), dtype=torch.long)
        if not torch.equal(model.motion_token_ids.cpu(), expected_table):
            raise RuntimeError("Resume checkpoint motion-token classifier table does not match tokenizer")
        if list(model.config.mimi_codebooks_used) != list(mimi_codebooks_used):
            raise RuntimeError(
                "Resume checkpoint Mimi codebooks do not match the requested configuration: "
                f"checkpoint={model.config.mimi_codebooks_used}, requested={mimi_codebooks_used}"
            )
        return model, tokenizer, []

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    language_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
    )
    added = ensure_step1_special_tokens(tokenizer, language_model)
    language_model.config.use_cache = False
    table = motion_token_id_table(tokenizer)
    audio_placeholder_id = int(tokenizer.convert_tokens_to_ids(MIMI_FRAME_TOKEN))
    planner_config = MimiQwenPlannerConfig(
        language_model_config=language_model.config.to_dict(),
        audio_placeholder_id=audio_placeholder_id,
        motion_token_ids=table,
        mimi_cardinality=2_048,
        mimi_codebooks_stored=8,
        mimi_codebooks_used=mimi_codebooks_used,
    )
    model = MimiQwenPlanner(planner_config, language_model=language_model)
    model.tie_weights()
    return model, tokenizer, added


def build_dataset(
    names: list[str],
    *,
    tokenizer: Any,
    paths: Mapping[str, Path],
    text_map: Mapping[str, str],
    data_config: Mapping[str, Any],
    neutral_seed: Optional[tuple[int, ...]],
    training: bool,
) -> Step1FixedGapDataset:
    generated_anchor_dir_value = data_config.get("generated_anchor_dir") if training else None
    generated_anchor_dir = project_path(generated_anchor_dir_value) if generated_anchor_dir_value else None
    return Step1FixedGapDataset(
        names,
        tokenizer=tokenizer,
        motion_token_dir=paths["motion_token_dir"],
        mimi_token_dir=paths["mimi_token_dir"],
        text_map=text_map,
        fixed_gap=int(data_config.get("fixed_gap", 3)),
        max_length=int(data_config.get("max_length", 2_048)),
        seed_mode=str(data_config.get("seed_mode", "observed")),
        neutral_seed_tokens=neutral_seed,
        neutral_seed_probability=float(data_config.get("neutral_seed_probability", 0.5)),
        previous_seed_probability=float(data_config.get("previous_seed_probability", 0.5)),
        generated_anchor_dir=generated_anchor_dir,
        generated_prefix_probability=(
            float(data_config.get("generated_prefix_probability", 0.0)) if training else 0.0
        ),
        random_seed=int(data_config.get("random_seed", 42)),
        require_causal_motion=bool(data_config.get("require_causal_motion", True)),
        max_duration_mismatch_seconds=float(data_config.get("max_duration_mismatch_seconds", 0.12)),
        mimi_codebooks_used=data_config.get("mimi_codebooks_used", [0]),
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "attention_mask", "audio_codes", "target_slots", "motion_local_labels")
    return {key: batch[key].to(device=device, non_blocking=True) for key in keys}


def move_condition_metadata(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    keys = ("text_mask", "audio_anchor_ids", "target_anchor_ids")
    return {key: batch[key].to(device=device, non_blocking=True) for key in keys}


def corrupted_model_inputs(
    inputs: Mapping[str, torch.Tensor], corruption: ConditionCorruption
) -> dict[str, torch.Tensor]:
    indices = corruption.selected_indices
    result = {
        key: value.index_select(0, indices)
        for key, value in inputs.items()
    }
    result["input_ids"] = corruption.input_ids.index_select(0, indices)
    result["audio_codes"] = corruption.audio_codes.index_select(0, indices)
    return result


def reduce_sums(values: torch.Tensor, distributed: bool) -> torch.Tensor:
    values = values.detach()
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    distributed: bool,
    use_bf16: bool,
    condition_alignment: Optional[Mapping[str, Any]] = None,
    alignment_seed: int = 42,
) -> dict[str, Any]:
    model.eval()
    base_count = 3 + 2 * len(BODY_SLOTS)
    # Base metrics followed by audio gap/count and text gap/count.
    totals = torch.zeros(base_count + 4, dtype=torch.float64, device=device)
    autocast_enabled = use_bf16 and device.type == "cuda"
    alignment_enabled = bool(
        condition_alignment is not None and condition_alignment.get("evaluate", False)
    )
    for batch_index, batch in enumerate(loader):
        inputs = move_batch(batch, device)
        metadata = move_condition_metadata(batch, device) if alignment_enabled else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model(**inputs, return_token_losses=alignment_enabled)
        count = output.count.to(torch.float64)
        totals[0] += output.loss.detach().to(torch.float64) * count
        totals[1] += output.correct.to(torch.float64)
        totals[2] += count
        totals[3 : 3 + len(BODY_SLOTS)] += output.per_slot_correct.to(torch.float64)
        totals[3 + len(BODY_SLOTS) : base_count] += output.per_slot_count.to(torch.float64)
        if alignment_enabled:
            assert condition_alignment is not None and metadata is not None
            selected_indices = deterministic_condition_indices(
                batch["names"],
                float(condition_alignment["eval_fraction"]),
                seed=alignment_seed,
                epoch=0,
                batch_index=batch_index,
            )
            for modality in condition_alignment["eval_modalities"]:
                corruption = build_condition_corruption(
                    modality,
                    inputs=inputs,
                    metadata=metadata,
                    names=batch["names"],
                    selected_indices=selected_indices,
                    alignment=condition_alignment,
                    seed=alignment_seed,
                    epoch=0,
                    batch_index=batch_index,
                )
                if not corruption.selected_indices.numel():
                    continue
                negative_inputs = corrupted_model_inputs(inputs, corruption)
                selected_mask = corruption.target_mask.index_select(
                    0, corruption.selected_indices
                )
                positive_loss = output.per_token_loss.index_select(
                    0, corruption.selected_indices
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    negative_output = model(
                        **negative_inputs, return_token_losses=True
                    )
                    _, gap = counterfactual_likelihood_loss(
                        positive_token_loss=positive_loss,
                        negative_token_loss=negative_output.per_token_loss,
                        target_mask=selected_mask,
                        margin_nats=float(condition_alignment["margin_nats"]),
                    )
                offset = base_count + (0 if modality == "audio" else 2)
                totals[offset] += gap.to(torch.float64).sum()
                totals[offset + 1] += gap.numel()
    totals = reduce_sums(totals, distributed)
    model.train()
    denominator = max(1.0, float(totals[2]))
    slot_correct = totals[3 : 3 + len(BODY_SLOTS)]
    slot_count = totals[3 + len(BODY_SLOTS) : base_count]
    per_slot = {}
    for slot, spec in enumerate(BODY_SLOTS):
        per_slot[f"{spec.part}_q{spec.quantizer}"] = (
            float(slot_correct[slot]) / max(1.0, float(slot_count[slot]))
        )
    condition_gaps = {}
    for modality, offset in (("audio", base_count), ("text", base_count + 2)):
        if float(totals[offset + 1]) > 0:
            condition_gaps[modality] = float(totals[offset]) / float(totals[offset + 1])
    return {
        "loss": float(totals[0]) / denominator,
        "slot_accuracy": float(totals[1]) / denominator,
        "per_slot_accuracy": per_slot,
        "condition_gap_nats_per_token": condition_gaps,
    }


def optimizer_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    output_dir: Path,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    source_config: Mapping[str, Any],
    distributed: bool,
    rank: int,
    final: bool = False,
    checkpoint_name: Optional[str] = None,
    best_eval_loss: float = math.inf,
    epochs_without_improvement: int = 0,
    curriculum_activation_epoch: Optional[int] = None,
    best_rollout_accuracy: float = -math.inf,
) -> Path:
    if final and checkpoint_name is not None:
        raise ValueError("final and checkpoint_name cannot both be set")
    checkpoint_dir = output_dir / (
        checkpoint_name or ("final" if final else f"checkpoint-{global_step}")
    )
    if is_main(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "best_eval_loss": float(best_eval_loss),
                "epochs_without_improvement": int(epochs_without_improvement),
                "curriculum_activation_epoch": curriculum_activation_epoch,
                "best_rollout_accuracy": float(best_rollout_accuracy),
            },
            checkpoint_dir / "training_state.pt",
        )
        (checkpoint_dir / "phase1_source_config.json").write_text(
            json.dumps(source_config, indent=2), encoding="utf-8"
        )
        (output_dir / "latest_checkpoint.txt").write_text(str(checkpoint_dir), encoding="utf-8")
        print(f"Saved checkpoint: {checkpoint_dir}")
    barrier(distributed)
    return checkpoint_dir


def load_training_state(
    checkpoint: Optional[Path],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> tuple[int, int, int, float, int, Optional[int], float]:
    if checkpoint is None:
        return 0, 0, 0, math.inf, 0, None, -math.inf
    state_path = checkpoint.resolve() / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume training state not found: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(state["scheduler"])
    return (
        int(state["global_step"]),
        int(state["epoch"]),
        int(state["batch_in_epoch"]),
        float(state.get("best_eval_loss", math.inf)),
        int(state.get("epochs_without_improvement", 0)),
        (
            int(state["curriculum_activation_epoch"])
            if state.get("curriculum_activation_epoch") is not None
            else None
        ),
        float(state.get("best_rollout_accuracy", -math.inf)),
    )


def initialize_wandb(
    config: Mapping[str, Any], *, rank: int, output_dir: Path
):
    monitoring = section(config, "monitoring")
    if not is_main(rank) or str(monitoring.get("report_to", "none")) != "wandb":
        return None
    import wandb  # pylint: disable=import-outside-toplevel

    mode = str(monitoring.get("wandb_mode", "online"))
    return wandb.init(
        project=str(monitoring.get("wandb_project", "sentiavatar-step1")),
        entity=monitoring.get("wandb_entity"),
        name=str(monitoring.get("wandb_run_name", "step1_multipart_fixed_gap3")),
        tags=list(monitoring.get("wandb_tags", ["phase1", "step1", "mimi", "fixed-gap3"])),
        mode=mode,
        dir=str(output_dir),
        config=dict(config),
    )


def validate_generated_history_config(config: Mapping[str, Any]) -> dict[str, Any]:
    generated = section(config, "generated_history")
    generated.setdefault("enabled", False)
    generated.setdefault("decoding", "greedy")
    generated.setdefault("minimum_teacher_epochs", 5)
    generated.setdefault("ramp_epochs", 10)
    generated.setdefault("max_probability", 0.5)
    generated.setdefault("rollout_microbatch_size", 8)
    generated.setdefault("rollout_eval_clips", 64)
    generated.setdefault("teacher_forced_ce_max", 5.4)
    generated.setdefault("rollout_accuracy_min", 0.01)
    generated.setdefault("rollout_q0_accuracy_min", 0.03)
    generated.setdefault("milestone_epochs", [5, 15, 25, 35, 50])
    generated.setdefault("rollout_eval_epochs", [5, 15, 25, 35, 50])
    generated.setdefault("start_immediately", False)
    if generated["decoding"] != "greedy":
        raise ValueError("Only greedy generated-history decoding is currently supported")
    for key in ("minimum_teacher_epochs", "ramp_epochs"):
        generated[key] = int(generated[key])
        if generated[key] < 0:
            raise ValueError(f"generated_history.{key} must be non-negative")
    for key in ("rollout_microbatch_size", "rollout_eval_clips"):
        generated[key] = int(generated[key])
        if generated[key] <= 0:
            raise ValueError(f"generated_history.{key} must be positive")
    generated["start_immediately"] = bool(generated["start_immediately"])
    generated["max_probability"] = float(generated["max_probability"])
    if not 0 <= generated["max_probability"] <= 1:
        raise ValueError("generated_history.max_probability must be in [0, 1]")
    for key in ("teacher_forced_ce_max", "rollout_accuracy_min", "rollout_q0_accuracy_min"):
        generated[key] = float(generated[key])
    for key in ("milestone_epochs", "rollout_eval_epochs"):
        values = [int(value) for value in generated[key]]
        if any(value <= 0 for value in values):
            raise ValueError(f"generated_history.{key} must contain positive epochs")
        generated[key] = sorted(set(values))
    return generated


def validate_auxiliary_loss_config(config: Mapping[str, Any]) -> dict[str, Any]:
    auxiliary = section(config, "auxiliary_loss")
    auxiliary.setdefault("type", "none")
    auxiliary.setdefault("weight", 0.0)
    auxiliary.setdefault("warmup_epochs", 0.0)
    auxiliary.setdefault("apply_to", "clean")
    auxiliary_type = str(auxiliary["type"])
    if auxiliary_type not in {"none", "expected_distortion"}:
        raise ValueError("auxiliary_loss.type must be 'none' or 'expected_distortion'")
    auxiliary["weight"] = float(auxiliary["weight"])
    auxiliary["warmup_epochs"] = float(auxiliary["warmup_epochs"])
    if auxiliary["weight"] < 0 or auxiliary["warmup_epochs"] < 0:
        raise ValueError("auxiliary loss weight and warmup_epochs must be non-negative")
    if auxiliary_type == "none":
        auxiliary["weight"] = 0.0
    elif auxiliary["weight"] <= 0:
        raise ValueError("expected_distortion requires a positive auxiliary_loss.weight")
    if auxiliary["apply_to"] not in {"clean", "all"}:
        raise ValueError("auxiliary_loss.apply_to must be 'clean' or 'all'")
    checkpoints = auxiliary.get("codec_checkpoints", {})
    if auxiliary_type == "expected_distortion":
        if not isinstance(checkpoints, dict):
            raise ValueError("auxiliary_loss.codec_checkpoints must be a mapping")
        missing = [part for part in ("upper", "lower", "feet", "hands") if part not in checkpoints]
        if missing:
            raise ValueError(f"Missing auxiliary codec checkpoints: {missing}")
        auxiliary["codec_checkpoints"] = {
            part: project_path(checkpoints[part]) for part in ("upper", "lower", "feet", "hands")
        }
    return auxiliary


def validate_condition_alignment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    alignment = section(config, "condition_alignment")
    alignment.setdefault("enabled", False)
    alignment.setdefault("evaluate", alignment["enabled"])
    alignment.setdefault("modalities", [])
    alignment.setdefault("eval_modalities", alignment["modalities"])
    alignment.setdefault("sub_batch_fraction", 0.25)
    alignment.setdefault("eval_fraction", 0.25)
    alignment.setdefault("weight", 0.03)
    alignment.setdefault("margin_nats", 0.05)
    alignment.setdefault("warmup_start_epoch", 5.0)
    alignment.setdefault("ramp_epochs", 3.0)
    alignment.setdefault("audio_past_shift_anchors", 2)
    alignment["enabled"] = bool(alignment["enabled"])
    alignment["evaluate"] = bool(alignment["evaluate"])
    modalities = [str(value) for value in alignment["modalities"]]
    eval_modalities = [str(value) for value in alignment["eval_modalities"]]
    for key, values in (("modalities", modalities), ("eval_modalities", eval_modalities)):
        unknown = sorted(set(values) - {"audio", "text"})
        if unknown or len(values) != len(set(values)):
            raise ValueError(f"Invalid or duplicate condition_alignment {key}: {values}")
    if alignment["enabled"] and not modalities:
        raise ValueError("Enabled condition alignment requires training modalities")
    if alignment["evaluate"] and not eval_modalities:
        raise ValueError("Condition alignment evaluation requires eval_modalities")
    alignment["modalities"] = modalities
    alignment["eval_modalities"] = eval_modalities
    for key in ("sub_batch_fraction", "eval_fraction"):
        alignment[key] = float(alignment[key])
        if not 0 < alignment[key] <= 1:
            raise ValueError(f"condition_alignment.{key} must be in (0, 1]")
    for key in ("weight", "margin_nats", "warmup_start_epoch", "ramp_epochs"):
        alignment[key] = float(alignment[key])
        if alignment[key] < 0:
            raise ValueError(f"condition_alignment.{key} must be non-negative")
    if alignment["enabled"] and alignment["weight"] <= 0:
        raise ValueError("Enabled condition_alignment requires a positive weight")
    alignment["audio_past_shift_anchors"] = int(alignment["audio_past_shift_anchors"])
    if alignment["audio_past_shift_anchors"] <= 0:
        raise ValueError("condition_alignment.audio_past_shift_anchors must be positive")
    return alignment


def condition_alignment_weight(epoch_progress: float, config: Mapping[str, Any]) -> float:
    if not bool(config["enabled"]):
        return 0.0
    start = float(config["warmup_start_epoch"])
    if epoch_progress <= start:
        return 0.0
    ramp = float(config["ramp_epochs"])
    progress = 1.0 if ramp <= 0 else min(1.0, (epoch_progress - start) / ramp)
    return float(config["weight"]) * max(0.0, progress)


def build_condition_corruption(
    modality: str,
    *,
    inputs: Mapping[str, torch.Tensor],
    metadata: Mapping[str, torch.Tensor],
    names: Sequence[str],
    selected_indices: Sequence[int],
    alignment: Mapping[str, Any],
    seed: int,
    epoch: int,
    batch_index: int,
) -> ConditionCorruption:
    if modality == "audio":
        return corrupt_audio_with_causal_past(
            input_ids=inputs["input_ids"],
            audio_codes=inputs["audio_codes"],
            audio_anchor_ids=metadata["audio_anchor_ids"],
            target_anchor_ids=metadata["target_anchor_ids"],
            selected_indices=selected_indices,
            shift_anchors=int(alignment["audio_past_shift_anchors"]),
        )
    if modality == "text":
        return corrupt_text_condition(
            input_ids=inputs["input_ids"],
            audio_codes=inputs["audio_codes"],
            text_mask=metadata["text_mask"],
            target_anchor_ids=metadata["target_anchor_ids"],
            names=names,
            selected_indices=selected_indices,
            seed=seed,
            epoch=epoch,
            batch_index=batch_index,
        )
    raise ValueError(f"Unsupported condition modality: {modality}")


def auxiliary_weight_at_epoch(
    epoch_progress: float, *, maximum: float, warmup_epochs: float
) -> float:
    if maximum <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return float(maximum)
    return float(maximum) * min(1.0, max(0.0, float(epoch_progress) / warmup_epochs))


def run_rank0_rollout_evaluation(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    *,
    device: torch.device,
    rank: int,
    distributed: bool,
    use_bf16: bool,
) -> GeneratedHistoryBatchStats:
    """Run rollout on rank zero, then broadcast its scalar metrics."""

    values = torch.zeros(8, dtype=torch.float64, device=device)
    if is_main(rank):
        if loader is None:
            raise ValueError("Rank zero rollout evaluation requires a loader")
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        was_training = unwrapped.training
        unwrapped.eval()
        stats = rollout_quality_metrics(
            unwrapped,
            loader,
            device=device,
            use_bf16=use_bf16,
        )
        unwrapped.train(was_training)
        values = torch.tensor(
            [
                stats.clips,
                stats.anchors,
                stats.tokens,
                stats.correct,
                stats.q0_correct,
                stats.q0_tokens,
                stats.confidence_sum,
                stats.entropy_sum,
            ],
            dtype=torch.float64,
            device=device,
        )
    if distributed:
        dist.broadcast(values, src=0)
    return GeneratedHistoryBatchStats(
        clips=int(values[0].item()),
        anchors=int(values[1].item()),
        tokens=int(values[2].item()),
        correct=int(values[3].item()),
        q0_correct=int(values[4].item()),
        q0_tokens=int(values[5].item()),
        confidence_sum=float(values[6].item()),
        entropy_sum=float(values[7].item()),
    )


def main() -> None:
    args = parse_args()
    if args.resume_from_checkpoint is not None and args.init_from_checkpoint is not None:
        raise ValueError("Choose either --resume_from_checkpoint or --init_from_checkpoint")
    config = load_config(args.config.resolve())
    paths = resolve_data_paths(config)
    if args.output_dir is not None:
        paths["output_dir"] = args.output_dir.resolve()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    training = section(config, "training")
    if args.num_train_epochs is not None:
        if args.num_train_epochs <= 0:
            raise ValueError("--num_train_epochs must be positive")
        training["num_train_epochs"] = int(args.num_train_epochs)
    data_config = section(config, "data")
    mimi_codebooks_used = mimi_codebooks_from_config(config)
    data_config["mimi_codebooks_used"] = mimi_codebooks_used
    generated_history = validate_generated_history_config(config)
    auxiliary_loss = validate_auxiliary_loss_config(config)
    condition_alignment = validate_condition_alignment_config(config)
    if bool(generated_history["enabled"]) and (
        data_config.get("generated_anchor_dir")
        or float(data_config.get("generated_prefix_probability", 0.0)) > 0
    ):
        raise ValueError(
            "On-policy generated_history cannot be mixed with the legacy disk-cache curriculum"
        )
    seed = int(training.get("seed", 42))
    seed_everything(seed, rank)
    resume = args.resume_from_checkpoint.resolve() if args.resume_from_checkpoint else None
    init_checkpoint = (
        args.init_from_checkpoint.resolve() if args.init_from_checkpoint else None
    )
    model_checkpoint = resume or init_checkpoint
    validate_paths(paths, model_checkpoint)

    use_bf16 = bool(training.get("bf16", True))
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but this CUDA device does not support it")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    if is_main(rank):
        print(f"Loading tokenizer/model on rank {rank}, device={device}, dtype={dtype}")
    model, tokenizer, added_tokens = build_model_and_tokenizer(
        base_model=paths["base_model"],
        resume=model_checkpoint,
        dtype=dtype,
        mimi_codebooks_used=mimi_codebooks_used,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.language_model.config.use_cache = False
    if auxiliary_loss["type"] == "expected_distortion":
        # The archived expected-distortion experiment depends on the full
        # causal-codec stack. Import it only when that archived loss is
        # explicitly selected so active CE/condition-alignment training and
        # evaluation do not require codec-only packages such as einops.
        from utils.step1_expected_distortion import (
            load_normalized_codebook_distance_table,
        )

        if is_main(rank):
            print("Loading frozen causal-codec geometry for expected-distortion loss")
        distances = load_normalized_codebook_distance_table(
            auxiliary_loss["codec_checkpoints"]
        )
        model.set_motion_codebook_distances(distances)
        del distances
    model.to(device)

    text_map = load_text_map(paths["text_json"])
    train_names = read_split_names(paths["train_split"])
    eval_names = read_split_names(paths["eval_split"])
    if args.max_train_clips is not None:
        train_names = train_names[: args.max_train_clips]
    if args.max_eval_clips is not None:
        eval_names = eval_names[: args.max_eval_clips]
    neutral_seed = load_neutral_seed(data_config.get("neutral_seed_json"))
    data_config["random_seed"] = seed
    train_dataset = build_dataset(
        train_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=neutral_seed,
        training=True,
    )
    eval_dataset = build_dataset(
        eval_names,
        tokenizer=tokenizer,
        paths=paths,
        text_map=text_map,
        data_config=data_config,
        neutral_seed=neutral_seed,
        training=False,
    )
    # Fail before DDP training if the serialization contract is broken.
    if len(train_dataset):
        first = train_dataset[0]
        if is_main(rank):
            print(
                "First serialized train example:",
                {
                    "name": first["name"],
                    "sequence_length": len(first["input_ids"]),
                    "anchor_times": first["anchor_times"],
                    "audio_boundaries": first["audio_boundaries"],
                    "supervised_tokens": sum(slot >= 0 for slot in first["target_slots"]),
                },
            )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    ) if distributed else None
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None
    collator = Step1PlannerCollator(tokenizer.pad_token_id, pad_to_multiple_of=8)
    workers = int(data_config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("per_device_train_batch_size", 2)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(data_config.get("persistent_workers", workers > 0)),
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training.get("per_device_eval_batch_size", 2)),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(data_config.get("persistent_workers", workers > 0)),
        collate_fn=collator,
    )

    rollout_eval_loader = None
    if is_main(rank) and bool(generated_history["enabled"]):
        rollout_count = min(int(generated_history["rollout_eval_clips"]), len(eval_dataset))
        rollout_eval_loader = DataLoader(
            Subset(eval_dataset, range(rollout_count)),
            batch_size=int(generated_history["rollout_microbatch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    accumulation = int(training.get("gradient_accumulation_steps", 8))
    epochs = int(training.get("num_train_epochs", 10))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * epochs
    warmup_steps = round(total_steps * float(training.get("warmup_ratio", 0.03)))
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(training.get("weight_decay", 0.01))),
        lr=float(training.get("learning_rate", 2e-5)),
        betas=tuple(float(v) for v in training.get("adam_betas", [0.9, 0.95])),
        eps=float(training.get("adam_epsilon", 1e-8)),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    (
        global_step,
        start_epoch,
        resume_batch,
        best_eval_loss,
        epochs_without_improvement,
        curriculum_activation_epoch,
        best_rollout_accuracy,
    ) = load_training_state(resume, optimizer, scheduler, device)
    if (
        resume is None
        and bool(generated_history["enabled"])
        and bool(generated_history["start_immediately"])
    ):
        # -1 makes epoch_progress=0 immediately exceed the activation point,
        # including when ramp_epochs=0 requests a fixed probability.
        curriculum_activation_epoch = -1

    if is_main(rank):
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print("=" * 76)
        print("Phase 1 fixed-gap multipart planner")
        print(f"Base/init/resume:  {model_checkpoint or paths['base_model']}")
        print(f"Output:            {paths['output_dir']}")
        print(f"Motion tokens:     {paths['motion_token_dir']}")
        print(f"Mimi tokens:       {paths['mimi_token_dir']}")
        print(f"Mimi codebooks:    {mimi_codebooks_used}")
        print(f"Train/eval clips:  {len(train_dataset)}/{len(eval_dataset)}")
        print(f"World size:        {world_size}")
        print(f"Added controls:    {added_tokens}")
        print(f"Parameters:        {trainable:,} trainable / {total:,} total")
        print(f"Updates:           {total_steps} ({updates_per_epoch}/epoch)")
        print(
            "Generated history: "
            f"enabled={bool(generated_history['enabled'])}, "
            f"activation_epoch={curriculum_activation_epoch}, "
            f"ramp={generated_history['ramp_epochs']} epochs, "
            f"p_max={generated_history['max_probability']}"
        )
        print(
            "Auxiliary loss:    "
            f"type={auxiliary_loss['type']}, weight={auxiliary_loss['weight']}, "
            f"warmup={auxiliary_loss['warmup_epochs']} epochs, "
            f"apply_to={auxiliary_loss['apply_to']}"
        )
        print(
            "Condition loss:    "
            f"enabled={condition_alignment['enabled']}, "
            f"evaluate={condition_alignment['evaluate']}, "
            f"modalities={condition_alignment['modalities']}, "
            f"eval_modalities={condition_alignment['eval_modalities']}, "
            f"weight={condition_alignment['weight']}, "
            f"warmup={condition_alignment['warmup_start_epoch']}+"
            f"{condition_alignment['ramp_epochs']} epochs"
        )
        print("=" * 76)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(config, rank=rank, output_dir=paths["output_dir"])
    log_steps = int(training.get("logging_steps", 20))
    eval_steps = int(training.get("eval_steps", 500))
    save_steps = int(training.get("save_steps", 500))
    save_best = bool(training.get("save_best", True))
    early_stopping_patience = int(training.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0))
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    autocast_enabled = use_bf16 and device.type == "cuda"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    base_metric_count = 3 + 2 * len(BODY_SLOTS)
    # clean loss/correct/count, generated loss/correct/count, eight detached
    # rollout statistics, then CE sum, expected-distortion sum/count.
    running = torch.zeros(base_metric_count + 17, dtype=torch.float64, device=device)
    # Counterfactual loss sum, correct-minus-corrupt log-likelihood gap sum,
    # example count, audio examples, text examples.
    alignment_running = torch.zeros(5, dtype=torch.float64, device=device)
    run_start = time.perf_counter()

    completed_epochs = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_batch_start = resume_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < epoch_batch_start:
                continue
            group_start = (batch_index // accumulation) * accumulation
            group_end = min(group_start + accumulation, len(train_loader))
            group_size = group_end - group_start
            should_step = batch_index + 1 == group_end
            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_step
                else contextlib.nullcontext()
            )
            inputs = move_batch(batch, device)
            condition_metadata = move_condition_metadata(batch, device)
            validate_generated_labels(inputs)
            epoch_progress = epoch + batch_index / max(1, len(train_loader))
            current_auxiliary_weight = auxiliary_weight_at_epoch(
                epoch_progress,
                maximum=float(auxiliary_loss["weight"]),
                warmup_epochs=float(auxiliary_loss["warmup_epochs"]),
            )
            current_condition_weight = condition_alignment_weight(
                epoch_progress, condition_alignment
            )
            condition_modality = None
            if current_condition_weight > 0:
                modalities = condition_alignment["modalities"]
                condition_modality = modalities[
                    (epoch * len(train_loader) + batch_index) % len(modalities)
                ]
            generated_probability = (
                generated_history_probability(
                    epoch_progress,
                    activation_epoch=curriculum_activation_epoch,
                    ramp_epochs=int(generated_history["ramp_epochs"]),
                    max_probability=float(generated_history["max_probability"]),
                )
                if bool(generated_history["enabled"])
                else 0.0
            )
            generated_indices = deterministic_generated_indices(
                batch["names"],
                generated_probability,
                seed=seed,
                epoch=epoch,
                batch_index=batch_index,
            )
            generated_mask = torch.zeros(
                inputs["input_ids"].shape[0], dtype=torch.bool, device=device
            )
            rollout_stats = GeneratedHistoryBatchStats()
            if generated_indices:
                generated_mask[generated_indices] = True
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                was_training = unwrapped.training
                unwrapped.eval()
                generated_input_ids, rollout_stats = apply_generated_history(
                    unwrapped,
                    inputs,
                    generated_indices,
                    microbatch_size=int(generated_history["rollout_microbatch_size"]),
                    use_bf16=use_bf16,
                )
                unwrapped.train(was_training)
                inputs["input_ids"] = generated_input_ids
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    auxiliary_example_mask = (
                        ~generated_mask
                        if auxiliary_loss["apply_to"] == "clean"
                        else None
                    )
                    output = model(
                        **inputs,
                        expected_distortion_weight=current_auxiliary_weight,
                        expected_distortion_example_mask=auxiliary_example_mask,
                        return_token_losses=current_condition_weight > 0,
                    )
                    scaled_loss = output.loss / group_size
                scaled_loss.backward()
            counterfactual_loss = None
            condition_gap = None
            condition_examples = 0
            if condition_modality is not None:
                selected_indices = deterministic_condition_indices(
                    batch["names"],
                    float(condition_alignment["sub_batch_fraction"]),
                    seed=seed,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                corruption = build_condition_corruption(
                    condition_modality,
                    inputs=inputs,
                    metadata=condition_metadata,
                    names=batch["names"],
                    selected_indices=selected_indices,
                    alignment=condition_alignment,
                    seed=seed,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                if corruption.selected_indices.numel():
                    negative_inputs = corrupted_model_inputs(inputs, corruption)
                    selected_target_mask = corruption.target_mask.index_select(
                        0, corruption.selected_indices
                    )
                    positive_token_loss = output.per_token_loss.index_select(
                        0, corruption.selected_indices
                    ).detach()
                    negative_sync_context = (
                        model.no_sync()
                        if isinstance(model, DistributedDataParallel) and not should_step
                        else contextlib.nullcontext()
                    )
                    with negative_sync_context:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ):
                            negative_output = model(
                                **negative_inputs,
                                return_token_losses=True,
                            )
                            counterfactual_loss, condition_gap = (
                                counterfactual_likelihood_loss(
                                    positive_token_loss=positive_token_loss,
                                    negative_token_loss=negative_output.per_token_loss,
                                    target_mask=selected_target_mask,
                                    margin_nats=float(condition_alignment["margin_nats"]),
                                )
                            )
                            scaled_counterfactual = (
                                current_condition_weight * counterfactual_loss / group_size
                            )
                        scaled_counterfactual.backward()
                    condition_examples = int(corruption.selected_indices.numel())
            count = output.count.detach().to(torch.float64)
            running[0] += output.loss.detach().to(torch.float64) * count
            running[1] += output.correct.detach().to(torch.float64)
            running[2] += count
            running[3 : 3 + len(BODY_SLOTS)] += output.per_slot_correct.detach().to(torch.float64)
            running[
                3 + len(BODY_SLOTS) : 3 + 2 * len(BODY_SLOTS)
            ] += output.per_slot_count.detach().to(torch.float64)
            clean_mask = ~generated_mask
            split_offset = base_metric_count
            if bool(clean_mask.any()):
                running[split_offset] += output.per_example_loss_sum[clean_mask].detach().sum().to(torch.float64)
                running[split_offset + 1] += output.per_example_correct[clean_mask].detach().sum().to(torch.float64)
                running[split_offset + 2] += output.per_example_count[clean_mask].detach().sum().to(torch.float64)
            if bool(generated_mask.any()):
                running[split_offset + 3] += output.per_example_loss_sum[generated_mask].detach().sum().to(torch.float64)
                running[split_offset + 4] += output.per_example_correct[generated_mask].detach().sum().to(torch.float64)
                running[split_offset + 5] += output.per_example_count[generated_mask].detach().sum().to(torch.float64)
            rollout_values = (
                rollout_stats.clips,
                rollout_stats.anchors,
                rollout_stats.tokens,
                rollout_stats.correct,
                rollout_stats.q0_correct,
                rollout_stats.q0_tokens,
                rollout_stats.confidence_sum,
                rollout_stats.entropy_sum,
            )
            running[split_offset + 6 : split_offset + 14] += torch.tensor(
                rollout_values, dtype=torch.float64, device=device
            )
            running[split_offset + 14] += output.ce_loss.detach().to(torch.float64) * count
            auxiliary_count = output.expected_distortion_count.detach().to(torch.float64)
            running[split_offset + 15] += (
                output.expected_distortion_loss.detach().to(torch.float64) * auxiliary_count
            )
            running[split_offset + 16] += auxiliary_count
            if counterfactual_loss is not None and condition_gap is not None:
                alignment_running[0] += (
                    counterfactual_loss.detach().to(torch.float64) * condition_examples
                )
                alignment_running[1] += condition_gap.to(torch.float64).sum()
                alignment_running[2] += condition_examples
                alignment_running[3 if condition_modality == "audio" else 4] += (
                    condition_examples
                )

            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_steps == 0:
                totals = reduce_sums(running.clone(), distributed)
                alignment_totals = reduce_sums(alignment_running.clone(), distributed)
                denominator = max(1.0, float(totals[2]))
                if is_main(rank):
                    split_offset = base_metric_count
                    clean_count = max(1.0, float(totals[split_offset + 2]))
                    generated_count = max(1.0, float(totals[split_offset + 5]))
                    rollout_tokens = max(1.0, float(totals[split_offset + 8]))
                    rollout_q0_tokens = max(1.0, float(totals[split_offset + 11]))
                    expected_distortion_count = max(1.0, float(totals[split_offset + 16]))
                    alignment_count = max(1.0, float(alignment_totals[2]))
                    train_metrics = {
                        "train/loss": float(totals[0]) / denominator,
                        "train/cross_entropy": float(totals[split_offset + 14]) / denominator,
                        "train/slot_accuracy": float(totals[1]) / denominator,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (batch_index + 1) / max(1, len(train_loader)),
                        "curriculum/generated_history_probability": generated_probability,
                        "curriculum/generated_clips": float(totals[split_offset + 6]),
                        "curriculum/generated_anchors": float(totals[split_offset + 7]),
                        "auxiliary/weight": current_auxiliary_weight,
                        "condition/weight": current_condition_weight,
                    }
                    if float(alignment_totals[2]) > 0:
                        train_metrics.update(
                            {
                                "condition/counterfactual_loss": float(alignment_totals[0])
                                / alignment_count,
                                "condition/gap_nats_per_token": float(alignment_totals[1])
                                / alignment_count,
                                "condition/audio_examples": float(alignment_totals[3]),
                                "condition/text_examples": float(alignment_totals[4]),
                            }
                        )
                        train_metrics["train/objective"] = (
                            train_metrics["train/loss"]
                            + current_condition_weight
                            * train_metrics["condition/counterfactual_loss"]
                        )
                    if float(totals[split_offset + 16]) > 0:
                        train_metrics["auxiliary/expected_distortion"] = (
                            float(totals[split_offset + 15]) / expected_distortion_count
                        )
                    if float(totals[split_offset + 2]) > 0:
                        train_metrics.update(
                            {
                                "train_clean/loss": float(totals[split_offset]) / clean_count,
                                "train_clean/slot_accuracy": float(totals[split_offset + 1]) / clean_count,
                            }
                        )
                    if float(totals[split_offset + 5]) > 0:
                        train_metrics.update(
                            {
                                "train_generated/loss": float(totals[split_offset + 3]) / generated_count,
                                "train_generated/slot_accuracy": float(totals[split_offset + 4]) / generated_count,
                                "rollout/accuracy": float(totals[split_offset + 9]) / rollout_tokens,
                                "rollout/q0_accuracy": float(totals[split_offset + 10]) / rollout_q0_tokens,
                                "rollout/mean_confidence": float(totals[split_offset + 12]) / rollout_tokens,
                                "rollout/mean_entropy": float(totals[split_offset + 13]) / rollout_tokens,
                            }
                        )
                    print(
                        f"step={global_step}/{total_steps} epoch={epoch + 1}/{epochs} "
                        f"loss={train_metrics['train/loss']:.5f} "
                        f"slot_acc={train_metrics['train/slot_accuracy']:.4%} "
                        f"p_gen={generated_probability:.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"elapsed={time.perf_counter() - run_start:.1f}s"
                    )
                    if wandb_run is not None:
                        wandb_run.log(train_metrics, step=global_step)
                running.zero_()
                alignment_running.zero_()

            if eval_steps > 0 and global_step % eval_steps == 0:
                eval_metrics = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    distributed=distributed,
                    use_bf16=use_bf16,
                    condition_alignment=condition_alignment,
                    alignment_seed=seed,
                )
                if is_main(rank):
                    print(
                        f"[eval] step={global_step} loss={eval_metrics['loss']:.5f} "
                        f"slot_acc={eval_metrics['slot_accuracy']:.4%}"
                    )
                    print("[eval slots]", json.dumps(eval_metrics["per_slot_accuracy"], sort_keys=True))
                    if eval_metrics["condition_gap_nats_per_token"]:
                        print(
                            "[eval condition gaps]",
                            json.dumps(
                                eval_metrics["condition_gap_nats_per_token"],
                                sort_keys=True,
                            ),
                        )
                    if wandb_run is not None:
                        wandb_payload = {
                            "eval/loss": eval_metrics["loss"],
                            "eval/slot_accuracy": eval_metrics["slot_accuracy"],
                        }
                        wandb_payload.update(
                            {
                                f"eval_slot/{name}": value
                                for name, value in eval_metrics["per_slot_accuracy"].items()
                            }
                        )
                        wandb_payload.update(
                            {
                                f"eval_condition/{name}_gap_nats_per_token": value
                                for name, value in eval_metrics[
                                    "condition_gap_nats_per_token"
                                ].items()
                            }
                        )
                        wandb_run.log(wandb_payload, step=global_step)

            if save_steps > 0 and global_step % save_steps == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    output_dir=paths["output_dir"],
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_index + 1,
                    source_config=config,
                    distributed=distributed,
                    rank=rank,
                    best_eval_loss=best_eval_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    curriculum_activation_epoch=curriculum_activation_epoch,
                    best_rollout_accuracy=best_rollout_accuracy,
                )
        resume_batch = 0

        eval_metrics = evaluate(
            model,
            eval_loader,
            device=device,
            distributed=distributed,
            use_bf16=use_bf16,
            condition_alignment=condition_alignment,
            alignment_seed=seed,
        )
        if is_main(rank):
            print(
                f"[epoch eval] epoch={epoch + 1} loss={eval_metrics['loss']:.5f} "
                f"slot_acc={eval_metrics['slot_accuracy']:.4%}"
            )
            print("[epoch eval slots]", json.dumps(eval_metrics["per_slot_accuracy"], sort_keys=True))
            if eval_metrics["condition_gap_nats_per_token"]:
                print(
                    "[epoch eval condition gaps]",
                    json.dumps(
                        eval_metrics["condition_gap_nats_per_token"], sort_keys=True
                    ),
                )
            if wandb_run is not None:
                wandb_payload = {
                    "eval/loss": eval_metrics["loss"],
                    "eval/slot_accuracy": eval_metrics["slot_accuracy"],
                    "eval/completed_epoch": epoch + 1,
                }
                wandb_payload.update(
                    {
                        f"eval_slot/{name}": value
                        for name, value in eval_metrics["per_slot_accuracy"].items()
                    }
                )
                wandb_payload.update(
                    {
                        f"eval_condition/{name}_gap_nats_per_token": value
                        for name, value in eval_metrics[
                            "condition_gap_nats_per_token"
                        ].items()
                    }
                )
                wandb_run.log(wandb_payload, step=global_step)

        completed_epochs = epoch + 1
        improved = eval_metrics["loss"] < best_eval_loss - early_stopping_min_delta
        if improved:
            best_eval_loss = float(eval_metrics["loss"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        rollout_metrics = None
        activation_check = (
            bool(generated_history["enabled"])
            and curriculum_activation_epoch is None
            and completed_epochs >= int(generated_history["minimum_teacher_epochs"])
        )
        scheduled_rollout_eval = (
            bool(generated_history["enabled"])
            and completed_epochs in generated_history["rollout_eval_epochs"]
        )
        if activation_check or scheduled_rollout_eval:
            rollout_metrics = run_rank0_rollout_evaluation(
                model,
                rollout_eval_loader,
                device=device,
                rank=rank,
                distributed=distributed,
                use_bf16=use_bf16,
            )
            if is_main(rank):
                print(
                    f"[generated rollout] epoch={completed_epochs} "
                    f"clips={rollout_metrics.clips} "
                    f"accuracy={rollout_metrics.accuracy:.4%} "
                    f"q0_accuracy={rollout_metrics.q0_accuracy:.4%} "
                    f"confidence={rollout_metrics.mean_confidence:.4f} "
                    f"entropy={rollout_metrics.mean_entropy:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval_rollout/accuracy": rollout_metrics.accuracy,
                            "eval_rollout/q0_accuracy": rollout_metrics.q0_accuracy,
                            "eval_rollout/mean_confidence": rollout_metrics.mean_confidence,
                            "eval_rollout/mean_entropy": rollout_metrics.mean_entropy,
                            "eval_rollout/clips": rollout_metrics.clips,
                        },
                        step=global_step,
                    )

        if activation_check:
            gate_passed = bool(
                eval_metrics["loss"] <= float(generated_history["teacher_forced_ce_max"])
                and rollout_metrics is not None
                and rollout_metrics.accuracy >= float(generated_history["rollout_accuracy_min"])
                and rollout_metrics.q0_accuracy
                >= float(generated_history["rollout_q0_accuracy_min"])
            )
            if gate_passed:
                curriculum_activation_epoch = completed_epochs
                if is_main(rank):
                    print(
                        "[generated-history gate] PASS: curriculum activates at the next epoch; "
                        f"activation_epoch={curriculum_activation_epoch}"
                    )
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    output_dir=paths["output_dir"],
                    global_step=global_step,
                    epoch=completed_epochs,
                    batch_in_epoch=0,
                    source_config=config,
                    distributed=distributed,
                    rank=rank,
                    checkpoint_name="teacher_warmup",
                    best_eval_loss=best_eval_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    curriculum_activation_epoch=curriculum_activation_epoch,
                    best_rollout_accuracy=best_rollout_accuracy,
                )
            elif is_main(rank):
                print(
                    "[generated-history gate] HOLD: "
                    f"ce={eval_metrics['loss']:.5f}/"
                    f"{generated_history['teacher_forced_ce_max']:.5f}, "
                    f"rollout={rollout_metrics.accuracy:.4%}/"
                    f"{generated_history['rollout_accuracy_min']:.4%}, "
                    f"q0={rollout_metrics.q0_accuracy:.4%}/"
                    f"{generated_history['rollout_q0_accuracy_min']:.4%}"
                )

        if rollout_metrics is not None and rollout_metrics.accuracy > best_rollout_accuracy:
            best_rollout_accuracy = rollout_metrics.accuracy
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                output_dir=paths["output_dir"],
                global_step=global_step,
                epoch=completed_epochs,
                batch_in_epoch=0,
                source_config=config,
                distributed=distributed,
                rank=rank,
                checkpoint_name="best_rollout",
                best_eval_loss=best_eval_loss,
                epochs_without_improvement=epochs_without_improvement,
                curriculum_activation_epoch=curriculum_activation_epoch,
                best_rollout_accuracy=best_rollout_accuracy,
            )

        if improved and save_best:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                output_dir=paths["output_dir"],
                global_step=global_step,
                epoch=completed_epochs,
                batch_in_epoch=0,
                source_config=config,
                distributed=distributed,
                rank=rank,
                checkpoint_name="best",
                best_eval_loss=best_eval_loss,
                epochs_without_improvement=epochs_without_improvement,
                curriculum_activation_epoch=curriculum_activation_epoch,
                best_rollout_accuracy=best_rollout_accuracy,
            )
        if is_main(rank):
            print(
                f"[early stopping] best_loss={best_eval_loss:.5f} "
                f"epochs_without_improvement={epochs_without_improvement}/"
                f"{early_stopping_patience or 'disabled'}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/best_loss": best_eval_loss,
                        "eval/epochs_without_improvement": epochs_without_improvement,
                    },
                    step=global_step,
                )
        if (
            bool(generated_history["enabled"])
            and completed_epochs in generated_history["milestone_epochs"]
        ):
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                output_dir=paths["output_dir"],
                global_step=global_step,
                epoch=completed_epochs,
                batch_in_epoch=0,
                source_config=config,
                distributed=distributed,
                rank=rank,
                checkpoint_name=f"epoch-{completed_epochs:04d}",
                best_eval_loss=best_eval_loss,
                epochs_without_improvement=epochs_without_improvement,
                curriculum_activation_epoch=curriculum_activation_epoch,
                best_rollout_accuracy=best_rollout_accuracy,
            )
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            if is_main(rank):
                print(f"Early stopping after {completed_epochs} completed epochs")
            break

    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        output_dir=paths["output_dir"],
        global_step=global_step,
        epoch=completed_epochs,
        batch_in_epoch=0,
        source_config=config,
        distributed=distributed,
        rank=rank,
        final=True,
        best_eval_loss=best_eval_loss,
        epochs_without_improvement=epochs_without_improvement,
        curriculum_activation_epoch=curriculum_activation_epoch,
        best_rollout_accuracy=best_rollout_accuracy,
    )
    if is_main(rank):
        suffix = " (early stopped)" if stopped_early else ""
        print(f"Training complete{suffix} in {time.perf_counter() - run_start:.1f}s")
        if wandb_run is not None:
            wandb_run.finish()
    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

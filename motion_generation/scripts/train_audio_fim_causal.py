#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Train the compact Step 2 audio-aware causal FIM transformer.

This is the small-model sibling of train_vllm_infill.py. It does not use the
Qwen tokenizer or vLLM checkpoint. It trains AudioFIMCausalLM from scratch with:

    motion tokens: compact RVQ ids, old Step 2 style
    audio:         continuous HuBERT layer9 features
    objective:     causal FIM, predict the middle motion frames only

First target:
    classic gap with step=4:
        left frame t, right frame t+4, predict frames t+1, t+2, t+3.

The explicit [LEN_N] path and --step argument are kept so variable gaps can be
enabled later without changing the checkpoint format.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed


THIS_DIR = Path(__file__).resolve().parent
MOTION_GENERATION_DIR = THIS_DIR.parent
PROJECT_DIR = MOTION_GENERATION_DIR.parent
sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_fim_causal_model import (  # noqa: E402
    AudioFIMCausalCollator,
    AudioFIMCausalConfig,
    AudioFIMCausalDataset,
    AudioFIMCausalLM,
    AudioFIMTokenMapper,
)
from configs.default_config import Config  # noqa: E402
from models.rvqvae import RVQVAE  # noqa: E402
from utils.constants import (  # noqa: E402
    BODY_JOINTS_ID,
    LEFT_HAND_JOINTS_ID,
    RIGHT_HAND_JOINTS_ID,
)
from utils.rotation_utils import sixd_to_quaternion  # noqa: E402


def format_fps_for_dir(fps: float) -> str:
    if float(fps).is_integer():
        return str(int(fps))
    return str(fps).replace(".", "p")


@contextmanager
def timed_stage(name: str, enabled: bool = True):
    if not enabled:
        yield
        return

    start = time.perf_counter()
    print(f"[Timing] {name} ...")
    try:
        yield
    finally:
        print(f"[Timing] {name}: {time.perf_counter() - start:.3f}s")


def read_split_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    normalized = []
    for name in names:
        name = name.replace("\\", "/").strip().strip("/")
        suffix = Path(name).suffix
        if suffix in {".wav", ".npy", ".json"}:
            name = name[: -len(suffix)]
        normalized.append(name)
    return normalized


def load_token_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"tokens": data}
    raise ValueError(f"Unsupported token JSON format: {path}")


def extract_action_text(raw_text: str) -> Optional[str]:
    tags = re.findall(r"ã€(.+?)ã€‘", raw_text)
    if not tags:
        return None

    last_tag = tags[-1]
    if last_tag == "åŠ¨ä½œï¼šæ— åŠ¨ä½œ":
        for tag in tags:
            if tag.startswith("è¡¨æƒ…ï¼š") and tag != "è¡¨æƒ…ï¼šæ— è¡¨æƒ…":
                expression = tag.replace("è¡¨æƒ…ï¼š", "")
                return expression if "åŠ¨ä½œ" in expression else f"åŠ¨ä½œï¼š{expression}"
    return last_tag


def load_action_text_map(path: Optional[str]) -> Dict[str, str]:
    # This compact model does not consume text yet. We still parse the map so
    # sequence metadata stays parallel with train_vllm_infill.py for later use.
    if path is None or not Path(path).exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        motion2text = json.load(f)

    result: Dict[str, str] = {}
    for name, raw_text in motion2text.items():
        action_text = extract_action_text(raw_text)
        if not action_text:
            continue
        normalized = name.replace("\\", "/").strip().strip("/")
        suffix = Path(normalized).suffix
        if suffix in {".wav", ".npy", ".json"}:
            normalized = normalized[: -len(suffix)]
        result[normalized] = action_text
    return result


def discover_names(
    motion_token_dir: Path,
    audio_feat_dir: Path,
    split_names: Optional[Sequence[str]],
) -> List[str]:
    if split_names is not None:
        names = list(split_names)
    else:
        motion_names = {
            path.relative_to(motion_token_dir).with_suffix("").as_posix()
            for path in motion_token_dir.rglob("*.json")
        }
        audio_names = {
            path.relative_to(audio_feat_dir).with_suffix("").as_posix()
            for path in audio_feat_dir.rglob("*.npy")
        }
        names = sorted(motion_names & audio_names)

    available = []
    for name in names:
        if (motion_token_dir / f"{name}.json").exists() and (
            audio_feat_dir / f"{name}.npy"
        ).exists():
            available.append(name)
    return available


def load_sequences(
    names: Sequence[str],
    motion_token_dir: Path,
    audio_feat_dir: Path,
    action_text_map: Dict[str, str],
    *,
    max_samples: Optional[int] = None,
    audio_fps: float = 10.0,
    motion_fps: float = 20.0,
) -> List[Dict[str, Any]]:
    sequences: List[Dict[str, Any]] = []

    for name in names:
        if max_samples is not None and len(sequences) >= max_samples:
            break

        motion_payload = load_token_json(motion_token_dir / f"{name}.json")
        if not motion_payload:
            continue

        motion_tokens = motion_payload.get("tokens")
        if not motion_tokens:
            continue

        audio_path = audio_feat_dir / f"{name}.npy"
        if not audio_path.exists():
            continue

        audio_features = np.load(audio_path).astype(np.float32)
        if audio_features.ndim != 2 or audio_features.shape[0] == 0:
            continue

        sequences.append(
            {
                "name": name,
                "motion_tokens": motion_tokens,
                "audio_features": audio_features,
                "motion_fps": motion_payload.get("fps") or motion_fps,
                "audio_fps": audio_fps,
                "action_text": action_text_map.get(name),
            }
        )

    return sequences


def split_train_eval(
    sequences: List[Dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if eval_ratio <= 0 or len(sequences) < 2:
        return sequences, None

    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[eval_size:], shuffled[:eval_size]


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def configure_wandb(args: argparse.Namespace, default_run_name: str) -> Optional[str]:
    """Configure optional Weights & Biases logging for HuggingFace Trainer."""

    if args.report_to != "wandb":
        return None

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested with --report_to wandb, but wandb is not "
            "installed. Install wandb or run with --report_to none."
        ) from exc

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    return args.wandb_run_name or default_run_name


def _token_color(value: int) -> tuple[int, int, int]:
    """Stable color for compact 0..511 RVQ token values."""

    value = int(value) % 512
    return (
        40 + (value * 37) % 176,
        40 + (value * 67) % 176,
        40 + (value * 97) % 176,
    )


def render_token_comparison_video(
    *,
    gt_motion: Sequence[Sequence[int]],
    pred_motion: Sequence[Sequence[int]],
    title: str,
    fps: int = 4,
    repeat_per_frame: int = 4,
) -> np.ndarray:
    """
    Build a small GT-vs-pred RVQ token comparison video for W&B.

    This is intentionally token-space visualization. It is cheap enough to run
    during evaluation and catches whether the infill model is generating the
    right middle frames before we add heavier RVQVAE/BVH rendering.

    Returns:
        uint8 video array shaped (T, C, H, W), as expected by wandb.Video.
    """

    del fps
    gt = [list(frame) for frame in gt_motion]
    pred = [list(frame) for frame in pred_motion]
    frames = max(len(gt), len(pred), 1)
    quantizers = max(
        max((len(frame) for frame in gt), default=4),
        max((len(frame) for frame in pred), default=4),
    )

    width, height = 860, 420
    cell_w, cell_h = 88, 48
    left_x, right_x = 110, 500
    top_y = 112
    frame_images: List[np.ndarray] = []

    token_matches = 0
    token_total = 0
    for frame_idx in range(frames):
        gt_frame = gt[frame_idx] if frame_idx < len(gt) else []
        pred_frame = pred[frame_idx] if frame_idx < len(pred) else []
        for q_idx in range(min(len(gt_frame), len(pred_frame))):
            token_total += 1
            token_matches += int(int(gt_frame[q_idx]) == int(pred_frame[q_idx]))
    token_acc = token_matches / max(1, token_total)

    for current in range(frames):
        image = Image.new("RGB", (width, height), (248, 248, 246))
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), title[:92], fill=(20, 20, 20))
        draw.text(
            (24, 44),
            f"middle frame {current + 1}/{frames} | token acc {token_acc:.3f}",
            fill=(70, 70, 70),
        )
        draw.text((left_x, 84), "Ground truth", fill=(20, 20, 20))
        draw.text((right_x, 84), "Prediction", fill=(20, 20, 20))

        for row in range(frames):
            y0 = top_y + row * (cell_h + 18)
            row_outline = (30, 110, 220) if row == current else (185, 185, 185)
            draw.text((24, y0 + 14), f"F{row + 1}", fill=(30, 30, 30))
            gt_frame = gt[row] if row < len(gt) else []
            pred_frame = pred[row] if row < len(pred) else []

            for q_idx in range(quantizers):
                gt_val = int(gt_frame[q_idx]) if q_idx < len(gt_frame) else None
                pred_val = int(pred_frame[q_idx]) if q_idx < len(pred_frame) else None

                for panel_x, value in (
                    (left_x, gt_val),
                    (right_x, pred_val),
                ):
                    x0 = panel_x + q_idx * (cell_w + 6)
                    x1 = x0 + cell_w
                    y1 = y0 + cell_h
                    fill = (230, 230, 230) if value is None else _token_color(value)
                    draw.rectangle(
                        [x0, y0, x1, y1],
                        fill=fill,
                        outline=row_outline,
                        width=3 if row == current else 1,
                    )
                    text = "--" if value is None else str(value)
                    draw.text((x0 + 8, y0 + 15), text, fill=(0, 0, 0))

                if gt_val is not None and pred_val is not None:
                    match = gt_val == pred_val
                    status = "OK" if match else "ERR"
                    fill = (30, 125, 50) if match else (170, 40, 40)
                    draw.text(
                        (right_x + quantizers * (cell_w + 6) + 14, y0 + 15),
                        f"q{q_idx + 1} {status}",
                        fill=fill,
                    )

        for _ in range(max(1, repeat_per_frame)):
            frame_images.append(np.asarray(image, dtype=np.uint8))

    video = np.stack(frame_images, axis=0)
    return video.transpose(0, 3, 1, 2)


def _parse_rvqvae_opt_value(value_str: str) -> Any:
    value_str = value_str.strip()
    if value_str == "True":
        return True
    if value_str == "False":
        return False
    if value_str == "None":
        return None
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            return ast.literal_eval(value_str)
        except (SyntaxError, ValueError):
            return value_str
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        return value_str


def _parse_rvqvae_opt_txt(opt_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    with open(opt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = _parse_rvqvae_opt_value(value)
    return result


def load_rvqvae_config_from_checkpoint(checkpoint_path: Path) -> Config:
    opt_path = checkpoint_path.parent.parent / "opt.txt"
    if not opt_path.exists():
        raise FileNotFoundError(f"RVQVAE opt.txt not found: {opt_path}")

    opt = _parse_rvqvae_opt_txt(opt_path)
    config = Config(
        name=opt.get("name", "VQVAE_v2"),
        dataset_name=opt.get("dataset_name", "quat63nodes_v2"),
        checkpoints_dir=opt.get("checkpoints_dir", "./checkpoints"),
        log_dir=opt.get("log_dir", "./log/vq"),
        gpu_id=opt.get("gpu_id", 0),
        local_rank=opt.get("local_rank", 0),
        seed=opt.get("seed", 3407),
        debug=opt.get("debug", False),
    )

    config.data.data_root = opt.get("data_root", "")
    config.data.body_parts = opt.get("body_parts", ["body", "left", "right", "positions"])
    config.data.body_joints_num = opt.get("body_joints_num", 24)
    config.data.left_joints_num = opt.get("left_joints_num", 20)
    config.data.right_joints_num = opt.get("right_joints_num", 20)
    config.data.total_joints_num = opt.get("total_joints_num", 63)
    config.data.body_dim = opt.get("body_dim", 153)
    config.data.left_dim = opt.get("left_dim", 120)
    config.data.right_dim = opt.get("right_dim", 120)
    config.data.whole_dim = opt.get("whole_dim", 393)
    config.data.window_size = opt.get("window_size", 64)
    config.data.batch_size = opt.get("batch_size", 128)
    config.data.num_workers = opt.get("num_workers", 4)
    config.data.fps = opt.get("fps", 20)

    config.model.nb_code = opt.get("nb_code", 512)
    config.model.code_dim = opt.get("code_dim", 512)
    config.model.down_t = opt.get("down_t", 1)
    config.model.stride_t = opt.get("stride_t", 2)
    config.model.width = opt.get("width", 512)
    config.model.depth = opt.get("depth", 3)
    config.model.dilation_growth_rate = opt.get("dilation_growth_rate", 3)
    config.model.vq_act = opt.get("vq_act", "relu")
    config.model.vq_norm = opt.get("vq_norm", None)
    config.model.vq_cnn_depth = opt.get("vq_cnn_depth", 3)
    config.model.num_quantizers = opt.get("num_quantizers", 4)
    config.model.shared_codebook = opt.get("shared_codebook", False)
    config.model.quantize_dropout_prob = opt.get("quantize_dropout_prob", 0.8)
    config.model.quantize_dropout_cutoff_index = opt.get(
        "quantize_dropout_cutoff_index",
        1,
    )
    config.model.use_whole_encoder = opt.get("use_whole_encoder", False)
    config.model.mu = opt.get("mu", 0.99)
    config.unit_length = config.model.down_t * 2
    config.save_root = os.path.join(config.checkpoints_dir, config.dataset_name, config.name)
    config.model_dir = os.path.join(config.save_root, "model")
    config.meta_dir = os.path.join(config.save_root, "meta")
    config.eval_dir = os.path.join(config.save_root, "animation")
    config.log_path = os.path.join(config.log_dir, config.dataset_name, config.name)
    return config


def load_rvqvae_model_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RVQVAE, Config]:
    rvq_config = load_rvqvae_config_from_checkpoint(checkpoint_path)
    rvqvae = RVQVAE(
        config=rvq_config,
        input_dim=rvq_config.data.whole_dim,
        nb_code=rvq_config.model.nb_code,
        code_dim=rvq_config.model.code_dim,
        output_dim=rvq_config.model.code_dim,
        down_t=rvq_config.model.down_t,
        stride_t=rvq_config.model.stride_t,
        width=rvq_config.model.width,
        depth=rvq_config.model.depth,
        dilation_growth_rate=rvq_config.model.dilation_growth_rate,
        activation=rvq_config.model.vq_act,
        norm=rvq_config.model.vq_norm,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    rvqvae.load_state_dict(state_dict)
    return rvqvae.to(device).eval(), rvq_config


def load_motion_dict(path: Path) -> Dict[str, np.ndarray]:
    motion = np.load(path, allow_pickle=True)
    if isinstance(motion, np.ndarray) and motion.dtype == object:
        motion = motion.item()
    if not isinstance(motion, dict):
        raise ValueError(f"Unsupported motion_data format: {path}")
    return motion


def slice_motion_dict(
    motion_dict: Dict[str, np.ndarray],
    start_idx: int,
    frames: int,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for key, value in motion_dict.items():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            result[key] = arr[start_idx : start_idx + frames]
        else:
            result[key] = arr
    return result


def resample_motion_tensor(
    x: torch.Tensor,
    *,
    src_fps: float,
    tgt_fps: float,
) -> torch.Tensor:
    if abs(float(src_fps) - float(tgt_fps)) < 1e-6 or x.shape[1] <= 1:
        return x

    new_frames = max(1, int(round(x.shape[1] * float(tgt_fps) / float(src_fps))))
    batch, _, joints, dims = x.shape
    flat = x.permute(0, 2, 3, 1).contiguous().view(batch, joints * dims, x.shape[1])
    flat = F.interpolate(flat, size=new_frames, mode="linear", align_corners=True)
    return flat.view(batch, joints, dims, new_frames).permute(0, 3, 1, 2).contiguous()


def pad_or_trim_hand_motion(hand: torch.Tensor, frames: int) -> torch.Tensor:
    if hand.shape[1] == 0:
        return torch.zeros(
            hand.shape[0],
            frames,
            hand.shape[2],
            dtype=hand.dtype,
            device=hand.device,
        )
    if hand.shape[1] == frames:
        return hand
    if hand.shape[1] > frames:
        return hand[:, :frames]

    pad = frames - hand.shape[1]
    return F.pad(hand.permute(0, 2, 1), (0, pad), mode="replicate").permute(0, 2, 1)


@torch.no_grad()
def decode_body_tokens_to_features(
    rvqvae: RVQVAE,
    body_tokens: Sequence[Sequence[int]],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tokens = np.asarray(body_tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError(f"Expected body tokens shaped (frames, quantizers), got {tokens.shape}")

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    decoded = rvqvae.forward_decoder({"body": token_tensor})
    decoded = decoded[0] if decoded.ndim == 3 else decoded.sum(0)
    return decoded.float() * std + mean


def body_features_to_quat_motion(
    body_features: torch.Tensor,
    motion_dict: Dict[str, np.ndarray],
    device: torch.device,
    *,
    src_fps: float,
    tgt_fps: float,
) -> Dict[str, np.ndarray]:
    body_features = body_features.to(device=device, dtype=torch.float32)
    frames = int(body_features.shape[0])

    offset_frame0 = torch.tensor([0.0, 0.0, 102.0], device=device)
    offset_vel = body_features[:, :3].clone()
    for frame_idx in range(1, offset_vel.shape[0]):
        offset_vel[frame_idx] = offset_vel[frame_idx] + offset_vel[frame_idx - 1]
    offset = (offset_vel + offset_frame0).reshape(1, frames, 1, 3)

    body_6d = body_features[:, 3:].reshape(1, frames, 25, 6)

    left_raw = motion_dict.get("left")
    right_raw = motion_dict.get("right")
    if left_raw is None:
        left_raw = np.zeros((frames, 20 * 6), dtype=np.float32)
    if right_raw is None:
        right_raw = np.zeros((frames, 20 * 6), dtype=np.float32)

    left = torch.tensor(left_raw, dtype=torch.float32, device=device).unsqueeze(0)
    right = torch.tensor(right_raw, dtype=torch.float32, device=device).unsqueeze(0)
    left = pad_or_trim_hand_motion(left, frames).reshape(1, frames, 20, 6)
    right = pad_or_trim_hand_motion(right, frames).reshape(1, frames, 20, 6)

    body_6d = resample_motion_tensor(body_6d, src_fps=src_fps, tgt_fps=tgt_fps)
    left = resample_motion_tensor(left, src_fps=src_fps, tgt_fps=tgt_fps)
    right = resample_motion_tensor(right, src_fps=src_fps, tgt_fps=tgt_fps)
    offset = resample_motion_tensor(offset, src_fps=src_fps, tgt_fps=tgt_fps)

    out_frames = int(body_6d.shape[1])
    body_quat = sixd_to_quaternion(body_6d.reshape(-1, 6)).reshape(
        1,
        out_frames,
        25,
        4,
    )
    left_quat = sixd_to_quaternion(left.reshape(-1, 6)).reshape(1, out_frames, 20, 4)
    right_quat = sixd_to_quaternion(right.reshape(-1, 6)).reshape(
        1,
        out_frames,
        20,
        4,
    )

    merged = torch.zeros(out_frames, 63, 4, device=device)
    merged[:, BODY_JOINTS_ID] = body_quat[0]
    merged[:, LEFT_HAND_JOINTS_ID[1:]] = left_quat[0, :, 1:]
    merged[:, RIGHT_HAND_JOINTS_ID[1:]] = right_quat[0, :, 1:]

    return {
        "offset": offset.reshape(out_frames, 3).detach().cpu().numpy(),
        "quat": merged.detach().cpu().numpy(),
    }


def quat_motion_to_joint_positions(
    motion: Dict[str, np.ndarray],
    postprocesser: Any,
    device: torch.device,
) -> np.ndarray:
    from actions.postprocess import process_batch_data
    from utils.visualization_torch.Animation import positions_global
    from utils.visualization_torch.Quaternions import Quaternions

    input_quat = torch.as_tensor(motion["quat"], dtype=torch.float32, device=device)
    input_offset = torch.as_tensor(motion["offset"], dtype=torch.float32, device=device)
    if input_quat.ndim == 3:
        input_quat = input_quat.unsqueeze(0)
    if input_offset.ndim == 2:
        input_offset = input_offset.unsqueeze(0)

    final_quats, final_root_pos = process_batch_data(
        input_quat,
        input_offset,
        postprocesser.anim,
        postprocesser.skel.src_joint_dict,
        shape="wxyz",
    )
    final_quats = final_quats.detach().cpu()
    final_root_pos = final_root_pos.detach().cpu() if final_root_pos is not None else None

    num_frames = int(final_quats.shape[1])
    base_pos = postprocesser.anim.positions[0].detach().cpu().clone()
    current_pos = base_pos.unsqueeze(0).repeat(num_frames, 1, 1)
    if final_root_pos is not None:
        current_pos[:, 0, :] = final_root_pos[0]

    postprocesser.anim.rotations = Quaternions(final_quats[0])
    postprocesser.anim.positions = current_pos
    positions = positions_global(postprocesser.anim)
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    return np.asarray(positions, dtype=np.float32)


def render_decoded_motion_comparison_video(
    *,
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    skeleton_edges: Sequence[tuple[int, int]],
    joint_names: Sequence[str],
    title: str,
    fps: int,
    source_frames: int,
    middle_start: int,
    middle_end: int,
) -> np.ndarray:
    width, height = 1280, 720
    panel_w, panel_h = 580, 560
    left_origin = (40, 110)
    right_origin = (660, 110)
    frame_count = max(len(gt_positions), len(pred_positions), 1)

    def project(points: np.ndarray) -> np.ndarray:
        x = points[..., 0] + 0.25 * points[..., 2]
        y = -points[..., 1] + 0.12 * points[..., 2]
        return np.stack([x, y], axis=-1)

    gt_2d = project(gt_positions)
    pred_2d = project(pred_positions)
    combined = np.concatenate([gt_2d.reshape(-1, 2), pred_2d.reshape(-1, 2)], axis=0)
    finite = np.isfinite(combined).all(axis=1)
    combined = combined[finite]
    if combined.size == 0:
        combined = np.zeros((1, 2), dtype=np.float32)
    min_xy = combined.min(axis=0)
    max_xy = combined.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    center = (min_xy + max_xy) * 0.5
    scale = min((panel_w - 80) / span[0], (panel_h - 100) / span[1])

    def to_screen(points_2d: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        x0, y0 = origin
        out = np.empty_like(points_2d)
        out[..., 0] = x0 + panel_w * 0.5 + (points_2d[..., 0] - center[0]) * scale
        out[..., 1] = y0 + panel_h * 0.5 + (points_2d[..., 1] - center[1]) * scale
        return out

    gt_screen = to_screen(gt_2d, left_origin)
    pred_screen = to_screen(pred_2d, right_origin)
    frame_images: List[np.ndarray] = []
    joint_count = max(
        int(gt_screen.shape[1]) if gt_screen.ndim >= 2 else 0,
        int(pred_screen.shape[1]) if pred_screen.ndim >= 2 else 0,
        1,
    )
    valid_edges = [
        (int(parent), int(child))
        for parent, child in skeleton_edges
        if 0 <= int(parent) < joint_count and 0 <= int(child) < joint_count
    ]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    marker_indices = [
        name_to_idx[name]
        for name in ("pelvis", "head", "hand_l", "hand_r", "foot_l", "foot_r")
        if name in name_to_idx and name_to_idx[name] < joint_count
    ]
    if not marker_indices:
        marker_indices = [0]

    def frame_at(seq: np.ndarray, idx: int) -> np.ndarray:
        if len(seq) == 0:
            return np.zeros((joint_count, 2), dtype=np.float32)
        return seq[min(idx, len(seq) - 1)]

    def draw_skeleton(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int]) -> None:
        for parent, child in valid_edges:
            if parent >= len(points) or child >= len(points):
                continue
            ax, ay = points[parent]
            bx, by = points[child]
            draw.line([(float(ax), float(ay)), (float(bx), float(by))], fill=color, width=4)
        for joint_idx in marker_indices:
            if joint_idx >= len(points):
                continue
            x, y = points[joint_idx]
            r = 5 if joint_idx == 0 else 4
            draw.ellipse(
                [float(x - r), float(y - r), float(x + r), float(y + r)],
                fill=color,
                outline=(15, 15, 15),
            )

    def draw_panel(
        draw: ImageDraw.ImageDraw,
        origin: tuple[int, int],
        label: str,
        points: np.ndarray,
        trajectory: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        x0, y0 = origin
        draw.rectangle(
            [x0, y0, x0 + panel_w, y0 + panel_h],
            fill=(248, 248, 246),
            outline=(180, 180, 180),
            width=2,
        )
        draw.text((x0 + 18, y0 + 16), label, fill=(20, 20, 20))
        root_path = trajectory[: min(current + 1, len(trajectory)), 0]
        if len(root_path) >= 2:
            draw.line(
                [(float(x), float(y)) for x, y in root_path],
                fill=(160, 160, 160),
                width=2,
            )
        draw_skeleton(draw, points, color)

    for current in range(frame_count):
        image = Image.new("RGB", (width, height), (236, 236, 232))
        draw = ImageDraw.Draw(image)
        source_pos = (current + 0.5) * max(1, source_frames) / max(1, frame_count)
        in_middle = middle_start <= source_pos < middle_end
        region = "INFILL" if in_middle else "context"
        accent = (200, 62, 42) if in_middle else (70, 90, 110)

        draw.text((34, 26), title[:110], fill=(20, 20, 20))
        draw.text(
            (34, 54),
            f"frame {current + 1}/{frame_count} | {fps} fps | region: {region}",
            fill=accent,
        )
        draw.rectangle([34, 82, width - 34, 92], fill=(205, 205, 200))
        progress_x = 34 + int((width - 68) * (current + 1) / frame_count)
        draw.rectangle([34, 82, progress_x, 92], fill=accent)

        draw_panel(
            draw,
            left_origin,
            "Ground truth",
            frame_at(gt_screen, current),
            gt_screen,
            (42, 104, 185),
        )
        draw_panel(
            draw,
            right_origin,
            "Prediction",
            frame_at(pred_screen, current),
            pred_screen,
            (188, 76, 50),
        )
        frame_images.append(np.asarray(image, dtype=np.uint8))

    video = np.stack(frame_images, axis=0)
    return video.transpose(0, 3, 1, 2)


def motion_token_accuracy(
    gt_motion: Sequence[Sequence[int]],
    pred_motion: Sequence[Sequence[int]],
) -> float:
    matches = 0
    total = 0
    for gt_frame, pred_frame in zip(gt_motion, pred_motion):
        for gt_token, pred_token in zip(gt_frame, pred_frame):
            matches += int(int(gt_token) == int(pred_token))
            total += 1
    return matches / max(1, total)


class WandbEvalVideoCallback(TrainerCallback):
    """Log cheap GT-vs-pred token videos for fixed eval examples."""

    def __init__(
        self,
        *,
        eval_dataset: AudioFIMCausalDataset,
        num_examples: int,
        every_n_evals: int,
        fps: int,
    ):
        self.eval_dataset = eval_dataset
        self.num_examples = max(0, int(num_examples))
        self.every_n_evals = max(1, int(every_n_evals))
        self.fps = max(1, int(fps))
        self._eval_calls = 0

    def on_evaluate(self, args, state, control, **kwargs):
        del args, control

        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if self.num_examples <= 0 or self.eval_dataset is None:
            return

        self._eval_calls += 1
        if self._eval_calls % self.every_n_evals != 0:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        model = kwargs.get("model")
        if model is None:
            return
        base_model = model.module if hasattr(model, "module") else model
        if not hasattr(base_model, "generate_infill"):
            return

        was_training = base_model.training
        log_payload: Dict[str, Any] = {}

        for sample_idx in range(min(self.num_examples, len(self.eval_dataset))):
            example = self.eval_dataset[sample_idx]
            pred_motion = base_model.generate_infill(
                history_motion=example.history_motion,
                left_anchor=example.left_anchor,
                right_anchor=example.right_anchor,
                middle_audio_features=example.middle_audio_features,
                left_audio_feature=example.left_audio_feature,
                right_audio_feature=example.right_audio_feature,
                history_audio_features=example.history_audio_features,
                temperature=0.0,
            )
            video = render_token_comparison_video(
                gt_motion=example.middle_motion,
                pred_motion=pred_motion,
                title=f"{example.name} | step {state.global_step}",
                fps=self.fps,
            )
            log_payload[f"eval_video/token_compare_{sample_idx}"] = wandb.Video(
                video,
                fps=self.fps,
                format="mp4",
            )

        if log_payload:
            wandb.log(log_payload, step=state.global_step)

        if was_training:
            base_model.train()


class WandbEvalMotionVideoCallback(TrainerCallback):
    """Log decoded GT-vs-pred motion videos for fixed eval examples."""

    def __init__(
        self,
        *,
        eval_dataset: AudioFIMCausalDataset,
        motion_data_dir: Path,
        rvqvae_ckpt: Path,
        mean_path: Path,
        std_path: Path,
        num_examples: int,
        every_n_evals: int,
        fps: int,
        scan_examples: int = 200,
    ):
        self.eval_dataset = eval_dataset
        self.motion_data_dir = motion_data_dir
        self.rvqvae_ckpt = rvqvae_ckpt
        self.mean_path = mean_path
        self.std_path = std_path
        self.num_examples = max(0, int(num_examples))
        self.every_n_evals = max(1, int(every_n_evals))
        self.fps = max(1, int(fps))
        self._eval_calls = 0
        self._rvqvae: Optional[RVQVAE] = None
        self._rvq_config: Optional[Config] = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._postprocesser: Any = None
        self._missing_warned: set[str] = set()
        self.sample_indices = self._select_sample_indices(scan_examples)

    def _select_sample_indices(self, scan_examples: int) -> List[int]:
        scored: List[tuple[int, int]] = []
        max_scan = min(len(self.eval_dataset), max(self.num_examples, int(scan_examples)))
        for idx in range(max_scan):
            try:
                example = self.eval_dataset[idx]
            except Exception:
                continue
            scored.append((len(example.history_motion), idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored[: self.num_examples]]

    def _ensure_decode_stack(self, device: torch.device) -> None:
        if self._rvqvae is None:
            self._rvqvae, self._rvq_config = load_rvqvae_model_for_eval(
                self.rvqvae_ckpt,
                device,
            )
            self._mean = torch.tensor(
                np.load(self.mean_path),
                dtype=torch.float32,
                device=device,
            )
            self._std = torch.tensor(
                np.load(self.std_path),
                dtype=torch.float32,
                device=device,
            )
            from actions.postprocess import MotionPostprocesser

            self._postprocesser = MotionPostprocesser()
            print(
                "[W&B motion video] loaded RVQVAE decoder: "
                f"{self.rvqvae_ckpt}"
            )
            return

        if next(self._rvqvae.parameters()).device != device:
            self._rvqvae = self._rvqvae.to(device)
            if self._mean is not None:
                self._mean = self._mean.to(device)
            if self._std is not None:
                self._std = self._std.to(device)

    def on_evaluate(self, args, state, control, **kwargs):
        del args, control

        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if self.num_examples <= 0 or self.eval_dataset is None:
            return

        self._eval_calls += 1
        if self._eval_calls % self.every_n_evals != 0:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        model = kwargs.get("model")
        if model is None:
            return
        base_model = model.module if hasattr(model, "module") else model
        if not hasattr(base_model, "generate_infill"):
            return

        device = next(base_model.parameters()).device
        self._ensure_decode_stack(device)
        if (
            self._rvqvae is None
            or self._rvq_config is None
            or self._mean is None
            or self._std is None
            or self._postprocesser is None
        ):
            return

        was_training = base_model.training
        log_payload: Dict[str, Any] = {}

        for sample_slot, dataset_idx in enumerate(self.sample_indices):
            example = self.eval_dataset[dataset_idx]
            motion_path = self.motion_data_dir / f"{example.name}.npy"
            if not motion_path.exists():
                key = str(motion_path)
                if key not in self._missing_warned:
                    print(f"[W&B motion video] missing motion_data file, skip: {motion_path}")
                    self._missing_warned.add(key)
                continue

            try:
                pred_middle = base_model.generate_infill(
                    history_motion=example.history_motion,
                    left_anchor=example.left_anchor,
                    right_anchor=example.right_anchor,
                    middle_audio_features=example.middle_audio_features,
                    left_audio_feature=example.left_audio_feature,
                    right_audio_feature=example.right_audio_feature,
                    history_audio_features=example.history_audio_features,
                    temperature=0.0,
                )

                prefix = [list(frame) for frame in example.history_motion]
                prefix.append(list(example.left_anchor))
                gt_clip = prefix + [list(frame) for frame in example.middle_motion]
                gt_clip.append(list(example.right_anchor))
                pred_clip = prefix + [list(frame) for frame in pred_middle]
                pred_clip.append(list(example.right_anchor))

                src_fps = float(self._rvq_config.data.fps)
                gt_features = decode_body_tokens_to_features(
                    self._rvqvae,
                    gt_clip,
                    self._mean,
                    self._std,
                    device,
                )
                pred_features = decode_body_tokens_to_features(
                    self._rvqvae,
                    pred_clip,
                    self._mean,
                    self._std,
                    device,
                )
                clip_start_idx = max(0, int(example.left_idx) - len(example.history_motion))
                unit_length = int(getattr(self._rvq_config, "unit_length", 2))
                motion_dict = slice_motion_dict(
                    load_motion_dict(motion_path),
                    clip_start_idx * unit_length,
                    int(gt_features.shape[0]),
                )
                gt_motion = body_features_to_quat_motion(
                    gt_features,
                    motion_dict,
                    device,
                    src_fps=src_fps,
                    tgt_fps=float(self.fps),
                )
                pred_motion = body_features_to_quat_motion(
                    pred_features,
                    motion_dict,
                    device,
                    src_fps=src_fps,
                    tgt_fps=float(self.fps),
                )
                gt_positions = quat_motion_to_joint_positions(
                    gt_motion,
                    self._postprocesser,
                    device,
                )
                pred_positions = quat_motion_to_joint_positions(
                    pred_motion,
                    self._postprocesser,
                    device,
                )
                token_acc = motion_token_accuracy(example.middle_motion, pred_middle)
                middle_start = len(example.history_motion) + 1
                middle_end = middle_start + len(example.middle_motion)
                joint_names = list(getattr(self._postprocesser.anim, "names", []))
                skeleton_edges = [
                    (int(parent), int(child))
                    for child, parent in enumerate(self._postprocesser.anim.parents)
                    if int(parent) >= 0
                ]
                video = render_decoded_motion_comparison_video(
                    gt_positions=gt_positions,
                    pred_positions=pred_positions,
                    skeleton_edges=skeleton_edges,
                    joint_names=joint_names,
                    title=(
                        f"{example.name} | step {state.global_step} | "
                        f"middle token acc {token_acc:.3f}"
                    ),
                    fps=self.fps,
                    source_frames=len(gt_clip),
                    middle_start=middle_start,
                    middle_end=middle_end,
                )
                log_payload[f"eval_video/motion_compare_{sample_slot}"] = wandb.Video(
                    video,
                    fps=self.fps,
                    format="mp4",
                )
                log_payload[f"eval_motion/token_acc_{sample_slot}"] = token_acc
            except Exception as exc:
                print(
                    "[W&B motion video] failed to render "
                    f"sample={example.name}: {exc}"
                )

        if log_payload:
            wandb.log(log_payload, step=state.global_step)

        if was_training:
            base_model.train()


@torch.no_grad()
def run_loss_sanity_check(
    model: AudioFIMCausalLM,
    dataset: AudioFIMCausalDataset,
    collator: AudioFIMCausalCollator,
    *,
    num_examples: int,
    use_bf16: bool,
    use_fp16: bool,
) -> None:
    """
    Print a first-batch CE/logit sanity check before Trainer takes over.

    For vocab_size=2075, all-zero logits should give CE ~= log(2075)=7.637.
    If zero-logit loss is normal but model loss is huge, inspect logit scale.
    If zero-logit loss is huge, labels/shift/vocab are wrong.
    """

    was_training = model.training
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda" and use_bf16:
        dtype = torch.bfloat16
    elif device.type == "cuda" and use_fp16:
        dtype = torch.float16

    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)

    examples = [dataset[i] for i in range(min(num_examples, len(dataset)))]
    batch = collator(examples)
    batch = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }

    outputs = model(**batch)
    logits = outputs.logits.detach().float()
    labels = batch["labels"]
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    supervised = int(valid.sum().item())

    zero_logits = torch.zeros(
        logits[:, :-1, :].shape,
        dtype=torch.float32,
        device=device,
    )
    zero_loss = torch.nn.functional.cross_entropy(
        zero_logits.view(-1, zero_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    target_ids = shift_labels[valid]
    shared_embed_head = (
        model.embed_tokens.weight.data_ptr() == model.out_head.weight.data_ptr()
    )

    print("=" * 70)
    print("[AudioFIM loss sanity]")
    print(f"examples:                    {len(examples)}")
    print(f"seq_len:                     {batch['input_ids'].shape[1]}")
    print(f"audio_bank_len:              {batch['audio_features'].shape[1]}")
    print(f"supervised labels:           {supervised}")
    print(f"target id min/max:           {int(target_ids.min())}/{int(target_ids.max())}")
    print(f"vocab_size:                  {model.config.vocab_size}")
    print(f"expected uniform CE:         {math.log(model.config.vocab_size):.4f}")
    print(f"zero-logit CE:               {float(zero_loss.detach().cpu()):.4f}")
    print(f"model CE:                    {float(outputs.loss.detach().float().cpu()):.4f}")
    print(
        "logits mean/std/min/max:     "
        f"{float(logits.mean().cpu()):.4f}/"
        f"{float(logits.std().cpu()):.4f}/"
        f"{float(logits.min().cpu()):.4f}/"
        f"{float(logits.max().cpu()):.4f}"
    )
    print(f"tie_word_embeddings config:  {model.config.tie_word_embeddings}")
    print(f"embed/out_head share memory: {shared_embed_head}")
    print("=" * 70)

    if was_training:
        model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train compact Step 2 audio-aware causal FIM transformer"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/audio_fim_causal"),
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data"),
        help="Dataset root containing motion_token_data and audio feature dirs",
    )
    parser.add_argument("--motion_token_dir", type=str, default=None)
    parser.add_argument("--audio_feat_dir", type=str, default=None)
    parser.add_argument("--motion2text_json", type=str, default=None)
    parser.add_argument("--train_split_file", type=str, default=None)
    parser.add_argument("--eval_split_file", type=str, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_windows_per_sequence", type=int, default=None)

    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Classic gap uses step=4, predicting 3 middle frames.",
    )
    parser.add_argument("--audio_fps", type=float, default=10.0)
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--min_history_frames", type=int, default=0)
    parser.add_argument("--max_history_frames", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--debug_examples", type=int, default=0)
    parser.add_argument(
        "--debug_loss_sanity",
        type=int,
        default=0,
        help=(
            "Run a first-batch CE/logit sanity check on this many examples "
            "before training. Use this when loss scale looks suspicious."
        ),
    )
    parser.add_argument("--profile_startup", action="store_true")
    parser.add_argument("--profile_collator_batches", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument(
        "--max_gap_frames",
        type=int,
        default=16,
        help="Reserve [LEN_1]..[LEN_N] tokens for future variable-gap training.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="Trainer reporting backend. Use 'wandb' to enable W&B logging.",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags, e.g. audio_fim,debug,step2.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Use offline when the server has no network.",
    )
    parser.add_argument(
        "--wandb_log_eval_videos",
        action="store_true",
        help="Log cheap GT-vs-pred token comparison videos during evaluation.",
    )
    parser.add_argument("--wandb_video_examples", type=int, default=2)
    parser.add_argument(
        "--wandb_video_every_n_evals",
        type=int,
        default=1,
        help="Log eval videos every N evaluation calls.",
    )
    parser.add_argument("--wandb_video_fps", type=int, default=4)
    parser.add_argument(
        "--wandb_log_eval_motion_videos",
        action="store_true",
        help="Decode eval predictions through RVQVAE and log GT-vs-pred motion videos.",
    )
    parser.add_argument("--wandb_motion_video_examples", type=int, default=2)
    parser.add_argument(
        "--wandb_motion_video_every_n_evals",
        type=int,
        default=1,
        help="Log decoded motion videos every N evaluation calls.",
    )
    parser.add_argument("--wandb_motion_video_fps", type=int, default=20)
    parser.add_argument(
        "--rvqvae_ckpt",
        type=str,
        default=str(PROJECT_DIR / "checkpoints/rvqvae/model/epoch_30.pth"),
        help="Step 1 RVQVAE checkpoint used to decode motion tokens for eval videos.",
    )
    parser.add_argument(
        "--rvqvae_mean_path",
        type=str,
        default=str(MOTION_GENERATION_DIR / "meta/mta_gen_demo/mean.npy"),
    )
    parser.add_argument(
        "--rvqvae_std_path",
        type=str,
        default=str(MOTION_GENERATION_DIR / "meta/mta_gen_demo/std.npy"),
    )
    parser.add_argument(
        "--motion_data_dir",
        type=str,
        default=None,
        help="Original motion_data directory used for hand placeholders in decoded eval videos.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start = time.perf_counter()

    if args.step - 1 > args.max_gap_frames:
        raise ValueError("--max_gap_frames must be >= --step - 1")

    data_dir = Path(args.data_dir)
    motion_token_dir = Path(args.motion_token_dir or data_dir / "motion_token_data")
    motion_data_dir = Path(args.motion_data_dir or data_dir / "motion_data")
    audio_fps_tag = format_fps_for_dir(args.audio_fps)
    audio_feat_dir = Path(
        args.audio_feat_dir
        or data_dir / f"audio_features_hubert_layer9_fps{audio_fps_tag}"
    )
    motion2text_json = args.motion2text_json or str(
        data_dir / "text_data/motion2text.json"
    )

    if args.report_to == "wandb" and args.wandb_log_eval_motion_videos:
        required_paths = {
            "rvqvae checkpoint": Path(args.rvqvae_ckpt),
            "rvqvae mean": Path(args.rvqvae_mean_path),
            "rvqvae std": Path(args.rvqvae_std_path),
            "motion_data dir": motion_data_dir,
        }
        missing = [
            f"{name}: {path}"
            for name, path in required_paths.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Decoded W&B motion videos need these files/dirs:\n"
                + "\n".join(f"  - {item}" for item in missing)
            )

    with timed_stage("load action text map", args.profile_startup):
        action_text_map = load_action_text_map(motion2text_json)

    with timed_stage("read/discover train split", args.profile_startup):
        train_split_names = read_split_file(args.train_split_file)
        train_names = discover_names(
            motion_token_dir,
            audio_feat_dir,
            train_split_names,
        )

    with timed_stage("load train sequences", args.profile_startup):
        train_sequences = load_sequences(
            train_names,
            motion_token_dir,
            audio_feat_dir,
            action_text_map,
            max_samples=args.max_samples,
            audio_fps=args.audio_fps,
            motion_fps=args.motion_fps,
        )

    eval_sequences = None
    with timed_stage("read/load eval split", args.profile_startup):
        eval_split_names = read_split_file(args.eval_split_file)
        if eval_split_names is not None:
            eval_names = discover_names(
                motion_token_dir,
                audio_feat_dir,
                eval_split_names,
            )
            eval_sequences = load_sequences(
                eval_names,
                motion_token_dir,
                audio_feat_dir,
                action_text_map,
                max_samples=args.max_samples,
                audio_fps=args.audio_fps,
                motion_fps=args.motion_fps,
            )
        else:
            train_sequences, eval_sequences = split_train_eval(
                train_sequences,
                eval_ratio=args.eval_ratio,
                seed=args.seed,
            )

    with timed_stage("build train FIM windows", args.profile_startup):
        train_dataset = AudioFIMCausalDataset(
            train_sequences,
            step=args.step,
            audio_fps=args.audio_fps,
            motion_fps=args.motion_fps,
            min_history_frames=args.min_history_frames,
            max_history_frames=args.max_history_frames,
            max_windows_per_sequence=args.max_windows_per_sequence,
            seed=args.seed,
        )

    eval_dataset = None
    if eval_sequences:
        with timed_stage("build eval FIM windows", args.profile_startup):
            eval_dataset = AudioFIMCausalDataset(
                eval_sequences,
                step=args.step,
                audio_fps=args.audio_fps,
                motion_fps=args.motion_fps,
                min_history_frames=args.min_history_frames,
                max_history_frames=args.max_history_frames,
                max_windows_per_sequence=args.max_windows_per_sequence,
                seed=args.seed + 1,
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No training windows were built. Check data paths/splits.")

    config = AudioFIMCausalConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        audio_feat_dim=args.audio_feat_dim,
        max_gap_frames=args.max_gap_frames,
        dropout=args.dropout,
    )
    model = AudioFIMCausalLM(config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params, trainable_params = count_parameters(model)
    default_run_name = (
        f"audio_fim_causal_step{args.step}_"
        f"afps{format_fps_for_dir(args.audio_fps)}_"
        f"mfps{format_fps_for_dir(args.motion_fps)}"
    )
    wandb_run_name = configure_wandb(args, default_run_name)

    print("=" * 70)
    print("Step 2 compact AudioFIM causal training")
    print(f"Output dir:       {args.output_dir}")
    print(f"Motion tokens:    {motion_token_dir}")
    if args.report_to == "wandb" and args.wandb_log_eval_motion_videos:
        print(f"Motion data:      {motion_data_dir}")
        print(f"RVQVAE decode:    {args.rvqvae_ckpt}")
    print(f"Audio features:   {audio_feat_dir}")
    print(f"Train sequences:  {len(train_sequences)}")
    print(f"Train windows:    {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval windows:     {len(eval_dataset)}")
    print(f"Gap setup:        step={args.step}, predict={args.step - 1} frames")
    print(f"History frames:   {args.min_history_frames}-{args.max_history_frames}")
    print(
        "Architecture:     "
        f"L={config.num_layers}, H={config.hidden_size}, "
        f"heads={config.num_heads}, ffn={config.intermediate_size}, "
        f"vocab={config.vocab_size}"
    )
    print(f"Parameters:       {total_params:,} total / {trainable_params:,} trainable")
    print(f"Tie embeddings:   {config.tie_word_embeddings}")
    print(
        "Shared emb/head:  "
        f"{model.embed_tokens.weight.data_ptr() == model.out_head.weight.data_ptr()}"
    )
    print(f"Report to:        {args.report_to}")
    if wandb_run_name:
        print(f"W&B run:          {wandb_run_name}")
    if args.report_to == "wandb" and args.wandb_log_eval_videos:
        print(f"W&B token video:  every {args.wandb_video_every_n_evals} eval(s)")
    if args.report_to == "wandb" and args.wandb_log_eval_motion_videos:
        print(
            "W&B motion video: "
            f"{args.wandb_motion_video_examples} sample(s), "
            f"{args.wandb_motion_video_fps} fps, "
            f"every {args.wandb_motion_video_every_n_evals} eval(s)"
        )
    print("=" * 70)

    collator = AudioFIMCausalCollator(
        config,
        max_length=args.max_length,
        debug_examples=args.debug_examples,
        profile_batches=args.profile_collator_batches,
    )

    if args.debug_loss_sanity > 0:
        run_loss_sanity_check(
            model,
            train_dataset,
            collator,
            num_examples=args.debug_loss_sanity,
            use_bf16=args.bf16,
            use_fp16=args.fp16,
        )

    training_args_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": args.report_to,
    }
    if wandb_run_name:
        training_args_kwargs["run_name"] = wandb_run_name

    if eval_dataset is not None and len(eval_dataset) > 0:
        training_args_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": args.eval_steps,
            }
        )

    with timed_stage("create TrainingArguments", args.profile_startup):
        training_args = TrainingArguments(**training_args_kwargs)

    callbacks = []
    if (
        args.report_to == "wandb"
        and args.wandb_log_eval_videos
        and eval_dataset is not None
        and len(eval_dataset) > 0
    ):
        callbacks.append(
            WandbEvalVideoCallback(
                eval_dataset=eval_dataset,
                num_examples=args.wandb_video_examples,
                every_n_evals=args.wandb_video_every_n_evals,
                fps=args.wandb_video_fps,
            )
        )
    if (
        args.report_to == "wandb"
        and args.wandb_log_eval_motion_videos
        and eval_dataset is not None
        and len(eval_dataset) > 0
    ):
        callbacks.append(
            WandbEvalMotionVideoCallback(
                eval_dataset=eval_dataset,
                motion_data_dir=motion_data_dir,
                rvqvae_ckpt=Path(args.rvqvae_ckpt),
                mean_path=Path(args.rvqvae_mean_path),
                std_path=Path(args.rvqvae_std_path),
                num_examples=args.wandb_motion_video_examples,
                every_n_evals=args.wandb_motion_video_every_n_evals,
                fps=args.wandb_motion_video_fps,
            )
        )

    with timed_stage("create Trainer", args.profile_startup):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )

    with timed_stage("trainer.train", True):
        trainer.train()

    with timed_stage("save model/token map", True):
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero():
            mapper = AudioFIMTokenMapper(config)
            mapper.save_json(Path(args.output_dir) / "compact_token_map.json")

    print(f"Saved compact AudioFIM checkpoint to: {args.output_dir}")
    print(f"[Timing] total script runtime: {time.perf_counter() - run_start:.3f}s")


if __name__ == "__main__":
    main()

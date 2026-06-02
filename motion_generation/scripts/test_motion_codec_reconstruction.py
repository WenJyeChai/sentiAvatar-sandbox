#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test RVQVAE motion encoder/decoder reconstruction on one dataset sample.

The script selects one sample, loads its motion/audio/text, encodes the body
motion with RVQVAE, decodes the tokens back to motion features, and writes both
the original motion and reconstructed motion as BVH files.
"""

import argparse
import ast
import json
import random
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
sys.path.insert(0, str(MODULE_DIR))

from actions.postprocess import MotionPostprocesser
from configs.default_config import Config
from models.rvqvae import RVQVAE
from utils.constants import BODY_JOINTS_ID, LEFT_HAND_JOINTS_ID, RIGHT_HAND_JOINTS_ID
from utils.rotation_utils import sixd_to_quaternion


def fixseed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_value(value_str: str):
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


def parse_opt_txt(opt_path: Path) -> Dict:
    config = {}
    with open(opt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = parse_value(value)
    return config


def load_config_from_checkpoint(checkpoint_path: Path) -> Config:
    opt_path = checkpoint_path.parent.parent / "opt.txt"
    if not opt_path.exists():
        raise FileNotFoundError(f"Config file not found: {opt_path}")

    opt = parse_opt_txt(opt_path)
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
    config.model.quantize_dropout_cutoff_index = opt.get("quantize_dropout_cutoff_index", 1)
    config.model.use_whole_encoder = opt.get("use_whole_encoder", False)
    config.model.mu = opt.get("mu", 0.99)
    return config


def load_rvqvae_model(checkpoint_path: Path, config: Config, device: torch.device) -> RVQVAE:
    model = RVQVAE(
        config=config,
        input_dim=config.data.whole_dim,
        nb_code=config.model.nb_code,
        code_dim=config.model.code_dim,
        output_dim=config.model.code_dim,
        down_t=config.model.down_t,
        stride_t=config.model.stride_t,
        width=config.model.width,
        depth=config.model.depth,
        dilation_growth_rate=config.model.dilation_growth_rate,
        activation=config.model.vq_act,
        norm=config.model.vq_norm,
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
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / (kernel.sum() + 1e-12)


def smooth_motion_gaussian(x: torch.Tensor, kernel_size: int = 7, sigma: float = 2.0):
    if x.shape[1] <= 1:
        return x
    kernel_size = min(kernel_size, x.shape[1] if x.shape[1] % 2 == 1 else x.shape[1] - 1)
    kernel_size = max(kernel_size, 3)
    pad = kernel_size // 2

    bsz, frames, joints, dims = x.shape
    y = x.permute(0, 2, 3, 1).contiguous().view(bsz, joints * dims, frames)
    kernel = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    weight = kernel.view(1, 1, kernel_size).repeat(joints * dims, 1, 1)
    pad_mode = "reflect" if frames > pad else "replicate"
    y = F.pad(y, (pad, pad), mode=pad_mode)
    y = F.conv1d(y, weight=weight, groups=joints * dims)
    return y.view(bsz, joints, dims, frames).permute(0, 3, 1, 2).contiguous()


def resample_fps(x: torch.Tensor, src_fps: float, tgt_fps: float):
    bsz, frames, joints, dims = x.shape
    new_frames = max(1, int(round(frames * (tgt_fps / src_fps))))
    y = x.permute(0, 2, 3, 1).contiguous().view(bsz, joints * dims, frames)
    y = F.interpolate(y, size=new_frames, mode="linear", align_corners=False)
    return y.view(bsz, joints, dims, new_frames).permute(0, 3, 1, 2).contiguous()


def smooth_then_resample(x: torch.Tensor, src_fps: float, tgt_fps: float):
    return resample_fps(smooth_motion_gaussian(x), src_fps, tgt_fps)


def sample_path(root: Path, subdir: str, name: str, suffix: str) -> Path:
    return root / subdir / Path(*PurePosixPath(name).parts).with_suffix(suffix)


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")


def load_motion_dict(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected motion npy dict, got {type(data)} from {path}")
    for key in ("body", "left", "right"):
        if key not in data:
            raise KeyError(f"Motion file missing key '{key}': {path}")
    return data


def load_name_list(split_path: Path) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def load_text_map(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def valid_samples(dataset_root: Path, names: Iterable[str], text_map: Dict[str, str]) -> List[str]:
    valid = []
    for name in names:
        motion_path = sample_path(dataset_root, "motion_data", name, ".npy")
        wav_path = sample_path(dataset_root, "wav_data", name, ".wav")
        if motion_path.exists() and wav_path.exists() and name in text_map:
            valid.append(name)
    return valid


def choose_sample(args, dataset_root: Path, text_map: Dict[str, str]) -> str:
    if args.sample_name:
        return args.sample_name.strip().replace("\\", "/")

    split_path = dataset_root / "split" / f"{args.split}_file_list.txt"
    names = load_name_list(split_path)
    candidates = valid_samples(dataset_root, names, text_map)
    if not candidates:
        raise RuntimeError(f"No valid samples found in split file: {split_path}")

    rng = random.Random(args.seed)
    return rng.choice(candidates)


def preprocess_body_for_codec(
    motion_dict: Dict[str, np.ndarray],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    body = torch.tensor(motion_dict["body"], dtype=torch.float32, device=device).clone()
    body[:, 2] = body[:, 2] - body[0, 2]
    body[1:, :3] = body[1:, :3] - body[:-1, :3]
    return ((body - mean) / std).unsqueeze(0)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def encode_decode_once(model, body_input: torch.Tensor, device: torch.device):
    synchronize_if_needed(device)
    t0 = time.perf_counter()
    encoded = model.encode(body_input)
    synchronize_if_needed(device)
    t1 = time.perf_counter()

    code_idx = {"body": encoded["code_idx"]["body"]}
    decoded = model.forward_decoder(code_idx)
    synchronize_if_needed(device)
    t2 = time.perf_counter()

    return encoded, decoded, (t1 - t0), (t2 - t1)


def pad_or_trim_hand_motion(hand: torch.Tensor, frames: int) -> torch.Tensor:
    if hand.shape[1] == frames:
        return hand
    if hand.shape[1] > frames:
        return hand[:, :frames]

    pad = frames - hand.shape[1]
    return F.pad(hand.permute(0, 2, 1), (0, pad), mode="replicate").permute(0, 2, 1)


def body_features_to_bvh_motion(
    body_features: torch.Tensor,
    motion_dict: Dict[str, np.ndarray],
    device: torch.device,
    src_fps: float,
    tgt_fps: float,
) -> Dict[str, np.ndarray]:
    body_features = body_features.to(device)
    frames = body_features.shape[0]

    offset_frame0 = torch.tensor([0.0, 0.0, 102.0], device=device)
    offset_vel = body_features[:, :3].clone()
    for i in range(1, offset_vel.shape[0]):
        offset_vel[i] = offset_vel[i] + offset_vel[i - 1]
    offset = (offset_vel + offset_frame0).reshape(1, frames, 1, 3)

    body_6d = body_features[:, 3:].reshape(1, frames, 25, 6)

    left = torch.tensor(motion_dict["left"], dtype=torch.float32, device=device).unsqueeze(0)
    right = torch.tensor(motion_dict["right"], dtype=torch.float32, device=device).unsqueeze(0)
    left = pad_or_trim_hand_motion(left, frames).reshape(1, frames, 20, 6)
    right = pad_or_trim_hand_motion(right, frames).reshape(1, frames, 20, 6)

    body_6d = smooth_then_resample(body_6d, src_fps=src_fps, tgt_fps=tgt_fps)
    left = smooth_then_resample(left, src_fps=src_fps, tgt_fps=tgt_fps)
    right = smooth_then_resample(right, src_fps=src_fps, tgt_fps=tgt_fps)
    offset = smooth_then_resample(offset, src_fps=src_fps, tgt_fps=tgt_fps)

    out_frames = body_6d.shape[1]
    body_quat = sixd_to_quaternion(body_6d.reshape(-1, 6)).reshape(1, out_frames, 25, 4)
    left_quat = sixd_to_quaternion(left.reshape(-1, 6)).reshape(1, out_frames, 20, 4)
    right_quat = sixd_to_quaternion(right.reshape(-1, 6)).reshape(1, out_frames, 20, 4)

    merged = torch.zeros(out_frames, 63, 4, device=device)
    merged[:, BODY_JOINTS_ID] = body_quat[0]
    merged[:, LEFT_HAND_JOINTS_ID[1:]] = left_quat[0, :, 1:]
    merged[:, RIGHT_HAND_JOINTS_ID[1:]] = right_quat[0, :, 1:]

    return {
        "offset": offset.reshape(out_frames, 3).detach().cpu().numpy(),
        "quat": merged.detach().cpu().numpy(),
    }


def reconstruction_metrics(target: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    frames = min(target.shape[0], pred.shape[0])
    dims = min(target.shape[1], pred.shape[1])
    diff = pred[:frames, :dims] - target[:frames, :dims]
    return {
        "compare_frames": int(frames),
        "compare_dims": int(dims),
        "mae": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "max_abs": float(diff.abs().max().item()),
    }


def parse_args():
    default_dataset = PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs"
    default_ckpt = PROJECT_DIR / "checkpoints" / "rvqvae" / "model" / "epoch_30.pth"
    default_out = PROJECT_DIR / "output" / "motion_codec_test"

    parser = argparse.ArgumentParser(
        description="Random-sample RVQVAE encoder/decoder BVH reconstruction test."
    )
    parser.add_argument("--dataset_root", type=Path, default=default_dataset)
    parser.add_argument("--checkpoint_path", type=Path, default=default_ckpt)
    parser.add_argument("--output_dir", type=Path, default=default_out)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--src_fps", type=float, default=None)
    parser.add_argument("--tgt_fps", type=float, default=30.0)
    parser.add_argument("--save_json", action="store_true", help="Also save UE-style anim JSON files.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint_path.resolve()

    text_path = dataset_root / "text_data" / "motion2text.json"
    text_map = load_text_map(text_path)
    name = choose_sample(args, dataset_root, text_map)

    motion_path = sample_path(dataset_root, "motion_data", name, ".npy")
    wav_path = sample_path(dataset_root, "wav_data", name, ".wav")
    text = text_map.get(name, "")
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    config = load_config_from_checkpoint(checkpoint_path)
    src_fps = float(config.data.fps if args.src_fps is None else args.src_fps)
    fixseed(args.seed)

    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")

    model = load_rvqvae_model(checkpoint_path, config, device)
    mean = torch.tensor(
        np.load(MODULE_DIR / "meta" / "mta_gen_demo" / "mean.npy"),
        dtype=torch.float32,
        device=device,
    )
    std = torch.tensor(
        np.load(MODULE_DIR / "meta" / "mta_gen_demo" / "std.npy"),
        dtype=torch.float32,
        device=device,
    )

    motion_dict = load_motion_dict(motion_path)
    body_input = preprocess_body_for_codec(motion_dict, mean, std, device)
    body_target_denorm = body_input[0] * std + mean

    with torch.no_grad():
        encoded, decoded_norm, encode_time, decode_time = encode_decode_once(model, body_input, device)

    decoded_body_denorm = decoded_norm[0] * std + mean
    metrics = reconstruction_metrics(body_target_denorm, decoded_body_denorm)

    sample_out = output_dir / safe_name(name)
    sample_out.mkdir(parents=True, exist_ok=True)
    postprocesser = MotionPostprocesser()

    original_motion = body_features_to_bvh_motion(body_target_denorm, motion_dict, device, src_fps, args.tgt_fps)
    reconstructed_motion = body_features_to_bvh_motion(decoded_body_denorm, motion_dict, device, src_fps, args.tgt_fps)

    original_bvh = sample_out / "original.bvh"
    reconstructed_bvh = sample_out / "reconstructed.bvh"
    postprocesser.save_quat_motion_to_bvh(original_motion, save_path=str(original_bvh))
    postprocesser.save_quat_motion_to_bvh(reconstructed_motion, save_path=str(reconstructed_bvh))

    if args.save_json:
        with open(sample_out / "original.json", "w", encoding="utf-8") as f:
            json.dump(postprocesser.convert_quat_motion_to_ue_from_bvh(original_motion), f, indent=2, ensure_ascii=False)
        with open(sample_out / "reconstructed.json", "w", encoding="utf-8") as f:
            json.dump(postprocesser.convert_quat_motion_to_ue_from_bvh(reconstructed_motion), f, indent=2, ensure_ascii=False)

    copied_wav = sample_out / "audio.wav"
    shutil.copy2(wav_path, copied_wav)

    tokens = encoded["code_idx"]["body"].squeeze(0).detach().cpu().numpy()
    metadata = {
        "sample_name": name,
        "motion_path": str(motion_path),
        "audio_path": str(wav_path),
        "text": text,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "src_fps": src_fps,
        "tgt_fps": args.tgt_fps,
        "body_frames": int(motion_dict["body"].shape[0]),
        "token_shape": list(tokens.shape),
        "encode_time_sec": encode_time,
        "decode_time_sec": decode_time,
        "body_feature_metrics_denorm": metrics,
        "note": "RVQVAE reconstructs body channels; left/right hand channels are copied from the original sample for BVH export.",
        "outputs": {
            "original_bvh": str(original_bvh),
            "reconstructed_bvh": str(reconstructed_bvh),
            "audio": str(copied_wav),
        },
    }
    with open(sample_out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(sample_out / "tokens.json", "w", encoding="utf-8") as f:
        json.dump({"body": tokens.tolist()}, f)

    print("\nDone.")
    print(f"Sample: {name}")
    print(f"Text: {text}")
    print(f"Encode: {encode_time * 1000:.2f} ms, decode: {decode_time * 1000:.2f} ms")
    print(f"MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}")
    print(f"Output: {sample_out}")


if __name__ == "__main__":
    main()

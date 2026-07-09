# %% [markdown]
# # RVQ-VAE Codec Comparison: Original Body Codec vs Multipart Codec
#
# This notebook compares the original SentiAvatar RVQ-VAE checkpoint against the
# new multipart RVQ-VAE checkpoints.
#
# Main report metrics:
#
# - evaluator-latent FID/diversity
# - ChronAccRet R@K semantic retrieval
#
# Diagnostic metrics:
#
# - body/part reconstruction RMSE/MAE
# - velocity and acceleration RMSE
# - 6D rotation geodesic error
# - root trajectory drift
# - codebook usage/perplexity
# - model size and token density
#
# Important evaluator caveat: the official evaluator config uses `nfeats: 153`,
# and `PredMotionTextDataset` currently feeds only `body` to the evaluator. These
# metrics judge body/semantic motion quality, not hand-finger fidelity. Hand
# quality is covered by the diagnostic reconstruction section.

# %%
from __future__ import annotations

import ast
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display
from tqdm.auto import tqdm


def find_project_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "motion_generation").exists() and (candidate / "evaluation").exists():
            return candidate
    raise RuntimeError("Could not find project root containing motion_generation/ and evaluation/")


PROJECT_DIR = find_project_root()
MOTION_GENERATION_DIR = PROJECT_DIR / "motion_generation"
EVALUATION_DIR = PROJECT_DIR / "evaluation"

for path in [str(MOTION_GENERATION_DIR), str(PROJECT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.default_config import Config  # noqa: E402
from models.multipart_rvqvae import MultiPartRVQVAE  # noqa: E402
from models.rvqvae import RVQVAE  # noqa: E402
from utils.multipart_motion import (  # noqa: E402
    FEET_JOINTS,
    LOWER_BODY_JOINTS,
    PART_DIMS,
    PART_ORDER,
    UPPER_BODY_JOINTS,
    PartNormalizer,
    canonicalize_body_root,
    load_motion_dict,
    load_name_list,
    merge_parts_to_legacy_motion,
    motion_path_for_name,
    split_motion_parts,
)

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)


# %% [markdown]
# ## Config
#
# Defaults are set for this repository layout. If your multipart checkpoints
# live somewhere else, edit `NEW_PART_CKPTS` or set these environment variables:
# `RVQ_UPPER_CKPT`, `RVQ_LOWER_CKPT`, `RVQ_FEET_CKPT`, `RVQ_HANDS_CKPT`.

# %%
DEVICE = torch.device(os.environ.get("RVQ_COMPARE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 1234

DATA_DIR = PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs"
MOTION_DIR = DATA_DIR / "motion_data"
EVAL_SPLIT_FILE = DATA_DIR / "split" / "val_file_list.txt"
MOTION2TEXT_JSON = DATA_DIR / "text_data" / "motion2text.json"

OLD_RVQVAE_CKPT = PROJECT_DIR / "checkpoints" / "rvqvae" / "model" / "epoch_30.pth"
OLD_RVQVAE_MEAN_PATH = MOTION_GENERATION_DIR / "meta" / "mta_gen_demo" / "mean.npy"
OLD_RVQVAE_STD_PATH = MOTION_GENERATION_DIR / "meta" / "mta_gen_demo" / "std.npy"

DEFAULT_MULTIPART_ROOT = PROJECT_DIR / "checkpoints" / "multipart_rvqvae"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


NEW_PART_CKPTS = {
    "upper": env_path("RVQ_UPPER_CKPT", DEFAULT_MULTIPART_ROOT / "rvq_upper_512x4_bs256_cosine" / "model" / "best.pth"),
    "lower": env_path("RVQ_LOWER_CKPT", DEFAULT_MULTIPART_ROOT / "rvq_lower_512x4_bs256_cosine" / "model" / "best.pth"),
    "feet": env_path("RVQ_FEET_CKPT", DEFAULT_MULTIPART_ROOT / "rvq_feet_512x4_bs256_cosine" / "model" / "best.pth"),
    "hands": env_path("RVQ_HANDS_CKPT", DEFAULT_MULTIPART_ROOT / "rvq_hands_512x4_bs256_cosine" / "model" / "best.pth"),
}

OUT_ROOT = PROJECT_DIR / "motion_generation" / "outputs" / "rvqvae_codec_compare"
GT_DIR = OUT_ROOT / "gt"
OLD_DIR = OUT_ROOT / "old_rvqvae_body"
NEW_DIR = OUT_ROOT / "new_multipart_full"

EVALUATOR_CKPT = PROJECT_DIR / "checkpoints" / "eval_model" / "best_model.pt"
EVALUATOR_CFG_PATH = EVALUATION_DIR / "config" / "train_bert_orig.yaml"
EVALUATOR_STATS_DIR = EVALUATION_DIR / "stats" / "humanml3d" / "guoh3dfeats"

# ChronAccRet TextEncoder hardcodes the original author's bert-base-chinese path.
# Set this to a local folder if the machine is offline.
EVALUATOR_TEXT_MODEL_PATH = os.environ.get("EVALUATOR_TEXT_MODEL_PATH", "bert-base-chinese")

# Use None for the full val split. Set a small integer for a smoke test.
MAX_EVAL_CLIPS: Optional[int] = None

# The released dataset has mixed root schemas. This comparison canonicalizes GT
# and model inputs to frame-delta root before reconstruction/export.
ROOT_ABS_THRESHOLD = 10.0
OLD_ROOT_MODE = "canonical"  # "canonical" or "legacy_preprocess"

# Heavy stages. Export first, inspect manifest/diagnostics, then flip these on.
RUN_EXPORT_RECONSTRUCTIONS = True
RUN_CODEBOOK_USAGE = True
RUN_EVALUATOR_METRICS = False
RUN_RETRIEVAL_RK = False

EVALUATOR_BATCH_SIZE = 64
DIVERSITY_TIMES = 300

print("PROJECT_DIR:", PROJECT_DIR)
print("DEVICE:", DEVICE)


# %%
def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def torch_load_trusted(path: Path, map_location=None):
    """Load a trusted local checkpoint across PyTorch 2.5/2.6 defaults.

    PyTorch 2.6 changed torch.load's default to weights_only=True. These local
    checkpoints intentionally store small metadata objects such as pathlib.Path
    inside args, so the comparison notebook must opt into full unpickling.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def auto_find_part_checkpoint(part: str) -> Optional[Path]:
    roots = [PROJECT_DIR / "checkpoints", PROJECT_DIR / "tmp", OUT_ROOT]
    patterns = [
        f"**/*{part}*/model/best.pth",
        f"**/*{part}*best*.pth",
        f"**/*{part}*/model/final.pth",
    ]
    found: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(root.glob(pattern))
    found = sorted(set(path.resolve() for path in found if path.is_file()))
    return found[0] if len(found) == 1 else None


resolved_part_ckpts = {}
for part, ckpt_path in NEW_PART_CKPTS.items():
    if ckpt_path.exists():
        resolved_part_ckpts[part] = ckpt_path.resolve()
    else:
        auto_path = auto_find_part_checkpoint(part)
        resolved_part_ckpts[part] = auto_path if auto_path is not None else ckpt_path

path_rows = [
    {"item": "data_dir", "path": DATA_DIR, "exists": DATA_DIR.exists()},
    {"item": "motion_dir", "path": MOTION_DIR, "exists": MOTION_DIR.exists()},
    {"item": "eval_split_file", "path": EVAL_SPLIT_FILE, "exists": EVAL_SPLIT_FILE.exists()},
    {"item": "motion2text", "path": MOTION2TEXT_JSON, "exists": MOTION2TEXT_JSON.exists()},
    {"item": "old_rvqvae", "path": OLD_RVQVAE_CKPT, "exists": OLD_RVQVAE_CKPT.exists()},
    {"item": "old_mean", "path": OLD_RVQVAE_MEAN_PATH, "exists": OLD_RVQVAE_MEAN_PATH.exists()},
    {"item": "old_std", "path": OLD_RVQVAE_STD_PATH, "exists": OLD_RVQVAE_STD_PATH.exists()},
    {"item": "evaluator_ckpt", "path": EVALUATOR_CKPT, "exists": EVALUATOR_CKPT.exists()},
    {"item": "evaluator_cfg", "path": EVALUATOR_CFG_PATH, "exists": EVALUATOR_CFG_PATH.exists()},
]
for part, path in resolved_part_ckpts.items():
    path_rows.append({"item": f"new_{part}", "path": path, "exists": Path(path).exists()})

display(pd.DataFrame(path_rows))


# %% [markdown]
# ## Model Loading Helpers

# %%
def parse_opt_value(value_str: str):
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


def parse_opt_txt(opt_path: Path) -> Dict[str, object]:
    result = {}
    with open(opt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = parse_opt_value(value)
    return result


def load_old_rvqvae_config(checkpoint_path: Path) -> Config:
    opt_path = checkpoint_path.parent.parent / "opt.txt"
    require_path(opt_path, "old RVQVAE opt.txt")
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
    config.model.quantize_dropout_prob = opt.get("quantize_dropout_prob", 0.0)
    config.model.quantize_dropout_cutoff_index = opt.get("quantize_dropout_cutoff_index", 1)
    config.model.use_whole_encoder = opt.get("use_whole_encoder", False)
    config.model.mu = opt.get("mu", 0.99)
    config.unit_length = config.model.down_t * config.model.stride_t
    return config


def load_old_rvqvae_model(checkpoint_path: Path, device: torch.device) -> Tuple[RVQVAE, Config]:
    config = load_old_rvqvae_config(checkpoint_path)
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
    checkpoint = torch_load_trusted(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model.to(device).eval(), config


def ckpt_arg(args: Mapping[str, object], key: str, default):
    value = args.get(key, default)
    if isinstance(value, Path):
        return value
    return value


@dataclass
class LoadedMultipartModel:
    part: str
    checkpoint_path: Path
    model: MultiPartRVQVAE
    normalizer: PartNormalizer
    nb_code: int
    num_quantizers: int
    unit_length: int


def infer_part_from_path(path: Path) -> Optional[str]:
    lower = str(path).lower()
    for part in PART_ORDER:
        if part in lower:
            return part
    return None


def load_multipart_checkpoint(checkpoint_path: Path, device: torch.device) -> LoadedMultipartModel:
    checkpoint_path = require_path(checkpoint_path, "multipart checkpoint")
    checkpoint = torch_load_trusted(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}

    part_order = model_config.get("part_order") or args.get("parts") or [infer_part_from_path(checkpoint_path)]
    if isinstance(part_order, str):
        part_order = [part_order]
    part_order = [part for part in part_order if part is not None]
    if len(part_order) != 1:
        raise ValueError(
            f"This comparison expects one checkpoint per part. Got part_order={part_order} for {checkpoint_path}"
        )
    part = part_order[0]

    nb_code = int(args.get("codebook_size", model_config.get("nb_code", 512)))
    code_dim = int(args.get("code_dim", model_config.get("code_dim", 512)))
    num_quantizers = int(args.get("num_quantizers", model_config.get("num_quantizers", 4)))
    down_t = int(args.get("down_t", 1))
    stride_t = int(args.get("stride_t", 2))

    model = MultiPartRVQVAE(
        part_dims=PART_DIMS,
        part_order=[part],
        nb_code=nb_code,
        code_dim=code_dim,
        num_quantizers=num_quantizers,
        down_t=down_t,
        stride_t=stride_t,
        width=int(args.get("width", 512)),
        depth=int(args.get("depth", 3)),
        dilation_growth_rate=int(args.get("dilation_growth_rate", 3)),
        activation=str(args.get("activation", "relu")),
        norm=args.get("norm", None),
        vq_cnn_depth=int(args.get("vq_cnn_depth", 3)),
        shared_codebook=bool(args.get("shared_codebook", False)),
        quantize_dropout_prob=float(args.get("quantize_dropout_prob", 0.0)),
        quantize_dropout_cutoff_index=int(args.get("quantize_dropout_cutoff_index", 1)),
        mu=float(args.get("mu", 0.99)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    normalizer_path = Path(str(checkpoint.get("normalizer_path", ""))).expanduser()
    if not normalizer_path.exists():
        normalizer_path = checkpoint_path.parent.parent / "meta" / "normalizer.npz"
    normalizer = PartNormalizer.load(require_path(normalizer_path, f"{part} normalizer"))

    return LoadedMultipartModel(
        part=part,
        checkpoint_path=checkpoint_path,
        model=model,
        normalizer=normalizer,
        nb_code=nb_code,
        num_quantizers=num_quantizers,
        unit_length=down_t * stride_t,
    )


# %% [markdown]
# ## Motion and Metric Helpers

# %%
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trim_motion(motion: Mapping[str, np.ndarray], frames: int) -> Dict[str, np.ndarray]:
    return {
        "body": np.asarray(motion["body"], dtype=np.float32)[:frames],
        "left": np.asarray(motion["left"], dtype=np.float32)[:frames],
        "right": np.asarray(motion["right"], dtype=np.float32)[:frames],
    }


def canonical_motion_dict(motion: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    body, schema, mean_norm = canonicalize_body_root(
        np.asarray(motion["body"], dtype=np.float32),
        abs_threshold=ROOT_ABS_THRESHOLD,
    )
    left = np.asarray(motion["left"], dtype=np.float32)
    right = np.asarray(motion["right"], dtype=np.float32)
    frames = min(len(body), len(left), len(right))
    return trim_motion({"body": body, "left": left, "right": right}, frames), {
        "root_schema": schema,
        "root_mean_norm": mean_norm,
        "frames": frames,
    }


def old_legacy_preprocess_body(motion: Mapping[str, np.ndarray]) -> np.ndarray:
    body = np.asarray(motion["body"], dtype=np.float32).copy()
    if len(body):
        body[:, 2] = body[:, 2] - body[0, 2]
    if len(body) > 1:
        body[1:, :3] = body[1:, :3] - body[:-1, :3]
    return body


def match_length_np(value: np.ndarray, target_length: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if len(value) == target_length:
        return value
    if len(value) > target_length:
        return value[:target_length]
    if len(value) == 0:
        return np.zeros((target_length, value.shape[-1]), dtype=np.float32)
    pad = np.repeat(value[-1:], target_length - len(value), axis=0)
    return np.concatenate([value, pad], axis=0)


def match_length_tensor(value: torch.Tensor, target_length: int) -> torch.Tensor:
    if value.shape[1] == target_length:
        return value
    if value.shape[1] > target_length:
        return value[:, :target_length]
    pad = target_length - value.shape[1]
    tail = value[:, -1:, :] if value.shape[1] else torch.zeros(
        value.shape[0], 1, value.shape[2], dtype=value.dtype, device=value.device
    )
    return torch.cat([value, tail.repeat(1, pad, 1)], dim=1)


@torch.no_grad()
def reconstruct_old_body(
    model: RVQVAE,
    motion: Mapping[str, np.ndarray],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    mode: str = "canonical",
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "canonical":
        canonical_motion, _ = canonical_motion_dict(motion)
        body = canonical_motion["body"]
    elif mode == "legacy_preprocess":
        body = old_legacy_preprocess_body(motion)
    else:
        raise ValueError("mode must be 'canonical' or 'legacy_preprocess'")

    x = torch.tensor(body, dtype=torch.float32, device=device)
    x_norm = ((x - mean.to(device)) / std.to(device)).unsqueeze(0)
    encoded = model.encode(x_norm)
    code_idx = encoded["code_idx"]["body"]
    decoded_norm = model.forward_decoder({"body": code_idx})
    decoded_norm = match_length_tensor(decoded_norm, x_norm.shape[1])
    decoded = decoded_norm[0].float() * std.to(device) + mean.to(device)
    return decoded.detach().cpu().numpy().astype(np.float32), code_idx.detach().cpu().numpy()


@torch.no_grad()
def reconstruct_new_parts(
    gt_parts: Mapping[str, np.ndarray],
    part_models: Mapping[str, LoadedMultipartModel],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rec_parts: Dict[str, np.ndarray] = {}
    code_indices: Dict[str, np.ndarray] = {}
    for part in PART_ORDER:
        loaded = part_models[part]
        normalizer = loaded.normalizer
        x_np = normalizer.normalize(part, gt_parts[part])
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        code_idx = loaded.model.encode({part: x})[part]
        decoded_norm = loaded.model.decode({part: code_idx})[part]
        decoded_norm = match_length_tensor(decoded_norm, x.shape[1])
        decoded = normalizer.denormalize_tensor(part, decoded_norm)[0]
        rec_parts[part] = decoded.detach().cpu().numpy().astype(np.float32)
        code_indices[part] = code_idx.detach().cpu().numpy()
    return rec_parts, code_indices


def body_joint_features(body: np.ndarray, joints: Sequence[int]) -> np.ndarray:
    return np.concatenate(
        [body[:, 3 + joint * 6 : 3 + (joint + 1) * 6] for joint in joints],
        axis=-1,
    ).astype(np.float32, copy=False)


def body_parts_from_motion(motion: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    parts, _ = split_motion_parts(motion, abs_threshold=ROOT_ABS_THRESHOLD, force_root_schema="delta")
    return parts


def save_motion_npy(path: Path, name: str, motion: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = min(len(motion["body"]), len(motion["left"]), len(motion["right"]))
    payload = {
        "name": name,
        "body": np.asarray(motion["body"], dtype=np.float32)[:frames],
        "left": np.asarray(motion["left"], dtype=np.float32)[:frames],
        "right": np.asarray(motion["right"], dtype=np.float32)[:frames],
    }
    np.save(path, payload)


def clean_eval_dir(path: Path, suffix: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old_file in path.glob(f"*_{suffix}.npy"):
        old_file.unlink()


def feature_error(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    frames = min(len(gt), len(pred))
    gt = gt[:frames]
    pred = pred[:frames]
    diff = pred - gt
    out = {
        "frames": int(frames),
        "feature_mae": float(np.mean(np.abs(diff))),
        "feature_mse": float(np.mean(diff ** 2)),
        "feature_rmse": float(np.sqrt(np.mean(diff ** 2))),
    }
    if frames > 1:
        vel_diff = np.diff(pred, axis=0) - np.diff(gt, axis=0)
        out["velocity_rmse"] = float(np.sqrt(np.mean(vel_diff ** 2)))
    else:
        out["velocity_rmse"] = np.nan
    if frames > 2:
        acc_diff = np.diff(pred, n=2, axis=0) - np.diff(gt, n=2, axis=0)
        out["acceleration_rmse"] = float(np.sqrt(np.mean(acc_diff ** 2)))
    else:
        out["acceleration_rmse"] = np.nan
    return out


def sixd_to_rotmat(x6: torch.Tensor) -> torch.Tensor:
    x6 = x6.reshape(-1, 6)
    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_degrees(gt6: np.ndarray, pred6: np.ndarray) -> np.ndarray:
    gt6 = np.asarray(gt6, dtype=np.float32)
    pred6 = np.asarray(pred6, dtype=np.float32)
    frames = min(len(gt6), len(pred6))
    gt6 = gt6[:frames].reshape(-1, 6)
    pred6 = pred6[:frames].reshape(-1, 6)
    if gt6.size == 0:
        return np.asarray([], dtype=np.float32)
    with torch.no_grad():
        r_gt = sixd_to_rotmat(torch.from_numpy(gt6))
        r_pred = sixd_to_rotmat(torch.from_numpy(pred6))
        rel = torch.matmul(r_pred.transpose(-1, -2), r_gt)
        trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
        cos = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        deg = torch.acos(cos) * (180.0 / math.pi)
    return deg.cpu().numpy().astype(np.float32)


def geodesic_metrics(gt6: np.ndarray, pred6: np.ndarray) -> Dict[str, float]:
    deg = geodesic_degrees(gt6, pred6)
    if deg.size == 0:
        return {"rotation_geodesic_deg_mean": np.nan, "rotation_geodesic_deg_p95": np.nan}
    return {
        "rotation_geodesic_deg_mean": float(np.mean(deg)),
        "rotation_geodesic_deg_p95": float(np.percentile(deg, 95)),
    }


def integrate_root_delta(body: np.ndarray, start: Sequence[float] = (0.0, 0.0, 102.0)) -> np.ndarray:
    root_delta = np.asarray(body[:, :3], dtype=np.float32)
    return np.cumsum(root_delta, axis=0) + np.asarray(start, dtype=np.float32).reshape(1, 3)


def root_metrics(gt_body: np.ndarray, pred_body: np.ndarray) -> Dict[str, float]:
    frames = min(len(gt_body), len(pred_body))
    if frames == 0:
        return {"root_delta_rmse": np.nan, "root_traj_rmse": np.nan, "root_final_drift_cm": np.nan}
    gt_body = gt_body[:frames]
    pred_body = pred_body[:frames]
    root_diff = pred_body[:, :3] - gt_body[:, :3]
    gt_traj = integrate_root_delta(gt_body)
    pred_traj = integrate_root_delta(pred_body)
    traj_diff = pred_traj - gt_traj
    return {
        "root_delta_rmse": float(np.sqrt(np.mean(root_diff ** 2))),
        "root_traj_rmse": float(np.sqrt(np.mean(traj_diff ** 2))),
        "root_final_drift_cm": float(np.linalg.norm(traj_diff[-1])),
    }


def part_reconstruction_row(
    codec: str,
    name: str,
    part: str,
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, object]:
    row: Dict[str, object] = {"codec": codec, "name": name, "part": part}
    row.update(feature_error(gt, pred))
    if part == "lower":
        row.update(geodesic_metrics(gt[:, 3:], pred[:, 3:]))
    else:
        row.update(geodesic_metrics(gt, pred))
    return row


def codebook_usage_rows(
    label: str,
    arrays: Sequence[np.ndarray],
    nb_code: int,
) -> List[Dict[str, object]]:
    if not arrays:
        return []
    flat = []
    for value in arrays:
        arr = np.asarray(value)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            arr = arr.reshape(-1, arr.shape[-1])
        else:
            raise ValueError(f"Unexpected code index shape for {label}: {arr.shape}")
        flat.append(arr)
    idx = np.concatenate(flat, axis=0)
    rows = []
    for q in range(idx.shape[-1]):
        values = idx[:, q]
        values = values[values >= 0]
        if values.size == 0:
            rows.append({
                "codec_part": label,
                "quantizer": q,
                "tokens": 0,
                "active_codes": 0,
                "dead_code_fraction": 1.0,
                "entropy": 0.0,
                "perplexity": 0.0,
            })
            continue
        counts = np.bincount(values.astype(np.int64), minlength=nb_code).astype(np.float64)
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        rows.append({
            "codec_part": label,
            "quantizer": q,
            "tokens": int(values.size),
            "active_codes": int((counts > 0).sum()),
            "dead_code_fraction": float(1.0 - (counts > 0).sum() / nb_code),
            "entropy": entropy,
            "perplexity": float(np.exp(entropy)),
        })
    return rows


def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    total_bytes += sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return total_bytes / (1024 ** 2)


# %% [markdown]
# ## Load Models and Eval Names

# %%
seed_everything(RANDOM_SEED)

require_path(OLD_RVQVAE_CKPT, "old RVQVAE checkpoint")
require_path(OLD_RVQVAE_MEAN_PATH, "old RVQVAE mean")
require_path(OLD_RVQVAE_STD_PATH, "old RVQVAE std")
require_path(EVAL_SPLIT_FILE, "eval split")
require_path(MOTION2TEXT_JSON, "motion2text json")

old_model, old_config = load_old_rvqvae_model(OLD_RVQVAE_CKPT, DEVICE)
old_mean = torch.from_numpy(np.load(OLD_RVQVAE_MEAN_PATH)).float()
old_std = torch.from_numpy(np.load(OLD_RVQVAE_STD_PATH)).float()

part_models = {part: load_multipart_checkpoint(Path(resolved_part_ckpts[part]), DEVICE) for part in PART_ORDER}

text_map = load_json(MOTION2TEXT_JSON)
all_eval_names = load_name_list(EVAL_SPLIT_FILE)
eval_names = [
    name for name in all_eval_names
    if motion_path_for_name(MOTION_DIR, name).exists() and name in text_map
]
if MAX_EVAL_CLIPS is not None:
    eval_names = eval_names[:MAX_EVAL_CLIPS]

summary_rows = [
    {
        "codec": "old_rvqvae_body",
        "parts": "body",
        "params_m": sum(p.numel() for p in old_model.parameters()) / 1e6,
        "model_size_mb": model_size_mb(old_model),
        "quantizers_per_token_frame": int(old_config.model.num_quantizers),
        "motion_frames_per_token_frame": int(old_config.unit_length),
        "code_ids_per_motion_frame": int(old_config.model.num_quantizers) / float(old_config.unit_length),
    }
]
for part, loaded in part_models.items():
    summary_rows.append({
        "codec": f"new_{part}",
        "parts": part,
        "params_m": sum(p.numel() for p in loaded.model.parameters()) / 1e6,
        "model_size_mb": model_size_mb(loaded.model),
        "quantizers_per_token_frame": loaded.num_quantizers,
        "motion_frames_per_token_frame": loaded.unit_length,
        "code_ids_per_motion_frame": loaded.num_quantizers / float(loaded.unit_length),
    })
summary_rows.append({
    "codec": "new_multipart_full",
    "parts": ",".join(PART_ORDER),
    "params_m": sum(row["params_m"] for row in summary_rows[1:]),
    "model_size_mb": sum(row["model_size_mb"] for row in summary_rows[1:]),
    "quantizers_per_token_frame": sum(loaded.num_quantizers for loaded in part_models.values()),
    "motion_frames_per_token_frame": min(loaded.unit_length for loaded in part_models.values()),
    "code_ids_per_motion_frame": sum(loaded.num_quantizers / float(loaded.unit_length) for loaded in part_models.values()),
})

model_summary_df = pd.DataFrame(summary_rows)
display(model_summary_df)
print(f"Valid eval clips: {len(eval_names)} / split rows {len(all_eval_names)}")


# %% [markdown]
# ## Export GT, Old, and New Reconstructions
#
# Output layout:
#
# - `gt/*_gt.npy`
# - `old_rvqvae_body/*_pred.npy`
# - `new_multipart_full/*_pred.npy`
#
# Each `.npy` is a dict with `name`, `body`, `left`, `right`.

# %%
def export_reconstructions() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    clean_eval_dir(GT_DIR, "gt")
    clean_eval_dir(OLD_DIR, "pred")
    clean_eval_dir(NEW_DIR, "pred")

    manifest_rows = []
    recon_rows = []
    clip_rows = []
    per_frame_errors: Dict[str, Dict[str, np.ndarray]] = {}
    code_indices: Dict[str, List[np.ndarray]] = {"old_body": []}
    for part in PART_ORDER:
        code_indices[f"new_{part}"] = []

    for idx, name in enumerate(tqdm(eval_names, desc="export reconstructions")):
        raw_motion = load_motion_dict(motion_path_for_name(MOTION_DIR, name))
        gt_motion, root_meta = canonical_motion_dict(raw_motion)
        gt_parts = body_parts_from_motion(gt_motion)

        old_body, old_idx = reconstruct_old_body(
            old_model,
            raw_motion,
            old_mean,
            old_std,
            DEVICE,
            mode=OLD_ROOT_MODE,
        )
        old_body = match_length_np(old_body, len(gt_motion["body"]))
        old_motion = trim_motion(
            {"body": old_body, "left": gt_motion["left"], "right": gt_motion["right"]},
            len(gt_motion["body"]),
        )

        new_parts, new_idx = reconstruct_new_parts(gt_parts, part_models, DEVICE)
        new_motion = merge_parts_to_legacy_motion(new_parts)

        target_len = min(
            len(gt_motion["body"]),
            len(old_motion["body"]),
            len(new_motion["body"]),
            len(new_motion["left"]),
            len(new_motion["right"]),
        )
        if target_len < 2:
            continue

        gt_motion = trim_motion(gt_motion, target_len)
        old_motion = trim_motion(old_motion, target_len)
        new_motion = trim_motion(new_motion, target_len)
        gt_parts = {part: match_length_np(value, target_len) for part, value in gt_parts.items()}
        old_parts = body_parts_from_motion(old_motion)
        new_parts = body_parts_from_motion(new_motion)

        stem = f"{idx:06d}"
        save_motion_npy(GT_DIR / f"{stem}_gt.npy", name, gt_motion)
        save_motion_npy(OLD_DIR / f"{stem}_pred.npy", name, old_motion)
        save_motion_npy(NEW_DIR / f"{stem}_pred.npy", name, new_motion)

        code_indices["old_body"].append(old_idx)
        for part in PART_ORDER:
            code_indices[f"new_{part}"].append(new_idx[part])

        for part in ("upper", "lower", "feet"):
            recon_rows.append(part_reconstruction_row("old_rvqvae_body", name, part, gt_parts[part], old_parts[part]))
        for part in PART_ORDER:
            recon_rows.append(part_reconstruction_row("new_multipart_full", name, part, gt_parts[part], new_parts[part]))

        old_body_metrics = {"codec": "old_rvqvae_body", "name": name, "part": "body_153"}
        old_body_metrics.update(feature_error(gt_motion["body"], old_motion["body"]))
        old_body_metrics.update(root_metrics(gt_motion["body"], old_motion["body"]))
        old_body_metrics.update(geodesic_metrics(gt_motion["body"][:, 3:], old_motion["body"][:, 3:]))
        recon_rows.append(old_body_metrics)

        new_body_metrics = {"codec": "new_multipart_full", "name": name, "part": "body_153"}
        new_body_metrics.update(feature_error(gt_motion["body"], new_motion["body"]))
        new_body_metrics.update(root_metrics(gt_motion["body"], new_motion["body"]))
        new_body_metrics.update(geodesic_metrics(gt_motion["body"][:, 3:], new_motion["body"][:, 3:]))
        recon_rows.append(new_body_metrics)

        old_frame_l2 = np.sqrt(np.mean((old_motion["body"] - gt_motion["body"]) ** 2, axis=1))
        new_frame_l2 = np.sqrt(np.mean((new_motion["body"] - gt_motion["body"]) ** 2, axis=1))
        per_frame_errors[name] = {
            "old_body_rmse": old_frame_l2,
            "new_body_rmse": new_frame_l2,
            "gt_body": gt_motion["body"],
            "old_body": old_motion["body"],
            "new_body": new_motion["body"],
        }
        clip_rows.append({
            "name": name,
            "frames": target_len,
            "root_schema": root_meta["root_schema"],
            "root_mean_norm": root_meta["root_mean_norm"],
            "old_body_rmse": float(np.sqrt(np.mean((old_motion["body"] - gt_motion["body"]) ** 2))),
            "new_body_rmse": float(np.sqrt(np.mean((new_motion["body"] - gt_motion["body"]) ** 2))),
        })

        manifest_rows.append({
            "file_stem": stem,
            "name": name,
            "frames": target_len,
            "root_schema": root_meta["root_schema"],
            "root_mean_norm": root_meta["root_mean_norm"],
            "gt_path": str(GT_DIR / f"{stem}_gt.npy"),
            "old_path": str(OLD_DIR / f"{stem}_pred.npy"),
            "new_path": str(NEW_DIR / f"{stem}_pred.npy"),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    recon_df = pd.DataFrame(recon_rows)
    clip_df = pd.DataFrame(clip_rows)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(OUT_ROOT / "manifest.csv", index=False)
    recon_df.to_csv(OUT_ROOT / "reconstruction_metrics.csv", index=False)
    clip_df.to_csv(OUT_ROOT / "clip_body_rmse.csv", index=False)

    usage_rows = []
    if RUN_CODEBOOK_USAGE:
        usage_rows.extend(codebook_usage_rows("old_body", code_indices["old_body"], int(old_config.model.nb_code)))
        for part in PART_ORDER:
            usage_rows.extend(codebook_usage_rows(f"new_{part}", code_indices[f"new_{part}"], part_models[part].nb_code))
    usage_df = pd.DataFrame(usage_rows)
    if not usage_df.empty:
        usage_df.to_csv(OUT_ROOT / "codebook_usage.csv", index=False)

    return manifest_df, recon_df, usage_df, per_frame_errors


if RUN_EXPORT_RECONSTRUCTIONS:
    manifest_df, recon_df, codebook_usage_df, per_frame_errors = export_reconstructions()
    display(manifest_df.head())
    display(recon_df.groupby(["codec", "part"]).mean(numeric_only=True).round(6))
    if not codebook_usage_df.empty:
        display(codebook_usage_df.groupby("codec_part").mean(numeric_only=True).round(4))
    print("exported clips:", len(manifest_df))
    print("output:", OUT_ROOT)
else:
    print("Set RUN_EXPORT_RECONSTRUCTIONS = True to export comparison npys.")


# %% [markdown]
# ## Official Evaluator Helpers

# %%
def load_official_evaluator_helpers():
    helper_names = [
        "ChronTMR",
        "PredMotionTextDataset",
        "compute_128_sample_metrics",
        "compute_fid_diversity_metrics",
    ]
    saved_path = list(sys.path)
    conflict_prefixes = ("models", "datasets", "evaluate_pred_motion_v2")
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in conflict_prefixes or name.startswith("models.") or name.startswith("datasets.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(EVALUATION_DIR))
    try:
        module = importlib.import_module("evaluate_pred_motion_v2")
        return {name: getattr(module, name) for name in helper_names}
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name in conflict_prefixes or name.startswith("models.") or name.startswith("datasets."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


_eval_helpers = load_official_evaluator_helpers()
ChronTMR = _eval_helpers["ChronTMR"]
PredMotionTextDataset = _eval_helpers["PredMotionTextDataset"]
compute_128_sample_metrics = _eval_helpers["compute_128_sample_metrics"]
compute_fid_diversity_metrics = _eval_helpers["compute_fid_diversity_metrics"]


def load_yaml_namespace(path: Path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def convert(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: convert(val) for key, val in value.items()})
        return value

    return convert(data)


def prepare_evaluator_stats_dir() -> Path:
    mean_pt = EVALUATOR_STATS_DIR / "mean.pt"
    std_pt = EVALUATOR_STATS_DIR / "std.pt"
    if mean_pt.exists() and std_pt.exists():
        return EVALUATOR_STATS_DIR

    fallback_dir = OUT_ROOT / "evaluator_stats_from_old_rvqvae"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    torch.save(old_mean.float().cpu(), fallback_dir / "mean.pt")
    torch.save(old_std.float().cpu(), fallback_dir / "std.pt")
    print("Evaluator stats missing; using fallback stats from old RVQVAE normalization:", fallback_dir)
    return fallback_dir


ACTIVE_EVALUATOR_STATS_DIR = prepare_evaluator_stats_dir()


def evaluator_length_to_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    lengths = lengths.to(device=device, dtype=torch.long)
    positions = torch.arange(max_length, device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def load_evaluator_motion_encoder(device: torch.device):
    from evaluation.models.actor import ACTORStyleEncoder

    require_path(EVALUATOR_CKPT, "evaluator checkpoint")
    require_path(EVALUATOR_CFG_PATH, "evaluator config")
    cfg = load_yaml_namespace(EVALUATOR_CFG_PATH)
    encoder = ACTORStyleEncoder(
        cfg.motion_encoder.nfeats,
        cfg.motion_encoder.vae,
        cfg.motion_encoder.latent_dim,
        cfg.motion_encoder.ff_size,
        cfg.motion_encoder.num_layers,
        cfg.motion_encoder.num_heads,
        cfg.motion_encoder.dropout,
        cfg.motion_encoder.activation,
    )
    ckpt = torch_load_trusted(EVALUATOR_CKPT, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    motion_state = {
        key.replace("motion_encoder.", "", 1): value
        for key, value in ckpt.items()
        if key.startswith("motion_encoder.")
    }
    missing, unexpected = encoder.load_state_dict(motion_state, strict=False)
    if missing or unexpected:
        print("motion encoder missing keys:", missing)
        print("motion encoder unexpected keys:", unexpected)
    encoder.to(device).eval()
    return cfg, encoder


def load_evaluator_body(path: Path) -> Tuple[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        return raw.get("name", path.stem), np.asarray(raw["body"], dtype=np.float32)
    return path.stem, np.asarray(raw, dtype=np.float32)


def normalize_pad_body_for_evaluator(
    body: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_len: int,
) -> Tuple[torch.Tensor, int]:
    body = np.asarray(body, dtype=np.float32)
    length = min(len(body), max_len)
    body_tensor = torch.tensor(body[:length], dtype=torch.float32)
    body_dim = body_tensor.shape[1]
    body_tensor = (body_tensor - mean[:body_dim]) / (std[:body_dim] + 1e-12)
    if length < max_len:
        pad = torch.zeros((max_len - length, body_dim), dtype=torch.float32)
        body_tensor = torch.cat([body_tensor, pad], dim=0)
    return body_tensor, length


@torch.no_grad()
def encode_evaluator_latents(
    motion_dir: Path,
    suffix: str,
    cfg,
    encoder,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:
    files = sorted(motion_dir.glob(f"*_{suffix}.npy"))
    if not files:
        raise FileNotFoundError(f"No *_{suffix}.npy files found in {motion_dir}")
    mean = torch_load_trusted(ACTIVE_EVALUATOR_STATS_DIR / "mean.pt", map_location="cpu").float()
    std = torch_load_trusted(ACTIVE_EVALUATOR_STATS_DIR / "std.pt", map_location="cpu").float()
    max_len = int(cfg.dataset.max_motion_length)

    names: List[str] = []
    latents: List[torch.Tensor] = []
    batch_motions: List[torch.Tensor] = []
    batch_lengths: List[int] = []

    for path in tqdm(files, desc=f"encode {motion_dir.name}"):
        name, body = load_evaluator_body(path)
        motion, length = normalize_pad_body_for_evaluator(body, mean, std, max_len)
        names.append(name)
        batch_motions.append(motion)
        batch_lengths.append(length)
        if len(batch_motions) >= batch_size:
            x = torch.stack(batch_motions, dim=0).to(device)
            lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)
            mask = evaluator_length_to_mask(lengths, max_len, device)
            encoded = encoder({"x": x, "mask": mask})[:, 0]
            latents.append(encoded.detach().cpu())
            batch_motions.clear()
            batch_lengths.clear()

    if batch_motions:
        x = torch.stack(batch_motions, dim=0).to(device)
        lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)
        mask = evaluator_length_to_mask(lengths, max_len, device)
        encoded = encoder({"x": x, "mask": mask})[:, 0]
        latents.append(encoded.detach().cpu())

    return names, torch.cat(latents, dim=0).numpy()


# %% [markdown]
# ## Evaluator-Latent FID and Diversity

# %%
if RUN_EVALUATOR_METRICS:
    cfg_eval, motion_encoder = load_evaluator_motion_encoder(DEVICE)
    gt_names, gt_latents = encode_evaluator_latents(
        GT_DIR,
        "gt",
        cfg_eval,
        motion_encoder,
        DEVICE,
        batch_size=EVALUATOR_BATCH_SIZE,
    )

    fid_rows = []
    for codec, motion_dir in {
        "old_rvqvae_body": OLD_DIR,
        "new_multipart_full": NEW_DIR,
    }.items():
        pred_names, pred_latents = encode_evaluator_latents(
            motion_dir,
            "pred",
            cfg_eval,
            motion_encoder,
            DEVICE,
            batch_size=EVALUATOR_BATCH_SIZE,
        )
        metrics = compute_fid_diversity_metrics(
            gt_latent_motions=gt_latents,
            pred_latent_motions=pred_latents,
            diversity_times=DIVERSITY_TIMES,
        )
        row = {key: float(value) for key, value in metrics.items()}
        row["codec"] = codec
        row["num_clips"] = int(len(pred_latents))
        row["diversity_gap"] = abs(row["Diversity_Gen"] - row["Diversity_GT"])
        fid_rows.append(row)

    fid_df = pd.DataFrame(fid_rows).set_index("codec")
    fid_df.to_csv(OUT_ROOT / "fid_diversity_metrics.csv")
    display(fid_df.round(6))
else:
    print("Set RUN_EVALUATOR_METRICS = True after exporting reconstructions.")


# %% [markdown]
# ## R@K Semantic Retrieval

# %%
def resolve_hf_path(path_like) -> str:
    path = Path(str(path_like)).expanduser()
    return str(path) if path.exists() else str(path_like)


def load_chrontmr_retrieval_model(device: torch.device):
    from transformers import AutoTokenizer
    import transformers

    cfg = load_yaml_namespace(EVALUATOR_CFG_PATH)
    cfg.model.text_model_name = resolve_hf_path(EVALUATOR_TEXT_MODEL_PATH)

    original_from_pretrained = transformers.AutoModel.from_pretrained

    def redirected_from_pretrained(model_name, *args, **kwargs):
        name = str(model_name)
        if name == "bert-base-chinese" or "bert-base-chinese" in name or name.startswith("/data/home/jinch/"):
            return original_from_pretrained(cfg.model.text_model_name, *args, **kwargs)
        return original_from_pretrained(model_name, *args, **kwargs)

    transformers.AutoModel.from_pretrained = redirected_from_pretrained
    try:
        model = ChronTMR(cfg, vae=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not load ChronAccRet text model. Set EVALUATOR_TEXT_MODEL_PATH "
            "to a local bert-base-chinese folder, or allow Hugging Face loading."
        ) from exc
    finally:
        transformers.AutoModel.from_pretrained = original_from_pretrained

    ckpt = torch_load_trusted(EVALUATOR_CKPT, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print("ChronTMR missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        print("ChronTMR unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    return cfg, model, tokenizer


def make_retrieval_dataset(cfg, motion_dir: Path, suffix: str):
    return PredMotionTextDataset(
        cfg=cfg,
        motion_dir=str(motion_dir),
        motion2text_path=str(MOTION2TEXT_JSON),
        stats_dir=str(ACTIVE_EVALUATOR_STATS_DIR),
        motion_type=suffix,
    )


if RUN_RETRIEVAL_RK:
    require_path(EVALUATOR_CKPT, "evaluator checkpoint")
    require_path(EVALUATOR_CFG_PATH, "evaluator config")
    require_path(MOTION2TEXT_JSON, "motion2text json")

    cfg_rk, chrontmr_model, chrontmr_tokenizer = load_chrontmr_retrieval_model(DEVICE)
    retrieval_rows = []
    for codec, motion_dir, suffix in [
        ("raw_gt", GT_DIR, "gt"),
        ("old_rvqvae_body", OLD_DIR, "pred"),
        ("new_multipart_full", NEW_DIR, "pred"),
    ]:
        dataset_rk = make_retrieval_dataset(cfg_rk, motion_dir, suffix)
        if len(dataset_rk) == 0:
            print(f"skip {codec}: no valid samples")
            continue
        seed_everything(int(cfg_rk.train.seed))
        metrics = compute_128_sample_metrics(cfg_rk, chrontmr_model, dataset_rk, DEVICE, chrontmr_tokenizer)
        row = {"codec": codec, "num_clips": int(len(dataset_rk))}
        for key in ["t2m/R01", "t2m/R02", "t2m/R03", "t2m/R05", "t2m/R10", "t2m/MedR"]:
            row[key] = metrics.get(key, np.nan)
        retrieval_rows.append(row)

    retrieval_df = pd.DataFrame(retrieval_rows).set_index("codec")
    retrieval_df.to_csv(OUT_ROOT / "retrieval_rk_metrics.csv")
    display(retrieval_df.round(4))
else:
    print("Set RUN_RETRIEVAL_RK = True after exporting reconstructions.")


# %% [markdown]
# ## Visualizations

# %%
def maybe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


if "recon_df" not in globals():
    loaded_recon = maybe_load_csv(OUT_ROOT / "reconstruction_metrics.csv")
    if loaded_recon is not None:
        recon_df = loaded_recon

if "codebook_usage_df" not in globals():
    loaded_usage = maybe_load_csv(OUT_ROOT / "codebook_usage.csv")
    if loaded_usage is not None:
        codebook_usage_df = loaded_usage

if "fid_df" not in globals():
    loaded_fid = maybe_load_csv(OUT_ROOT / "fid_diversity_metrics.csv")
    if loaded_fid is not None:
        fid_df = loaded_fid.set_index("codec")

if "retrieval_df" not in globals():
    loaded_rk = maybe_load_csv(OUT_ROOT / "retrieval_rk_metrics.csv")
    if loaded_rk is not None:
        retrieval_df = loaded_rk.set_index("codec")


if "recon_df" in globals() and not recon_df.empty:
    agg = recon_df.groupby(["codec", "part"]).mean(numeric_only=True).reset_index()
    body_parts = agg[agg["part"].isin(["body_153", "upper", "lower", "feet", "hands"])]
    display(body_parts.round(5))

    plt.figure(figsize=(11, 4))
    pivot = body_parts.pivot(index="part", columns="codec", values="feature_rmse")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title("Feature RMSE by Part")
    plt.ylabel("RMSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 4))
    pivot = body_parts.pivot(index="part", columns="codec", values="rotation_geodesic_deg_mean")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title("Mean Rotation Geodesic Error by Part")
    plt.ylabel("degrees")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


if "codebook_usage_df" in globals() and not codebook_usage_df.empty:
    usage_pivot = codebook_usage_df.pivot(index="codec_part", columns="quantizer", values="perplexity")
    plt.figure(figsize=(9, 4))
    plt.imshow(usage_pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="perplexity")
    plt.yticks(range(len(usage_pivot.index)), usage_pivot.index)
    plt.xticks(range(len(usage_pivot.columns)), usage_pivot.columns)
    plt.title("Codebook Perplexity by Quantizer")
    plt.xlabel("quantizer")
    plt.tight_layout()
    plt.show()

    active_pivot = codebook_usage_df.pivot(index="codec_part", columns="quantizer", values="active_codes")
    display(active_pivot)


if "fid_df" in globals() and not fid_df.empty:
    display(fid_df.round(6))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fid_df[["FID_norm_by_GT", "FID_raw"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Evaluator-Latent FID")
    axes[0].set_ylabel("lower is better")
    fid_df[["Diversity_GT", "Diversity_Gen", "diversity_gap"]].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Latent Diversity")
    axes[1].set_ylabel("closer gap is better")
    plt.tight_layout()
    plt.show()


if "retrieval_df" in globals() and not retrieval_df.empty:
    display(retrieval_df.round(4))
    rk_cols = ["t2m/R01", "t2m/R02", "t2m/R03", "t2m/R05", "t2m/R10"]
    retrieval_df[rk_cols].T.plot(marker="o", figsize=(9, 4))
    plt.title("R@K Semantic Retrieval")
    plt.ylabel("recall (%)")
    plt.xlabel("K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Worst-Case Body Error Inspection

# %%
if "per_frame_errors" in globals() and per_frame_errors:
    clip_df = pd.read_csv(OUT_ROOT / "clip_body_rmse.csv")
    clip_df["new_minus_old_body_rmse"] = clip_df["new_body_rmse"] - clip_df["old_body_rmse"]
    display(clip_df.sort_values("new_body_rmse", ascending=False).head(10))
    display(clip_df.sort_values("new_minus_old_body_rmse", ascending=False).head(10))

    worst_name = clip_df.sort_values("new_body_rmse", ascending=False).iloc[0]["name"]
    errors = per_frame_errors[worst_name]
    plt.figure(figsize=(12, 4))
    plt.plot(errors["old_body_rmse"], label="old body")
    plt.plot(errors["new_body_rmse"], label="new multipart body")
    plt.title(f"Per-frame Body RMSE: {worst_name}")
    plt.xlabel("frame")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    gt_traj = integrate_root_delta(errors["gt_body"])
    old_traj = integrate_root_delta(errors["old_body"])
    new_traj = integrate_root_delta(errors["new_body"])
    plt.figure(figsize=(6, 6))
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], label="GT")
    plt.plot(old_traj[:, 0], old_traj[:, 2], label="old")
    plt.plot(new_traj[:, 0], new_traj[:, 2], label="new")
    plt.title(f"Integrated Root Trajectory X/Z: {worst_name}")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    geo = geodesic_degrees(errors["gt_body"][:, 3:], errors["new_body"][:, 3:]).reshape(len(errors["gt_body"]), 25)
    plt.figure(figsize=(12, 5))
    plt.imshow(geo.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="degrees")
    plt.title(f"New Multipart Body Rotation Error Heatmap: {worst_name}")
    plt.xlabel("frame")
    plt.ylabel("body joint index")
    plt.tight_layout()
    plt.show()
else:
    print("Run export in this notebook session to enable per-frame/worst-case plots.")


# %% [markdown]
# ## Reading the Results
#
# Recommended report table:
#
# - `fid_diversity_metrics.csv`: compare `FID_norm_by_GT`, `FID_raw`,
#   `Diversity_Gen`, and `diversity_gap`.
# - `retrieval_rk_metrics.csv`: compare `t2m/R01`, `t2m/R05`, `t2m/R10`,
#   and `t2m/MedR`.
# - `reconstruction_metrics.csv`: use body and part reconstruction diagnostics
#   to explain why FID/R@K moved.
# - `codebook_usage.csv`: verify codebook health. Perplexity too close to zero
#   means collapse; high reconstruction error with very high perplexity can mean
#   the codec is using many codes but not learning a clean manifold.
#
# Because the official evaluator currently uses only 153D body features, a new
# hands codec can be excellent without improving FID/R@K. Judge hands with the
# `hands` rows in `reconstruction_metrics.csv`.

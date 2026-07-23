"""Shared deterministic inference-math configuration.

This utility deliberately has no model/codec imports so token-only evaluators
can configure CUDA math without loading the multipart RVQ stack.
"""

from __future__ import annotations

import os
from typing import Dict

import torch


def configure_strict_inference_math(device: torch.device) -> Dict[str, object]:
    """Disable TF32 so full-prefix and streaming token decisions stay stable."""

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
    return {
        "device_type": device.type,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

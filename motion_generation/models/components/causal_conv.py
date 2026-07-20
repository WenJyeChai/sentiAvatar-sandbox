from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_single_int(value: int | Tuple[int]) -> int:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Expected a one-dimensional value, got {value}")
        return int(value[0])
    return int(value)


class CausalConv1d(nn.Conv1d):
    """Conv1d with explicit left padding and no access to future samples.

    For a strided convolution, ``stride_end_aligned=True`` delays the first
    output until a complete stride-sized input chunk has arrived. With kernel
    size 4 and stride 2, output ``j`` ends at input frame ``2*j + 1`` and an
    input of length ``T`` produces ``floor(T / 2)`` outputs. This makes a 10 Hz
    codec token represent a completed 100 ms chunk of the 20 Hz motion stream.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int],
        stride: int | Tuple[int] = 1,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        *,
        stride_end_aligned: bool = False,
    ) -> None:
        if padding not in (0, (0,)):
            raise ValueError("CausalConv1d manages padding internally; pass padding=0")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        kernel_size = _as_single_int(self.kernel_size)
        stride = _as_single_int(self.stride)
        dilation = _as_single_int(self.dilation)
        receptive_history = dilation * (kernel_size - 1)
        alignment_offset = stride - 1 if stride_end_aligned else 0
        self.left_padding = receptive_history - alignment_offset
        if self.left_padding < 0:
            raise ValueError(
                "Kernel/dilation does not provide enough history for stride-end "
                f"alignment: kernel={kernel_size}, stride={stride}, dilation={dilation}"
            )
        self.stride_end_aligned = bool(stride_end_aligned)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding:
            x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)

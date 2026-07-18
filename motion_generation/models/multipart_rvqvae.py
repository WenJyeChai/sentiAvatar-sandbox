from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.decoder import Decoder
from models.components.encoder import Encoder
from models.components.residual_vq import ResidualVQ
from utils.multipart_motion import PART_DIMS, PART_ORDER


class MultiPartRVQVAE(nn.Module):
    """Independent RVQ codecs for selected SentiAvatar body or face streams."""

    def __init__(
        self,
        part_dims: Optional[Mapping[str, int]] = None,
        part_order: Optional[Sequence[str]] = None,
        nb_code: int = 512,
        code_dim: int = 512,
        num_quantizers: int = 4,
        down_t: int = 1,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm: Optional[str] = None,
        vq_cnn_depth: int = 3,
        shared_codebook: bool = False,
        quantize_dropout_prob: float = 0.0,
        quantize_dropout_cutoff_index: int = 1,
        mu: float = 0.99,
    ) -> None:
        super().__init__()
        source_dims = dict(part_dims or PART_DIMS)
        self.part_order = tuple(part_order or PART_ORDER)
        unknown = [part for part in self.part_order if part not in source_dims]
        if unknown:
            raise ValueError(f"Unknown motion part(s): {unknown}")
        self.part_dims = {part: int(source_dims[part]) for part in self.part_order}
        self.nb_code = int(nb_code)
        self.code_dim = int(code_dim)
        self.num_quantizers = int(num_quantizers)
        self.unit_length = int(down_t * stride_t)

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.quantizers = nn.ModuleDict()

        for part in self.part_order:
            input_dim = int(self.part_dims[part])
            self.encoders[part] = Encoder(
                input_dim=input_dim,
                output_dim=code_dim,
                down_t=down_t,
                stride_t=stride_t,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
                vq_cnn_depth=vq_cnn_depth,
            )
            self.decoders[part] = Decoder(
                input_dim=input_dim,
                output_dim=code_dim,
                down_t=down_t,
                stride_t=stride_t,
                width=width * 2,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
                vq_cnn_depth=vq_cnn_depth,
            )
            self.quantizers[part] = ResidualVQ(
                num_quantizers=num_quantizers,
                shared_codebook=shared_codebook,
                quantize_dropout_prob=quantize_dropout_prob,
                quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
                nb_code=nb_code,
                code_dim=code_dim,
                mu=mu,
            )

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float()

    @staticmethod
    def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        if x.shape[1] == target_length:
            return x
        if x.shape[1] > target_length:
            return x[:, :target_length]
        pad = target_length - x.shape[1]
        return F.pad(x.permute(0, 2, 1), (0, pad), mode="replicate").permute(0, 2, 1)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        return_idx: bool = False,
    ) -> Dict[str, object]:
        rec: Dict[str, torch.Tensor] = {}
        code_idx: Dict[str, torch.Tensor] = {}
        commit_loss: Dict[str, torch.Tensor] = {}
        perplexity: Dict[str, torch.Tensor] = {}

        for part in self.part_order:
            x = inputs[part]
            latent = self.encoders[part](self.preprocess(x))
            quantized, indices, loss, ppl = self.quantizers[part](
                latent,
                sample_codebook_temp=0.5,
            )
            decoded = self.decoders[part](quantized)
            rec[part] = self._match_length(decoded, x.shape[1])
            code_idx[part] = indices
            commit_loss[part] = loss
            perplexity[part] = ppl

        output: Dict[str, object] = {
            "rec": rec,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
        }
        if return_idx:
            output["code_idx"] = code_idx
        return output

    @torch.no_grad()
    def encode(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        code_idx: Dict[str, torch.Tensor] = {}
        for part in self.part_order:
            latent = self.encoders[part](self.preprocess(inputs[part]))
            code_idx[part] = self.quantizers[part].quantize(latent)
        return code_idx

    def decode(self, code_idx: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rec: Dict[str, torch.Tensor] = {}
        for part in self.part_order:
            latent = self.quantizers[part].get_codebook_entry(code_idx[part])
            rec[part] = self.decoders[part](latent)
        return rec

    def forward_decoder(self, code_idx: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.decode(code_idx)

    def config_dict(self) -> Dict[str, object]:
        return {
            "part_order": list(self.part_order),
            "part_dims": dict(self.part_dims),
            "nb_code": self.nb_code,
            "code_dim": self.code_dim,
            "num_quantizers": self.num_quantizers,
            "unit_length": self.unit_length,
        }

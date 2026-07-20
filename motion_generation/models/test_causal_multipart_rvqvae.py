from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.components.causal_conv import CausalConv1d
from models.components.quantizer import Quantizer
from models.multipart_rvqvae import MultiPartRVQVAE


def tiny_causal_codec() -> MultiPartRVQVAE:
    model = MultiPartRVQVAE(
        part_dims={"upper": 6},
        part_order=["upper"],
        nb_code=16,
        code_dim=8,
        num_quantizers=2,
        down_t=1,
        stride_t=2,
        width=16,
        depth=2,
        dilation_growth_rate=2,
        activation="relu",
        norm=None,
        vq_cnn_depth=2,
        quantize_dropout_prob=0.0,
        causal=True,
    )
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in model.quantizers["upper"].layers:
            layer.codebook.copy_(torch.randn(layer.codebook.shape, generator=generator))
            layer.code_sum.copy_(layer.codebook)
            layer.code_count.fill_(1.0)
            layer.init = True
    return model.eval()


def test_stride_end_aligned_conv_waits_for_complete_chunks():
    torch.manual_seed(3)
    conv = CausalConv1d(2, 3, kernel_size=4, stride=2, stride_end_aligned=True)
    x = torch.randn(1, 2, 11)
    full = conv(x)
    assert full.shape[-1] == 5

    for source_frames in (2, 4, 6, 8, 10):
        prefix = conv(x[..., :source_frames])
        torch.testing.assert_close(prefix, full[..., : source_frames // 2])


def test_causal_encoder_prefix_and_future_perturbation_equivalence():
    torch.manual_seed(5)
    model = tiny_causal_codec()
    x = torch.randn(1, 21, 6)
    full_latent = model.encoders["upper"](model.preprocess(x))
    full_codes = model.encode({"upper": x})["upper"]
    assert full_latent.shape[-1] == 10
    assert full_codes.shape == (1, 10, 2)

    for source_frames in (2, 6, 12, 20):
        expected_tokens = source_frames // 2
        prefix_x = x[:, :source_frames]
        prefix_latent = model.encoders["upper"](model.preprocess(prefix_x))
        prefix_codes = model.encode({"upper": prefix_x})["upper"]
        torch.testing.assert_close(prefix_latent, full_latent[..., :expected_tokens])
        assert torch.equal(prefix_codes, full_codes[:, :expected_tokens])

    changed_future = x.clone()
    changed_future[:, 12:] = torch.randn_like(changed_future[:, 12:]) * 100.0
    perturbed_latent = model.encoders["upper"](model.preprocess(changed_future))
    perturbed_codes = model.encode({"upper": changed_future})["upper"]
    torch.testing.assert_close(perturbed_latent[..., :6], full_latent[..., :6])
    assert torch.equal(perturbed_codes[:, :6], full_codes[:, :6])


def test_causal_decoder_prefix_and_streaming_equivalence():
    model = tiny_causal_codec()
    code_idx = torch.randint(0, model.nb_code, (1, 10, model.num_quantizers))
    full = model.decode({"upper": code_idx})["upper"]
    assert full.shape == (1, 20, 6)

    streamed_chunks = []
    for token_frames in range(1, code_idx.shape[1] + 1):
        prefix = model.decode({"upper": code_idx[:, :token_frames]})["upper"]
        source_frames = token_frames * model.unit_length
        torch.testing.assert_close(prefix, full[:, :source_frames])
        streamed_chunks.append(prefix[:, -model.unit_length :])
    torch.testing.assert_close(torch.cat(streamed_chunks, dim=1), full)

    changed_future = code_idx.clone()
    changed_future[:, 6:] = (changed_future[:, 6:] + 1) % model.nb_code
    perturbed = model.decode({"upper": changed_future})["upper"]
    torch.testing.assert_close(perturbed[:, :12], full[:, :12])


def test_causal_forward_drops_only_an_incomplete_source_tail():
    model = tiny_causal_codec()
    x = torch.randn(2, 21, 6)
    output = model({"upper": x}, return_idx=True)
    assert output["rec"]["upper"].shape == (2, 20, 6)
    assert output["code_idx"]["upper"].shape == (2, 10, 2)


@pytest.mark.parametrize("norm", ["BN", "GN"])
def test_time_aggregating_norms_are_rejected(norm: str):
    with pytest.raises(ValueError, match="strictly causal"):
        MultiPartRVQVAE(
            part_dims={"upper": 6},
            part_order=["upper"],
            nb_code=8,
            code_dim=8,
            num_quantizers=1,
            down_t=1,
            stride_t=2,
            width=32,
            depth=1,
            norm=norm,
            vq_cnn_depth=1,
            causal=True,
        )


def test_ema_state_survives_resume_and_old_codebooks_still_load():
    quantizer = Quantizer(nb_code=8, code_dim=4)
    with torch.no_grad():
        quantizer.codebook.normal_()
        quantizer.code_sum.copy_(quantizer.codebook * 2.0)
        quantizer.code_count.copy_(torch.arange(1, 9, dtype=torch.float32))
        quantizer.init = True

    resumed = Quantizer(nb_code=8, code_dim=4)
    resumed.load_state_dict(quantizer.state_dict())
    assert resumed.init
    torch.testing.assert_close(resumed.code_sum, quantizer.code_sum)
    torch.testing.assert_close(resumed.code_count, quantizer.code_count)

    old_state = {"codebook": quantizer.codebook.clone()}
    legacy_loaded = Quantizer(nb_code=8, code_dim=4)
    legacy_loaded.load_state_dict(old_state, strict=True)
    assert legacy_loaded.init
    torch.testing.assert_close(legacy_loaded.codebook, quantizer.codebook)
    torch.testing.assert_close(legacy_loaded.code_sum, quantizer.codebook)
    torch.testing.assert_close(legacy_loaded.code_count, torch.ones(8))

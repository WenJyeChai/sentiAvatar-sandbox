"""Hard-forward frozen-Step-2 guidance for Step 1 anchor fine-tuning.

Experiment A keeps the temporal schedule fixed.  Step 1 supplies one predicted
right boundary per clip, while the left boundary and missing-token targets stay
canonical.  The right boundary is hard in the forward pass and differentiable
in the backward pass through a straight-through categorical embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.audio_motion_model import AudioMotionTransformer


@dataclass
class FrozenStep2GuidanceOutput:
    loss: torch.Tensor
    hard_ce: torch.Tensor
    correct: torch.Tensor
    count: torch.Tensor
    q0_correct: torch.Tensor
    q0_count: torch.Tensor
    stage_ce: torch.Tensor
    hard_local_ids: torch.Tensor


class FrozenStep2AnchorGuidance(nn.Module):
    """Evaluate predicted right anchors with a frozen C2F Step 2 model."""

    def __init__(
        self,
        model: AudioMotionTransformer,
        *,
        stage_weights: Sequence[float],
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Straight-through temperature must be positive")
        self.model = model
        self.temperature = float(temperature)
        self.codebook_size = int(model.config.codebook_size)
        self.tokens_per_frame = int(model.config.num_tokens_per_frame)
        self.num_quantizers = int(model.config.num_quantizers_per_part)
        if self.tokens_per_frame != 16:
            raise ValueError(
                "Experiment A expects 16 multipart body IDs per frame, got "
                f"{self.tokens_per_frame}"
            )
        if self.codebook_size != 512:
            raise ValueError(
                f"Experiment A expects 512-way body codebooks, got {self.codebook_size}"
            )
        if len(stage_weights) != self.num_quantizers:
            raise ValueError(
                "stage_weights must match Step 2 quantizer count "
                f"{self.num_quantizers}"
            )
        weights = torch.as_tensor(stage_weights, dtype=torch.float32)
        if bool((weights < 0).any()) or float(weights.sum()) <= 0:
            raise ValueError("stage_weights must be non-negative with positive sum")
        self.register_buffer("stage_weights", weights / weights.sum(), persistent=False)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Path,
        *,
        stage_weights: Sequence[float],
        temperature: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FrozenStep2AnchorGuidance":
        model = AudioMotionTransformer.from_pretrained(
            checkpoint,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        return cls(
            model,
            stage_weights=stage_weights,
            temperature=temperature,
        ).to(device)

    def train(self, mode: bool = True) -> "FrozenStep2AnchorGuidance":
        # Step 2 remains deterministic and frozen even while Step 1 trains.
        super().train(False)
        self.model.eval()
        return self

    def _straight_through_boundary(
        self,
        selected_anchor_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            selected_anchor_logits.shape[0],
            self.tokens_per_frame,
            self.codebook_size,
        )
        if tuple(selected_anchor_logits.shape) != expected:
            raise ValueError(
                "selected_anchor_logits must be [B,16,512], got "
                f"{tuple(selected_anchor_logits.shape)}"
            )
        probabilities = F.softmax(
            selected_anchor_logits.float() / self.temperature,
            dim=-1,
        )
        hard_local_ids = probabilities.argmax(dim=-1)
        hard = F.one_hot(hard_local_ids, num_classes=self.codebook_size).to(
            probabilities.dtype
        )
        straight_through = hard - probabilities.detach() + probabilities
        embedding_weight = self.model.embed_tokens.weight[
            : self.tokens_per_frame * self.codebook_size
        ].reshape(self.tokens_per_frame, self.codebook_size, -1)
        boundary_embeddings = torch.einsum(
            "bsv,svh->bsh",
            straight_through.to(embedding_weight.dtype),
            embedding_weight,
        )
        return hard_local_ids, boundary_embeddings

    def _prepare_step2_inputs(
        self,
        *,
        hard_local_ids: torch.Tensor,
        boundary_embeddings: torch.Tensor,
        motion_tokens: torch.Tensor,
        audio_features: torch.Tensor,
        frame_mask: torch.Tensor,
        gap_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if motion_tokens.ndim != 3 or motion_tokens.shape[-1] != self.tokens_per_frame:
            raise ValueError("motion_tokens must be [B,F,16]")
        batch, frames, _ = motion_tokens.shape
        if tuple(audio_features.shape[:2]) != (batch, frames):
            raise ValueError("audio_features must begin with the same [B,F] shape")
        if tuple(frame_mask.shape) != (batch, frames):
            raise ValueError("frame_mask must have shape [B,F]")
        if tuple(gap_lengths.shape) != (batch,):
            raise ValueError("gap_lengths must have shape [B]")
        if int(audio_features.shape[-1]) != int(self.model.config.audio_feat_dim):
            raise ValueError(
                f"Expected {self.model.config.audio_feat_dim}-D Step 2 audio features, "
                f"got {audio_features.shape[-1]}"
            )
        expected_frames = gap_lengths + 2
        if not torch.equal(frame_mask.sum(dim=-1).to(expected_frames.dtype), expected_frames):
            raise ValueError("Every guidance window must contain gap+2 valid frames")

        device = motion_tokens.device
        offsets = (
            torch.arange(self.tokens_per_frame, device=device) * self.codebook_size
        ).view(1, 1, -1)
        gt_ids = motion_tokens.clamp(0, self.codebook_size - 1) + offsets
        mask_token_id = int(
            getattr(self.model.config, "mask_token_id", self.model.config.vocab_size - 1)
        )
        gt_ids = gt_ids.masked_fill(~frame_mask.unsqueeze(-1), mask_token_id)

        frame_indices = torch.arange(frames, device=device).view(1, -1)
        right_frames = gap_lengths.view(-1, 1) + 1
        middle_frames = (
            frame_indices.gt(0)
            & frame_indices.lt(right_frames)
            & frame_mask
        )
        right_frame_mask = frame_indices.eq(right_frames) & frame_mask
        if not torch.equal(
            right_frame_mask.sum(dim=-1),
            torch.ones(batch, device=device, dtype=torch.long),
        ):
            raise ValueError("Every guidance window must contain exactly one right boundary")

        current_ids = gt_ids.clone()
        current_ids[middle_frames] = mask_token_id
        right_global = hard_local_ids + offsets.view(1, -1)
        current_ids[right_frame_mask] = right_global

        sequence_length = frames * self.tokens_per_frame
        flat_current = current_ids.reshape(batch, sequence_length)
        flat_gt = gt_ids.reshape(batch, sequence_length)
        attention_mask = (
            frame_mask.unsqueeze(-1)
            .expand(-1, -1, self.tokens_per_frame)
            .reshape(batch, sequence_length)
        )
        middle_mask = (
            middle_frames.unsqueeze(-1)
            .expand(-1, -1, self.tokens_per_frame)
            .reshape(batch, sequence_length)
        )
        override_mask = (
            right_frame_mask.unsqueeze(-1)
            .expand(-1, -1, self.tokens_per_frame)
            .reshape(batch, sequence_length)
        )

        hidden_size = boundary_embeddings.shape[-1]
        flat_overrides = boundary_embeddings.new_zeros(
            batch * sequence_length, hidden_size
        )
        override_indices = override_mask.reshape(-1).nonzero(
            as_tuple=False
        ).squeeze(-1)
        flat_overrides = flat_overrides.index_copy(
            0,
            override_indices,
            boundary_embeddings.reshape(-1, hidden_size),
        )
        return {
            "input_ids": flat_current,
            "gt_ids": flat_gt,
            "audio_features": audio_features,
            "attention_mask": attention_mask,
            "middle_mask": middle_mask,
            "gap_lengths": gap_lengths,
            "token_embedding_overrides": flat_overrides.reshape(
                batch, sequence_length, hidden_size
            ),
            "token_embedding_override_mask": override_mask,
        }

    def forward(
        self,
        selected_anchor_logits: torch.Tensor,
        guidance_batch: Mapping[str, torch.Tensor],
    ) -> FrozenStep2GuidanceOutput:
        required = {
            "motion_tokens",
            "audio_features",
            "frame_mask",
            "gap_lengths",
        }
        missing = sorted(required.difference(guidance_batch))
        if missing:
            raise KeyError(f"Guidance batch is missing: {missing}")
        hard_local_ids, boundary_embeddings = self._straight_through_boundary(
            selected_anchor_logits
        )
        tensors = self._prepare_step2_inputs(
            hard_local_ids=hard_local_ids,
            boundary_embeddings=boundary_embeddings,
            motion_tokens=guidance_batch["motion_tokens"],
            audio_features=guidance_batch["audio_features"],
            frame_mask=guidance_batch["frame_mask"],
            gap_lengths=guidance_batch["gap_lengths"],
        )
        current = tensors["input_ids"].clone()
        gt_ids = tensors["gt_ids"]
        middle = tensors["middle_mask"]
        slots = torch.arange(current.shape[1], device=current.device).remainder(
            self.tokens_per_frame
        )
        quantizers = slots.remainder(self.num_quantizers).view(1, -1)

        # Audio has no trainable Experiment-A path; caching its frozen encoding
        # avoids rebuilding the same graph four times.
        with torch.no_grad():
            encoded_audio = self.model.audio_encoder(tensors["audio_features"])

        stage_losses: list[torch.Tensor] = []
        ce_sum = selected_anchor_logits.new_zeros((), dtype=torch.float32)
        count = torch.zeros((), device=current.device, dtype=torch.long)
        correct = torch.zeros_like(count)
        q0_count = torch.zeros_like(count)
        q0_correct = torch.zeros_like(count)
        for stage in range(self.num_quantizers):
            logits = self.model(
                input_ids=current,
                audio_features=None,
                encoded_audio=encoded_audio,
                attention_mask=tensors["attention_mask"],
                middle_mask=middle,
                gap_lengths=tensors["gap_lengths"],
                c2f_stage=stage,
                token_embedding_overrides=tensors["token_embedding_overrides"],
                token_embedding_override_mask=tensors[
                    "token_embedding_override_mask"
                ],
            )
            valid = middle & quantizers.eq(stage)
            if not bool(valid.any()):
                raise ValueError(f"Guidance batch has no targets for C2F stage {stage}")
            selected_losses = F.cross_entropy(
                logits[valid].float(),
                gt_ids[valid],
                reduction="none",
            )
            stage_loss = selected_losses.mean()
            stage_losses.append(stage_loss)
            predictions = logits.argmax(dim=-1)
            stage_count = valid.sum()
            stage_correct = predictions[valid].eq(gt_ids[valid]).sum()
            ce_sum = ce_sum + selected_losses.sum()
            count = count + stage_count
            correct = correct + stage_correct
            if stage == 0:
                q0_count = stage_count
                q0_correct = stage_correct
            # C2F inference uses hard, detached earlier-level predictions.
            current[valid] = predictions[valid].detach()

        stage_ce = torch.stack(stage_losses)
        loss = (stage_ce * self.stage_weights.to(stage_ce.device)).sum()
        return FrozenStep2GuidanceOutput(
            loss=loss,
            hard_ce=ce_sum / count.clamp_min(1),
            correct=correct,
            count=count,
            q0_correct=q0_correct,
            q0_count=q0_count,
            stage_ce=stage_ce,
            hard_local_ids=hard_local_ids,
        )

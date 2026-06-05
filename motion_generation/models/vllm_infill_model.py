#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Training-side model code for the Step 2 autoregressive infill LLM.

Important naming note:
    vLLM is an inference engine. During training, we train the same kind of
    HuggingFace causal LM checkpoint that vLLM later serves.

Architecture idea:
    1. Load the Step 1 motion-planning LLM checkpoint as initialization.
    2. Keep the same tokenizer/vocabulary: [audio_N], [res_1_N] ... [res_4_N].
    3. Fine-tune on a new downstream objective:

        left anchor + right anchor + middle audio tokens -> middle motion tokens

    4. Save this as a separate Step 2 checkpoint/LoRA. The Step 1 planner is
       not overwritten.

This file is intentionally written as a readable draft. It is complete enough
to plug into a PyTorch training loop or HuggingFace Trainer, but the exact
training script, distributed setup, and LoRA choices can be added separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100


FrameTokens = List[int]       # One motion frame: [res_1, res_2, res_3, res_4]
MotionSequence = List[FrameTokens]


@dataclass
class MotionInfillExample:
    """
    One supervised training example for Step 2.

    Example for the current step=4 setup:
        left_anchor         = motion frame t
        right_anchor        = motion frame t+4
        middle_audio_tokens = audio tokens at t+1, t+2, t+3
        middle_motion       = motion frames t+1, t+2, t+3

    The model sees the anchors/audio as prompt and learns to generate only
    middle_motion.
    """

    left_anchor: FrameTokens
    right_anchor: FrameTokens
    middle_audio_tokens: List[int]
    middle_motion: MotionSequence
    action_text: Optional[str] = None


@dataclass
class MotionInfillBatch:
    """
    Batch object returned by the collator.

    input_ids:
        Prompt + target token ids.
    attention_mask:
        1 for real tokens, 0 for padding.
    labels:
        Same length as input_ids. Prompt and padding tokens are -100, so the LM
        loss is computed only on target middle-motion tokens.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "MotionInfillBatch":
        """Small helper for plain PyTorch training loops."""

        return MotionInfillBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


def _as_int_audio_token(token: Any) -> int:
    """
    Normalize audio token values.

    Existing data helpers sometimes use either 123 or [123]. Training should not
    care which representation was loaded from JSON.
    """

    if isinstance(token, list):
        if not token:
            raise ValueError("Cannot serialize an empty audio token list")
        return int(token[0])
    return int(token)


def format_audio_tokens(audio_tokens: Iterable[Any]) -> str:
    """Convert audio token ids into the added-token string format."""

    return "".join(f"[audio_{_as_int_audio_token(token)}]" for token in audio_tokens)


def format_motion_frame(frame: Sequence[int]) -> str:
    """
    Convert one RVQ motion frame into four motion-code tokens.

    Your Step 1 tokenizer contains tokens like:
        [res_1_123][res_2_45][res_3_67][res_4_89]
    """

    if len(frame) != 4:
        raise ValueError(f"Expected 4 motion tokens per frame, got {len(frame)}")

    return "".join(f"[res_{i + 1}_{int(value)}]" for i, value in enumerate(frame))


def format_motion_sequence(frames: Iterable[Sequence[int]]) -> str:
    """Convert a sequence of motion frames into LM text tokens."""

    return "".join(format_motion_frame(frame) for frame in frames)


def build_infill_prompt(example: MotionInfillExample) -> str:
    """
    Prompt used for Step 2 training.

    I use plain text task labels here because the current checkpoint already
    understands text and already has the motion/audio special tokens. Later, you
    can add compact special tokens like [infill], [left_motion], [right_motion],
    etc. if you want a cleaner tokenizer vocabulary.
    """

    action_line = ""
    if example.action_text:
        action_line = f"Action: {example.action_text}\n"

    gap_length = len(example.middle_audio_tokens)
    task_text = (
        "Task: audio-conditioned motion infilling.\n"
        "Given the left motion anchor, right motion anchor, and middle audio, "
        "predict the missing middle motion frames.\n"
        f"{action_line}"
        f"Gap length: [len_{gap_length}]\n"
        f"Left anchor: {format_motion_frame(example.left_anchor)}\n"
        f"Right anchor: {format_motion_frame(example.right_anchor)}\n"
        f"Middle audio: {format_audio_tokens(example.middle_audio_tokens)}\n"
        "Middle motion:"
    )

    # Match the repo's current vllm_server.py style instead of using Qwen's
    # chat template. This keeps training and inference formatting aligned.
    return f"Human: {task_text}<|im_end|>\nAssistant:"


def build_infill_completion(example: MotionInfillExample) -> str:
    """
    Target text for Step 2.

    The completion should contain only motion tokens and the stop token. This
    makes generation parsing much easier at inference time.
    """

    return format_motion_sequence(example.middle_motion) + "<|im_end|>"


class MotionInfillSFTDataset(Dataset):
    """
    Dataset that converts dense motion/audio sequences into infill windows.

    Expected input item format:
        {
            "motion_tokens": [[r1, r2, r3, r4], ...],
            "audio_tokens": [a0, a1, a2, ...],
            "action_text": optional string
        }

    You can build this list from the existing JSON files in:
        data/motion_token_data/
        data/audio_tokens_hubert_layer9_fps10/
    """

    def __init__(
        self,
        sequences: Sequence[Dict[str, Any]],
        *,
        step: int = 4,
        audio_fps: Optional[float] = None,
        motion_fps: Optional[float] = None,
        max_windows_per_sequence: Optional[int] = None,
    ):
        if step < 2:
            raise ValueError("step must be >= 2")

        self.examples: List[MotionInfillExample] = []

        for item in sequences:
            motion_tokens = item["motion_tokens"]
            audio_tokens = item["audio_tokens"]
            action_text = item.get("action_text")
            item_audio_fps = item.get("audio_fps", audio_fps)
            item_motion_fps = item.get("motion_fps", motion_fps)

            # If audio_fps and motion_fps are known, audio and motion do not
            # need to have the same number of tokens. For example, Step 2 can
            # use 50fps audio tokens while motion tokens are 20fps.
            if item_audio_fps is None or item_motion_fps is None:
                num_frames = min(len(motion_tokens), len(audio_tokens))
            else:
                num_frames = len(motion_tokens)
            made_for_this_sequence = 0

            for left_idx in range(0, num_frames - step, step):
                right_idx = left_idx + step
                middle_slice = slice(left_idx + 1, right_idx)
                middle_audio_tokens = self._slice_middle_audio(
                    audio_tokens,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    audio_fps=item_audio_fps,
                    motion_fps=item_motion_fps,
                )

                self.examples.append(
                    MotionInfillExample(
                        left_anchor=list(motion_tokens[left_idx]),
                        right_anchor=list(motion_tokens[right_idx]),
                        middle_audio_tokens=middle_audio_tokens,
                        middle_motion=[
                            list(frame) for frame in motion_tokens[middle_slice]
                        ],
                        action_text=action_text,
                    )
                )

                made_for_this_sequence += 1
                if (
                    max_windows_per_sequence is not None
                    and made_for_this_sequence >= max_windows_per_sequence
                ):
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MotionInfillExample:
        return self.examples[idx]

    @staticmethod
    def _slice_middle_audio(
        audio_tokens: Sequence[Any],
        *,
        left_idx: int,
        right_idx: int,
        audio_fps: Optional[float],
        motion_fps: Optional[float],
    ) -> List[int]:
        """
        Select the audio-token interval between two motion anchors.

        Old aligned path:
            audio token i corresponds to motion frame i.

        High-detail Step 2 path:
            audio_fps may be larger than motion_fps. We convert motion-frame
            indices into time, then into audio-token indices, so every infill
            window receives all audio tokens between the anchors.

        Example:
            motion_fps = 20, audio_fps = 50, left_idx=0, right_idx=4
            middle interval is time [1/20, 4/20), so audio indices [2, 10).
        """

        if audio_fps is None or motion_fps is None:
            return [
                _as_int_audio_token(t)
                for t in audio_tokens[left_idx + 1 : right_idx]
            ]

        start = int(round((left_idx + 1) * float(audio_fps) / float(motion_fps)))
        end = int(round(right_idx * float(audio_fps) / float(motion_fps)))
        start = max(0, min(start, len(audio_tokens)))
        end = max(start, min(end, len(audio_tokens)))

        selected = [_as_int_audio_token(t) for t in audio_tokens[start:end]]

        # Very short clips or rounding can produce an empty interval. Add the
        # closest available token so the prompt still contains audio context.
        if not selected and audio_tokens:
            fallback_idx = min(start, len(audio_tokens) - 1)
            selected = [_as_int_audio_token(audio_tokens[fallback_idx])]

        return selected


class MotionInfillCollator:
    """
    Tokenize and pad Step 2 examples for causal-LM supervised training.

    The key part is label masking:
        prompt tokens     -> -100, no loss
        completion tokens -> token id, LM loss is applied
        padding tokens    -> -100, no loss

    HuggingFace causal LMs shift labels internally, so labels should be aligned
    with input_ids in the usual way.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int = 2048,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.tokenizer.pad_token_id is None:
            # Qwen checkpoints often use eos/endoftext as padding. This keeps
            # batching simple without changing the vocabulary.
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode_one(self, example: MotionInfillExample) -> Dict[str, List[int]]:
        prompt = build_infill_prompt(example)
        completion = build_infill_completion(example)

        # Tokenize prompt and target separately so we know exactly which labels
        # should be ignored.
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        completion_ids = self.tokenizer(
            completion,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids[:]

        # Keep the target end if the example is too long. For this task, losing
        # part of the prompt is usually worse than losing examples entirely, so
        # in a real trainer you may prefer filtering long examples beforehand.
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def __call__(self, examples: Sequence[MotionInfillExample]) -> Dict[str, torch.Tensor]:
        encoded = [self._encode_one(example) for example in examples]
        max_len = max(len(item["input_ids"]) for item in encoded)

        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in encoded:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
            batch_labels.append(item["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class MotionInfillCausalLM(nn.Module):
    """
    Thin training wrapper around the Step 1 causal LM checkpoint.

    This is the Step 2 "model.py" idea:
        - it is not a new transformer architecture from scratch;
        - it reuses the Step 1 motion-planning LLM weights;
        - it changes the supervised task and loss mask to infilling.
    """

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm

    @classmethod
    def from_step1_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device_map: Optional[str | Dict[str, Any]] = None,
        local_files_only: bool = True,
        gradient_checkpointing: bool = True,
    ) -> "MotionInfillCausalLM":
        """
        Initialize Step 2 from the Step 1 model checkpoint.

        checkpoint_path should usually be:
            checkpoints/llm

        Save the trained result somewhere else, for example:
            checkpoints/llm_infill
        """

        lm = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # Training does not use KV cache. Disabling it avoids warnings and
        # saves memory, especially with gradient checkpointing.
        lm.config.use_cache = False

        if gradient_checkpointing:
            lm.gradient_checkpointing_enable()

        return cls(lm)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Standard causal-LM forward.

        If labels are provided, HuggingFace computes next-token cross entropy.
        Because the collator sets prompt labels to -100, the loss trains only
        the generated middle-motion region.
        """

        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def save_pretrained(self, output_dir: str, **kwargs) -> None:
        """Save the Step 2 infill checkpoint in HuggingFace format."""

        self.lm.save_pretrained(output_dir, **kwargs)


def load_tokenizer_for_infill(
    checkpoint_path: str,
    *,
    local_files_only: bool = True,
):
    """
    Load the Step 1 tokenizer for Step 2 training.

    This is important: Step 2 should use the exact same added token vocabulary,
    so [audio_N] and [res_i_N] mean the same thing as in Step 1.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def maybe_enable_lora(
    model: MotionInfillCausalLM,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> MotionInfillCausalLM:
    """
    Optional LoRA hook.

    Use this if you want Step 2 to be a lightweight adapter initialized from the
    Step 1 checkpoint. This keeps the base weights frozen and trains only small
    low-rank matrices.

    Requires:
        pip install peft

    For Qwen2, the common target modules are q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, and down_proj.
    """

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is not installed. Install peft or train full fine-tuning."
        ) from exc

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model.lm = get_peft_model(model.lm, peft_config)
    return model


def training_step(
    model: MotionInfillCausalLM,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Minimal plain-PyTorch training step.

    You do not need this if using HuggingFace Trainer, but it shows the exact
    data flow:
        batch -> model -> loss -> backward -> optimizer
    """

    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())

#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Experimental autoregressive infill helpers.

This module intentionally lives outside the original AudioMotionTransformer
class. It reuses the trained mask transformer weights, but changes only the
inference schedule so it is easy to compare against the original pipeline.
"""

import numpy as np
import torch


def _build_window_tokens(frame_tokens, mask_token_id, offsets, num_tokens_per_frame):
    input_tokens = []
    for frame in frame_tokens:
        if frame is None:
            input_tokens.extend([mask_token_id] * num_tokens_per_frame)
            continue

        for j in range(num_tokens_per_frame):
            input_tokens.append(frame[j] + offsets[j])

    return input_tokens


def _get_audio_window(audio_features, start_audio_idx, window_size=5):
    end_audio_idx = start_audio_idx + window_size
    if end_audio_idx <= audio_features.shape[0]:
        return audio_features[start_audio_idx:end_audio_idx]

    if start_audio_idx >= audio_features.shape[0]:
        last_frame = audio_features[-1:]
        return np.tile(last_frame, (window_size, 1))

    available = audio_features[start_audio_idx:]
    pad_len = window_size - available.shape[0]
    if pad_len > 0:
        padding = np.tile(available[-1:], (pad_len, 1))
        return np.concatenate([available, padding], axis=0)

    return available[:window_size]


def _build_framewise_causal_mask(num_frames, num_tokens_per_frame, device):
    """
    Build an anchored causal mask for a 5-frame infill window.

    Frame 0 and frame 4 are anchors and only attend to anchors. Middle frame k
    attends to both anchors and middle frames up to k. This avoids future masked
    frames leaking into target predictions through updated anchor states.
    """
    seq_len = num_frames * num_tokens_per_frame
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

    for query_frame in range(num_frames):
        if query_frame == 0 or query_frame == num_frames - 1:
            allowed_frames = {0, num_frames - 1}
        else:
            allowed_frames = {0, num_frames - 1}
            allowed_frames.update(range(1, query_frame + 1))

        query_start = query_frame * num_tokens_per_frame
        query_end = query_start + num_tokens_per_frame

        for key_frame in allowed_frames:
            key_start = key_frame * num_tokens_per_frame
            key_end = key_start + num_tokens_per_frame
            mask[query_start:query_end, key_start:key_end] = False

    return mask


def generate_framewise_infill_window(
    model,
    left_keyframe,
    right_keyframe,
    window_audio,
    generate_steps=6,
    on_frame=None,
):
    """
    Generate the 3 middle frames using direct next-frame decoding.

    One forward pass predicts all 4 RVQ tokens for frame2, writes that frame
    back, then the next pass predicts frame3, then frame4.

    `generate_steps` is accepted for API compatibility with the old helper but
    is not used by this framewise path.
    """
    ntpf = model.config.num_tokens_per_frame
    codebook_size = model.config.codebook_size
    offsets = [codebook_size * i for i in range(ntpf)]
    mask_token_id = model.config.vocab_size - 1
    device = next(model.parameters()).device
    attention_mask = _build_framewise_causal_mask(
        model.config.num_frames, ntpf, device
    )

    frame_slots = [list(left_keyframe), None, None, None, list(right_keyframe)]
    input_tokens = _build_window_tokens(
        frame_slots, mask_token_id, offsets, ntpf
    )
    input_ids = torch.tensor([input_tokens], device=device)
    audio_feat = torch.tensor(
        window_audio, dtype=torch.float32, device=device
    ).unsqueeze(0)

    generated_frames = []

    for target_frame_idx in range(1, 4):
        with torch.no_grad():
            logits = model(
                input_ids,
                audio_features=audio_feat,
                attention_mask=attention_mask,
            )

        frame_tokens = []
        for token_idx in range(ntpf):
            # Position of this RVQ token inside the flattened 5-frame window.
            # Example: frame 1 token 0 -> pos 4, frame 1 token 1 -> pos 5, etc.
            target_pos = target_frame_idx * ntpf + token_idx

            # Pick the highest-scoring vocabulary token predicted by the model
            # for this position.
            pred_token_id = logits[0, target_pos].argmax(dim=-1)

            # Write the predicted global token ID back into the input sequence so
            # later AR steps can condition on this generated token.
            input_ids[0, target_pos] = pred_token_id

            # Convert from global vocab ID back to the raw RVQ codebook ID by
            # removing this quantizer's offset, then store it in the output frame.
            frame_tokens.append(int(pred_token_id.item() - offsets[token_idx]))

        generated_frames.append(frame_tokens)
        if on_frame is not None:
            on_frame(target_frame_idx, frame_tokens)

    return generated_frames


def interpolate_sequence_ar_framewise(
    model,
    keyframe_tokens,
    audio_features,
    generate_steps=6,
    on_frame=None,
):
    """
    Frame-level autoregressive variant of pipeline_infer.interpolate_sequence.

    Each 3-frame gap is generated as frame2 -> frame3 -> frame4. Each frame
    prediction is one forward pass that writes all 4 RVQ tokens back together.
    """
    num_keyframes = len(keyframe_tokens)
    if num_keyframes < 2:
        return keyframe_tokens

    all_output_frames = []

    for i in range(num_keyframes - 1):
        frame1_tokens = keyframe_tokens[i]
        frame5_tokens = keyframe_tokens[i + 1]
        start_audio_idx = i * 4
        window_audio = _get_audio_window(audio_features, start_audio_idx)

        if i == 0:
            all_output_frames.append(list(frame1_tokens))
            if on_frame is not None:
                on_frame(len(all_output_frames) - 1, list(frame1_tokens), "known")

        interp_frames = generate_framewise_infill_window(
            model,
            frame1_tokens,
            frame5_tokens,
            window_audio,
            generate_steps=generate_steps,
            on_frame=(
                None
                if on_frame is None
                else lambda local_idx, frame: on_frame(
                    len(all_output_frames) + local_idx - 1,
                    frame,
                    f"generated_f{local_idx + 1}",
                )
            ),
        )

        all_output_frames.extend(interp_frames)
        all_output_frames.append(list(frame5_tokens))
        if on_frame is not None:
            on_frame(len(all_output_frames) - 1, list(frame5_tokens), "known")

    return all_output_frames

# Phase 1 Fixed-Gap Multipart Planner Implementation

**Status:** implemented and locally smoke-tested on 2026-07-20  
**Scope:** causal Mimi preprocessing, causal fixed-gap Step 1 dataset/model, 16-slot anchor CE, DDP training, checkpointing, and generated-prefix cache consumption.  
**Out of scope:** learned gap selection, Step 2 modification, latent/decoded anchor loss, and end-to-end Step 2 rollout.

## 1. Locked Phase 1 contract

- Qwen initialization: `checkpoints/llm` (Qwen2, hidden size 896).
- Complete expression tag, action tag, and transcript are known at utterance start.
- Audio arrives causally as Mimi tokens at 12.5 Hz.
- Preprocessing stores q0-q7, each in `[0, 2047]`; the first baseline consumes q0 only.
- Body tokens come from four newly trained causal codecs at 10 Hz.
- Anchor layout is exactly:

```text
upper q0 q1 q2 q3
lower q0 q1 q2 q3
feet  q0 q1 q2 q3
hands q0 q1 q2 q3
```

- Every anchor contains 16 local ids in `[0, 511]`.
- The existing Qwen checkpoint already has `[body_0]` through `[body_8191]`.
- Logical mapping is `global_body_id = slot * 512 + local_id`.
- Phase 1 uses runtime-supplied `[gap_3]`, not a learned scheduler.
- A final `[gap_0]`, `[gap_1]`, or `[gap_2]` reaches the exact last token when the clip length is not `1 mod 4`.
- The seed is always a known current state:
  - `[seed_previous]`: previous utterance's final anchor;
  - `[seed_neutral]`: configured conversation-start pose; or
  - `[seed_observed]`: known current pose/debug path.
- Initial training uses `mixed_known`, which exposes both observed and previous-state tags with real first-frame anchors. `mixed_all` additionally requires a verified neutral seed JSON.

## 2. Causal sequence seen by Qwen

The serialized training stream is interleaved instead of placing the complete audio before the completion:

```text
Human: [step1_planner]<complete annotation><|im_end|>
Assistant:
[motion_start]
[seed_previous or seed_observed][anchor]<16 known seed ids>

[gap_3]
[mimi_frame] x 5
[anchor]<16 target body ids>

[gap_3]
[mimi_frame] x 5
[anchor]<16 target body ids>
...
[gap_tail]
[mimi_frame] x remaining causal audio
[anchor]<16 final target body ids>
[audio_end][motion_end]<|im_end|>
```

The placeholder id `[mimi_frame]` is replaced by a learned `Embedding(2048, 896)` value. It is not represented by the legacy `[audio_*]` tokenizer symbols. Qwen's normal causal mask guarantees that an anchor can see complete text, known previous anchors, and audio received up to that anchor, but not later audio.

Exactly 25 new control tokens are added to the old tokenizer. The 8,192 body symbols are reused rather than added again.

## 3. Phase 1 loss

Only target-anchor slots are supervised. Prompt text, seed ids, audio placeholders, gap controls, and end controls are masked.

For slot `s`, the classifier is restricted to that slot's 512 existing Qwen output rows:

```text
allowed(s) = Qwen ids for [body_(s*512)] ... [body_(s*512+511)]
logits_s   = hidden_before_target @ output_embedding[allowed(s)].T
loss       = mean CE over every predicted anchor slot
```

This is a real 512-way loss, not a 225k-vocabulary CE. Random-uniform reference values are:

- CE: `ln(512) = 6.2383`
- slot accuracy: `1/512 = 0.1953%`

Training reports aggregate accuracy and all 16 part/quantizer accuracies. Latent and decoded-anchor losses are intentionally deferred until token CE converges.

## 4. Files

```text
motion_generation/utils/adaptive_anchor_tokens.py
motion_generation/models/step1_mimi_planner.py
motion_generation/scripts/precompute_mimi_audio_tokens.py
motion_generation/scripts/export_step1_seed_anchor.py
motion_generation/scripts/validate_step1_fixed_gap_data.py
motion_generation/scripts/train_step1_multipart_fixed_gap3.py
motion_generation/configs/step1_multipart_fixed_gap3.yaml
motion_generation/models/test_step1_mimi_planner.py
motion_generation/notebooks/phase1_mimi_preflight.ipynb
```

## 5. Remote prerequisite

Install the local Moshi package in the same environment used for preprocessing:

```bash
python -m pip install -e /path/to/moshi/moshi
```

The training process itself does not import Moshi; it reads precomputed `.npz` files.

## 6. Precompute all eight Mimi levels

One GPU is sufficient:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  motion_generation/scripts/precompute_mimi_audio_tokens.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --output_dir SuSuInterActs/SuSuInterActs/audio_tokens_mimi_12p5hz_8cb \
  --moshi_repo /path/to/moshi \
  --mimi_weight checkpoints/mimi/tokenizer-e351c8d8-checkpoint125.safetensors \
  --device cuda:0 \
  --batch_size 16 \
  --max_padded_batch_seconds 120 \
  --verify_existing
```

For four independent preprocessing jobs, use `--num_shards 4` and one of `--shard_id 0`, `1`, `2`, or `3` in each process. When each process sees only one GPU through `CUDA_VISIBLE_DEVICES`, keep `--device cuda:0`.

Each clip produces:

```text
audio_tokens_mimi_12p5hz_8cb/<relative clip name>.npz
    codes: uint16 [8, T_audio]
    sample/frame/cardinality/source metadata
```

The script is resumable by default and writes one manifest per shard.

## 7. Audit and export the new causal motion tokens

After all four Phase 0 codecs finish, audit representative clips first:

```bash
python motion_generation/scripts/audit_causal_body_codecs.py \
  --upper_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth \
  --lower_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth \
  --feet_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth \
  --hands_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth \
  --device cuda:0 \
  --max_clips 32 \
  --output_json checkpoints/causal_multipart_rvqvae/phase0_audit.json
```

Then export all dense 10 Hz body tokens:

```bash
python motion_generation/scripts/export_multipart_motion_tokens.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --split_file SuSuInterActs/SuSuInterActs/split/all_file_list.txt \
  --output_dir SuSuInterActs/SuSuInterActs/motion_token_data_multipart_causal_512x4 \
  --upper_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth \
  --lower_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth \
  --feet_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth \
  --hands_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth \
  --device cuda:0
```

The Step 1 loader rejects any payload whose `body_causal` is not exactly `true`, whose part order differs, or whose slot/rate/codebook metadata is wrong.

## 8. Optional neutral conversation-start seed

Choose and visually verify a genuinely neutral frame from the causal token export, then write its 16 ids:

```bash
python motion_generation/scripts/export_step1_seed_anchor.py \
  --motion_token_json SuSuInterActs/SuSuInterActs/motion_token_data_multipart_causal_512x4/<clip>.json \
  --frame_index 0 \
  --output motion_generation/configs/step1_neutral_seed.json
```

After verification, change the YAML to:

```yaml
seed_mode: mixed_all
neutral_seed_json: motion_generation/configs/step1_neutral_seed.json
```

Do not call an arbitrary first frame neutral without visual verification.

## 9. Four-GPU training

Validate every training and validation record before allocating four GPUs:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_multipart_fixed_gap3.yaml \
  --output_json checkpoints/step1_multipart_fixed_gap3/data_preflight.json
```

This catches missing shards, noncausal motion exports, duration mismatches, token-range errors, and sequences over `max_length`.

First run a small integration job:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_fixed_gap3.yaml \
  --max_train_clips 64 \
  --max_eval_clips 32
```

Then run the complete training by removing the two `--max_*_clips` arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_fixed_gap3.yaml
```

Resume all ranks from the same planner checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_fixed_gap3.yaml \
  --resume_from_checkpoint checkpoints/step1_multipart_fixed_gap3/checkpoint-500
```

The default global anchor-token batch is `2 clips/GPU x 4 GPUs x 8 accumulation = 64 clips/update`. Adjust only after measuring actual sequence lengths and GPU memory.

## 10. Checkpoint format

Every saved directory is a standalone `MimiQwenPlanner` checkpoint containing:

- the complete resized Qwen weights;
- the separate Mimi embedding;
- the fixed 16x512 allowed-token table;
- the tokenizer and new controls;
- optimizer/scheduler/global-step state; and
- the source training YAML serialized as JSON.

`latest_checkpoint.txt` points to the most recent save. `final/` is written after all epochs.

## 11. Generated-prefix curriculum

The first run uses GT previous anchors. The dataset already supports scheduled sampling from a rollout cache. A cache file has either form:

```json
{"anchors": [{"time": 4, "tokens": [16 local ids]}, {"time": 8, "tokens": [16 local ids]}]}
```

or:

```json
{"anchor_tokens_by_time": {"4": [16 local ids], "8": [16 local ids]}}
```

Set `generated_anchor_dir` and gradually raise `generated_prefix_probability` only after the GT-prefix baseline converges. Input anchor ids are replaced by generated ids while CE targets remain GT, producing genuine generated-prefix exposure without changing the sequence grammar.

The rollout-cache producer is intentionally deferred until the first checkpoint exists.

## 12. Validation gates before Phase 2

1. Anchor CE must improve materially below the random `6.2383` reference.
2. Report all 16 slot accuracies; do not accept a result driven only by easy lower-body slots.
3. Compare text-only, text+q0, and text+multiple Mimi levels before declaring q0 sufficient.
4. Measure GT-prefix versus generated-prefix degradation.
5. Decode predicted anchors through the new causal body codecs and inspect kinematics.
6. Run predicted anchors through the frozen Step 2 reference before implementing adaptive gaps.

Passing the causal Mimi preflight proves representation correctness, not downstream motion predictability.

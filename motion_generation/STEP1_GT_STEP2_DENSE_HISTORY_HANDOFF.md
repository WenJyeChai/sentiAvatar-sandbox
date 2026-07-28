# Step 1 Stage 1: GT-boundary Step 2 dense history

## Purpose

This experiment changes both motion history and audio visibility for the
Stage 1 anchor predictor. Previous sparse anchors are blocked from later
targets. Each target receives a recent contiguous suffix of motion completed
by frozen Step 2 and only its own aligned audio interval.

It tests one precise hypothesis:

> Does local dense motion continuity make the next multipart anchor easier to
> predict than a sparse sequence of earlier anchors?

It does **not** learn gap placement. Every next gap is a supplied, unsupervised
control. Only the 16 IDs of each next anchor receive cross-entropy.

## Exact causal contract

For cached schedule anchors `a0, a1, ..., ak`, frozen Step 2 is run offline on
each interval `(a(i-1), ai)` with its GT left and right boundaries. The cache
retains:

1. every greedy Step 2 missing frame in that interval; and
2. the ordinary GT right endpoint `ai`.

Before predicting `a(i+1)`, the serializer samples a contiguous suffix of
1--15 frames from this completed interval. The suffix always ends at `ai`.
There is no special GT-boundary token: the last frame has the same
representation as every other dense motion frame.

All intervals remain physically packed in one utterance record:

```text
[seed_observed] <16 seed IDs>

<Nano q0-q3 audio aligned from a0 to a1>
[gap_g0]                         # input only; ignored label
[anchor] <16 target a1 IDs>      # supervised

[step2_history_start]
  [motion_history_frame] <16 Step 2 IDs>
  ...
  [motion_history_frame] <16 IDs ending at a1>
[step2_history_end]
<Nano q0-q3 audio aligned from a1 to a2>
[gap_g1]                         # input only; ignored label
[anchor] <16 target a2 IDs>      # supervised
...
```

For each Nano frame, q0, q1, q2 and q3 are ordinary vocabulary tokens in
time-major order. The sum of all interval chunks equals the utterance audio,
but the segment-causal attention mask prevents a target from reading any
other interval.

The mask permits a target to read:

- complete structured text;
- the observed seed;
- previous supplied gap controls;
- its current Step 2 history;
- its own aligned audio interval;
- earlier slots of the same 16-ID anchor.

It blocks previous sparse anchors, previous audio intervals, and obsolete
history blocks. Thus earlier anchor IDs remain physically present for their
own teacher-forced loss but are not causal context for a later target.

Text, seed, audio, gaps, history, and control markers all have ignored labels.
The preflight checks that supervised positions remain exactly:

```text
(number of anchors - 1) * 16
```

There is no gap CE, latent loss, frozen-Step-2 online loss, or generated
history loss in this Stage 1 run.

## Robustness corruption

Training uses a 50/50 clean/corrupted replay mixture. On a selected cached
interval, 5--15% of eligible history IDs are corrupted with:

- slot-specific mask: weight 0.5;
- same-slot value from another cached frame: weight 0.3;
- previous-frame hold in the same slot: weight 0.2.

The most recent endpoint is preserved. Corruption and history-length draws are
deterministic functions of seed, epoch, clip, and interval, so a run is
reproducible while still changing across epochs. Evaluation serialization is
clean.

This is deliberately not self-generated-history training. The existing
online rollout implementation remains untouched for a later experiment.

## Required inputs

Starting from raw SuSuInterActs, first produce the inputs documented in
`motion_generation/STEP1_OFFLINE_DP_FULL_AUDIO_TEACHER_HANDOFF.md`, sections
1--3:

- causal multipart body tokens:
  `motion_token_data_multipart_causal_512x4`;
- Nano q0--q15 tokens:
  `audio_tokens_moss_nano_48k_12p5hz_16cb`;
- Nano all-16 quantized features:
  `audio_features_moss_nano_all16_12p5hz_768d`.

The following checkpoints must also exist:

```text
checkpoints/llm/
checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15/
```

## Generate the offline history cache

Open four terminals and run one command per GPU. They write disjoint shards to
the same directory; do not merge the manifests.

GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/cache_step1_gt_step2_history.py \
  --step2_config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --step2_checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_gt_boundary_step2_history_uniform_gap_seed42 \
  --device cuda:0 --batch_size 256 --clip_batch_size 128 \
  --schedule_seed 42 --min_gap 3 --max_gap 15 \
  --num_shards 4 --shard_id 0
```

GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python motion_generation/scripts/cache_step1_gt_step2_history.py \
  --step2_config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --step2_checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_gt_boundary_step2_history_uniform_gap_seed42 \
  --device cuda:0 --batch_size 256 --clip_batch_size 128 \
  --schedule_seed 42 --min_gap 3 --max_gap 15 \
  --num_shards 4 --shard_id 1
```

GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 python motion_generation/scripts/cache_step1_gt_step2_history.py \
  --step2_config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --step2_checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_gt_boundary_step2_history_uniform_gap_seed42 \
  --device cuda:0 --batch_size 256 --clip_batch_size 128 \
  --schedule_seed 42 --min_gap 3 --max_gap 15 \
  --num_shards 4 --shard_id 2
```

GPU 3:

```bash
CUDA_VISIBLE_DEVICES=3 python motion_generation/scripts/cache_step1_gt_step2_history.py \
  --step2_config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --step2_checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_gt_boundary_step2_history_uniform_gap_seed42 \
  --device cuda:0 --batch_size 256 --clip_batch_size 128 \
  --schedule_seed 42 --min_gap 3 --max_gap 15 \
  --num_shards 4 --shard_id 3
```

Every manifest must report `missing_or_bad: 0`. Re-running the same command is
safe: existing clip caches are checked by presence and skipped unless
`--overwrite` is passed.

## Preflight

First validate a smoke subset:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_stage1_gt_step2_dense_history_uniform_gap100.yaml \
  --max_train_clips 32 --max_eval_clips 32 \
  --output_json checkpoints/step1_stage1_gt_step2_dense_history_uniform_gap100/data_preflight_smoke.json
```

Then validate every train/validation record:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_stage1_gt_step2_dense_history_uniform_gap100.yaml \
  --output_json checkpoints/step1_stage1_gt_step2_dense_history_uniform_gap100/data_preflight.json
```

Do not train unless the result ends in `GO`. Inspect the reported dense-history
frame and corrupted-token distributions, target-interval audio lengths,
`supervised_gap_tokens` (which must be zero), and maximum sequence length.

## One-GPU smoke training

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_gt_step2_dense_history_uniform_gap100.yaml \
  --max_train_clips 64 --max_eval_clips 32 \
  --num_train_epochs 1 --max_train_steps 2 \
  --output_dir checkpoints/step1_stage1_gt_step2_dense_history_smoke
```

The header must say:

```text
Base/init/resume: checkpoints/llm
Supplied schedule: ...step1_gt_boundary_step2_history_uniform_gap_seed42
Step 2 history: cache=..., suffix_frames=1-15, boundary=GT, labels=ignored
Dense corruption: enabled=True, examples=50%, rate=5%-15%, preserve_endpoint=True
Gap supervision: disabled
```

The serialized example and config must also report
`segment_causal+interval_audio_isolated`.

## Full four-GPU candidate

Do not pass `--init_from_checkpoint`; the YAML intentionally initializes from
`checkpoints/llm`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun \
  --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_gt_step2_dense_history_uniform_gap100.yaml
```

## Matched sparse-history control

The control reads the same cached schedules, uses the same target-aligned
audio, segment mask, targets and added vocabulary, and starts independently
from `checkpoints/llm`. Its sole intentional difference is omission of the
dense history blocks. Because previous sparse anchors are blocked in both
runs, this is a deliberately difficult but clean history ablation.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun \
  --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_cached_schedule_sparse_control100.yaml
```

## Interpretation boundary

This cache is a best-case/bridge distribution: Step 2 sees GT boundaries.
Success would establish that dense completed motion is useful. It would not
establish deployment robustness, because deployment uses predicted boundaries.
The candidate must later be evaluated in two separate modes:

1. GT-boundary Step 2 history, matching this training distribution;
2. closed-loop history in which Step 1 predicts the endpoint and Step 2 fills
   the interval before the next anchor is requested.

Do not use the ordinary sparse generated-history evaluator as a substitute for
the second protocol.

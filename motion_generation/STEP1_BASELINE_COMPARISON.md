# Step 1 Baseline Comparison Protocol

**Status:** implemented; run on the validation split after the self-forcing job
was manually stopped around epoch 30. The test split remains sealed.

## 1. Systems under comparison

### Direct multipart comparison

These checkpoints predict the same fixed-`[gap_3]`, 16-slot causal multipart
anchor representation, so CE, top-k accuracy, generated-rollout accuracy,
calibration, and per-slot metrics are directly comparable:

1. `checkpoints/step1_multipart_fixed_gap3_self_forcing_q0q3_full/best`
2. `checkpoints/step1_multipart_fixed_gap3_self_forcing_q0q3_full/best_rollout`
3. `checkpoints/step1_multipart_fixed_gap3_main6000/final`

The first two consume Mimi q0-q3 and were trained with an on-policy generated-
history curriculum. The third is the earlier 6,000-clip, Mimi-q0,
teacher-forced-only baseline.

### Released SentiAvatar comparison

`checkpoints/llm` is the released Qwen2-0.5B SentiAvatar motion-token planner.
It consumes action text plus 10 Hz HuBERT tokens and predicts four legacy
whole-body RVQ IDs per sparse frame.

The legacy and multipart IDs are different learned representations. Therefore:

- compare raw CE and accuracy only among multipart checkpoints;
- compare each representation against its own uniform and persistence
  references;
- use persistence ratio, output coverage, horizon degradation, latency, and
  decoded/end-to-end motion as cross-system evidence; and
- do not claim that a larger raw token percentage wins across representations.

## 2. Evaluation controls

- Split: official 635-clip validation split only.
- Subset seed: 42.
- Schedule: fixed interval four motion-token frames.
- Decoding: greedy.
- Multipart seed: known observed/previous seed, matching training.
- Released planner: has no motion seed and declares interval-end anchors at
  frame 3, 7, 11, ... for step four. It is scored at those native frame labels
  rather than shifted onto the multipart seed-at-0 / anchors-at-4 schedule.
- Missing or malformed released-planner anchors count as incorrect. Matched-only
  accuracy is diagnostic and is never the primary score.
- Previous-GT-anchor copy is always computed on the exact evaluated subset.

The pilot uses all validation clips for multipart teacher forcing and the same
deterministic 128 clips for generated rollout in both representations. After
the pilot is inspected, use all 635 clips for the final generated-rollout table.

## 3. Prerequisites

The multipart exports are already required by Phase 1. The released planner
additionally needs these native legacy exports:

```text
SuSuInterActs/SuSuInterActs/motion_token_data/
SuSuInterActs/SuSuInterActs/audio_tokens_hubert_layer9_fps10/
```

If they are absent, generate them with the released RVQVAE, Chinese HuBERT, and
K-means checkpoints. Four independent terminals may process shards in parallel:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/preprocess_data.py --all --data_dir SuSuInterActs/SuSuInterActs --device cuda:0 --num_shards 4 --shard_id 0
```

```bash
CUDA_VISIBLE_DEVICES=1 python motion_generation/scripts/preprocess_data.py --all --data_dir SuSuInterActs/SuSuInterActs --device cuda:0 --num_shards 4 --shard_id 1
```

```bash
CUDA_VISIBLE_DEVICES=2 python motion_generation/scripts/preprocess_data.py --all --data_dir SuSuInterActs/SuSuInterActs --device cuda:0 --num_shards 4 --shard_id 2
```

```bash
CUDA_VISIBLE_DEVICES=3 python motion_generation/scripts/preprocess_data.py --all --data_dir SuSuInterActs/SuSuInterActs --device cuda:0 --num_shards 4 --shard_id 3
```

The exporters write clip files directly into the shared directory and skip
existing outputs. No manifest merge is required.

## 4. Pilot execution

The notebook is the recommended entry point:

```text
motion_generation/notebooks/evaluate_step1_baseline_comparison.ipynb
```

The equivalent multipart command is:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/evaluate_step1_multipart_comparison.py \
  --teacher_max_clips 0 \
  --rollout_max_clips 128 \
  --rollout_batch_size 8 \
  --subset_seed 42
```

The released SentiAvatar command is:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/evaluate_sentiavatar_legacy_step1.py \
  --checkpoint checkpoints/llm \
  --data_dir SuSuInterActs/SuSuInterActs \
  --max_clips 128 \
  --batch_size 8 \
  --subset_seed 42
```

If either command runs out of memory, reduce only its rollout/generation batch
size first. This does not change the evaluated examples or decoding policy.

For the full validation rollout, change both `128` values to `0`.

## 5. Outputs

All outputs are written beneath:

```text
motion_generation/outputs/step1_baseline_comparison/
```

Important files:

- `multipart_teacher_forced.csv`
- `multipart_generated_rollout.csv`
- `multipart_generated_rollout_per_slot.csv`
- `multipart_generated_rollout_horizon.csv`
- `multipart_reference_baselines.csv`
- `legacy_sentiavatar_rollout.csv`
- `legacy_sentiavatar_rollout_per_clip.csv`
- `legacy_sentiavatar_rollout_horizon.csv`
- `legacy_sentiavatar_raw_generations.jsonl`
- `cross_representation_summary.csv` (created by the notebook)

## 6. Baseline selection rule

Select the multipart checkpoint using generated rollout, not teacher-forced CE
alone. The self-forcing baseline should:

1. improve generated-prefix accuracy and q0 accuracy over `q0_6k_final`;
2. reduce the teacher-forced-to-rollout accuracy drop;
3. reduce horizon collapse and expected calibration error;
4. improve its margin relative to exact-subset previous-anchor copying; and
5. retain acceptable latency and peak memory.

Beating the released planner's native token accuracy is not a valid final claim.
After selecting the multipart checkpoint, the common scientific comparison is
decoded raw motion and frozen-Step-2 output, with each system using its own
codec. That table must include codec-oracle error, incremental anchor damage,
infilled motion quality, latency, and anchor count.

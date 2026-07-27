# Step 1 Offline-DP Full-Audio Teacher

## Experiment contract

The teacher is initialized from `checkpoints/llm`, not from a completed Step 1
checkpoint. Its known condition is serialized as:

```text
structured text
all MOSS Nano q0-q3 audio frames
[audio_end]
[motion_start] seed-mode [anchor] 16 known seed IDs
---------------------------------------------------- prefix boundary
[gap_g] [anchor] 16 target body IDs
[gap_g] [anchor] 16 target body IDs
...
```

The condition above the boundary uses prefix-LM attention: every condition
token can attend every other condition token. Plan tokens below the boundary
attend the full condition and earlier plan tokens only. No target motion token
is allowed inside the bidirectional prefix.

Training uses:

```text
anchor CE + curriculum-weighted offline-DP soft gap CE
```

It does not use online Step 2 guidance, generated history, visited-state
rollout, expected distortion, or condition-alignment loss.

Main configuration:

```text
motion_generation/configs/step1_offline_dp_full_audio_teacher50.yaml
```

## Starting from raw SuSuInterActs

The raw dataset root must contain:

```text
SuSuInterActs/SuSuInterActs/
  motion_data/
  wav_data/
  text_data/motion2text.json
  split/all_file_list.txt
  split/train_file_list.txt
  split/val_file_list.txt
```

The following trained artifacts are also prerequisites; raw data cannot
reconstruct them:

```text
checkpoints/llm/
checkpoints/moss_audio_tokenizer_nano/
checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth
checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth
checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth
checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth
checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15/
```

All commands below are run from the repository root. Four preprocessing
processes write disjoint modulo shards into one shared output directory.

### 1. Export causal multipart motion tokens

Launch one command per GPU, changing both `CUDA_VISIBLE_DEVICES` and
`--shard_id` from 0 through 3:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/export_multipart_motion_tokens.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --split_file SuSuInterActs/SuSuInterActs/split/all_file_list.txt \
  --output_dir SuSuInterActs/SuSuInterActs/motion_token_data_multipart_causal_512x4 \
  --upper_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth \
  --lower_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth \
  --feet_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth \
  --hands_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth \
  --device cuda:0 --num_shards 4 --shard_id 0
```

After all four shards finish, verify every record and create the consolidated
manifest required by Step 2:

```bash
python motion_generation/scripts/verify_multipart_motion_token_export.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --split_file SuSuInterActs/SuSuInterActs/split/all_file_list.txt \
  --output_dir SuSuInterActs/SuSuInterActs/motion_token_data_multipart_causal_512x4 \
  --num_shards 4
```

Do not pass `--allow_noncausal_body`.

### 2. Export all 16 MOSS Nano audio codebooks

Again launch shards 0, 1, 2, and 3 on GPUs 0, 1, 2, and 3:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/precompute_moss_nano_audio_tokens.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --split_file SuSuInterActs/SuSuInterActs/split/all_file_list.txt \
  --model_dir checkpoints/moss_audio_tokenizer_nano \
  --output_dir SuSuInterActs/SuSuInterActs/audio_tokens_moss_nano_48k_12p5hz_16cb \
  --device cuda:0 --num_shards 4 --shard_id 0 \
  --batch_size 8 --compute_dtype bf16 --attention_implementation sdpa
```

The `.npz` files contain q0-q15. Step 1 reads q0-q3; keeping all 16 avoids
re-encoding audio for Step 2. Shard manifests do not need to be merged.

### 3. Export all-16 Nano quantized latents for frozen Step 2

Launch four shards as above:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/export_moss_nano_all16_features.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --split_file SuSuInterActs/SuSuInterActs/split/all_file_list.txt \
  --model_dir checkpoints/moss_audio_tokenizer_nano \
  --token_dir SuSuInterActs/SuSuInterActs/audio_tokens_moss_nano_48k_12p5hz_16cb \
  --output_dir SuSuInterActs/SuSuInterActs/audio_features_moss_nano_all16_12p5hz_768d \
  --device cuda:0 --num_shards 4 --shard_id 0 \
  --batch_size 128 --feature_dtype float16 --compute_dtype bf16
```

### 4. Cache frozen-Step-2 interval costs

This is the expensive offline oracle pass. It scores every legal interval for
the train and validation clips with GT left/right boundaries. Launch four
shards:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/cache_step2_interval_costs.py \
  --config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_adaptive_gap_interval_costs \
  --device cuda:0 --batch_size 256 --num_shards 4 --shard_id 0
```

Calibration refuses incomplete shard manifests, so every shard must report
`missing_or_bad: 0` and cover all assigned clips.

### 5. Solve and materialize the global DP schedules

```bash
python motion_generation/scripts/calibrate_step1_adaptive_gap.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50.yaml \
  --cost_dir checkpoints/step1_adaptive_gap_interval_costs \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_json checkpoints/step1_adaptive_gap_oracle/calibration.json \
  --calibration_max_clips 512 \
  --ce_weight 1.0 --latent_weight 0.1
```

This writes one schedule `.npz` for each DP curriculum phase plus
`calibration.json`. The schedules cover all 19,019 training and 635 validation
clips; only penalty calibration uses the 512-clip subset.

### 6. Preflight the exact teacher serialization

First run a small check:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50.yaml \
  --max_train_clips 32 --max_eval_clips 32 \
  --output_json checkpoints/step1_offline_dp_full_audio_teacher50/data_preflight_smoke.json
```

Then validate every phase and every record:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50.yaml \
  --output_json checkpoints/step1_offline_dp_full_audio_teacher50/data_preflight.json
```

Proceed only after `GO: every selected Phase 1 record passed serialization and
alignment checks`.

### 7. Two-update model smoke test

This uses the real 50-epoch configuration but exits cleanly after two optimizer
updates:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50.yaml \
  --max_train_clips 64 --max_eval_clips 32 \
  --max_train_steps 2 \
  --output_dir checkpoints/step1_offline_dp_full_audio_teacher50_smoke
```

The header must show:

```text
Base/init/resume:  .../checkpoints/llm
Planner context:   attention=prefix_lm, layout=full_audio_prefix
Generated history: enabled=False
Visited state:     enabled=False
Frozen Step 2:     enabled=False
```

### 8. Full four-GPU teacher training

Do not pass `--init_from_checkpoint`; the YAML already selects
`checkpoints/llm`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50.yaml
```

With 19,019 clips and global batch size 128, this is approximately 149 updates
per epoch and 7,450 updates over 50 epochs.

## Expected checkpoint/evaluation behavior

The model checkpoint records `planner_attention_mode=prefix_lm`. During
adaptive rollout, the complete condition prefix is prefetched once with
bidirectional attention and retained in the KV cache; generated gap/anchor
tokens are then appended causally. Audio is not inserted again during rollout.

This model is an offline upper-bound teacher. It cannot be deployed as the
online causal student because it consumes the complete utterance audio before
the first gap decision.

## Ordinary q0-q3 vocabulary-token ablation

The matched ablation config is:

```text
motion_generation/configs/step1_offline_dp_full_audio_teacher50_nano_q0q3_vocab.yaml
```

It reuses the same Nano `.npz` exports and offline DP calibration. No audio or
motion preprocessing needs to be repeated. The only modeling change is the
audio input representation: each 12.5 Hz frame becomes four ordinary Qwen
vocabulary positions in `q0,q1,q2,q3` order instead of one custom fused-frame
embedding. It still initializes from `checkpoints/llm`, uses GT anchor history,
and retains the full-audio prefix-LM teacher context and the same 50-epoch DP
curriculum.

Run the full preflight first:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50_nano_q0q3_vocab.yaml \
  --output_json checkpoints/step1_offline_dp_full_audio_teacher50_nano_q0q3_vocab/data_preflight.json
```

The report must show `audio_vocabulary_token_count: 4096`, all records valid,
and per-clip vocabulary-token positions equal to four times the audio-frame
count. Supervised targets must never be truncated.

Then run a two-update smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50_nano_q0q3_vocab.yaml \
  --max_train_clips 64 --max_eval_clips 32 \
  --max_train_steps 2 \
  --output_dir checkpoints/step1_offline_dp_full_audio_teacher50_nano_q0q3_vocab_smoke
```

The header must show `input=ordinary_tokens`,
`attention=prefix_lm, layout=full_audio_prefix`, and all generated-history and
online-guidance features disabled.

Finally, start the full run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_offline_dp_full_audio_teacher50_nano_q0q3_vocab.yaml
```

The per-device batch is 8 with four accumulation microsteps, preserving the
original four-GPU global batch of 128. Do not initialize this ablation from the
previous fused-audio teacher: its tokenizer, sequence layout at audio frames,
and audio input parameters have a different contract.

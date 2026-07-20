# Phase 0: Causal Multipart Body Codec

## Goal and frozen interface

Phase 0 replaces only the four body tokenizers with codecs trained from
scratch. The face codec remains unchanged and is not part of the causal claim.

Each body part keeps the current downstream interface:

- parts, in slot order: `upper`, `lower`, `feet`, `hands`
- one independent codec per part
- input motion: 20 Hz
- output tokens: 10 Hz
- four residual quantizers per part
- 512 entries per quantizer
- 16 body token IDs per 10 Hz token frame after concatenating all parts
- the existing part split, root canonicalization, and per-part normalization

The new temporal convention is explicit: token `j` is emitted only after the
20 Hz frames `2*j` and `2*j+1` have arrived. It represents that completed
100 ms chunk. An odd final source frame is retained in the source metadata but
does not emit a token until another frame arrives.

## What changed

- `CausalConv1d` uses left-only padding.
- The strided encoder convolution is aligned to the end of a complete stride
  chunk, producing `floor(T / 2)` tokens rather than exposing a partial chunk.
- Every encoder convolution and residual block is causal.
- Every decoder convolution and residual block is causal. Nearest-neighbor
  upsampling produces exactly two motion frames per token.
- `BN` and `GN` are rejected for causal codecs because they aggregate over the
  time axis. The supplied configuration uses no normalization; channel-wise
  `LN` is also permitted.
- EMA codebook state and initialization status are now checkpointed. Old
  checkpoints containing only `codebook` remain loadable.
- DDP code replacement is synchronized across ranks.
- Reconstruction and velocity losses mask dataset padding. For causal clips an
  unpaired odd final source frame is also excluded from the objective.
- Checkpoints and token exports record `causal` and temporal-alignment metadata.

The old noncausal model path is still the default unless `causal: true` or
`--causal` is supplied.

## Architecture and objective

The supplied configuration retains the old controlled baseline architecture:

- code dimension 512
- width 512
- depth 3
- dilation growth 3
- VQ CNN depth 3
- four non-shared 512-entry EMA codebooks
- reconstruction Smooth-L1 weight 1.0
- velocity Smooth-L1 weight 1.0
- commitment weight 0.02

Its causal encoder receptive field is 118 source frames (5.9 seconds at 20
Hz). Training therefore uses 128-frame crops instead of the old 64-frame crops.
The default per-GPU batch is 64; reduce it if a particular remote environment
runs out of memory.

## Local or remote preflight

From the repository root:

```bash
python -m pytest motion_generation/models/test_causal_multipart_rvqvae.py -q
python motion_generation/scripts/train_multipart_rvqvae.py \
  --causal --parts upper --max_train_clips 2 --max_val_clips 1 \
  --max_stat_clips 2 --window_size 8 --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 --num_workers 0 --codebook_size 16 \
  --num_quantizers 2 --code_dim 8 --width 16 --depth 1 \
  --vq_cnn_depth 1 --epochs 1 --dry_run_batches 1 \
  --output_dir /tmp/sentiavatar_phase0_smoke --experiment_name smoke \
  --device cpu --disable_tqdm
```

The first command covers token alignment, encoder/decoder prefix invariance,
future perturbations, odd-tail behavior, streaming equivalence, normalization
guards, and EMA checkpoint compatibility. The second command exercises one
actual dataset batch.

## Four RTX 4090 training launch

Open four terminals from the repository root and run one command in each. Each
YAML is self-contained; no shell launcher is required.

GPU 0, upper body:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_generation/scripts/train_multipart_rvqvae.py \
  --config motion_generation/configs/causal_rvq_upper_512x4.yaml \
  --device cuda:0
```

GPU 1, lower body:

```bash
CUDA_VISIBLE_DEVICES=1 python motion_generation/scripts/train_multipart_rvqvae.py \
  --config motion_generation/configs/causal_rvq_lower_512x4.yaml \
  --device cuda:0
```

GPU 2, hands:

```bash
CUDA_VISIBLE_DEVICES=2 python motion_generation/scripts/train_multipart_rvqvae.py \
  --config motion_generation/configs/causal_rvq_hands_512x4.yaml \
  --device cuda:0
```

GPU 3, feet:

```bash
CUDA_VISIBLE_DEVICES=3 python motion_generation/scripts/train_multipart_rvqvae.py \
  --config motion_generation/configs/causal_rvq_feet_512x4.yaml \
  --device cuda:0
```

Inside each process the selected physical GPU is exposed as logical `cuda:0`.
To reduce memory, append for example
`--per_device_train_batch_size 48 --per_device_eval_batch_size 48`. To use
offline W&B logging, append `--wandb_mode offline`.

Expected best checkpoints:

```text
checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth
checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth
checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth
checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth
```

To resume an interrupted part, repeat its command and append
`--resume checkpoints/causal_multipart_rvqvae/<experiment>/model/latest.pth`.

## Mandatory post-training causality audit

Run this before exporting any dataset tokens:

```bash
python motion_generation/scripts/audit_causal_body_codecs.py \
  --upper_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth \
  --lower_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth \
  --feet_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth \
  --hands_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth \
  --device cuda:0 --max_clips 8 --max_source_frames 128 \
  --output_json checkpoints/causal_multipart_rvqvae/causality_audit.json
```

The command exits nonzero if any of these contracts fail:

1. Encoding a prefix differs from the corresponding full-sequence latent or
   token prefix.
2. Perturbing future motion changes an earlier latent or token ID.
3. Decoding a token prefix differs from the same frames of a full decode.
4. Perturbing future token IDs changes earlier decoded motion.
5. Repeated prefix streaming differs from one-shot encoding or decoding.
6. Token and decoded lengths violate the completed-chunk convention.

## Reconstruction and codebook evaluation

Use `motion_generation/notebooks/compare_rvqvae_codecs_metrics.py`. Set
`RVQ_UPPER_CKPT`, `RVQ_LOWER_CKPT`, `RVQ_FEET_CKPT`, and `RVQ_HANDS_CKPT` to
the four causal checkpoints. The loader now reconstructs causal checkpoints
from their saved metadata.

At minimum, compare the new causal codecs against the current noncausal
multipart reference on the validation split:

- normalized and physical reconstruction RMSE/MAE
- velocity and acceleration RMSE
- 6D rotation geodesic error
- root trajectory drift
- hands reconstruction metrics
- active codes, dead-code fraction, and perplexity for every part and RVQ level
- evaluator FID/diversity and semantic retrieval when the evaluator environment
  is available

Do not select checkpoints using the test split. Inspect per-level code usage;
an average perplexity alone can hide collapse in a later RVQ level.

## Token export after acceptance

```bash
python motion_generation/scripts/export_multipart_motion_tokens.py \
  --upper_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_upper_512x4_scratch/model/best.pth \
  --lower_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_lower_512x4_scratch/model/best.pth \
  --feet_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_feet_512x4_scratch/model/best.pth \
  --hands_ckpt checkpoints/causal_multipart_rvqvae/causal_rvq_hands_512x4_scratch/model/best.pth \
  --split_file SuSuInterActs/SuSuInterActs/split/all_file_list.txt \
  --output_dir SuSuInterActs/SuSuInterActs/motion_token_data_causal_multipart_512x4 \
  --device cuda:0
```

The export manifest must report `body_causal: true` and all four entries in
`causal_by_part` must be true. Keep the old token directory unchanged. Step 2
must be retrained on this new export; its existing HuBERT features and model
architecture may remain unchanged.

## Phase 0 completion gate

Phase 0 is complete only when all four codecs have usable validation
reconstruction, healthy per-level codebook usage, a passing strict-causality
audit, and a complete new token export. The old frozen Step 2 is not compatible
with these newly learned codebook meanings.

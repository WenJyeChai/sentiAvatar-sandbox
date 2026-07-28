# Step 1 Stage 1: Supplied-Gap Anchor-CE Pretraining

## Purpose

Stage 1 learns only the conditional anchor model:

\[
p(a_{k+1}\mid \text{text},\text{audio},a_{\le k}^{GT},g_k)
\]

The dataset supplies \(g_k\). The model receives no target or loss for choosing
that gap. Every target anchor contains 16 multipart body RVQ IDs, and anchor CE
is the sole planner objective.

## Data contract

Required:

- causal multipart body tokens in
  `SuSuInterActs/SuSuInterActs/motion_token_data_multipart_causal_512x4`
- MOSS Nano token files in
  `SuSuInterActs/SuSuInterActs/audio_tokens_moss_nano_48k_12p5hz_16cb`
- structured text source
  `SuSuInterActs/SuSuInterActs/text_data/motion2text.json`
- train and validation split files

Not required:

- frozen Step 2 checkpoints or audio features
- Step 2 interval-cost caches
- adaptive-gap calibration or DP schedule files
- generated-anchor rollout caches

## Schedule contract

- every normal gap is sampled uniformly from 3 through 15
- the random schedule is resampled for each training clip each epoch
- validation uses one deterministic fixed random schedule
- gaps 0 through 2 are allowed only for the final EOS interval
- supplied gap tokens are inputs and have no target mask
- previous anchors are ground truth

## Configuration

`motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab.yaml`

Important settings:

```yaml
provided_gap_training:
  enabled: true
  distribution: uniform
  min_gap: 3
  max_gap: 15
  resample_each_epoch: true

adaptive_gap:
  enabled: false

training:
  num_train_epochs: 100
  learning_rate: 2.0e-5
  warmup_ratio: 0.03
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
```

On four GPUs, the effective global batch is 128.

## Preflight

Small preflight:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab.yaml \
  --max_train_clips 32 \
  --max_eval_clips 32 \
  --output_json checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab/data_preflight_smoke.json
```

Full preflight:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab.yaml \
  --output_json checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab/data_preflight.json
```

The report must show:

- `provided_gap_training.enabled: true`
- gap counts only in 3--15, apart from EOS tails 0--2
- no failed or truncated records
- ordinary audio token positions equal four times the Nano frame count

## One-GPU smoke training

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab.yaml \
  --max_train_clips 64 \
  --max_eval_clips 32 \
  --max_train_steps 2 \
  --output_dir checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_smoke
```

The header must report:

```text
Stage 1 supplied-gap anchor-CE pretraining
Supplied gaps: distribution=uniform, range=3-15
Gap supervision: disabled
Generated history: enabled=False
Visited state: enabled=False
Frozen Step 2: enabled=False
```

Evaluation must report `gap_loss=0`. Because every example is clean and no
auxiliary objective is active:

```text
train/loss ~= train/cross_entropy ~= train_clean/loss
```

## Full four-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab.yaml
```

Do not pass `--init_from_checkpoint`; the configuration initializes from
`checkpoints/llm`.

If memory is insufficient, change the configuration to per-device batch 4 and
gradient accumulation 8 to preserve the global batch of 128.

## Matched causal-student ablation

The causal comparison uses:

`motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_causal.yaml`

Everything matches the full-audio run except:

```yaml
planner_context:
  attention_mode: causal
  sequence_layout: causal_interleaved
```

The complete text remains available at the start. Audio frames are inserted
chronologically after each supplied gap and before its target anchor, so anchor
\(k\) sees audio only through its own boundary.

Small preflight:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_causal.yaml \
  --max_train_clips 32 \
  --max_eval_clips 32 \
  --output_json checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_causal/data_preflight_smoke.json
```

One-GPU smoke training:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_causal.yaml \
  --max_train_clips 64 \
  --max_eval_clips 32 \
  --max_train_steps 2 \
  --output_dir checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_causal_smoke
```

Full four-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab_causal.yaml
```

Use the same validation clips and deterministic validation schedule when
comparing the two runs. Primary comparisons are validation anchor CE,
per-gap-bin CE, generated-anchor FID, and frozen-Step-2 performance under the
same externally supplied schedules.

## Portable teacher/causal evaluation

Use:

`motion_generation/notebooks/evaluate_step1_stage1_teacher_causal_step2.ipynb`

The notebook accepts any one or more local `LABEL=CHECKPOINT` selections. It
exports each checkpoint independently so the teacher and causal student do not
need to exist on the same server. Every `<label>/` result bundle contains:

- teacher-forced and generated-history Stage 1 metrics;
- the deterministic supplied gap schedule and its SHA-256 fingerprint;
- matched GT and generated-anchor rollout caches;
- frozen-Step-2 likelihood and decoded-motion metrics;
- anchor-substitution and Step-2-infilling FID inputs;
- complete decoded body/hand motions for synchronized rendering.

Copy the complete `<label>/` bundle from one server to the other and register
it in the notebook's `COPIED_BUNDLES` mapping. Comparison is rejected unless
the selected clip list, supplied schedule, frozen Step 2 weights, and all four
causal body codecs have matching fingerprints.

## Twenty-epoch generated-history post-training

The completed teacher checkpoint was trained for 50 clean-history epochs with
32 examples per GPU on four GPUs (global batch 128). Post-training starts a
new optimizer/scheduler from that checkpoint; it does not resume the old
optimizer state.

The generated-history configuration is:

`motion_generation/configs/step1_stage1_teacher_generated_history_posttrain20.yaml`

The curriculum keeps the supplied uniform 3--15 gaps and anchor CE objective:

| Epochs | Generated-history examples | History used by generated examples | Clean replay |
|---|---:|---|---:|
| 1--3 | 20% | coherent suffix with 1--2 generated previous anchors | 80% |
| 4--7 | 40% | coherent suffix with 1--4 generated previous anchors | 60% |
| 8--12 | 60% | half local suffix, half complete generated prefix | 40% |
| 13--20 | 60% | complete generated prefix from the known seed | 40% |

The target RVQ IDs remain ground truth. Generated IDs are produced greedily
under `torch.inference_mode()` and fed back as input context; the following
normal forward pass carries the anchor-CE gradients.

Run a one-GPU smoke test first:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_teacher_generated_history_posttrain20.yaml \
  --init_from_checkpoint checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab/best \
  --max_train_clips 64 \
  --max_eval_clips 32 \
  --num_train_epochs 1 \
  --max_train_steps 2 \
  --output_dir checkpoints/step1_stage1_teacher_generated_history_posttrain20_smoke
```

The header must show the four `History phase` lines, prefix-LM mode, and
`Generated history: enabled=True`. During epoch 1, logs must show
`p_gen=0.200`.

Run the full four-GPU post-train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29516 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_teacher_generated_history_posttrain20.yaml \
  --init_from_checkpoint checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab/best
```

Use `--init_from_checkpoint`, not `--resume_from_checkpoint`: this experiment
needs a fresh 20-epoch cosine schedule at `5e-6`.

The matched clean-continuation control is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29517 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_stage1_teacher_clean_control20.yaml \
  --init_from_checkpoint checkpoints/step1_stage1_anchor_ce_uniform_gap100_nano_q0q3_vocab/best
```

Both runs use per-device batch 32, gradient accumulation 1, and therefore the
same global batch 128 as the completed teacher pretraining. The prefix-LM
generated rollout is intentionally microbatched one clip at a time because
full-audio prefix lengths differ; it is substantially slower than the clean
control.

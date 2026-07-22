# Step 1 6K Loss-Only Condition Alignment

## Fixed decisions

- Step 1 pretraining is teacher-forced: every previous motion anchor is GT.
- Self-forcing is a separate weights-only fine-tuning stage.
- No conditioning adapter, projection head, or inference-time module is added.
- Expected distortion is archived and disabled in every active config.
- Text/audio reliance is trained and measured through correct-versus-corrupted
  target likelihood while targets and motion history remain fixed.

## Objective

For each selected example or causal audio interval:

```text
s_pos = mean log P(GT anchor IDs | correct condition, fixed motion history)
s_neg = mean log P(GT anchor IDs | corrupt condition, fixed motion history)

L_CF = softplus(margin - (s_pos - s_neg))
L    = L_CE + lambda_CF L_CF
```

`margin=0.05` nats/token and `lambda_CF=0.03` are initial screening
values. CE trains every example. The extra wrong-condition forward uses 25% of
each local batch and alternates text/audio by minibatch in the combined arm.

### Audio corruption

All Mimi q0-q3 streams for an eligible interval are replaced together by the
same clip's audio interval two anchors (about 0.8 seconds) in the past. Early
anchors without sufficient past context are excluded from this loss. Only
anchors whose audio was actually replaced are scored. Future audio is never
used.

### Text corruption

The transcript is replaced by a different transcript from the local batch.
Donor token IDs are deterministically resampled to the exact original text
span length. This preserves every motion/audio/control token position and
prevents sequence length or RoPE position shifts from solving the task. Exact
token-sequence duplicates are rejected.

## Pretraining arms

| Arm | Config | Training loss |
|---|---|---|
| P0 | `configs/step1_multipart_gap3_6k_pretrain_p0_ce.yaml` | CE |
| P1 | `configs/step1_multipart_gap3_6k_pretrain_p1_audio_cf.yaml` | CE + audio CF |
| P2 | `configs/step1_multipart_gap3_6k_pretrain_p2_text_cf.yaml` | CE + text CF |
| P3 | `configs/step1_multipart_gap3_6k_pretrain_p3_audio_text_cf.yaml` | CE + alternating audio/text CF |

Every arm:

- starts independently from `checkpoints/llm`;
- uses the same balanced 6,000 clips and full validation split;
- uses Mimi q0-q3 and fixed `[gap_3]` anchors;
- trains for 50 epochs with global batch 128;
- uses no generated-history inputs;
- evaluates both audio and text likelihood gaps, including P0;
- saves every 235 updates (approximately every five epochs).

Alignment training begins after five CE-only epochs and ramps over epochs 6-8.

## Recommended execution order

Run P0 first. Then run P1 and P2 separately. Run P3 only if at least one
single-modality arm improves its intended validation gap without unacceptable
CE/decoded-quality degradation.

### Data preflight

The serialized data are identical across arms, so one complete preflight is
sufficient:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_multipart_gap3_6k_pretrain_p0_ce.yaml \
  --output_json checkpoints/step1_gap3_6k_pretrain_p0_ce/data_preflight.json
```

### Counterfactual-path smoke test

Seven tiny epochs are used so the epoch-5 warm-up ends and the negative branch
actually runs. The output is isolated from the real experiment.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
WANDB_MODE=disabled torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_gap3_6k_pretrain_p3_audio_text_cf.yaml \
  --max_train_clips 256 \
  --max_eval_clips 32 \
  --num_train_epochs 7 \
  --output_dir checkpoints/smoke_step1_condition_alignment
```

At epoch 7, logs must contain nonzero:

```text
condition/weight
condition/counterfactual_loss
condition/gap_nats_per_token
```

### Full P0 command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_gap3_6k_pretrain_p0_ce.yaml
```

For P1/P2/P3, replace only the config and use a different free master port.

## Evaluation

Training reports the teacher-forced quantities:

```text
eval_condition/audio_gap_nats_per_token
eval_condition/text_gap_nats_per_token
```

A positive gap means the wrong condition gives the GT anchors higher NLL, as
desired. Compare checkpoints with the same corruption policy; absolute values
from different negative construction are not comparable.

Evaluate both GT history and one fixed generated prefix on the complete 635
validation clips:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  motion_generation/scripts/evaluate_step1_condition_alignment.py \
  --checkpoint checkpoints/step1_gap3_6k_pretrain_p1_audio_cf/best \
  --max_clips 0 \
  --batch_size 8 \
  --output_json motion_generation/outputs/p1_audio_cf_condition_alignment.json
```

Generated histories are produced once under the correct condition, frozen,
and reused for correct and corrupt likelihood. Thus the generated-prefix gap
cannot be explained by different histories.

Do not select a winner from condition gap alone. Require:

- intended gap improves beyond P0;
- validation CE/top-1 does not materially regress;
- full generated rollout remains stable;
- anchor-substitution decoded error/FID does not materially regress;
- later, Step 2 output also benefits.

## Separate self-forcing fine-tune

After choosing a teacher-forced checkpoint, initialize a fresh optimizer and
fine-tune it with generated histories. Do not use `--resume_from_checkpoint`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29518 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_gap3_6k_finetune_self_forcing_cf.yaml \
  --init_from_checkpoint checkpoints/step1_gap3_6k_pretrain_p3_audio_text_cf/best
```

The fine-tune starts generated-history probability near 0.25 and reaches 0.5
at epoch 2. It uses ten epochs, LR `5e-6`, and retains the condition likelihood
loss so generated motion history does not erase conditioning reliance.

## Interpretation

- Retrieval/InfoNCE heads are deliberately deferred. This experiment asks the
  more direct question: does the existing generator assign higher probability
  to the same GT gesture under the correct condition?
- No model architecture changes at inference.
- P0 measures how much conditioning sensitivity CE learns naturally.
- P1/P2 isolate local causal audio and global untimed text.
- P3 tests coexistence only after isolated evidence.
- The stopped expected-distortion/self-forcing run remains an exploratory
  ablation and is not a canonical pretraining checkpoint.


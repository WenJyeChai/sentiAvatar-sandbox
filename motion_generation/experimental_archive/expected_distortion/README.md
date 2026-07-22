# Archived Step 1: 6K CE + Codec Expected-Distortion Experiment

This experiment is intentionally inactive while the loss-only condition-alignment
suite is evaluated. Its implementation is retained for reproducibility.

## Objective

Test whether frozen causal-RVQ codebook geometry improves generated anchors
over categorical cross-entropy alone. This is a matched, from-Qwen experiment
on the balanced 6,000-clip training split.

"From scratch" here means that each arm independently initializes from
`checkpoints/llm`, then creates a fresh Step 1 planner, optimizer, cosine
scheduler, and epoch counter. Neither arm loads a trained Step 1 checkpoint,
including `self_forcing_best_rollout`.

## Matched arms

The arms differ only in `auxiliary_loss` and their output/monitoring names:

| Arm | Config | Training objective |
|---|---|---|
| Control | `experimental_archive/expected_distortion/step1_multipart_gap3_6k_scratch_ce_control.yaml` | CE |
| Treatment | `experimental_archive/expected_distortion/step1_multipart_gap3_6k_scratch_expected_distortion.yaml` | CE + expected distortion |

Both use:

- the balanced `train_step1_main_6000.txt` split;
- causal Mimi q0-q3 audio;
- fixed `[gap_3]` anchors and all 16 multipart RVQ targets;
- per-device batch size 32 on four GPUs (global batch 128);
- 50 epochs, AdamW, LR `2e-5`, and cosine decay;
- five teacher-forced epochs, a rollout-quality gate, then a ten-epoch cosine
  ramp to generated-history probability 0.5;
- the same validation split, seed, ordering rules, and rollout evaluations.

## Treatment loss

For slot `s`, GT code `y`, frozen codebook entries `e`, and planner
distribution `p`, the auxiliary is:

```text
D_s(y, k) = mean_d((e_y - e_k)^2) / mean_all_pairs_distance_s
L_ED       = mean_tokens sum_k p(k) D_s(y, k)
L_total    = L_CE + lambda(t) L_ED
```

The distance normalization gives an average random-pair cost close to one for
every part and RVQ level. `lambda(t)` ramps from 0 to 0.1 during epoch 1 and
then remains 0.1. Codecs are frozen; the distance table is a nonpersistent
model buffer and receives no gradients.

CE remains active on all training rows. In this first experiment, expected
distortion applies only to GT-history (clean) rows. Once generated q0 differs
from GT, the original q1-q3 residual IDs are not necessarily the canonical
residual decomposition for that generated prefix, so applying codec geometry
there would mix two effects.

## 1. Audit codec geometry first

Run on one GPU. The script samples local windows from the 6K split, substitutes
near-to-far codes at each of the 16 slots, decodes them with the causal codec,
and correlates embedding distance with decoded feature/geodesic damage.

```bash
CUDA_VISIBLE_DEVICES=0 python \
  motion_generation/experimental_archive/expected_distortion/audit_step1_codebook_geometry.py \
  --device cuda:0 \
  --samples_per_slot 32 \
  --alternatives_per_sample 8 \
  --output_json checkpoints/step1_codebook_geometry_audit_6k.json
```

The default exploratory gate reports `GO` when every slot has RMSE Spearman
rho at least 0.20 and the median across slots is at least 0.50. `NO_GO` is a
valid audit result rather than a crashed command. Inspect individual slots;
do not hide a failing RVQ level behind the aggregate median.

## 2. Data preflight

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/experimental_archive/expected_distortion/step1_multipart_gap3_6k_scratch_ce_control.yaml \
  --output_json checkpoints/step1_multipart_gap3_6k_scratch_ce/data_preflight.json

python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/experimental_archive/expected_distortion/step1_multipart_gap3_6k_scratch_expected_distortion.yaml \
  --output_json checkpoints/step1_multipart_gap3_6k_scratch_expected_distortion/data_preflight.json
```

Both must report 6,000 assigned/valid training clips and zero errors. The
treatment additionally checks codec files when training starts.

## 3. Train the matched arms

Run these sequentially because each command consumes all four GPUs. Do not add
`--resume_from_checkpoint`.

Control:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/experimental_archive/expected_distortion/step1_multipart_gap3_6k_scratch_ce_control.yaml
```

Treatment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/experimental_archive/expected_distortion/step1_multipart_gap3_6k_scratch_expected_distortion.yaml
```

The treatment logs `train/cross_entropy`,
`auxiliary/expected_distortion`, and `auxiliary/weight` separately from total
`train/loss`. A falling auxiliary value is not sufficient evidence: selection
still depends on held-out rollout and decoded-motion evaluation.

## 4. Compare checkpoints

Start with each run's `best_rollout` checkpoint. Compare `best` and `final`
only as secondary checkpoint-selection diagnostics.

```bash
CUDA_VISIBLE_DEVICES=0 python \
  motion_generation/scripts/evaluate_step1_multipart_comparison.py \
  --checkpoint ce_6k_scratch=checkpoints/step1_multipart_gap3_6k_scratch_ce/best_rollout \
  --checkpoint ed_6k_scratch=checkpoints/step1_multipart_gap3_6k_scratch_expected_distortion/best_rollout \
  --output_dir motion_generation/outputs/step1_6k_expected_distortion_comparison \
  --teacher_max_clips 0 \
  --rollout_max_clips 0 \
  --rollout_batch_size 8 \
  --subset_seed 42 \
  --write_rollout_cache
```

Then run the existing anchor-substitution evaluation on the common 635-clip
validation set. The treatment is promising only if it improves generated
rollout decoded metrics/FID without materially worsening teacher-forced CE,
rollout stability, or diversity. Exact token accuracy alone is not the primary
decision metric for this experiment.

## Stop/go interpretation

- **Geometry NO_GO:** do not treat this codebook distance as a perceptual loss;
  redesign the metric (for example, decoded local damage lookup or learned
  cost) before the 50-epoch treatment.
- **Geometry GO, treatment loses to matched CE:** reject lambda 0.1 or this
  metric; do not compare only against the older full-data checkpoint.
- **Treatment improves only teacher-forced metrics:** exposure behavior did not
  improve; it is not a successful anchor-planning result.
- **Treatment improves full rollout and decoded/FID metrics:** proceed to a
  small lambda ablation around the observed gradient scale before changing
  Step 2.

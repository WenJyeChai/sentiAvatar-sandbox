# Step 1 Fixed-Gap Planner Evaluation Report

**Evaluation date:** 2026-07-21  
**Status:** teacher-forced learning confirmed; generated-prefix reliability not yet acceptable  
**Selected checkpoint for further diagnosis:** `checkpoints/step1_multipart_fixed_gap3_main6000/final`

## 1. Training contract

The evaluated planner was trained with:

- 6,000 clips from the official training split;
- the complete 635-clip validation split for model selection;
- fixed `[gap_3]` scheduling (one anchor every four 10 Hz motion-token positions, plus the exact tail);
- causal Mimi q0 conditioning at 12.5 Hz;
- complete text known at utterance start;
- 16 body IDs per anchor: upper/lower/feet/hands × q0-q3;
- `mixed_known` seeds;
- GT previous-anchor prefixes only (`generated_prefix_probability: 0.0`);
- equal 512-way CE over all 16 anchor slots;
- 50 configured epochs, 32 clips/GPU on four GPUs, no gradient accumulation; and
- effective global batch size 128.

The historical run stopped after epoch 44 because its remote configuration still used early stopping. The checked-in configuration now disables early stopping for future runs.

## 2. Evaluation protocol and validity

The evaluation used `motion_generation/notebooks/evaluate_step1_fixed_gap3_planner.ipynb` and did not touch the test split.

| Evaluation | Clips | Anchors | Status |
|---|---:|---:|---|
| Teacher-forced best/final comparison | 635 | 10,024 | Valid |
| Uniform, train-unigram, previous-anchor baselines | 635 | 10,024 | Valid |
| Teacher-forced condition shuffle | 128 | 1,975 | Valid paired sensitivity test |
| Greedy generated-prefix rollout | 128 | 1,975 | Valid pilot; not full validation |
| Generated-prefix condition shuffle | 32 | 471 | Exploratory due to small subset |
| Codec oracle-gap anchor substitution | 32 | mean 14.72/clip | Decoded metrics valid; original latent column invalid |

Important limitations:

1. The supplied 128- and 32-clip results used the first clips in validation order, not a deterministic random or balanced subset. The notebook has since been corrected to use deterministic random subsets and to compute persistence baselines on the exact rollout subset. A final report should use all 635 clips for generated-prefix rollout.
2. `previous_gt_anchor_copy` has no CE/perplexity because it is a deterministic hard predictor rather than a probability distribution. Its accuracy is valid.
3. The original codec table incorrectly repeated `predicted_anchor_latent_rmse` on the copy-baseline rows. Ignore that column in the supplied run. The decoded feature, rotation, velocity, and root metrics are independently computed and valid. The helper now calculates a separate `anchor_latent_rmse` for each variant.
4. Text/Mimi shuffling is an evaluation-time sensitivity test, not a retrained ablation. It supports conclusions about reliance but cannot isolate the maximum value of either modality.
5. Codec decoding retains GT tokens at all non-anchor positions. It isolates anchor substitution damage under an oracle gap; it is not Step 2 or end-to-end generated motion.

## 3. Reference baselines

The full validation set contains 10,024 target anchors and 160,384 supervised IDs.

| Predictor | Overall | q0 | q1 | q2 | q3 | CE | Perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform reference | 0.195% | 0.195% | 0.195% | 0.195% | 0.195% | 6.2383 | 512.0 |
| Train-unigram majority | 0.904% | 1.107% | 0.950% | 0.798% | 0.758% | 5.8150 | 335.3 |
| Previous GT anchor copy | **11.295%** | **27.497%** | **10.041%** | **4.861%** | **2.781%** | N/A | N/A |

Persistence is a much stronger baseline than uniform or token frequency. Any useful online planner must be compared against copying the most recent known anchor.

## 4. Teacher-forced checkpoint results

| Checkpoint | Accuracy | Top-5 | CE | Perplexity | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `final` | **7.423%** | **18.723%** | **4.96946** | **143.95** | **19.421%** | **5.774%** | 2.733% | **1.766%** |
| `best` | 7.397% | 18.721% | 4.96965 | 143.98 | 19.351% | 5.741% | **2.738%** | 1.758% |

`final` is marginally better overall and is the preferred checkpoint, although the difference is too small to be practically important.

The strongest teacher-forced slots were `feet_q0` (23.20%) and `lower_q0` (22.27%). The weakest was `upper_q3` (0.88%). Later RVQ levels remain difficult, as expected, but all slots exceed uniform chance.

### Interpretation

- The planner learned substantially more than token frequency: 7.42% versus 0.90% accuracy and CE 4.97 versus 5.81.
- It did **not** beat simple persistence: 7.42% versus 11.29% overall.
- q0 also lost to persistence: 19.42% versus 27.50%.
- Teacher-forced performance therefore demonstrates representation learning, not a superior anchor predictor.

## 5. Conditioning sensitivity under teacher forcing

The paired 128-clip subset produced:

| Condition | Accuracy | CE | Δ accuracy | Δ CE |
|---|---:|---:|---:|---:|
| Full | 7.851% | 4.92198 | — | — |
| Shuffled text | 7.427% | 4.95525 | -0.424 pp | +0.03328 |
| Shuffled Mimi q0 | 7.674% | 4.92751 | -0.177 pp | +0.00554 |

Text has a measurable but modest effect. Mimi q0 has a much smaller effect. Because GT motion prefixes remain present, this test suggests that the model relies mainly on motion history and uses text more than audio.

This does not mean q0 contains no useful information. It means the trained planner can largely minimize teacher-forced CE without depending strongly on it.

## 6. Generated-prefix rollout

Greedy rollout retained the known seed but generated every subsequent anchor and kept generated anchors in the causal prefix.

| Metric | Teacher forced, same subset | Generated prefix | Change |
|---|---:|---:|---:|
| Overall accuracy | 7.851% | **2.196%** | -5.655 pp (-72.0%) |
| q0 accuracy | 19.861% | **4.987%** | -14.874 pp (-74.9%) |
| q1 accuracy | 6.468% | 1.722% | — |
| q2 accuracy | 3.215% | 1.317% | — |
| q3 accuracy | 1.861% | 0.760% | — |

Generated-prefix accuracy remains above uniform (0.195%) and unigram accuracy (0.904%), but it is far below the full-validation previous-anchor-copy baseline (11.295%). The corrected notebook will compute the copy result on the exact same rollout subset; this is not expected to reverse the large gap.

### Horizon behavior

| Relative horizon | Overall | q0 | Confidence | Entropy |
|---|---:|---:|---:|---:|
| 0-25% | 2.795% | 7.421% | 10.57% | 4.620 |
| 25-50% | 2.114% | 4.412% | 15.37% | 4.344 |
| 50-75% | 1.840% | 3.652% | 16.36% | 4.292 |
| 75-100% | 1.954% | 4.126% | 16.48% | 4.281 |

Accuracy degrades after the first quarter while confidence rises and entropy falls. This is the signature of self-reinforcing exposure bias: after consuming its own wrong anchors, the model becomes more certain rather than recovering. Mean confidence over the rollout was 14.57% while exact accuracy was only 2.20%, indicating severe overconfidence. The corrected evaluator now records 10-bin ECE and a top-1 Brier score.

## 7. Conditioning sensitivity during generated rollout

The 32-clip result is exploratory because the subset is small:

| Condition | Overall | q0 | Δ accuracy |
|---|---:|---:|---:|
| Full | 1.234% | 2.813% | — |
| Shuffled text | 1.035% | 1.964% | -0.199 pp |
| Shuffled Mimi q0 | 1.234% | 2.813% | 0.000 pp |

Text retains a small influence. Shuffling Mimi did not change overall or q0 top-1 accuracy on these clips. Combined with the teacher-forced shuffle result, the current model appears to underuse causal audio, especially after rollout errors begin.

## 8. Codec-space oracle-gap diagnostics

The following table compares predicted-anchor substitution against copying the previous GT anchor while retaining GT non-anchor tokens.

| Part | Feature RMSE predicted | Feature RMSE copy | Mean geodesic predicted | Mean geodesic copy |
|---|---:|---:|---:|---:|
| Feet | 0.01970 | **0.01129** | 1.434° | **0.718°** |
| Hands | 0.06430 | **0.03273** | 3.974° | **1.554°** |
| Lower | 0.04805 | **0.03112** | 3.049° | **1.242°** |
| Upper | 0.07448 | **0.03191** | 4.996° | **2.072°** |

For lower-body root trajectory:

| Variant | Mean trajectory error | Final drift |
|---|---:|---:|
| Predicted anchors | 2.646 | 3.973 |
| Previous-anchor copy | **0.985** | **1.027** |

Copying is better for every decoded part and every reported error family. Predicted anchors approximately double mean rotation error for feet and more than double it for hands, lower body, and upper body. Lower-root final drift is about 3.9× larger.

The original latent-RMSE comparison is excluded until the corrected codec cell is rerun.

## 9. Decision

### What passed

- Data and causal serialization are valid.
- Both checkpoints learned well beyond uniform and unigram baselines.
- `final` is marginally the stronger teacher-forced checkpoint.
- Text has measurable predictive value.
- KV-cached autoregressive decoding was unit-tested against full-prefix recomputation.

### What failed

- Teacher-forced accuracy does not beat previous-anchor copying.
- Generated-prefix rollout loses roughly 72% of teacher-forced accuracy.
- Errors compound over time while confidence increases.
- Mimi q0 sensitivity is weak and disappears in the small free-running ablation.
- Predicted anchors cause substantially more codec-space damage than copying.

**Current decision:** the checkpoint is a useful teacher-forced baseline but is **not yet a reliable online anchor planner**. Do not treat its greedy anchors as better than persistence, and do not start adaptive-gap training on the assumption that Step 1 is solved.

Frozen Step 2 evaluation may still be run as a diagnostic, but it must compare at least GT anchors, predicted anchors, and previous-anchor-copy anchors. Based on the standalone results, predicted anchors are unlikely to beat the copy baseline without additional training.

## 10. Required next work

1. Rerun the corrected notebook on a deterministic random pilot, then set `ROLLOUT_MAX_CLIPS = None` for all 635 validation clips.
2. Regenerate the codec section to obtain valid per-variant `anchor_latent_rmse`.
3. Produce generated-prefix caches for the 6,000 training clips using `final`.
4. Train a scheduled-sampling curriculum, beginning with a modest generated-prefix probability and increasing it only when validation rollout improves.
5. Add previous-anchor corruption/dropout so the model cannot minimize CE by relying almost entirely on perfect motion history.
6. Investigate a persistence-aware decoder or copy gate: the learned planner should change an anchor only when it can beat holding the previous state.
7. Strengthen audio/text reliance with controlled condition dropout or contrastive mismatching, then repeat the shuffle tests.
8. Require generated-prefix accuracy and decoded oracle-gap metrics to beat previous-anchor copying before moving to adaptive schedules.
9. Only then evaluate frozen Step 2 infilling and generated-prefix curriculum variants on the untouched test split.

## 11. Reproducibility artifacts

- Notebook: `motion_generation/notebooks/evaluate_step1_fixed_gap3_planner.ipynb`
- Evaluation helpers: `motion_generation/utils/step1_planner_evaluation.py`
- Evaluation tests: `motion_generation/scripts/test_step1_planner_evaluation.py`
- Machine-readable outputs: `motion_generation/outputs/step1_fixed_gap3_evaluation/`
- Planner checkpoint: `checkpoints/step1_multipart_fixed_gap3_main6000/final`


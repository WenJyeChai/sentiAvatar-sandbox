# SentiAvatar Step 2: Multipart Variable-Gap C2F Handoff

**Status date:** 2026-07-19

**Purpose:** self-contained technical handoff for continuing the project in a new conversation.
**Current recommendation:** treat the face-enabled Stage 2 checkpoint as the current combined Step 2 model, retain the body-only Stage 2 checkpoint as the strongest body reference, and stop changing Step 2 unless a final controlled evaluation exposes a concrete defect.

---

## 1. What Step 2 Does

Step 2 is a **bidirectional audio-conditioned motion infiller**. It receives:

- a left motion-token anchor;
- a right motion-token anchor;
- HuBERT audio features aligned to both anchors and the missing interval; and
- mask tokens for every missing motion-token frame.

It predicts the missing interval noncausally. This is not the causal Step 2 model and is not the old fixed-three-frame `generate_sbs` setup.

The selected system combines four ideas:

1. Independent multipart 512x4 RVQ codecs.
2. Variable temporal gaps from 1 through 15 codec frames.
3. Coarse-to-fine generation over RVQ levels `q0 -> q1 -> q2 -> q3`.
4. A second training stage using self-generated RVQ prefixes and **soft residual-recovery supervision**.

The latest combined model additionally appends face as a fifth RVQ stream.

### 1.1 Completed outcomes

The Step 2 work has reached the following concrete state:

- diagnosed and normalized the mixed root-channel schemas in the released motion data;
- replaced the unusable old motion codec with independently trained upper/lower/feet/hands 512x4 RVQ codecs;
- verified that the new codec preserves evaluator motion and semantic retrieval far better than the old codec;
- replaced fixed three-frame infilling with clip-balanced variable-gap training over gaps 1-15;
- implemented q0-to-q3 coarse-to-fine generation for all parts;
- implemented generated-prefix training to expose later RVQ stages to inference-time context;
- audited hard residual recoverability and identified why hard adaptive relabeling can hurt distributional motion quality;
- developed soft residual recovery while retaining canonical CE supervision;
- established that soft recovery primarily improves medium/long gaps rather than gap 3;
- evaluated token, latent, decoded-motion, FID, diversity, retrieval, seam, and beat behavior;
- trained a separate 512x4 face codec and appended face as a fifth RVQ stream;
- completed face Stage 1 and Stage 2 training and showed that Stage 2 recovers most of the body and face degradation;
- added full-clip body and schematic ARKit visualization, including a body-only fallback for clips without face data;
- rejected several higher-complexity audio branches that did not justify their parameter or engineering cost.

---

## 2. Dataset Contract and Preprocessing

### 2.1 Raw rates and representations

- Raw body and hand motion: 20 FPS.
- Body pose: root channel plus 25 body joints in 6D rotation.
- Hands: left and right hand 6D rotations.
- Face: 51 ARKit blendshape coefficients at 20 FPS where available.
- HuBERT layer-9 features: 768 dimensions at 10 FPS.
- Multipart codec temporal downsampling: `unit_length = down_t * stride_t = 2`.
- Motion tokens therefore run at 10 FPS and align one-to-one with the stored HuBERT features.
- A gap of 1-15 token frames corresponds to approximately 0.1-1.5 seconds of missing motion.

### 2.2 Root-channel split discovered in the release

The released body data contains two incompatible root conventions:

- roughly 12,400 older clips store frame displacement/velocity in `body[:, :3]`;
- roughly 8,800 newer clips store absolute root/pelvis position, including a pelvis height near 101 cm;
- the pipeline cutover is around 2025-09-17.

This is upstream in the release, not local corruption.

`motion_generation/utils/multipart_motion.py` handles this with:

- `classify_root_channel()`;
- `canonicalize_body_root()`; and
- `split_motion_parts()`.

The codec contract is always root-frame delta. Absolute-root clips are differenced before normalization and codec training. The first delta frame is set to zero.

**Evaluation caveat:** the historical full-clip FID protocol used `canonicalize_raw_root=False` to remain directly comparable with earlier numbers. A root-canonicalized rerun would be a different protocol and must be reported separately.

### 2.3 Multipart feature layout

| Part | Features | Dimension |
|---|---|---:|
| `upper` | 16 upper-body joints x 6D | 96 |
| `lower` | root delta (3) + 5 lower-body joints x 6D | 33 |
| `feet` | 4 foot/ball joints x 6D | 24 |
| `hands` | 19 left + 19 right non-wrist joints x 6D | 228 |
| `face` | ARKit blendshape coefficients | 51 |

The two hands are deliberately combined into one hands stream. There are not separate left/right hand codebooks.

### 2.4 Face-data availability

Face is not present for the complete motion release.

- Complete split census: 21,133 names.
- Face-token export reported 12,367 exported and 8,766 missing.
- The face-available validation intersection used for final face evaluation contains 372 clips, compared with 635 body validation clips.

Missing raw ARKit implies missing face RVQ tokens. It cannot be repaired by rerunning token export. Do not substitute zero facial coefficients for quantitative comparison.

---

## 3. Multipart RVQ Codec Stack

### 3.1 Selected codec design

Each part has an independent CNN RVQ-VAE:

- codebook size: 512;
- residual quantizers: 4;
- latent/code dimension: 512;
- temporal downsampling: 2;
- CNN width: 512;
- depth: 3;
- EMA decay `mu`: 0.99;
- independent codebooks across both body parts and RVQ levels.

For one part, the latent approximation is:

\[
\hat z = e_0[k_0] + e_1[k_1] + e_2[k_2] + e_3[k_3].
\]

The codec implementation is `motion_generation/models/multipart_rvqvae.py`. Training is in `motion_generation/scripts/train_multipart_rvqvae.py`.

### 3.2 Codec checkpoints

Body:

```text
checkpoints/multipart_rvqvae/rvq_upper_512x4_bs256_cosine/model/best.pth
checkpoints/multipart_rvqvae/rvq_lower_512x4_bs256_cosine/model/best.pth
checkpoints/multipart_rvqvae/rvq_feet_512x4_bs256_cosine/model/best.pth
checkpoints/multipart_rvqvae/rvq_hands_512x4_bs256_cosine/model/best.pth
```

Face:

```text
checkpoints/multipart_rvqvae/rvq_face_512x4_bs256_cosine/model/best.pth
```

The repository face-codec YAML currently records 100 epochs. If the final run overrode this to 200 from the CLI, the checkpoint's saved `args` are authoritative and this discrepancy should be resolved before publication.

### 3.3 Codec training objective

For each selected part:

\[
L_{codec} = L_{rec} + L_{vel} + 0.02L_{commit}.
\]

The stored configuration uses cosine learning-rate decay, 3% warmup, BF16, gradient clipping at 1.0, and batch size 256 per device.

### 3.4 Why retraining the codec was necessary

The old paper codec was not adequate for evaluating improvements to the infiller. Reported test-set evaluator results were:

| Reconstruction source | FID normalized by GT | FID raw | Diversity gen | Clips |
|---|---:|---:|---:|---:|
| Old codec | 9.5223 | 7.7056 | 22.2654 | 1,455 |
| Multipart codec GT | 0.9905 | 0.7706 | 22.7542 | 1,479 |

Semantic retrieval was nearly preserved by the new codec:

| Source | R@1 | R@2 | R@3 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|
| Real GT | 61.73 | 72.95 | 78.97 | 84.52 | 90.33 |
| Multipart codec GT | 59.77 | 73.63 | 78.57 | 83.98 | 90.06 |

The codec is enabling infrastructure, not the intended primary novelty.

---

## 4. Motion-Token Representation

### 4.1 Slots per frame

The token order is part-major and RVQ-level-minor.

Body-only, 16 slots per token frame:

```text
upper q0 q1 q2 q3
lower q0 q1 q2 q3
feet  q0 q1 q2 q3
hands q0 q1 q2 q3
```

Body+face, 20 slots per token frame:

```text
upper q0 q1 q2 q3
lower q0 q1 q2 q3
feet  q0 q1 q2 q3
hands q0 q1 q2 q3
face  q0 q1 q2 q3
```

### 4.2 Global vocabulary

Each local code ID in `[0, 511]` is offset by `slot * 512`.

- Body vocabulary: `16 * 512 + 1 = 8,193`; mask ID = 8,192.
- Body+face vocabulary: `20 * 512 + 1 = 10,241`; mask ID = 10,240.

The final models constrain every slot's logits to its own 512-code interval. This prevents a token assigned to one part/RVQ slot from selecting another slot's vocabulary block.

Token export is implemented in `motion_generation/scripts/export_multipart_motion_tokens.py`.

---

## 5. Current Step 2 Transformer

Implementation: `motion_generation/models/audio_motion_model.py`.

### 5.1 Core architecture

| Component | Setting |
|---|---:|
| Transformer type | Bidirectional `nn.TransformerEncoder` |
| Encoder layers | 8 |
| Hidden width | 512 |
| Attention heads | 16 |
| FFN width | 1,536 |
| Activation | GELU |
| Dropout | 0.2 |
| Positional table | 512 positions, sinusoidal initialization then trainable |
| Input normalization | RMSNorm |
| Output | shared linear head over global token vocabulary |

There is no causal mask. Padding is excluded with a key-padding mask.

Parameter counts computed from the current implementation:

| Layout | Tokens/frame | Vocabulary | Parameters |
|---|---:|---:|---:|
| Body | 16 | 8,193 | 30,331,392 |
| Body+face | 20 | 10,241 | 32,428,544 |

Appending face adds 2,097,152 parameters, almost entirely from the enlarged token embedding and output head.

### 5.2 Input sequence lengths

For gap `g`, the model receives `g + 2` token frames: left anchor, `g` masked frames, and right anchor.

- Body maximum: `17 * 16 = 272` tokens.
- Body+face maximum: `17 * 20 = 340` tokens.
- Both fit under the 512-position table.

### 5.3 Audio conditioning retained in the final model

The selected mode is `legacy_additive`.

HuBERT features are projected through:

```text
Linear(768, 512)
LayerNorm
GELU
Dropout
Linear(512, 512)
LayerNorm
```

One projected audio vector is repeated across all 16 or 20 tokens belonging to the same token frame and added to the token-plus-position representation:

\[
h^0_{t,s} = E_{token}(x_{t,s}) + E_{pos}(tS+s) + A(a_t).
\]

Here `S` is slots per frame. In `legacy_additive` mode there are no active explicit part, RVQ-level, or C2F-stage embeddings. Slot identity is encoded by the global token ID range and absolute flattened position.

### 5.4 Audio branches that are not active

- The large audio residual-posterior branch was removed after adding about 88.4% trainable parameters for only small, inconsistent gains.
- L3/L6 temporal audio cross-attention adapters were removed.
- Routed additive audio remains implemented as an optional mode but is not used by the selected checkpoints.
- The final active model uses only input-level legacy additive HuBERT conditioning.

---

## 6. Variable-Gap Dataset and Batching

Implementation: `VariableGapMaskDataset`, `EpochResamplingLengthGroupedSampler`, and `VariableGapC2FCollator` in `motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py`.

### 6.1 Sampling

- Every eligible clip is used.
- Up to 8 unique windows are sampled per clip per training epoch.
- Windows are resampled at every epoch with a deterministic epoch-dependent seed.
- Validation uses 4 fixed windows per clip.
- Gap range: 1-15 token frames.
- Gap bucket probabilities: `[0.35, 0.35, 0.30]` for short, medium, and long buckets.
- Within each bucket, weights are divided by the number of available lengths so a bucket's probability is not dominated by having more integer gap lengths.
- Similar-length examples are grouped to reduce padding.

### 6.2 Collation

- Both anchors remain visible.
- Every slot in every interior frame is replaced by the global mask token.
- Variable windows are padded by complete token frames.
- The attention mask removes padded tokens.
- Audio is aligned by the ratio of audio FPS to token FPS and padded along the frame dimension.

### 6.3 Distributed training

The standard final setup is:

- 4 GPUs;
- 256 examples per GPU;
- gradient accumulation 1;
- effective batch size 1,024;
- BF16;
- gradient checkpointing;
- cosine learning-rate schedule;
- 3% warmup;
- max gradient norm 1.0.

The number of optimizer steps depends on the number of loaded clips and `train_windows_per_sequence`. The earlier 1,900-step run used only one window per clip; the final eight-window setup is intentionally much larger.

---

## 7. Coarse-to-Fine RVQ Generation

### 7.1 Inference

`generate_quantizer_coarse_to_fine()` performs four full transformer passes:

1. Predict q0 for every masked frame and every part.
2. Insert generated q0 tokens, then predict q1.
3. Insert generated q1 tokens, then predict q2.
4. Insert generated q2 tokens, then predict q3.

All parts at the same RVQ level are predicted in parallel. Temporal positions are also predicted in parallel within a stage. Only RVQ depth is sequential.

The old `generate_sbs()` confidence-filling path still exists for compatibility with old fixed-gap checkpoints but is not the selected decoder.

### 7.2 Training-stage sampling

Training does not execute all four stages for every optimizer batch. One stage is sampled using:

```text
q0: 0.35
q1: 0.25
q2: 0.20
q3: 0.20
```

Evaluation always performs the complete q0-to-q3 rollout and averages the four stage losses.

### 7.3 Why C2F was introduced

The MSD/error audit found that predictability differences were concentrated in coarse RVQ levels. Q1-to-Q4 MSD-stratum changes in teacher NLL were largest at q0/q1 and generally small at q3. The residual-recoverability audit also showed that later levels could substantially compensate for an imperfect coarse prefix when selected conditionally.

C2F by itself should not be presented as the main novelty; it is a natural way to respect residual quantization hierarchy.

---

## 8. Self-Forcing-Inspired Prefix Training

The implementation borrows the **principle** of conditioning on the model's own predictions, but it is not a faithful reproduction of the complete Self-Forcing paper objective or its distribution-matching method.

For a sampled stage `q > 0`:

- teacher-prefix batch: fill q0 through `q-1` with ground-truth tokens;
- self-forced batch: generate q0 through `q-1` greedily under `torch.no_grad()`, detach them, and use them as context for stage q.

The schedule in the selected Stage 2 configuration is:

- first 10% of total training: probability 0;
- next 30%: linear ramp;
- remaining training: probability 0.5;
- q0 never needs self-forcing because it has no RVQ prefix.

This addresses RVQ-level exposure bias: at inference q1-q3 condition on generated prefixes, whereas pure teacher-prefix training exposes them only to correct prefixes.

---

## 9. Soft Residual Recovery

This is the selected main Step 2 contribution.

### 9.1 Motivation

If q0 is wrong, the original q1-q3 labels may no longer be the best residual correction. Hard nearest-code relabeling reduced local latent discrepancy in audits but produced rare or off-manifold token combinations and worsened motion FID.

Soft recovery keeps the canonical token label as the main supervision and adds a distributional alternative only when the generated prefix is wrong.

### 9.2 Definitions

For one part, let the canonical target latent be:

\[
z^* = \sum_{r=0}^{3} e_r[k_r^*].
\]

At stage `q`, let the generated prefix latent be:

\[
p_q = \sum_{r<q} e_r[\hat k_r].
\]

The residual that the current level should ideally explain is:

\[
r_q = z^* - p_q.
\]

The implementation finds the `K=8` nearest vectors in codebook q to `r_q`. If their squared distances are `d_j`, it subtracts the nearest distance and estimates a local variance from the median shifted neighbor distance. The soft target is:

\[
w_j = \operatorname{softmax}\left(-\frac{d_j-d_{min}}{2\sigma^2}\right).
\]

The recovery loss is distributional cross-entropy over that local pool:

\[
L_{soft} = -\sum_{j \in \mathcal N_8(r_q)} w_j\log p_\theta(j\mid\text{generated prefix, anchors, audio}).
\]

### 9.3 When it is applied

In the selected Stage 2 settings:

- `adaptive_target_mode: never`;
- canonical CE labels are never replaced;
- soft recovery is active only on self-forced batches;
- it applies only to q1-q3;
- it applies only where the generated prefix differs from the canonical prefix;
- `soft_recovery_weight = 0.1`;
- `soft_recovery_topk = 8`;
- `soft_recovery_sigma_scale = 1.0`.

This distinction is essential: **soft recovery is not hard adaptive relabeling**.

---

## 10. Complete Training Loss

At a sampled stage q:

\[
L_q = L_{canonical\ CE}
+0.1L_{embedding}
+0.1L_{final\ latent}
+0.1L_{soft\ recovery}.
\]

Terms that do not apply at a stage are zero.

### 10.1 Canonical CE

The primary target remains the original codec ID `k_q*`, even under a generated prefix.

### 10.2 Expected-code embedding loss

For the local q-stage distribution `p(i)`, compute:

\[
\bar e_q = \sum_i p(i)e_q[i].
\]

The embedding loss is L1 distance between `bar(e_q)` and the current hard target code vector. In the selected fixed-target mode, that target is the canonical code vector.

### 10.3 Final-latent loss

Only at q3:

\[
L_{final\ latent}=\left\|p_3+\bar e_3-z^*\right\|_1.
\]

This encourages the completed residual sum to approach the canonical codec latent.

### 10.4 Soft-recovery loss

Only at q1-q3, only on selected self-forced batches with an incorrect prefix, as described above.

---

## 11. Final Two-Stage Training Curriculum

### 11.1 Body-only checkpoint lineage

Stage 1:

```text
checkpoints/mask_multipart_variable_c2f_fixed_targets_no_sf_gap1_15
```

- variable gaps 1-15;
- C2F generation;
- fixed canonical targets;
- teacher prefixes;
- no self-forcing;
- no soft recovery;
- trained from scratch.

Stage 2:

```text
checkpoints/mask_multipart_variable_c2f_soft_recovery_sf05_gap1_15
```

- initialized from body Stage 1;
- canonical targets retained;
- self-forcing ramps to 0.5;
- soft recovery weight 0.1;
- hard adaptive targets disabled.

The exact body YAML used for these final checkpoints is no longer present in `motion_generation/configs`. Preserve the checkpoint `config.json` and `trainer_state.json`; they are the authoritative record. Recreate body YAMLs before a reproducibility release.

### 11.2 Face-enabled checkpoint lineage

Stage 1:

```text
checkpoints/mask_multipart_face_variable_c2f_fixed_targets_no_sf_scratch_gap1_15
```

Config:

```text
motion_generation/configs/audio_c2f_face_fixed_targets_no_sf_scratch.yaml
```

This model is trained from scratch on 20-slot body+face tokens with variable gaps, C2F, fixed targets, and no self-forcing.

Stage 2, selected combined checkpoint:

```text
checkpoints/mask_multipart_face_variable_c2f_soft_recovery_sf05_stage2_gap1_15
```

Config:

```text
motion_generation/configs/audio_c2f_face_soft_recovery_sf05_stage2.yaml
```

This continues from face Stage 1 at learning rate `5e-5`, enables self-forcing up to 0.5, and adds soft recovery.

Compute-matched control config, not yet represented in the recorded final result table:

```text
motion_generation/configs/audio_c2f_face_fixed_targets_no_sf_stage2_control.yaml
```

Run this control if a publication-quality attribution of Stage 2 gains is required.

---

## 12. Reported Body-Only Results

These values were reported during development and are not stored in the local Windows checkout as CSV files. Preserve the server-side notebook outputs before cleanup.

### 12.1 FID

| Gap | Codec floor | Stage 1 fixed/no-SF | Stage 2 soft/SF0.5 |
|---:|---:|---:|---:|
| 3 | 1.3242 | **1.9658** | 2.0309 |
| 7 | 1.3247 | 3.5412 | **3.4167** |
| 15 | 1.3321 | 8.0359 | **7.0612** |

Interpretation:

- soft recovery slightly trades short-gap FID;
- it improves medium and especially long-gap FID;
- the benefit is consistent with recovery becoming useful when generated-prefix errors accumulate.

### 12.2 Semantic retrieval R@1

| Gap | Stage 1 fixed/no-SF | Stage 2 soft/SF0.5 |
|---:|---:|---:|
| 3 | 61.26 | **63.15** |
| 7 | 58.74 | **59.21** |
| 15 | 51.34 | **52.91** |

At gap 15, R@10 improved from 85.83 to 87.24.

### 12.3 Ablation interpretation

- Self-forcing without recovery produced only small changes.
- Hard adaptive recovery was harmful, particularly at long gaps: normalized FID reached 11.0655 at gap 15.
- Soft recovery gave the best long-gap tradeoff among the tested recovery variants.

---

## 13. Reported Face-Enabled Results

Evaluation used the 372 face-available validation clips.

### 13.1 Full-clip normalized FID

| Gap | Multipart codec floor | Body-only Stage 2 | Face Stage 1 | Face Stage 2 |
|---:|---:|---:|---:|---:|
| 3 | 0.7960 | 1.8664 | 3.6857 | **2.0400** |
| 7 | 0.8091 | 3.8997 | 6.8053 | **4.3464** |
| 15 | 0.8385 | 9.2743 | 13.2044 | **10.2138** |

Face Stage 2 recovers most of the Stage 1 degradation and approaches the body-only model on the same restricted subset.

### 13.2 Retrieval R@1

| Gap | Body-only Stage 2 | Face Stage 1 | Face Stage 2 |
|---:|---:|---:|---:|
| 3 | 43.01 | 41.13 | **43.01** |
| 7 | 41.13 | 36.83 | **40.86** |
| 15 | 36.83 | 34.14 | **38.98** |

The absolute retrieval values are lower than the earlier 635-clip table because this is a different 372-clip face-available subset.

### 13.3 Body retention

| Gap | Body-only RMSE | Face Stage 1 RMSE | Face Stage 2 RMSE |
|---:|---:|---:|---:|
| 3 | 0.04112 | 0.05578 | 0.04291 |
| 7 | 0.06168 | 0.07738 | 0.06352 |
| 15 | 0.09191 | 0.11223 | 0.09579 |

Stage 2 substantially restores body quality after adding face.

### 13.4 Face quality

| Gap | Stage 1 face accuracy | Stage 2 face accuracy | Stage 1 face RMSE | Stage 2 face RMSE |
|---:|---:|---:|---:|---:|
| 3 | 0.05175 | **0.06877** | 0.07610 | **0.06625** |
| 7 | 0.04099 | **0.05002** | 0.09646 | **0.08935** |
| 15 | 0.03289 | **0.04028** | 0.11488 | **0.10923** |

Face codec floor:

```text
face RMSE:              0.015086
face velocity RMSE:     0.011476
lip RMSE:               0.015200
lip velocity RMSE:      0.011806
non-lip RMSE:           0.015005
non-lip velocity RMSE:  0.011324
```

The infiller remains well above the face codec floor, so face prediction rather than face quantization is now the limiting factor.

---

## 14. Diagnostics That Led to the Final Design

### 14.1 MSD complexity audit

The normalized MSD mean `Omega` was not a reliable monotonic difficulty score. Correlations with teacher NLL were weak and sometimes negative. Spectral energy and physical FK speed were more informative:

| Part | Energy vs NLL rho | FK speed vs NLL rho |
|---|---:|---:|
| Feet | 0.325 | 0.228 |
| Hands | 0.383 | 0.141 |
| Lower | 0.425 | 0.318 |
| Upper | 0.248 | 0.227 |

MSD remains a diagnostic only. It is not currently used for masking, attention, curriculum, or loss weighting.

### 14.2 Residual-recoverability audit

Observed q0 accuracies:

| Part | q0 accuracy |
|---|---:|
| Feet | 0.5945 |
| Hands | 0.4895 |
| Lower | 0.6130 |
| Upper | 0.6423 |

The audit showed:

- q0 errors dominate reconstruction damage;
- later residual levels can compensate for many q0 errors;
- beam-selected residual tails were generally more stable than purely greedy tails;
- when q0 was correct, recovered tails could approach codec-GT reconstruction;
- geometric recovery alone does not guarantee common or semantically valid token combinations.

This motivated a soft candidate distribution instead of assigning one hard replacement token.

---

## 15. Experiments Deliberately Not Selected

### 15.1 Hard adaptive residual targets

Mechanism: replace q1-q3 canonical labels with nearest residual codes conditioned on the generated prefix.

Outcome: adaptive-token accuracy improved, but FID and retrieval worsened, especially at long gaps. Likely causes include discontinuous hard relabeling and rare cross-level code combinations. Keep `adaptive_target_mode: never` in the selected system.

### 15.2 Self-forcing without recovery

Outcome: mostly neutral, with a modest long-gap FID improvement. It establishes that generated-prefix exposure alone does not explain the full soft-recovery gain.

### 15.3 Audio residual posterior

Outcome: approximately 88.4% more trainable parameters with only small and inconsistent metric improvement. Removed.

### 15.4 Routed additive audio

Outcome: no robust advantage over legacy additive conditioning. It remains implemented for ablation but is inactive.

### 15.5 L3/L6 audio cross-attention adapters

Outcome: full-data cross-attention helped some isolated cells, but adding it to soft recovery did not improve the selected baseline consistently.

Normalized FID comparison:

| Gap | Soft recovery | L3/L6 cross-attention | Soft recovery + L3/L6 |
|---:|---:|---:|---:|
| 3 | 2.0309 | 1.8865 | 2.0503 |
| 7 | 3.4167 | 3.2953 | 3.4260 |
| 15 | **7.0612** | 7.1333 | 7.0989 |

The L3/L6 implementation and notebook branches were removed to keep the active architecture coherent.

---

## 16. Evaluation Stack

Core utility: `motion_generation/utils/variable_c2f_evaluation.py`.

### 16.1 Deterministic window metrics

- token accuracy;
- exact-frame and exact-gap accuracy;
- per-part accuracy;
- per-quantizer accuracy;
- decoded body MAE/RMSE;
- velocity, acceleration, and jerk RMSE;
- face/lip/non-lip RMSE and velocity RMSE when face is present;
- invalid-token fraction;
- hard latent RMSE after the complete C2F rollout.

### 16.2 Full-clip metrics

- evaluator FID, raw and normalized by GT;
- generated and GT diversity;
- text-to-motion R@1/R@2/R@3/R@5/R@10 and median rank;
- seam acceleration and jerk statistics;
- BAS, BHR, and ESD beat metrics.

Primary full-clip gaps are 3, 7, and 15.

### 16.3 Notebooks

`motion_generation/notebooks/compare_face_infill_metrics.ipynb`

- authoritative current body-vs-face checkpoint comparison;
- full-clip FID/diversity and R@K;
- face codec floor;
- body and face window metrics;
- body+face visualization;
- automatic body-only visualization fallback when a requested clip has no ARKit data.

`motion_generation/notebooks/compare_variable_c2f_infill_metrics.ipynb`

- contains the generic variable-gap evaluator;
- its current `MODEL_SPECS` still point to early pretrained/scratch/fixed checkpoints;
- update those specs before using it for the final soft-recovery body comparison.

`motion_generation/notebooks/multipart_residual_recoverability_audit.ipynb`

- q0 and residual-tail recoverability audit.

`motion_generation/notebooks/multipart_msd_complexity_error_audit.ipynb`

- MSD, energy, speed, teacher-NLL, and generated-correctness analysis.

---

## 17. Visualization

Implementation: `motion_generation/utils/face_infill_visualization.py`.

The full-clip renderer shows:

- raw motion GT;
- multipart codec GT;
- Stage 1 output;
- Stage 2 output;
- generated versus anchor/context regions;
- original audio waveform envelope;
- a 2D diagnostic ARKit face driven by all 51 coefficients when face exists.

For a clip without ARKit data, the notebook now loads the 16-slot body tokens and renders:

- `Raw GT`;
- `Codec GT`;
- `Body soft recovery`.

It omits the face panel rather than fabricating neutral coefficients.

`VISUALIZATION_FRAME_STEP=2` writes 10 rendered frames per second while preserving the real-time duration. Use 1 for final 20 FPS output.

---

## 18. Reproduction Commands

Run from the repository root on the Linux server.

### 18.1 Face codec

```bash
CUDA_VISIBLE_DEVICES=0 \
python motion_generation/scripts/train_multipart_rvqvae.py \
  --config motion_generation/configs/face_rvqvae_512x4.yaml
```

### 18.2 Face Stage 1: scratch, fixed targets, no self-forcing

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py \
  --config motion_generation/configs/audio_c2f_face_fixed_targets_no_sf_scratch.yaml
```

### 18.3 Face Stage 2: selected soft recovery + SF0.5

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29515 \
  motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py \
  --config motion_generation/configs/audio_c2f_face_soft_recovery_sf05_stage2.yaml
```

### 18.4 Optional compute-matched Stage 2 control

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 --master_port=29516 \
  motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py \
  --config motion_generation/configs/audio_c2f_face_fixed_targets_no_sf_stage2_control.yaml
```

### 18.5 Evaluation order

In `compare_face_infill_metrics.ipynb`:

1. Load codecs and the paired face-available subset.
2. Run window metrics.
3. Set `RUN_FULL_CLIP_EXPORT=True` once and export gaps 3/7/15.
4. Disable export after it completes.
5. Enable FID/diversity and R@K cells.
6. Run visualization separately; it need not repeat full-clip metric export.

Trainer checkpoint directories are named `checkpoint-6000`, not `checkpoints-6000`. The root output directory is valid only after `trainer.save_model()` has written `config.json` and model weights there.

---

## 19. W&B Signals Worth Retaining

General:

- `loss`, `eval/loss`, `learning_rate`, `grad_norm`, epoch, global step.

Per stage:

- `train_c2f/q{q}_ce`;
- `train_c2f/q{q}_embed`;
- `train_c2f/q3_final_latent`;
- `train_c2f/q{q}_original_acc`;
- `train_c2f/{part}_q{q}_adaptive_acc`.

Self-forcing and recovery:

- `train_c2f/self_forcing_probability`;
- `train_c2f/self_forced_batch`;
- `train_c2f/soft_recovery_batch`;
- `train_c2f/q{q}_soft_recovery`;
- `train_c2f/q{q}_recovery_pool_entropy`;
- `train_c2f/q{q}_recovery_top1_original_rate`;
- `train_c2f/q{q}_recovery_samples`.

Evaluation rollout:

- `eval_c2f/final_original_acc`;
- per-gap original accuracy;
- overall and per-part hard latent RMSE;
- hard latent RMSE split by q0 correct/wrong and gap buckets.

The metric name `adaptive_acc` is legacy. When `adaptive_target_mode=never`, it is accuracy against canonical targets.

---

## 20. Known Limitations and Remaining Controls

1. **Face coverage:** only 12,367/21,133 released clips have usable ARKit; face results use 372 validation clips.
2. **Face error:** final face RMSE is still far above the face codec floor.
3. **Short-gap tradeoff:** soft recovery improves long gaps but is slightly worse in gap-3 FID.
4. **Compute-matched attribution:** run the face fixed-target/no-SF Stage 2 control before making a strong causal claim about Stage 2 gains.
5. **Protocol split:** historical FID uses noncanonicalized raw roots for comparability; do not mix it with a canonicalized-root table.
6. **Body reproducibility configs:** final body YAML files should be reconstructed from checkpoint metadata.
7. **Evaluation artifacts:** final result CSVs exist on the Linux run environment, not in this Windows checkout. Archive them.
8. **Self-forcing terminology:** describe the implementation as self-forcing-inspired generated-prefix training unless the full paper objective is implemented.
9. **Production inference:** `motion_generation/pipeline_infer.py` still contains legacy fixed-window assumptions and `generate_sbs` paths. It is not yet fully wired to the 16/20-slot variable-gap C2F checkpoints.
10. **End-to-end Step 1 integration:** the next pipeline must ensure Step 1 keyframes use the same multipart token layout and codec checkpoints as Step 2.

---

## 21. Recommended Next Work

The next conversation should begin outside Step 2 architecture search and focus on integration:

1. Freeze the selected body and face Stage 2 checkpoints.
2. Archive checkpoint configs, trainer states, W&B run URLs, and final CSV outputs.
3. Reconstruct missing body YAMLs from checkpoint metadata.
4. Run the face compute-matched Stage 2 control if required for the paper.
5. Wire the new multipart token layout into end-to-end inference.
6. Replace legacy fixed-three-frame `generate_sbs` calls with variable-gap `generate_quantizer_coarse_to_fine`.
7. Define how Step 1 emits anchors for all parts, including face where available.
8. Decide the runtime policy for clips/audio without facial anchors.
9. Run final held-out test evaluation only after the full Step 1 -> Step 2 pipeline is frozen.

---

## 22. Suggested New-Conversation Prompt

```text
Read motion_generation/STEP2_MULTIPART_C2F_HANDOFF.md and inspect the referenced
code before proposing changes. Treat the current Step 2 architecture as frozen:
multipart 512x4 RVQ, variable gaps 1-15, q0->q3 coarse-to-fine generation,
legacy additive HuBERT conditioning, and Stage 2 canonical CE plus self-forced
soft residual recovery. The selected combined checkpoint is
checkpoints/mask_multipart_face_variable_c2f_soft_recovery_sf05_stage2_gap1_15.

My next goal is to integrate this Step 2 model into the end-to-end SentiAvatar
Step 1 -> Step 2 inference/training pipeline. First inspect the current Step 1,
pipeline_infer.py, token layouts, and checkpoint contracts. Identify every
legacy fixed-gap or four-token assumption and propose a migration plan before
editing code. Do not redesign Step 2 unless an integration blocker requires it.
```

---

## 23. Primary Code Map

| Responsibility | File |
|---|---|
| Transformer and C2F generation | `motion_generation/models/audio_motion_model.py` |
| Variable-gap training, self-forcing, losses | `motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py` |
| Multipart RVQ model | `motion_generation/models/multipart_rvqvae.py` |
| Multipart RVQ training | `motion_generation/scripts/train_multipart_rvqvae.py` |
| Root normalization and part layout | `motion_generation/utils/multipart_motion.py` |
| Multipart token export | `motion_generation/scripts/export_multipart_motion_tokens.py` |
| Evaluation and full-clip export | `motion_generation/utils/variable_c2f_evaluation.py` |
| Face/body visualization | `motion_generation/utils/face_infill_visualization.py` |
| Face Stage 1 config | `motion_generation/configs/audio_c2f_face_fixed_targets_no_sf_scratch.yaml` |
| Face Stage 2 config | `motion_generation/configs/audio_c2f_face_soft_recovery_sf05_stage2.yaml` |
| Compute-matched control | `motion_generation/configs/audio_c2f_face_fixed_targets_no_sf_stage2_control.yaml` |
| Face codec config | `motion_generation/configs/face_rvqvae_512x4.yaml` |
| Final face evaluation notebook | `motion_generation/notebooks/compare_face_infill_metrics.ipynb` |
| Generic variable-gap evaluation | `motion_generation/notebooks/compare_variable_c2f_infill_metrics.ipynb` |
| Residual audit | `motion_generation/notebooks/multipart_residual_recoverability_audit.ipynb` |
| MSD audit | `motion_generation/notebooks/multipart_msd_complexity_error_audit.ipynb` |

---

## 24. MOSS Nano All-16-RVQ Retraining Branch (2026-07)

This section is newer than the HuBERT integration recommendations above. The
current controlled experiment retrains Step 2 with MOSS Audio Tokenizer Nano
while leaving the motion side and Step 2 Transformer architecture unchanged.

### 24.1 Audio representation

- Input codec: causal MOSS Audio Tokenizer Nano, 48 kHz.
- Stored codes: all 16 residual codebooks, q0 through q15, 1,024 entries each.
- Native token rate: 12.5 Hz.
- Step 2 feature: the frozen Nano quantizer decodes all q0-q15 contributions,
  sums the residual-codebook embeddings, and applies its frozen output
  projection.
- Resulting tensor: one 768-D continuous quantized latent per Nano frame.
- Motion alignment: nearest physical-time Nano frame for each 10 Hz motion
  token frame; no latent interpolation.
- Step 2 still trains its existing 768-to-512 audio projection. Nano and its
  quantizer remain frozen and are not loaded by training workers.

This uses every RVQ layer without adding 16 new trainable embedding tables or
expanding the Step 2 sequence length.

### 24.2 Offline feature export

The existing Nano token export is expected at:

`SuSuInterActs/SuSuInterActs/audio_tokens_moss_nano_48k_12p5hz_16cb`

Decode it with:

`motion_generation/scripts/export_moss_nano_all16_features.py`

The output is:

`SuSuInterActs/SuSuInterActs/audio_features_moss_nano_all16_12p5hz_768d`

Feature files are float16 on disk and are converted to float32 by the existing
Step 2 loader. Per-shard manifests record the model hashes, all-16 contract,
feature rate, dimension, and alignment rule. The exporter is safe to resume
without `--overwrite`.

### 24.3 Two-stage training protocol

Stage 1 is a new model trained from scratch with fixed canonical targets:

`motion_generation/configs/audio_c2f_face_moss_nano_all16_fixed_targets_no_sf_scratch.yaml`

Stage 2 must initialize from that Nano Stage 1 output, never from the HuBERT
checkpoint:

`motion_generation/configs/audio_c2f_face_moss_nano_all16_soft_recovery_sf05_stage2.yaml`

All motion-side settings are held fixed to the selected face/body Step 2
protocol: multipart 512x4 tokens, gaps 1-15, q0-to-q3 generation, canonical CE,
latent auxiliaries, and the established Stage 2 self-forcing/soft-recovery
schedule.

Both configs require complete Nano audio coverage for every split clip that has
a multipart face-motion token file. Missing face-motion clips remain excluded,
matching the existing face-coverage protocol. Training also rejects feature
arrays whose second dimension is not 768.

### 24.4 Inference status

The training path and checkpoint metadata are wired. Production inference is
not yet migrated: legacy inference code still extracts Chinese HuBERT
features. Do not feed those features into a checkpoint whose
`audio_representation` is `moss_nano_quantized_latent_q0_q15`. Runtime Nano
feature extraction and checkpoint-driven dispatch remain a separate
integration task.

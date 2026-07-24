# Adaptive Multipart Anchor Planning and Infilling Handoff

**Status date:** 2026-07-24
**Scope:** Step 1 planner design, Step 2 integration, losses, training curriculum, inference contract, implementation plan, and evaluation gates.  
**Detailed Step 2 reference:** [`STEP2_MULTIPART_C2F_HANDOFF.md`](STEP2_MULTIPART_C2F_HANDOFF.md)  
**Literature review:** [`../learned_sparse_anchor_planning_literature_review.md`](../learned_sparse_anchor_planning_literature_review.md)
**Phase 1 implementation:** [`PHASE1_FIXED_GAP3_IMPLEMENTATION.md`](PHASE1_FIXED_GAP3_IMPLEMENTATION.md)

---

## 1. Executive Decision

The proposed system is a **rate-adaptive plan-then-infill motion generator**:

1. Step 1 is a heavy causal language model initialized from the existing Qwen motion planner.
2. It receives semantic text, discrete audio tokens, target duration, and previous sparse anchors.
3. It autoregressively emits a sparse sequence of:
   - one discrete gap token; then
   - one complete multipart RVQ anchor containing 16 body token IDs.
4. Step 1 does **not** emit one `CONTINUE` or mask token per motion frame.
5. Step 1 itself emits all 16 RVQ IDs. There is no additional anchor-content model or small multipart decoder in Version 1.
6. The runtime constructs mask frames from each emitted gap token and calls Step 2 between adjacent anchors.
7. Step 2 is the existing bidirectional, audio-conditioned, variable-gap, coarse-to-fine infiller.
8. The selected Step 2 checkpoint remains permanently frozen as a **reference and initial judge**, but a copied Step 2 may later be adapted to predicted Step 1 anchors. Step 2 does not have to remain frozen for the entire project.

The central decision rule is:

> Spend an anchor only when its expected reduction in downstream infilling risk is greater than the cost of predicting that anchor.

The desired result is not simply maximum temporal distance. It is the **widest safe next interval**, subject to semantic correctness, anchor predictability, Step 2 quality, a maximum gap, and a latency budget.

---

## 2. Why This System Exists

The original fixed-step planner spends the same number of anchors in every motion regime. That is inefficient:

- a slow hold or simple conversational motion may not need anchors every four token frames;
- a fast semantic transition or multipart coordination event may require closer anchors;
- a long gap reduces expensive Step 1 output but is harder for Step 2 and increases right-anchor buffering latency;
- an apparently useful GT anchor may be too difficult for Step 1 to predict reliably.

The research question is therefore:

> Can a condition-only planner predict where a frozen or adapted infiller will need help, generate reliable multipart anchors at those times, and improve the quality-rate-latency frontier over strong fixed schedules?

This is best understood as **rate-constrained conditional subgoal planning through a generative motion channel**.

---

## 3. Current Project State

### 3.1 Motion and audio rates

- Raw body, hand, and face motion: 20 FPS.
- Multipart codec temporal downsampling: 2.
- Motion-token rate: 10 FPS.
- Stored HuBERT layer-9 features: 768 dimensions at 10 FPS.
- Discrete HuBERT/K-means tokens for Step 1: 10 FPS before optional fixed-grid subsampling.
- Motion and HuBERT therefore align one-to-one at the token-frame level.

### 3.2 Multipart motion representation

Four independent body codecs are trained, each with codebook size 512 and four residual quantizers:

| Part | Features | RVQ tokens per anchor |
|---|---:|---:|
| `upper` | 16 upper-body joints x 6D | 4 |
| `lower` | root delta + 5 lower-body joints x 6D | 4 |
| `feet` | 4 foot/ball joints x 6D | 4 |
| `hands` | both hands, excluding wrists | 4 |

The canonical body anchor layout is:

```text
upper q0 q1 q2 q3
lower q0 q1 q2 q3
feet  q0 q1 q2 q3
hands q0 q1 q2 q3
```

This is 16 local IDs in `[0, 511]`, or 16 slot-offset global IDs.

The face-enabled layout appends:

```text
face q0 q1 q2 q3
```

for 20 IDs per anchor. Face is available for only 12,367 of 21,133 released clips, so Version 1 should establish the body-only system first unless combined body-face production is a hard requirement.

### 3.3 Existing Step 2

The selected Step 2 architecture is:

- bidirectional/noncausal within an infill interval;
- conditioned on both boundary anchors;
- conditioned on dense 10 FPS HuBERT features through legacy additive fusion;
- trained on missing gaps 1 through 15;
- generated coarse-to-fine over `q0 -> q1 -> q2 -> q3`;
- trained with canonical CE, generated-prefix exposure, and soft residual recovery.

Reported body-only validation results illustrate why gap selection matters:

| Missing gap | Normalized FID | Retrieval R@1 |
|---:|---:|---:|
| 3 | 2.0309 | 63.15 |
| 7 | 3.4167 | 59.21 |
| 15 | 7.0612 | 52.91 |

Longer gaps reduce the number of Step 1 anchors but are measurably harder for Step 2.

### 3.4 Existing Step 1 and inference pipeline

The current legacy path in [`pipeline_infer.py`](pipeline_infer.py):

- samples audio at a fixed step, normally 4;
- asks the vLLM planner to emit only four residual IDs per sparse frame;
- parses `res_1` through `res_4`;
- assumes a fixed schedule;
- uses legacy interpolation helpers and `generate_sbs` paths;
- does not understand 16-slot multipart anchors or variable gap tokens.

This path is a migration source, not the new contract.

---

## 4. Terminology and the Gap Off-by-One Rule

Let the motion-token sequence have frames `0 ... T-1`. Let consecutive anchor indices be `i < j`.

The Step 2 handoff defines:

\[
g = j-i-1
\]

where `g` is the number of masked interior token frames. Therefore:

\[
j-i = g+1.
\]

Examples:

| Gap token | Masked interior frames | Anchor distance | Approximate time at 10 FPS |
|---|---:|---:|---:|
| `[gap_3]` | 3 | 4 token frames | 0.4 s |
| `[gap_7]` | 7 | 8 token frames | 0.8 s |
| `[gap_15]` | 15 | 16 token frames | 1.6 s |

Every serializer, decoder, dataset, evaluator, and test must use this definition. Do not use `gap` to mean anchor distance in one component and interior-mask count in another.

---

## 5. End-to-End Architecture

```text
action/motion text -------------------------+
                                             |
discrete audio tokens on a fixed grid -------+--> Step 1 causal LM
                                             |       |
target motion-token length ------------------+       | sparse output
                                                     v
previous sparse anchors ------------------------> [anchor, gap, anchor, ...]
                                                     |
                                                     | runtime inserts masks
                                                     v
dense HuBERT features + adjacent anchors ------> Step 2 C2F infiller
                                                     |
                                                     v
                                             dense multipart tokens
                                                     |
                                                     v
                                             frozen RVQ decoders
                                                     |
                                                     v
                                                full motion
```

There is one shared temporal schedule for all body parts because the current Step 2 expects complete 16-slot boundary frames. Part-specific asynchronous schedules would require a different Step 2 contract and are out of scope for Version 1.

---

## 6. Step 1 Contract

### 6.1 Selected Version 1 causality mode

The recommended Version 1 is an **utterance-level or bounded-lookahead sparse causal decoder**:

- Step 1 is autoregressively causal over its generated gaps and anchors.
- The semantic text and schedule-independent discrete audio-token sequence are available as an input prefix.
- The model generates only sparse gap and anchor tokens.
- It does not execute a heavy Qwen forward step for every 10 FPS motion frame.

This preserves the original reason for sparse planning: expensive Step 1 compute scales with the number of anchors rather than dense motion length.

This is not strict live-microphone causality with respect to future audio. A strict-online variant would need a lightweight per-frame boundary controller and would still incur buffering because Step 2 requires a future right anchor. That variant is deferred.

### 6.2 Step 1 inference inputs

Required:

1. **Semantic text condition**
   - Version 1 should retain the paper-like action/motion label because it is available in the dataset and current pipeline.
   - Replacing it with only the spoken transcript is a separate harder experiment.
2. **Discrete audio tokens**
   - Use all aligned 10 FPS discrete tokens initially, or a fixed schedule-independent coarse grid.
   - Do not sample audio only at the unknown learned anchor locations; that creates a circular input contract.
3. **Target motion-token length `T`**
   - Normally equal to the aligned 10 FPS HuBERT length.
   - The runtime uses `T` to constrain gap decoding and force the final anchor.
4. **Previous generated gaps and anchors**
   - Supplied naturally through the causal LM prefix/KV cache.
5. **Optional continuation context**
   - Previous utterance's final anchors, character ID, or dialogue state.

Training additionally has dense GT multipart motion tokens and decoded motion. GT motion is never an inference-time scheduler input.

### 6.3 Step 1 vocabulary

The legacy vocabulary of four token families is insufficient because each part has independent codebooks.

Recommended human-readable special-token families:

```text
[upper_q0_0] ... [upper_q0_511]
[upper_q1_0] ... [upper_q1_511]
...
[hands_q3_0] ... [hands_q3_511]
```

This is 16 x 512 = 8,192 body motion tokens. Each maps one-to-one to Step 2's global slot-offset token ID:

\[
global\_id = slot \times 512 + local\_id.
\]

Control tokens:

```text
[motion_start]
[motion_end]
[gap_0] ... [gap_15]
```

Normal adaptive choices should initially be `[gap_3]` through `[gap_15]`. `[gap_0]` through `[gap_2]` remain available for final-tail arithmetic, difficult transitions, and fallback behavior:

- `gap_0`: adjacent anchors, no Step 2 call;
- `gap_1` and `gap_2`: supported by the current Step 2.

The Step 2 mask ID is not a Step 1 output token.

### 6.4 Step 1 output grammar

```text
[motion_start]

<16 tokens for anchor at t=0>

[gap_g]
<16 tokens for anchor at t=g+1>

[gap_h]
<16 tokens for the next anchor>

...

<16 tokens for final anchor at t=T-1>
[motion_end]
```

Example:

```text
[motion_start]
[upper_q0_42]...[hands_q3_106]
[gap_7]
[upper_q0_11]...[hands_q3_73]
[gap_3]
[upper_q0_91]...[hands_q3_208]
[motion_end]
```

The runtime derives absolute anchor times cumulatively:

\[
t_{k+1}=t_k+g_k+1.
\]

The parser should convert the generated text into one canonical machine-readable object before Step 2 is called:

```json
{
  "token_length": 63,
  "layout": "body_16slot_512x4",
  "anchors": [
    {"time": 0, "tokens": [42, 6, 36, 77, 21, 2, 79, 1, 29, 41, 97, 12, 21, 78, 61, 6]},
    {"time": 8, "tokens": [11, 8, 32, 70, 24, 5, 73, 9, 20, 45, 90, 18, 27, 71, 60, 4]},
    {"time": 12, "tokens": [91, 7, 39, 75, 28, 3, 76, 8, 25, 48, 93, 14, 22, 74, 63, 5]},
    {"time": 28, "tokens": [64, 9, 30, 72, 26, 6, 70, 4, 23, 43, 88, 16, 29, 69, 58, 7]},
    {"time": 44, "tokens": [55, 4, 33, 68, 31, 8, 74, 2, 27, 46, 95, 19, 25, 72, 66, 3]},
    {"time": 60, "tokens": [47, 5, 35, 73, 22, 7, 78, 6, 21, 40, 92, 15, 24, 76, 59, 8]},
    {"time": 62, "tokens": [43, 6, 37, 76, 20, 4, 77, 5, 28, 42, 96, 13, 23, 75, 62, 6]}
  ],
  "gaps": [7, 3, 15, 15, 15, 1]
}
```

Each `tokens` array contains exactly 16 local IDs in canonical slot order. The JSON object is an internal runtime representation; the Qwen completion remains the token sequence above.

### 6.5 Constrained decoding

Generation must use a grammar/logit mask rather than relying on prompt compliance:

1. After `[motion_start]`, require the 16 canonical slots in order.
2. At slot `s`, allow only that slot's 512 tokens.
3. After a complete anchor, allow only a valid gap token or `[motion_end]`.
4. A candidate gap is valid only when:

   \[
   t_k+g+1\le T-1.
   \]

5. Permit `[motion_end]` only after an anchor exactly at `T-1`.
6. Force a gap no larger than Step 2's technical limit and the production latency limit.
7. If no normal gap 3-15 fits the remaining duration, allow the short-tail tokens 0-2.

Malformed or incomplete generations must fail loudly; do not silently truncate different slot streams to their shortest length as the legacy parser does.

### 6.6 No additional anchor decoder in Version 1

The same heavy Step 1 causal LM emits:

- each gap token; and
- all 16 multipart anchor token IDs.

Do not introduce a separate small kinematic or multipart decoder now. This keeps model management, checkpointing, and training simpler. A factorized decoder remains a future speed optimization only if Step 1 output latency becomes the measured bottleneck.

---

## 7. Step 2 Contract

### 7.1 Inputs per interval

For adjacent anchors `i` and `j` with gap `g=j-i-1`, Step 2 receives:

- complete left 16-slot anchor;
- `g` complete 16-slot mask frames;
- complete right 16-slot anchor;
- dense HuBERT features for frames `i ... j`;
- padding mask where required.

The flattened body sequence length is:

\[
(g+2)\times16.
\]

### 7.2 Output

Step 2 predicts all `g x 16` missing token IDs through four passes:

1. all-parts `q0` in parallel;
2. all-parts `q1` in parallel;
3. all-parts `q2` in parallel;
4. all-parts `q3` in parallel.

The interval includes the original anchors plus the generated interiors. Adjacent intervals share exactly one anchor and are stitched into the full token sequence.

### 7.3 Frozen reference versus adaptive production copy

Training Step 2 first was the correct bootstrap because anchor usefulness is defined relative to an infiller. The current selected Step 2 should be preserved as:

```text
Step2_reference
```

It serves as:

- a reproducible baseline;
- the first schedule-cost oracle;
- a stable judge during early Step 1 training;
- protection against catastrophic co-adaptation.

If predicted Step 1 anchors expose a serious train-inference mismatch, create:

```text
Step2_adaptive
```

initialized from the reference checkpoint. Adapt it on a mixture of GT and predicted anchors. During alternating training, use a lagged or periodically copied `Step2_target` to provide stable schedule costs while `Step2_adaptive` changes.

Freezing is therefore a **phase-level optimization rule**, not necessarily a permanent architectural requirement.

### 7.4 What unfreezing cannot solve

Unfreezing can improve robustness to predicted anchors and nonuniform schedules. It cannot remove the structural requirement for a future right anchor. Strict-online output must still buffer an interval until the right boundary exists unless Step 2 is redesigned as a causal or dual-mode generator.

---

## 8. Planning Objective

Let the schedule be:

\[
A=\{(t_0,r_0),\ldots,(t_K,r_K)\},
\qquad 0=t_0<\cdots<t_K=T-1.
\]

The deployment-aligned objective is:

\[
\begin{aligned}
L_{system}={}&L_{schedule}
+\alpha L_{anchor}
+\beta L_{fill}
+\gamma L_{semantic}\\
&+\delta L_{seam}
+\lambda L_{rate}
+\zeta L_{risk}.
\end{aligned}
\]

Hard constraints:

\[
0\le g_k\le15,
\qquad g_k+1\le G_{latency},
\qquad t_K=T-1.
\]

The technical maximum is not automatically the production maximum. At 10 FPS, `[gap_15]` requires the future anchor 1.6 seconds after the left anchor.

This equation is the conceptual system objective, not a claim that every term is differentiable through hard sampled gaps and RVQ IDs. The first implementation optimizes causal-LM CE and oracle-distillation losses. Latent/decoded losses require probability-weighted codec embeddings or a straight-through relaxation, while hard full-pipeline semantic and Step 2 risks require candidate-level minimum-risk or policy-gradient training. Do not silently detach a loss and report it as Step 1 supervision.

---

## 9. Oracle Cost and Schedule Supervision

### 9.1 Pairwise edge cost

For candidate anchors at `i < j`, define:

\[
C_{ij}=L_{fill}(i,j)+\kappa P^{fail}_{ij}.
\]

`L_fill(i,j)` should combine:

- canonical Step 2 token CE;
- final latent error;
- decoded motion/velocity error;
- boundary/seam error;
- per-part errors normalized so upper body and hands do not dominate;
- optional worst-part or CVaR penalty for rare catastrophic failures.

The MSD atlas shows that upper body and hands dominate raw latent energy and that hands are relatively independent of lower body and feet. A single unnormalized combined score is therefore not an acceptable schedule cost.

### 9.2 Anchor prediction/node cost

Let:

\[
P_j=L_{anchor}(j)
\]

measure how difficult Step 1 finds the candidate anchor at `j`. This prevents an oracle from selecting spectacular GT poses that greatly help Step 2 but cannot be inferred reliably from text/audio.

### 9.3 Dynamic program

With a fixed number of anchors:

\[
D[k,j]=P_j+\min_i\{D[k-1,i]+C_{ij}\}
\]

over valid predecessors satisfying the gap constraint.

With a variable rate, use shortest path with an anchor penalty:

\[
\min_A
\sum_{(i,j)\in A}C_{ij}
+\mu\sum_{j\in A}P_j
+\lambda |A|.
\]

The marginal value of inserting anchor `m` between `i` and `j` is:

\[
V(m\mid i,j)=
C_{ij}-C_{im}-C_{mj}-\mu P_m.
\]

Insert it only when:

\[
V(m\mid i,j)>\lambda.
\]

### 9.4 Two different oracles

Always report both:

1. **GT-content oracle**: interval endpoints are GT tokens. This measures theoretical placement sensitivity.
2. **Predicted-content oracle**: interval endpoints come from Step 1 or a calibrated corruption model. This measures deployable placement sensitivity.

If only the GT-content oracle beats a fixed schedule, anchor prediction is the bottleneck. If neither beats the fixed schedule, adaptive scheduling is not justified for the current Step 2/regime.

### 9.5 Soft schedule targets

There is usually no unique correct schedule. Convert oracle cost-to-go into a soft next-gap target:

\[
q(g\mid state)
\propto
\exp\left(-\frac{C(g)+V_{future}(g)}{\tau}\right).
\]

Train Step 1 with:

\[
L_{schedule}=KL(q\Vert p_\theta).
\]

A hard oracle schedule and ordinary gap-token CE are acceptable for the first implementation. Soft cost-sensitive targets are the preferred refinement.

---

## 10. Step 1 Losses

### 10.1 Causal LM/token loss

Mask the input prompt from the LM loss. Supervise only the motion-plan completion.

Separate metrics and optional weights for:

- control/gap-token CE;
- each part;
- each RVQ level;
- first/final anchors;
- teacher-forced versus generated-prefix examples.

For a selected time `t_k`:

\[
L_{anchor\_token}
=\sum_{p,q}w_q\,
CE(\hat r_{k,p,q},r^{GT}_{t_k,p,q}).
\]

### 10.2 Latent and decoded anchor losses

Exact residual IDs are not identical to motion equivalence. Once token CE is stable, add:

\[
L_{anchor\_latent}
=\sum_p w_p\,
Huber(\hat z_{k,p},z^{GT}_{t_k,p})
\]

and optional decoded pose/velocity loss:

\[
L_{anchor\_motion}
=\|\hat x_{t_k}-x^{GT}_{t_k}\|_1
+\rho_v\|\Delta\hat x_{t_k}-\Delta x^{GT}_{t_k}\|_1.
\]

The combined anchor loss is:

\[
L_{anchor}= 
L_{anchor\_token}
+\rho_zL_{anchor\_latent}
+\rho_xL_{anchor\_motion}.
\]

To backpropagate these auxiliary losses into Step 1, construct expected codec latents from the slot logits and frozen codebook embeddings, or use an explicitly documented straight-through estimator. Argmax token IDs break the gradient. These losses are optional until the basic token predictor is reliable.

### 10.3 Training after a predicted gap

During initial teacher forcing:

```text
oracle gap -> supervise anchor at oracle target time
```

During generated-gap training:

```text
predicted gap g
-> target time t + g + 1
-> supervise against the GT multipart tokens at that predicted time
```

Do not keep the oracle anchor target after the model selects a different time.

### 10.4 Semantic loss

The dataset mostly supplies sentence-level action labels, not timestamped semantic events. Do not force every individual anchor to match the entire sentence independently.

Apply semantics to the ordered sparse plan or final infilled motion:

\[
L_{semantic}=InfoNCE(E_{motion}(\hat X),E_{text}(T_{text})).
\]

Use hard negatives that change direction, action identity, or event order. Validate semantic conditioning with text swaps and ablations rather than attention magnitude.

If `E_motion` consumes a hard Step 2 rollout, this loss is a sequence-level reward rather than an ordinary differentiable loss. Use it for candidate reranking/minimum-risk training, or construct a documented soft-latent surrogate. Keep retrieval metrics on held-out hard generations as the final semantic evidence.

### 10.5 Rate and latency

\[
L_{rate}=\frac{K+1}{T}
\]

where `K+1` consistently counts all emitted anchors, including endpoints if Step 1 generates them.

Anchor count is a proxy, not a complete speed metric. Report and optionally optimize measured costs:

\[
C_{runtime}=
aN_{Step1\ tokens}
+bN_{Step2\ calls}
+c\sum_k C_{Step2}(g_k).
\]

Fewer anchors reduce Step 1 output length and Step 2 call count, but longer Step 2 windows may cost more per call. Benchmark rather than assume total speedup.

### 10.6 Boundary loss

Penalize decoded velocity discontinuity around shared anchors:

\[
L_{seam}=\sum_k
\|\Delta\hat x_{t_k^-}-\Delta\hat x_{t_k^+}\|_1.
\]

---

## 11. Step 2 Adaptation Loss

If a trainable copy is required, freeze Step 1 while adapting Step 2 on a mixture of:

- GT anchors and random variable gaps;
- fixed-schedule predicted Step 1 anchors;
- adaptive-schedule predicted Step 1 anchors;
- calibrated anchor corruption matching observed Step 1 errors.

Use:

\[
L_{Step2\ adaptive}
=L_{canonical\ CE}
+aL_{final\ latent}
+bL_{soft\ recovery}
+cL_{decoded}
+dL_{seam}
+eL_{GT\ replay}.
\]

GT replay is required to prevent Step 2 from forgetting its general infilling behavior or inventing a private token protocol with Step 1.

Potential future extension: pass calibrated endpoint confidence to Step 2 so it can learn how strongly to trust each predicted anchor. This changes the Step 2 interface and is not Version 1.

---

## 12. Training Curriculum

### Phase 0: lock contracts and archive references

1. Preserve the selected body-only and face-enabled Step 2 checkpoints and configs.
2. Preserve the four body codec checkpoints and token-export manifest.
3. Define the exact Step 1 special-token vocabulary and global-ID mapping.
4. Add unit tests for gap arithmetic, slot order, parsing, and exact final length.
5. Do not use held-out test data while selecting the planner design.

### Phase 1: strongest fixed-schedule Step 1 baseline

1. Initialize from the existing Qwen motion-planning checkpoint.
2. Resize the tokenizer/model for 8,192 body motion tokens plus gap/control tokens.
3. Use a fixed interval of 4 (`[gap_3]`) initially.
4. Train Step 1 to emit all 16 body IDs per anchor.
5. Use anchor token CE first; add latent/decoded anchor losses after basic convergence.
6. Train with GT previous anchors, then introduce generated previous-anchor prefixes.

This phase establishes whether multipart anchor content is predictable before schedule learning is attempted.

### Phase 2: oracle-headroom gate with `Step2_reference`

1. Cache feasible interval costs for gaps 1-15 on validation/training subsets.
2. Build the GT-content DP oracle.
3. Compare it with fixed interval 4, 8, and 16 under equal anchor rate.
4. If the GT-content oracle cannot beat fixed placement, stop adaptive scheduling work.

### Phase 3: predicted-content gate

1. Run the fixed-schedule Step 1 model to measure real anchor error.
2. Build predicted-content or calibrated-corruption edge costs.
3. Solve the predicted-content DP oracle.
4. If predicted-content placement cannot beat uniform placement, improve anchor content or Step 2 robustness before training a scheduler.

### Phase 4: optional Step 2 adaptation

If predicted endpoints are the bottleneck:

1. Copy `Step2_reference` to `Step2_adaptive`.
2. Freeze Step 1.
3. Adapt Step 2 on mixed GT/predicted endpoints and mixed schedules.
4. Preserve canonical CE and original-distribution replay.
5. Recompute predicted-content oracle costs.

### Phase 5: train variable-gap Step 1

1. Serialize oracle schedules as gap-and-anchor sequences.
2. Train gap tokens with hard CE first, or soft cost targets when ready.
3. Constrain all training and inference schedules to the Step 2 gap contract.
4. Progress from oracle previous gaps/anchors to generated prefixes.
5. When a generated gap differs, supervise the anchor at the generated target time.

### Phase 6: alternating co-adaptation

Repeat a small number of major cycles:

```text
freeze Step 2 target
-> update Step 1 content and schedule

freeze Step 1
-> roll out current plans
-> update Step2_adaptive on those plans plus GT replay

refresh Step2_target and oracle/risk costs
-> update Step 1 again
```

Do not naïvely train both discrete modules from scratch at the same time. A moving judge makes schedule learning unstable and permits degenerate co-adaptation.

### Phase 7: optional hard minimum-risk refinement

Only after supervised training succeeds:

1. sample several hard gap-and-anchor plans per condition;
2. run full Step 2 rollouts;
3. score whole-motion semantic, boundary, quality, rate, and latency outcomes;
4. apply minimum-risk or self-critical sequence training;
5. monitor reward hacking and static/easy-anchor collapse.

This is optional and compute-heavy. It is not required for the first working system.

---

## 13. Inference Algorithm

### 13.1 Step 1 sparse plan generation

```text
input:
    action/motion text
    discrete audio tokens on a fixed grid
    target token length T
    optional previous-utterance anchors

generate initial 16-slot anchor at t=0
current_t = 0

while current_t < T-1:
    mask invalid gap tokens using remaining duration and production limits
    generate one gap token g
    next_t = current_t + g + 1
    generate exactly 16 slot-constrained multipart IDs
    append (next_t, anchor)
    current_t = next_t

require current_t == T-1
generate [motion_end]
```

### 13.2 Step 2 full-clip generation

```text
for each adjacent anchor pair (i, j):
    g = j - i - 1
    if g == 0:
        append the adjacent right anchor; do not call Step 2
    else:
        construct [left][MASK x g][right]
        slice dense HuBERT features i ... j
        run generate_quantizer_coarse_to_fine()
        append generated interior and shared right anchor

assert dense token length == T
decode all four parts with the matching codec checkpoints
```

The new pipeline must not call legacy `generate_sbs()` for selected variable-gap C2F checkpoints.

---

## 14. Implementation Work Breakdown

The following paths are proposals until created.

### 14.1 Token and schedule contract

Proposed:

```text
motion_generation/utils/adaptive_anchor_tokens.py
```

Responsibilities:

- slot definitions and order;
- local/global ID mapping;
- Qwen special-token string mapping;
- gap/anchor-distance conversion;
- plan serialization and strict parsing;
- constrained-decoding state machine;
- final-length validation;
- body versus body-face layouts.

### 14.2 Step 1 dataset and trainer

Proposed:

```text
motion_generation/scripts/train_step1_adaptive_anchor.py
motion_generation/configs/step1_multipart_fixed_gap3.yaml
motion_generation/configs/step1_multipart_adaptive_gap.yaml
```

Responsibilities:

- load action text, discrete audio tokens, multipart GT tokens, and length;
- select fixed, oracle, or generated schedules;
- build prompt/completion pairs;
- mask prompt labels;
- compute gap and slot-specific metrics;
- support GT and generated previous anchors;
- save tokenizer and model together for vLLM.

Version 1 should reuse the existing Qwen causal LM directly. Do not create a second anchor-content model.

### 14.3 Oracle builder

Proposed:

```text
motion_generation/scripts/build_step1_anchor_oracle.py
motion_generation/utils/anchor_schedule_dp.py
```

Responsibilities:

- enumerate feasible edges for gaps 1-15;
- run `Step2_reference` or `Step2_target`;
- cache GT-content and predicted-content edge costs;
- normalize per-part metrics;
- solve fixed-budget and rate-penalized schedules;
- export one-best, top-L, cost-to-go, and soft next-gap targets.

### 14.4 Step 2 adaptation

Extend rather than duplicate:

```text
motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py
```

Required additions only if Phase 4 is reached:

- load predicted Step 1 anchor datasets or online rollouts;
- mix GT and predicted endpoint batches;
- record endpoint corruption/error strata;
- preserve canonical CE and GT replay;
- save separate reference, live, and target checkpoint identities.

### 14.5 End-to-end inference migration

Refactor:

```text
motion_generation/pipeline_infer.py
```

Required changes:

1. replace fixed sampled-index prompt construction with the new schedule-independent audio input contract;
2. replace the four-stream `res_1...res_4` parser with strict gap-and-16-slot parsing;
3. derive anchor indices from gap tokens;
4. enforce exact target length before Step 2;
5. build variable mask windows;
6. call `generate_quantizer_coarse_to_fine()`;
7. decode with the four matching multipart codec checkpoints;
8. record separate Step 1, Step 2, codec, and total timing;
9. fail on malformed plans rather than silently truncating.

### 14.6 Tests

Proposed:

```text
motion_generation/scripts/test_adaptive_anchor_tokens.py
motion_generation/scripts/test_anchor_schedule_dp.py
motion_generation/scripts/test_step1_step2_pipeline.py
```

Minimum tests:

- all 16 slots round-trip local ID -> text token -> global ID;
- every gap 0-15 round-trips;
- `[gap_3]` produces exactly three mask frames and anchor distance four;
- invalid gap logits are masked near the tail;
- short-tail schedule reaches exactly `T-1`;
- malformed slot order is rejected;
- Step 1 never emits a Step 2 mask token;
- Step 2 receives `(g+2) x slots` tokens;
- adjacent `gap_0` skips Step 2;
- stitched output length equals audio/token length;
- fixed `[gap_3]` reproduces the fixed-step baseline contract;
- body-only and body-face mappings never share global ranges accidentally.

---

## 15. Evaluation and Go/No-Go Gates

### 15.1 Required factorial comparison

| Schedule | Anchor content | Step 2 | Purpose |
|---|---|---|---|
| Fixed interval 4/8/16 | GT | Reference | Placement upper-bound baselines |
| GT-content DP oracle | GT | Reference | Theoretical adaptive headroom |
| Fixed interval 4/8/16 | Predicted | Reference | Real fixed baselines |
| Predicted-content DP oracle | Predicted | Reference | Realistic adaptive headroom |
| Learned adaptive | Predicted | Reference | Scheduling contribution |
| Fixed | Predicted | Adaptive copy | Step 2 adaptation contribution |
| Learned adaptive | Predicted | Adaptive copy | Full co-adapted system |

All comparisons must match:

- average anchor count or explicitly compare rate-quality curves;
- maximum gap;
- Step 2 generation procedure;
- number of samples/candidates;
- codec and evaluator versions;
- face/body evaluation subset.

### 15.2 Primary gates

1. **GT oracle headroom:** if GT-content adaptive placement cannot beat fixed placement, stop schedule learning.
2. **Predicted oracle headroom:** if predicted-content placement cannot beat fixed placement, improve anchor prediction or Step 2 robustness first.
3. **Identifiability:** a condition-only student must predict oracle marginal gains/gaps better than duration-only and gap-only baselines.
4. **Deployment:** learned placement with predicted content must improve the rate-quality frontier, not just one cherry-picked metric.

A reasonable pre-registered research threshold is for the deployable student to recover a meaningful fraction, such as 30%, of the predicted-content oracle advantage. The exact threshold must be fixed before final experiments.

### 15.3 Metrics

**Step 1 anchor layer**

- gap-token NLL/accuracy and oracle regret;
- token top-1/top-k accuracy by part and RVQ level;
- anchor latent and decoded error;
- generated-prefix versus teacher-forced degradation;
- anchor count, mean/P95/max gap.

**Step 2 interval layer**

- canonical token CE/accuracy;
- per-part hard latent RMSE;
- decoded pose/velocity error;
- failure rate versus gap and endpoint corruption;
- seam/boundary discontinuity;
- confidence calibration.

**Whole motion**

- normalized FID under a fixed protocol;
- retrieval R@1/R@K;
- ESD/beat synchronization;
- diversity;
- per-part motion quality;
- blinded human semantic, naturalness, and synchronization preference.

**Efficiency and causality**

- Step 1 generated-token count and wall-clock time;
- Step 2 calls, total processed tokens, and wall-clock time;
- total end-to-end latency;
- right-anchor buffering delay;
- memory and KV-cache usage.

### 15.4 Counterfactual conditioning tests

Do not use high attention weights as evidence of conditioning. Test behavior:

- change action text while keeping audio fixed;
- shift or shuffle audio tokens while keeping text fixed;
- reverse action order in hard-negative prompts;
- remove text or audio separately;
- retain anchors but permute their schedule;
- corrupt one part or RVQ level of an anchor.

Text changes should primarily affect semantic anchor content. Audio timing changes should primarily affect gap placement and local rhythm.

---

## 16. Expected Goal and Research Claim

The primary target is **Pareto improvement**, not a guaranteed single-number gain:

- at equal anchor count, improve final motion quality, semantics, seams, or failure rate;
- at equal final quality, reduce Step 1 anchor tokens and planner runtime;
- maintain a hard Step 2 gap and production-latency guarantee;
- allocate dense anchors to complex/identifiable transitions and sparse anchors to simple motion.

For an average 63-token-frame clip:

| Anchor interval | Approximate anchors | Step 1 body motion IDs |
|---:|---:|---:|
| 4 | 17 | 272 |
| 8 | 9 | 144 |
| 16 | 5 | 80 |

The adaptive model should move between these rates according to condition-predictable motion difficulty rather than choosing one global interval.

Candidate novelty, subject to final literature verification:

1. condition-only variable-rate scheduling plus discrete multipart anchor generation;
2. explicit coupling of anchor predictability, frozen/adapted infiller risk, semantics, rate, and latency;
3. predicted-content oracle distillation instead of relying only on perfect GT endpoints;
4. phase-wise frozen judging followed by alternating co-adaptation;
5. deployment-aligned separation of placement, content, and infilling gains.

A concise claim is:

> We introduce a rate-adaptive multipart motion planner that autoregressively emits gap-conditioned discrete anchors and learns when an audio-conditioned infiller requires intervention, jointly accounting for anchor predictability, semantic fidelity, infilling risk, and generation cost.

---

## 17. Known Risks and Degenerate Solutions

### 17.1 Too many anchors

Step 1 may minimize fill loss by emitting nearly every frame. Prevent with rate loss, hard budgets, and equal-rate evaluation.

### 17.2 Step 2 ignores anchors

An adapted Step 2 may become audio-only. Retain endpoint-adherence losses and anchor-swap counterfactual tests.

### 17.3 Private Step 1-Step 2 token protocol

Joint modules may exploit unusual code combinations that no longer represent canonical motion. Keep codec/codebooks frozen, retain GT token CE and decoded anchor loss, and replay real anchors during Step 2 adaptation.

### 17.4 Oracle leakage

The DP teacher observes GT motion. The deployable Step 1 may use only text, audio, length, and generated history. Always report the oracle-student identifiability gap.

### 17.5 Nonunique valid motion

Single-reference token/pose loss can penalize valid alternatives. Keep distributional retrieval, diversity, and human evaluation alongside paired reconstruction losses.

### 17.6 Pairwise cost misses global behavior

DP assumes mostly additive interval costs. This assumption is more credible for fixed-skeleton motion than for general video, but must be checked through full-clip reranking and cross-segment seam metrics.

### 17.7 Longer gaps are not automatically faster

They reduce Step 1 output and Step 2 calls but increase missing frames, Step 2 window size, and buffering delay. Measure the full runtime frontier.

### 17.8 Face coverage

Body-first results use 635 validation clips; face-enabled evaluation uses only the 372 face-available validation intersection. Do not compare their absolute metrics as if they were the same set.

---

## 18. Current Decisions and Open Questions

### Decided for Version 1

- body-first 16-slot planner;
- one shared temporal schedule for all parts;
- current heavy Qwen planner emits both gap tokens and all 16 RVQ IDs;
- no separate anchor-content decoder/model;
- sparse gap-token output, not one heavy `CONTINUE` decision per dense frame;
- mask tokens are constructed by Step 2 runtime, never emitted by Step 1;
- gap means number of masked interior frames;
- keep Step 2 reference checkpoint permanently frozen for controls;
- permit a trainable Step 2 copy only after predicted-anchor mismatch is measured;
- train content before adaptive scheduling;
- preprocess Mimi/body tokens for all clips but train the first fixed-gap
  baseline on deterministic nested 512/2,000/6,000 train-only subsets;
- balance the subsets with raw body/hand rotation dynamics and metadata, not
  MSD outputs, while retaining the complete validation split;
- use GT and predicted-content oracle gates before expensive end-to-end optimization;
- do not optimize or interpret raw attention magnitude as grounding.

### Open before final production implementation

1. Is the full utterance/TTS audio available before motion planning, or is strict live-microphone causality required?
2. Is an explicit action/motion label available at production inference, or only transcript/dialogue context?
3. Should normal scheduling permit gaps 1-2, or reserve them only for tail/fallback cases?
4. What production buffering limit is acceptable: 0.4, 0.8, 1.6 seconds, or another value?
5. Is Version 1 body-only, or must Step 1 immediately emit 20 body-face IDs?
6. Which existing Qwen planner checkpoint is the canonical initialization, and are its pretraining assets reproducibly archived?
7. Does the final system use `Step2_reference`, `Step2_adaptive`, or a frozen core plus small adapters?
8. Which server-side result artifacts and checkpoint configs still need copying into this checkout?

These questions alter experiment scope or deployment semantics and should be resolved explicitly rather than hidden in defaults.

---

## 19. Developer Entry Checklist

Before editing code, a new developer should:

1. Read this document completely.
2. Read [`STEP2_MULTIPART_C2F_HANDOFF.md`](STEP2_MULTIPART_C2F_HANDOFF.md).
3. Inspect [`pipeline_infer.py`](pipeline_infer.py), especially the legacy four-token parser and fixed-step prompt construction.
4. Inspect [`models/audio_motion_model.py`](models/audio_motion_model.py), especially `generate_quantizer_coarse_to_fine()`.
5. Inspect [`scripts/train_audio_mask_multipart_variable_c2f.py`](scripts/train_audio_mask_multipart_variable_c2f.py).
6. Inspect [`utils/multipart_motion.py`](utils/multipart_motion.py) and [`scripts/export_multipart_motion_tokens.py`](scripts/export_multipart_motion_tokens.py).
7. Confirm the actual availability and metadata of all Step 1, Step 2, and codec checkpoints.
8. Implement and test the token/gap contract before changing training or inference.
9. Reproduce the fixed `[gap_3]` 16-slot baseline before implementing adaptive schedules.
10. Do not evaluate on the final held-out test set until architecture, losses, and schedule policy are frozen.

### Existing code map

| Responsibility | Existing path |
|---|---|
| Legacy Step 1 -> Step 2 pipeline | `motion_generation/pipeline_infer.py` |
| vLLM Step 1 service | `motion_generation/vllm_server.py` |
| Step 2 Transformer/C2F generation | `motion_generation/models/audio_motion_model.py` |
| Step 2 variable-gap training | `motion_generation/scripts/train_audio_mask_multipart_variable_c2f.py` |
| Step 2 evaluation | `motion_generation/utils/variable_c2f_evaluation.py` |
| Multipart codecs | `motion_generation/models/multipart_rvqvae.py` |
| Multipart codec training | `motion_generation/scripts/train_multipart_rvqvae.py` |
| Multipart token export | `motion_generation/scripts/export_multipart_motion_tokens.py` |
| Step 1 balanced subset builder | `motion_generation/scripts/build_step1_training_subsets.py` |
| Root and part preprocessing | `motion_generation/utils/multipart_motion.py` |
| Step 2 detailed handoff | `motion_generation/STEP2_MULTIPART_C2F_HANDOFF.md` |
| Multipart MSD atlas | `motion_generation/utils/msd/outputs/multipart_atlas_val/README.md` |

---

## 20. Immediate Next Action

Do not begin with adaptive scheduling or joint Step 1-Step 2 training.

The first implementation milestone is:

> Extend the existing Qwen Step 1 tokenizer and training data to emit a fixed `[gap_3]` schedule with complete 16-slot body anchors, then run those predicted anchors through the frozen body-only Step 2 reference using the selected variable-gap C2F decoder.

That milestone validates the new vocabulary, multipart anchor predictability, generated-prefix behavior, Step 2 interface, decoding, metrics, and timing. Only after it works should the project build DP oracles and learn variable gaps.

---

## 21. Implemented Frozen-Step-2 Curriculum (2026-07-24)

The fixed `[gap_3]` multipart baseline, causal body codecs, MOSS Nano audio
path, structured text, and body-only MOSS-Nano-all16 Step 2 now exist. The
next experiment has therefore moved beyond Section 20.

### 21.1 Exact experiment

The implemented 50-epoch run uses:

| Epochs | Schedule source | Allowed normal gaps | Mean-gap target | Gap-loss weight |
|---:|---|---:|---:|---:|
| 1-5 | deterministic random warm-up | 3-7 | none | 0 |
| 6-12 | frozen-Step-2 DP oracle | 3-7 | 4.5 | linear 0 -> 1 |
| 13-25 | frozen-Step-2 DP oracle | 3-11 | 6.0 | 1 |
| 26-50 | frozen-Step-2 DP oracle | 3-15 | 7.0 | 1 |

Gaps 0-2 are legal only when they land exactly on the final token frame.
They are forced EOS arithmetic, have no learned gap target, and do not
contribute to the reported mean normal gap.

The trainable loss is:

\[
L = L_{\text{anchor CE}} + w(e)L_{\text{soft gap CE}}.
\]

For a frozen Step 2 interval `(i,j)`, the detached edge risk is:

\[
C(i,j)=g\left(
  \operatorname{CE}_{\text{C2F rollout}}
  +0.1\operatorname{L1}_{\text{hard RVQ latent}}
\right),\qquad g=j-i-1.
\]

The DP minimizes:

\[
\sum_{(i,j)} C(i,j)+\lambda\,N_{\text{new anchors}}.
\]

`lambda` is calibrated independently for each DP phase by bisection on a
fixed 512-clip training subset. It is not manually or linearly ramped. The
soft gap target is a Boltzmann distribution over edge cost plus DP
cost-to-go. Step 2 remains frozen and detached throughout; no adapter or
gradient path is introduced.

This first oracle uses **GT boundary-anchor content**. Predicted-content
oracle costs and self-forcing remain later gates/fine-tuning stages.
Text/audio alignment losses remain deliberately disabled.

### 21.2 Code map

| Responsibility | Path |
|---|---|
| DP, legal-tail rules, lambda calibration | `utils/step1_adaptive_schedule.py` |
| Frozen Step 2 interval-cost export | `scripts/cache_step2_interval_costs.py` |
| Lambda calibration and consolidated schedule materialization | `scripts/calibrate_step1_adaptive_gap.py` |
| Adaptive dataset and soft gap CE | `models/step1_mimi_planner.py` |
| Training loop, phase-local best reset, metrics | `scripts/train_step1_multipart_fixed_gap3.py` |
| Four-phase/full-data configuration | `configs/step1_multipart_adaptive_gap_step2_curriculum50.yaml` |
| All-phase data preflight | `scripts/validate_step1_fixed_gap_data.py` |

### 21.3 One-time frozen-Step-2 cost export

The example below selects the completed soft-recovery Step 2. If the selected
production checkpoint is instead the fixed-target root, change both
`--config` and `--checkpoint` consistently. Never combine shards from
different Step 2 weights.

Run these in four terminals:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/cache_step2_interval_costs.py \
  --config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_adaptive_gap_oracle/step2_interval_costs \
  --num_shards 4 --shard_id 0 --device cuda:0 --batch_size 256
```

```bash
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/cache_step2_interval_costs.py \
  --config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_adaptive_gap_oracle/step2_interval_costs \
  --num_shards 4 --shard_id 1 --device cuda:0 --batch_size 256
```

```bash
CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/cache_step2_interval_costs.py \
  --config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_adaptive_gap_oracle/step2_interval_costs \
  --num_shards 4 --shard_id 2 --device cuda:0 --batch_size 256
```

```bash
CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python \
  motion_generation/scripts/cache_step2_interval_costs.py \
  --config motion_generation/configs/audio_c2f_body_causal_moss_nano_all16_soft_recovery_sf05_stage2.yaml \
  --checkpoint checkpoints/mask_multipart_body_causal_moss_nano_all16_variable_c2f_soft_recovery_sf05_stage2_gap1_15 \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_dir checkpoints/step1_adaptive_gap_oracle/step2_interval_costs \
  --num_shards 4 --shard_id 3 --device cuda:0 --batch_size 256
```

This is intentionally an offline operation: the full train+validation corpus
contains millions of candidate intervals. Existing per-clip files are skipped,
so an interrupted shard can be restarted with the same command. Do not use
`--overwrite` when merely resuming.

### 21.4 Calibrate lambda and materialize schedules

Run only after all four manifests report full coverage:

```bash
python motion_generation/scripts/calibrate_step1_adaptive_gap.py \
  --config motion_generation/configs/step1_multipart_adaptive_gap_step2_curriculum50.yaml \
  --cost_dir checkpoints/step1_adaptive_gap_oracle/step2_interval_costs \
  --split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --output_json checkpoints/step1_adaptive_gap_oracle/calibration.json \
  --calibration_max_clips 512 \
  --ce_weight 1.0 \
  --latent_weight 0.1
```

This command rejects missing shards, incomplete coverage, mixed Step 2
weights, missing clips, and stale schedule contracts. It writes one
consolidated schedule file for each DP phase; no manifest merge is needed.

### 21.5 Preflight

The validator serializes every selected train/eval clip once per curriculum
phase and verifies schedule endpoints, legal gaps, audio consumption, target
counts, and maximum sequence length:

```bash
python motion_generation/scripts/validate_step1_fixed_gap_data.py \
  --config motion_generation/configs/step1_multipart_adaptive_gap_step2_curriculum50.yaml \
  --output_json checkpoints/step1_adaptive_gap_oracle/data_preflight.json
```

Use `--max_train_clips 32 --max_eval_clips 32` first for a quick smoke test.

### 21.6 Four-GPU 50-epoch training

Initialize from the completed structured-text/Nano fixed-gap content model.
This retains the learned 16-ID anchor predictor while starting a fresh
optimizer and 50-epoch curriculum:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun \
  --nproc_per_node=4 --master_port=29514 \
  motion_generation/scripts/train_step1_multipart_fixed_gap3.py \
  --config motion_generation/configs/step1_multipart_adaptive_gap_step2_curriculum50.yaml \
  --init_from_checkpoint checkpoints/step1_gap3_6k_nano_q0q3_structured_text_ce/best
```

For a Qwen-only ablation, omit `--init_from_checkpoint`; that is a different
experiment and should not overwrite the configured output directory.

The trainer saves milestones at epochs 5, 12, 25, and 50. The `best` metric is
reset at each curriculum boundary because composite losses from different
gap ranges/weights are not directly comparable. Thus, after epoch 26, `best`
means best within the final 3-15/mean-7 phase.

### 21.7 Required monitoring

At every epoch inspect:

- `eval/anchor_ce` and 16-slot accuracy;
- `eval/gap_loss` and `eval/gap_accuracy`;
- `eval/mean_normal_gap` versus the active phase target;
- `eval/tail_fraction`;
- phase-boundary changes at epochs 6, 13, and 26.

A calibrated oracle mean near seven does not prove the learned planner emits
mean seven during free rollout. After training, the mandatory next evaluator
must autoregressively sample `next_gap_logits()` plus 16 anchor IDs, report the
generated gap histogram/oracle regret, then run frozen Step 2 and anchor FID.

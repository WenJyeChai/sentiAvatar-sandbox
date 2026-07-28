# Step 1 Stage 1 Planner Failure Analysis

**Status:** diagnostic report after the supplied-gap Stage 1 teacher and causal
student evaluations  
**Scope:** anchor generation only; gap placement is intentionally outside this
stage  
**Primary checkpoint discussed:** the full-audio prefix-LM teacher trained from
`checkpoints/llm`

## 1. Executive conclusion

The Stage 1 model is not random, but it is not learning the deployed task.

It currently learns:

\[
p_\theta(a_{k+1}\mid x_{\text{text}},x_{\text{audio}},
a^{GT}_{0:k},g_k)
\]

where the supplied gap \(g_k\), every previous anchor, and every earlier slot
inside the current 16-ID anchor are ground truth.

At deployment, Step 1 and Step 2 require:

\[
p_\theta(a_{k+1}\mid x_{\text{text}},x_{\text{audio}},
\hat a_{0:k},g_k)
\]

where both the previous anchors and the already emitted IDs inside the current
anchor are generated. The training objective never exposes the clean Stage 1
model to that state distribution.

This mismatch is large enough to explain the main result:

- teacher-forced anchor accuracy: **11.59%**
- fully generated-history anchor accuracy: **2.47%**
- previous-GT-anchor copy baseline: **8.98%**
- frozen Step 2 CE with GT endpoints: **4.30**
- frozen Step 2 CE with generated endpoints: **9.39**
- decoded raw-GT RMSE after Step 2: **0.102 with GT endpoints** versus
  **0.199 with generated endpoints**

The generated anchors are therefore unusable as reliable Step 2 boundaries,
even though their decoded anchor-only error is not catastrophic. Step 2
amplifies modest endpoint errors because it was trained mainly around
ground-truth boundary states.

There is also a likely initialization problem: `checkpoints/llm` predicts four
legacy whole-body RVQ IDs, whereas the new system predicts 16 IDs from four
independent causal multipart codecs trained from scratch. Both layouts happen
to occupy 8,192 body-token rows, but the row meanings do not match. Reusing
those rows is not equivalent to initializing from a compatible motion
foundation model.

The present teacher should not yet be distilled into the causal student. Full
audio lookahead improves clean one-step CE only slightly and does not improve
the rollout sufficiently.

## 2. What is actually being trained

### 2.1 Target representation

Each 10 Hz body anchor contains 16 categorical IDs:

1. upper q0, q1, q2, q3
2. lower q0, q1, q2, q3
3. feet q0, q1, q2, q3
4. hands q0, q1, q2, q3

Every slot has 512 possible IDs. The four part codecs are independent causal
RVQ-VAEs trained from scratch.

### 2.2 Supplied-gap data schedule

For every training clip and epoch, the dataset creates a new feasible random
anchor path:

- normal gaps are sampled from 3 through 15;
- gaps 0 through 2 are reserved for the final EOS tail;
- the path is resampled every training epoch;
- validation uses a fixed deterministic path;
- the gap is an input, not a prediction target.

This is a legitimate way to train the conditional question, “given this gap,
what is the next anchor?” It does not train “where should the next anchor be?”

### 2.3 Sequence presented to the full-audio teacher

Conceptually, the input is:

```text
[structured expression/action/transcript]
[all Nano audio q0-q3 tokens]
[audio_end]
[motion_start]
[seed mode]
[anchor] seed_upper_q0 ... seed_hands_q3
[gap_g1] [anchor] target_1_upper_q0 ... target_1_hands_q3
[gap_g2] [anchor] target_2_upper_q0 ... target_2_hands_q3
...
```

Text, complete audio, controls, and the seed form a bidirectional prefix. The
gap/anchor plan after the seed is autoregressive.

### 2.4 Loss

Only the 16 motion IDs per target anchor receive loss:

\[
\mathcal L_{\text{anchor-CE}}
=
\frac{1}{N}
\sum_{k,s}
-\log p_\theta(a^{GT}_{k,s}\mid\cdot)
\]

The classifier is restricted to the relevant 512-entry slot vocabulary. All
16 slots are weighted equally.

There is:

- no gap-placement loss;
- no generated-history training;
- no within-anchor scheduled sampling;
- no latent or decoded anchor loss;
- no frozen-Step-2 compatibility loss;
- no text/audio reliance loss;
- no visited-state training.

This matches the intended narrow Stage 1 specification, but the narrow
objective is not sufficient to produce stable endpoints.

## 3. The most important train–inference mismatch

### 3.1 Cross-anchor teacher forcing

During training, every previous anchor is the exact GT anchor. During rollout,
the next anchor is conditioned on all previous generated anchors. A mistake at
anchor \(k\) changes the state used to predict anchors \(k+1,k+2,\ldots\).

The observed drop from 11.59% to 2.47% is direct evidence that this exposure
bias is severe.

### 3.2 Within-anchor teacher forcing

The problem also occurs inside each anchor.

The 16 IDs are serialized autoregressively. When training:

- upper q1 sees GT upper q0;
- upper q2 sees GT upper q0-q1;
- lower q0 sees all four GT upper IDs;
- hands q3 sees the preceding 15 GT IDs.

During rollout, those preceding IDs are generated. An early q0 or part error
therefore contaminates the remaining slots before the model even reaches the
next anchor.

The evaluation is consistent with this cascade:

| History | q0 accuracy | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| Teacher forced | 23.68% | 11.75% | 6.68% | 4.26% |
| Generated | 5.90% | 1.97% | 1.19% | 0.84% |

The current “history” discussion has mostly focused on previous anchors, but
the 16-token intra-anchor history is another large source of exposure bias.

### 3.3 Checkpoint selection uses the wrong state distribution

The clean Stage 1 run saves its best checkpoint using teacher-forced validation
loss. Generated rollout evaluation is not the selection criterion for this
configuration.

The checkpoint can therefore improve the metric it is selected on while
remaining poor for the actual Step 1-to-Step 2 interface.

## 4. Input representation audit

### 4.1 Audio is visible but weakly time-addressable

Nano q0-q3 are registered as 4,096 ordinary Qwen tokens. For each 12.5 Hz
audio frame, the sequence contains:

```text
audio_q0(frame t), audio_q1(frame t), audio_q2(frame t), audio_q3(frame t)
```

The advantages are simplicity and direct self-attention. The weaknesses are:

- four simultaneous RVQ streams become four successive language positions;
- there is no explicit audio-frame boundary token;
- there is no learned time embedding tying the four tokens to one timestamp;
- the 4,096 audio rows are new and were not learned by the released Qwen;
- the prefix becomes four times longer than the Nano frame sequence.

For the full-audio teacher, all audio appears before all planned anchors. At an
anchor query, there is no explicit pointer to the target audio time. The model
must infer current motion time by summing all preceding gap tokens, convert
between 10 Hz motion and 12.5 Hz audio, and then attend back to the appropriate
location in a long prefix.

Full future visibility is therefore not the same as good temporal grounding.

The teacher-versus-causal result supports this interpretation:

| Model | TF CE | TF accuracy | rollout accuracy | Step 2 generated CE |
|---|---:|---:|---:|---:|
| Full-audio teacher | 4.626 | 11.59% | 2.47% | 9.391 |
| Causal student | 4.710 | 10.46% | 2.58% | 9.424 |

Lookahead gives a modest clean one-step gain, but almost no end-to-end gain.
The main bottleneck is not simply the absence of future audio.

### 4.2 Text is structured but utterance-level

Expression, action, and transcript are separated with explicit field markers.
This is better than an ambiguous raw string, but all fields describe the whole
utterance. They do not state when a described gesture phase should happen.

The model can therefore learn the overall gesture class while still lacking a
strong timestamp-specific signal for the next anchor.

Earlier condition-counterfactual experiments on related checkpoints found a
much larger text likelihood gap than audio gap and an audio gap that vanished
under generated history. Those results are not a direct measurement of the
present checkpoint, but they warn that perfect motion history can become a
shortcut and allow the model to underuse speech.

### 4.3 Seed modes are not semantically implemented

With `seed_mode: mixed_known`, both `observed` and `previous` use the same first
GT anchor. Only the control token changes.

This does not simulate a true previous-conversation pose. It presents identical
motion under two semantic labels and encourages the model to ignore the
distinction. This is probably not the main failure, but it should be corrected.

## 5. Initialization mismatch

The released `checkpoints/llm` model was trained to predict four legacy
whole-body RVQ IDs per sparse frame. The new target is 16 IDs from four
independent causal multipart codecs trained from scratch.

The layouts both use 8,192 body vocabulary rows:

- legacy: approximately 4 whole-body streams × 2,048 codes;
- current: 16 part/quantizer slots × 512 codes.

That numerical equality does not imply semantic compatibility. RVQ code
indices are arbitrary and permutation-sensitive. For example, a row that used
to mean “legacy whole-body q0 code 512” may now be interpreted as “upper q1
code 0.”

Consequences:

- inherited body input embeddings have the wrong meaning;
- inherited output classifier rows have the wrong meaning;
- tied input/output embeddings make the mismatch affect both directions;
- the model must relearn the current motion vocabulary from only the SuSu
  training set.

The generic Qwen layers and language knowledge can still transfer. The current
checkpoint should, however, not be described as a motion-foundation
initialization for the new codec unless the exact codec provenance proves
otherwise.

This is a high-confidence architectural risk and must be tested with a matched
initialization ablation.

## 6. What the evaluation says

### 6.1 The one-step model learned something

Uniform chance is:

- accuracy: \(1/512 = 0.195\%\)
- CE: \(\log(512)=6.238\)

Teacher-forced accuracy of 11.59% and CE 4.626 are clearly non-random. The
model has learned a useful conditional distribution under clean history.

However, it only slightly beats previous-anchor copying under that favorable
state:

- teacher-forced model: 11.59%
- previous-GT-anchor copy: 8.98%

Under generated history it falls to 2.47%, far below the copy reference. It is
not a stable autoregressive motion prior.

### 6.2 Longer gaps are harder under clean history

| Gap bin | TF accuracy | TF CE | generated accuracy |
|---|---:|---:|---:|
| small, 3-6 | 13.48% | 4.507 | 2.71% |
| medium, 7-10 | 7.81% | 4.827 | 2.42% |
| large, 11-15 | 7.85% | 4.857 | 2.09% |

The clean decline is expected: a farther future anchor is less constrained by
the current pose. Under generated history, all bins collapse into roughly the
same low-performance regime. History failure dominates the normal gap effect.

The easy EOS tail should not be used to judge normal gap performance.

### 6.3 Step 2 works with GT endpoints

With GT boundaries, Step 2 shows the expected difficulty curve:

| Gap bin | Step 2 CE | missing-token accuracy | decoded raw-GT RMSE |
|---|---:|---:|---:|
| small, 3-6 | 3.912 | 21.61% | 0.0868 |
| medium, 7-10 | 4.339 | 14.78% | 0.1055 |
| large, 11-15 | 4.458 | 13.86% | 0.1212 |

Step 2 is therefore capable of using good endpoints. Larger gaps are simply
harder.

### 6.4 Generated endpoints break Step 2

| Gap bin | Generated-endpoint CE | CE increase | accuracy | decoded raw-GT RMSE |
|---|---:|---:|---:|---:|
| small, 3-6 | 9.999 | +6.087 | 3.10% | 0.2126 |
| medium, 7-10 | 9.306 | +4.967 | 3.37% | 0.2029 |
| large, 11-15 | 9.155 | +4.698 | 2.77% | 0.2140 |

Across all gaps:

- Step 2 CE rises from 4.297 to 9.391;
- perplexity rises from about 73.5 to 11,979, a factor of about 163;
- missing-token accuracy falls from 15.99% to 3.15%;
- missing q0 accuracy falls from 38.54% to 7.94%;
- decoded raw-GT RMSE rises from 0.102 to 0.199.

The especially large short-gap CE penalty is plausible. In a short interval,
Step 2 is tightly constrained by both endpoints, so a slightly incompatible
endpoint makes the original GT middle highly unlikely. Over a longer interval,
audio and learned motion priors have more room to dominate.

This does not mean large gaps are better overall. With GT endpoints, large gaps
remain worse. It means bad endpoints contaminate tight infilling especially
strongly.

### 6.5 Exact token accuracy is not the entire problem

Generated anchor substitution alone has:

- codec-relative RMSE: 0.0286;
- raw-GT RMSE: 0.0473, compared with the codec floor of 0.0371.

Thus different token IDs can decode to a pose that is not extremely far away
in average feature space. Exact RVQ accuracy understates this partial success.

Nevertheless, Step 2 still fails badly. The generated pose can be:

- locally close in average coordinates;
- on a code combination or trajectory state uncommon during Step 2 training;
- inconsistent with the particular GT middle used as the CE target;
- wrong in a coarse q0 component that matters disproportionately;
- inconsistent across independently predicted body parts.

The correct conclusion is not merely “token accuracy is unfair.” The endpoints
are genuinely incompatible with the current Step 2 interface.

## 7. Ranked root causes

### Critical: training only on clean autoregressive histories

Evidence: 11.59% teacher-forced accuracy collapses to 2.47% rollout accuracy.
This includes both cross-anchor and within-anchor exposure bias.

### Critical: Step 2 is brittle to off-manifold boundary anchors

Evidence: a modest anchor-only decoded error becomes a roughly 163-fold Step 2
perplexity increase and nearly doubles decoded RMSE. Step 2's internal
self-forcing does not necessarily train it for corrupted external endpoints.

### High: incompatible legacy motion-token initialization

Evidence: the inherited model and current scratch codecs use different learned
representations despite sharing 8,192 row positions.

### High: weak temporal addressability of full-prefix audio

Evidence: future audio improves clean CE only modestly and does not improve
rollout. The model receives no explicit target-time pointer into the audio
prefix.

### High: CE weights all RVQ slots equally

q1-q3 account for 12 of 16 target tokens and therefore 75% of the loss, while
q0 errors appear much more consequential for Step 2. The objective is not
aligned with endpoint utility.

### Medium-high: no codebook geometry or multipart consistency objective

CE treats every wrong ID as equally wrong and never checks whether the 16
predicted IDs decode to a coherent anchor.

### Medium: one-to-many gesture supervision

Speech admits multiple valid gestures, but the loss recognizes only one
recorded token path. This limits exact accuracy. It does not by itself explain
the severe Step 2 failure.

### Low-medium: seed-mode label mismatch

`observed` and `previous` currently contain the same pose, so the mode token
does not represent a real change in input state.

## 8. Recommended recovery sequence

### Phase A: diagnose before another long run

1. **Verify motion-vocabulary provenance.** Confirm which codec and exact slot
   mapping produced the body rows in `checkpoints/llm`.
2. **Run an initialization ablation.** Compare:
   - current reused body rows;
   - reinitialized 8,192 motion input/output rows with the transformer retained;
   - a model pretrained on the exact causal multipart tokens, if available.
3. **Decompose exposure bias.** Evaluate four modes on the same clips:
   - full teacher forcing;
   - generated earlier slots inside each anchor, but GT previous anchors;
   - GT slots inside each anchor, but generated previous anchors;
   - full rollout.
4. **Measure condition use on this checkpoint.** Shuffle/drop audio and text
   under both GT and generated history.
5. **Report copying, code usage, and horizon degradation per part and
   quantizer.**

These diagnostics are cheaper and more informative than another 50-100 epoch
run.

### Phase B: make anchor generation robust

1. Retain clean-history replay, but introduce model-generated or realistic
   corrupted previous-anchor histories.
2. Address **within-anchor** exposure explicitly. Options:
   - slot-level scheduled sampling/corruption; or
   - predict all 16 slots from a shared anchor-query state using 16
     slot-specific heads, removing the 16-step cascade.
3. Weight q0 more strongly than q1-q3, with normalized total weight.
4. After CE stabilizes, add codebook-latent or decoded-anchor supervision.
5. Select checkpoints using rollout q0 accuracy, frozen-Step-2 generated
   endpoint CE, and decoded motion error—not clean CE alone.

Random donor-anchor corruption can be a fast robustness screen, but it is not
equivalent to generated-history training. Unrealistic noise may teach the
model to ignore motion history instead of recover from plausible errors.

### Phase C: repair temporal conditioning

For the teacher, distinguish global future context from local target evidence.
At each anchor query, provide an explicit target-time or target-audio boundary,
or a local audio chunk associated with the supplied gap.

Also compare the ordinary q0-q3 token stream with a frame-level fused audio
embedding. The comparison should keep the dataset, model initialization,
schedule, and optimization fixed.

### Phase D: make Step 2 compatible with predicted endpoints

Once Step 1 improves, fine-tune Step 2 using boundary perturbations drawn from
actual Step 1 rollouts while retaining GT-boundary replay. Otherwise Step 2
will remain a sharp amplifier of small endpoint errors.

This should follow—not replace—improving Step 1. Training Step 2 to tolerate
arbitrary poor anchors would weaken the purpose of anchor planning.

## 9. Suggested go/no-go gates

Before gap-placement training or teacher-to-student distillation:

- generated-history accuracy must clearly exceed the current 2.47%;
- generated q0 accuracy must improve materially beyond 5.90%;
- the generated model should beat previous-anchor persistence on a matched
  rollout metric;
- frozen-Step-2 CE with generated endpoints should move substantially below
  9.39;
- generated-endpoint decoded raw-GT RMSE should move from about 0.199 toward
  0.15 or below on the same protocol;
- audio and text counterfactual gaps should remain positive under generated
  history;
- improvement must hold across small, medium, and large gaps;
- final confirmation should include whole-clip FID and visual inspection.

The numeric targets are engineering gates, not final research claims. They are
intended to prevent spending compute on adaptive gap placement while endpoint
generation remains the dominant bottleneck.

## 10. Immediate decision

Do not proceed directly to adaptive gap learning or distillation from the
current teacher.

The highest-value next action is a short diagnostic matrix covering
initialization compatibility and the two distinct forms of exposure bias.
After that, run a matched clean-control robustness post-training experiment
that includes within-anchor as well as cross-anchor errors. Only when generated
endpoints become useful to frozen Step 2 should gap placement become the next
optimization target.

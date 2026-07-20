# Learned Sparse-Anchor Planning for Conditional Video Generation

## A critical literature review and research design, through 20 July 2026

### Executive judgment

Learned sparse-anchor planning is technically plausible, but the exact problem posed here is not solved by the current literature. No reviewed work found jointly (i) chooses a variable number of temporal positions from text and sparse signals, (ii) predicts discrete visual content at those positions without observing the target video, (iii) hands the predicted anchors to a frozen interval infiller, and (iv) optimizes an explicit anchor-rate/maximum-safe-gap objective. The missing contribution is therefore real, but it lies in the coupling of scheduling, predictable content, and frozen-decoder risk—not in keyframe generation by itself.

The closest architectural precedent is Hsin-Ping Huang, Yu-Chuan Su, and Ming-Hsuan Yang's WACV 2025 long-take system: it generates discrete VQ-VAE keyframe tokens, then invokes an existing masked-token model between every adjacent pair. Its keyframe times and guidance length are fixed, however. Time-Agnostic Prediction (TAP) supplies the strongest conceptual precedent for choosing *predictable* future content, but it minimizes over target-video time during training and does not learn an explicit deployable schedule. KeyVID is the closest timing-aware generative prototype: it predicts audio-derived motion timing and selects keyframes, but uses a fixed 12-of-48 budget, continuous diffusion latents, and a jointly adapted interpolator; its ICLR 2026 submission was desk rejected and should be treated as unreviewed evidence.

The central empirical question is not whether adaptive anchors can outperform an arbitrarily weak fixed baseline. It is whether a deployable planner with *predicted* anchor content beats a strong uniform schedule at the same anchor count, endpoint convention, infiller calls, and sampling compute. Published negative evidence makes this nontrivial: adaptive MRI acquisition can converge to non-adaptive equispaced policies; hierarchical learned codecs often retain fixed group-of-pictures structures; and variable-boundary models can over-segment. Uniform anchors are likely to win on short, smooth, stationary clips, when timing signals are weak, when the infiller's risk is governed mainly by maximum gap, or when schedule mistakes make anchor prediction harder than the saved interpolation cost.

The most credible development path is staged. First, measure a frozen infiller's *oracle schedule advantage* with a dynamic program. If oracle placement cannot beat uniform placement under equal budget, there is little reason to train a planner. If it can, distill the oracle into a condition-only schedule model, train anchor content under predicted/corrupted context, and only then consider minimum-risk fine-tuning with full frozen-infiller rollouts. The decisive test is a factorial decomposition of placement (uniform, oracle, learned) and content (ground truth, predicted), not a single end-to-end score.

### Scope and terminology

This is a structured critical review rather than a statistical meta-analysis. It covers peer-reviewed work and clearly marked preprints available by 20 July 2026 in video generation and infilling, video compression, hierarchical planning, active acquisition, subset selection, discrete stochastic optimization, token pruning, and sequence-level risk training. Papers were included when they illuminate at least one of four decisions: **where** to place anchors, **what** content an anchor should contain, **how many** anchors to use, or **how** predicted anchors interact with a downstream generator.

An **anchor** is a discrete visual representation attached to an explicit time index. A **schedule** is the ordered set

\[
 A=\{(t_m,z_m)\}_{m=0}^{M},\qquad 0=t_0<t_1<\cdots<t_M=T-1,
\]

where \(z_m\) is an anchor token sequence and \(M-1\) is the number of interior anchors. A **planner** is deployable only if it predicts \(A\) from the conditioning variables \(c\)—text, audio, pose, event tags, or other sparse temporal signals—without seeing the target video \(y_{0:T-1}\). A method that selects frames from an already observed target video is an oracle, teacher, or analysis method, not an inference-time planner for generation.

This distinction removes a common category error. Video summarization, adaptive masking, and compression decide which *observed* samples to retain. Conditional generation must decide both position and unobserved content. Their objectives transfer; their information assumptions do not.

## 1. Taxonomy of the design space

| Axis | Main choices | Consequence for this problem |
|---|---|---|
| Position source | Uniform/fixed; event supplied by user; selected from target video; predicted from conditions | Only the last choice is fully deployable; target-video selection is an oracle training signal. |
| Anchor content | Ground-truth frame; continuous latent; discrete code tokens; semantic/layout state | Discrete tokens make likelihood training and caching convenient, but token accuracy is not identical to downstream usefulness. |
| Cardinality | Fixed \(K\); fixed maximum gap; learned stopping/hazard; rate-penalized | Variable cardinality requires an explicit rate term or hard budget; otherwise over-segmentation is attractive. |
| Downstream model | Jointly trained; fine-tuned for schedule; frozen; black-box API | A frozen infiller prevents co-adaptation and makes planner errors more consequential, but permits honest modular evaluation. |
| Optimization | Supervised fixed targets; target-video oracle; differentiable relaxation; policy gradient/minimum risk; search/DP | The best method depends on whether interval cost is additive and whether gradients through the infiller are available or trustworthy. |
| Time representation | Frame index; normalized time; interval; relative gap; event clock | Absolute index alone does not extrapolate reliably to variable lengths; a planner should encode both normalized and relative time. |
| Risk granularity | Anchor reconstruction; independent interval loss; whole-video loss; semantic/event loss | Independent interval costs enable DP but miss identity drift and cross-segment coupling. Whole-video risk is faithful but expensive. |
| Uncertainty | Point schedule; stochastic schedules; calibrated risk; robust/worst-case | Multi-modal futures require expected or risk-sensitive objectives; a single ground-truth schedule is not uniquely correct. |

Three families dominate the literature:

1. **Fixed hierarchical generation.** A coarse model generates uniformly spaced keyframes and another model interpolates. Video LDM, NUWA-XL, [FusionFrames](https://arxiv.org/abs/2311.13073) (**2023 preprint**), and the WACV 2025 long-take system establish feasibility, but not adaptive scheduling.
2. **Externally timed generation.** The user, a scene detector, audio peaks, or annotated events supply time intervals. MinT, SEINE, STAGE, Captain Cinema, and KeyVID show that explicit timing improves control, but they largely avoid the inference-time selection problem.
3. **Observed-data subset selection.** Summarizers, learned codecs, adaptive sensing, and token-pruning methods optimize coverage or reconstruction from data already present. They offer algorithms and failure modes, but using them directly would leak the target video.

## 2. Ten unresolved tensions

### 2.1 Semantic importance versus infiller difficulty

An anchor may be narratively important yet easy to infer from its neighbors, or visually mundane yet essential because it resolves a difficult transition. Query-focused summarization and storyboard systems emphasize semantic salience; learned compression and adaptive masking emphasize reconstruction error. Neither alone is sufficient. A useful edge cost should combine semantic event coverage with the frozen infiller's empirical failure probability. Otherwise a planner selects climactic frames that the infiller already handles while ignoring identity changes, occlusions, or abrupt motion.

### 2.2 Salient versus predictable anchors

TAP's key insight is that a model should predict the future state it can predict reliably, not necessarily a fixed time. That is attractive for anchors, but unconstrained predictability can collapse to static or recurring states. Conversely, selecting high-motion peaks as in KeyVID may create valuable temporal control while making the anchor generator's job harder. The correct quantity is the *joint* cost of predicting an anchor and infilling around it, not a salience score.

### 2.3 Placement quality versus content quality

Most papers vary one axis while holding the other implicit. In the proposed system, a perfectly placed but incorrectly generated anchor can be worse than no anchor: the frozen infiller is forced to reconcile inconsistent endpoints. Evaluation must therefore cross placement \(\times\) content. Ground-truth-anchor results are upper bounds, not evidence that a learned anchor generator works.

### 2.4 Rate minimization versus maximum safe gap

A Lagrangian penalty \(\lambda M\) encourages fewer anchors but does not prevent one catastrophic long interval. A hard constraint \(t_{m+1}-t_m\leq G_{\max}\) prevents such failures but can force redundant anchors in easy regions. Practical systems need both: a safety constraint derived from infiller calibration and a rate term that reallocates the remaining budget.

### 2.5 Pairwise interval cost versus global coherence

If infiller risk decomposes over adjacent anchors, optimal schedules can be found by shortest path or dynamic programming. Real generators share identity, camera, lighting, and narrative state across segments; independent pairwise scores can choose locally good endpoints that form a globally inconsistent chain. DP is valuable as an oracle, but it needs a global reranker or stateful cost when cross-segment drift is material.

### 2.6 Oracle observability versus deployment observability

Compression, summarization, and adaptive masking observe the target frames when deciding what to keep. A generative planner sees only \(c\). An oracle may exploit motion or novelty that is not predictable from the prompt. Distillation succeeds only to the extent that oracle decisions are identifiable from available conditions. The irreducible oracle–student gap should be reported, not hidden inside an end-to-end result.

### 2.7 Teacher-forced versus predicted anchors

VideoAuteur's noise/mask/shuffle regularization and MAGI's complete-teacher-forcing analysis both show that training on ideal context creates exposure problems. A frozen infiller trained on real endpoints may be even more brittle to generated anchors. Planner and content models should be trained and evaluated with rollouts, corrupted anchors, and scheduled replacement by model predictions. Ground-truth-only interval costs are optimistic.

### 2.8 Differentiability versus objective fidelity

Gumbel-Softmax, hard-concrete gates, SoftSort, and optimal-transport top-\(k\) relaxations supply gradients, but introduce temperature bias and soft/hard mismatch. Score-function estimators and minimum-risk training optimize hard schedules and black-box metrics directly, but have high variance and require expensive samples. A sensible sequence is supervised/oracle warm-start followed by hard-schedule risk fine-tuning, not relaxation-only optimization.

### 2.9 Variable duration versus positional generalization

Absolute frame embeddings bind a policy to training lengths; normalized time loses absolute velocity; relative gaps alone lose global narrative phase. A variable-length planner should encode all three, constrain gaps in physical or frame time, and train across length distributions. EOS must mean “the terminal anchor is now safe,” not simply “the model prefers to stop.”

### 2.10 Metric convenience versus causal control

FVD can improve while event order, prompt faithfulness, or boundary continuity worsens. VBench and EvalCrafter broaden measurement, while newer reference-based benchmarks explicitly score event temporal consistency, but automatic metrics remain correlational. Attention maps do not establish that temporal signals caused a schedule. Counterfactual prompt/order/signal interventions and blinded human comparisons are necessary.

## 3. Closest video-generation evidence

### 3.1 The exact architectural chain, but fixed timing

[Generating Long-Take Videos via Effective Keyframes and Guidance](https://openaccess.thecvf.com/content/WACV2025/html/Huang_Generating_Long-Take_Videos_via_Effective_Keyframes_and_Guidance_WACV_2025_paper.html) (Huang, Su, and Yang, WACV 2025, pp. 3709–3720) is the strongest direct precedent. It generates layout guidance, autoregressively predicts discrete VQ-VAE keyframe tokens, and uses an existing masked-token model to generate frames between every pair of consecutive keyframes. Its reported EPIC-KITCHENS FVD improves from 258.4 to 214.7 with ground-truth layouts and to 174.1 with ground-truth keyframes. That large oracle gap is important: better anchor information helps, but it also shows that predicted conditioning is the bottleneck. The schedule length and positions are fixed, so it does not answer where or how many anchors to generate.

[Time-Agnostic Prediction](https://openreview.net/forum?id=SyzVb3CcFX) (Jayaraman et al., ICLR 2019) replaces fixed-time regression with a minimum over possible target times. On three robot-prediction tasks it reports lower \(\ell_1\) error than fixed-time prediction, with relative reductions of 21%, 26.5%, and 53.2%. The result supports predictability-aware anchors and bidirectional recursive decomposition. However, the minimizing time is found using the target video during training, no explicit timestamp need be emitted at inference, and the objective can favor easy/static states. It is a component idea, not a schedule solution.

[NUWA-XL](https://arxiv.org/abs/2303.12346) (Yin et al., arXiv:2303.12346, 2023, **unreviewed preprint**) uses global diffusion to establish sparse global keyframes and local diffusion to fill the hierarchy. [Align Your Latents](https://openaccess.thecvf.com/content/CVPR2023/html/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.html) (Blattmann et al., CVPR 2023, pp. 22563–22575) similarly combines sparse keyframe synthesis with temporal interpolation. Both validate coarse-to-fine generation, but retain fixed sampling patterns and jointly engineered stages.

### 3.2 Infilling capability is mature; schedule learning is not

[MCVD](https://openreview.net/forum?id=hX5Ia-ION8Y) (Voleti et al., NeurIPS 2022) unifies prediction, interpolation, and generation through masked conditional diffusion. [SEINE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html) (Chen et al., ICLR 2024) learns random-mask transitions and short-to-long generation. [Generative Inbetweening](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4bbdef62653d8088717640e7660a1ebb-Abstract-Conference.html) (Wang et al., ICLR 2025) adapts image-to-video models to user-supplied endpoint keyframes with dual-directional sampling. These works show that an interval generator can condition on arbitrary endpoints. They take positions as given, train or fine-tune the infiller, and do not price additional anchors.

### 3.3 Timing is controllable when someone else supplies it

[Mind the Time](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Mind_the_Time_Temporally-Controlled_Multi-Event_Video_Generation_CVPR_2025_paper.html) (Wu et al., CVPR 2025, pp. 23989–24000) binds event captions to user-provided intervals through recurrent rotary position embeddings. It shows that explicit interval conditioning helps event order and duration; it does not infer those intervals. [STAGE](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_STAGE_Storyboard-Anchored_Generation_for_Cinematic_Multi-shot_Narrative_CVPR_2026_paper.html) (Zhang et al., CVPR 2026, pp. 659–669) predicts a start/end frame pair for each predetermined shot. [Captain Cinema](https://openreview.net/forum?id=zlNZBxQZIC) (Xiao et al., ICLR 2026 poster) plans interleaved shot keyframes and then synthesizes multi-keyframe video. Both advance narrative coherence, but inherit shot boundaries or a fixed storyboard structure rather than learning a rate-constrained frame schedule.

[VideoAuteur](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_VideoAuteur_Towards_Long_Narrative_Video_Generation_ICCV_2025_paper.pdf) (Xiao et al., ICCV 2025) generates actions, captions, visual embeddings, keyframes, and shots in a hierarchy. Its ablation is directly relevant to exposure bias: adding Gaussian noise, masking, and shuffling to visual conditions improves reported CLIP-T from 26.4 to 27.3 and FVD from 554.3 to 520.7. Yet it uses a fixed shot sequence and a video generator trained for those conditions, not a frozen generic infiller.

### 3.4 The closest adaptive-timing prototype remains inconclusive

[KeyVID](https://openreview.net/forum?id=oijKOpfSmX) (Wang et al., ICLR 2026 **desk-rejected submission; unreviewed**) derives motion pseudo-labels from RAFT flow, predicts an audio-conditioned motion curve, and picks peaks/valleys plus uniform completion. With 12 anchors in 48 frames, its reported FVD is 262.34 versus 273.40 for 12 uniform anchors, while AlignSync is 24.08 versus 23.53. The comparison supports modest nonuniform benefit under equal count, but the method fixes \(K=12\), uses latent diffusion rather than discrete anchor tokens, and jointly adapts a multi-anchor interpolator. At 48 anchors the gap becomes small. It is useful hypothesis-generating evidence, not confirmation of the proposed architecture.

## 4. Shortlist of the most relevant papers

| Rank | Work | What transfers | What does not |
|---:|---|---|---|
| 1 | [Huang et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Huang_Generating_Long-Take_Videos_via_Effective_Keyframes_and_Guidance_WACV_2025_paper.html) | Discrete predicted keyframes followed by pairwise masked-token infilling; strong oracle-anchor ablation | Fixed keyframe schedule and count |
| 2 | [Jayaraman et al., ICLR 2019](https://openreview.net/forum?id=SyzVb3CcFX) | Predictability-aware target time; bidirectional recursive anchors | Training sees target video; no explicit schedule/rate/frozen infiller |
| 3 | [KeyVID, ICLR 2026 rejected submission](https://openreview.net/forum?id=oijKOpfSmX) | Condition-predicted timing curve; equal-count uniform ablation | Fixed budget, continuous latents, jointly adapted infiller; unreviewed |
| 4 | [Blattmann et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.html) | Sparse keyframes plus reused interpolation model | Uniform/fixed hierarchy; stages co-designed |
| 5 | [NUWA-XL, 2023 preprint](https://arxiv.org/abs/2303.12346) | Global sparse structure plus recursive local filling | Fixed hierarchy; unreviewed and not frozen-module planning |
| 6 | [SEINE, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html) | Random-mask, arbitrary endpoint transition model | Positions supplied; infiller trained jointly for task |
| 7 | [Generative Inbetweening, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4bbdef62653d8088717640e7660a1ebb-Abstract-Conference.html) | Strong two-endpoint image-to-video adaptation | Only supplied endpoints; no scheduler |
| 8 | [VideoAuteur, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_VideoAuteur_Towards_Long_Narrative_Video_Generation_ICCV_2025_paper.pdf) | Predicted visual conditions; explicit noise/mask/shuffle robustness | Shot-level fixed plan; downstream model fine-tuned |
| 9 | [Captain Cinema, ICLR 2026](https://openreview.net/forum?id=zlNZBxQZIC) | Top-down keyframe plan, bottom-up multi-keyframe synthesis | Shot boundaries/data pipeline supply granularity |
| 10 | [STAGE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_STAGE_Storyboard-Anchored_Generation_for_Cinematic_Multi-shot_Narrative_CVPR_2026_paper.html) | Start/end shot anchors and cross-shot memory | Fixed two anchors per predetermined shot |
| 11 | [Mind the Time, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Mind_the_Time_Temporally-Controlled_Multi-Event_Video_Generation_CVPR_2025_paper.html) | Explicit interval/event binding and order evaluation | User supplies intervals; full generator is fine-tuned |
| 12 | [MCVD, NeurIPS 2022](https://openreview.net/forum?id=hX5Ia-ION8Y) | A general masked conditional infiller | No anchor scheduler or budget objective |

## 5. Evidence map: 29 representative papers

“P?” means explicit positions/boundaries are learned; “C?” means missing or anchor content is generated. “Target?” asks whether the selector inspects the target sequence at deployment. A condition-only generative planner needs P=yes, C=yes, and Target=no. “Frozen?” refers to the relevant downstream generator/task model while selection is learned. Most papers meet only a subset of these conditions.

| # | Work | Domain | Mechanism | P? | C? | Target? | Budget/gaps | Frozen? | Optimization and losses | Baseline evidence | Limitation and relevance |
|---:|---|---|---|:---:|:---:|:---:|---|:---:|---|---|---|
| 1 | [Long-Take Keyframes](https://openaccess.thecvf.com/content/WACV2025/html/Huang_Generating_Long-Take_Videos_via_Effective_Keyframes_and_Guidance_WACV_2025_paper.html), WACV 2025 | Video gen. | Fixed-time guidance; AR discrete keyframe tokens; adjacent-pair infill | No | Yes | No | Fixed count/positions | Yes, infiller | Layout/keyframe token likelihood; generation loss | GT layouts/keyframes lower EPIC FVD 258.4→214.7→174.1 | Exact chain, but no timing/rate learning; large predicted-condition gap |
| 2 | [TAP](https://openreview.net/forum?id=SyzVb3CcFX), ICLR 2019 | Video prediction | Min over possible target times; recursive bidirectional prediction | Partial | Yes | No deploy.; yes for training argmin | Fixed recursion; no rate | N/A | Min-time \(\ell_1\); optional CVAE | Beats fixed-time prediction on three robot tasks | Emits no explicit time; can select easy/static states |
| 3 | [KeyVID](https://openreview.net/forum?id=oijKOpfSmX), ICLR 2026 rejected | Audio-video gen. | Predict motion curve; peaks/valleys plus uniform completion; multi-key infill | Yes | Yes | No | Fixed 12/48; endpoints/indexes explicit | No | Motion-label \(\ell_2\); diffusion denoising; sync metrics | At 12 anchors FVD 262.34 vs 273.40 uniform | Fixed rate/continuous latents/joint infiller; unreviewed rejected submission |
| 4 | [NUWA-XL](https://arxiv.org/abs/2303.12346), 2023 preprint | Long video gen. | Global coarse keys then recursive local diffusion | No | Yes | No | Fixed hierarchy | No | Nested diffusion losses | Compared with long-video generators | Shows hierarchy, not adaptive scheduling; unreviewed |
| 5 | [Video LDM](https://openaccess.thecvf.com/content/CVPR2023/html/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.html), CVPR 2023 | Video gen. | Sparse keyframe stage plus temporal interpolation LDM | No | Yes | No | Fixed uniform/coarse stages | Reused, co-designed | Latent diffusion denoising | Component and quality ablations | Scalable chain; no selection/rate objective |
| 6 | [MCVD](https://openreview.net/forum?id=hX5Ia-ION8Y), NeurIPS 2022 | Video infill | Mask past/future blocks; conditional diffusion | No | Yes | Context only | Supplied mask/block; arbitrary length by blocks | No | Conditional denoising | Prediction, generation, interpolation baselines | Capable infiller, not a planner |
| 7 | [SEINE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html), ICLR 2024 | Video infill | Random-mask diffusion between supplied frames | No | Yes | Context only | Supplied/random endpoints | No | Masked diffusion denoising | Transition/prediction/I2V comparisons | Flexible interval model; schedule external |
| 8 | [Generative Inbetweening](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4bbdef62653d8088717640e7660a1ebb-Abstract-Conference.html), ICLR 2025 | Video infill | Reverse I2V adaptation; dual-directional sampling | No | Yes | Supplied keys | Two endpoints | No | Diffusion adaptation/sampling objective | I2V/interpolation baselines | Strong endpoints; no position/cardinality learning |
| 9 | [Mind the Time](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Mind_the_Time_Temporally-Controlled_Multi-Event_Video_Generation_CVPR_2025_paper.html), CVPR 2025 | Temporal video gen. | ReRoPE binds event captions to supplied intervals | No | Yes | No | User fixes intervals | No | Diffusion loss with interval-aware attention | Multi-event temporal-control baselines | Establishes timing value; does not infer timing |
| 10 | [VideoAuteur](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_VideoAuteur_Towards_Long_Narrative_Video_Generation_ICCV_2025_paper.pdf), ICCV 2025 | Narrative video | AR actions/captions/visual embeddings/keys then shots | No frame schedule | Yes | No | Fixed shot sequence | No | Multimodal next-token + video generation; noisy-condition training | Noise/mask/shuffle improves CLIP-T/FVD | Predicted-condition robustness transfers; rate planning does not |
| 11 | [Captain Cinema](https://openreview.net/forum?id=zlNZBxQZIC), ICLR 2026 | Movie gen. | Interleaved shot keyframes then multi-key diffusion | No frame schedule | Yes | No | Shot-defined | No | Planning likelihood + diffusion denoising | Long-movie and component comparisons | Top-down plan; inherited granularity, joint stack |
| 12 | [STAGE](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_STAGE_Storyboard-Anchored_Generation_for_Cinematic_Multi-shot_Narrative_CVPR_2026_paper.html), CVPR 2026 | Storyboard video | Predict start/end image pair for each shot; memory pack | No time schedule | Yes | No | Exactly two per supplied shot | No | Image/video generation losses | Multi-shot/storyboard baselines | Content planning without global anchor-rate choice |
| 13 | [MAGVIT](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html), CVPR 2023 | Discrete video | 3D tokenizer; masked token transformer | No | Yes | Task-dependent context | Supplied masks | No | Tokenizer + masked-token likelihood | Ten generation tasks/tokenizer ablations | Strong discrete representation; no planner |
| 14 | [Phenaki](https://openreview.net/forum?id=vOEXS39nOF), ICLR 2023 | Long video gen. | C-ViViT tokens; AR generation across prompt sequence | No | Yes | No | Variable AR length, not sparse anchors | No | Tokenizer + next-token likelihood | Long/open-domain video baselines | Variable length transfers; no frozen infill interface |
| 15 | [VideoPoet](https://proceedings.mlr.press/v235/kondratyuk24a.html), ICML 2024 | Multimodal video | Decoder-only AR multimodal discrete tokens | No | Yes | No | Variable token sequence | No | Next-token likelihood | Generation/editing task suite | Flexible conditioning; no temporal allocation |
| 16 | [FramePack](https://openreview.net/forum?id=J8JCF64aEn), NeurIPS 2025 | Long video gen. | Time/feature-aware packing of generated history | No | Yes | Generated history | Fixed packed context | No | Diffusion denoising; heuristic packing | Packing/drift ablations | Useful memory/exposure design, not anchor scheduling |
| 17 | [MAGI](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html), CVPR 2025 | AR video gen. | Masked next-chunk generation with complete teacher forcing | No | Yes | Generated history | Fixed chunks | No | Masked token/generation loss | Reports 23% FVD gain over masked teacher forcing | Direct exposure-bias evidence; no schedule learning |
| 18 | [MaskViT](https://arxiv.org/abs/2206.11894), ICLR 2023 | Masked video gen. | Iterative masked visual-token refinement | No | Yes | Context frames | Arbitrary supplied context/mask | No | Masked-token cross-entropy | Prediction baselines/mask ablations | Consumes context but does not decide unseen anchors |
| 19 | [B-EPIC](https://openaccess.thecvf.com/content/ICCV2021/papers/Pourreza_Extending_Neural_P-Frame_Codecs_for_B-Frame_Coding_ICCV_2021_paper.pdf), ICCV 2021 | Compression | Fixed-GOP bidirectional interpolation + residual coding | No | Yes, recon. | Yes | Fixed hierarchical midpoint GOP | N/A | Rate–distortion | Learned P/B codec comparisons; longer GOP costly, little gain | Pairwise RD transfers; observed frames and fixed grid do not |
| 20 | [Neural Rate Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html), ICLR 2024 | Compression | Content-dependent per-frame bit allocation | Partial, rate not time | Yes, recon. | Yes | Target total bitrate | No | Target-rate + rate–distortion losses | Fixed/equal allocation and learned codec baselines | Strong budget analogy; observes frames/transmits residuals |
| 21 | [L3P](https://proceedings.mlr.press/v139/zhang21x.html), ICML 2021 | Robot planning | Learn latent landmarks and reachability graph | Yes, state landmarks | Yes, latent goals | Training data | Sparse graph, not frame budget | No | Reachability/Q distillation and planning | Goal-conditioned RL/planning baselines | “Reachable anchor” analogy; no temporal tokens/frozen dynamics |
| 22 | [CompILE](https://proceedings.mlr.press/v97/kipf19a.html), ICML 2019 | Imitation/segmentation | Differentiable boundaries + latent segment codes | Yes | Yes, codes | Yes, demonstrations | Variable up to max slots | N/A | Variational objective; Gumbel boundary relaxation | Segmentation/imitation baselines | Learns both, but over-segments and observes target trajectory |
| 23 | [Compositional Foundation Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html), NeurIPS 2023 | Robot planning | LLM symbolic plan grounded via video/dynamics modules | Partial, subgoals | Yes | No deploy. | Task-dependent hierarchy | No | Module likelihoods + iterative consistency | Long-horizon planning baselines | Semantic hierarchy transfers; modules repair/co-adapt |
| 24 | [Search-Map-Search](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Search-Map-Search_A_Frame_Selection_Paradigm_for_Action_Recognition_CVPR_2023_paper.html), CVPR 2023 | Video understanding | Search best observed frame subset; distill selection map | Yes | No | Yes | Fixed selection count | Yes classifier | Downstream classification reward + map distillation | Uniform/random/selection baselines | Excellent oracle-distillation template; invalid inference information for generation |
| 25 | [GP Sensor Placement](https://www.jmlr.org/papers/v9/krause08a.html), JMLR 2008 | Active sensing | Greedy mutual-information subset | Yes | No | Field model | Fixed \(K\) | Yes | Submodular mutual information | Random/heuristic placements; \(1-1/e\) guarantee | Guarantee needs monotone submodular risk, unlikely with bad anchors |
| 26 | [Adaptive MRI Acquisition](https://arxiv.org/abs/2203.16392), 2022 manuscript | Active sensing | Sequential k-space acquisition policy | Yes | No | Partial scan | Fixed samples | Usually yes/reused | Reconstruction objective with straight-through/RL-style policy | Equispaced/nonadaptive; +1.89 SSIM points at 8×, but best policies can be nonadaptive | Strong negative precedent for assuming adaptivity helps |
| 27 | [Sequential Subset Selection](https://proceedings.neurips.cc/paper/2017/hash/8fecb20817b3847419bb3de39a609afe-Abstract.html), NeurIPS 2017 | Summarization | Facility location plus transition dynamics; integer/message passing | Yes | No | Yes | Fixed/penalized subset | N/A | Coverage + transition cost | Summarization/selection baselines | Ordered coverage transfers; observed target and no content generation |
| 28 | [Submodular Mixtures](https://openaccess.thecvf.com/content_cvpr_2015/html/Gygli_Video_Summarization_by_2015_CVPR_paper.html), CVPR 2015 | Summarization | Learned mixture of representative/salient subset objectives | Yes | No | Yes | Summary budget | N/A | Supervised submodular mixture | Summarization baselines | Coverage/diversity useful, but salience is not infiller risk |
| 29 | [AdaMAE](https://openaccess.thecvf.com/content/CVPR2023/html/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.html), CVPR 2023 | Token selection | Policy selects observed tokens with high expected recon. error | Yes | No | Yes | Fixed visible ratio | No | MAE reconstruction + policy gradient | Uniform/random masking | Selects hard observed evidence; planner must generate predictable evidence |

## 6. What transfers from adjacent fields—and what fails

### 6.1 Learned video compression: use rate–distortion logic, not codec assumptions

Learned compression supplies the cleanest analogy. [Rippel et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html) optimize a learned codec end to end; B-frame work explicitly interpolates between two reference frames; and [Neural Rate Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html) allocates a finite bitrate across frames according to spatial and temporal characteristics. This motivates a rate term, per-interval distortion, hierarchical bidirectional decoding, and a Pareto curve rather than a single arbitrary anchor count.

The analogy breaks in three places. A codec observes and transmits information about the actual frame; the proposed planner must hallucinate it. Codec distortion has a paired ground truth; conditional generation has many valid futures. Finally, perception and distortion are not interchangeable: [Blau and Michaeli's rate–distortion–perception analysis](https://proceedings.mlr.press/v97/blau19a.html) shows that optimizing expected distortion can degrade perceptual quality. Thus “anchor bits” are not just a compression rate, and pixel error alone is a poor interval cost.

Fixed GOP structures are also a warning, not merely an old baseline. B-EPIC reports that training with longer GOPs became dramatically slower without significant performance improvement. A strong fixed hierarchical schedule can be an architectural optimum when bidirectional interpolation is stable and content variation is not inferable ahead of time.

### 6.2 Robotics and hierarchical planning: choose reachable landmarks

[L3P](https://proceedings.mlr.press/v139/zhang21x.html) constructs latent landmarks connected by learned reachability, suggesting that an anchor should be valuable when the frozen infiller can reliably traverse both adjacent intervals. Visual novelty alone is analogous to choosing a spectacular but unreachable subgoal. [Compositional Foundation Models for Hierarchical Planning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html) shows the value of symbolic-language plans grounded through video, but its iterative modules repair one another; a frozen infiller removes that escape route.

TAP and [CompILE](https://proceedings.mlr.press/v97/kipf19a.html) address variable temporal abstraction. TAP favors predictability; CompILE learns differentiable boundaries and latent segment codes. CompILE's tendency toward excess boundaries when given too many slots is a direct reason to include an anchor penalty, stopping calibration, and minimum-gap constraints.

### 6.3 Active sensing: adaptivity has to beat a surprisingly strong grid

Classical sensor placement maximizes information gain. Under Gaussian-process assumptions, [Krause, Singh, and Guestrin](https://www.jmlr.org/papers/v9/krause08a.html) establish submodularity and a \(1-1/e\) greedy guarantee. That result does not automatically apply: frozen-infiller risk may be non-monotone (a wrong anchor can increase loss), correlated across intervals, and non-submodular.

The MRI result is more cautionary. [Bakker et al.](https://arxiv.org/abs/2203.16392) learn adaptive acquisition policies with a reconstruction model; adaptivity gives a reported 1.89-point SSIM gain at 8× acceleration, yet the best policies can be explicitly non-adaptive and equispaced. When the reconstruction prior already handles missing data well, observation-dependent allocation is unnecessary. The video analogue is a strong infiller whose error depends mostly on gap length. Before building a planner, estimate how much residual risk is predictable from the permitted conditions after controlling for gap.

Training with privileged information can still help. [Sidekick Policy Learning](https://openaccess.thecvf.com/content_ECCV_2018/html/Santhosh_Kumar_Ramakrishnan_Sidekick_Policy_Learning_ECCV_2018_paper.html) uses full observability during training to shape a partially observed exploration policy. This supports oracle schedule distillation, provided evaluation never gives target-video features to the deployable planner.

### 6.4 Subset selection: coverage and diversity are ingredients, not the objective

Submodular mixtures, sequential facility location, determinantal point processes, and query-focused summarization formalize coverage, diversity, and order. [Sequential subset selection](https://proceedings.neurips.cc/paper/2017/hash/8fecb20817b3847419bb3de39a609afe-Abstract.html) is particularly relevant because its cost includes transition dynamics rather than treating frames as an unordered set. [Query-focused summarization](https://openaccess.thecvf.com/content_cvpr_2017/html/Sharghi_Query-Focused_Video_Summarization_CVPR_2017_paper.html) also shows how language can condition a subset.

But summarizers inspect the video. They favor representative or interesting frames, not frames a condition-only generator can predict. DPP diversity can also fight the intended objective: mutually different anchors may be difficult for a frozen infiller to connect. Coverage/diversity should be regularizers or oracle features, never treated as proof of deployability.

### 6.5 Token pruning: efficiency mechanisms preserve observed information

[TokenLearner](https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html), [DynamicViT](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html), and [Token Merging](https://openreview.net/pdf?id=JroZRaRw7Eu) learn or infer compact token sets. They demonstrate useful gates, staged pruning, and the advantage of merging over dropping. Yet their selector sees the tokens it compresses, and their goal is to preserve task features, not to generate unknown visual states. The transferable lesson is architectural—ordered gates, differentiable masks, compute regularization—not informational.

### 6.6 Speech, music, and language: separate what from when

Language generation provides a useful two-stage decomposition. [Hua and Wang](https://aclanthology.org/D19-1055/) first select and order keyphrases in a sentence-level content plan, then perform surface realization; [Trisedya et al.](https://aclanthology.org/2021.findings-emnlp.166/) explicitly combine content selection and plan generation. This maps cleanly to “which events and in what order” before visual realization. It does not supply physical duration: textual order is only a partial temporal plan.

Speech supplies timing evidence. [Dai et al.](https://www.isca-archive.org/interspeech_2022/dai22_interspeech.html) infer prosodic boundary labels from paired text and audio and outperform text-only boundary baselines, illustrating why sparse audio/prosody can make boundary timing identifiable when text alone cannot. In music-conditioned video, beat or motion peaks can likewise propose boundaries, but peak detection is not a rate-aware visual plan and may over-anchor repetitive rhythms.

Sequence models also face mismatch between local teacher-forced likelihood and deployment-time quality. [Minimum Risk Training for Neural Machine Translation](https://aclanthology.org/P16-1159/) directly optimizes sentence-level metrics over sampled candidates. [Stochastic Computation Graphs](https://proceedings.neurips.cc/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html) formalizes score-function and pathwise estimators. These support schedule-level training against a black-box frozen infiller after supervised warm-start. They also predict the main difficulty: variance and expensive candidate generation.

### 6.7 Transferability and implementation-risk summary

| Family | Transfers directly | Does not transfer | Required video-specific modification | Expected implementation risk |
|---|---|---|---|---|
| Learned compression | Rate–distortion frontier, bidirectional segments, global budget, rate control | Codec observes exact frames and sends residual bits; distortion has one target | Replace transmitted rate with generated-anchor cost; use perceptual/semantic/stochastic risk; model predicted-anchor error | Medium: objectives are mature, but the information and perception gaps are fundamental |
| Robotics/control | Reachable subgoals, temporal abstraction, receding-horizon recovery | Environment can provide feedback; subgoal states need not be photorealistic tokens | Define reachability through frozen-infiller failure; add identity/event state; plan open-loop or explicitly permit feedback | High: long-horizon costs are non-additive and failures compound |
| Active sensing | Value of information, calibrated uncertainty, budgeted acquisition, oracle teachers | Policy observes partial target measurements | Predict value from conditions only; distinguish epistemic risk from unavoidable future ambiguity | High: oracle value may be unidentifiable; uniform can remain optimal |
| Subset selection | Coverage, diversity, event boundaries, transition-aware facility location | Selector sees all candidate frames; selected content is exact | Make usefulness conditional on anchor predictability and infiller cost; treat target selection as teacher only | Medium–high: easy oracle, potentially large distillation gap |
| Discrete selection | Hard constraints, gates/top-\(k\), policy gradients, minimum risk | Relaxed selectors may use soft mixtures unavailable at inference | Enforce monotone integer times/gaps/EOS; fine-tune on hard schedules and black-box rollouts | High for end-to-end; medium for oracle distillation |
| Token pruning/memory | Compute penalties, staged selection, merging, compressed history | Retained tokens are observed and already meaningful | Generate missing anchor tokens; preserve explicit timestamps and error calibration | Medium architecturally, high semantically |
| Speech/music/language | Boundary cues, latent alignment, “what/order” plan before realization | Linguistic units do not determine physical motion duration or visual reachability | Fuse prosody/audio/event intervals with gap-aware visual cost; counterfactually test timing response | Medium when signals are aligned; high for text-only timing |

### 6.8 Critical synthesis checkpoint

**Recurring successful patterns.** Across video generation and adjacent fields, robust systems (i) establish global endpoints or coarse state before local synthesis, (ii) represent relative time/gap explicitly, (iii) allocate resources under an explicit rate constraint, (iv) choose states for downstream reachability rather than novelty alone, and (v) expose downstream models to predicted or corrupted context. Hierarchical midpoint schedules remain strong because they satisfy gap bounds, reuse bidirectional models, and reduce search.

**Contradictory findings.** KeyVID's fixed-count experiment suggests modest value from motion-aware nonuniform anchors, while adaptive MRI shows that a learned acquisition policy can rationally collapse to equispaced sampling. TAP favors easy-to-predict times; AdaMAE favors hard, high-reconstruction-error observations. The contradiction disappears once information is considered: an observed-data encoder should retain hard evidence, whereas a generator should request evidence it can predict and that the infiller needs. Ground-truth keyframes consistently help video generation, but the large WACV 2025 oracle gap shows that this does not imply predicted keys will help.

**Negative evidence.** CompILE can over-segment when excess boundary slots are available; B-EPIC finds much slower longer-GOP training with little performance gain; learned MRI policies can be non-adaptive; and KeyVID's direct adaptive result is modest, fixed-rate, and unreviewed after rejection. No peer-reviewed paper in this review provides the decisive comparison of learned condition-only placement versus uniform placement with the same predicted anchor content, frozen infiller, and equal compute.

**Common evaluation mistakes.** The most serious are target-video leakage into the selector; unequal anchor counts, maximum gaps, or denoising steps; reporting only ground-truth-anchor results; fine-tuning the “frozen” infiller for the proposed schedule but not baselines; selecting the best of more samples; treating FVD or token cross-entropy as sufficient; ignoring boundary failures; and presenting attention weights as causal grounding. Another subtle error is counting endpoints for one method but not another.

**Unresolved questions.** Can frozen-infiller failure be predicted and calibrated from conditions before frames exist? Is pairwise interval cost accurate enough for DP, or is global identity state essential? Which discrete-token errors are harmless versus catastrophic? How should a stochastic planner represent multiple valid event timings? When is closed-loop replanning permitted? Can one schedule policy generalize across infillers, durations, frame rates, and tokenizers? These questions are more consequential than choosing among similar top-\(k\) relaxations.

## 7. A formulation that matches deployment

Let \(c=(x,s,T)\) contain text \(x\), sparse temporal signals \(s\), and requested duration \(T\). A schedule policy \(p_\theta(\tau\mid c)\) emits ordered indices \(\tau=(t_0,\ldots,t_M)\); a content model \(q_\phi(z_m\mid c,t_m,z_{<m})\) emits discrete visual tokens. The frozen infiller \(F_\psi\), with fixed \(\psi\), samples each interval:

\[
 \hat y_{t_m:t_{m+1}}\sim
 F_\psi(\hat z_m,\hat z_{m+1},\Delta_m,c;\epsilon_m),
 \qquad \Delta_m=t_{m+1}-t_m.
\]

The generated intervals are stitched into \(\hat y\). A deployment-aligned objective is

\[
\begin{aligned}
\min_{\theta,\phi}\;\mathbb E_{(c,y),\tau,\hat z,\epsilon}\big[&
L_{\text{video}}(\hat y,y,c)
+\alpha L_{\text{event}}(\hat y,x,s)
+\beta L_{\text{boundary}}(\hat y,\tau)\\
&+\mu\sum_{m=0}^{M}L_{\text{anchor}}(\hat z_m,y_{t_m},c)
+\lambda(M-1)\big],
\end{aligned}
\]

subject to

\[
t_0=0,\quad t_M=T-1,\quad
G_{\min}\le t_{m+1}-t_m\le G_{\max},\quad M-1\le K_{\max}.
\]

Endpoints should be counted consistently: throughout this review, “anchor budget” means *interior* anchors unless explicitly stated. If the system must also generate the endpoints, their compute and error are reported separately even though they are mandatory.

The terms have distinct roles:

- \(L_{\text{video}}\) measures whole-sample perceptual, motion, semantic, and possibly reference loss over actual predicted anchors and stochastic infiller samples.
- \(L_{\text{event}}\) evaluates whether named events occur, in the right order and temporal windows. It must not be replaced by text–video similarity alone.
- \(L_{\text{boundary}}\) penalizes jumps in appearance, flow, trajectory, or identity across stitched interval boundaries.
- \(L_{\text{anchor}}\) prices how hard the selected content is for the anchor generator. A schedule that is ideal with ground-truth frames can be bad when its anchors are unpredictable.
- \(\lambda\) traces a rate–quality frontier. The hard \(G_{\max}\) constraint is a safety property, not a substitute for the rate term.

For genuinely multi-modal futures, a single reference \(y\) can punish valid generations. Report both paired reconstruction-style losses, where appropriate, and reference-free conditional/human scores over multiple samples. Risk-sensitive variants such as CVaR can penalize rare catastrophic segment failures:

\[
L_{\text{risk}}=\operatorname{CVaR}_{q}
\left[L_{\text{video}}+\alpha L_{\text{event}}+\beta L_{\text{boundary}}\right].
\]

This is preferable to optimizing only the mean when one bad interval invalidates the video.

### 7.1 A tractable offline oracle

For an observed training video, define an empirical edge cost for placing anchors at \(i<j\):

\[
C_{ij}=\mathbb E_{\tilde z_i,\tilde z_j,\epsilon}
\left[
L_{ij}\big(F_\psi(\tilde z_i,\tilde z_j,j-i,c;\epsilon),y_{i:j}\big)
\right]+\kappa P_{ij}^{\text{fail}}.
\]

Crucially, \(\tilde z\) should be sampled from the actual anchor predictor—or at least a calibrated corruption model—not always taken from ground truth. Let \(P_j\) be the prediction cost for anchor \(j\). With a fixed \(K\), an additive oracle is

\[
D[k,j]=P_j+\min_{i:\,G_{\min}\le j-i\le G_{\max}}
\{D[k-1,i]+C_{ij}\}.
\]

The answer is \(D[K+2,T-1]\) when two endpoints and \(K\) interior anchors are counted. Complexity is \(O(KTG_{\max})\) after edge costs have been cached. With a Lagrangian rather than fixed \(K\), shortest path in a directed acyclic graph yields the Pareto frontier over \(\lambda\). Event order can be handled by augmenting DP state with a small event automaton; top-\(L\) schedules can be globally reranked using a whole-video coherence model.

The oracle has two possible meanings and they should not be conflated:

1. **GT-content oracle:** \(C_{ij}\) uses true endpoint frames. This measures the frozen infiller's theoretical placement sensitivity.
2. **Predicted-content oracle:** \(C_{ij}\) uses predicted/noisy endpoint tokens. This measures placement sensitivity of the deployable content interface.

If only the first beats uniform, anchor prediction—not scheduling—is the real research problem. If neither beats uniform, the adaptive-planning hypothesis is falsified for that infiller/data regime.

### 7.2 Identifiability limit

Suppose two target videos share identical \(c\) but place a transition at different times. No condition-only planner can know which instance-specific time will occur. It can learn the population distribution \(p(t\mid c)\), not the hidden realization. Adaptive placement can beat uniform only when timing difficulty is predictable from text/sparse signals, when stochastic schedule sampling improves expected risk, or when the planner is allowed feedback. This limit should be diagnosed by measuring oracle predictability from \(c\), not attributed to model capacity.

## 8. Optimization choices for discrete positions and variable cardinality

| Method | Cardinality | Gradient | Strength | Main failure here |
|---|---|---|---|---|
| Autoregressive gap tokens + EOS | Variable | Token likelihood; later policy gradient | Naturally ordered, hard constraints can be masked | Exposure error; stopping calibration; sequential latency |
| Hazard/continue process | Variable | Likelihood or survival loss | Clean variable-length semantics | Local hazards can miss global budget and event interactions |
| [Hard-concrete \(L_0\) gates](https://arxiv.org/abs/1712.01312) | Variable | Biased pathwise | Direct expected-cardinality penalty | Independent gates do not enforce order/gap constraints; soft/hard mismatch |
| [Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) | Usually fixed choices | Biased pathwise | Simple warm-start for categorical times | Temperature instability and duplicate/invalid positions |
| [SOFT top-\(k\)](https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html) | Fixed \(K\) | Entropic-OT relaxation | Ordered differentiable subset | Fixed count; Sinkhorn cost; relaxed optimum may not survive hard rounding |
| [SoftSort](https://proceedings.mlr.press/v119/prillo20a.html) | Fixed items/count | Differentiable sorting | Efficient order relaxation | Still needs hard selection/cardinality and cannot expose black-box infiller gradients |
| REINFORCE/self-critical | Variable/hard | Unbiased in principle | Optimizes black-box frozen-infiller reward | High variance and expensive rollouts; reward hacking |
| Minimum-risk candidate training | Variable/hard | Sample-weighted score gradient | Direct sequence-level metric alignment | Candidate coverage and metric quality limit learning |
| DP/search oracle + distillation | Fixed or variable | Supervised student | Stable, interpretable lower/upper bounds | Expensive target-video teacher; oracle decisions may not be inferable from \(c\) |

The recommended default is **masked autoregressive gap tokens plus EOS**, warm-started by oracle distillation. It guarantees monotone time, can mask gaps outside \([G_{\min},G_{\max}]\), and can represent variable duration. Use minimum-risk fine-tuning on hard sampled schedules if the oracle study demonstrates headroom. Differentiable relaxations are useful for pretraining or ablations, not as the sole evidence that a discrete deployment policy is optimized.

Anchor content can be trained with cross-entropy over a frozen visual tokenizer's codes, but token accuracy should be weighted or supplemented by decoded perceptual/semantic loss. Equivalent-looking code errors may have very different effects on the infiller. Timestamp embeddings should combine absolute frame/time, normalized phase \(t/(T-1)\), and relative gap; event embeddings should retain interval boundaries, not just event order.

## 9. Four concrete system designs

### Design A: fixed schedule, learned discrete content

**Architecture.** Choose a strong uniform or midpoint-hierarchical schedule satisfying \(G_{\max}\). A condition encoder fuses text and sparse temporal signals. A timestamp-aware autoregressive transformer predicts VQ tokens for each anchor. The frozen infiller processes adjacent pairs.

**Training.** Cross-entropy on target anchor codes; decoded DINO/LPIPS or task-semantic auxiliary loss; replace a growing fraction of previous ground-truth anchors with model predictions; apply VideoAuteur-style noise/mask/shuffle perturbations that match measured predictor errors.

**Cost.** Lowest training and inference complexity. No schedule sampling or infiller rollout is required for initial supervised training.

**Advantage.** It is the indispensable control. It may be the best production system when timing is weakly identifiable or infiller risk is mainly a function of gap.

**Risk.** Uniform anchors waste budget in easy intervals and may straddle abrupt events. Improving content can mask the absence of genuine schedule learning.

### Design B: offline predicted-anchor oracle DP, then distillation

**Architecture.** Precompute edge costs by running the frozen infiller over feasible training-video intervals. Include separate GT-content and predicted/noisy-content cost tables. Solve constrained shortest paths/DP for multiple \(K\) or \(\lambda\). Train a condition-only schedule transformer to emit gap tokens and EOS, plus the anchor-content transformer from Design A.

**Training.** Distill one or several near-optimal schedules with likelihood, cost-sensitive classification, or distribution matching. Use oracle cost differences as weights so harmless deviations are not punished like catastrophic ones. Add DAgger-style data: roll out the student, query the oracle from reached positions, and train on recovery actions. Finally train content on student-selected times.

**Cost.** The expensive stage is \(O(NTG_{\max})\) frozen-infiller interval evaluations, multiplied by stochastic samples; caching endpoint latents and using a learned edge-cost surrogate can reduce this. DP itself is cheap.

**Advantage.** Gives the cleanest scientific answer about whether adaptive schedules contain usable signal and produces interpretable supervision without differentiating through the infiller.

**Risk.** Edge additivity can miss global coherence. A teacher that uses target-video information can be impossible for the student to imitate. Training on GT-content edge costs alone optimizes the wrong interface.

### Design C: end-to-end minimum-risk planner with a frozen infiller

**Architecture.** An autoregressive policy emits \((\Delta_m,z_m)\) blocks until EOS, with endpoint and gap masks enforcing feasibility. For each condition, sample \(B\) complete hard schedules; generate anchors and full videos with the frozen infiller. A learned value/edge model supplies a baseline and optional cheap prescreening.

**Training.** Start from Design B. Apply minimum-risk or self-critical sequence training using whole-video, event-order, boundary, and rate reward. Use multiple infiller seeds. Normalize rewards within the candidate set, clip extreme gradients, and periodically evaluate with held-out metrics to detect reward hacking.

**Cost.** Approximately \(B\) complete generations per example, often the dominant project expense. A two-stage funnel—many cheap cost-model candidates, few true rollouts—is likely necessary.

**Advantage.** Directly matches the hard discrete deployment policy and can optimize a non-differentiable frozen module and global, non-additive metrics.

**Risk.** High-variance gradients, metric exploitation, mode collapse toward easy/static anchors, and severe compute demand. This should not be the first design attempted.

### Design D: safe uniform scaffold with learned insertions

**Architecture.** Begin with the sparsest uniform scaffold satisfying a conservative \(G_{\max}\). A condition-only risk predictor scores each interval. Repeatedly split the highest-risk interval—predicting one new discrete anchor—until a budget, uncertainty threshold, or marginal-gain threshold is reached.

**Training.** Distill marginal gains from the predicted-content DP oracle; calibrate the stop rule on held-out full infiller rollouts. Midpoint is the default split, with a small learned offset when evidence supports it.

**Cost.** Logically simple, anytime, and naturally variable-rate; inference can stop early. Its schedule search is cheaper than free autoregression.

**Advantage.** Preserves a worst-case gap guarantee and degrades gracefully to uniform. It is the most credible production compromise.

**Risk.** Greedy splits are suboptimal when event costs interact, and early wrong splits consume the budget. Confidence calibration is essential.

### 9.1 Interface tests before planner training

Treat the frozen infiller as an empirical channel. Before building any policy, map:

- quality and failure probability versus gap length;
- sensitivity to endpoint-token error, semantic contradiction, spatial misalignment, and time-index jitter;
- whether the model uses the supplied gap/absolute time or assumes its training interpolation ratio;
- cross-segment identity drift when intervals are generated independently;
- support for stochastic valid paths between identical endpoints.

If the infiller cannot consume predicted discrete anchors in-distribution, a planner cannot repair that contract. An adapter may be necessary, but training one changes the claim from “frozen infiller” to “frozen core with learned interface” and should be named honestly.

## 10. The decisive equal-budget experiment

### 10.1 Hypotheses and pre-registered gates

The main hypothesis is:

> At the same interior-anchor count and generation compute, a condition-only learned schedule with predicted discrete anchors improves whole-video temporal/semantic quality over a strong uniform schedule with the same anchor-content model and frozen infiller.

Run two gates before end-to-end planner training:

1. **Oracle-headroom gate.** At equal \(K\), the predicted-content DP oracle must significantly outperform uniform placement. If it does not, stop: the fixed schedule is adequate for this infiller/regime.
2. **Identifiability gate.** A condition-only model must predict oracle marginal gains or next splits better than duration/event-frequency baselines on held-out videos. If it cannot, the available conditioning lacks timing information.

Only a pass on both gates justifies Design B or C.

### 10.2 Factorial comparison

Keep the tokenizer, anchor-content model, frozen infiller, prompts, random seeds, duration, resolution, and total infiller sampling steps identical. Cross the following factors:

| Placement | Anchor content | Role |
|---|---|---|
| Uniform | Ground truth | Infiller upper-bound baseline |
| Uniform | Predicted | **Primary fixed baseline** |
| Random feasible, matched gap histogram | Predicted | Detects gains due merely to nonuniformity or regularization |
| Midpoint hierarchy | Predicted | Strong fixed bidirectional baseline |
| Condition-only heuristic: event boundaries/audio peaks/predicted motion | Predicted | Tests whether learning is needed |
| Target-video motion/change-point heuristic | Ground truth and predicted | Privileged heuristic upper bound; never call it deployable |
| DP oracle using GT-content edge costs | Ground truth | Pure placement/infill headroom |
| DP oracle using predicted-content edge costs | Predicted | Realistic oracle headroom |
| Learned distilled schedule | Ground truth | Isolates placement quality |
| Learned distilled schedule | Predicted | **Primary proposed system** |
| Minimum-risk or greedy-insertion schedule | Predicted | Tests benefit beyond distillation |

Evaluate multiple rates, for example \(K\in\{0,1,2,4,8\}\) interior anchors for a fixed duration and equivalent anchors-per-second for variable durations. Always report mean gap, 95th-percentile gap, and maximum gap. Match the number of generated candidates and infiller denoising steps; if the learned method consumes extra planner compute, report latency and energy separately.

### 10.3 Data regimes

Use at least three strata rather than one aggregate benchmark:

- **Smooth/stationary:** gradual motion, one event, limited occlusion. This is the regime in which uniform should be hardest to beat.
- **Transition-rich:** multiple ordered actions, entrances/exits, contact changes, camera cuts or rapid motion. This tests allocation around true difficulty.
- **Timing-identifiable:** audio beats, timestamped event phrases, sparse pose/trajectory observations, or supplied temporal tags. This tests whether sparse signals causally drive placement.

Include held-out duration ranges and compositions of familiar events. A controlled diagnostic set should pair prompts with the same nouns but different verbs/order (for example, “open then pour” versus “pour then open”), and should shift event times without changing event identity. Natural video can supply realism; controlled or procedurally composed sequences supply causal ground truth.

### 10.4 Metrics: measure the chain, not just the final frame distribution

**Anchor layer**

- discrete-token negative log-likelihood and top-\(k\) accuracy;
- decoded LPIPS/DINO or equivalent perceptual distance;
- semantic identity/action accuracy at the selected time;
- calibration: predicted anchor uncertainty versus actual infiller degradation.

**Interval and boundary layer**

- optical-flow/trajectory error where references are meaningful;
- flicker, identity drift, endpoint adherence, and a learned segment-failure classifier;
- discontinuity in appearance, motion, and identity across stitched boundaries;
- failure rate as a function of gap and endpoint-token corruption.

**Whole-video layer**

- FVD only as one distributional measure, with sample count and confidence intervals;
- relevant dimensions from [VBench](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) and [EvalCrafter](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.html);
- event presence, order, start/end localization mAP or IoU, and temporal consistency, informed by [Ref4D-VideoBench](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html);
- blinded human pairwise judgments of prompt faithfulness, temporal logic, realism, and preference;
- rate–quality curves, area under the curve, infiller calls, wall-clock latency, and memory.

The main statistical unit is the prompt/condition, not the generated sample. Use multiple seeds per condition, paired bootstrap confidence intervals, and a hierarchical or mixed-effects analysis across conditions and data regimes. Correct for the limited set of pre-registered primary endpoints; treat the remainder as diagnostic.

### 10.5 Counterfactual tests for causal use of timing signals

Do not infer causality from attention visualization; [Jain and Wallace](https://aclanthology.org/N19-1357/) show that attention weights can be weakly related to feature importance and that alternative attentions can preserve predictions. Instead intervene:

1. swap the order of two event phrases while preserving vocabulary;
2. time-shift the sparse signal while holding text fixed;
3. mute each modality separately;
4. replace the signal with a constant or shuffled sequence;
5. jitter, delete, or contradict one predicted anchor;
6. permute a learned schedule while retaining the same anchor count/content generator;
7. compare prompts with identical nouns but different action order.

A valid planner should move schedule mass with the shifted event, preserve irrelevant regions, and lose its advantage when the informative signal is destroyed. If its schedule barely changes, any quality gain probably comes from content-model capacity or regularization rather than temporal planning.

### 10.6 Decision table

| Result | Interpretation | Next action |
|---|---|---|
| Neither GT-content nor predicted-content oracle beats uniform | Frozen infiller is schedule-insensitive or metric is inadequate | Keep fixed schedule; improve infiller/metric, not planner |
| GT-content oracle wins; predicted-content oracle does not | Anchor prediction error destroys placement benefit | Improve content/interface robustness before scheduling |
| Predicted-content oracle wins; condition-only student cannot imitate it | Oracle uses target-only information; timing is not identifiable | Add temporal conditioning/feedback or accept fixed schedule |
| Student wins with GT content but not predicted content | Exposure/content quality is the bottleneck | Train on predicted anchors and calibrate uncertainty |
| Learned schedule wins only with more compute or larger max gap | Comparison is confounded | Re-run with equal compute and safety constraints |
| Learned schedule wins mainly on transition-rich/identifiable strata | Conditional value is real but regime-specific | Deploy a gated hybrid, not universal adaptation |
| Learned schedule significantly beats uniform across rates with predicted content | Core claim supported | Proceed to minimum-risk/global-coherence refinement |

A useful success criterion is not a cherry-picked FVD change. Require a statistically reliable improvement over uniform in event/temporal human preference and segment failure at the same rate, with no material loss in visual quality, plus recovery of a meaningful fraction (for example, at least 30%) of the predicted-content oracle gap. The exact threshold should be set before experiments.

## 11. When fixed anchors are expected to win

The literature and formulation identify seven concrete cases:

1. **Stationary risk:** after conditioning on gap length, infiller failure is almost independent of content or temporal signals.
2. **Weak timing information:** prompts describe events but not when they occur, and sparse signals do not disambiguate phase.
3. **Short horizons:** the maximum safe gap already covers most or all of the clip.
4. **Uniform-trained infiller:** the frozen model is calibrated only on a particular interpolation ratio or regular frame grid.
5. **High anchor-prediction noise:** adaptive policies select transitions precisely where discrete content is hardest to predict.
6. **Rate dominated by safety:** a tight \(G_{\max}\) constraint nearly determines the schedule, leaving little allocation freedom.
7. **Planner overhead/variance:** schedule sampling, search, or stochastic errors cost more than the marginal infilling gain.

These are not embarrassing edge cases. They are plausible operating regimes, supported by non-adaptive learned MRI policies and persistent fixed GOP hierarchies in compression. A publishable result may be a calibrated *gating rule* that predicts when adaptation is worthwhile, rather than a claim that learned schedules always dominate.

## 12. Recommended research sequence

1. **Freeze and characterize the infiller.** Produce gap/corruption/failure curves and verify explicit time conditioning.
2. **Build Design A.** This establishes the strongest fixed baseline and measures predicted-anchor exposure bias.
3. **Construct both DP oracles.** Compare GT-content and predicted-content schedules over equal budgets.
4. **Run the two gates.** Abort adaptive planning if there is no headroom or no condition-level identifiability.
5. **Distill Design B and evaluate the full factorial.** Do not use target-video motion at deployment.
6. **Add Design D for safe variable rate.** Calibrate stopping and worst-case gaps.
7. **Attempt Design C only if residual oracle headroom remains.** Use hard sampled schedules and global metrics.
8. **Stress test causality and distribution shift.** Include unseen lengths, event compositions, signal jitter, and corrupted anchors.

## 13. Conclusion

The proposed direction is best understood as **rate-constrained conditional subgoal planning through a frozen generative channel**. Existing video work establishes all of its pieces separately: discrete anchor generation, endpoint-conditioned infilling, narrative keyframes, explicit event timing, and long-horizon hierarchy. Adjacent fields contribute rate–distortion objectives, reachability-aware landmarks, oracle subset distillation, and hard discrete risk optimization. What they do not establish is that a condition-only model can predict where a frozen infiller will need help *and* generate reliable discrete content there.

The main scientific contribution should therefore be a decomposition, not merely a new architecture: quantify schedule headroom; separate placement from content; replace target-video selection with condition-only inference; price anchors and maximum gap; train under predicted endpoints; and demonstrate causal response to timing signals. If the predicted-content oracle cannot beat a uniform grid, the honest conclusion is that sparse-anchor planning is unnecessary for that regime. If it can, and a condition-only student closes a substantial part of the gap under equal budget, the result would fill a genuine gap in the literature.

## Bibliographic source index

The links below point to canonical proceedings or project records wherever available. “Preprint” and “rejected submission” labels are intentional; those works should not be weighted as peer-reviewed evidence.

### Core video generation, infilling, and temporal control

- Jayaraman, D.; Ebert, F.; Efros, A. A.; Levine, S. “Time-Agnostic Prediction: Predicting Predictable Video Frames.” ICLR 2019. [OpenReview](https://openreview.net/forum?id=SyzVb3CcFX); [arXiv:1808.07784](https://arxiv.org/abs/1808.07784).
- Huang, Hsin-Ping; Su, Yu-Chuan; Yang, Ming-Hsuan. “Generating Long-Take Videos via Effective Keyframes and Guidance.” WACV 2025, pp. 3709–3720. [CVF proceedings](https://openaccess.thecvf.com/content/WACV2025/html/Huang_Generating_Long-Take_Videos_via_Effective_Keyframes_and_Guidance_WACV_2025_paper.html).
- Blattmann et al. “Align Your Latents: High-Resolution Video Synthesis With Latent Diffusion Models.” CVPR 2023, pp. 22563–22575. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.html).
- Voleti et al. “MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation.” NeurIPS 2022. [OpenReview](https://openreview.net/forum?id=hX5Ia-ION8Y).
- Chen et al. “SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction.” ICLR 2024. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html).
- Wang et al. “Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation.” ICLR 2025. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4bbdef62653d8088717640e7660a1ebb-Abstract-Conference.html).
- Wu et al. “Mind the Time: Temporally-Controlled Multi-Event Video Generation.” CVPR 2025, pp. 23989–24000. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Mind_the_Time_Temporally-Controlled_Multi-Event_Video_Generation_CVPR_2025_paper.html).
- Xiao et al. “VideoAuteur: Towards Long Narrative Video Generation.” ICCV 2025. [CVF paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_VideoAuteur_Towards_Long_Narrative_Video_Generation_ICCV_2025_paper.pdf).
- Xiao et al. “Captain Cinema: Towards Short Movie Generation.” ICLR 2026 poster. [OpenReview](https://openreview.net/forum?id=zlNZBxQZIC).
- Zhang et al. “STAGE: Storyboard-Anchored Generation for Cinematic Multi-shot Narrative.” CVPR 2026, pp. 659–669. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_STAGE_Storyboard-Anchored_Generation_for_Cinematic_Multi-shot_Narrative_CVPR_2026_paper.html).
- Wang et al. “Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation.” ICLR 2026 desk-rejected submission, **unreviewed**. [OpenReview](https://openreview.net/forum?id=oijKOpfSmX).
- Yin et al. “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation.” arXiv:2303.12346, 2023, **preprint**. [arXiv](https://arxiv.org/abs/2303.12346).
- Arkhipkin, V.; Shaheen, Z.; Vasilev, V.; Dakhova, E.; Kuznetsov, A.; Dimitrov, D. “FusionFrames: Efficient Architectural Aspects for Text-to-Video Generation Pipeline.” arXiv:2311.13073, 2023, **preprint**. [arXiv](https://arxiv.org/abs/2311.13073).
- Yu et al. “MAGVIT: Masked Generative Video Transformer.” CVPR 2023. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html).
- Villegas et al. “Phenaki: Variable Length Video Generation From Open Domain Textual Descriptions.” ICLR 2023. [OpenReview](https://openreview.net/forum?id=vOEXS39nOF).
- Kondratyuk et al. “VideoPoet: A Large Language Model for Zero-Shot Video Generation.” ICML 2024, PMLR 235. [PMLR](https://proceedings.mlr.press/v235/kondratyuk24a.html).
- Zhang et al. “Frame Context Packing and Drift Prevention for Long Video Generation.” NeurIPS 2025 spotlight. [OpenReview](https://openreview.net/forum?id=J8JCF64aEn).
- Zhou et al. “Taming Teacher Forcing for Masked Autoregressive Video Generation.” CVPR 2025. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html).

### Compression, planning, sensing, and selection

- Rippel et al. “Learned Video Compression.” ICCV 2019. [CVF proceedings](https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html).
- Pourreza et al. “Extending Neural P-Frame Codecs for B-Frame Coding.” ICCV 2021. [CVF paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Pourreza_Extending_Neural_P-Frame_Codecs_for_B-Frame_Coding_ICCV_2021_paper.pdf).
- Zhang et al. “Neural Rate Control for Learned Video Compression.” ICLR 2024. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html).
- Blau, Y.; Michaeli, T. “Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff.” ICML 2019, PMLR 97. [PMLR](https://proceedings.mlr.press/v97/blau19a.html).
- Zhang, L.; Yang, G.; Stadie, B. C. “World Model as a Graph: Learning Latent Landmarks for Planning.” ICML 2021, PMLR 139. [PMLR](https://proceedings.mlr.press/v139/zhang21x.html).
- Kipf et al. “CompILE: Compositional Imitation Learning and Execution.” ICML 2019, PMLR 97. [PMLR](https://proceedings.mlr.press/v97/kipf19a.html).
- Ajay et al. “Compositional Foundation Models for Hierarchical Planning.” NeurIPS 2023. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html).
- Zhao et al. “Search-Map-Search: A Frame Selection Paradigm for Action Recognition.” CVPR 2023. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Search-Map-Search_A_Frame_Selection_Paradigm_for_Action_Recognition_CVPR_2023_paper.html).
- Krause, A.; Singh, A.; Guestrin, C. “Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies.” JMLR 9, 2008. [JMLR](https://www.jmlr.org/papers/v9/krause08a.html).
- Bakker et al. “On Learning Adaptive Acquisition Policies for Undersampled Multi-Coil MRI Reconstruction.” 2022 manuscript. [arXiv:2203.16392](https://arxiv.org/abs/2203.16392).
- Ramakrishnan and Grauman. “Sidekick Policy Learning for Active Visual Exploration.” ECCV 2018. [ECCV proceedings](https://openaccess.thecvf.com/content_ECCV_2018/html/Santhosh_Kumar_Ramakrishnan_Sidekick_Policy_Learning_ECCV_2018_paper.html).
- Elhamifar and Kaluza. “Subset Selection and Summarization in Sequential Data.” NeurIPS 2017. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/8fecb20817b3847419bb3de39a609afe-Abstract.html).
- Gygli et al. “Video Summarization by Learning Submodular Mixtures of Objectives.” CVPR 2015. [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2015/html/Gygli_Video_Summarization_by_2015_CVPR_paper.html).
- Bandara et al. “AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning With Masked Autoencoders.” CVPR 2023. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.html).

### Discrete optimization, risk training, and evaluation

- Hua, X.; Wang, L. “Sentence-Level Content Planning and Style Specification for Neural Text Generation.” EMNLP-IJCNLP 2019, pp. 591–602. [ACL Anthology](https://aclanthology.org/D19-1055/).
- Trisedya, B. D.; Wang, X.; Qi, J.; Zhang, R.; Cui, Q. “Grouped-Attention for Content-Selection and Content-Plan Generation.” Findings of EMNLP 2021, pp. 1935–1944. [ACL Anthology](https://aclanthology.org/2021.findings-emnlp.166/).
- Dai, Z.; Yu, J.; Wang, Y.; Chen, N.; Bian, Y.; Li, G.; Cai, D.; Yu, D. “Automatic Prosody Annotation with Pre-Trained Text-Speech Model.” Interspeech 2022, pp. 5513–5517. [ISCA Archive](https://www.isca-archive.org/interspeech_2022/dai22_interspeech.html).
- Jang, Gu, and Poole. “Categorical Reparameterization with Gumbel-Softmax.” ICLR 2017. [OpenReview paper](https://openreview.net/pdf?id=rkE3y85ee).
- Louizos, Welling, and Kingma. “Learning Sparse Neural Networks through \(L_0\) Regularization.” ICLR 2018. [arXiv:1712.01312](https://arxiv.org/abs/1712.01312).
- Xie et al. “Differentiable Top-k with Optimal Transport.” NeurIPS 2020. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html).
- Prillo and Eisenschlos. “SoftSort: A Continuous Relaxation for the argsort Operator.” ICML 2020, PMLR 119. [PMLR](https://proceedings.mlr.press/v119/prillo20a.html).
- Schulman et al. “Gradient Estimation Using Stochastic Computation Graphs.” NeurIPS 2015. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html).
- Shen et al. “Minimum Risk Training for Neural Machine Translation.” ACL 2016. [ACL Anthology](https://aclanthology.org/P16-1159/).
- Huang et al. “VBench: Comprehensive Benchmark Suite for Video Generative Models.” CVPR 2024. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html).
- Liu et al. “EvalCrafter: Benchmarking and Evaluating Large Video Generation Models.” CVPR 2024. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.html).
- Wei et al. “Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models.” CVPR 2026. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html).
- Jain, S.; Wallace, B. C. “Attention is not Explanation.” NAACL-HLT 2019. [ACL Anthology](https://aclanthology.org/N19-1357/).

# Joint Sparse-Observation Planning for a Frozen Sequence Completer

## An architecture-oriented critical literature review

**Evidence cutoff:** 26 July 2026  
**Purpose:** enumerate defensible architecture options and the experiments that distinguish them; this is a selective critical review, not a bibliometric survey.

### Executive decision

The most accurate established name for the problem is:

> **constrained semi-Markov rate-distortion planning with learned sparse discrete observations and a frozen neural decoder**

Other useful names expose different parts of the problem:

- **active observation scheduling** describes choosing when to observe, but is incomplete because this planner predicts observation values instead of measuring them;
- **task-aware or semantic compression** describes optimizing sparse messages for a downstream reconstruction task;
- **amortized decision-focused inference through a frozen differentiable program** describes training the planner with the completion loss;
- **structured latent-variable learning** describes the latent schedule and code sequence;
- **stochastic shortest-path control** describes the causal, variable-duration rollout with an optional STOP action.

It is **not inherently bilevel optimization**. With fixed \(\phi\), training is a single-level problem:

\[
\min_\theta\; {\cal L}(F_\phi(S_\theta,c),x), \qquad \phi\ \text{constant}.
\]

It becomes bilevel only if the completion model, an adapter, an oracle, or a learned risk model is re-optimized in an inner loop as \(\theta\) changes.

The exact intersection of requirements is poorly covered. [KeyIn](https://proceedings.mlr.press/v120/pertsch20a.html) is the closest peer-reviewed structural match: it predicts keyframe times and values, freezes an inpainter, and trains the keyframe model with inpainting loss. However, it fixes the number of keyframes, relaxes time into expected target frames, uses continuous visual states rather than multiple discrete codebooks, and has no explicit observation-rate control. Recent systems such as [SKIP](https://arxiv.org/abs/2606.00664) and [KeyWorld](https://arxiv.org/abs/2509.21027) validate sparse-keyframe/interpolator decompositions, but learn timing and content in separate stages and are currently preprints. Active sensing learns positions but normally **measures** their values; neural compression learns discrete values but normally fixes their temporal lattice. Therefore, a complete solution will be a synthesis rather than a reproduction of one paper.

The recommended first implementation is an **offline frozen-completer oracle plus causal distillation**, followed by a small amount of hard downstream-risk fine-tuning. Do not begin with end-to-end actor-critic training. Before building any learned scheduler, run the falsification experiment in Section 12: if an oracle using ground-truth values cannot beat uniform placement at equal rate, adaptive placement has insufficient headroom and the project should focus on value prediction or completer robustness instead.

---

## 1. Evidence labels and assumptions

Claims and recommendations use these labels:

- **[D] Direct evidence:** a peer-reviewed work instantiates most of the claimed mechanism under materially similar assumptions.
- **[A] Adaptation:** the mechanism is supported in an adjacent problem, but at least one important assumption differs.
- **[S] Synthesis/speculation:** an implementable proposal assembled from evidence, not a published solution to the exact problem.
- **[P] Preprint evidence:** relevant public work not yet peer reviewed as of the cutoff.

The review assumes:

1. \(F_\phi\) is differentiable with respect to continuous embeddings at its input, even though its weights are frozen.
2. The planner chooses gaps from a finite set \(G=\{1,\ldots,G_{\max}\}\), plus an optional STOP action.
3. Each predicted state has \(Q\) codebooks, \(a_k=(a_k^1,\ldots,a_k^Q)\), with vocabulary sizes \(V_q\).
4. Training sequences provide a ground-truth discrete encoder \(e(x_t)\), or another unambiguous target for observation fidelity.
5. Some conditioning is available ahead of time and some is causal. No training procedure may expose future conditioning to a deployable policy unless that variant is explicitly called an oracle.
6. The final duration may be unknown. A policy therefore needs either a STOP decision, a terminal event, or a rolling safety rule rather than a fixed-\(T\) top-\(K\) mask alone.

If \(F_\phi\) is a black-box API or includes non-differentiable decoding after its token input, the pathwise designs below reduce to score-function, minimum-risk, distillation, or surrogate-risk methods.

---

## 2. Formalization: what is actually being optimized?

### 2.1 A semi-Markov control view

At event \(k\), define the causal planner state

\[
h_k = \left(c_{\le t_k},\, S_{\le k},\, b_k,\, \tau_k\right),
\]

where \(b_k\) is remaining budget and \(\tau_k\) contains relative time, normalized elapsed time when available, and any terminal signal. The action is

\[
u_k=(g_k,a_{k+1}^{1:Q})\quad\text{or STOP}.
\]

Because an action advances time by \(g_k\), this is naturally a **semi-Markov decision process** rather than an ordinary one-step MDP. With unknown duration it is a stochastic shortest-path or average-cost problem. The terminal cost is the actual sequence distortion:

\[
L_{\rm seq}(S)=D_{\rm sequence}(F_\phi(S,c),x).
\]

The training state is fully observable because \(x\) is available for losses; the deployment state may be partially observable because the future sequence and future causal conditioning are not. This is why oracle schedules can be useful teachers but are not valid deployed policies.

### 2.2 A constrained rate-distortion view

A more interpretable objective than an unconstrained \(\lambda |S|\) is:

\[
\begin{aligned}
\min_\theta\quad
&\mathbb E\left[
L_{\rm seq}
+ \alpha L_{\rm obs}
+ \beta L_{\rm prior}
+ \gamma L_{\rm robust}
\right]\\
\text{s.t.}\quad
&\mathbb E[K/T]\le \rho,\qquad
\Pr(g_k>G_{\rm safe})\le\epsilon.
\end{aligned}
\]

Here:

- \(L_{\rm obs}=\sum_{k,q}{\rm CE}(p_\theta(a_k^q\mid h_k),e_q(x_{t_k}))\) is a proper token loss;
- \(L_{\rm prior}=-\sum_k\log p_\psi(a_k\mid a_{<k},g_{\le k},c)\) penalizes low-density code tuples and transitions;
- \(L_{\rm robust}\) measures sensitivity to plausible code corruption or disagreement among frozen-completer samples;
- \(\rho\) is a target observation density;
- \(G_{\rm safe}\) is calibrated from the frozen completer rather than chosen aesthetically.

The Lagrangian uses a learned multiplier:

\[
{\cal L}=
L_{\rm seq}+\alpha L_{\rm obs}+\beta L_{\rm prior}
+\lambda(\mathbb E[K/T]-\rho),\qquad
\lambda\leftarrow[\lambda+\eta_\lambda(\widehat{K/T}-\rho)]_+.
\]

This is preferable to hand-tuning a different \(\lambda\) for every dataset and duration. The simple count \(|S|\) is only a compute proxy. If actual bandwidth matters, the rate must include both positions and values:

\[
R(S)\approx-\sum_k\log p(g_k\mid h_k)
-\sum_{k,q}\log p(a_k^q\mid h_k,g_k,a_k^{<q}).
\]

Two schedules with the same number of observations need not have the same bit cost.

### 2.3 A shortest-path special case

If \(F_\phi\) completes every interval independently and sequence distortion is additive, define the ground-truth-boundary edge cost

\[
C(i,j)=D\!\left(F_\phi((i,e(x_i)),(j,e(x_j)),c),x_{i:j}\right).
\]

For a count penalty, the oracle schedule is a shortest path:

\[
V(j)=\min_{i<j,\ j-i\le G_{\max}}\{V(i)+C(i,j)+\lambda\}.
\]

For exactly \(K\) intervals:

\[
V(k,j)=\min_i\{V(k-1,i)+C(i,j)\}.
\]

This dynamic program is exact only if interval costs do not depend on the full schedule. Predicted autoregressive values, shared random seeds, global identity state, and cross-interval consistency break that decomposition. In those cases the DP is still a useful upper-bound oracle, but not the final training objective.

### 2.4 Why location and value are not symmetric discrete variables

A codebook choice can often be made differentiable at the input embedding:

\[
\tilde y = {\rm stopgrad}(y_{\rm hard}-p)+p,\qquad
z=\tilde y^\top E.
\]

The forward value is a real code embedding, while the backward derivative follows \(p\). A time choice changes indexing and control flow: a hard gather at \(t+g\) has no derivative with respect to \(g\). A single straight-through expression does not magically differentiate that control flow. Location learning needs one of:

- explicit evaluation of candidate intervals in **loss space**;
- a differentiable temporal interpolant, accepting soft/hard mismatch;
- differentiable dynamic programming over edge costs;
- a score-function estimator;
- oracle imitation;
- or a learned risk surrogate.

This distinction should shape the architecture.

---

## 3. Taxonomy of solution families

| Family | What it optimizes | Typical gradient | Actual hard downstream input? | Best use here | Principal mismatch |
|---|---|---|---|---|---|
| Fixed-budget keyframe generator + infiller | Predicted values at predetermined times | CE, reconstruction gradient | Usually yes for values | Establish content and completer baselines | Does not learn locations or rate |
| Offline edge-cost search / DP | Ground-truth or cached interval risk | No planner gradient; supervised distillation | Yes | Prove placement headroom; create labels | Oracle can see information unavailable causally |
| Adaptive sensing / feature acquisition | Which value to reveal next | RL, greedy VOI, oracle imitation | Yes | Scheduling, stopping, cost control | Values are measured, not predicted |
| Differentiable mask / subset selection | A fixed-\(K\) subset | Gumbel, hard concrete, exact marginals, soft top-\(K\) | Varies | Offline fixed-duration selection | Weak fit to causal variable-cardinality rollout |
| Differentiable DP | Globally structured path/subset | Smoothed max or entropy marginal | Soft trajectory or hard forward/proxy backward | Additive interval cost and bounded gaps | History-dependent values destroy small-state DP |
| Hard candidate risk marginalization | Expected downstream loss across a shortlist | Exact derivative of \(\sum p_iL_i\) | Yes | Low-bias gap updates with small candidate sets | Requires \(B\) downstream calls per decision |
| Score-function / actor-critic | Expected loss of hard causal schedules | REINFORCE, RELAX, actor-critic, PPO | Yes | General non-decomposable objective | High variance, critic error, expensive rollouts |
| Minimum-risk / \(N\)-best training | Sequence metric on sampled or beam schedules | Renormalized sample risk | Yes | Final fine-tuning after imitation | Candidate truncation bias and multiple \(F\) calls |
| Rate-distortion / task-aware compression | Rate plus reconstruction or task loss | Pathwise through codec; quantization proxy | Often | Rate control, multi-codebook design | Usually fixed temporal grid |
| Token pruning / adaptive computation | Task loss at lower token count | soft mask, ST mask, halting loss | Often | Learned stopping and target-rate penalties | Drops observed tokens rather than inventing future states |
| Surrogate risk / learned critic | Predicted downstream cost | Supervised risk regression | Yes during data generation, not every update | Cheap action ranking and long horizons | Planner can exploit critic error |
| Robust decoding / conditioning augmentation | Performance under imperfect context | Noise/replay/adapters | Yes | Close clean-to-predicted input gap | Usually requires modifying or wrapping \(F_\phi\) |

No family solves all axes. The architecture must choose where to accept approximation: oracle observability, soft relaxation, local risk, policy-gradient variance, or critic bias.

---

## 4. Evidence matrix: systems that learn placement, content, or rate

Notation: \(D\) = reconstruction distortion, \(R\) = rate/count, \(L_{\rm task}\) = downstream task loss, \(C_{\rm acq}\) = acquisition cost. "Frozen" refers to the downstream reconstructor/task model during the relevant upstream optimization. Cost is relative and omits one-time pretraining.

| Paper | Domain | Mathematical objective | Positions learned? | Values learned? | Discrete gradient | Downstream frozen? | Actual downstream loss trains upstream? | Approx. cost | Main limitation and relevance |
|---|---|---|---|---|---|---|---|---|---|
| [KeyIn, Pertsch et al. (2020)](https://proceedings.mlr.press/v120/pertsch20a.html) | Video prediction/planning | keyframe reconstruction + \(\lambda_I\sum_t\|\hat x_t-x_t\|_2\) | **Yes**, temporal-offset distributions | **Yes**, predicted keyframes | Expected target frame/time; differentiable relaxation | **Yes**, inpainter frozen in keyframe stage | **Yes** | One inpainter rollout per sequence after pretraining | **[D] Closest precedent.** Fixed keyframe count and maximum offset; expected-frame relaxation; continuous pixels/latents; no rate term. |
| [SKIP (2026 preprint)](https://arxiv.org/abs/2606.00664) | Embodied video world models | sparse keyframe generation + gap prediction + action-conditioned interpolation | Indirectly; robot-aware keyframes/gap predictor | Yes | Separate supervised modules | No, modules trained for pipeline | Not jointly | Sparse generation plus interpolation | **[P]** Very close decomposition and reported efficiency, but placement and value are not jointly optimized by final loss. |
| [KeyWorld (2025 preprint)](https://arxiv.org/abs/2509.21027) | Robot world models | oracle keyframe reconstruction/generation + learned gap interpolation | Oracle RDP selection; gap learned | Yes | Supervised modular training | No | No | About keyframe fraction plus interpolation | **[P]** Shows a pragmatic oracle-first architecture; no causal joint selection. |
| [LOUPE, Bahadir et al. (2019/2020)](https://arxiv.org/abs/1907.11374) ([official code](https://github.com/cagladbahadir/LOUPE)) | MRI | \(\mathbb E_m D(R_\omega(m\odot y),x)+\lambda\|m\|_1\) | Yes, global sampling mask | No; sampled values are measured | Bernoulli probability relaxation | Usually jointly trained | Yes | One reconstruction per mask sample | **[A]** Strong rate-distortion precedent. Mask is dataset-level and values are observed, not predicted. |
| [LOUPE-ST (2020 preprint/meeting work)](https://arxiv.org/abs/2007.14450) | Multi-coil MRI | reconstruction loss under learned binary mask | Yes | No | Hard forward, straight-through mask | Usually no | Yes | Unrolled reconstruction | **[P/A]** Evidence for hard sensing masks and biased STE; not causal or value-generative. |
| [Yin et al. (2021)](https://arxiv.org/abs/2105.06460) | Sequential MRI | end-to-end expected reconstruction error after adaptive samples | Yes, per-instance sequential | No | Differentiable sequential sampler | No | Yes | Reconstructor repeatedly used during rollout | **[A]** Direct adaptive-location evidence; joint co-adaptation and measured values differ. |
| [Pineda et al. (2020)](https://arxiv.org/abs/2007.10469) ([official code](https://github.com/facebookresearch/active-mri-acquisition)) | Active MRI | expected reconstruction-improvement reward under acquisition budget | Yes, sequential | No | Policy gradient / RL | **Yes** in key experiments | **Yes**, as reward | One or more reconstructions per acquisition | **[D/A]** Best frozen-reconstructor scheduling precedent, but acquisition reveals truth. |
| [ASMR (2024)](https://arxiv.org/abs/2406.04318) | MRI for pathology prediction | task reward minus sampling cost | Yes, sequential | No | RL | Task/reconstruction components fixed by stage | Yes, task reward | Sequential acquisitions and predictor calls | **[A]** Shows task-aware placement can differ from pixel-reconstruction placement. |
| [Acquisition Conditioned Oracle, Valancius et al. (2024)](https://proceedings.mlr.press/v235/valancius24a.html) | Active feature acquisition | expected prediction/decision loss + \(\alpha |A|\) conditional on acquired features | Yes | No | Nonparametric nongreedy oracle; optional behavior cloning | Predictor can be fixed | Indirectly through oracle risk | Search/inference over conditional subsets | **[A]** Strong support for oracle-first training and a warning that "cheating" oracles can be hard to imitate. |
| [Generative-Surrogate AFA, Li & Oliva (2021)](https://arxiv.org/abs/2010.02433) | Active feature acquisition | MDP reward shaped by conditional information gain/task risk | Yes | No | RL with generative surrogate | Task predictor may be fixed | Usually via proxy/shaped reward | Cheap policy steps after surrogate training | **[A]** Supports proxy risk and auxiliary rewards; proxy mismatch is central. |
| [Non-Myopic AFA via Pathwise Policy Gradients (2026 preprint)](https://arxiv.org/abs/2605.05511) | Active feature acquisition | expected terminal task loss + acquisition cost | Yes, full trajectory | No | Hard forward / soft backward pathwise rollout, temperature staging | Predictor treated as downstream evaluator | Yes | One rollout; relaxation backprop through trajectory | **[P]** Highly relevant estimator, but values are revealed and peer review is pending. |
| [AdaFrame, Wu et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.pdf) | Video recognition | recognition reward with sequential frame selection and stopping | Yes | No | Policy gradient | Recognition backbone participates by stage | Yes | Selected-frame backbone evaluations | **[A]** Causal frame selection and stopping; selects existing frames for classification. |
| [Adaptive Keyframe Sampling (2025 preprint)](https://arxiv.org/abs/2502.21271) | Long-video understanding | prompt relevance + temporal coverage at fixed \(K\) | Yes | No | Structured/algorithmic selection | Frozen VLM setting | Proxy, not reconstruction | Feature scoring plus selection | **[P/A]** Useful coverage prior; fixed count and observed video. |
| [DynamicViT, Rao et al. (2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html) | Efficient vision transformers | classification loss + token-ratio target | Yes, keep/drop tokens | No | Differentiable attention masking | No | Yes | Reduced later-layer FLOPs | **[A]** Good target-rate and hard-token precedent; not temporal generation. |
| [End-to-End Optimized Compression, Balle et al. (2017)](https://openreview.net/pdf?id=rJxdQ3jeg) | Image compression | \(R+\lambda D\) | Fixed lattice | Yes, quantized latents | Additive-noise quantization proxy | Decoder jointly learned | Yes, reconstruction | One codec pass | **[A]** Foundational rate-distortion objective; no adaptive timing or frozen decoder. |
| [SoundStream, Zeghidour et al. (2021)](https://research.google/pubs/soundstream-an-end-to-end-neural-audio-codec/) | Audio compression | adversarial + reconstruction losses under RVQ bitrate | Fixed time grid; variable active quantizers | Yes, multiple residual codebooks | Quantizer STE; structured quantizer dropout | No | Yes | One streaming codec pass | **[A]** Strong evidence for multi-codebook states and one model across rates; not sparse temporal placement. |
| [Deep Video Codec Control for Vision Models (2024)](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Reich_Deep_Video_Codec_Control_for_Vision_Models_CVPRW_2024_paper.html) | Task-aware video coding | bandwidth/quality plus downstream vision performance | Coding controls rather than arbitrary sparse times | Codec representation learned/controlled | Differentiable task-aware control | Downstream vision model can be fixed | Yes | Codec plus vision-model pass | **[A]** Direct task-aware rate allocation; conventional codec structure constrains actions. |
| [TransTIC, Chen et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_TransTIC_Transferring_Transformer-based_Image_Compression_from_Human_Perception_to_Machine_ICCV_2023_paper.pdf) | Compression for machine vision | rate plus frozen machine-task loss via transferable prompts | Fixed spatial lattice | Yes | Pathwise codec gradient | **Yes**, base codec/task components largely fixed | **Yes** | Codec plus task pass | **[A]** Supports upstream adaptation through frozen downstream modules; no scheduling. |
| [Identity-Preserving Learned Compression (2022)](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Xiao_Identity_Preserving_Loss_for_Learned_Image_Compression_CVPRW_2022_paper.html) | Face compression/recognition | rate-distortion plus identity loss from recognizer | Fixed lattice | Yes | Pathwise through codec | **Yes**, recognition model | **Yes** | Codec plus recognition pass | **[D/A]** Clear peer-reviewed workshop precedent for frozen-task gradients, but continuous spatial codec. |
| [VQ-VAE, van den Oord et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | Discrete representation learning | reconstruction + codebook + commitment losses | Fixed lattice | Yes, categorical codes | Hard nearest neighbor with STE | No | Yes | Encoder/decoder plus nearest-neighbor lookup | **[A]** Foundational hard discrete-value path; valid IDs still do not guarantee valid code sequences. |
| [Finite Scalar Quantization (2024)](https://iclr.cc/virtual/2024/poster/19317) | Discrete representation learning | reconstruction using bounded, rounded scalar product codes | Fixed lattice | Yes | Round STE | No | Yes | Simpler than learned VQ codebooks | **[A]** Avoids codebook collapse in a representation redesign; not directly applicable if \(F_\phi\)'s codebooks are already fixed. |

### Critical reading of the matrix

1. **Joint location and predicted-value learning is rare.** KeyIn is the clearest direct example, but its relaxation and fixed cardinality leave most of the current problem open.
2. **Frozen downstream loss is not technically exotic.** Active MRI and task-aware compression show that a model may be frozen while its input-producing policy or encoder is optimized. Freezing weights must not be confused with detaching its input graph.
3. **Adaptive sensing overstates transfer.** In sensing, a chosen action reveals a true value. Here, choosing a difficult or remote time also creates a harder prediction target. Location and value errors are coupled.
4. **Compression provides the right trade-off, not the right action space.** Rate-distortion objectives, dual rate control, and multi-codebook dropout transfer well; fixed sampling grids do not.
5. **Recent keyframe systems support modular engineering, not joint optimality.** SKIP and KeyWorld are evidence that separate keyframe/gap/interpolation modules are practical. They do not establish that their positions are optimal for the downstream model.

---

## 5. Evidence matrix: discrete optimization and robustness tools

| Paper | Domain | Objective / mechanism | Positions? | Values? | Gradient method | Frozen downstream? | Actual downstream loss? | Cost | Limitation and relevance |
|---|---|---|---|---|---|---|---|---|---|
| [Gumbel-Softmax, Jang et al. (2017)](https://arxiv.org/abs/1611.01144) | Discrete latent variables | differentiable categorical sample with temperature | Generic | Generic | Continuous reparameterization; optional ST | Compatible | Compatible | One path | Biased for the hard objective; soft states may be off-manifold. |
| [Hard Concrete \(L_0\), Louizos et al. (2018)](https://arxiv.org/abs/1712.01312) | Sparse neural networks | task loss + expected number of nonzero gates | Binary gates | No | Stretched concrete with hard clipping | Compatible | Yes if task loss used | One path | Good count control; not fixed-\(K\), ordered, or multi-categorical by itself. |
| [Differentiable DP, Mensch & Blondel (2018)](https://proceedings.mlr.press/v80/mensch18a.html) | Structured prediction | replace DP max/min with smooth operator; differentiate expected structure | Structured path | Generic edge labels | Exact gradient of smoothed DP | Compatible | Yes if edge costs derive from \(F\) | DP over state graph | Requires decomposable costs and manageable state; smooth/hard gap remains. |
| [REBAR (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/ebd6d2f5d60ff9afaeda1a81fc53e2d0-Abstract.html) / [RELAX (2018)](https://openreview.net/forum?id=SyzKd1bCW) | Discrete stochastic computation | unbiased score-function gradient with continuous control variate | Generic | Generic | Unbiased, learned/analytic control variate | Yes | Yes | Multiple conditional evaluations plus critic | More complex and often slower than ST; variance can remain high. |
| [SIMPLE, Ahmed et al. (2023)](https://openreview.net/forum?id=GPJVuyX4p_h) ([official code](https://github.com/UCLA-StarAI/SIMPLE)) | \(k\)-subset sampling | hard subset forward; exact subset marginals as backward proxy | Fixed-\(k\) subset | No | Discrete forward, marginal proxy gradient | Compatible | Yes | Marginal computation + one task path | Attractive for offline fixed-\(K\) selection, but not variable-length causal schedules. |
| [Fast Differentiable Sparse Top-\(k\), Sander et al. (2023)](https://proceedings.mlr.press/v202/sander23a.html) | Ranking/subset selection | regularized sparse top-\(k\) operator | Fixed-\(k\) | No | Differentiable \(p\)-norm regularization | Compatible | Yes | Sorting/top-\(k\) overhead | Efficient but still a relaxed, noncausal fixed-\(k\) object. |
| [Minimum Risk Training, Shen et al. (2016)](https://aclanthology.org/P16-1159/) | Sequence generation | normalized expected task risk over sampled candidates | Sequence choices | Yes | Sample/beam risk gradient | Yes | **Yes**, even for nondifferentiable metrics | \(B\) task evaluations per example | Candidate truncation and sampling bias; strong final-stage option. |
| [DARTS (2019)](https://arxiv.org/abs/1806.09055), [FairDARTS (2020)](https://arxiv.org/abs/1911.12126) | Neural architecture search | validation loss over soft operation mixtures | Architecture edges | Operations | Continuous mixtures / discretization regularizer | Often shared weights | Yes | Much cheaper than enumerating architectures | Published warning: a good soft mixture need not yield a good hard architecture. Direct analogy to expected code embeddings. |
| [DAgger, Ross et al. (2011)](https://proceedings.mlr.press/v15/ross11a.html) | Imitation learning | aggregate states visited by current policy and query expert labels | Sequential action | Generic | Supervised on on-policy states | Oracle fixed | Oracle action, not task loss | Rollouts + relabeling | Strong exposure-bias remedy if a valid oracle can label visited states. |
| [Scheduled Sampling, Bengio et al. (2015)](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) | Autoregressive sequence learning | mix ground-truth and predicted history | No | Yes | Standard supervised gradient | N/A | No | One rollout | [Huszar's critique](https://arxiv.org/abs/1511.05101) shows the objective can be statistically inconsistent; use as curriculum, not final principle. |
| [Cascaded Diffusion Conditioning Augmentation, Ho et al. (2022)](https://jmlr.org/papers/v23/21-0635.html) | Conditional image generation | corrupt lower-resolution conditioning during training | No | Conditioning values | Standard differentiable training | No | Yes | Ordinary training with noise | **[A]** Direct support for matching predicted/noisy conditioning, but modifying a frozen completer needs an adapter or limited fine-tuning. |
| [MOPO, Yu et al. (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html) | Offline model-based RL | model return minus uncertainty penalty | Actions | Model states | Policy optimization | Learned model fixed during policy stage | Penalized proxy | Model rollouts + ensembles | **[A]** Supports pessimism against model exploitation; uncertainty estimates can themselves be unreliable. |

---

## 6. Comparing the discrete-decision options

### 6.1 Decision table

| Method | Hard state seen by \(F_\phi\) during training? | Bias | Variance | \(F_\phi\) evaluations | Handles causal variable \(K\)? | Recommended role |
|---|---:|---:|---:|---:|---:|---|
| Soft expected embeddings | No | High for hard inference objective | Low | 1 | Yes | Diagnostic only; avoid as the main final method |
| ST Gumbel / categorical STE | Yes in forward | Biased | Low-medium | 1 | Yes | Codebook values; not sufficient for hard time indexing |
| Hard-concrete gates | Usually | Biased | Low-medium | 1 | Yes for stop/keep gates | Sparsity/halting, with calibrated hard inference |
| Soft or Gumbel top-\(K\) | Often soft | Biased | Low-medium | 1 | Usually fixed \(K\), offline | Fixed-duration ablation or proposal shortlist |
| SIMPLE exact-marginal proxy | Yes | Proxy-gradient bias | Often lower than ST in tested settings | 1 | No, natively fixed \(K\) | Offline fixed-\(K\) location selector |
| Differentiable DP | Soft structure or hard/proxy variant | Controlled smoothing bias | Low | edge-cost table | Yes if state graph includes stopping | Additive interval objectives |
| Exact loss-space marginal over \(B\) candidates | **Yes** | Exact for enumerated one-step expectation; myopic otherwise | Low | \(B\) per decision | Yes | Strong location update when \(B\) is small |
| REINFORCE | **Yes** | Unbiased Monte Carlo gradient | High | 1 per rollout | **Yes** | General baseline and final hard-objective audit |
| REBAR / RELAX | **Yes** | Unbiased under estimator assumptions | Medium | \(>1\) plus critic | Yes | When plain REINFORCE variance is prohibitive |
| Actor-critic / PPO | **Yes** | Critic/bootstrapping/clipping bias | Medium | terminal or sampled exact audits | **Yes** | Long-horizon ambitious design |
| Minimum-risk \(N\)-best | **Yes** | Candidate-set bias | Medium | \(B\) per sequence | Yes | Final hard-schedule fine-tuning |
| Oracle-policy distillation | **Yes** at deployment | Bias toward oracle and imitation error | Low | Offline oracle only | Yes | First implementation |
| Learned risk critic | **Yes** when collecting labels | Critic approximation bias | Low planner-update variance | Periodic audits | Yes | Candidate pruning and long-horizon scaling |

### 6.2 The most important implementation rule

**[A] Keep the downstream model's forward inputs hard.** Do not pass a convex combination of codebook embeddings to a model trained only on code embeddings and then assume the resulting gradient describes hard inference. A frozen completer may use those mixtures as an unintended side channel.

For codebook \(q\), use a hard-forward STE:

\[
y_q^{\rm ST}=y_q^{\rm hard}+p_q-\operatorname{stopgrad}(p_q),\qquad
z_q=(y_q^{\rm ST})^\top E_q.
\]

The forward tensor equals the chosen embedding. The gradient is biased, so it must be checked against hard-rollout improvements and token-level supervision.

For gaps, prefer **loss-space marginalization** when affordable. Let \(\mathcal B(h_k)\) be a shortlist of \(B\) hard next-gap candidates. Evaluate:

\[
L_b=D\!\left(F_\phi(S_{\le k}\cup\{(t_k+g_b,a_b)\},c),x\right),
\qquad
\bar L=\sum_{b\in\mathcal B}\tilde p_b L_b.
\]

Every call to \(F_\phi\) sees a real hard time and real hard code IDs. The derivative with respect to gap logits is exact for this finite one-step mixture. It is not an exact gradient of the full autoregressive schedule unless the interval loss decomposes or future actions are marginalized too.

### 6.3 When differentiable dynamic programming is genuinely appropriate

Use differentiable DP when:

- intervals are conditionally independent given adjacent observations;
- candidate gaps are bounded;
- an edge cost can be computed without rolling out the entire future;
- value prediction can be teacher-forced or summarized in a small state;
- and global rate/count constraints fit in the DP state.

Do not use it merely because schedules look like paths. If an endpoint token depends on all earlier predicted tokens, exact state would include an exponentially large history. A beam or learned edge-cost state then makes the method approximate.

### 6.4 When policy gradients are justified

REINFORCE for a hard schedule has the form:

\[
\nabla_\theta J =
\mathbb E\left[(L(S)-b(h))\nabla_\theta\log\pi_\theta(S\mid c)\right].
\]

One hard whole-sequence \(F_\phi\) evaluation can supervise all location and code actions. The estimator is unbiased if the baseline is action independent and the sample is on-policy, but terminal reward has high variance and assigns credit poorly. Use:

- oracle/imitation initialization;
- per-interval reward differences when valid;
- learned state-value baselines;
- advantage normalization;
- multiple stochastic completer seeds treated as environmental noise;
- and an auxiliary observation CE that is never fully removed.

Actor-critic becomes attractive only after the policy reaches a sensible region. Starting it from random schedules invites sparse rewards, dense-selection collapse, and downstream-model exploitation.

---

## 7. Frozen downstream gradients, robustness, and model exploitation

### 7.1 Propagating loss through a frozen model

Freezing parameters means:

```text
requires_grad(phi) = false
```

It must not mean:

```text
input_to_F = detach(input_to_F)
```

The computation graph from \(L_{\rm seq}\) through \(F_\phi\) to the planner's input embeddings must remain live. [KeyIn](https://proceedings.mlr.press/v120/pertsch20a.html), frozen-reconstructor active MRI, and task-aware learned compression provide direct or adjacent evidence for this pattern.

Practical controls:

1. Put \(F_\phi\) in evaluation mode so dropout and normalization do not drift unless stochasticity is intentionally sampled.
2. Disable parameter gradients, but preserve activation gradients to inputs.
3. Use activation checkpointing or interval-level calls if \(F_\phi\) is large.
4. Verify a finite-difference directional derivative on a continuous input embedding before debugging the discrete estimator.
5. Log clean-boundary and predicted-boundary losses separately.

### 7.2 A valid code is not necessarily an on-manifold observation

Multi-codebook IDs individually drawn from valid vocabularies can still form:

- code tuples never produced by the original encoder;
- temporally impossible transitions;
- conflicting coarse/fine residual codes;
- states inconsistent with external conditioning;
- adversarial boundary pairs that induce a deceptively low training loss from \(F_\phi\).

Use a hierarchy of safeguards:

1. **[D/A] Token fidelity floor:** retain CE to \(e(x_t)\), not just downstream loss.
2. **[A] Frozen prior:** penalize \(-\log p_\psi(a_k\mid h_k,g_k)\) from a model trained only on encoded ground-truth sequences.
3. **[A] Factor-respecting decoding:** predict residual codebooks in the same order and conditioning structure used by the original codec.
4. **[A] Trust region:** constrain \(D_{\rm KL}(\pi_\theta\|\pi_{\rm teacher})\) during downstream-risk fine-tuning.
5. **[A] Uncertainty/pessimism:** penalize ensemble disagreement or high completer sensitivity, analogous to [MOPO](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html).
6. **[S] Audit-and-repair:** reject low-prior tuples or pass them through a small token repair model before \(F_\phi\).

The 2026 theory preprint [Imperfect World Models are Exploitable](https://arxiv.org/abs/2605.15960) formalizes why optimizing over a large policy set against an imperfect learned model can reverse true preferences. It is not direct evidence for sequence completers, but it makes the "frozen means safe" assumption untenable.

### 7.3 Making the completer robust without destroying clean performance

There are three levels:

**Level 0: keep \(F_\phi\) completely frozen.** Train the planner on hard free-running observations, add prior/uncertainty penalties, and optionally repair inputs. This preserves the original model exactly but may leave an irreducible clean-to-predicted gap.

**Level 1: frozen core plus input adapter [A].** Insert a small residual adapter \(A_\eta\) before \(F_\phi\), initialized to identity:

\[
z' = z + s\,A_\eta(z,h),\qquad s(0)=0.
\]

Train on a mixture of ground-truth, corrupted, and planner-predicted observations. Preserve clean behavior with:

\[
L_{\rm preserve} =
D\!\left(F_\phi(A_\eta(S_{\rm clean})),F_\phi(S_{\rm clean})\right)
\eta\|A_\eta(S_{\rm clean})-S_{\rm clean}\|^2.
\]

**Level 2: limited completer adaptation [A].** Fine-tune only normalization, LoRA, or boundary-conditioning blocks with clean replay and output distillation. Conditioning augmentation in cascaded diffusion supports the general principle that training on degraded upstream conditions reduces cascaded mismatch, but the exact preservation recipe is an adaptation.

Always report a 2-by-2 matrix:

| Completer input | Original \(F_\phi\) | Robust wrapper/adapted \(F\) |
|---|---:|---:|
| Ground-truth observations | clean reference | clean regression check |
| Planner-predicted observations | deployment mismatch | robustness gain |

An intervention is unacceptable if it gains on predicted inputs by materially degrading the clean-input row without an explicitly accepted trade-off.

---

## 8. Efficient candidate evaluation

The naive action count at one step is:

\[
|G|\prod_{q=1}^Q V_q,
\]

which is intractable. The solution is architectural factorization, not exhaustive enumeration.

### 8.1 Factor the policy

\[
\pi(u_k\mid h_k)
=\pi_g(g_k\mid h_k)
\prod_{q=1}^Q\pi_q(a_k^q\mid h_k,g_k,a_k^{<q}).
\]

First shortlist \(B_g\) gaps, then sample or beam-search only \(B_a\) code tuples per gap. Preserve codebook order when the codec is residual. A single flat head over all joint gap-code combinations should be rejected except in tiny toy problems.

### 8.2 Two-stage scoring

Use a cheap local score:

\[
\hat C_\omega(h,g,a)=
\widehat D_{\rm interval}
+\alpha \widehat D_{\rm obs}
+\beta \widehat L_{\rm prior},
\]

to prune to \(B\) candidates. Run \(F_\phi\) only on those candidates. Train the score on exact residual targets \(C-\hat C\), maintain a held-out calibration set, and include uncertainty in ranking:

\[
C_{\rm pessimistic}=\mu_\omega+\kappa\sigma_\omega.
\]

The critic must not be allowed to become the sole judge indefinitely; exact audits prevent reward hacking.

### 8.3 Cache what is invariant

- With ground-truth boundaries and additive intervals, cache \(C(i,j)\) once.
- If \(F_\phi\) exposes boundary encoders, cache boundary features and rerun only the interval decoder.
- Batch candidate intervals across examples and gaps.
- Share conditioning encodings across all candidates.
- Cache deterministic completer outputs by boundary-token tuple only if memory and collision rates make this useful.

### 8.4 Amortize global evaluation

For policy-gradient training, one terminal whole-sequence completion provides a reward for every action. Use cheap dense auxiliary losses at intermediate steps and the exact terminal loss periodically. This changes wall-clock cost from \(O(KB C_F)\) toward \(O(C_F+K C_{\rm critic})\), at the price of higher variance and critic bias.

---

## 9. Failure modes and diagnostic signatures

| Failure | Diagnostic signature | Likely cause | Mitigation |
|---|---|---|---|
| Degenerate dense selection | \(K/T\) rises to limit; tiny gaps everywhere | Weak/poorly scaled rate penalty; downstream always benefits from more anchors | Hard budget or dual target rate; charge actual compute; compare matched-rate curves |
| Degenerate sparse collapse | Immediate STOP or maximum gaps | Rate multiplier too high; reward scale mismatch | Normalize distortion; warm-start at fixed budget; gradually tighten target rate |
| Uniform-policy collapse | Gap entropy vanishes at one constant gap | Conditions do not predict oracle schedule; strong regularization; uniform is genuinely optimal | Measure oracle-student gap; add causal signals; accept uniform if headroom is absent |
| Off-manifold code values | Low \(L_{\rm seq}\), poor token likelihood or implausible decoded observations | Planner exploits \(F_\phi\); soft mixtures or unusual code tuples | Hard forward; prior/trust region; human/independent-model audit |
| Multi-codebook collapse | Later codebooks constant or unused | Dominant early codebooks; weak residual supervision | Per-codebook CE/entropy, structured dropout, residual-order conditioning |
| Soft/hard mismatch | Soft validation improves while argmax rollout worsens | Expected embeddings/times not seen at inference | Hard forward; loss-space candidate risk; report soft-hard gap every epoch |
| Exposure bias | Error grows superlinearly with number of planner steps | Teacher-forced histories dominate training | DAgger-style visited-state training; free-running curriculum; predicted-context replay |
| Token/sequence disagreement | CE improves but \(D_{\rm seq}\) stalls, or vice versa | Token distance is misaligned with completer utility | Retain both losses; measure causal token swaps and downstream sensitivity |
| Boundary conflict | Each anchor looks plausible but intervals have discontinuities | Independently predicted endpoints; global consistency absent | History-conditioned values, transition prior, overlap consistency loss |
| Biased-estimator failure | Training surrogate falls; true hard risk does not | STE/Gumbel/soft-DP bias | Small-problem exact-gradient audit; REINFORCE comparison; hard validation |
| High-variance RL | Loss spikes; policy entropy and rate oscillate | Sparse terminal reward, long horizon | Imitation warm-start, critic, per-interval advantages, reward normalization |
| Critic exploitation | Predicted risk falls while exact audits worsen | Planner moves out of critic training support | Uncertainty penalty, conservative ranking, active relabeling, trust region |
| Frozen-completer brittleness | Good with ground-truth boundaries, sharp failure with predicted ones | Clean-to-predicted conditioning shift | Corruption curriculum, input adapter, repair model, clean replay |
| Unknown-duration failure | Premature STOP or overrun near termination | Weak terminal signal; fixed-length positional encoding | Hazard/STOP head, normalized + relative time, terminal safety constraint |
| Causality leakage | Offline metrics excellent; streaming deployment fails | Future conditioning or target-derived oracle features leaked | Timestamped information audit; causal feature construction; streaming replay |
| Unstable "bilevel" training | Moving target and oscillations | \(F\), critic, and planner all updated together | Freeze stages; slower target critic; replay; do not call a fixed-\(F\) setup bilevel |
| Metric gaming | Better reconstruction metric but worse semantics/perception | Narrow \(D_{\rm seq}\) | Independent metrics, task checks, blinded human review, counterfactual tests |

Mode collapse has two meanings here and both should be logged: collapse of the schedule distribution to one pattern, and collapse of code values to a small set. A deterministic good policy is not pathological; collapse is pathological when it removes necessary conditional variation or harms held-out hard risk.

---

## 10. Three implementable training designs

### Design 1 - Conservative: offline completion-cost oracle plus causal distillation

**Status:** **[D/A]** Directly supported in pieces by KeyIn, active MRI, ACO, DAgger, and rate-distortion training. The exact combination is an adaptation.

#### Architecture

1. Keep \(F_\phi\) frozen.
2. Encode all ground-truth states \(e(x_t)\).
3. Build interval costs \(C(i,j)\) with hard ground-truth boundary codes.
4. Solve a shortest-path/DP oracle for several target rates or count penalties.
5. Train a causal gap/STOP policy to imitate the oracle using only deployable information.
6. Train the multi-codebook value model with token CE at both oracle and policy-visited positions.
7. Aggregate policy-visited histories and relabel them, DAgger-style, when an oracle label remains well defined.

#### Objective

\[
\begin{aligned}
L_{\rm D1} =
&\;{\rm CE}(\pi_g(g_k\mid h_k),g_k^\star)
+\sum_q{\rm CE}(\pi_q(a_k^q\mid h_k,g_k),e_q(x_{t_k+g_k}))\\
&+\mu\,{\rm KL}(\pi_g\|\pi_g^\star)
+\nu\,L_{\rm rate}
+\xi\,L_{\rm prior}.
\end{aligned}
\]

The oracle should be generated at multiple \(\lambda\) or \(K\) values and the policy conditioned on a requested rate \(\rho\). This yields one rate-controllable model rather than one model per budget.

#### Gradient path

No downstream gradient is required during ordinary planner updates. \(F_\phi\) supplies cached costs and validation risk. Gradients are ordinary supervised gradients through gap, STOP, and codebook heads.

#### Discrete decisions

Training uses categorical CE; inference uses hard argmax or sampling. No soft state is sent to \(F_\phi\). The DP uses hard positions and codes.

#### Complexity

- Cost construction: \(O(N\,T\,G_{\max}\,C_{F,\rm interval})\), highly batchable and one-time.
- DP: \(O(N\,T\,G_{\max})\) for a Lagrangian path, or \(O(N\,K\,T\,G_{\max})\) for an exact-count DP.
- Planner training: \(O(NK C_\pi)\), with no \(F_\phi\) in the inner loop.
- Storage: up to \(O(NTG_{\max})\) scalar costs; reduce by storing only valid gaps or recomputing rare edges.

#### Curriculum

1. Calibrate \(G_{\rm safe}\) from ground-truth-boundary completion curves.
2. Train value prediction at uniform/random positions.
3. Train fixed-budget oracle imitation with teacher-forced state history.
4. Replace history tokens progressively with planner predictions.
5. Add visited-state aggregation and multi-rate conditioning.
6. Only after hard-rollout validation stabilizes, add a small amount of minimum-risk fine-tuning.

#### Expected failures

- The oracle uses target information not identifiable from \(c_{\le t}\); the student averages incompatible schedules.
- Ground-truth boundary costs underestimate predicted-boundary risk.
- Additive interval costs miss global identity drift.
- Oracle imitation optimizes labels, not the true loss after student errors.

#### Necessary ablations

1. Uniform vs random vs DP oracle positions, with ground-truth values.
2. Oracle vs distilled positions, with ground-truth values.
3. Ground-truth vs predicted values, at the same positions.
4. Ground-truth-history vs predicted-history training.
5. Per-interval DP cost vs whole-sequence reranked oracle.
6. Fixed \(K\), fixed \(G_{\max}\), and dual target-rate control.
7. Future-aware oracle features vs strictly causal oracle/student features.

#### Judgment

This design is the best first system because it separates "is adaptive placement useful?" from "can a discrete estimator learn it?" It also creates the data needed to train a risk critic later.

---

### Design 2 - Differentiable online: hard code values plus loss-space gap marginalization

**Status:** **[A/S]** Hard-forward code gradients and candidate risk are established tools; their combination for a causal multi-codebook scheduler is a synthesis.

#### Architecture

At each event:

1. The gap head proposes top-\(B_g\) hard gaps.
2. For each gap, the value head proposes one or \(B_a\) hard multi-codebook tuples.
3. \(F_\phi\) evaluates the resulting \(B=B_gB_a\) hard candidate intervals, batched.
4. Gap probabilities are trained by loss-space expected risk.
5. Codebook heads use hard-forward STE plus token CE and a frozen-prior penalty.
6. A hard sampled/argmax schedule is rolled out and evaluated by a whole-sequence loss.

#### Objective

\[
\begin{aligned}
L_{\rm D2} =
&\sum_k\sum_{b\in\mathcal B_k}
\tilde p_{\theta,b}\left[
D_{{\rm interval},b}
+\alpha D_{{\rm obs},b}
+\beta L_{{\rm prior},b}
\right]\\
&+\zeta D_{\rm sequence}(F_\phi(S_{\rm hard},c),x)
+\lambda(\widehat{K/T}-\rho)
+\chi L_{\rm consistency}.
\end{aligned}
\]

\(L_{\rm consistency}\) compares overlapping interval predictions or global state embeddings across neighboring completions. If the whole-sequence term is non-decomposable, its gradient to locations is supplied by a small REINFORCE term or a risk critic; its gradient to hard-forward code embeddings can use STE.

#### Gradient path

- Candidate loss \(\rightarrow\) candidate probabilities \(\rightarrow\) gap logits: exact for the finite shortlist expectation.
- Sequence/interval loss \(\rightarrow F_\phi\) input embeddings \(\rightarrow\) code logits: hard-forward STE, biased.
- Observation CE and prior losses: ordinary categorical gradients.
- Rate constraint: differentiable expected STOP/gap probabilities plus hard-rate monitoring.
- \(F_\phi\)'s parameters remain frozen.

#### Discrete decisions

\(F_\phi\) never receives expected code embeddings or fractional timestamps. All downstream calls use hard code IDs and hard positions. Soft probabilities exist only as weights over scalar hard-candidate losses. Inference is identical to the hard rollout used in training.

#### Complexity

\[
O(N\,K\,B\,C_{F,\rm interval})+
O(N\,C_{F,\rm sequence})
\]

per training sweep. Reduce cost with cached conditioning/boundary encodings, top-\(B\) pruning, mixed precision, and a delayed whole-sequence loss. This is practical when \(B\in[2,8]\) and interval calls are much cheaper than full completion.

#### Curriculum

1. Initialize from Design 1.
2. Start with ground-truth history and one-step candidate losses.
3. Introduce predicted code histories over short horizons.
4. Increase rollout horizon and the proportion of hard free-running examples.
5. Activate the sequence term and dual rate controller.
6. Anneal gap entropy and STE temperature, but keep an entropy floor until rate stabilizes.
7. Periodically validate exact hard rollouts at all target rates.

#### Expected failures

- Local candidate loss is myopic and selects endpoints that conflict globally.
- STE improves a surrogate gradient while hard code choices do not improve.
- Candidate pruning drops the truly best gap early.
- Cost grows linearly with candidate count and event count.
- Gap/value co-adaptation becomes unstable when both change rapidly.

#### Necessary ablations

1. Hard loss-space marginalization vs soft expected embeddings.
2. STE vs REINFORCE for code values on a small exact-enumeration task.
3. \(B=1,2,4,8\); measure quality and wall-clock Pareto curves.
4. Interval-only vs interval + whole-sequence loss.
5. With/without token CE, prior, and overlap consistency.
6. Frozen top-\(B\) proposal vs jointly changing proposal.
7. Teacher-forced vs free-running histories.
8. Fixed \(\lambda\) vs dual target-rate calibration.

#### Judgment

This is the preferred second-stage architecture when interval evaluation can be batched. Its core advantage is objective fidelity: the completion model sees the same hard objects at training and inference. Its cost, not conceptual soundness, is the main risk.

---

### Design 3 - Ambitious: hard semi-Markov actor-critic with a pessimistic completion-risk critic

**Status:** **[S]** The components are established in active acquisition, sequence risk, and offline/model-based RL, but there is no direct published validation for the full combination.

#### Architecture

1. An autoregressive actor factorizes gap, STOP, and residual codebooks.
2. Every training rollout uses hard actions and hard inputs to \(F_\phi\).
3. The exact terminal reward is:

\[
r_{\rm terminal}=
-D_{\rm sequence}
-\alpha D_{\rm obs}
-\beta L_{\rm prior}
-\lambda K.
\]

4. A state/action critic predicts downstream distortion reduction from a candidate event.
5. An ensemble or distributional critic supplies uncertainty; selection uses pessimistic cost \(\mu+\kappa\sigma\).
6. The critic proposes or ranks many cheap candidates; \(F_\phi\) exactly audits terminal rollouts and a small active subset of high-uncertainty candidates.
7. A replay buffer stores hard schedules, exact losses, per-interval residuals, clean/predicted-boundary flags, and completer seeds.

#### Objective

Actor:

\[
L_{\rm actor}=
-\sum_k \hat A_k\log\pi_\theta(u_k\mid h_k)
+\lambda(\widehat{K/T}-\rho)
+\beta L_{\rm prior}
+\alpha L_{\rm obs}
+\tau_{\rm KL}{\rm KL}(\pi_\theta\|\pi_{\rm teacher}).
\]

Critic:

\[
L_{\rm critic}=
\operatorname{Huber}\!\left(Q_\omega(h_k,u_k),\widehat R_k^{F}\right)
+\eta_{\rm cal}L_{\rm calibration}.
\]

Use Monte Carlo returns initially. Introduce bootstrapping only after critic calibration. RELAX can replace an ordinary baseline if its extra conditional samples are affordable.

#### Gradient path

- Gap, STOP, and code actions receive hard score-function/advantage gradients.
- Token CE and prior losses give dense supervised gradients to value heads.
- The critic is trained from exact frozen-completer labels.
- No pathwise gradient through discrete actions is required, although a code STE auxiliary may be retained.

#### Discrete decisions

The actor samples categorical hard actions. Training/inference support matches exactly. Multi-codebook actions are factorized autoregressively, not flattened. A hard maximum gap and budget are enforced by the action mask.

#### Complexity

After critic warm-up:

\[
O(N\,R\,C_{F,\rm sequence})+
O(N\,K\,B\,C_{\rm critic}),
\]

where \(R\) is the number of exact hard rollouts per example and need not scale with \(B\). Periodic active relabeling adds exact interval calls. Memory cost includes the replay buffer and critic ensemble.

#### Curriculum

1. Pretrain the actor with Design 1 labels and value CE.
2. Pretrain the critic on cached DP edges plus deliberately corrupted/predicted boundaries.
3. Train on short fixed horizons with dense interval rewards.
4. Switch to full hard rollouts while retaining a KL trust region to the teacher.
5. Increase horizon and allow STOP.
6. Activate uncertainty-aware candidate search.
7. Relax the teacher KL only when exact audit loss, not critic loss, improves.
8. Continuously add high-disagreement and actor-visited cases to the critic dataset.

#### Expected failures

- High-variance credit assignment over long schedules.
- Critic exploitation and distribution shift.
- Nonstationarity when actor, value generator, and critic learn together.
- Conservative uncertainty suppresses useful novel schedules.
- Policy entropy or rate oscillation.
- Terminal \(F_\phi\) cost remains too high for sufficient rollouts.

#### Necessary ablations

1. Actor initialized from imitation vs random.
2. Exact terminal reward vs critic-only reward.
3. Single critic vs ensemble/pessimistic critic.
4. Whole-sequence vs interval-shaped advantages.
5. REINFORCE, actor-critic, and RELAX on a small controlled task.
6. With/without teacher KL and observation CE.
7. Critic active relabeling vs static cached data.
8. Factorized codebooks vs a limited joint-code baseline.
9. One vs multiple stochastic \(F_\phi\) samples.

#### Judgment

This design is warranted only if Design 1 shows large oracle headroom, Design 2 is too expensive or too myopic, and long-range non-additive effects materially change the best schedule. It offers the most general objective but the weakest optimization reliability.

---

## 11. Evaluation protocol that separates the causes of success

### 11.1 Placement-by-value factorial

At every matched budget, evaluate:

| Placement | Values | Question answered |
|---|---|---|
| Uniform | Ground truth | Baseline capability of \(F_\phi\) at this rate |
| Random (gap-constrained) | Ground truth | Difficulty of the placement space |
| Oracle DP/search | Ground truth | Maximum accessible placement headroom |
| Learned | Ground truth | Placement quality independent of value prediction |
| Uniform | Predicted | Value-generator quality without adaptive placement |
| Oracle | Predicted | Whether the best locations are too difficult to predict |
| Learned | Predicted | Actual system |
| Learned | Corrupted/shuffled | Sensitivity and off-manifold diagnostics |

This is the minimum design capable of distinguishing placement from value quality. Reporting only "uniform + predicted" versus "learned + predicted" confounds both.

### 11.2 Four gaps to report

1. **Oracle placement gap**
   \[
   D({\rm uniform},a^{GT})-D({\rm oracle},a^{GT}).
   \]
   Measures whether adaptive timing is worth learning.

2. **Distillation gap**
   \[
   D({\rm learned},a^{GT})-D({\rm oracle},a^{GT}).
   \]
   Measures causal predictability and policy quality.

3. **Value gap**
   \[
   D({\rm placement},a^{pred})-D({\rm placement},a^{GT}).
   \]
   Measures content prediction damage at fixed placement.

4. **Robustness gap**
   Compare original and robust-wrapped \(F\) under clean and predicted values.

### 11.3 Rate-quality evaluation

Report complete curves, not one \(\lambda\):

- \(D_{\rm sequence}\) vs observations per unit time;
- \(D_{\rm sequence}\) vs measured wall-clock/FLOPs;
- true coded bits if relevant;
- area under the rate-distortion curve;
- Pareto dominance and bootstrap confidence intervals;
- worst-decile and CVaR distortion, not only the mean;
- maximum gap and gap-distribution calibration.

Train at multiple requested rates if possible. Fixed-budget comparisons should include exactly matched endpoints, maximum-gap rules, completer calls, decoding samples, and search compute.

### 11.4 Observation-value metrics

- per-codebook top-1/top-\(k\) accuracy and NLL;
- decoded observation-space distortion;
- code prior NLL and tuple/transition support;
- utilization and entropy per codebook;
- downstream sensitivity to replacing one predicted code with ground truth;
- downstream damage per token error, stratified by gap and time.

Token accuracy alone is inadequate because codebooks and times can have very different downstream leverage.

### 11.5 Exposure and long-horizon metrics

- teacher-forced vs free-running curves;
- distortion as a function of planner event index;
- error growth versus elapsed time and number of reused predictions;
- recovery after injecting one wrong observation;
- schedule divergence from an identical prefix;
- STOP calibration and terminal-overrun rate.

### 11.6 Estimator and critic audits

On a small problem where every action can be enumerated:

- compute the exact gradient of expected hard risk;
- compare cosine similarity, bias, variance, and hard-risk progress for STE, Gumbel, REINFORCE, and candidate marginalization;
- log soft-objective versus hard-objective validation.

For a risk critic:

- calibration error and rank correlation on actor-visited candidates;
- regret of top-\(B\) pruning;
- exact-audit loss versus predicted loss over training;
- out-of-distribution detection on deliberately unusual code tuples.

### 11.7 Causality audit

For each planner input, record the timestamp at which it becomes available. Replay the policy in a streaming harness. Any feature derived from \(x_{>t_k}\), final duration, future conditioning, or an offline encoder is oracle-only unless deployment truly provides it. This audit is as important as model accuracy.

---

## 12. Concrete recommendation and minimal falsification experiment

### Recommendation

Attempt **Design 1 first**, but implement it as a scientific gate rather than a permanent commitment:

1. Calibrate the frozen completer over gap length and boundary corruption.
2. Construct a hard, ground-truth-boundary interval-cost table.
3. Compare DP oracle placement to uniform placement at matched rates.
4. Only if oracle headroom exists, distill a causal multi-rate gap/STOP policy.
5. Train multi-codebook values with both token fidelity and predicted-history exposure.
6. Add whole-sequence minimum-risk or Design 2 fine-tuning only after the factorial analysis identifies placement as a material bottleneck.

This ordering avoids spending months optimizing a discrete scheduler when the frozen completer may already prefer uniform spacing or when predicted value error dominates every placement gain.

### Minimal falsification experiment

Use a representative held-out subset large enough for paired bootstrap intervals (roughly 500-1,000 sequences is usually adequate, but determine the final count from observed variance). Choose three rates, for example 5%, 10%, and 20% of candidate times, with identical endpoints and \(G_{\max}\).

Evaluate:

1. uniform positions + ground-truth values;
2. DP-oracle positions + ground-truth values;
3. uniform positions + current predicted values;
4. DP-oracle positions + predicted values;
5. distilled causal positions + ground-truth values;
6. distilled causal positions + predicted values.

Pre-register success as:

- the oracle reduces \(D_{\rm sequence}\) relative to uniform by at least
  \[
  \delta=\max(5\%\ \text{relative},\,2\times\text{paired bootstrap SE})
  \]
  at two of three rates; and
- at least half of that gain remains when oracle placements use predicted values; and
- the distilled policy recovers at least half of the ground-truth-value oracle gain without increasing observation rate or downstream calls.

**Falsification rules:**

- If criterion 1 fails, reject the central adaptive-placement hypothesis for the current \(F_\phi\), data, and rate range. Use uniform gaps and focus on code values or completer quality.
- If criterion 1 passes but criterion 2 fails, location learning is not the immediate bottleneck. Improve value prediction and boundary robustness before joint optimization.
- If criteria 1-2 pass but criterion 3 fails, the oracle relies on noncausal/unpredictable information. Add better causal conditioning or accept a simpler schedule; changing gradient estimators will not solve an information deficit.
- Only if all three pass should Design 2 or Design 3 be justified.

---

## 13. Architecture selection guide

| Observed condition | Choose | Reason |
|---|---|---|
| Oracle placement barely beats uniform | Uniform schedule + better value model | No scheduling headroom |
| Oracle strong, student weak | Better causal features / multi-modal policy / uncertainty | Information or distillation bottleneck |
| Student strong with GT values, weak with predicted values | Value model + robust input adapter | Content/robustness bottleneck |
| Interval costs additive and \(G_{\max}\) modest | DP oracle or differentiable DP | Exploit exact structure |
| \(F_\phi\) interval calls are cheap and batchable | Design 2 | Hard candidate marginalization is faithful |
| \(F_\phi\) is black-box or nondifferentiable | Minimum-risk / REINFORCE / distillation | No pathwise gradient |
| Global coherence changes schedule rankings | Whole-sequence reranking or Design 3 | Pairwise edge costs are insufficient |
| Multi-codebook product is huge | Gap-first, residual-code factorization | Avoid exponential action head |
| Soft validation improves but hard rollout does not | Hard-forward training and estimator audit | Relaxation mismatch |
| Critic and exact risk diverge | Active exact relabeling or remove critic | Critic exploitation |
| Clean \(F_\phi\) is good, predicted-input \(F_\phi\) is brittle | Identity-initialized adapter with clean replay | Preserve clean behavior while closing shift |

---

## 14. Bottom line

The architecture should not be selected by asking whether Gumbel-Softmax, RL, or differentiable DP is "best" in general. The decision is controlled by three empirical facts:

1. **Does nonuniform placement have oracle value at the relevant rate?**
2. **Is that placement predictable from information available causally?**
3. **Does its advantage survive imperfect predicted observation values?**

The literature supports every individual component--rate-distortion control, hard discrete codes, adaptive acquisition, frozen-task gradients, oracle distillation, downstream-risk training, and conditioning-robust decoding--but not their entire conjunction. The lowest-risk contribution is therefore a staged system with explicit headroom tests, hard-interface training, and causal factorial evaluation. A monolithic end-to-end policy should be treated as the last option, not the default.

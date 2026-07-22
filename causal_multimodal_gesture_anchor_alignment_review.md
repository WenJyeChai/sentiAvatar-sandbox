# Causal multimodal alignment for sparse gesture-anchor planning

## Critical technical literature review and implementation recommendation

**Coverage:** 1 January 2024–22 July 2026  
**Target system:** Step 1 Qwen causal planner; complete untimed transcript; causal Mimi q0–q3 at 12.5 Hz; four-region, four-level motion RVQ anchors at about 0.4 s; 6,000 clips; four 24 GB RTX 4090 GPUs  
**Evidence convention:** **Reported** means stated in a paper, supplement, or official code. **Recommendation** means a design choice inferred for this project. **NR** means the source did not report the requested field or it could not be verified in the official artifact. Retrieval improvements are not described as generation improvements.

---

## 1. Executive conclusion

Cross-modal alignment is likely to help, but symmetric InfoNCE on its own is not a sufficient remedy for this planner's weak conditioning. It can teach speech, text, and motion representations to be distinguishable. It does **not** by itself make the autoregressive output distribution change when the condition changes, and it can succeed through speaker, style, static pose, or clip identity while the generator continues to copy motion history.

The strongest direct evidence now comes from two CVPR 2026 systems. [MIBURI](https://openaccess.thecvf.com/content/CVPR2026/papers/Mughal_MIBURI_Towards_Expressive_Interactive_Gesture_Synthesis_CVPR_2026_paper.pdf) is unusually close to this project: it is causal, consumes a 12.5 Hz speech-language token stream, uses body-part-aware multi-level RVQ, and autoregressively predicts temporal and residual-level tokens. Its GT-versus-predicted-motion InfoNCE improves FGD, BeatAlign, and diversity. However, that objective does not contrast correct versus wrong speech/text. It primarily regularizes expressiveness. Its published straight-through Gumbel construction also exposes the exact problem in the brief: residual-level logits are teacher-forced, then independently sampled and summed into an apparently coherent RVQ latent even when a sampled earlier level differs from the GT prefix that produced later logits.

[LiveGesture](https://openaccess.thecvf.com/content/CVPR2026/html/Saleem_LiveGesture_Streamable_Co-Speech_Gesture_Generation_Model_CVPR_2026_paper.html) shows that zero-look-ahead causal audio, region experts, causal fusion, and uncertainty-guided history masking can produce strong streaming motion. Yet its text-removal ablation changes FGD only from 4.57 to 4.60, leaves diversity essentially unchanged, and slightly improves its beat metric. This is direct evidence that causality, cross-attention, and good generation metrics still do not establish semantic text reliance.

The most relevant transferable result is not another retrieval loss but **counterfactual conditional likelihood**. [Condition Contrastive Alignment (CCA)](https://openreview.net/forum?id=kGvXIlIVLM) improves discrete autoregressive visual generation by explicitly raising likelihood under the correct condition and lowering it under shuffled conditions. [Chronologically Accurate Retrieval (CAR)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7570_ECCV_2024_paper.php) shows that event-shuffled hard negatives expose temporal failures missed by ordinary retrieval; its reinforced model prefers the correct rather than shuffled text under the generator's token likelihood much more often (61.9% to 89.9%). These objectives act on the output distribution and are therefore better matched to the observed failure than attention supervision.

### Bottom-line recommendation

For the 6K experiment, implement a **history-masked condition-refresh adapter** and train it with three separately measurable signals:

1. **Local causal audio-to-GT-motion multi-positive InfoNCE.** Reuse the existing q0–q3 Mimi embeddings, pool only the last 0.64 s of available audio, and align this condition branch to frozen upper-body/hands GT RVQ latents containing both absolute codes and temporal differences.
2. **Utterance-level text-to-GT-trajectory InfoNCE.** Pool the complete transcript and the full GT anchor trajectory. Treat this only as semantic/discourse alignment; without word timestamps it is not token-level temporal alignment.
3. **Correct-versus-corrupted conditional likelihood.** On a 25% sub-batch, keep the target and motion prefix fixed and compare correct text/audio against a duplicate-safe shuffled transcript or a causal past-shifted audio window. This is the component that directly trains condition sensitivity.

Inject the same condition features through a zero-initialized gated residual into the Qwen hidden state immediately before each of the 16 slot classifiers. The condition branch must not read motion tokens. Continue generated-history/self-forcing, add moderate motion-history corruption, and independently drop text/audio during training. Do **not** begin with soft or Gumbel-composed predicted RVQ latents. Use frozen GT codec latents as stable targets and planner condition-only hidden states as the trainable side. If a predicted-token auxiliary is later justified, perform a genuinely sequential 16-step relaxed rollout in which every later logit is recomputed from the relaxed/hard sampled earlier token.

This design is expected to improve:

- local sensitivity to current/past prosody and rhythm;
- utterance-level semantic correspondence;
- correct-versus-shuffled text/audio likelihood gaps;
- resistance to a pure motion-history shortcut;
- generated-prefix condition sensitivity.

It will **not** by itself solve:

- missing word-level timing;
- the one-to-many ambiguity of valid gestures;
- errors in the frozen RVQ representation or Step 2 infiller;
- exposure bias unless generated-history training remains active;
- semantic events whose audible evidence occurs after the causal decision;
- speaker/style shortcuts unless negatives and evaluations control for them.

---

## 2. Search methodology

### Sources and search process

Primary sources were searched in CVF Open Access (CVPR/ICCV), ECVA/ECCV, NeurIPS proceedings, PMLR/ICML, ICLR/OpenReview, ACL Anthology, ACM Transactions on Graphics/project pages, IEEE/ICASSP author repositories, and arXiv. Official project pages and official GitHub repositories were consulted for code and implementation details.

Query families combined terms from the following groups:

- `co-speech gesture`, `speech driven motion`, `semantic gesture`, `streaming gesture`, `causal gesture`, `gesture RVQ`;
- `audio motion contrastive`, `speech gesture alignment`, `temporal contrastive`, `audio visual synchronization`;
- `text motion alignment`, `chronological negative`, `weak temporal alignment`, `late interaction`, `multiple instance`;
- `condition contrastive alignment`, `mismatched condition`, `counterfactual conditioning`, `modality dropout`, `gradient balance`;
- `autoregressive discrete generation`, `Gumbel softmax RVQ`, `differentiable autoregressive sampling`, `residual quantization`;
- `cross attention`, `gated conditioning`, `AdaLN`, `ControlNet consistency`, `modality dominance`.

### Date range and tally

- Search window: **2024-01-01 through 2026-07-22**.
- Manual screening tally after obvious duplicate removal: **104 title/abstract records**.
- Full paper, supplement, or official code artifacts inspected: **43**.
- Papers retained in the evidence table: **25** — 16 directly relevant and 9 transferable.
- Older papers appear only in the separate foundational section.

This is a structured critical review, not a PRISMA systematic review or statistical meta-analysis. The tally records the search session used for this report; it should not be interpreted as an exhaustive census of every paper in every database.

### Inclusion criteria

A paper was retained if it provided at least one of the following:

- generated co-speech/body motion from text, audio, or both;
- strict causal/streaming motion generation;
- an explicit cross-modal or correct-versus-corrupted alignment objective;
- weakly supervised local temporal alignment without dense labels;
- a mechanism with evidence of increased conditioning control;
- a discrete/RVQ autoregressive construction directly relevant to the 16-slot anchor;
- a diagnostic or training method for modality dominance or temporal hard negatives.

### Exclusion criteria

The review excluded papers whose only relevance was generic multimodal classification, papers without enough primary-source detail to support a technical claim, target-video keyframe selection that is not deployable for conditional generation, and pre-2024 work unless it introduced a necessary foundational method. Retrieval papers were retained only when their loss or diagnostics transfer, and are explicitly labeled as retrieval rather than generation evidence.

---

## 3. Taxonomy of approaches

| Family | Representative mechanism | What it can establish | What it cannot establish alone |
|---|---|---|---|
| Cross-modal representation alignment | Symmetric InfoNCE, sigmoid pairwise loss, cosine, triplet/ranking | Paired representations become more separable; useful retrieval geometry | That generated tokens causally depend on the condition |
| Temporal hard negatives | Same-clip shifts, event reorder, partial caption, action replacement | Sensitivity to chronology or synchronization that random cross-clip negatives miss | Causal deployability if the encoder itself sees future context |
| Weak temporal alignment | Phrase-level contrast plus max/LogSumExp late interaction, MIL | Localizes useful words/segments without frame labels | Word timing when the method still relies on forced alignment or timestamps |
| Condition-outcome alignment | Correct-vs-shuffled likelihood, condition reconstruction, cycle consistency | Direct effect of a condition on generated output or likelihood | Naturalness/diversity unless balanced with the base objective |
| Conditioning injection | Token cross-attention, gated residuals, adapters, ControlNet branches, AdaLN | A path through which a condition can affect generation | Actual reliance; a strong history path can ignore it |
| Shortcut prevention | Modality dropout, history masking, context corruption, negative condition swapping | Removes easy paths and exposes inference-like errors | Which modality supplies the missing information unless separately tested |
| Modality balancing | Gradient conflict measurement, Pareto/common gradients, alternating modality updates | Mitigates one objective or modality dominating optimization | Semantic/temporal correctness if the losses themselves are weak proxies |
| Discrete differentiable alignment | Soft expected code embeddings, straight-through Gumbel, differentiable feedback | Gradient from a continuous loss to categorical logits | A coherent autoregressive sample unless later logits use the sampled/relaxed earlier tokens |
| Causal discrete generation | Hierarchical/2D AR over time, body region, and RVQ level | Streaming and factorized prediction with frozen codecs | Strong speech/text reliance without explicit counterfactual evidence |

### 3.1 Contrastive objective choices

- **Symmetric InfoNCE** is the best understood baseline for the local audio-motion and utterance text-motion heads. It benefits from the global batch of 128 and can use several anchor segments per clip. It is sensitive to false negatives and correlated anchors from the same utterance.
- **Sigmoid pairwise/SigLIP-style loss** is a reasonable second experiment if the batch-softmax denominator becomes unstable or duplicate masking removes many negatives. It does not eliminate false negatives; they still require masking.
- **Triplet or margin ranking** is most useful for deliberate corruptions: correct condition should outrank a past-shifted audio window or shuffled transcript by a margin. This is a different question from general retrieval.
- **Multi-positive InfoNCE** fits uncertain gesture latency. Several strictly causal preceding audio windows can be positives for one anchor. A single exact audio-to-anchor match incorrectly assumes zero latency.
- **VICReg/Barlow Twins/BYOL** avoid explicit negatives but offer less direct leverage for the desired correct-versus-wrong condition test. They are fallback anti-collapse objectives, not the recommended first line.
- **Optimal transport, CTC, and soft-DTW** can impose fine temporal structure, but transcript timestamps and duration are absent and the dataset is small. Their alignment paths can become arbitrary. They should not precede a successful utterance-level and audio-local baseline.

### 3.2 Weak temporal alignment without word timestamps

Three levels must remain distinct:

1. The full transcript can be aligned to the **whole anchor trajectory**. This supports an utterance-level semantic claim.
2. A current causal audio query can cross-attend over the known transcript to select content relevant to the present acoustic state. This is a learned timing bridge, but without labels it still does not prove the selected token is spoken at that moment.
3. Word-level or frame-level claims require timestamps, forced alignment, CTC-like supervision, or an independently validated alignment path. JEGAL's impressive local gesture-word coupling uses word boundaries obtained from aligned speech; it cannot be transferred as timestamp-free evidence.

### 3.3 Conditioning mechanisms

Token-level cross-attention is preferable to pure global AdaLN for the transcript. GenTron reports that cross-attention outperforms AdaLN for free-form text, consistent with the need to preserve token-level content. Global AdaLN or FiLM remains suitable for speaker, style, or utterance-level emotion. A gated, zero-initialized residual is preferable for retrofitting this Qwen planner because it is small, initially preserves the baseline, and exposes a measurable condition contribution.

### 3.4 Shortcut prevention

Motion history is both necessary and dangerous. It should remain in the base Qwen path for causal coherence, but the new alignment branch must be unable to read it. Selective token/region masking from LiveGesture provides direct evidence that corrupted-history training improves streaming robustness. Independent text/audio dropout from ConvoFusion and JEGAL provides evidence that removing a modality during training can reduce dominance, although the rates used in large datasets should not be copied uncritically to 6K clips.

Attention magnitude is not causal reliance. An attention map can be high while values are ignored by later layers, and supervised attention can force a visually plausible map without changing the output. The preferred evidence is an intervention: keep target, prefix, RNG, and all other conditions fixed; shuffle or shift one condition; then measure token likelihood and generated output change.

### 3.5 Discrete autoregressive alignment

For the factorization

\[
p(y_i\mid c,h)=\prod_{j=1}^{16}p(y_{i,j}\mid y_{i,<j},c,h),
\]

the logits for slot \(j\) are conditional on the actual prefix supplied to the model. If training supplies GT \(y_{i,<j}\), but an auxiliary independently samples \(\tilde y_{i,<j}\) from earlier logits and combines it with the later teacher-forced distributions, the resulting 16-code latent is not distributed as a sample from this factorization. It is a set of marginals computed under mutually inconsistent prefixes.

MIBURI makes this issue concrete. It forms a predicted RVQ latent by summing straight-through Gumbel samples from all level logits, with Gumbel temperature 0.4, while the kinematic transformer is trained with teacher forcing. The paper demonstrates useful generation regularization, but it does not resolve the joint-sample inconsistency. For this project, a differentiable generated latent is valid only if slot 1 is sampled/relaxed, fed back, slot 2 logits are recomputed, and so on through slot 16.

---

## 4. Evidence table

### Reading the table

Every requested implementation field is either reported or marked **NR**. “Temperature” distinguishes an InfoNCE temperature from a Gumbel temperature. “Compute” reports published hardware only where verified. “Causal” refers to the complete condition-to-output path, not merely a fast sampler.

### 4.1 Directly relevant gesture, speech-motion, and discrete-motion papers

| Paper / status / links | Task, deployment, modalities, output | Alignment representations and granularity | Objective; positives; negatives; false-negative handling | Projection, temperature, weight, batch/distribution, stage, compute | GT vs predicted; relaxation | Reported evidence and ablations | Transferability and critical limitation |
|---|---|---|---|---|---|---|---|
| **MIBURI: Towards Expressive Interactive Gesture Synthesis** (2026, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Mughal_MIBURI_Towards_Expressive_Interactive_Gesture_Synthesis_CVPR_2026_paper.pdf), [project/code link](https://vcai.mpi-inf.mpg.de/projects/MIBURI/) | Online, real-time, fully causal. Moshi semantic/acoustic states at 12.5 Hz + identity -> face, upper, lower multi-level RVQ; two-dimensional temporal/kinematic AR. | Segment-level GT RVQ latent vs predicted RVQ latent; **not** speech-motion contrast. Voice-activation head on hidden state. | One-way InfoNCE: matching GT/predicted segment positive; other batch samples negative. False-negative handling NR. CE and BCE also used. | RVQ latent dimension/codebook details vary by region; projection head NR. Gumbel temp **0.4**; InfoNCE \(\tau\) NR. \(\alpha_{con}=0.1\), \(\beta_{voice}=0.01\). Batch/distributed negatives NR. Joint training after codecs. Training hardware/total compute NR. | GT latent vs ST-Gumbel predicted latent. Teacher-forced AR logits. | Relative to CE+voice: FGD 0.499->0.480, BeatAlign 0.450->0.461, diversity 10.25->10.44; direct latent MSE worsens FGD. | Closest factorization and rate. Supports a small latent auxiliary, but not condition reliance. Published latent composition has the exact teacher-forced q-level coherence problem. Uses a large Moshi foundation model and no correct/shuffled condition test. |
| **LiveGesture: Streamable Co-Speech Gesture Generation Model** (2026, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Saleem_LiveGesture_Streamable_Co-Speech_Gesture_Generation_Model_CVPR_2026_paper.html), [project](https://m-usamasaleem.github.io/publication/LiveGesture/LiveGesture.html); official code URL NR | Strict zero-look-ahead streaming. Causal audio + optional online text + motion history -> discrete per-region motion, then causal decode. | No explicit contrastive alignment. Region experts and causal spatiotemporal fusion; audio-motion cross-attention. | AR CE. Uncertainty-guided token masking, random whole-region masking, and classifier-free modality dropout/guidance. Pair definitions/FN not applicable. | Streamable audio encoder 0.5M params. Alignment projection/temp not applicable. Local/fusion loss best at 0.3/1.0. Batch NR. UGM mask range sampled up to 0.5 performs best. Training hardware/total compute NR; inference <50 ms per 200 ms chunk. | GT discrete targets; corrupted histories. No Gumbel. | Full: FGD 4.57, BC 0.794, diversity 13.97. Without UGR: FGD 4.98, BC 0.723. Without text: FGD 4.60, BC 0.796, diversity 13.96. | Strong evidence for causal history corruption and small audio modules. Equally strong evidence that streaming architecture does not force text use. Its training tokenizer encoder is bidirectional even though generated tokens decode causally; target construction and deployment causality should not be conflated. |
| **HolisticSemGes: Semantic Grounding ... with Contrastive Flow-Matching** (2026, arXiv preprint). [Paper](https://arxiv.org/abs/2603.26553); official code URL NR | Offline holistic co-speech flow matching. Audio + text -> body motion. | Composite text/audio/holistic-motion latent; trajectory/flow level. | Cosine and contrastive objectives; mismatched audio-text conditions are negatives that repel incongruent motion trajectories. Exact positive sampling, FN handling NR. | Projection dimension, temperatures, weights, batch, distributed negatives, activation schedule, hardware and compute NR in accessible primary metadata. | Continuous generated trajectory/velocity field; no discrete relaxation. | Reports objective and user-study gains on BEAT2 and SHOW; detailed numerical ablation NR here. | Most direct recent support for mismatched-condition generation training. Preprint, offline flow model, and continuous outputs; cannot establish feasibility or stability for a 16-slot causal RVQ planner. |
| **SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning** (2025, ICCV). [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_SemGes_Semantics-aware_Co-Speech_Gesture_Generation_using_Semantic_Coherence_and_Relevance_ICCV_2025_paper.pdf), [project/code](https://semgesture.github.io/) | Offline audio+text+speaker -> body/hands VQ gestures. Frozen HuBERT audio; FastText+TCN text semantics; separate body/hands VQ-VAEs. | Global text-to-GT motion latent cosine; semantic relevance for annotated gestures; generated quantized multimodal code matched to GT motion code. | Published Eq. 6 is positive cosine only for body and hands. Prose mentions mismatches, but explicit negative term is absent from that equation. FN/batch-negative fields not applicable/NR. | Motion/text latent 256; VQ commitment 0.25. Second stage: 8-layer, 8-head, hidden 768. BEAT: batch 128, 40 epochs, lr 3e-4; single A100. Temperature NR. Main loss weights NR. | GT frozen motion features for coherence; predicted multimodal quantized representation for consistency. No Gumbel. | Removing coherence or relevance degrades FGD/diversity/SRGR. Misaligned text/noisy audio yields more generic/rhythmic motion. | Strong evidence for separate semantic and quantization targets with small heads. Offline HuBERT and aligned text semantics can leak future timing. No strict generated-prefix or correct/shuffled likelihood evidence. |
| **Understanding Co-speech Gestures in-the-wild (JEGAL)** (2025, ICCV oral). [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Hegde_Understanding_Co-speech_Gestures_in-the-wild_ICCV_2025_paper.html), [project/code](https://www.robots.ox.ac.uk/~vgg/research/jegal/) | **Retrieval/understanding**, not generation. Video gestures + speech + text. Offline bidirectional encoders. | Global phrase embedding plus local gesture-word late interaction. Audio and text word features concatenated; gesture frames pooled near word. | Global in-batch InfoNCE plus max-over-word local coupling contrastive loss. Matching phrase positive; other phrases negatives. Duplicate/FN policy NR. | Text encoder 3 layers, gesture encoder 6; hidden 512, FFN 2048, 8 heads; AdamW 5e-5, wd 1e-4. 50% random drop of either speech or text. Temperature/loss weight/batch-distributed strategy and training hardware/compute NR in the main paper. | GT video/audio/text representations. No predicted generation or Gumbel. | Sequence-only retrieval R@10 23.6; word-only 14.6; both 30.8. Word spotting rises from 20.83/52.46 to 63.6 with both; speech/text capture complementary signals. | Excellent evidence for global+local complementarity and modality dropout. Local coupling uses word boundaries/forced alignment and about ±10 video frames, so it is not a timestamp-free method. Huge 718.4 h dataset and retrieval gains do not imply generation reliance. |
| **SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis** (2025, ICCV). [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_SemTalk_Holistic_Co-speech_Motion_Generation_with_Frame-level_Semantic_Emphasis_ICCV_2025_paper.html); official code URL NR | Offline holistic speech-to-motion. Separates rhythmic base motion from sparse semantic motion, then fuses. | Frame-level rhythm consistency and learned semantic score; sparse semantic branch. | Rhythmic consistency and semantic emphasis objectives. Exact contrastive pairs/negatives/FN: not applicable or NR. | Projection, dimensions, temperatures, weights, batch, training stage, hardware and compute NR in the official landing-page evidence reviewed. | Continuous/latent generated motion; no verified Gumbel. | Reports stronger semantic richness and overall quality; ablation details NR here. | The base-plus-sparse decomposition is highly relevant to anchors. Frame-level semantic cues require timing machinery unavailable in the current untimed transcript; do not import its “frame-level” claim without that supervision. |
| **GestureHYDRA** (2025, ICCV). [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_GestureHYDRA_Semantic_Co-speech_Gesture_Synthesis_via_Hybrid_Modality_Diffusion_Transformer_ICCV_2025_paper.html); official code URL NR | Offline diffusion with audio/motion hybrid inputs; semantic hand gesture activation through a per-subject retrieval repository and timestamp adjustment. | Retrieval-level semantic gesture identity; explicit injected motion segments; audio synchronization. | No cross-modal contrastive objective verified. Hybrid masking, 3D keypoint loss, retrieval and adaptive timestamp synchronization. | Pretrain 120k steps batch 512, then 30k with keypoint loss; four 40 GB A100s for ~3 days; 50-step DDIM. Alignment temperature not applicable. | Retrieved/GT gesture controls and diffusion output; no Gumbel. | Streamer seen: FGD 3.24, semantic activation 84.82%; unseen: FGD 15.43, activation 81.36%. | Shows that explicit retrieval can guarantee rare gesture activation better than implicit generation. Too expensive and structurally different for 4x4090; requires a semantic repository, subject annotations, and timestamp adjustment. Retrieval is not learned generation. |
| **HOP: Heterogeneous Topology-based Multimodal Entanglement** (2025, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_HOP_Heterogeneous_Topology-based_Multimodal_Entanglement_for_Co-Speech_Gesture_Generation_CVPR_2025_paper.html), [project/code](https://star-uu-wang.github.io/HOP/) | Offline trimodal co-speech generation. Gesture motion + audio rhythm + text semantics. | Spatiotemporal heterogeneous graph; audio-action alignment; audio-text semantic reprogramming. | Alignment/entanglement losses reported by the method; explicit InfoNCE pair construction, FN policy, distributed negatives NR. | Projection dimensions, temperatures, weights, batch, training stage, hardware and compute NR in the official landing-page evidence reviewed. | GT-aligned multimodal training and generated motion; discrete relaxation not a core mechanism. | Reports SOTA objective/subjective generation and coordinated gestures. | Supports modeling rhythm and semantics through distinct relations rather than one pooled vector. The offline graph and reported metrics do not prove correct-condition causal sensitivity. |
| **The Language of Motion** (2025, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_The_Language_of_Motion_Unifying_Verbal_and_Non-verbal_Language_of_CVPR_2025_paper.html), [project](https://languageofmotion.github.io/); official code URL NR | Unified multimodal language model: text, speech, motion in/out; offline generation and understanding. | Shared discrete/generative token modeling; task/utterance level. | Generative pretraining, not a verified local cross-modal contrastive loss. Pair/negative/FN fields not applicable. | Model/projection/batch/compute details are task-dependent; alignment temperature/weight and comparable hardware NR. | GT multimodal tokens with generated tokens; no verified Gumbel bridge. | Reports SOTA co-speech generation with less task-specific data and supports editing/emotion tasks. | Evidence that shared-token generative pretraining can transfer across modalities. It relies on a much larger pretrained model/data regime and does not isolate motion-history shortcut prevention. |
| **ConvoFusion** (2024, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Mughal_ConvoFusion_Multi-Modal_Conversational_Diffusion_for_Co-Speech_Gesture_Synthesis_CVPR_2024_paper.html), [code](https://github.com/m-hamza-mughal/convofusion) | Offline latent diffusion from audio, text, speaker, and interlocutor context. | Separate modality cross-attention; modality guidance; word-excitation guidance on text-motion attention. | No contrastive loss. Each modality is randomly null-replaced 10% for selective classifier-free guidance. Word guidance maximizes token-to-motion attention during sampling. | Attention dimensions/temperature not a contrastive setting. Drop rate 10%. Batch/compute NR here. | Diffusion latent; no discrete relaxation. | Word guidance improves semantic recall from 0.34 to 0.40. Contribution-norm analysis shows audio dominates other modalities. | Direct evidence for modality dropout and dominance measurement. Attention maximization assumes attention corresponds to semantic control and is not causal proof; method is offline and can use future context. |
| **EMAGE** (2024, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_CVPR_2024_paper.html), [code/project](https://pantomatrix.github.io/EMAGE/) | Offline holistic audio+masked-motion -> face, upper, hands, lower motion. Four compositional VQ-VAEs. | Audio content/rhythm and time-aligned transcript embeddings; adaptive fusion; regional motion priors. | Joint masked-gesture reconstruction and audio-to-gesture generation. Codec reconstruction, velocity, acceleration, commitment losses. No verified contrastive negatives. | Region latent dimensions/codebook details in paper; loss weights for VQ terms reported as 1 in the inspected equations. Contrastive temperature/batch-negative fields not applicable. Comparable training hardware/total compute NR. | GT masked hints and GT VQ targets; no Gumbel. | Masked training and content-rhythm adaptive fusion improve holistic metrics; separate regional VQs help. | Strong precedent for four-region targets and absolute+velocity/acceleration representations. Word features are time-aligned and thus unavailable here; heavy masked motion hints can reinforce history reliance if conditions are not counterfactually tested. |
| **DiffSHEG** (2024, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_DiffSHEG_A_Diffusion-Based_Approach_for_Real-Time_Speech-driven_Holistic_3D_Expression_and_CVPR_2024_paper.pdf), [code](https://github.com/JeremyCJM/DiffSHEG) | Fast chunked/outpainting diffusion for expression+gesture. Mel/audio + frozen HuBERT, style. | Feature/time concatenation and diffusion conditioning; no explicit cross-modal contrast. | Noise, velocity, Huber losses with reported weights 10/1/1. FOPPAS outpainting for continuity. | Projection/temp/batch-negative fields not applicable. Code training examples use 5 GPUs and very large effective frame batches; exact comparable clip batch NR. | Continuous diffusion outputs; no Gumbel. | Runs over 30 FPS on a 3090 and reports strong holistic metrics. | Fast generation is not strict per-anchor causality: frozen HuBERT and full/chunk speech context are bidirectional as used. Useful for velocity targets and chunk continuity, not condition reliance. |
| **MambaTalk** (2024, NeurIPS). [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/23c9c94227f937cfb50592a15e7fbb63-Abstract-Conference.html), [project/code](https://kkakkkka.github.io/MambaTalk/) | Efficient holistic speech gesture synthesis with selective state-space models and two-stage discrete motion priors. | Hybrid multimodal fusion; local/global scans; no verified correct-vs-corrupt alignment. | Generation/reconstruction objectives; explicit contrastive pairs, negatives and FN handling NR. | Alignment projections/temp/weight/distributed negatives, comparable batch, hardware and total compute not applicable or NR. | GT discrete priors and generated motion; no verified Gumbel. | Reports improved subjective/objective quality and efficiency over diffusion baselines. | Supports efficient long causal-style state processing and discrete priors. “Selective scan” does not by itself establish strict zero-look-ahead or modality reliance in the released generation setting. |
| **LLM Knows Body Language, Too (GesTran)** (2024, ACL). [Paper](https://aclanthology.org/2024.acl-long.273/); official code URL NR | Offline speech-to-gesture translation. Transformer autoencoder discretizes gestures; pretrained LLM maps speech to gesture symbols. | Sequence-level speech/gesture token translation. | AR token likelihood; no explicit cross-modal contrastive pairs/negatives. | Projection/temp/loss weight/batch/distributed negatives and comparable hardware/compute NR/not applicable. | GT discrete symbols under teacher forcing; generated symbols at inference; no Gumbel. | Reports SOTA results on TED and TED-Expressive. | Demonstrates discrete gesture-as-language generation. Uses a large LLM and offline speech representation; no history-shuffle or generated-prefix condition-sensitivity evidence. |
| **T3M: Text Guided 3D Human Motion Synthesis from Speech** (2024, Findings NAACL). [Paper](https://aclanthology.org/2024.findings-naacl.74/), [code](https://github.com/Gloria2tt/naacl2024) | Offline speech-driven motion with additional textual control. | Text-guided global motion semantics; audio-motion generation. | The paper uses pretrained cross-modal components and generation losses; exact relevant contrastive positive/negative/FN configuration NR in the reviewed primary artifacts. | Projection, temperature, alignment weight, batch/distributed strategy and comparable training hardware/compute NR for the transferable component. | GT motion training; generated continuous motion; no verified Gumbel. | Reports quantitative and qualitative improvements over speech-only methods. | Evidence that text adds control beyond audio. It does not address untimed complete transcripts, strict streaming audio, or a dominant generated-motion prefix. |
| **Semantic Gesticulator** (2024, ACM TOG / SIGGRAPH). [Paper](https://doi.org/10.1145/3658134), [project](https://pku-mocca.github.io/Semantic-Gesticulator-Page/), [code](https://github.com/LuMen-ze/Semantic-Gesticulator-Official) | Offline semantics-aware co-speech synthesis using a semantic gesture library, retrieval, and language-model reasoning. | Explicit utterance/phrase semantics mapped to retrieved gesture exemplars. | Retrieval/matching and generation pipeline; no verified end-to-end cross-modal contrastive loss on generated motion. | Large pretrained language/retrieval components; projection/temp/weight/batch and comparable training hardware/compute fields NR/not applicable. | Retrieved motion exemplars plus generated transitions; no Gumbel. | Reports improved semantic correspondence and perceptual results. | Shows the value of explicit semantic gesture grounding. Requires an external library/LLM and is retrieval-augmented rather than evidence that a compact planner learns causal condition use. |

### 4.2 Loosely transferable alignment, conditioning, and discrete-AR papers

| Paper / status / links | Task and core mechanism | Alignment construction and implementation | Reported evidence | Transfer and limitation |
|---|---|---|---|---|
| **Synchformer: Efficient Synchronization from Sparse Cues** (2024, ICASSP). [Paper/project/code](https://www.robots.ox.ac.uk/~vgg/research/synchformer/) | Offline audio-video synchronization. AST audio and Motionformer video encode 0.64 s segments. Linear projections, L2 normalization, symmetric InfoNCE with trainable temperature. Positive: same interval. Negatives: other intervals from the same video and other videos. FN handling NR. Pretraining uses 2 videos × 14 segments = 28 pairs; larger video batch/momentum queue did not help. A separate 3-layer, 8-head, 768-d offset transformer is trained afterward. | Demonstrates that within-clip temporal negatives and modest effective batches can learn sparse synchronization. | Directly supports 0.64 s causal windows and same-utterance shifted negatives. Encoders are bidirectional/offline and the task is synchronization retrieval/classification, not gesture generation. |
| **Chronologically Accurate Retrieval for Temporal Grounding of Motion-Language Models** (2024, ECCV). [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7570_ECCV_2024_paper.php) | Text-motion retrieval and transfer to text-to-motion. Adds event-shuffled descriptions as hard negatives to symmetric InfoNCE. TMR motion features include relative joint positions and accelerations. Batch 32, AdamW 1e-4; temperature numeric value NR; distributed negatives/FN policy NR. GT motion/text; no Gumbel. | Baselines are near 60% on correct chronology; reinforced models exceed 90% in CAR. A generator prefers the correct over shuffled text by token likelihood 61.9% before and 89.9% after reinforcement; generation metrics also improve. | Strongest evidence for deliberate text hard negatives and a correct-vs-shuffled likelihood diagnostic. Offline text-to-motion and LLM-produced event decompositions do not provide word timing or causal speech alignment. |
| **VideoComp** (2025, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_VideoComp_Advancing_Fine-Grained_Compositional_and_Temporal_Alignment_in_Video-Text_Models_CVPR_2025_paper.html) | Video-text retrieval/compositionality. Standard symmetric InfoNCE plus hierarchical hinge preference: correct > temporal reorder > action replacement/partial/compound disruption. Trainable temperature; pretraining batch 128, finetuning batch 32. Exact projection/loss weight and FN handling are paper-specific; no generation or Gumbel. | Temporal-reorder accuracy improves from 56.6 to 65.4 with compositional loss and to 68.2 with compositional pretraining+loss. | Supports a graded corruption hierarchy rather than treating every negative equally. Requires temporally localized event captions and proves retrieval understanding, not generated gesture reliance. |
| **Toward Guidance-Free AR Visual Generation via Condition Contrastive Alignment (CCA)** (2025, ICLR Oral). [Paper](https://openreview.net/forum?id=kGvXIlIVLM), [code](https://github.com/thu-ml/CCA) | Discrete autoregressive image generation. Fine-tunes a target model relative to a frozen reference so paired conditions raise sequence likelihood and batch-shuffled conditions lower it. No projection head; the generator log likelihood is the score. Conditions are shuffled in-batch. False-negative/duplicate policy NR. \(\beta\) and negative weight are method hyperparameters; exact model-specific values should be taken from code. No Gumbel. | One epoch (~1% of pretraining epochs) substantially improves guidance-free generation on LlamaGen and VAR and can halve sampling cost relative to CFG. | Most direct transferable mechanism for making discrete AR output depend on its condition. Images/classes differ from gestures/text/audio, and a live frozen reference model may exceed memory; cached reference scores or a simpler paired margin are feasible variants. |
| **ControlNet++** (2024, ECCV). [Paper](https://arxiv.org/abs/2404.07987), [code](https://github.com/liming-ai/ControlNet_Plus_Plus) | Conditional image diffusion with explicit output-to-condition cycle consistency. A pretrained discriminative reward model recovers the condition from a one-step denoised output; total loss is base diffusion plus weighted reward. No contrastive negatives or Gumbel. Naive full differentiable sampling is reported around 340 GB vs about 6.8 GB for the one-step strategy at batch 1. | Improves over ControlNet by 11.1% mIoU, 13.4% SSIM, and 7.6% RMSE on different controls. | Demonstrates that accepting a condition is weaker than outcome-level feedback. A gesture version would need a trustworthy condition-from-motion reward encoder, exactly the additional large model this project wants to avoid. |
| **GenTron: Diffusion Transformers for Image and Video Generation** (2024, CVPR). [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GenTron_Diffusion_Transformers_for_Image_and_Video_Generation_CVPR_2024_paper.html) | Text-to-image/video DiT. Direct ablation finds token cross-attention better than AdaLN-Zero for free-form text because global modulation loses token/spatial granularity. No contrastive pairs, negatives, or discrete relaxation. | Cross-attention wins the reported text-image composition metrics over AdaLN-Zero. | Supports token cross-attention for the complete transcript and reserves AdaLN/gating for global style. Offline diffusion evidence does not prove causal condition sensitivity. |
| **MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance** (2024, ICML). [Paper](https://proceedings.mlr.press/v235/wei24d.html), [code](https://github.com/GeWu-Lab/MMPareto_ICML2024) | Multimodal classification. Measures gradient conflict between multimodal and unimodal objectives and computes a Pareto-common gradient with enhanced magnitude. Projection/temp/pair fields are task-specific; no generated outputs or Gumbel. | Improves several multimodal recognition benchmarks and diagnoses objective-gradient conflict. | Useful only after this planner logs per-loss gradient norm and cosine. It is unnecessary complexity for the first 6K run and cannot replace correctly constructed condition negatives. |
| **Mogo: RQ Hierarchical Causal Transformer for High-Quality 3D Human Motion Generation** (2024, arXiv preprint). [Paper](https://arxiv.org/abs/2412.07797) | Text-to-motion with RVQ-VAE and a hierarchical causal transformer: base motion AR plus residual-layer inference. Standard discrete generation; no verified contrastive condition objective, negative mining, or Gumbel. Implementation fields for alignment are not applicable/NR; official code NR. | Reports HumanML3D FID 0.079 vs 0.116 for T2M-GPT and generation beyond the dataset's usual sequence length. | Supports causal hierarchical RVQ factorization and streaming-style decoding. Preprint, text-only domain, and no evidence that residual-level generation remains sensitive under a dominant prefix. |
| **LLM-Codec: Neural Audio Codec Meets Language Model Objectives** (2026, Findings ACL). [Paper](https://aclanthology.org/2026.findings-acl.1308.pdf), [code](https://github.com/voidful/llm-codec) | Speech codec training. Adds future-token prediction and memory-bank audio-text contrastive semantic alignment. A differentiable Gumbel bridge sends gradients to the codec encoder. Positive: paired speech/text; negatives from memory bank. Projection, temperature, loss weights, bank size and batch are specified in the paper/code but are not copied here without a project-matched role; distributed strategy NR. | SALMon coherence reaches 61.6% (+12.1 points) and LM perplexity falls 35× while reconstruction remains competitive. | Evidence that discrete codec representations can receive semantic gradients through a Gumbel bridge. Here the codecs are frozen and the problem is coherent generation across 16 teacher-forced RVQ slots; its bridge does not solve that factorization mismatch. |

---

## 5. Critical comparison across domains

### 5.1 What gesture papers establish

Recent gesture systems establish five useful facts:

1. **Causal discrete generation is feasible.** MIBURI and LiveGesture remove the argument that future audio or a diffusion model is necessary for good co-speech motion.
2. **Regional tokenization is useful.** EMAGE, MIBURI, LiveGesture, and MambaTalk all preserve body-part structure rather than forcing one monolithic code.
3. **History corruption helps rollout robustness.** LiveGesture's UGR ablation directly supports training on imperfect histories.
4. **Text and audio carry different information.** JEGAL's local/global results and ConvoFusion's contribution analysis support distinct paths rather than naive concatenation.
5. **Semantic retrieval can activate rare gestures.** Semantic Gesticulator and GestureHYDRA can make explicit actions more reliable, but their repository/LLM/timestamp assumptions are outside the desired compact planner.

What most gesture papers do **not** establish is that the generated motion changes for the correct semantic reason. FGD, beat alignment, diversity, and even human naturalness can improve while shuffled text has little effect. LiveGesture's text-removal ablation is a particularly clear warning. MIBURI's contrastive loss contrasts real and predicted motion, not correct and wrong speech. SemGes aligns semantics to motion representations, but its published semantic-coherence equation has no explicit negative term even though surrounding prose discusses mismatches.

### 5.2 What retrieval/synchronization papers add

Synchformer, JEGAL, CAR, and VideoComp offer better negative construction than most generators:

- same-utterance distant intervals remove easy speaker/background cues;
- event-reordered text tests chronology rather than bag-of-words similarity;
- partial/action-replacement corruptions create graded semantic difficulty;
- late interaction exposes sparse local correspondence that global pooling hides.

Their limitation is fundamental: a representation can retrieve the right pair while the generator ignores that representation. Therefore retrieval must be an intermediate metric, followed by correct-vs-corrupted likelihood and generated-output interventions.

### 5.3 What image/video conditioning papers add

CCA and ControlNet++ supervise **outcomes**, not attention. CCA is especially transferable because the target planner is also a discrete autoregressive model. It needs no differentiable decoding: sequence log likelihood is already differentiable. ControlNet++ demonstrates the same principle for diffusion, but its extra reward encoder and denoising path are unnecessary here.

GenTron's cross-attention result transfers at the architectural level: full free-form text benefits from token-level access. It does not imply that cross-attention alone forces use. Classifier-free/modality dropout helps, but only a paired intervention establishes reliance.

### 5.4 Causal compatibility audit

| Method component | Deployable as-is? | Reason |
|---|---|---|
| Complete transcript cross-attention | Yes | Text is known before generation; seeing all text is not leakage. |
| Word-aligned local text loss from JEGAL/EMAGE/SemTalk | No | Word timestamps or forced alignment are unavailable. |
| Causal raw Mimi embedding window | Yes | Use only frames whose boundary is at or before the anchor. |
| Frozen HuBERT/wav2vec features from full utterance | No | Common encoders are bidirectional and leak future audio as used. |
| Synchformer 0.64 s window | Only after causalization | Window length transfers; its encoders do not. |
| Full-utterance diffusion gesture models | No | They use future condition context and noncausal generation. |
| MIBURI/LiveGesture causal factorization | Yes in principle | Their causal attention/token schedules are compatible; foundation-model and codec details differ. |
| Event-shuffled text hard negatives | Yes for utterance likelihood | They test order/semantics, but no anchor-level timing claim is allowed. |
| Future-shifted audio negative | No for training the deployable path | A negative may not expose the anchor to audio that had not arrived. Use past shifts/cross-clip audio; reserve future exposure only for a leakage audit. |

### 5.5 Segment representation

Only absolute pose/code embeddings are a weak contrastive target. A head can distinguish clips through body proportions, stance, camera reconstruction artifacts, speaker style, or long-lived pose without learning expressive motion. Recent motion systems repeatedly use velocity, acceleration, or phase/rhythm features; CAR includes relative positions and accelerations; EMAGE explicitly reconstructs velocity and acceleration.

For this planner, the most economical target is the concatenation of frozen absolute region latents and first differences. It is already available from the four causal RVQ codebooks and needs no new encoder. Local speech alignment should emphasize upper body and hands, where co-speech evidence is strongest, while the global text trajectory can use all four regions. Feet/lower-body condition gates should be learned and reported rather than forced to match speech equally.

### 5.6 Why attention supervision is not recommended

An attention weight is a routing coefficient, not a causal effect. A high weight can multiply an uninformative value; later residual paths can cancel the result; and supervising a token-frame map without timestamps injects an unverifiable alignment target. ConvoFusion's word-excitation guidance improves semantic recall, so attention guidance can be a useful control heuristic. It should not be used as the primary evidence that the planner uses text. The decisive quantities are:

\[
\Delta_{text}=\operatorname{NLL}(y\mid T_{shuf},A,h)-\operatorname{NLL}(y\mid T,A,h)
\]

and the paired change in generated anchors when only \(T\) is intervened on. The same applies to audio.

---

## 6. Recommended architecture

### 6.1 Verified workspace dimensions

The recommendation below is tied to the current workspace rather than a generic Transformer. The existing implementation uses:

| Symbol | Meaning | Verified value / shape |
|---|---|---|
| \(B_d\) | per-device training batch | 32 |
| \(B\) | global batch on four GPUs | 128 |
| \(L\) | serialized Qwen length | at most 2,560 in the q0–q3 config |
| \(d_h\) | Qwen hidden size | 896 |
| \(Q_a\) | Mimi levels used | 4: q0–q3 |
| \(V_a\) | Mimi cardinality per level | 2,048 |
| \(f_a\) | Mimi rate | 12.5 Hz |
| \(R\) | motion regions | 4: upper, lower, feet, hands |
| \(Q_m\) | motion RVQ levels per region | 4 |
| \(V_m\) | motion codebook size | 512 |
| \(d_c\) | frozen motion code dimension | 512 |
| \(J\) | IDs emitted per anchor | \(R Q_m=16\) |
| \(f_m\) | motion-token rate | 10 Hz |
| \(\Delta_i\) | default anchor spacing | 4 motion frames, about 0.4 s |
| \(d\) | proposed shared alignment dimension | 256 |

The current `MimiQwenPlanner` embeds each selected Mimi level into 896 dimensions, concatenates the four embeddings, and applies a learned linear 3,584->896 fusion only at real audio positions. Qwen predicts each motion slot from the hidden state immediately preceding that slot and restricts the classifier to the corresponding 512 vocabulary rows. These are favorable integration points: no second speech encoder is required, and the auxiliary can share the exact audio embeddings used by generation.

### 6.2 Required batch metadata

Extend the data collator with indices, not dense feature tensors:

- `text_mask: Bool[B_d,L]`: transcript token positions before `<|im_end|>`; exclude role/control tokens.
- `audio_anchor_id: Int[B_d,L]`: for each real `[mimi_frame]`, the next anchor index it can condition; `-1` elsewhere.
- `anchor_id: Int[B_d,L]`: anchor group for each of the 16 supervised target positions; `-1` elsewhere.
- `slot_id: Int[B_d,L]`: existing `target_slots`, values 0–15.
- `anchor_time: Int[B_d,N]`: 10 Hz token index for every anchor, padded with `-1`.
- `clip_id`, `speaker_id` when available, and normalized transcript hashes for negative masking.

The metadata must be derived from the same `causal_audio_boundaries` used to serialize the input. No feature window should be reconstructed from duration rounding independently; doing so risks a one-frame future leak.

### 6.3 History-masked condition-refresh branch

The base Qwen path remains unchanged and can see complete text, causal audio, and previous motion. The new branch sees text and audio but **never motion tokens**.

#### Causal audio representation

Let \(E^a_q\in\mathbb{R}^{2048\times896}\) be the existing trainable Mimi embedding for level \(q\), and \(W_f\in\mathbb{R}^{896\times3584}\) the existing q0–q3 fusion. For Mimi frame \(s\):

\[
x^a_s=W_f[ E^a_0(a_{s,0});E^a_1(a_{s,1});E^a_2(a_{s,2});E^a_3(a_{s,3})]
\in\mathbb{R}^{896}.
\]

For anchor \(i\) at time \(t_i\), collect only the last \(S=8\) available Mimi frames:

\[
A_i=[x^a_s]_{s=\max(0,b_i-7)}^{b_i}\in\mathbb{R}^{S_i\times896},
\qquad b_i=\max\{s:\text{Mimi frame }s\text{ arrived by }t_i\}.
\]

Eight frames correspond to 0.64 s, matching the useful sparse-sync scale in Synchformer and spanning more than the five new Mimi frames normally received between 0.4 s anchors. This value is a **recommended starting point**, not a reported optimum for this dataset.

Project and attention-pool with one learned query:

\[
\bar A_i=\operatorname{LN}(A_i W^a_{in})\in\mathbb{R}^{S_i\times256},
\quad
c^a_i=\operatorname{MHA}(q^a_i,\bar A_i,\bar A_i)\in\mathbb{R}^{256}.
\]

Use four heads and a 256->512->256 FFN. The query contains a learned anchor-index embedding and a 5-bin relative-window embedding. It contains no motion hidden state. All padding and keys after \(b_i\) are masked. A unit test must demonstrate that appending arbitrary future Mimi codes leaves every earlier \(c^a_i\) bitwise unchanged in evaluation mode.

#### Complete-text representation

Let \(T_b\in\mathbb{R}^{L_b^t\times896}\) be Qwen hidden states at transcript positions **before the motion sequence begins**. These states cannot contain motion history because of the causal sequence order. Although each token state is left-to-right, pooling over all transcript positions exposes the complete known transcript.

Project them:

\[
\bar T_b=\operatorname{LN}(T_bW^t_{in})\in\mathbb{R}^{L_b^t\times256}.
\]

Use the causal audio summary as a weak timing bridge:

\[
q^t_i=W_q[c^a_i;e_i^{anchor}],
\qquad
c^t_i=\operatorname{MHA}(q^t_i,\bar T_b,\bar T_b)\in\mathbb{R}^{256}.
\]

This allows different anchors to retrieve different transcript content even though timestamps are absent. It does **not** justify saying that a selected token is spoken at anchor \(i\). For an utterance-level text loss, separately attention-pool \(\bar T_b\) with a learned global query to obtain \(g^t_b\in\mathbb{R}^{256}\).

#### Injection before the 16 slot classifiers

Let \(h_{b,i,j}\in\mathbb{R}^{896}\) be the Qwen hidden state immediately before motion slot \(j\in\{0,\ldots,15\}\). It already contains the causal motion prefix and, for \(j>0\), previous within-anchor RVQ IDs.

Use learned slot-specific scalar gates, initially 0.5, and zero-initialized output projections:

\[
\tilde h_{b,i,j}=h_{b,i,j}
+\sigma(\gamma^a_j)W^a_o c^a_{b,i}
+\sigma(\gamma^t_j)W^t_o c^t_{b,i},
\]

where \(W^a_o,W^t_o\in\mathbb{R}^{896\times256}\) start at zero. The existing 512-row slot classifier consumes \(\tilde h\). This keeps initial logits identical to the baseline, then lets alignment/counterfactual gradients build a condition residual.

Do not make the first gates functions of \(h_{b,i,j}\). A history-conditioned gate can learn to close whenever motion history is predictive, recreating the shortcut. A later ablation may replace the scalars with condition-only vector gates.

Estimated new trainable size is approximately 3–5M parameters depending on whether audio/text attention share projections. The largest proposed projection, 4,096->512 for the whole-motion target, adds about 2.1M parameters. This is small relative to Qwen and does not add a pretrained encoder.

### 6.4 Frozen GT motion targets

Let the frozen codebook for region \(r\) and RVQ level \(q\) be

\[
C_{r,q}\in\mathbb{R}^{512\times512}.
\]

For GT anchor IDs \(y_{i,r,q}\), construct each regional residual sum:

\[
e^{gt}_{i,r}=\sum_{q=0}^{3} C_{r,q}[y_{i,r,q}]\in\mathbb{R}^{512}.
\]

All \(C\) tensors and codec decoders are frozen and detached.

For **local audio alignment**, emphasize the regions most plausibly tied to speech:

\[
x^{loc}_i=[e_{i,upper};e_{i,hands};
e_{i,upper}-e_{i-1,upper};e_{i,hands}-e_{i-1,hands}]
\in\mathbb{R}^{2048},
\]

\[
m^{loc}_i=\operatorname{norm}(P^{loc}_m(x^{loc}_i))\in\mathbb{R}^{256}.
\]

For **global text alignment**, use all regions:

\[
x^{all}_i=[e_{i,upper};e_{i,lower};e_{i,feet};e_{i,hands}]
\in\mathbb{R}^{2048},
\]

\[
m^{all}_i=\operatorname{norm}(P^{all}_m[x^{all}_i;x^{all}_i-x^{all}_{i-1}])
\in\mathbb{R}^{256}.
\]

Each projection is `LayerNorm -> Linear(input,512) -> SiLU -> Linear(512,256) -> L2 normalize`. This absolute-plus-difference design makes it harder to solve the objective from a static stance alone. Acceleration can be tested later; it should not be added before velocity proves useful.

Pool \(m^{all}_{b,1:N_b}\) with a learned 256-d attention query to obtain the utterance trajectory \(g^m_b\in\mathbb{R}^{256}\). Because this branch uses GT motion, it is a stable training target, not a claim about generated trajectory alignment.

### 6.5 Gradient routing

| Component | CE gradient | audio InfoNCE | text InfoNCE | counterfactual likelihood | Frozen? |
|---|---:|---:|---:|---:|---:|
| Qwen backbone | yes | no direct AM gradient | yes through pre-motion transcript states | yes on correct/wrong forward | no |
| Mimi q0–q3 embeddings + fusion | yes | yes | no direct TM gradient | yes for audio corruption | no |
| Audio attention pool | yes through injection | yes | no | yes for audio corruption | no |
| Text attention/global pools | yes through injection | no | yes | yes for text corruption | no |
| Slot gates and output residuals | yes after zero-init unlocks | no direct InfoNCE gradient | no direct InfoNCE gradient | yes | no |
| Existing slot logits/output embedding rows | yes | no direct GT-target gradient | no direct GT-target gradient | yes | no |
| Motion target projections | no | yes | yes | no | no |
| Frozen RVQ codebooks/decoders | no | **no; stop-gradient** | **no; stop-gradient** | no | yes |
| Soft/Gumbel predicted code latents | not used | not used | not used | not required | n/a |

The shared condition representations are both aligned and injected. InfoNCE shapes those representations; CE and especially the counterfactual likelihood train the gates/output residuals to make them affect logits. This separation is deliberate. An auxiliary head attached to an otherwise unused audio representation can achieve retrieval gains without affecting generation.

### 6.6 Conditioning and history corruption

Use three independent training corruptions, recorded in the batch log:

- Drop text with probability 0.10.
- Drop Mimi condition embeddings with probability 0.10.
- Corrupt 10–15% of prior motion-anchor IDs, plus drop an entire immediately previous anchor with probability 0.05.

Never drop both text and audio on more than 1–2% of ordinary CE examples. On counterfactual-ranking examples, the correct target modality must be present. These rates are recommendations for a 6K dataset; ConvoFusion's 10% modality nulling and LiveGesture's moderate history masking motivate the scale, but do not report an optimum for this planner.

History corruption complements, rather than replaces, the existing generated-history curriculum. Self-forcing exposes realistic model errors; random corruption broadens error coverage and prevents the model from assuming every prefix token is reliable.

---

## 7. Recommended losses

### 7.1 Main autoregressive loss

For generated or GT prefix \(\hat y_{<i}\):

\[
\mathcal L_{CE}=-\frac{1}{M}
\sum_{b,i,j}\log p_\theta
(y^{gt}_{b,i,j}\mid y^{gt}_{b,i,<j},\hat y_{b,<i},T_b,A_{b,\le t_i}).
\]

Keep the existing slot-restricted 512-way CE and generated-history schedule unchanged for the first comparison.

### 7.2 Local causal audio-motion multi-positive InfoNCE

Normalize the condition and target projections:

\[
u_{b,i}=\operatorname{norm}(P_a(c^a_{b,i})),\qquad
v_{b,i}=m^{loc}_{b,i}.
\]

For each anchor, create up to three audio views that all obey causality: windows ending at \(t_i\), \(t_i-0.16\) s, and \(t_i-0.32\) s. They cover plausible gesture delay without using future speech. Let \(\mathcal P(n)\) be the audio views for the same motion anchor, and \(\mathcal K(n)\) all valid gathered candidates after masking.

\[
\mathcal L_{a\rightarrow m}=-\frac1N\sum_n
\log\frac{\sum_{p\in\mathcal P(n)}
\exp(u_n^\top v_p/\tau_a)}
{\sum_{k\in\mathcal K(n)}\exp(u_n^\top v_k/\tau_a)},
\]

with an analogous motion-to-audio term whose positive set contains the multiple causal audio views:

\[
\mathcal L_{AM}=\tfrac12(\mathcal L_{a\rightarrow m}+\mathcal L_{m\rightarrow a}).
\]

#### Negative construction

Use a mixture of:

- anchors from other clips in the global batch;
- same-clip anchors at least 0.8 s away;
- same-speaker cross-clip anchors when speaker labels exist, because they are harder and reduce a speaker-identity shortcut;
- past-shifted audio windows by 0.8 or 1.2 s.

Mask:

- same-clip anchors closer than 0.8 s;
- the other positive latency views;
- exact duplicate clips/segments;
- known duplicate transcript-condition groups where available.

Subsample at most four anchors per clip per step. Global batch 128 then provides up to 512 base anchor pairs before multi-positive views, which is ample and reduces correlated within-clip negatives.

### 7.3 Global text-trajectory InfoNCE

Let \(g^t_b\) be the normalized complete-transcript embedding and \(g^m_b\) the normalized GT trajectory embedding. Use symmetric InfoNCE over clips:

\[
\mathcal L_{TM}= -\frac{1}{2B}\sum_b
\left[
\log\frac{e^{g_b^{t\top}g_b^m/\tau_t}}
{\sum_{k\in\mathcal V(b)}e^{g_b^{t\top}g_k^m/\tau_t}}
+
\log\frac{e^{g_b^{m\top}g_b^t/\tau_t}}
{\sum_{k\in\mathcal V(b)}e^{g_b^{m\top}g_k^t/\tau_t}}
\right].
\]

`V(b)` excludes exact normalized transcript duplicates. Near-duplicate masking should be enabled only after manually validating a text-cosine threshold on a sample; indiscriminate semantic masking can remove legitimately different gestures described with similar words.

This loss supports **utterance-level semantic alignment only**. Do not report it as evidence that a particular text token aligns to a particular anchor.

### 7.4 Correct-versus-corrupted conditional likelihood

Define the mean target log likelihood for one modality condition \(c\) while holding target and motion prefix fixed:

\[
s_b(c)=\frac{1}{M_b}\sum_{i,j}
\log p_\theta(y^{gt}_{b,i,j}\mid y^{gt}_{b,i,<j},\hat y_{b,<i},c).
\]

Construct a corrupted condition \(c_b^-\):

- text: a batch derangement, rejecting identical transcript hashes and, where possible, keeping speaker fixed;
- audio: a same-clip past shift of at least 0.8 s or cross-clip same-speaker Mimi; never reveal future audio at an anchor.

The paired margin objective is:

\[
\mathcal L_{CF}=\frac1{|\mathcal B_{cf}|}\sum_b
\operatorname{softplus}\left(
\mu-[s_b(c_b^+)-s_b(c_b^-)]
\right).
\]

Set \(\mu=0.05\) nats/token initially. This is a project recommendation, not a literature-reported optimum. Alternate text and audio corruptions by step so each receives equal updates.

For minimum memory, use a sequential two-pass variant on 25% of the batch:

1. Perform the ordinary correct-condition forward/backward and save `s_pos.detach()`.
2. Before the optimizer step, run only the selected examples with wrong conditions and backpropagate `softplus(mu - (s_pos_detached - s_neg))`.

CE already raises correct-condition likelihood; the second pass explicitly lowers the wrong condition. This avoids retaining two full Qwen graphs. A stronger moderate design retains gradients through both sides or uses CCA's frozen-reference likelihood ratio.

### 7.5 Total objective and initial hyperparameters

\[
\mathcal L=
\mathcal L_{CE}
+\lambda_a\mathcal L_{AM}
+\lambda_t\mathcal L_{TM}
+\lambda_{cf}\mathcal L_{CF}.
\]

Recommended starting ranges, after every loss is normalized as written:

| Hyperparameter | Start | Small sweep | Status |
|---|---:|---:|---|
| InfoNCE \(\tau_a\) | 0.07 fixed | 0.05, 0.10 | project recommendation |
| InfoNCE \(\tau_t\) | 0.07 fixed | 0.05, 0.10 | project recommendation |
| \(\lambda_a\) | 0.03 | 0.02, 0.05 | project recommendation |
| \(\lambda_t\) | 0.015 | 0.01, 0.03 | project recommendation |
| \(\lambda_{cf}\) | 0.03 | 0.02, 0.05 | project recommendation |
| CF margin \(\mu\) | 0.05 nats/token | 0.02, 0.10 | project recommendation |
| CF sub-batch fraction | 0.25 | 0.125, 0.50 | project recommendation |

Do not learn temperature in the first run. A learned logit scale can mask collapse by racing toward an extreme; fixed temperatures make the first comparison easier to interpret. If learned later, clamp the effective temperature to 0.03–0.20 and log it.

### 7.6 Activation schedule

- Train CE alone for the first 5 epochs, matching the current minimum teacher-forcing phase.
- Linearly ramp \(\lambda_a\) or \(\lambda_t\) from zero to target over epochs 6–8 in the corresponding single-loss experiment.
- Start history corruption at 5% and ramp to the target by epoch 10.
- Activate counterfactual likelihood only after teacher-forced CE is below the existing gate and generated-prefix accuracy is measurably non-random; ramp over three epochs.
- Keep the current self-forcing activation gate. Do not activate all auxiliary losses in the first run.

### 7.7 Four-GPU negatives

All-gather only 256-d normalized embeddings, clip/anchor IDs, times, transcript hashes, and speaker IDs. Do not gather Qwen hidden tensors or logits. With up to 512 anchors, one fp32 embedding matrix is about 0.5 MB.

Use an autograd-capable `all_gather` for the query/key embeddings or explicitly document a local-gradient/remote-stop-gradient variant. Construct the mask after gathering so the denominator is identical on every rank. Tests should compare one-process and four-process losses on the same synthetic batch.

---

## 8. Three ranked experimental designs

### Rank 1 — Minimal-risk, recommended for the 6K run

**Components**

- History-masked 256-d audio/text condition-refresh branch.
- Zero-initialized injection before all 16 slot classifiers.
- One auxiliary at a time: local AM or global TM InfoNCE on frozen GT latents.
- After each single auxiliary passes, add 25% sequential negative-only counterfactual likelihood.
- 10% independent condition dropout and moderate history masking.
- No soft predicted motion, no Gumbel, no new pretrained encoder.

**Expected cost (engineering estimate, profile before launch)**

- Parameters: +3–5M.
- Memory: +5–10% for heads/extra states; CF stays near this range with sequential backward.
- Runtime: +5–12% without CF; +25–35% with a 25% wrong-condition pass.
- Communication: negligible relative to Qwen DDP because only 256-d embeddings are gathered.

**Why ranked first:** it directly targets condition use, keeps frozen codecs, avoids incoherent sampling, and fits four 4090s. Every component has an isolated diagnostic.

### Rank 2 — Stronger, moderately complex

**Additional components**

- Multi-positive latency views and both same-clip and same-speaker hard negatives.
- Condition-only vector gates per region/RVQ level.
- Correct and wrong likelihood both receive gradient on a 50% sub-batch.
- Optional cached frozen-baseline log probabilities implementing a CCA-style reference correction:

\[
d_b=[s_\theta(c^+)-s_\theta(c^-)]-[s_{ref}(c^+)-s_{ref}(c^-)],
\qquad \mathcal L=\operatorname{softplus}(\mu-\beta d_b).
\]

- Apply losses on both teacher-forced and generated-prefix examples, stratified in the logs.

**Expected cost**

- Parameters: +5–8M.
- Memory: +15–30% if paired graphs overlap; keep under control with checkpointing/microbatches.
- Runtime: +40–70%, depending on the fraction of wrong-condition forwards.

**Risk:** more interactions make a negative result harder to attribute; reference-score caching and false-negative logic add data plumbing.

### Rank 3 — Most principled, research-intensive

**Components**

- A truly sequential differentiable 16-slot rollout for predicted-latent alignment.
- At slot \(j\), draw a soft or ST-Gumbel token, convert it to an embedding, append it to the Qwen cache, and recompute slot \(j+1\) logits from that sampled prefix.
- Align the resulting coherent predicted anchor/trajectory to audio/text, optionally decode through the frozen causal codec for a lightweight condition-reconstruction reward.
- Combine with reference-corrected CCA and generated-prefix rollouts.

**Expected cost**

- Parameters: small, but activation graph large.
- Memory: +30–80% depending on cache differentiability and truncation.
- Runtime: about 1.5–2.5x for auxiliary examples because 16 sequential token steps and cache graphs must be retained or recomputed.

**Risk:** ST-Gumbel bias/variance, temperature sensitivity, cache autograd complexity, and compounded exposure errors. Use only after Rank 1 demonstrates that condition-sensitive likelihood predicts better generated-prefix motion.

---

## 9. Controlled ablation plan

Keep data order, optimizer, self-forcing schedule, evaluation clips, and total update count fixed. Use one seed for screening, then three seeds only for arms that pass the prespecified threshold.

### 9.1 Primary arms

| Arm | CE | audio InfoNCE | text InfoNCE | CF likelihood | expected distortion | Purpose |
|---|---:|---:|---:|---:|---:|---|
| A0 | yes | no | no | no | no | current q0–q3 self-forcing baseline |
| A1 | yes | yes | no | no | no | representation-level local audio effect |
| A2 | yes | no | yes | no | no | utterance semantic effect |
| A3 | yes | yes | yes | no | no | whether local/global heads coexist |
| A4 | yes | no | no | yes | no | direct condition-sensitivity effect |
| A5 | yes | yes | yes | yes | no | only after A1/A2/A4 individually pass |
| E1 | yes | no | no | no | yes | separate codec expected-distortion experiment |
| E2 | yes | best proven alignment only | as proven | as proven | yes | only after E1 and the chosen alignment arm each pass |

Do not start with A5 or E2. A combined improvement would otherwise be uninterpretable, and a combined failure would not reveal whether gradients conflict.

### 9.2 Mechanism ablations on the winning alignment arm

1. Remove condition-refresh injection but keep the retrieval heads. If retrieval remains high and generation sensitivity vanishes, the injection/shared path is necessary.
2. Let the alignment query see Qwen motion-history hidden states. If metrics rise but shuffle sensitivity falls, it is exploiting the shortcut.
3. Remove absolute motion features; retain differences only.
4. Remove differences; retain absolute features only.
5. Whole-body local AM vs upper+hands AM.
6. Cross-clip negatives only vs cross-clip + same-clip past shifts.
7. Single exact positive vs multi-positive causal latency window.
8. No history corruption vs random token corruption vs existing self-forcing only vs both.
9. Fixed scalar gates vs condition-only vector gates.
10. GT latent target vs condition-only hidden target. Test soft expected predicted latents only after these.

### 9.3 Discrete representation comparison

If the first-stage experiments succeed, compare the candidate choices from the brief:

| Choice | Recommendation | Reason |
|---|---|---|
| A. Frozen GT motion latents | **Use for AM and TM targets** | Stable, cheap, frozen-codec faithful; must share/inject condition path so the generator benefits. |
| B. Soft expected predicted latents | Later diagnostic only | Differentiable and cheap per slot, but later marginals remain teacher-forced and do not form one coherent anchor. |
| C. ST-Gumbel predicted latents | Do not use in Rank 1 | MIBURI supports possible expressiveness gains, but the exact RVQ prefix inconsistency applies. Valid only with sequential recomputation. |
| D. Internal planner hidden states | **Use on the condition side** | Direct gradient to Qwen and no discrete mismatch; must be constructed without motion history. |
| E. GT and predicted through separate objectives | **Best overall interpretation** | Use GT target + condition-only hidden alignment + output likelihood ranking. Add a coherent predicted-latent objective only later. |

The choice need not be identical for speech and text. Audio benefits from a short GT motion segment with dynamics. Text benefits from a global GT trajectory and correct-vs-shuffled output likelihood.

---

## 10. Evaluation protocol

### 10.1 Split and sampling controls

- Freeze the test split and every condition corruption before comparing checkpoints.
- Evaluate teacher-forced and full generated-prefix modes separately.
- For generated interventions, fix the seed, top-p/temperature, starting seed anchor, and all non-intervened modalities.
- Report results per region and RVQ level as well as globally; q0 accuracy alone can hide semantic changes in hands/residual levels.
- Use the same number of decoded clips for every arm and the same Step 2 checkpoint.

### 10.2 Representation metrics

1. **Local audio-motion retrieval:** R@1/5/10 and median rank among held-out anchor windows, including same-clip distractors.
2. **Global text-trajectory retrieval:** R@1/5/10 with duplicate transcripts collapsed or masked.
3. **Temporal-shift classification:** choose correct audio among past shifts of 0.4, 0.8, and 1.2 s.
4. **Collapse diagnostics:** per-dimension variance, mean pairwise cosine, singular-value spectrum, and projection norm.
5. **Shortcut probes:** speaker prediction from the alignment embeddings; high speaker accuracy with weak content retrieval is a warning.

These are representation diagnostics, not the primary success criterion.

### 10.3 Causal conditioning metrics

For text and audio separately, report:

\[
\Delta\mathrm{NLL}_{mod}=
\mathrm{NLL}(y\mid c^-_{mod})-
\mathrm{NLL}(y\mid c^+_{mod}).
\]

A positive value means the correct condition receives higher likelihood. Report it for:

- teacher-forced motion history;
- the exact same generated prefix supplied to both conditions;
- full free-running generation after the intervention point.

Also report an output intervention curve:

- fraction of the 16 anchor IDs that change under text/audio intervention;
- change by region and RVQ level;
- frozen-codebook expected distance between paired anchors;
- decoded joint/velocity difference while holding RNG fixed;
- persistence of the effect over the next 1, 2, 4, and 8 anchors.

An ideal model is neither invariant nor chaotic: the correct modality should produce localized, plausible changes rather than entirely unrelated motion.

### 10.4 Required condition tests

- Correct vs shuffled transcript, rejecting duplicates and near duplicates.
- Correct vs shuffled Mimi across clips, with same-speaker negatives where possible.
- Correct vs same-clip Mimi shifted into the past by 0.4/0.8/1.2 s.
- Text-only, audio-only, both, and neither.
- Motion-history replacement: GT, generated, randomly corrupted, and previous-anchor dropped.
- A leakage audit that appends/randomizes future Mimi after each boundary and confirms earlier logits are unchanged. Future audio may be used for this **audit only**, never as a training input to an earlier anchor.

### 10.5 Motion and pipeline metrics

- Existing token CE, top-1, and q-level/region accuracy.
- Anchor-substitution FID/FGD: predicted anchors, GT-substituted anchors, and condition-corrupted predicted anchors.
- Frozen-codebook expected distortion and decoded anchor joint error.
- Velocity/acceleration error at anchor times.
- Diversity across repeated samples and cross-clip distribution.
- Repetition: immediate anchor-copy rate, n-gram/token-loop rate, long static-run duration, and codebook usage entropy.
- Beat/prosody alignment computed with a causal-compatible evaluator.
- Semantic human evaluation on a small balanced set of iconic/metaphoric vs beat-only utterances.
- Downstream Step 2 FID/FGD, joint/velocity error, foot sliding, transition discontinuity, and perceptual preference.

### 10.6 Statistical protocol

- Use paired bootstrap resampling over clips with 10,000 replicates and report 95% confidence intervals for every paired intervention delta.
- For retrieval, bootstrap by utterance, not by correlated anchor.
- Run at least three independent seeds for the final baseline and any claimed winner.
- Report the mean, seed standard deviation, paired effect size, and CI; do not rely on a single best checkpoint.
- Prespecify the main success metric as generated-prefix \(\Delta\mathrm{NLL}\) plus downstream Step 2 quality, not retrieval R@K.

### 10.7 Suggested go/no-go criteria

Advance an auxiliary from screening to three-seed evaluation only if it meets all of the following on validation:

1. Generated-prefix text or audio \(\Delta\mathrm{NLL}\) increases beyond the baseline 95% paired CI.
2. The matching correct-vs-corrupt generated-anchor distance increases without more than a 5% relative worsening in decoded anchor error/FGD.
3. Repetition does not increase.
4. Teacher-forced gain does not disappear under a fixed generated prefix.
5. Local/global retrieval improves for the intended modality, confirming that the head learned something coherent.

The 5% quality tolerance is a project decision and should be adjusted if downstream Step 2 is more sensitive than Step 1 metrics suggest.

---

## 11. Failure modes and falsification criteria

### 11.1 Results that falsify the main hypothesis

The alignment objective is ineffective for this planner if any of the following persists across three seeds:

- Retrieval improves materially but correct-vs-shuffled/shifted generated-prefix \(\Delta\mathrm{NLL}\) stays inside the baseline CI.
- Teacher-forced condition sensitivity improves but vanishes when the same generated prefix is used.
- Step 1 anchor metrics improve but Step 2 output is unchanged or worse.
- Shuffled conditions change projection embeddings but not token logits or sampled anchors.
- The model changes anchors under any perturbation, including semantically equivalent transcripts, indicating instability rather than meaningful reliance.

### 11.2 Representation collapse

Symptoms include near-constant embeddings, singular values concentrated in one dimension, temperature/logit scale saturating, or all samples achieving similar cosine. Prevent by L2 normalization, fixed temperature initially, monitoring embedding variance, and retaining diverse cross-clip negatives. If collapse occurs, first reduce auxiliary weight; do not immediately add a more complex self-supervised objective.

### 11.3 False-negative damage

Natural co-speech gesture is many-to-many. Two clips with similar speech may legitimately have different gestures, and two different transcripts may support the same beat gesture. Symptoms are worse generation/diversity despite higher retrieval, high loss on duplicate/near-duplicate captions, or over-separated semantically similar motion. Mitigate exact duplicates, a temporal guard band, multi-positive latency windows, same-speaker hard negatives, and a later sigmoid-pairwise ablation.

### 11.4 Motion-history shortcut survives

The condition branch may be history-free while the base Qwen path still dominates. Indicators are gates/projection norms near zero, unchanged output under condition swaps, or sensitivity only when history is removed. The counterfactual likelihood term and history corruption are the remedies. Increasing InfoNCE weight alone may only improve the side head.

### 11.5 Speaker/style shortcut

Audio and motion share speaker identity, recording session, body proportions, and habitual pose. Random cross-clip negatives make these easy discriminators. Diagnose same-speaker retrieval, speaker prediction from the 256-d embeddings, and cross-speaker condition swaps. Use same-speaker negatives and upper/hands temporal differences. Do not infer semantic alignment from high cross-speaker retrieval without these controls.

### 11.6 Retrieval-generation disconnect

This is the central risk. A frozen GT motion target lets projections learn a beautiful joint space while Qwen logits remain history-driven. The shared injection path and likelihood intervention are mandatory. If removing injection leaves retrieval unchanged but destroys generated sensitivity, that is expected and confirms the distinction.

### 11.7 Gumbel/RVQ mismatch

With teacher forcing, independent Gumbel samples across q0–q3 can look plausible after codebook summation but are not a sample from the model. Symptoms include auxiliary loss improvement with worse free-running CE, discontinuous temperature dependence, or predicted-latent metrics that cannot be reproduced by hard sequential decoding. The falsification test is direct: compare the “parallel” auxiliary latent with a sequentially sampled anchor under identical logits/prefixes. If later logits differ after feeding the sampled earlier ID, the parallel latent was incoherent.

### 11.8 Condition corruption becomes too easy

Cross-speaker shuffled audio can be rejected by voice identity; random text may be rejected by topic. The model can win the margin without learning timing or semantics. Track performance by corruption difficulty and prioritize same-speaker shifts, event-order corruptions, and semantically similar transcript derangements.

### 11.9 Gradient conflict

Log the L2 norm and cosine between CE, AM, TM, and CF gradients on the audio fusion, condition adapter, and a late Qwen block once per epoch. If auxiliary gradients exceed 20–30% of CE norm or have persistently negative cosine while quality worsens, reduce/ramp weights. Only then consider GradNorm or MMPareto; complex balancing should respond to measured conflict, not be the default.

---

## 12. Final recommendation

### 12.1 Implement this first

Build **Rank 1** as a small extension of `motion_generation/models/step1_mimi_planner.py`:

1. Expose the already fused real Mimi embeddings before they replace `[mimi_frame]` placeholders.
2. Add collator index tensors for transcript positions, audio-to-anchor grouping, and anchor/slot grouping.
3. Add the 256-d history-free audio pool, text cross-attention pool, and zero-initialized condition residual.
4. Load the four frozen 512x4, 512-d codec codebooks only for target construction; detach them and exclude them from checkpoints/optimizer.
5. Implement local upper+hands absolute/difference targets and whole-body trajectory targets.
6. Implement global-embedding all-gather with explicit masks.
7. Run A1 and A2 separately. Do not add CF until their heads and causal masks pass unit tests.
8. Add A4 counterfactual likelihood on a sequential 25% wrong-condition pass.
9. Only after A1/A2/A4 individually improve their intended generated-prefix metric, run A3 and then A5.
10. Keep expected distortion in its separate E1 arm until both lines of evidence are established.

### 12.2 Pseudocode for the recommended loss

```python
# Shapes on each GPU:
# input_ids, attention_mask, target_slots: [Bd, L], Bd=32, L<=2560
# audio_codes: [Bd, L, 4]
# qwen_hidden: [Bd, L, 896]
# anchor_count N varies by clip; sampled alignment anchors <=4/clip

inputs_embeds, fused_audio_at_real_positions = planner.prepare_embeddings(
    input_ids, audio_codes, return_fused_audio=True
)
qwen_hidden = planner.qwen_base(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    use_cache=False,
).last_hidden_state

# Condition-only states. Audio uses input embeddings, not Qwen audio hidden
# states, because the latter already contain motion history.
audio_windows = gather_last_causal_frames(
    fused_audio_at_real_positions,
    audio_anchor_id,
    max_frames=8,
)                                      # ragged/padded [Bd,N,8,896]
c_audio = audio_pool(audio_windows)    # [Bd,N,256]

text_states = gather(qwen_hidden, text_mask)  # ragged [Bd,Lt,896]
c_text = text_pool(
    query=torch.cat([c_audio, anchor_index_embedding], dim=-1),
    key_value=text_states,
)                                      # [Bd,N,256]

# Inject into the hidden immediately preceding each target slot.
h_slot = gather_predecessor_hidden(qwen_hidden, anchor_id, target_slots)
# h_slot: [M,896], with matching anchor_idx[M] and slot_idx[M]
h_cond = (
    h_slot
    + sigmoid(gate_audio[slot_idx])[:, None] * out_audio(c_audio[anchor_idx])
    + sigmoid(gate_text[slot_idx])[:, None] * out_text(c_text[anchor_idx])
)
slot_logits = restricted_512_logits(h_cond, slot_idx)
loss_ce, per_example_logp = slot_cross_entropy(slot_logits, labels)

with torch.no_grad():
    # Each region: sum four frozen 512-d RVQ code vectors.
    e_gt = frozen_rvq_sum(gt_anchor_ids)  # [Bd,N,4,512]

motion_local = torch.cat([
    e_gt[:, :, UPPER], e_gt[:, :, HANDS],
    diff(e_gt[:, :, UPPER]), diff(e_gt[:, :, HANDS]),
], dim=-1)                               # [Bd,N,2048]
motion_all = torch.cat([
    e_gt.flatten(-2), diff(e_gt.flatten(-2))
], dim=-1)                               # [Bd,N,4096]

u_audio = l2norm(proj_audio(c_audio))
v_motion_local = l2norm(proj_motion_local(motion_local))
loss_am = distributed_multi_positive_infonce(
    u_audio, v_motion_local,
    positives=causal_latency_positive_mask,
    valid_negatives=negative_mask,
    temperature=0.07,
)

g_text = l2norm(global_text_pool(text_states))       # [Bd,256]
g_motion = l2norm(global_motion_pool(proj_motion_all(motion_all)))
loss_tm = distributed_symmetric_infonce(
    g_text, g_motion,
    valid_negatives=~duplicate_transcript_mask,
    temperature=0.07,
)

loss_main = loss_ce + lambda_a * loss_am + lambda_t * loss_tm
loss_main.backward()  # normal DDP accumulation; keep optimizer step pending

# Alternate text/audio corruption. Run on 25% of examples sequentially so a
# second full Qwen graph does not overlap the first one.
idx = counterfactual_subset(Bd, fraction=0.25)
wrong_batch = replace_one_condition(
    batch[idx],
    modality=alternating_modality,
    text_derangement=duplicate_safe_derangement,
    audio_corruption=past_shift_or_same_speaker,
)
s_neg = planner.mean_target_logp(wrong_batch)        # [len(idx)]
s_pos = per_example_logp[idx].detach()
loss_cf = F.softplus(0.05 - (s_pos - s_neg)).mean()
(lambda_cf * loss_cf).backward()

clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### 12.3 Staged implementation and verification checklist

#### Stage A — data and causality

- [ ] Add text/audio/anchor grouping indices without changing serialized `input_ids`.
- [ ] Assert every audio frame assigned to anchor \(i\) has source index \(\le b_i\).
- [ ] Append random future Mimi and verify all earlier audio pools/logits are unchanged.
- [ ] Verify text states are taken only before `[motion_start]` and therefore cannot contain motion history.
- [ ] Unit-test padded clips and final short audio windows.

#### Stage B — frozen motion targets

- [ ] Load all four causal codec checkpoints and verify codebook shape `[4,512,512]` per region.
- [ ] Compare summed codebook latent against the codec quantizer output for sampled GT IDs.
- [ ] Confirm every codec parameter has `requires_grad=False` and zero gradient after backward.
- [ ] Verify temporal differences reset/mask at the first anchor rather than crossing clips.
- [ ] Cache only codebook matrices, not full codecs, during planner training if equivalence is proven.

#### Stage C — condition adapter

- [ ] With zero-initialized output projections, logits and CE match the baseline to numerical tolerance.
- [ ] Confirm audio/text condition tensors have shapes `[B,N,256]` and no motion-token input edge.
- [ ] Confirm all 16 slot positions receive the intended anchor condition and slot gate.
- [ ] Log gate values and condition residual/base-hidden norm ratio by slot/region.
- [ ] Verify gradients reach Mimi embeddings, text-side Qwen states, adapter, and selected Qwen/output parameters.

#### Stage D — distributed losses

- [ ] Compare single-GPU and four-GPU InfoNCE on a deterministic synthetic batch.
- [ ] Verify positive, guard-band, duplicate, and padding masks after all-gather.
- [ ] Assert every query has at least one positive and one valid negative.
- [ ] Log the number of positives/negatives removed by each mask.
- [ ] Test that remote embeddings receive gradients if that is the selected implementation.

#### Stage E — counterfactual likelihood

- [ ] Transcript derangement has no fixed points and rejects exact duplicates.
- [ ] Past-shifted Mimi never reads a frame later than the current causal boundary.
- [ ] Correct and wrong examples share identical target anchors and motion prefixes.
- [ ] Validate `s_pos`/`s_neg` against manually summed slot log probabilities.
- [ ] Run the wrong-condition backward sequentially before one shared optimizer step.

#### Stage F — experiments

- [ ] Reproduce A0 before changing losses.
- [ ] Run A1, A2, and A4 separately with one screening seed.
- [ ] Evaluate teacher-forced, fixed generated-prefix, and full rollout condition gaps.
- [ ] Run A3/A5 only if individual arms pass.
- [ ] Keep E1 separate; combine with alignment only after both pass individually.
- [ ] Run three seeds and paired bootstrap CIs for the final claim.
- [ ] Decode identical clip sets through frozen Step 2.

### 12.4 Decisions requiring developer input

These choices are unresolved by the literature and materially affect implementation. Defaults are provided so they need not block a first branch.

| Decision | Default for first run | Why it remains open |
|---|---|---|
| Qwen update scope | update full Qwen as current training does; keep codecs frozen | Freezing lower Qwen layers may save memory but could prevent text/audio adaptation. |
| Audio window | 8 Mimi frames (0.64 s) | Dataset-specific gesture latency has not been measured. |
| Same-clip negative guard | 0.8 s | Nearby anchors may remain semantically/kinematically equivalent; tune from autocorrelation. |
| Local target regions | upper + hands | Lower/feet may carry beat/posture cues; forcing equal alignment risks style shortcuts. |
| Text pooling | token cross-attention queried by causal audio; global learned pool for TM | Final transcript token is cheaper but may lose token-specific content. |
| Counterfactual fraction | 25%, alternating modality | Compute/sensitivity trade-off must be profiled on 4090s. |
| Correct-side CF gradient | detach in Rank 1 | CE already raises correct likelihood; full paired gradients cost more memory. |
| Near-duplicate text mask | exact normalized duplicates first | Semantic threshold requires manual calibration to avoid overmasking. |
| Gate granularity | one scalar per modality per slot | Vector gates add expressiveness but can overfit 6K clips. |
| Gumbel experiment | defer | It requires a new sequential cached autograd path and should follow causal-sensitivity success. |

### Final judgment

The tentative `CE + audio InfoNCE + text InfoNCE` formulation is a sound representation-learning baseline but not a complete answer to the observed shortcut. The recommended formulation is:

\[
\boxed{
\mathcal L=
\mathcal L_{CE}
+\lambda_a\mathcal L_{AM}^{GT\leftrightarrow condition}
+\lambda_t\mathcal L_{TM}^{GT\leftrightarrow condition}
+\lambda_{cf}\mathcal L_{correct>corrupt}
}
\]

with a history-free condition branch that is injected into every output slot, frozen GT RVQ targets using absolute and difference features, strict causal audio windows, duplicate-aware global negatives, moderate history/condition corruption, and generated-prefix evaluation. The counterfactual likelihood term is the key addition: without it, retrieval can improve while the planner remains functionally invariant to speech and text.

---

## Foundational background (pre-2024, deliberately limited)

- [Contrastive Predictive Coding / InfoNCE](https://arxiv.org/abs/1807.03748) (2018) introduced the contrastive predictive formulation used by later cross-modal work.
- [CLIP](https://proceedings.mlr.press/v139/radford21a.html) (ICML 2021) established large-batch symmetric image-text InfoNCE.
- [SigLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) (ICCV 2023) replaces the batch softmax with independent sigmoid pair classification.
- [Gumbel-Softmax](https://openreview.net/forum?id=rkE3y85ee) (ICLR 2017) supplies differentiable categorical relaxation but does not by itself make a teacher-forced autoregressive sample coherent.
- [Differentiable Scheduled Sampling](https://aclanthology.org/P17-1110/) (ACL 2017) relaxes feedback decisions so later states can depend on differentiable earlier choices—the relevant principle for a valid sequential 16-slot rollout.
- [TMR](https://openaccess.thecvf.com/content/ICCV2023/html/Petrovich_TMR_Text-to-Motion_Retrieval_Using_Contrastive_3D_Human_Motion_Synthesis_ICCV_2023_paper.html) (ICCV 2023) is the text-motion retrieval backbone strengthened by CAR.
- [DiT](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) and [ControlNet](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) (ICCV 2023) are the modulation/control baselines discussed by recent conditioning papers.

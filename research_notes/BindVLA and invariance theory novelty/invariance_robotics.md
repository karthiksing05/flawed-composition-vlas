# Invariance penalties in robot imitation / VLAs, and position-free binding keys: novelty check (as of 2026-10-27)

Legend: [full text] = arXiv HTML read through WebFetch (a summarising fetch, so the main body was read but not every appendix); [abstract] = abs page only; [snippet] = search-result snippet only; [prior] = from model memory, NOT re-verified in this session (check before citing).

Components under test:
- C1: a V-REx-style variance-of-loss penalty, computed across scene layouts, on a VLA's flow-matching loss.
- C2: binding attention whose keys have the positional embedding removed, with position entering only through a relative-geometry channel. Motivation: we measured that ViT patch features linearly encode absolute position (R^2 0.72/0.52 SigLIP, 0.89/0.85 DINOv2), and a binder then binds "left" as a remembered location.

## (a) Have invariance penalties (IRM / V-REx / REx / Fishr / group DRO / ICP) been used across environments in robot imitation learning or VLAs?

### Takeaway
I found no paper, through October 2026, that applies IRM, V-REx, Fishr or group DRO across scene layouts (or any environment partition) to a VLA or a flow-matching/diffusion policy. The closest prior work has three branches:
- **IRMv1 in adversarial imitation** on MuJoCo (Ovinnikov & Buhmann 2023), with environments defined as experts or replay-buffer policies.
- **Invariant-representation imitation** (ICIL, NeurIPS 2021) on control and healthcare tasks.
- **Recent VLA robustness work** that uses consistency or contrastive losses and explicitly leaves layout out: RoVLA (2605.19678) and DILL (2609.37165). DILL's limitations section names object layout as an unaddressed shortcut.

The layout shortcut itself is well documented: LIBERO-Plus Finding 5 calls it "positional bias", and the LIBERO-PRO and AFI papers call it a "memory trap". Novelty for C1 is therefore real in application, but it is an application of an existing penalty.

### Cited Findings
- **Ovinnikov & Buhmann, "Regularizing Adversarial Imitation Learning Using Causal Invariance", arXiv 2308.09189** [full text]. [Source](https://arxiv.org/html/2308.09189v1)
  - Penalty: IRMv1 (gradient-norm penalty) on the discriminator's BCE loss, weighted by lambda. Not REx.
  - Environments: "settings" are either a varied set of expert demonstrations (in 2D, each expert has a different intermediate goal) or replay buffers from earlier policy iterations (SAC).
  - Tasks: 2D MoveP-v0 and MuJoCo Ant, HalfCheetah, Hopper and Walker2d, with 10 expert trajectories.
  - Results are mixed in 2D. For example, GAIL with 2 updates scores -22320 under ERM and -27169 under IRM with lambda=0.01. In MuJoCo, GAIL-IRM beats GAIL-ERM on Ant (4291 vs 2163) and Hopper (3316 vs 3080).
  - The authors call the results "preliminary". They list extending to the image domain as future work. No vision, no VLA, no layouts.
- **Bica, Jarrett & van der Schaar, "Invariant Causal Imitation Learning for Generalizable Policies" (ICIL), arXiv 2311.01489, NeurIPS 2021** [abstract]. [Source](https://arxiv.org/abs/2311.01489)
  - Learns a causal-feature representation shared across training environments and disentangled from environment-specific noise. Adds an energy-based regularizer on the imitator's next states.
  - Environments differ in observable features and dynamics. Evaluated on control and healthcare tasks, not image-based manipulation.
  - The abstract gives no invariance-penalty formula.
- **RoVLA, "Multi-Consistency Constraints for Robust Vision-Language-Action Models", arXiv 2605.19678 (19 May 2026; Luo, Wen, Bai, Song, Liu, Lin; HCPLab-SYSU)** [full text]. [Source](https://arxiv.org/html/2605.19678v1)
  - Model: a flow-matching VLA (InternVL3.5-2B feeding a 32-layer DiT; chunk 16; 8 Euler steps).
  - Losses:
    - IC: paraphrase augmentation only, with about 15 Qwen3-8B paraphrases per trajectory.
    - EC: ||v̂^τ1 − v̂^τ2||² across flow timesteps.
    - OC: adversarial feature-space perturbation of visual features and robot state, with the output pulled toward the stop-gradient clean prediction.
    - Total: L = L_SFT + λ·½(L_EC+L_OC), with λ = 1/(1+L_ema).
  - No IRM, REx or variance across environments. No layout or object-position mechanism.
  - LIBERO-Plus Layout: 73.0 zero-shot and 74.1 when trained on LIBERO-Plus. The authors say π0.5 is better on Layout and Robot. Ablation (LIBERO-Plus total): backbone 77.1, full model 82.0.
- **DILL, "Disentangling Spurious Correlations in VLA Models via Predicting Domain-Invariant Latent Lookahead", arXiv 2609.37165 (submitted 29 Sep 2026; Kim, ..., Kang, Byoung-Tak Zhang; CoRL 2026)**. The ID is verified. [full text, first ~100k of 147k chars; Appendix B not read]. [Source](https://arxiv.org/html/2609.37165v1)
  - Method: a Task-Domain Encoder trained with a task InfoNCE loss, a domain InfoNCE loss and a SIGReg Gaussian disentanglement loss. The policy predicts future task latents.
  - "Domain" means viewpoint, appearance, camera heterogeneity and visual degradation. Layout is not included.
  - No IRM, REx or DRO. It cites de Haan's causal confusion.
  - Its counterfactual test binds Task-L and Task-R to left and right view ranges during training, then swaps them at test time.
  - Main LIBERO-Plus table reports Camera, Light, BG and Noise only. Average 69.1%, +11.4pp over the strongest baseline.
  - **Limitations explicitly name object layouts, initial states and language templates as non-visual shortcuts it does not cover.**
- **LIBERO-Plus, arXiv 2510.13626** [full text]. [Source](https://arxiv.org/html/2510.13626v1)
  - Layout has two parts: O1 adds distractors drawn from 416 objects; O2 changes the target's initial position and orientation.
  - Finding 5: "Current VLAs exhibit positional bias rather than genuine semantic understanding of objects." Adding distractors costs little, but moving the target costs a lot; the models may have "merely learned the positional information of the target objects".
  - Layout drops: UniVLA 95.2→34.3, OpenVLA 76.5→31.6, π0 94.2→70.4, π0-Fast 85.5→70.3, OpenVLA-OFT 97.1→77.1.
  - The paper does not attribute the bias to the visual encoder's position encoding. Its only remedy is mixed fine-tuning on the generalized data, which moves layout from 68.7 to 77.6.
- **LIBERO-PRO (arXiv 2510.03827)** and **Affordance Field Intervention (arXiv 2512.07472, CVPR 2026)** [snippet]. [LIBERO-PRO](https://arxiv.org/pdf/2510.03827); [AFI](https://www.arxiv.org/pdf/2512.07472)
  - Under position perturbations, policies reach toward memorized locations. AFI calls this a "memory trap".
  - AFI is training-free. It reports +21.7% on Spatial and +16.8% on Object for LIBERO-Pro position perturbations.
- **Selected Diffusion Noise, arXiv 2606.14084 (Jun 2026)** [abstract]. [Source](https://arxiv.org/abs/2606.14084)
  - A test-time noise-selection method for "spurious-robust" diffusion VLAs (π0, GR00T N1.5/N1.6), with about +8% in simulation and +10% real.
  - No environment partition and no invariance penalty.
- **G3VLA, arXiv 2606.24472** [snippet]. [Source](https://arxiv.org/html/2606.24472v1)
  - Adds camera-aware projective positional encoding (PRoPE) to VLA visual tokens. Removing PRoPE drops 87.0→85.9.
  - It adds geometric position. It does not remove absolute position.
- **Two extended searches found no VLA IRM, REx or DRO paper.** One searched IRM/REx/DRO with VLA, layout and spurious correlation. The other searched V-REx, risk extrapolation and group DRO with imitation. The search tool summaries said so explicitly. [Search 1 result: DILL](https://arxiv.org/html/2609.37165v1); [REx paper](https://arxiv.org/pdf/2003.00688)
- **V-REx (Krueger et al., arXiv 2003.00688, ICML 2021)** [abstract/snippet]. [Source](https://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)
  - V-REx penalizes the variance of training risks. MM-REx extrapolates over affine combinations of risks, where group DRO uses only convex combinations.
  - REx is claimed to beat IRM when causally induced and covariate shift co-occur.
- **Pareto IRM / PAIR, arXiv 2206.07766 (ICLR 2023)** [abstract]. [Source](https://arxiv.org/abs/2206.07766)
  - OOD objectives relaxed into penalties added to ERM are weakened by the relaxation, conflict with ERM, and need careful weight tuning. PAIR treats this as a multi-objective problem. Results are on WILDS.
  - The abstract does not itself name V-REx or a "few environments" failure. Those claims sit in the body, which I did not read.
- [prior] **Further objections to cite** (not re-verified this session):
  - Rosenfeld, Ravikumar & Risteski, "The Risks of Invariant Risk Minimization" (arXiv 2010.05761): with fewer environments than spurious dimensions, IRM/REx-type objectives can fail.
  - Kamath et al. 2101.01134: IRMv1 can miss the invariant predictor.
  - Gulrajani & Lopez-Paz, DomainBed (2007.01434): tuned ERM matches IRM, V-REx and group DRO.
  - Fishr (Rame et al., 2109.02934): matches gradient variances across domains.
  - de Haan et al., "Causal Confusion in Imitation Learning" (1905.11979): nuisance correlates hurt behavior cloning; addressed with targeted interventions.

### Inferences
- **C1 novelty.** A variance-of-loss penalty across layout environments, applied to the per-environment flow-matching loss of a VLA, appears unpublished. The nearest work uses IRMv1 in GAIL on MuJoCo with no vision. RoVLA and DILL target robustness in 2026 VLAs with non-environment losses and explicitly do not handle layout.
  - So the claim "the first invariance-penalty (REx) treatment of the documented VLA layout shortcut" looks defensible.
  - "A new invariance method" would not be defensible.
- **Strongest reviewer objection to C1.** A layout partition gives few, similar environments.
  - Rosenfeld et al. (and Pareto IRM's relaxation argument) predict V-REx then reduces to ERM plus noise, or fails to remove a spurious feature. Absolute target position has many degrees of freedom compared with a handful of layouts.
  - A second objection is that layout randomization or data augmentation is the obvious competing baseline. LIBERO-Plus already gets +8.9 layout points from mixed fine-tuning.
  - The paper needs (i) ERM with the same layout-diverse data, and (ii) group DRO and Fishr as alternatives. DomainBed-style tuning of the ERM baseline is essential.
- **Loss-scale confound.** In flow matching, the per-environment loss is dominated by noise level τ and action variance. A raw loss variance may penalise environments that are intrinsically harder (longer reaches) rather than spurious features. Matching the τ sampling per environment, or normalising per environment, is needed. No paper addresses this, which is both a gap and a reviewer hook.

### Gaps
- Not read: DILL Appendix B (counterfactual design and the per-dimension LIBERO-Plus table with Layout), and the Pareto IRM body.
- ICIL was read at the abstract level only. Full-text penalty details are not confirmed.
- No 2025–2026 VLA paper using IRM/REx/DRO across tasks or embodiments was found. Absence is from search, not exhaustive arXiv listing.
- I could not verify de Haan follow-ups (e.g. object-aware regularisation, "Fighting Copycat Agents"); they were not fetched.

## (b) Position-free / position-disentangled attention keys and role-filler separation for systematicity; is "ViT features encode absolute position, causing layout shortcuts in grounding" known?

### Takeaway
Each half of C2 has precedent:
- **Role/filler split.** In Russin et al. syntactic attention and Chakravarthy, Russin & O'Reilly NAACL-SRW 2022, a role stream drives Q/K and a filler stream drives V.
- **Relative-only position in robot attention.** Act3D and 3D Diffuser Actor use 3D rotary relative attention because "the coordinate system is arbitrary".
- **Slot-centric relative frames.** Invariant Slot Attention, 2302.04973.
- **Linear decodability of absolute position from ViT patch features** is known in general (multiple probing papers). A 2026 paper shows it is causally used (permuting APE features costs 24–31 top-1 points).

What I found no paper doing:
- measuring absolute-position decodability in the SigLIP/DINOv2 features of VLA encoders and linking it causally to the LIBERO layout "positional bias";
- building a binder whose keys are stripped of absolute position.

LIBERO-Plus documents the behavior but does not attribute it to the encoder. The specific combination appears novel. The general idea, "remove absolute position, keep relative geometry", is not.

### Cited Findings
- **Chakravarthy, Russin & O'Reilly, "Systematicity Emerges in Transformers when Abstract Grammatical Roles Guide Attention", NAACL-SRW 2022** [snippet]. [Source](https://aclanthology.org/2022.naacl-srw.1/)
  - A role stream sets the attention distributions (queries/keys) and a filler stream sets the values. Systematic generalization improves when abstract role labels are given to the role stream.
- [prior] **Russin, Jo, O'Reilly & Bengio, "Compositional generalization in a deep seq2seq model by separating syntax and semantics" (syntactic attention), arXiv 1904.09708**. Attention is computed from a syntactic stream and values come from a semantic stream, giving large gains on SCAN add-jump. Not re-fetched this session.
- **"Attention as Binding: A Vector-Symbolic Perspective", arXiv 2512.14709** [snippet]. [Source](https://arxiv.org/pdf/2512.14709)
  - A theoretical reading: keys are roles, values are fillers, and binding works best with near-orthogonal keys. No empirical systematicity test.
- **NoPE, Kazemnejad et al., arXiv 2305.19466** [snippet]. [Source](https://arxiv.org/pdf/2305.19466)
  - A Transformer without positional encoding trained with SGD learns position representations resembling T5 relative PE, and length generalization improves.
  - Relevant objection: with causal masking, position can re-enter even when it is not given explicitly.
- **Neural Data Router, arXiv 2110.07732** [snippet]. [Source](https://arxiv.org/pdf/2110.07732)
  - The model falls back to position-based attention on some tasks. The fix lets heads choose between absolute and relative PE.
- **Act3D, arXiv 2306.17817 (CoRL 2023)** [snippet]. [Source](https://arxiv.org/pdf/2306.17817)
  - Relative attention implemented as an inner product in an extended position space via rotary embeddings, justified because the 3D coordinate system is arbitrary. +10% absolute over the previous 2D multi-view SOTA on 74 RLBench tasks.
- **3D Diffuser Actor, arXiv 2402.10885** [snippet]. [Source](https://arxiv.org/html/2402.10885v2)
  - Uses 3D relative attention between end-effector pose tokens and visual/language tokens.
  - These two papers are the direct precedent for "position only via a relative-geometry channel" in manipulation. They use 3D point features from depth, not a VLM's 2D ViT tokens, and they do not strip absolute position from pretrained ViT features.
- **Invariant Slot Attention, arXiv 2302.04973** [abstract]. [Source](https://arxiv.org/abs/2302.04973)
  - Slot-centric reference frames; equivariance is obtained by translating, scaling and rotating position encodings relative to each slot.
  - The abstract does not give the mechanics (full text not read). [prior] Each slot has a position; keys and values receive relative-grid encodings, so slot content is position-free.
- **Mannes et al., "How Vision Transformers Represent Position: APE vs RoPE", arXiv 2610.13992 (26 Sep 2026)** [abstract + snippet]. [Source](https://arxiv.org/abs/2610.13992)
  - SAEs recover row- and column-specific features in APE ViTs.
  - Permuting the APE positional features cuts top-1 accuracy by 24–31 points, so position is used, not just decodable.
  - RoPE ViTs rebuild position inside attention, and disabling rotations removes decodable information.
  - Snippet: zeroing the identified SAE features drops exact-position recovery from 99.7% to 22.9% in supervised ViT-B.
  - It probes generic ViTs. The models are not named as SigLIP or DINOv2 on the abs page. No shortcut framing.
- **"What DINO saw: ALiBi positional encoding reduces positional bias in Vision Transformers", arXiv 2603.16840** [snippet]. [Source](https://arxiv.org/pdf/2603.16840)
  - Linear probes of left-right, up-down, diagonal and radial ramps on DINO features show positional bias. ALiBi reduces it.
  - This is the closest prior to our R^2 measurement. Read it before claiming novelty for the measurement.
- **Shi, Birsak, Cui, Li & Wonka, "Geometry without Position? When Positional Embeddings Help and Hurt Spatial Reasoning", arXiv 2601.22231 (Jan 2026)** [abstract]. [Source](https://arxiv.org/abs/2601.22231)
  - Across 14 foundation ViTs, PEs act as a causal geometric prior on representation structure. Not robotics, not grounding.
- **Search snippet: a paper on positional awareness in ViT foundation models** reports that models with explicit APE or RPE (DINO, DeiT, SAM) localize almost perfectly, and that all PE schemes allow linear recovery of position. Exact ID not verified. [snippet via search](https://arxiv.org/html/2610.13992)
- [prior] **DINOv2 registers (Darcet et al., arXiv 2309.16588)**: high-norm artifact tokens in DINOv2 and others hold global, not local, information. Registers remove them.
  - It does not show or remove absolute-position encoding. It is only tangential, as a known "ViT tokens carry non-local information" result.
- **LIBERO-Plus Finding 5 (positional bias)** [full text] and **LIBERO-PRO / AFI "memory trap"** [snippet]. Links are in (a).
  - These show the behavioral layout shortcut. None attributes it to linearly decodable absolute position in the vision encoder's features, and none tests that link.
- **G3VLA (2606.24472)** [snippet] adds projective PE to VLA tokens. This is the opposite direction (more geometric position), with a small effect (−1.1 when removed).

### Inferences
- **"ViT patch features linearly encode absolute position"** is known: probing papers, 2603.16840, 2610.13992, and the foundation-model positional-awareness work.
- **"VLAs bind or grasp toward remembered locations"** is known behaviorally: LIBERO-Plus, LIBERO-PRO, AFI.
- **The causal link between the two, measured in SigLIP/DINOv2 features of VLA encoders, appears unpublished.** The link is that absolute position decodable from the encoder features lets a binder or the policy bind a role ("left") to a remembered location.
  - Our R^2 numbers plus a binder intervention would be a new diagnostic.
  - To be credible it needs a causal ablation (remove or permute the position subspace and show the layout gap shrinks), because 2610.13992 stresses that decodability does not establish use.
- **C2 novelty is moderate.** The recipe "content-only keys + relative geometry" is Act3D/3DDA (3D) plus Invariant Slot Attention plus role-filler attention (Russin).
  - Novel parts: applying it to a pretrained 2D VLM encoder's tokens, removing the absolute-position subspace from keys post hoc, and targeting VLA layout shortcuts.
  - Frame it as transferring a known inductive bias with a new measured motivation, not as a new attention mechanism.
- **Strongest reviewer objections to C2:**
  1. Position is entangled nonlinearly in the features (R^2 0.72–0.89 linear only). Projecting out the linear PE or position subspace leaves residual position, as with NoPE, where position re-emerges. Show nonlinear-probe R^2 after removal.
  2. Act3D and 3D Diffuser Actor already do relative-only geometric attention and beat 2D policies, so reviewers will ask why not just use 3D relative attention.
  3. Some LIBERO tasks legitimately need absolute position (e.g. "the left plate"). Removing absolute position from keys must not remove task-relevant egocentric relations, so the relative-geometry channel must carry them, and this needs a demonstration.

### Gaps
- Not fetched: Russin 1904.09708, Invariant Slot Attention full text, DINOv2 registers, the "Compositional generalization through abstract representations" role/filler paper, POCR, and NoPE beyond the snippet. These are marked [prior] or [snippet].
- I found no specific study of position bias in CLIP/SigLIP grounding (e.g. referring expressions) within budget. This is a remaining search target.
- 2603.16840 and the positional-awareness foundation-model paper were seen only as snippets. Their models and R^2 values are unverified, and they must be read before claiming our probe is new.

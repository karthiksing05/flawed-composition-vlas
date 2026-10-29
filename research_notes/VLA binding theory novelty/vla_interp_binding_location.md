# Prior work on WHERE and WHEN instruction-to-referent binding lives in VLAs (novelty check for C1 and C3)

Scope: published and preprint work through Oct 2026. Every entry below was checked against its arXiv abstract or HTML full text on 2026-10-26, using WebFetch, which returns a model-written summary of the page. Numbers are as extracted from those pages. Anything marked UNVERIFIED was not confirmed in the primary text.

## Q1. Which VLA papers do activation patching, K/V splicing, attention knockout or causal tracing, and what do they find about where language goes?

### Takeaway
The closest prior result to C1 is VLA-Trace (arXiv 2605.30117), which uses attention knockout on pi0.5. During action generation, blocking the action expert's access to text barely hurts pi0.5 on LIBERO-Goal, Spatial and Object. Blocking the visual-language attention during prefill collapses LIBERO-Goal. Read together, these results imply that instruction information reaches the action pathway through the visual tokens. VLA-Trace never states that conclusion. It does not patch by layer band, splice image K/V across instructions, test object-region versus whole-image patches, or measure when information leaves the word positions. No paper found does whole-image vs region K/V splicing, and none reports "binding leaves word positions by layer 2". C1's specific localisation appears novel, but VLA-Trace must be cited and positioned as partial prior evidence.

### Cited Findings
- **VLA-Trace** (Shi, Ren, Zhang, et al.; arXiv 2605.30117; v1 28 May 2026, v3 18 Oct 2026). Models are pi0.5 and OpenVLA on LIBERO. Methods are CKA tracing, attention-knockout interventions and rollout probes. The abstract's conclusion is that VLAs are "strong at visually grounded trajectory generation but limited in fine-grained semantic following." [abs](https://arxiv.org/abs/2605.30117)
  - All-layer knockout, success % in the order LIBERO-10 / Goal / Spatial / Object. The column order was inferred from the text, which is a caveat. [html](https://arxiv.org/html/2605.30117)
    - pi0.5 baseline: 93.5 / 96.0 / 98.5 / 99.5.
    - "Prefill: V-L attention blocked": 0 / 11.5 / 77.0 / 71.5.
    - "Gen: no text": 39.0 / 96.5 / 99.0 / 98.0.
    - "Gen: no image": 0 / 4.0 / 0 / 0.
    - OpenVLA goes to 0 on every suite under "Gen: no text".
  - Layer-wise on LIBERO-10, pi0.5 has a "narrow visual bottleneck". Removing the image during generation drops success to 4.00% at the most vulnerable layer. "Gen: no text" stays close to baseline across layers. [html](https://arxiv.org/html/2605.30117)
  - Internal inconsistencies:
    - The text says dropping instruction tokens costs pi0.5 "only 1.3 points" (its Tab. 6), which conflicts with the 93.5 -> 39.0 drop on LIBERO-10.
    - The pi0.5 LIBERO-10 baseline is 93.5 in Tab. 2 but 75.00 in Tab. 4.
    - The direction of "V-L attention blocked" is not explicit in the extract, i.e. whether vision tokens are blocked from attending to language or the reverse. UNVERIFIED. [html](https://arxiv.org/html/2605.30117)
  - Checkpoint IDs were not found in the portion read (first 100k of 165k characters).
- **Not All Features Are Created Equal** (Grant, Zhao, Wang; arXiv 2603.19233; v1 19 Mar 2026, v4 20 Oct 2026). [abs](https://arxiv.org/abs/2603.19233)
  - Models: six VLAs, namely pi0.5 (3B), OpenVLA-OFT (7B), X-VLA (1B), SmolVLA (450M), GR00T N1.5 (3B) and ACT (80M). Benchmarks are LIBERO, MetaWorld, SimplerEnv and ALOHA, plus a UR10e arm. The paper trains 520+ SAEs and reports 420k+ rollouts. No exact HF checkpoint IDs are given. [html](https://arxiv.org/html/2603.19233)
  - Method: activation injection overwrites the residual stream ONLY at the 50 action-token positions. Image and prompt tokens are untouched. The production condition replaces the expert-pathway residual at layers 16 and 17 on every step. [html](https://arxiv.org/html/2603.19233)
  - Headline: injecting task A makes pi0.5 execute A's motor trajectory in 99.6% of episodes (n=1,968). [abs](https://arxiv.org/abs/2603.19233)
  - The injected program is "tied to absolute workspace coordinates" and ignores the visible scene. On the UR10e, the arm grasped at the recorded coordinates in 50/50 trials after the object moved 10-15 cm. [abs](https://arxiv.org/abs/2603.19233)
  - Instruction weight: "every architecture encodes the instruction; four of five weight it below vision when the scene alone identifies the goal". Goal selection and motor generation "separate into VLM and action-expert pathways". [abs](https://arxiv.org/abs/2603.19233)
  - Null-prompt result on pi0.5: injecting baseline *visual-pathway* activations recovers 80% goal, 70% object and 78% spatial success, against 6-62% without injection (Tab. 2). "Visual pathway" appears to mean the VLM/prefix pathway, not image tokens specifically, so this is ambiguous. [html](https://arxiv.org/html/2603.19233)
  - An edit to the VLM pathway *at instruction positions* restores goal selection with no instruction on 4/5 architectures. [html](https://arxiv.org/html/2603.19233)
  - SmolVLA: expert-pathway injection overrides 78.6% vs 25.6% for the VLM pathway. [html](https://arxiv.org/html/2603.19233)
  - App. F.4 ("where spatial binding forms during fine-tuning") is a per-layer cosine comparison of base OpenVLA vs OFT. It finds L1-L15 cosine > 0.999 and L31 0.917-0.938, i.e. fine-tuning reshapes the final third of the backbone. It concerns OpenVLA representation drift, not token-type localisation. [html](https://arxiv.org/html/2603.19233)
  - It does NOT patch image-token positions and does NOT localise binding in image tokens. [html](https://arxiv.org/html/2603.19233)
- **Do VLA Models Understand Instructions?** (Wulff, Cangelosi; arXiv 2610.10178; 7 Oct 2026). [abs](https://arxiv.org/abs/2610.10178)
  - Checkpoints: pi0.5 `pi05_libero_finetuned_v044` (lerobot) and GR00T N1.7 `gr00t17-lerobot-libero_10-640` (nvidia), both LIBERO-10 fine-tuned. [html](https://arxiv.org/html/2610.10178)
  - Method: activation and attribution patching on the action-expert residual stream after self-attention o_proj, against five instruction corruptions (synonym, semantic scaling, directional, random object, empty). [abs](https://arxiv.org/abs/2610.10178)
  - Findings:
    - Both models are insensitive to rephrasing and to references to nonexistent objects, and sensitive to empty and directional instructions.
    - In pi0.5, sensitivity is spread over the earliest layers plus selected later layers. Under directional perturbation, layers 2, 3, 5 and 14 show activation-patching effects of about -0.05 to -0.35.
    - Attribution-patching vs activation-patching correlation is r=0.946 for GR00T N1.7 but r=0.089 for pi0.5.
    - No token-type split, no image-token analysis, no within-episode timing; steps are sampled from 10,000 offline-demo steps.
    [html](https://arxiv.org/html/2610.10178)
- **Mechanistic Interpretability for Steering VLA Models** (Haon, Stocking, Chuang, Tomlin; arXiv 2509.00328; CoRL 2025; project page vla-mech-interp.github.io).
  - Models: pi0, pi0-FAST, pi0-FAST-DROID and OpenVLA; LIBERO-Long and a UR5.
  - Method: projects FFN value vectors onto the vocabulary and steers semantic neuron clusters ("up", "slow"). Mostly qualitative.
  - No token-type localisation and no timing. [project page](https://vla-mech-interp.github.io/)
- **SAEs Reveal Interpretable and Steerable Features in VLA Models** (Swann, McGranahan, Buurmeijer, Kennedy, Schwager; arXiv 2603.19183; v2 6 Jun 2026).
  - SAEs on hidden-layer activations, motion-primitive and semantic features, and steering on LIBERO and DROID hardware.
  - The abstract names no model and gives no numbers. No image-token localisation or timing appears in the abstract. [abs](https://arxiv.org/abs/2603.19183)

### Inferences
- VLA-Trace's pattern for pi0.5 matches C1's mechanism: generation survives without text but not without image on Goal, Spatial and Object, while blocking prefill V-L attention kills Goal. That pattern is what we would expect if instruction content is written into image tokens during prefill and read from there. C1 goes beyond it in four ways:
  - It splices whole-image K/V across instructions and moves binding (+0.44 of +0.50).
  - It shows region-only splices are inert (-0.005).
  - It localises the effect to deep layers 10-17.
  - It shows word positions lose the binding by layer 2.
- Frame C1 as the positive, localised, causal version of VLA-Trace's knockout hint.
- 2603.19233's "absolute workspace coordinates" and "ignores the visible scene" finding concerns the action-expert motor program, not the instruction-to-referent choice. It is compatible with C1: a scene-wide, non-object-local code. Cite it as complementary.
- The LIBERO-10 "no text" drop in VLA-Trace (93.5 -> 39.0) suggests that long multi-step tasks still read text during generation. C1 should state that it was measured on our LIBERO-90 compositional split and does not claim generality to LIBERO-10-style tasks.

### Gaps
- VLA-Trace layer-window and per-suite supplements and its checkpoint IDs were not read (second half of the HTML).
- The direction of its "V-L" prefill block is UNVERIFIED.
- 2603.19183's model is unnamed in the abstract; the full text was not read.
- No VLA paper using K/V splicing specifically was found. Searches for RDT and Octo interpretability returned nothing relevant (absence noted, not proven).

## Q2. Is there VLM work showing binding or relation info is written into image tokens, or moves from text to visual tokens?

### Takeaway
The VLM literature mostly documents the OPPOSITE direction: image information flows into the question/text tokens, mid-layers carry it, and later layers move it to the last position. One line, Binding IDs in VLMs, puts a shared binding identifier on an object's own image tokens, which is object-local. C1's "scene-wide code in deep image tokens, object region inert" therefore contrasts with both lines. That is a selling point, but it needs careful positioning because pi0.5's prefix is bidirectional (image tokens can attend to text), unlike causal LLaVA-style VLMs, where image tokens precede text and cannot read it.

### Cited Findings
- **Cross-modal Information Flow in MLLMs** (Zhang, Yadav, Han, Shutova; arXiv 2411.18620; CVPR 2025; LLaVA series). It finds two integration stages: lower layers move whole-image features into question tokens, middle layers move question-relevant object info into question positions, and higher layers move the result to the last position. [abs](https://arxiv.org/abs/2411.18620)
- **What's in the Image?** (Kaduri et al.; arXiv 2411.17491; CVPR 2025; InternVL2 and LLaVA-1.5). Attention knockout shows query text tokens act as compressed global image descriptors, and cross-modal flow is dominated by middle layers. [CVF pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Kaduri_Whats_in_the_Image_A_Deep-Dive_into_the_Vision_of_CVPR_2025_paper.pdf)
  - A secondary summary gives a ~25% layer band, and LLaVA-1.5 layers 4-20. These band numbers are UNVERIFIED in the primary text.
- **Towards Interpreting Visual Information Processing in VLMs** (arXiv 2410.07149; ICLR 2025). It uses knockout to test whether the final token reads object tokens directly or through aggregation. Details beyond the search snippet are UNVERIFIED. [pdf](https://arxiv.org/pdf/2410.07149)
- **Pathways of Visual Information Flow in VLMs** (arXiv 2607.03358; Qwen3-VL-4B). Blocking query tokens from attending to the image has the largest impact, while blocking direct image->last-token attention is negligible. This was taken from a search snippet and not fetched. [html](https://arxiv.org/html/2607.03358)
- **Investigating Mechanisms for In-Context Vision Language Binding** (Saravanan, Tapaswi, Gandhi; arXiv 2505.22200; CVPRW MIV 2025 oral). On a synthetic Shapes task, VLMs assign "a distinct Binding ID" to an object's image tokens and its textual references. The abstract names no models and gives no numbers. [abs](https://arxiv.org/abs/2505.22200)
- **Uncovering Grounding IDs** (Izadi et al.; arXiv 2509.24072; ICML 2026 per v5 comment, 14 Oct 2026). External partition cues (grids, symbols) induce latent "Grounding IDs" that align visual and textual components and strengthen attention between them. Layers and patching numbers were not in the abstract. [abs](https://arxiv.org/abs/2509.24072)

### Inferences
- The VLM consensus is text <- image, mid-layers, object-local. That makes C1 (text -> image, deep layers 10-17, whole-scene, and the region splice inert) a notable contrast.
- The likely explanation is architectural: pi0.5's bidirectional prefix attention lets image tokens read the prompt. This mechanism is our inference and is not shown in any cited paper.
- Binding-ID work predicts that patching the object's own region should move binding. C1's -0.005 for the region splice directly contradicts the "object-local ID" picture for this policy. Cite 2505.22200 as the hypothesis C1 tests and rejects.

### Gaps
- None of the VLM papers found studies a bidirectional-prefix VLM (PaliGemma-style prefix-LM) with text -> image-token flow. A targeted search for PaliGemma prefix-LM information flow was not run.
- No paper on where spatial relations ("left of") live in image vs text tokens was verified.

## Q3. Has anyone measured WHEN a robot policy commits to a goal or referent within an episode (mode commitment, sample-diversity collapse)?

### Takeaway
No paper found measures commitment to a referent at the grasp, or coverage@k of the correct behaviour before versus during the carry. The closest is DiVeR (arXiv 2610.04933, pi0 and pi0.5). It finds that sample divergence, which it calls criticality, is sparse and temporally localised at pre-grasp and pre-placement states, and that candidates look alike during post-grasp transport. That is qualitatively consistent with C3, but it is not framed as referent binding and has no per-phase numbers. C3 appears novel as stated; cite DiVeR as converging evidence.

### Cited Findings
- **DiVeR: Decision-Critical Verifier Learning for VLA Test-Time Scaling** (arXiv 2610.04933v1; 4 Oct 2026; author list not in the extract, so UNVERIFIED). [html](https://arxiv.org/html/2610.04933v1)
  - Models: pi0 and pi0.5 (main), OpenVLA (App. C.2). No checkpoint IDs are given.
  - Criticality u_t is the mean per-coordinate variance of K sampled action-expert representations.
  - Examples of critical states: "pre-grasp or pre-placement alignment". Examples of routine states: free-space motion and post-grasp transport.
  - Fig. 1(b): critical states are "sparse and temporally localized" over 500 LIBERO-Long trajectories. No numeric timing is given.
  - u_t correlates with oracle criticality at Spearman rho=0.493 on 500 RoboCasa states (pi0.5).
  - It reports success vs N=4/8/16/32, not per-timestep pass@k.
- **Flow-Support-Weighted Proposals for Test-Time Verified VLA Policies** (arXiv 2610.14588). It argues that fine-tuning collapses flow-matching VLAs to a dominant mode, so sampled candidates are near-duplicates. Taken from a search snippet; not fetched in full. [html](https://arxiv.org/html/2610.14588)
- **What Frozen VLAs Already Know About Success** (Zhang, Nie, Lao, Cheng, Liu, Jiang, Huang; arXiv 2605.28527; 27 May 2026). [abs](https://arxiv.org/abs/2605.28527)
  - Models: OpenVLA, pi0.5, DINOv2 and CLIP frozen features on LIBERO-Goal. No checkpoint IDs are on the abstract page.
  - Linear probes predict Monte-Carlo success, with ~92% pairwise ordering accuracy for pi0.5 and shuffled labels at chance.
  - As a test-time selector over pi0.5 action prefixes, it raises push-plate success from 26.7% to 44.3%.
  - It controls for timestep but does not report how predictability evolves within an episode, and does not localise by layer or token in the abstract.
- **2603.19233 temporal analyses** [html](https://arxiv.org/html/2603.19233):
  - GR00T N1.5 temporal ablation pooled over suites: early-window ablation -29.7 pp, mid -13.8 pp, late -13.6 pp. The paper concludes GR00T is "most sensitive during trajectory initiation".
  - Phase-aligned DTW: raw-index injection overrides 100%, DTW 94.4%, gripper-phase alignment 83.3%.
  - Dose-response: at alpha=0.25 the commanded task degrades first (19/35 destination success); at alpha=0.5, 34/35 episodes are source-classified.
  - None of this measures referent commitment or sample coverage.
- **Diffusion Policy** (Chi et al.; arXiv 2303.04137). Mode selection comes from initial noise at each sampling call, and the policy commits to an action chunk for a fixed horizon before replanning. There is no within-episode commitment measurement. [pdf](https://arxiv.org/pdf/2303.04137)

### Inferences
- DiVeR's "divergent at pre-grasp, routine after grasp" is the variance-side twin of C3. C3's coverage@16 (0.53 before the grasp, 0.20 during the carry) adds that the CORRECT referent is among samples pre-grasp and disappears after it, i.e. commitment to a referent and not just reduced action variance.
- Position C3 against DiVeR and 2610.14588, which concern diversity and mode collapse, and against 2605.28527, which shows the good continuation is often already in the sample set.

### Gaps
- DiVeR's tail (last ~12k characters), its authors and its checkpoints were not verified.
- 2610.14588 was not fetched.
- No search was run for "goal inference timing" in imitation policies outside VLAs, e.g. intent-prediction literature in HRI.

## Re-check summary of the five named items
- **2603.19233.** Injects only at action-token positions in expert layers 16-17. It does not localise binding in image tokens. Its timing work (GR00T early-window sensitivity, DTW) does not measure referent commitment. Its "absolute coordinates, ignores scene" finding concerns the motor program. [html](https://arxiv.org/html/2603.19233)
- **2610.10178.** Patches only the action-expert residual, with no token-type split and no timing. In pi0.5, sensitivity is in early layers (2, 3, 5) plus layer 14. [html](https://arxiv.org/html/2610.10178)
- **2605.28527.** Success probing and test-time selection only. It has no image-token localisation and no commitment-timing curve. [abs](https://arxiv.org/abs/2605.28527)
- **vla-mech-interp.github.io (2509.00328).** FFN neuron steering in pi0 and OpenVLA. No binding or timing. [page](https://vla-mech-interp.github.io/)
- **2603.19183.** SAE steering features. The abstract has no localisation or timing. [abs](https://arxiv.org/abs/2603.19183)
- **Not named, but most relevant to C1: VLA-Trace (2605.30117).** It is the only paper whose pi0.5 knockouts imply that instruction info reaches the action pathway through visual tokens. It must be cited. [html](https://arxiv.org/html/2605.30117)

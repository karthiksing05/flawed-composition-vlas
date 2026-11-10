# VLA compositional failure: benchmarks, diagnoses, fixes, theory (2026)

Method note: ~19 web calls. Numbers below come from arXiv abstract pages or project pages fetched by a small summarizer model, plus search snippets. Items marked [UNVERIFIED] come from search snippets only. Full-text tables were not read except where stated. "CoRL 2026" is stated only where a fetched page said so.

## (a) Benchmarks and evaluations under changed instructions, positions, unseen combinations, distractors

### Takeaway
Every 2026 benchmark shows large drops on unseen composites or perturbed layouts. MESA (CoRL 2026) is the cleanest recent table: pi0.5 is best, GR00T-N1.6 is weaker, and composite suites lose 16-31 points relative to ID. CRAFT (arXiv 2610.00524) measures exactly the user's pairing setting and reports near-zero held-out success for plain fine-tuning (0.3-8.4% except pi0.5 LoRA/Frozen). LIBERO-Plus/PRO report that models ignore language.

### Cited Findings
- MESA / BiMESA (Wilcox, Chang, Nguyen, Chakraborty, Collins, Saxena, Joffe, Karamcheti, Garg; Georgia Tech; CoRL 2026). MESA: Franka, 70 tasks, 7,000 demos. BiMESA: bimanual YAM, 57 tasks, 6,833 demos. Five suites: ID, Spatial, Instance, Category, Composite (unseen composites of ID subtasks). Distractor objects and predicates probe language following. MESA-Gen builds on MimicGen with subtask stitching. — [MESA page](https://www.pair.toronto.edu/MESA/)
- MESA success (%), as ID / Spatial / Instance / Composite / Category / Avg. pi0.5: 62.8 / 68.6 / 55.4 / 46.7 / 26.4 / 52.0. pi0: 43.1 / 42.2 / 34.0 / 27.5 / 11.6 / 31.7. FAST: 46.5 / 53.6 / 39.1 / 28.9 / 8.4 / 35.3. GR00T-N1.6: 25.0 / 23.2 / 22.7 / 15.0 / 14.1 / 20.0. PG-FM: 30.7 / 35.0 / 22.7 / 17.7 / 7.7 / 22.8. DP: 1.6 / 1.0 / 1.9 / 0.3 / 0.7 / 1.1. — [MESA page](https://www.pair.toronto.edu/MESA/)
- BiMESA success (%), same order. pi0.5: 68.7 / 70.5 / 48.1 / 37.5 / 18.6 / 48.7. GR00T-N1.6: 47.0 / 46.2 / 36.7 / 22.1 / 26.1 / 35.6. pi0: 45.9 / 49.2 / 23.9 / 18.4 / 10.6 / 29.6. FAST: 53.7 / 58.1 / 36.7 / 28.8 / 19.8 / 39.4. PG-FM: 43.8 / 50.8 / 28.6 / 17.9 / 10.9 / 30.4. PG-Bin avg 18.3. DP avg 10.6. — [MESA page](https://www.pair.toronto.edu/MESA/)
- MESA language-following on composite tasks: pi0.5 reaches 81.5% (MESA-Composite) and 80.1% (BiMESA-Composite), so the authors' reading is that the gap is execution, not instruction understanding. This matches the user's "grasp right, place wrong" pattern. — [MESA page](https://www.pair.toronto.edu/MESA/)
- CRAFT held-out-pairing benchmark (see section c for method). Built on LIBERO. Pick-Place: 16 object-target combinations, 4 demonstrated, 12 held out. Pick-Place-Press: 8 combinations, 2 demonstrated, 6 held out. 50 demos per demonstrated combination. Undemonstrated success (%), as PP pi0.5 / pi0 / GR00T, then PPP pi0.5 / pi0 / GR00T. FT(Full): 8.4 / 2.8 / 1.1, 0.7 / 0.3 / 1.1. FT(LoRA): 46.3 / 1.0 / 2.6, 15.4 / 0.7 / 1.0. FT(Frozen): 21.1 / 3.0 / 2.2, 0.9 / 1.5 / 1.1. Demonstrated-combination success is about 94-99.7%. — [arXiv 2610.00524](https://arxiv.org/html/2610.00524v1)
- Robust Skills, Brittle Grounding (Emukpere, Deffayet, Renders; arXiv 2602.24143, 27 Feb 2026; preprint). Controlled multi-object picking with increasing placement variability and held-out object-location pairings, on SmolVLA and pi0.5. Primitive execution stays much more reliable than instruction-conditioned success. They propose "task ladders" and decomposed metrics. No numbers were visible on the abstract page. — [arXiv](https://arxiv.org/abs/2602.24143)
- LIBERO-Plus (Fei et al., Fudan; arXiv 2510.13626, late-2025 reference point; CVPR 2026). Seven perturbation dimensions: object layout, camera, robot init, language, light, background, noise. 10,030 tasks. Performance drops "from 95% to below 30% under modest perturbations" (camera, robot init). Models are largely insensitive to language and "tend to ignore language instructions completely." — [arXiv](https://arxiv.org/abs/2510.13626); [CVPR 2026 page](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)
- LIBERO-Plus per-model list and "fine-tuning on a generalized dataset lifts OpenVLA-OFT to 79.6%" — search snippet only, [UNVERIFIED] — [themoonlight review](https://www.themoonlight.io/review/libero-plus-in-depth-robustness-analysis-of-vision-language-action-models)
- LIBERO-PRO (Zhou et al.; arXiv 2510.03827, Oct 2025, v2 May 2026; preprint). Perturbs objects, initial states, instructions, environment. Models above 90% on standard LIBERO "collapse to 0.0%" under the generalized setting. Authors blame rote memorization of action sequences and layouts. Models keep grasping when the target is swapped for an irrelevant item, and outputs are unchanged under corrupted instructions or nonsense tokens. — [arXiv](https://arxiv.org/abs/2510.03827)
- LIBERO-CF (CoRL 2026 Spotlight): alternative feasible instructions under the same LIBERO layouts. See section b. — [arXiv 2602.17659](https://arxiv.org/abs/2602.17659)
- LangGap (Hou, Zhao; arXiv 2603.00592, 28 Feb 2026; preprint). 99 tasks, same layout supports different instructions. pi0.5 only: single-task training lifts success 0% to 90%, multi-task training 0% to 28%. Claim: SOTA VLAs (>95% on standard benchmarks) largely ignore language. — [arXiv](https://arxiv.org/abs/2603.00592)
- ICBench (in IGAR paper): contradictory instructions on an unchanged LIBERO scene, 30 tasks, Pi0, Pi0.5, OpenVLA-OFT. Models often complete tasks even when the instruction is impossible for the scene. — [arXiv 2603.06001](https://arxiv.org/abs/2603.06001)
- RoboCombine / HSE (Zhang, Yang, Guo, Yin, Zhang; NeurIPS 2026 listing): skill recombination into long-horizon sequences. Training on limited combination sequences gives 40.0% ordered success on covered combinations but 7.7% on recombination. "Atomic competence alone does not ensure ordered execution." — [NeurIPS page](https://neurips.cc/virtual/2026/poster/139255)
- Also surfaced, not read [UNVERIFIED]: ACG-Bench (arm-wise compositional generalization, dual-arm, arXiv 2610.06184 per [opentrain listing](https://www.opentrain.ai/papers/arm-wise-compositional-generalization-in-dual-arm-vision-language-action-models--arxiv-2610.06184/)); VLA-Arena (near-zero success on long-horizon and compositional tasks across architectures, per [search snippet](https://www.emergentmind.com/papers/2512.22539)).

### Inferences
- MESA suggests the composite deficit is mostly in execution, while CRAFT and the user's data show the failure sits in binding the instruction to the held-out pairing. Both fit "grasp OK, place wrong."
- pi0.5 beats GR00T-N1.6 on composites in MESA, but GR00T wins Category on BiMESA. Do not assume a fixed ranking.

### Gaps
- I did not find per-model LIBERO-Plus and LIBERO-PRO tables with verified numbers.
- I did not find a benchmark report for position-shift numbers for OpenVLA-OFT in 2026 beyond the above snippets.

## (b) Diagnoses of the cause

### Takeaway
The recurring diagnosis is a vision shortcut: when scene, object or location predicts the instruction, the model drops language. Mechanistic work finds that instruction is encoded but weighted below vision, that motor programs are scene-bound, and that most SAE features are episode memorization.

### Cited Findings
- CAG / LIBERO-CF (Fang, Feng, Jing, Liu, Yang, Wei, Szafir, Ding; arXiv 2602.17659, v3 4 Nov 2026; CoRL 2026 Spotlight). VLAs act on "vision shortcuts induced by dataset biases": repeating well-learned behaviors and picking objects frequently seen in training. — [arXiv](https://arxiv.org/abs/2602.17659)
- LangForce / BayesianVLA (Lian et al.; arXiv 2601.15197; ICML 2026 per comments field). "Information Collapse": instructions are predictable from vision alone, so conditional mutual information between instruction and action vanishes. Snippet: on RoboCasa a vision-only model scores 44.6% against 47.8% language-conditioned [UNVERIFIED, snippet]. — [arXiv](https://arxiv.org/abs/2601.15197); [search result](https://arxiv.org/pdf/2601.15197v4)
- CRAFT appendix A: failed held-out rollouts mostly completed demonstrated tasks sharing the object or the target with the instruction, and an empty instruction still produced demonstrated behaviors. Authors say this is behavioral evidence only. — [arXiv 2610.00524](https://arxiv.org/html/2610.00524v1)
- Not All Features Are Created Equal (Grant, Zhao, Wang; arXiv 2603.19233, v6 2 Nov 2026). Activation injection into the action expert, pi0.5 plus four other VLAs, over 420,000 rollouts, 520+ SAEs. pi0.5 executed task A's trajectory in 99.6% of injected episodes (n=1,968). The program is tied to absolute workspace coordinates and ignores the visible scene. Every architecture encodes the instruction, but four of five weight it below vision when the scene identifies the goal. Goal selection and motor generation separate into VLM and action-expert pathways. — [arXiv](https://arxiv.org/abs/2603.19233)
- Dr. VLA (Swann, McGranahan, Buurmeijer, Kennedy, Schwager; arXiv 2603.19183). SAEs on VLA hidden layers: the large majority of features correspond to memorized training-demo sequences, a smaller set are general steerable primitives. SFT on small robotics datasets amplifies memorization, while diverse data (DROID) or knowledge insulation yields more general features. Steering tested on LIBERO and DROID hardware. No exact fractions on the abstract page. — [arXiv](https://arxiv.org/abs/2603.19183); [snippet](https://www.emergentmind.com/papers/2603.19183)
- Primitive Subspaces Mediate Few-Shot Transfer (Singh et al.; arXiv 2605.30695, 29 May 2026). OpenVLA and pi0.5, REASSEMBLE, 6 held-out object-task combinations. Primitive-trained models reach 78% of the upper bound at m=3 demos, flat-trained need m=10. Ablating the primitive-decodable subspace cuts transfer by 32 points, a random subspace has no effect. — [arXiv](https://arxiv.org/abs/2605.30695)
- IGAR (Zhang, Zhu, Zhou, Chen; arXiv 2603.06001): "linguistic blindness," visual priors override instruction semantics in action generation. — [arXiv](https://arxiv.org/abs/2603.06001)
- LIBERO-PRO: memorization of action sequences and layouts (above). — [arXiv](https://arxiv.org/abs/2510.03827)
- Robust Skills, Brittle Grounding: skill acquisition decoupled from instruction following; policies may rely on object-location correlations (stated as likely, not proven). — [arXiv](https://arxiv.org/abs/2602.24143)
- ACT-VLA (arXiv 2607.00351): attributes failure to overfitting to specific behavioral patterns. — [arXiv](https://arxiv.org/abs/2607.00351)
- Data-level snippet: "fewer than 2% of instructions are linguistically unique" in audited VLA datasets [UNVERIFIED, secondhand] — [search result](https://humansignal.com/blog/why-your-vla-model-ignores-language-and-the-cheapest-fix-in-robotics)

### Inferences
- The "one-target object fails more than two-target object" result is predicted by the vision-shortcut story: an object with one partner lets the object identity predict the target, so language carries no information about the target. LangForce's Information Collapse formalizes this. I found no paper that tests it directly.
- The user's "hide the held object's identity from the place skill" fix is the same move as CAG's language-unconditioned branch and CRAFT's counterfactual instruction, applied to the object channel. Nothing I found does exactly this; the closest is CRAFT's skill-token design.

### Gaps
- No paper found that probes object/target/skill binding specifically (e.g. a binding-ID analysis) in a VLA. Closest: 2603.19233 (pathway separation) and 2605.30695 (primitive subspace).
- Per-paper SAE memorization fractions not obtained.

## (c) Fixes

### Takeaway
Training-free fixes give single-digit to low-double-digit gains (CAG +9.7 grounding, +3.6 success). Fine-tuning fixes that add counterfactual instruction supervision (CRAFT) move held-out combinations from near zero to 30-84%.

### Cited Findings
- CRAFT (Yang, Na, Cho, Yoon; KAIST; arXiv 2610.00524, 30 Sep 2026; venue not stated). Adds learnable skill and state query tokens to the VLM. Contrastive skill loss reuses skill representation across executions. Counterfactual loss pairs an observation with a changed instruction, using a demonstrated execution of the required skill as the action target. Held-out success: PP pi0.5 84.2, pi0 64.3, GR00T 72.1. PPP pi0.5 62.6, pi0 40.7, GR00T 31.9. Real Piper arm pi0.5: 43/60 against 9/60 for full fine-tuning. Other baselines: CAG-VA 66.5 (PP pi0.5), Skill Composition 25.7. — [arXiv](https://arxiv.org/html/2610.00524v1)
- CAG (training-free TF-CAG on pi0.5): +9.7% language grounding, +3.6% task success on LIBERO-CF. VA-CAG (adds a vision-action module, training status unconfirmed): +15.5 grounding, +8.5 success. Real world VA-CAG: +9.4 grounding, +17.2 success. Dual branch: amplify the difference between language-conditioned and language-unconditioned actions. — [arXiv](https://arxiv.org/abs/2602.17659)
- LangForce (training objective, no new data): maximizes conditional PMI between action and instruction; +11.3% on SimplerEnv OOD. — [arXiv](https://arxiv.org/abs/2601.15197)
- IGAR: training-free attention recalibration toward instruction tokens; evaluated on Pi0, Pi0.5, OpenVLA-OFT. No numbers retrieved. — [arXiv](https://arxiv.org/abs/2603.06001)
- ACT-VLA (Peng, Lu, Peng; arXiv 2607.00351, 1 Jul 2026): offline augmentation using the model's latent task representations (text latent interpolation per search snippet) to synthesize demos for unseen compositions. Abstract gives no numbers. — [arXiv](https://arxiv.org/abs/2607.00351); [snippet](https://www.emergentmind.com/papers/2607.00351)
- AC-VLA (Peng, Peng, Lu, Lian, Yu, Wang; arXiv 2607.15714, 17 Jul 2026): LLM instruction decomposition plus mixed full/decomposed training on pi0.5, plus masking the wrist view while the gripper is closed. About +28% absolute on LIBERO compositional OOD, ID near-perfect. — [arXiv](https://arxiv.org/abs/2607.15714)
- PRTI (RoboCombine): ordered success on recombination 2.11% to 26.32% under oracle timed HSE; transition conversion 2.63% to 60.53%. Failure attributed to a start-state gap at subtask transitions. — [NeurIPS page](https://neurips.cc/virtual/2026/poster/139255)
- VLS (Vision-Language Steering, arXiv 2602.03973): steers frozen generative policies at inference with VLM-produced stages and rewards; about 40% success on novel objects where frozen baseline fails [snippet only, UNVERIFIED]. TAG (target-agnostic guidance), OmniGuide, VLA Grounder (2607.04517), LFP (2606.12299), ATA (2603.01490, up to +10%) also surfaced [snippet only]. — [search result](https://arxiv.org/pdf/2602.03973); [search result](https://arxiv.org/pdf/2606.12299); [search result](https://arxiv.org/pdf/2603.01490)
- PrimitiveVLA (arXiv 2605.28634): 6x improvement on unseen tasks for OpenVLA [UNVERIFIED, snippet]. — [search result](https://arxiv.org/pdf/2605.28634v1)
- DTG-VLA (NeurIPS 2026 listing): dynamic target guidance for task-compositional generalization [UNVERIFIED, no details]. — [NeurIPS](https://neurips.cc/virtual/2026/164339)
- ReGuide: no paper by this name found in three searches. Closest training-free guidance: CAG, VLS, IGAR, TAG. Flag as unlocated.

### Inferences
- In CRAFT, FT(Frozen) and the CAG-VA baseline help pi0.5 but barely help GR00T and pi0, so pi0.5's VLM-derived language path is what makes training-free fixes work. The user's repair may be architecture-dependent.
- CRAFT's counterfactual target (reuse a demonstrated execution of the skill, swap the instruction) is closest to the user's "mask identity from the place skill."

### Gaps
- No exact numbers for ACT-VLA, IGAR, or object-centric-input fixes.

## Pairing diversity law and theory

### Takeaway
I found no robot paper that varies partners per object and reports a law or threshold. The nearest evidence is data-scaling and coverage theory. This is a genuine gap relevant to the user's one-vs-two-target result.

### Cited Findings
- Data Scaling Laws in Imitation Learning (Lin et al.; arXiv 2410.18647, ICLR 2025, late-2025 or earlier reference): generalization to new objects/environments follows a power law in the number of training objects, environments, or environment-object pairs. Diversity beats demos per object. 32 environments each with one object and 50 demos reached about 90% on new environments and objects. Demos per object saturate beyond a threshold. — [arXiv](https://arxiv.org/html/2410.18647v3)
- Gao et al., "Efficient Data Collection for Robotic Manipulation via Compositional Generalization" (arXiv 2403.05110, 2024 reference): policies show compositional generalization across factors such as object types; no partner threshold seen. — [arXiv](https://arxiv.org/html/2403.05110v2)
- Coverage Principle (arXiv 2505.20278, 2025 reference): coverage is necessary; a pattern-matching learner generalizes only to inputs reachable by substituting functionally equivalent fragments. — [arXiv](https://arxiv.org/html/2505.20278v1)
- NeurIPS 2026 poster: transformer compositional performance degrades monotonically as module coverage shift grows; composition equivalence matters. — [NeurIPS](https://neurips.cc/virtual/2026/poster/139422)
- arXiv 2606.18089 (June 2026): latent-variable model separating atomic skills from routing; SFT must cover all atomic modules via compositional traces; RL addresses compositions outside SFT support. — [HF](https://huggingface.co/papers/2606.18089.md)
- ICML 2026 paper (title not retrieved): finite-sample theory, latent composition operator unique up to change of basis, PAC sample-complexity separation. — [ICML](https://icml.cc/virtual/2026/69362) [UNVERIFIED, snippet]
- arXiv 2608.26465: category-theoretic account of compositional generalization via identification (Kan extension), on COGS. — [arXiv](https://arxiv.org/pdf/2608.26465) [UNVERIFIED, snippet]

### Inferences
- With one partner, an object's identity and its target are perfectly confounded, so the held-out pairing is outside the "functionally equivalent substitution" reach the Coverage Principle requires. With two partners, the model has at least seen the object separated from one target. This is my reading, not a result in these papers.

### Gaps
- No quantitative pairing-diversity law for VLAs found. The user's one-vs-two-partner finding appears novel relative to what I could locate.
- Theory papers are mostly language-model or generic; none robotics-specific retrieved.

# Prior work on instruction-to-scene binding failures in flow VLAs (novelty check for the failure atlas)

Scope note: about 15 tool calls. Verification depth differs by paper. ReGuide (2609.38616) and ECT (2609.39971) were read in full HTML, but only the first ~100k of ~170k characters, so their appendices were not read. The other papers were checked from the abstract only. Papers marked "[prior knowledge, not re-verified]" were not fetched in this session and need checking before anything about them is cited.

## Q1. Which 2024 to Oct-2026 papers analyse VLA failures by referring-expression complexity, spatial/relational language, instance disambiguation or part/sub-region grounding?

### Takeaway
Several 2026 papers show the broad claim: VLAs keep their motor skill but bind the instruction to the wrong target, through shortcuts or retrieval of a demonstrated task. They cover pi0.5, SmolVLA, GR00T N1.5/N1.7, X-VLA and pi0. None of the papers checked stratifies binding error by referring-expression depth (type < instance < relational sub-region such as a drawer level or shelf level). The depth-graded part of the atlas looks open. The general claim "grounding is brittle and skills are robust" is already taken.

### Cited Findings
- **Robust Skills, Brittle Grounding (Emukpere, Deffayet, Renders; arXiv 2602.24143).**
  - What it does: a controlled multi-object picking study that increases placement randomization and holds out object-location pairings.
  - Main result: for SmolVLA and pi0.5, "manipulation skill acquisition is decoupled from instruction following". The primitive stays reliable while instruction-conditioned success drops.
  - It recommends task ladders and decomposed metrics.
  - What it does NOT do (abstract only): no referring-depth taxonomy, no drawers or shelves, no MCD.
  - Overlap: HIGH on the claim that skills are robust and grounding is brittle. LOW on depth.
  - [arXiv 2602.24143](https://arxiv.org/abs/2602.24143)
- **ReGuide, "Correcting WHERE, Preserving HOW" (Zhang et al.; arXiv 2609.38616).**
  - Benchmark: a LIBERO-based compositional benchmark with four one-factor axes (Target Region, Target Object, Unseen Object, Task Scene) in 16 composition cells.
  - Bare-policy Comp. Avg.: OpenVLA-OFT 14.0, pi0 8.5, pi0.5 39.0, X-VLA 20.6, GR00T N1.7 13.0.
  - What it does NOT do: no split by category vs instance vs relational expression, and no drawer, shelf or part targets (in the portion read).
  - Overlap: HIGH on "VLAs fail at global grounding while keeping local skill" across 5 backbones. NONE on depth.
  - [arXiv 2609.38616](https://arxiv.org/html/2609.38616)
- **ECT, "When Instructions Retrieve Trajectories" (arXiv 2609.39971).**
  - Models: fine-tuned pi0.5 and GR00T-N1.7 on LIBERO-PRO and CALVIN.
  - Failures are grouped by perturbation cell (Semantic, Object, Swap, Task) and by LIBERO suite. There is no breakdown by referring-expression type or sub-region, and drawers and shelf levels are not mentioned in the portion read.
  - [arXiv 2609.39971](https://arxiv.org/html/2609.39971)
- **VLA-Trace (arXiv 2605.30117).**
  - Models: pi0.5 and OpenVLA.
  - Uses rollout-level probes for grounding, shortcut dependence and semantic following.
  - Finding: policies are good at "visually grounded trajectory generation" but limited in "fine-grained semantic following".
  - Abstract only: no referring-depth or sub-region stratification is stated.
  - Overlap: MEDIUM.
  - [arXiv 2605.30117](https://arxiv.org/abs/2605.30117)
- **arXiv 2603.19233.**
  - Analysis is mainly on pi0.5 (plus UR10e physical tests), across six VLAs including SmolVLA, on LIBERO, MetaWorld, SimplerEnv and ALOHA.
  - Finds that goal selection and motor generation separate into the VLM and action-expert pathways.
  - An injected program is "bound to absolute workspace coordinates": shifting the basket about 5 cm makes the policy drop the object.
  - Four of five architectures weight the instruction below vision when the scene alone identifies the goal.
  - What it does NOT do: no referring-depth axis and no MCD (abstract only).
  - Overlap: MEDIUM-HIGH on the binding-vs-motor separation, through mechanism.
  - [arXiv 2603.19233](https://arxiv.org/abs/2603.19233)
- **RoboSemanticBench (arXiv 2606.02277).**
  - The robot must grasp the block that shows the correct answer to a math or knowledge multiple-choice question (4 or 10 choices).
  - Selection is scored "after controlling for grasp success". Many policies grasp candidates fine but select at near-random or below-random rates.
  - What it does NOT do: semantic reasoning is not referring depth, and it has no spatial or part targets.
  - Overlap: MEDIUM. It separates selection from grasping, but on a different axis.
  - [arXiv 2606.02277](https://arxiv.org/abs/2606.02277)
- **DiVeR (arXiv 2610.04933).** A verifier that reweights training toward "decision-critical states", evaluated on LIBERO, RoboCasa and a Franka FR3. It is not a binding-failure analysis. Overlap: LOW. [arXiv 2610.04933](https://arxiv.org/abs/2610.04933)
- **EquiVLA (arXiv 2606.19784).** Built on GR00T N1.5 (78.1% LIBERO baseline cited), evaluated on LIBERO, CALVIN and Mobile ALOHA. The abstract reports only aggregate success, with no failure taxonomy. Overlap: LOW. [arXiv 2606.19784](https://arxiv.org/abs/2606.19784)
- **Pointing-VLA (arXiv 2608.23138).**
  - Adds typed spatial-grounding interfaces to a pi0.5 policy.
  - Real PiPER arm: 79/150 for the baseline vs 121/150 with guidance.
  - Search snippet only: it is a fix, and no depth stratification was seen.
  - [arXiv 2608.23138](https://arxiv.org/pdf/2608.23138)
- **LIBERO-Goal already contains sub-region instructions** such as "open the top drawer and put the bowl inside" and "open the middle drawer of the cabinet". Sub-region targets therefore exist in standard benchmarks but are not analysed as a depth level. [RD-VLA page listing LIBERO-Goal tasks](https://rd-vla.github.io/)
- **Non-VLA analogue (referring segmentation).** InstanceBench (arXiv 2610.09478) separates target-selection errors from mask errors. Replacing expressions with category names cuts mIoU by 36.95 to 44.91. This is a methodological precedent for separating selection from execution in vision models. [arXiv 2610.09478](https://arxiv.org/html/2610.09478)
- **[prior knowledge, not re-verified]**
  - LIBERO-PRO and LIBERO-Plus (both late 2025) report that LIBERO-trained VLAs keep succeeding under corrupted or perturbed instructions and object/position changes, which the authors read as trajectory memorization and language insensitivity. Neither stratifies by referring depth, as far as I recall.
  - He et al. arXiv 2505.05787 was NOT checked. Its content is unknown here.

### Inferences
- The claim "VLAs bind wrong while moving right" is now crowded: 2602.24143, 2609.38616, 2609.39971, 2603.19233, 2605.30117 and 2606.02277. The atlas cannot claim it as new.
- What is still uncovered is a graded referring-depth axis (type -> instance -> relational sub-region) with error rising monotonically, measured across all five flow VLAs. LIBERO-Goal's drawer instructions make the sub-region level testable without a new environment.

### Gaps
- The appendices of ReGuide and ECT (the last ~70k characters of each) were not read. A depth or sub-region breakdown could still be there.
- Full texts of 2602.24143, 2605.30117, 2603.19233 and 2606.02277 were not read.
- He et al. 2505.05787, LIBERO-PRO and LIBERO-Plus were not fetched.
- arXiv 2610.15425 ("Exploring the Limits of Combinatorial Generalization in Embodied Manipulation") showed up in search but was not checked. It is a likely close neighbour and must be read: [link](https://arxiv.org/html/2610.15425).

## Q2. Do the existing fixes (ReGuide, ECT, CRAFT, RoboPoint, RoboRefer, SoFar, MOKA, CAG-style guidance) report results by referring depth or by sub-region targets?

### Takeaway
No fix checked reports results stratified by referring depth, and none evaluates drawer, shelf-level or part targets as a separate condition. The claim "existing fixes also degrade with depth" is unclaimed. It would need to be shown experimentally; it cannot be cited.

### Cited Findings
- **ReGuide** is a training-free wrapper:
  - It takes object poses from a grounding module, rebinds the end-effector semantically and geometrically toward a demonstration-supported configuration of the referent, then hands control back to the frozen policy.
  - Comp. Avg. with ReGuide: OpenVLA-OFT 70.8, pi0.5 94.5, GR00T N1.7 69.1. Real robot: 15% to 90%.
  - It reports no depth or sub-region stratification (portion read).
  - Its approach depends on object poses. Whether part or sub-region referents (a drawer level) are supported is not shown.
  - [arXiv 2609.38616](https://arxiv.org/html/2609.38616)
- **ECT** results:
  - LIBERO-PRO position-swap mean rises from 36% to 59%.
  - Real UR5e unseen positions: 8% to 88%.
  - Reports CALVIN five-task completion.
  - No depth or sub-region stratification.
  - [arXiv 2609.39971](https://arxiv.org/abs/2609.39971)
- **CRAFT (arXiv 2610.00524)** is a training method, not a benchmark. It transfers supervision from demonstrated executions of a skill to counterfactual pairs, for undemonstrated skill combinations. It uses "three VLA models" and "two simulation benchmarks", not named in the abstract. Its axis is skill combination, not referring depth. [arXiv 2610.00524](https://arxiv.org/abs/2610.00524)
- **Pointing-VLA** improves pi0.5 from 79/150 to 121/150 with typed spatial guidance. No depth stratification was seen. [arXiv 2608.23138](https://arxiv.org/pdf/2608.23138)
- **[prior knowledge, not re-verified]**
  - RoboPoint (2406.10721) predicts spatial-affordance points, including free-space relational regions.
  - RoboRefer (2506.04308) introduces RefSpatial-Bench, which as I recall grades by the number of spatial reasoning steps. That is the closest existing "depth" axis, but it measures VLM pointing accuracy, not VLA binding.
  - SoFar (orientation-aware spatial reasoning) and MOKA (mark-based affordance prompting) are VLM front-ends.
  - None of these, as far as I recall, evaluates a frozen flow VLA stratified by depth.

### Inferences
- RoboRefer's step-count grading is the nearest precedent for a depth axis. The atlas should cite it and set itself apart: it measures closed-loop VLA binding, not VLM pointing, and its levels are type/instance/sub-region, not reasoning-step count.
- ReGuide is the strongest fix to test at the sub-region level. If it grounds by object pose, it may have no pose for "top drawer". A pose-based wrapper would then plausibly degrade at the sub-region level, but this is untested.

### Gaps
- The RoboPoint, RoboRefer, SoFar, MOKA and CAG papers were not fetched this session.
- CRAFT's full text (its models and whether it uses drawer tasks) was not read.

## Q3. Does any robotics work use MCD-style compound-divergence splits or show exemplar/nearest-neighbour binding?

### Takeaway
- MCD: I found no robotics or VLA paper that uses the Keysers et al. compound-divergence metric or MCD splits.
- Exemplar binding: ECT is direct prior work. It shows nearest-trajectory retrieval behaviour in pi0.5 and causal prefix-KV interventions. The atlas cannot claim "VLAs bind by retrieving exemplars" as new. What remains open is measuring binding error as a function of compound divergence to tell holistic binding from factorised binding.

### Cited Findings
- **ECT on pi0.5:**
  - Behaviour: 69% of 381 probes "follow a familiar behavior". Excluding two rules where the familiar behaviour stays valid, 87% of 291 probes are retrieval-like.
  - A nearest-trajectory check agrees with the human label on 88% of usable Goal retrieval cases.
  - Prefix-KV readout ranks the executed task first in 34/45 cases and in the top three in 43/45. Text-only TF-IDF ranks it first in 22/45.
  - Replacing the prefix-KV cache moves the arm toward the target task's object in 109/200 rollouts, vs 0/200 for controls.
  - No MCD in the portion read.
  - [arXiv 2609.39971](https://arxiv.org/html/2609.39971)
- **Search for compound-divergence splits in robot manipulation returned none.** Nearest neighbours:
  - "Diagnosing Compositional Generalization in Sequential Robot Tasks" (2607.29687) bounds the generalization gap with an "instruction-compositional shift" term. This is conceptually close to compound divergence but not framed as MCD (search snippet).
  - AC-VLA (2607.15714) names "trajectory overfitting", where models overfit to holistic trajectory patterns rather than sub-skill semantics, which is a holistic-binding claim.
  - SkillIndex-VLA, MA-VLA (2608.25864) and 2610.15425 use held-out recombination splits.
  - [Search results: 2607.29687](https://arxiv.org/html/2607.29687), [AC-VLA 2607.15714](https://arxiv.org/html/2607.15714), [MA-VLA 2608.25864](https://arxiv.org/pdf/2608.25864), [2610.15425](https://arxiv.org/html/2610.15425)

### Inferences
- An MCD axis (fixed atom distribution, swept compound divergence) with binding error as the dependent variable seems new for VLAs. The prediction that holistic binders degrade with compound divergence while factorised binders stay flat would be the distinguishing contribution.
- 2607.29687's bound should be checked for how close its compositional-shift term is to compound divergence.

### Gaps
- 2607.29687 and 2610.15425 were not read in full. Either could contain a divergence-graded split.

## Q4. Does any work decompose success into P(correct binding) x motor competence, or separate binding failures from motion failures?

### Takeaway
Yes. The factorisation is already published: ReGuide writes success as Pr(S_k) = beta_k x gamma_k, binding times competence. Others separate selection from grasping (RoboSemanticBench), or primitive reliability from instruction success (2602.24143). The atlas must cite these. It can only claim novelty for measuring the two factors per depth level and per backbone, which ReGuide does not report per baseline.

### Cited Findings
- **ReGuide Eq. 1:** Pr(S_k) = beta_k x gamma_k, where beta_k (binding) is reaching the correct interaction and gamma_k (competence) is completing it.
  - Supporting diagnostics: grasp success rises from 64.5% to 95.8% on eight recomposed cells with pre-contact guidance. Hand-back replacement drops grasp success from 98.6% to 44.0% (default descent-and-close) or 33.5% (hold).
  - The product is not reported per baseline.
  - [arXiv 2609.38616](https://arxiv.org/html/2609.38616)
- **RoboSemanticBench** scores semantic selection "after controlling for grasp success". [arXiv 2606.02277](https://arxiv.org/abs/2606.02277)
- **2602.24143** separates primitive reliability from instruction-conditioned success and recommends decomposed metrics. [arXiv 2602.24143](https://arxiv.org/abs/2602.24143)
- **2603.19233** separates goal selection (VLM) from motor generation (action expert) mechanistically. [arXiv 2603.19233](https://arxiv.org/abs/2603.19233)
- **ECT** shows the policy "adjusts execution while retaining the wrong task-level choice". Only 4% of rollouts are unstructured collapse. A scripted oracle reaches 90/100 on the Object Task cell. There is no explicit factorisation. [arXiv 2609.39971](https://arxiv.org/html/2609.39971)

### Inferences
- Overall verdict on novelty:
  - Taken: "binding vs motor" separation, "skills robust, grounding brittle", and "instructions retrieve trajectories" (exemplar binding).
  - Apparently open:
    - (a) a referring-depth gradient (type < instance < relational sub-region) measured across all five flow VLAs;
    - (b) showing that existing fixes (ReGuide, ECT, pointing/affordance front-ends) also degrade with depth;
    - (c) a compound-divergence (MCD) sweep that separates holistic/exemplar binding from factorised binding;
    - (d) a beta x gamma estimate reported per cell, per depth and per backbone.
- The atlas should present beta x gamma as ReGuide's factorisation and measure it at each depth, not present it as new.

### Gaps
- Not verified whether ReGuide's appendix reports beta and gamma per baseline or per axis.
- SmolVLA appears in 2602.24143 and 2603.19233. GR00T N1.5 appears only as the EquiVLA base. No paper checked covers all of pi0, pi0.5, SmolVLA, X-VLA and GR00T N1.5 together; ReGuide covers pi0, pi0.5, X-VLA and GR00T N1.7 but not SmolVLA or N1.5.

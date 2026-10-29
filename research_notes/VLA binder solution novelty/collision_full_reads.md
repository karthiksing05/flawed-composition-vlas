# Collision check (full-text reads): planned VLA binding paper vs 7 named papers + searches (through Oct 2026)

Planned contributions under test:
- C1: failure atlas, binding error rising with referring-expression depth (type < instance < relational sub-region, e.g. "top drawer of the cabinet") and with compound divergence (MCD).
- C2: union-bound / factorised systematicity guarantee for binding; holistic (nearest-neighbour) binders provably fail as compound divergence grows.
- C3: unsupervised, prototype-based word-to-referent binder learned from demo grasp/release events, plugged into a frozen VLA.
- C4: once-trained adapter that writes role masks into a frozen VLA's deep image-token K/V.
- C5: referring-expression grounding of the base VLM vs the robot-fine-tuned VLA; the base VLM used as a binder.

Method: arXiv HTML full text, read in two chunks where the page was over 100k characters (GSR, 2609.39971, ReGuide). A small model answered targeted questions against each chunk, so the "quotes" below are its extractions with section numbers, not my own verbatim transcription. Verify any sentence before quoting it in the paper.

## Q1. GSR, arXiv 2608.02497, "Grounded Semantic Re-Binding for Robust Instruction Generalization in VLA Models" (Yin, Zhang et al., SJTU AutoLab / Anyverse; v1 3 Aug 2026)

### Takeaway
GSR is about robustness to paraphrase. A frozen T5-large re-encodes the instruction, and the paper injects it per architecture: sidecar K/V in VLA-Adapter's action blocks, the language slots in SmolVLA, and the PaliGemma task slots in pi0.5. The action head is retrained on LIBERO-Goal. "Binding" here means instruction-to-action re-binding. It has no referent binding, no depth taxonomy, no MCD, no theory and no masks on image tokens. Verdict: C1 none, C2 none, C3 none, C4 partial (mechanism neighbour only), C5 none.

### Cited Findings
- Method: frozen T5-large plus a learned projection (Sec. 3.1, App. B.1). For VLA-Adapter, Qwen gets the fixed prompt "perform the task" and the T5 tokens enter each action block "as a sidecar K/V source with its own softmax and gate" (App. B.2.1). SmolVLA uses grounding attention into the native language slots (Eq. 7, App. B.3). pi0.5 grounds T5 tokens into the PaliGemma task slots and adds a fresh 300M action expert (App. B.4). The action head is re-initialized and trained on canonical LIBERO-Goal demonstrations (Sec. 3.1). — [arXiv 2608.02497](https://arxiv.org/html/2608.02497)
- Results on LIBERO-Para (4,092 episodes): VLA-Adapter 46.82 -> 70.94%, SmolVLA 4.47 -> 49.12%, pi0.5 73.60 -> 75.59% (Table 2). The final Bridge-Attention block recovers 96.8% of the action gap in VLA-Adapter (Sec. 4.1). Route-conflict hijacking appears in Table 3. ParaVLA (0.33B, frozen T5 + DINOv2) scores 92% canonical and 91% paraphrase (Sec. 5, Table 4). — [arXiv 2608.02497](https://arxiv.org/html/2608.02497)
- The closest thing to C1 is a split of object-reference wording changes (Obj, 259 episodes) from combined changes (Comp, 2,963) in Table 5. There is no type/instance/sub-region taxonomy and no MCD. App. D.3 says no model met the criteria for a universal cross-architecture semantic breakpoint. — [arXiv 2608.02497](https://arxiv.org/html/2608.02497)
- Related work it cites on VLA grounding: Zhang et al. 2026 "Restoring linguistic grounding ... train-free attention recalibration", Pugacheva 2025 "Bring the apple, not the sofa", Zhan 2026 "Stable language guidance", Wu 2026 "Do what you say", Swann 2026 (SAE features in VLAs), LIBERO-Pro, CLIPort, VLMbench, Wang 2023 "Programmatically grounded, compositionally generalizable manipulation". — [arXiv 2608.02497 refs](https://arxiv.org/html/2608.02497)

### Inferences
- C4 is the main risk. GSR already "adds a K/V source to a frozen-ish VLA," so reviewers will see a resemblance. The planned adapter differs in three ways: it writes role masks onto image-token K/V deep in the backbone, it is trained once and stays frozen, and the VLA is unchanged. GSR adds language K/V, sends it to the action blocks, and retrains the action head. Cite GSR and state these differences.

### Gaps
- The full author list was truncated.

## Q2. arXiv 2610.15425, "Exploring the Limits of Combinatorial Generalization in Embodied Manipulation" (SJTU AutoLab)

### Takeaway
This paper is about sequencing skills (ordered two-stage compositions of atomic RoboTwin skills) and the "start state gap." It does not touch referent binding. Verdict: no overlap on C1 through C5.

### Cited Findings
- Training on full instructions: 40.0% ordered success on seen compositions (Set A) vs 15.0% (Set B) and 4.4% (Set C). The authors say the model may "memorize sequence templates" (Sec. 4.1). — [arXiv 2610.15425](https://arxiv.org/html/2610.15425)
- HSE is a diagnostic: an LLM decomposer, a GRU handoff cue, and matched oracle modes (Sec. 3.2, App. C). RoboCombine has 50 atomic tasks and 19 unseen two-stage compositions x 20 rollouts (Sec. 3.1, App. B). — [arXiv 2610.15425](https://arxiv.org/html/2610.15425)
- PRTI samples starts from late segments of demonstrations, mixed 1:1 with canonical starts; pi0.5 is fully fine-tuned with no adapters (Sec. 4.2, App. D). Ordered success goes from 2.63% to 18.16% and conversion from 7.52% to 39.88% (Table 2). — [arXiv 2610.15425](https://arxiv.org/html/2610.15425)
- App. A (limitations): compositions use only familiar atomic labels. — [arXiv 2610.15425](https://arxiv.org/html/2610.15425)

### Inferences
- Its "memorized sequence templates" finding is a sequence-level version of the holistic-binder argument in C2. Cite it as supporting evidence. It is not a collision.

### Gaps
- None material.

## Q3. arXiv 2607.29687, "Diagnosing Compositional Generalization in Sequential Robot Tasks" (Wang, Wu, Sun, ..., Tomizuka; UC Berkeley; v2 15 Oct 2026)

### Takeaway
Prop. 4.1 is a generic decomposition of distribution shift. Under a bounded loss, the change in test loss is bounded by marginal shift + instruction-compositional shift + context-action shift, all as L1 distances. It is not a factorised systematicity guarantee for binding and says nothing about nearest-neighbour binders. Verdict: C2 partial (closest prior theory; cite and contrast), C1/C3/C4/C5 none.

### Cited Findings
- The bound: assuming 0 <= L <= M, |Delta_q(theta)| is bounded by ||q(l_i) - p(l_i)||_1 + E_{l_i~p}[||q(l_-i | l_i) - p(l_-i | l_i)||_1] + E_{l~p}[||q(z,a | l) - p(z,a | l)||_1] (Sec. 4, Prop. 4.1; proof in App. A). The middle term is the "instruction-compositional shift," described as "whether familiar subtasks are recombined in unfamiliar ways." — [arXiv 2607.29687](https://arxiv.org/html/2607.29687)
- The term is never estimated. Sec. 5.2 uses "seen pair count" as a proxy: "pairwise coverage mitigates the instruction-compositional discrepancy." PP saturates around B=8 and PPP around B=16 (Sec. 5.2). One demonstration per task raises OOD success from 0.4% to 54.7% (Sec. 5.3). — [arXiv 2607.29687](https://arxiv.org/html/2607.29687)
- Benchmarks are robomimic PP/PPP/2S-PP and an Isaac Lab human-demonstration task with 27 combinations. There is no MCD, no referring-expression depth and no binding. — [arXiv 2607.29687](https://arxiv.org/html/2607.29687)

### Inferences
- The planned C2 must be presented as different in kind. It bounds binding error by a sum over per-factor (part/role) errors, so the guarantee holds under compound divergence. Separately, it shows a lower bound for holistic or nearest-neighbour binders that grows with divergence. 2607.29687's bound grows with compositional shift for any model, so it cannot separate factorised from holistic binders. Say this in one sentence in the paper.

### Gaps
- I did not check whether App. A contains further corollaries, such as a per-factor version.

## Q4. arXiv 2609.39971, "When Instructions Retrieve Trajectories" (Chen, Hou, ..., Min Sun, Chun-Yi Lee; NTHU/NTU), Appendix E

### Takeaway
The prefix-KV swap replaces the whole prefix K/V cache: image, language and state tokens are not separated. Table 16 compares only the final layer with all layers. Nothing is stratified by referring-expression depth or sub-region. Verdict: C1 none (its "instruction-action binding" is trajectory retrieval and has no depth strata), C4 partial at most (whole-cache K/V intervention used as a diagnostic, not a role-mask adapter), C2/C3/C5 none.

### Cited Findings
- E.2 replaces "the prefix K/V cache of each of the 45 perturbed inputs," i.e. the whole prefix. The action expert's suffix-token states are excluded, and decoding runs "with the action expert unchanged" (E.1-E.2). — [arXiv 2609.39971](https://arxiv.org/html/2609.39971)
- Layers (Table 16): Source/Executed x final/all. "All-layer patching changes the chunk more but worsens the first-step distance." No subset of layers is tested. — [arXiv 2609.39971](https://arxiv.org/html/2609.39971)
- The readout ranks candidates by "the final-layer prefix key-value (KV) representation read by pi0.5's action tokens" (Sec. 4.1). For GR00T-N1.7 the swap targets "the backbone feature sequence read by the action head" (Sec. 4.1). Results are reported by suite and task pair (e.g. 50 Object-suite pairs) with no depth or sub-region strata. Tables 16-17 are pooled over 45 rollouts. — [arXiv 2609.39971](https://arxiv.org/html/2609.39971)
- The fix is ECT (Equivariant Counterfactual Training), which needs retraining. — [arXiv 2609.39971](https://arxiv.org/html/2609.39971)

### Inferences
- It is safe to say that this paper uses whole-cache swaps for diagnosis, while the planned work writes token-selective role masks into image-token K/V at chosen deep layers.

### Gaps
- Appendix D Table 10 (human labels by rule, R_g,4 multi-field change) was only summarised.

## Q5. ReGuide, arXiv 2609.38616, appendices

### Takeaway
ReGuide wraps a frozen VLA as an executor and steers end-effector commands. An external GroundingDINO detector does the grounding. In two composition cells (A2/A3) the drawer is only a placement destination ("the open top drawer"). Articulated manipulation is not evaluated, and there are no per-depth results. Verdict: C1 partial-weak (contains a "top drawer" destination cell but no depth stratification), C3 partial-weak (frozen VLA plus external binder, but supervised off-the-shelf detector, no prototypes, no grasp/release learning), C4 none, C5 none, C2 none.

### Cited Findings
- "ReGuide leaves the model, its training, and its decoding untouched and wraps any of these policies as a frozen executor" (Sec. 2.1). Grounding outputs "entity poses and axis-aligned bounding boxes" (Sec. 3.1). Guidance acts on end-effector commands. — [arXiv 2609.38616](https://arxiv.org/html/2609.38616)
- Real robot: "open-vocabulary detector with depth and measured object dimensions" (Sec. 4.1). Simulation uses oracle simulator state. The perception-based variant is 7.0 points below oracle grounding (Sec. 4.2). D.5/C.6: GroundingDINO (tiny; box and text thresholds 0.25) prompted with entity names, with wrist-camera refinement (0.10 m objects / 0.25 m destinations). It is a separate module, not the VLA's base VLM. — [arXiv 2609.38616](https://arxiv.org/html/2609.38616)
- Recombination axes: Target Region, Target Object, Unseen Object, Task Scene, varied "one factor at a time" (Sec. 4.1). Table 6 cells A2 and A3 place the object in "the open top drawer" of a wooden cabinet. On pi0.5, ReGuide scores 48/50 and 46/50 vs 5/50 and 42/50 for the bare policy. App. E: articulated manipulation is not evaluated. — [arXiv 2609.38616](https://arxiv.org/html/2609.38616)
- "Habitual" retrieval: "retrieve the training scene nearest to the first frame under a frozen image encoder" (Sec. 3.2), a nearest-neighbour step. — [arXiv 2609.38616](https://arxiv.org/html/2609.38616)

### Inferences
- The A2 cell (5/50 bare pi0.5 for "top drawer" placement) is outside evidence that relational sub-region references break pi0.5, which is consistent with C1. ReGuide is the closest "frozen VLA + external binder" baseline and should be included as one.

### Gaps
- I did not check whether drawer-cell success is reported separately under perception-based grounding.

## Q6. TwinBrainVLA, arXiv 2601.14133

### Takeaway
A frozen "Left Brain" VLM supplies stop-gradient K/V to a trainable "Right Brain" through AsyMoT, and the action expert reads the Right Brain. The only before/after measurement uses general VL benchmarks (POPE, SEED, ScienceQA, MME), not grounding or referring expressions. Verdict: C5 partial (the frozen base VLM is kept as a "semantic anchor," but nothing measures grounding and the VLM is not used as a binder), C4 partial-weak (the frozen VLM's K/V enters joint attention, but no role masks and no image-token writes), C1-C3 none.

### Cited Findings
- "the POPE score of Qwen3-VL drops from 88.87% to near-zero (0.04%)" after VLA training (Sec. 3.2, Fig. 2b). No RefCOCO or pointing evaluation appears. — [arXiv 2601.14133](https://arxiv.org/html/2601.14133)
- Left Brain -> Right Brain via stop-gradient K/V in joint attention (Eqs. 4-6, Sec. 4.1). The action expert is conditioned on H_R (Sec. 4.2). Fig. 5 shows attention maps with no grounding metric (Sec. 6.1). — [arXiv 2601.14133](https://arxiv.org/html/2601.14133)

### Inferences
- The planned C5 measurement (base-VLM vs VLA referring-expression grounding, stratified by depth) is not in TwinBrainVLA. Cite it as evidence that forgetting happens (POPE collapse).

### Gaps
- None material.

## Q7. SPARC, arXiv 2606.13497 (Blank, Mattes, ..., Reuss, Lioutikov; KIT/NVIDIA), related work

### Takeaway
SPARC uses grasp cycles to auto-label spatial annotations. The referent comes from an off-the-shelf text-prompted detector (LLMDet + SAM2), so no word-to-referent binding is learned. Its related work cites no contact/grasp-event word-learning papers, and the only grasp-event supervision it cites is Robo2VLM. Verdict: C3 partial (same grasp/release signal, but supervised detection with no prototypes and no frozen-VLA plug-in), C1/C2/C4/C5 none.

### Cited Findings
- Pipeline: grasp-cycle detection + LLM subtask parse -> LLMDet + SAM2 + AllTracker + MoGe-2 -> reliability score combining phase-aware motion, 3D gripper proximity, a robot-overlap penalty and detector confidence (Eq. 3, weights hand-set on validation). — [arXiv 2606.13497](https://arxiv.org/html/2606.13497)
- "Following grasp-phase detection [25], we extract maximal closed-gripper intervals" (Sec. 3.1), where [25] = Robo2VLM. Table 5 lists Robo2VLM's signal as "Grip. phases." Downstream VQA and VLA models are fine-tuned (Qwen3.5-4B; 0.8B VLA backbone), not frozen. — [arXiv 2606.13497](https://arxiv.org/html/2606.13497)

### Inferences
- In SPARC's lineage, the contact-event signal is used to label data for supervised training, not for unsupervised word learning. The C3 claim stays open but must cite SPARC and Robo2VLM. The developmental word-learning literature (cross-situational learning, infant handling) still needs a separate search.

### Gaps
- The last 42k characters (late references and appendix) were not read.

## Q8. Searches: binding/referring/sub-region/part-level VLA papers, and SigLIP feature clustering in VLAs

### Takeaway
No 2026 paper turned up that stratifies VLA binding by referring-expression depth or MCD, gives a factorised binding guarantee, learns an unsupervised grasp-event prototype binder, writes role masks into image-token K/V, or clusters a VLA's own SigLIP features into objects or parts. Several neighbours need citing.

### Cited Findings
- 2608.29967 "What Can One Sentence Fix?" (Kwon, Ortega-Kral, Bucker, Jean Oh; CMU; v2 23 Oct 2026): labels 133 failing LIBERO-90 pairs (OpenVLA-OFT, pi0.5) as task misunderstanding 47/50% (relative spatial reference resolved in the image frame, e.g. "the middle bowl"), perception failure 25/22% (wrong object), execution misalignment 23/22%, and multi-step 5/7% (Sec. 3). There is no type/instance/sub-region depth, no MCD, no base-VLM comparison and no K/V. It notes frozen pi0.5 "gets stuck at the top drawer." C1 partial-weak. — [arXiv 2608.29967](https://arxiv.org/html/2608.29967)
- RoboSemanticBench 2606.02277 (Bin Yu et al., same group as TwinBrainVLA): tests whether action targets follow math/commonsense semantics through lettered blocks. 95.92% of QwenGR00T's grasp-success failures are wrong-target selections (Table 5), and VLA cotraining lowers TSR to 8.2% (Table 4). It has no depth strata, no MCD, no base-VLM evaluation and no binder. No overlap, but useful as wrong-target evidence. — [arXiv 2606.02277](https://arxiv.org/html/2606.02277)
- 2603.06001 IGAR, "Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration" (Zhang, Zhu, Zhou, Chen; reportedly ECCV 2026): recalibrates attention at inference with no training, on ICBench (contradictory instructions). Only the search snippet was read. C4 partial-weak: it intervenes in attention in a frozen VLA, but without learned role masks or K/V writes. — [arXiv 2603.06001](https://arxiv.org/abs/2603.06001)
- 2510.25616 "Don't Blind Your VLA": reports that fine-tuned VLAs have diffuse or misplaced attention on key entities while the pretrained VLM attends to task objects, and that representations collapse. Only the search snippet was read. C5 partial: a base-VLM vs VLA attention comparison, with no referring-expression depth and no binder. — [arXiv 2510.25616](https://arxiv.org/html/2510.25616v1)
- Point-VLA 2512.18933 (bounding-box cues added to instructions) and VAP "Bring My Cup!" 2512.20014 (training-free input-side highlighting for frozen VLAs via detection + embedding matching, instance-level) work on referent disambiguation in the input or pixel space, not on parts. C3/C4 partial-weak. — [arXiv 2512.20014](https://arxiv.org/html/2512.20014v2)
- SlotVLA 2511.06754 applies slot attention to dense encoder patch features to get object-centric slots. Nearest prior for "clustering VLA vision features into objects," but it is a trained slot module, not an unsupervised clustering of the frozen VLA's own SigLIP features into parts. — [arXiv 2511.06754](https://arxiv.org/html/2511.06754)

### Inferences
- Overall novelty, by contribution:
  - C1: open. Neighbours are 2608.29967 (relative references), the ReGuide A2 cell and GSR Table 5 (Obj vs Comp).
  - C2: open, but contrast with 2607.29687 Prop. 4.1.
  - C3: open. Neighbours are SPARC/Robo2VLM (grasp phases used for supervised labels) and VAP (embedding matching).
  - C4: open, but distinguish from GSR's sidecar K/V, 2609.39971's whole-cache swaps, TwinBrainVLA's frozen-VLM K/V and IGAR's attention recalibration.
  - C5: the measurement is open. Cite TwinBrainVLA (POPE collapse) and "Don't Blind Your VLA" (attention comparison).

### Gaps
- IGAR (2603.06001), "Don't Blind Your VLA" (2510.25616), Point-VLA and VAP were seen only through search snippets, not full text. IGAR and "Don't Blind" are the two worth reading in full next, for C4 and C5.
- The PDF fetch of RoboSemanticBench failed; the HTML version was read instead.
- Search engines may miss very recent (Oct 2026) arXiv listings. A direct arXiv listing search for "binding" + "VLA" was not done.

# Novelty check: stage-resolved failure atlas of VLAs on held-out pairings

Date: 2026-11-12. Scope: arXiv from 2026-01, CoRL/ICML/IROS 2026 listings. Method: arXiv abstract pages fetched for every paper below. A second pass (section 3b) read the arXiv HTML full texts of ATOM-Bench, Harness VLA, VLA-Corrector and Failing Forward. LIBERO-Gen has no arXiv listing, so only its ICML poster page was read. Verdicts marked [FT] rest on those full texts; unmarked verdicts rest on abstracts and snippets.

## 1. Id verification

| Cited as | arXiv id | Resolves to | Matches claim |
|---|---|---|---|
| CRAFT | 2610.00524 | "Same Scene, Different Task: Skill Alignment for Compositional Generalization in VLAs" (Yang, Na, Cho, Yoon; submitted 2026-09-30; cs.RO; abstract names the CRAFT method) | Yes. Three VLAs, two sims, vision shortcut. The abstract names no venue. |
| AC-VLA | 2607.15714 | "AC-VLA: Robust Out-of-Distribution Action Execution via Compositional Learning" (Peng et al.; 2026-07-17) | Yes. Abstract says "hides wrist-view inputs during closed-gripper phases" (the earlier review noted the abstract and Sec. 3.3 disagree on closed-gripper vs place). About +28 points on pi0.5. |
| ACT-VLA | 2607.00351 | "Unleashing More Actions via Action Compositional Training for VLA Models" (Peng, Lu, Peng; 2026-07-01) | Yes. Offline augmentation from latent task representations. |
| ReGuide | 2609.38616 | "Correcting WHERE, Preserving HOW: Compositional Generalization for VLAs via Referential Guidance" (Zhang et al.; 2026-09-29) | Yes. Training-free wrapper, frozen policy, up to +56.8 sim and +75.0 real. |
| LIBERO-CF | 2602.17659 | "When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs" (Fang et al.; 2026-02-19; later revisions) | Yes. LIBERO-CF benchmark and CAG. The abstract page names no venue; the review's "CoRL spotlight" is unconfirmed here. |
| Brittle Grounding | 2602.24143 | "Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in VLA Policies via Multi-Object Picking" (Emukpere, Deffayet, Renders; 2026-02-27) | Yes. SmolVLA and pi0.5, held-out object-location pairs. |
| Grant et al. | 2603.19233 | "Not All Features Are Created Equal: A Mechanistic Study of VLA Models" (Grant, Zhao, Wang; v1 2026-03-19, v6 2026-11-02) | Yes. Activation injection redirects pi0.5 in 99.6% of 1,968 episodes; five VLAs, 420k rollouts. |
| Wang et al. | 2607.29687 | "Diagnosing Compositional Generalization in Sequential Robot Tasks" (Wang, Wu, Sun, ..., Tomizuka; v1 2026-07-31, v2 2026-10-15) | Yes. A community CoRL 2026 list (#136) also carries it. Abstract: a structured subset of demonstrations matches full enumeration when it covers action-relevant dependencies; one demo per task lifts OOD from 0.4% to 54.7%. |

All eight ids resolve to the claimed papers.

## 2. Closely related papers and exact overlap

Already in the earlier review (CRAFT, AC-VLA, ACT-VLA, ReGuide, LIBERO-CF, Brittle Grounding, Grant, Wang) is not repeated beyond the overlap lines below.

### Overlap with the atlas (stage decomposition, multi-model)

1. **Brittle Grounding** (2602.24143, preprint). Measures reach, grasp-anything, and instruction-conditioned success separately for SmolVLA and pi0.5 on held-out object-location pairs. Overlap: stage-wise decomposition and grasp-level binding failure on held-out pairings, two models. Gap: grasp-only task (no carry, release, or physics stage), no hand-off stage, two models, no excess-over-partner measure, no reference system.
2. **CRAFT** (2610.00524). Table A.4 splits held-out LIBERO-Goal rollouts into correct task, share-object task (38.4% for pi0.5: right object, trained partner), and other. Overlap: same benchmark, same models (pi0.5, pi0, GR00T), and an object-grasped-then-wrong-destination outcome class. Gap: outcome classes only; no first-broken-stage assignment, no release or physics stage, no partner-excess baseline.
3. **ATOM-Bench** (2606.16826, preprint, community CoRL list #44). Real-world, 30 atomic and 24 held-out compositional tasks, five policies, 2,700 rollouts. Reports Atomic Score and Compositional Failure Share, which separate weak skills from poor skill reuse. Overlap: multi-model, held-out compositions, a measure of failures attributable to composition versus skill. Gap: failure split is skill-vs-composition at the task level; no per-stage first-break assignment; Pi0.5 15.8% compositional on Franka. Check full text for stage breakdowns (*).
4. **Benchmarking VLAs on SO-101** (2606.08881, preprint). pi0.5, SmolVLA, Wall-X, ACT on four real tasks. Two-level taxonomy (semantic vs execution), "execution instability dominant". A blog summary says the taxonomy includes grasp instability, repetition loops, state mismatch. Overlap: multi-model failure taxonomy with grasp-type categories. Gap: not compositional or held-out; no pairing excess.
5. **Failing Forward** (2605.08434). In a grasp-and-place task, grasp failures are 63.0% of logged failures and misaligned placement 1.9% (per search snippet). Overlap: a per-stage failure share. Gap: single task, training method paper, no held-out pairings.
6. **LIBERO-Gen** (ICML 2026 poster 64080). Compositional tiers (Spatial-CG, Task-CG; pi0.5 64.0 and 21.2). Names perceptual instability, action binding collapse, shortcut reliance. Overlap: held-out compositions on LIBERO, binding failure named. Gap: no stage-level analysis on the page.
7. **VLA-Corrector** (2609.06508). Defines five stages (approach, alignment, grasp, transport, placement) for a verifier. Overlap: a stage vocabulary close to ours. Gap: used for runtime recovery under perturbation; no held-out pairings; model unnamed.
8. **Grant et al.** (2603.19233). Five VLAs, goal selection vs motor generation separate into VLM and action-expert pathways. Overlap: multi-model diagnosis and a "which module fails" split. Gap: mechanistic, no episode-level stage labels.
9. **LiLo-VLA** (2602.21531) and **Harness VLA** (2607.08448; LIBERO-Pro +38.6 points). Both hand transport and release to non-learned primitives and use a VLA for contact. Overlap: a hand-off from a VLA to a scripted or object-blind component, which is our reference/hand-over fix. Gap: built for long-horizon reliability; no held-out pairing diagnosis, no per-stage atlas. Harness VLA abstract says nothing on failure decomposition (*).
10. **Demystifying VLA failures in contact-rich tasks** (2608.01402, CoRL 2026 spotlight). Two failure modes (precision, force). Overlap: a multi-cause VLA failure account. Gap: contact-rich, not compositional.
11. **VLA-Trace** (2605.30117) pi0.5 and OpenVLA; representation and behavior tracing; compositional control listed as future work. Little overlap.
12. **Wang et al.** (2607.29687) says stage-wise policies that cannot see inactive instructions fail; sparse training fails because the policy misreads the instruction, not for lack of skills. Overlap: low-level skills intact, binding fails. Gap: small custom policy, no stage labels.

### Overlap with the fixes

- **AC-VLA** (2607.15714): whole-wrist masking during closed-gripper phases. Overlaps with role views; our result that role views match AC-VLA masking confirms equivalence and gives no priority for the masking idea.
- **ReGuide** (2609.38616): frozen pi0.5 39.0 to 94.5 on LIBERO compositions via a pose-based hand-off. Overlaps with the hand-over fix (.50 to .96).
- **Cloak** (2606.22836, CoRL 2026) masks the end-effector in the wrist view for cross-embodiment transfer; **MaskVLA** (2609.23565, IROS 2026) masks the main camera randomly; **FoveaVLA** (2610.20064) drops the wrist camera. Related masking, different purpose.
- **LiLo-VLA** masks non-target objects in the wrist view for a local policy.

### Searches that returned nothing on point

- Chunk length or gripper timing as a cause of compositional failure: only adaptive chunking (CVPR 2026, Liang et al.), VGAS, and CheckVLA appeared. None tests chunking against compositional failure. AC-VLA and the earlier review's VP-VLA use gripper events as phase cues, with no timing ablation of failures.
- Forced gripper closure as a release test: no hit.
- X-VLA failure analysis on LIBERO-Goal held-out pairings: no hit.
- Partner-count law refuted across models: only Wang et al. (small policy) states the law.

## 3. Verdict per element

| Element | Verdict | Reason |
|---|---|---|
| First-broken-stage assignment of every failed episode (grasp, hand-off, transport, release, physics) | PARTLY PRIOR [FT] | Failing Forward Table S9 already assigns each failed episode its earliest failure event (grasp 63.0%, released too early 3.8%, placement 1.9%) for one model on one task. No full-text paper does it on held-out pairings across several VLAs, and "hand-off" as a stage has no prior. |
| Excess failure over known partner tasks, per stage | NOVEL [FT] | No paper found measures a pairing's per-stage failure as excess over its partner tasks; CRAFT's share-object class is the nearest and is an outcome, not an excess. |
| Reference system (scripted servo / oracle parts policy) succeeding on the same states | PARTLY PRIOR [FT] | Harness VLA, LiLo-VLA and ReGuide use analytic components as fixes; VLA-Corrector has a privileged rule verifier as a verifier upper bound. None runs a scripted policy on the same states as an existence proof in a failure atlas. |
| All five fail new pairings in a familiar scene | PRIOR [FT] | CRAFT, ReGuide, Brittle Grounding, LIBERO-Gen, ATOM-Bench report it for pi0.5, pi0, GR00T, SmolVLA; X-VLA adds a model. |
| pi0 and SmolVLA grasp the habitual object; pi0.5 and GR00T grasp right and fail at hand-off and release | PARTLY PRIOR | Brittle Grounding shows grasp-level binding failure for SmolVLA; CRAFT shows pi0.5 and GR00T grasp the right object and use its trained partner. ATOM-Bench full text gives no stage split, so the model split by stage and the hand-off and release breakdown stay new. |
| New layouts fail at the grasp for every system, including an exact-position planner, with no pairing-specific excess | NOVEL | No hit for a planner-with-exact-positions control on layouts; Brittle Grounding's jitter results involve policies only. |
| Ruled out: action chunk length | NOVEL [FT] | None of the four full texts ablates chunk length (fixed or config-only); no abstract-level hit elsewhere. |
| Ruled out: release timing (forcing the gripper shut) | NOVEL [FT] | No gripper-timing test in the four full texts ("released too early" is only a label in Failing Forward); gripper events appear as phase cues in AC-VLA and VP-VLA. |
| Ruled out: partner-count law across models | NOVEL | Only Wang et al. state the law, on a small custom policy; a refutation across pretrained VLAs has no prior. |
| Role views (hide non-active object identity) raise a matched VLA .025 to .425 | PARTLY PRIOR | LiLo-VLA masks non-target objects; AC-VLA masks the wrist view (indistinguishable, per your result). The identity-only hiding with a matched-VLA ablation is the new part. |
| Hand-over to an object-blind place policy: pi0.5 .50 to .96 | PRIOR | ReGuide (39.0 to 94.5, frozen pi0.5), Harness VLA, and LiLo-VLA hand control to non-learned components. The 0.96 number differs; the idea does not. |


## 3b. Full-text audit (second pass, 2026-11-12)

Each paper was checked for (i) a per-stage or first-failure table on held-out compositions, (ii) a reference, planner or oracle control on the same states, (iii) a test of chunk length or gripper timing as a cause. Page-length caveats are noted.

| Paper | (i) per-stage / first-failure on held-out compositions | (ii) reference control on same states | (iii) chunking / gripper timing |
|---|---|---|---|
| ATOM-Bench 2606.16826 (full HTML) | No. Sec. 4.4 sorts failures into five types (motor, reference, constraint, safety abort, timeout) over all tasks, with only "over 90%" for motor+reference+timeout and no per-category numbers. Partial-progress score PSR (Table 2) is not mapped to grasp/transport/place/release. Models: Pi0.5, LingBot-VLA, GR00T N1.6, SmolVLA, Motus (real robot, 24 held-out compositional tasks). Table 2 composition SR (Franka): Pi0.5 15.8, Motus 10.8, LingBot 3.3, GR00T 3.3, SmolVLA 0.0; Compositional Failure Share 73.7 for Pi0.5. | None | Chunk length not tested (Table 10 lists optimizer settings only) |
| Harness VLA 2607.08448 (full HTML v3, read in two parts, all 182k characters) | No. Results are aggregate success (Table 3: LIBERO-Pro 82.4 vs 50.0 for pi-RLinf; Table 5 per-task Goal). Fig. 6 attributes the primitive that fires completion, for successes only. | No oracle. Comparisons are against direct frozen-VLA rollouts. | No ablation. Chunk size 50 appears only as a config entry (Table 14). `release` is a primitive with no timing test. |
| VLA-Corrector 2609.06508 (full HTML) | No. Five stages (approach, align, grasp, move, place) route recovery; no per-stage failure table; no held-out composition split (episodes split 60/20/20 by episode). Only pi0.5 (OpenPI pi05_libero). | Partial. "Rule + Prompt" uses a privileged rule verifier as an upper bound for the verifier, not a scripted policy on the same states. | No. replan_steps fixed at 5 (3 in recovery); no ablation. |
| Failing Forward 2605.08434 (full HTML; last 18k characters of supplement read separately) | Partly. Table S9 assigns every failed episode its earliest failure event: grasp failed 267 (63.0%), no action 96 (22.6%), wrong-object 22 (5.2%), released too early 16 (3.8%), misaligned reach 12 (2.8%), misaligned placement 8 (1.9%), unstable stack 2 (0.5%), dropped in transport 1 (0.2%); total 424. The setting is a single pick-place/stacking task with pi0.5 + AFIL on ManiSkill; no held-out compositions. The text says 150 failed episodes / 424 segments while the caption says 424 episodes; unreconciled. | No control; motion planner only generates correction data | No chunk-length ablation (horizon 30 fixed); no gripper-timing test ("released too early" is a label only) |
| LIBERO-Gen (ICML 2026 poster 64080; no arXiv id found) | Not checkable: only the poster abstract is public. It names perceptual instability and action binding collapse, no stage analysis stated. | Unknown | Unknown |

Correction to section 2: item 5 cited Failing Forward's 63.0% / 1.9% figures from a search snippet. The full text confirms them (Table S9, supplement). It is a first-failure table, so the first-broken-stage scheme has a prior (single task, one model, no held-out pairings). The 150/424 failed-episode study that my first search could not place is this paper.

Remaining gaps: LIBERO-Gen full text is unavailable; ATOM-Bench Sec. 4.4 may hold a figure with category shares that the fetch tool summarised as text only.

## 4. Positioning advice

- Lead with the atlas as a measurement: per-stage excess over partners, five models, a same-state reference. That combination has no match in the sources found.
- Cite Brittle Grounding and CRAFT Table A.4 as the closest prior diagnosis; state that they stop at the grasp or at the outcome class.
- Cite ATOM-Bench for the skill-vs-composition split; read its stage tables before claiming the atlas is the first stage-level view on compositions.
- Frame the fixes as confirmations that match AC-VLA and ReGuide; they carry no priority claim.
- Unchecked: full texts of ATOM-Bench, LIBERO-Gen, Harness VLA, and VLA-Corrector. Any could hold a stage table that narrows the "first-broken-stage" verdict.

## Sources

- arxiv.org/abs/2610.00524, 2607.15714, 2607.00351, 2609.38616, 2602.17659, 2602.24143, 2603.19233, 2607.29687 (all fetched)
- arxiv.org/abs/2606.08881 (SO-101), 2608.01402 (contact-rich), 2607.08448 (Harness VLA), 2605.30117 (VLA-Trace), 2609.23565 (MaskVLA), 2609.06508 (VLA-Corrector), 2607.21725 (Pigey; abstract only), 2608.02497 (GSR; paraphrase, no stage analysis) (fetched)
- icml.cc/virtual/2026/poster/64080 (LIBERO-Gen, fetched)
- arxiv.org/pdf/2606.16826 (ATOM-Bench), 2605.08434 (Failing Forward), 2602.21531 (LiLo-VLA), 2606.22836 (Cloak), 2610.20064 (FoveaVLA): seen in search results only

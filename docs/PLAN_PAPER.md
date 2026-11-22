# Plan: the paper, and a novel architectural fix (2026-11-16)

User, 2026-11-16 (verbatim): "Goal is to really fix the weaknesses of the paper - want to propose a comprehensive account of the various VLA failures and how they can be fixed and produce some novelty in the fix, hopefully to an architectural change".

## 1. Weaknesses and what answers each

| # | weakness | answer |
|---|---|---|
| W1 | Fix novelty. ReGuide is prior art. §70's composite is the sum of its parts. | A1: an architectural change inside the policy, trained on the same data as a plain retrain, that removes F1/F1m/F2/F7/F6 by construction (section 2). |
| W2 | "Architecture-rooted" rests on criterion C3 only. There are no probes inside the model. | A2: one probe per failure on frozen pi0.5, e.g. action-expert attention or activation patching at the place step: does the destination follow the held object's tokens? Grant et al. 2603.19233 is the prior to beat. |
| W3 | Uneven models: X-VLA has no fix; GR00T is below its floor in scene 2. | A3: report the floor as a result. Port X-VLA only if A1 is done. |
| W4 | Two LIBERO scenes only. | A4: a third LIBERO scene from a different suite, with the same atlas protocol. Optional: CALVIN or RoboCasa. |
| W5 | The paper draft is stale (paper/DRAFT.md, 2026-10-21, an earlier project). | A5: a rewrite from ARCH_FAILURES.md and the proposal's Final account. The old draft moves to docs/history/. |

## 2. Candidate architecture: role-addressed action (working name RAA)

The diagnosis in one line: the action expert picks its sub-goal from visual cues that covaried with it in training (held object, start side, nearer or salient plate), because every action token attends to the whole scene. Every fix that works binds a role to an entity and moves the arm in that entity's frame: ReGuide, the hand-over and U. THEORY_COMPOSITION.md P5 states the principle: identity is an address, and the motor computation never takes it as an input. RAA builds P5 into the policy.

1. **Entity tokens.** Pool the VLM's patch features under each segment of a learned segmenter, giving one token per object (as in r2l / §43; no simulator state).
2. **Role pointer.** Two learned queries (object, target), conditioned on the instruction's language tokens, attend over the entity tokens and output a pointer: a softmax over entities.
   - Supervision comes from the demos' own contact events (L4 rule, LADDER_RESULTS §5): the grasped entity is the one under the gripper at close; the target is the one under the held object at release.
   - No new information enters (C3).
   - The pointer latches while its entity is occluded. That is the F6 memory, built in.
3. **Role-frame action.**
   - The action expert receives the active role's entity: its pose and extent, plus a wrist crop. The active role is the object while the gripper is open and the target while it is closed.
   - It predicts end-effector deltas in that entity's frame.
   - The other entities reach it only as geometry, by an occupancy map without identity.
4. **Claim.** For any pairing whose fillers each appear in training, the action expert's input matches a trained input up to geometry. So new pairings reduce to trained ones by construction. The theory predicts which failures this removes (F1, F1m, F2, F7, F6) and which it does not (new layouts, sparse exceptions such as the bottle grasp).

**Test (C3 with matched data):** R0, a plain retrain of the same backbone on the same demos, against RAA. Both are scored on new pairings in scene 1 and scene 2, with the F7 and F6 tasks, against frozen pi0.5, ReGuide, and CRAFT-style counterfactual data at matched data.

**Kill:** RAA must beat R0 on new pairings with no trained-task cost, and match or beat ReGuide without demo statistics at test time. If the lit check finds the same design published, drop RAA or narrow it to the part that is new.

## 2b. Decision after the novelty check (2026-11-16; reports/novelty_raa.md, abstract level)

No single RAA component is new:
- object-frame actions: FOCI 2609.08743, TAX-Pose;
- object slots with an address/content split: OA-WAM 2605.06481;
- role-indexed object tokens: POT-VLA 2607.18016;
- phase masking: AC-VLA, LiLo-VLA.

No paper we found combines them, and none tests a combination on held-out pairings against a plain retrain at matched data. So the claim is the theory-driven decomposition:
- the diagnosis predicts that the place step needs role-addressed geometry;
- the ladder shows that input purity fixes the grasp (R2 grasps right .975) but not the place (.436 given a right grasp);
- one added channel, the active role entity's position relative to the gripper, should close the place gap at matched data.

**RAA-min, built from the R2 path (inventory: SR §34, §37; lifelong_ood/policy/l90c_config.py; experiments/c_train_pi05.py, sc_rv.py, sc_pu.py, sc_seg.py, pp_p_train.demo_sites):**
- **Training.** Same backbone, data and recipe as R0/R2: pi05_base, action expert only, the 8 known tasks, 6,000 steps, one A40.
- **Role channel.** Each frame gets the active role entity's position minus the end-effector position, in the robot base frame (3 numbers, plus 1 for the role id).
  - In training the entity comes from the demo's own contact events: the grasp site g before the hold and the release site r during it (pp_p_train.demo_sites).
  - It enters through the state input. That is the only architectural edit, and it adds no information.
- **Test-time pointer.**
  - Words: the contact-induced lexicon (parts_flow.role_lexicon / role_parse, LADDER_RESULTS §5).
  - Location: learned masks (sc_seg) plus the sc_pu Estimator. Rendered depth and calibration are privileged, as for ReGuide-perc.
  - The Estimator holds the last estimate when its entity is occluded, which gives F6 memory.
- **Arms at matched data:**
  - R0 and R2 already exist and are re-read on the same states;
  - Rrel = R0 + role channel, testing whether geometry addressing alone suffices;
  - RAA = R2 + role channel;
  - frozen pi0.5 and rgp ReGuide as references.
- **Primary contrast:** RAA − R2 on N1-N6, placed given a right grasp, and success.
- **Secondary contrasts:** Rrel − R0, and RAA − rgp. The K1-K6 non-inferiority margin is −.05.
- **Result (SR §73 phase 2, 2026-11-17): SUPPORTED.** RAA − R2 on N1-N6: success +.267 [+.092, +.467], placed given right grasp +.256 [+.077, +.461] (RAA .656 vs R2 .400; right grasp 1.000 vs .972). Rrel − R0 +.236 [−.006, +.553] (NO CLEAR DIFFERENCE); RAA − Rrel +.378 [+.133, +.658]. RAA − rgp −.242 [−.464, −.033] (rgp .897; not non-inferior). K1-K6: RAA − R2 +.086 [+.011, +.172] non-inferior; Rrel − R0 not shown. The place gap shrinks, it does not close.
- **Follow-up (SR §76, 2026-11-18): RAA-o (target channel measured from the held object) NOT SUPPORTED.** RAA-o − RAA on N1-N6: success −.061 [−.264, +.108], placed given right grasp −.054 [−.261, +.114]; RAA-o .594 vs RAA .656 vs rgp .897 (RAA-o − rgp −.303 [−.481, −.111]); K1-K6 non-inferiority not shown (+.042 [−.067, +.175]). Exploratory: the failed-release error no longer correlates with the in-hand offset (N4 r .14 vs .57) but is as large (median 6.5 cm).
- **Follow-up (SR §77, 2026-11-18): RAA-c (the channel as a continuous state token instead of text) NOT SUPPORTED.** RAA-c − RAA-o on N1-N6: success −.092 [−.306, +.072], placed given right grasp −.088 [−.301, +.077]; RAA-c .503 vs RAA-o .594 vs RAA .656 vs rgp .897 (RAA-c − rgp −.394 [−.608, −.158]); K1-K6 non-inferiority not shown (−.067 [−.233, +.061]). Exploratory: the failed-release error grew (N4 median 11.8 cm, N6 11.0, against 6.5 and 7.0), so number format is not what limits the place step. Write-up in docs/notes/sr77.md.
- **Follow-up (SR §79, 2026-11-18): WA module-removal test on RAA, primary NOT SHOWN.** Removing the resolver R lowers F7 (full .369 vs noR .156, +.213 [+.131, +.300], SUPPORTED); removing the pose memory M does not lower F6_b (+.050 [0, +.117], NOT SUPPORTED; full is at the executor floor, F6_a .083 NOT TESTABLE); noM - full on F7 -.075 [-.144, -.006] (NOT EQUIVALENT); DK non-inferior (+.008 [0, +.025]); full far below rc (F7 -.594, F6_b -.750). docs/notes/sr79.md.
- **Follow-up (SR §81, 2026-11-18): MT moved-target test, "the arm goes to the trained coordinates" NOT SUPPORTED for frozen pi0.5 (M1), seen for R2.** Plate moved .14-.29 m (5-11 x the demo range), 40 states, K1: I releases at a trained plate position in 0 of 40 (success .975 [.925, 1.000]; M1 0 [0, 0] NOT SUPPORTED); R2 releases there in 31 of 40 (.775 [.625, .900], success 0); RAA 1 of 40 but success .100 (M2 RAA − I −.875 [−.975, −.750] REVERSED, M3 not testable); rgp .950. Bowl moved far: I .375/.400 (K1/K2), R2 0/.075, RAA .125/.050, rgp .850/.925. docs/notes/sr81.md.
- **Later:** scene 2, where position is the cue, and the F7/F6 tasks.

## 2c. The novelty pillar: predicting new-pairing failure from trained tasks (user choice, 2026-11-16)

After the full-text reads the user picked a failure predictor to carry the novelty. RAA-min stays as an ablation: restriction alone (R2), offset alone (Rrel), and both (RAA), at matched data.

**Idea.** The diagnosis says that a policy fails a new pairing when its sub-goal follows a trained cue instead of the instruction's role. Measure that on trained tasks only, with no new-pairing rollout, and predict held-out success across every policy we have.

**Probe: steerability by counterfactual instruction.** The probe is open-loop. It is a side query that is never executed, inside closed-loop rollouts of trained tasks K.
- **Grasp steerability g_s.** Take each query before the first hold. Re-query with the instruction's object word swapped to another movable object present in the scene. Score the share of the action chunk's endpoint displacement that moves toward the swapped object, against the trained one: a signed projection in [−1, 1], mapped to [0, 1].
- **Place steerability p_s.** Take each query during the hold, before release. Swap the target word to another target present in the scene, and score the same way toward the new target.
- The probe passes through any front end or wrapper the policy uses: a resolver, role views, or ReGuide. So it scores the whole system as deployed.
- **Prediction with no fitting:** ŝ = g_s × p_s. This is the theory's product form s(o, t) = g(o) · p(t), with "competence" read as following the role.

**Registered tests.**

Goal, 2026-11-16: "Want to establish a failure atlas, design a failure predictor for provided tasks, and then find ways to repair VLAs with an adaptation to the architecture". So the primary test is per task.
- **T0, primary.** Take a provided new task (o, t) and a policy. Predict its success from g_s(o) and p_s(t). Measure g_s(o) on trained tasks whose prompt is swapped to name o; measure p_s(t) on trained tasks whose prompt is swapped to name t. The prediction is ŝ(o, t) = g_s(o) · p_s(t).
  - Score with AUC, using the cell's measured success ≥ .5 as the label, and with Spearman over all (policy, new task) cells in both scenes.
  - Pass if the AUC lower bound is > .5 and the AUC is ≥ .75.
  - The cells are bootstrapped jointly by task and by policy.
- **T1.** Spearman ρ between ŝ and measured new-pairing success across all policies in scene 1. The probe is run fresh on K tasks; success is re-read or taken from existing sections on the same task set. Pass if ρ ≥ .6 and the bootstrap lower bound is > 0.
- **T2.** Leave-one-model-family-out. Fit a one-parameter calibration on the other families and check its error on the held-out family.
- **T3.** Scene 2, where the cue is position. Use p_s from the plate-word swap on K1/K2 against N1/N2 success. The diagnosis predicts p_s ≈ 0 for every input-only front end, and p_s high for ReGuide and Hp.
- **Baselines for the predictor:**
  - trained-task success alone, which the diagnosis predicts does not rank new-pairing success;
  - a language-sensitivity score that swaps to a nonsense word (LIBERO-PRO style).

**Additions from the novelty check (reports/novelty_predictor.md, abstract level):**
- **Wrapper-free subset.** Report T0 and T1 on the wrapper-free subset as the headline. ReGuide, Hp and RC inherit steerability by design, so they appear only in the full set.
- **Within-family ρ.** Report ρ inside each family as well as across families. T2, leave-one-family-out, is co-primary.
- **Validity control.** Swap to an object or target word that is absent from the scene. A valid probe gives about 0 displacement toward any entity. This separates steering from a detector reflex.
- **Prior art to cite:**
  - the product form: Okawa et al. 2310.09336;
  - action sensitivity as a diagnostic: STAGE 2609.13458 and LIBERO-PRO;
  - policy ranking by world-model evaluators, e.g. Veo 2512.10675. We compare cost; we do not compare accuracy.
- **Scope of the claim:** pick-place object-target families only.

**Policies:**
- frozen pi0.5, GR00T, pi0, X-VLA and SmolVLA;
- the front ends tv, tvp and tvpb;
- the ladder: R0, R1, R2, R2b, R2w and r2l, then Rrel and RAA when trained;
- the wrappers: ReGuide rgp, Hp, and RC.

That is about 15-20 policies, with roughly 5 model families. Fewer servable policies narrow T1; the prereg fixes the list before any probe runs.

**Result (SR §74, m3_prereg steer, 2026-11-17).** The probe ran on 16 scene-1 and 3 scene-2 policy arms (10 trained-task states per task; the wrappers rgp, Hp, RC were not probed because their controllers take over the episode; X-VLA has no target; Rrel, RAA later). T0 headline on the wrapper-free subset (24 cells, 4 policies): AUC .550 [.000, 1.000], Spearman .227 [-.440, .812], NOT SUPPORTED. On all 90 scene-1 cells: AUC .742 [.526, .907], Spearman .442 [.125, .642], under the .75 bar by .008, and above trained-task success alone (.645) and a nonsense-word score (.462). T1 across 15 policies rho .193 [-.424, .736], NOT SUPPORTED. T2 leave-one-family-out beats the constant in 4 of 4 families, by .011 MAE on the pi0.5 family. T3: pooled plate steering .35 / .42 / .72 for pi0.5 I / pi0 I / tvp, not <= .25, NOT SUPPORTED. Intact policies are steerable in the place phase (.33-.55), so F1 is not visible as p_s near 0; the absent-word control is within .15 of the true prompt for all intact policies, and the angle rule has a .22-.28 floor on the true prompt. g_s and p_s depend heavily on which entity is swapped in. Consistent with a predictor that ranks front-end and rung variants of one base, not one that ranks unrelated frozen models from four systems.

**Why it is new (to check):** no prior paper predicts held-out-pairing success across many VLA systems from a trained-task probe. Grant et al. steer activations, and LIBERO-PRO perturbs language, but neither predicts. Novelty check queued.

## 3. Order

1. Novelty check of RAA (Sonnet, web) and an inventory of the retrain infrastructure (Sonnet, read-only), in parallel.
2. My design decision. Then the RAA prereg and implementation (Sonnet, from the ladder templates).
3. The paper rewrite, starting with the diagnosis sections, which do not depend on RAA.
4. A2 probes, then A4 scene 3.

**Exploratory follow-up (5fec75f7, post-hoc, results/flawed/steer_v2_explore.json).**
- Subtracting the true-prompt floor (V-a) raises all-90 AUC from .742 to .770. Multiplying in trained success (k×a) reaches .787 and Spearman .579. Both gains are inside v1's CI.
- Entity normalisation lowers all-90 AUC to about .60–.65. So v1's pooled signal is mostly entity difficulty: the bottle and bowl tasks are hard for every policy.
- On the wrapper-free four, no variant beats trained success alone (.800).
- Decision: the predictor stays a reported negative. The paper's claim is that steerability tracks which entities a family can be steered to, not which policies generalise.
- Planned: one confirmatory test of k×a on new policies (Rrel and RAA, probed in §73 phase 2). Its prereg row must be written before that probe runs.
- **Result (SR §75, 2026-11-17): k×a confirmed on new policies.** C1 AUC on the 12 Rrel/RAA × N1-N6 cells .944 (PASS, CI [.667, 1.0]); C2 pooled AUC over 102 cells .805, lower bound .603 (PASS); C3 sign test PASS (one pair). Trained-task success alone gives AUC .50 on these cells. Still two policies, six tasks, and one scene.

## Venue (user decision, 2026-11-21)

Target **ICML 2027**: abstract registration Jan 14, 2027 and full paper Jan 22, 2027 (AoE), per icml.cc/Conferences/2027 quotas page and mldeadlines; confirm against the official CFP. Format: 8 pages of main text plus an unlimited appendix, ICML LaTeX style. Fallback is CoRL 2027 (deadline expected late May 2027), which fits after ICML decisions in early May. Not RSS 2027 (stage-1 review overlaps Jan 22, so it would be a dual submission).

Framing for ML reviewers: a study of how VLAs compute where to act (failure atlas, probes, module removal, training-data routes). The paper has no real-robot claims.

User, 2026-11-21: "I'd like a lot of illustrations for each of the failures as well as architecture and other stuff". Chosen style: sim frames from rollouts (start / grasp / release, gripper path overlaid) plus a small schematic of the cue that drives the choice. Figure plan: docs/notes/fig_plan.md.

## Reframe (2026-11-22)

The user approved the analysis-paper framing proposed in docs/notes/novelty_review_adv.md (Section 3) and the two extra arms: a token-only oracle arm and a pi0 replication (plan to be written as docs/notes/plan_oracle_pi0.md; the file does not exist yet), plus the second training seed of RAA-sL already running.

- New thesis: behaviour-cloned VLA experts do not take up explicit geometry, so where the object goes must be owned outside the learned network. Evidence order: failure atlas, matched-data ladder, dose-by-servo grid (central result), W5 and ReGuide convergence as corroboration, moved plate, separable modular decomposition.
- paper/DRAFT.md is rewritten (main text about 5,600 words incl. tables; appendix A-I holds every removed result). The predictor moved to Appendix G; the W1-W6 chronology is one table (Table C1); SR pointers and capitalised verdicts are appendix-only.
- Archived draft: docs/history/DRAFT_2026-11-22_pre_reframe.md (verbatim).
- Open TODOs in the draft: Grant et al. version check (99.6% vs 0% transfer), read w2VLA in full, RoboMME and LIBERO-Mem citations, ReGuide plus resolver arm, targeted layer-14 patching test, F7 conflict with Wulff and Cangelosi.

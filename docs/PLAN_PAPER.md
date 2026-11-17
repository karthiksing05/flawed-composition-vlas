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
- **Later:** scene 2, where position is the cue, and the F7/F6 tasks.

## 3. Order

1. Novelty check of RAA (Sonnet, web) and an inventory of the retrain infrastructure (Sonnet, read-only), in parallel.
2. My design decision. Then the RAA prereg and implementation (Sonnet, from the ladder templates).
3. The paper rewrite, starting with the diagnosis sections, which do not depend on RAA.
4. A2 probes, then A4 scene 3.

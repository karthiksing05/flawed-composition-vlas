# [Working title] Routed by Cues: Architecture-Rooted Failures of Vision-Language-Action Models and a Role-Addressed Fix

Skeleton, 2026-11-16. The previous draft (a different project) is at docs/history/DRAFT_2026-10-21_units.md. Plan: docs/PLAN_PAPER.md. Numbers come from docs/SCAFFOLD_RESULTS.md (SR), docs/ARCH_FAILURES.md and docs/ARCHITECTURE_PROPOSAL.md; each claim below names its sections. Prose follows the user's paper rules: active voice, no "X, not Y" constructions, one idea per paragraph, contributions written as a paragraph.

## Abstract (write last)

The claim in one line: current VLAs choose their sub-goals by visual cues that covaried with those sub-goals in training. The instruction's roles do not choose them. This one routing defect explains five failures that persist across five models. A policy that addresses its actions by role removes them.

## 1 Introduction

- VLAs succeed on trained tasks and fail on new pairings of known parts (CRAFT, LIBERO-PRO, LIBERO-CF).
- Prior work either names the failure or patches it. It does not say which part of the architecture causes it.
- Criteria for "architecture-rooted":
  - C1: the information is in the input, and a reference planner succeeds.
  - C2: the failure persists across models.
  - C3: a structural change that adds no information removes it.
- Contributions paragraph:
  - the failure account (F1, F1m, F2, F6, F7) on up to five CoRL-era VLAs and two scenes;
  - the routing rule in scene 2 (pre-registered majority);
  - the prediction that input-only fixes fail where the routing cue is task-relevant, confirmed;
  - [RAA] a role-addressed architecture that removes the failures at matched data.

## 2 Related work

- Compositional generalisation in VLAs.
- Test-time wrappers: ReGuide, Harness VLA, LiLo-VLA, VP-VLA.
- Data fixes: CRAFT, ACT-VLA, AC-VLA.
- Mechanistic studies: Grant et al.
- Memory in VLAs: MEM, pi0.7.

Source: docs/NOVELTY.md.

## 3 Setting and criteria

- Models: pi0.5, pi0, GR00T N1.x, X-VLA, SmolVLA.
- Scenes:
  - LIBERO-goal descendant, 11 tasks;
  - LIBERO-10 LIVING_ROOM_SCENE5.
- New pairings, held-out states, the reference planner, and the testability floor.
- Pre-registration and paired bootstrap.

## 4 Failures (the account)

| failure | claim | sources |
|---|---|---|
| F1 | The place follows a cue: 4 of 5 models in scene 1; 0 of 720 in scene 2. | SR §45.x, §50; ARCH_FAILURES |
| | In scene 2 the cue is decided by a majority of {nearer, salient, trained} (.912 of 445). | §57-§61 |
| F1m | The held object's outline picks the carry. The box gives +.188, and GR00T's carry is overdetermined. | §51, §53, §56, §71 |
| F2 | The habit grasp is the failure on pi0 and SmolVLA (68%, 83% of their failures). It survives a put-back (240/240). | §65, §64 |
| F7 | Relational and negated language is ignored: the relational choice equals the ambiguous choice on 92-100% of states. | §45.1, §45.3, §70 |
| F6 | One view, then a cover, costs .38 and .60. | §45.5, §45.6 |
| limit | New layouts fail every system, including the planner (C1 fails). We report this as a task limit. | ARCH_FAILURES |
| [A2] | Mechanistic probe per failure. | to run |

## 5 What fixes them, and what does not

- Input-only fixes (views, boxes, phrases):
  - they work where the cue is task-irrelevant (scene 1, pi0.5: .938, §55);
  - they fail where it is task-relevant (scene 2, §50, §66) and on F2-limited models (§65).
- Fixes that act on the transport work everywhere tested:
  - ReGuide-class: scene 1 on pi0.5, pi0, SmolVLA and GR00T (§52, §55, §65, §71);
  - scene 2 on pi0.5 and pi0 (§68, §69).
- F7 needs a resolver; F6 needs a memory slot (§70).
- N1 residual: [§72 pending].

## 6 [RAA] Role-addressed action

Design, matched-data test against R0, CRAFT-style data and ReGuide. To come, depending on the novelty check and the build.

## 7 Conclusion and limitations

- Two LIBERO scenes.
- GR00T below its floor in scene 2.
- X-VLA has no fix.
- The fixes still need depth and calibration.

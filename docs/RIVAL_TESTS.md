# Rival-cause tests (PLAN_PARTS §10 item 18)

These tests serve the claim the user chose on 2026-11-04: compare rival causes of new-pairing failure in real VLAs, on the way to a solution that resolves composition within and across chunks. All are post hoc and development reads. Nothing here changes an earlier verdict.

## 1. V1: phase-switched instruction on pi0.5 (design fixed 2026-11-04, before any code or data; m3_prereg tag pp_rival1)

### 1.1 The question

pi0.5 keeps new pairings' grasps fairly well and loses the place (familiar layout, F3's intact read: g 0.95, p 0.51). Does it place when, once it holds the object, it receives an instruction it was trained on?

The test needs no training and no intervention inside the model. Only the prompt changes, so its reading does not depend on a purification being clean.

### 1.2 Arms

The policy is the frozen pi0.5 checkpoint, served as in F3's intact arm with the same sampler and seeds. "Held" is the oracle contact signal, used here only to time the switch. The prompt follows the current hold state, so a dropped object returns to the pre-hold prompt.

| arm | before the hold | while held | runs on |
|---|---|---|---|
| intact | the new pairing's instruction | the same | both layouts |
| A, primary | the new pairing's instruction | the known instruction sharing the target | both layouts |
| S, full switch | the known instruction sharing the object | the known instruction sharing the target | both layouts |
| W, wrong target | the new pairing's instruction | a known instruction with another target | familiar only |

Partners: cheese K6, bottle K4 before the hold. Plate K1, stove K2, cabinet K3, bowl K6 while held. W uses K2 when the true target is the plate, and K1 otherwise. A's episodes match intact until the switch, so A does not change who grasps.

**Episodes.** N1-N6, N2 main states 4-23, N2's seeds: 20 per task and layout. Rough cost: 840 episodes, about 5 A40-h, which is more than my earlier estimate of 2-3 because this uses twice F3's states and four arms.

### 1.3 Pre-registration

**Estimators**, familiar layout, pooled over N1-N6. g = success or the instructed object moved first; p = successes over g, as F3's read. CIs are the paired bootstrap over tasks and states (2,000 draws).
- `Δp_A`: p in A minus p in intact.
- `Δsteer`: among grasped episodes, the share whose object ends within 0.10 m horizontally of W's target, in W minus in intact.

**Two calls:**
- RECOVERS iff `Δp_A` has its CI lower bound above 0 and exceeds 0.10.
- STEERS iff `Δsteer` has its CI lower bound above 0 and exceeds 0.10.

**Verdict:**

| RECOVERS | STEERS | verdict | reading | direction for the solution |
|---|---|---|---|---|
| yes | yes | **INSTRUCTION BINDING** | the place skill exists and language reaches it. The failure lies in how a new instruction binds | compose at the language or conditioning side, across chunks |
| no | yes | **HELD-OBJECT BINDING** | the instruction steers the place, but a known place instruction does not work with this held object. This fits a vision-language-action binding inside the chunk | compose inside the chunk |
| no | no | **NOT THE INSTRUCTION** | while held, the prompt does not control the place. This points at the visual state or a memorised program | look outside language |
| yes | no | **RECOVERS WITHOUT STEERING** | ambiguous: the prompt helps with its own target yet does not redirect | no direction from this test |

**Reported with no verdict:** the new layout under the same rule; S's success against intact, which asks whether phase switching alone solves the whole task; per-task results; agreement of the intact arm with N2's recorded episodes.

**Prediction (mine, before data; my record here is poor).** INSTRUCTION BINDING, led by the cheese pairings, with the bottle pairings already high in intact.

**What this cannot show.**
- A gain could be a replay of a memorised program, as published work suggests. W's steering helps here only partly.
- Six tasks on one scene, one model. GR00T does not run in this pass.
- The switch uses oracle contact. A solution would need its own phase signal.
- At 20 states, the bootstrap over six tasks will be wide.

## 2. L1: library floors (design fixed 2026-11-04, before any code or data; same prereg tag)

ADVERSARIAL_STUDY §3 asks what the library adds. These run on CPU, with no training.

**L1a, no part (k = 0).** Take P_min's k = 1 creation checkpoints, set the created atom's slot embedding and low-rank delta to zero, and run the same S3 tasks, states and seeds: bottle K4, N4, N5, N6; plate K1, N1, N4; both layouts. For each atom and layout, `Δ` is k = 1 success minus k = 0 success, pooled, paired state bootstrap.
- PART ADDS iff the CI lower bound is above 0 and Δ > 0.10.
- Otherwise BACKBONE SUFFICES.

The start radius, target radius and m stay at the k = 1 values, so this removes the part's weights and keeps its fitted numbers.

**L1b, scripted servo.** The v2 servoing planner (`pp_n4d.plan_episode(servo=True)`) on P_min's evaluation states, K1-K6 and N1-N6, both layouts, with three planner seeds. It uses demo-fitted keypoints: known tasks use their own, and a new pairing combines the grasp of the known task sharing its object with the carry and release of the known task sharing its target. So it is itself a scripted composition. For new pairings and for known pairings in each layout, `Δ` is P_min's success minus the servo's, paired state bootstrap.
- P_MIN BEATS SERVO iff the CI lower bound is above 0 and Δ > 0.10.
- SERVO BEATS P_MIN iff the CI upper bound is below 0 and Δ < −0.10.
- Otherwise TIED.

**Prediction (mine, before data).** L1a: PART ADDS for the plate and the bottle in the familiar layout. L1b: SERVO BEATS P_MIN in the new layout and TIED in the familiar one.

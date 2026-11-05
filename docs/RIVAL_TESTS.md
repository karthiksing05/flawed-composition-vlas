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

**L1b, scripted servo.** The v2 servoing planner (`pp_n4d.plan_episode(servo=True)`) on P_min's evaluation states, K1-K6 and N1-N6, both layouts, with three planner seeds. It uses demo-fitted keypoints: known tasks use their own, and a new pairing applies `pp_n4d.keypoints`' existing rule for N4 to every new pairing: the object-sharing known task's grasp, carry and release rotation, with release offset = its grasp offset + (the target-sharing task's release offset − its grasp offset). So it is itself a scripted composition. For new pairings and for known pairings in each layout, `Δ` is P_min's success minus the servo's, paired state bootstrap.
- P_MIN BEATS SERVO iff the CI lower bound is above 0 and Δ > 0.10.
- SERVO BEATS P_MIN iff the CI upper bound is below 0 and Δ < −0.10.
- Otherwise TIED.

**Prediction (mine, before data).** L1a: PART ADDS for the plate and the bottle in the familiar layout. L1b: SERVO BEATS P_MIN in the new layout and TIED in the familiar one.

## 3. Results (2026-11-04; registered in commit 8fc4764e; results/flawed/pp_v1_read.json, pp_l1_read.json)

### 3.1 V1: HELD-OBJECT BINDING

840 episodes, no errors. Arm W's job was preempted twice and requeued. Its finished tasks were kept, and the interrupted task reran from its start. The intact arm agrees with N2's recorded episodes on success in all 240 episodes.

| layout | arm | s | g | p |
|---|---|---|---|---|
| familiar | intact | 0.50 | 0.97 | 0.52 |
| familiar | A | 0.53 | 0.97 | 0.55 |
| familiar | S | 0.53 | 0.98 | 0.55 |
| familiar | W | 0.00 | 0.96 | 0.00 |
| new | intact | 0.30 | 0.64 | 0.47 |
| new | A | 0.30 | 0.63 | 0.47 |
| new | S | 0.27 | 0.65 | 0.41 |

- **RECOVERS: no.** Δp_A = +0.03 [−0.16, +0.23] in the familiar layout. The interval is wide and still contains a gain above the 0.10 threshold, so the call is "no recovery shown". It does not establish zero. New layout: +0.01 [−0.11, +0.15].
- **STEERS: yes.** Δsteer = +0.67 [+0.45, +0.86]. A known instruction naming another target moves the held object to that target.
- **Verdict: HELD-OBJECT BINDING.** My prediction, INSTRUCTION BINDING, was wrong.
- **S against intact, no verdict.** Familiar Δs = +0.03 [−0.27, +0.30]. New −0.03 [−0.16, +0.07]. Phase switching alone does not solve the task.

Per-task familiar success, no verdict:

| arm | N1 | N2 | N3 | N4 | N5 | N6 |
|---|---|---|---|---|---|---|
| intact | 0.10 | 0.20 | 0.60 | 0.85 | 0.85 | 0.40 |
| A | 0.15 | 0.50 | 0.30 | 0.80 | 0.95 | 0.50 |
| S | 0.30 | 0.20 | 0.05 | 0.75 | 1.00 | 0.90 |

The pooled null hides opposing moves: N2 gains under A, N3 loses under both, N6 gains under S. Twenty states per task cannot separate these from noise, and I did not register a per-task call.

**A correction to how I have quoted this model.** The pi0.5 figure "0.99 to 0.38, g 0.79, p 0.47" in ADVERSARIAL_STUDY is N2's "new" cell over N1-N8, which includes the drawer and push tasks. On the six pick-and-place pairings tested here, intact is s 0.50, g 0.97, p 0.52. The bottle on the plate and the stove already reach 0.85. The failure concentrates in N1 and N2 (cheese on plate, on stove).

### 3.2 L1a: PART ADDS, both atoms, both layouts

| atom | layout | k = 0 | k = 1 | Δ [95% CI] |
|---|---|---|---|---|
| bottle | familiar | 0.04 | 0.83 | +0.78 [+0.64, +0.91] |
| bottle | new | 0.05 | 0.38 | +0.33 [+0.19, +0.47] |
| plate | familiar | 0.56 | 0.94 | +0.38 [+0.28, +0.47] |
| plate | new | 0.23 | 0.44 | +0.21 [+0.13, +0.30] |

My prediction held for the familiar layout and the mapping also returns PART ADDS for the new one. The zeroed checkpoint keeps the part's fitted numbers (start radius, target radius, m), so k = 0 is a floor on the weights alone. On the plate, the backbone with those numbers already reaches 0.56 in the familiar layout.

### 3.3 L1b: mixed; my prediction was half right

| pairings | layout | servo | P_min | Δ [95% CI] | verdict |
|---|---|---|---|---|---|
| known | familiar | 0.83 | 0.90 | +0.07 [+0.01, +0.13] | TIED |
| known | new | 0.67 | 0.46 | −0.21 [−0.27, −0.14] | SERVO BEATS P_MIN |
| new | familiar | 0.62 | 0.92 | +0.29 [+0.21, +0.38] | P_MIN BEATS SERVO |
| new | new | 0.31 | 0.37 | +0.06 [−0.03, +0.14] | TIED |

Two tasks carry two of these calls. These exclusions are post hoc and change no verdict.
- **K5.** Known/new: K5 is servo 0.85 against P_min 0.00. Without K5, Δ is −0.08.
- **N3.** The servo gets 0.12 in both layouts, against P_min's 1.00 and 0.42. A composed-keypoint fault in the servo is possible for this pairing; I have not diagnosed it. Without N3, new/familiar is still +0.18 and new/new is +0.01.

**The fact to carry forward.** The scripted planner reads exact object and target positions and has no learned binding. It still falls from 0.62 to 0.31 on new pairings when the layout moves, and from 0.83 to 0.67 on known ones. It has no obstacle avoidance, and I have not traced why its episodes fail. So some of the new-layout gap is not evidence about binding at all, and P_min's new-layout scores should be read against this reference rather than against 1.0.

## 4. V1 arrival read (design fixed 2026-11-04, after §3, before this read; m3_prereg tag pp_rival2)

HELD-OBJECT BINDING leaves one question that decides the solution's shape: when pi0.5 grasps and fails, does the object reach the target and fail to seat, or does it never arrive? V1's track files already hold the final positions. No new episodes.

**Estimator.** Intact arm, familiar layout, pooled N1-N6, episodes with g = 1 and no success. `arrive` is the share whose object ends within 0.10 m horizontally of the true target. The CI is the bootstrap over tasks and states, 2,000 draws.

**Mapping.**
- ARRIVES, FAILS TO SEAT iff the CI lower bound is above 0.5.
- DOES NOT ARRIVE iff the CI upper bound is below 0.5.
- MIXED otherwise.

**Reported with no verdict.** The same share for A; the new layout; the share within 0.03 m (the success check's horizontal radius); the share still held at the end; per-task values.

**Prediction (mine).** MIXED, with the bottle's failures arriving and the cheese's not.

**What this cannot show.** A final position is one instant. An object that passed the target and was carried away counts as not arriving. The 0.10 m radius is V1's steering radius; the plate is about that wide, so an object resting on the plate's rim counts as arrived.

### 4.1 Result: MIXED, with the direction opposite to my prediction

Registered in m3_prereg `pp_rival2`; results/flawed/pp_v1_arrive.json. Intact, familiar: 56 grasped failures, `arrive` 0.55 [0.16, 0.81]; within 0.03 m 0.13; still held at the end 0.05. Arm A: 52 failures, `arrive` 0.67 [0.32, 0.95].

| intact, familiar | N1 | N2 | N3 | N4 | N5 | N6 |
|---|---|---|---|---|---|---|
| grasped failures | 18 | 15 | 5 | 3 | 3 | 12 |
| within 0.10 m | 14 | 11 | 4 | 0 | 0 | 2 |
| within 0.03 m | 6 | 1 | 0 | 0 | 0 | 0 |

- **Cheese:** 29 of 38 failures end within 0.10 m of the target and 7 within 0.03 m. The cheese arrives and misses by a few centimetres.
- **Bottle:** 2 of 18 failures arrive. N4 and N5 rarely fail in the familiar layout. N6's 12 failures end elsewhere.
- **New layout:** intact, 41 failures, `arrive` 0.24 [0.03, 0.50], none within 0.03 m. The two layouts fail differently: in the new layout the object mostly does not get there.

The per-object split is descriptive. It was reported with no verdict, and I read it after the pooled call.

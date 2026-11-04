# Place diagnostic: does placing depend on the object-target pair? (PLAN_PARTS §10 item 15)

The user decided this on 2026-11-04, after seeing the coverage arm's descriptive table (COVERAGE_RESULTS §3.5). It is post hoc and
labelled so everywhere. It is a diagnostic: it builds no model, changes no criterion and does not change any GO or ladder verdict.
Every number is a development read: seeds 10-12 and N2 states 4-23. No confirmatory or sealed read is run, and no demo is executed.

## 1. Design note (fixed 2026-11-04, before any code or data of this diagnostic)

### 1.1 The question

3F predicts a new pairing as hold(object) × arrival(object) × place-on-arrival(target), so it assumes a target has one place rate.
COVERAGE_RESULTS §3.5 shows new-layout place rates that differ by held object, in every arm:

| target | object | task | P_min | P_cov | P_cov_h |
|---|---|---|---|---|---|
| plate | bowl | K1 | .46 | .61 | .59 |
| plate | cheese | N1 | .84 | .91 | .94 |
| plate | bottle | N4 | .57 | .78 | .71 |
| bowl | cheese | K6 | .93 | .91 | .79 |
| bowl | bottle | N6 | .70 | .65 | .63 |

### 1.2 Three facts from the code, which shape the test

1. **P_min's target term has no input that names the held object.** With `tsize` off (P_min's setting), `PartsFlow._canon` feeds the
   target term the foot (the held object's position with its rest height removed), the last command, the *target's* hand-set size,
   the drawer opening and the rotation since the hold's onset. `_tscale` adds the distance at the hold's onset.
   - **Correction.** My description to the user on 2026-11-04 said the target term reads the held object's size, and proposed a probe
     that swaps it. That is wrong for P_min: the size it reads is the target's. The swap probe is replaced by §1.3 test T3.
   - **Consequence.** The object can change the target term's command only through those states. Two objects with equal inputs get
     equal commands. So "the target term treats objects differently" is not a separate explanation here: it is a difference in state.
2. **The success check is one rule with a narrow tolerance.** K1 and K6 use LIBERO's `On` (N1, N4 and N6 are registered to be
   confirmed by T4), and `ObjectState.check_ontop` requires the target's z ≤ the object's z, contact between the two bodies, and their
   body origins within **0.03 m** horizontally. An episode ends the step the check passes. A bowl and a cheese land differently within
   that same rule.
3. **Arrival uses a per-pair radius.** `radius(o, t) = max(RAD[t], calibrated mean dxy + 0.03)`. So two pairs at one target can differ in
   who counts as arrived, and the place rates in §1.1 could differ partly for that reason.

### 1.3 The tests

**T0: failure taxonomy, from saved rollouts (no simulation).** For P_min, P_cov and P_cov_h, every K1-K6 and N1-N6 task in both
layouts, each episode that arrived is classed by its saved fields: success; still held at the end (`held_end`); final horizontal
distance object-target ≥ 0.03 m; otherwise (within 0.03 m and no success). Also `held_segments` > 1 and the radii used. Descriptive.

**T1: traced re-run.** P_min, tasks K1, N1, N4, K6, N6, both layouts, seeds 10-12, states 4-23, the same seeds (60 episodes per task,
600 in all). `fa1h_roll.episode` is unchanged; `env.step` is wrapped to read the state after each step. No input to the policy changes.
Rollouts are not bit-reproducible across hosts (PARTS_RESULTS), so agreement with the stored outcomes is reported, and T1's own
outcomes are what the verdict uses.
- **Release state.** The first step with an open command (`a[6] < 0`) at or after the hold first reaches the radius. The state read
  is the one just before that command: foot distance d and foot height h from the target's origin, the object's tilt, and the target's
  displacement since the episode's start. That is the state the policy opened at, before the outcome.
- **No open command** is its own stratum.

**T2: matched release states.** For each contrast, place rates are standardised over release strata:
`none` (no open command); `tilted` (tilt > 15°); else d in [0, 0.015), [0.015, 0.03), [0.03, ∞) × h below or at/above the median
release h pooled over the contrast's two tasks (median fixed without looking at outcomes). This gives at most 8 strata. Weights are
the pooled share of arrived episodes; strata with no episode in one task are dropped and their mass is reported as lost.

**T3: exactness check for fact 1.** On each P_min seed, for held rows from the demo batch: set the other objects' slots equal to the
held object's, change `oi` and compare the velocity. The check passes iff the difference is exactly 0 in every case.

**T4: drop map, same state with no hand.** No policy. Each of the five pairs, both layouts, states 4-7. The object is placed by the
simulator upright at its starting orientation, with zero velocity, with its foot at d ∈ {0, 0.01, 0.02, 0.03, 0.04} m (8 bearings for
d > 0) from the target's origin. h ∈ {0.01, 0.04} m for the plate and {0.05, 0.08} m for the bowl, which brackets the demos' release
heights in COVERAGE_RESULTS §1.2. It then steps 60 times with zero motion and an open gripper, recording the check. Drops where a
finger touches the object are excluded, and their share is reported. T4 also logs each task's goal predicate.

### 1.4 What each outcome means

| outcome | reading | for the composition claim |
|---|---|---|
| NOISE | the raw gap's CI contains 0 | no gap established for this contrast |
| ARRIVAL-DEFINITION | the gap goes away with a common radius | an estimator artefact of per-pair radii |
| RELEASE-STATE | objects open at different states, and those explain the gap | the shared target term gives object-dependent releases: an interface question (the foot alone does not suffice) |
| SAME-STATE: PHYSICS | the gap survives matching, and the drop map shows it with no hand | the same state gives different outcomes: 3F needs a pair-level term at the place, and the target term is not at fault |
| SAME-STATE: UNMEASURED | the gap survives matching, and the drop map does not show it | something the strata and the drop map miss, such as the fingers opening or the object term after release |
| UNDETERMINED | under half the mass overlaps | the tasks' release states barely overlap, so matching cannot decide |

T1's cost is CPU only. T4 is CPU only and does not use a checkpoint. No GPU is used.

## 2. Pre-registration (fixed 2026-11-04, before any T0-T4 run; m3_prereg tag pp_place)

**Contrasts** (first minus second): plate, N1 − K1 and N4 − K1; bowl, N6 − K6. Each in each layout. The new layout is where the gap was
seen, and those three verdicts are the headline. Familiar-layout verdicts come from the same rule and are reported beside them.

**Estimators**, all on T1's episodes:
- `G`: place-on-arrival difference, each task with its own pair radius (the frozen estimator's arrival). Its CI is the paired state
  bootstrap over the 20 states (2,000 draws, generator seed 0, seeds kept).
- `G_c`: the same with both tasks using the smaller of the contrast's two radii.
- `A`: the standardised difference (T2), with arrival at the common radius.
- `D`: drop-map success, first task minus second, over cells with d ≤ 0.02 m, all states and heights in that layout.

**Verdict per contrast, first match wins:**
1. CI of `G` contains 0: **NOISE**.
2. CI of `G_c` contains 0: **ARRIVAL-DEFINITION**.
3. Overlapping strata hold less than half of the pooled arrived episodes: **UNDETERMINED**.
4. |A| ≤ 0.5 |G_c|: **RELEASE-STATE**.
5. Otherwise, `D` has the sign of `G_c` and |D| ≥ 0.10: **SAME-STATE: PHYSICS**. Else **SAME-STATE: UNMEASURED**.

**T3 gate.** If T3 is not exact, the diagnostic stops, since fact 1 is then false and §1.4's readings don't hold. It is reported as such.

**Reported with no verdict:** T0; A's bootstrap CI (cells are small, so A is noisy); outcome agreement with the stored rollouts;
release-state distributions per task; and, per PLAN_PARTS §10.14, P_min's m-conditioned 3F beside the frozen one, labelled post-hoc.

**Prediction (mine, before data).** New layout: N1 − K1 is SAME-STATE: PHYSICS, as a wide bowl can rest on the plate's rim outside 3 cm
while a small cheese settles toward the centre. N4 − K1 is RELEASE-STATE, given the bottle's history of tilting and touching down
early. N6 − K6 is SAME-STATE: PHYSICS, as a tall bottle in a bowl can tip.

**Amendment 1 (2026-11-04, after T3's first run and T0, before the final read).** T3's first version included the bowl in its list of
object slots. At K6, where the bowl is the target, it overwrote the target's position with the held object's, and its control then moved
both together, so the control showed no change. Every swap difference was exactly 0, on K1's rows and on K6's degenerate rows, but K6's
check was not informative. The corrected check never touches the target's slot, compares only non-target objects, and writes
`pp_place_t3_a1.json`, which the read now uses for the gate. The control must also change the velocity. The first output stays in place.
The contrasts, estimators and verdict rule are unchanged. T0 and the last 7 lines of one K1 trace log were seen before this amendment.

**What this cannot show.**
- The strata are coarse. A SAME-STATE verdict means these strata did not explain the gap, not that no state could.
- Standardising on a state reached during the episode describes association within the rollouts. It is not a controlled intervention.
  T4 is the only intervention here, and it has no hand.
- Each target has only a few partners, so a verdict covers these pairs and not objects in general.

## 3. Results (2026-11-04; m3_prereg pp_place; results/flawed/pp_place_read.json, pp_place_t0.json, pp_place_t3_a1.json; pp_place_trace, pp_place_drop)

### 3.1 Summary

- **The verdicts.** New layout: N1 − K1 and N6 − K6 are SAME-STATE: UNMEASURED; N4 − K1 is NOISE. All three of my predictions were wrong.
- **Ruled out for these pairs.** The target term cannot tell objects apart (T3 exact). Per-pair radii aren't the cause, as each contrast's two
  radii are equal. Landing physics at a matched pose isn't the cause either: inside 2 cm, with no hand, the drop map gives 1.00 for almost every cell.
- **Not established.** What does cause the gaps. The registered matching left most of the gap in place, but it was weak (§3.4), so the state
  reached in the hand remains an untested candidate.
- **Scope.** No GO, ladder or coverage verdict changes.

### 3.2 Verdicts

| layout | contrast | G [95% CI] | G_c | A [95% CI] | overlap | D | verdict |
|---|---|---|---|---|---|---|---|
| new | N1 − K1 | +0.33 [0.04, 0.58] | +0.33 | +0.25 [−0.13, 0.62] | 1.00 | 0.00 | SAME-STATE: UNMEASURED |
| new | N4 − K1 | +0.08 [−0.21, 0.38] | +0.08 | +0.16 [−0.08, 0.40] | 0.97 | 0.00 | NOISE |
| new | N6 − K6 | −0.23 [−0.50, −0.00] | −0.23 | −0.13 [−0.41, 0.21] | 0.91 | −0.03 | SAME-STATE: UNMEASURED |
| familiar | N1 − K1 | −0.02 [−0.05, 0.00] | −0.02 | 0.00 [0.00, 0.00] | 0.95 | 0.00 | NOISE |
| familiar | N4 − K1 | −0.16 [−0.31, −0.04] | −0.16 | −0.07 [−0.13, 0.00] | 0.91 | 0.00 | RELEASE-STATE |
| familiar | N6 − K6 | −0.13 [−0.24, −0.04] | −0.13 | −0.06 [−0.14, 0.03] | 0.92 | −0.03 | RELEASE-STATE |

- N6 − K6's upper CI bound in the new layout is −0.004, so that gap only just clears NOISE. Both new-layout A intervals contain 0.
- T3 (amendment 1): every swap difference is exactly 0 on all seeds and both tasks, and the control changes the velocity on all of them.
- T1's outcomes match the stored rollouts in 95-100% of episodes per task.

### 3.3 What the failures look like

T1's place-on-arrival, with arrived failures split by how they end. "Median tilt" is at the open command, among episodes that have one.

| layout | task | object → target | arrived | place | still held at the end | ended ≥ 3 cm away | ended < 3 cm away | no open command | median tilt |
|---|---|---|---|---|---|---|---|---|---|
| new | K1 | bowl → plate | 45 | 0.51 | 13 | 9 | 0 | 22 | 24° |
| new | N1 | cheese → plate | 25 | 0.84 | 3 | 1 | 0 | 19 | 10° |
| new | N4 | bottle → plate | 29 | 0.59 | 0 | 12 | 0 | 10 | 36° |
| new | K6 | cheese → bowl | 29 | 0.93 | 0 | 2 | 0 | 4 | 11° |
| new | N6 | bottle → bowl | 27 | 0.70 | 0 | 8 | 0 | 0 | 13° |
| familiar | K1 | bowl → plate | 60 | 1.00 | 0 | 0 | 0 | 55 | 10° |
| familiar | N1 | cheese → plate | 59 | 0.98 | 0 | 1 | 0 | 55 | 13° |
| familiar | N4 | bottle → plate | 57 | 0.84 | 0 | 9 | 0 | 36 | 14° |
| familiar | K6 | cheese → bowl | 60 | 0.98 | 0 | 1 | 0 | 6 | 8° |
| familiar | N6 | bottle → bowl | 56 | 0.86 | 0 | 8 | 0 | 0 | 8° |

Descriptive readings, post hoc:
- **No failure in this table ended within 3 cm.** The contact and height parts of the check are not what fails here.
- **Bowl on plate is a new-layout problem.** It is 60 of 60 in the familiar layout. In the new layout 13 of its 22 failures are still holding
  the bowl at the end, and K1's trace log includes a bowl at 175° of tilt, 33 cm above the plate.
- **The bottle falls short in both layouts.** Every N4 and N6 failure ends away from the target and none is still held, and most are still in
  contact with it (familiar: 9 of 9 and 8 of 8). So the two shortfalls look like different problems.
- **P_cov and P_cov_h show the same split in T0:** K1 new has 15 still held in each, and no target-tolerance failures to speak of.

### 3.4 Drop map, and a weakness in the registered matching

**Drop map.** Inside 2 cm every object succeeds in every cell, with one exception: the bottle over the bowl at d = 0, h = 0.05 m fails at all states
(so N6's rate is 0.97). Excluded drops: 0%. Beyond the registered cells the objects do differ, for example at d = 0.03 m: bowl on plate 0.25 at the lower
height and 0 at the higher, cheese on plate 0.91-1.00, bottle on plate 1.00. The drop has no hand and starts upright, so it says nothing about a
tilted or still-gripped object.

**Weakness in the matching.** I defined the release state at the first open command. In many episodes there is none, because the check passes on
touch-down with the gripper still closed: 55 of 60 in K1 familiar. Those episodes share the `none` stratum with episodes that never let go. In the new
layout, `none` succeeds in 11 of K1's 22 episodes against 17 of N1's 19, so most of N1 − K1's gap sits inside one mixed stratum. SAME-STATE here therefore
means "the registered strata did not explain it", and is weaker evidence than its name suggests.

### 3.5 3F with m conditioning (PLAN_PARTS §10.14; post-hoc, no verdict; P_min)

| layout | | N1 | N2 | N3 | N4 | N5 | N6 |
|---|---|---|---|---|---|---|---|
| new | frozen | +0.129 | −0.063 | −0.067 | +0.164 | +0.276 | +0.107 |
| new | m-conditioned | +0.129 | −0.063 | −0.067 | +0.061 | +0.068 | −0.102 |
| familiar | frozen | −0.033 | 0.000 | 0.000 | −0.033 | +0.133 | +0.011 |
| familiar | m-conditioned | −0.033 | 0.000 | 0.000 | −0.167 | 0.000 | −0.118 |

Intervals are in pp_place_t0.json. The frozen column reproduces the published values, and the frozen estimator still decides every criterion.
- **N5, the frozen estimator's one familiar failure, goes to 0.000.** So that failure comes from pooling the bottle's hold across grasp modes.
- **Familiar N4 and N6 now miss**, with intervals excluding 0 (N4 [−0.317, −0.050], N6 [−0.234, −0.021]). In the frozen estimator K5's low rates pull the
  bottle's prediction down, which offsets the bottle's place shortfall, so these two may pass the frozen 3F partly because two errors cancel.

### 3.6 What this does and does not change

- **The law's factorisation at the place remains open.** The bottle's shortfall at both targets is what a pair term would capture, but the cause isn't known.
- **A testable candidate, not a result:** the target term sees the foot and the hand's rotation since onset, not the object's own tilt or its pose in the hand. K1 new and
  N4 new have the largest tilts in §3.3. An in-hand-state test would need its own registration, and should drop the open-command event in favour of
  states along the carry.

## 4. In-hand test: design note (fixed 2026-11-04, before any code or data of this test; PLAN_PARTS §10 item 16)

The user approved this next step after reading §3. It is post hoc, because §3's tilts suggested it, and §3 was seen before its thresholds were set. It builds no model and
changes no criterion or verdict. Dev states and seeds only, CPU only.

**The question.** The target term sees the foot and the hand's rotation since onset, not the object's own tilt. Does the tilt a held object has reached by arrival explain
the place gaps, and do the target term's own rotation commands cause it?

**The event.** The first step the hold reaches the pair's radius (the frozen estimator's arrival). Unlike §1.3's open command, every arrived episode has it, and it comes
before the descent and before any contact with the target.

**Tilt.** The angle between the object's body z axis at arrival and at the episode's start, which removes each object's resting orientation. §3.3's tilt was absolute.

**I1: traced re-run.** As T1 with a richer trace: P_min, K1, N1, N4, K6, N6, both layouts, seeds 10-12, states 4-23.

**I2: intervention (arm `norot`).** The same episodes, with the commanded rotation `a[3:6]` set to 0 at every step where the oracle hold is on. This only asks whether those commands cause
the tilt and the failures. It is a diagnostic and not a candidate fix, and adopting anything like it would be the user's decision. Only the rotation is changed, and only while held.

**Reported with no verdict.** Where the tilt comes from: tilt at the hold's onset, the hand's rotation from onset to arrival, and the object's slip relative to the hand over the same span. Also the object's
offset shift in the hand; place rate by tilt stratum within each task; the state at the end of K1's still-held failures; agreement with T1's and the stored outcomes. This test computes no 3F; §3.5's
frozen and m-conditioned values stand.

### 4.1 Pre-registration (fixed 2026-11-04, before any I1 or I2 run; m3_prereg tag pp_inhand)

**Contrasts** as §2: N1 − K1, N4 − K1, N6 − K6, in each layout. The new layout is the headline.

**Estimators**, on arrived episodes:
- `G`: place-on-arrival difference in the base arm, with the paired state bootstrap of §2 (2,000 draws, generator seed 0). Each contrast's two radii are equal.
- `A`: the difference standardised over tilt at arrival, strata [0, 15), [15, 30), [30, ∞) degrees, weighted and dropped as in §1.3.
- `Δ`: for the contrast's lower-place task in the base arm, place-on-arrival in `norot` minus base, same bootstrap.

**Observational verdict per contrast, first match wins:**
1. CI of `G` contains 0: **NO GAP**.
2. Overlapping strata hold less than half of the pooled arrived episodes: **UNDETERMINED**.
3. |A| ≤ 0.5 |G|: **TILT-MEDIATED**.
4. Otherwise: **NOT EXPLAINED BY TILT**.

**Intervention verdict per contrast:** CI lower bound of `Δ` above 0 and Δ > 0.10 (the rounds' "improves" rule): **ROTATION-CAUSAL**. Otherwise **NO EFFECT**.

**What each pair means:**

| observational | intervention | reading |
|---|---|---|
| TILT-MEDIATED | ROTATION-CAUSAL | the target term's own rotation commands tilt the object and cost the place. A question for the target part (condition P or S), not for the law |
| TILT-MEDIATED | NO EFFECT | tilt goes with the gap but these commands don't drive it: look at the grasp or at slip. The interface has no such input (condition S) |
| NOT EXPLAINED BY TILT | either | tilt at arrival isn't the answer for this contrast. ROTATION-CAUSAL would still show those commands matter by another route |
| NO GAP | either | nothing to explain in this run; the intervention's effect on that task is reported |

**Prediction (mine, before data; my last three were wrong).** New layout: N1 − K1 is TILT-MEDIATED with NO EFFECT, N4 − K1 is NO GAP, N6 − K6 is NOT EXPLAINED BY TILT, since §3.3's tilts for N6 and K6 are close.

**What this cannot show.**
- Standardising is association only. `norot` is the one intervention, and it removes all held rotation, not just the harmful part.
- With about 25-45 arrived episodes per task, A's interval will be wide, and the rule reads its point estimate.
- If `norot` changes who arrives, the Δ groups differ. Arrival and total success are reported beside it.
- Tilt is one scalar. A NOT EXPLAINED verdict leaves other in-hand quantities open.

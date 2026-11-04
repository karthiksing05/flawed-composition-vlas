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

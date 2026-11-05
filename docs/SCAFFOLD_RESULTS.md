# Scaffold architecture: protocols and results

Plan: `docs/PLAN_SCAFFOLD.md`. The user approved the build on 2026-11-05, and I told them T5 stays off until they lift the occupancy rule. All reads here are development reads. N2 main states 4-23 have shaped earlier work, so no result here is confirmatory.

## 1. T0: recorded tracks (protocol fixed 2026-11-05, before the read; m3_prereg sc_t03)

`experiments/sc_t0.py`. CPU, atlas tracks only, no rollouts. Systems: pi0.5, GR00T, P_min seeds 10-12, servo. Tasks: N1-N6, K1-K4, K6 (the atlas has no K5). Episodes replan every 5 actions.

### 1.1 Read A: do fatal gripper events line up with query boundaries?

A query comes before actions 0, 5, 10, and so on. Phase = action index mod 5. For an episode with a hold, its event is the first closed-to-open command after the hold's onset.

- **Fatal set:** N1-N6 episodes the atlas labels HANDOFF, TRANSPORT or RELEASE.
- **Control set:** successful episodes, all tasks of that layout and system.
- **Statistic:** Δ = share of fatal events at phase 0 − the same share in the control set. CI by resampling states, 2000 draws.

**Verdict per system and layout, first match.** TOO FEW iff either set holds under 10 events. BOUNDARY iff the CI's lower bound is above 0 and Δ > 0.10. Otherwise NOT AT BOUNDARIES. The familiar cells of pi0.5 and GR00T are the primary ones, because the atlas's breaks live there.

Reported with no verdict: phase histograms, the phase of the hold's end, and hand jerk by phase, free and held.

**What it decides.** BOUNDARY on either VLA makes T1 a priority and raises the rival that variable chunking fixes a VLA without schemas. NOT AT BOUNDARIES says nothing about other chunk lengths, and T1 is still needed for that.

**Prediction (mine; my record is poor).** NOT AT BOUNDARIES on both VLAs.

### 1.2 Read B: where does the new layout's loss sit?

Four cumulative gates per episode. A later gate passing counts as passing the earlier ones.

| gate | passes iff | timescale |
|---|---|---|
| reach | the hand comes within 0.05 m of object + `grasp_off` before any hold (Read E's `a_t`) | long |
| hold | any hold | short |
| arrive | object within 0.10 m of the target during a hold (the atlas's ARRIVE) | long |
| place | success | short |

Each gate's conditional rate uses (k + 0.5) / (n + 1), pooled over tasks, seeds and states. L_gate = log rate(new) − log rate(familiar). **Long share** = (L_reach + L_arrive) / sum of all four. CI by resampling tasks (the same draw in both layouts) and states (independently per layout), 2000 draws.

**Verdict per system, pooled over the 11 tasks.** LONG iff the share is at least 2/3 and the CI's lower bound is at least 0.5. SHORT iff the share is at most 1/3 and the CI's upper bound is at most 0.5. Otherwise MIXED. P_min is the primary system, because the design builds on it.

Reported with no verdict: the share with a 0.07 m reach, known and new groups separately, and each gate's rate.

**What it decides.** LONG supports PLAN_SCAFFOLD §3.3's hypothesis. SHORT counts against it, and T3 decides. MIXED means the split into transit and contact can only recover part of the gap.

**Limit.** Reach uses each task's demo keypoint, and a VLA gripping elsewhere reads as not reaching (HAND_OVER 6.2). Hold is conditional on reach, so a reach that the hand cannot finish because of a fixture counts as short here.

**Prediction.** MIXED on P_min, with the long share between .4 and .7.

### 1.3 Read C: clearance, descriptive

New layout, per object and system: hold and success by the object's start distance to the nearest of stove, cabinet and rack, in bins below .10, .10-.20 and at least .20 m. These are site positions and ignore the fixtures' shapes. No verdict.

## 2. T3: are contact parts layout-invariant? (protocol fixed 2026-11-05, before any dev rollout; m3_prereg sc_t03)

`experiments/sc_t3.py`. P_min, seeds 10-12, oracle state, oracle contact, the same 300-step policy budget, K1-K6 and N1-N6, both layouts, dev states 4-23 with their seeds. 2,880 episodes. Each arm replaces the part of the episode outside what it tests with the servo (`pp_n4d.plan_episode`'s controller, phases and keypoints), which reads simulator positions.

| arm | the servo does | P_min does | measure |
|---|---|---|---|
| A | moves the hand to the pre-pose: object + `grasp_off` + 0.10 m up, grasp rotation | everything from there | hold, among reached |
| B | grasps, lifts and carries to above the release point at carry height | everything from there | success, among reached |

**Reached.** In A, the hand ends within 0.015 m and 0.10 rad of the pre-pose within 120 steps. In B, when the carry phase converges or times out at 120 steps, the oracle reads a hold and the object lies within 0.10 m of the target (the atlas's ARRIVE). The smoke states showed why B does not need pose convergence: on familiar N4 the hand stalled about 0.05 m short with the bottle held, and the planner's own timeout tolerates that. I changed this rule after the smoke and before registration. The servo takes no account of obstacles. An unreached state counts toward the reach rate, and P_min does not run on it. The seed's rng draws the servo's noise and offset jitter, as in `pp_l1_floor`.

**Contrasts.** D_A = hold(new) − hold(familiar) in A. D_B = success(new) − success(familiar) in B. CI by resampling tasks (same draw) and states (per layout), 2000 draws.

**Verdict, first match.**
1. INSTRUMENT FAILS iff familiar hold in A or familiar success in B is below 0.70. P_min then does not accept the servo's states, and T3 says nothing about layouts.
2. CONTACT FAILS iff either contrast's CI lies wholly below −0.10.
3. CONTACT INVARIANT iff both CI lower bounds are at or above −0.10.
4. UNRESOLVED otherwise.

Reported with no verdict: reach rates, success in A, unconditional success, intact P_min, known and new groups, per task, and states near (under .10 m) and far from a fixture.

**What it decides.** PLAN_SCAFFOLD's stop rule: CONTACT FAILS means the layout hypothesis as written is dead, and I report to the user before more layout work. CONTACT INVARIANT lets T4 split contact from transit. A reach rate that falls in the new layout is itself the transit gap.

**Limits.**
- Conditioning on reached drops blocked states, so invariance holds only among states a straight servo can reach. The reach rates and the near and far strata show the size of this.
- The servo's approach and carry differ from a demo's, so a low familiar rate would indict the hand-over. The 0.70 floor catches that.
- B's success still includes the descent and release, which is what the place parts do. It excludes everything the arrival state does to the object under P_min's own carry, so a gap between B and P_min's place on arrival (PLACE_DIAG 3.3) points at the arrival state.
- Fixtures do not move and yaw stays constant, so this tests moved objects in one kitchen.

**Prediction.** CONTACT FAILS on A because of near-fixture descents, with D_B inside the margin. Familiar rates clear the floor.

**Equivalence check, before any verdict.** `sc_t3.py equiv`: mode none on dev states 4-7, K6 familiar and N1 new, seed 10, against the atlas's P_min outcomes. It exercises the copied episode. Disagreement on more than 1 of 8 stops T3 until I explain it. **Result: 8 of 8 agree.**

### 1.4 T0 results (2026-11-05; results/flawed/sc_t0.json)

**Read A: NOT AT BOUNDARIES on every system and layout.** My prediction held.

| familiar, N1-N6 | fatal events | Δ share at phase 0 | CI |
|---|---|---|---|
| pi0.5 | 57 | −.03 | [−.18, +.13] |
| GR00T | 75 | −.13 | [−.32, +.08] |
| P_min | 19 | +.04 | [−.09, +.20] |

- The intervals are wide, so an effect of about +.10 stays possible on pi0.5.
- GR00T opens at phase 0 on 53% of fatal events and 66% of controls. It tends to open at a query, whether or not the episode then fails.
- Hand jerk is higher in phases 0-1 than 2-4 on both VLAs and P_min, by roughly 1.2 to 2 times, and flat on the servo. So query boundaries are visible in motion, without being where fatal openings concentrate.

This reads only the current 5-action replan. T1 still owns how failures change with chunk length.

**Read B: MIXED on every system.** My predicted range for P_min (.4-.7) just missed.

| system | long share | CI | L_reach | L_hold | L_arrive | L_place |
|---|---|---|---|---|---|---|
| P_min | .37 | [.21, .55] | −.16 | −.28 | −.11 | −.19 |
| pi0.5 | .60 | [.37, .88] | −.27 | −.21 | −.21 | −.10 |
| GR00T | .74 | [.44, 1.15] | −.54 | −.31 | −.47 | −.05 |
| servo | .52 | [.13, 1.01] | −.22 | −.13 | −.02 | −.10 |

On P_min, the short-timescale gates carry most of the loss. The VLAs lean long. With a 0.07 m reach, P_min's share is .34.

**Read C, descriptive.** Cheese hold rises steeply with clearance on all systems: P_min 0 (under .10 m, n = 12), .36 (.10-.20, n = 156), .94 (at least .20, n = 72). Bottle hold stays at .50 on P_min and the servo in both populated bins, so clearance to these sites does not explain it. Site positions ignore fixture shapes.

## 3. S1: the lexicon's refusal (protocol fixed 2026-11-05, before the read; m3_prereg sc_lex)

`experiments/sc_lex.py`, CPU, no rollouts. `lifelong_ood/scaffold/stores.py`'s `Lexicon` keeps L4's rule and refuses a parse when a token is unknown or its referent is still ambiguous. Situations come from demos through L1's event, as in L4. Given roles score the parses and nothing else. This test uses the instructions that shaped L4's rule, so it is weak evidence about language in general.

Three reads: the full lexicon; leaving each known task out; lexicons from one demo per task (50 indices). Read tasks: K1-K6, N1-N6.

**Verdict, first match.** OVER-REFUSES iff the full lexicon is exact on fewer than 12 read tasks. SILENT WRONG iff any leave-one-out or one-demo lexicon gives a wrong parse on a read task. Otherwise REFUSES SAFELY.

**Prediction.** REFUSES SAFELY on the full lexicon and most leave-one-out cases, with SILENT WRONG surviving somewhere, because a token grounded to the wrong slot passes both refusal checks.

**Result: REFUSES SAFELY.** My prediction failed. `results/flawed/sc_lex.json`.

| lexicon | read tasks exact | refused | wrong |
|---|---|---|---|
| full | 12 | 0 | 0 |
| without K1 / K2 | 9 / 9 | 3 / 3 | 0 |
| without K3 | 12 | 0 | 0 |
| without K4 | 7 | 5 | 0 |
| without K5 | 11 | 1 | 0 |
| without K6 | 7 | 5 | 0 |
| without K7 | 12 | 0 | 0 |
| one demo per task, 50 indices | mean 12.0 | 0 | 0 |

Refusal removes L4's silent wrong parses, including K4's own and K7's wrong target. K8 and N8 refuse on the full lexicon, because push has no event and so gives no situation. That keeps a known gap visible, and it means this lexicon cannot yet ground a push. The evidence is narrow: 16 instructions with few templates, which shaped L4's rule.

## 4. T3 results (2026-11-05; results/flawed/sc_t3_read.json; rollouts results/flawed/sc_t3)

**Registered verdict: UNRESOLVED.** Equivalence 8 of 8. Familiar rates clear the floor (.91 and .85). My prediction failed on A's label, because A's interval overlaps the margin.

| contrast | familiar | new | D | CI |
|---|---|---|---|---|
| A: hold from the pre-pose | .91 | .79 | −.12 | [−.25, −.02] |
| B: success from the servo's carry | .85 | .85 | −.00 | [−.10, +.09] |
| A: success (reported) | .82 | .52 | −.30 | [−.44, −.15] |

| | familiar | new |
|---|---|---|
| A reached | 1.00 | .79 |
| B reached | .76 | .51 |
| A unconditional success | .82 | .41 |
| B unconditional success | .64 | .43 |
| intact P_min | .91 | .42 |

**Readings, post hoc.**
- **Oracle transit to the pre-pose alone does not raise new-layout success.** A's unconditional success (.41) matches intact (.42). The transit half of PLAN_SCAFFOLD §3.3's hypothesis, as a route to recovering the new layout, gets no support here.
- **Pre-pose reach fails by object.** Cheese .62 and bottle .75 in the new layout; bowl 1.00. The straight servo cannot reach about 8 of 20 cheese states. HAND_OVER 7.4 saw the hand stop on the rack at about this height, so an obstacle-aware path to the same pose may not exist either.
- **Hold from the pre-pose drops by object:** bowl about .94 from 1.00 (K1-K3), cheese .81 from .95, bottle .58 from .82, while K5's bottle mode rises to .92.
- **The strongest signal lies between grasp and arrival.** Having held, P_min's own continuation loses much of the new layout, and the servo's does not: K1 from the pre-pose scores hold .97, success .28; from the servo's carry .93. PLACE_DIAG 3.1 left the state reached in the hand as an untested candidate, and this supports it.

**Flaws I found in this read.**
1. **The near and far strata are uninformative.** My `clear` measure uses site positions, so almost no state counts as near, and those states never reach. Near reads NaN and far repeats the whole. FC 5.29's rack clearance differs from this measure.
2. **The task resampling overstates independence.** The object term ignores the target, so hold is exactly equal across tasks sharing an object and a mode (.81 across K6 and N1-N3; .58 across K4 and N4-N6). That confirms locality, and it makes 12 tasks closer to 4 groups for A's hold. The known and new groups mainly compare object mixes.
3. **B's composed keypoints are flawed on new pairings.** N3 never reaches in either layout (B reached 0.00), so it does not enter D_B. N4 scores .02 and .08 from B against .78 intact in the familiar layout. `pp_l1_floor.kp_for` keeps the object task's carry height and release rotation. I suspect this carries the cheese into the cabinet and takes the bottle out of the plate part's support. I have not confirmed the cause. D_B compares layouts, so a defect in both does not move it much, but B's pooled familiar rate rests on these cells.

The verdict stands as registered. T3 did not fail, so the stop rule does not fire. It also did not establish invariance.

## 5. T3b: who owns the loss after the pre-pose? (protocol fixed 2026-11-05, before any dev rollout; m3_prereg sc_t3b)

T3 suggests the new-layout loss follows what happens between the grasp and the arrival. `experiments/sc_t3b.py` separates grasp from carry. Every episode starts from T3's pre-pose. P_min, seeds 10-12, K1-K6 and N1-N6, both layouts, dev states 4-23. 5,760 episodes.

| cell | grasp by | carry by | then |
|---|---|---|---|
| PP | P_min | P_min | P_min to the end (T3's A, the equivalence check) |
| PS | P_min | servo | P_min from above the release point |
| SP | servo | P_min | P_min from the hold's onset |
| SS | servo | servo | P_min from above the release point |

- **Grasp by servo:** T3's phases 1-3 until the oracle reads a hold. Then the carry owner takes over. P_min's steps total at most 300.
- **Carry by servo:** phases 3-4, 120 steps each at most, and then P_min runs regardless. No arrival filter.
- **Keypoint fix, new pairings only:** carry height = the target task's + (object task's `grasp_off` z − target task's `grasp_off` z). So the carried object rides at about the height the target's demos use. Known tasks keep their own. Checked on smoke states before registration.
- **Logged at arrival within 0.10 m and at the first open command:** object tilt and object-to-target distance. Also rack clearance at the start.

**Primary outcome: success given a hold, among episodes that reached the pre-pose, new layout, pooled.** This asks what the continuation does with an object already in the hand.

- G = s|hold(SP) − s|hold(PP): does a servo grasp help when P_min still carries?
- C = s|hold(PS) − s|hold(PP): does a servo carry help when P_min still grasps?

CI by resampling the 4 object groups (bowl, cheese, bottle in K4 and new pairings, bottle in K5) and the states, 2000 draws. This corrects T3's flaw 2. A group's tasks are kept together.

**Verdict, first match.** Registered effect iff the CI's lower bound is above 0 and the estimate exceeds 0.10.
- INSTRUMENT FAILS iff PP disagrees with T3's A on more than 5% of episodes.
- BOTH iff both G and C are registered effects.
- CARRY OWNS IT iff only C is.
- GRASP OWNS IT iff only G is.
- NEITHER otherwise.

Reported with no verdict: unconditional success, hold rate, the familiar layout's G and C, tilt at arrival and release, per object, and rack strata.

**What it decides.** CARRY OWNS IT says the architecture needs a shared, object-indifferent carry, the long-timescale store, even though the pre-pose transit recovered nothing. GRASP OWNS IT says the object part's grip, read at contact, is the fault, and the short-timescale parts must change. BOTH or NEITHER leaves the split without support, and I report that before building T4.

**Limit.** With four object groups, the CI has few independent units, and the bootstrap will be rough. Servo carries use oracle positions.

**Prediction.** CARRY OWNS IT, with C about +.25 and G within the margin. My record here is poor.

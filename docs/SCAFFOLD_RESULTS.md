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

- **Grasp by servo:** T3's phases 1-3 until the oracle reads a hold. Then the carry owner takes over. P_min's steps total at most 300. If the servo's grasp fails, P_min still runs, and a later pick-up by P_min does not count as a hold in that cell's primary outcome (added after the smoke, before registration).
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

## 6. T3b results (2026-11-05; results/flawed/sc_t3b_read.json; rollouts results/flawed/sc_t3b)

**Registered verdict: NEITHER.** My prediction failed. PP disagrees with T3's A on 0.2% of episodes. As registered, this leaves the split into contact and transit without support, and I report to the user before T4.

| new layout, success given an owned hold | estimate | CI |
|---|---|---|
| G: servo grasp, P_min carries | +.09 | [−.01, +.17] |
| C: P_min grasps, servo carries | +.07 | [−.05, +.28] |
| SS against PP (reported) | +.19 | [+.02, +.42] |

With four object groups, the CI has few independent units.

| cell | familiar: hold | s given hold | all episodes | new: reached | hold | s given hold | all episodes |
|---|---|---|---|---|---|---|---|
| PP | .91 | .90 | .82 | .79 | .79 | .65 | .40 |
| PS | .91 | .84 | .76 | .79 | .79 | .72 | .45 |
| SP | .85 | .93 | .87 | .79 | .78 | .74 | .52 |
| SS | .86 | .94 | .88 | .79 | .78 | .84 | .58 |

Familiar reach is 1.00 in every cell. Intact P_min scores .91 and .42.

**Readings, post hoc, on development states of one kitchen.**

| new layout, all episodes | PP | PS | SP | SS | what limits it |
|---|---|---|---|---|---|
| bowl, K1-K3 | .61 | .61 | .77 | .83 | the grasp, then the carry: s given hold .65, .64, .81, .88 |
| cheese, K6 N1-N3 | .39 | .43 | .43 | .44 | getting a hold: reach .62, and hold .81 (P_min) or .65 (servo). No owner helps |
| bottle, K4 N4-N6 | .36 | .34 | .54 | .52 | P_min's grasp: hold .58 against the servo's .73, with reach .75 |
| bottle side-on, K5 | .02 | .47 | .07 | .65 | the carry: P_min's own carry almost never arrives |

- **The new layout has no single owner.** Each object group loses at a different place, and pooling hides it. A split by timescale does not sort them, and three of the four causes sit at or before the grasp.
- **Oracle control of everything before the final descent reaches .58 against .88 familiar.** In SS, reach (.79) and hold (.78 against .86) cost more than the continuation (.84 against .94). So on this hand, this servo and this fixed grasp pose, much of the gap comes before any policy question about carrying.
- **The cheese cap may be physical.** No state within 0.10 m of the rack by site distance reaches the pre-pose, and cheese stays near .44 in every cell. FC 5.29 found that only 5 of 10 near layouts give the generator a grasp that holds, and HAND_OVER 7.4 found the hand stopping on the rack. No test yet separates a collision-free grasp that a policy could learn from no such grasp existing.
- **Tilt tracks the bowl's failure.** Bowl tilt at the first open command has median 31° in new PP and 17° in SS, against 14° and 13° familiar. It is an association only.
- **A servo carry can hurt in the familiar layout.** N6 falls .68 to .18 and N1 .95 to .65 under PS. A part does not always accept a state its demos never produced, which is a limit on swapping skills at a seam.
- **The design consequence.** PLAN_SCAFFOLD §3.3's hypothesis, as written, is not supported: oracle transit to the pre-pose recovers nothing alone (.40 against .42 intact), and neither oracle grasp nor oracle carry passes as the owner. I withdraw it as the basis for T4's split. The record does not refute a role for a shared carry, which K5 shows clearly, and it leaves the bottle and bowl grasps as measured targets.

## 7. F: how much of the layout gap is physically closable? (protocol fixed 2026-11-05, before any dev rollout; m3_prereg sc_feas)

The user approved this on 2026-11-05. `experiments/sc_feas.py`. No training. `fa1l_cgen.episode`, unchanged: a scripted oracle-state planner with eight grasp options (yaw 0, 90, ±45; tilt 25° and 40° away from the nearest fixture). Objects on their own known tasks: bowl K1, cheese K6, bottle K4, both layouts, dev states 4-23, 3 noise seeds per option, 2,880 episodes. Grasp options are for measurement only, and nothing here enters any data file. The tilt's direction uses simulator positions and hand-set sizes. The approach is still a straight line.

**Per state:** an option counts if it holds through the lift in at least 2 of 3 noise seeds. A hold means the planner's `held_lift` or a full-task success, because the smoke showed `held_lift`, sampled once, reading false on episodes that succeed (changed after the smoke, before registration). `hold_any` = share of states with some counting option. `hold_base` = the same with the fixed pose (0, 0) alone. Gain = hold_any − hold_base, CI over states, 2000 draws.

**Verdict, first match.** INSTRUMENT FAILS iff familiar `hold_any` is below .90 on any object. GAIN iff some object's new-layout gain is at least .10 with the CI's lower bound above 0. Otherwise NO GAIN.

**Reported with no verdict:** full-task success under any option and the fixed pose, each option's hold rate, and the blocked states.

**What it decides.** GAIN opens T5 (entity-based obstacle terms under one rule) as the user allowed. NO GAIN closes it. `sc_feas_states.json` lists the new-layout states with a counting option, and those define the reachable set for the restated claim.

**Limits.** The options are not exhaustive and the path is straight, so hold_any is a lower bound on physical feasibility, and a blocked state may be open to some other approach. Three noise seeds per option are few.

**Prediction.** GAIN on the cheese, with `hold_any` about .75 against about .45 fixed; no gain on the bowl; familiar clears the floor, with the bottle at risk.

## 8. D: does the part inventory come out of the demos? (protocol fixed 2026-11-05, before the read; m3_prereg sc_disc)

The user's requirement: "learn parts in a very unsupervised and reusable way". `experiments/sc_disc.py`, GPU, no rollouts.

**Build U40.** `pp_p_train.py build --tag sc_u40 --steps 30000 --rank 32 --rtclip --psupport --tscale --decouple --rtransport --bbnom --bbnosize --pgate --iroles --nosize --demos 40`, seeds 10-12. L4's arm with three things removed: the hand-set sizes, the grasp message m, and demos 40-49 (kept for the decisions). The event's threshold still reads all 50 demos' widths. Starting with one candidate part per (role, moved or nearby slot), which comes from the demos through L4's sites. What it still assumes: the entity list and oracle positions.

**Decisions, all on held-out demos 40-49, with common noise, per seed.**
- **k = 0:** the part zeroed, to see what it adds at all.
- **Split gain:** refit one part on one of its tasks' training rows, 4000 steps, with everything else frozen. gain = (loss(shared) − loss(refit)) / loss(shared) on that task's held-out rows.
- **Noise floor:** the same refit for parts with only one task. They have nothing to split, so their gain is what refitting alone gives. Floor = the largest such gain that seed.
- **Split** iff a part with several tasks has gain above the floor and a demo-resampled CI above 0.
- **Merge cost:** use part p on part q's held-out rows, cost = (loss(p) − loss(q)) / loss(p). **Merge** iff the CI's upper bound is at or below the floor's magnitude.

A split or merge stands iff at least 2 of 3 seeds agree. Given roles and atom names score the result, and nothing else.

**Verdict on the inventory, first match.** UNDER-SPLITS iff no bottle object split stands (K4 or K5). OVER-SPLITS iff at least 2 splits stand on anything other than the bottle. Otherwise RECOVERS. Merges are reported with no verdict: a merge is what reusable parts should produce, and it needs a behavioural test.

**What it decides.** RECOVERS: the build's inventory replaces the hand-listed atoms and m, with one rule. UNDER-SPLITS: this rule misses the one pair dependence we know is real, so held-out flow loss is the wrong criterion for parts, and I report that before T4. OVER-SPLITS: parts do not come out as atoms by this rule, and that counts against sparse reuse.

**Next, registered only after this read.** T4: behavioural builds on all 50 demos from the discovered inventory; removal controls for k = 0, psupport, tscale, rtransport and rank; the reference is P_min, and reuse is S3 creation from held-out atoms under the same unsupervised recipe.

**Limits.** Flow loss averages over all rows, including far rows where a part is off, which dilutes gains. Low loss does not guarantee success. These instructions and demos shaped the earlier design. The refit noise floor is one heuristic among several, and six single-task parts make a small sample.

**Prediction.** UNDER-SPLITS is my main fear. I predict RECOVERS with the K5 bottle split at low margin, with few merges among objects and some among fixture targets.

## 9. F and D results (2026-11-05; results/flawed/sc_feas_read.json, sc_feas_states.json, sc_disc_read.json)

### 9.1 F: GAIN

Registered verdict: **GAIN**, through the bottle. My prediction named the wrong object.

| new layout | hold, fixed pose | hold, any option | gain | CI | success, fixed | success, any | blocked states |
|---|---|---|---|---|---|---|---|
| bowl | .90 | 1.00 | +.10 | [.00, +.25] | .55 | .95 | 0 of 20 |
| cheese | .40 | .40 | .00 | [.00, .00] | .40 | .40 | 12 of 20 |
| bottle | .60 | 1.00 | +.40 | [+.20, +.60] | .60 | 1.00 | 0 of 20 |

Familiar hold_any is 1.00 on every object, and the fixed pose already reaches it.

- **The cheese's limit may be physical, up to these eight straight-line options.** No option helps, and 12 of 20 states are blocked: 4, 5, 9, 10, 12, 13, 14, 15, 18, 19, 20, 22. So under the restated claim, reachable new-layout cheese states are 8 of 20 at this lower bound.
- **The bottle and bowl are physically open.** With the right yaw the scripted planner completes the bottle on all 20 states and the bowl on 19. Bottle yaw ±45 or 90 holds at .68-.78 against .57 at 0. Bowl success rises from .55 to .95, more than its hold does.
- **This opens T5** under the user's condition. Grasp options stay out of training.

A choice among poses has to read the scene, so a policy cannot get this gain with a fixed pose. Whether it needs obstacle terms, or a wider set of demos, stays open.

### 9.2 D: OVER-SPLITS

Registered verdict: **OVER-SPLITS**. My prediction failed. Splits standing in 2 of 3 seeds: bowl K1, bowl K7, bottle K4, bottle K5. No merges in any seed.

| refit gain (seeds 10, 11, 12) | |
|---|---|
| bowl K7 | +.041, +.047, +.039 |
| bottle K4 | +.025, +.024, +.024 |
| bowl K1 | +.020, +.021, +.020 |
| bowl K2 | +.013, +.017, +.017 (CI spans 0) |
| bowl K3 | +.011, +.014, +.008 (CI spans 0) |
| bottle K5 | +.014, +.013, +.014 |
| cabinet K3 | +.013, +.014, +.016 (CI spans 0) |
| cabinet K4 | −.029, −.034, −.034 |
| single-task parts (the floor) | −.02 to −.05 |

**Reading, post hoc.**
- **The criterion fails its one known case.** K5's 81° grasp is the pair dependence the record is surest of, and its gain is the smallest of the four splits. The rule does not separate it from K1, where nothing is known to differ. Flow loss averages over every row, and the turn occupies a small share of them.
- **The floor is badly designed.** It sits below zero, because a 4000-step refit does worse than the 30,000-step joint fit, so almost any positive gain clears it. I chose it, and it did not act as a noise estimate.
- **K7's split is believable.** Its object rows include opening the drawer, which is a different skill. That agrees with the sequencing evidence, and it is the one case where flow loss flags a real structural difference at a clear margin.
- **Parts are distinct and they matter.** No substitution comes within the noise on any pair. Zeroing a part costs target parts .53-.65 of their rows' loss and object parts .12-.22, on seeds 10-12.

**Consequence.**
- **Flow loss alone will not produce the inventory.** I do not patch the rule on these same demos. The slot-keyed inventory (one part per role and per slot that demos move or approach) stays, and it comes from L4's sites.
- **A correction to PLAN_SCAFFOLD's amendment.** I listed m as hand-given, and that overstated it. Its inventory and values are fitted from demos by `grasp_inventory`, with no labels. What was chosen by hand is the feature, the grasp's yaw turn, and I chose it after seeing K5. So m stays in T4 as a demo-fitted device with a removal control, and its form is disclosed as post hoc.

## 10. T4: the rebuild under the unsupervised recipe (protocol fixed 2026-11-05, before any non-smoke build; m3_prereg sc_t4)

`experiments/sc_t4.py`. The user approved T4 and asked for parts learned in a "very unsupervised and reusable way". Seeds 10-12, all 50 demos, 30,000 steps, K1-K7 by L4's rule (K8 has no event). Rollouts: K1-K6, N1-N6, both layouts, dev states 4-23, oracle positions. No transit split. No hold oracle or given roles enter the policy.

**Arm U.** L4's arm without hand-set sizes: `--rank 32 --rtclip --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize --pgate --iroles --nosize`. What it still assumes: the entity list and oracle positions, and the features chosen for m and the frames.

| removal | how | question |
|---|---|---|
| Unm | no `--imode` | does the demo-fitted grasp message earn its place? |
| Unp | no `--psupport` | per-part radii, never ablated |
| Unt | no `--tscale` | carry scale frame |
| Unr | no `--rtransport --bbnom --bbnosize` | role-level transport, never ablated |
| Ur8 | `--rank 8` | rank 32, never ablated |
| U0 | U's checkpoint with every part zeroed | do parts add anything over the backbones? (the plan's k = 0) |

**Verdicts.** Each cell is a layout × task group (known K1-K6, new pairings N1-N6). Differences are in success, with a CI by resampling tasks and states, paired, 2000 draws. Task resampling overstates independence where tasks share an object (T3's flaw 2), so these intervals run somewhat narrow.

- **V1, U against L4** (`pp_L4_roll/P2L4`, hand-set sizes kept), familiar layout. First match: COSTS iff either familiar cell's CI lies wholly below −.10. HOLDS iff both lower bounds are at or above −.10. Otherwise UNRESOLVED. New-layout cells are reported with the same labels and no decision.
- **V2, each removal against U.** KEEP iff in any of the four cells the difference is below −.10 and its CI's upper bound is below 0. Otherwise DROP (no measured loss). Six tests at 4 cells each, uncorrected, so a KEEP on one cell is weak.
- **V3, reuse.** Builds without the bottle (`--exclude K4+K5`) and without the plate (`--exclude K1`), then creation from k = 1, 5, 10 demos of K4 or K1. Cells: K4, N4, N5, N6 and K1, N1, N4. Difference = success under the created part − success under full U, by layout and k, familiar and new. TRANSFERS at k = 1 iff the familiar lower bounds are at or above −.10 on both atoms. Otherwise NOT YET. Old tasks' parameters stay bitwise unchanged, which the creation code asserts.

**Storage** (no verdict): `lifelong_ood/scaffold/stores.py`'s report for U, counting shared backbones.

**What it decides.** HOLDS means hand-set sizes can go, and the recipe's labels drop to the entity list and positions. DROP removes a device from the final store. KEEP keeps it, and that device needs a stated principle. TRANSFERS is the reuse claim for one object and one target. It says nothing about pixels, which T2 and T7 own.

**Limits.** Development states have shaped all of this. Fixed pose: T5 has not run, so new-layout cells carry F's gap. Cheese is capped at 8 of 20 states at F's lower bound. Two held-out atoms are few.

**Prediction.** V1 HOLDS. KEEP on Unm (K5), Unr and U0; DROP on Unp and Ur8; KEEP on Unt through N4. V3 TRANSFERS on the plate and NOT YET on the bottle.

**Smoke note.** The smoke changed the command separator from `+` to `~`, because `K4+K5` was being split, and removed LIBERO from the openpi venv's path, which had caused a silent exit 1. Both fixes came before the ledger row.

## 11. T4 results (2026-11-05; results/flawed/sc_t4_read.json; rollouts results/flawed/sc_t4)

All 673 tasks completed: 24 builds, 18 creations, 630 rollout files, 12,600 episodes, no episode errors. Every episode induced roles from the lexicon.

### 11.1 V1: HOLDS, at the edge

| U minus L4 | U | L4 | difference | CI |
|---|---|---|---|---|
| known, familiar | .872 | .872 | .000 | [−.056, +.056] |
| new pairings, familiar | .894 | .928 | −.033 | [−.100, +.028] |
| known, new layout | .433 | .433 | .000 | [−.089, +.095] |
| new pairings, new layout | .381 | .403 | −.022 | [−.089, +.039] |

The new-pairing lower bound sits exactly on the −.10 margin, and these intervals run somewhat narrow, so the pass is a narrow one. The two exact zeros are coincidences of the group mean: per-task values differ (K4 1.00 → .93, K5 .32 → .38). The rule removes hand-set sizes. The cost, if real, sits on the bottle: N4 .88 → .78 and N5 1.00 → .93.

### 11.2 V2: three devices stay, three go

| removal | verdict | cell that decided it | difference | CI | my prediction |
|---|---|---|---|---|---|
| U0, parts zeroed | KEEP | new pairings, familiar | −.725 | [−.903, −.519] | right |
| Unr, no role-level transport | KEEP | known, new layout | −.194 | [−.336, −.067] | right |
| Unm, no grasp message | KEEP | known, new layout | −.111 | [−.231, −.011] | right verdict, wrong reason |
| Unt, no carry scale | DROP | none | | | wrong |
| Unp, no per-part radii | DROP | none | | | right |
| Ur8, rank 8 | DROP | none | | | right |

Per-task success. Familiar layout:

| | L4 | U | Unm | Unp | Unt | Unr | Ur8 | U0 |
|---|---|---|---|---|---|---|---|---|
| K1 | .98 | 1.00 | .98 | 1.00 | 1.00 | .98 | .90 | .70 |
| K2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .98 | 1.00 |
| K3 | 1.00 | .98 | .98 | .95 | .98 | .97 | .98 | .95 |
| K4 | 1.00 | .93 | .75 | .93 | .93 | .88 | .98 | .42 |
| K5 | .32 | .38 | .42 | .32 | .40 | .63 | .42 | .00 |
| K6 | .93 | .93 | .95 | .93 | .97 | .93 | .80 | .03 |
| N1 | 1.00 | .98 | .98 | .98 | .88 | .97 | .90 | .03 |
| N2 | .95 | .98 | .98 | .98 | .98 | .98 | .88 | .07 |
| N3 | .98 | .98 | .98 | 1.00 | 1.00 | .98 | .87 | .07 |
| N4 | .88 | .78 | .58 | .85 | .47 | .85 | .87 | .30 |
| N5 | 1.00 | .93 | .72 | .95 | .93 | .88 | .98 | .43 |
| N6 | .75 | .70 | .55 | .70 | .70 | .73 | .78 | .12 |

New layout:

| | L4 | U | Unm | Unp | Unt | Unr | Ur8 | U0 |
|---|---|---|---|---|---|---|---|---|
| K1 | .43 | .35 | .25 | .37 | .27 | .23 | .50 | .28 |
| K2 | .55 | .63 | .53 | .58 | .75 | .33 | .63 | .55 |
| K3 | .65 | .77 | .52 | .62 | .73 | .37 | .68 | .65 |
| K4 | .55 | .47 | .27 | .48 | .50 | .18 | .40 | .27 |
| K5 | .00 | .00 | .00 | .00 | .13 | .00 | .00 | .00 |
| K6 | .42 | .38 | .37 | .38 | .42 | .32 | .30 | .00 |
| N1 | .30 | .32 | .32 | .30 | .40 | .28 | .33 | .00 |
| N2 | .33 | .28 | .37 | .32 | .35 | .30 | .37 | .02 |
| N3 | .48 | .50 | .43 | .52 | .52 | .42 | .32 | .03 |
| N4 | .33 | .32 | .17 | .28 | .23 | .13 | .27 | .12 |
| N5 | .60 | .50 | .27 | .50 | .52 | .18 | .40 | .32 |
| N6 | .37 | .37 | .20 | .35 | .33 | .10 | .23 | .07 |

**Reading, post hoc.**
- **Parts carry new pairings.** With every part zeroed, N1-N3 fall to .03-.07 and K6 to .03. This is the strongest removal result in the study: the sparse store is what makes cheese tasks and their recombinations work.
- **The backbone stores the bowl, against the sparse claim.** With every part zeroed, K2 stays at 1.00 and K3 at .95. New-layout K2 and K3 lose only .08 and .12. The majority object lives in the shared store, so the bowl part and these two target parts add little behaviourally. D found that zeroing target parts costs .53-.65 of flow loss, so the loss and the behaviour disagree here.
- **The grasp message earns its place on the upright bottle, and it does nothing for K5.** I predicted K5 would carry this verdict. K5 is .38 with m and .42 without. The loss falls on K4, N4, N5 and N6 instead, in both layouts. The device was motivated by the case it fails to help, so its principle still needs stating.
- **K5 remains unsolved.** It stays at .38 familiar and .00 new in U. Removing role-level transport raises it to .63 familiar, which is the largest gain from any removal and goes against that device's overall verdict. Transport helps new layouts generally (known .43 → .24 without it) and costs K5's familiar scenes.
- **Two DROPs hide task-level costs.** Rule V2 tests group means. Unt loses .31 on N4 familiar and Ur8 loses about .10 on N1-N3 and K6 familiar, offset elsewhere. Both verdicts stand as registered, and neither removal has been tested jointly with the others.

### 11.3 V3: NOT YET

| created minus full U, k = 1 | created | U | difference | CI |
|---|---|---|---|---|
| bottle, familiar (K4, N4, N5, N6) | .858 | .838 | +.021 | [−.129, +.175] |
| plate, familiar (K1, N1, N4) | .939 | .922 | +.017 | [−.061, +.133] |
| bottle, new layout | .388 | .413 | −.025 | [−.175, +.129] |
| plate, new layout | .350 | .328 | +.022 | [−.122, +.178] |

Success by demos, per cell:

| | U | k = 1 | k = 5 | k = 10 |
|---|---|---|---|---|
| K4 familiar | .93 | .90 | .85 | .82 |
| N4 familiar | .78 | .85 | .80 | .78 |
| N5 familiar | .93 | .88 | .85 | .82 |
| N6 familiar | .70 | .80 | .62 | .60 |
| K4 new | .47 | .43 | .67 | .62 |
| N4 new | .32 | .32 | .35 | .35 |
| N5 new | .50 | .45 | .67 | .62 |
| N6 new | .37 | .35 | .40 | .50 |
| K1 familiar | 1.00 | 1.00 | 1.00 | 1.00 |
| N1 familiar | .98 | .97 | .95 | .97 |
| N4 familiar (plate) | .78 | .85 | .90 | .92 |
| K1 new | .35 | .27 | .25 | .27 |
| N1 new | .32 | .33 | .27 | .32 |
| N4 new (plate) | .32 | .45 | .25 | .30 |

The plate passes. The bottle fails on interval width with a positive point estimate: its lower bound is −.129 against a −.10 margin, over 4 tasks. The registered label stands, and my prediction matched it.

**Limits on what this establishes.**
- **The control is missing.** V3 compares a created part with a full build. It never measures the same build with the created part switched off. U0 shows that K1, K2 and K3 survive without parts, so a created plate could pass while doing little. Crediting the part needs a k = 0 arm.
- **K5 is outside the bottle's cells.** The side grasp, the bottle's hardest use, never entered this test.
- **More demos do not help familiar scenes.** Bottle familiar falls from k = 1 to k = 10 on all four cells, and the same seeds do not rescue it. New-layout K4 and N5 rise to .67, above full U's .47-.50. A bottle part fitted to K4 alone may beat one shared with K5, which agrees with D's bottle split. This is a post-hoc hypothesis from one comparison.

### 11.4 Storage (registered, no verdict)

| | total parameters | one object part | one target part | backbones |
|---|---|---|---|---|
| U | 1,781,632 | 82,928 | 82,608 | 871,664 |
| Ur8 | 1,099,288 | 20,744 | 20,664 | 871,664 |

One part is 4.7% of U's total, and backbones are 49%. Adding an atom costs one part. Reaching that atom's cells needs the backbones already in place.

### 11.5 New layouts on F's reachable states (post hoc, under the user's restated claim)

| U, new layout | all 20 states | F-reachable states | F-blocked states |
|---|---|---|---|
| bowl tasks | .58 | .58 (20) | none |
| cheese tasks | .37 | .70 (8) | .15 (12) |
| bottle tasks | .41 | .41 (20) | none |
| pooled | | .53 | |

F's upper limit on reachable states is close to 1.00, against .53 here, so roughly half of the restated claim's gap remains. The policy succeeds on 15% of the "blocked" cheese states, so F's blocked list over-counts: it reflects the eight scripted options and these states are not physically impossible.

### 11.6 What T4 decides

Hand-set sizes go. Per-part radii, rank 32 and the carry scale frame go under the registered rule, pending a joint test because each was removed alone. Parts, role-level transport and the grasp message stay. The labels left in the recipe are the entity list, oracle positions, and the features chosen for m and the frames. New layouts stay the largest gap.

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
- **More demos do not help familiar scenes.** Bottle familiar falls from k = 1 to k = 10 on all four cells. New-layout K4 and N5 rise to .67, above full U's .47-.50. A bottle part fitted to K4 alone may beat one shared with K5, which agrees with D's bottle split. This is a post-hoc hypothesis from one comparison.

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

## 12. T5a and T4b (protocol fixed 2026-11-05, before any non-smoke rollout of T5a or any build of T4b; m3_prereg sc_t5)

### 12.1 Where new-layout episodes fail, and two checks that shaped T5a

U's failures, 20 states, 3 seeds:

| | success | no hold | hold, then fail | median hold step |
|---|---|---|---|---|
| bowl, familiar | .99 | .00 | .01 | 49 |
| bowl, new | .58 | .14 | .27 | 80 |
| cheese, familiar | .97 | .00 | .03 | 52 |
| cheese, new | .37 | .53 | .10 | 77 |
| bottle, familiar | .75 | .13 | .12 | 48 |
| bottle, new | .33 | .45 | .22 | 73 |

**A clearance rule fails offline, so I do not register it.** The rule rotated the finger axis to maximise the distance from the fingertips to every other entity's position. Scored against F's recorded yaw options on new-layout states:

| scripted success | fixed pose | rule | random angle | best per state |
|---|---|---|---|---|
| bowl | .53 | .63 | .63 | .88 |
| bottle | .57 | .75 | .70 | .95 |
| cheese | .40 | .18 | .14 | .40 |

The rule matches or barely beats a random angle and it halves cheese, because rotating the cheese grasp fails in familiar scenes too (90° holds at .00). An entity-distance rule with no footprints does not earn the user's allowance for obstacle terms.

**Demos cannot say which objects tolerate rotation.** The bowl's demos show it: its yaw varies (doubled-angle R .82-.88) while the finger axis stays fixed in the world (.98). Bottle and cheese yaw barely vary in demos (.99), so their demos carry no evidence either way.

**Smoke.** A rotated frame produced no hold in 6 of 6 episodes, including K1 familiar at +90°. Repeating theta = 0 succeeded on both its episodes. My reading: a turned frame places the hand and wrist outside the states the part trained on.

### 12.2 T5a: frames and retries

`experiments/sc_t5.py`. No training. U's checkpoints, K1-K6 and N1-N6, both layouts, dev states 4-23, seeds 10-12.

- **Rotation arms:** the object part's yaw frame offset by +45°, −45° or +90° until the hold.
- **Repeat arms:** theta = 0 under three fresh noise offsets. Retrying alone also gains by luck, so rotation must beat this.

Union = success in any attempt on that task, seed and state, with U's own episode included. Each attempt runs from a fresh scene, so unions are upper bounds on what a sequential retry could reach. A real retry also has to survive whatever the failed attempt knocked over.

**Verdict, first match.** PARTS DO NOT TURN iff, on both bowl and bottle, the best rotation's familiar hold falls below U's hold by more than .10. ROTATION PAYS iff on bowl or bottle, new layout, union(rotations) − union(repeats) ≥ +.10 with CI lower bound above 0. Otherwise RETRY ONLY. The retry ceiling, union(repeats) − U, is reported per object with no verdict.

### 12.3 T4b: the controls T4 lacked

`experiments/sc_t4b.py`.

- **Z.** Same created checkpoints at k = 1 with the created part's row of emb and B zeroed, on V3's cells. Created − Z, familiar, per atom: CREDITED iff ≥ +.10 with CI lower bound above 0. Otherwise NOT CREDITED.
- **J.** Uj removes `--psupport` and `--tscale` and uses rank 8, three seeds, all 12 tasks, both layouts. Uj − U, familiar cells: COSTS iff either CI lies wholly below −.10. HOLDS iff both lower bounds are at or above −.10. Otherwise UNRESOLVED.
- **Jc.** Uj's own reuse builds (`--exclude K4+K5`, `--exclude K1`), k = 1, V3's cells. Created − full Uj, familiar: TRANSFERS iff the lower bound is at or above −.10 on both atoms. Otherwise NOT YET.

**What it decides.** CREDITED lets the reuse claim name the part. HOLDS on J makes Uj the recipe. COSTS or UNRESOLVED keeps U and restores the devices, and I look at task-level costs before removing any again.

**Prediction.** T5a PARTS DO NOT TURN. Z CREDITED on the bottle and NOT CREDITED on the plate. J UNRESOLVED, on Ur8's cheese loss and Unt's N4 loss. Jc NOT YET.

## 13. T5a and T4b results (2026-11-05; results/flawed/sc_t5_read.json, sc_t4b_read.json; rollouts sc_t5, sc_t4b)

All jobs completed. Three of my four predictions held; the plate's credit was the miss.

### 13.1 T5a: PARTS DO NOT TURN

| hold | U | +45° | −45° | +90° |
|---|---|---|---|---|
| bowl, familiar | 1.00 | .30 | .23 | .00 |
| bowl, new | .86 | .42 | .17 | .01 |
| cheese, familiar | 1.00 | .04 | .26 | .00 |
| cheese, new | .48 | .08 | .45 | .00 |
| bottle, familiar | .87 | .01 | .38 | .00 |
| bottle, new | .55 | .03 | .22 | .00 |

A trained part does not work in a frame turned away from its demos, even on objects a scripted grasp can take at those angles. So F's pose gain stays out of reach for a part fitted to one pose. Rotation's union also never beats repeating: new-layout differences are bowl −.07, cheese −.10, bottle −.22.

**Retry ceiling** (U plus three repeats, any success; descriptive, attempts run from fresh scenes):

| | one attempt | best of four | gain |
|---|---|---|---|
| bowl, new | .58 | .79 | +.21 |
| cheese, new | .37 | .53 | +.16 |
| bottle, new | .33 | .59 | +.26 |
| bottle, familiar | .75 | .98 | +.24 |

Identical scenes succeed or fail by sampling noise, and this is the first scene-agnostic source of gain the study has found. It needs testing as a real sequence, which T5b does.

### 13.2 T4b

| verdict | result | numbers |
|---|---|---|
| Z, bottle | CREDITED | created .86, switched off .03, +.83 [+.70, +.94] |
| Z, plate | CREDITED | created .94, switched off .36, +.58 [+.42, +.74] |
| J, Uj − U | UNRESOLVED | known −.03 [−.09, +.03]; new pairings −.11 [−.29, +.04] |
| Jc, reuse in Uj | NOT YET | bottle −.00 [−.12, +.12]; plate +.16 [−.02, +.43] |

- **One demo's part does the work.** Part off, every bottle cell sits at .02-.05. This closes the control T4 lacked. My prediction that the plate would add little was wrong: K1 falls to .52, N1 to .42, N4 to .13.
- **U stays the recipe.** The registered consequence of UNRESOLVED applies, and N4 familiar shows why: .78 in U, .30 in Uj. Per-part radii, carry scale and rank 32 return. T4's three single DROPs did not survive being combined.
- **A post-hoc pattern in new layouts.** In Uj, a created bottle part beats the full build (.56 vs .31, CI [+.04, +.48]), and so does the plate (.38 vs .23, CI [+.03, +.29]). In U the same comparison is flat at k = 1 and rises for K4 and N5 at k = 5. Parts fitted to fewer tasks may generalise better to new layouts. This is one comparison with no registered test.

## 14. T5b: a real sequential retry (protocol fixed 2026-11-05, before any non-smoke rollout; m3_prereg sc_t5b)

`experiments/sc_t5b.py`. No training. U's policy, 12 tasks, both layouts, dev states 4-23, seeds 10-12, the same 300-step limit as every other arm.

**The rule, one in every scene.** If the library's proprioceptive event stays off for T steps, the attempt ends: the gripper opens, the hand rises straight up to its start height, returns to its start pose (at most 60 steps), and the policy resumes on fresh noise. T is the 99th percentile of the demos' first event step, fitted without any rollout (147 on seed 10). The rule reads no object, entity or scene constant, and it trains nothing. In the store's terms, it is a schema-level failure transition.

**Verdict, Ur − U, first match.** HARMS iff either familiar cell's CI lies wholly below −.10. PAYS iff both familiar lower bounds are at or above −.10 and, in either new-layout cell (known, new pairings), the difference is at least +.10 with CI lower bound above 0. Otherwise NOT YET. Same bootstrap as T4.

**Reported with no verdict:** per-object differences, attempt counts, success by attempt count.

**Limits.** Return takes about 50 steps, so 300 steps allow about two attempts, and T5a's four-attempt ceiling is out of reach. A failed attempt can move objects, and the retry inherits that scene. Episodes that hold and then fail never trigger a return unless the event stays off for T afterwards. Oracle positions still enter the policy. The smoke read states 0-1 on four combos.

**Prediction.** NOT YET, with a positive difference on bowl and bottle. T5a's ceiling assumes independent fresh scenes and four tries, and this has two tries in scenes that may be disturbed.

### 14.1 T5b result: NOT YET (2026-11-05; results/flawed/sc_t5b_read.json; rollouts sc_t5b)

| Ur − U | Ur | U | difference | CI |
|---|---|---|---|---|
| known, familiar | .872 | .872 | .000 | [−.011, +.011] |
| new pairings, familiar | .892 | .894 | −.003 | [−.017, .000] |
| known, new layout | .400 | .433 | −.033 | [−.092, .000] |
| new pairings, new layout | .383 | .381 | +.003 | [.000, +.011] |

The verdict matched my prediction and the reason I gave did not: bowl fell (−.07 [−.15, −.01]) and bottle sat at .00.

| new layout | share retried | success after one attempt | success after a retry |
|---|---|---|---|
| bowl | .25 | .69 | .00 |
| cheese | .47 | .70 | .00 |
| bottle | .30 | .47 | .00 |

No second attempt succeeded, out of about 246. The rule costs bowl tasks late first-attempt holds, which U keeps. A return at T = 147 usually starts the second attempt near step 200, leaving about 100 steps against a median success of 107-128 steps. So this read cannot separate a too-short budget from a disturbed scene. The familiar guard works: retries fire on under 1% of bowl and cheese familiar episodes, and they cost nothing there.

## 15. T5c: the same rule with room to work (protocol fixed 2026-11-05, before any non-smoke rollout; m3_prereg sc_t5c)

`experiments/sc_t5c.py`. U's policy, 12 tasks, new layout, dev states 4-23, seeds 10-12. Both arms get 600 steps: **Ur600** with T5b's unchanged rule and T = the same demo fit, **U600** alone, so late holds count for U as well.

**Verdict.** RETRY PAYS iff Ur600 − U600 is at least +.10 with CI lower bound above 0 in either cell (known, new pairings). Otherwise RETRY DOES NOT PAY.

**Reported with no verdict:** per-object differences, attempts, success and holds after a retry, U600's late holds.

**What it decides.** RETRY PAYS adds a schema-level failure transition to the architecture, with the time cost stated. RETRY DOES NOT PAY closes retry as the way to reach T5a's ceiling, because scenes after a failed attempt differ from fresh ones. I then stop this line and report to the user. 600 steps is twice the budget of every earlier read, so these numbers do not compare with them.

**Prediction.** RETRY DOES NOT PAY on the group cells, with some gain on the bottle. U600 itself should gain from the extra time.

### 15.1 T5c result: RETRY DOES NOT PAY (2026-11-05; results/flawed/sc_t5c_read.json; rollouts sc_t5c)

| Ur600 − U600 | Ur600 | U600 | difference | CI |
|---|---|---|---|---|
| known | .486 | .486 | .000 | [−.053, +.050] |
| new pairings | .444 | .419 | +.025 | [−.019, +.075] |
| bowl | .644 | .667 | −.022 | [−.100, +.061] |
| cheese | .458 | .425 | +.033 | [−.038, +.113] |
| bottle | .363 | .347 | +.017 | [−.020, +.060] |

| Ur600 by attempts used | share | success |
|---|---|---|
| bowl: 1, 2, 3 | .63, .16, .21 | .86, .57, .08 |
| cheese: 1, 2, 3, 4 | .43, .15, .04, .38 | .91, .44, .00, .00 |
| bottle: 1, 2, 3 | .38, .25, .35 | .88, .08, .02 |

**Reading, post hoc.**
- **The budget was not the main limit.** With three or four tries available, retries rescue little, and U600 gets most of the same gain from waiting. Time alone moves U from .433 to .486 on known tasks and from .381 to .419 on new pairings.
- **T5a's ceiling does not carry over to a sequence.** Fresh-scene repeats reached +.16 to +.26. A sequence reaches about +.02. The likely reason is disturbed scenes, since 42% of U's no-hold bowl failures move something, but this read did not measure scene state after a failed attempt.
- **Cheese's 38% that use four attempts never succeed.** That share is close to F's scripted-grasp block, so the rule cannot reach those states.
- **Retry is closed.** No schema-level failure transition enters the architecture on this evidence.

## 16. T8a and T9 (protocol fixed 2026-11-05, before any non-smoke build; m3_prereg sc_t8)

The user went to bed with "keep iterating towards our goals" and did not choose among the options in §18. So these reads need no new rule. A real VLA retrain waits, because U runs on oracle positions and a pixel VLA does not, so that comparison has no equal inputs until parts run from pixels.

`experiments/sc_t8.py`. Seeds 10-12, 30,000 steps, K1-K6 and N1-N6, both layouts, dev states 4-23.

**T8a, the oracle rungs of C1's ladder.** Same demos and oracle inputs.

| arm | what changes from U |
|---|---|
| Ulc | a small MLP combines the two role velocities, the hold and flow time, in place of the fixed rule; roles train jointly |
| RF | one role-routed head, atom identity as input (`pp_baselines`) |
| HF | one holistic head (`pp_baselines`) |

RF and HF get contact labels and given roles, so they receive more supervision than U. RF and HF have 888k and 967k parameters against U's 1.78M. Parameter budgets differ, and the plan's equal budget is unmet.

**Verdict per arm, U − arm, first match.** UNDERFIT iff on known tasks, familiar, the difference exceeds +.10 with CI lower bound above 0 (so the arm cannot do even its training tasks). FALLS iff on new pairings, familiar, the CI lower bound exceeds +.10. Otherwise NO FALL. Overall: C1 RUNGS HOLD iff RF and HF both read FALLS. Otherwise C1 RUNGS FAIL. Ulc's label is reported separately, because it keeps the parts.

**T9, a lifelong stream.** A base build without K1, K4 and K5, so with no bottle and no plate. Two parts then arrive one at a time from one demo each, in both orders: bp (bottle from K4, then plate from K1) and pb (the reverse). The second creation starts from the first's checkpoint. N4, bottle on plate, then pairs two parts neither of which was in the base.

- **Stream, each order's final store − full U, on K1, K4, N1, N4, N5, N6, familiar.** STREAM HOLDS iff the lower bound is at or above −.10 in both orders. Otherwise NOT YET.
- **Forgetting, each order's final store − the base, on K2, K3, K6, N2, N3, familiar.** NOT FORGOTTEN iff the lower bound is at or above −.10 in both orders. Otherwise FORGETS. Old parameters stay bitwise unchanged, which the creation code asserts, so only a change in the lexicon's parses could move these.
- **Reported with no verdict:** N4's success in both orders, the new layout, and a table of every instruction's parse under the base and final stores. The smoke showed the base parsing "put the cream cheese on the plate" as cheese into bowl, a silent wrong parse that S1's refusal did not catch when two tasks are withheld.

**Prediction.** T8a C1 RUNGS HOLD, with Ulc at NO FALL, because Ulc reads only the role velocities and can learn the switch. T9 STREAM HOLDS is unlikely on six tasks of mixed difficulty: NOT YET in at least one order, through N4. NOT FORGOTTEN.

## 17. T6: chains no demo shows (protocol fixed 2026-11-05, before any non-smoke rollout; m3_prereg sc_t6)

`experiments/sc_t6.py`. No training, U's checkpoints, seeds 10-12, dev states 4-23, both layouts.

**The schema, one rule.** Split the instruction at " then ", parse each clause with the library's lexicon, run the first skill until the release event (hold on, then off with the command open), open the gripper, rise, return to the start pose, and run the second skill. At most 300 steps per skill and 60 for the return. Success means both goals are true at once, judged by a BDDL goal joined from the two tasks.

| chain | first | second |
|---|---|---|
| C1 | K1 bowl → plate | N2 cheese → stove |
| C2 | K4 bottle → cabinet | K1 bowl → plate |
| C3 | K2 bowl → stove | K4 bottle → cabinet |
| C4 | N1 cheese → plate | K3 bowl → cabinet |
| C5 | N4 bottle → plate | N2 cheese → stove |
| C6 | N3 cheese → cabinet | K2 bowl → stove |

The " then " split is a hand-written grammar rule, and the chains were chosen so each object and target is free when its skill runs. Both are disclosed limits.

**Verdict, familiar.** Chain success − product of the two skills' U outcomes, on matching seeds and state indices. BREAKS iff the CI lies wholly below −.10. HOLDS iff the lower bound is at or above −.10. Otherwise UNRESOLVED. New layout reported with no verdict.

**Reported with no verdict:** the first goal at the transition and at the end, the second skill's hold, and U alone on K7, N7, K8, N8. This read tests sequence by transition. The drawer needs a schema with three phases and stays open.

**Prediction.** UNRESOLVED, with a first skill that holds and a second that loses something to the first's moved objects.

## 18. Where the new-layout line stands

Three scene-blind routes tried, none closes the gap: the frame turned offline by clearance (failed), the frame turned by a part (PARTS DO NOT TURN), and retry (DOES NOT PAY). What remains known: F shows bowl and bottle states are physically open at other poses, and a part fitted to one pose cannot reach them. Giving a part those poses needs either new demos showing them, or geometric data that amounts to a grasp-option teacher, which the rules bar in training. That choice belongs to the user.

## 19. T2: entity keys on frozen VLM features (protocol fixed 2026-11-05, before any non-smoke read; m3_prereg sc_t2)

`experiments/sc_t2.py`. No training. pi05_libero's image features before language fusion, on LOCATOR's frames, heads, event labels and camera matrices.

**Key, k = 1.** From one demo's first frame, the token under the demo's event label, averaged with its four neighbours (a store of one vector per entity). Three different demos give three repeats. A mean over 10 demos is reported as k = 10.

**Locate.** The most similar token in a development frame, refined by a softmax over its 3 × 3 window at temperature .02, then lifted to the head's mean label height. One token covers 16 px, about .06 m. Errors subtract the simulator's constant label shift, as in LOCATOR's verdict, and raw errors are reported beside them.

**Verdict, LOCATOR's mapping, on the median over repeats of each head's 95th-percentile horizontal error at k = 1, states 4-23.** WITHIN BUDGET iff every head is at or below .02 m in both layouts. BEATS A TABLE, OVER BUDGET iff every head in the new layout is at or below half a constant's error. Otherwise NO BETTER THAN A TABLE.

**What it decides.** Whether the entity store can sit on the VLM, per the plan's stop rule. A failure keeps it on the small encoder, which weakens the tie to the VLM.

**Limits.** No fine-tuned head and no refinement beyond the token window. The camera matrices, height and the label shift stay oracle.

**Prediction.** NO BETTER THAN A TABLE, on the plate and on confusion between similar objects, with the median error well below the 95th percentile.

## 20. T2, T8a and T9 results (2026-11-05 night; results/flawed/sc_t2_read.json, sc_t8_read.json; rollouts sc_t8)

All three are development reads on states that have shaped earlier work. The user has not reviewed the rules.

### 20.1 T2: NO BETTER THAN A TABLE (predicted)

Median over three one-demo keys of each head's 95th-percentile horizontal error, in metres, against a 2 cm budget.

| head | familiar | new | constant table, familiar | constant table, new | median error, familiar | median error, new |
|---|---|---|---|---|---|---|
| cheese | .332 | .319 | .021 | .271 | .05-.27 | .09-.19 |
| bottle | .348 | .447 | .016 | .371 | .04-.15 | .18-.25 |
| bowl | .099 | .543 | .017 | .261 | .03-.05 | .16-.32 |
| plate | .089 | .425 | .016 | .322 | .03-.09 | .07-.24 |
| bowl as target | .195 | .383 | .019 | .258 | .02-.04 | .24-.29 |

- **Every head is worse than a constant at the 95th percentile, in both layouts.** A key from 10 demos does not rescue it (new layout .31-.58).
- **My prediction named the plate.** The plate is one of the better heads in the familiar layout, so that part was wrong. The cheese and bottle fail hardest there.
- **The stop rule fires.** The entity store stays on a small encoder, which weakens the tie to the VLM. This read tests one key per entity on pooled token features, with no fine-tuned head. It does not rule out VLM features with a trained head.

### 20.2 T8a: C1 RUNGS FAIL, by HF's label; the rungs themselves fall hard

| arm | known familiar | new pairings familiar | known new | new pairings new | U − arm, new pairings familiar | label |
|---|---|---|---|---|---|---|
| U | .872 | .894 | .433 | .381 | | |
| Ulc | .931 | .358 | .089 | .092 | +.536 [+.378, +.692] | FALLS |
| RF | .742 | .011 | .028 | .003 | +.883 [+.756, +.978] | FALLS |
| HF | .658 | .003 | .000 | .000 | +.892 [+.772, +.975] | UNDERFIT |

- **The registered overall reads C1 RUNGS FAIL.** HF trails U on its own training tasks by +.214 [+.017, +.414], so the first-match rule labels it UNDERFIT and its new-pairing loss cannot count as a pure generalization loss. My prediction of C1 RUNGS HOLD was wrong on that rung.
- **Ulc is the informative arm, and my prediction there was wrong too.** I predicted NO FALL. Ulc keeps every part and every role and replaces the fixed rule with a small MLP, and roles train jointly. It matches or beats U on known tasks (−.058 [−.192, +.031]) and loses .54 on new pairings. It loses most on the bottle pairings (N4 .08, N5 .22, N6 .00) and less on cheese (N1 .55, N2 .72, N3 .58). It also loses in the new layout on known tasks, +.344 [+.183, +.511].
- **The fixed rule carries it (post hoc, existing rollouts, same bootstrap).** T4b's Uj trains jointly and keeps the fixed rule, so Uj − Ulc changes the rule alone, and U − Uj changes the training alone.

| new pairings unless marked | U − Uj (joint training) | Uj − Ulc (the rule) |
|---|---|---|
| familiar | +.106 [−.036, +.289] | **+.431 [+.217, +.647]** |
| new layout | +.069 [−.042, +.175] | **+.219 [+.122, +.333]** |
| known tasks, familiar | +.033 [−.033, +.094] | −.092 [−.222, +.008] |
| known tasks, new layout | +.061 [−.056, +.186] | **+.283 [+.167, +.422]** |

  With parts, roles and training held equal, learning the combination costs nothing measurable on trained tasks and loses .43 on new pairings. Parts alone do not generalize; the fixed rule carries them. This is the sharpest removal result the thesis has. It remains a development read with oracle inputs, and it compares against one learned-rule form.
- **Limits.** All arms have oracle inputs and the same demos. RF and HF have about half U's parameters, and they receive contact labels and given roles. No retrained VLA is in the ladder yet, so C1's refuting case remains untested.

### 20.3 T9: NOT YET (predicted), by .006; NOT FORGOTTEN (predicted)

| final store | − full U, six stream tasks, familiar | new layout | − base, five old tasks, familiar | N4, double new, familiar / new |
|---|---|---|---|---|
| bp | −.011 [−.106, +.083] | +.042 [−.072, +.158] | .000 [.000, .000] | .82 / .35 |
| pb | −.011 [−.106, +.083] | +.039 [−.067, +.147] | +.003 [.000, +.020] | .82 / .37 |

- **The registered verdict is NOT YET.** The lower bound −.106 misses −.10 by .006, with a point estimate of −.011. The margin stays where it was registered.
- **N4 works with both parts created after the base.** It reaches .82 familiar and .35-.37 new, with no demo of bottle on plate and no task in the base that contains either part.
- **Order makes almost no difference.** Familiar rates match between orders on every task but K6 (.967 and .983), and new-layout rates differ by at most .02.
- **Old tasks hold.** Parameters stay bitwise unchanged, and no old task's parse moves.

**The parse table shows a defect.** Before the parts arrive, the base lexicon silently misparses four instructions it should refuse.

| task | instruction's content | base parse | final parse |
|---|---|---|---|
| K4 | bottle → cabinet | **bowl → cabinet** | correct |
| N1 | cheese → plate | **cheese → bowl** | correct |
| N5 | bottle → stove | **bowl → stove** | correct |
| N6 | bottle → bowl | **cheese → bowl** | correct |
| K1, N4, K5 | | refused | K1, N4 correct, K5 refused (no rack part) |

**Correction to §16, post hoc.** §16 says S1's refusal did not catch this parse. That was wrong: the table above comes from `role_parse`, the rule without refusal, which is what rollouts call. Run on the same stores, S1's `Lexicon` refuses all four, and on U it agrees with `role_parse` on K1-K8 and N1-N7. It also refuses N8 ("push the bowl..."), which `role_parse` on U silently runs as bowl → stove. So the defect is wiring: the rollout path bypasses the refusing store. T6b routes clauses through `Lexicon`.

## 21. T6 result: BREAKS (2026-11-05 night; results/flawed/sc_t6_read.json; rollouts sc_t6)

**Amendment to the read.** The first read crashed on an assert in the descriptive block, because U's lexicon refuses K8. I changed that block to report refusals, after the chain rollouts and before any verdict had printed. The verdict's code did not change, and the chain block keeps its assert. The registered sha of `experiments/sc_t6.py` is the pre-amendment file.

**Registered verdict, familiar: BREAKS.** Chain .494 against a product of .931, −.436 [−.636, −.244]. My prediction of UNRESOLVED was wrong. New layout, no verdict: .086 against .158, −.072 [−.169, +.019].

| chain | chain | product | first alone | first goal at transition | first goal at end | second alone | second held | second goal at end |
|---|---|---|---|---|---|---|---|---|
| C1 K1→N2 | .67 | .98 | 1.00 | .85 | .87 | .98 | 1.00 | .72 |
| C2 K4→K1 | .13 | .93 | .93 | .53 | .52 | 1.00 | .82 | .18 |
| C3 K2→K4 | .82 | .93 | 1.00 | .95 | .95 | .93 | 1.00 | .87 |
| C4 N1→K3 | .62 | .97 | .98 | .77 | .80 | .98 | .95 | .78 |
| C5 N4→N2 | .12 | .78 | .78 | .63 | .65 | .98 | 1.00 | .18 |
| C6 N3→K2 | .62 | .98 | .98 | .93 | .93 | 1.00 | .92 | .67 |

The two chains that start with the bottle fail hardest, at the second skill. U alone on the descriptive tasks: K7 .30, N7 .00, N8 .00, K8 refused in every episode (no parse), all familiar. New layout: K7 .03, the others 0.

### 21.1 What broke, post hoc: my transition

Per-episode breakdown, familiar. Second skill's goal at the end, split by whether the scripted return arrived inside its 60 steps.

| chain | return ran out | second goal, return arrived | second goal, return ran out |
|---|---|---|---|
| C1 | .10 | .80 | .00 |
| C2 | .80 | .75 | .04 |
| C3 | .00 | .87 | |
| C4 | .00 | .78 | |
| C5 | .88 | .86 | .09 |
| C6 | .33 | .93 | .15 |

Three defects sit in the schema I wrote, and the verdict stands as registered with them in it.

- **The return stalls.** A trace on three smoke states of C2 and C5 shows the hand stopping at the same point, about (.079, 0, 1.188), from different starts, with x growing while the command asks for −x. The script rises to the start height first, and that height far from the base extends the arm fully. My reading, from kinematics and with no joint angles logged, is that the elbow reaches its straight limit, where a retract command produces no motion. A second skill then starts far from any start pose it has seen.
- **The hand leaves too early.** The release event fires with the fingers still near closed (width .02-.03 in the traces), and the script rises at once. The bottle gets dragged: the first goal goes from true to false within 10 steps in three of six traces, and one of those recovers.
- **The last skill stops at its release.** U alone runs until the goal or 300 steps. So the chain and the product measure different things. Between .02 and .22 of familiar chains end with the last skill released and its goal false.

These defects mean T6 did not test whether skills compose in sequence. It tested this script. The read says nothing yet about moved objects interfering, which was my prediction's mechanism.

## 22. T6b: the boundary rebuilt on hand state (protocol fixed 2026-11-06, before any non-smoke rollout; m3_prereg sc_t6b)

**Weakness, measured in 21.1.** The scripted boundary. **One principle.** The boundary ends on hand state, and a skill runs as it runs alone.

`experiments/sc_t6b.py`. No training, U's checkpoints, seeds 10-12, dev states 4-23, both layouts, T6's six chains and joined goals. Clauses parse through S1's `Lexicon`.

**The stall, measured on smoke states.** A probe of joint angles confirms 21.1's reading. During a stalled return, joint 4 sits at −.07, the Panda's elbow limit, with no contacts. A retract command swings the locked arm upward until it rests near (.016, 0, 1.35). The elbow's link offsets put the limit past the true straight point (my reading from the arm's geometry), so an inward pull presses it against its stop. The grasps of the tall bottle reach this pose because the wrist sits high.

**Boundary, both arms.** At the release event, hold still with the command open until the finger width stops changing (under 1 mm per step, at most 20 steps). Rise 8 cm straight up. The rise height is a hand-set constant, one value in every scene.

| arm | after the rise |
|---|---|
| R | a straight line in space to the start pose, until within 2 cm (at most 150 steps), then the second skill. While the elbow sits within .02 rad of its limit, push outward horizontally until it has bent by .25 rad |
| D | the second skill starts at once, from wherever the hand is, with no elbow rule |

The elbow rule reads the arm's joint angle and its own limit, and no scene input. It names one joint of this arm, so it is robot-specific. D scripts less: no return, so only open and rise. It asks more of the parts, which have seen only one start pose. The last skill runs until the goal or 300 steps, as U alone does.

**Verdict, each arm, familiar, chain − product of U's alone outcomes on matching seeds and states.** BREAKS iff the CI lies wholly below −.10. HOLDS iff the lower bound is at or above −.10. Otherwise UNRESOLVED. T6's verdict stays on the record and this read does not replace it. Reported with no verdict: D − R, the new layout, and per-chain rows for the first goal at release and at the transition, arrival, the second hold and each goal at the end.

**What the smoke showed before this row.** Smoke states 0-2, seed 10, familiar. Before the elbow rule: R succeeds on all nine episodes of C1, C3 and C4, stalls on C2 and C5, and D succeeds only on C4. With it: R succeeds on nine of nine on C1, C2 and C5. The elbow rule and its two constants came from these states. The development states 4-23 have not seen this schema.

**Prediction.** R HOLDS, with a point estimate near −.05, since the smoke is clean and 20 states will be harder. D BREAKS, because the parts have seen one start pose.

**What it decides.** R's HOLDS would say sequence composes at the product of skill competences on two-skill chains, under a scripted boundary. That is C3's chain half, with the drawer still open. A scripted boundary counts as structure the schema holds, so it needs a removal control, which D provides.

### 22.1 T6b result: R UNRESOLVED, D BREAKS (2026-11-06; results/flawed/sc_t6b_read.json; rollouts sc_t6b)

| arm | familiar chain | product | − product [95% CI] | label | new chain | new product | − product |
|---|---|---|---|---|---|---|---|
| R | .869 | .931 | −.061 [−.108, −.017] | UNRESOLVED | .181 | .158 | +.022 [−.056, +.100] |
| D | .219 | .931 | −.711 [−.917, −.453] | BREAKS | .033 | .158 | −.125 [−.250, −.022] |

- **My prediction on R missed by .008.** I predicted HOLDS near −.05. The point estimate is −.061 and the lower bound is −.108 against −.10. The margin stays where it was registered.
- **The boundary was most of T6's break.** Familiar chains go from .494 to .869 on the same skills, states and seeds. No skill changed, so the loss in T6 belonged to my transition.
- **In the new layout, chains reach their product.** That product is low, because the skills themselves fail there. Sequence adds no measurable loss on top.
- **The return earns its place.** D − R is −.650 [−.858, −.375] familiar and −.147 [−.267, −.044] new. Parts have seen one start pose and cannot take over from another. This is a limit on the parts, and it means the schema holds real structure.

| chain, familiar | R chain | product | first goal at release | first goal at transition | first goal at end | return arrived | second goal at end |
|---|---|---|---|---|---|---|---|
| C1 K1→N2 | .97 | .98 | .72 | .98 | .98 | 1.00 | .98 |
| C2 K4→K1 | .87 | .93 | .70 | .87 | .87 | .98 | .95 |
| C3 K2→K4 | .88 | .93 | .93 | 1.00 | 1.00 | 1.00 | .88 |
| C4 N1→K3 | .87 | .97 | .48 | .95 | .95 | .98 | .92 |
| C5 N4→N2 | .68 | .78 | .42 | .72 | .73 | .97 | .92 |
| C6 N3→K2 | .95 | .98 | .32 | .98 | .98 | 1.00 | .97 |

**Reading, post hoc.** The goal is often still false at the release event and true once the object settles, which is what waiting for the fingers buys. The first skill now keeps its goal through the rest of the chain: first goal at the transition and at the end agree within .01 on all six chains. So I saw no sign of the second skill disturbing the first object, which was the mechanism in my T6 prediction.

**Limits.** The " then " split, the return, the rise height and the elbow rule are all hand-written, and the elbow rule came from smoke states. The chains were chosen so targets stay free. All inputs are oracle.

## 23. T6c: three skills no demo shows (protocol fixed 2026-11-06, before any non-smoke rollout; m3_prereg sc_t6c)

`experiments/sc_t6c.py`. T6b's arm R, unchanged, on chains of three skills. Seeds 10-12, dev states 4-23, both layouts. No training, U's checkpoints. Three objects go to three different targets, and every order of the objects appears once.

| chain | first | second | third |
|---|---|---|---|
| L1 | K1 bowl → plate | N2 cheese → stove | K4 bottle → cabinet |
| L2 | K4 bottle → cabinet | N1 cheese → plate | K2 bowl → stove |
| L3 | N4 bottle → plate | K3 bowl → cabinet | N2 cheese → stove |
| L4 | K2 bowl → stove | N4 bottle → plate | N3 cheese → cabinet |
| L5 | N1 cheese → plate | N5 bottle → stove | K3 bowl → cabinet |
| L6 | N3 cheese → cabinet | K1 bowl → plate | N5 bottle → stove |

**Verdict, familiar, chain − product of the three skills' U outcomes on matching seeds and states.** BREAKS iff the CI lies wholly below −.10. HOLDS iff the lower bound is at or above −.10. Otherwise UNRESOLVED. New layout reported with no verdict, with each goal at the end per chain.

**Prediction.** UNRESOLVED, with a chain near .70 against a product near .82, since each boundary costs about .05 and a later skill now meets two placed objects.

**Limits.** As in T6b. No demo holds more than one pick and place here, so this read tests length past anything trained, and the objects and targets are all known.

### 23.1 T6c result: UNRESOLVED (predicted) (2026-11-06; results/flawed/sc_t6c_read.json; rollouts sc_t6c)

**Amendment to the read.** The first read crashed on an assert: a long episode ran past the environment's 1000-step horizon, and the next step raised an error. I changed the read to count that one error as a failure with its goals unmeasured, and to report the share. Any other error still stops the read. Three skills at 300 steps plus two boundaries can reach 1310 steps, so this horizon cuts off some long episodes that T6b never reached. It costs .00-.03 of familiar episodes per chain and .08-.28 in the new layout.

**Familiar: chain .817, product .869, −.053 [−.133, +.036].** The label is UNRESOLVED, as predicted, and the chain beats my guess of .70. New layout, no verdict: .028 against .042, −.014 [−.064, +.042]. Both rates are near the floor, since a product of three failing skills is small.

| chain, familiar | chain | product | goals at the end: first, second, third | alone: first, second, third |
|---|---|---|---|---|
| L1 bowl, cheese, bottle | .93 | .92 | .98, .98, .97 | 1.00, .98, .93 |
| L2 bottle, cheese, bowl | .78 | .92 | .88, .88, .93 | .93, .98, 1.00 |
| L3 bottle, bowl, cheese | .65 | .78 | .70, .93, .90 | .78, .98, .98 |
| L4 bowl, bottle, cheese | .72 | .77 | 1.00, .72, .97 | 1.00, .78, .98 |
| L5 cheese, bottle, bowl | .90 | .92 | .95, .93, .95 | .98, .93, .98 |
| L6 cheese, bowl, bottle | .92 | .92 | .98, .95, .95 | .98, 1.00, .93 |

**Reading, post hoc.** The third skill does not lower the gap: −.053 at three skills against −.061 at two, with intervals that overlap a lot. On this read, length past anything trained costs little beyond the boundaries themselves. The largest losses sit in the chains that start with the bottle and on the bottle's own place (L2, L3). The bottle's skill alone is the weakest of the three, which the product already charges for.

## 24. T11: a push is a carry without a close (protocol fixed 2026-11-06, before any non-smoke build; m3_prereg sc_t11)

**Weakness, measured on the demos, offline.** The drawer's and the push's first phases never fire the hold event, because the gripper stays open. In 42 of 50 K7 demos the only close is the bowl's. Every K8 demo stays open, so K8 has no situation, no parse and no rows. The plan reads this as sequence's weakest evidence (§3.5).

**Scene-motion read of the demos, no labels.** An entity counted as moving if it moved over 5 mm across 5 steps. Motion with the hold off and over 2 cm net:

| tasks | demos with it |
|---|---|
| K1, K2, K3, K4 | 0 of 50 each |
| K5, K6 | 3, 4 of 50, small settling moves |
| K7 | 50 of 50, drawer, starting near step 35 |
| K8 | 50 of 50, plate first, starting near step 54 |

**One principle.** A hold is the hand moving an entity. Closing is only one way to do that. So the gate generalizes: held, or the hand moves the parsed object while the command stays open.

**The event.** `HandMoved`. On once the parsed object sits 2 cm from its rest position with the command open. Off once it has moved under 5 mm in 15 steps. The first close command disarms it for the episode. Without that rule it also fired on objects falling after a release (K6 40 of 50 demos, K7 47 of 50). With it the event fires in 0 of 50 demos on K1-K6, 1 of 50 on K7's bowl, and 50 of 50 on K8. Its constants are hand-set, one value everywhere. It reads the object's position, which is oracle like every position here.

**The lexicon.** A demo with no hold takes its situation from the first entity it moves: K8 reads (plate, stove front). That alone broke the lexicon. "stove" sits next to the stove-front target in K8 and the cook region in K2, so its candidate set empties and K2, N2 and N5 lost their parse. The fix is a rule from cross-situational learning: a slot its situation leaves unexplained goes to that situation's rarest contradicted word. "stove" then takes the cook region ahead of "on", "put" and "the". With it all 16 instructions parse correctly on the demos, including N8's target. This rule is new, and I wrote it after seeing this failure. It runs only under this flag.

**Arm Um.** U's recipe plus `--mgate`, seeds 10-12, 30,000 steps. K8 now trains as a task: plate object part, stove-front target part, and no new role. Rollouts on K1-K6, N1-N6, K8, N8, both layouts, dev states 4-23. U's own rollouts serve as the comparator on the first twelve. U scores 0 on K8 and N8.

**Verdicts, familiar.**
- **Push, K8 over states.** PUSH WORKS iff the mean is at or above .50 and the CI lower bound at or above .25. PUSH PARTIAL iff the lower bound exceeds .05. Otherwise NO PUSH.
- **Cost, Um − U on known K1-K6 and on new pairings N1-N6.** NO COST iff both lower bounds are at or above −.10. Otherwise COST.
- **Reported with no verdict.** N8, which pushes the bowl under a part that has only learned to grasp it; the new layout; how often the event fires in rollouts.

**Prediction.** PUSH PARTIAL, with K8 near .35. A part in the plate's frame has to find the contact point from one approach in the demos. NO COST, with the event quiet on grasp tasks. N8 near 0, because a verb-specific approach is missing.

**The drawer stays for later.** K7's two phases need different objects within one task, which this arm does not do. Its first phase still trains under the bowl's part.

### 24.1 Result: PUSH WORKS, NO COST (2026-11-06; results/flawed/sc_t11_read.json; rollouts sc_t11)

Development read, oracle positions, one kitchen, 3 seeds, 20 states per task. The seed-12 build lost its node once and requeued. All 84 rollouts started after the requeued build finished.

**Push.** K8 familiar .867 [.783, .950]. Registered bar: mean .50, lower bound .25. **PUSH WORKS.** U scores 0 on this task. My prediction was PUSH PARTIAL near .35, which was wrong: a part in the plate's frame found the contact point far more reliably than I expected.

**Cost.** Um − U, familiar: known +.019 [−.039, +.103]; new pairings −.011 [−.069, +.047]. **NO COST**, as predicted.

| group | layout | Um | U | Um − U |
|---|---|---|---|---|
| known K1-K6 | familiar | .892 | .872 | +.019 [−.039, +.103] |
| new pairings N1-N6 | familiar | .883 | .894 | −.011 [−.069, +.047] |
| known K1-K6 | new | .475 | .433 | +.042 [−.022, +.125] |
| new pairings N1-N6 | new | .400 | .381 | +.019 [−.072, +.117] |

**Reported with no verdict.**

| task | layout | Um | event fired |
|---|---|---|---|
| K8 plate push | familiar | .867 [.783, .950] | .983 |
| K8 plate push | new | .233 [.083, .400] | 1.00 |
| N8 bowl push | familiar | .033 [0, .100] | 1.00 |
| N8 bowl push | new | .067 [0, .183] | .983 |

**What this does and does not establish.** K8 trained on its own 50 demos, so K8 measures the generalized event and shows nothing about composition. The composition test in this block is N8, the push verb on an object the push demos never touch, and N8 fails as predicted. The new layout falls for K8 as it does for grasps. The rarest-word rule and the event's constants came after I read the demos, and the user has not reviewed them.

## 25. T12: the verb as a slot (protocol fixed 2026-11-06, before any non-smoke rollout; m3_prereg sc_t12)

**Weakness, read from T11's rollouts.** On N8 the bowl's part closes the hand near step 47, because grasping is all it has learned. The hold event then hands over to the push target part, which commands an open hand, because K8 never closes. The bowl drops a few centimetres along: it ends near y = .05 against a target near y = .21. K8's demos show the hand dragging the plate by its rim with the fingers open, and K1's grasp site on the bowl sits at the same kind of rim offset. So the bowl's part already reaches a site from which an open hand can drag. The word "push" never reaches the gripper.

**One principle.** How the hand moves an object is one more slot of the situation, learned the way the entity slots are.

**The rule, three steps, no retraining.**
1. **Mode per instruction, from the demos.** A task reads `close` if over half its demos show a hold event, `open` otherwise. K1-K7 read `close`, K8 reads `open`.
2. **Word binding.** A token binds to a mode if every instruction containing it has that mode. An instruction takes the mode of its rarest bound token, the same preference as T11's rarest rule. All 16 instructions parse as expected: only K8 and N8 read `open`.
3. **Execution under `open`.** The gripper command stays open for the whole episode. The first close the policy commands marks arrival and turns the gate on for the rest of the episode, so the target part takes over. Under `close` the code path is the earlier episode.

Arrival reuses the part's own close command as a signal. It needs no contact label, no grasp-option teacher and no scene constant.

**A defect the smoke caught.** My first version excluded the entity lexicon's words from binding. The lexicon had bound "push", "to" and "front" to the stove-front target, since those three only occur in K8, so the exclusion erased the verb and both push tasks parsed to no mode. I replaced the exclusion with the rarest-token rule before registering.

**Limits I can name now.** One push task cannot separate "push" from "to" and "front": all three bind `open`, so "put the bowl to the front of the stove" would also read `open`. N8 is one task on one object, in a scene whose bowl-to-target direction may suit a drag. The goal predicate reads the object's final position, so it scores the outcome and does not check the verb; the mode rule itself guarantees an open hand.

**Arm V.** T11's Um checkpoints with the mode applied, seeds 10-12, K8 and N8, both layouts, dev states 4-23. T11's Um rollouts serve as the comparator.

**Verdicts, familiar.**
- **N8 over states.** VERB COMPOSES iff the mean is at or above .50 and the CI lower bound at or above .25. VERB PARTIAL iff the lower bound exceeds .05. Otherwise VERB DOES NOT COMPOSE. The same bars as T11's push.
- **K8, V − Um.** K8 HOLDS iff the CI lower bound is at or above −.10. Otherwise K8 HURT.
- **Reported with no verdict.** The new layout, how often arrival fires, where the object ends.

**Prediction, after reading smoke.** The smoke read seed 10 on states 0-1: N8 2 of 2, K8 2 of 2. I predict VERB COMPOSES with N8 near .70 familiar, K8 HOLDS with a delta near 0, and N8 near .20 on the new layout, in line with K8's .23 there.

### 25.1 Result: VERB COMPOSES, K8 HOLDS (2026-11-06; results/flawed/sc_t12_read.json; rollouts sc_t12)

Development read, oracle positions, one kitchen, 3 seeds, 20 states per task.

| task | layout | V | Um (T11) | V − Um | arrival fired |
|---|---|---|---|---|---|
| N8 bowl push | familiar | .833 [.683, .950] | .033 | +.800 [+.633, +.933] | 1.00 |
| N8 bowl push | new | .517 [.317, .700] | .067 | +.450 [+.250, +.650] | 1.00 |
| K8 plate push | familiar | .850 [.750, .933] | .867 | −.017 [−.083, +.050] | 0 |
| K8 plate push | new | .267 [.133, .433] | .233 | +.033 [−.033, +.100] | .03 |

**VERB COMPOSES** and **K8 HOLDS**, as predicted. My numbers were low: N8 familiar came in above .70, and N8 new far above .20. N8 on the new layout beats K8 there, .517 against .267, which I did not expect and cannot yet explain.

**What this establishes.** A verb demonstrated on the plate alone now drives the bowl, using the bowl's approach, the push's motion and a mode read from the words. No new demo and no retraining enter. It is the first result in this block where a whole new kind of action composes with an existing part.

**What it leaves open.**
- One task on one object. The bowl's grasp site sits at the rim, which suits a drag, and the bowl's direction to this target may suit it too. The bowl ends about 1.6 cm above its rest height on average, so it may tilt while dragged. T13 tests other objects.
- "Push", "to" and "front" stay confounded (§25).
- Success reads the object's final position. The open hand comes from the rule, and I have not watched the episodes.
- The arrival rule and the word rule came after I read T11's failure, and the user has not reviewed them.

## 26. T13: how far the push verb reaches (protocol fixed 2026-11-06, before any non-smoke rollout; m3_prereg sc_t13)

**Question.** T12 rests on one object whose rim suits a drag. Does the same rule work on objects with other geometry? Two new instructions: P1 "push the cream cheese to the front of the stove" (a low box) and P2 "push the wine bottle to the front of the stove" (a tall object that can tip). Neither appears in any demo or in T12. Each BDDL is the shared template with its language and goal replaced, and each uses the same 24 start states as K8.

**A change to the word rule.** Working through P1 by hand, "cream" and "push" tie as rarest tokens with opposite modes, so T12's rule would give no mode. A word the object slot already explains should not vote on how the hand moves. So tokens with `pg` > 0 in the entity lexicon no longer bind a mode. On the smoke run all 16 existing instructions keep their mode, and P1 and P2 read `open`. T12's committed read used the earlier rule, and the two agree on K8 and N8.

**Arm V.** T11's Um checkpoints, no retraining, seeds 10-12, P1 and P2, both layouts, dev states 4-23. No comparator: without the mode, a push has no way to stay open.

**Verdicts, familiar, per task.** VERB COMPOSES iff the mean is at or above .50 and the CI lower bound at or above .25. VERB PARTIAL iff the lower bound exceeds .05. Otherwise VERB DOES NOT COMPOSE. **Reach:** GENERAL iff both tasks are at least partial, SOME OBJECTS iff one is, BOWL ONLY iff neither.

**Reported with no verdict.** The new layout, how often arrival fires, and where the object ends.

**Prediction, after reading smoke.** The smoke read seed 10 on states 0-1: P1 2 of 2, P2 0 of 2, with the bottle knocked away and arrival firing early, near step 27. I predict P1 VERB COMPOSES near .70, P2 VERB DOES NOT COMPOSE near .05, reach SOME OBJECTS.

### 26.1 Result: SOME OBJECTS (2026-11-06; results/flawed/sc_t13_read.json; rollouts sc_t13)

Development read, oracle positions, one kitchen, 3 seeds, 20 states per task.

| task | layout | V | arrival fired | object's mean end (x, y) |
|---|---|---|---|---|
| P1 cheese push | familiar | .983 [.950, 1.00] | 1.00 | −.045, .172 |
| P1 cheese push | new | .133 [.033, .250] | .67 | −.101, .042 |
| P2 bottle push | familiar | 0 [0, 0] | 1.00 | −.207, −.048 |
| P2 bottle push | new | 0 [0, 0] | .93 | −.012, .069 |

P1 **VERB COMPOSES**, P2 **VERB DOES NOT COMPOSE**, reach **SOME OBJECTS**, all as predicted. P1 came in above my .70.

**Read P1 familiar with care.** The cheese's familiar position sits near (−.05, .13) and the target region starts near y = .17, so the task needs a drag of about 4 cm. It ends at y = .172, barely inside. The bowl's task needs about 17 cm. So .983 mostly shows the verb on an easy case, and it should not count as equal evidence to N8. On the new layout, where the distance is larger, P1 falls to .133 and arrival fires in only two thirds of episodes. N8 holds .517 there.

**P2.** The bottle ends near its familiar start, about (−.20, −.05). So on average the hand barely moves it, and on the smoke's two states it pushed the bottle further from the target. Arrival fires early on the smoke, near step 27, and I have not separated an early arrival from an open hand's inability to drag a tall object at its grasp site.

**What the verb's reach shows, post hoc.** The verb transfers where the object's learned site lets an open hand drag it: the bowl's rim, and the cheese over a short distance. It does not transfer to a tall object held near its neck. The parts carry a grasp site and nothing about what that site can do with an open hand, so this boundary comes from the part design.

## 27. T14: chains that mix verbs (protocol fixed 2026-11-06, before any non-smoke rollout; m3_prereg sc_t14)

**Question.** Do a verb from one demo set and a sequence no demo shows compose together? T6b tested grasp chains and T12 tested a single push. T14 joins them under one instruction per clause, on T11's Um checkpoints with no retraining.

**Clause rule.** Each clause parses its own roles and mode, and runs its own hold event and arrival. A `close` clause ends at its release, as in T6b. An `open` clause has no release, so it ends once its object has moved 2 cm from where the clause began and then moved under 5 mm over 15 steps, the same constants as `HandMoved`. Boundary between clauses: T6b's arm R. The last clause runs to the goal or 300 steps. Parses use `role_parse` on Um's rarest-rule lexicon, which reads all 18 instructions correctly; this path bypasses S1's refusing `Lexicon`.

**Chains, verdict.** M1 (K8, K2), M2 (K8, N3), M3 (N8, N1), M4 (K4, K8), M5 (K4, N8), M6 (N3, N8). No demo shows any of them, and each pairs a push with a grasp, in both orders. M3, M5 and M6 also use N8's new pairing.

**Chains, descriptive.** D1 (K8, K1) and D2 (K8, N1) place an object on the plate the first clause just pushed. The second skill then sees its target far from where its demos had it. Reported with no verdict.

**Verdict, familiar, over the six chains.** Chain minus the product of each skill alone, the same statistic and margin as T6b: BREAKS iff the CI upper bound falls below −.10; HOLDS iff the lower bound is at or above −.10; otherwise UNRESOLVED. A push alone comes from T12's V and a grasp alone from T11's Um.

**Known bias in the product.** A single push episode stops the instant the object enters the region, while a chain needs the goal to stay true after the hand moves on. So a push's solo rate overstates its chance in a chain. I fixed the statistic to match T6b and flag this before the read.

**Prediction, after reading smoke.** The smoke read seed 10 on states 0-1: M1 1 of 2, M3 1 of 2, D1 1 of 2. On state 0 the first goal was false when the push clause ended. I predict UNRESOLVED, with the chains near .55 against a product near .75, and D1 and D2 near .40.

### 27.1 Result: BREAKS (2026-11-06; results/flawed/sc_t14_read.json; rollouts sc_t14)

Development read, oracle positions, one kitchen, 3 seeds, 20 states per chain.

**Familiar, six chains: .411 against a product of .817, −.406 [−.617, −.189]. BREAKS.** I predicted UNRESOLVED near −.20, which was wrong. New layout: .128 against .186, −.058 [−.139, +.011]. The dependent chains D1 and D2: .283 against .817, new layout .042 against .125.

| chain | clauses | chain | product | first goal at transition | first goal at end | second goal at end |
|---|---|---|---|---|---|---|
| M1 | plate push, bowl to stove | .283 | .850 | .283 | .283 | 1.00 |
| M2 | plate push, cheese to cabinet | .233 | .817 | .267 | .267 | .833 |
| M3 | bowl push, cheese to plate | .033 | .750 | .433 | .433 | .050 |
| M4 | bottle to cabinet, plate push | .767 | .850 | .933 | .933 | .767 |
| M5 | bottle to cabinet, bowl push | .600 | .833 | .933 | .933 | .617 |
| M6 | cheese to cabinet, bowl push | .550 | .800 | .917 | .917 | .567 |
| D1 | plate push, bowl onto plate | .283 | .833 | .283 | .283 | .833 |
| D2 | plate push, cheese onto plate | .283 | .800 | .283 | .300 | .783 |

**Reading, post hoc.** The six chains split by order. With the push last, the chains sit near their product. With the push first, the push's own goal holds at the transition only about .28 of the time, against .85 alone, while the second skill runs well: M1's second goal reaches 1.00. D1 and D2 show the second skill finding a plate the first clause moved, at .83 and .78. That dependent placement is the strongest thing in this table, and the pushed plate's own goal is what sinks both chains.

**The bias I flagged is the likely cause.** A solo push stops when the object enters the region. The push clause keeps running until the object goes still. M3's second goal also falls to .05, and its geometry suggests why: the bowl's path to the region passes near the cheese's start at (−.05, .13). I have not checked either account, and §28 tests the first.

## 28. T15: a push clause that stops (protocol fixed 2026-11-06, before any non-smoke rollout; m3_prereg sc_t15)

**Trace, smoke states only, no stop at the goal.** K8 state 0: the plate enters the goal at step 152, sits 8 mm from the target's centre at step 160, and has left the region by step 180. N8 state 0: the bowl enters at step 90 and has left by step 120. K8's demos end at the goal, with the plate a median 1.8 cm from the target and still moving, so no row teaches the hand to stop. The trace also bears on M3: dragging the bowl moves the cheese from (−.03, .14) to (.04, .34) on state 0, so that chain's second goal starts from a displaced object. That is a physical conflict in the chain I chose, and the arms below do not touch it.

**One principle.** A carry ends where its demos' carries ended. The close mode already has this through its release. The open mode needs it stated.

**Arms, same chains, checkpoints and boundary as T14.** An open clause that is not last still ends when the object goes still, and it now also ends:
- **R:** once the object sits within rho of its target. Rho is the 90th percentile, over K8's 50 demos, of that distance at the demo's last row: 3.57 cm. Only K8 supplies open-mode demos, so one value serves.
- **G:** on the clause's own goal predicate. This is a ceiling, and it matches how a solo episode stops. A robot has no such signal.

**Verdict, familiar, six chains, per arm.** Chain minus product, as in T14: BREAKS iff the CI upper bound falls below −.10; HOLDS iff the lower bound is at or above −.10; otherwise UNRESOLVED. R carries the claim and G shows the ceiling. **Also reported:** each arm against T14 on the same chains, D1 and D2, and the new layout, with no verdict.

**Prediction, after reading smoke.** The smoke read seed 10 on states 0-1: both arms M1 1 of 2 and D1 1 of 2, passing state 0, which failed in T14. I predict R UNRESOLVED, chains near .65 against .82, a gain near +.25 over T14; G within .05 of R; D1 and D2 near .65; M3 stays low.

### 28.1 Result: BOTH BREAK (2026-11-06; results/flawed/sc_t15_read.json; rollouts sc_t15)

| arm | six chains, familiar | − product | vs T14 | new layout | D1, D2 familiar |
|---|---|---|---|---|---|
| R, demo radius | .489 | −.328 [−.525, −.155] | +.078 [−.028, +.217] | .136 | .500 |
| G, goal predicate | .483 | −.333 [−.533, −.125] | +.072 [−.011, +.189] | .144 | .467 |

**R BREAKS and G BREAKS.** I predicted R UNRESOLVED near .65 with a gain near +.25, which was wrong. G within .05 of R held, and M3 stayed low as predicted, at .017 for R. D1 and D2 reached about .50 against my .65.

**First goal at the transition, R:** M1 .55, M2 .52, M3 .67, against T14's .28, .27, .43. The stop roughly doubles it, and the demo radius matches the goal predicate, so the deployable rule loses nothing to the ceiling. Yet the ceiling itself reaches only about .53 on plate chains, against .85 for K8 alone.

**So overshoot was part of the cause.** Next suspect, post hoc: the stillness rule I kept in both arms. A solo push may pause and resume, as the gate's two segments in T11's episodes suggest, and the clause then ends too early. On smoke state 1, a clause ended by stillness at step 168, the step at which a solo K8 reached its goal on that state in T12's smoke. §29 removes the rule.

## 29. T15b: no end by stillness (protocol fixed 2026-11-06, before any rollout; m3_prereg sc_t15b)

**One change from T15's R.** An open clause that is not last ends only within rho of its target, or at 300 steps. A last clause runs as before. Same chains, states and checkpoints. No smoke read: this arm differs from R by one removed condition.

**Verdict, familiar, six chains.** Chain minus product, as before: BREAKS iff the CI upper bound falls below −.10; HOLDS iff the lower bound is at or above −.10; otherwise UNRESOLVED. **Also reported:** the arm against R, D1 and D2, the new layout, the rate at which the first clause ends, and episodes past the env's horizon.

**Prediction.** UNRESOLVED, chains near .60. On M1, M2, D1 and D2 the first goal at the transition rises from about .54 to about .75. M3 stays near 0 because of the cheese.

**This closes the block on these chains.** T14, T15 and T15b have all read the same chains and states, each fix designed on the last failure. I will not iterate further on them whatever the label, because the states have stopped being evidence for a stop rule.

### 29.1 Result: UNRESOLVED (2026-11-06; results/flawed/sc_t15b_read.json; rollouts sc_t15b)

**Familiar, six chains: .567 against .817, −.250 [−.472, −.072]. UNRESOLVED**, as predicted near .60. Against R: +.078 [+.006, +.183]. New layout: .158 against .186, −.028 [−.106, +.036].

| chain | clauses | T14 | R | T15b | product | first goal at transition, T15b |
|---|---|---|---|---|---|---|
| M1 | plate push, bowl to stove | .283 | .533 | .783 | .850 | .817 |
| M2 | plate push, cheese to cabinet | .233 | .483 | .633 | .817 | .750 |
| M3 | bowl push, cheese to plate | .033 | .017 | .017 | .750 | .700 |
| M4 | bottle to cabinet, plate push | .767 | .700 | .767 | .850 | .933 |
| M5 | bottle to cabinet, bowl push | .600 | .617 | .617 | .833 | .917 |
| M6 | cheese to cabinet, bowl push | .550 | .583 | .583 | .800 | .933 |
| D1 | plate push, bowl onto plate | .283 | .533 | .783 | .833 | .817 |
| D2 | plate push, cheese onto plate | .283 | .467 | .617 | .800 | .817 |

**Both removed causes were real.** The first goal at the transition rises on the plate chains from about .28 (T14) to .54 (R) to about .80 (T15b), above my predicted .75. Stopping at the demo's radius and refusing to end by stillness each contributed.

**Dependent chains.** D1 and D2: .700 against .817, −.117 [−.283, 0]. The second skill places an object on a plate at a position where its demos never saw one. This counts as a strong sign that parts read the moved target. It is a descriptive read with no verdict, and it was reached by iterating on these states.

**What remains, post hoc.** M3 stays at .017 because the bowl's drag plows the cheese away, a physical conflict in my chain selection and no failure of a part. Its removal would raise the mean, but the statistic stands as registered. M5 and M6 lose about .22, with the push as the last clause, and I have not explained that. A last clause inherits a scene changed by the first skill and an arm that has already returned, and the solo push's success at .83 starts from a fixed pose. On new layouts the push's own weakness dominates.

**Status of this block.** T11-T15b add a verb to the library with no retraining beyond one event rule, on one kitchen with oracle positions. The stop rule is the only part tuned across repeated reads of the same chains. These states have now shaped it, so any claim about it needs fresh states.

## 30. T8b: the retrained VLA (protocol fixed 2026-11-06, before training starts; m3_prereg sc_t8v)

**Why now.** PLAN_SCAFFOLD §7's stop rule reads: "If T8's retrained VLA matches the full system, the thesis as written fails, and we report it." Every VLA arm so far is frozen, and T8a had no VLA in its ladder. This read is the one that can refute the thesis.

**The arm.** pi0.5 trained from `pi05_base` on K1-K8 only, with `pi05_l90c`'s recipe and its own norm stats: `pi05_goal8` in `lifelong_ood/policy/l90c_config.py`. Batch 64, 3,000 steps, four A40s, about 45 A40-h, one seed, read at step 3,000. That length gives about 24k samples per task and 4.6 passes over the data. `pi05_l90c@5000` reached .959 parity at about 4k samples per task, so this budget is generous by that comparison.

**The data, and how it differs from U's.**
- The demos come from `physical-intelligence/libero`, the set `pi05_libero` trained on: 335 episodes, 41,569 frames (K1 49, K2 48, K3 46, K4 47, K5 36, K6 40, K7 36, K8 33). That set dropped some raw demos, while U fits all 50 per task. So the VLA sees about 16% fewer demos, which departs from the plan's "equal demos".
- The VLA reads pixels, while U reads oracle positions, also a departure from the plan's "same VLM". U's inputs are the easier ones.
- One seed against U's three. Comparison intervals resample tasks and states, with U averaged over its seeds.

So a VLA that matches U is decisive against the thesis, because the VLA carries the handicaps. A VLA below U leaves input difference as a live rival: it supports the rung and does not establish it. T7 (the stack from pixels) would be needed to close that.

**Rollouts.** Through the frozen read's harness: `experiments/serve_guided.py` in mode `ours` with seed 13, and `fc_n2_client.py` with `n2_branches_main.json`, all 16 compounds in both layouts, dev states 4-23.

**Verdict, new pairings N1-N6, familiar, VLA − U.** VLA MATCHES iff the CI lower bound is at or above −.10. RUNG HOLDS iff the CI upper bound falls below −.10. Otherwise UNRESOLVED. If the same comparison on known K1-K6 has its upper bound below −.10 and the label would read RUNG HOLDS, the label reads UNDERFIT, as T8a's rule does.

**Reported with no verdict.** The new layout; VLA − frozen `pi05_libero` on the same cells; K7, K8, N7, N8.

**Prediction.** RUNG HOLDS: VLA near .95 on known, near .45 on new pairings against U's .894; new layout near .15 on new pairings against U's .381. Near .45 follows the frozen read's .50 for pi0.5 on these pairings, with less data to generalize from.

## 31. K7's two phases: measured, not built (2026-11-06, offline on the demos)

**What the demos show.** The slot `wooden_cabinet_1_top_region` moves with the drawer, by .156 m along +y before the bowl's hold, and no other slot moves over 1.3 cm. The drawer starts moving near step 40 and stops near step 59. The hand sits at (−.008, +.151, +.015) from the slot throughout, with a spread of 7 mm in y. The first hold comes near step 127. Pre-hold rows carry a close command 9% of the time, so most demos pull with an open hand.

**What this means.** Opening is `HandMoved` on the drawer's slot, with the mode T12 already reads, so one event covers the grasp, the push and the pull. It differs from the push in one way: "open the top drawer" names no target entity. The clause's target is where the moved slot ended, relative to where it started.

**Why I stopped.** That needs one new slot, a target defined relative to the moved slot's start, and rows split at the first hold. A new slot changes the entity store's shape and invalidates every checkpoint's slot encoding unless it is guarded, and the object part for a drawer changes the atom lists. I am leaving this decision for the user, together with whether N7 belongs in the claim. U alone reads K7 .30 and N7 .00.

**User's decisions, 2026-11-06.** K7 and N7 join the claim, so the slot gets built. Grasp poses come from new demos that show them. T1 runs, because no such experiment exists in the repo. Fresh states are allowed. On moved objects, in the user's words: "if something's moved, it's assumed that it should move because of the robot (no other active actors in the scene)". I read that as keeping scenes with fixtures moved at reset out of the claim. The goal, in their words: "adopt structure into the traditional VLA routine to promote native composition in the hopes of being better than traditional VLAs at it". So the oracle library counts as evidence for mechanisms, and the claim rests on a VLA trained with that structure against a plain retrained VLA.

## 32. T1: actions per query on frozen VLAs (protocol fixed 2026-11-06, before any rollout; m3_prereg sc_t1)

**Question.** Does the composition failure follow chunk length? The plan's reading: if failures follow chunks, a VLA with variable chunking may fix them, which weakens the case for structure.

**Arms.** Frozen `pi05_libero` and `ar0s/groot_libero`, `serve_guided.py` in mode `ours`, seed 13, at 2 and 10 actions per query. The frozen read at 5 is the comparator. K1-K6 and N1-N6, both layouts, dev states 4-23. Same harness as the frozen read apart from `--replan`. Replanning at 1 is left out for cost.

**Verdict, per model and setting, new pairings, familiar, against 5.** GAIN iff the CI lower bound exceeds +.10. LOSS iff the upper bound falls below −.10. Otherwise NO CLEAR CHANGE. **Overall:** CHUNK-SENSITIVE iff any GAIN, otherwise NO GAIN FROM CHUNKING. **Also reported:** known tasks, the new layout, and the rate of grasping the right object.

**Prediction.** NO GAIN FROM CHUNKING, all four settings NO CLEAR CHANGE. The break atlas places the failures at the carry's binding, which a shorter query does not supply.

### 32.1 Result: NO GAIN FROM CHUNKING (2026-11-06; results/flawed/sc_t1_read.json; rollouts sc_t1)

| model | actions per query | new pairings, familiar | against 5 | grasped the right object | known, familiar | new pairings, new layout |
|---|---|---|---|---|---|---|
| pi0.5 | 5 (frozen read) | .500 | | .967 | .992 | .300 |
| pi0.5 | 2 | .450 | −.050 [−.250, +.150] | .950 | .983 | .183 |
| pi0.5 | 10 | .458 | −.042 [−.200, +.100] | .967 | .992 | .308 |
| GR00T | 5 (frozen read) | .167 | | .992 | .958 | .083 |
| GR00T | 2 | .158 | −.008 [−.117, +.100] | .975 | .950 | .067 |
| GR00T | 10 | .250 | +.083 [−.025, +.233] | .992 | .933 | .100 |

All four settings NO CLEAR CHANGE, as predicted. Overall **NO GAIN FROM CHUNKING**.

**Reading.** On new pairings in familiar scenes both VLAs grasp the right object at .95-.99 and then succeed far less often, at every setting. So query length does not move the failure, which comes after the grasp. This agrees with the break atlas and gives T16 its target.

**Limits.** The intervals allow an effect up to about +.15 for pi0.5 and +.23 for GR00T, so this rules out a large gain and leaves a modest one open. Replanning at 1 stayed out for cost. One lower bound falls under zero outside the verdict: pi0.5 at 2 on new pairings, new layout, −.117 [−.250, −.025].

**A bug in my job script, no data lost.** The client names a video folder from `--out`, so `x.json.tmp` produced a folder `x.json`, and my `mv` placed each result at `x.json/x.json.tmp`. The first read crashed on this, and I changed its path lookup after registration. The statistics did not change.

### 30.1 Training incident (2026-11-06)

T8b lost its node at 05:58 with a checkpoint at step 500. Its three chained retries and all of T16s's first attempts ran out of GPU memory at startup on gundam and kitt, within minutes. On starrysky the same config ran at 6.9 s/step. My job script exited 0 regardless, so Slurm showed them COMPLETED, and I reported training as running while nothing ran for about nine hours. I do not know why those two nodes ran out of memory. The replacement script logs GPU memory at start, refuses a GPU already holding over 1 GB, returns the real exit code, and excludes both nodes. T8b resumed from step 500 on cyborg at 15:34, on the same recipe and data.

## 33. T16: structure inside the VLA's routine (protocol fixed 2026-11-06, before training starts; m3_prereg sc_t16)

**The question the user set.** Does a VLA trained with composition's structure in its routine compose better than the same VLA trained the traditional way? Arm S against T8b's VLA is that comparison: same model, base weights, frames, batch, steps and harness, both from pixels.

**The one change, with two sides.**
- **Training.** Each frame's prompt names one role. Before the hold event: "object: the bowl". During it: "target: the plate". Frames are identical to T8b's, and only the per-frame text differs.
- **Test time.** The instruction parses into the same two prompts, and the event picks one at each query from the proprioceptive state and the last command.

So "put the cream cheese on the plate" reaches the policy as "object: the cream cheese" and then "target: the plate", both trained. At prompt level a new pairing holds nothing new. The carry never sees the object's name, and the reach never sees the target's. That targets the measured failure: V1 and the break atlas found the frozen VLA's carry bound to the held object, and hand-over at the hold repaired it.

**Where the structure comes from.** The lexicon is T11's, read from demos with no labels; the event is `ProprioHeld`, one threshold fit on this dataset's own proprioception. Both the lexicon's situations and T12's modes read simulator positions from recorded demo states, as U's do, and nothing reads them at test time: the policy sees pixels and proprioception, the same inputs as T8b's VLA.

**Limits I can name now.**
- K7, K8, N7 and N8 keep their whole instruction. Second clauses and open-hand verbs are not routed yet, and §34's K7 work comes after this read.
- K6's object phrase reads "the cream cheese in", because cheese appears in one training task. Test prompts read "the cream cheese". The mismatch handicaps S on the cheese pairings.
- The role tags "object:" and "target:" are hand-written. They are the one designed piece of text.
- Query-level event: the gate updates every 5 steps at test time, against every step in training.
- One seed per arm. N2 main states have shaped U and the lexicon rules, though neither VLA has ever read them through training.

**Verdicts, S − T8b's VLA.**
- **Composition, new pairings N1-N6, familiar.** STRUCTURE WINS iff the CI lower bound exceeds +.10. STRUCTURE LOSES iff the upper bound falls below −.10. Otherwise NO CLEAR DIFFERENCE.
- **Cost, known K1-K6, familiar.** NO COST iff the lower bound is at or above −.10. Otherwise COST.

**Reported with no verdict.** The new layout, per-task rates, K7, K8, N7, N8.

**Prediction.** STRUCTURE WINS: S near .75 on new pairings against the VLA near .45; NO COST, with S near .90 on known; on the new layout S near .25 against the VLA near .15, because the grasp limit there stays untouched.

### 33.1 Order changed: supervised first (2026-11-06; m3_prereg sc_t16s)

**User, in their words:** "This should start as supervised and then be unsupervised". I cancelled the training jobs for the arm above before either started, so no steps ran. The row `sc_t16` stays in the ledger, and that arm, with its dataset already built, becomes stage two.

**Stage one, arm Ss.** Everything above, with one change: role phrases come from each task's definition, hand-named from its goal atom ("object: the cream cheese", "target: the plate"). This removes K6's "in" artifact. Dataset `lol/libero_goal8_rsup`, config `pi05_goal8_rsup`.

**What stays unsupervised in this stage, and why.** The hold event stays proprioceptive, in training and at test. A contact label does not exist for these frames: they come from a different filtering of the demos than `demos_b.npz`, and test-time contact would need the simulator, which T8b's VLA does not have. K7, K8, N7 and N8 stay whole, because no phase label exists for these frames either.

**Verdicts and comparator.** The same as above, on Ss − T8b's VLA: composition on new pairings familiar, and cost on known familiar.

**Prediction.** STRUCTURE WINS, Ss near .78 on new pairings against the VLA near .45, a little above the unsupervised arm's predicted .75. NO COST, near .90 on known. New layout near .25.

**Stage two, once stage one reads.** The registered arm above, on its own dataset. Their difference is the cost of learning the phrases, which on these instructions is one word.

## 34. The single-GPU ladder: rungs R0 and R1 (protocol fixed 2026-11-10, before any rollout; m3_prereg sc_t34)

**Why.** T8b and T16 need four A40s on one node. Between 2026-11-06 and 2026-11-10 that queue gave T8b 80 training steps: the jobs waited two days, one node handed over occupied GPUs, and the one run that started lost its node after 18 minutes. A read that cannot train cannot test the theory. One A40 schedules within minutes, so the ladder of docs/THEORY_COMPOSITION.md §3 runs under a recipe that fits one. T8b and T16 stay queued as the full fine-tune forms.

**The recipe, shared by every rung.** pi0.5 from `pi05_base`, with SigLIP and the PaliGemma LLM frozen; the action expert and its projections train. Batch 32, 6,000 steps (192k samples, T8b's count), constant 5e-5 after 1,000 warmup steps, no EMA, one seed, read at step 6,000. Configs `pi05_goal8_ae` and `pi05_goal8_rsup_ae` in `lifelong_ood/policy/l90c_config.py`; each uses its parent's dataset and norm stats, so the frames match T8b's and T16's exactly. I chose a frozen VLM over LoRA because openpi's converter to PyTorch does not merge LoRA weights, and the rollout harness serves PyTorch.

**Two queues, one run read.** Each rung is submitted twice with the same config and seed: experiment `main` on overcap and experiment `tl` on tail-lab. The first of the two to save step 6,000 is the run that gets exported and read, and I cancel the other. This is a scheduling device and adds no second seed.

**Changed 2026-11-10, before any rollout.** The `tl` runs started within a minute and the `main` runs did not, so I cancelled `main` for both rungs. Later the same evening I cancelled R1 here near step 1,900, unread, to free its GPU: §37's ladder has its own R1 on frames matched to R2. R0 here was to run on as a check on §37's render pipeline; I paused it too, near step 1,900, because overcap's estimate for a single A40 was a day and a half and §37's fourth rung needed the lab GPU. Its checkpoints stay, and it resumes if a GPU frees. This section's R1 − R0 verdict will not be read.

**Rungs.**
- **R0.** Whole instructions, as T8b.
- **R1.** T16's supervised stage: given role phrases, switched by `ProprioHeld`, in training and at test.

**Rollouts.** T8b's and T16's harness and seeds, all 16 compounds in both layouts, dev states 4-23. Results in `results/flawed/sc_t34`, named by rung.

**Verdicts, R1 − R0, paired bootstrap over tasks and states (2,000 draws).**
- **Fit gate.** If R0 scores under .80 on known K1-K6, familiar, the label is UNDERFIT and nothing below is read as a statement about composition.
- **Composition, new pairings N1-N6, familiar.** STRUCTURE WINS iff the CI lower bound exceeds +.10. STRUCTURE LOSES iff the upper bound falls below −.10. Otherwise NO CLEAR DIFFERENCE.
- **Cost, known K1-K6, familiar.** NO COST iff the lower bound is at or above −.10. Otherwise COST.

**Reported with no verdict.** Each rung's success on the cheese pairings (N1-N3) and on the bottle pairings (N4-N6); grasp and place rates; the new layout; K7, K8, N7, N8; each rung against frozen `pi05_libero` and, when they exist, against T8b and T16.

**Predictions.** They differ from §33's, and I leave §33's standing. By the theory (THEORY_COMPOSITION §2), role phrases close the language channel and leave the held object in view, and V1's arm S already showed that phrase switching alone does not repair the frozen model. So: **NO CLEAR DIFFERENCE** and NO COST. R0 near .45 on new pairings, with the cheese pairings N1 and N2 at or under .25 and the bottle pairings N4 and N5 above .70. R1 within .10 of R0. Both at or above .90 on known tasks.

**Limits.**
- A frozen VLM is a different baseline from T8b's full fine-tune. CRAFT reports that lighter fine-tuning composes better than full fine-tuning (46.3% against 8.4% for pi0.5), so R0 here may sit above T8b. Every comparison stays inside this recipe.
- One seed per rung. States 4-23 have shaped U and the lexicon; neither VLA has trained on them.
- R2 (role views) needs a re-rendered dataset with entity masks and gets its own protocol.

## 35. E0: do partner counts order new-pairing success? (protocol fixed 2026-11-10, before the read; m3_prereg sc_e0)

**Question.** THEORY_COMPOSITION §2 claims a VLA composes where the training data broke the tie between object and target. In K1-K6 the cream cheese has one target and the wine bottle two. So the bottle pairings (N4-N6) should succeed more often than the cheese pairings (N1-N3).

**What I had seen.** The pattern came from frozen pi0.5's intact arm, per task (RT §3.1). I had seen GR00T's pooled figure on new pairings (.167) and pi0.5's at 2 and 10 actions per query (.450, .458), and none of their per-task values. Those arms carry the test; pi0.5 intact is reported as descriptive.

**Arms, existing tracks, no rollouts.** GR00T frozen at 5, 2 and 10 actions per query; pi0.5 at 2 and 10. Familiar layout, dev states 4-23.

**Estimator.** `gap` = mean success on the bottle pairings minus mean success on the cheese pairings. The CI resamples states jointly across tasks (2,000 draws); tasks stay fixed, because each group has three.

**Verdict.** Per arm: PARTNER ORDER HOLDS iff the CI lower bound exceeds 0; REVERSED iff the upper bound falls below 0; otherwise UNCLEAR. Overall: **SUPPORTED** iff GR00T's three settings averaged HOLD and no arm is REVERSED; **REFUTED** iff GR00T's average is REVERSED; otherwise UNRESOLVED.

**Reported with no verdict.** Per-task success, the rate of grasping the right object, success given that grasp, the new layout, and N3 (cheese on the cabinet, the one target with two holders) against the mean of N1 and N2.

**Prediction.** SUPPORTED, every arm HOLDS. GR00T's gap is smaller than pi0.5's because its level sits near the floor.

**Limits.** Two objects stand for the two partner counts, so the object and its count are confounded: the cheese could be harder to place for reasons of shape. E2, which gives the cheese a second partner, is the test that separates them. Six tasks on one scene.

### 35.1 Result: REFUTED (2026-11-10; results/flawed/sc_e0_read.json; ledger row sc_e0 at c51e5412)

My prediction, SUPPORTED, was wrong. GR00T reverses the order at all three settings.

| arm | N1 | N2 | N3 | N4 | N5 | N6 | cheese | bottle | gap [95% CI] | label |
|---|---|---|---|---|---|---|---|---|---|---|
| GR00T, 5 per query | .10 | .20 | .70 | .00 | .00 | .00 | .333 | .000 | −.333 [−.433, −.250] | REVERSED |
| GR00T, 2 | .20 | .15 | .60 | .00 | .00 | .00 | .317 | .000 | −.317 [−.433, −.217] | REVERSED |
| GR00T, 10 | .15 | .45 | .85 | .00 | .05 | .00 | .483 | .017 | −.467 [−.567, −.383] | REVERSED |
| GR00T, pooled | | | | | | | .378 | .006 | −.372 [−.428, −.317] | REVERSED |
| pi0.5, 2 | .20 | .25 | .65 | .45 | .90 | .25 | .367 | .533 | +.167 [.000, +.333] | UNCLEAR |
| pi0.5, 10 | .10 | .10 | .70 | .85 | .90 | .10 | .300 | .617 | +.317 [+.167, +.467] | HOLDS |
| pi0.5, 5 (seen before; descriptive) | .10 | .20 | .60 | .85 | .85 | .40 | .300 | .700 | +.400 [+.233, +.583] | |

Overall **REFUTED**: GR00T's pooled arm is REVERSED.

**What this takes away.** The object's partner count does not order success across models. GR00T grasps the bottle at .95-1.00 and then places it on a new target 1 time in 180. So the sentence in THEORY_COMPOSITION §2 that I proposed to centre the paper on, "a compositional architecture should need one partner where the VLA needs two or more", has no support as a law about objects. I withdraw it there.

**What the same table shows, with no verdict.** N3, cheese on the cabinet, beats the mean of N1 and N2 in all six rows, by +.43 to +.60. The cabinet is the one target that two different objects were carried to in training. I registered this contrast as descriptive, and I had seen pi0.5's value before, so it stands as an observation on five new rows and one old one. It fits the place side of the shortcut account: a place that training showed with two held objects accepts a third. The bottle on the plate and the stove breaks even that reading for pi0.5 (.85-.90 on targets with one holder), so no count rule fits both models.

**What survives.** The failure sits after the grasp in every row (grasped-right .85-1.00 on familiar scenes). V1 and the hand-over still say that the held object's identity is what the place binds to. The ladder tests that directly and does not depend on a count law. Its diagnostic changes: the read is new pairings as a whole, and the cheese-against-bottle split is reported only.

**New layout, with no verdict.** GR00T's gap is −.133 [−.250, −.033] at 5 per query and unclear at 2 and 10; grasps of the right object fall to .20-.60 there for both models.

## 36. Q0: the prototype operator on a policy with pure parts (protocol fixed 2026-11-10, before any search or read; m3_prereg sc_q0)

**Why.** The user wants the lab's prototype-discovery paper used as the composition operator, and expects a hyperparameter search to matter (THEORY_COMPOSITION §4b). Plan v4 ran that operator on VLAs at one setting and it failed. Two causes were possible: the operator, or the VLA's field. U's parts are pure by construction, so running the operator on U separates them. It is also the first search of the operator in action space.

**Task.** One human demonstration of a known task, no instruction. Work out which entity fills the object role and which the target role.

**Operator, as in the paper.** At subsampled rows of the demo, for each candidate assignment and noise level: ascend U's score from several starts, fit a diagonal Gaussian at each mode, and score each candidate by how well its prototypes cover the demo's action chunk per dimension. The object comes from rows before the hold and the target from rows during it; the phase comes from the proprioceptive hold event. `experiments/sc_q0.py`.

**Baselines on the same rows.** The paper's retrieval rule (lowest flow-matching loss); the repo's motion heuristic, which reads recorded positions; chance.

**Split.** Per task K1-K6: demos 0-9 for the search, on the seed-10 checkpoint; demos 25-49 for the read, on all three checkpoints; demos 10-24 unused. U trained on all of these demos, so the read tests the operator on a model that knows the tasks. It says nothing about new pairings.

**Search.** 50 settings fixed in code before any search run: plan v4's setting, the paper's ColorMNIST setting, and 48 draws with seed 0 from a grid over levels, noise range, starts, initialisation, steps, step size, K, temperature, and three fields added after the smoke run below: the scoring rule, the extent estimate and its floor. The setting with the highest joint accuracy on the search split is picked, ties to the cheaper one.

**The three added fields.**
- **Scoring rule.** `uncapped`: a candidate's score is the per-dimension best over all its prototypes. `capped`: the same over at most K prototypes per candidate. `pooled`: the paper's own operator, greedy selection of K over every candidate's prototypes, then each candidate gets the product-of-experts weight of its selected prototypes.
- **Extent.** `posterior` is the paper's (the Tweedie Jacobian at the mode). `marginal` is the curvature of the noisy marginal. `deconv`, plan v4's, is that curvature with the noise variance subtracted. For a sharp isolated mode, `posterior` and `deconv` go to zero and `marginal` goes to σ²/α².
- **Floor** on the variance: 1e-4, 1e-3 or 1e-2.

**What I saw before registering.** A smoke run on two search-split demos (K1 and K5, demo 0), with six decisions per setting.
- Plan v4's setting got 3 of 6 right. The retrieval rule and the motion heuristic got all of them.
- The cause, on those demos: 23-31% of the variances sat at the 1e-4 floor, so a human demo's ordinary distance from a sharp mode cost the true part more than a wrong part's scattered modes gained. The true part did have the nearest prototype (rmse .07-.11 against .17-.28).
- Raising the floor to 1e-3, or the `marginal` extent, or `pooled` scoring, each turned the misses into hits. The paper's setting and two cheap draws with floor 1e-2 got 6 of 6.
- Plan v4's own read reported "46% of extents sit at the 1e-4 floor" on pi0.5 (FRAMEWORK, history). So part of that failure may have been this setting. Q1 tests it.

**Verdicts, read split, bootstrap over demos (2,000 draws).**
- OPERATOR PARSES iff the picked setting's pooled joint accuracy (object and target both right, target scored with the inferred object) has its CI lower bound above .80. OPERATOR FAILS iff the upper bound falls below .50. Otherwise PARTIAL.
- SEARCH MATTERS iff picked minus plan v4's setting has its CI lower bound above +.10. Otherwise SEARCH DOES NOT MATTER.

**Prediction, written before the smoke run.** OPERATOR PARSES, at or above .90. SEARCH DOES NOT MATTER: each pure part's field has one mode toward its own entity, so four starts should find it. The retrieval rule also scores above .90, so on U the operator shows no advantage over flow loss.

**Prediction, revised after the smoke run and before any search.** OPERATOR PARSES at the picked setting, at or above .90. **SEARCH MATTERS**: plan v4's setting lands near .5 on joint accuracy, because of the floor. The picked setting uses floor 1e-3 or 1e-2, or `pooled` scoring. The retrieval rule still matches the picked setting within .05, so the operator's gain on U is over its own old setting and not over flow loss.

**Limits.** Oracle positions. Known tasks only. A pass shows the operator can read parts off a pure policy; it does not show composition of anything new.

### 36.1 Result: OPERATOR PARSES, SEARCH MATTERS (2026-11-10; results/flawed/sc_q0_read.json; ledger row sc_q0 at 4a1a65c0)

450 demo reads (25 held-out demos × 6 tasks × 3 checkpoints), no instruction.

| | object | target, inferred object | joint [95% CI] |
|---|---|---|---|
| plan v4's setting | .900 | .444 | .436 [.387, .484] |
| picked setting (index 31) | 1.000 | .998 | **.998 [.993, 1.000]** |
| retrieval rule (flow-matching loss) | 1.000 | 1.000 | 1.000 |
| motion heuristic (recorded positions) | 1.000 | 1.000 | 1.000 |
| chance | .25 | .155 | .039 |

Picked minus plan v4's setting: **+.562 [+.513, +.611]**. On the search split the two scored 1.000 and .317.

**The picked setting.** Three noise levels between .05 and .40, four starts (half from noise), 50 steps of size .02, K = 5, temperature .05, `capped` scoring, `marginal` extent, floor 1e-2. It costs under a third of plan v4's setting (600 against 1,920 field evaluations per start). With the same prototypes, `uncapped` scoring gives .998 and the paper's `pooled` scoring .989.

**Against my predictions.** The one written before the smoke run said SEARCH DOES NOT MATTER and was wrong. The revised one held on every count: the picked setting at or above .90, plan v4's near .5, a floor of 1e-2, and the retrieval rule within .05 of the operator.

**Reading.**
- The user's expectation holds on this policy: the setting decides whether the operator works. Plan v4's extent collapses to the floor at sharp modes, and the operator then misreads the target in more than half of the demos.
- So plan v4's offline failure on VLAs had two possible causes, and this removes neither: the setting was bad, and the VLA's field may also be impure. Q2 runs the operator on the ladder's checkpoints at the picked kind of setting to separate them. (The October closed-loop pilot, `q3_pilot`, is a separate case: its variances were clipped at 1e-3 and it failed for a reason its docs put on holding one prototype set over a whole phase. See THEORY_COMPOSITION §4b's correction.)
- On U the operator gains nothing over flow-loss retrieval. Both read the parts off, because U's parts can be addressed one role at a time. The paper's advantage over retrieval rests on retrieval returning whole trained combinations, which is the VLA's situation and not U's.
- With no verdict: the product-of-experts mean sits closer to the demo's chunk (rmse .04-.05) than U's own sample under the true parts (.16-.24). It conditions on the demo, so this shows the composer reproduces a chunk it was given.

**Limits.** Known tasks, demos U trained on, oracle positions. The search and the read share tasks and differ in demos.

## 37. The matched ladder on re-rendered frames: R0, R1, R2, R2b (protocol fixed 2026-11-10, before training starts; m3_prereg sc_t37)

**Why a second ladder.** Role views need entity masks, and the published frames have none. `experiments/sc_rv.py` replays all 400 raw demos of K1-K8 state by state and renders RGB and instance masks together: 52,181 frames after the no-op rule, against 41,569 in the published set. Every rung trains on these same frames, so the rungs differ only in the prompt and the paint. §34's R0 on the published frames keeps running as a check on this render pipeline; its R1 is redundant with the R1 here.

**Rungs, all under §34's recipe (frozen VLM, trainable action expert, batch 32, 6,000 steps, one seed).**
- **R0** `pi05_g8v_plain_ae`: whole instructions, images untouched.
- **R1** `pi05_g8v_rsup_ae`: given role phrases switched by `ProprioHeld`, images untouched.
- **R2** `pi05_g8v_rview2_ae`: R1's prompts, and on every routed frame, in both cameras, every movable object except the active role's filler is painted flat magenta by its exact mask. Fixtures are never painted. Before the hold the active filler is the task's object. During it the active filler is the target when the target is movable; when the target is a fixture all four movable objects are painted, the held one included.
- **R2b** `pi05_g8v_rbox_ae`: R2 with each painted object's mask replaced by its bounding rectangle, so shape reduces to position and extent.
- K7 and K8 stay whole and unpainted in every rung.

**A rule I built first and dropped before training.** The first role-view dataset (`rview`) painted only the other role's filler: the target before the hold, the held object during it. Looking at its contact sheets I saw two faults in my rule. Painting the target makes the object phase differ by target, although unpainted it looks the same for every target, because each scene shows all the objects. And painting only the held object leaves a tell during the hold: which objects remain on the table. The rule above makes each phase's view independent of the other role's filler, up to the painted shapes. The `rview` dataset stays on disk, untrained.

**At test.** The simulator's instance masks reach the server with each query, and the server paints by the same rule from the parsed roles and the event. This is the supervised stage: masks and entity names are given. The unsupervised stage would take masks from the locator.

**Rollouts.** §34's harness, all 16 compounds in both layouts, dev states 4-23, results in `results/flawed/sc_t37`.

**Verdicts, paired bootstrap over tasks and states (2,000 draws), each on N1-N6 familiar with §34's margins.**
- **Fit gate.** R0 under .80 on known K1-K6 familiar reads UNDERFIT, and nothing below is read as a statement about composition.
- **Primary: R2 − R0.** STRUCTURE WINS iff the CI lower bound exceeds +.10. STRUCTURE LOSES iff the upper bound falls below −.10. Otherwise NO CLEAR DIFFERENCE.
- **Which part carries it: R2 − R1 and R1 − R0**, same labels.
- **Shape: R2b − R2**, same labels.
- **Cost, known K1-K6 familiar, each of R1, R2, R2b against R0.** NO COST iff the lower bound is at or above −.10.

**Reported with no verdict.** Grasp and place rates, per-task success, the new layout, K7, K8, N7, N8, and each rung against frozen `pi05_libero`, against §34's R0 and against U (.894 on new pairings, from simulator positions).

**Predictions.** R0 near .45 on new pairings, at or above .90 on known. R1 − R0: NO CLEAR DIFFERENCE (§34's reasoning). **R2 − R0: STRUCTURE WINS**, with R2 near .70, because hiding identity from the place is what the hand-over did; I put this at a little better than even. R2b − R2: NO CLEAR DIFFERENCE, with R2b a few points higher if silhouette shape still gives the held object away. NO COST for R1; R2 and R2b lose under .10 on known tasks.

**What would count against the theory.** R2 and R2b both land within .10 of R0 while R0 passes the fit gate. Then hiding identity in the image does not buy composition in a VLA, and the hand-over's gain came from something else the external part had, such as its entity frame.

**Limits.**
- Masks are privileged at test time. R2 gets them and R0 does not.
- A painted held object still shows its outline in R2 and its extent in R2b, and the fingers' opening still differs by object.
- The paint changes what a frozen vision tower sees; the tower cannot adapt, only the action expert can.
- One seed per rung, six pairings, one scene, development states.

### 37.1 Result on R0, R1 and R2: STRUCTURE WINS (2026-11-11; results/flawed/sc_t37_read_r012.json; rollouts sc_t37)

**Order of reads, disclosed.** R2b's training lost three segments to nodes that had no GPU (heistotron, cyborg, major: "No devices were found"), so its export failed. I resumed it from step 2,500 with those nodes excluded. The primary verdict (R2 − R0) does not involve R2b, so I ran the registered estimator on R0-R2 now (`--tag r012`). The four-rung read runs, as registered, when R2b's rollouts exist. Nothing in the mapping changed.

**Fit gate passes.** R0 scores .85 on known K1-K6 familiar, against a threshold of .80.

| N1-N6, familiar | R0 plain | R1 role phrases | R2 role views |
|---|---|---|---|
| success | .025 | .225 | **.425** |
| grasped the right object | .408 | .925 | .975 |
| placed, given the right grasp | .061 | .243 | .436 |
| cheese (N1-N3) / bottle (N4-N6) | .05 / .00 | .35 / .10 | .52 / .33 |
| known K1-K6 | .85 | .758 | .758 |

| verdict | composition, N1-N6 familiar | cost, K1-K6 familiar |
|---|---|---|
| **R2 − R0 (primary)** | **+.400 [+.167, +.625]: STRUCTURE WINS** | −.092 [−.308, +.117]: COST |
| R1 − R0 | +.200 [+.025, +.433]: NO CLEAR DIFFERENCE | −.092 [−.258, +.067]: COST |
| R2 − R1 | +.200 [−.058, +.458]: NO CLEAR DIFFERENCE | +.000 [−.142, +.158]: COST |

Under the registered rule, COST means the lower bound fell below −.10. Each point estimate is under .10, and the intervals are wide.

**My predictions.**
- **Primary verdict:** right (STRUCTURE WINS).
- **Levels:** wrong. I expected R0 near .45 and R2 near .70. R0 nearly never composes (.025), and R2 reaches .425.
- **R1 − R0:** right (NO CLEAR DIFFERENCE).
- **Costs:** wrong. I predicted NO COST, and the rule labels all three COST.

**What the ladder shows, rung by rung.**
- **R0, trained on the eight known tasks alone, fails new pairings at both stages.** It grasps the wrong object on more than half of them, the shortcut taken all the way back to the grasp, and places .06 of what it grasps.
- **Role phrases (R1) fix the grasp:** .41 → .93.
- **Role views (R2) double the place given a right grasp:** .24 → .44.
- Each part removes a different break in the atlas's sense. The theory's split, language purity for the grasp and visual purity for the place, fits these numbers. R2 − R1 alone does not clear the margin.

**Against frozen pi0.5** (reported with no verdict): .50 on new pairings, trained on 40 LIBERO tasks with the full model. R2, trained on 8 tasks with a frozen vision-language model, sits −.075 [−.417, +.258] from it. In the new layout every rung is near 0, against .30 for frozen pi0.5: the eight-task recipe does not cover new placements.

### 37.2 Registered four-rung read and the T37w control (2026-11-12; sc_t37_read.json, sc_t37_read_w*.json)

**The four-rung read** gives the same primary verdict as §37.1: STRUCTURE WINS, R2 − R0 = +.400 [+.167, +.625].

**R2b, boxes, against R2, silhouettes:** −.275 [−.458, −.108], STRUCTURE LOSES. R2b scores .150 on new pairings and .683 on known tasks. My prediction (NO CLEAR DIFFERENCE) was wrong. Covering each non-active object's whole extent, which removes its shape along with its identity, costs about two thirds of R2's gain.

**T37w (§41), the AC-VLA-style control.**

| N1-N6 familiar | success | known K1-K6 |
|---|---|---|
| R1 | .225 | .758 |
| R2w, wrist grey on held frames, filled at test | .242 | .617 |
| R2w0, same checkpoint, unfilled at test | .175 | .308 |
| R2 | .425 | .758 |

- **Primary, r2 − r2w:** +.183 [−.092, +.475], NO CLEAR DIFFERENCE. I gave ENTITY VIEW WINS 45%, conditional on r2 − r0 winning; it did not happen. On this evidence R2's gain cannot be told apart from a whole-wrist mask.
- **r2w − r1:** +.017 [−.292, +.283], NO CLEAR DIFFERENCE, as predicted.
- **r2w0 − r1:** −.050 [−.217, +.100], NO CLEAR DIFFERENCE, as predicted.
- **The read script printed UNDERFIT** for these two reads because it gates on their first rung (R1, .758). The registration applies T37's gate on R0 (.85, passes), so the composition labels stand.
- **Serving R2w without the fill it was trained on** drops known tasks to .308, so AC-VLA's train-only rule fails in this pixel form.

**Reading.** Hiding identity in the image helps a VLA on new pairings (R2 − R0). Which form of hiding matters is unresolved at this sample size: silhouettes beat boxes, and silhouettes do not clearly beat a whole-wrist mask.

## 38. PU: the parts policy from perceived poses (protocol fixed 2026-11-10, before any non-smoke rollout; m3_prereg sc_pu)

**Question.** U's new-pairing success (.894 familiar, §11) reads object and site poses from the simulator. Does it keep that success when every movable object's pose comes from the agentview camera instead? This removes the privileged input that separates U from the VLA rungs of §37.

**Arms.** `experiments/sc_pu.py roll`, U's checkpoints `pp_sc_U_s{10,11,12}`, sc_t4's combos (K1-K6, N1-N6, both layouts), N2 main states.
- **oracle:** simulator poses, through sc_pu's own copy of fa1h_roll's episode. This is the sanity arm.
- **percep:** movable objects from agentview instance masks plus depth, back-projected with the camera's intrinsics and extrinsics, with per-object offsets fitted on demos 0-24 of K1-K6. The held object comes from its mask (bowl, cheese) or from rigid attachment to the hand (bottle), as calibration chose. Partly hidden objects keep their last visible estimate. Fixture sites come from their masks, because rollout envs jitter fixtures by 1-2 cm. Yaw is the training mean for round objects. Calibration choices are in `results/flawed/sc_pu_calib.json`.
- The pose errors on demos 25-49 are in `sc_pu_check.json`. Medians are .03-.49 cm. The p90 reaches 2.1 cm during the hold and 6.9 cm for the bottle after release.

**Read.** `experiments/sc_pu_read.py`. Four cells, known / new pairings × familiar / new layout. Each cell gives percep − oracle, bootstrapped over tasks and states (2,000 draws).
- **Sanity:** oracle must agree with sc_t4's U episodes on at least .90 of episodes, else INVALID.
- **COSTS** if any cell's upper bound is below −.10. **HOLDS** if every lower bound is at or above −.10. **UNRESOLVED** otherwise.

**Prediction.** HOLDS, with differences of −.03 to −.05 and the largest per-task costs on the bottle tasks (N4, N5). Second most likely is UNRESOLVED, from a bottle cell or the new layout.

**Seen before registration.** Calibration, the check, and 12 smoke episodes. Smoke ran seed 10, states 0-1, K1 and N4 familiar, under ref, oracle and percep; all succeeded, and oracle matched ref step for step. sc_t4's U results were also read.

**Limits.**
- Simulator masks stand in for a segmenter. The test removes pose privilege, not mask privilege. This is the same privilege R2 has in §37, so the two can be compared.
- The drawer stays simulator-read; no task here uses it.
- Calibration uses familiar-layout demos only.

### 38.1 Result: HOLDS (2026-11-10; results/flawed/sc_pu_read.json; rollouts sc_pu; ledger row sc_pu)

All 144 rollout files completed: 2,880 episodes, no errors. The oracle arm agrees with sc_t4's U episodes on .985 of episodes, so the sanity gate passes.

| cell | percep | oracle | percep − oracle | CI |
|---|---|---|---|---|
| known, familiar | .853 | .878 | −.025 | [−.069, +.011] |
| new pairings, familiar | .883 | .897 | −.014 | [−.061, +.025] |
| known, new layout | .417 | .408 | +.008 | [−.042, +.058] |
| new pairings, new layout | .383 | .378 | +.006 | [−.044, +.058] |

**My prediction (HOLDS) held, but its detail was wrong.** The bottle tasks did not carry the cost: N4 moved +.02 and N5 −.02. The largest per-task drops were K5 (−.07, the bottle on the rack, where the check's post-release error was largest) and N6 (−.05).

**What this establishes.** U's new-pairing success no longer depends on simulator poses. On the same N2 main states and seeds, frozen pi0.5 scores .50 and frozen GR00T .17 on the new pairings in the familiar layout (HOLD_GATE §5, intact). The parts policy, reading object poses from one camera, scores .883.

**What it does not establish.**
- **Masks:** they still come from the simulator, the same privilege R2 has in §37.
- **Instructions:** U reads its roles from the instruction through a lexicon, while a VLA reads the raw instruction.
- **The new layout:** it stays near .38-.42 for every system (pi0.5 .30). The atlas places that failure at the grasp, tied to where objects start, and pairing does not explain it.

## 39. Q2: the prototype operator on the ladder's VLAs (protocol fixed 2026-11-10, before any search or read; m3_prereg sc_q2)

**Question.** Q0 (§36.1) showed that the paper's operator parses a demo into parts when the policy's parts are pure (U). Q2 asks the same of a VLA. From one rendered demo of a known task and no instruction, does prototype discovery over a §37 rung's action field pick the phrases the demo is made of? For the object role it reads 6 rows before the grasp; for the target role, 6 rows during the hold.

**Rungs and candidates** (`experiments/sc_q2.py`).
- **r0:** the six K1-K6 instructions as whole prompts, over all 12 rows.
- **r1, r2, r2b:** 3 object phrases and 5 target phrases. On r2 and r2b each candidate gets its own painted view.

**Search.** Ten settings, run on demos 0-4 of each task: plan v4's, Q0's picked setting 31, and eight one-field variations of it. The pick is the best joint accuracy, with ties going to the lower index.

**Read.** Demos 25-34 of each task, run once, at the picked setting and at setting 0. The baselines are flow-loss ranking (the same noise for every candidate) and chance.

**Verdicts per rung, on joint accuracy.**
- **Parsing:** PARSES iff the lower bound is at least .50; FAILS iff the upper bound is below .50.
- **Against retrieval** (picked − flow loss): OPERATOR BEATS RETRIEVAL iff the lower bound exceeds +.10; RETRIEVAL BEATS OPERATOR iff the upper bound is below −.10; otherwise NO CLEAR DIFFERENCE.
- **Search** (picked − setting 0): SEARCH MATTERS iff the lower bound exceeds +.10.
- Comparisons across rungs carry no verdict.

**Prediction.**
- r1, r2 and r2b PARSE, with joint accuracy .60-.85, at a setting whose levels reach σ .5. The smoke showed that role phrases move the field only there.
- r0 is UNRESOLVED, because the scene sets the mode.
- NO CLEAR DIFFERENCE against retrieval on every rung.
- SEARCH MATTERS on the role rungs.

If this holds, the operator is a parser with the same power as retrieval, and the contribution is the role channel that makes parsing possible.

**Seen before registration.** Smoke on the 500-step checkpoint (§ prereg row), and Q0.

**Limits.**
- Known tasks only. Parsing a recorded demo of a new pairing (`sc_q2.py demo`) waits on initial states that neither trial list uses.
- One seed per rung.

### 39.1 Result on r0, r1 and r2 (2026-11-11; results/flawed/sc_q2/<rung>/read.json; r2b pending, see 37.1)

Each rung picked setting 3: sc_q0's setting 31 with noise levels .3-.9.

| rung | operator, picked (joint) | flow loss | picked − flow loss | setting 0 | picked − setting 0 |
|---|---|---|---|---|---|
| r0 (instructions) | 1.000 | 1.000 | .000 | .800 | +.200 [+.117, +.283] |
| r1 | .900 [.833, .967] | 1.000 | −.100 [−.167, −.033] | .383 | +.517 [+.400, +.634] |
| r2 | 1.000 | 1.000 | .000 | .600 | +.400 [+.300, +.500] |

**Verdicts.**
- **Parsing:** PARSES on all three.
- **Against retrieval:** NO CLEAR DIFFERENCE on all three. On r1 the operator trails flow loss by .10, and the upper bound (−.033) misses the −.10 bar.
- **Search:** SEARCH MATTERS on all three.

**My predictions.**
- PARSES on r1 and r2: right.
- Tie with retrieval: right.
- Setting 3 picked: right.
- r0 UNRESOLVED: wrong. Its whole-instruction candidates parse known demos perfectly.
- Search NOT SHOWN on r0: wrong.

**Reading.**
- The search turned the paper's operator from failing (setting 0, plan v4's: .38-.80) into a parser.
- The levels that matter sit at noise .5 and above, which the smoke had shown.
- On known tasks it never beats retrieving by flow loss. As on U (§36.1), the operator adds nothing over retrieval when the candidates are given.

## 40. PU-L: the parts policy from learned masks (protocol fixed 2026-11-10, before any non-smoke rollout; m3_prereg sc_seg_pu)

**Question.** §38 removed pose privilege, and its masks still came from the simulator. Does U keep its success when the masks come from a learned segmenter?

**The segmenter** (`experiments/sc_seg.py`) is a 4.9M-parameter U-Net trained from scratch on the re-rendered K1-K8 demos 0-39, both cameras, with copy-paste and affine augmentation, for 46 min on one A40. Its classes are background, the four movable objects and the three fixtures.

| movable-object IoU | bowl | cheese | bottle | plate | mIoU |
|---|---|---|---|---|---|
| validation (demos 40-49) | .991 | .986 | .988 | .994 | .990 |
| held-out frames, familiar layout | .988 | .981 | .974 | .992 | .984 |
| held-out frames, new layout | .984 | .954 | **.715** | .942 | .899 |

The held-out frames come from smoke states 0-3 with scripted reaches. In the new layout the bottle fails mostly in the wrist camera (.678), where it is labelled as the cabinet. In agentview, the camera U reads, the bottle scores .972, and the plate's detection rate is .865 when the arm partly hides it.

**Arm.** `sc_seg.py pu` runs sc_pu's percep pipeline unchanged, with `seg_std` replaced in-process by the segmenter's prediction on the RGB render. Depth, calibration and the Estimator stay the same, and so do the combos and states.

**Read.** `experiments/sc_seg_pu_read.py`. The primary comparison, per cell, is learned − percep: the same pipeline with only the mask source changed. COSTS if any cell's upper bound is below −.10; HOLDS if every lower bound is at or above −.10; otherwise UNRESOLVED. learned − oracle is reported with no verdict.

**Prediction.** HOLDS in the familiar layout, at about −.03 or better. In the new layout, a cost of −.03 to −.08 on the plate-target tasks and the bottle tasks. Overall: UNRESOLVED 55%, HOLDS 40%, COSTS 5%.

**Seen before registration.** The IoU tables; one learned-arm smoke (N4 familiar, states 0-1, 2/2 succeeded); learned-mask R2/R2b smokes on a format checkpoint (uninformative); §38.1.

**Limits.**
- The segmenter is trained with simulator labels on simulator renders, so this is the supervised stage.
- Depth is still the simulator's rendered depth.
- U still reads its roles from the instruction through the lexicon.

### 40.1 Result: HOLDS (2026-11-11; results/flawed/sc_seg_pu_read.json; rollouts sc_seg/pu_learned; ledger row sc_seg_pu)

All 72 files and 1,440 episodes completed with no errors. Preemption requeued two jobs after they had written their files; the requeued runs found the files and exited.

| cell | learned | sim masks (percep) | oracle | learned − percep | CI | learned − oracle |
|---|---|---|---|---|---|---|
| known, familiar | .853 | .853 | .878 | +.000 | [−.050, +.039] | −.025 |
| new pairings, familiar | .881 | .883 | .897 | −.003 | [−.042, +.039] | −.017 |
| known, new layout | .406 | .417 | .408 | −.011 | [−.069, +.033] | −.003 |
| new pairings, new layout | .375 | .383 | .378 | −.008 | [−.050, +.031] | −.003 |

**My prediction was half right.** HOLDS in the familiar layout came true. I expected a new-layout cost of −.03 to −.08 and gave UNRESOLVED the higher odds; the cost was −.01. The new layout's ceiling is the grasp (BREAK_ATLAS §8), not perception. Per task, the largest drops were K2 new (−.10) and N1 familiar (−.05, a plate target). They sit inside the cell CIs.

**What this establishes.** The parts policy reaches .881 on new pairings in the familiar layout using one camera, a learned segmenter and calibrated depth: no simulator masks and no simulator poses. Frozen pi0.5 scores .50 and frozen GR00T .17 on the same states and seeds.

**What remains privileged or hand-made.**
- The depth render.
- The lexicon that maps an instruction to roles. It is learned from the demos (§3: words grounded through the hold event; one demo per task suffices; it refuses safely), but its rule was designed on these 16 instructions.
- The parts' typed state, which is built by hand.
- The segmenter's training labels, which come from the simulator.

## 41. T37w: the whole-wrist-mask control (protocol fixed 2026-11-10, before training starts; m3_prereg sc_t37w)

**Why.** AC-VLA (arXiv 2607.15714) is the closest prior repair. It masks the wrist view's tokens during the place, at training time only, and so hides the held object together with all of the wrist camera's geometry. R2 instead paints entities and keeps geometry. Without this control, a gain for R2 could be AC-VLA's gain.

**Rung R2w** (`experiments/sc_rv_w.py`, arm `rwrist`; config `pi05_g8v_rwrist_ae` inserted at runtime by `experiments/c_train_pi05_w.py`).
- R1's role phrases. On every held frame of K1-K6, the whole wrist image is replaced by grey 128 (about 0 in openpi's pixel range). Agentview stays unchanged, and K7 and K8 stay whole.
- The dataset equals rsup's on every other frame: 13,327 of 16,240 held frames are filled. Its norm stats are byte-identical to rsup's.
- The training recipe is T37's, and the config differs from R1's in name and data only.

**Two test-time arms of one checkpoint.**
- **r2w:** filled at test, as trained. The fill is gated by the `ProprioHeld` step that switches the prompt.
- **r2w0:** served unfilled, which is AC-VLA's own rule.

The fill is a pixel proxy for AC-VLA's token mask, not a copy of it; a token-drop version was not built.

**Read.** `sc_t34.py read --dir sc_t37`, T37's estimator and margins, N1-N6 familiar.
1. **Primary, r2 − r2w:** ENTITY VIEW WINS iff the lower bound exceeds +.10; WHOLE-VIEW MASK WINS iff the upper bound is below −.10; otherwise NO CLEAR DIFFERENCE.
2. **r2w − r1 and r2w0 − r1:** whether whole-wrist masking adds to role phrases.
3. **Cost on known tasks,** r2w against r0.

T37's fit gate applies.

**Prediction.**
- r2w − r1 and r2w0 − r1: NO CLEAR DIFFERENCE. Agentview still shows the held object in every carry frame, so the shortcut stays open.
- r2 − r2w: ENTITY VIEW WINS exactly when T37's r2 − r0 wins (45%), NO CLEAR DIFFERENCE 50%, WHOLE-VIEW MASK WINS 5%.

**Limits.**
- This is a pixel proxy for a token mask.
- AC-VLA's other parts, LLM decomposition and its phase cut at gripper change, are not reproduced.
- One seed.

### 41.1 Result: see 37.2

## 42. Q2-N: parsing a demo of a new pairing (protocol fixed 2026-11-11, before any run; m3_prereg sc_q2n)

**Question.** Q2 parses demos of known tasks. The paper's claim is about unseen combinations: from one demo of a pairing no training task holds, does the operator recover its parts?

**Demos.** One per new pairing, N1-N6, familiar layout. U recorded them (oracle, `experiments/sc_pu_record.py`) from extra start state 24 (`experiments/fc_n2_extra.py`: LIBERO's sampler at seed 600000 + k, k = 24-35, after a check that it reproduces stored states 0-1 exactly). All six succeeded on the first attempt. No trial list uses these states.

**Run.** `experiments/sc_q2_demo.py`: sc_q2's `parse_demo`, unchanged, at each rung's picked setting and at setting 0, with the flow-loss baseline on the same rows.

**Verdict per rung,** on the joint count J out of 6 at the picked setting: PARSES NEW iff J ≥ 5; FAILS NEW iff J ≤ 2; otherwise MIXED. Flow loss is reported beside it with no verdict. r0's candidates are trained instructions, so its joint count is 0 by construction; its object and target matches are reported only.

**Prediction.**
- r1, r2 and r2b are MIXED or PARSE NEW, equal to flow loss within one demo.
- If a rung stores the pair (P1), the cheese demos parse their target worse than the bottle demos.
- r0 matches the object on about half.

**Limits.** Six demos, one state each. This is parsing only: no closed-loop composition from the parse.

### 42.1 Result on r0, r1 and r2 (2026-11-11; results/flawed/sc_q2/<rung>/demo_new.json; r2b pending)

| rung | operator, picked: joint J of 6 | flow loss J | verdict | misses |
|---|---|---|---|---|
| r0 | 0 (by construction); object 2, target 4 | 0 | (none) | picks K1, K2, K6 or K4 |
| r1 | **5** | 6 | **PARSES NEW** | N6: target "plate" instead of "bowl" |
| r2 | **5** | 5 | **PARSES NEW** | N4: target "bowl" instead of "plate" (flow loss makes the same miss) |

At setting 0, r1 gets 4 of 6 and r2 gets 3 of 6.

**Prediction.** MIXED or PARSES NEW, and equal to flow loss within one demo: right. That the cheese demos would parse worse than the bottle demos: wrong. All three cheese demos parse on both rungs, and each miss is a bottle target.

**Reading.**
- On role-channel VLAs, one demo of a pairing no training task holds is decomposed into its trained parts. Plain r0 cannot express the answer: its nearest trained instruction keeps the object or the target, never both.
- The operator and retrieval over role phrases do equally well. The parts in the role vocabulary make the new pairing parseable; the operator is one way to read them.

## 43. T37-L: R2 with learned masks (protocol fixed 2026-11-11, before any rollout; m3_prereg sc_t37l, commit ad0271da)

**Question.** Does R2's checkpoint keep its gain when the role views are painted from the learned segmenter (§40) instead of the simulator? `sc_seg.py serve`: the simulator maps reach only an IoU log.

**Read.** `sc_t34.py read --dir sc_t37 --rungs r0 r2 r2l --primary r2l:r2 --tag l`, on N1-N6 familiar.
- MASKS HOLD iff the lower bound of r2l − r2 is at or above −.10.
- MASKS COST iff the upper bound is below −.10.
- Otherwise UNRESOLVED.
- r2l − r0 is reported with sc_t34's labels.

**Prediction.** UNRESOLVED 50%, MASKS HOLD 35%, MASKS COST 15%. Wrist masks of the bottle are weak, and R2 paints both cameras. r2l − r0: STRUCTURE WINS, near +.30.

## 44. HP: the hand-over hybrid without simulator state (protocol fixed 2026-11-11, before any main-state rollout; m3_prereg sc_hp)

**Question.** HAND_OVER §6.1 (H) raised new pairings from .50 to .96 on pi0.5 and from .17 to .93 on GR00T. Its place part read simulator poses, and the hand-over fired on the simulator's contact signal. HP removes both.

**The system** (`experiments/sc_hp.py`).
1. The frozen VLA acts from the start, with the same seeds, states and prompts as H.
2. At the first onset of U's proprioceptive hold (ProprioHeld), control passes to U (`pp_sc_U_s10`) for the rest of the episode.
3. From then on, U reads poses from learned masks plus depth (sc_seg pred, sc_pu Estimator), stepped from step 10 of the episode.

**What still comes from the simulator at test time:**
- the depth render, drawn with sites hidden;
- the camera matrices;
- the entity list and the fixture-to-site map.

At training time, the calibration constants and the segmenter's labels also come from the simulator.

**Read** (`experiments/sc_hp_read.py`; its `--selftest` reproduces §6.1). Per model, N1-N6 familiar:
- **Primary, HP − intact:** BEATS THE VLA iff the lower bound exceeds +.10.
- **HP − H:** NO PRIVILEGE COST iff the lower bound is at or above −.10; PRIVILEGE COST iff the upper bound is below −.10; otherwise UNRESOLVED.
- **Known tasks:** NO KNOWN COST iff the lower bound of HP − intact is at or above −.10.

**Prediction.**
- **pi0.5:** BEATS THE VLA (about .88 against .50), NO PRIVILEGE COST.
- **GR00T:** BEATS THE VLA (about .80 against .17). HP − H UNRESOLVED, because the bottle's held-object estimate is off by 3-4 cm in rigid mode.

**Disclosed before registration.** In the smokes, ProprioHeld fired on failed cheese lifts in 3 of 4 cheese episodes. U then made the grasp itself, so on those episodes HP is "VLA approach + U grasp and place". I added no debounce after seeing this. The read reports the share of early hand-overs.

**Limits.**
- One U seed, one scene, 20 states.
- The partly privileged inputs listed above.


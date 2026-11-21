# RAA-F Route A+ generator: why S0-5 yield was 1/30, and a PROPOSED rule change

Written 2026-11-20.  Diagnosis only: no full 50-per-object set, no prereg row, no training.  fa1g_plan.py is untouched.
Code: experiments/raafp_gen.py (copy of fa1g_plan.py `gen`'s working path; modes `area`, `dry`).  Results: results/flawed/raaf/planner/
(area*.json, dry_*.json); per-step states of every kept dry-run demo: /coc/flash10/ksingara3/.claude_tmp/raaf/planner/<rule>/<obj>/demo_<k>.npz.
Seeds are the plan's (93_000_000 + 1000 obj + k), so every rule sees the same 30 base layouts per object.

**PROPOSED deviation from docs/notes/raaf_plan.md 3.2 / 3.4.  Needs orchestrator approval before any use.**

**APPROVED 2026-11-20 by the orchestrator (rule h_j), attributed to the orchestrator.  Reasons recorded for the launch record: (1) the plan's clearance (12 cm beyond other footprints) was unsatisfiable: free area 0.03-0.2% of the layout box, a spot in 10-27% of layouts, S0-5 yield 1/30; (2) the release keypoint bug: the K-task release offset (hand relative to the target site) put the object 3-18 cm off the spot or released it in the air; (3) the 40-step transit cap was too short for 30-45 cm carries.  The wine bottle is dropped by the plan's own k1 rule (yield 23% < 30%), leaving bowl + cheese: A+ as registered with 2 objects.**

## 1. Verdict

Two independent defects, both in how Route A+ was specified, neither in the planner:

1. **The spot rule is nearly unsatisfiable** (question a).  The 12 cm clearance around the other movables is the killer.
2. **The plan's release keypoint puts the OBJECT at spot + (K-task release offset), not on the spot** (question b).  This is why even
   found spots mostly failed once the spot rule was relaxed.  Plus a 40-step cap on the transit phase that is too short for 30-45 cm carries,
   and a reach limit on the east edge of the N2 box.

raaf_plan_dry.py is a faithful copy of the fa1g working path (checked below), so its 1/30 is real.

## 2. (a) Free table area

Box = N2 layout box, 41.0 x 34.0 cm = **1394 cm2**.  Measured on a 1 cm grid in the 30 real layouts per object (same seeds as S0-5).
Fraction of the box excluded by each constraint ALONE (orig rule, mean over layouts, all three objects similar):
start 20 cm: 48-54%; K targets 20 cm: 29-70% (bowl 70%); fixtures (FIXR + 12): 42%; other movables (MOVR + 12): **90-93%**; stove front +5 cm: 12%.

| rule | bowl mean / median cm2, layouts with any spot | cheese | bottle |
|---|---|---|---|
| orig (plan, MOVR + 12 cm) | 0.5 / 0, 10% | 3.2 / 0, 27% | 3.1 / 0, 18% |
| removing only the K-target term | 1.2, 23% | 4.2, 28% | 3.1, 18% |
| removing only the stove-front term | 1.1, 20% | 6.4, 31% | 5.2, 26% |
| edge gap 2 cm (center >= R_other + R_own + 2 cm), box + [7,5,0,0] cm | 25 / 14, 70% | 49 / 24, 89% | 51 / 28, 92% |
| same, box exactly N2's | 10 / 0, 46% | 25 / 14, 79% | 42 / 19, 85% |

So under the plan's rule the free area is **0.03-0.2% of the box, the spot exists in 10-27% of layouts** (dry runs here: bowl 4/30, cheese 9/30,
bottle 5/30 layouts with a spot, 18/90 against S0-5's 7/30).  The rule as written (12 cm beyond the footprint, ignoring the object's own radius,
so the real edge gap is 12 cm minus its radius, and 18-22 cm center-to-center from every other movable) cannot be met in a box that holds four
movables and three fixtures at 2 cm gaps.  No single term is to blame except the movable clearance: relaxing any one other term leaves < 7 cm2.

## 3. (b) Why the attempts fail

### 3.1 Faithfulness of raaf_plan_dry.py
Same phases, thresholds, waypoint jitter, controller (FP.ctrl), layout (FP.layout), keypoints (a1g_keypoints.json), 300-step cap.  Differences are
only the destination (fixed spot at zref) and the stop (run to phase 8 then 20 settle steps instead of stopping at LIBERO's success predicate).
Control (rule `ktarget` in raafp_gen.py: the same loop, destination = the K task's own target, success = LIBERO predicate, 10 attempts, same seeds):
bowl 8/10, cheese 6/10, bottle 6/10 (1 no-layout), against fa1g's past 59% / 67% / 35%.  The copy reproduces fa1g within sampling error.

### 3.2 Failure classes (orig rule, 30 attempts per object, dry_orig.json)
| | no_spot | grasp_miss | drop_in_transit | release_off_spot | tilt | no_layout | kept |
|---|---|---|---|---|---|---|---|
| bowl | 26 | 0 | 1 | 1 | 0 | 0 | 2 |
| cheese | 20 | 2 | 1 | 1 | 1 | 1 | 4 |
| bottle | 22 | 4 | 1 | 0 | 0 | 3 | 0 |

There is no IK in this controller (OSC delta commands); no timeout class occurred under the orig rule.
Of 90 attempts, 68 are lost to "no spot" (the rule), 6 kept; the rest is the planner.

### 3.3 With the spot rule relaxed (rules prop*, dry_prop*.json): the planner's own faults
Relaxing the spot rule alone (edge gap 3 cm, box +5 cm all round) gave 8/4/0 kept of 30 (bowl / cheese / bottle).  The failures were:
- **Release keypoint puts the object off the spot.**  The K-task release offset is the hand relative to the TARGET SITE, and it contains an xy part
  (K4: -5.8, +3.4 cm) and a z part measured against the target's height.  Using it at a table spot lands the object 3-9 cm from the spot, or releases it
  in the air (cheese: hand 7 cm above the table).  `drop_in_transit` / `release_off_spot` were 12-23 of 30 per object.  Bottle: 0 kept in every
  such variant (16 of 27 attempts with a spot ended 6-18 cm off).
- **Reach**: spots east of x = 0.14 or north of y = 0.22 (the box extended on all sides) always failed (arm stalls 6-12 cm short, phase 4/5 timeout).
  Starts east of x = 0.04 fail for bottle (0/20) and half the time for cheese (the lift does not clear).
- **Phase-4 cap**: carries of 30-45 cm (new spots, vs 10-33 cm in the K demos) hit the 40-step limit.
- **grasp_miss**, independent of the spot: bottle 30-50% (the same as fa1g's K4, which needed 142 attempts for 50), cheese 20-35%, bowl 7-13%.
  This is the planner's ceiling, not a spot problem.

## 4. PROPOSED rule change (needs orchestrator approval)

Rule `h_j` in experiments/raafp_gen.py.  Smallest set of changes that fixed the measured causes; the A+ purpose is kept: shift >= 20 cm from the
object's start and from every one of its K targets, not N8's stove-front, >= 20 cm moves, prompt "target: the table".

1. **Clearance**: replace "12 cm beyond the other movables' / fixtures' footprint" by an **edge gap of 2 cm** (center >= R_other + R_own + 0.02), the
   same gap N2's own layout sampler uses between footprints.  (Physical need: the object must not touch a neighbour.)  This change alone raises the
   spot-exists rate from 10-27% to 46-85% of layouts.
2. **Sampling box**: N2's box grown by [xlo 7, ylo 5, xhi 0, yhi 0] cm (towards the robot; the east / north edge is at the reach limit).  Spot-exists 70-92%.
3. **Release by object, not hand**: in phases 4-5 the hand target is (spot + the live hand - object offset), at object height = its resting height + 1 cm.
   Replaces "K-task release offset applied to the spot".  The K-task release ROTATION and carry height are kept.  (This is the plan's fallback variant
   with the reference changed from the target site to the object; it removes the xy offset and the target-height dependence.)
4. **Transit cap** 100 steps for phase 4 (was 40) and episode cap 360 (was 300).  Kept demos average 112 / 139 / 145 steps, so rollouts' 300-step
   horizon is not affected.
5. **Layout redraw** (cheap, no sim rollout): redraw the layout when no spot exists, and when the object starts east of x = 0.04 (reach).  The
   redraws are counted: **layout draws per attempt are 1.8 / 1.8 / 1.4**.  Yield below is per ROLLOUT attempt, as S0-5 defined it; per layout draw it is
   19/54, 11/53, 7/42.

Not changed: d_start = d_tgt = 20 cm, stove-front +5 cm, keypoints, planner phases, success filter (3 cm, 20 deg, 1 cm settle, 2 cm others), seeds.

### Validated yield (30 attempts per object, seeds 93_000_000 + 1000 obj + k, rule h_j, dry_h_j.json)
| object | kept / 30 | yield | steps (kept mean) | classes |
|---|---|---|---|---|
| bowl | 19 | **63%** | 112 | grasp_miss 4, release_off 3, drop 2, collision 1, tilt 1 |
| cream cheese | 11 | **37%** | 139 | grasp_miss 11, collision 3, release_off 3, tilt 2 |
| wine bottle | 7 | 23% | 145 | grasp_miss 13, drop 6, tilt 3, release_off 1 |

**>= 30% on 2 of 3 objects (bowl, cheese), so K1 (plan section 9) does not kill A+ but drops the bottle** under this rule.  All 37 kept demos have a
shift of 19-38 cm (median 26) from the object's start.  Wilson 95% for 11/30 is 22-54%, so cheese is above 30% in the point estimate only.

Variants tried (all 30 per object, results/flawed/raaf/planner/dry_*.json), kept bowl / cheese / bottle:
orig 2/4/0 (of 30; 6 total); prop (gap 3, box +5 all round, K-task release) 8/4/0; prop_e10 7/6/0; prop_g02 6/5/0;
h_a..h_d (hold release, 40-step cap) 12/-/8, 14/6/6, 8/-/-, 7/7/4; h_e / h_f / h_g (hold + 100-step cap) 13/6/8, 14/6/6, 11/7/7;
h_i (h_j without the reach filter) 19/7/7; **h_j 19/11/7**.  The hold release is the large step for bowl (8 -> 14) and bottle (0 -> 6-8);
the redraw for bowl and cheese (13-14 -> 19; 6 -> 7-11).  A start-gap filter (5 cm to neighbours, h_h) gave bowl 20/30 but needed about 4x more layout
draws and was stopped before cheese and bottle finished; it is not part of the proposal.

### Bottle
23% is at fa1g's own K4 ceiling (35% with the K4 target and no spot rule): half of the attempts fail at the grasp (bottle at the box edge, zmax lift
0).  If A+ must keep three objects, the cheapest further change is a bottle-only start filter (x <= 0.04 and y >= -0.08 or so), not measured here.

## 5. Cost of 50 kept demos per object (rule h_j), CPU only (the generator uses no GPU)

Measured per rollout attempt incl. layout redraws: bowl 10.8 s, cheese 12 s, bottle 11.6 s.  Per kept demo: bowl 17 s, cheese 33 s, bottle 50 s.
- Bowl 50 kept: about 80 attempts, 0.24 CPU-h.  Cheese: about 136 attempts, 0.46 CPU-h.  Bottle: about 217 attempts, 0.70 CPU-h.
- Bowl + cheese only (100 demos): about 0.7 CPU-h; all three: about 1.4 CPU-h (plan: 2.5 CPU-h).  Each is one single-process sbatch job (-c 4, <= 1 h).
- GPU: none for generation.  Render / convert of 100 (or 150) demos stays at the plan's about 2 GPU-h (1.4 for 100); not re-measured.
- Per-step states are saved (states_0 = sim state BEFORE action t, actions_0, init_state, spot, seed; last states row = final state); the 37 kept dry-run demos
  are in /coc/flash10/ksingara3/.claude_tmp/raaf/planner/h_j/.  sc_rv.render needs the K task's own bddl env (results/flawed/n2_bddl/K{1,4,6}.bddl, as generated), and a
  10-step dummy wait precedes row 0 (`wait_steps` saved); the render step must `set_init_state(init_state)` and step 10 dummies, or start from states_0[0].
  I did NOT run a replay check (stored vs replayed ee within 1 mm, plan k3).

## 6. What I did not check
- Replay fidelity of the saved states (k3), and any sc_rv render on them.
- Whether training on the h_j demos is acceptable: the objects start only in the west 3/4 of the box (x <= 0.04), and spots avoid the east edge.
- Spot coverage: spots cluster toward the robot side (west) of the box, 0.07 m beyond N2's box; N1-N8 test targets are untouched by construction (they are K-target exclusions
  + stove front), but the spot distribution has not been compared to the N-set target positions beyond the 20 cm K-target exclusion.
- A bottle-only filter, or a fixed grasp fix.

## 7. LAUNCH RECORD (2026-11-20): Route A+ and A+ctrl built, registered and launched

Approved deviation (orchestrator, attributed to the orchestrator): spot rule h_j.  Reasons: (1) the plan's clearance was unsatisfiable (free area 0.03-0.2% of the box); (2) the release keypoint bug (the K-task release offset put the object 3-18 cm off the spot, or released it in the air); (3) the 40-step transit cap.  The wine bottle is dropped by the plan's own k1 rule (23% < 30%), so A+ has two objects: bowl, cream cheese.

- **Demos** (raafp_gen.py dry --rule h_j --keep 50 --base 94000000, jobs 4819295 / 4819296): bowl 50 kept in 79 rollout attempts (63%, 131 layout draws), cream cheese 50 in 163 (31%, 230 draws).  Seeds 94,000,000 + 1000 obj + k (obj 0 bowl, 1 cheese), layout redraws + 50,000,000 j; the dry run used 93,000,000 + ... (disjoint; no other generator used 94M).  Failure classes bowl: collision 6, release_off 7, grasp_miss 7, drop 5, tilt 2, unsettled 2; cheese: grasp_miss 59, release_off 21, collision 16, drop 10, tilt 7.  Per-step states in /coc/flash10/ksingara3/.claude_tmp/raaf/gen_aplus/<obj>/demo_<k>.npz; summaries results/flawed/raaf/planner/gen_hj_*.json.
- **Replay check (k3)** (raafp_rv.py render, job 4819464, replay_check_gen_aplus.json): restoring each stored state reproduces its end effector to **0.000 mm in 100 of 100 demos** (the sc_rv chan check).  Action replay does not: one-step median 15.3 mm (max 101 mm), open-loop median 21 mm (max 686 mm; cheese demos 24-72 mm).  Genuine K1 demos give the same order (one-step 9.8-11.7 mm, open-loop 3.4-18 mm, replay_check job 4819324 log), so non-replayability from a state is a property of the simulator state / controller, not of the planner; the images and tokens come from the stored states and the labels are the executed actions.  The strict reading "stored vs replayed by actions within 1 mm" fails for genuine demos too and was not applied.
- **Datasets** (raafp_rv.py convert / check, jobs 4819540 / 4819541 / 4819914 / 4819915): lol/libero_g8v_raacp (A+) and lol/libero_g8v_rview2p (A+ctrl), each 500 episodes = 400 base + 100 planner, **61,649 frames** (52,181 base + 9,468 planner, of which 5,306 held frames carry "target: the table").  Checker: base episodes identical to lol/libero_g8v_raac / rview2 on state and actions (400 of 400), planner state dims, finite, |action xyz| <= 1, token role 2 exactly on the held frames, painted views present; token |dx dy dz| on the last held row median 1.1 cm (max 9.7 cm).  Target-role token = spot at the object's resting height - hand - in-hand offset.
- **Configs** (additive): pi05_g8v_raas_lp_ae (RAA-sL's model and servo constants, RAA-c norm stats, weights from pi05_libero, data lol/libero_g8v_raacp), pi05_g8v_rview2_lp_ae (rview2 model, rview2 stats, weights from pi05_libero, data lol/libero_g8v_rview2p); `--g8v raas_lp | rview2_lp` in c_train_pi05.py (raas_lp uses raac_export).  Edits made after Route B's launch commit a4712e75 landed and the diff on both files was empty.
- **Smoke** (20 steps, exp smoke, tail-lab): A+ job 4819964 step-0 loss **.0486** (RAA-sL .0556, k limit .1112), state_proj in the parameter list, assets raac; A+ctrl job 4819990 step-0 loss **.0475**, assets rview2.
- **Prereg**: row tag **raafp**, ledger/m3_prereg/m3_prereg__libero_goal__raafp__s0__d6e484d0.json (job 4819998), predictions experiments/prereg/raafp_pred.txt: A+ passes probe 28% (point share .50), A+ dev N within +-.05 of RAA-sL 55% / above by > .05 15%, A+ dev K >= RAA-sL - .05 70%, A+ctrl dev N > .70 40%, no route passes 52%.
- **Chains**: experiments copies /coc/flash10/ksingara3/.claude_tmp/pp/raafp_train.sh, raafp_chain.sh (6 x 4 h afterany, overcap, exp main, 6,001 steps, batch 32).  A+ (raas_lp): 4820007-4820012; A+ctrl (rview2_lp): 4820013-4820018.  No tail-lab chains: three of my jobs already run there (raaf_tl1, raafb_tl1, ctx_s3f), and the fourth slot is left to Routes A / B.
- Not done: the probe, dev rollouts, the A+ vs A+ctrl contrast; nothing here is a result.  The 5 early A+ test demos in /coc/flash10/ksingara3/.claude_tmp/raaf/rvp_test and results/flawed/raaf/planner/replay_check_h_j.json (2 demos x 2 objects, the render test) are test files.

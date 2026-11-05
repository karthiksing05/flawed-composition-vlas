# The break atlas (PLAN_PARTS §10 items 18-19)

The user's goal: "find where VLAs break, and see how a partonomic architecture can fix them", in terms of within-chunk binding errors and cross-chunk composition errors. This test assigns every failed episode to the first stage that breaks, on two real VLAs, and asks which stages break because the pairing is new.

Design fixed 2026-11-04, before any atlas rollout. m3_prereg tag `pp_atlas`. A post hoc development read with no training. It changes no earlier verdict.

## 1. Runs

| system | tasks | states and seeds | role |
|---|---|---|---|
| pi0.5, frozen | N1-N6, K1, K2, K3, K4, K6; both layouts | N2 main states 4-23, N2's seeds | verdict |
| GR00T, frozen | the same | the same | verdict |
| P_min, seeds 10-12 | the same | the same | reported, no verdict |
| scripted servo, planner seed 10 | the same | the same start states | reported, no verdict |

That is 880 VLA episodes, about 7 A40-h at V1's rate. K5 runs nowhere here because no new pairing shares its target. The simulator state is read after each step for analysis, and no system's inputs change.

## 2. The stage rule

Definitions, for the instructed object: `held` is `FB.Held`, the signal used across this project. `d` is the object's horizontal distance to the instructed target. The hold's onset is the first held step, and `dmin` is the smallest `d` over held steps. Each failed episode gets exactly one stage, tested in this order.

| stage | rule | group |
|---|---|---|
| **GRASP** | never held | within-chunk |
| **HAND-OFF** | held; `dmin` > 0.10 m; `d` shrank by less than half from the onset | cross-chunk |
| **TRANSPORT** | held; `dmin` > 0.10 m; `d` shrank by at least half | cross-chunk |
| **RELEASE** | arrived (`dmin` ≤ 0.10 m), then either still held at the end, or last held more than 0.03 m from the target | within-chunk |
| **PHYSICS** | arrived, and last held within 0.03 m of the target, and still failed | neither |

**Why these groups.** This is my reading of the user's two terms, and the user may correct it. A pairing joins the object's motor chunk to the target's. Hand-off and transport are failures to get from one chunk into the other. Grasp and release happen inside one role's chunk, where the action must fit the object and target it is bound to.

## 3. The control and the verdict

A stage counts as a composition error only if it breaks more on the new pairing than on the known tasks that share its parts. Each new pairing is compared with its two known partners, so the object mix matches:

| new | object partner | target partner |
|---|---|---|
| N1 cheese → plate | K6 | K1 |
| N2 cheese → stove | K6 | K2 |
| N3 cheese → cabinet | K6 | K3 |
| N4 bottle → plate | K4 | K1 |
| N5 bottle → stove | K4 | K2 |
| N6 bottle → bowl | K4 | K6 |

**Estimator.** For a model, layout and stage or group, `E` is the mean over N1-N6 and states of: the new pairing's rate minus the mean of its two partners' rates. Rates are per episode, counting all episodes. The CI is the bootstrap over the six pairings and the 20 states jointly, 2,000 draws, paired.

**Rule.** A stage or group BREAKS iff the CI lower bound is above 0 and `E` > 0.10.

**Verdict, for each model and layout**, from the two groups:

| within-chunk breaks | cross-chunk breaks | verdict |
|---|---|---|
| yes | no | **WITHIN-CHUNK** |
| no | yes | **CROSS-CHUNK** |
| yes | yes | **BOTH** |
| no | no | **NEITHER** |

**Reported with no verdict:** each stage's `E`; per-task stage rates; the rate at each stage among episodes that reached it; subtypes (grasp: another object moved, or none; release: never let go, or let go away from the target); at the last held step, the end effector and the object relative to the target and the object's tilt; agreement with recorded episodes; P_min's and the servo's stage rates and `E`.

**Prediction (mine; my record on this project is poor).** WITHIN-CHUNK for both models in the familiar layout, carried by RELEASE. NEITHER in the new layout, because layout already breaks the known partners.

## 4. What this cannot show

- The stage is where the failure first shows. An earlier cause can surface later: a poor grasp can cause a failed release.
- Competing stages mask each other: an episode lost at GRASP cannot fail at RELEASE. The reached-stage rates help with this, and they carry no verdict.
- 0.03 m is the horizontal check for bodies. Stove and cabinet targets are regions, so PHYSICS and RELEASE may trade episodes there. The known partners share these targets, which limits the effect on `E`.
- The bowl appears as a target only in N6 and K6, and K6 is both a partner and the cheese's object source.
- The 0.10 m arrival radius admits objects on the plate's rim.
- P_min runs on oracle state and the servo on exact positions, so neither row shows that parts fix a VLA's vision. Their rows show which stages those structures clear.
- This pass does not re-test the object × target reversal in RIVAL_TESTS §5.4, which needs arm W with tracking.
- One scene and two models.

## 5. Registered result (2026-11-04; results/flawed/pp_atlas_read.json), held pending §6

Rollouts agree with N2's recorded success on 1.00 of pi0.5's episodes and 0.995 of GR00T's.

| model | layout | within-chunk E [95% CI] | cross-chunk E [95% CI] | verdict |
|---|---|---|---|---|
| pi0.5 | familiar | +0.18 [+0.05, +0.33] | +0.33 [+0.13, +0.53] | **BOTH** |
| GR00T | familiar | +0.39 [+0.15, +0.66] | +0.43 [+0.19, +0.67] | **BOTH** |
| pi0.5 | new | +0.05 [−0.13, +0.21] | −0.00 [−0.15, +0.14] | **NEITHER** |
| GR00T | new | +0.04 [−0.16, +0.24] | +0.04 [−0.09, +0.18] | **NEITHER** |

My prediction for the familiar layout (WITHIN-CHUNK) was wrong under this mapping, and my prediction for the new layout (NEITHER) held. In the familiar layout HANDOFF and RELEASE break for both models; GRASP, TRANSPORT and PHYSICS do not.

Two features of this output need checking before the familiar verdict carries an interpretation. §6 registers those checks.

## 6. Two checks on the atlas (design fixed 2026-11-04, after §5; m3_prereg tag pp_atlas2)

Both use the recorded tracks. No new episodes. They change no registered verdict in §5; check A decides how the cross-chunk half of it may be read.

### 6.1 Check A: does HANDOFF mean what its label says?

**Why.** The rule calls an episode HANDOFF when, over held steps, the object stayed over 0.10 m from the target and `d` shrank by less than half. It carries the reading "held, and never went to the target". But N1 is HANDOFF on 0.55 of pi0.5's episodes, and RIVAL_TESTS §4.1 found 14 of its 18 grasped failures ending within 0.10 m. The rule looks only at held steps, so an object that drops early and still lands near the target would carry this label.

**Estimator.** VLA HANDOFF episodes, familiar layout, N1-N6, per model: the share whose object ends within 0.10 m of the target.

**Mapping, per model.**
- LABEL FAILS iff the share is above 0.5.
- LABEL HOLDS iff the share is below 0.25.
- AMBIGUOUS otherwise.

**Reported with no verdict.** Per task: the same share; `d` at the onset; `dmin` over held steps and over all steps; final `d`; lift and the end effector's distance to the object at the last held step; whether the policy had commanded the gripper open by the last held step.

**Prediction (mine).** LABEL FAILS for pi0.5, because of the conflict above. No prediction for GR00T.

### 6.2 Check B: is there a state-locked grasp ceiling in the new layout?

**Why.** In the new layout the servo's GRASP rate is 0.50 on each of N1-N6, K4 and K6, and P_min's is 0.48-0.50. Identical rates across tasks suggest that one set of start states blocks the grasp of the cheese and the bottle whatever the system.

**Estimator.** New layout, the eight cheese and bottle tasks. Per state, the GRASP rate over those tasks for the servo and for P_min (three seeds). `L` holds states with both rates ≥ 0.9, and `C` holds states with both rates ≤ 0.1.

**Mapping.**
- STATE-LOCKED iff |L| ≥ 8 and |L| + |C| ≥ 18.
- NOT STATE-LOCKED otherwise.

**Reported with no verdict.** The states in `L`; each system's success and GRASP on `L` and `C`, including K1-K3; the same sets computed for the familiar layout; start positions in `L` and `C`; in the servo's episodes, the object's largest lift and the end effector's closest approach to it.

**Prediction (mine).** STATE-LOCKED.

**What this cannot show.** A state-locked failure shared by a planner and P_min says the state is hard for those two. It does not show why, or that no system could grasp there. A VLA doing better on `L` would disprove a strict ceiling, and I report that comparison.

### 6.3 Results (registered in m3_prereg `pp_atlas2`; results/flawed/pp_atlas_checks.json)

**Check A: pi0.5 AMBIGUOUS, GR00T LABEL HOLDS.** 18 of pi0.5's 37 HANDOFF episodes (0.49) end within 0.10 m of the target; my prediction was LABEL FAILS. 2 of GR00T's 47 (0.04) do.

Medians over HANDOFF episodes, reported with no verdict:

| model, task | n | `d` at onset | `dmin` held | final `d` | lift at last hold | commanded open by last hold |
|---|---|---|---|---|---|---|
| pi0.5 N1 | 11 | 0.18 | 0.12 | 0.03 | 0.06 | yes |
| pi0.5 N2 | 12 | 0.23 | 0.23 | 0.05 | 0.06 | yes |
| pi0.5 N3 | 4 | 0.38 | 0.25 | 0.07 | 0.06 | yes |
| pi0.5 N4 | 3 | 0.26 | 0.20 | 0.30 | 0.28 | yes |
| pi0.5 N5 | 3 | 0.27 | 0.25 | 0.40 | 0.31 | yes |
| pi0.5 N6 | 4 | 0.12 | 0.11 | 0.22 | 0.28 | yes |
| GR00T N2 | 16 | 0.24 | 0.23 | 0.23 | 0.07 | yes |
| GR00T N4 | 9 | 0.26 | 0.24 | 0.24 | 0.27 | yes |
| GR00T N5 | 12 | 0.26 | 0.25 | 0.37 | 0.26 | yes |
| GR00T N6 | 10 | 0.13 | 0.12 | 0.19 | 0.27 | half |

What these medians establish:
- **HANDOFF is mostly not "held and idle".** In almost every row the object is lifted and the policy has commanded the gripper open before it has carried the object far: the hold ends with `d` close to its onset value. The exception is GR00T N6, which often keeps holding to the end of the episode.
- **The stage's accurate description is "the hold ends before the object is carried to the target".** I keep the registered name and rule, and change this reading.
- **Unexplained.** On pi0.5's cheese, the median object ends 0.03-0.07 m from the target after a hold that ended 0.12-0.25 m away. Something moves it about 0.1-0.2 m after the hold signal ends. The hold signal also goes off when contact is lost, so this could be a real drop followed by a shove, or a carry the signal misses. These tracks can answer that, and I have not read them for it.

**This corrects RIVAL_TESTS §4.1.** I reported there that the cheese "arrives and misses by centimetres" and later told the user cross-chunk composition looked mostly intact on the cheese. Final position alone cannot show that the object was carried there, and most of N1 and N2's failures end their hold well short of the target. That reading is withdrawn. V1's steering effect (RIVAL_TESTS §3.1) uses final positions as well, so it shows where the object ends under a prompt, and it does not by itself show a carry.

**Check B: NOT STATE-LOCKED.** |L| = 2 and |C| = 2 in the new layout; my prediction was wrong. The registered check used the wrong grain. The servo's per-state rates over the eight tasks are almost all 0, 0.5 or 1, with 0.5 on 12 of 20 states. The eight tasks are four cheese and four bottle, so this suggests each state blocks the cheese, the bottle, both or neither. That is a post hoc observation from the printed rates. I have not tested the object × state version.

The servo's blocked episodes do reach above the object (horizontal approach about 0.02 m) and never lift it. On the cheese the end effector stops about 0.12 m from the object's origin, where the familiar layout reaches 0.01 m. That rests on only two states (`L`), and I have not established the cause. So the possible ceiling on new-layout success is neither shown nor removed.

### 6.4 A flaw in the control

§3 says the two partners make the object mix match. That holds across a pairing's two roles, but not per stage. At GRASP the target partner holds a different object, usually the bowl, and the bowl grasps far better than the cheese and bottle for P_min and the servo. This inflates GRASP's `E` on those two rows, whose printed "WITHIN-CHUNK" labels in the new layout should not be read. They carried no verdict, as registered. More generally `E` is not a pure interaction: if a stage's rate is additive in object and target, `E` equals half the object difference plus half the target difference. (Another session working on this repo made this point first, in eb550475.)

**Effect on the VLA verdicts: none visible.** In the familiar layout the known partners almost never fail at any stage, so any control gives the same answer. In the new layout, comparing GRASP with the object partner alone (post hoc, from the per-task table) also shows no excess:

| model | cheese N1, N2, N3 | K6 | bottle N4, N5, N6 | K4 |
|---|---|---|---|---|
| pi0.5 | 0.40, 0.30, 0.50 | 0.40 | 0.35, 0.40, 0.25 | 0.50 |
| GR00T | 0.65, 0.65, 0.55 | 0.70 | 0.55, 0.65, 0.60 | 0.60 |

## 7. What stands

- **Familiar layout, both VLAs: new pairings break at two stages where known tasks do not.** The hold ends before the object is carried to the target (`E` +0.31 for pi0.5, +0.39 for GR00T), and objects that do arrive are released badly (`E` +0.16, +0.38). Grasps do not break.
- **New layout, both VLAs: composition adds no measurable break** over the known partners, which already fail at similar rates, mostly at the grasp. The CIs allow up to about +0.2.
- **P_min (oracle state, no verdict):** in the familiar layout its excess at the two VLA stages is +0.00 and +0.04, on the same states. In the new layout it fails to grasp the cheese and the bottle in about half of its episodes, as does a servo with exact positions.

## 8. Reads C and D (design fixed 2026-11-04, after §7, before these reads; m3_prereg tag pp_atlas3)

The user is away and asked me to keep iterating, so I make these calls on the standing rules. Recorded tracks only, no new episodes.

### 8.1 Read C: on pi0.5's cheese, a drop or a missed carry?

**Estimator.** VLA HANDOFF episodes ending within 0.10 m of the target, familiar layout, per model. After the last held step, an object step counts as carried if the object is lifted at least 0.01 m, the gripper is commanded closed, and the end effector is within the episode's median held gap + 0.02 m. An episode is a missed carry iff over half its object travel after the last held step is carried.

**Mapping, per model.** Share of episodes above 2/3: MISSED CARRY, and the cross-chunk `E` overstates that model. Below 1/3: REAL DROP, and the object was moved after the hold ended. Otherwise MIXED.

**Prediction (mine).** REAL DROP on pi0.5. GR00T has only 2 such episodes, so its call means little.

### 8.2 Read D: does each state block one object?

**Estimator.** New layout, per object (cheese N1, N2, N3, K6; bottle N4, N5, N6, K4), per state: the grasp-failure rate for the servo and for P_min. `L` holds states with both rates ≥ 0.9, and `C` holds states with both ≤ 0.1.

**Mapping.** OBJECT-STATE LOCKED iff for both objects |L| ≥ 6 and |L| + |C| ≥ 16. Otherwise NOT LOCKED.

**Reported with no verdict.** Each system's grasp failure and success on `L` and `C`, which tests a ceiling directly because a VLA holding the object on `L` would disprove one; states where no system ever holds the object; start position, yaw and nearest other movable in `L` and `C`; the end effector's closest approach.

**Prediction (mine).** OBJECT-STATE LOCKED, with `L` states showing a close neighbour or an unusual yaw.

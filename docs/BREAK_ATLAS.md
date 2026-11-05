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

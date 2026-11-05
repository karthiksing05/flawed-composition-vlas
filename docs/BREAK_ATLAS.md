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

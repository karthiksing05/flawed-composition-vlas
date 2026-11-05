# The hold gate (PLAN_PARTS §10 items 18-19; BREAK_ATLAS §7)

Design fixed 2026-11-04, before any run. m3_prereg tag `pp_gate`. The user is away and asked me to keep iterating, so this design has not had their review. A post hoc development read with no training. It changes no earlier verdict.

## 1. The question

The atlas found that, on new pairings in the familiar layout, both VLAs often end the hold before the object reaches the target, with the gripper commanded open. If that premature opening causes the failure, then a frozen VLA whose gripper stays shut until arrival should succeed more. This is the smallest test of a structural fix on a real VLA: one rule, separating when to release from what the network outputs.

## 2. Arms

The models are frozen pi0.5 and GR00T, served as in the atlas with the same seeds and start states. `Held` and `d` come from the simulator, so this is an oracle signal and labelled as such.

| arm | rule | tasks |
|---|---|---|
| intact | the atlas's own runs | N1-N6; K1, K2, K3, K4, K6 |
| G, gate | while the object is held and `d` > 0.05 m, the gripper command is forced shut | N1-N6 and the same K tasks, both layouts |
| GA, gate + known target prompt | G, plus while held the prompt is the known instruction sharing the target (V1's arm A) | N1-N6, both layouts |

0.05 m is registered once. It sits between the check's 0.03 m and the atlas's 0.10 m arrival radius. All the policy's motion is unchanged, and it sees its own gripper state in later queries. 1,160 episodes, about 9 A40-h.

## 3. Estimators and verdict

Per model, familiar layout, pooled N1-N6, success `s`; CIs bootstrap tasks and states jointly, paired, 2,000 draws.

- **GATE RECOVERS** iff `s(G) − s(intact)` has CI lower bound above 0 and exceeds 0.10.
- **PROMPT ADDS** iff `s(GA) − s(G)` has CI lower bound above 0 and exceeds 0.10.

| GATE RECOVERS | PROMPT ADDS | verdict | reading |
|---|---|---|---|
| yes | yes | **TIMING AND BINDING** | both a release-timing part and a target binding part help |
| yes | no | **TIMING** | separating release timing is enough for what this recovers |
| no | yes | **BINDING NEEDS THE GATE** | the known target prompt helps only with the gate; the gate alone does not |
| no | no | **NOT TIMING** | premature opening is a symptom. Holding the object shut does not lead to a place |

**Damage control, no verdict unless it fails.** On K1, K2, K3, K4, K6 familiar, `s(G)` must stay within 0.05 of intact. A larger loss is reported as GATE DAMAGES KNOWN TASKS and qualifies the verdict.

**Reported with no verdict.** The new layout under the same rules; the atlas's stages and `E` under each arm, with the same arm's K tasks as partners; per-task success; how often the gate fired; the share of gated episodes where the object stays held to the end without arriving.

**Prediction (mine; my record on this project is poor).** pi0.5 TIMING. GR00T NOT TIMING, because its release failures are as common as its early ends.

## 4. What this cannot show

- The gate uses oracle contact and an oracle target position. A real system would need its own signals.
- If the policy stops moving once its chunk has "finished", the gate cannot help, and a failure then does not separate motion from timing beyond that.
- Read C showed the hold signal can miss a carry on pi0.5. The gate only acts while `Held` is on.
- One threshold, one scene, two models.

## 5. Result (2026-11-05)

All eight run jobs and the read completed with no requeue. `results/flawed/pp_gate_read.json`.

**Registered verdict: NOT TIMING for both VLAs.** Known tasks took no damage. My pi0.5 prediction (TIMING) failed, narrowly: the point estimate clears 0.10 and the CI lower bound sits exactly at 0, which the rule does not count as above 0.

| familiar, N1-N6 | intact | G | GA | G − intact | GA − G | known, intact → G |
|---|---|---|---|---|---|---|
| pi0.5 | .50 | .63 | .68 | +.13 [.00, +.35] | +.05 [−.09, +.21] | 1.00 → 1.00 |
| GR00T | .17 | .23 | .28 | +.06 [−.02, +.15] | +.05 [−.27, +.33] | .98 → .98 |

The new layout does not move: pi0.5 .30, .32, .30; GR00T .08, .09, .08. Its failures sit at GRASP, ahead of the gate.

### 5.1 Reported with no verdict

Success per task, familiar layout:

| | N1 | N2 | N3 | N4 | N5 | N6 |
|---|---|---|---|---|---|---|
| pi0.5 intact → G → GA | .10 .30 .45 | .20 .75 .95 | .60 .65 .45 | .85 .85 .80 | .85 .85 .95 | .40 .40 .50 |
| GR00T intact → G → GA | .10 .10 .15 | .20 .35 .90 | .70 .85 .25 | 0 .05 0 | 0 0 .35 | 0 0 0 |

Excess `E`, familiar layout, each arm's K tasks as partners (GA uses G's):

| | cross-chunk, intact → G → GA | within-chunk, intact → G → GA |
|---|---|---|
| pi0.5 | +.33 → +.18 → +.09 [.00, +.23] | +.18 → +.19 → +.23 [+.07, +.41] |
| GR00T | +.43 → +.30 → +.21 [+.06, +.40] | +.39 → +.46 → +.50 [+.25, +.76] |

Share of episodes that fail with the object still held and more than 0.05 m from the target: pi0.5 .03 → .28 → .17; GR00T .29 → .66 → .57.

### 5.2 What this means

1. **Opening early is a symptom.** The gate keeps the object in hand, and most of those episodes still fail. The policy holds the object and does not finish the place. The held-to-the-end share rises by about the amount the HANDOFF share falls.
2. **Failures move from the cross-chunk column to the within-chunk column.** The gate and prompt shrink cross-chunk `E`, and within-chunk `E` grows. On these numbers the carry's missing piece and the place's missing piece are one problem: after the grasp, the frozen VLA has no reliable way to finish this object at this target. This reading is post hoc.
3. **The known target prompt is target-specific.** N2 (stove) rises on both VLAs, to .95 and .90. N3 (cabinet) falls on both, to .45 and .25. N1 (plate) stays low. Six tasks and post hoc, so this is a lead and nothing more.

The next test replaces the VLA's whole place chunk with a part: `docs/HAND_OVER.md`.

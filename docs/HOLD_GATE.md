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

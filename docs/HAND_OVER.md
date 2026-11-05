# Hand-over: a VLA's pick chunk composed with a place part (PLAN_PARTS §10 items 18-19; HOLD_GATE §5)

Design fixed 2026-11-05, before any run. m3_prereg tags `pp_hand` and `pp_gate_reads`. The user is away and asked me to keep iterating, so this design has not had their review. Post hoc development reads with no training. They change no earlier verdict.

## 1. The question

The atlas measured two breaks on new pairings in the familiar layout: within-chunk and cross-chunk. Both VLAs grasp the object almost every time (GRASP .02). The hold gate showed that keeping the gripper shut does not rescue what follows. The user's goal is to find where VLAs break and see how a partonomic architecture fixes them, so the direct test is this: keep the frozen VLA for the chunk it does well, hand the rest to a part, and ask whether the breaks go away.

## 2. Arm H

- From the episode's start the frozen VLA acts exactly as in the atlas, with the same seeds, states and prompt. Each episode is therefore paired with its atlas run up to the hand-over.
- At the first step where `Held` turns on, control passes to P_min (`pp_full_naf_s10/P.npz`, trained on the K tasks only) for the rest of the episode. It does not pass back.
- P_min runs with the inputs, tracking and seeds of `fa1h_roll.episode`: `prev`, `Held`, `Canon`, `rot0`, `d0t`, `ee0` are stepped from the start of the episode, and its noise seed is `seed_base + calls`. No P_min retraining, no change to any weight in either model.
- If the object drops after the hand-over, P_min keeps control and may regrasp, as it does in its own episodes.

Tasks: N1-N6 and K1, K2, K3, K4, K6, both layouts, both VLAs. 880 episodes, about 7 A40-h. One P_min seed, because this is a first read.

## 3. Estimators and verdict

Per VLA, familiar layout, pooled N1-N6. CIs bootstrap tasks and states jointly, paired, 2,000 draws, as in the gate.

- **PART RAISES** iff `s(H) − s(intact)` has CI lower bound above 0 and exceeds 0.10.
- **Each break under H:** `E` for within-chunk and cross-chunk with the atlas's stage rule and the atlas's partners, the partners also under H. A break remains iff its CI lower bound is above 0 and `E` exceeds 0.10.

| PART RAISES | breaks remaining | verdict |
|---|---|---|
| yes | none | **FIXED** |
| yes | within | **RAISED, WITHIN STILL BREAKS** |
| yes | cross | **RAISED, CROSS STILL BREAKS** |
| yes | both | **RAISED, BOTH STILL BREAK** |
| no | any | **NO FIX** |

**Damage control, no verdict unless it fails.** On K1, K2, K3, K4, K6 familiar, `s(H)` must stay within 0.05 of intact. A larger loss is reported as KNOWN COST and qualifies the verdict.

**Reported with no verdict.** The new layout under the same rules; per-task stages; the share of episodes handed over and success among them; `s(H)` against P_min alone on seed 10, atlas tracks.

**Prediction (mine; my record is poor).** FIXED on both VLAs, familiar `s(H)` about .85. New layout capped by grasping, about .55 on pi0.5 and .35 on GR00T.

## 4. Reads on existing tracks (tag `pp_gate_reads`)

No rollouts. CPU. `experiments/pp_gate_reads2.py` → `results/flawed/pp_gate_reads2.json`.

**Read E: why new-layout grasps fail.** Systems pi0.5, GR00T, P_min s10, servo. Atlas tracks of N1-N6, new layout, every episode the atlas labels GRASP. With `a_t` = distance from the end effector to object + `grasp_off` (the task's keypoint) and the object's displacement from its first tracked position, the first matching class applies:

1. PUSHED: the object moves more than 0.03 m before `a_t` first reaches 0.05 m, or at any time if it never does.
2. NEVER REACHED: `a_t` never reaches 0.05 m.
3. NO CLOSE: `a_t` reaches 0.05 m, and the gripper is never commanded closed while within it.
4. CLOSED, NO HOLD: closed while within 0.05 m, with no hold.

Verdict per system: the class holding more than half of its GRASP failures, otherwise MIXED. `grasp_off` comes from demos. A VLA gripping elsewhere on the object can read as NEVER REACHED, so I report the distribution of minimum `a_t` alongside. Prediction: PUSHED on the servo, NEVER REACHED on the VLAs.

**Read F: does the gated policy stall.** Tracks of arms G and GA, familiar, N1-N6, per VLA: the failed episodes still held at the end with `d` > 0.05 m. An episode STALLS when the object's net horizontal motion over its last 50 tracked steps is under 0.02 m. Verdict per VLA on arm G: STALLS iff that share is above one half, otherwise KEEPS MOVING. Reported with it: arm GA, median final `d`, and the share ever within 0.03 m. Prediction: STALLS on both.

## 5. What this cannot show

- P_min reads oracle object and target positions and arm H hands over on oracle contact. This shows where to cut and what a part recovers. It does not show a deployable system.
- H replaces the VLA's place chunk. It does not repair it. A FIXED verdict says the VLA's pick chunk composes with an external place chunk, and leaves open whether the VLA's own place chunk can be made to work.
- The VLA's grasp may differ from a P_min grasp, so the hand-over tests P_min on grasps it did not train on. This is part of what is being tested.
- `E`'s control carries the flaw noted in BREAK_ATLAS §6.4.
- One P_min seed, one scene, two models, 20 states.

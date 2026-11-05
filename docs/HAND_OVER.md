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

## 6. Results (2026-11-05)

All eight run jobs and both read jobs completed with no requeue. `results/flawed/pp_hand_read.json`, `results/flawed/pp_gate_reads2.json`.

### 6.1 Hand-over

**Registered verdict: FIXED on both VLAs. No KNOWN COST.** My prediction held on the familiar layout and undershot it.

| familiar, N1-N6 | intact | H | H − intact | within `E`, intact → H | cross `E`, intact → H | known, intact → H |
|---|---|---|---|---|---|---|
| pi0.5 | .50 | .96 | +.46 [+.18, +.73] | +.18 → +.02 [−.04, +.10] | +.33 → +.01 [.00, +.05] | 1.00 → .98 |
| GR00T | .17 | .93 | +.76 [+.53, +.93] | +.39 → +.04 [−.05, +.16] | +.43 → +.01 [.00, +.05] | .98 → .98 |

Success per task under H, familiar layout (N1 to N6): pi0.5 1.00, 1.00, .85, 1.00, 1.00, .90; GR00T 1.00, 1.00, 1.00, .80, .95, .80. P_min alone on seed 10 scores .92, so H sits level with it: +.04 [−.07, +.18] and +.01 [−.10, +.12]. The VLAs hand over in 98% of episodes, and 97% and 94% of those succeed. The grasps of two VLAs with different vision and action models, which P_min never trained on, carry straight into its place part.

**New layout, no verdict.** The atlas found no break here, so the printed FIXED label carries no meaning.

| new layout, N1-N6 | intact | H | H − intact | handed over | success when handed over | P_min s10 alone |
|---|---|---|---|---|---|---|
| pi0.5 | .30 | .51 | +.21 [+.05, +.40] | .63 | .80 | .35 |
| GR00T | .08 | .31 | +.23 [+.10, +.37] | .39 | .79 | .35 |

Grasping sets this ceiling. On pi0.5, H's point estimate sits above P_min alone (+.16 [−.03, +.33]), which is a lead that the CI does not settle.

**What this does and does not establish.** On this scene the two measured breaks sit entirely after the first hold: replacing that one chunk removes both, and the pick chunk composes with no change. The fix runs on oracle contact and oracle positions, and it replaces the VLA's place chunk without repairing it. It is one P_min seed on development states.

### 6.2 Read E: why new-layout grasps fail

| system | GRASP failures | PUSHED | NEVER REACHED | NO CLOSE | CLOSED, NO HOLD | verdict | median min `a` |
|---|---|---|---|---|---|---|---|
| pi0.5 | 44 | .02 | .55 | .09 | .34 | NEVER REACHED | .073 |
| GR00T | 73 | .01 | .63 | .21 | .15 | NEVER REACHED | .085 |
| P_min s10 | 66 | .05 | .41 | .05 | .50 | MIXED | .045 |
| servo | 60 | 0 | .55 | .05 | .40 | NEVER REACHED | .058 |

My prediction held on the VLAs and failed on the servo: nothing gets pushed. Median closest approaches sit near the 0.05 m line, so the split between NEVER REACHED and CLOSED, NO HOLD depends on that threshold. The servo has oracle positions and a demo keypoint and still fails on half its failures this way, so vision does not explain all of this.

### 6.3 Read F: does the gated policy stall

**pi0.5 STALLS** (21 of 33 under G, median net motion 0.011 m, final `d` 0.20 m, none ever within 3 cm). **GR00T KEEPS MOVING** (17 of 79 stall, median net motion 0.060 m, final `d` 0.14 m, 8% ever within 3 cm). My prediction held on pi0.5 only. Under the gate pi0.5 stops 20 cm short, and GR00T keeps moving without getting there. Either way the frozen VLA does not deliver the object.

## 7. The second round: replication, less oracle, a perception budget (m3_prereg `pp_hand2`)

Design fixed 2026-11-05 after §6, before any of these runs. Post hoc, development states, no training. The user has not reviewed it. L2 of the ladder (position perception) stays paused for the user's decision (LADDER_RESULTS §4), because its plan-level choices are theirs. §7.3 only measures what such a perception step would have to deliver.

Client `experiments/pp_hand2_client.py`, read `experiments/pp_hand2_read.py`. Everything below uses the mapping in §3 where it says "same mapping". All arms run from the same starts and seeds as §2.

### 7.1 Replication on other part seeds

Arm H on `pp_full_naf_s11` and `pp_full_naf_s12`, both VLAs, all 11 tasks, both layouts, pooled over the two seeds. **Verdict per VLA, familiar: REPLICATES iff the same mapping returns FIXED.** Prediction: REPLICATES on both.

### 7.2 Hand-over without simulator contact

Arm H1 uses L1's build `pp_full_L1_s10` (LADDER_RESULTS §1, the rung that held). The hand-over and P's gate both follow L1's event from gripper width and proprioception, stepped exactly as `fa1h_roll.episode` steps it. Oracle `Held` stays the read's instrument only. Both VLAs, all 11 tasks, both layouts. An event firing on an empty close hands over early, and P may then grasp on its own.

**Verdict per VLA, familiar: CONTACT NOT NEEDED iff the same mapping returns FIXED and `s(Hs10) − s(H1)` ≤ 0.10. Otherwise CONTACT MATTERS.** Prediction: CONTACT NOT NEEDED on both.

### 7.3 Perception budget

Arm H on seed 10, pi0.5, N1-N6, familiar. Each episode draws one horizontal direction from `seed_base`, and it stays fixed through the episode. Arms T01, T02, T04 move the target position P sees by 0.01, 0.02, 0.04 m in that direction. Arms O01, O02, O04 move the held object's position and grasp offset P sees by the same amounts. Everything else stays oracle. This is a constant error, the case a calibration fault or a poor locator would produce, and it says nothing about jitter.

**Verdict per factor: the budget is the largest radius, going up from 0.01 m with no skipped radius, whose `s` stays within 0.10 of `s(Hs10)`. None qualifying reads BELOW 0.01 M.** Prediction: 0.02 m on both, because LIBERO's `On` allows 0.03 m.

### 7.4 Read G: is the new-layout grasp failure a physical block

The servo's reach phases time out after 40 steps and close wherever the hand is, so Read E's NEVER REACHED on the servo may reflect that timer. `experiments/pp_read_g.py` reruns phases 0-3 of `pp_n4d.plan_episode` on the states where the servo's atlas track on N1 (cheese) and N4 (bottle), new layout, reads GRASP. It uses the same seeds and the timeout as the only change: 40 (control) and 120 steps. The episode cap stays at 300. Reported with no verdict: contacts at the closest approach, whether a joint is within 0.05 rad of its limit, and the closest approach itself.

**Verdict per object, on the 120-step hold rate over those states: NOT A BLOCK above 0.5; BLOCK below 0.2; otherwise MIXED.** A control hold rate well above 0 would mean this loop does not reproduce the atlas's servo, and I report it as such. Prediction: NOT A BLOCK on both.

3,360 rollouts, about 11 A40-h, plus Read G on CPU.

## 8. Results of the second round (2026-11-05)

All 30 run jobs and both reads completed. `results/flawed/pp_hand2_read.json`, `results/flawed/pp_read_g.json`. Audits 0/0 at the commit before this section.

### 8.1 Replication and contact

**REPLICATES on both VLAs. CONTACT NOT NEEDED on both VLAs.** Predictions held.

| familiar, N1-N6 | intact | H, s10 | H, s11 + s12 | H1, event | H1 − H s10 | within `E`, H1 | cross `E`, H1 | known, H1 |
|---|---|---|---|---|---|---|---|---|
| pi0.5 | .50 | .96 | .94 | .88 | −.08 [−.18, .00] | +.06 [−.03, +.17] | +.02 [.00, +.06] | .98 |
| GR00T | .17 | .93 | .93 | .92 | −.01 [−.10, +.08] | +.05 [−.03, +.17] | +.01 [.00, +.05] | .99 |

H1 uses the build's own weights, trained with the event. It still reads oracle positions: the object's and target's positions, and rest height from the object's pose, all remain. P does not read the grasp offset `b["off"]`, so that oracle quantity in `fa1h_roll.episode`'s call never reaches it. H1 removes contact from the hand-over and from P's gate, and from nothing else.

New layout, no verdict: pi0.5 .30 → .54 (s11 + s12), .49 (H1); GR00T .08 → .30, .34. Known tasks in that layout: pi0.5 .37 → .47, .54; GR00T .20 → .34, .42.

### 8.2 Perception budget

**Target: 0.02 m. Held object: 0.02 m.** Predictions held. pi0.5 only, `s(Hs10)` = .96. Because P does not read `b["off"]`, the O arms act through the object position alone.

| constant error | 0.01 m | 0.02 m | 0.04 m |
|---|---|---|---|
| target position | .91 | .88 | .72 |
| held object position | .92 | .88 | .63 |

Tasks differ, with no verdict on this: at 0.04 m, N2 and N5 (stove) stay at 1.00 under both factors, and N1, N4 (plate) and N6 (bowl) fall to .15-.55.

### 8.3 Read G: is the new-layout grasp failure a physical block

**BLOCK on both objects, against my prediction.** Hold rate over the servo's failed states is 0 of 10 for the cheese and 0 of 10 for the bottle, at 120 steps as at 40 steps. So the timer does not explain the servo's failures, and this loop reproduces the atlas's servo (control 0).

Contacts at the closest approach, 120-step run, no verdict:

- **Cheese.** On 7 of 10 states the hand touches the wine rack and stops 0.09-0.12 m from the grasp point.
- **Bottle.** On 3 states the hand touches the cabinet, about 0.05 m short. On 3 others it reaches the grasp point within 0.005 m with a joint at its limit, touching nothing. The rest brush the bottle or stop short.

This is a block for one approach: the demo's fixed grasp rotation, from directly above. It is not shown to be a block for every approach, because the VLAs sometimes succeed on these object × state pairs (BREAK_ATLAS §8.3: pi0.5 .16 on cheese, .08 on bottle). It matches the post hoc spatial split in read D, and it gives that split a mechanism: in the new layout some start positions place the object under the rack or cabinet, or at the arm's limit. Getting past this needs an obstacle-aware approach. The standing rule bars giving an executor occupancy or free-space inputs, so I stop here and leave this to the user. A VLA already sees the obstacle, which makes this a place where its vision could earn its keep.

### 8.4 Where this leaves the goal

| | familiar layout | new layout |
|---|---|---|
| Where VLAs break on new pairings | within-chunk and cross-chunk, all after the first hold | no excess over known tasks; both fail, mostly at the grasp |
| Does a part fix it | yes on both VLAs, three part seeds, simulator contact removed | raises success; grasp sets the ceiling, and part of that is a physical block for a fixed approach |
| What the fix still takes from the simulator | object and target positions (budget 0.02 m), rest height, sizes, roles | the same |

## 9. Read H: how much perception would a fix need (m3_prereg `pp_read_h`)

Design fixed 2026-11-05, before the read. No rollouts. `experiments/pp_read_h.py` → `results/flawed/pp_read_h.json`.

The budget says P tolerates about 0.02 m. This read asks whether positions move by more than that, using first-tracked positions from the atlas tracks over the 20 states.

- **Target.** Per target and layout, the 95th percentile of distance from that target's mean horizontal position, over the tasks using it. **Verdict per layout: A TABLE SUFFICES iff every target is within 0.02 m, otherwise NEEDS PERCEPTION.**
- **Object.** The same on the first-tracked position of the cheese and the bottle. Same verdict rule.
- **Grasp offset**, reported with no verdict: object minus end effector at the hand-over, horizontal, from the H tracks, per VLA and object.

The mean comes from the same states it is scored on, so this is the most favourable case for a table. Passing it would not show a table works on unseen states. Prediction: NEEDS PERCEPTION for objects in both layouts; for targets, NEEDS PERCEPTION in both, because the scene's placement regions span several centimetres.

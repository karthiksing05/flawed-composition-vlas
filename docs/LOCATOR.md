# Locator: positions from demos and the robot's own state (HAND_OVER §10 item 1)

Design fixed 2026-11-05, before any training or read. m3_prereg tag `pp_loc`. The user chose the route: a demo-labelled locator, with a simulator-labelled locator as the reference. The user has not reviewed the details below. Offline development read, no rollouts. It changes no earlier verdict.

## 1. The question

The part needs object and target positions, and it tolerates about 0.02 m of constant error (HAND_OVER §8.2). Can a small model find them in one camera image, trained with labels that come from the demos and the robot's own state?

## 2. Why the design has this shape

**The demos barely vary where objects sit.** All demos come from the familiar layout, where positions stay within about 0.02 m of a constant (HAND_OVER §9.1). A model that regresses world coordinates from the whole image would learn a constant, and that constant would already pass in the familiar layout. So the informative test is the new layout, and every arm is compared against a constant.

**So the model finds appearance.** A fully convolutional network outputs one heatmap per head, with a receptive field of 61 pixels and no coordinate inputs. Training shifts and rescales each image and moves the label with it, so absolute image position carries no information. The network has to recognise the thing.

**One start-of-episode look may be enough.** Objects and targets sit still until the grasp, so the locator runs once per episode. The held object's position during the carry can follow from the hand: the located start position minus the hand position at the hand-over gives the offset. That removes the per-replan vision pass LADDER_RESULTS §4 feared, and it replaces HAND_OVER §9.1's suggestion of measuring the bottle's offset separately. It holds only if the grasp barely moves the object, and §5 measures that.

## 3. What differs from what the user was told, and from the plan

1. **The arms use the simulator's camera matrices.** I described route C as needing only demos and the robot's own state. It needs a camera model too: to place 3-D labels into the image and to lift pixels back to the table. A calibrated camera is an ordinary robot-side given, but it remains an oracle here. PLAN_PARTS §5.3's self-calibration is unbuilt, and this read does not build it.
2. **The input is a small network on raw pixels.** PLAN_PARTS §5 names the VLA's visual tokens as L2's source. LADDER_RESULTS §4(d) lists the small encoder as the cheaper path that is "not the VLA's visual tokens". It stays independent of which VLA runs.
3. **Scope.** Positions only. Rest height, sizes, yaw and roles stay as they are. No rollout uses these outputs yet.

## 4. Design (`experiments/pp_loc.py`)

**Data.** Recorded states of the K1-K6 demos, set and rendered, never executed. One agentview image at 256 px, upright, every 5th kept step. Demos 0-39 train; demos 40-49 are a held-out sanity check. No development state enters training.

**Events, from proprioception only.** L1's `ProprioHeld`, threshold by its fixed rule on K1-K6 rows. In each demo: the first hold; the close row, where the closing command that led to that hold began; the release row, the first open command after the hold. Labels use the hand's 3-D position at those rows.

**Five heads,** one per entity and role, because the two conventions differ:

| head | tasks | valid frames, arm C label |
|---|---|---|
| cheese, object | K6 | before the close row; hand at close |
| bottle, object | K4, K5 | same |
| bowl, object | K1, K2, K3 | same |
| plate, target | K1 | before the release row; hand at release |
| bowl, target | K6 | same |

**Arms.** **C** uses those labels. **A** uses the same frames, network and training, with the simulator's entity position at that row as the label. C and A both project labels into the image with the simulator's camera matrix. Three seeds each.

**Training, frozen.** Batch 32 with heads sampled evenly, 4,000 AdamW steps at 1e-3 with cosine decay. Augmentation: translation up to ±64 px, scale 0.85-1.15, brightness and contrast ±0.15. Cross-entropy over a 128×128 heatmap against a Gaussian of 1.5 cells.

**Lifting.** The heatmap's peak, refined over a 7×7 window, gives a pixel. A ray through it meets a horizontal plane at that head's mean training label height, which assumes the entity rests at the same height in both layouts. The read reports the oracle heights so this can be checked.

**Evaluation.** Development states 4-23 of each layout, after the same 10 settle steps as N2's client. 20 states, one image each.

## 5. Estimators and verdict

Arm C's label is where the hand closed or opened, which is a fixed distance from the body's centre. A part retrained on these positions would absorb a constant shift, but the frozen part would not. So the verdict removes it, and the raw error is reported with it.

- `b` per head: the mean over training demos of (label − oracle position), horizontal. Near zero for A. The simulator appears here as an instrument only.
- `e` per head and layout: the 95th percentile over the 20 states of |prediction − `b` − oracle|, horizontal. Median over three seeds.
- `T` per head and layout: the same with the prediction replaced by the mean training label, which is the constant a table would give.

**Verdict per arm.**

| condition, first match | verdict |
|---|---|
| every head has `e` ≤ 0.02 m in both layouts | **WITHIN BUDGET** |
| every head has `e` ≤ `T` / 2 in the new layout | **BEATS A TABLE, OVER BUDGET** |
| otherwise | **NO BETTER THAN A TABLE** |

**How the two arms read together, no separate verdict.** A passes and C fails: the demo labels cost the accuracy. Both fail in the same way: the data or the network does, and the labels are not the problem.

**Reported with no verdict.** Raw error; median error; per-seed values; demos 40-49 error; label noise (spread of label − oracle per head); metres per pixel at the table; oracle heights. **Grasp displacement:** from the hand-over tracks, horizontal distance between the object's position at the hand-over and at the start, 95th percentile per VLA, layout and object. If it stays under 0.02 m, the start look gives the held object's offset.

**Prediction (mine; my record is poor).** Both arms BEATS A TABLE, OVER BUDGET. Familiar `e` near 0.02 m, new-layout `e` near 0.05 m, with occlusion by the rack and cabinet setting the worst states. Grasp displacement under 0.02 m on the familiar layout and over it on the new one.

## 6. What this cannot show

- Passing does not show the fix works on perceived positions. That needs rollouts, and the budget was measured on a constant error, on pi0.5 only.
- 95th percentile of 20 states is close to the worst state, so one bad state decides a head.
- It leaves the camera matrices, rest height, sizes, yaw and roles oracle.
- Removing `b` assumes the shift stays constant. A rim grasp on a rotated bowl would break that. The bowl's object head matters only for known tasks.
- The new layout's states are a development set, and one scene.

## 7. Results (2026-11-05)

All jobs completed. `results/flawed/pp_loc.json`. One K4 demo gave no full hold and was dropped. 3,644 demo frames. One pixel is 0.0037 m at the table, so the budget is about 5 pixels.

**Registered verdict: arm C NO BETTER THAN A TABLE. Arm A NO BETTER THAN A TABLE.** My prediction failed on both.

| `e`, 95th percentile, m (new-layout median error in brackets) | cheese | bottle | bowl, object | plate, target | bowl, target |
|---|---|---|---|---|---|
| A, familiar | .001 | .004 | .001 | .001 | .001 |
| A, new | .020 (.007) | .006 (.003) | .021 (.005) | **.304** (.005) | .019 (.005) |
| C, familiar | .019 | .035 | .013 | .008 | .034 |
| C, new | **.260** (.017) | .038 (.021) | **.176** (.012) | **.310** (.009) | **.163** (.018) |
| table, new | .271 | .371 | .261 | .322 | .258 |

Reading across arms, as registered: A does much better than C on four heads, and A's plate head fails in the same way. So the demo labels cost accuracy, and something else also breaks the plate.

Other reported numbers:

- **Label noise,** p95 of label − oracle − `b`: cheese .033, bottle .032, bowl object .030, plate .024, bowl target .114. Every head's labels are noisier than the budget.
- **A's plate head** varies by seed: .038, .305, .304.
- **Heights.** New-layout oracle heights vary by up to 0.019 m for cheese, plate and bowl, so "same height in both layouts" fails on some states.
- **Grasp displacement,** p95, m. Familiar: pi0.5 .016 cheese, .027 bottle, .021 bowl; GR00T .013, .015, .022. New: pi0.5 .026, .067, .032; GR00T .040, .026, .034. Medians run .004-.019. My prediction failed on the familiar side: pi0.5's bottle and both bowls exceed 0.02 m. The start look does not fully give the offset at the 95th percentile.

## 8. Diagnostic: a few states, or a broad failure (m3_prereg `pp_loc_diag`)

Design fixed after §7, before this read. Post hoc. No training, and it changes no verdict in §7. `experiments/pp_loc_diag.py` → `results/flawed/pp_loc_diag.json`, `pp_loc_diag.png`. The same checkpoints and 20 new-layout states.

A state is **bad** for a head and seed when its horizontal error, after removing `b`, exceeds 0.05 m. **Verdict per arm and head: FEW STATES iff every seed has at most 3 bad states of 20 and its median error is at most 0.02 m. Otherwise BROAD.**

Reported with no verdict: each bad state's index; whether it repeats across seeds and arms; whether the prediction sits within 0.05 m of another entity; that state's oracle height. The figure shows the worst states with oracle and predicted points, for a visual check of occlusion.

Prediction: FEW STATES on all of A's heads, and on C's heads except the bottle, with bad states repeated across seeds and caused by occlusion.

### 8.1 Result (2026-11-05)

`results/flawed/pp_loc_diag.json`, `pp_loc_diag.png`. **Arm A: FEW STATES on all five heads. Arm C: FEW STATES on the bowl's object head, BROAD on the other four.** My prediction held on A and on C's bottle, and failed on C's cheese, plate and bowl target heads. It also failed on the cause.

| bad states of 20 per seed, new layout | cheese | bottle | bowl, object | plate, target | bowl, target |
|---|---|---|---|---|---|
| A | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 1, 3, 2 | 0, 0, 0 |
| C | 3, 6, 6 | 1, 1, 1 | 2, 3, 1 | 4, 5, 9 | 2, 2, 6 |

- **The failure is confusion, and the figure shows little occlusion.** Most bad predictions land on or near another object: C's plate head always lands within 0.05 m of the bowl, its bowl heads mostly on the cheese, and its cheese head mostly near the bottle or bowl. In all 13 states drawn, the head's own target is visible. No training image shows these objects anywhere else, so the network may be using surroundings or scale to tell them apart. That explanation is a reading of the figure, untested.
- **A's plate failures come from two kinds of state.** On states 13 and 19 the plate rests tilted against the stove (oracle height up to 0.014 m above its flat height), and its appearance changes. On state 20 the bottle partly covers it. A's other four heads have no bad state on any seed.
- **C's bottle head has one bad state, repeated across seeds, at .055-.065 m.** It fails on its median error, .017-.022 m, and the label height, 0.105 m above the bottle's body origin, may contribute.

**Where this leaves route C.** With exact labels, appearance alone places four of five heads within about 0.02 m in a layout the network never saw, and A's verdict rests on the plate alone. So the architecture can locate these objects, and C's costs come from its labels. Label noise also blurs what the network learns about each object, which fits C's confusions, though this read does not separate the two. All of it uses the simulator's camera matrices, and the 95th percentile still decides §7's verdict, which stands.

## 9. Round two: objects in new places, cleaner labels (m3_prereg `pp_loc2`)

Design fixed 2026-11-05 after §8.1, before any of this training, with the user's go-ahead for the two changes. Offline development read, no rollouts. These 20 states have now shaped the design, so a pass here is not evidence on unseen states. `experiments/pp_loc2.py` → `results/flawed/pp_loc2.json`.

**Mechanism P, moved objects.** No training image shows an object anywhere else. So for 80% of samples, all four objects move to random positions in the workspace, which is the bounding box of the hand's horizontal positions over the K1-K6 demos. The rest of training stays as §4.

- **Cutouts.** An object is what changed when the robot moved it. For the cheese, bottle and bowl, each training demo's first and last frames differ where the object left. The connected region near that demo's label, with holes filled, is the cutout, with no simulator mask. No K1-K6 demo moves the plate, so its cutout is a disc of radius 40 px, large enough to hold the whole plate under arm C's off-centre label.
- **The empty scene.** A background image fills each object's usual place from the median of late frames, and fills the plate's disc by interpolating across each row. Every moved sample starts from its own frame with those places replaced.
- **Pasting.** A random cutout of each object lands at a random position on the head's label height, rescaled by the ratio of camera depths. The trained head's object goes last, so nothing covers it. Its new label is that head's mean offset from the cutout's centre, added to the position. So arm C's pasted labels no longer carry per-demo label noise, except on the plate's disc.

A first draft used 96 px square patches. A smoke-run drawing showed them carrying neighbouring objects, which left unlabelled copies of each head's target. I replaced it before registration and before any full training. No error was measured on either version.

**Mechanism S, cleaner labels.** A first network trains on the arm's labels. For each demo and head, the median of its peak over the demo's training frames replaces the label, unless it lies more than 20 px (about 0.07 m) from that frame's original label, in which case the original stays. A fresh network then trains on the new labels, from new weights. `b` and `T` are recomputed from those labels per seed, lifted at the same height.

**Arms, three seeds each.**

| arm | labels | P | S | role |
|---|---|---|---|---|
| **CPS** | demo | yes | yes | **the verdict's arm** |
| CP | demo | yes | no | removal control for S; CPS's first network |
| CS | demo | no | yes | removal control for P; first network is §7's C |
| AP | simulator | yes | no | what P gives with exact labels |

**Verdict: §5's mapping, unchanged, on CPS alone.** WITHIN BUDGET, BEATS A TABLE, OVER BUDGET, or NO BETTER THAN A TABLE. The other arms get the same label as a report, and picking the best of them would not count. §8's FEW STATES or BROAD is reported for every arm with no further standing.

**Mechanism reading, no verdict.** Compare new-layout `e` and bad-state counts across C, CP, CS and CPS. Compare A and AP on the plate.

**Prediction (mine; my record is poor).** CPS BEATS A TABLE, OVER BUDGET, with most of the gain from P. AP also BEATS A TABLE, because moving a flat plate's patch cannot show it tilted.

**What this cannot show.** Any verdict here has the §6 limits. A pass needs a check on states no design has seen, and confirmatory states need the user's approval. The rescale ignores the change in viewing angle. P already removes most label noise on pasted samples, so S has less to do in CPS than in CS, and the two controls do not cleanly split their credit. The plate cannot be seen tilted, and the plate's disc carries a ring of table. The smoke drawing `results/flawed/pp_loc2_patches_CP.png` still shows small remnants of the bottle in the emptied scene.

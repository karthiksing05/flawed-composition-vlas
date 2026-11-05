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

### 9.1 Results (2026-11-05)

All jobs completed. `results/flawed/pp_loc2.json`.

**Registered verdict on CPS: NO BETTER THAN A TABLE.** My prediction failed: only the plate head stops CPS from beating a table, and its `e` of .563 m is worse than a constant. As reports, CP and CS read the same, and AP reads BEATS A TABLE, OVER BUDGET. My prediction for AP held, though its reason did not: AP's plate has 0, 1, 0 bad states, so moved flat plates help even on tilted ones.

| new layout, `e`, m (median error in brackets) | cheese | bottle | bowl, object | plate, target | bowl, target | bad states per seed, plate |
|---|---|---|---|---|---|---|
| C, §7 | .260 (.017) | .038 (.021) | .176 (.012) | .310 (.009) | .163 (.018) | 4, 5, 9 |
| CS | .267 (.014) | .034 (.020) | .038 (.006) | .309 (.006) | .030 (.011) | 6, 5, 5 |
| CP | .015 (.005) | .005 (.003) | .028 (.023) | .318 (.012) | .033 (.027) | 4, 3, 5 |
| **CPS** | .014 (.007) | .005 (.002) | .027 (.022) | **.563** (.014) | .033 (.024) | 5, 5, 5 |
| A, §7 | .020 (.007) | .006 (.003) | .021 (.005) | .304 (.005) | .019 (.005) | 1, 3, 2 |
| AP | .015 (.004) | .012 (.008) | .020 (.017) | .025 (.005) | .022 (.020) | 0, 1, 0 |

| familiar layout, `e`, m | cheese | bottle | bowl, object | plate, target | bowl, target |
|---|---|---|---|---|---|
| CP | .012 | .026 | .007 | .009 | .017 |
| CPS | .013 | .029 | .006 | .011 | .012 |
| AP | .003 | .003 | .002 | .001 | .002 |

**Mechanism reading, no verdict.**

- **P does most of the work on demo labels.** Going from C to CP, the cheese falls from .260 to .015 and the bottle from .038 to .005, both inside the budget on every seed. The bowl's heads fall from .176 and .163 to .028 and .033. So my prediction on where the gain comes from held. Moved objects do address the confusion §8.1 found.
- **S adds little on top of P.** CP and CPS sit level on four heads, and S makes the plate worse. One CPS seed has 2 bad cheese states that CP lacks. On its own, S helps the bowl's heads and leaves the cheese and plate failing.
- **The plate is C's remaining failure.** It keeps 3-5 bad states under P on demo labels, against 0-1 under AP. AP's disc sits centred on the plate; C's label sits about 0.04 m off centre (§7's `b`), so its disc is lopsided. That is a likely cause, and it is untested.
- **P adds a shift on the bowl.** In the new layout the bowl's median error rises under P from .005-.018 to .017-.027 on both label types, with `e` close to the median. That looks like a near-constant shift in this layout, with no bad states. The cause is unknown.
- **C's familiar bottle exceeds the budget,** at .026-.029, and exceeds the table's .016, with one seed at .010-.013.

**Where this leaves route C.** The two objects the new pairings carry, the cheese and the bottle, sit within 0.015 m in the new layout on demo labels and the simulator's camera. Plate and bowl targets do not yet. The stove and cabinet need no perception. So tasks N2, N3 and N5 could run closed-loop on located objects today, with the familiar bottle's error as the known risk.

## 10. Closed loop on located objects (m3_prereg `pp_hand3`)

Design fixed 2026-11-05 after §9.1, before any of these runs, with the user's go-ahead for the test. Development states, no training. The 20 states have shaped the locator, so this does not test unseen states. Client `experiments/pp_hand3_client.py`, read `experiments/pp_hand3_read.py`.

**The question.** Does the hand-over survive when the part's object position comes from one look with the demo-labelled locator?

**Arm LS.** H1 from HAND_OVER §7.2 (the frozen VLA until L1's proprioceptive event, then the L1 build `pp_full_L1_s10`), with one change to what P sees about the object.

1. **One look.** At the first tracked step, after the same settle steps, CP's network (§9) reads one agentview image and gives the object's horizontal position. The head is the cheese's or the bottle's, chosen by the task's role.
2. **No shift removed.** The raw output goes in, with no `b`, because `b` comes from the simulator. So P sees where a demo's hand closes, up to the network's error. §9.1's raw errors in the new layout are .019 m for the cheese and .014 m for the bottle.
3. **During a hold.** At each onset of the event, the offset is the current estimate minus the hand's horizontal position. While the event stays on, the estimate is the hand plus that offset.
4. **After a hold.** The estimate stays frozen at its last value, with no second look. A dropped object's estimate goes stale.

Held-object error therefore combines the locator's error with how far the grasp moved the object (§7).

**What stays oracle.** The object's height, and so its rest height; its yaw and size; every target's position; the role assignment; the camera matrices, through the network's labels and lifting. The stove and cabinet never move, so their target positions could come from a table, which the standing rule bars as a fix. Read's Rec still uses simulator contact as an instrument only.

**Pairing.** P never acts before the hand-over, so every episode matches its H1 episode until then, with the same seeds. Any difference comes from what P does with the estimate.

**Tasks.** N2, N3, N5 (cheese→stove, cheese→cabinet, bottle→stove), with K4 and K6 as known controls. K6 places on the bowl, whose position stays oracle here. Both layouts, both VLAs, 20 states, three locator seeds: 1,200 episodes, about 10 A40-h. One P seed, as in H1.

**Verdict per VLA and layout,** on N2, N3, N5 pooled over locator seeds. CIs bootstrap tasks and states jointly, 2,000 draws, paired; three tasks make these coarse. `RAISES`: `s(LS) − s(intact)` has CI lower bound above 0 and exceeds 0.10. `LOSS`: `s(H1) − s(LS)`.

| RAISES | LOSS | verdict |
|---|---|---|
| yes | ≤ 0.10 | **PERCEPTION HOLDS** |
| yes | > 0.10 | **PERCEPTION COSTS** |
| no | any | **NO FIX** |

**Damage control, no verdict unless it fails.** On K4 and K6, `s(LS)` must stay within 0.05 of `s(H1)`. A larger loss is reported as KNOWN COST and qualifies the verdict.

**Reported with no verdict.** Per task and locator seed; the share handed over and success among those; stage counts; the estimate's horizontal error against the oracle at the hand-over and its maximum over the carry, 50th and 95th percentile per object and layout.

**Prediction (mine; my record is poor).** PERCEPTION HOLDS on all four, because N2 and N5 stayed at 1.00 under a 0.04 m constant error (HAND_OVER §8.2). The bottle in the new layout does worst, because the grasp moves it most.

**What this cannot show.** No plate or bowl target, so it says nothing about N1, N4 or N6. It does not remove the camera, the height or the target positions. N2, N3 and N5 are among the easier new pairings, as §8.2 shows. Three tasks on one scene, with development states that shaped the locator.

### 10.1 Results (2026-11-05)

All 12 run jobs and the read completed. `results/flawed/pp_hand3_read.json`.

**Registered verdicts: GR00T familiar PERCEPTION HOLDS, with KNOWN COST. The other three NO FIX.** My prediction held on one of four.

| N2, N3, N5 | intact | H1 | LS | LS − intact [95% CI] | H1 − LS [95% CI] | known, H1 → LS | verdict |
|---|---|---|---|---|---|---|---|
| pi0.5, familiar | .55 | .92 | .87 | +.32 [−.07, +.75] | .05 [.00, +.20] | .95 → .87 | NO FIX, KNOWN COST |
| GR00T, familiar | .30 | .98 | .98 | +.68 [+.35, +.95] | .00 [.00, .00] | .98 → .88 | PERCEPTION HOLDS, KNOWN COST |
| pi0.5, new | .38 | .53 | .44 | +.06 [−.17, +.31] | .09 [.00, +.22] | .28 → .24 | NO FIX |
| GR00T, new | .12 | .38 | .29 | +.17 [−.01, +.36] | .09 [−.03, +.27] | .28 → .19 | NO FIX, KNOWN COST |

| success per task, H1 → LS | N2 | N3 | N5 | K4 | K6 |
|---|---|---|---|---|---|
| pi0.5, familiar | 1.00 → 1.00 | .75 → .60 | 1.00 → 1.00 | 1.00 → 1.00 | .90 → .73 |
| GR00T, familiar | 1.00 → 1.00 | 1.00 → 1.00 | .95 → .95 | 1.00 → 1.00 | .95 → .77 |
| pi0.5, new | .55 → .47 | .50 → .35 | .55 → .50 | .15 → .15 | .40 → .33 |
| GR00T, new | .30 → .30 | .50 → .27 | .35 → .30 | .20 → .18 | .35 → .20 |

**How to read the verdicts.**

- **The registered loss never exceeds 0.10.** It stays at .00-.09 on all four, so no verdict reads PERCEPTION COSTS. The three NO FIX verdicts fail on `RAISES`. On pi0.5 familiar the point estimate is +.32, and three tasks leave the CI wide. In the new layout, grasping limits both H1 and LS.
- **Locator seeds barely differ.** Per-task success varies by at most .05 across the three seeds.
- **Estimate error,** horizontal, 50th and 95th percentile. At the hand-over: cheese .014-.018, .020-.034; bottle .011-.019, .018-.034. Maximum over the carry: cheese .016-.018, .025-.038; bottle .013-.021, .027-.037. So 95th percentiles sit above the 0.02 m budget.

**Where the cost falls, no verdict.**

- **Tasks with tight targets lose most.** The stove tasks N2 and N5 lose .00-.08. The cabinet task N3 and the bowl task K6 lose .07-.23, apart from N3 on GR00T familiar, which loses nothing. This fits HAND_OVER §8.2, where the stove tolerated 0.04 m and the bowl did not, though §8.2 tested constant errors on pi0.5 alone.
- **A likely design flaw in the estimate, untested.** In the new layout pi0.5 hands over in .88 of LS episodes, and only .50 of those succeed. The event can fire on an empty close (LADDER_RESULTS §4 reports this on 25% of its new-layout episodes). In that case my estimate moves with the empty hand and then freezes in the wrong place, so a regrasp heads to the wrong spot. H1 reads the true position and can recover. GRASP failures do rise under LS on all four, by .00-.05. I did not separate this cause from the rest.

## 11. Looking while the hand is open (m3_prereg `pp_hand4`)

Design fixed 2026-11-05 after §10.1, with the user's go-ahead and their instruction to keep iterating. Development states, no training. Client `experiments/pp_hand4_client.py`, read `experiments/pp_hand4_read.py`.

**Arm LR.** LS with one change: the estimate keeps updating whenever no hold is on and the gripper command is open.

- Every 5 steps under those conditions, the network looks once. The estimate is the elementwise median of up to the last 3 looks.
- At an onset of the event, the offset comes from that estimate and the buffer clears. During a hold the estimate is the hand plus the offset, as in LS.
- After a hold ends, the estimate stays put until the next look.

This addresses two things together, and the read cannot split them: the stale estimate after a false event, and the object being nudged before the grasp. The looks run during the VLA's approach too, though P still acts only after the hand-over. No look runs while the gripper closes, because the network's training frames stop at the close.

**Everything else as §10:** tasks, seeds, states, what stays oracle. The same 1,200 episodes.

**Verdict: §10's mapping, unchanged, with LR in place of LS.** Reported with no verdict: LR − LS, and everything §10 reports.

**Prediction (mine; my record is poor).** LR recovers most of LS's loss in the new layout and on K6. PERCEPTION HOLDS on both familiar cases; NO FIX stays in the new layout, because grasping limits it.

## 12. A plate cutout from figure and ground (m3_prereg `pp_loc3`)

Design fixed 2026-11-05, before this training. Offline development read. `experiments/pp_loc3.py` → `results/flawed/pp_loc3.json`.

**The change.** §9.1 blames the plate's lopsided disc, untested. No K1-K6 demo moves the plate, so first-against-last differencing cannot cut it out. Instead, in each K1 demo's first frame, the table's colour is the median colour of a ring 60-70 px from the label. The plate is the connected region near the label that differs from that colour by more than 20/255, with gaps closed, holes filled and thin bridges opened. Its anchor and head offset then work as for the other objects. The empty scene stays as in §9.

**Arms.** **CP3**: demo labels, as CP with this cutout; three seeds. **AP3**: simulator labels, as a reference. Label cleaning is dropped, because §9.1 found it adds little and makes the plate worse.

**Verdict: §5's mapping, unchanged, on CP3 alone.** AP3 takes the label as a report.

**Prediction (mine; my record is poor).** CP3 BEATS A TABLE, OVER BUDGET: the plate within .05 m, with the bowl's shift and the familiar bottle still over budget.

**Limit.** The rule works on an object that contrasts with the table and sits apart from others, as the plate does in the familiar layout. It would not separate touching objects. The 20 states keep shaping the design, so unseen states stay untested.

## 13. Results of §11 and §12 (2026-11-05)

All jobs completed. `results/flawed/pp_hand4_read.json`, `results/flawed/pp_loc3.json`.

**§11, registered verdicts on arm LR.** PERCEPTION HOLDS on three of four, each with KNOWN COST. My prediction held on both familiar cases, failed on GR00T new, and overstated the recovery.

| N2, N3, N5 | intact | H1 | LS | LR | LR − intact [95% CI] | H1 − LR | LR − LS [95% CI] | K4 and K6, H1 → LR | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pi0.5, familiar | .55 | .92 | .87 | .91 | +.36 [+.03, +.75] | .01 | +.04 [.00, +.14] | .95 → .86 | PERCEPTION HOLDS, KNOWN COST |
| GR00T, familiar | .30 | .98 | .98 | .98 | +.68 [+.35, +.95] | .00 | .00 | .98 → .87 | PERCEPTION HOLDS, KNOWN COST |
| pi0.5, new | .38 | .53 | .44 | .49 | +.11 [−.08, +.34] | .04 | +.05 [.00, +.12] | .28 → .21 | NO FIX, KNOWN COST |
| GR00T, new | .12 | .38 | .29 | .31 | +.19 [+.05, +.35] | .07 | +.02 [−.05, +.14] | .28 → .18 | PERCEPTION HOLDS, KNOWN COST |

- **K6 does not recover.** It stays at .72, .73, .27, .18, against H1's .90, .95, .40, .35. K4 holds level with H1. So the known cost sits on the bowl target, and the stale estimate does not explain it.
- **Looks sometimes go badly wrong.** At the 95th percentile the estimate's error over the carry reaches .32 m on the cheese and .12 m on the bottle, against .03-.04 m under LS. Median errors stay near LS's, except GR00T's new-layout cheese at .037 m, and the familiar bottle rises to .024-.026 m at the hand-over. The hand near the object probably confuses the head. Success still rises, so the gain outweighs these errors on these tasks.

**§12, registered verdict on CP3: NO BETTER THAN A TABLE.** My prediction failed. The plate worsens to `e` .312 m, median .049 m, with 7-10 bad states per seed. CP had median .012 m and 3-5 bad states. AP3 reads BEATS A TABLE, OVER BUDGET, with the plate at .030 m and 0, 1, 0 bad states. The other heads barely move. A tight cutout with a label about 12 px off its centre does worse than a loose disc. So the off-centre label looks like the problem, and the cutout's shape does not.

## 14. Labels at the object's centre (m3_prereg `pp_loc4`)

Design fixed 2026-11-05 after §13, before this training. Offline development read. `experiments/pp_loc4.py` → `results/flawed/pp_loc4.json`.

**The reasoning.** Three problems may share one cause: where the hand closes or opens is not the object's centre. The plate's label sits off its centre (§13). The raw locator carries that shift into P, which trained on body positions, and that may explain K6's cost, because the bowl tolerates little error (HAND_OVER §8.2). The bowl's constant shift under P (§9.1) may come from a fixed pixel offset between label and cutout. A centre label removes that offset.

**Arm CC.** As CP3, with one change to the labels. Each demo's label is the centre of its own cutout, in pixels. The hand no longer supplies the position, only the demo events and the region to search.

- **Cheese, bottle, bowl as object:** the centre of the first-against-last cutout, per demo. A demo with no cutout takes its head's mean.
- **Plate as target:** the centre of the figure-and-ground cutout, per K1 demo, with the same fallback.
- **Bowl as target:** K6 demos do not move the bowl, so this head takes the bowl's mean centre from K1-K3 on every K6 frame.

**Lifting heights, still from the hand.** Each object head uses its mean hand height at the close. The bowl-as-target head uses the bowl's. No demo grasps the plate, so the plate uses the lowest of those three heights, as the nearest thing to the table the hand has reached. The plate's true height is about 0.016 m lower, which adds a small constant shift.

**Verdict: §5's mapping, unchanged, on CC.** Reported with no verdict: raw error and `b`, because raw error is what P will see. Three seeds. No simulator-labelled arm: arm A's labels are already at the body positions.

**Prediction (mine; my record is poor).** BEATS A TABLE, OVER BUDGET, with the plate under .05 m. Raw error below CP's on the plate and the bowl. The familiar bottle stays near .02 m.

### 14.1 Result (2026-11-05)

All jobs completed. `results/flawed/pp_loc4.json`. All 279 demos gave a cutout. **Registered verdict on CC: BEATS A TABLE, OVER BUDGET.** My prediction held on the verdict and on raw error, and failed on the plate's `e` (.059 m).

| CC, m | cheese | bottle | bowl, object | plate, target | bowl, target |
|---|---|---|---|---|---|
| `e`, new | .015 | .021 | .032 | .059 | .034 |
| raw, new | .015 | .051 | .013 | .018 | .014 |
| raw median, new | .013 | .043 | .005 | .006 | .006 |
| `e`, familiar | .004 | .006 | .029 | .057 | .013 |
| raw, familiar | .011 | .056 | .008 | .098 | .040 |

**Reading, untested.** In the new layout, raw errors sit near or inside the budget on every head except the bottle, and that is the first read where the plate gets there. `e` looks worse than raw because `b`, estimated on familiar-layout demos, moves correct predictions away: the plate's `b` is .052 m, and its 14-17 bad states per seed come with a raw median of .006 m. The familiar plate reads .098 m raw, so the network or its labels may be wrong around the plate's familiar position, perhaps because the figure-and-ground cutout merges with something nearby. The bottle's raw shift of about .05 m may come from lifting its centre at the grasp height. I did not diagnose either. This work is now paused under `docs/PLAN_SCAFFOLD.md`.

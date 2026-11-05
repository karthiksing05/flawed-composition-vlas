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

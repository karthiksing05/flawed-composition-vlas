# SR 86 diagnostic: why W3 loses K5 and the unique-name controls (EXPLORATORY, not registered, no new arm, nothing implemented)

Written 2026-11-20 after SR 85.  All numbers come from existing SR 82 / SR 85 rollouts (Phase B `wasl` vs `wasw3`, 60 / 40 paired states per cell) plus 40 new instrumented K5 episodes (20 W3, 20 RAA-sL, the SR 85 diagnostic states trials 24-43, jobs 4811457 / 4811458).
Scripts (all `experiments/sr86_*.py`; outputs `results/flawed/sr86_*`): `sr86_instr.py` (copy of raa_instr.py adding named object / robot contacts and the rack geometry), `sr86_q1a..e.py`, `sr86_q2.py`, `sr86_q2t.py`, `sr86_q2b..d.py`.  Analysis jobs 4811452-4811608 (CPU, overcap).
One seed, one trial per state, 20 states for K5; no label is attached to any number here.

## Q1. K5 (rack): why does W3 stall after the xy gate opens (.067 vs RAA-sL .683)?

### Finding
The module's target is the wrong point for the rack.  The channel's target reference for K5 (the site `wine_rack_1_top_region`) is **12.5 cm in xy from where the demos release**, and W3 drives the object to the reference itself.  The held bottle parks on the reference, the arm's wrist presses the rack, the expert's descent is blocked, and the episode runs out the clock.

### Evidence
(a) Demo release point in the channel frame (target minus object, cm; first open command while role = target; training demos, `sr86_q1d.json`):

| target class | n | dx | dy | dz | xy median | xy p95 |
|---|---|---|---|---|---|---|
| plate | 50 | -0.5 | -0.2 | +0.1 | 0.9 | 2.1 |
| bowl | 50 | +0.3 | 0.0 | -6.9 | 1.4 | 3.3 |
| stove | 50 | -3.3 | -0.8 | -2.3 | 3.5 | 5.5 |
| cabinet | 100 | +4.2 | -3.2 | 0.0 | 5.3 | 8.7 |
| **rack** | 50 | **-8.9** | **+8.1** | +2.2 | **12.5** | 15.4 |

The rack offset is stable (xy p5 9.2, p95 15.4; dx p5-p95 -11.8 .. -5.6, dy +4.4 .. +12.0), so it is a property of the rack's reference, not noise.  RAA-sL's 16 successes release at channel rel (-7, +10.5, -1) cm (median), the demos' point (sr86_q1d.json, `chan_rel_at_success_cm`).

(b) What W3 does (20 episodes, success 1 / 20; `sr86_q1b/q1d/q1e`).  The module owns xy from env step ~65-80 in 19 / 20 episodes.  Its commanded setpoint is channel rel_xy = 0: the median own-phase channel rel_xy is 0.8 cm.  12 / 20 episodes hover at the reference (>= 10 queries with the module owning xy and channel rel_xy < 1.5 cm), at c = 4-9 cm above the reference, chan rel (0, 0, -4 .. -10).  The expert's xy, which the module overrides, has cosine 0.62 (median) with the module's and norm 0.69 vs 0.41: the expert is still trying to go elsewhere.

(c) The blocker is the arm, not the bottle.  During the hover (19 episodes with one, about 1,500 hover steps pooled) the **wrist link `robot0_link7` is in contact with the rack in 18 / 19 episodes** (link6 in one, the hand in 3); the bottle itself touches nothing but the gripper (`obj non-gripper contacts: {}` in all 19).  The expert's z command during the hover is negative (median mean -0.36: it keeps asking to descend) and the object rises or stays (median +4.8 cm over the hover); 6 / 20 episodes have the arm on the rack in the last 100 steps.  SR 85 saw only one W3 episode with target contact because its flag is bottle-rack contact; the bottle never touches the rack, the arm does.

(d) Where RAA-sL goes instead: the bottle ends 10 cm in +x from the reference with its side against the rack plate (success rows: first non-gripper contact `wine_rack_1_g1` in 16 / 16 successes; origin rel at success (-8.4, +2.1, -3.0) cm), exactly the demos' pocket.  Geometry (`sr86_q1b.json` `_static`): the rack collision boxes lie in one vertical plane x = -0.2566 m, y -0.33 .. -0.15, z 0.93 .. 1.17.

(e) Side facts: the channel's object reference is not the body origin (median channel-minus-origin rel (-0.2, +8.3, +4.1) cm, sd 1-4 cm), which is why the npz rel and the channel rel differ in the plots; the channel's target error on K5 is small (median 0.7 cm, p95 1.3), so this is not a perception error.  SR 85's reading "cause unknown, perhaps overshoot / reference-height issue" is replaced: it is the xy aim point.

### Cause
`WHERE` = "the target reference" is right for plate (offset 0.9 cm) and bowl, borderline for stove / cabinet (3.5 / 5.3 cm, inside those targets' success tolerances: K3 .967, K4 .983), and wrong for the rack (12.5 cm, outside the pocket and pushing the arm into the frame).  The expert knows the pocket from the demos; the module does not.

### Proposed change (ONE; not implemented): the module aims at the release point, not at the reference
In `where_own_apply`, replace the servo input rel_xy by rel_xy - mu_xy, where mu = the median channel rel at the release frame over the **training demos of the same target class** (the table above: rack (-8.94, +8.13), plate (-0.47, -0.15), ...); the class comes from the role phrase the channel already carries ("target: the rack"), mu is computed once from the demo set, no evaluation state is used.  The module still owns only WHERE (now defined as where the demos let go); the expert still owns lift, descent, rotation, gripper.  Unknown classes get mu = 0 (today's behaviour).  Prediction to register: K5 up toward RAA-sL's .68 (the arm no longer reaches the frame), K3 / K4 / N3 / F6_b unchanged within noise (their mu is 0-5 cm); the risk is the +-3 cm spread of the rack offset (dx / dy p5-p95), which the expert's own last-centimetres control would absorb only if C2 below is also adopted.

## Q2. Why did W3 cost the unique-name controls (.600 -> .458), the F7 pool (.744 -> .669) and F6_a (.983 -> .883)?

### Paired result (same states; `sr86_q2.json`, full arm, success counts SS / lose (RAA-sL ok, W3 fail) / gain / FF)
| cell | RAA-sL | W3 | SS | lose | gain | FF |
|---|---|---|---|---|---|---|
| F7 relL (tid 0) | 1.000 | .825 | 33 | 7 | 0 | 0 |
| F7 relR (1) | .900 | .775 | 31 | 5 | 0 | 4 |
| F7 pos (2) | .075 | .075 | 3 | 0 | 0 | 37 |
| F7 neg (3) | 1.000 | 1.000 | 40 | 0 | 0 | 0 |
| **F7 uniL (4)** | .925 | **.500** | 20 | **17** | 0 | 3 |
| F7 uniR (5) | .875 | .875 | 34 | 1 | 1 | 4 |
| F7 uniP (6) | 0 | 0 | 0 | 0 | 0 | 40 |
| F6_a (9) | .983 | .883 | 53 | 6 | 0 | 1 |
| K1t (10) | .950 | .883 | 53 | 4 | 0 | 3 |
| K2t (7) | .975 | 1.000 | 39 | 0 | 1 | 0 |

The controls gate (tids 4-6) fell because of uniL alone (17 flips; uniR and uniP unchanged, uniP is 0 for both arms).  The F7 pool loss is relL (-7) and relR (-5).  There are 40 success-to-failure flips and 2 gains in total.  Same cells, same states, same checkpoint: the only difference is who owns xy while c >= 3 cm.

### Finding
The channel's **target** is right on these cells (the resolver picks the right entity: held-query median error of the plate 0.04 cm, p90 0.23; of the held bowl 0.95 cm, p90 2.8).  The flips happen at **release**: 33 of the 36 non-K1t flips release the bowl for the first time >= 3 cm (LIBERO's body-target xy tolerance, instr docs) from the plate centre; 2 never release a carry; 1 releases inside 3 cm.  The module nulls the channel's estimate of the bowl-plate offset, so the true offset left at release is the channel's held-object error plus the servo's overshoot; the expert uses the image and ends within 1-2 cm.

### Evidence (`sr86_q2b.json`, `sr86_q2c`, `sr86_q2d` outputs in the job logs 4811588, 4811590, 4811591)
First-release true xy offset of the bowl to the plate (cm), W3 vs RAA-sL, same cells:

| cell | W3 median | RAA-sL median | W3 frac >= 3 cm | RAA-sL frac >= 3 cm |
|---|---|---|---|---|
| F7 relL | 3.29 | 1.25 | .67 | .12 |
| F7 relR | 3.28 | 1.52 | .62 | .45 |
| F7 uniL | 4.42 | 2.28 | .93 | .39 |
| F7 uniR | 3.11 | 1.89 | .50 | .44 |
| F6_a | 4.00 | 2.97 | .78 | .50 |

Success vs the episode's median channel error on held queries (pooled F7 relL / relR / uniL / uniR and F6_a):

| channel error (cm) | W3 success (n) | RAA-sL success (n) |
|---|---|---|
| < 0.5 | .983 (60) | 1.000 (59) |
| 0.5-1.0 | .919 (37) | .949 (39) |
| 1.0-1.5 | .787 (47) | .943 (53) |
| 1.5-2.0 | .683 (41) | .972 (36) |
| >= 2.0 | .438 (32) | .867 (30) |

RAA-sL is nearly flat in the channel error (the expert corrects with vision); W3 falls from .98 to .44.  The error is the bowl's: per-episode median held-object error 0.95 cm (p90 2.8) vs plate 0.04 cm.  uniL has the largest (median 1.74 vs .94-1.34 for the other cells), which is the cell with the most flips.  Example: uniL trial 28, channel says rel_xy 0.3 cm for 80 steps while the bowl sits 3-6 cm off the plate (err 2.15 cm every query), the bowl hangs 10-13 cm above and the episode times out.

Overshoot (second component): the first-release offset is 1.9 cm larger than the channel error in W3 (median offset - error 1.86 vs 0.33 for RAA-sL; the correlation with the channel error is weak, Pearson .23, so the channel error is necessary but not the whole story).  Chunk-level: with the channel rel_xy between 1 and 4 cm, a W3 module-owned 5-step chunk moves the object 2.8 cm (p90 5.8) and **45 %** of such chunks pass through the target (|rel| >= 1 cm on the other side), vs 1.4 cm and 11 % for RAA-sL (n = 401 / 314); far from the target the module moves 7.1 cm per chunk vs 4.5.  (Timeline uniL trial 6: x offset -5.9 -> -1.3 -> +2.8 cm in 10 steps.)  Timelines: `results/flawed/sr86_q2t.txt` (uniL trials 6, 7, 28, 10, both arms).

### Cause
Not the resolver and not a collision: the module owns xy down to the last centimetre, but its measurement of the held object (p90 error 2.8 cm) is coarser than the 3 cm success tolerance and coarser than the expert's visual alignment, and its 5-step open-loop chunk at the demos' clip speed overshoots when it is within 4 cm.  The module owns a decision (the last 3 cm) that it measures worse than the expert.  On F6_b the same ownership is a gain because the expert's error there is 20 cm (wrong plate); the loss is the price of owning the part of WHERE that is already finer than the module's resolution.

### Proposed change (ONE; not implemented): the module owns xy only where its resolution is finer than the task demand
Add a distance condition to the ownership gate: own = held AND c >= CLEAR AND rel_xy > R, with R = 3 cm, fixed from the measured held-object error (p90 2.4-2.8 cm over the 217 W3 / 217 RAA-sL episodes) and equal to the plate's success tolerance, not tuned on F6_a / controls.  Inside R the chunk is the expert's (RAA-sL exactly), so the expert owns the last 3 cm (alignment and release: HOW), and the module owns the long-range decision (WHERE), which is where it fixed F6_b.  This removes both components at once (bias and overshoot only act inside R).  Prediction to register: unique-name controls and F6_a return toward RAA-sL (.60 / .98), F6_b stays near W3's .72 (its error is 20 cm, outside R); the risk is F6_b cases where the module's last 3 cm decided the plate choice (none seen: the target error is 0.04 cm).

## Interaction
C1 (aim at the release point) and C2 (hand back inside R) are independent: C1 shifts the setpoint, C2 limits where the module acts around it.  For the rack, C2 around the shifted setpoint leaves the last 3 cm of the pocket approach to the expert, which also covers the +-3 cm spread of the rack offset.

## Limits
20 states per K5 arm; K5 contacts come from one run per state; hover is defined by the channel (own and rel_xy < 1.5 cm); the demo offset is the first open command while role = target; for Q2 the first release is the first open command of the gripper after a lifted, closed carry, from the simulator track (tids 0, 1, 4, 5, 9 only; K1t's 4 flips were not analysed; K2t / K1t have site targets); the success tolerance 3 cm is the LIBERO body-target check from the instr docstring, not re-measured; a 5-step chunk displacement is read from the channel's rounded integer centimetres (+-0.5); the error bins are over episodes, not queries; cause statements are correlational for Q2 (no intervention was run) and observational-causal for Q1 (contact names, geometry and demos agree), also not intervened.
Jobs: instrumented K5 4811457 (W3), 4811458 (RAA-sL); analysis 4811452-4811461, 4811587-4811591, 4811603-4811608.

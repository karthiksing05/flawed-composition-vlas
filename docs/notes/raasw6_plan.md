# RAA-sL-W6 (SR 90): the module owns WHERE until the object is inside the release envelope; the expert then owns HOW.  Third fresh-state confirmatory run.

Written 2026-11-20, before the prereg row and any W6 rollout.  Predictions with probabilities: experiments/prereg/raasw6_pred.txt (copied into the row).  Origin: SR 88 diagnostic (docs/notes/sr88_diag.md), proposed changes 1 and 4 (1 in its envelope form).
No training; same checkpoint as SR 82-87 (pi05_g8v_raas_l_ae main/6000).

## Principle
The module owns WHERE: where the object goes AND where it is let go, until the object is inside the release envelope that the training demonstrations show for that target class; then the expert owns HOW (final alignment, release).
SR 88 found W5's fixed R = 3 cm hand-off fails because it hands off too high (N4: 11.7 cm above the plate, the expert presses the cabinet), too fast (K6 / K1 overshoot) and with a target estimate that jumps after hand-off (N6).
W6 answers the first and the third with quantities measured on the training demos; the second (speed) is SR 88 change 2 and is NOT implemented here (future work).
Envelope of a class = (height above the target's reference <= h_class) AND (xy distance from the release point <= r_class): h_class = the demos' median release height + their p90 spread; r_class = the p90 of the demos' xy scatter around mu.
The release gate is the same envelope: the module owns the gripper command too, so the object is not let go outside the envelope.

## Arms (test-time only; the module changes xy [dims 0-1] and the open -> closed gripper command; z and rotation stay the expert's)
| arm | where_own | aim | module owns xy while | gate (open -> closed) | target latch |
|---|---|---|---|---|---|
| W6 | 8 | rel - mu | held, c >= 3 cm (CLEAR, W4's), and OUTSIDE the envelope (c > h_class or \|rel - mu\| > r_class) | held and outside the envelope | yes |
| W6nl | 9 | as W6 | as W6 | as W6 | no |
| W5 | 5 | rel - mu | held, c >= 3 cm, \|rel - mu\| > 3 cm | \|rel - mu\| > 3 cm | no |
| W5c1 | 7 | rel - mu | held, c >= 3 cm | \|rel - mu\| > 2 cm | no |
| RAA-sL | 0 | - | expert | none | no |
Unknown class (target phrase not in the table): W6 / W6nl == W4 bit for bit (tested).  Inside the envelope the chunk is exactly RAA-sL's.  CLEAR is kept (W4's component of W5c1); the gate does not use CLEAR (as W4's).
Class = the role phrase (as SR 87); class tables in lifelong_ood/policy/raasw6_env.json, computed ONCE by experiments/raasw6_env.py from the training demos (lol/libero_g8v_raac for mu, c, r; data/fa1/rv chan_*.npz of the K1-K6 demos for tau) and committed before any rollout.
Target latch (server-side, lifelong_ood/policy/w6_latch.py; raa_w6.py / wa_sw6.py): the held-target channel is rel = target_estimate - eef - offset; the target estimate is latched at carry start (first target-role query after a non-target query, or a new episode); a later estimate replaces it only if |new - latched| <= tau = the p99 over the training demos of the target pose's 5-step change (the serving replan interval); else the latched value stays.  Everything downstream (expert, module) reads the latched channel.  W6nl uses the unlatched estimate.

## Constants (training demos only; no rollout)
Printed in the pred file and SR 90.  tau = 0.437 cm (p50 = p90 = 0: the demos' target is static; the p99 comes from the few demos where a movable target is nudged; max 3.6 cm).  Not tuned on any rollout.

## Fresh states (the THIRD set; none used in SR 70-88)
Phase A: n2 familiar states k = 640-699 (60), harness trial = k - 640, branch seed_base 13000000 + 1000 k, sampler np.random.seed = env.seed = 600000 + k (fc_n2_wy.py = fc_n2_wx.py with K0 = 640; it reproduces stored k 0, 23, 540, 579, 580, 639 exactly and refuses any state allclose to a stored set incl. SR 87's wx).
Phase B: rg_comp_y.py (= rg_comp_x.py with F7 candidates k = 520-599 (80), F6 candidates k = 620-699 (80)); first 4 accepted of each = smoke rows, UNUSED; next 60 = main per cell; seeds F7 820000 + k / layout 821000 + k, F6 810000 + k / plate 811000 + k, branch seed_base 14000000 + 1000 k.  Accepted k lists: SR 90.
All compared arms are re-rolled on these states (W6, W6nl, W5, W5c1, RAA-sL, rgp, frozen I, W6 noR / noM).  SR 87 rollouts are not reused.

## Registered reads (paired bootstrap B = 2000, rng 0; experiments/raasw6_read.py = raasw5_read.py with the arms changed; the same resampling as SR 87)
Phase A:  A1 PRIMARY-A W6 - RAA-sL N1-N6 (SUPPORTED iff lower > 0, REVERSED iff upper < 0, else NOT SUPPORTED);  A2a W6 - rgp N1-N6 NON-INFERIOR iff lower > -.10;  A2b W6 beats rgp iff lower > 0;  AK W6 - RAA-sL K1-K6 NON-INFERIOR iff lower > -.05 (COST iff upper < 0);
  A4 W6 - W5 on N1-N6 (SUPPORTED iff lower > 0, REVERSED iff upper < 0);  L W6 - W6nl on N1-N6 (the latch; SUPPORTED iff lower > 0, REVERSED iff upper < 0).
Phase B (wa_read on b_w6): D1' full - noR on the F7 pool; D2' full - noM on F6_b; D3a noR - full on F6_b and D3b noM - full on F7 pool (EQUIVALENT iff inside +-.10); DK full - noR on K1t + K2t (NON-INFERIOR iff lower > -.05); gates on full W6: F6_a >= .50, unique-name controls (tids 4-6) >= .50, K floor >= .70; gated PRIMARY; B1 full_W6 - full RAA-sL F6_b and B2 full_W6 - frozen I F6_b (SUPPORTED iff lower > 0).
Descriptive (no label weight): per-cell tables for all arms; W6 - W5, W6 - W5c1 (envelope), W6 - W6nl (latch), W6nl - W5c1 on N, K, controls, F6_a, F7, F6_b; per-cell Y contrasts on N4, N6, K1, K6, K5.
Scoring: every prediction in raasw6_pred.txt is scored as written; deviations in SR 90.  One seed, one trial per state, one checkpoint; no multiplicity correction.

## Limits stated in advance
The envelope's h_class for the bowl (9.2 cm) is close to W5's measured hand-off heights (9-10 cm), so W6 may change little for bowl targets (N6, K1, K6); speed (K6 / K1 overshoot) is not addressed.  The release gate blocks open commands outside the envelope: a stall (arm hovering with a closed gripper) is possible and is not guarded against.  The latch freezes the estimate at carry start (tau 0.44 cm), so a wrong estimate at carry start is carried through the whole carry.  mu / h / r are class medians of 50-100 demos (the rack's spread +-3 cm).  New states: baselines are re-rolled.

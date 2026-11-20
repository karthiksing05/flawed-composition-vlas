# RAA-sL-W3 / W4 (SR 85): the module owns WHERE only once the held object is clear of the target height; the expert owns HOW

Written 2026-11-20, before the prereg row and any main rollout.  Predictions with probabilities: experiments/prereg/raasw3_pred.txt (copied into the row).  Diagnostic: docs/notes/sr85_diag.md.
Code: lifelong_ood/policy/backbone_w.py (modes 3 / 4 added to `where_own_apply`; modes 0 / 1 / 2 untouched), experiments/raa_w.py and wa_sw.py (only the argparse `choices` tuple extended to 0-4), experiments/raasw3_test.py (identity + semantics),
experiments/raasw3_read.py (copy of raasw_read.py, arms raasw3 / raasw4), experiments/prereg/raasw3.py; scripts /coc/flash10/ksingara3/.claude_tmp/pp/raaw3_roll.sh, wasw3_roll.sh, raasw3_test.sh, raasw3_prereg.sh (copies of the SR 84 scripts).  No training; same checkpoint as SR 82 / 84.

## Principle
The module owns WHERE (the xy of a held-target transport); the expert owns HOW (lift, approach, descent, rotation, gripper, and every motion in which the object is not yet above the target's height).  SR 84's W1 gave the module xy from the
first held step.  The diagnostic (sr85_diag.md) shows what that does on K3 / K5: the object has moved 3 cm sideways after 4-5 steps, ~6 cm above its start and ~10 cm below the target's reference height, pushes into the cabinet / rack and stalls
(K3: 20 / 20 episodes with target contact > 5 cm out, 20 / 20 stalls; K5: 16 / 20 stalls; RAA-sL lifts 12-15 cm first).  WHERE is only well defined once the object can travel horizontally without hitting the target: W3 hands xy to the module at that point
and leaves it to the expert before.

## W3 (`--where_own 3`)
    c   = -dz,  dz = the channel's target-relative z (target minus held object, metres; the third state number, un-normalised as in raas_model),
    own = (role id = 2) AND (c >= CLEAR),     CLEAR = 0.03 m
    own:      action xy (dims 0, 1, all 10 chunk rows) = clip(K_w * rel_xy, +-servo_max)   [W1's controller, K_w = 46.875 per m, unchanged]
    not own:  the chunk is exactly RAA-sL's (expert xy included); m = 0 frames are bit-identical to RAA-sL as before.
z, rotation, gripper are the expert's in every case.  The decision is a function of the current query's channel only (no memory): when the object descends below CLEAR near the release the expert has xy again.
## W4 (`--where_own 4`)
W3 + SR 84's release gate, unchanged: while m = 1 and |rel_xy| > 2 cm an open gripper command is replaced by closed (raw +1), everywhere (also where the expert owns xy; this is a risk on elevated targets, stated, not tuned).

## CLEAR = 3 cm (fixed here, before any read; not tuned on any registered cell, no cell was run with any other value)
Derivation, from things stated: (a) the demos release on elevated targets at c <= +1.3 cm (p90 over 150 demos, sr85_clear.json: p5 -3.8 cm, median -0.4 cm), so an object at c >= 3 cm is above where the demos themselves let go; (b) the channel's median
xy error is 1.3 cm (SR 82 diagnostic), so one such error of margin on top of the 1.3 cm gives 2.6, rounded up to the next whole centimetre = 3 cm; (c) 3 cm is also the value the lead suggested before any reading.  The diagnostic's rollouts
(20 of the 60 registered states per cell for K3 / K5) were NOT used to choose it.  Known: for non-elevated targets the demos release at median c = +2.4 cm and p90 +7.9 cm (some at c < 3), and for elevated targets the demos start sideways
motion while still 3-8 cm below the target's reference (c at 3 cm of xy progress: median -7.5, p75 -3.4 cm), so W3 does NOT replace the expert on the approach of elevated targets.  What it does there depends on whether the expert's own carry ever reaches c >= 3 cm; if it
does, the module drives xy at the clip from that moment and the expert's z is not coordinated with it (risk).

## Reads (identical to SR 84's registered set; paired bootstrap B = 2000, rng 0; 60 states per Phase A task; SR 82 states and arms; machinery raa_read / wa_read via experiments/raasw3_read.py)
Phase A (raaw3_roll.sh, ARM=raac, CONFIG=pi05_g8v_raas_l_ae, the RAA-sL checkpoint; NAME raasw3 / raasw4 into results/flawed/raa; main and rx states; tids 0+2+4+6+8+10 (K1-K6) and 16+18+20+22+24+26 (N1-N6)).  For W in {W3, W4}:
- A1 PRIMARY-A  W - RAA-sL (SR 82 rollouts), N1-N6: SUPPORTED iff lower > 0; REVERSED iff upper < 0; else NOT SUPPORTED.
- A2a  W - rgp, N1-N6: NON-INFERIOR iff lower > -.10; A2b W beats rgp iff lower > 0.
- AK   W - RAA-sL, K1-K6: NON-INFERIOR iff lower > -.05.  Fit gate R0 K1-K6 >= .80.
- Unlabelled: W - RAA-s(78b), W - RAA-L, bottle / cheese, placed-given-grasp, W - rgp on K, W4 - W3 (N, K), W - W1 (SR 84) on N and K; per task success vs RAA-sL / rgp / W1 (K3, K4, K5, N3 are the elevated cells where W1 lost .65-.97).
Phase B (wasw3_roll.sh, results/flawed/wasw3 and wasw4, SR 82's cells: full tids 0-10, noR 0-3 + 7 + 8 + 10, noM 0-3 + 8; the arms full / noR / noM; job split as SR 84):
- wa_read labels D1' full - noR on F7 pool, D2' full - noM on F6_b (SUPPORTED iff lower > 0), D3a, D3b (EQUIVALENT iff inside (-.10, +.10)), DK (NON-INFERIOR iff lower > -.05), gates (F6_a >= .50, unique-name controls >= .50, K floor >= .70), gated PRIMARY (SR 80 / 82 rule).
- B1 full_W - full RAA-sL (SR 82 wasl), F6_b: SUPPORTED iff lower > 0.  B2 full_W - frozen I (.517), F6_b: SUPPORTED iff lower > 0 (REVERSED iff upper < 0).  B3 descriptive: full_W - RAA-sL on F7 pool, F6_a, K1t + K2t; full_W - rgp; full_W - full W1 (SR 84) on F6_b, F7, F6_a, K; noM replication state by state.
Exploratory (not registered, labelled): none planned beyond what the diagnostic already did; a per-cell release / stall instrumented run on K3 / K5 under W3 may be added after the read, as a mechanism check only.
Scoring: every prediction in raasw3_pred.txt is scored as written; deviations are listed in SR 85.  The selftest reproduces SR 82's P3a / P1 / D1 / D2 and SR 84's A1 (W1) from the files before the read.
One seed, one trial per state, one checkpoint, no multiplicity correction across W3 / W4 and labels (as SR 84).

## Limits stated in advance
The gate has no memory (instantaneous c); c uses the target's reference point (a body origin or a site) and the channel's z, whose error is unmeasured; the demos leave elevated targets with c near 0 so W3 can only help where the expert's carry exceeds CLEAR;
the module's saturating servo in 5-step open-loop chunks can overshoot (as in SR 84); W4's gripper gate also acts where the expert owns xy.

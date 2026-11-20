# RAA-sL-W (SR 84): the WHERE of transport owned by the module, test time only

Written 2026-11-19, before the prereg row and any main rollout.  Predictions with probabilities: experiments/prereg/raasw_pred.txt (copied into the row).
Code: lifelong_ood/policy/backbone_w.py (new; PolicyBackboneW), experiments/raa_w.py and experiments/wa_sw.py (copies of raa.py / wa_s.py with one guarded line, flag `--where_own 0|1|2`,
default 0), experiments/raasw_test.py (bit-identity and semantics), experiments/raasw_read.py (copy of raasl_read.py), experiments/prereg/raasw.py; scripts
/coc/flash10/ksingara3/.claude_tmp/pp/raaw_roll.sh, wasw_roll.sh, raasw_test.sh, raasw_prereg.sh (copies of raa_roll.sh, wasl_roll.sh, raasb_test.sh, raasl_prereg.sh).
No existing config, script, file or checkpoint is edited; no training.

## Principle: de-distribute WHERE
In RAA-s / RAA-sL the xy of a transport step is a sum of two parties: the module's proportional servo (K = 2.85 per m, a ~0 step in the last 3-5 cm) and the flow expert's residual.  Near the
target the expert's xy opposes the servo (cos -.6 to -.9 in F6_b, sr82_diag.md), so the bowl stalls and the expert's descend-and-release fires 3-6 cm short; the where is shared and the
weaker, more reliable party loses.  SR 84 asks what happens when the where belongs to one party: while a target is held (m = 1, role id 2) the action's xy (dims 0, 1) come ONLY from the module;
the expert keeps z, rotation and gripper (the HOW).  Everything else (m = 0, object role, no channel) is bit-identical to RAA-sL.

## The controller (fixed here, before any read)
    xy = clip(K_w * rel_xy, -max_step, +max_step)        rel_xy = the channel's target-relative xy in metres (the same quantity the existing servo reads), all 10 chunk rows
    K_w = max_step / D_SAT = 0.9375 / 0.02 m = 46.875 per metre
max_step = .9375 raw action units is the existing servo's M (results/flawed/raas_gate.json: pooled p99 of |demo action xyz| on held rows = LIBERO's own per-step clip).  D_SAT = 2 cm is the
registered distance beyond which the command is at the demos' own clip; the old gain K = 2.851 reaches M only at 33 cm, so K_w is 16.4 x the old gain and its linear zone is |rel| < 2 cm
(within the channel's 1.3 cm median error on F6_b).  Units: the command is mapped into the normalised action space by the same affine as raas_model.servo_norm_torch
(x 2 / (q99 - q01 + 1e-6) on dims 0, 1, the quantile offset cancels).  z keeps what RAA-sL gives (expert residual + the old proportional z servo); rotation and gripper are the expert's.
K_w and D_SAT are not tuned on any registered cell; no cell was run with any other value.  Known risks (stated, not tuned): the executed chunk is 5 steps open loop (replan 5), so a
command at the clip can carry the object several cm in one chunk and overshoot a 2 cm target; the servo moves the held object sideways from the first held step (the expert lifts first,
and avoids obstacles with its xy).

## Arms
- W1 (`--where_own 1`): xy only, as above.
- W2 (`--where_own 2`): W1 plus a release gate: while m = 1 and |rel_xy| > 2 cm, a gripper command with raw value below 0 (open; LIBERO: -1 open, +1 closed, checked against the demos in
  raasw_test.py) is replaced by the closed command (raw +1; LIBERO -1 open / +1 closed, verified on the demos).  The 5-step chunk uses the rel at the query, so a release can be delayed by up to 5 steps
  and, within a chunk, can happen after the object has entered the 2 cm zone.  Registered as a separate arm so the two pieces are separable.

## Implementation and the identity guarantee
backbone.py is not edited.  PolicyBackboneW(PolicyBackbone) overrides `sample` only: it calls the base sample with full=True (so the base path, servo included, runs unchanged),
then applies `where_own_apply` (replace dims 0-1 of every chunk row by the saturating servo where m = 1; for W2 the gripper gate) and returns real rows or the full chunk as asked.
raa_w.py / wa_sw.py construct PolicyBackboneW only when `--where_own` is 1 or 2; at 0 (default) `_make_bk` is a literal `PolicyBackbone(name, checkpoint_dir=ckpt, device="cuda")`.
Test (experiments/raasw_test.py, sbatch): the SR 82 row's hashes of raa.py / wa_s.py / backbone.py / raas_model.py / raac_model.py / serve.py match the files; the copies differ from their
originals by the one replaced line plus added lines; with where_own 0 `_make_bk` builds the plain class with the identical call; on the RAA-sL checkpoint PolicyBackboneW at m = 0 gives
np.array_equal outputs to the base sample (default, full, 4 steps), and at m = 1 every dim except xy (W1) / except xy and gripper (W2) is bit-equal to the base sample, xy equals an
independent numpy controller and is independent of the noise.  Output is quoted in SR 84.

## Registered reads (paired bootstrap B = 2000, rng 0; the SR 82 states and arms; machinery raa_read / wa_read, via experiments/raasw_read.py, a copy of raasl_read.py)
Phase A (SR 73's 60 states per task: dev trials 4-23 + new k 540-579; rollouts raa_roll-style through raaw_roll.sh, ARM=raac, CONFIG=pi05_g8v_raas_l_ae, CKPT the RAA-sL export; NAME raasw1 /
raasw2 into results/flawed/raa; main and rx states, 12 tasks, one trial per state).  For each W in {W1, W2} (no multiplicity correction; each is its own read):
- A1 PRIMARY-A  W - RAA-sL (SR 82 rollouts), N1-N6 success: SUPPORTED iff lower bound > 0; REVERSED iff upper < 0; else NOT SUPPORTED.
- A2a  W - rgp, N1-N6: NON-INFERIOR iff lower > -.10; A2b beats rgp iff lower > 0.
- AK   W - RAA-sL, K1-K6: NON-INFERIOR iff lower > -.05.  Fit gate as SR 73 (R0 on K1-K6 >= .80).
- Unlabelled: W - RAA-s(78b), W - RAA-L on N; W2 - W1 on N and K; per task (and cheese N1-N3 / bottle N4-N6) success vs RAA-sL / rgp / RAA-s(78b) / RAA-L.
Phase B (wa_sw.py arms full / noR / noM, SR 82's cells: full tids 0-10, noR 0-3 + 7 + 8 + 10, noM 0-3 + 8; rg_comp rgc main states; wasw_roll.sh; results/flawed/wasw1 and wasw2):
- B-labels from wa_read on each directory: D1' full - noR on F7 pool (SUPPORTED iff lower > 0); D2' full - noM on F6_b (SUPPORTED iff lower > 0); D3a noR - full on F6_b and D3b noM - full on
  F7 (EQUIVALENT iff inside (-.10, +.10)); DK full - noR on K1t + K2t (NON-INFERIOR iff lower > -.05); gates as SR 80 / 82 (F6_a >= .50, unique-name controls >= .50, K floor >= .70);
  gated PRIMARY as SR 82.
- B1 full_W - full RAA-sL (SR 82 wasl), F6_b: SUPPORTED iff lower > 0.  B2 full_W - frozen I (.517), F6_b: SUPPORTED iff lower > 0 (REVERSED iff upper < 0).  B3 (descriptive) full_W - RAA-sL full on
  F7 pool, F6_a, K1t + K2t; full_W - rgp, rc on F6_b and F7.
- noM replication check (descriptive): noM never engages the module, so noM_W should equal SR 82's noM state by state; the number of identical states is reported.
Exploratory (not registered, labelled): release xy error on F6_b (last held query's |rel xy| and the final bowl-plate xy from the track files) and on N4 / N6 via logged runs, W1 / W2 vs RAA-sL.
Scoring: every prediction in raasw_pred.txt scored as written; deviations listed in SR 84.  Selftest reproduces SR 82's P3a and D2 from the SR 82 files before the read.
One seed, one trial per state, one checkpoint.  W2's gate uses the same channel as the module; with a bad channel (rel off by > 2 cm) the gate can hold the object closed until the cap.

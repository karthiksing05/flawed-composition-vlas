# WA (SR 79): module-removal test of the WHERE architecture on RAA (prereg tag `wa`)

Written 2026-11-18, before the prereg row and any main rollout.  Code: experiments/wa.py (server + client), experiments/wa_read.py, experiments/prereg/wa.py;
rollouts /coc/flash10/ksingara3/.claude_tmp/pp/wa_roll.sh.  Predictions with probabilities: experiments/prereg/wa_pred.txt (copied into the row).

## Question
RAA (SR 73, pi05_g8v_raa_ae_tl6000, no training here) with a referent resolver R (fx_r's resolver picks the bowl instance; it fills the object slot, the role view
and the prompt) and a pose memory M (sc_pu.Estimator's hold of the last good pose).  Do the modules dissociate: R needed for F7 only, M for F6 only?

## Arms (one server; the client decides)
- full: R + M.  R engaged on the 4 relational templates only (F7R's unique prompt "put the bowl on the plate|stove", roles (bowl_1, parsed target); the other bowl
  painted magenta in both cameras by fx_r.Front's distractor mask, applied on the raw renders by the server).  The Estimator sees only the resolved bowl's pixels
  (each bowl pixel goes to the nearer of two tracked 3D centres).  Off template: RAA's lexicon.
- noR: M, no R.  What RAA does with two identical bowls: the learned map has one bowl class and sc_pu takes the median over all its pixels, so it AVERAGES them.
  noR therefore binds one instance by rgp's rule (rg_comp.Episode.instances: largest 8-connected bowl component at the first query, tracked with Front's
  update rule; with one component nothing is restricted = RAA as is).  No paint.  The lexicon's roles equal R's on all four templates (job 4790725), so the prompt is the same
  as full's: R's only contribution is the instance and the distractor paint.  On K1t, K2t, F6_b R is not engaged (one bowl), so noR IS full there; noR is rolled there
  as a replicate (same code path), which makes D3a and DK a rollout-noise comparison rather than an identity.  Plain RAA = noR (no separate run).
- noM: R, no M.  Precisely: for the ACTIVE role entity (object before the server's ProprioHeld, target while held) with an estimator source in {occl, last, prior}
  (a held estimate: the occlusion rule's last good pose when the mask is under occl_frac of its first-query size, or the last / prior pose when the mask is
  empty in both cameras), the channel suffix is omitted (prompt = the role phrase alone, "target: the plate") and the server paints no role view on that query (the
  client image, which already carries R's distractor paint, goes through).  Measured entities (agent, wrist, rigid) and fixtures (site) are exactly full's.
- noMe (descriptive): as noM but only {last, prior} are withheld; the occlusion rule's hold is kept.
- References (descriptive): I, rgp, rc of SR 70 on frozen pi0.5 (results/flawed/rg_comp_roll); same states, branch files and start snapshots (checked by trial id and
  start snapshot in the read), different policy.

## Cells (rgc main states, rg_comp unchanged)
F7 tids 0-3 (40 states each), F6_b tid 8, F6_a tid 9 (60), K2t tid 7 (40), K1t tid 10 (60).  Gate cells: full on the unique-name controls tids 4-6 (40 each).
Rollouts: full 0-10 (500 episodes), noR 0-3, 7, 8, 10 (320), noM 0-3, 8 (220), noMe 0-3, 8 (220).  Client seed 7, server seed 13, one trial per state.

## Registered contrasts (paired by state; bootstrap B = 2000, default_rng(0), percentile 2.5 / 97.5, wa_read.py = rg_comp.bootm)
D1 full - noR on the F7 pool (mean of tids 0-3, one draw of the 40 state indices for all four): SUPPORTED iff lo > 0 (REVERSED iff hi < 0, else NOT SUPPORTED).
D2 full - noM on F6_b: the same labels.  D3a noR - full on F6_b and D3b noM - full on the F7 pool: EQUIVALENT iff the whole interval lies inside (-.10, +.10).
DK full - noR on K1t + K2t (each task's states resampled separately): NON-INFERIOR iff lo > -.05, COST iff hi < 0, else NOT SHOWN.
PRIMARY: DISSOCIATION SHOWN iff D1 and D2 SUPPORTED and D3a, D3b EQUIVALENT; PARTIAL iff D1 and D2 SUPPORTED but D3 not both EQUIVALENT; else NOT SHOWN.
Gates (labels on the executor, rg_comp's rule restated; do not change the primary rule): floor full's K >= .70; F7 mean of full's tids 4-6 >= .50; F6 full's F6_a >= .50;
a contrast on a NOT TESTABLE cell is reported with that annotation.
Descriptive: full vs rc and vs I (F7 pool, F6_b), full F6_b vs F6_a, noMe vs full, per-task full - noR and noM - full, resolver / binding accuracy, omission shares per
arm, cell and role (server log), channel error vs simulator truth per role.

## Smoke (4 smoke states per cell, outputs outside results/; jobs 4790728, 4790729, 4790846)
See the SEEN paragraph in the prereg row and wa_pred.txt.  Findings that shaped the arms before the row: (1) the hold in F6_b is the occlusion rule's (src occl on the
plate, 134 of 134), which led to defining noM over all held estimates and noMe as the empty-mask variant; (2) noR on K / F6 is a replicate; (3) gate cells added.

## Information
R reads fx_r's template parser, the learned U-Net masks and the rendered depth + calibration (privileged, as SR 45 / 70).  M reads the same.  The simulator is read for
the per-query error log and the gates' diagnostics only, never fed.  Deviation from the plain-RAA protocol: wa.py copies raa.RAAServer's arm-raa logic and
sc_rv.RoleView.infer instead of importing raa.py (another agent is editing raa.py); the text format, painting and Estimator are SR 73's.

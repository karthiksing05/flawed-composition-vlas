# SR 91: registered replication of W5 (where_own 5, the SR 87 design, unchanged) on a FOURTH disjoint set of fresh start states

Written 2026-11-20, before the prereg row (m3_prereg tag raasw5r) and before any SR 91 rollout.  Predictions with probabilities: experiments/prereg/sr91_pred.txt (copied into the row).
No training, no code change to W5: same checkpoint (pi05_g8v_raas_l_ae main/6000), same serving code (experiments/raa_w.py for Phase A, wa_sw5z.py = wa_sw5.py with its two `rg_comp_x` references changed to `rg_comp_z` for Phase B), same mu table (lifelong_ood/policy/raasw5_mu.json), where_own 5, R = 3 cm, gate 3 cm, CLEAR 3 cm.
The prereg script asserts that the W5 code path (raa_w.py, wa_sw.py, wa_sw5.py, backbone_w.py, raasw5_mu.json) is byte-identical to the files hashed in the SR 87 / SR 90 ledger rows (backbone_w.py itself gained modes 8 / 9 for SR 90; SR 90's identity test, job 4813893, showed modes 1-7 torch.equal / np.array_equal to the SR 87 function and checkpoint; the file is unchanged since).

## Why a replication
SR 87 evaluated W5 on a fresh set and found the dissociation SHOWN, B1 SUPPORTED, K5 repaired (.950), but the paper's repair claim, non-inferiority to ReGuide-perc on N1-N6 at -.10 (A2a), NOT SHOWN (point -.022, lower -.114).  SR 90 re-rolled W5 on a third set as a comparator for W6 (N1-N6 W5 .931, rgp .883; A2a was not computed for W5).
Step 0 below computes the SR 90 W5 contrasts.  W5 has not been changed since it was frozen (SR 87 plan).  This run asks whether A2a replicates, on states nobody has seen.

## Step 0 (EXPLORATORY, seen before registration)
From the existing SR 90 rollouts (results/flawed/raasw6, third fresh set k 640-699; script experiments/sr91_step0.py, job 4817131, results/flawed/sr91_step0.json; paired bootstrap B = 2000, rng 0, tasks then states resampled, raa_read.diff, same resampling as SR 87).  Not registered; nothing below is a test.

| contrast (SR 90 states) | arm a | arm b | delta [95% CI] |
|---|---|---|---|
| W5 - rgp, N1-N6 | .931 | .883 | +.047 [-.017, +.125] |
| W5 - rgp, K1-K6 | .964 | .972 | -.008 [-.067, +.050] |
| W5 - RAA-sL, N1-N6 | .931 | .853 | +.078 [+.011, +.161] |
| W5 - RAA-sL, K1-K6 | .964 | .939 | +.025 [-.067, +.150] |

(Lower bound against the SR 91 margins, for reference only: W5 - rgp N1-N6 lower -.017 > -.10; W5 - RAA-sL K1-K6 lower -.067 < -.05.)  The W5 gated module-removal dissociation cannot be computed from SR 90: results/flawed/raasw6/b_w5 holds only the full arm (files: full, segdiag), no noR / noM rollouts (SR 90 rolled removal arms for W6 only).
A further exploratory dry run of the pooled routine of this plan over SR 87 + SR 90 (two sets, seen; job 4817136): P-A1 W5 - RAA-sL N +.047 [-.006, +.099]; P-A2a W5 - rgp N +.013 [-.046, +.064]; P-AK K +.017 [-.056, +.107]; K5 W5 - RAA-sL +.300 [+.217, +.392]; F6_b W5 - RAA-sL +.533 [+.433, +.625]; F6_b W5 - I +.050 [-.083, +.183].  These feed predictions 10 only as priors.

## Fresh states (fourth set; disjoint from SR 70-90 by index range and, checked by job, by np.allclose)
Used so far (SR 70-90): N1-N6 / K1-K6 n2 familiar states k 0-23, 24-195, 236-275, 500-539 (old sets), 540-579 (rx; SR 70-86), 580-639 (wx; SR 87), 640-699 (wy; SR 90).  F7: rg_comp candidates 300-379, rg_comp_x 380-519, rg_comp_y 520-599; F6: rg_comp 400-498, rg_comp_x 500-619, rg_comp_y 620-699; fx / f6o2 0-193.  SR 88 / 89 diagnostics used SR 87 rollouts and scene-2 states (seed 1090000 + k): no overlap.
- Phase A (N1-N6, K1-K6, all familiar-layout tasks share the states): n2 familiar states **k = 700-759** (60), harness trial t = k - 700, branch seed_base 13000000 + 1000 k, sampler seed np.random.seed = env.seed = 600000 + k (experiments/fc_n2_wz.py = fc_n2_wy.py with K0 = 700 and the wy set added to the refusal list; reproduces stored states k 0, 23, 540, 579, 580, 639, 640, 699 exactly; refuses any state np.allclose to a stored one).  New seed base: 13000000 + 1000 k with k >= 700 (not used by any earlier set).
- Phase B (rgc): experiments/rg_comp_z.py = rg_comp_y.py with F7 candidates k = 600-679, F6 candidates k = 700-779 (80 each); the first 4 accepted of each are the smoke rows (UNUSED), the next 60 are the main rows.  Seeds: F7 sampler 820000 + k, layout 821000 + k; F6 sampler 810000 + k, plate 811000 + k; branch seed_base 14000000 + 1000 k (as SR 87 / SR 90; the k's are new).  The accepted k's are listed in results/flawed/rg_comp_z/rg_comp_z_tasks.json and in the as-built appendix below.
- Disjointness by job (experiments/raasw5r_disjoint.py, np.allclose, CPU): every rg_comp_z state row vs rg_comp, rg_comp_x, rg_comp_y (same tid); n2_states_wz vs every other n2_states_* set.  Result in the appendix; the run is refused if any row is equal.
- Smoke rows (first 4 accepted of F7 and F6) are not used anywhere.  The pcheck (policy-free perception check) is not rerun on the fresh rgc states (as SR 87 / SR 90, stated).

## Arms (all re-rolled on the new states; nothing from SR 70-90 rollouts is reused as an SR 91 comparator)
Phase A: W5 (raa_w.py --where_own 5), RAA-sL (--where_own 0), rgp (ReGuide-perc, rg5r_roll.sh copy of SR 87's rg5_roll.sh); 12 tasks (N1-N6 = tids 16-26, K1-K6 = tids 0-10) x 60 states.
Phase B (wa_sw5z.py on the rgc cells): W5 full (tids 0-10), W5 noR (tids 0-3, 7, 8, 10), W5 noM (tids 0-3, 8), RAA-sL full (tids 0-10); frozen I (tids 0-3, 8: the F7 pool and F6_b, where it is used) and rgp (tids 0-3, 8) through rg_comp_z.py.  (SR 87 also rolled I on all tids and rgp on 7, 10; not needed by any registered or secondary contrast.)
W4, W5c1, W5c2, W6 are not rerun (not part of this question).

## Registered contrasts (per set, SR 91 only; paired bootstrap B = 2000, rng 0, tasks and states resampled by raa_read.diff; Phase B by wa_read.bootm; one seed, one trial per state, no multiplicity correction)
Phase A (60 fresh states per task):
- **A2a PRIMARY-R** W5 - rgp on N1-N6: NON-INFERIOR iff lower > -.10 (COST iff upper < -.10, else NOT SHOWN).  This is the paper's repair claim.  **A2b** W5 beats rgp: SUPPORTED iff lower > 0.
- **A1** W5 - RAA-sL on N1-N6: SUPPORTED iff lower > 0, REVERSED iff upper < 0, else NOT SUPPORTED.
- **AK** W5 - RAA-sL on K1-K6: NON-INFERIOR iff lower > -.05, COST iff upper < 0, else NOT SHOWN.
Phase B (wa_read.read on b_w5; 60 states per cell):
- D1' full - noR on the F7 pool (tids 0-3): SUPPORTED iff lower > 0.  D2' full - noM on F6_b (tid 8): SUPPORTED iff lower > 0.  D3a noR - full on F6_b and D3b noM - full on the F7 pool: EQUIVALENT iff the CI lies inside +-.10.  DK full - noR on K1t + K2t (tids 10, 7): NON-INFERIOR iff lower > -.05.
- Gates on the full W5 arm: K floor (K1t + K2t) >= .70; unique-name controls (tids 4-6 mean) >= .50; F6_a (tid 9) >= .50.  Gated PRIMARY: DISSOCIATION SHOWN iff D1', D2' SUPPORTED, D3a, D3b EQUIVALENT and all three gates TESTABLE; UNTESTABLE if those four hold and a gate fails; else wa_read's PARTIAL / NOT SHOWN.
- **B1** full W5 - full RAA-sL on F6_b: SUPPORTED iff lower > 0.  **B2** full W5 - frozen I on F6_b: SUPPORTED iff lower > 0, REVERSED iff upper < 0.
Descriptive (no label weight): per-cell tables for W5, RAA-sL, rgp, I, noR, noM; K5 alone; W5 - RAA-sL on F7 pool, controls, F6_a, K1t + K2t; W5 - rgp F6_b / F7; placed given right grasp; bottle / cheese pairings.

## SECONDARY: pooled analysis over the three fresh sets W5 was evaluated on without design changes after it was frozen
Sets: SR 87 (second fresh set; where W5 was first confirmed: dissociation SHOWN, A2a NOT SHOWN), SR 90 (third set; the W5 arm re-rolled as a comparator), SR 91 (this run).  **SR 87 and SR 90 data were SEEN before this registration** (their per-set results and, for SR 90, the Step 0 numbers above); only SR 91 is new.  Because of that the pooled analysis is SECONDARY and cannot upgrade a failed primary.
Estimator: for each set the paired difference of the two arms' mean success over its tasks and 60 states; per bootstrap draw (B = 2000, one default_rng(0)) EACH SET independently resamples its tasks with replacement, then its states with replacement (the raa_read.diff scheme applied within set); the pooled statistic is the mean over the sets of the per-set differences (equal weights, 60 states per cell per set).  Interval = 2.5 / 97.5 percentiles.  Per-set deltas reported beside it.
Pooled contrasts and labels (same rules as above): P-A1 W5 - RAA-sL N1-N6; P-A2a W5 - rgp N1-N6 (NI -.10); P-A2b (beats); P-AK W5 - RAA-sL K1-K6 (NI -.05); P-B1 W5 - RAA-sL F6_b; P-B2 W5 - frozen I F6_b (states only: one task).  Descriptive pooled: W5 - rgp K1-K6, K5 alone W5 - RAA-sL.
The pooled module-removal dissociation is not computed over three sets (SR 90 has no W5 noR / noM arms); D1' / D2' are SR 87 and SR 91 only and are not pooled.
Code: experiments/raasw5r_read.py `pooled` (dry-run on SR 87 + SR 90 before registration, job 4817136, exploratory).

## What counts as what (stated now)
Replication of the repair claim = A2a NON-INFERIOR on the SR 91 states.  Supporting: P-A2a NON-INFERIOR.  Not a replication: A2a NOT SHOWN or COST here.  K1-K6 (AK) and A1 are reported with their labels; the pre-registered expectation (sr91_pred.txt) is that AK stays NOT SHOWN and A1 NOT SUPPORTED.

## Limits stated in advance
One seed, 60 states per cell, one checkpoint; W5's class-median constants come from the training demos and unknown target phrases get mu = 0; the pooled analysis includes two seen sets and weights the sets equally; the SR 87 and SR 90 W5 episodes were run in different job batches (same code, same checkpoint), SR 91's too; the fresh rgc states have no pcheck; the first accepted 4 states per rgc build are unused smoke rows; nothing from SR 91 changes W5.

## Run plan
Builds: fc_n2_wz.py (job 4817133), rg_comp_z.py f7 / f6 (4817134 / 4817135), then `assemble`; disjointness job; plan + predictions + code committed; prereg row (experiments/prereg/raasw5r.py, ledger m3_prereg raasw5r); then Phase A (12 jobs), Phase B (22 jobs), read, tables, sr91.md.
Job scripts in /coc/flash10/ksingara3/.claude_tmp/pp/ (raaw5r_roll.sh, rg5r_roll.sh, wasw5r_roll.sh, rgcz_roll.sh, w5r_states.sh, raasw5r_prereg.sh), all sbatch on overcap with explicit -p / -A and the standard --exclude list.

## Appendix: fresh states as built (before the prereg row)
- n2 familiar states **k = 700-759** (job 4817133; reproduces stored k 0, 23, 540, 579, 580, 639, 640, 699 with max |diff| 0; 60 new states distinct from each other and from 486 stored rows); results/flawed/n2_states_wz, n2_branches_wz.json (trial t = k - 700, seed_base 13000000 + 1000 k).
- F7 cells (job 4817134, assemble 4817149): candidates k 600-679, **64 accepted = k 600-663 (0 rejected)**; k 600-603 smoke, UNUSED; **main k 604-663 (60)**.  F6 cells (job 4817135): candidates k 700-779, **64 accepted = k 700-763 (0 rejected)**; k 700-703 smoke, unused; **main k 704-763 (60)**.  results/flawed/rg_comp_z.
- Disjointness (job 4817150, experiments/raasw5r_disjoint.py, np.allclose on every state row): rg_comp_z rgc_0 ... rgc_10 vs rg_comp, rg_comp_x, rg_comp_y: 0 equal rows in each of the 33 comparisons; n2_states_wz vs fresh, fresh2, fresh3, fresh4, gx, record, rx, s1x, wx, wy: 0 equal; TOTAL EQUAL ROWS 0.  By index range: N k 700-759 > every earlier n2 set's maximum (wy 699); F7 600-663 > rg_comp_y's 520-599; F6 700-763 > rg_comp_y's 620-699.  New seed bases: branch 13000000 + 1000 k (k >= 700), 14000000 + 1000 k (k >= 600 F7 / >= 700 F6), sampler seeds 600000 + k, 820000 + k, 821000 + k, 810000 + k, 811000 + k with these new k.
- Code diffs against the SR 90 / SR 87 originals (diff checked): fc_n2_wz.py vs fc_n2_wy.py: K0 700, output names, refusal list + wy; rg_comp_z.py vs rg_comp_y.py: docstring, OUT / META names, F7_K0 / F6_K0 = 600 / 700 (80 candidates each), the `_setup` assertion paths; wa_sw5z.py vs wa_sw5.py: the two `rg_comp_x` references (import, tasks json path).  Job scripts are sed copies of SR 87's (raaw5_roll.sh, rg5_roll.sh, wasw5_roll.sh, rgcx_roll.sh) with the states paths, the ledger tag and wa_sw5z.py / rg_comp_z.py substituted.

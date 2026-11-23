# Oracle arm (Exp 1) and pi0 replication (Exp 2): results

Executed 2026-11-22 from docs/notes/plan_oracle_pi0.md (commit 940c8c68).  Pre-registration rows (ledger/m3_prereg): `pz` (b715f7dc), `orc` (1c6977ab), `orc_arm` (d7404080).
Predictions: experiments/prereg/pz_pred.txt, orc_pred.txt.  One seed per arm, development states only (as every RAA-F route).

## 1. Jobs
| step | job id(s) | note |
|---|---|---|
| cfgcheck (CPU) | 4831812 | all 33 names construct (after the add-only l90c_config edit, made while raaslS2 links 4831420 / 4831426 were running) |
| pz_check 1-3 (GPU) | 4831831 | Pi0ZS(k=0) == openpi Pi0 exactly; servo identities; tree = pi0_libero's (50 tensors); state_proj bit-equal; quantile norm, no DeltaActions |
| pz smokes (20 steps, tail-lab) | 4831829 (sL), 4831830 (A0) | step-0 loss .1718 (sL), .1781 (A0): below the k3 line .556; 0 truncation warnings |
| smoke export + pz_check item 4 | 4831993, 4832011 | torch servo = reference for role 2 (on), role 1 and zeros (off) |
| pz serving smoke | 4832012 | server up, finite chunks, role-1 tokens in the log |
| pz prereg | 4832088 | row `pz` |
| pz training chains (tail-lab, exp tl) | sL 4832104-4832109; A0 4832110-4832115 | ~4.4 s/step; step 6000 reached |
| exports | 4835407 (sL), 4835948 (A0) | pytorch/pi0_g8v_raas_l_ae_tl6000, pi0_g8v_raac_l_ae_tl6000 |
| captures | sL 4835408 (K) / 4835409 (N); A0 4835949 / 4835951 | results/flawed/pz/a2/cap_pzsL, cap_pzA0 |
| probes (rows full) | sL 4835488 / 4835489; A0 4835950 / 4835952 | probe_pzsL_{N,K}, probe_pzA0_{N,K} |
| reads, contrasts, stateproj_diag_pz | 4835584, 4836392; 4836394-4836396; 4836412 | |
| orc_gen test / pairs | 4831816 / 4831817 | |
| render test set | 4832412 | replay check: stored states restore the end effector to 0.000 mm for 48 of 48 demos |
| orc_obs, selftest, prereg `orc` | 4832649, 4832650, 4832655 | 131 observations, 23 layouts; selftest 1 / 0 / .5 |
| orc_probe (A+0, A0, A+, RAA-sL, A) | 4832677 | first submission 4832660 cancelled (identity tolerance, deviation 4) |
| orc_read (5 models) and 3 contrasts | 4832778-4832782, 4832790-4832792 | results/flawed/orc/orc_read_*.json, orc_contrast_*.json |
| ORC data: render pairs, link, convert, check | 4835855, links by shell, 4835932, 4836229 | lol/libero_g8v_raacpo: 400 base + 139 planner episodes (67 bowl + 72 cheese), 7,659 held frames (A+0: 5,306); CHECK PASSED |
| ORC smoke, prereg `orc_arm`, chain | 4836238, 4836307, 4836312-4836317 | step-0 loss .0781 (<= .111) |

Later jobs: pzdev prereg 4839556, 14 dev rollout jobs 4839560-4839573, pzdev read 4840607; ORC export 4840606, orc_probe 4840671, orc_read 4840700, contrasts 4840706 / 4840707, ORC captures 4840710 / 4840712, probes 4840711 / 4840713, read 4840939.

## 2. Deviations from the plan
1. **Test set smaller than target.**  The registered cap of 250 attempts per object stopped generation at 16 complete pairs (bowl) and 8 (cream cheese) instead of 20 + 20.  Selection gave 131 observations on 23 layouts (>= 120 passes k1; target was 240 / 40 clusters).  Rules not changed.
2. **Pair yield for the ORC training data far below plan (k3).**  New spot >= 15 cm from the first: 17 of 50 (bowl) and 22 of 50 (cream cheese); 31 / 50 bowl demos had `no_spot` (free area too small), the rest failed classes.  The 15 cm rule was not relaxed.  Plan risk 6 says "below 50 pairs per object, ORC is not trained"; k3 says "built from the pairs that exist".  The coordinator told me to launch ORC from the pairs that exist.  ORC therefore adds **39 paired demos to A+0's 100 (not 100)**: ORC - A+0 isolates "39 paired-spot demos (+44 % held frames)", not the plan's "every layout paired".
3. **pz serving smoke showed role-1 tokens only.**  A 20-step model never grasps, so no role-2 token reaches the server; the role-2 path was checked directly in pz_check item 4 (policy input transform + PI0PytorchZS on the exported smoke checkpoint) and the serving code is the unchanged raa.py path of A0.  No identity probe (`--limit 2 --ident`) was possible without captured held queries; replay-vs-server of the real probes is .002-.003.
4. **orc_probe identity tolerance** loosened from 1e-3 to 2e-2 (batch-4 vs batch-1 bf16 gives max |diff| .004-.009 on all models; SR 83 / raaf validations report .003-.02).  orc_probe.py was edited after prereg `orc` hashed it (the file hash in the row is the pre-edit version); no registered quantity changes.
5. Captures / probes for pz use `rows full`; no budget cut was applied.  Observation: pzsL and pzA0 K success are within the K gate; N success is reported, not gated.
7. ORC's conditional N + K probe used `--rows inputs`; the pzdev rollouts (coordinator request, own prereg row) write to results/flawed/raaf/dev_pzsL / dev_pzA0 (the dev reader's fixed layout) with the RAA-F job scripts copied as pzdev_roll.sh / pzdev_wa.sh.  Total GPU-h overran the approved budget (section 8).
6. Plan section 3.4 item 6 `a2fx_capture.sh` default OUTD is results/flawed/pz/a2/cap_<label>; reads use a2f_read.sh and `raaf_probe_read.py --route pzsL|pzA0` (label only).

## 3. Exp 1, Stage 1a: follow share phi on held-out paired table-spot frames (131 observations, 23 layouts; clustered bootstrap B = 2000, rng 0)
| model | phi (95 % CI) | label | c_clean | phi_zero | sigma_demo |
|---|---|---|---|---|---|
| A+0 (servo off, planner dose) | **.771 [.605, .887]** | PARTIAL | .676 | .166 | 1.61 |
| A (small relabel, servo off) | .662 [.542, .783] | PARTIAL | .447 | .043 | 1.33 |
| A0 (no dose, servo off) | -.001 [-.002, .000] | NOT LEARNED | -.329 | .001 | -.002 |
| A+ phi_net (servo kept) | -.068 [-.210, .042] | NOT LEARNED | .821 | -.024 | -.168 |
| RAA-sL phi_net (servo kept) | -.006 [-.008, -.005] | NOT LEARNED | .892 | .001 | -.014 |
phi_total (token and servo from the counterfactual): A+ .830 [.737, .911], RAA-sL .957 [.906, 1.006] (the servo carries the where).
Contrasts (cluster-paired): phi(A+0) - phi(A0) = +.772 [+.606, +.887]; c_clean(A+0) - c_clean(A0) = +1.006 [+.819, +1.189]; phi_net(A+) - phi(A+0) = -.839 [-1.029, -.665] (servo effect on oracle frames); phi(A+0) - phi(A) = +.109 [-.018, +.219].
**Gate:** A+0 is not LEARNED (CI lower .605 < .70), so the registered gate says train ORC.

Labels for the registered predictions:
- (o1) A0 phi CI upper < .30: **TRUE** (0.000).  (o2) RAA-sL phi_net CI upper < .30: **TRUE**.  (o3) A+ phi_net CI upper < .30: **TRUE** (.042).
- (o4) A+0 phi point .50 (range .15-.85): **in range, point .771**; A+0 LEARNED (30 %): **FALSE**; A+0 NOT LEARNED (25 %): **FALSE** (PARTIAL).
- (o5) A+0 c_clean > A0 c_clean, CI lower > 0: **TRUE**.

## 4. Exp 2: pi0 (native trained state_proj), SR 83 design, pooled N + K, rows full
Fit gate (plan 3.7: K-pool capture success >= .70 and T CI lower > 0):
| arm | K success (tids 0, 2, 6, 10) | K pool | N success (16, 22, 24, 26) | N pool | T | gate |
|---|---|---|---|---|---|---|
| pi0-sL (servo kept) | .95 1.0 .95 .90 | .950 | .90 .55 .95 .35 | .6875 | 4.272 [3.939, 4.610] | **PASS** |
| pi0-A0 (servo off) | 1.0 1.0 .85 1.0 | .9625 | .80 .00 .65 .15 | .400 | 3.044 [2.668, 3.444] | **PASS** |
| reference RAA-sL (pi0.5) | 1.0 .95 .95 1.0 | .975 | .65 .85 1.0 .70 | .800 | | |
| reference A0 (pi0.5) | 1.0 .95 .75 1.0 | .925 | .60 .40 .90 .20 | .525 | | |
Both pi0 arms pass the K fit gate (the coordinator's requirement to apply it before reading in.tok: applied, K >= .70 in both).

| quantity | pi0-sL (S_kept) | pi0-A0 (S_off) |
|---|---|---|
| pooled in.tok share | **-.150 [-.174, -.127]** | **.542 [.478, .609]** |
| N / K in.tok | -.147 / -.153 | .755 / .371 |
| in.prefix pooled | .190 | .387 |
| in.servo (sL only) | .960 [.928, .994] | |
| abl.tok drop (K) | -.457 | +.271 |
Cross-backbone contrasts (in.tok): pzA0 - pzsL = +.683 [+.608, +.767]; pzA0 - A0(pi0.5) = +.531 [+.463, +.607]; pzsL - RAA-sL = -.143 [-.168, -.119].
state_proj change from pi0_libero's init (live columns 0-3, relative Frobenius): pzsL .0158, pzA0 .0173 (pad columns exactly 0).

Decision rules: **R1 (shortcut replicates): FALSE** (S_off CI upper .609 >= .30).  **R2 (no-dose read, S_off CI lower >= .30): TRUE** (.478); strong form (>= .70): FALSE.  **R3 (servo suppresses): TRUE** (S_off - S_kept = +.68 [+.61, +.77]).  **R4 (servo fails to suppress): FALSE** (S_kept CI lower is negative).
Labels: (z1) gates pass: sL TRUE, A0 TRUE.  (z2) S_kept point .00 (range -.05 to .20): **point -.150 is outside the range** (CI upper < .30: TRUE).  (z3) S_off point .08 (range -.05 to .45): **.542 outside**; CI upper < .30: **FALSE**; CI lower >= .30: **TRUE**; CI lower >= .70: FALSE.  (z4) R1: FALSE.  (z5) R3: TRUE.  (z6) R4: FALSE.  (z7) in.servo >= .70: TRUE (.960).  (z8) sL K within +-.10 of RAA-sL's: TRUE (.950 vs .975); A0 N below sL N: TRUE (.40 < .6875).  (z9) A0 relative state_proj change larger than sL: TRUE (.0173 vs .0158; descriptive, tiny difference).
Caveat (plan risk 2): the live columns of pi0_libero's state_proj were trained on eef xyz and axis-angle x; the negative pzsL share (-.150) is leftover dependence with the sign opposite to the counterfactual target, present only where the servo already supplies the correction.

## 5. pi0 full development-state rollouts (prereg `pzdev`, ledger 8fccf20a; experiments/prereg/pzdev_pred.txt; reader experiments/pzdev_read.py, results/flawed/pz/pzdev_read.json)
Same states and pipeline as the RAA-F dev read: Phase A K1-K6 and N1-N6, 60 states each (trials 4-23 + 24-63), Phase B F7 pool / F6_b / F6_a; paired clustered bootstrap B = 2000, rng 0.  Jobs 4839560-4839573 (4839564 was cancelled after its cells already existed; a requeue artefact).  Dirs results/flawed/raaf/dev_pzsL, dev_pzA0.
| arm | dev K | dev N | cheese | bottle | F7 pool | F6_b | F6_a |
|---|---|---|---|---|---|---|---|
| pi0-sL | .811 | .761 | .939 | .583 | .638 | .000 | .733 |
| pi0-A0 | .906 | .506 | .783 | .228 | .575 | .200 | .967 |
| RAA-sL (pi0.5) | .936 | .878 | .872 | .883 | .744 | .017 | .983 |
| A0 (pi0.5) | .831 | .686 | .856 | .517 | .700 | .383 | 1.000 |
Registered contrasts on N (60 states x 6 tasks):
- **(d1) pi0-A0 - pi0-sL = -.256 [-.386, -.128]** (CI upper < 0).  K difference +.094 [+.003, +.208]; **(d4) K non-inferiority at -.05: holds** (pi0-A0 is not worse on K).
- **(d2) pi0-A0 - A0 (pi0.5) = -.181 [-.344, -.008]**.  **(d3) pi0-sL - RAA-sL (pi0.5) = -.117 [-.306, +.067]** (CI straddles 0).  K: pi0-sL - RAA-sL = -.125 [-.258, -.017].  pi0-A0 - RAA-sL on N = -.372 [-.633, -.081].
- Phase B: F7 pool pi0-A0 - pi0-sL = -.063 [-.137, +.019]; F6_b +.200 [+.100, +.300]; F6_a +.233 [+.117, +.350].  pi0-A0 - A0: F7 -.125 [-.194, -.056], F6_b -.183 [-.317, -.033].
- Per task N (pi0-sL / pi0-A0): N1 .88/.77, N2 .93/.63, N3 1.0/.95, N4 .53/.08, N5 .87/.53, N6 .35/.07.  The pi0-A0 deficit sits in N2, N4, N5, N6.
**Rule "read but not in control" (registered: d1 CI upper < 0 AND S_off CI lower >= .30 AND d4 holds): TRUE.**  pi0-A0 reads the token (in.tok .542 [.478, .609]) and still sits 26 points below pi0-sL on N and 37 points below RAA-sL (whose servo supplies the target), with K intact.
Labels: (p1) d1 point -.20, CI upper < 0 (75 %): **point -.256, TRUE**; CI lower > 0: FALSE.  (p2) d2 point -.08, CI upper < 0 (30 %): **point -.181, TRUE** (the less likely branch).  (p3) d3 CI upper < 0 (55 %): **FALSE**.  (p4) d4 (80 %): **TRUE**.  (p5) pi0-sL dev N .70 (.55-.85): **.761 in range**; pi0-A0 dev N .45 (.25-.65): **.506 in range**.  (p6) pi0-sL dev K >= .886 (65 %): **FALSE (.811)**; pi0-A0 dev K >= .886 (60 %): **TRUE (.906)**.  (p7) F7 pool d CI upper < 0 (50 %): **FALSE (+.019)**.  (p8) rule TRUE (65 %): **TRUE**.
Caveat: one seed per arm; the capture-subset N for pi0-A0 (.40) and the dev N (.506) agree; the pi0-sL capture subset N (.6875) is a little below its dev N (.761).

## 6. Exp 1, Stage 1b: ORC (pi05_g8v_raac_lo_ae; prereg `orc_arm`) trained on A+0's data + 39 paired-spot demos
Trained 6,001 steps (chain 4836312-4836317; one link, 4836313, hung in a step-3000 checkpoint save for 20 min and was cancelled; the chain resumed from step 2750, no step lost).  Export 4840606.  Step-0 loss .0781.
| model | phi (95 % CI) | label | c_clean | phi_zero | N + K in.tok share | K / N capture success |
|---|---|---|---|---|---|---|
| ORC | **.777 [.628, .873]** | PARTIAL | .727 | .033 | **.195 [.124, .278]** (N .338, K .077; rows inputs; in.prefix .565) | .825 / .4375 |
| A+0 | .771 [.605, .887] | PARTIAL | .676 | .166 | .124 [.089, .164] (raaf_sel) | |
phi(ORC) - phi(A+0) = **+.005 [-.058, +.066]**; c_clean +.051 [-.022, +.128]; phi(ORC) - phi(A0) = +.777 [+.629, +.873].  Adding 39 paired-spot demos (+44 % held frames) did not move phi.  ORC dev rollouts were not run (registered only if the N + K share CI lower >= .70).
Labels: (o6) ORC phi point .75: **.777 in range**; ORC LEARNED (55 %): **FALSE** (CI lower .628 < .70); ORC NOT LEARNED (10 %): **FALSE**.  (o7) phi(ORC) - phi(A+0) CI lower > 0 (65 %): **FALSE**.  (o8) ORC N + K share point .20: **.195**; CI upper < .30 (55 %): **TRUE**; CI lower >= .70 (7 %): **FALSE**.  (o9) within-model dissociation (oracle model LEARNED and N + K CI upper < .30), A+0 or ORC (45 %): **FALSE** (both PARTIAL; the N + K half holds for both).
Note on the probe: the ORC N + K probe used `--rows inputs` (share unchanged, plan cut 1); the "route" label in the read file is `Aplus` (a label of the reader, not the model).  The reader's VERDICT line ("FAILED (share CI upper < .30)") is the pi0.5 selection rule and is not part of this plan.

## 7. Reading against "where must be owned outside the learned network"
Neither experiment refutes the claim and neither supports it cleanly; the plan's conditions come out as follows.  **Exp 1 refutation (ORC or A+0 LEARNED and N + K share >= .70 and dev N >= RAA-sL - .05): not met** (phi PARTIAL for both, .77 with CI lower .61-.63; N + K share .12-.20).  "ORC NOT LEARNED" did not occur either: the random-init state_proj is learned in part within the shared recipe (phi .77 where only the token informs), so the capacity / optimisation version of O4 fails, and the network follows the token on the frames that need it while ignoring it on N + K (.12-.20), the within-model pattern the plan calls the strongest outcome, short of the registered .70 bar for LEARNED.  **Exp 2 refutation (pi0-A0 S_off CI lower >= .70 and N >= RAA-sL - .05): not met**; the weaker condition **R2 is met**: pi0-A0 reads the token at .542 [.478, .609] with no dose and no servo, so the pi0.5 null (A0 -.001) is specific to a new random-init pathway, and "the network reads the where-input only when the data demand it" does not hold on pi0 as stated.  But the registered reading "read but not in control" holds: pi0-A0 reads the token and its dev N (.506) is below pi0-sL (.761; d1 -.256 [-.386, -.128]) and RAA-sL (.878) with K intact; the exact target, once read, is not turned into aim as well as the explicit servo provides.  Taken together: the inductive-bias artefact explains the pi0.5 null of the in.tok share (R2), but not the behavioural gap; the claim that the where must be owned outside the learned network survives in its control form (a servo or equivalent external owner buys N, reading does not) and is weakened in its reading form (a trained state pathway does read an exact where-token from ordinary data).  One seed per arm, development states only, test set of 131 observations on 23 layouts, and ORC with 39 rather than 100 paired demos limit all of this.

## 8. Budget
Approximate A40 hours: pi0 training 2 x ~8 = 16; ORC training ~10; pi0 captures + probes ~5.5; ORC captures + probes ~1.5; renders ~1.5; Stage 1a probes ~0.3; pi0 dev rollouts (14 jobs) ~12.  Total about 47 GPU-h against the approved ~30-33; the overrun is the ORC arm (gate fired, instructed to launch) and the pzdev request.

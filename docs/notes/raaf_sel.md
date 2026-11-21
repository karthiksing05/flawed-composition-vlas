# RAA-F selection (Routes A, B, A0), PROVISIONAL (A+ pending)

Executed 2026-11-21 per docs/notes/raaf_plan.md sections 6, 9 and the ORCHESTRATOR sections; commands from docs/notes/raaf_probe_runbook.md.
No label below was changed after the numbers were seen.  Not built, not run: the 5th fresh state set, the confirmatory read, A+ / A+ctrl.

## Outcome (section 6.3)
**No route passes the probe, so there is no winner (6.3 rule 5), PROVISIONAL (A+ pending).**  A: NOT PASSED (share .274, upper .314 is not below .30).
B: FAILED (selection PASS = `pass_alt_nocrop` = False; share CI upper .089 < .30).  A0: FAILED.  A is not closed by kill k5 (its CI upper is .314, just above .30).
Dev behaviour is reported below as observation only; no route beats RAA-sL on dev N.  A+ (step about 3500 of 6000) is added later and can change this.

## Checkpoints (tail-lab chain, `--exp tl`; the overcap copies were cancelled; both chains share one config, so the tl checkpoint is the registered one)
| route | config | checkpoint | pytorch export |
|---|---|---|---|
| A | pi05_g8v_raac_lf_ae | jax/.../tl/6000 | pytorch/pi05_g8v_raac_lf_ae_tl6000 |
| B | pi05_g8v_raac_lb_ae (servo OFF, see plan correction) | tl/6000 | pytorch/pi05_g8v_raac_lb_ae_tl6000 |
| A0 | pi05_g8v_raac_l_ae | tl/6000 | pytorch/pi05_g8v_raac_l_ae_tl6000 |
Every capture ran on an A40 (`--gres=gpu:a40:1` in the scripts).  A probe checkpoint tag `tl6000` was passed as `--ckpt` (a2_probe_f.py's registry hard-codes `main6000`; its `--ckpt` option is the documented override, no file edited).

## 6.1 Probe (pooled N + K, 96 observations per route, B = 2000, rng 0, clustered; files results/flawed/raaf/a2/raaf_probe_{A,B,A0}.json)
| | where-input | pooled share [95% CI] | N set | K set | in.prefix pooled (RAA-sL .089) | T [CI] | label |
|---|---|---|---|---|---|---|---|
| A | in.tok | .274 [.239, .314] | .426 [.352, .512] | .170 [.141, .203] | .751 (N .631, K .834) | 2.961 [2.615, 3.313] | NOT PASSED |
| B | in.tokcrop | .022 [-.046, .089] | .031 [-.177, .199] | .017 [-.012, .049] | literal 1.001; nocrop 1.026 (K .819) | 2.111 [1.737, 2.470] | FAILED (`pass_alt_nocrop` False; `PASS` literal False) |
| A0 | in.tok | -.001 [-.001, .000] | -.000 [-.002, .001] | -.001 [-.002, -.000] | 1.002 (N 1.000, K 1.003) | 2.627 [2.225, 3.055] | FAILED |
RAA-sL reference (same code, existing captures): in.tok -.007 [-.008, -.006]; in.prefix .089 (N + K pool); T 4.474.  Gate T lower > 0 holds for all three.
Reported always: A top1 .166 [.139, .191], PRn .196 [.159, .251], abl.tok drop N .267 [.156, .630], K .131 [.103, .162].  B top1 .190, PRn .156, abl.tok drop K -.002 (N not available).
A0 top1 .172, PRn .175, abl.tok drop N .002 [-.012, .018], K -.003 [-.005, -.002].  B extras: sum(in.tok + in.crop) .023, in.crop alone .021 [-.047, .088].
**Prefix share (the second half of the pass rule) is above .089 for every route.**  The where-share half fails for all three as well, so the prefix half decided nothing.

### Contrasts of the learned where-share (results/flawed/raaf/a2/contrast_{A,B}_A0.json; experiments/raaf_probe_contrast.py, new; same clustered bootstrap, B = 2000, rng 0; clusters = dev states, 95 clusters common to both routes; observations differ per policy because the 3rd held query is on the policy's own rollout, so pairing is by cluster)
- **A - A0: +.272 [.238, .312]** (A .272, A0 -.001).  The relabel data raised the token share clearly from zero, but to about .27, not .70.
- **B - A0: +.030 [-.034, .089]** (the crop adds nothing detectable).
- in.prefix: A - A0 -.250 [-.285, -.218] (A reads less through the prefix than A0, still .75); B - A0 -.001 [-.003, .001].

### Observed defect of the prefix half of the pass rule (descriptive; no label changed)
RAA-sL's where-effect runs mostly through the servo, which sits outside the network, so RAA-sL's network-internal in.prefix share is small (.089).  A servo-free model has no outside path, so its effect must run through network inputs, and its in.prefix cannot fall below RAA-sL's.  The servo-matched reference is A0: in.prefix **1.002** (N 1.000, K 1.003).  Against it A is .751 (lower by .250 [.218, .285]) and B is 1.001.  The prefix criterion as written therefore fails every servo-free route by construction.  Labels above use the rule as registered.

## Predictions vs outcomes
- raaf_pred (A): item 1 A passes 18%: NO.  Point share about .35 (range .05-.80): observed .274, in range.  K share above N share (65%): NO (K .170, N .426).  Item 6 A dev N below RAA-sL (75%, point .74): YES (.778).  A dev K >= .70 (85%): YES (.994).  Items 2, 4, 5 (A+, A0 in plan, no-route) see below.
- plan item 3 / raafb_pred (B): B passes 6%: NO.  B crop + token share point < .30 (80%): YES (.022).  Dev K >= RAA-sL - .05 (45%): NO (.850 vs .936, -.086).  Dev K >= .70 (80%): YES.  Dev N within +-.05 of RAA-sL (20%): NO; below by more than .05 (70%): YES (.644, -.233).  Step-0 loss <= 2x (85%): YES (build smoke).
- raaf0_pred (A0): in.tok passes by itself (4%): NO.  in.tok share point below .30 (93%): YES, point -.001 (predicted about .05, range -.02 to .35: inside).  Reading rule: A0 did not pass, so the servo removal alone did not move the share; the rise in A is attributable to the relabel data (A - A0 above).
- Plan item 5 "no route passes" (52%, A+ still pending): holds for A, B, A0.
- Principle (plan 0): "only if" direction: B (dose 0) and A0 (dose 0) fail as predicted, so it is not falsified.  "If" direction: A has a small dose (K-delta variance share 3.4% of held xyz labels) and moves the share from -.001 to .274; read as a dose result, not a falsification.  A+ (large dose) decides it.

## 6.2 Dev behaviour (seen sets, selection only; files results/flawed/raaf/raaf_dev_read.json, experiments/raaf_dev_read.py new reader importing raa_read and wa_read)
Phase A: SR 73's 60 states per task (dev trials 4-23 plus rx trials 24-63), K1-K6 / N1-N6, B = 2000, rng 0, tasks and states resampled (raa_read.diff).  RAA-sL comparator = existing SR 82 rollouts (results/flawed/raa raasl_*).
| arm | N1-N6 | K1-K6 | cheese N1-N3 | bottle N4-N6 | per-task K5 / N4 / N6 |
|---|---|---|---|---|---|
| RAA-sL | .878 | .936 | .872 | .883 | .683 / .883 / .767 |
| A | .778 | .994 | .844 | .711 | .983 / .717 / .433 |
| B | .644 | .850 | .767 | .522 | .167 / .467 / .167 |
| A0 | .686 | .831 | .856 | .517 | .267 / .417 / .200 |
| route - RAA-sL | dev N [CI] | dev K [CI] | K non-inf -.05 |
|---|---|---|---|
| A | -.100 [-.261, +.050] | +.058 [-.006, +.167] | NON-INFERIOR |
| B | -.233 [-.447, -.019] REVERSED | -.086 [-.272, +.022] | NOT SHOWN |
| A0 | -.192 [-.406, +.019] | -.106 [-.250, +.003] | NOT SHOWN |
| route - A0 (matched) | dev N | dev K | bottle |
|---|---|---|---|
| **A - A0** | +.092 [-.019, +.222] | +.164 [+.006, +.403] | +.194 [+.050, +.367] |
| **B - A0** | -.042 [-.183, +.094] | +.019 [-.056, +.092] | +.006 [-.117, +.144] |
Phase B (rg_comp rgc main states, wa_s.py for A and A0, wa_sb.py for B, full arm tids 0-10; comparator = SR 82 results/flawed/wasl; bootm, states resampled; noR / noM not rolled on dev, as registered, so wa_read's dissociation labels are NOT SHOWN with missing arms and are not used):
| arm | F7 pool | F6_b | F6_a | K floor (K1t + K2t) | unique controls (tids 4-6) |
|---|---|---|---|---|---|
| RAA-sL | .744 | .017 | .983 | .962 | .600 |
| A | .706 | .433 [.317, .567] | .933 | .988 | .633 |
| B | .694 | .600 [.467, .717] | .950 | .963 | .583 |
| A0 | .700 | .383 [.250, .517] | 1.000 | .975 | .625 |
| route - RAA-sL | F7 pool | F6_b |
|---|---|---|
| A | -.037 [-.081, .000] | +.417 [.300, .550] SUPPORTED |
| B | -.050 [-.106, .000] | +.583 [.450, .700] SUPPORTED |
| A0 | -.044 [-.100, .013] | +.367 [.233, .500] SUPPORTED |
| A - A0 | +.006 [-.044, .056] | +.050 [-.100, .217] NOT SUPPORTED |
| B - A0 | -.006 [-.069, .050] | +.217 [.050, .383] SUPPORTED |
Reading (observational): the F6_b rise over RAA-sL (about .4) is already present in A0, so it comes from removing the servo (or the base data), not from the relabel; B adds about .22 on F6_b over A0 (crop effect on the memory cell, with no learned where-share: the crop is read through the prefix).  A's relabel adds +.16 on K and +.19 on bottle pairings over A0.  No servo-free route recovers RAA-sL's release aim on N (A .778, B .644, A0 .686 vs .878), as the plan predicted.

## 9 Kill checks
- A k3 (clip drops > 5%): 0.7% at build, clear.  A k5 (share CI upper < .30): upper .314, NOT triggered; A stays open (A0 was already trained unconditionally).
- B k3 (dev K below RAA-sL - .10): .850 vs .936 = -.086, NOT triggered.  B k4 (share CI upper < .30): upper .089, TRIGGERED, B closed for the probe.
- A0: FAILED (CI upper < .30), as predicted; used as the servo-matched comparator.

## 6.3 winner rule applied
Eligible = PASS and dev K >= .70 and K diff >= -.05.  No route has PASS, so none is eligible.  Rule 5: no winner, no confirmatory read, 5th set stays unbuilt; the reportable line so far is "the shortcut survives the data at these doses: shares .27 (A), .02 (B), -.00 (A0), dev N below RAA-sL for all".  A route beating RAA-sL on dev N without passing would be a behavioural observation only; none does.  PROVISIONAL until A+.

## Deviations and notes
1. Checkpoints are the tail-lab chain (tl6000), as instructed; labelled `tl6000` instead of the runbook's `main6000` (probe `--ckpt`).  No pytorch dir was overwritten (checked absent first).
2. B's selection PASS is `pass_alt_nocrop` (ORCHESTRATOR clarification); the literal `PASS` is also False.  B is servo-off (plan correction), so B is compared with A0 as well; the registered comparator (RAA-sL) is used in the winner rule.
3. New files: experiments/raaf_dev_read.py (needed: raa_read.grid reads rx files only from results/flawed/raa), experiments/raaf_probe_contrast.py.  Job scripts (copies, in .claude_tmp/pp): raaf_roll.sh (of raa_roll.sh, RAAPY option), raaf_wa.sh (of wasl_roll.sh, WAPY option, raaf prereg check), raaf_devread.sh, raaf_contrast.sh.  No existing script or reader edited.
4. Smokes outside results/ before each route's main rollouts: A raafA_smoke 4823936 (roles 1 and 2 in the server log) and wa 4823973 (roles 1, 2); B raafB_smoke 4824430 (roles present) and wa_sb 4824431 (crop centre non-None on all 120 logged queries, e.g. [138.2, 104.2]; raa_b.py logs no crop centre, crop presence for Phase A is evidenced by the capture npz via the probe's crop assertion and by wa_sb); A0 raaf0_smoke 4824890, wsmoke 4824891 (roles present).
5. Failed attempts, no results affected: first dev read 4825497 (comma in `--routes` split by sbatch --export), 4825515 cancelled (duplicate `read` argument); contrast jobs 4825248/49 and 4825295/96 (observation pairing by position failed because each policy's 3rd held query differs; replaced by cluster pairing, 4825308/09).  The same session also had one denied command containing a stray `python3 -` line (login-node python; refused by the permission layer, executed nothing).
6. Remaining tail-lab chain links: raafb_tl6 (4819414) cancelled after B reached 6000; raaf0 tl links ended on their own; raaf0_tl6 (4820177) ran briefly as a restore at step 6000.  Raw captures (cap_*, 570 MB) and probe npz are not committed (paths above).

## Job ids
A: export 4823926; capture K 4823934, N 4823935; probe K 4823962, N 4823963; read 4824204; Phase A 4823969-72 (main m1/m2, rx x1/x2); Phase B 4824032-34; dev read (A only) 4824516.
B: export 4824395; capture 4824428-29; probe 4824432-33; read 4824567; Phase A 4824481-84; Phase B 4824485-87.
A0: export 4824862; capture 4824885-86; probe 4824887-88; read 4824889; Phase A 4824953-56; Phase B 4824957-59.
Contrasts 4825308 (A - A0), 4825309 (B - A0).  Final dev read 4825516.

---

# A+ (appended 2026-11-21; Route A+ = pi05_g8v_raas_lp_ae, servo kept, planner table-spot demos)

Checkpoint: the overcap `main` run, jax/pi05_g8v_raas_lp_ae/main/6000, exported to pytorch/pi05_g8v_raas_lp_ae_main6000 (job 4827061; no pre-existing dir).  Registered config, prereg tag raafp (experiments/prereg/raafp_pred.txt; deviations: spot rule h_j, bottle dropped, 2 objects / 100 demos; nothing in the row changes the probe or read rules).  Captures on A40, same sets, B = 2000, rng 0.  A+ctrl is not run (still training).

## Outcome: **no route passes the probe; A+ is NOT eligible; PROVISIONAL only for A+ctrl, which is not a route.**  No winner, no 5th set, no confirmatory read.
| | where-input | pooled share [CI] | N | K | in.prefix pooled (RAA-sL .089, matched: A+ keeps the servo) | T [CI] | label |
|---|---|---|---|---|---|---|---|
| A+ | in.tok | **-.007 [-.013, -.000]** | -.005 [-.016, .007] | -.009 [-.013, -.005] | .143 (N .129 [.069, .191], K .159 [.098, .214]) | 4.742 [4.372, 5.123] | FAILED (CI upper < .30) |
Gate T lower > 0 holds.  top1 .092 [.070, .135], PRn .313 [.258, .352]; abl.tok drop N -.030 [-.043, -.015], K -.036 [-.043, -.030] (ablating the token changes nothing or slightly helps).  A+ is the RAA-sL pattern again: the token is not read (RAA-sL -.007), and the prefix half (.143 vs .089) is also not below, though here it is a matched comparison.  Kill k4 (CI upper < .30) is triggered.
Contrasts (results/flawed/raaf/a2/contrast_Aplus_A.json, contrast_Aplus_A0.json; clustered bootstrap, 94 common clusters): **A+ - A = -.282 [-.322, -.247]**; **A+ - A0 = -.006 [-.012, +.001]**; in.prefix A+ - A -.607 [-.657, -.553], A+ - A0 -.858 [-.904, -.815].  The large planner dose (19-38 cm shifts on 100 table-spot demos, 1/12 of frames) did not raise the token share at all; the small-dose servo-free relabel (A) did (+.27).  Possible reasons, not tested: the kept servo supplies the where-effect on the table frames (the label a - K rel depends on the spot through rel, which the servo computes outside the network), and the table-spot frames use a new prompt phrase.  The planned "if" direction of the principle (large dose => learned use) is therefore NOT supported at this dose with the servo present; plan 9 plan-level rule: A and A+ both fail, but A's dose is below the S0 thresholds' meaning of "large" and A+ keeps the servo, so the plan-level falsification is not claimed here.

## Dev behaviour (results/flawed/raaf/raaf_dev_read_v2.json; same readers and states as before; smoke 4827139 / 4827140 showed roles 1 and 2 before the main rollouts)
| arm | N1-N6 | K1-K6 | cheese | bottle | F7 pool | F6_b | F6_a | K floor | unique |
|---|---|---|---|---|---|---|---|---|---|
| A+ | .828 | .947 | .861 | .794 | .688 [.638, .731] | .033 [.000, .083] | .417 | 1.000 | .567 |
| RAA-sL | .878 | .936 | .872 | .883 | .744 | .017 | .983 | .962 | .600 |
A+ - RAA-sL: N -.050 [-.192, +.061] (NOT SUPPORTED); K +.011 [-.028, +.067] (NON-INFERIOR at -.05); cheese -.011 [-.150, .106]; bottle -.089 [-.300, .078]; F7 pool -.056 [-.106, -.013] (REVERSED); F6_b +.017 [-.033, .083] (NOT SUPPORTED).
A+ - A: N +.050 [-.097, +.211]; K -.047 [-.136, +.003]; cheese +.017; bottle +.083 [-.167, .350]; F7 pool -.019 [-.069, .031]; **F6_b -.400 [-.533, -.267] (REVERSED)**.  A+ - A0: N +.142 [-.017, .350]; K +.117 [-.006, .283]; bottle +.278 [.039, .567]; F7 -.012; F6_b -.350 [-.483, -.200] (REVERSED).
Reading: A+ behaves like RAA-sL (servo kept): N and K near it, F6_b near zero (the F6 memory cell is only rescued by the servo-free models A, B, A0).  F6_a (the F6 gate, .417 < .50) is poor for A+, new relative to RAA-sL (.983); the table-spot data may have shifted F6_a behaviour.  Dissociation labels are not used (noR / noM not rolled on dev).

## 6.3 with A, B, A0, A+
No route has PASS (A .274 NOT PASSED, B / A0 / A+ FAILED), so none is eligible even though A+ meets the two behaviour conditions (dev K .947 >= .70; K - RAA-sL +.011 >= -.05).  Rule 5: no winner; 5th fresh set unbuilt; no confirmatory read.  Reportable line: "the shortcut survives the data at these doses": shares A .274, A+ -.007, B .022, A0 -.001; RAA-sL -.007.

## Labels vs raafp predictions
- Item 2 A+ passes (28%): NO.  Point share about .50 (range .10-.90): observed -.007, far below the range.
- Item 5 no route passes (52%): YES (all four).
- Item 7 A+ dev N within +-.05 of RAA-sL (55%): at the edge (point -.050, CI [-.19, .06]; counts as within by CI, outside by a rounding of the point -.0500000000000000044); above by > .05 (15%): NO.  A+ dev K >= .886 (70%): YES (.947).
- Counter-evidence rule (probe passes but dev N > .10 below): not applicable, probe failed.
- Item 9 and "A+ fails while A+ctrl - RAA-sL on dev N >= 0": pending A+ctrl.
- Kill k4: triggered.

## Deviations / notes (A+)
Export tag main6000 (overcap main run), probe via `--model Aplus --config pi05_g8v_raas_lp_ae --ckpt <dir>`.  raaf_dev_read.py edited (untracked-then-committed new file, my own): added the A+ - A and A+ - A0 pairs; the earlier output files are unchanged and the new read is a new file (raaf_dev_read_v2.json).  One of my commands contained a stray `python3 -c 1` login-node line; the permission layer denied it and nothing ran.
Jobs: export 4827061; capture 4827132-33; probes 4827134-35; read 4827136; contrasts 4827137-38; smokes 4827139-40; Phase A 4827161-64; Phase B 4827165-67; dev read v2 4827520.

---

# A+ctrl (appended 2026-11-21; pi05_g8v_rview2_lp_ae: rview2 format, no token, no servo; same frames / prompts / views as A+)

Not a route and not probed (no where-input).  Checkpoint jax/pi05_g8v_rview2_lp_ae/main/6000, exported to pytorch/pi05_g8v_rview2_lp_ae_main6000 (job 4828331; no pre-existing dir).  Dev read: results/flawed/raaf/raaf_dev_read_v3.json (new file; v1 / v2 untouched; job 4828975); reader extended with the pairs A+ctrl - RAA-sL, A+ctrl - A+ and an F6_a contrast.  The plan names no matched rview2 arm (an R2-from-pi05_libero model is not trained), so A+ctrl is compared descriptively with RAA-L (.575 N / .831 K in raasl_read.json) and R2 (.389 N / .753 K, a different base model).

## Serving paths (deviation to note)
- Phase A: plan path, sc_rv.py serve --arm rview2 with `--config pi05_g8v_rview2_lp_ae` (the R2 path: exact masks, client --seg), through a copy of raa_ref_roll.sh (.claude_tmp/pp/raaf_ref.sh, adds --config), same 60 states / seeds as A+ (main n2_states + rx).  Smoke 4828344 ran before (role prompts, config confirmed in the server log).
- Phase B: the plan names no path.  I used a NEW copy of wa_s.py, experiments/wa_s_ctrl.py (the only change: the observation state is replaced by 8 zeros, since the rview2 norm stats are 8-dim and the model reads no state), full arm tids 0-10 with learned masks, the same pipeline as A+.  Smoke 4828345 (state all zeros, role prompts).  Caveat: Phase A uses exact masks, Phase B learned masks, so the two phases are not the same perception.
Jobs: Phase A 4828365-68, Phase B 4828369-71, dev read 4828975.

## Dev results (A+ctrl vs RAA-sL [N .878, K .936, bottle .883, cheese .872, F7 .744, F6_b .017, F6_a .983] and vs A+ [N .828, K .947, bottle .794, cheese .861, F7 .688, F6_b .033, F6_a .417])
| A+ctrl | value | - RAA-sL [CI] | - A+ [CI] |
|---|---|---|---|
| N1-N6 | .631 | -.247 [-.483, -.030] REVERSED | -.197 [-.431, +.017] |
| K1-K6 | .814 | -.122 [-.328, +.006] NOT SHOWN | -.133 [-.361, +.006] |
| cheese | .689 | -.183 [-.617, .111] | -.172 [-.600, .122] |
| bottle | .572 | -.311 [-.528, -.033] | -.222 [-.483, -.011] |
| F7 pool | .413 [.356, .469] | -.331 [-.388, -.269] REVERSED | -.275 [-.338, -.213] REVERSED |
| F6_b | .000 | -.017 [-.050, .000] | -.033 [-.083, .000] |
| F6_a | **.000** | -.983 [-1.0, -.950] REVERSED | -.417 [-.533, -.300] REVERSED |
Other cells: K1t .933, K2t .975 (K floor .962), unique-name controls .183.  Per task K5 .083, N2 .35, N4 .45, N6 .30.

## Pending raafp items
- Item 9, A+ctrl dev N above R2-type levels (> .70) (40%): **NO** (.631; above R2 .389 and about RAA-L .575).
- "A+ fails while A+ctrl - RAA-sL on dev N >= 0 (data help without the channel)" (20%): **NO** (A+ failed the probe; A+ctrl - RAA-sL is -.247 [-.483, -.030], so the planner data alone do not help; A+ - A+ctrl N is +.197 [-.017, .431], in favour of keeping the servo and channel, though the CI includes 0).

## F6_a: is A+ctrl's low like A+'s .417?  Yes, lower (.000), and the numbers point to the table-spot data, not the servo
F6_a (tid 9, 60 states): RAA-sL .983 (servo, token), A .933 (no servo), A0 1.000 (no servo, base data), B .950 (no servo), A+ .417 (servo kept, planner data), A+ctrl .000 (no servo, planner data).  The servo-free base-data models (A0, A, B) are high, and the servo-kept base-data model (RAA-sL) is high, so neither presence nor absence of the servo explains the drop; the two models trained on the planner table-spot data are the only low ones, and the one without a servo / token is lower still.  Most likely cause: the table-spot demos (a new "target: the table" phrase, 5,306 held frames, 100 demos) degrade F6_a, the cell where the plate is the target and the bowl is the object; the servo / token partly protect it (A+ .417 vs A+ctrl .000).  Caveats: one seed per model, A+ctrl has no matched base-data twin from the same init (the A0 is the servo-free token twin but is not an rview2 model), and the Phase B path for A+ctrl is a new copy (wa_s_ctrl.py).  K1t / K2t are .93 / .98 so the pipeline itself works; unique-name controls (.18) and F7 (.41) are also low, so A+ctrl's Phase B weakness is not confined to F6_a.

Outcome of selection is unchanged: no route passes the probe, no winner, no 5th set, no confirmatory read.

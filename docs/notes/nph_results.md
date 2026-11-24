# NPH results (plan: docs/notes/plan_necessity_power.md, tag `nph`)

## RESUME NOTE (written 2026-11-23 21:20; the run is unattended from here)

**State.**  Head fit and gates passed; checks and smoke passed; registration row committed (ledger/m3_prereg/m3_prereg__libero_goal__nph__s0__66315325.json, commit 7865e062, 2026-11-23 17:47:54);
6th set built (results/flawed/n2_states_w6, n2_branches_w6.json, k 820-879; disjointness job 4847576: 0 equal rows, geometry results/flawed/rva1/geom_w6.json).  pi0-H* training is running.

**Job IDs and dependency order** (all gates are inside the job scripts: a gated job exits 0 without work; scripts in docs/notes/nph_jobs/, run by `.claude_tmp/pp/nph_pipeline.sh`, ids in `.claude_tmp/pp/nph_pipeline_ids.txt`):
1. Training chain nph_tl1..tl6 = 4847565-4847570 (tail-lab, afterany chain, 6,001 steps at ~4.4 s/step, ~7.3 h; was at step ~2900 at 21:17).
2. 4848465 export (afterany 4847570; runs only if jax/pi0_g8v_raah_l_ae/tl/6000 exists) -> robot/checkpoints_l90c/pytorch/pi0_g8v_raah_l_ae_tl6000.
3. 4848466 pcheck (tail-lab GPU; raah0_check.py --torch-ckpt: PI0PytorchZH = PI0Pytorch + head; PolicyBackboneH0 constructs).
4. 4848467, 4848468 pi0-H* K jobs (K1-K3, K4-K6), cap guard 21.5 wall-h of nph_* jobs.
5. 4848469 readgate -> results/flawed/nph6/nph_read_gate.json (`"passes": true` iff pi0-H* K1-K6 >= .70; otherwise UNTESTABLE and every later gated job skips).
6. 4848470, 71 pi0-H* N jobs; 4848472-75 pi0-A0 (4 groups); 4848476-79 pi0-sL (4 groups); all afterany readgate, run only if the gate passed and the cap holds.  Outputs results/flawed/nph6/a_<pzH|pzA0|pzsL>/<arm>_main_<tid>.json (+ chan_/where_ logs).
7. 4848480 readmain -> results/flawed/nph6/nph_read_main.json (n1 primary, n1r, n2, d1'', nK, sensitivities, Holm); runs only if all 36 cells exist (kill k4).
8. 4848481-86 E1 (results/flawed/nph6/e1_<arm>.json and e1_dec_<a>_<b>.json; pi0-H* class-H share and decomposition vs pi0-A0).

**How to read.**  Look at nph_read_gate.json first (`gate.K_mean`, `passes`); then nph_read_main.json: `contrasts` keys "n1 PRIMARY ..." (label SUPPORTED / FLAGGED / REVERSED / EQUIVALENT / NOT SHOWN, `ci`, `flagged_tasks`), "n1r ...", "n2 ...", "d1'' ...", "nK ...", "sens ...", `holm`.
Compare to the predictions in experiments/prereg/nph_pred.txt (p0-p8).  Then fill the sections below.

**What to check.**  (a) `sacct -u $USER -S 2026-11-23 -X --format=JobName,Elapsed,State | grep nph_` : GPU-h used vs the 23 cap (wrappers skip rollouts above 21.5 wall-h of nph_* jobs; training chain ~7.4 h).  (b) any FAILED roll job (kill k4: no read until all cells exist; resubmit with the same script, a finished cell is skipped).
(c) where_pzH_*.jsonl logs show nonzero head_abs_max (serving smoke); pcheck output in .claude_tmp/pp/nph_gated_nph_pcheck_*.out.  (d) If the chain ended before step 6000 (node trouble), resubmit `nph_chain.sh` (resumable) and rerun the pipeline script after fixing the dependency ids.
Do not edit any .sh under a queued job.  Do not touch results/l90c/split_sealed.json.  Nothing was pushed.

## Gate metrics (seen at registration)
Head gates on the 29 validation demos (1,320 held rows; results/flawed/raah0/fit.json): H* G1 mean cos .9709 (>= .80), G2 median ratio .8888 (>= .60), G3 median |h| .0330 (<= .05): all pass.  Registered H on the same rows: G1 .6777, G2 4.611, G3 .3890 (fails all three).
Val R2 of the chunk target .9613; npz sha256 ee217430c8ba3fcd85384d99cc76524ac1ff357000eafe87cfd9e1593a29fabe.  Checks (job 4847548) and smoke (4847523, step-0 loss .1743 vs pi0-sL .1718) passed.

## RESULT (2026-11-24): all 26 pipeline jobs COMPLETED; every registered quantity read

Source files: results/flawed/nph6/nph_read_gate.json, nph_read_main.json, e1_pz{H,A0,sL}.json, e1_dec_*.json.  Arm names: pi0-H* = pzH, pi0-A0 = pzA0, pi0-sL = pzsL.  Set: 6th held-out set (k 820-879, 60 states per task, 360 episodes per arm and group).

**Gate.**  pi0-H* K1-K6 mean .925 (K1 .983, K2 1.0, K3 1.0, K4 .967, K5 .633, K6 .967) >= .70: PASSES; N jobs ran.

**Success.**  N (held-out pairings): pi0-H* .600 (cheese .894, bottle .306), pi0-A0 .506 (cheese .783, bottle .228), pi0-sL .714 (cheese .900, bottle .528).  K (trained tasks): .925 / .906 / .839.
Per task N, H* / A0 / sL: N1 .683/.700/.750; N2 1.0/.667/.950; N3 1.0/.983/1.0; N4 .167/.050/.500; N5 .667/.567/.800; N6 .083/.067/.283.

### Registered quantities and labels

| quantity | estimate [95% interval] | label |
|---|---|---|
| gate K (pi0-H*) | .925 | passes (>= .70) |
| n1 PRIMARY: pi0-H* - pi0-A0, N (seed-widened, tasks fixed) | +.094 [-.010, +.199], raw one-sided p .039, no flagged task | NOT SHOWN |
| n1r: same, registered tasks+states bootstrap | +.094 [-.017, +.222], p .047 | NOT SHOWN |
| sensitivity sigma_s .05 | +.094 [-.053, +.242] | (not shown) |
| sensitivity FPC 6/8 at sigma_s .033 / .05 | [-.023, +.212] / [-.062, +.251] | reported |
| nK: pi0-H* - pi0-A0, K (NI -.05) | +.019 [-.122, +.161] | NOT SHOWN |
| n2: pi0-H* - pi0-sL, N (NI -.10) | -.114 [-.222, -.006]; registered bootstrap [-.247, +.003] | NOT SHOWN: non-inferiority at -.10 not shown; the point lies below the servo |
| d1'': pi0-A0 - pi0-sL, N | -.208 [-.317, -.100]; registered bootstrap [-.347, -.069] | SUPPORTED under both rules (flagged task N4: A0 .05 vs sL .50); third replication of d1' |
| d1'' K (reported) | +.067 [-.033, +.228] | reported |
| n1 by pairing: cheese / bottle | +.111 [-.061, +.333] / +.078 [-.017, +.183] | reported |
| Holm over {d1', r1, h1, n1} (adjusted p; survives iff < .025) | d1' .0045 yes; r1 .0020 yes; h1 .262 no; n1 .077 no | two survive |
| n1p pooled 5th + 6th | not in the reader output; not computed | not run |

Per-task n1 differences (H* - A0): N1 -.017, N2 +.333, N3 +.017, N4 +.117, N5 +.100, N6 +.017; no flag.  N4 and N6 (bottle pairings) remain near the floor for pi0-H* (.167, .083).

### E1 failure classes (6th set, 360 episodes per arm)

| arm | failures | G | W | O | H (held, never released) | M |
|---|---|---|---|---|---|---|
| pi0-H* | 144 | 2 (.014) | 0 | 3 (.021) | 97 (.674 [.070, .900]) | 42 (.292) |
| pi0-A0 | 178 | 7 (.039) | 3 (.017) | 81 (.455) | 43 (.242) | 44 (.247) |
| pi0-sL | 103 | 8 (.078) | 0 | 16 (.155) | 35 (.340) | 44 (.427) |

Shares are of each arm's failures.  The class-H share of pi0-H*'s failures is .674 (97 of 144; cheese .053, bottle .768), against E4's registered head H at .786 (55 of 70 on the 5th set) and pi0-A0 at .242.  Prediction p7 (share <= .30) is false: the "owner vanishes only at the release point" form does not remove the hold-to-the-cap failure on the failures that remain.  pi0-H* does remove the class-O failures of pi0-A0 (81 to 3) ; its cheese-pairing failures (19 of 144) are mostly class M.
Decomposition pi0-A0 minus pi0-H* (episode fraction; positive = A0 fails more in that class; total +.094): O +.217 [+.036, +.442], H -.150 [-.408, +.058], M +.006 [-.092, +.100], G +.014, W +.008; rule MIXED (neither "read but imprecise" nor "routing").  So H* trades A0's class-O failures for class-H failures, with a net gain of +.094.  E4's H against A0 was H -.122, M +.114, O +.092.
Decomposition pi0-sL minus pi0-H*: H -.172 [-.378, .000], O +.036, M +.006, total -.114; the servo-kept policy fails less than pi0-H* mainly in class H (it releases).  pi0-A0 minus pi0-sL: O +.181 [+.028, +.375], H +.022, total +.208; the servo's gain sits in class O, as on the earlier sets.

### Predictions as written (nph_pred.txt) vs outcome

| id | prediction | outcome |
|---|---|---|
| p0 | head gate G1-G3 pass 70%; registered H fails G1 80% | true, true (seen at registration) |
| p1 | pi0-H* 6th-set K >= .70 85%, point .88 | true; .925 |
| p2 | N: A0 .52 (.35-.68), sL .75 (.60-.88), H* .72 (.50-.88) | A0 .506 inside; sL .714 inside; H* .600 inside (below the point) |
| p3 | n1 SUPPORTED 50%, FLAGGED 7%, NOT SHOWN 30%, EQUIVALENT 10%, REVERSED 3% | NOT SHOWN (point +.094, just under .10) |
| p4 | n1r SUPPORTED 40% | false; NOT SHOWN |
| p5 | n2 NON-INFERIOR 30%, COST 10% | false; NOT SHOWN, point -.114, below the servo |
| p6 | d1'' SUPPORTED 80% (fixed rule), 65% (registered) | true under both |
| p7 | class-H share of H*'s failures <= .30, 65% | false; .674 |
| p8 | nK NON-INFERIOR 60% | false; NOT SHOWN (lower bound -.122) |

Kill rules k1-k4: none triggered (k4: all 36 cells exist).

### GPU-h per item (sacct Elapsed x GPUs allocated; cap 23)

| item | jobs | GPU-h |
|---|---|---|
| model checks (3 attempts, two failed) | 4847522, 4847535, 4847548 | .25 (.06 + .10 + .09) |
| 20-step smoke | 4847523 | .09 |
| training chain tl1-tl6 (tl1, tl2 TIMEOUT at 4 h; tl3-tl6 continued) | 4847565-70 | 8.43 (4.00 + 4.00 + .36 + .03 + .02 + .02) |
| 6th-set state build | 4847571 | .02 |
| export + torch parity check | 4848465-66 | .10 |
| pi0-H* K (K1-K6) | 4848467-68 | 1.24 |
| pi0-H* N | 4848470-71 | 1.63 |
| pi0-A0 (K and N) | 4848472-75 | 3.41 |
| pi0-sL (K and N) | 4848476-79 | 3.08 |
| reads, E1, geometry, disjointness, head fit | CPU | 0 |
| **total** | | **18.25** (cap 23) |

### Deviations (honest list)

Items 1-6 from the resume note stand (copied here).  New:

1. An attempt to run `python3` on the login node (an empty heredoc) was issued by mistake and blocked by the permission layer before it ran; nothing was executed.
2. The registration asserts "jax/pi0_g8v_raah_l_ae absent" was changed to "jax/pi0_g8v_raah_l_ae/tl absent": the 20-step smoke (a plan step before registration) created the `smoke` experiment in that directory.
3. Check attempts 1 and 2 failed (GPU OOM at model creation; a too-strict loss test that a float64 shift broke under bf16) and were re-run (3 attempts, .1-.2 GPU-h each); the SEEN field says so.
4. nph_read.py imports rvw5_read.py instead of copying it; the plan's `c_train_x.py --name` became the copy c_train_nph.py (the original restricts --name).
5. The unattended pipeline wrappers (nph_gated.sh, nph_pipeline.sh) were written after registration at the orchestrator's request; they carry gates and a 21.5 wall-h cap only, no statistics.  They are not hashed in the row.
6. fit.json and other results/ files are untracked (results/ is gitignored); only the npz and code are committed.
7. The training chain needed six links (tl1 and tl2 ended by the 4 h wall time, as expected; tl3 to tl6 were short resume and closing links (22, 2, 1.5 and 1.5 min)); the final checkpoint is step 6000, as for E4.
8. The d1'' contrast carries the N4 flag (A0 .05, sL .50) under the flag rule; the label is SUPPORTED because the interval excludes zero under both rules.
9. The option "n1p pooled 5th + 6th" in the registration is not in the reader and was not computed.
10. Pooling pi0.5 H (E4) and pi0-H* below is exploratory; it was not registered.
11. GPU-h are computed as Elapsed x allocated GPUs from sacct, not from the wrappers' wall-hour guard (21.5 wall-h), which counts all nph_* jobs including CPU jobs.
12. While writing this note the agent ran `python3 --version` once on the login node (a version query only; no script, no data read).

## Reading

Two independent learned separate owners (pi0.5 H, pi0 H*) both land about +.09 over their in-network comparators (+.089 [-.208, +.346] pooled; +.094 [-.010, +.199] on the 6th set), and neither is significant.  The fixed-form servo owner beats the in-network network three times on pi0 (development -.256, 5th set -.222, 6th set -.208; each interval excludes zero).  H* recovers about 45% of the servo's gain ((.600 - .506) / (.714 - .506)) and sits significantly below the servo (-.114, seed-widened interval [-.222, -.006]; the registered bootstrap [-.247, +.003] touches zero); non-inferiority at -.10 is not shown.  So the data support "a separate owner with a fitted fixed form beats the in-network route on pi0".  "Separation alone (learned form) suffices" is not shown: both learned owners are positive and unresolved, and H* trails the servo.  Two thirds of H*'s failures have the hold-to-the-cap shape that H showed (class H .674 against .786).  Any pooling of the two learned heads (about +.09 each) is exploratory and not registered.

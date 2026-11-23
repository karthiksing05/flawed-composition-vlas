# Review-atlas experiments: results (executor note, 2026-11-23)

Plan: docs/notes/plan_review_atlas.md (657f25d6).  Prereg rows (ledger/m3_prereg): `rva1` (01e9d6e8), `rvh` (504debe0), `rvw5` (687c2b00).  Raw outputs: results/flawed/rva1/, results/flawed/rvw5/, results/flawed/raah/.
**STATUS: E1, E2, E3 complete.  E4 (the learned head H) is NOT complete: the head was fit and checked, H training was started and then STOPPED at about step 2250 of 6001 because the projected
GPU-h exceeded the 42 GPU-h hard budget.  h1, h2, h3, hK, hF are therefore NOT READ (no H rollouts exist).**  Resume = `bash /coc/flash10/ksingara3/.claude_tmp/pp/rvh_chain.sh` (c_train resumes from the newest checkpoint of jax/pi05_g8v_raah_l_ae/tl).

## 0. Headline

| quantity | result |
|---|---|
| E1 gate (chan-log classes vs raa_instr, RAA-sL rx 24-43, N1/N4/N6) | PASS: 60/60 coarse classes agree; median abs d diff .40 cm but over only n = 3 released failures (weak) |
| E1 rule, pi0-A0 vs pi0-sL (dev) | MIXED (not P, not R); the servo gain is carried by class O (release >= 10 cm away, not at a known partner): d_O +.194 [+.039, +.386], d_M +.025 [-.111, +.142], d_W +.014 [0, +.039] |
| E1 rule, fresh set | same: MIXED; d_O +.167 [+.028, +.353], d_M -.003 [-.106, +.089] |
| d1' (pi0-A0 - pi0-sL, N, fresh) | **-.222 [-.383, -.069] SUPPORTED**; d4' (K) +.078 [-.019, +.228] NON-INFERIOR; **REPLICATES = true**; Holm-surviving (see 3) |
| c0 (A0 - RAA-sL, N) | -.167 [-.356, +.017] NOT SHOWN; cA (A - RAA-sL, N) -.086 [-.244, +.056] NOT SHOWN |
| r1 (W5 - noW, N) | **+.678 [+.492, +.864] SUPPORTED**; r2 (F6_b) +.400 [+.233, +.567] SUPPORTED; r3 (noW - A0, N) -.503 [-.806, -.217] **DEPENDENT** |
| h1 / h2 / h3 / hK | NOT RUN (E4 stopped, budget) |
| GPU-h | about 38.7 used by the stop (budget 42; completing E4 projected about 54) |

## 1. E1: failure split (CPU, rva1)

Classifier as registered (rva1_read.py; selftest passed on synthetic sequences for G, W, O, H, M bins, J, E, requeue duplicates).  Join: (task text, trial) between the episode json and chan log; **0 join failures (J = 0) in every arm** (kill k1 not triggered).
Geometry: results/flawed/rva1/geom_dev.json / geom_w5.json (initial positions; plate / bowl can move: stated limit).  Shares are shares of failures, [95% clustered CI] in results/flawed/rva1/rva1_<arm>.json.

### 1.1 Development states (N1-N6, 60 states each)
| arm | success | failures | G | W | O | H | M | M<3 / 3-6 / 6-10 (counts) |
|---|---|---|---|---|---|---|---|---|
| pi0-sL | .761 | 86 | .19 | .00 | .16 | .23 | .42 | 5 / 22 / 9 |
| pi0-A0 | .506 | 178 | .08 | .03 | .47 [.22, .66] | .17 | .25 [.08, .60] | 1 / 15 / 29 |
| RAA-sL | .878 | 44 | .02 | .00 | .30 | .20 | .48 | 3 / 14 / 4 |
| A | .778 | 80 | .01 | .24 | .14 | .08 | .54 | 6 / 31 / 6 |
| A0 | .686 | 113 | .04 | .12 | .35 | .08 | .42 | 12 / 23 / 13 |
| RAA-sL s43 (extra) | .822 | 64 | .00 | .00 | .08 | .41 | .52 | 6 / 24 / 3 |
| A+0 (extra) | .603 | 143 | .01 | .15 | .35 | .19 | .29 | 8 / 25 / 9 |
| B (extra) | .644 | 128 | .04 | .45 | .30 | .08 | .13 | 3 / 10 / 4 |

By pairing (shares of failures, W / O / M): pi0-A0 cheese 0 / .26 / .67, bottle .04 / .53 / .14; RAA-sL cheese 0 / .43 / .52, bottle 0 / .14 / .43; A0 cheese .04 / .08 / .81, bottle .14 / .43 / .31; A cheese .04 / 0 / .93, bottle .35 / .21 / .33.

### 1.2 Servo-gain decomposition (arm off minus arm kept, per-episode rates over all episodes; Sum_c d_c = success difference)
| pair | set | d_G | d_W | d_O | d_H | d_M | total | rule |
|---|---|---|---|---|---|---|---|---|
| pi0-A0 vs pi0-sL (primary) | dev | -.006 | +.014 [0, .039] | +.194 [.039, .386] | +.028 | +.025 [-.111, .142] | +.256 [.128, .386] | **MIXED** |
| pi0-A0 vs pi0-sL | fresh | see json | +.008 [0, .031] | +.167 [.028, .353] | | -.003 [-.106, .089] | +.222 [.069, .383] | **MIXED** |
| A0 vs RAA-sL | dev | +.008 | +.036 [0, .094] | +.072 | 0 | +.075 [.006, .161] | +.192 [-.019, .406] | MIXED |
| A0 vs RAA-sL | fresh | | +.025 | +.075 | | +.061 | +.167 | MIXED |
| A vs RAA-sL | dev | 0 | +.053 [0, .144] | -.006 | -.008 | +.061 | +.100 [-.05, .261] | MIXED |
| A vs RAA-sL | fresh | | +.044 | -.008 | | +.042 | +.086 | MIXED |
| noW vs W5 (fresh) | | +.008 | +.039 | +.381 | +.189 | +.061 | +.678 | MIXED |

Reading: neither registered story holds for pi0.  The servo gain is not a centimetre-precision gain (rule P needs d_M >= half of total with CI above 0; d_M is about 0) and not a known-partner mis-routing gain (rule R needs d_W; d_W is about 0).
Most of the pi0 gain is class O: the held object is released or dropped 10 cm or more from the commanded target and not near a known partner's target.  This does not decide between "wrong target" and "lost in transit": the class is "not at the target, not at a known partner".
The atlas label "read but not in control" is not contradicted (the object ends up far from the read target), but the wording "wrong known target" is not supported by the data (W is about 3% of pi0-A0's failures).  For A0 and A against RAA-sL the M share carries the gain (d_M +.061 to +.075), with wide CIs.

### 1.3 Phase B (tracks; F6_b only; rva1_phaseb.py)
F6_b failure split (dev; G never lifted, L > 10 cm, M3-10, M<3): pi0-sL success 0.0, failures G 10 / L 12 / M3-10 38 / M<3 0; pi0-A0 (success .20) G 0 / L 10 / M3-10 38; RAA-sL (.017) G 7 / L 3 / M3-10 49; A (.433) G 0 / L 3 / M3-10 31; A0 (.383) G 0 / L 18 / M3-10 19.
Fresh set: pi0-sL (.083) G 16 / L 13 / M3-10 26; pi0-A0 (.30) 0 / 8 / 34; RAA-sL (.05) 9 / 4 / 44; noW (.40) 0 / 3 / 33; W5 (.80) 4 / 0 / 8.  M<3 is 0 everywhere.
The F7 pool and K1t are not split (the second bowl and the stove site are not in the tracks).  The servo-kept arms miss the covered plate by 3-10 cm, not by more than 10 cm.

## 2. E2 / E3 on the fresh set rvw5 (Phase A: N1-N6 and K1-K6, 60 states; Phase B: rgc tids 7-10, F6_b = tid 8)
Fresh-set success (N / K): pi0-sL .739 / .833; pi0-A0 .517 / .911; RAA-sL .867 / .933; A0 .700 / .839; A .781 / .983; W5 .875 / .956; noW .197 / .272; RAA-sL s43 .822 / .872.
Phase B: W5 F6_b .80, noW .40, RAA-sL .05; K2t W5 .98, noW .08, RAA-sL 1.00; K1t W5 .92 / noW .85 / RAA-sL .95; F6_a W5 .92 / noW .88 / RAA-sL .97; pi0-sL F6_b .083, F6_a .70; pi0-A0 F6_b .30, F6_a .87.

| quantity | point | 95% CI | rule outcome |
|---|---|---|---|
| d1' pi0-A0 - pi0-sL, N (primary) | -.222 | [-.383, -.069] | SUPPORTED; raw one-sided p .0015 |
| d1' by pairing | cheese -.061 [-.172, +.045]; bottle -.383 [-.533, -.233] | | |
| d4' same, K | +.078 | [-.019, +.228] | NON-INFERIOR (> -.05) |
| REPLICATES (d1' SUPPORTED and d4' NI) | | | **true** |
| d2' pi0-A0 - A0, N | -.183 | [-.358, +.019] | reported |
| d3' pi0-sL - RAA-sL, N | -.128 | [-.292, +.022] | reported |
| c0 A0 - RAA-sL, N | -.167 | [-.356, +.017] | NOT SHOWN |
| cA A - RAA-sL, N | -.086 | [-.244, +.056] | NOT SHOWN |
| r1 W5 - noW, N (primary) | +.678 | [+.492, +.864] | SUPPORTED; raw one-sided p .0005 |
| r2 W5 - noW, F6_b | +.400 | [+.233, +.567] | SUPPORTED |
| r3 noW - A0, N | -.503 | [-.806, -.217] | **DEPENDENT** (upper < 0) |
| rK noW - W5, K | -.683 | [-.964, -.356] | reported: noW also collapses on the trained tasks (K2t .08) |
| ladder W5 - RAA-sL (owner removal), N | +.008 | [-.092, +.086] | descriptive; F6_b +.750 [+.633, +.867] |
| ladder RAA-sL - noW (servo removal), N | +.669 | [+.494, +.836] | descriptive; F6_b -.350 [-.483, -.217] (the servo is worse than none on the covered plate) |

Holm over the registered family {d1', r1, h1}: h1 does not exist, so only {d1', r1} could be adjusted.  Raw one-sided p: r1 .0005, d1' .0015.  Adjusted with m = 3 (counting h1 at p = 1, the most conservative case): r1 .0015, d1' .003; both survive (< .025, the one-sided equivalent of the 95% interval).  The reader's own Holm output exists only for parts h / all, which were not run.
Reading of E3: removing the servo from the RAA-sL checkpoint returns a failure WORSE than the native network (noW N .197 vs A0 .700; r3 DEPENDENT): by the plan's own reading rule this is engineering dependence of a residual-trained expert, not evidence that a network cannot own the where.  Removing the owner alone costs nothing on N (+.008).

## 3. E4: the learned head H (partial)
- Fit (experiments/raah_fit.py, results/flawed/raah/fit.json, npz sha256 8975f769eb8fc6eec14c53b3953af65a08b8cc5a35456c08643948601a9f607a): held rows 12,007 train / 1,320 val (13,327 total, as the plan), demos 271 / 29, best step 2900.
  **Validation R2 (chunk-consistent): head .8296, linear-saturating refit .4012 (K 2.840 vs gate 2.851), linear at gate K .4014; head - linear +.428.**  The head is not "almost the servo": it carries much more than the linear form (it is per-row, so it absorbs timing within the chunk).
  |h| along x at 2 / 5 / 10 / 20 cm (mean over rows): .017 / .051 / .321 / .890 vs servo .057 / .143 / .285 / .570: the head is weaker than the servo near the target and stronger far away.
- Checks (raah_check.py logs hashed in the rvh row): linear-form head equals servo_norm on 1,000 states (np 2.7e-7, jax 0, torch 6e-8); head np / jax / torch agree (2.4e-6 / 5e-6); zero off role 2 and at rel = 0; parameter tree equals RAA-c's (53 tensors, 32 frozen, state_proj trainable); Pi0H with the linear-form head equals Pi0S exactly (loss and sample diff 0).  Config check passed (raah_cfgcheck.py).
- Smoke (20 steps): step-0 loss .0685 (finite, <= 2 x .0556).
- Training: chain tl (tail-lab) started 05:07; link 1 ran 4 h to about step 2250 (7 s/step, slower than the plan's 5.4); loss .014 at step 2250.  Links 2-6 were cancelled while pending to stop the spend.  Checkpoints: robot/checkpoints_l90c/jax/pi05_g8v_raah_l_ae/tl (newest step kept).  No export, no H rollout.
- To finish (estimate): about 6.7 h more training, export, dev Phase A (3.3 h) and B (1 h), fresh-set Phase A (3.3 h) and B (1 h), about 15 GPU-h more.

## 4. Predictions: written vs outcome
rva1 (pred file as written):
| item | written | outcome |
|---|---|---|
| e1 gate passes | 85% | PASS (60/60; the distance criterion rests on n = 3) |
| e2 pi0-A0 shares | M .55 (.30-.80); W .12 (0-.35); G .10; H + O .23 | M .25 (outside range); W .03 (inside); G .08; H + O .64 (far off) |
| e3 rule | P 50%, R 15%, MIXED 35% | MIXED (dev and fresh) |
| e4 RAA-sL M >= .60 (75%); A0, A M >= .50 (65% each) | | RAA-sL .48 FALSE; A0 .42 FALSE; A .54 TRUE |
| e5 bottle carries more of pi0-A0's W (60%) | | TRUE but tiny (W: bottle .04, cheese 0) |
| e6 servo-kept arms' F6_b failures mostly L (70%) | | FALSE (L .20 pi0-sL, .05 RAA-sL; M3-10 dominates) |
rvw5:
| item | written | outcome |
|---|---|---|
| q1 d1' point -.18, SUPPORTED 70%, REVERSED 2%, d4' NI 75%, REPLICATES 55%, survives Holm 60% | | point -.222, SUPPORTED, d4' NI, REPLICATES, survives |
| q2 N: pi0-sL .74 (.60-.86), pi0-A0 .52 (.35-.70), RAA-sL .87, A0 .70, A .79, W5 .89 | | .739, .517, .867, .700, .781, .875 (all inside) |
| q3 c0 SUPPORTED 50%, cA 25% | | both NOT SHOWN |
| q4 noW N .45 (.15-.75); r1 SUPPORTED 92%; r2 SUPPORTED 80% (W5 .80, noW .35); r3 DEPENDENT 55%, EQUIVALENT 15%; noW K below W5 by > .05: 70% | | noW N .197 (below range); r1 SUPPORTED; r2 SUPPORTED (W5 .80, noW .40); r3 DEPENDENT; noW K .272 vs W5 .956: TRUE |
| q5, q6 (h1, h2, h3, hK, joint) | | NOT TESTED |
rvh: h0 (head val R2 exceeds linear by >= .02, 70%): TRUE (+.428; point predicted .45 vs .36).  h0b (dev K >= .70; dev N point .82): NOT TESTED.  h0c (step-0 loss below .0556, 70%): FALSE (.0685), and this was visible at registration (smoke ran before the row): it is not a clean prediction.

## 5. Deviations (honest list)
1. **E4 stopped, h1-h3/hK/hF unread; budget.**  Plan unit costs were too low: Phase A cost .66-.83 GPU-h per job (about 3.3 GPU-h per arm, plan 2), noW 5.6 (slow nodes).  Measured usage at the stop: Phase A 26.6, Phase B 4.8, set build 2.6, checks / smoke / nowcheck .6, H link 1 4.0 = about 38.7.  Finishing E4 projects to about 54 (> 42), so the chain was cancelled.
2. noW serving check re-designed (rvw5_nowcheck.py v4): two separately loaded backbones of the SAME config give different samples with identical seed and noise (v3: dims 0-2 up to .44, gripper about 2), so the plan's cross-instance comparison cannot discriminate (v1 failed for that reason, not because of a servo mismatch).  v4: input identity between configs (0.0), and in-instance servo toggle (chunk with servo - chunk with servo_k = 0 == servo_norm, diff 0.0, dims 3+ 0.0) on 8 held and 1 object-role synthetic states (not a2 captures).  Passed.  The earlier failed runs (nowcheck.json, _v2, _v3) are kept.  Consequence: instance-to-instance sampling nondeterminism exists in serving (noted, not investigated).
3. rg_comp_5.py: the plan's seed-disjointness assertion failed as first written (F6 sampler seed 810000 + k for k >= 1000 equals the F6 plate seed 811000 + k' of an earlier set; a cross-role equality).  Changed after the rvw5 row to compare seeds within each role; recipe and ranges unchanged (the row hashes the earlier file).  Disjointness job: 0 equal rows (n2 w5 vs 11 sets; rgc vs 4 sets).  F7 k 860-923 and F6 k 940-1003, 0 rejected.
4. rva1 Phase B lives in rva1_phaseb.py (separate file, not in the rva1 row's hash); only F6_b is classified.  The 5th-set E1 and its geometry use rvw5_geom.py (new file).
5. E1 gate: the median-distance criterion is on n = 3 episodes (few released failures among the 60).
6. Jobs were submitted for Phase A before the disjointness job (the n2 states are asserted distinct from 546 stored rows at build); the rvw5 row, set build and rollouts order otherwise as planned.  Commit and sbatch times: rva1 row committed 04:47:56, gate submitted 04:47:56 (same second, commit first); rvh row 05:07:16 and chain 05:07:19; rvw5 row 05:07:53 and build 05:07:56.  Not a violation but not a margin.
7. Phase A job logs are named raaw5r_roll_rvw5_a_* (inherited log name).  Option runs: RAA-sL s43 Phase A and pi0 F6_b / F6_a cells on the fresh set were run (descriptive).  H seed 2, the H probe and E1-opt were not run (the gate passed).
8. The rvh row's SEEN field does not mention that the smoke step-0 loss (.0685) was seen before registration.
9. Holm: only two of the three primaries exist (see 2).

## 6. Files
experiments/: rva1_geom.py, rva1_read.py, rva1_phaseb.py, rvw5_geom.py, rvw5_read.py, rvw5_nowcheck.py, rvw5_disjoint.py, fc_n2_w5.py, rg_comp_5.py, wa_sw5_5.py, raa_wh.py, wa_sh.py, wa_sh5.py, raah_fit.py, raah_check.py, raah_cfgcheck.py, c_train_h.py, prereg/{rva1,rvh,rvw5}{.py,_pred.txt}; lifelong_ood/policy/{raah_model.py, backbone_h.py, raah_head.npz}; l90c_config.py (add-only: pi05_g8v_raah_l_ae).
Job scripts (outside git): .claude_tmp/pp/rvw5_roll.sh, rvw5_wa.sh, rvw5_states.sh, rvw5_submit_a.sh, rvw5_submit_b.sh, rvh_train.sh, rvh_chain.sh, rvh_check.sh, rvh_roll.sh, rvh_wa.sh, rvh_export.sh, rv_gpu.sh.
Results: results/flawed/rvw5/{rvw5_read_d.json, rvw5_read_r.json, nowcheck*.json, a_*, b_*}; results/flawed/rva1/*.json; results/flawed/raah/fit.json.

## 7. E4 completed (orchestrator decision: finish H as registered; cap 16 GPU-h for E4; appended 2026-11-23, sections 0-6 above are the state at the earlier stop)

Training resumed with rvh_chain.sh from step 2250 (tail-lab), ran to step 6000 (links tl1 to ~4750, tl2 to ~5750, tl3 to 6000; final loss about .010), exported as pytorch/pi05_g8v_raah_l_ae_tl6000, served with PolicyBackboneH (server log "[rvh] PolicyBackboneH").  The serving smoke was folded into the first dev rollout (finite chunks, role-1 queries logged, no failures); no separate smoke job.
Dev rollouts: results/flawed/raaf/dev_H (Phase A, 60 states per task) and dev_H/wa (Phase B tids 7-10); fresh set: rvw5/a_H, rvw5/b_H.  Reads: results/flawed/rvw5/rvw5_read_h.json, rvw5_hF.json, results/flawed/rva1/rva1_w5_H.json and decomps.

**Gate:** dev K1-K6 of H = .897 (>= .70): PASS, H is testable.  Dev N = .758.  Fresh set: N .806, K .894.  Fresh Phase B: F6_b .583, K2t 1.00, F6_a .917, K1t .95.

| quantity | point | 95% CI | rule outcome |
|---|---|---|---|
| h1 PRIMARY H - A0, N, pooled dev + fresh (equal weight) | +.089 (dev +.072, fresh +.106) | [-.208, +.346] | NOT SHOWN (raw one-sided p .262) |
| h1 dev alone / fresh alone | +.072 / +.106 | [-.339, +.414] / [-.353, +.467] | NOT SHOWN |
| h2 H - RAA-sL, N, fresh | -.061 | [-.417, +.172] | NOT SHOWN (neither NI, lower > -.10, nor COST) |
| h3 H - A, N, fresh | +.025 | [-.395, +.347] | NOT SHOWN |
| hK H - RAA-sL, K, fresh | -.039 | [-.125, +.006] | NOT SHOWN (lower < -.05) |
| hF H - RAA-sL, F6_b | +.533 | [+.400, +.650] | reported (H .583, RAA-sL .05) |
| hF H - noW, F6_b | +.183 | [+.017, +.350] | reported |
| hF W5 - H, F6_b | +.217 | [+.067, +.383] | reported (W5 .80) |
| hF H - A0, F6_b | | | not available: A0 has no Phase B rollouts on the fresh set |

CIs on N are wide (about +-.25-.30 with six tasks).  Reading by the plan's table 6.9: h1 is NOT SHOWN and the point (+.089) is not within +-.05 of A0, so neither "separation, not form" (needs h1 SUPPORTED) nor "separation does not help" (needs h1 within +-.05) is established; the experiment is underpowered for the +.09 effect.  The head does not hurt (h1 not REVERSED) and does not clearly match the servo (h2 point -.061, CI wide).  The head does recover the covered-plate cell the servo kills (F6_b .583 vs .05).
E1 on H (fresh): success .806; 70 failures: G 4, W 0, O 5, H (held to the end, step cap) 55, M 6.  79% of H's failures are class H, a different failure shape from every other arm: the learned head leaves the object held at the target without releasing (its output is weaker than the servo near the target, 5 cm: .051 vs .143).  Decomposition A0 vs H: MIXED (d_H -.122, d_M +.114, d_O +.092; total +.106 [-.353, +.467]); RAA-sL vs H total -.061 [-.417, +.172] (the P / R rules are not meaningful for a negative total).
F6_b track split (fresh): H S 35, L 4, M3-10 21, G 0.

**Holm over the full registered family {d1', r1, h1}** (one-sided p in the registered direction, step-down, survives iff adjusted < .025):
| primary | raw p | Holm-adjusted p | survives |
|---|---|---|---|
| r1 (W5 - noW, N) | .0005 | .0015 | yes |
| d1' (pi0-A0 - pi0-sL, N) | .0015 | .0030 | yes |
| h1 (H - A0, N, pooled) | .2624 | .2624 | no |
(The reader's own `holm` field in rvw5_read_h.json contains only h1 because parts d and r were read in separate runs; the table above combines the three raw p's from rvw5_read_d.json, rvw5_read_r.json and rvw5_read_h.json.)

**Predictions (E4), written vs outcome:** h0 (head val R2 > linear + .02): TRUE (+.428).  h0b dev K >= .70: TRUE (.897); dev N point .82 (.65-.92): .758, inside.  h0c step-0 loss below .0556: FALSE (.0685, seen at registration).  q5: h1 point +.12 / SUPPORTED 45% / REVERSED 3%: point +.089, NOT SHOWN, not reversed; h2 NI 55% / COST 7%: NOT SHOWN; h3 SUPPORTED 15%: NOT SHOWN; hK NI 50%: NOT SHOWN.  q6: joint "separation, not form" 30%: not met; "separation does not help" (h1 within +-.05) 20%: not met.
Effect of the full outcome on the claim (plan section 9, E4 row): the learned separate head neither confirms nor refutes "separation, not form"; the necessity claim keeps E2 (d1', Holm-surviving) and E3 (r1, with r3 DEPENDENT) and gets no independent support from a learned head at this recipe and power.

**GPU-h for E4 since the resume:** 13.7 (sacct with duplicates: training 6.1 over tl1-tl3, dev Phase A about 2.2 plus a requeued first attempt of the rx K job, dev Phase B .9, fresh Phase A 2.8, fresh Phase B .8, export 0).  Under the 16 GPU-h cap.  Run total about 38.7 + 13.7 = 52.4 GPU-h.

**Deviations added in this part:** (a) the rvh row's SEEN field omits that the smoke step-0 loss (.0685) was seen before registration (also section 5, item 8); (b) the commit and submit of the rva1 row fell in the same second, 04:47:56, commit first (section 5, item 6) - the rvh and rvw5 rows have 3-4 s margins (05:07:16 / 05:07:19 and 05:07:53 / 05:07:56); (c) hF "H - A0" is unavailable; (d) the fresh-set H jobs were submitted once the dev main K results (.89 over 20 states) were in and before the dev rx K files finished (the gate then passed at .897 over 60), a small risk taken to save wall time; (e) no separate serving-smoke job (see above); (f) experiments/rvw5_hF.py is a new small reader written after the registration for the hF cells.

## 78b. RAA-s re-serve: the SR 78 checkpoint served WITH its servo (ledger row m3_prereg__libero_goal__raas_b__s0__5122d4b7, commit d1ade69c; plan docs/notes/raas_b_plan.md, predictions experiments/prereg/raas_b_pred.txt)

**SR 78 = RAA-s served WITHOUT its servo (a serving bug); SR 78b = the same checkpoint served as designed.** SR 78's registered read (N1-N6 .028, K1-K6 .092, NOT SUPPORTED) stands as the read of the served-without-servo system and is not edited or overwritten (results/flawed/raas_* files untouched). It must not be quoted as evidence about the servo. SR 78b is a re-serve of pi05_g8v_raas_ae_tl6000 (no retraining, same export) with the registered design, contrasts, states and labels of SR 78 unchanged.

**Result.** S1 primary PARTIAL (S1b only, lower bound .008; S1a lower bound -.019). RAA-s as designed: N1-N6 success .717 (RAA-o .594, RAA .656, RAA-c .503, rgp .897), K1-K6 .844 (RAA-o .881, RAA-c .814, R0 .853). Grasp right N1-N6 .967, K1-K6 .986. R0 fit gate .853 (passes). Selftest (job 4799985) reproduced SR 73's V1a / V1b and SR 76's W1a / W1b / WK exactly. The servo roughly closes the gap to RAA-o on the bottle pairings (+.261 over RAA-o) and beats RAA-c by +.214 on N1-N6; it does not reach rgp.

**The fix.** lifelong_ood/policy/backbone.py (PolicyBackbone.sample, after the Euler loop, about lines 296-303): when `getattr(model.config, "servo_k", 0)` is nonzero, add `servo_norm_torch(model.config, prefix.state, A)[..., :3]` to dims 0-2 of every chunk step in the normalised full chunk, before it is returned or unnormalised. Twelve inserted lines; configs without servo_k never enter the branch. sample_composed / sample_poe untouched (raas is served with hooks None, i.e. only through bk.sample).
Evidence (experiments/raas_b_test.py, job 4799556, PASS; the pre-change function is the git HEAD copy bound to the same model instance):
- bit identity, np.array_equal and max |diff| 0 on fixed noise (default, full=True, 4 steps): pi05_libero (frozen pi0.5, 9 samples), pi05_g8v_plain_ae (9), pi05_g8v_raac_ae (18).
- raas: patched backbone.sample equals PI0PytorchS.sample_actions on the same noise (max |diff| 0, six frames incl. an object-role frame and a zero-state frame where the servo is 0); patched equals old + servo (0); GuidedPolicy.infer equals unnormalize(PI0PytorchS.sample_actions) to 5.9e-8 action units.
- raas_diag.py rerun unchanged (job 4799549): served RAA-s mean |xyz - demo| over the 15 held-target frames now .176 (chunk-bias norm .292), RAA-c .167; the old "served + servo" row now double-counts (.302) and is meaningless.
- RAA-c state_proj (SR 77 not affected by the same class of bug): PI0PytorchC.embed_suffix applies state_proj(state) and denoise_step receives prefix.state, which backbone.v_full passes. Job 4799556: the served path (backbone.sample) on the raac checkpoint equals PI0PytorchC.sample_actions on the same noise (max |diff| 0, four held-target frames), and zeroing the prefix state changes the served chunk by up to .052 / .090 / .075 / .372 (so the token is read). RAA-c needs no state-dependent post-term, so nothing was missing.

**Smoke (before the main rollouts, after the prereg row).** Job 4799600, 4 episodes (tasks 0 and 22, smoke states 0-1), 0 server errors, 3 of 4 succeeded (K1 2 of 2, N4 1 of 2; SR 78 had N4 0 of 2).

**Pooled (60 states per task).**

| arm | N1-N6 success | right grasp | placed given right grasp | K1-K6 |
|---|---|---|---|---|
| RAA-s 78b (with servo) | .717 | .967 | .741 | .844 |
| RAA-s 78 (no servo, bug) | .028 | .958 | .029 | .092 |
| RAA-c | .503 | .981 | .513 | .814 |
| RAA-o | .594 | .989 | .601 | .881 |
| RAA | .656 | 1.000 | .656 | .839 |
| rgp | .897 | .989 | .907 | .950 |
| R2 | .389 | .972 | .400 | .753 |

Cheese N1-N3: RAA-s 78b .844 (RAA-o .861, RAA .844, RAA-c .644, rgp .961); bottle N4-N6: .589 (RAA-o .328, RAA .467, RAA-c .361, rgp .833).

**Per cell, success (right grasp in brackets for RAA-s 78b).**

| cell | RAA-s 78b | RAA-o | RAA | RAA-c | rgp |
|---|---|---|---|---|---|
| N1 | .817 (.95) | .850 | .683 | .700 | .967 |
| N2 | .717 (.95) | .783 | .850 | .233 | 1.000 |
| N3 | 1.000 (1.0) | .950 | 1.000 | 1.000 | .917 |
| N4 | .600 (.967) | .300 | .217 | .317 | .867 |
| N5 | .700 (.967) | .417 | .933 | .417 | .933 |
| N6 | .467 (.967) | .267 | .250 | .350 | .700 |
| K1 | .933 | .883 | .850 | .950 | 1.000 |
| K2 | .983 | .983 | .967 | .950 | 1.000 |
| K3 | .933 | .967 | .983 | .900 | .983 |
| K4 | .950 | .817 | .950 | .933 | .983 |
| K5 | .367 | .683 | .383 | .250 | .733 |
| K6 | .900 | .950 | .900 | .900 | 1.000 |

**Registered verdicts (paired bootstrap B = 2000, rng 0, tasks and states; labels from experiments/raas_b_read.py = raas_read.py with the arm swapped; results/flawed/raas_b_read.json).**

| contrast | delta | 95% CI | label |
|---|---|---|---|
| S1a RAA-s - RAA-o, N1-N6 success | +.122 | [-.019, +.264] | NO CLEAR DIFFERENCE |
| S1b RAA-s - RAA-o, placed given right grasp | +.140 | [+.008, +.280] | RAA-s beats RAA-o |
| **S1 primary** | | | **PARTIAL** |
| S2 RAA-s - RAA, N1-N6 | +.061 | [-.117, +.253] | NO CLEAR DIFFERENCE |
| S3a RAA-s - rgp, N1-N6 (non-inferiority -.10) | -.181 | [-.300, -.055] | COST (not non-inferior) |
| S3b RAA-s - rgp (beats) | -.181 | [-.300, -.055] | rgp beats RAA-s |
| SK RAA-s - RAA-o, K1-K6 (non-inferiority -.05) | -.036 | [-.178, +.075] | NOT SHOWN (lower bound below -.05) |
| X1 RAA-s - R2, N1-N6 (no label) | +.328 | [+.164, +.514] | none |
| X2 cheese N1-N3, RAA-s - RAA-o | -.017 | [-.133, +.067] | no clear difference |
| X3 bottle N4-N6, RAA-s - RAA-o | +.261 | [+.133, +.400] | RAA-s beats |
| X4 RAA-s - R0, K1-K6 | -.008 | [-.181, +.131] | NOT SHOWN |
| X5 RAA-s - RAA-c, N1-N6 (no label) | +.214 | [+.075, +.381] | none |
| X6 RAA-s - RAA-c, K1-K6 (no label) | +.031 | [-.022, +.097] | none |
| X7 78b - 78, N1-N6 (no label) | +.689 | [+.522, +.842] | none |
| X8 78b - 78, K1-K6 (no label) | +.753 | [+.558, +.933] | none |

The registered label is PARTIAL; SUPPORTED needs both lower bounds above 0 and S1a's is -.019. Say it as: placed-given-right-grasp beats RAA-o by a margin whose interval just clears zero (+.008), overall success is higher by .12 with an interval that includes zero, and the gain is concentrated on the bottle pairings.

**Channel at test (4 chan logs, 21,710 queries; simulator read only for the log).** Object role median .42 cm (n 15,601); target role median 1.04 cm (n 6,109, 86.5% within 2 cm). Sampled server-log state lines (3,118 + 3,723 + 6,401 + 8,468): role 1.0 in 15,601 and role 2.0 in 6,109, none other.

**Exploratory release-error check (NOT registered; raa_instr via raa_instr_raa.sh, N1 / N4 / N6 / K5 x 20 states, trials 24-43, NAME=raas_b, read by experiments/raas_b_instr_read.py -> results/flawed/raas_b_instr.json; other rows reused, none re-rolled).** Identity: the logged runs equal the unlogged raas_b_rx record in steps and success in 80 of 80 episodes. Success of 20: RAA-s 78b N1 17, N4 13, N6 10, K5 5 (RAA-c 15, 6, 9, 5; RAA-o 16, 4, 7, 13; rgp 19, 18, 16, 14; RAA-s 78 1, 2, 2, 0).

| task | arm | failed-episode releases (n) | median release xy error, failed (cm) | releases per episode |
|---|---|---|---|---|
| N1 | RAA-s 78b / 78 / RAA-c / RAA-o / rgp | 2 / 23 / 5 / 4 / 3 | 4.27 / 13.95 / 4.26 / 3.35 / 12.35 | .75 / 1.20 / .90 / .55 / 1.10 |
| N4 | RAA-s 78b / 78 / RAA-c / RAA-o / rgp | 10 / 26 / 19 / 25 / 2 | 6.47 / 11.09 / 11.76 / 6.47 / 2.72 | 1.00 / 1.40 / 1.20 / 1.35 / .45 |
| N6 | RAA-s 78b / 78 / RAA-c / RAA-o / rgp | 14 / 21 / 8 / 14 / 4 | 3.82 / 13.59 / 11.02 / 6.97 / 9.72 | 1.00 / 1.15 / .80 / 1.10 / .90 |
| K5 | RAA-s 78b / 78 / RAA-c / RAA-o / rgp | 14 / 15 / 15 / 6 / 3 | 14.74 / 57.23 / 14.81 / 18.47 / 5.36 | .85 / .75 / .90 / .85 / .65 |

Reading (descriptive): with the servo, failed releases on N4 land 6.5 cm from the target (RAA-c 11.8, RAA-o 6.5) and on N6 3.8 cm (RAA-c 11.0, RAA-o 7.0), at most about half of RAA-c's miss, so the servo does tighten release aim relative to RAA-c; N4 is level with RAA-o. K5 stays 14.7 cm (K5 success .367, below RAA-o's .683). N1 has only two failed-episode releases, so its median is not informative.

**Prediction scorecard (experiments/prereg/raas_b_pred.txt, scored as written; my probabilities were far too low).**
1. S1 SUPPORTED 7 / PARTIAL 8 / NOT SUPPORTED 85: PARTIAL occurred (the 8% bucket). Point estimate S1a -.07 (N1-N6 about .52, range .38-.68): MISSED (+.122, .717 outside the range).
2. S1a lower bound > 0 alone (5%): did not occur (-.019). X3 > X2 in the point estimate (50%): HELD (+.261 against -.017).
3. S2 beats RAA (4%): did not occur, HELD.
4. S3 non-inferior 1%, beats < 1%: neither occurred, HELD.
5. SK non-inferior (12%): did not occur (lower bound -.178), HELD. K1-K6 at least .80 (55%): HELD (.844). R0 fit gate (99%): HELD (.853).
6. X5 point above RAA-c's .503 (50%): occurred (.717). X5 interval excludes 0 (15%): occurred, in the positive direction. X6 point above .814 (40%): occurred (.844), interval includes 0.
7. X7 positive with lower bound > 0 (99%): HELD (+.689, [.522, .842]).
8. Plumbing (90%): HELD for the parts checked: served chunk bit-identical to PI0PytorchS.sample_actions on the same noise before the rollouts; no sampled state line outside roles 1 / 2; held-target channel median 1.04 cm (under 2 cm). The m = 1 gate was not separately logged at the model; role 2.0 appeared in 6,109 sampled lines and the channel logs have 6,109 target-role queries.
9. Exploratory: N4 failed-release median at most 4.5 cm (10%): MISSED (6.47). N1 and N4 both at most 2.5 cm (2%): MISSED. N4 and N6 not below RAA-o's (75%): MISSED on N6 (3.8 against 7.0 cm), N4 equal (6.47 against 6.47).
Held: 2 (X3 part), 3, 4, 5, 7, 8; missed: 1 (point), 9. The prediction that the servo would leave RAA-s near RAA-c (about .50) was wrong: the offline chunk error matched RAA-c's, but closed-loop success is .214 higher. "What would count against the claim" (S1a lower bound <= 0 with N4 and N6 not below RAA-o) did not hold in full (N6 is far below); "what would count for it" (S1a lower bound > 0 with N4 below 5 cm) did not hold either. The result is partial support: the servo helps over RAA-c and over RAA-o on the bottle pairings, matches RAA-o on cheese, and does not demonstrably beat RAA-o overall on 60 states per task. It loses on K5 and does not approach rgp.

**Deviations.** (1) Same checkpoint, no training touched; the only changed code path is PolicyBackbone.sample (plus new test / read scripts). (2) The registered S1-SK, X1-X6 contrasts are SR 78's; X7 / X8 (78b minus 78) were added unlabelled in the prereg; raas_read.py was copied, not edited, to raas_b_read.py (arm raasb = raas_b_{main,rx}_<tid>.json, SR 78's raas rows read as a reference). (3) raa_roll.sh and raa_instr_raa.sh unchanged (ARM=raac CONFIG=pi05_g8v_raas_ae NAME=raas_b; the exploratory output dir results/flawed/raa_instr/raas_b); smoke used raao_smoke_branches.json, results in .claude_tmp/pp/raas_b_smoke. (4) The fix test compares against the git HEAD copy of backbone.py bound to the same model instance (duck typing, one model in memory) rather than two loaded models. (5) raas_diag.py was rerun unchanged; its "+ servo" row now double-counts. (6) The patched check of "m = 1 on every held-target query" is read from the server state log (role 2.0 lines), not from a model-internal log. (7) Exploratory rollouts ran after the read as in SR 78. (8) Ledger rows are written dirty=true because the working tree has foreign uncommitted edits, as in earlier prereg rows. (9) One command attempt at the very start, an empty python invocation on the login node, was refused by the permission layer and produced no output; no python ran on the login node.
Jobs: test 4799556 (first attempt 4799548 failed on a module-registration bug in the test and was rerun), diag 4799549, prereg 4799597, smoke 4799600, rollouts 4799665-68, selftest 4799985, read 4799986, exploratory 4800013-14, instr read 4800216, channel summary 4800218.

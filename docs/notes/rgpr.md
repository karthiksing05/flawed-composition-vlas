# RGPR: ReGuide given our F7 referent resolver (evaluation only)

Registered: m3_prereg tag rgpr, ledger row `ledger/m3_prereg/m3_prereg__libero_goal__rgpr__s0__6810460b.json` (code and predictions commit 6810460b, row commit 5c3b19f8, written before any main rollout; smoke job 4841110 ran before the row, read for crashes and binding only). Predictions: `experiments/prereg/rgpr_pred.txt`. Code: `experiments/rgpr.py`, `experiments/rgpr_roll.sh`, `experiments/prereg/rgpr.py`. No existing file changed (`rg_comp.py` is imported; `Ctx` and `fx_r.Front` are patched in-process).

## Design
- Arm rgpr = SR 70's rgp (ReGuide RG-perc, unchanged) whose bowl instance is the F7R resolver R's pick (template parser + learned segmenter + depth; the other bowl is a non-instructed movable, as in rc). The VLA gets the ORIGINAL instruction and UNMODIFIED images: no unique-name prompt, no magenta role view, no memory slot. Smoke logs confirm prompt = task language, slot False, resolve and binding True.
- rc differs from rgpr only in those VLA-facing parts, so r2 measures them.
- Frozen pi0.5, same 40 main F7 states, same branches (br7_main), client seed 7, one trial per state, as rgp and rc in SR 70 (paired). pi0 not run (its NG and F6 gates failed in SR 70).
- Jobs: smoke 4841110, prereg 4841193, main rollouts 4841208 (overcap, one job, ~50 min), read 4841940. Outputs: `results/flawed/rgpr_roll`, `results/flawed/rgpr_read.json`. Paired clustered bootstrap, 40 states resampled together across the four tasks, B=2000.

## Results (pi0.5, F7, n=40 per task)
| cell | rgp | rgpr | rc |
|---|---|---|---|
| relL | .800 | .950 | .950 |
| relR | .050 | .775 | 1.000 |
| pos | .000 | .000 | .900 |
| neg | .575 | 1.000 | 1.000 |
| SP | .425 | .863 | .975 |
| NG | .287 | .500 | .950 |
| F7 pool | .356 | .681 [.644, .713] | .963 |

| contrast | point [95% CI] | label |
|---|---|---|
| r1 rgpr - rgp, F7 pool | +.325 [+.281, +.369] | RESOLVER HELPS |
| r2 rc - rgpr, F7 pool | +.281 [+.238, +.325] | RC LEADS |
| rc - rgp, F7 pool | +.606 [+.556, +.656] | |
| r1 SP / NG | +.438 [+.375, +.487] / +.213 [+.138, +.287] | HELPS / HELPS |
| r2 SP / NG | +.112 [+.037, +.200] / +.450 [+.400, +.488] | RC LEADS / RC LEADS |
| r2 relL / relR / pos / neg | .000 [-.10, +.10] / +.225 [+.10, +.351] / +.900 [+.80, +.975] / .000 | EQUIV / RC LEADS / RC LEADS / EQUIV |

rgpr closes .536 of the rc - rgp gap on the pool (point).

Mechanism (descriptive): rgpr binds the referent on 40/40 in every task and R is correct 40/40; bound-bowl registration error is .17-1.2 cm (rgp on pos / neg: 5.1 cm). On pos the VLA still goes to the table bowl (referent grasp 0 of 13 grasps), ReGuide flags "wrong binding", transports for about 11 steps (~2 transports per episode), hands control back, and the VLA returns to the wrong bowl until the step cap (310). Our ReGuide has no post-hand-back escort, so a binding that is right does not guarantee a right grasp when the unrewritten instruction pulls the VLA to the other bowl. relR shows the same effect in milder form (.775).

## Predictions against outcomes
- Held: r1 HELPS (95%), point +.325 inside the predicted +.20 to +.60 (below the +.44 point); r2 RC LEADS (50%), point +.281 inside the predicted +.00 to +.38 (above the +.16 point); r2 on NG > r2 on SP; binding >= 38/40 and R 40/40 everywhere; registration within 1 cm of rc; pool .681 within .55-.95; SP .863; NG .50.
- Missed: pos predicted .65, outcome .000; share of gap closed predicted >= .70 (55%) and >= .90 (25%), outcome .536.

## Reading for the paper (consistent with, not proven)
- The F7 gain of rc over rgp (+.606) splits: giving ReGuide the resolver recovers .325 (about 54%), and the remaining +.281 sits with rc's VLA-facing parts (unique-name prompt, role view painted in both cameras, plate memory slot), concentrated on the positional / negated "on the plate" families (pos +.900, NG +.450) and on relR (+.225). On relL and neg, ReGuide plus R is already equal to rc (.950, 1.000).
- So the SR 70 F7 margin of +.550 / +.663 was partly unfair, as suspected (SP margin falls to +.112, NG to +.450), but it does not disappear: ReGuide's transport cannot use a correct referent when the frozen VLA still reads the ambiguous instruction and sees both bowls, with no escort after hand-back. This supports "explicit modules repair F7" in two parts: the referent resolver, and a channel that tells the VLA which entity (the view and prompt). The Appendix L expectation "a template parser added to ReGuide would remove the F7 margin" (TODO(pending)) is NOT supported at this strength: it removes about half; replace it by the measured numbers.
- The `rc - f7r` SR 70 result (+.000 SP, +.087 NG) and this one agree: the VLA-facing view carries most of the repair; the ReGuide transport adds little on top of it.
- Caveats: one simulator, one VLA (pi0.5), one client seed, one trial per state; rgpr runs under rc's cell of rg_comp's client with patches, so its `_rc.json` side files carry arm "rc"; the W5-vs-ReGuide F7 pool (.750 against .375, different states) was not rerun with rgpr.

## Deviations
- None from the registration. Scratch wrappers (`pp/rgpr_job.sh`, `rgpr_read.sh`, `rgpr_prereg.sh`) were made from the rg_comp ones; one `sed -i` was run on the new scratch file `rgpr_read.sh` (not a tracked or running script) and one on `/dev/null` by mistake (failed, no effect).

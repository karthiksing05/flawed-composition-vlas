# RAA-sL-W5 (SR 87): the module owns WHERE (the release point, and only until the expert can finish); the expert owns HOW.  Confirmatory run on FRESH states.

Written 2026-11-20, before the prereg row and any W5 rollout.  Predictions with probabilities: experiments/prereg/raasw5_pred.txt (copied into the row).  Origin: the SR 86 diagnostic (docs/notes/sr86_diag.md) and its two proposed changes.
No training; same checkpoint as SR 82 / 84 / 85 (pi05_g8v_raas_l_ae main/6000).
Code: lifelong_ood/policy/backbone_w.py (modes 5 / 6 / 7 added to `where_own_apply`, class lookup in `PolicyBackboneW.observation`; modes 0-4 untouched and tested bit-identical to the SR 85 code), lifelong_ood/policy/raasw5_mu.json (the table),
experiments/raasw5_mu.py (computes it), experiments/raa_w.py / wa_sw.py (argparse `choices` extended to 0-7), experiments/wa_sw5.py (wa_sw.py with two lines changed: it imports rg_comp_x), experiments/rg_comp_x.py (rg_comp.py with the fresh k ranges),
experiments/fc_n2_wx.py (fresh N2 states), experiments/raasw5_test.py, experiments/raasw5_read.py, experiments/prereg/raasw5.py; job scripts in /coc/flash10/ksingara3/.claude_tmp/pp/ (raaw5_roll.sh, wasw5_roll.sh, rg5_roll.sh, rgcx_roll.sh, w5_states.sh, raasw5_test.sh, raasw5_prereg.sh).

## Principle
SR 85's W3 / W4 pass non-inferiority to ReGuide-perc on N1-N6 but lose K5 (.683 -> .067) and the F7 unique-name controls / F6_a (.600 -> .46, .983 -> .88).  SR 86 traced both to ownership of WHERE that is wrong in two places:
(i) the module aims at the target's REFERENCE point, but the demos let go somewhere else: 12.5 cm from the rack's reference (3.5 cm stove, 5.3 cm cabinet, <= 1.4 cm plate / bowl), so the held bottle parks on the reference, the wrist presses the rack, and the expert's descent is blocked (K5);
(ii) the module owns the last centimetres, where its held-object estimate (p90 2.8 cm) is coarser than the 3 cm success tolerance and its 5-step open-loop chunk overshoots (45% of chunks at 1-4 cm pass through the target vs 11% for RAA-sL), so releases land >= 3 cm off (33 of 36 flips).
The module should own WHERE = where the demos let go, and hand the object to the expert once it is within the task's tolerance of that point; lift, descent, final alignment and the gripper stay the expert's.

## C1, release-point aim
Servo input e = rel_xy - mu_class (instead of rel_xy), where mu_class = the median, over the TRAINING demos of the same target class, of the channel's rel xy (target minus object, metres, base-frame axes) at the demo's release frame
(first open command while the role id = target).  Class = the role phrase the channel carries ("target: the rack" ..., the prompt at the server; `PolicyBackboneW.observation` reads obs["prompt"], strips any ". rel:" suffix, looks it up); an unknown phrase -> mu = 0 (today's behaviour).
mu is computed ONCE by experiments/raasw5_mu.py from lol/libero_g8v_raac (the RAA-sL checkpoint's training demos, 300 held episodes, no evaluation state, no rollout) and committed as lifelong_ood/policy/raasw5_mu.json before any W5 episode exists:
plate (-0.47, -0.15) cm, bowl (+0.29, -0.03), stove (-3.31, -0.82), cabinet (+4.23, -3.23), rack (-8.94, +8.13); n = 50 / 50 / 50 / 100 / 50.  The table equals SR 86's (sr86_q1d.json), which was computed from the same demos; it was not tuned.
## C2, handoff radius R
Ownership: the module owns xy only while held AND c = -dz >= CLEAR (3 cm, unchanged) AND |e| = |rel_xy - mu| > R = 3 cm.  Inside R the chunk is exactly RAA-sL's (expert xy).  R = 3 cm = the LIBERO body-target xy tolerance and SR 86's own derivation
(the module's held-object error p90 2.4-2.8 cm); fixed from SR 86's numbers, not from any rollout of W5, not tuned on F6_a / controls / K5.  Memoryless (a function of the current channel).
## Release gate (the decision asked for in the brief)
W4's gate (while held and |rel_xy| > 2 cm an open command is replaced by closed) becomes, in W5, a gate on |e| = |rel_xy - mu| (C1: the release point, not the reference) with radius = R = 3 cm, not 2 cm.  Reasons, all from SR 86:
(a) C2's contract is that inside R the chunk is exactly RAA-sL's; a 2 cm gate would alter the gripper command in the 2-3 cm band where the expert owns; (b) the failure SR 86 measured is a release at >= 3 cm (33 of 36 flips), i.e. exactly what a gate with radius 3 cm blocks, while RAA-sL's own successful first releases sit at 1.25-2.28 cm median offset with 12-45% of them at >= 3 cm
(frac >= 3 cm: .12 / .45 / .39 / .44 / .50 over F7 relL / relR / uniL / uniR / F6_a) and are not blocked by a gate of 3 cm; a 2 cm gate would block roughly half of the expert's releases that are within tolerance and risk a stall with the arm hovering in the band;
(c) the gate then acts only where the module owns xy (or would, at c < CLEAR) and the expert owns every release that is within R of the release point.  This is a design decision stated before any read, not a tuned value.  (Where mode 7, W5c1, has no R, its gate keeps W4's 2 cm, measured from the release point.)
## Arms (test-time only; same checkpoint, same harness; the module changes only xy [dims 0-1] and the open -> closed gripper command; z, rotation are the expert's)
| arm | where_own | aim | ownership of xy | gate |
|---|---|---|---|---|
| W5 | 5 | rel - mu | held, c >= 3 cm, \|rel - mu\| > 3 cm | \|rel - mu\| > 3 cm |
| W5c2 (C2 only) | 6 | rel | held, c >= 3 cm, \|rel\| > 3 cm | \|rel\| > 3 cm |
| W5c1 (C1 only) | 7 | rel - mu | held, c >= 3 cm | \|rel - mu\| > 2 cm |
| W4 (SR 85, re-run on fresh states) | 4 | rel | held, c >= 3 cm | \|rel\| > 2 cm |
| RAA-sL | 0 | - | expert | none |
W5c2 is "W4 + C2": the handoff radius governs both ownership and the gate, so W5c2 - W4 contains the gate radius change 2 -> 3 cm; W5 - W5c2 is C1.  W5c1 is "W4 + C1".  rgp = ReGuide-perc (the SR 52 / 73 arm); I = the frozen pi0.5.
## Fresh states (the run must not reuse any state used to design or read SR 70-86)
SR 82-86 iterated designs on the same states.  Used so far, by the code and notes: Phase A (N1-N6, K1-K6, all familiar-layout tasks share the states): fc_n2_tasks k 0-23 (harness trials 4-23 = k 4-23, the dev states) and fc_n2_rx k 540-579 (trials 24-63), plus the older state sets
fc_n2_extra / fresh / fresh2 / fresh3 / fresh4 / s1x / gx (k 24-195, 236-275, 500-539) in earlier SRs (which of them SR 70-86 touched was not audited: none is used here); SR 85's diagnostic and SR 86's instrumented runs used trials 24-43 of the same rx set.  Phase B (the rgc suite): rg_comp F7 candidates k 300-379 (44 accepted, 4 smoke + 40 main), F6 candidates k 400-498 (64 accepted, 4 smoke + 60 main); fx / f6o2 k 0-193.
Fresh: Phase A: fc_n2_rx.py's own docstring leaves k 580-639 free; no script, note or ledger row in the repo uses a state of that range; experiments/fc_n2_wx.py draws k = 580-639 by the same rule (np.random.seed = env.seed = 600000 + k), checks it reproduces stored k 0, 23, 540, 579 exactly, and refuses any state that
np.allclose-matches a stored one: 60 states, harness trials 0-59 (= k - 580), branch seed_base 13000000 + 1000 k (N2's convention): 60 per task, supply sufficient.  Phase B: experiments/rg_comp_x.py = rg_comp.py with F7 candidates k = 380-519 (fx's two-bowl rule, sampler seed 820000 + k, layout 821000 + k) and F6 candidates k = 500-619 (sampler 810000 + k, plate 811000 + k)
-- the first 4 accepted of each are the (unused) smoke rows and the next 60 are main rows (SR 70 had 40 main rows per F7 cell; 60 here); seeds do not overlap any SR 70 / fx / f6o2 / s2rg / mt range.  The accepted k's are listed in results/flawed/rg_comp_x/rg_comp_x_tasks.json and in SR 87.
The policy-free perception check (pcheck) of SR 70 is not rerun on the fresh rgc states (deviation, stated; the build's own render / duplicate / park checks are).
## Baselines re-rolled on the fresh states (nothing from SR 70-86 is reused as a comparator)
Phase A: W5, W5c2, W5c1, W4, RAA-sL (raa_w.py, --where_own 5 / 6 / 7 / 4 / 0), rgp (rg_roll.sh copy); 12 tasks x 60 states each.  The R0 fit gate of earlier reads (the unprompted R0 model on K1-K6 >= .80) is not rerun (no R0 episodes on fresh states); K1-K6 RAA-sL (the executor) plays that role and is reported.
Phase B (wa_sw5.py on the rgc cells): W5 full (tids 0-10), noR (0-3, 7, 8, 10), noM (0-3, 8); W5c2, W5c1, W4 and RAA-sL full (0-10); I (0-10) and rgp (0-3, 7, 8, 10) via rg_comp_x.py.  W5c2 / W5c1 / W4 / RAA-sL have no removal arms (the registered D contrasts are W5's).
## Reads (paired bootstrap B = 2000, rng 0; experiments/raasw5_read.py; tasks and states resampled as raa_read.diff; Phase B as wa_read.bootm)
Phase A:  A1 PRIMARY-A  W5 - RAA-sL, N1-N6: SUPPORTED iff lower > 0, REVERSED iff upper < 0, else NOT SUPPORTED.   A2a  W5 - rgp N1-N6: NON-INFERIOR iff lower > -.10; A2b W5 beats rgp iff lower > 0.   AK  W5 - RAA-sL K1-K6: NON-INFERIOR iff lower > -.05 (COST iff upper < 0, else NOT SHOWN).
  A3  W5 - W4 on K1-K6 (the C1 + C2 effect, mostly K5): SUPPORTED iff lower > 0.  Note on power (stated now): K5 is one of six resampled tasks, so in about a third of draws it is absent and the interval's lower end sits near 0 unless the other cells move; the K5-only contrast (X2, states resampled only) is reported beside it, unlabelled.
Phase B (wa_read on b_w5): D1' full - noR on the F7 pool (SUPPORTED iff lower > 0); D2' full - noM on F6_b; D3a noR - full on F6_b and D3b noM - full on F7 (EQUIVALENT iff inside +-.10); DK full - noR on K1t + K2t (NON-INFERIOR iff lower > -.05);
  gates on full W5: F6_a >= .50, unique-name controls (tids 4-6 mean) >= .50, K floor (K1t + K2t) >= .70;  gated PRIMARY: DISSOCIATION SHOWN iff D1', D2' SUPPORTED and D3a, D3b EQUIVALENT and all three gates TESTABLE; UNTESTABLE if the four hold and a gate fails; else wa_read's PARTIAL / NOT SHOWN.
  B1 full_W5 - full RAA-sL, F6_b (SUPPORTED iff lower > 0);  B2 full_W5 - frozen I, F6_b (SUPPORTED iff lower > 0, REVERSED iff upper < 0).
  C2 attribution (no label): full_W5c2 - full_W4 on the unique-name controls (tids 4-6 pooled) and F6_a (and F7 pool, F6_b, K1t + K2t); C1 attribution: W5 - W5c2, W5c1 - W4 on the same cells.
  Descriptive (no label weight): per-cell tables for W5, W5c2, W5c1, W4, RAA-sL, rgp, I; W5 - W4 on F6_b, F7, F6_a, K; full_W5 - rgp.
Scoring: every prediction in raasw5_pred.txt is scored as written; deviations go in SR 87.  One seed, one trial per state, one checkpoint; no multiplicity correction.
## Limits stated in advance
Release points are class medians of demos (the rack's spread is +-3 cm, dx p5-p95 -11.8 .. -5.6, dy +4.4 .. +12.0, which R = 3 cm hands to the expert only partly); the class is read from the role phrase, so a target phrase outside the table gets mu = 0; the gate and the ownership are memoryless;
the channel's z is unmeasured; W5c1 / W5c2 differ from W5 in more than one number where noted (gate radius); the fresh F7 / F6 states have no pcheck; fresh baselines can differ from SR 82-85's numbers (new states), which is why they are re-rolled.

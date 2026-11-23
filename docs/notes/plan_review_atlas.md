# Plan: four experiments for the remaining objections of docs/notes/review_atlas_adv.md (failure-atlas paper)

Written 2026-11-23.  Design only: nothing here was run, no job was submitted, no file other than this one was written.  For the executor agent.
Goal it serves: show the VLA weaknesses and test whether a NON-DISTRIBUTED architecture is necessary, where "non-distributed" means a separate module
owns the where (it can be learned).  A learned separate head that works counts FOR the proposition; the experiments are built so that it can also
count against it.

Inputs read: docs/notes/review_atlas_adv.md (all), raaf_sel.md, raaf_plan.md sections 6-11, oracle_pi0_results.md, plan_oracle_pi0.md, sr82_s43.md,
sr91.md, raas_plan.md (phase 0), paper/icml2027/main.tex lines 170-216, experiments/prereg/pzdev_pred.txt, raafp0_pred.txt, prereg/pzdev.py,
experiments/pp_atlas.py (Rec, classify), raa.py (server diag, client meta), wa_s.py (track sink), raa_instr.py / raa_instr_read.py headers,
lifelong_ood/policy/raas_model.py, backbone_w.py, backbone.py (servo loop), l90c_config.py (RAA-sL / A0 entries), .claude_tmp/pp/raa_roll.sh,
raasw5r_submit.sh, wasw5r_roll.sh; ls of the result dirs named below; bounded greps of the raaf* / pz* ledger rows for "5th".

---

## 0. Summary

| # | experiment | answers (review) | GPU-h | value per GPU-h |
|---|---|---|---|---|
| E1 | failure-mode split of the held-out-pairing failures from existing logs (pi0-sL, pi0-A0, RAA-sL, A, A0; A+0, B, RAA-sL s43 free) | §3.1(ii), §3.3 item 1, O2(a): wrong target vs centimetre miss vs grasp | **0** (CPU about 1 h); 2.5 only if its validity gate fails | highest |
| E2 | confirmatory "read but not in control" on the sealed 5th fresh set; pi0.5 references on the same states | §3.3 items 5-6, O2(b), O5 (Holm) | **10 + 2.5 set build** | high |
| E3 | removal cell for the where-module: W5 with the owner AND the in-model servo removed at serving (noW), on the 5th set, N and F6_b | O3, matrix F1 "not in the removal test" | **6.5** | high |
| E4 | a LEARNED separate where-head (BC-fit MLP on the token) in the servo's slot, expert retrained on the residual; vs RAA-sL and vs A0 / A | §3.1(i), O1 experiment | **about 16** | lower per GPU-h, largest effect on the claim |
| | **core total** | | **about 35** | |
| | options (cut first) | | +7.4 (or +19 with an H seed 2) | |

Order of launch (section 8): E4's training starts first because it is the long pole (about 10.5 h wall); E1 runs on CPU the same day; the 5th set is
built and E2 / E3 roll on overcap while E4 trains; E4's held-out rollouts come last.  The three held-out experiments share ONE fresh state set and ONE
pre-registration row (tag `rvw5`) written before the set is built, so every held-out contrast is paired on the same states.

---

## 1. Facts this design rests on (checked 2026-11-23)

1. **The "hand-set" servo's gain is already fit by behaviour cloning.**  raas_plan.md phase 0: K = 2.851 is the least-squares fit of demo action xyz at
   t + j (j = 0..9) on the channel rel_xyz at t over the 13,327 held rows of K1-K6 (pooled chunk-consistent R² .361); M = .9375 is the p99 of |demo
   action xyz| on held rows.  Only the FORM is chosen by hand: linear, saturating, translation only, gated by role = target, the same value on every
   chunk row.  The servo is a 1-gain learned head of fixed form, outside the network.  Two consequences: (a) the paper can say so now (rewrite, 0 GPU-h);
   (b) E4's learned head changes the FORM (free-form MLP, per-chunk-row output), not "hand-set vs learned" from scratch.
2. **Servo-off serving of the RAA-sL checkpoint is a config swap, no code.**  l90c_config: `pi05_g8v_raas_l_ae` and `pi05_g8v_raac_l_ae` (A0) are the
   same model class (RS.Pi0ConfigS, pi05, discrete_state_input False), the same AE assets / data (`("pi05_g8v_raac", "lol/libero_g8v_raac")`), the same
   weight loader (RaacLoader(LIBERO_PARAMS)); they differ ONLY in servo_fields(k=0.0).  backbone.PolicyBackbone.sample adds servo_norm_torch iff
   servo_k != 0 (backbone.py line 296).  So `--config pi05_g8v_raac_l_ae --ckpt pytorch/pi05_g8v_raas_l_ae_main6000` serves RAA-sL's expert residual
   with no servo.  PolicyBackboneW asserts servo_k != 0 (backbone_w.py line 102), so a servo-off arm must use where_own 0.
3. **Phase A rollouts (N1-N6, K1-K6) log enough for a failure split; no per-step track exists.**
   - Episode json (`<name>_{main,rx}_<tid>.json`): trial, success, steps, calls, `bound` (the object the gripper bound to) and `bound_step`.
   - Server channel log (`chan_<name>_<jobid>.jsonl`, written by raa.py serve --chan-diag, every raa_roll.sh copy passes it): one line per query
     (every 5 env steps) with ep (= trial), t, text, role ("object" / "target" = the client's held detector), `true_cm` = simulator truth of the
     channel (target minus held object while role = target; object minus eef otherwise), whole cm.  Checked in results/flawed/raaf/dev_pzA0
     (`chan_raafpzA0_4839567.jsonl`, 1.1 MB) and present for RAA-sL (results/flawed/raa/chan_raasl_*), A (raaf/dev_A), A0 (raaf/dev_A0), pi0-sL / pi0-A0
     (raaf/dev_pzsL, dev_pzA0), and RAA-sL seed 2 (raa/chan_raasl43_*).
   - pp_atlas.py's Rec (`_track.npz` with held / obj / tgt / ee per step) is used only by the atlas clients (pi05, groot, pmin, servo); none of the
     RAA-family rollouts carries it.
4. **Phase B rollouts carry per-step tracks.**  wa_s.py (and its copies wa_sw5z.py, pzdev_wa) write `<cell>_track.npz`: every object's body position per
   step (`pos`, names in `names`), eef, gripper command a6, hidden flag, success.  Present in results/flawed/wasl, raaf/dev_*/wa, raasw5r/b_*.
5. **An exact per-step re-roll instrument exists and reproduces episodes.**  raa_instr.py (observer under raa.py client) + raa_instr_read.py (categories
   OFF_XY / Z_FAIL / NO_CONTACT / UNDER_FALSE / TIPPED); raasl_instr.json: RAA-sL on N1, N4, N6, K5 x trials 24-43, re-rolls identical in steps and
   success on 79 of 80 episodes.  This is E1's validity check at 0 GPU-h and its fallback at about 2.5 GPU-h.
6. **The 5th fresh set may be used.**  raaf_plan.md 6.3 rule 5: "If no route passes the probe ... The 5th fresh set stays unbuilt / sealed for a later
   design."  raaf_sel.md (final, A+0 included): no route passed, no winner, the set is unbuilt.  The raafp0 row reserves it for A+0 only "if it is
   eligible" (it is not).  plan_oracle_pi0.md risk 10 names it as the place for a confirmatory pi0 read.  No other registration claims it (bounded greps
   of the raaf, raaf0, raafb, raafp, raafp0, pz, pzdev rows).  It is unbuilt: no fc_n2_w5.py, rg_comp_5.py, n2_states_w5 exist.  Its recipe is fixed in
   raaf_plan.md 7.1 and is used verbatim.  The new row must state that it takes the set over and that raafc will never be written.
7. **The owner-removal half of the where-module removal already exists on held-out sets.**  W5 is the RAA-sL checkpoint plus the W5 owner; RAA-sL full
   (with resolver R and pose memory M) is exactly "W5 minus the owner".  SR 91: W5 - RAA-sL on N +.017 [-.125, +.128] (A1), on F6_b +.767 [+.667, +.867]
   (B1); pooled over three sets N +.037 [-.020, +.090], F6_b +.611.  The owner adds the covered-plate aim, not N.  What has never been removed is the
   in-model servo.  E3 removes it.
8. **Unit costs** (raaf_plan 10, SR 91 sacct): Phase A about .5 GPU-h per job, 4 jobs per arm for 12 tasks x 60 states = 2 GPU-h per arm; Phase B .3-1.0
   GPU-h per job (tids 7-10 about .8 per arm); training 6,001 steps x 5.4 s = 9 h + about .5 restarts on one A40 chain.

---

## 2. E1: failure-mode split from logged rollouts (0 GPU-h; CPU sbatch only)

### 2.1 Question and claims at stake
Are the in-network failures on held-out pairings (N1-N6) routing errors (the right object carried to a wrong, known-partner target), centimetre misses at
the right target, or grasp failures?  The paper's label "read but not in control" (pi0) and the matrix's "missing function: where the object goes" need
routing errors; a precision deficit gives "read but imprecise".  The split also decomposes the servo's gain (pi0-sL minus pi0-A0, RAA-sL minus A0 / A)
by failure class.

### 2.2 Arms and data (all existing, development states: Phase A trials 4-23 main + 24-63 rx, 60 per task; Phase B rg_comp main)
| arm | Phase A dir (episode json + chan logs) | Phase B tracks |
|---|---|---|
| pi0-sL | results/flawed/raaf/dev_pzsL | dev_pzsL/wa |
| pi0-A0 | raaf/dev_pzA0 | dev_pzA0/wa |
| RAA-sL (seed 1) | raa (raasl_main_*, raasl_rx_*, chan_raasl_4806984-87) | wasl |
| A | raaf/dev_A | dev_A/wa |
| A0 | raaf/dev_A0 | dev_A0/wa |
| free extras (descriptive) | RAA-sL s43 (raa/raasl43_*), A+0 (dev_Aplus0), B (dev_B), W5 on held-out sets (raasw5r/a_w5, b_w5) | |

### 2.3 Classifier (registered before it runs on any arm; thresholds fixed here)
Per failed N episode (success false, no error):
- **G, grasp failure**: `bound` is not the instructed object (another object or none).
- Otherwise take the chan-log queries of that episode (join on job id, text, ep = trial; assert the query count equals the episode's `calls`).
  Held segments = maximal runs of role "target".  The **release query** is the last query of the last held segment.
  d = |true_cm[:2]| at the release query (xy distance, held object to the commanded target, cm).
- **H, held to the end**: the last query of the episode has role "target" and steps = the cap (300).
- **W, wrong target**: d >= 10 cm and the held object at release lies within 8 cm (xy) of a known-partner target of that object
  (object position relative to the commanded target = -true_cm; partner offsets from 2.4).
- **O, off-route**: d >= 10 cm and not W (dropped in transit, released elsewhere).
- **M, centimetre miss**: d < 10 cm, split M<3 (inside LIBERO's 3 cm xy tolerance, failed on height / contact / tilt), M3-6, M6-10.
- No held segment after a right grasp: counted in O with a flag `never_carried`.
Thresholds: 3 cm = LIBERO's On xy tolerance for body targets; 10 cm and 8 cm separate centimetre misses from mis-pairings, which are 20-40 cm apart
(raaf_plan 11).  Resolution limits, stated with the result: queries every 5 env steps (the object moves at most a few cm between the release query and
the release), whole-cm rounding, and the rigid in-hand offset assumed by the channel's truth.

### 2.4 Partner geometry (CPU job)
`experiments/rva1_geom.py` (new, LIBERO venv, CPU): for every N task and dev state (n2_states main 4-23, n2_states_rx 24-63) it restores the init state
and records the xy position of the commanded target and of every known-partner target of the instructed object (the targets that object is paired with
in K1-K6, from n2_tasks.json / fc_n2_tasks.PAIRS) through fa1_common.scene.  Fixtures do not move; the plate can be pushed, which is a stated limit.
Output results/flawed/rva1/geom_dev.json (refuse to overwrite).  About .5 CPU-h.

### 2.5 Phase B split (exact, from tracks)
F6_b (tid 8) and the F7 pool (tids 0-3): the final resting xy distance of the instructed object to its target body or site from `pos` at the last step,
plus "never lifted" from the object's z.  Classes: G (never lifted / other object lifted), L (> 10 cm: lost the target), M3-10, M<3.  For F6_b the review
cites frozen pi0.5's "44 of 50 failures release within 3 cm"; this gives the same split for the trained arms.

### 2.6 Validity gate (0 GPU-h)
Run the chan-log classifier on RAA-sL's rx trials 24-43 of N1, N4, N6 and compare with the per-step categories of the same episodes in
results/flawed/raa_instr/raasl (raa_instr_read functions imported unchanged): the coarse class {success, G, release within 10 cm, release beyond 10 cm,
held to end} must agree on >= 90% of the 60 episodes, and the median |d_chan - d_instr| over released failures must be <= 2 cm.
If the gate fails, run **E1-opt** (2.5 GPU-h): raa_instr.py re-rolls of pi0-A0, pi0-sL, A, A0 on N2, N4, N5, N6 x trials 24-43 through copies of the
arms' roll scripts (raa_instr_raa.sh pattern), and classify from the per-step npz instead (raa_instr_read categories plus distance to the partner).

### 2.7 Registered quantities and rules (`rva1`)
- Per arm: shares of failures in G, W, O, H, M (and M bins), 95% CI by clustered bootstrap (tasks, then states), B = 2000, rng 0.  Also by pairing
  (cheese N1-N3, bottle N4-N6).
- Servo-gain decomposition: for class c, Δ_c = rate_c(servo-off arm) - rate_c(servo-kept arm) per episode, paired by (task, state); Σ_c Δ_c = the
  success difference.  Pairs: pi0-A0 vs pi0-sL (primary), A0 vs RAA-sL, A vs RAA-sL.
- **Rule P, "read but imprecise"** (pi0): Δ_M CI lower > 0 AND Δ_M >= half of the total Δ AND Δ_W CI upper < .10.
- **Rule R, "routing"** (pi0): Δ_W CI lower > 0 AND Δ_W >= half of the total Δ.
- **MIXED** otherwise.  The same rules are reported for A0 and A against RAA-sL.
- What it means: P -> the atlas row is renamed "read but imprecise"; the necessity claim is about a module owning the EXACT release point, and the
  matrix's "missing function: where the object goes" is replaced by "missing function: the release point".  R -> "read but not in control" stands in
  its strong form (the network reads the target and still carries the object elsewhere).  MIXED -> report both shares.

### 2.8 Files
experiments/rva1_geom.py, experiments/rva1_read.py (new, CPU numpy; `selftest` on synthetic chan sequences covering every class; outputs
results/flawed/rva1/rva1_<arm>.json, rva1_decomp_<a>_<b>.json, rva1_gate.json, opened with "x").  Prereg: experiments/prereg/rva1_pred.txt +
experiments/prereg/rva1.py (copy of pzdev.py: hashes the two scripts, geom_dev.json, the pred file; asserts results/flawed/rva1/rva1_*.json absent;
written after the geometry job and the selftest, before the classifier reads any arm).

---

## 3. The shared held-out set and its registration (tag `rvw5`)

### 3.1 Build (exactly raaf_plan 7.1)
- N1-N6, K1-K6: n2 familiar k = 760-819 (60), trial t = k - 760, sampler seed 600000 + k, branch seed_base 13000000 + 1000 k.
  `experiments/fc_n2_w5.py` = copy of fc_n2_wz.py with K0 = 760 and wz added to the refusal list (fc_n2_wz.py read, not edited).
  Outputs results/flawed/n2_states_w5, n2_branches_w5.json (+ _smoke).
- F7: `experiments/rg_comp_5.py` (copy of rg_comp_z.py) candidates k = 860-939; F6: candidates k = 940-1019.  First 4 accepted of each = smoke (unused),
  next 60 = main.  Seeds F7 sampler 820000 + k / layout 821000 + k; F6 810000 + k / plate 811000 + k; the build asserts no seed equals an earlier set's.
- Disjointness job: copy of raasw5r_disjoint.py run as a new file; np.allclose of every new row against every stored n2_states_* and rg_comp* row = 0.
- About 2.5 GPU-h (raaf_plan budgets about 3 including CPU).
- The `rvw5` row is written BEFORE the build (raaf_plan 7.1: "Built only after ... the confirmatory prereg row is written"), and nobody rolls on the set
  before it.

### 3.2 Arms on the set (Phase A = N1-N6 + K1-K6, 60 states; Phase B cells named per experiment)
| arm | checkpoint (pytorch/) | serving | used by |
|---|---|---|---|
| pi0-sL | pi0_g8v_raas_l_ae_tl6000 | raa.py serve, config pi0_g8v_raas_l_ae (pzdev_roll.sh pattern) | E2 |
| pi0-A0 | pi0_g8v_raac_l_ae_tl6000 | config pi0_g8v_raac_l_ae | E2 |
| RAA-sL | pi05_g8v_raas_l_ae_main6000 | where_own 0 | E2, E3, E4 |
| A0 | pi05_g8v_raac_l_ae_tl6000 | | E2, E4 |
| A | pi05_g8v_raac_lf_ae_tl6000 | | E2 (reference) |
| W5 | pi05_g8v_raas_l_ae_main6000 + raasw5_mu.json | raa_w.py --where_own 5 | E3 |
| noW | pi05_g8v_raas_l_ae_main6000 | raa.py / wa_sw5_5.py, where_own 0, **--config pi05_g8v_raac_l_ae** (fact 2) | E3 |
| H | pi05_g8v_raah_l_ae_tl6000 (E4) | raa_wh.py (section 6.5) | E4 |
Phase A job scripts: copies of raaw5r_roll.sh (SR 91: WHERE_OWN, CONFIG, CKPT, FRESH, BR; tid groups 0+2+4, 6+8+10, 16+18+20, 22+24+26), FRESH =
n2_states_w5, BR = n2_branches_w5.json.  Phase B: `experiments/wa_sw5_5.py` = wa_sw5z.py with the two rg_comp_z references -> rg_comp_5 (as SR 91
made wa_sw5z), through a copy of wasw5r_roll.sh (it already takes CKPT, CONFIG, WHERE_OWN, CELLS).  Every server log keeps --chan-diag, so E1's
classifier runs on all of these rollouts at no cost.
Remember: sbatch --export splits on commas (use '+' in TIDS / CELLS), and never edit a .sh that a queued job will read.

### 3.3 Reader
`experiments/rvw5_read.py` (new; copy of raasw5r_read.py with the arm table above): Phase A by raa_read.diff (paired, B = 2000, rng 0, tasks then
states); Phase B by wa_read.bootm; Holm over the three primaries d1', r1, h1 (raw and adjusted p from the bootstrap, both reported); E1's classifier
on every arm (secondary).  It reads only the arms whose files all exist (the E4 arm is read later by `--part h`, same code); outputs opened with "x".

---

## 4. E2: confirmatory "read but not in control" (pi0) on fresh states, with pi0.5 references (10 GPU-h + the build)

### 4.1 Design
No training.  Both pi0 arms and three pi0.5 references on the 5th set, Phase A only (the pi0 F6 cells are an option).  The registered contrasts copy
pzdev's, renamed with a prime.  The dev probe's S_off (.542 [.478, .609]) is a property of the checkpoint and is carried over, not re-measured.

### 4.2 Registered contrasts
- **d1' PRIMARY**: pi0-A0 - pi0-sL on N1-N6.  SUPPORTED iff CI upper < 0; REVERSED iff lower > 0; else NOT SHOWN.  Holm family {d1', r1, h1}.
- d4': pi0-A0 - pi0-sL on K1-K6, NON-INFERIOR iff lower > -.05.
- Rule "read but not in control REPLICATES on fresh states" iff d1' SUPPORTED and d4' NON-INFERIOR (S_off already >= .30).
- d2' pi0-A0 - A0 (pi0.5), d3' pi0-sL - RAA-sL on N: reported, no rule (cross-backbone).
- **c0** A0 - RAA-sL on N (pi0.5: the expert learns the geometry vs the servo, same data, matched): SUPPORTED-for-module iff CI upper < 0.
- **cA** A - RAA-sL on N (the best in-network route, the review's -.100): same rule.  These two answer the review's §3.1 point 1 on fresh states,
  where the servo's constants have no home advantage.
- E1 classes on every arm (shares and the servo-gain decomposition), reported.

### 4.3 What the outcomes mean
| d1' | d4' | reading |
|---|---|---|
| SUPPORTED | NI | the most-cited number replicates on states nobody has seen; paper keeps the row, quotes the fresh-state value, Holm-adjusted |
| NOT SHOWN, point < -.10 | any | direction holds, size not confirmed; the row is "development states; fresh-state direction only" |
| NOT SHOWN, point >= -.05 | | the dev contrast does not replicate: drop "read but not in control" as a claim, keep pi0's token read (R2) only |
| REVERSED | | refuted |
c0 / cA SUPPORTED on fresh states would close the review's point that A's -.100 had a CI through 0; NOT SHOWN keeps the current scoped wording.

---

## 5. E3: a removal cell for the where-module (6.5 GPU-h)

### 5.1 Design
The removal ladder on one checkpoint, one set, everything else unchanged (resolver R and pose memory M kept in Phase B):
W5 (owner + servo) -> RAA-sL (servo only; owner removed; rolled for E2) -> **noW** (owner and servo removed: RAA-sL's expert residual alone, fact 2).
A0 (a network trained with the same data and no servo) is the reference for "the network owning the where itself".
Phase A: W5, noW (4 GPU-h).  Phase B: tids 7, 8, 9, 10 (K2t, F6_b, F6_a, K1t) for W5, noW and RAA-sL (about 2.5 GPU-h; F7 tids 0-3 are not needed for
this question).  Serving check before rollouts (`experiments/rvw5_nowcheck.py`, about .2 GPU-h): on 8 held queries from existing RAA-sL captures
(results/flawed/raaf/a2/cap_raasl*), the noW chunk plus servo_norm_torch(state) equals the RAA-sL chunk on dims 0-2 and the chunks are equal on dims
3-6, same seed and noise, within bf16 tolerance .02.

### 5.2 Registered contrasts
- **r1 PRIMARY**: W5 - noW on N1-N6, SUPPORTED iff lower > 0 (Holm family).
- r2: W5 - noW on F6_b, SUPPORTED iff lower > 0.
- r3 (circularity): noW - A0 on N: DEPENDENT iff CI upper < 0 (the expert, trained around the servo, does worse without it than a network trained
  without one); EQUIVALENT iff the CI lies inside +-.10; else NOT SHOWN.
- rK: noW - W5 on K1-K6 (the servo also acts on the trained tasks' held phases): reported.
- Ladder (descriptive): W5 - RAA-sL (owner removal, re-measured here) and RAA-sL - noW (servo removal) on N and F6_b; E1 classes per rung.

### 5.3 Reading
r1 SUPPORTED with r3 EQUIVALENT: removing the where-module returns the held-out-pairing failure at the level of a network trained without it; the matrix
F1 row gets a removal cell ("removal returns the failure").  r1 SUPPORTED with r3 DEPENDENT: removal returns a failure WORSE than the native network,
which is engineering dependence of a residual-trained expert; report it as such ("in the system we built"), never as evidence that the network cannot
own the where (that is E2's c0 and E4's h1).  r1 NOT SHOWN: the servo is not what carries N in W5, and the F1 removal cell reads "no effect".

---

## 6. E4: a learned separate where-head (about 16 GPU-h)

### 6.1 Hypothesis and the claim it tests
A small network outside the action expert, trained by behaviour cloning on the same token and the same demonstrations, placed in the servo's slot (the
expert trains on the residual), matches the fixed-form servo (RAA-sL) and beats letting the expert learn the same geometry from the same data (A0; and
the dosed route A).  If so, separation, not the hand-chosen form, is what carries the result, and "explicit owner" becomes an architectural claim:
the where is owned by a separate module with its own input (the token) and its own output slot (the translation residual).

### 6.2 Supervision: what the demos give
- The servo's own correction is computable from demo states (clip(K rel) on held rows), but distilling it reproduces the servo: no information.  It is
  used ONLY as the identity test (6.6 item 1).
- The demos' own actions on held rows, regressed on the token: the same supervision that fixed K (fact 1), now with a free-form function.  **This is
  the training target.**  The head learns E[a_xyz(t + j) | rel(t)] for j = 0..9, which also absorbs where the demonstrator slows and stops relative to the
  target (the release-point information that W5 takes from its mu table, here learned implicitly and class-blind).
- The demos' release points per class (raasw5_mu.json) need a class input; a class-conditioned head is the W5 analogue and is listed as a follow-up
  (6.9), not run, to keep the head's information equal to the servo's.

### 6.3 Head (tiny; fit once on CPU, then frozen)
- Input: rel_xyz (metres, scaled by 1 / .20), only on frames with role = target (m = 1); output 0 elsewhere, as the servo.
- MLP 3 -> 64 -> 64 -> 30 (10 chunk rows x 3), SiLU; output M tanh(u) with M = .9375; centred so h(0) = 0 (h(r) = g(r) - g(0), then clipped to +-M),
  so the correction vanishes at the target as the servo's does.  About 6.4k parameters.
- Data: lol/libero_g8v_raac, the 13,327 held rows of K1-K6 with chunk-consistent targets (raas_gate.py's loader imported unchanged).  Split by demo, 90 / 10,
  by a fixed hash of the episode index.
- Fit: MSE in raw action units over the 10 rows, AdamW lr 1e-3, weight decay 1e-4, full batch, 3000 steps, seed 0, step chosen by validation loss
  (checked every 100).  Fixed now, never tuned after.
- Reported at fit time: validation R² (chunk-consistent) of the head and of the linear-saturating form refit on the train split (check: its K within
  .05 of 2.851); slices h(rel) against the servo along x and y; the head's output magnitude at 2, 5, 10, 20 cm.
- `experiments/raah_fit.py` (new, openpi venv, CPU sbatch, minutes) -> lifelong_ood/policy/raah_head.npz + results/flawed/raah/fit.json.

### 6.4 Model, config, training (the RAA-sL recipe with the servo function swapped)
- `lifelong_ood/policy/raah_model.py` (new): `head_norm_{np,jax,torch}(cfg, state, action_dim)` from the npz (un-normalise the state as servo_norm does,
  apply the head, map into normalised action units by * 2 / (aq99 - aq01 + EPS), shape [b, 10, A]); `Pi0H(RS.Pi0S)` overrides `servo()` with
  head_norm_jax (Pi0S's compute_loss and sample_actions then subtract / add it unchanged); `Pi0ConfigH(RS.Pi0ConfigS)` adds `head_npz: str` and keeps
  **servo_k = 0** (so backbone.py's built-in servo loop adds nothing), create -> Pi0H, load_pytorch -> PI0PytorchH (sample_actions adds head_norm_torch).
  The head weights are module-level constants loaded from the npz, not nnx parameters, so the parameter tree equals RAA-c's and the converter and
  raac_export.py need no change (the executor verifies export with the check below).
- `lifelong_ood/policy/backbone_h.py` (new): `PolicyBackboneH(PolicyBackbone)` overrides `sample` only: x = super().sample(..., full=True), add
  head_norm_torch(cfg, prefix.state)[..., :3] to x[:, :, :3] row by row, return full or real (pattern of backbone_w.py).
- l90c_config.py, add-only: name `pi05_g8v_raah_l_ae` = pi05_g8v_raas_l_ae exactly (data lol/libero_g8v_raac, AE assets, RaacLoader(LIBERO_PARAMS),
  frozen VLM, no EMA) with model Pi0ConfigH(head_npz=<npz path>, **servo_fields(k=0.0)).  Make the edit when no chain link of anyone's run is about to
  restart, then submit the CPU cfgcheck at once (plan_oracle_pi0 step 1 pattern).
- Train with experiments/c_train_x.py --name pi05_g8v_raah_l_ae (c_train_pi05.py not edited): 6,001 steps, batch 32, lr 5e-5 cosine, 1k warmup, seed 42,
  ONE chain of 6 x 4 h afterany on tail-lab (exp tl), save 250 / keep 1000.  Export tag tl6000.

### 6.5 Serving
- Phase A: `experiments/raa_wh.py` = copy of raa_w.py whose backbone builder returns PolicyBackboneH when the config's model is Pi0ConfigH (the only
  change; where_own stays 0).  Phase B: `experiments/wa_sh.py` (copy of wa_s.py, dev rg_comp states) and `experiments/wa_sh5.py` (copy of wa_sw5_5.py,
  5th set) with the same one change.  Job scripts: copies of raaf_roll.sh / raaf_wa.sh (dev) and of the 5th-set scripts of 3.2.
- Serving smoke (2 states, K1 and N4, outside results/): role-1 and role-2 tokens in the server log, finite chunks, and on role-2 queries the logged
  chunk minus the head term equals the residual (assert in the smoke client log).

### 6.6 Checks before the `rvh` row (raah_check.py, a copy of raas_check.py retargeted; GPU .3 h on tail-lab)
1. Identity: a linear-form npz (slope K, clip M, constant over rows) makes head_norm equal servo_norm (max |diff| < 1e-6 on 1,000 dataset states), and
   Pi0H with it gives Pi0S's loss and samples on one batch.
2. head_norm_np == head_norm_jax == head_norm_torch on 1,000 states (< 1e-5); zero where role != 2; zero at rel = 0.
3. Parameter tree equals RAA-c's (no missing or extra keys); the npz sha256 equals the registered one.
4. Smoke (20 steps, exp smoke): step-0 loss finite and <= 2 x .0556; state_proj in the trainable set.

### 6.7 Readout (no selection among variants: one head, one arm)
- Dev (unseen for H when registered): Phase A N / K on the RAA-F dev states (2 GPU-h); Phase B tids 7-10 (.7).  Gate: dev K1-K6 >= .70, else UNTESTABLE
  (no held-out read; report the recipe failure).
- 5th set: Phase A (2) and Phase B tids 7-10 (.7), read with rvw5_read.py --part h.
- Option (1.5): a2 capture + probe (rows inputs) of H for the expert's token share.  a2_probe's Patcher adds no head when servo_k = 0, so this measures
  the expert's own token use only; descriptive.

### 6.8 Registered contrasts (in `rvw5`, fixed before H's training finished and before the set was built)
- **h1 PRIMARY** H - A0 on N1-N6, pooled over the dev set and the 5th set (equal weight, mean of the per-set differences, each set resampled tasks then
  states, as SR 91's pooled routine).  Both H read-outs are unseen at registration; A0's dev numbers were seen, which does not bias a contrast whose
  other arm is new.  SUPPORTED iff lower > 0; REVERSED iff upper < 0.  Holm family.  Per-set values reported.
- **h2** H - RAA-sL on N1-N6, 5th set only (removes the servo's dev-state home advantage): NON-INFERIOR iff lower > -.10; COST iff upper < -.10.
- h3 H - A on N, 5th set: SUPPORTED iff lower > 0 (secondary).
- hK H - RAA-sL on K1-K6, 5th set: NON-INFERIOR iff lower > -.05.
- hF H - RAA-sL and H - A0 on F6_b (5th set): reported (the servo kills the covered-plate cell; does a learned head too?).
- E1 classes for H and the decomposition H vs A0.

### 6.9 What each outcome means (including refutations)
| h1 | h2 | reading |
|---|---|---|
| SUPPORTED | NON-INFERIOR | **separation, not the hand-chosen form**: a learned separate where-module matches the fixed-form servo and beats the distributed route at matched data.  The paper's claim becomes "an architecturally separate owner beats the distributed one at matched learning" (review: 5 -> 6) |
| SUPPORTED | NOT SHOWN / COST | separation helps, the free form is less precise than the saturating linear form: the claim holds for separation, and the form adds precision.  Say both |
| NOT SHOWN, H point within +-.05 of A0 | any | **refutes "separation is what matters" at this recipe**: a learned separate head gives no measurable gain over the expert learning the geometry; RAA-sL's edge is then its fixed form.  The necessity claim narrows to "a fixed-form controller on the exact offset", which the review calls unsurprising |
| REVERSED | | **refutes**: the separate learned head hurts |
| UNTESTABLE (dev K < .70) | | recipe failure; no claim; report losses |
h3 SUPPORTED adds "beats even the dosed in-network route"; h3 NOT SHOWN with h1 SUPPORTED leaves A as an open in-network alternative (state it).
If the head's validation R² is within .02 of the linear form's, the head is nearly the servo and h2 is uninformative about form; h1 still tests separation.
Follow-ups not run: a class-conditioned head (the W5 analogue), an end-to-end-trained head (flow-loss gradients into the head), a second seed of H.

---

## 7. Pre-registration text (format of experiments/prereg/pzdev_pred.txt; probabilities are the designer's, for the orchestrator to adjust)

### 7.1 experiments/prereg/rva1_pred.txt
```
RVA1 (docs/notes/plan_review_atlas.md 2): failure-mode split of held-out-pairing (N1-N6) failures from EXISTING development-state rollouts; no GPU.
Arms: pi0-sL, pi0-A0 (raaf/dev_pzsL, dev_pzA0), RAA-sL seed 1 (raa raasl_*), A (raaf/dev_A), A0 (raaf/dev_A0); free extras RAA-sL s43, A+0, B, W5 (held-out).
Classifier (experiments/rva1_read.py, fixed here): G bound != object; else release query = last query of the last held segment (chan log), d = xy |target - object| there;
H held at the last query and steps = 300; W d >= 10 cm and object within 8 cm of a known-partner target (rva1_geom.py); O d >= 10 cm not W; M d < 10 cm (bins <3, 3-6, 6-10).
Phase B (tracks): final rest distance, classes G, L (> 10 cm), M3-10, M<3, on F6_b and the F7 pool.
Gate: chan-log classes agree with raa_instr per-step classes on RAA-sL rx 24-43 of N1, N4, N6 in >= 90% of 60 episodes and median |d_chan - d_instr| <= 2 cm; else E1-opt re-rolls.
Rules (pi0-A0 vs pi0-sL, paired by task and state, B = 2000, rng 0): P "read but imprecise" iff dM CI lower > 0, dM >= half the total, dW CI upper < .10;
R "routing" iff dW CI lower > 0 and dW >= half the total; MIXED otherwise.  Same rules reported for A0 and A vs RAA-sL.
STATUS: success rates of every arm were seen (pzdev, raaf_sel); no failure class of any trained arm has been computed; RAA's SR 77 split (47 of 51 right object, right target, 3.7-7.9 cm) and raasl_instr categories were seen.
Predictions:
 (e1) gate passes: 85%.
 (e2) pi0-A0 failure shares: M point .55 (range .30-.80); W point .12 (0-.35); G .10; H + O .23.
 (e3) Rule P: 50%; Rule R: 15%; MIXED: 35%.
 (e4) RAA-sL failure M share >= .60: 75%.  A0 and A: M share >= .50: 65% each.
 (e5) bottle pairings carry most of pi0-A0's W (W share bottle > cheese): 60%.
 (e6) F6_b failures of the servo-kept arms (RAA-sL, pi0-sL) are mostly L (> 10 cm): 70%.
Kill: (k1) the join fails (a query count differs from the episode's calls for > 2% of episodes): fix the join, re-register, no read.
```

### 7.2 experiments/prereg/rvh_pred.txt (E4 training row; written after raah_fit, raah_check and the smoke, before the main chain)
```
RVH (docs/notes/plan_review_atlas.md 6): a learned separate where-head H in the servo's slot.  pi05_g8v_raah_l_ae = pi05_g8v_raas_l_ae exactly with the servo function
replaced by a frozen MLP head (raah_head.npz, sha256 in the row) fit by least squares of demo action xyz(t + j), j = 0..9, on rel(t) over the held rows of K1-K6
(the supervision that fixed K = 2.851), servo_k = 0, the expert trained on the residual; data lol/libero_g8v_raac, pi05_libero init, 6,001 steps, batch 32, lr 5e-5, seed 42, one chain.
Principle: if what matters is that a SEPARATE module owns the where, a learned head in that slot does as well as the fixed-form servo and better than the expert learning the geometry (A0).
STATUS: the head's fit metrics (validation R² of head and linear form) were seen at registration; no H checkpoint exists.
Predictions:
 (h0) head validation R² exceeds the linear form's by >= .02: 70% (point .45 vs .36).
 (h0b) dev K1-K6 >= .70: 90%; dev N point .82 (.65-.92).
 (h0c) step-0 loss below RAA-sL's .0556: 70%.
Kill: (k1) raah_check fails; (k2) step-0 loss NaN or > 2 x .0556; (k3) no role-2 token in the serving smoke.
```

### 7.3 experiments/prereg/rvw5_pred.txt (the held-out row; written before the 5th set is built)
```
RVW5 (docs/notes/plan_review_atlas.md 3-6): the 5th fresh state set of raaf_plan 7.1 (N/K k 760-819, F7 860-939, F6 940-1019), unbuilt and reserved "for a later design"
(raaf_plan 6.3 rule 5; no RAA-F route passed, raafc is void and will never be written).  This row takes the set over for E2, E3 and E4.
Arms: pi0-sL, pi0-A0, RAA-sL, A0, A, W5 (where_own 5), noW (RAA-sL checkpoint served with config pi05_g8v_raac_l_ae, where_own 0, R and M kept), H (pi05_g8v_raah_l_ae, after training).
Contrasts (Phase A by raa_read.diff, Phase B by wa_read.bootm; B = 2000, rng 0; Holm over the PRIMARY family {d1', r1, h1}, raw and adjusted reported):
 d1' PRIMARY pi0-A0 - pi0-sL, N: SUPPORTED iff upper < 0.  d4' K NI at -.05.  REPLICATES iff d1' SUPPORTED and d4' NI.
 c0 A0 - RAA-sL, N; cA A - RAA-sL, N: SUPPORTED-for-module iff upper < 0.  d2', d3' reported.
 r1 PRIMARY W5 - noW, N: SUPPORTED iff lower > 0.  r2 W5 - noW, F6_b.  r3 noW - A0, N: DEPENDENT iff upper < 0, EQUIVALENT iff inside +-.10.
 h1 PRIMARY H - A0, N, pooled dev + 5th set: SUPPORTED iff lower > 0.  h2 H - RAA-sL, N, 5th set: NI iff lower > -.10, COST iff upper < -.10.  h3 H - A, N.  hK H - RAA-sL, K, NI at -.05.
 H gate: dev K1-K6 >= .70, else UNTESTABLE.  E1 classes on every arm (secondary).
STATUS: every arm's dev-state success was seen except H; no arm has been rolled on these states; H's design is fixed by the rvh row.
Predictions:
 (q1) d1' point -.18; SUPPORTED 70%; REVERSED 2%.  d4' NI 75%.  REPLICATES 55%.  d1' survives Holm: 60%.
 (q2) pi0-sL N .74 (.60-.86); pi0-A0 N .52 (.35-.70); RAA-sL .87; A0 .70; A .79; W5 .89.
 (q3) c0 SUPPORTED 50%; cA SUPPORTED 25%.
 (q4) noW N point .45 (.15-.75); r1 SUPPORTED 92%; r2 SUPPORTED 80% (W5 F6_b .80, noW .35); r3 DEPENDENT 55%, EQUIVALENT 15%; noW K below W5 by > .05: 70%.
 (q5) h1 point +.12; SUPPORTED 45%; REVERSED 3%.  h2 NI 55%, COST 7%.  h3 SUPPORTED 15%.  hK NI 50%.
 (q6) joint "separation, not form" (h1 SUPPORTED and h2 NI): 30%.  "separation does not help" (h1 point within +-.05 of 0): 20%.
Kill: (k1) the disjointness job finds an equal row; (k2) the noW serving check fails (difference != servo); (k3) any rollout job fails on infrastructure (no read until all cells exist).
```

### 7.4 Ledger rows
Three scripts copied from experiments/prereg/pzdev.py (each through sbatch, CPU):
- `experiments/prereg/rva1.py` (tag rva1): hashes rva1_pred.txt, rva1_geom.py, rva1_read.py, geom_dev.json; asserts no results/flawed/rva1/rva1_*.json.
- `experiments/prereg/rvh.py` (tag rvh): hashes rvh_pred.txt, raah_fit.py, raah_head.npz, results/flawed/raah/fit.json, raah_model.py, backbone_h.py,
  l90c_config.py, raah_check log, the job scripts; asserts jax/pi05_g8v_raah_l_ae/{main,tl} absent.
- `experiments/prereg/rvw5.py` (tag rvw5): hashes rvw5_pred.txt, this plan, fc_n2_w5.py, rg_comp_5.py, wa_sw5_5.py, raa_wh.py, wa_sh5.py, rvw5_read.py,
  the job scripts and raah_head.npz; asserts results/flawed/n2_states_w5, n2_branches_w5.json and results/flawed/rg_comp_5 absent.  Its SEEN field
  records that raaf_plan 7.1's recipe is used verbatim and that the raafc confirmatory row is void.
Rows land as ledger/m3_prereg/m3_prereg__libero_goal__<tag>__s0__<hash>.json.  Commit each row before the jobs it guards (pzdev's 7 s ordering slip,
oracle_pi0_results "deviation added at review", is the thing to avoid).

---

## 8. Order of work, GPU-h and cuts

### 8.1 Order (value per GPU-h, with E4's training launched first for wall clock)
1. Day 0, CPU: rva1_geom, rva1_read selftest, `rva1` row, E1 read and gate (all CPU sbatch).  Same time: raah_fit (CPU), raah_check + smoke
   (tail-lab, .3 h), `rvh` row, H chain on tail-lab (1 of the 4 tail-lab GPUs, about 10.5 h).
2. Write fc_n2_w5.py, rg_comp_5.py, wa_sw5_5.py, raa_wh.py, wa_sh.py, wa_sh5.py, rvw5_read.py, rvw5_nowcheck.py; `rvw5` row; build the set and the
   disjointness job; noW serving check.
3. Overcap, in parallel (no tail-lab use): Phase A of pi0-sL, pi0-A0 (E2 core), RAA-sL, W5, noW, A0, then A; Phase B tids 7-10 of W5, noW, RAA-sL.
4. Read E2 and E3 (rvw5_read.py, parts d and r).
5. After H's step 6000: export, serving smoke, dev Phase A / B, gate, 5th-set Phase A / B, read part h.  Then the paper rows.
Wall clock about 1.5-2 days.  tail-lab peaks at 1-2 GPUs (H chain, checks); every rollout goes to overcap A40 with the usual exclude list; chains and
rollouts are requeue-safe.  Every waiter is capped with `timeout` and stopped before the session ends.

### 8.2 GPU-h
| item | GPU-h | experiment |
|---|---|---|
| E1 geometry + classifier (CPU) | 0 | E1 |
| 5th-set build + disjointness | 2.5 | shared |
| pi0-sL, pi0-A0 Phase A | 4 | E2 core |
| RAA-sL Phase A | 2 | E2 / E3 / E4 |
| A0 Phase A | 2 | E2 / E4 |
| A Phase A | 2 | E2 reference |
| W5, noW Phase A | 4 | E3 |
| W5, noW, RAA-sL Phase B tids 7-10 | 2.5 | E3 |
| noW serving check | .2 | E3 |
| head fit (CPU), raah_check + smoke | .5 | E4 |
| H training, one chain | 9.5 | E4 |
| H serving smoke | .2 | E4 |
| H dev Phase A + Phase B 7-10 | 2.7 | E4 |
| H 5th-set Phase A + Phase B 7-10 | 2.7 | E4 |
| **core total** | **about 35** | |
| options: E1-opt re-rolls (only if the gate fails) 2.5; pi0 arms F6 cells on the 5th set 1.4; RAA-sL seed 2 on the 5th set (seed spread on fresh states) 2; H probe 1.5 | +7.4 | |
| option: H seed 2 (train 9.5 + 5th-set Phase A 2), registered as a replication like raaslS2 | +11.5 | |
Core plus the small options is about 42 GPU-h, inside the 50 guide.  H seed 2 takes it to about 54 and needs approval.

### 8.3 What to cut first (decide BEFORE the rvw5 row is written: cuts change the registered contrasts)
1. H seed 2, then the H probe, RAA-sL seed 2 on the 5th set, the pi0 F6 cells (options; nothing registered depends on them).
2. A on the 5th set (cA becomes dev-only; -2).
3. H's dev Phase B and the 5th-set K1t / K2t cells (keep F6_b and F6_a; -1.5).
4. H's dev Phase A (-2): then h1 is read on the 5th set alone (half the power; write it that way in the row).
Never cut: E1; pi0-sL and pi0-A0 on the 5th set; W5 and noW Phase A and F6_b; H's training; H, A0 and RAA-sL Phase A on the 5th set.

---

## 9. How the four results bear on "a non-distributed architecture is necessary"

| result | supports the proposition | weakens or refutes it |
|---|---|---|
| E1 | Rule R (in-network failures are wrong-target routing although the token is read) | Rule P makes it a precision claim: the separate module owns the exact release point.  Still a module claim, but the atlas label and "missing function" wording change |
| E2 | d1' and c0 SUPPORTED on fresh states, Holm-surviving | d1' NOT SHOWN near 0: "read but not in control" was a development-state effect |
| E3 | r1 SUPPORTED, r3 EQUIVALENT: removing the module returns the failure at the native network's level | r3 DEPENDENT: the drop is engineering dependence only; r1 NOT SHOWN: the servo is not what carries N |
| E4 | h1 SUPPORTED with h2 NI: separation, not form | h1 null: the gain is the fixed form; h1 REVERSED: separation hurts |
Strongest joint outcome: E1 R or MIXED, d1' SUPPORTED, r1 with r3 EQUIVALENT, h1 with h2 NI.  Then the paper can claim, scoped to this recipe (frozen VLM,
6k-step expert fine-tune, LIBERO-Goal, privileged geometry for the module): a separate module that owns the where, learned or fixed-form, beats letting
the action expert learn the same geometry from the same data, on states nobody tuned on.  Weakest: E1 P, d1' null, h1 null.  Then the honest paper is an
atlas plus "a fixed-form controller on exact geometry is sufficient", and the necessity framing is dropped.
None of these answers the review's field-level counterexamples (CRAFT, ACT-VLA, task scrubbing, CofactVLA) or pi0's overwritten EE state (O2(d), about
32 GPU-h); both stay as scoped limits in the text.

---

## 10. Risks

1. **The 5th set is the last sealed set.**  Using it here leaves none.  If the orchestrator rules it must stay reserved, build a 6th set with the same
   recipe and fresh ranges: N / K k 820-879 (branch base 13000000 + 1000 k stays below 14,000,000), F7 candidates 1020-1099, F6 1100-1179, the build's
   own seed assertion against every earlier set (shift by +1000 on any collision) and the disjointness job.  Cost unchanged.
2. **noW is circular by construction** (a residual-trained expert loses its addend).  r3 against A0 measures how much; never quote r1 alone as necessity.
3. **The learned head may be almost the servo** (R² within .02): then h2 is uninformative about form; h1 still tests separation.  The head is class-blind
   by design (the servo's information); a class-conditioned head is a follow-up.
4. **Power.**  60 states x 6 tasks gives CIs of about +-.10 to .15 on N (task resampling dominates).  h1 is pooled over two sets for that reason; d1' and
   r1 have large expected effects.  Seed variance (RAA-sL: .056 on N between seeds) is not in any CI; H is one seed unless the option runs.
5. **E1 resolution** (5-step queries, whole cm, rigid-offset truth, the client's held detector).  The validity gate against raa_instr guards it; E1-opt is
   the fallback.  The plate can be pushed, so partner geometry from the init state is approximate for plate targets.
6. **Reading order.**  E2 / E3 numbers will be seen before H's held-out rollouts.  H's design and contrasts are frozen in rvh / rvw5 before either, so
   this does not bias h1-h3; state it in the paper.
7. **Shared-code edits.**  Only l90c_config.py is edited (add-only, cfgcheck at once).  backbone.py, raas_model.py, raa.py, raa_w.py, wa_s.py, wa_sw5z.py
   stay byte-identical (earlier rows hash them); everything new is a copy or a new file.
8. **Export of Pi0ConfigH** relies on raac_export.py treating it as the Pi0ConfigC family; raah_check item 3 and the serving smoke catch a mismatch.
9. **Overcap preemption and the 4-GPU tail-lab cap**: only the H chain and short checks use tail-lab.

## 11. Hard rules for the executor
No python on the login node (sbatch / srun only; login node = edits, git, sbatch / squeue, cat / grep of logs).  Every job with the exclude list of
raaf_plan 5 (sky1 and the throttled nodes) and explicit -p / -A.  No training or rollout before its prereg row is committed.  Never open
results/l90c/split_sealed.json or key files.  Outputs opened with "x" (no overwrite).  No recursive grep / find over the repo or data dirs.  Write
results to a new note (docs/notes/review_atlas_results.md) and label every prediction as written.

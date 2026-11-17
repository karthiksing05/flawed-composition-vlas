# Predictor targets: measured new-pairing success per policy (PLAN_PAPER §2c)

Read-only extraction. Numbers copied as written in the source doc. Nothing computed. Predictions in brackets in the sources are not copied. No jobs, no python, not committed.

Sources: `docs/SCAFFOLD_RESULTS.md` (SR, sections 11-72), `docs/ARCH_FAILURES.md`, `docs/ARCHITECTURE_PROPOSAL.md`. `docs/LADDER_RESULTS.md` has no R0/R1/R2 rungs; it is the L0-L4 PLAN_PARTS ladder, so it adds no rows.

## Scene and state-set key

- **Scene 1** = LIBERO-goal, new pairings N1-N6, trained tasks K1-K4 and K6 (K1-K6 for the §34-§37 ladder). Familiar layout unless noted.
- **Scene 2** = LIBERO-10 LIVING_ROOM_SCENE5, new pairings N1 and N2, trained tasks K1 and K2. Layout O unless noted.
- **Scene 1 state sets**
  - `main` = states 4-23, 20 states (SR §44, §46, §34-§37; §37 calls these dev states).
  - `fresh` = k 36-55, 20 states (SR §47, §48, §52).
  - `new k56-115` = 60 states (SR §49).
  - `new k116-155` = 40 states (SR §55).
  - `new k156-195` = 40 states (SR §56).
  - `new k236-275` = 40 states (SR §65).
  - `new k500-539` = 40 states (SR §71).
- **Scene 2 state sets**
  - `k4-63` = 60 main states (SR §50, §54).
  - `k4-43` = 40 states (SR §57; §59-§60 main states).
  - `new, seed 835000/845000` = 40 new states per task (SR §59, §60).
  - `new, seeds 890000/920000/950000/960000/970000/980000/990000/1000000` = 60 new states per task, one set per section (SR §62-§64, §66-§69, §72). Seed bases are listed in the section column.

## Table A: scene 1 (LIBERO-goal N1-N6)

| policy | scene | state set (n) | new-pairing success [CI] | trained K success | SR § | results JSON |
|---|---|---|---|---|---|---|
| frozen pi0.5 intact | 1 | main 4-23 (20) | .500 (no CI printed) | K1-K4,K6 1.00 | §46 | results/flawed/tv_frozen_read.json |
| frozen GR00T intact | 1 | main 4-23 (20) | .167 (no CI printed) | K1-K4,K6 .98 | §46 | results/flawed/tv_frozen_read.json |
| frozen pi0.5 intact | 1 | fresh k36-55 (20) | .450 [.258, .642] | K1-K4,K6 .99 | §48 (§47 gives .450 without CI) | results/flawed/tv_fresh_read.json, tv_attr_read.json |
| frozen GR00T intact | 1 | new k56-115 (60) | .219 [.025, .456] | K1-K4,K6 .99 | §49 | results/flawed/tv_g_read.json |
| frozen pi0.5 intact | 1 | new k116-155 (40) | .454 [.242, .654] | K1-K4,K6 .990 | §55 | results/flawed/tvpb_rg_read.json |
| frozen GR00T intact | 1 | new k156-195 (40) | .179 [.025, .371] | K1-K4,K6 .975 | §56 | results/flawed/groot_box_read.json |
| frozen pi0 intact | 1 | new k236-275 (40) | .021 [.000, .058] | K1 .98, K2 .95, K3 .98, K4 1.00, K6 .98 | §65 | results/flawed/s1_cross_read.json |
| frozen SmolVLA intact | 1 | new k236-275 (40) | .021 [.000, .075] | K1 .98, K2 1.00, K3 1.00, K4 1.00, K6 .75 | §65 | results/flawed/s1_cross_read.json |
| frozen GR00T intact | 1 | new k500-539 (40) | .212 [.017, .475] | K1 .975, K2 1.00, K3 .950, K4 1.00, K6 1.00 | §71 | results/flawed/groot_rg_read.json |
| tv (role views), frozen pi0.5 | 1 | main 4-23 (20) | .725 (no CI) | K1-K4,K6 .96 | §46 | tv_frozen_read.json |
| tvp (views + phrases), frozen pi0.5 | 1 | main 4-23 (20) | .925 (no CI) | K1-K4,K6 .99 | §46 | tv_frozen_read.json |
| w (whole-wrist grey), frozen pi0.5 | 1 | main 4-23 (20) | .542 (no CI) | K1-K4,K6 .87 | §46 | tv_frozen_read.json |
| tv, frozen GR00T | 1 | main 4-23 (20) | .400 (no CI) | K1-K4,K6 .95 | §46 | tv_frozen_read.json |
| tvp, frozen GR00T | 1 | main 4-23 (20) | .167 (no CI) | K1-K4,K6 .67 | §46 | tv_frozen_read.json |
| w, frozen GR00T | 1 | main 4-23 (20) | .158 (no CI) | K1-K4,K6 .72 | §46 | tv_frozen_read.json |
| tv, frozen pi0.5 | 1 | fresh k36-55 (20) | .700 [.475, .900] | K1-K4,K6 .98 | §48 | tv_attr_read.json (§47: tv_fresh_read.json) |
| tvp, frozen pi0.5 | 1 | fresh k36-55 (20) | .817 [.583, 1.00] | K1-K4,K6 .98 | §48 | tv_attr_read.json (§47: tv_fresh_read.json) |
| p (phrases only), frozen pi0.5 | 1 | fresh k36-55 (20) | .475 [.225, .733] | K1-K4,K6 .93 | §48 | tv_attr_read.json |
| wp (wrist grey + phrases), frozen pi0.5 | 1 | fresh k36-55 (20) | .375 [.108, .675] | K1-K4,K6 .76 | §48 | tv_attr_read.json |
| tv, frozen GR00T | 1 | fresh k36-55 (20) | .367 (no CI printed) | K1-K4,K6 .99 | §47 | tv_fresh_read.json |
| tv, frozen GR00T | 1 | new k56-115 (60) | .406 [.092, .731] | K1-K4,K6 .97 | §49 | tv_g_read.json |
| w, frozen GR00T | 1 | new k56-115 (60) | .169 [.061, .306] | K1-K4,K6 .65 | §49 | tv_g_read.json |
| tv, frozen GR00T | 1 | new k156-195 (40) | .379 [.050, .717] | K1-K4,K6 .975 | §56 | groot_box_read.json |
| tvb (tv + held-object box), frozen GR00T | 1 | new k156-195 (40) | .383 [.071, .708] | K1-K4,K6 .905 | §56 | groot_box_read.json |
| tvg (tv + cheese finger qpos), frozen GR00T | 1 | new k156-195 (40) | .392 [.079, .713] | K1-K4,K6 .980 | §56 | groot_box_read.json |
| tvbg, frozen GR00T | 1 | new k156-195 (40) | .383 [.062, .708] | K1-K4,K6 .885 | §56 | groot_box_read.json |
| tvpb (views + phrases + box), frozen pi0.5 | 1 | new k116-155 (40) | .938 [.850, .996] | K1-K4,K6 .980 | §55 | tvpb_rg_read.json |
| tvp, frozen pi0.5 | 1 | new k116-155 (40) | .838 [.637, .983] | K1-K4,K6 .980 | §55 | tvpb_rg_read.json |
| tv, frozen pi0 | 1 | new k236-275 (40) | .204 [.037, .450] | K1 1.00, K2 .98, K3 .93, K4 1.00, K6 1.00 | §65 | s1_cross_read.json |
| tvpb, frozen pi0 | 1 | new k236-275 (40) | .292 [.096, .517] | K1 .98, K2 .45, K3 .55, K4 .33, K6 .15 | §65 | s1_cross_read.json |
| tv, frozen SmolVLA | 1 | new k236-275 (40) | .021 [.000, .075] | K1 .85, K2 .98, K3 .93, K4 .83, K6 .08 | §65 | s1_cross_read.json |
| tvpb, frozen SmolVLA | 1 | new k236-275 (40) | .108 [.000, .350] | K1 .03, K2 .00, K3 .00, K4 .00, K6 .03 | §65 | s1_cross_read.json |
| rgp (ReGuide, perceived poses), frozen pi0.5 | 1 | fresh k36-55 (20) | .883 [.733, .992] | K1-K4,K6 .99 | §52 | results/flawed/rg_read.json |
| rgo (ReGuide, simulator poses), frozen pi0.5 | 1 | fresh k36-55 (20) | .933 [.825, 1.00] | K1-K4,K6 1.00 | §52 | rg_read.json |
| rgp, frozen pi0.5 | 1 | new k116-155 (40) | .929 [.854, .988] | K1-K4,K6 .995 | §55 | tvpb_rg_read.json |
| rgo, frozen pi0.5 | 1 | new k116-155 (40) | .933 [.825, 1.00] | K1-K4,K6 .990 | §55 | tvpb_rg_read.json |
| rgp, frozen GR00T | 1 | new k500-539 (40) | .838 [.567, .992] | K1 1.00, K2 1.00, K3 .950, K4 1.00, K6 .975 | §71 | groot_rg_read.json |
| rgo, frozen GR00T | 1 | new k500-539 (40) | .879 [.679, .992] | K1 1.00, K2 1.00, K3 .950, K4 1.00, K6 1.00 | §71 | groot_rg_read.json |
| rgp, frozen pi0 | 1 | new k236-275 (40) | .608 [.333, .858] | K1 1.00, K2 1.00, K3 .98, K4 1.00, K6 1.00 | §65 | s1_cross_read.json |
| rgp, frozen SmolVLA | 1 | new k236-275 (40) | .604 [.304, .863] | K1 .98, K2 1.00, K3 .98, K4 1.00, K6 .93 | §65 | s1_cross_read.json |
| H (sim-state hand-over to U), frozen pi0.5 | 1 | main 4-23 (20) | .958 (no CI printed) | not extracted | §44 | results/flawed/sc_hp_read.json |
| H, frozen GR00T | 1 | main 4-23 (20) | .925 (no CI printed) | not extracted | §44 | sc_hp_read.json |
| HP (H with learned perception, no sim state), frozen pi0.5 | 1 | main 4-23 (20) | .850 (no CI printed; HP − intact +.350 [+.017, +.675]) | HP − intact −.010 [−.060, .000] | §44 | sc_hp_read.json |
| HP, frozen GR00T | 1 | main 4-23 (20) | .842 (no CI printed; HP − intact +.675 [+.417, +.917]) | HP − intact −.010 [−.080, +.040] | §44 | sc_hp_read.json |
| U (parts policy, simulator poses) | 1 | familiar, states as sc_t4 (count not restated) | .894 (§11; U − L4 −.033 [−.100, .028]) | known familiar oracle .878 (§38) | §11, §38 | results/flawed/sc_t4_read.json, sc_pu_read.json |
| U, camera-perceived poses (percep) | 1 | sc_pu states (count not restated) | .883 (percep − oracle −.014 [−.061, +.025]) | known familiar percep .853 | §38 | results/flawed/sc_pu_read.json |
| U, learned segmenter masks | 1 | sc_seg_pu states | .881 (§40 table; learned − percep −.003 [−.042, +.039]) | known familiar learned .853 | §40 | results/flawed/sc_seg_pu_read.json |

### Table A1: scene 1 ladder (frozen-VLM retrain, §34/§37 recipe, one seed)

All rungs share one recipe: pi0.5 from `pi05_base`, frozen VLM, trainable action expert, 6,000 steps. State set: dev 4-23, familiar, 20 per task. Known K = K1-K6 for this ladder.

| policy | scene | state set (n) | new-pairing success [CI] | trained K1-K6 | SR § | results JSON |
|---|---|---|---|---|---|---|
| R0 (whole instructions; ckpt `pi05_g8v_plain_ae`) | 1 | dev 4-23, familiar (20) | .025 (no CI) | .85 | §37.1 | results/flawed/sc_t37_read_r012.json |
| R1 (role phrases; `pi05_g8v_rsup_ae`) | 1 | dev 4-23, familiar (20) | .225 (no CI; R1 − R0 +.200 [+.025, +.433]) | .758 | §37.1 | sc_t37_read_r012.json |
| R2 (role views, simulator masks; `pi05_g8v_rview2_ae`) | 1 | dev 4-23, familiar (20) | .425 (R2 − R0 +.400 [+.167, +.625]) | .758 | §37.1 | sc_t37_read_r012.json |
| R2b (boxes; `pi05_g8v_rbox_ae`) | 1 | dev 4-23, familiar (20) | .150 (R2b − R2 −.275 [−.458, −.108]) | .683 | §37.2 | results/flawed/sc_t37_read.json |
| R2w (whole-wrist grey filled at test; `pi05_g8v_rwrist_ae`) | 1 | dev 4-23, familiar (20) | .242 (no CI) | .617 | §37.2 (T37w, §41) | results/flawed/sc_t37_read_w*.json |
| R2w0 (same ckpt as R2w, unfilled at test) | 1 | dev 4-23, familiar (20) | .175 (no CI) | .308 | §37.2 | sc_t37_read_w*.json |
| r2l (R2 with learned masks) | 1 | dev 4-23, familiar (20) | .417 (r2l − r2 −.008 [−.133, +.117]; r2l − r0 +.392 [+.117, +.675]) | not printed | §43.1 | results/flawed/sc_t37_read_l.json |

Not measured in the ladder: §34 R0 and R1 on the published frames. The §34 read was paused and its verdict will not be read (SR §34). Do not use it as a row.

### Table A2: F7 bowl family (scene not named in §45; §70 names libero_goal)

These are not N1-N6. Each cell is the referent success on a new relational or negated instruction. Success columns are R (relational) and F7R (role view + resolver). Main states 4-23, 20 per cell, from SR §45.1-§45.3.

| policy | scene | state set (n) | R relational | F7R (resolver + view) | U (unique name) | SR § | results JSON |
|---|---|---|---|---|---|---|---|
| pi0.5 SP | other (F7 bowl) | main 4-23 (20) | .40 | 1.00 (delta +.60 [+.525, +.70], FIXED) | 1.00 | §45.1, §45.2 | results/flawed/fx_read_main.json, fx_r_read_main.json |
| pi0.5 NG | other (F7 bowl) | main 4-23 (20) | .20 | .775 (delta +.575 [+.425, +.70], FIXED) | .825 | §45.2 | fx_r_read_main.json |
| GR00T SP | other (F7 bowl) | main 4-23 (20) | .30 | .85 (delta +.55, FIXED) | .825 | §45.2 | fx_r_read_main.json |
| GR00T NG | other (F7 bowl) | main 4-23 (20) | .475 | .50 (delta +.025, NOT FIXED) | .50 | §45.2 | fx_r_read_main.json |
| pi0 SP | other (F7 bowl) | main 4-23 (20) | .225 | .425 (delta +.20 [+.025, +.375], PARTIAL) | .65 [.52, .80] | §45.3 | results/flawed/fx_more_r_read_main.json |
| pi0 NG | other (F7 bowl) | main 4-23 (20) | .20 | .50 (delta +.30, PARTIAL) | .50 | §45.3 | fx_more_r_read_main.json |
| X-VLA SP | other (F7 bowl) | main 4-23 (20) | .00 | .125 (NO POWER) | .10 [.025, .20] | §45.3 | fx_more_r_read_main.json |
| X-VLA NG | other (F7 bowl) | main 4-23 (20) | .00 | .50 (delta +.50, PARTIAL) | .55 | §45.3 | fx_more_r_read_main.json |
| SmolVLA SP | other (F7 bowl) | main 4-23 (20) | .05 [.00, .125] | .025 (NO POWER) | .05 [.00, .125] | §45.3 | fx_more_r_read_main.json |
| SmolVLA NG | other (F7 bowl) | main 4-23 (20) | .025 | .50 (delta +.475, PARTIAL) | .50 | §45.3 | fx_more_r_read_main.json |

§45.4 (f7rA, paint-all) has its own rows in `results/flawed/fx_ra_read_main.json`, not copied here.

### Table A3: scene 1, RC composite (§70, libero_goal ledger, F7 and F6 families, NOT N1-N6)

Success numbers as printed. Bracketed predictions are dropped. 40 states per F7 family, 60 per F6 cell and K1t.

| policy | scene | state set (n) | SP (relL + relR) | NG (pos + neg) | F6_b (displaced plate covered) | K (K1t, K2t mean) | SR § | results JSON |
|---|---|---|---|---|---|---|---|---|
| I (intact), pi0.5 | other (libero_goal F7/F6) | rg_comp states (40 F7, 60 F6) | .425 | .275 | .517 | 1.000 | §70 | results/flawed/rg_comp_read.json |
| rgp (ReGuide), pi0.5 | same | same | .425 | .287 | .633 | .992 | §70 | rg_comp_read.json |
| RC (rc), pi0.5 | same | same | .975 | .950 | .800 | .992 | §70 | rg_comp_read.json |
| f7r (resolver only, I executes), pi0.5 | same | same | .975 | .863 | (not run) | (not run) | §70 | rg_comp_read.json |
| mem (memory slot only), pi0.5 | same | same | (not run) | (not run) | .817 | (not run) | §70 | rg_comp_read.json |
| I, pi0 | same | same | .225 | .287 | .000 | .967 | §70 | rg_comp_read.json |
| rgp, pi0 | same | same | .312 | .338 | .183 | .938 | §70 | rg_comp_read.json |
| RC, pi0 | same | same | .688 | .500 | .217 | .929 | §70 | rg_comp_read.json |
| f7r, pi0 | same | same | .675 | .500 | (not run) | (not run) | §70 | rg_comp_read.json |
| mem, pi0 | same | same | (not run) | (not run) | .000 | (not run) | §70 | rg_comp_read.json |

§70 note: pi0 NG and pi0 F6_b are NOT TESTABLE under the registered gates. pi0 pos is 0.000 under every arm.

## Table B: scene 2 (LIBERO-10 LIVING_ROOM_SCENE5, N1 and N2)

Each row gives N1, N2 and the joint N1+N2 where the source gives it. K = K1, K2.

| policy | scene | state set (n) | N1 | N2 | N1+N2 [CI] | K1 | K2 | SR § | results JSON |
|---|---|---|---|---|---|---|---|---|---|
| frozen pi0.5 intact | 2 | k4-63 (60) | .000 | .000 | .000 | 1.00 | .983 | §50 (§54 table) | results/flawed/s2_read.json |
| pi0.5 tv | 2 | k4-63 (60) | .000 | .000 | .000 [.000, .000] | .983 | .933 | §50, §54 | s2_read.json, s2fix_read.json |
| pi0.5 tvp | 2 | k4-63 (60) | .000 | .000 | .000 [.000, .000] | .983 | .767 | §50, §54 | s2_read.json, s2fix_read.json |
| pi0.5 htvp (hand-over + tvp, PRIMARY) | 2 | k4-63 (60) | .450 | .450 | .450 [.333, .567] | 1.00 | .550 | §54 | results/flawed/s2fix_read.json |
| pi0.5 hI (hand-over + intact) | 2 | k4-63 (60) | .550 | .000 | .275 [.000, .633] | 1.00 | .000 | §54 | s2fix_read.json |
| frozen GR00T intact | 2 | k4-63 (60) | .000 | .000 | .000 | .600 | .417 | §50 | s2_read.json |
| GR00T tv | 2 | k4-63 (60) | .000 | .000 | .000 [.000, .000] | .917 | .000 | §50, §54 | s2_read.json, s2fix_read.json |
| GR00T hI | 2 | k4-63 (60) | .467 | .000 | .233 [.000, .534] | .617 | .017 | §54 | s2fix_read.json |
| frozen pi0 intact | 2 | k4-63 (60) | .000 | .000 | .000 | .867 | .75 | §50 (§57 quotes K1 .825) | s2_read.json |

Scene 2, hand-over and front-end arms on new states (layout O, 60 per task unless noted):

| policy | scene | state set (n) | N1 | N2 | N1+N2 [CI] | K1 | K2 | K1+K2 | SR § | results JSON |
|---|---|---|---|---|---|---|---|---|---|---|
| pi0.5 I | 2 | new, seeds 890000 (60) | .000 | .000 | .000 | 1.00 | 1.00 | 1.00 | §62 | results/flawed/s2fix2_read.json |
| pi0.5 H0 (= §54 hI) | 2 | new, seeds 890000 (60) | .600 | .000 | .300 | 1.00 | .017 | .508 | §62 | s2fix2_read.json |
| pi0.5 H (step-1 fix) | 2 | new, seeds 890000 (60) | .600 | .283 | .442 | 1.00 | .983 | .992 | §62 | s2fix2_read.json |
| pi0.5 G (gated hand-over) | 2 | new, seeds 890000 (60) | .600 | .283 | .442 [+.367, +.517] vs I | 1.00 | 1.00 | 1.00 | §62 | s2fix2_read.json |
| pi0.5 I | 2 | new, seeds 920000 (60) | .000 | .000 | .000 | .983 | 1.00 | .992 | §63 | results/flawed/s2fix3_read.json |
| pi0.5 H | 2 | new, seeds 920000 (60) | .617 | .367 | .492 | .983 | 1.00 | .992 | §63 | s2fix3_read.json |
| pi0.5 Hp (learned perception) | 2 | new, seeds 920000 (60) | .600 | .300 | .450 | .983 | 1.00 | .992 | §63 | s2fix3_read.json |
| pi0 I | 2 | new, seeds 920000 (60) | .000 | .000 | .000 | .900 | .683 | .792 | §63 | s2fix3_read.json |
| pi0 H | 2 | new, seeds 920000 (60) | .500 | .033 | .267 | .983 | .717 | .850 | §63 | s2fix3_read.json |
| GR00T I | 2 | new, seeds 920000 (60) | .000 | .000 | .000 | .533 | .433 | .483 | §63 | s2fix3_read.json |
| GR00T H | 2 | new, seeds 920000 (60) | .467 | .017 | .242 | .633 | .517 | .575 | §63 | s2fix3_read.json |
| pi0.5 I | 2 | new, seeds 950000 (60) | .000 | .000 | .000 | 1.00 | 1.00 | 1.00 | §64 | results/flawed/s2fix4_read.json |
| pi0.5 Hp | 2 | new, seeds 950000 (60) | .533 | .433 | .483 | 1.00 | 1.00 | 1.00 | §64 | s2fix4_read.json |
| pi0.5 Hb (put-back) | 2 | new, seeds 950000 (60) | .533 | .167 | .350 | 1.00 | .800 | .900 | §64 | s2fix4_read.json |
| pi0 I | 2 | new, seeds 950000 (60) | .000 | .000 | .000 | .933 | .733 | .833 | §64 | s2fix4_read.json |
| pi0 Hp | 2 | new, seeds 950000 (60) | .367 | .100 | .233 | .933 | .733 | .833 | §64 | s2fix4_read.json |
| pi0 Hb | 2 | new, seeds 950000 (60) | .367 | .000 | .183 | .933 | .050 | .492 | §64 | s2fix4_read.json |
| GR00T I | 2 | new, seeds 950000 (60) | .000 | .000 | .000 | .667 | .400 | .533 | §64 | s2fix4_read.json |
| GR00T Hp | 2 | new, seeds 950000 (60) | .467 | .017 | .242 | .717 | .450 | .583 | §64 | s2fix4_read.json |
| GR00T Hb | 2 | new, seeds 950000 (60) | .467 | .033 | .250 | .717 | .600 | .658 | §64 | s2fix4_read.json |
| pi0.5 I | 2 | new, seeds 960000 (60) | .000 | .000 | .000 | .983 | 1.00 | .992 | §66 | results/flawed/s2fix5_read.json |
| pi0.5 Hp | 2 | new, seeds 960000 (60) | .450 | .217 | .333 | 1.00 | 1.00 | 1.00 | §66 | s2fix5_read.json |
| pi0.5 Hv (views + phrases + Hp) | 2 | new, seeds 960000 (60) | .533 | .483 | .508 | 1.00 | .767 | .883 | §66 | s2fix5_read.json |
| pi0.5 Hv0 (views + Hp) | 2 | new, seeds 960000 (60) | .450 | .567 | .508 | .967 | .883 | .925 | §66 | s2fix5_read.json |
| pi0 I | 2 | new, seeds 960000 (60) | .000 | .000 | .000 | .900 | .717 | .808 | §66 | s2fix5_read.json |
| pi0 Hp | 2 | new, seeds 960000 (60) | .450 | .083 | .267 | .917 | .750 | .833 | §66 | s2fix5_read.json |
| pi0 Hv | 2 | new, seeds 960000 (60) | .550 | .083 | .317 | .833 | .067 | .450 | §66 | s2fix5_read.json |
| pi0.5 I | 2 | new, seeds 970000 (60) | .000 | .000 | .000 | 1.00 | 1.00 | 1.00 | §67 | results/flawed/s2fix6/hist.json, s2fix6_read.json |
| pi0.5 Hp | 2 | new, seeds 970000 (60) | .450 | .250 | .350 | 1.00 | 1.00 | 1.00 | §67 | s2fix6_read.json |
| pi0.5 Hc (clear-away) | 2 | new, seeds 970000 (60) | .450 | .100 | .275 | 1.00 | .967 | .983 | §67 | s2fix6_read.json |
| pi0 I | 2 | new, seeds 970000 (60) | .000 | .000 | .000 | .933 | .800 | .867 | §67 | s2fix6_read.json |
| pi0 Hp | 2 | new, seeds 970000 (60) | .517 | .133 | .325 | .950 | .783 | .867 | §67 | s2fix6_read.json |
| pi0 Hc | 2 | new, seeds 970000 (60) | .533 | .033 | .283 | .950 | .833 | .892 | §67 | s2fix6_read.json |
| pi0.5 Hp | 2 | new, seeds 980000 (60) | .483 | .233 | .358 | .983 | 1.00 | .992 | §68 | results/flawed/s2rg/rg_stats_s2.json, s2rg_read.json |
| pi0.5 rgp (ReGuide, perceived) | 2 | new, seeds 980000 (60) | .417 | 1.00 | .708 | 1.00 | 1.00 | 1.00 | §68 | s2rg_read.json |
| pi0.5 rgo (ReGuide, simulator) | 2 | new, seeds 980000 (60) | .833 | 1.00 | .917 | 1.00 | 1.00 | 1.00 | §68 | s2rg_read.json |
| pi0 I | 2 | new, seeds 980000 (60) | .000 | .000 | .000 | .933 | .800 | .867 | §68 | s2rg_read.json |
| pi0 Hp | 2 | new, seeds 980000 (60) | .550 | .167 | .358 | .917 | .867 | .892 | §68 | s2rg_read.json |
| pi0 rgp | 2 | new, seeds 980000 (60) | .433 | .967 | .700 | 1.00 | .967 | .983 | §68 | s2rg_read.json |
| pi0 rgo | 2 | new, seeds 980000 (60) | .717 | .983 | .850 | .983 | .983 | .983 | §68 | s2rg_read.json |
| pi0.5 I | 2 | new, seeds 990000 (k0-59, 60) | .000 | .000 | .000 | 1.00 | .983 | .992 | §69 | results/flawed/s2rgt/pcheck.json, s2rgt_read.json |
| pi0.5 rgp (labelled §68 replication) | 2 | new, seeds 990000 (60) | .417 | .983 | .700 | 1.00 | .983 | .992 | §69 | s2rgt_read.json |
| pi0.5 rgt (tracking, rendered depth) | 2 | new, seeds 990000 (60) | .550 | .400 | .475 | 1.00 | .983 | .992 | §69 | s2rgt_read.json |
| pi0.5 rgtp (tracking, table-plane depth) | 2 | new, seeds 990000 (60) | .433 | .967 | .700 | 1.00 | .983 | .992 | §69 | s2rgt_read.json |
| pi0 I | 2 | new, seeds 990000 (60) | .000 | .000 | .000 | 1.00 | .833 | .917 | §69 | s2rgt_read.json |
| pi0 rgp | 2 | new, seeds 990000 (60) | .333 | .933 | .633 | .967 | .883 | .925 | §69 | s2rgt_read.json |
| pi0 rgt | 2 | new, seeds 990000 (60) | .383 | .400 | .392 | .983 | .883 | .933 | §69 | s2rgt_read.json |
| pi0 rgtp | 2 | new, seeds 990000 (60) | .283 | .933 | .608 [.542, .675] | .967 | .833 | .900 | §69 | s2rgt_read.json |
| pi0.5 I | 2 | new, seeds 1000000 (k0-59, 60) | .000 | .000 | .000 | .983 | .983 | .983 | §72 | results/flawed/s2rgs_read.json |
| pi0.5 rgp (perception everywhere) | 2 | new, seeds 1000000 (60) | .500 | 1.00 | .750 | 1.00 | .983 | .992 | §72 | s2rgs_read.json |
| pi0.5 rgo (simulator everywhere) | 2 | new, seeds 1000000 (60) | .900 | 1.00 | .950 | 1.00 | 1.00 | 1.00 | §72 | s2rgs_read.json |

Scene 2 cue and manipulated-start rows (not the same state set as Table B; each row is a manipulated start):

| policy | scene | state set (n) | K1 | K2 | N1 | N2 | SR § | results JSON |
|---|---|---|---|---|---|---|---|---|
| pi0.5 I, §50 unmanipulated, k4-43 | 2 | k4-43 (40) | 1.00 | 1.00 | .000 | .000 | §57 | results/flawed/s2cue_read.json |
| pi0.5 tv, §50 unmanipulated, k4-43 | 2 | k4-43 (40) | .975 | .90 | .000 | .000 | §57 | s2cue_read.json |
| pi0.5 tv, swapped starts (P) | 2 | k4-43 (40) | .000 | .000 | .975 | 1.00 | §57 | s2cue_read.json |
| pi0.5 I, swapped starts (P) | 2 | k4-43 (40) | .000 | .000 | .950 | 1.00 | §57 | s2cue_read.json |
| pi0.5 C I (mugs on no side) | 2 | §58 manipulated (40) | .025 | .600 | .050 | .000 | §58 | results/flawed/s2cue2_read.json |
| pi0.5 C tv | 2 | §58 manipulated (40) | .125 | 1.00 | .375 | .000 | §58 | s2cue2_read.json |
| pi0.5 C′ I | 2 | §58 manipulated (40) | .250 | .350 | .075 | .000 | §58 | s2cue2_read.json |
| pi0.5 C′ tv | 2 | §58 manipulated (40) | .175 | .175 | .775 | .000 | §58 | s2cue2_read.json |
| pi0.5 M I | 2 | §58 manipulated (40) | 1.00 | 1.00 | .000 | .000 | §58 | s2cue2_read.json |
| pi0 P I (swapped starts) | 2 | §58 manipulated (40) | .000 | .000 | .425 | .900 | §58 | s2cue2_read.json |
| pi0 P tv | 2 | §58 manipulated (40) | .000 | .750 | .100 | .300 | §58 | s2cue2_read.json |
| pi0 M I | 2 | §58 manipulated (40) | .775 | .725 | .000 | .000 | §58 | s2cue2_read.json |
| pi0.5 Sm I (§59 layout) | 2 | new, seeds 835000 (40) | .000 | .025 | .975 | .000 | §59 | results/flawed/s2cue3_read.json |
| pi0.5 PSm I | 2 | new, seeds 835000 (40) | .000 | 1.00 | .050 | .000 | §59 | s2cue3_read.json |
| pi0 Sm I | 2 | new, seeds 835000 (40) | .000 | .000 | .400 | .000 | §59 | s2cue3_read.json |
| pi0 PSm I | 2 | new, seeds 835000 (40) | .000 | .050 | .000 | .000 | §59 | s2cue3_read.json |
| pi0.5 Sm tv | 2 | new, seeds 835000 (40) | .000 | 1.00 | 1.00 | .000 | §59 | s2cue3_read.json |
| pi0.5 PSm tv | 2 | new, seeds 835000 (40) | .000 | 1.00 | .475 | .000 | §59 | s2cue3_read.json |
| pi0 Sm tv | 2 | new, seeds 835000 (40) | .000 | .000 | .025 | .000 | §59 | s2cue3_read.json |
| pi0 PSm tv | 2 | new, seeds 835000 (40) | .000 | .050 | .000 | .000 | §59 | s2cue3_read.json |
| pi0.5 Sx I (§60 layout) | 2 | new, seeds 845000 (40) | .975 | .975 | .000 | .000 | §60 | results/flawed/s2cue4_read.json |
| pi0.5 PSx I | 2 | new, seeds 845000 (40) | .000 | .800 | .075 | .075 | §60 | s2cue4_read.json |
| pi0 Sx I | 2 | new, seeds 845000 (40) | .475 | .000 | .275 | .000 | §60 | s2cue4_read.json |
| pi0 PSx I | 2 | new, seeds 845000 (40) | .000 | .000 | .000 | .000 | §60 | s2cue4_read.json |
| GR00T Sm I | 2 | new, seeds 845000 (40) | .075 | .025 | .700 | .000 | §60 | s2cue4_read.json |
| pi0.5 Sx tv | 2 | new, seeds 845000 (40) | 1.00 | .975 | .500 | .000 | §60 | s2cue4_read.json |
| pi0.5 PSx tv | 2 | new, seeds 845000 (40) | .000 | .825 | .725 | .700 | §60 | s2cue4_read.json |

§61 (s2_cue5) reports routing shares and a majority-rule test on pi0.5 only, not new-pairing success. It is not copied into these tables.

## Table C: other rows named in the record (not new-pairing success)

- SR §39 and §42 (Q2 and Q2-N) report prototype-operator parsing (J of 6), not success. Excluded.
- SR §36 (Q0) is an operator read. Excluded.
- SR §51 (N4 and N6 probe) gives N4+N6 held-at-hand-over rows: pi0.5 vla 1.00 [1.00, 1.00]; GR00T vla .35 [.21, .45]. This is a subset of N4 and N6 only, so it is not a row in Table A or B.
- SR §53 (f1m_cue, shape and proprioception cues) gives N4+N6 success on fresh and main states: tvp .625 [.475, .775]; sb .812 [.700, .925]; sf .550; g .637. This is a subset of N4 and N6 only, excluded from Table A.
- SR §37.2 and §41: r2w and r2w0 are in Table A1.
- ARCH_FAILURES.md §2.1 gives pi0.5 and GR00T at 2 and 10 actions per query, from the same new-pairing family (familiar): pi0.5 .450 and .458 against .500 at 5; GR00T .158 and .250 against .167 at 5 (SR §32.1, results/flawed/sc_t1_read.json). Not in Table A, since these are chunking variants of the intact arm.
- ARCHITECTURE_PROPOSAL.md and ARCH_FAILURES.md restate SR numbers and give no new measurements. Their U values (.894, .883, .881) match Table A.
- X-VLA intact new-pairing success is not measured in any of the four docs. X-VLA appears in the F7 bowl family (Table A2) and in the break-atlas excess table (ARCH_FAILURES.md, the "Update 2026-11-12" table; these are excess values, not success rates).
- New-layout success (not requested) is in ARCH_FAILURES: pi0.5 .30, GR00T .08, U .378 (SR §38.1).

## Table D: flags

### D1. Same policy, different state sets (not directly comparable as a single number)

- **Frozen pi0.5 intact:** .500 (main 4-23, §46); .450 [.258, .642] (fresh k36-55, §47/§48); .454 [.242, .654] (new k116-155, §55). Three different state sets.
- **Frozen GR00T intact:** .167 (main, §46); .219 [.025, .456] (new k56-115, §49); .179 [.025, .371] (new k156-195, §56); .212 [.017, .475] (new k500-539, §71). Four different state sets.
- **tv / tvp / tvpb, pi0.5:** tv .725 (main) and .700 [.475, .900] (fresh); tvp .925 (main) and .817 [.583, 1.00] (fresh) and .838 [.637, .983] (k116-155, §55 tvp). tvpb .938 is k116-155 only.
- **tv, GR00T:** .400 (main, §46); .367 (fresh, §47); .406 [.092, .731] (k56-115, §49); .379 [.050, .717] (k156-195, §56). Four state sets.
- **rgp, pi0.5:** .883 [.733, .992] (fresh, §52); .929 [.854, .988] (k116-155, §55). Two state sets. Scene 2 rgp is a separate block in Table B (§68 .708 and §69 .700 are on the same seed family 980000 and 990000, not the same states).
- **rgp, scene 2 pi0.5:** §68 (seeds 980000) N1+N2 .708 and §69 (seeds 990000, labelled "§68 replication") N1+N2 .700 with N2 .983 instead of 1.00. The §69 "rgp" row is not the §68 row: it is a re-run on different states, and the source labels it a replication. Flagged.
- **Hp / H, scene 2:** the perception hand-over arm named Hp appears in §63 (seeds 920000: N1+N2 .450), §64 (seeds 950000: .483), §66 (seeds 960000: .333), §67 (seeds 970000: .350) and §68 (seeds 980000: .358). Five different state sets. The name collides with HP in §44 (scene 1, different system: H with learned perception and no sim state, .850 pi0.5 / .842 GR00T); do not join them.
- **pi0 I, scene 2 K1:** SR §50 table gives .867 (K1) and §57 quotes .825 for the same arm (§50 pi0 I, k4-63 vs k4-43). Both are in the record; this is a conflict in the source text, flagged, not resolved.
- **§50 unmanipulated tv, scene 2 K2:** .933 (§54, k4-63) vs .90 (§57, k4-43). Same arm, different state subset. Flagged.
- **§54 vs §62 hand-over, scene 2 pi0.5:** hI N1 .550 (§54, k4-63) vs H0 (= §54 hI) N1 .600 (§62, seeds 890000). Different state sets.
- **Sim-state H (§44) vs scene-2 H (§62/§63):** different scenes and different state sets. Do not join.
- **Scene 1 ladder vs §34:** §34 R0/R1 on published frames were paused and have no read (SR §34). Only §37 rungs are numbers.
- **Scene 2 cue tables §57-§60:** K and N columns are on manipulated starts, not the unmanipulated §50 states. Flagged in Table B's last block.

### D2. Same policy measured on different scenes or task families

- **pi0.5 rgp / rgo:** scene 1 (§52, §55, N1-N6) and scene 2 (§68, §69, §72; N1, N2). Different scenes, not joined.
- **pi0.5 Hv / Hp / Hb / Hc:** scene 2 only (N1, N2).
- **pi0.5 I:** scene 1 (§46, §47, §55), scene 2 (§50, §54, §62-§72), F7 bowl family (§45) and RC family (§70). Different task families.
- **tvp / tvpb / tv:** scene 1 (N1-N6), scene 2 (N1, N2; §50, §54), F7 bowl families not used.

### D3. Partial rows (not full N1-N6 or N1+N2)

- SR §51 and §53: N4 and N6 only (listed in Table C).
- SR §65: pi0 and SmolVLA only (Table A).
- SR §56: GR00T only for tvb, tvg, tvbg (Table A).
- SR §44: pi0.5 and GR00T only.

## Table E: servers and launchers named for re-serving

| policy | server or launcher named | launcher script (SR section) | notes |
|---|---|---|---|
| frozen pi0.5 intact | `pp_serve_purify --model pi05 --mode intact --comp= --seed 13` (client seed 7, `--bind track`) | `experiments/tv_roll.sh` (§46 basis); `tv_fresh_roll.sh` (§47); `tv_attr_roll.sh` (§48); `s2_roll.sh` (§50) | server line in §57 is `pp_serve_purify --model pi05 --mode intact --comp= --seed 13` |
| frozen GR00T intact | `pp_serve_purify --model groot --mode intact --comp= --seed 13` (`pp_serve_purify.py` in §71) | `experiments/tv_g_roll.sh` (§49); `groot_box_roll.sh` (§56); `groot_rg_roll.sh` (§71) | GR00T uses the GR00T venv (§49 note) |
| frozen pi0 (openpi `pi0_libero`) | `serve_guided.py --mode ours --seed 13`, `BACKBONE_CONFIG=pi0_libero` | `experiments/pp_atlas_more_roll.sh` (§45.3); `s1_cross_roll.sh` (§65); `s2_roll.sh` (§50) | `serve_guided.py` is in `experiments/` |
| frozen X-VLA (`xvla-libero`) | `serve_guided.py --mode ours --seed 13` | `experiments/pp_atlas_more_roll.sh` (§45.3) | only F7 bowl family and atlas excess have numbers |
| frozen SmolVLA (`smolvla_libero`) | `serve_guided.py --mode ours --seed 13` | `experiments/pp_atlas_more_roll.sh` (§45.3); `s1_cross_roll.sh` (§65) | |
| tv / tvp / p / w / wp (role-view front end, pi0.5 and GR00T) | client `experiments/tv_frozen.py client --arm {I,tv,tvp,w,p,wp}` on the server above | `tv_roll.sh` (§46), `tv_fresh_roll.sh` (§47), `tv_attr_roll.sh` (§48), `tv_g_roll.sh` (§49) | |
| tvb / tvg / tvbg (GR00T) | client `experiments/f1m_cue.py client --arm …`, server as GR00T above | `experiments/groot_box_roll.sh` (§56) | |
| tvpb (views + phrases + box) | client `experiments/f1m_cue.py client --arm sb`; launcher `tvpb_roll.sh` | `tvpb_roll.sh` (§55); `s1_cross_roll.sh` (§65) | |
| ReGuide rgp / rgo | client `experiments/rg.py client --arm rgp` or `--arm rgo` | `rg_roll.sh` (§52, §55); `groot_rg_roll.sh` (§71); `s2rg_roll.sh` (§68); `s2rgt_roll.sh` (§69); `s2rgs_roll.sh` (§72) | rgp uses learned or perceived poses; rgo uses simulator poses |
| ladder R0, R1, R2, R2b, R2w, r2l (frozen VLM, §34/§37 recipe) | configs `pi05_g8v_plain_ae` (R0), `pi05_g8v_rsup_ae` (R1), `pi05_g8v_rview2_ae` (R2), `pi05_g8v_rbox_ae` (R2b), `pi05_g8v_rwrist_ae` (R2w); configs live in `lifelong_ood/policy/l90c_config.py` | (serve command not named; read via `sc_t34.py read --dir sc_t37`) | server command for these rungs is not named in SR §37 or §41; the checkpoints and config names are |
| H (sim-state hand-over) and HP | `experiments/sc_hp.py` (U checkpoint `pp_sc_U_s10`) | sc_hp.py (§44) | HP uses learned masks; H uses simulator state |
| U (parts policy) | `experiments/sc_pu.py roll`; checkpoints `pp_sc_U_s10/11/12` | sc_pu.py (§38); sc_seg_pu (§40) | |
| scene-2 hand-over arms (H, G, Hp, Hb, Hc, Hv, Hv0) | client `experiments/s2*.py` (s2fix*.py); servers as scene-2 frozen rows above | `s2fix_roll.sh` (§54), `s2fix2_roll.sh` (§62), `s2fix3_roll.sh` (§63), `s2fix4_roll.sh` (§64), `s2fix5_roll.sh` (§66), `s2fix6_roll.sh` (§67) | §63 Hp uses perception; §64 Hb is put-back |
| RC (composite, §70) | client `experiments/rg_comp.py`; launcher `rg_comp_roll.sh` | `rg_comp_roll.sh` (§70) | pi0.5 primary, pi0 secondary |
| F7 / F7R (bowl family) | `experiments/fx.py`, `fx_r`, `fx_more`, `fx_ra.py`; servers as in break atlas (§45.3) | `pp_atlas_more_roll.sh` (§45.3) | |
| scene-2 cue probes (§57-§60) | `experiments/s2cue.py`, `s2cue2.py`, `s2cue3.py`, `s2cue4.py`; job scripts under `.claude_tmp/pp/s2cue_*.sh` | `s2cue_*.sh` (§57-§60) | |

## Checks

- Numbers are copied as printed. CI brackets are from the source tables. Where the source gives no CI, the row says "no CI printed".
- Mapping and verdict labels from the sources (for example, FIX WORKS, FRONT END HELPS, RC WINS) are not restated, except where a row's note needs one.
- No value was computed from another value. The only derived-looking numbers are the ones the source printed (for example, "+.442 [+.367, +.517]").

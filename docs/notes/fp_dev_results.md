# Failure predictor, development fit on the LIBERO atlas (PLAN_GOAL3 pillar B), 2026-11-25

Existing rollouts only; no new episodes, CPU only. Scripts: experiments/fp_build.py, experiments/fp_fit.py (sbatch scripts in docs/notes/fp_jobs/). Outputs in results/flawed/fp_dev/ (table.csv, build_log.txt, fit.json, v1_frozen.json, v1_frozen.sha256).

## Data sources (per-episode json + track npz)
- Scene 1, development states 4-23, familiar layout, K1-K4, K6, N1-N6, five VLAs: results/flawed/pp_atlas/{pi05,pi0,groot,xvla,smolvla}_<tid>.json (+ _track.npz); tids from results/flawed/n2_tasks.json.
- Scene 1, fresh states 36-55 (same tasks, five VLAs): results/flawed/pp_atlas_fresh/.
- Scene 1 initial geometry: assembled from step-0 tracks of the K tasks (matches results/flawed/rva1/geom_dev.json to 1.7 cm on dev states; identical across models); rack = dev median (fixture).
- Scene 2, standard layout, K1 K2 N1 N2, states 4-63: results/flawed/s2_roll/{pi05,pi0,groot}_I_<tid>.json; positions from results/flawed/s2/s2_tasks.json, plate pixel area from results/flawed/s2cue5/s2cue5_states.json (orig, states 4-43 only).
- Scene 2 cue-law layouts (pi05 only), states 4-43: results/flawed/s2cue5_roll/{A2,PA2,PD2,Sx,PSx,Sm}/pi05_I_<tid>.json with layouts and plate pixel areas from s2cue5_states.json. Earlier cue sets (s2cue..s2cue4, 'O') not used.
- Class labels S/G/W/O/H/M come from a track-based reimplementation of rva1_read.classify (the channel-log E1 classifier exists only for the RAA arms, not for the frozen atlas runs). Not used: new layout (no geometry), F6/F7 tasks, X-VLA and SmolVLA in scene 2 (no rollouts).

Rows: 3,880. Scene 1: 440 per model (220 dev + 220 fresh), 2,200 total. Scene 2: pi05 1,200 (240 standard + 960 cue layouts), pi0 240, groot 240. Failure rate .542 (scene 1 .464, scene 2 .645); K rows fail .021 (scene 1) and .418 (scene 2, cue layouts included); N rows .833 and .873. Classes: S 1776, G 1051, W 577, H 293, O 97, M 86.

## Features (pre-rollout) and missingness
novelty (N task); log partner counts (object's trained targets, target's trained holders; THEORY_COMPOSITION table); trained-cue conflict (object has a trained partner receptacle other than the instruction's target); geometric: a non-instructed receptacle nearer to the object's start than the instructed one, signed nearness margin (m), rank of the instructed receptacle by distance; salience conflict (plate pixel area >= 1.5x the other, scene 2 only); language form and occlusion flags; known-partner base rate (the model's success on the partner K tasks of the same set, leave-state-out).
Missing: salience 62.9 percent (all of scene 1: no rendering, imputed to the mean plus a missing flag; scene-2 standard states >43). Constant (zero information): language form and occlusion (every task here is plain, unoccluded). No other missing.

## Results (pooled out-of-fold AUC for P(fail); 95% CI from 2,000 cluster bootstrap draws over states; macro = mean of per-fold AUCs)
| scheme / set | v0 | v1 | b1 novelty | b2 base rate | b1+b2 |
|---|---|---|---|---|---|
| LOMO, all rows | .776 [.764, .791] | .885 [.872, .895] | .763 [.742, .784] | .617 [.591, .640] | .879 [.868, .888] |
| LOMO macro, all | .793 | .927 [.918, .935] | .878 [.867, .888] | .614 [.603, .624] | .924 |
| LOMO macro, N tasks only | .445 [.394, .501] | .584 [.544, .623] | .500 | .553 [.488, .615] | .553 |
| LOSO, all rows | .776 [.765, .790] | .693 [.667, .720] | .775 [.752, .801] | .644 [.631, .657] | .891 [.874, .904] |
| LOSO, N tasks only (macro) | .692 [.675, .708] | .263 [.242, .285] | .500 | .349 [.331, .368] | .349 |
Paired differences, LOMO all: v1 - b2 +.265 [.235, .299]; v1 - b1 +.120 [.107, .133]; v1 - v0 +.105 [.089, .122]; v1 - (b1+b2) +.005 [.001, .010]. LOSO all: v1 - b2 +.049 [.030, .069]; v1 - b1 -.083 [-.128, -.037]; v1 - v0 -.086 [-.107, -.064]. Pooled N-only AUCs are distorted by folds with different intercepts; read the macro rows.
Per held-out model (LOMO AUC v0 / v1 / b1 / b2): groot .73/.91/.85/.63, pi0 .89/.97/.96/.53, pi05 .76/.84/.70/.75, smolvla .82/.97/.96/.67, xvla .77/.94/.92/.49. Scene-wise LOMO: scene 1 v1 .886 vs b1 .856 vs v0 .783; scene 2 v0 .886 vs v1 .804 vs b1 .713.
Class top-1 agreement over all rows (LOMO; chance-level majority S .458): v0 .432 (only S/W possible), v1 .573, b1 .593, b2 .517, b1+b2 .613. Among true failures (LOMO): v0 .274, v1 .374, b1 .448, b2 .353, b1+b2 .448. LOSO top-1: v1 .433, b1 .593. Class prediction is not better than the novelty baseline.
Frozen v1: results/flawed/fp_dev/v1_frozen.json, sha256 6bd3f41458d2d6010946139a6405d52e7cf8426345564fff1d935949ef2a8ebd (L2 lambda=10 on sum-scale NLL, fixed before the fit; lambda 1 / 10 / 100 give LOMO AUC .880 / .885 / .881). Coefficients (standardized): novelty +2.46, nearer-conflict +.96, near-margin +.49, base rate -.60, trained-cue conflict -1.29 (sign flipped by collinearity with novelty), instr-rank -.89, rest near 0.

## Assessment
1. v1 beats b2 by a wide margin in LOMO (+.27) and narrowly in LOSO (+.05). Against b1 (novelty alone) the gain is +.12 pooled in LOMO and the scene-wise gain is small (scene 1 .886 vs .856). Against novelty plus base rate together the gain is +.005, essentially nil. The predictor's skill is "new pairing fails", plus a model-level base rate that tells which models fail on known tasks. b2 alone is weak because known tasks are near ceiling (K fail .021) in scene 1.
2. Within new pairings (the case a new benchmark cares about), v1 reaches only macro AUC .58 in LOMO and anti-predicts under leave-scene-out (.26). v0, the cue-vote rule, is the better within-N predictor across scenes (.69 LOSO, .886 AUC in scene 2 overall) and worse in scene 1 (.78 against .886; salience missing and only nearer plus trained votes). The claim "v1 beats b2 on new benchmarks" is not established by this development fit, because b2 is nearly constant here; the stronger test is v1 vs b1+b2 on new benchmarks.
3. Features that carry v1: novelty (drop-one costs .017 AUC), base rate (-.021), instr-rank, trained-cue conflict and nearer-conflict (-.003 to -.006 each); the others cost nothing. Single-feature v1 within N rows: nearer-conflict .64 and near-margin .65 (scene 2 cue layouts drive this). Language and occlusion carry nothing because they are constant in this atlas.
4. Class prediction (top-1) does not beat the novelty baseline.
5. Leakage and design concerns: (a) no feature uses episode outcomes except the base rate, which uses other states' K episodes of the same model (leave-state-out) and never the row's own episode; it is outcome information from the same rollout campaign, available only if known-task rollouts are run on the new benchmark. (b) Scene-1 geometry varies little by state, so geometric features are nearly task-level constants; in LOMO the same tasks appear in train and test, so v1 can partly memorise task identity. Neither scheme holds out pairings. A leave-one-pairing-out CV is the missing honest test and should be added before the freeze is used. (c) Scene 2 has two held-out pairings, so its trained-cue and novelty features coincide, and 960 of its 1,200 pi05 rows come from layouts designed to separate the cues (a design that favours v0). (d) Salience is a missing feature in scene 1 (no rendering), so v0 votes with two cues there. (e) The track classifier differs from the channel-log E1 classifier. (f) Rows share states across tasks and layouts; CIs resample state clusters, but scene 1 has 40 states and scene 2 40 to 60.
6. b3 (VLM prompt predictor) not run.

# SR 89 build: steps 0 and 1 of docs/notes/scene2_plan.md (written 2026-11-20)

Scope: step 0 (exploratory zero-shot probe of the kitchen RAA-sL and W5 on scene 2) and step 1 (scene-2 role-view / role-channel data, config, smoke, two training chains).  No registered rollout was started, no prereg row exists, no SR 89 predictions file yet.  Code: experiments/s2_rv.py, experiments/wa_s2.py, experiments/sr89_probe_read.py; additive edits to lifelong_ood/policy/l90c_config.py (config `pi05_g8v_raas_ls2_ae`) and experiments/c_train_pi05.py (`--g8v raas_ls2`).  Job scripts (outside git, as SR 82's): /coc/flash10/ksingara3/.claude_tmp/pp/sr89_{libero,cpu,probe}.sh, raasls2_train.sh, raasls2_chain.sh.

## Scene-2 state ranges (recorded)
| range | use | state |
|---|---|---|
| seed 830000 + k | SR 50 | earlier |
| seed 980000 + k | SR 68 / 69 / 72 | earlier |
| seed 1,090,000 + k, k 0-19 (branches seed_base 47,000,000 + 1000 k) | step 0 probe states (K1, K2, N1, N2 share the 20 states) | USED by the exploratory probe; never to be used by registered rollouts |
| seed 1,080,000 + k, k 0-9 | calibration of the mask + depth channel estimator (rest poses) | USED |
| seed 1,100,000 + k, k 0-63 (plan: 0-3 smoke, 4-63 main) | registered SR 89 rollouts | RESERVED, not generated, not used |
Deviation from the task text: the plan named ~1,100,000 + k for the probe; the probe used 1,090,000 so the 1,100,000 range stays fresh for the registered main.  All probe and calibration states were checked unequal to every earlier scene-2 state file (s2, s2cue3-5, s2fix2-6, s2rg) and to each other (results/flawed/sr89/probe/ranges_used.json).  The scene-1 states reserved by SR 88 were not touched (scene 2 has its own env and seeds).

## Step 0: zero-shot probe (EXPLORATORY, no prereg row, one trial per state, 20 states per cell, same states for every cell)
Client experiments/wa_s2.py (copy of the wa_s.py channel path on s2.py's tvp client: simulator-mask role views, role phrases, state-token channel).  Channel positions: simulator masks + rendered depth + camera calibration (median of the entity's back-projected agentview mask points + a per-entity constant fitted at rest on the calibration states; occluded mask = last estimate seen with at least half of its first mask).  Checkpoint pi05_g8v_raas_l_ae_main6000 (RAA-sL), served as is (where_own 0) and with W5 (where_own 5; the plate phrases are not in raasw5_mu.json, so mu = 0, the channel's own rel_xy).  Jobs 4813459 (RAA-sL), 4813460 (W5); read 4813784 -> results/flawed/sr89/probe/probe_read.json.

| arm | K1 | K2 | N1 | N2 | K1+K2 | N1+N2 |
|---|---|---|---|---|---|---|
| kitchen RAA-sL zero-shot | 20/20 | 11/20 (.55) | 16/20 (.80) | 5/20 (.25) | 31/40 = .775 [.625, .877] | 21/40 = .525 [.375, .671] |
| kitchen RAA-sL + W5 | 20/20 | 12/20 (.60) | 16/20 (.80) | 4/20 (.20) | 32/40 = .800 [.652, .895] | 20/40 = .500 [.352, .648] |
(Wilson 95% intervals on the pooled cells.)  For reference only (different states, not a contrast): SR 50 reports views alone on frozen pi0.5 at N1+N2 .000, SR 68 rgp at .708.

Reading (descriptive): the K1+K2 floor (.70) holds out of suite for the kitchen executor, but K2 (.55-.60) is clearly below K1 (1.00).  N1 (.80) is well above the views-only .000, N2 (.20-.25) is low.  The channel moves N1 but the probe cannot say whether that is localisation, the init's prior (pi05_libero has seen the scene-2 compound), or the white-mug / left-plate habit.  W5 is indistinguishable from plain RAA-sL (one point in 40 either side).  This does NOT remove the need for training: N2 and K2 are the weak cells, and the plan's A1 / L1 contrasts need a model that fits K1 / K2.

Channel at test (client log, descriptive; the first-listed numbers are for RAA-sL): object-role channel error median .10-.13 cm (K1, K2, N2), 1.19 cm on N1; held-target channel error median 3.32 cm (K1), .74 cm (K2), 4.01 cm (N1), 2.89 cm (N2); share of held queries within 2 cm: .17 / .87 / .00 / .40.  The target PLATE position error is about 1 cm in all cells (.78-1.35 cm median), so the held-target channel error comes from the in-hand offset (the pre-hold mug estimate against the eef at the hold's onset, which for the white mug carries the lift between the last pre-hold query and the onset), not from the plate.  This estimator misses SR 77's 2 cm gate on three of four cells; a registered run would need a better held-offset rule (or the privileged `--chan sim`, which exists).  The probe's result is therefore a probe with a noisy channel.

## Step 1: data
Source: the 50 raw LIBERO-10 compound demos (s2_tasks.DUMP), replayed in the scene's env (jobs 4813454 render, about 265-290 raw steps per demo, 12,892 kept frames in all, only no-op steps dropped), both cameras with simulator segmentation (experiments/s2_rv.py render -> /coc/flash10/ksingara3/robot/data/fa1/rv_s2, 2.1 GB, not in git).  In the replay the compound goal holds at the last state in 48 of 50 demos (demos 1 and 11 do not end on the goal at the last recorded state; their 4 cycles are kept and flagged goal_at_end False in the manifest).
Split by s2rg.py's cycle rule (first rise of 3 cm, first open command): every one of the 50 demos is a white-mug cycle (K1, plate_1) followed by a yellow-and-white cycle (K2, plate_2); no demo has the other order (first_cycle_mug_per_demo in the manifest).  So every K2 cycle starts with the white mug already on the left plate and the hand above it (a start K2 never has at test, plan risk 60).
Prompts and views as the plan: "object: the white mug" / "target: the left plate" (K1), "object: the yellow and white mug" / "target: the right plate" (K2); rview2 rule over this scene's five movables, exact simulator masks, both cameras; channel [dx, dy, dz, role id], object role = mug - eef until the hold, target role while held = plate - eef - the in-hand offset fixed at the onset (sc_rv.held_target_rel), labels are simulator body positions; hold = ProprioHeld with the kitchen threshold (sc_rv_lex.json tw .002534, lift .01, unchanged).
Merge: kitchen lol/libero_g8v_raac copied frame by frame (400 episodes, 52,181 frames, PNG decoded and re-encoded, per-frame count verified) + 100 scene-2 cycles -> new repo lol/libero_g8v_raacs2 (6.4 GB under the LeRobot home, not in git).  Job 4813531 convert; manifest results/flawed/sr89/s2_rv_manifest.json.

| quantity | value |
|---|---|
| episodes (kitchen + scene 2) | 500 = 400 + 100 |
| frames | 65,073 = 52,181 + 12,892 (scene 2 = 19.8%) |
| scene-2 cycles | 100 (K1 50, K2 50); 12,892 frames, 5,181 held |
| frames per prompt | object: the white mug 2,973; target: the left plate 2,740; object: the yellow and white mug 4,738; target: the right plate 2,441 |
| demos skipped | 0 |
| N1, N2 frames | none (held out by construction) |

Channel sanity.  Role-channel error against simulator truth: 0 by construction in the data (the labels are the simulator's body positions); the mask + depth estimator's error at test is the probe's channel line above.  Replay checks: eef from the replay equals the recorded state by construction (set_init_state of each recorded state); every cycle has a hold event (no cycle without held rows); the detector's hold agrees with the simulator's (mug up > 2 cm and within 12 cm of the hand) on .889 of frames on average (min .70), the detector fires a median of 3 frames BEFORE the simulator rule (range 0 to 42 frames before, one outlier cycle) because it triggers on finger closure plus a 1 cm lift; the target channel at the last held row is 1.6 cm median in xy from the plate centre (the release offset) and the median in-hand offset is [+0.5, -4.7, -8.3] cm (x, y, z).  Paint: agent-view painted share .06-.08 of pixels, wrist .05-.15; empty paint in the wrist view on 168 K1-pre and 201 K2-pre frames (no other movable in view), none in the agent view.
Servo check (plan 1a asks to refit or check): the pooled p99 of |action xyz| on scene-2 held rows is .763 (per axis .750 / .774 / .745) against the kitchen servo_max .9375 and K 2.851; the constants are unchanged (RAA-sL's recipe), no refit.
Norm stats: the config keeps RAA-c's norm stats / assets (as RAA-sL does), NOT the union's (a deviation from plan 1a / 1b, which said recomputed on the union): the servo constants (st_q01 / st_q99, act_q01 / act_q99) and the init expert's normalisation are tied to them, and a union recompute would change the servo's de-normalisation.  Side numbers (results/flawed/sr89/s2_rv_norm_side.json): the state's dy q99 moves from .174 to .353 m on the union (scene-2 target offsets reach further in y); the other 4 + 7 quantiles move by under .02.  The scene-2 dy values therefore sit up to about 2x beyond the old normalised range; the quantile normalisation extends linearly.

## Config and trainer
`pi05_g8v_raas_ls2_ae` in lifelong_ood/policy/l90c_config.py: pi05_g8v_raas_l_ae's model, servo constants, freeze filter, optimiser, schedule, batch, weight loader (RaacLoader(LIBERO_PARAMS)) and assets; ONLY data.repo_id differs (S2_REPO = lol/libero_g8v_raacs2).  The existing configs are unchanged: job 4813444 (sr89_cfgtest.py) builds all 22 pre-existing configs from a copy of the pre-edit file and from the edited file, repr identical for all 22; NAMES old == new minus the new name; the new config equals raas_l on model, weight loader, freeze filter, optimiser, schedule, batch, EMA and steps.  `--g8v raas_ls2` in c_train_pi05.py (training; export through raac_export.py as raas_l).

## Smoke and training
Smoke job 4813641 (raasls2_train.sh, exp smoke, 20 steps, batch 32): loaded the 500-episode repo and RAA-c's norm stats, step 0 loss .0447, 5.39 s/step median, 20.8 GB peak, checkpoint written (jax/pi05_g8v_raas_ls2_ae/smoke/19).  Recipe unchanged: frozen VLM, action expert + state_proj trained, batch 32, lr 5e-5, warmup 1,000, 6,001 steps, save every 250, keep period 1,000, one seed, one A40, no EMA.
Chains (raasls2_chain.sh, two redundant 6 x 4 h afterany chains, as SR 82's):
- exp `main`, overcap: raasls2_main1-6 = jobs 4813659-4813664 (main1 started 12:05 on conroy).
- exp `tl`, tail-lab: raasls2_tl1-6 = jobs 4813665-4813670 (tl1 started 12:05 on conroy).
Expected: 6,001 steps x 5.4 s = about 9.0 h of compute per chain, 3 of the 6 jobs; if no node waits, step 6000 appears about 21:30-22:30 EST on 2026-11-20 (SR 82's chains lost about 4 h to waits: worst case early 2026-11-21).  First-to-appear rule: when jax/pi05_g8v_raas_ls2_ae/{main,tl}/6000 exists, scancel the other chain (raasls2_main* / raasls2_tl*), then export: t16_export.sh with ARGS `--g8v~raas_ls2~--exp~<main|tl>~--export~--step~6000~--tag~<main|tl>6000` -> .../checkpoints_l90c/pytorch/pi05_g8v_raas_ls2_ae_<main|tl>6000, then a serving smoke through wa_s2.py serve --config pi05_g8v_raas_ls2_ae.  Not done yet.

## Deviations and limits
1. Probe states from 1,090,000 (not 1,100,000), to keep the main range fresh.
2. Config keeps RAA-c's norm stats (see above), not a union recompute.
3. The probe's channel estimator is mine (sc_pu.Estimator is kitchen-bound); its held-offset rule gives 3-4 cm held-target channel error in the white-mug cells.
4. K1 / K2 cycles are not exactly the K1 / K2 test starts (K2 cycles always follow a K1 cycle in the demos).
5. No resolver or pose-memory analogue, no mu table for the plates, no wa_s2 arms noR / noM: step 2 of the plan, not asked.
6. Smoke dataset lol/libero_g8v_raacs2_smoke (3 episodes) and results/flawed/sr89/s2_rv_manifest_smoke.json remain from the format test; not used for training.

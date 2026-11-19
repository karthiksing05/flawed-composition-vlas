# RAA-L (SR 80): the RAA recipe initialised from the LIBERO-fine-tuned pi0.5

Written 2026-11-18, before the prereg row, any main training job and any RAA-L rollout.  Predictions with probabilities: experiments/prereg/raal_pred.txt (copied into the row).
Code: lifelong_ood/policy/l90c_config.py (config `pi05_g8v_raa_l_ae`, additive), experiments/c_train_pi05.py (`--g8v raa_l`, additive), experiments/raal_read.py,
experiments/prereg/raal.py; scripts /coc/flash10/ksingara3/.claude_tmp/pp/raal_train.sh, raal_chain.sh, wal_roll.sh (raa_roll.sh, t16_export.sh unchanged).
RAA-s (jobs raas_*) is training on the same machine; none of its files, raa.py, sc_rv.py or the raas_* scripts is edited.

## Why
SR 79 (docs/notes/sr79.md, results/flawed/wa_read.json): the WHERE modules' removal test on RAA was NOT SHOWN because RAA (pi05_base + 8 known tasks, 6000 steps, action
expert only) is at the floor on out-of-suite scenes: F6_b .050, F6_a .083, unique-name bowl controls .108, while frozen pi0.5 (I) gets .517 on F6_b and .967 on F6_a.
The resolver removal worked (D1 +.213).  We need an executor that keeps broad competence so the module removals are testable.

## The checkpoint
`/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero/params` (openpi's pi05_libero, JAX params; assets with its own norm stats in
.../pi05_libero/assets/physical-intelligence/libero).  It is the frozen "I" / pi05 of SR 45 / 52 / 70: experiments/pp_serve_purify.py --model pi05 (called by
experiments/rg_comp_roll.sh MODEL=pi05) builds `PolicyBackbone("pi05_libero")` (experiments/pp_f1_mix.py Pi05), whose PyTorch weights are
.../checkpoints/pi05_libero_pytorch, converted from this JAX checkpoint.  The model config is the one the RAA recipe already uses (Pi0Config pi05=True, action horizon 10,
discrete_state_input False), so the params load with openpi's CheckpointWeightLoader exactly as pi05_base's do (no new tensors; RAA has no state_proj, unlike RAA-c / -s).

## Leakage check (N1-N6)
openpi's pi05_libero was fine-tuned on physical-intelligence/libero = the LIBERO spatial / object / goal / 10 suites (40 tasks, 4 suites; LIBERO-90 is not in it).
Check (job 4793857, experiments/ copy at /coc/flash10/ksingara3/.claude_tmp/pp/raal_leak.py, LIBERO venv, `benchmark.get_benchmark_dict()`): exact instruction match of
each n2 task language (results/flawed/n2_tasks.json) against every task of libero_spatial, libero_object, libero_goal, libero_10 and libero_90:
- K1-K8 (tids 0, 2, 4, 6, 8, 10, 12, 14): each matches exactly one LIBERO-Goal task (bowl on plate, bowl on stove, bowl on top of cabinet, wine bottle on top of cabinet,
  wine bottle on rack, cream cheese in bowl, open top drawer and put bowl inside, push plate to front of stove).  These are the known tasks; the checkpoint has seen them
  (and the RAA training demos are the physical-intelligence/libero demos of these eight tasks, SR docs section 1149), as the RAA recipe's own K1-K8 data.
- N1-N8 (tids 16, 18, 20, 22, 24, 26, 28, 30): "put the cream cheese on the plate / on the stove / on top of the cabinet", "put the wine bottle on the plate / on the stove /
  in the bowl", "open the top drawer and put the cream cheese inside", "push the bowl to the front of the stove": NO match in any of the five suites (including libero_90).
- LIBERO-Goal tasks the checkpoint saw and RAA did not (open the middle drawer, turn on the stove) are not among N1-N6.
So N1-N6 are not in the checkpoint's training data as tasks.  Residual exposure, stated and not removable: the checkpoint has seen the other 32 LIBERO tasks' demos and the
other two Goal tasks, i.e. general grasping of cream cheese / bottle / bowl objects in the Goal scene; RAA-L - RAA therefore mixes a better init with more exposure
(prediction file, hypothesis).  N1-N6 as new pairings of seen objects and targets is unchanged.  No leak: do not stop.

## Norm statistics (decision)
The RAA-L config keeps RAA's own norm stats (assets dir pi05_g8v_raa, repo lol/libero_g8v_raa; the same files the raa config uses), NOT the checkpoint's.  Why: (1) one change
against RAA (the init), so L1 isolates it; (2) the served pipeline (raa.py --arm raa, wa.py) and the exported assets are identical to RAA's; (3) the two sets are close (the
action quantiles differ by .01-.18 of a unit: q01 x -.92 vs -.75, y -.77 vs -.80, rotation +-.15 vs +-.17 / .22 vs .14; the state is unused by pi0.5).  Cost: the init's action expert
outputs are in its own normalisation, so a small first-steps mismatch; the smoke's step-0 loss (.0421) shows it is small against what the init gains.  Training adapts it.

## The config
`pi05_g8v_raa_l_ae` = `pi05_g8v_raa_ae` (frozen VLM = SigLIP + PaliGemma LLM, action expert trained, no EMA, batch 32, lr 5e-5, 1,000 warmup, 6,001 steps, one seed, one A40, data
lol/libero_g8v_raa, RAA's norm stats) with `weight_loader = CheckpointWeightLoader(pi05_libero/params)`.  Frozen VLM weights are the LIBERO-fine-tuned ones (not pi05_base's),
so the VLM too is part of the init change.  Existing configs are unchanged (the edit adds a name to NAMES, an AE entry, one `if` before the generic return, one constant).

## Smoke (job 4793814, tail-lab, exp smoke, 20 steps, batch 32)
Step-0 loss .0421 (grad norm .179).  RAA (pi05_base init) step-0 loss .1028 / .1038 (raa_train_4757776 and later); RAA-s smoke .0991.  So the init cuts the step-0 flow loss by
about 60% on RAA's frames and actions.  (RAA reached .020 at step 6000.)

## Registered contrasts (paired bootstrap B = 2000, rng 0, labels as raa_read.py / wa_read.py; read: experiments/raal_read.py)
Phase A (SR 73's 60 states: dev trials 4-23 + new k 540-579; tasks K1-K6, N1-N6; rollouts raa_roll.sh ARM=raa NAME=raal CONFIG=pi05_g8v_raa_l_ae into results/flawed/raa):
- L1 RAA-L - RAA, success, N1-N6: SUPPORTED iff lower bound > 0.
- L2a RAA-L - rgp, N1-N6, NON-INFERIOR iff lower bound > -.10; L2b = beats rgp (lower > 0).
- LK RAA-L - RAA, K1-K6, non-inferior at -.05.  Fit gate as SR 73 (R0 on K1-K6 >= .80).
- Unlabelled: RAA-L - R2, - R0 (K), - RAA-o / RAA-c / RAA-s when their files exist, cheese vs bottle, placed given right grasp, per task.
Phase B (rg_comp rgc main states, wa.py arms full / noR / noM on the RAA-L checkpoint, results/flawed/wal, cells as SR 79: full 0-10, noR 0-3 + 7 + 8 + 10, noM 0-3 + 8):
- Gates first (full only): F6_a (tid 9) >= .50 for D2 and D3a to be TESTABLE; mean of the unique-name controls (tids 4-6) >= .50 for D1 and D3b to be TESTABLE; K1t + K2t >= .70 floor.
- D1 full - noR on the F7 pool (SUPPORTED iff lower > 0); D2 full - noM on F6_b (SUPPORTED iff lower > 0); D3a noR - full on F6_b and D3b noM - full on the F7 pool
  (EQUIVALENT iff the interval lies in (-.10, +.10)); DK full - noR on K1t + K2t (NON-INFERIOR iff lower > -.05).
- PRIMARY: DISSOCIATION SHOWN iff D1 and D2 SUPPORTED, D3a and D3b EQUIVALENT, and the gates TESTABLE; otherwise as SR 79's rule (PARTIAL iff D1 and D2 SUPPORTED with a D3 not
  EQUIVALENT, else NOT SHOWN), with "UNTESTABLE" reported when the label pattern passes and a gate fails.
- Descriptive: full vs I, rgp, rc of SR 70 (same states, results/flawed/rg_comp_roll), full vs SR 79's RAA full on the same cells, F6_b vs F6_a.
Read scripts are new (raal_read.py imports raa_read / raao_read / raac_read / raas_read / wa_read; none is edited).  Selftest (reproduces SR 73's V1a / V1b / V3a and SR 70's rc - I)
before the read.

## Launch (written after the prereg row; nothing below changes a registered quantity)
Training: two redundant 6 x 4 h afterany chains of raal_train.sh (raas_train.sh with the log name changed), launched by raal_chain.sh, `--g8v raa_l --steps 6001 --batch 32
--save-interval 250 --keep-period 1000`; overcap exp main (raal_main1-6), tail-lab exp tl (raal_tl1-6); checkpoints
/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raa_l_ae/{main,tl}/<step>.

LAUNCH RECORD (2026-11-18 ~20:00).  Prereg row ledger/m3_prereg/m3_prereg__libero_goal__raal__s0__cae04558.json (job 4793879; the row carries dirty=true from foreign uncommitted
doc edits in the tree, as earlier rows).  Smoke 4793814 (tail-lab) ok, checkpoint jax/pi05_g8v_raa_l_ae/smoke/19.  Chains (raal_chain.sh): overcap exp main jobs 4793899-4793904
(raal_main1-6), tail-lab exp tl jobs 4793905-4793910 (raal_tl1-6).  tail-lab holds this user's raas_tl and sb100k jobs (per-user 4-GPU cap): with raal_tl running that is 3 GPUs,
so both chains are on their registered partitions.  RAA-s's chains (raas_main / raas_tl) are untouched.  Expected: RAA-c / RAA-s needed 6-10 h wall from submission, about
5.4 s/step in the smoke (includes the first compile), so 6,001 steps ~ 6-9 h of A40 time over 2-3 jobs; step 6000 about 04:00-08:00 on 2026-11-19 depending on the queue.

WAITER PLAN (one background waiter, capped with `timeout`, re-armed on notification, stopped before the session ends; no python on the login node; all python through sbatch):
1. Poll `ls /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raa_l_ae/{main,tl}/6000`.  On the first to appear: scancel the other chain (jobs named raal_main* / raal_tl*).
2. Export: sbatch t16_export.sh with ARGS `--g8v~raa_l~--exp~<main|tl>~--export~--step~6000~--tag~<main|tl>6000` -> .../checkpoints_l90c/pytorch/pi05_g8v_raa_l_ae_<main|tl>6000
   (standard conversion, as raa; assets copied from the JAX checkpoint).
3. Phase A rollouts: raa_roll.sh ARM=raa NAME=raal CONFIG=pi05_g8v_raa_l_ae CKPT=<export> OUTD=results/flawed/raa, once with the main states / branches (PHASE main; FRESH
   results/flawed/n2_states, BR results/flawed/n2_branches_main.json) and once with the rx states (PHASE rx; FRESH results/flawed/n2_states_rx, BR n2_branches_rx.json), TIDS
   0+2+4+6+8+10+16+18+20+22+24+26 (as the raac rollouts: 4789144-4789147, split in two jobs per phase), overcap full exclude list.
4. Phase B rollouts: wal_roll.sh (CKPT=<export>) in the SR 79 split: full:0-3, full:4-7, full:8+9+10 + noR:7, noR:0-3, noM:0-3, noR:8+10 + noM:8 (noMe is not rolled), PHASE=main,
   OUTD default results/flawed/wal.  Check every cell exists with n_episodes = number of states.
5. Reads (sbatch, openpi venv): `python experiments/raal_read.py selftest`, then `read` (phase A, results/flawed/raal_read.json) and `readb` (phase B, raal_wa_read.json +
   raal_wa_gated.json), each once, no overwrite.
6. Write SR 80 only after the reads and score raal_pred.txt as written.  An exported checkpoint below step 6000 is not read.

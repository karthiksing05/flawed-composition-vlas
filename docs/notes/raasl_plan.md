# RAA-sL (SR 82): the RAA-s recipe (servo residual) initialised from the LIBERO-fine-tuned pi0.5

Written 2026-11-19, before the prereg row, any main training job and any RAA-sL rollout.  Predictions with probabilities: experiments/prereg/raasl_pred.txt (copied into the row).
Code: lifelong_ood/policy/l90c_config.py (config `pi05_g8v_raas_l_ae`, additive), experiments/c_train_pi05.py (`--g8v raas_l`, additive; training, and export through raac_export.py as raas),
experiments/raasl_read.py, experiments/wa_s.py (new, copy of wa.py), experiments/prereg/raasl.py; scripts /coc/flash10/ksingara3/.claude_tmp/pp/raasl_train.sh, raasl_chain.sh, wasl_roll.sh
(raa_roll.sh and t16_export.sh unchanged).  No existing config, script or file is edited (raal / raas / wa files untouched).

## Why
SR 78b: RAA-s (servo) N1-N6 .717 / K1-K6 .844 (bottle pairings .589, S1 PARTIAL) but RAA-s starts from pi05_base and, like RAA, fails out of suite.  SR 80: RAA-L (RAA from pi05_libero) kept
out-of-suite competence (F6_a .967, controls .558; Phase B gates TESTABLE) but lost on the held-out pairings (N1-N6 .575, bottle .333).  RAA-sL asks whether the two compose: the servo's
release aim on the bottle pairings plus the broad init's out-of-suite competence.  One change against RAA-s: the initial weights.

## The checkpoint, the config, the loading
Same init as SR 80: `/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero/params`.  `pi05_g8v_raas_l_ae` = `pi05_g8v_raas_ae` (Pi0ConfigS with servo_fields() from
results/flawed/raas_gate.json: K 2.851, M .9375; data lol/libero_g8v_raac; RAA-c's norm stats / assets dir pi05_g8v_raac; frozen VLM = SigLIP + PaliGemma LLM, action expert + state_proj
trained, no EMA, batch 32, lr 5e-5, 1,000 warmup, 6,001 steps, one seed, one A40) with `weight_loader = RaacLoader(LIBERO_PARAMS)`.  RaacLoader (raac_model.py) = CheckpointWeightLoader whose
`_merge_params(missing_regex=".*lora.*|.*state_proj.*")` leaves the parameters missing from the checkpoint at the model's own random initialisation: pi05_libero has no state_proj
(pi0.5 reads no state), so state_proj is initialised exactly as in RAA-s's training (raas used RaacLoader(BASE_PARAMS), the same missing set).  RAA-L's plain CheckpointWeightLoader could not be
reused (it would fail on the missing state_proj); RaacLoader is the existing loader of both RAA-c and RAA-s.  Norm stats are RAA-c's / RAA-s's own (one change against RAA-s; the servo's
normalised constants, st_q01 / act_q01 ..., come from them), as RAA-L kept RAA's.
Smoke (tail-lab job 4801501, exp smoke, 20 steps, batch 32): loads; state_proj kernel (32, 1024) / bias (1024,) are in the parameter list; step-0 loss .0556 (grad norm .201), against RAA-s's
smoke .0991 (pi05_base) and RAA-L's .0421; about 5.4 s/step.  The step-0 loss is lower than RAA-s's, higher than RAA-L's (the servo-subtracted target and the random state_proj).

## Serving: sampling path and the channel (finding)
PolicyBackbone.sample: wa.py's server is `GuidedPolicy(PolicyBackbone(name, ckpt), hooks=None)`, i.e. the same bk.sample that raa.py --arm raac / raas serves through.  The SR 78b fix is in
bk.sample, so the servo is added whenever servo_k != 0; nothing in the server loop bypasses it.  (Bit-identity tests: experiments/raas_b_test.py, unchanged.)  So the sampling path needs no fix.
BUT wa.py carries the role channel as RAA's TEXT suffix (`chan_text`, prompt = role phrase + "rel: dx dy dz cm") and its client sends no held-object offset.  RAA-s / RAA-sL read the channel as the
continuous 4-dim STATE token [dx, dy, dz (m), role id] (target channel = target - eef - held-object offset), with a 4-dim state the norm stats require.  Serving RAA-sL through wa.py would give
a text prompt the model never saw and an 8-dim eef state in a 4-dim normaliser: a wrong-model test.  Fix: experiments/wa_s.py, a COPY of wa.py (wa.py untouched, its SR 79 / 80 hashes unchanged) with only:
(1) the server's `final` sets `observation/state = chan_state(rel, on)` and keeps the role phrase as the prompt; (2) the client's `channel` sends meta/raa_off_ok / off / toff by raa.py's rule
(offset = the Estimator's object estimate minus the eef at the held onset, fixed while held; held target rel = target - eef - offset); (3) the noM skip sends the base prompt and the all-zero
state (no channel; K7 / K8 convention); a frame with no channel is the all-zero state; (4) warm-up with a 4-dim state.  Everything else, arms, cells, resolver, pose memory, log record,
is wa.py's text.  Plumbing check before the row (not a result): wasl_roll.sh on the SR 78 RAA-s checkpoint pi05_g8v_raas_ae_tl6000 with config pi05_g8v_raas_ae, PHASE smoke, cells full:0 + noM:0 + full:8,
OUTD outside results/ (job 4801514): checked from the server log for state lines [dx, dy, dz, role] with role 1 / 2 and the channel diag.

## Leakage and norm statistics
As SR 80 (docs/notes/raal_plan.md): N1-N8 are in no LIBERO suite (exact instruction match), K1-K8 are LIBERO-Goal tasks the init has seen; the init has seen the other 32 LIBERO tasks' demos; RAA-sL - RAA-s
mixes a better init with that exposure.  Norm stats: RAA-c's, unchanged (the init expert's action outputs are in the LIBERO normalisation: a small first-steps mismatch that training adapts).

## Registered contrasts (paired bootstrap B = 2000, rng 0, labels as raa_read.py / wa_read.py; read: experiments/raasl_read.py)
Phase A (SR 73's 60 states: dev trials 4-23 + new k 540-579; tasks K1-K6, N1-N6; rollouts raa_roll.sh ARM=raac NAME=raasl CONFIG=pi05_g8v_raas_l_ae into results/flawed/raa; "RAA-s(78b)" = the
raas_b_* rollouts of SR 78b, the registered reference, served with the servo):
- P1 PRIMARY RAA-sL - RAA-s(78b), success, N1-N6: SUPPORTED iff lower bound > 0 (REVERSED iff upper < 0; else NOT SUPPORTED).
- P2 RAA-sL - RAA-L (SR 80 rollouts), success, N1-N6: SUPPORTED iff lower bound > 0.
- P3a RAA-sL - rgp, N1-N6: NON-INFERIOR iff lower bound > -.10; P3b = beats rgp (lower > 0).
- PK RAA-sL - RAA-s(78b), K1-K6: non-inferior at -.05.  Fit gate as SR 73 (R0 on K1-K6 >= .80).
- Unlabelled: RAA-sL - RAA, - R2, - R0 (K), placed given right grasp vs RAA-s(78b) and RAA-L, cheese / bottle vs RAA-s(78b), - RAA-o / RAA-c / RAA-s(78), per task.
Phase B (wa_s.py arms full / noR / noM on the RAA-sL checkpoint, rgc main states, results/flawed/wasl; cells as SR 79 / 80: full 0-10, noR 0-3 + 7 + 8 + 10, noM 0-3 + 8):
- Gates first (full only): F6_a (tid 9) >= .50 for D2 and D3a TESTABLE; mean of the unique-name controls (tids 4-6) >= .50 for D1 and D3b TESTABLE; K1t + K2t >= .70 floor.
- D1 full - noR on the F7 pool (SUPPORTED iff lower > 0); D2 full - noM on F6_b (SUPPORTED iff lower > 0); D3a noR - full on F6_b and D3b noM - full on the F7 pool (EQUIVALENT iff the interval lies in
  (-.10, +.10)); DK full - noR on K1t + K2t (NON-INFERIOR iff lower > -.05).  PRIMARY gated as SR 80 (DISSOCIATION SHOWN iff D1 and D2 SUPPORTED, D3a and D3b EQUIVALENT and the gates TESTABLE; else
  wa_read's rule; "UNTESTABLE" when the label pattern passes and a gate fails).
- Descriptive: full F6_b vs frozen I (.517) and rgp, rc of SR 70 (same states); full vs SR 79's RAA full and SR 80's RAA-L full on the same cells; F6_b vs F6_a.
Read scripts are new (raasl_read.py imports raa_read / raao_read / raac_read / raas_read / raas_b_read / raal_read / wa_read; none is edited).  readb catches wa_read's print-loop KeyError (noMe contrasts, not
rolled; the SR 80 deviation) after the json is written.  Selftest (reproduces SR 73's V1a / V1b / V3a, SR 76's W1a / W1b / WK, SR 70's rc - I) before the read.

## Launch (written after the prereg row; nothing below changes a registered quantity)
Training: two redundant 6 x 4 h afterany chains of raasl_train.sh (raal_train.sh with the log name changed), launched by raasl_chain.sh, `--g8v raas_l --steps 6001 --batch 32 --save-interval 250
--keep-period 1000`; overcap exp main (raasl_main1-6), tail-lab exp tl (raasl_tl1-6); checkpoints /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raas_l_ae/{main,tl}/<step>.
(LAUNCH RECORD appended below after the launch.)

WAITER PLAN (one background waiter, capped with `timeout`, re-armed on notification, stopped before the session ends; no python on the login node; all python through sbatch):
1. Poll `ls /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raas_l_ae/{main,tl}/6000`.  On the first to appear: scancel the other chain (jobs named raasl_main* / raasl_tl*).
2. Export: sbatch t16_export.sh with ARGS `--g8v~raas_l~--exp~<main|tl>~--export~--step~6000~--tag~<main|tl>6000` -> .../checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_<main|tl>6000
   (raac_export.py conversion with state_proj, as raas; assets copied from the JAX checkpoint).
3. Serving smoke (outside results/): raa_roll.sh ARM=raac CONFIG=pi05_g8v_raas_l_ae CKPT=<export>, tids 0 and 22, 2 states (as SR 80's / SR 78b's smoke); check the server log for role 1 / 2 state lines.
4. Phase A rollouts: raa_roll.sh ARM=raac NAME=raasl CONFIG=pi05_g8v_raas_l_ae CKPT=<export> OUTD=results/flawed/raa, once with the main states / branches (PHASE main; FRESH results/flawed/n2_states, BR
   results/flawed/n2_branches_main.json) and once with the rx states (PHASE rx; FRESH results/flawed/n2_states_rx, BR n2_branches_rx.json), TIDS 0+2+4+6+8+10 and 16+18+20+22+24+26 as the raas_b / raal
   rollouts (4 jobs), overcap full exclude list.
5. Phase B rollouts: wasl_roll.sh (CKPT=<export>; default OUTD results/flawed/wasl; refuses without the raasl prereg row) in the SR 79 / 80 split: full:0-3, full:4-7, full:8+9+10 + noR:7, noR:0-3,
   noM:0-3, noR:8+10 + noM:8 (noMe is not rolled), PHASE=main.  Check every cell exists with n_episodes = number of states (40 for tids 0-7, 60 for 8-10).
6. Reads (sbatch, openpi venv, each once, no overwrite): `python experiments/raasl_read.py selftest`, then `read` (phase A, results/flawed/raasl_read.json) and `readb` (phase B, raasl_wa_read.json +
   raasl_wa_gated.json).
7. Write SR 82 only after the reads, scoring raasl_pred.txt as written.  An exported checkpoint below step 6000 is not read.  Report the backbone.py sha1 at launch as SR 80 did.

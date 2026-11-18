RAA-s (SR 78): RAA-c's continuous role channel plus a SERVO RESIDUAL action head.  Written 2026-11-18 BEFORE the prereg row, any main training job of RAA-s,
and any RAA-s rollout.  Known when written: SR 76 (RAA-o N1-N6 .594 against RAA .656, W1 NOT SUPPORTED; K1-K6 .881; rgp .897 on N1-N6), SR 77 (RAA-c N1-N6 .503, U1
NOT SUPPORTED; K1-K6 .814; target channel at test median .94 cm yet failed releases miss by 11-12 cm on N4 / N6).  The BC action expert does not turn exact target
geometry into centimetre releases; ReGuide's pose controller does.

Hypothesis (one).  The expert's weakness is the mapping geometry -> translation command, not the geometry.  Building that mapping into the action head as a fixed
proportional servo on the channel, and training the expert only on what the servo does not explain, should turn the channel's 1 cm accuracy into releases of
similar accuracy, because the servo's output is exact in the channel at every step and the expert cannot unlearn it (it is not a trained function).

The change (the only one beyond RAA-c).  RAA-s = RAA-c (same dataset lol/libero_g8v_raac, same norm stats, same state token via state_proj, same frames, role
phrases, painted views, recipe: frozen VLM, action expert + state_proj trained, batch 32, lr 5e-5, 1,000 warmup, no EMA, 6,001 steps, one seed, one A40) with the
action head replaced, translation dims only:

    action_xyz = clip(K * rel_xyz, -M, +M) * m  +  expert_xyz          rotation (dims 3-5) and gripper (dim 6): expert only, unchanged

rel_xyz = the channel's dx, dy, dz in metres (the target minus the held object, as RAA-o / RAA-c); m = 1 iff the channel's role id is 2 (target, while held), else 0
(object role, no channel, K7 / K8).  m is read from the same state vector the state token uses, inside the model (lifelong_ood/policy/raas_model.py): the state is
un-normalised with the dataset's own quantile stats (the inverse of openpi's Normalize), m = [role > 1.5], rel = the first three numbers.  The servo is applied to
every step of the 10-step action chunk (the channel is read once per chunk).  It is a fixed function inside the model, not a trained layer and not a test-time
wrapper: the training target of the flow-matching loss is (demo action - servo) and sampling returns (denoised residual + servo), both in the model's normalised
action space, where the servo is s * 2 / (q99 - q01 + 1e-6) on dims 0-2 (openpi's quantile Normalize is affine, so the offset cancels exactly).  openpi's pi0.5 LIBERO
actions are raw absolute delta-eef commands (extra_delta_transform False; pi0 predicts action-minus-state, pi0.5 does not), so the servo depends on the channel only.
No openpi file is edited; the arm needs no new data: it trains on the RAA-c dataset and serves through experiments/raa.py --arm raac with --config pi05_g8v_raas_ae
(the server's state token is identical).  New code: lifelong_ood/policy/raas_model.py (subclasses raac_model's config / JAX / PyTorch model), config entry
pi05_g8v_raas_ae, --g8v raas in c_train_pi05.py (training and export), experiments/raas_gate.py, raas_check.py, raas_read.py.  Existing arms are unchanged
(experiments/raas_check.py asserts the raac config and the K = 0 identity; raac_model.py, raa.py and sc_rv.py are not edited).

PHASE 0, the offline gate (experiments/raas_gate.py, job 4790531, results/flawed/raas_gate.json; no model, no rollout; the 400 RAA-c training demos' state and actions).
K and M come from the demos, never from a rollout.  M = pooled p99 of |demo action xyz| over held rows, per step = 0.9375 (LIBERO's own clip).  K = least squares of the
chunk-consistent pairs (held row t, j = 0..9: demo action xyz at t + j against the servo at t) = 2.851 action units per metre (the unclipped closed form is identical;
the clip never binds: no held row has K |rel| > M).  13,327 held rows (K1-K6; K7 / K8 have none).
  pooled, chunk-consistent: R^2 .361, residual variance ratio var(a - s) / var(a) .573 (mean-square ratio .587).
  pooled, per step (j = 0 only): R^2 .142, variance ratio .752.  R^2 by chunk position j = 0 .14 rising to j = 9 .50 (the demo's later steps follow the target more
  tightly than its first step).
  per task, chunk: variance ratio K1 .708, K2 .696, K3 .643, K4 .597, K5 .615, K6 .678; per task, per step: .879 1.034 .902 .856 .788 .884 (K2 above 1 at j = 0).
  Gate (set in the task brief): K > 0 and pooled variance ratio < .9: PASS (K = 2.851, ratio .573; the per-step ratio .752 also passes).
Reading: the servo is a modest prior, not a controller: it explains 36% of the held-phase translation variance of the demos (14% of the first step of a chunk).  Whatever
accuracy RAA-s gains at release must come through the servo's exactness near the target (where the demo's action also goes to zero) and the expert's cheaper residual,
not through a servo that matches the demos everywhere.  Per-task best K ranges 2.1 (K5) to 5.4 (K1); one pooled K is used (one gain, as registered).

Registered contrasts (read: experiments/raas_read.py = raac_read.py with the arm renamed, plus RAA-c as an unlabelled reference; the same 60 states as SR 73, 76 and 77 =
dev trials 4-23 + new k 540-579, paired bootstrap B = 2000, rng 0, tasks and states, raa_read machinery extended not changed):
S1  primary RAA-s - RAA-o on N1-N6: S1a success, S1b placed given right grasp.  SUPPORTED iff both lower bounds > 0; PARTIAL iff one; else NOT SUPPORTED.
S2  RAA-s - RAA, N1-N6 success: help (lower bound > 0).
S3a RAA-s - rgp, N1-N6 success, non-inferiority margin -.10 (NON-INFERIOR iff lower bound > -.10), reported either way; S3b = beats rgp (lower bound > 0).
SK  RAA-s - RAA-o on K1-K6 success, non-inferior at -.05.   Fit gate as SR 73 (R0 on K1-K6 >= .80).
Reported with no label: RAA-s - R2, RAA-s - R0 (K1-K6), RAA-s - RAA-c (N1-N6 and K1-K6), cheese vs bottle pairings, per-task success / right-grasp /
placed-given-right-grasp, channel error per query.
Exploratory (labelled, run after the read, NOT registered): release error check = raa_instr.py on N1, N4, N6, K5 x 20 states (trials 24-43, as SR 76 / 77): median failed-release xy
error, its correlation with the in-hand offset, releases per episode; RAA, RAA-o, RAA-c, rgp rows from results/flawed/raa_instr.json, raao_instr.json, raac_instr_id.json.

Predictions (before data; probabilities are mine).
1. S1 primary: SUPPORTED 22%, PARTIAL 14%, NOT SUPPORTED 64%.  Point estimate of S1a about +.06 (RAA-s N1-N6 about .65, range .45-.85).  Why not higher: the servo
   explains a third of held-phase translation variance, the expert still has to produce the approach, the transit height, the descent, rotation and the gripper
   timing, and in SR 77 the failed releases were not aim errors alone (releases per episode ~1.0-1.2: the policy releases at some point whatever the aim); a fixed
   gain K = 2.85 cannot stop at the target by itself.  Why not lower: the servo's sign and magnitude are exact in the channel, and the expert is trained on a residual with
   less target-direction variance to learn.
2. S1a lower bound > 0 alone: 20%.  Bottle pairings gain more than cheese (X3 > X2 in the point estimate vs RAA-o): 55%.
3. S2 (RAA-s beats RAA): 22%.
4. S3: RAA-s NON-INFERIOR to rgp (lower bound > -.10) 5%; beats rgp 1%.  rgp is about .90 on N1-N6.
5. SK: RAA-s non-inferior to RAA-o on K1-K6 (lower bound > -.05): 55%.  The servo is active on the known tasks' held phases too and demos disagree with it
   (per-task per-step variance ratio .79-1.03); the expert must cancel it, which it can, but 6,000 steps leave residual drift.  The R0 fit gate (.85) holds regardless.
6. Plumbing: the 12-task read finishes with no routed query that reached the model with an all-zero state, held-target channel median at test under 2 cm
   (SR 77: .94 cm), and the servo gate m = 1 on every held-target query (checked from the server's state log): 85%.
7. Exploratory mechanism (labelled): the median failed-release xy error on N4 is at most 4.5 cm (RAA-o 6.5, RAA-c 11.8): 30%; on N1 and N4 both at most 2.5 cm: 10%.  If
   S1a's lower bound is <= 0 and these errors stay at 6-12 cm, the hypothesis (the expert's mapping is the bottleneck and a built-in servo fixes it) is not supported at
   this gain and recipe: the release decision, not the commanded translation, sets where the object lands.
What would count against the claim: S1a lower bound <= 0 together with a failed-release xy error on N4 and N6 not below RAA-o's (6.5 and 7.0 cm).  What would count for it: S1a
lower bound > 0 with a median failed-release xy error below 5 cm on N4.  A gain with unchanged release error would mean the servo helps for another reason (for example it
shortens transit).  A loss on K1-K6 with a gain on N1-N6 would be a trade, not a fix.  A K1-K6 result near RAA-c's (.81) with N1-N6 near RAA-c's (.50) would say the state
token, not the head, dominates.

LAUNCH AND WAITER PLAN (written after the prereg row 2cf74b5e and the launch; nothing below changes a registered quantity).
Training: two redundant 6 x 4 h afterany chains of /coc/flash10/ksingara3/.claude_tmp/pp/raas_train.sh (RAA-c's script, log name changed), launched by raas_chain.sh:
overcap exp main jobs 4790861-4790866 (raas_main1-6), tail-lab exp tl jobs 4790867-4790872 (raas_tl1-6); args --g8v raas --steps 6001 --batch 32 --save-interval 250
--keep-period 1000; checkpoints /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raas_ae/{main,tl}/<step>.  Expected: RAA-c's chains needed about 5.5-6 h of A40
time over 2-3 jobs; RAA-s is the same model plus a few elementwise ops, so about 6-10 h wall from the start (2026-11-18 15:31 submission), i.e. step 6000 between
about 22:00 and 02:00 depending on the queue and the bad-node requeues of the RAA-c run.
Waiter (one background waiter, capped with `timeout`, re-armed on notification, stopped before the session ends): poll `ls .../pi05_g8v_raas_ae/main/6000 .../tl/6000`; on the
first to appear (1) scancel the other chain (jobs named raas_main* / raas_tl* still pending or running), (2) export with t16_export.sh, ARGS
`--g8v~raas~--exp~<main|tl>~--export~--step~6000~--tag~<main|tl>6000` -> .../pytorch/pi05_g8v_raas_ae_<main|tl>6000, (3) rollouts exactly as RAA-c (raa_roll.sh with
ARM=raac CONFIG=pi05_g8v_raas_ae NAME=raas, CKPT the export, the main states / branches and the rx states / branches, 12 tasks, 20 + 40 states, the overcap full exclude list),
(4) `python experiments/raas_read.py selftest` then `read` (sbatch; never on the login node), (5) the exploratory release-error check (raa_instr.py on N1 / N4 / N6 / K5 x 20
states, trials 24-43), (6) write SR 78 only after the read, and score the predictions above as written.  An exported checkpoint below step 6000 is not read.

# STEER-CONF: confirmatory test of k x a on Rrel and RAA (predictions, written 2026-11-17 before any probe of either policy)

Context: docs/PLAN_PAPER.md 2c ("Exploratory follow-up"), docs/SCAFFOLD_RESULTS.md 73 (Rrel, RAA; training jobs 4757776-99, checkpoints do not exist yet) and 74 (STEER).
SR 74's registered predictor (g_s x p_s) missed its bar by .008 on 90 cells. Its post-hoc variant k x a (5fec75f7, results/flawed/steer_v2_explore.json) reached AUC .787, Spearman .579 on the same 90 cells, chosen after seeing them among 6 variants x 3 multipliers, so that number is optimistic. This test uses policies that were not in any of those choices.

## Frozen definition (no refitting)
- Score: `k*a` of experiments/steer_v2_explore.py: K(policy) * a_grasp(o) * a_place(t); a = clip((mean steer(cf) - mean steer(true)) / (1 - floor), 0, 1) per policy x phase x swapped-in entity, floor = the policy's phase mean of steer(true); K = success share of the probe's own K1-K4, K6 episodes. The v2 file cannot be imported (it runs on import and refuses to overwrite its output), so experiments/steer_conf_read.py restates its lines verbatim inside `ka()`. Frozen by: sha256 of steer_v2_explore.py `37047092cba3bf45f8ccbdaa70adc8b6795e08b51f2c27ce7e5d6ac4d465f638`, of steer_read.py `541d7713151779a8dfc71b9a6d43458113f9fbb88aaee92e9978591aabba66f7` (the read refuses to run if either changed), and `steer_conf_read.py selftest`, which reproduces the k*a all-90 AUC, CI and Spearman of steer_v2_explore.json to 1e-9. The score of a policy depends on that policy's own records only, so the 90 old cells keep their v2 values.
- Probe: identical to SR 74 (scene 1; K1 K2 K3 K4 K6 = tids 0 2 4 6 10; the first 10 states of n2_branches_main, steer/branches_s1_main.json, k 4-13; same swap lists, same absent and nonsense words; d = summed xyz of the first 5 steps; steer by angle). Bootstrap B = 2000, numpy default_rng(0), tasks and policies jointly (steer_read.boot_cells / stat_block).
- Channel rule for side queries (new, needed because Rrel and RAA read a position suffix): the side query carries the channel the deployed system computes for the counterfactual instruction (lexicon parse of the swapped instruction, estimate of the swapped-in entity minus the end effector); for absent and nonsense words the true instruction's channel is kept. Side queries never step the client's Estimator; the executed episode is the unprobed one. Simulator positions go to the steer score only.
- Cells: Rrel, RAA x N1-N6, scene 1 = 12 new cells. Labels = SR 73 measured success >= .5 per (policy, task), mean over the 60 states of raa_read (dev trials 4-23 + new trials 24-63, familiar layout), computed by `raa_read.grid` from the rollout files. Old cells: targets.json.

## Registered tests (stated before any data)
- **C1.** AUC of k x a on the 12 new cells >= .75 (ties .5). The joint bootstrap interval is reported but does not gate (2 policies, 6 tasks).
- **C2.** Pooled AUC over the 90 old + 12 new cells, score frozen. PASS iff the lower bound of the joint bootstrap > .5.
- **C3.** Sign test: sign(mean k x a over N1-N6, Rrel - RAA) equals sign(mean measured N1-N6 success, Rrel - RAA). One pair, so chance is .5; C3 cannot carry a claim alone.
- **Fallback.** With 12 cells and possibly one class missing: fewer than 12 scoreable cells, or all 12 labels in one class, makes C1 NOT TESTABLE, and an undefined AUC is reported as NOT TESTABLE, never as a pass. A tie in C3 is NOT TESTABLE. C2 is NOT TESTABLE only if its AUC is undefined.
- **Overall.** "k x a confirmed on new policies" iff C1 and C2 PASS. Anything else is reported as it is and k x a stays a reported post-hoc variant. Whatever happens goes into SR 75 and PLAN_PAPER 2c.
- **Discipline.** The labels (SR 73 read) may be seen before the probe runs; that is harmless because the score is frozen and unfitted. The score code, the probe settings and the channel rule may not change after any label or any probe output of Rrel or RAA is seen. Plumbing fixes before the main probe need a smoke on K1 only and are logged as deviations.

## Predictions (probabilities mine, from the existing record)
Reasoning: k x a at policy level is nearly the same for Rrel and RAA if both fit K1-K6 (K ~ .9), so the 12-cell ranking is driven by the entity factors (wine bottle and bowl swaps low, cabinet high) and by whichever policy is more steerable; the SR 74 reading says the 90-cell signal was mostly entity difficulty, and the labels also fall on the bottle tasks, so entity structure helps AUC here too. The v2 point value is a selected maximum, so I shrink it.

- P1 Both classes present among the 12 labels (C1 testable): .85. (R0 N1-N6 .025 and R2 .425 in SR 37; Rrel and RAA each have at least one task >= .5 and at least one < .5.)
- P2 C1 outcome: PASS .38, FAIL .47, NOT TESTABLE .15. Conditional on testable, AUC >= .75 has P .45.
- P3 C2 PASS (lower bound > .5 on 102 cells) .70. Pooled point AUC in [.70, .82]: .65. Pooled point AUC below the old .787: .60.
- P4 C3 PASS .62 (RAA, with R2's painted views, is both more steerable and more successful than Rrel in the ladder pattern R2 > R0).
- P5 C1 and C2 both PASS (the registered confirmation): .27.
- P6 Measured: RAA mean N1-N6 success > Rrel's .75; RAA >= .40 .50; Rrel >= .25 .40; both K1-K6 success >= .85 .70; mean k x a RAA > Rrel .65.
- P7 Trained-task success K alone gives AUC within .15 of .5 on the 12 cells (K nearly equal for both): .70. Descriptive, not gated.
- P8 The channel suffix makes the counterfactual steer high for both policies (a above the 90-cell median for the grasp phase): .55; this would compress k x a differences between them and make C3 noisy.

## Runbook for phase 2 (run only after the checkpoints exist; commands from the repo root, nothing here has been run)
0. Preconditions (stop and report if any fails): the ledger row `ledger/m3_prereg/m3_prereg__libero_goal__steer_conf__s0__*.json` exists and is committed; `ls /coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/ | rg 'g8v_(rrel|raa)'` shows the exported dirs (one chain exported at step 6,000; the other cancelled); `results/flawed/steer/s1_rrel` and `s1_raa` do not exist; no edit to `.sh` files used by a running job.
1. Probe adapter (one piece of new code, plumbing only, smoke before main): the SR 74 client `steer_probe.py` knows arms I, IR, tv, tvp, tvpb, tvb. Rrel and RAA clients are `raa.py client` (Estimator stepped per query, channel from `meta/raa_*`). Add an arm `RA` to steer_probe.py that wraps `raa.py client` and implements the channel rule above, and a roll script `steer_conf_roll.sh` in /coc/flash10/ksingara3/.claude_tmp/pp/ (copy of steer_roll.sh; new name; never edit steer_roll.sh) with cases
   - `rrel`: `SRV="$SS $REPO/experiments/raa.py serve --arm rrel --ckpt $CK/<rrel dir> --seed 13 --seg-ckpt $SEGCKPT"`, ARMS=RA
   - `raa`: the same with `--arm raa`, ARMS=RA, SEGF=--seg
   The server is wrapped by `steer_serve.py experiments/raa.py serve ...` exactly as the other servers (side queries restore call counter, query index and role-detector state; RAAServer keeps only counters `n`, `noch` that do not affect outputs; check that `STATE` in steer_serve.py covers `RAAServer.cur`, which is overwritten by every query, and `LearnedMasks` state for raa).
   Smoke (K1, 2 states): `PHASE=smoke POLICY=rrel` and `raa` through the same sbatch line as step 2 with `--export=ALL,POLICY=<p>,SCENE=1,PHASE=smoke,TIDS=0` (output `results/flawed/steer_smoke/`); confirm side prompts carry the swapped entity's channel, the executed episode equals an unprobed `raa.py client` episode on the same state (same trial, same seed), and every client exits 0. Commit the adapter and smoke note before step 2.
2. Main probe, one job per policy (each 12 h, one A40; same exclude list as SR 74):
   ```
   cd /coc/flash10/ksingara3/.claude_tmp/pp
   for P in rrel raa; do sbatch -p overcap -A overcap --gres=gpu:a40:1 -c 6 --mem=48G -t 12:00:00 --requeue \
     --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax \
     --export=ALL,POLICY=$P,SCENE=1,PHASE=main -o steer_conf_mn_${P}_%j.out steer_conf_roll.sh; done
   ```
   Output `results/flawed/steer/s1_rrel` and `s1_raa` (a finished file is skipped on requeue; never overwrite). The main PHASE guard must look for the steer_conf row.
3. Labels: the SR 73 read must have run (`raa_read.py read`, which refuses to overwrite). `steer_conf_read.py` recomputes success from the same rollout files and does not need `raa_read.json`, but compare the two per-task numbers before reporting.
4. Check, then the confirmatory read (CPU, openpi venv, writes `results/flawed/steer_conf_read.json`, refuses to overwrite):
   ```
   cd /coc/flash10/ksingara3/.claude_tmp/pp
   sbatch -p overcap -A overcap -c 2 --mem=16G -t 00:30:00 --export=ALL,MODE=check steer_conf_cpu.sh
   sbatch -p overcap -A overcap -c 2 --mem=16G -t 00:30:00 --export=ALL,SCRIPT=experiments/steer_conf_read.py,ARGS=read steer_conf_cpu.sh
   ```
   The job reports COMPLETED even when the script fails: read `steer_conf_<jobid>.out` for `exit=0` and confirm the json exists with `ls`.
5. Write SR 75 (rows C1-C3 verbatim from the json, the predictions above scored one by one, deviations), update PLAN_PAPER 2c, commit by file name.

# NPH results (plan: docs/notes/plan_necessity_power.md, tag `nph`)

## RESUME NOTE (written 2026-11-23 21:20; the run is unattended from here)

**State.**  Head fit and gates passed; checks and smoke passed; registration row committed (ledger/m3_prereg/m3_prereg__libero_goal__nph__s0__66315325.json, commit 7865e062, 2026-11-23 17:47:54);
6th set built (results/flawed/n2_states_w6, n2_branches_w6.json, k 820-879; disjointness job 4847576: 0 equal rows, geometry results/flawed/rva1/geom_w6.json).  pi0-H* training is running.

**Job IDs and dependency order** (all gates are inside the job scripts: a gated job exits 0 without work; scripts in docs/notes/nph_jobs/, run by `.claude_tmp/pp/nph_pipeline.sh`, ids in `.claude_tmp/pp/nph_pipeline_ids.txt`):
1. Training chain nph_tl1..tl6 = 4847565-4847570 (tail-lab, afterany chain, 6,001 steps at ~4.4 s/step, ~7.3 h; was at step ~2900 at 21:17).
2. 4848465 export (afterany 4847570; runs only if jax/pi0_g8v_raah_l_ae/tl/6000 exists) -> robot/checkpoints_l90c/pytorch/pi0_g8v_raah_l_ae_tl6000.
3. 4848466 pcheck (tail-lab GPU; raah0_check.py --torch-ckpt: PI0PytorchZH = PI0Pytorch + head; PolicyBackboneH0 constructs).
4. 4848467, 4848468 pi0-H* K jobs (K1-K3, K4-K6), cap guard 21.5 wall-h of nph_* jobs.
5. 4848469 readgate -> results/flawed/nph6/nph_read_gate.json (`"passes": true` iff pi0-H* K1-K6 >= .70; otherwise UNTESTABLE and every later gated job skips).
6. 4848470, 71 pi0-H* N jobs; 4848472-75 pi0-A0 (4 groups); 4848476-79 pi0-sL (4 groups); all afterany readgate, run only if the gate passed and the cap holds.  Outputs results/flawed/nph6/a_<pzH|pzA0|pzsL>/<arm>_main_<tid>.json (+ chan_/where_ logs).
7. 4848480 readmain -> results/flawed/nph6/nph_read_main.json (n1 primary, n1r, n2, d1'', nK, sensitivities, Holm); runs only if all 36 cells exist (kill k4).
8. 4848481-86 E1 (results/flawed/nph6/e1_<arm>.json and e1_dec_<a>_<b>.json; pi0-H* class-H share and decomposition vs pi0-A0).

**How to read.**  Look at nph_read_gate.json first (`gate.K_mean`, `passes`); then nph_read_main.json: `contrasts` keys "n1 PRIMARY ..." (label SUPPORTED / FLAGGED / REVERSED / EQUIVALENT / NOT SHOWN, `ci`, `flagged_tasks`), "n1r ...", "n2 ...", "d1'' ...", "nK ...", "sens ...", `holm`.
Compare to the predictions in experiments/prereg/nph_pred.txt (p0-p8).  Then fill the sections below.

**What to check.**  (a) `sacct -u $USER -S 2026-11-23 -X --format=JobName,Elapsed,State | grep nph_` : GPU-h used vs the 23 cap (wrappers skip rollouts above 21.5 wall-h of nph_* jobs; training chain ~7.4 h).  (b) any FAILED roll job (kill k4: no read until all cells exist; resubmit with the same script, a finished cell is skipped).
(c) where_pzH_*.jsonl logs show nonzero head_abs_max (serving smoke); pcheck output in .claude_tmp/pp/nph_gated_nph_pcheck_*.out.  (d) If the chain ended before step 6000 (node trouble), resubmit `nph_chain.sh` (resumable) and rerun the pipeline script after fixing the dependency ids.
Do not edit any .sh under a queued job.  Do not touch results/l90c/split_sealed.json.  Nothing was pushed.

## Gate metrics (seen at registration)
Head gates on the 29 validation demos (1,320 held rows; results/flawed/raah0/fit.json): H* G1 mean cos .9709 (>= .80), G2 median ratio .8888 (>= .60), G3 median |h| .0330 (<= .05): all pass.  Registered H on the same rows: G1 .6777, G2 4.611, G3 .3890 (fails all three).
Val R2 of the chunk target .9613; npz sha256 ee217430c8ba3fcd85384d99cc76524ac1ff357000eafe87cfd9e1593a29fabe.  Checks (job 4847548) and smoke (4847523, step-0 loss .1743 vs pi0-sL .1718) passed.

## Predictions written vs outcome, deviations, GPU-h per item
TO BE FILLED when the pipeline finishes.

Deviations so far (honest list):
1. An attempt to run `python3` on the login node (an empty heredoc) was issued by mistake and blocked by the permission layer before it ran; nothing was executed.
2. The registration asserts "jax/pi0_g8v_raah_l_ae absent" was changed to "jax/pi0_g8v_raah_l_ae/tl absent": the 20-step smoke (a plan step before registration) created the `smoke` experiment in that directory.
3. Check attempts 1 and 2 failed (GPU OOM at model creation; a too-strict loss test that a float64 shift broke under bf16) and were re-run (3 attempts, .1-.2 GPU-h each); the SEEN field says so.
4. nph_read.py imports rvw5_read.py instead of copying it; the plan's `c_train_x.py --name` became the copy c_train_nph.py (the original restricts --name).
5. The unattended pipeline wrappers (nph_gated.sh, nph_pipeline.sh) were written after registration at the orchestrator's request; they carry gates and a 21.5 wall-h cap only, no statistics.  They are not hashed in the row.
6. fit.json and other results/ files are untracked (results/ is gitignored); only the npz and code are committed.

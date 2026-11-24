# NPHS results (multi-seed NPH, tag `nphs`; prereg experiments/prereg/nphs_pred.txt)

## RESUME NOTE (written 2026-11-24 10:15; the run is unattended from here)

**State.**  Row committed: ledger/m3_prereg/m3_prereg__libero_goal__nphs__s0__3e4670b2.json, commit 597a5cfe (10:09:22); pipeline submitted ~10:11 (137 jobs, all gated).  Nothing pushed.
Design: pi0-H*, pi0-A0, pi0-sL x seeds 42 (existing tl6000 checkpoints) + 43, 44, 45 (9 new trainings, exp `tl_s<seed>`, `--seed`; export tag `s<seed>_6000` -> robot/checkpoints_l90c/pytorch/<cfg>_s<seed>_6000); 7th set k 880-939 (results/flawed/n2_states_w7, n2_branches_w7.json).
Cap 130 GPU-h (guard in nphs_train.sh / nphs_gated.sh counts nphs_* GPU wall-h since 2026-11-24; starts skip above 130).  Expected ~115-125 GPU-h; wall ~15-20 h.

**DAG** (job ids in `.claude_tmp/pp/nphs_pipeline_ids.txt`; scripts docs/notes/nphs_jobs/, driver nphs_pipeline.sh):
1. states 4851989 -> disjoint 4851990 (afterok; kill k3 if equal row) -> geom 4851991.  Every rollout job has afterok:disjoint.
2. Training chains (6 x 4 h, afterany; last link): pzH43 4851997, pzH44 4852005, pzH45 4852013, pzA0_43 4852021, pzA0_44 4852028, pzA0_45 4852035, pzsL43 4852042, pzsL44 4852049, pzsL45 4852056.  Tail-lab: pzH43-45 + pzA0_43 (4 GPUs); the other five on overcap.
3. Exports (afterany last link): 4851998, 4852006, 4852014 (H*), 4852022, 4852029, 4852036 (A0), 4852043, 4852050, 4852057 (sL).  H* pcheck (overcap, deviation from tail-lab): 4851999, 4852007, 4852015.
4. Rollouts (48 jobs `nphs_a_<arm>_<seed>_<tids>`, ids in the ids file as roll_*): K groups 0+2+4, 6+8+10 and N groups 16+18+20, 22+24+26 for each of 12 checkpoints; seed 42 runs as soon as the set is checked.  H* N jobs run only if that seed's gate passed.
   Gates readgate_<s> = 4852060 (42), 4852073 (43), 4852086 (44), 4852099 (45) -> results/flawed/nphs7/nphs_gate_s<s>.json (`gate.passes`, `K_mean`).
5. readmain 4852110 (CPU) -> results/flawed/nphs7/nphs_read_main.json; runs only if every expected cell exists (k4).
6. E1: per (arm, seed) 4852111-4852114 (H*), 4852116-19 (A0), 4852121-24 (sL) -> e1_<arm>_s<seed>.json; pooled per arm 4852115 (H*), 4852120, 4852125 -> e1_<arm>_pooled.json (`share.H` = class-H share, prediction >= .50 for H*).

**Outputs.**  results/flawed/nphs7/a_<arm>_s<seed>/<arm>_s<seed>_main_<tid>.json (arm pzH|pzA0|pzsL; K tids 0 2 4 6 8 10, N tids 16 18 20 22 24 26) + chan_/where_ logs.
**How to read.**  nphs_gate_s*.json first (an H* seed with `passes: false` is UNTESTABLE, its pair dropped; fewer than 3 passing -> n1s UNTESTABLE).  Then nphs_read_main.json: `contrasts` keys "n1s PRIMARY ...", "n2s ...", "d1s ...", "nKs ..." (label, delta, ci, per_seed_delta), `holm`, `seed_sd` (per arm, N and K; replaces sigma .033), `success`, `valid_hstar_seeds`.  Compare with nphs_pred.txt.
**What to check.**  (a) `sacct -u $USER -S 2026-11-24 -X --format=JobName,Elapsed,State,AllocTRES | grep nphs_` : GPU-h vs 130.  (b) FAILED roll jobs: resubmit the same job (a finished cell is skipped), then resubmit readmain (readgate if a gate json is missing); FAILED/short training links: the chain is resumable (exp tl_s<seed>), resubmit nphs_chain.sh with the same CFG/SEED (a link skips when step 6000 exists).
(c) disjoint 4851990 must print TOTAL EQUAL ROWS 0; if it fails every rollout stays DependencyNeverSatisfied (scancel them).  (d) where_pzH_s*.jsonl head_abs_max nonzero; pcheck outputs `.claude_tmp/pp/nphs_check_*.out`.
Do not edit any .sh under a queued job.  Nothing touched in paper/ or l90c_config.py.

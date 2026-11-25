# NPHP results (pi0.5 four-seed H* vs A0 vs RAA-sL, tag `nphp`; prereg experiments/prereg/nphp_pred.txt)

## RESUME NOTE (written 2026-11-24 22:20; the run is unattended from here)

**State.**  Code commit 03e2f4f6; prereg row ledger/m3_prereg/m3_prereg__libero_goal__nphp__s0__03e2f4f6.json, commit 8d44ddbd (22:14:34); pipeline submitted 22:15 (143 nphp_* jobs).  Nothing pushed.
Design: pi0.5 H* (`pi05_g8v_raah0_l_ae` = A0 `pi05_g8v_raac_l_ae` + frozen head H* `raah0_head.npz`, add-only in l90c_config.py, model lifelong_ood/policy/raah0p_model.py), A0, RAA-sL (`pi05_g8v_raas_l_ae`), seeds 42-45.
Reused checkpoints: A0 s42 (pi05_g8v_raac_l_ae_tl6000), RAA-sL s42 (..._main6000) and s43 (..._s43_6000).  9 new trainings (exp `tl_s<seed>`, export tag `s<seed>_6000`): H* 42-45, A0 43-45, sL 44-45.
H* head: raah0_head.npz reused, NOT refit (raw-unit function of the object-to-target offset, fit on the same raac held rows with the same norm stats; G1-G3 passed at its fit, results/flawed/raah0/fit.json).  8th set k 940-999 (results/flawed/n2_states_w8, n2_branches_w8.json).
Cap 150 GPU-h (guard in nphp_train.sh / nphp_gated.sh counts nphp_* GPU wall-h since 2026-11-24; starts skip above 150).  Expected ~140 GPU-h (9 x ~11 train + ~40 rollouts) -- close to the cap; if it trips, trailing rollouts skip and readmain stays unread (k4).  Wall ~15-20 h.

**DAG** (job ids in `/coc/flash10/ksingara3/.claude_tmp/pp/nphp_pipeline_ids.txt`; scripts docs/notes/nphp_jobs/; drivers nphp_checks.sh then nphp_pipeline.sh):
0. Pre-flight: cfgcheck + py_compile ran by hand (passed, 36 names construct).  ident 4857201 (raah0p_check.py, GPU, tail-lab) -> smoke_train 4857202 (31 steps, exp smoke) -> smoke_export 4857203 -> smoke_pcheck 4857204 -> checks_ok 4857205 (writes marker `.claude_tmp/pp/nphp_checks_ok`).  Training links skip (exit 0) without the marker.
1. states -> disjoint (afterok; TOTAL EQUAL ROWS must be 0, else kill k3) -> geom (rva1/geom_w8.json).  Every rollout job has afterok:disjoint.
2. Training chains (6 x 4 h, afterany; first link afterok:checks_ok): H* 42-45 on tail-lab (4 GPUs); A0 43-45, sL 44-45 on overcap.  Exports afterany last link; H* pcheck (overcap) after each export.
3. Rollouts: 48 jobs `nphp_a_<arm>_<seed>_<tids>` (K 0+2+4, 6+8+10; N 16+18+20, 22+24+26), arms pqH / pqA0 / pqsL; existing checkpoints start at once.  readgate_<s> (H* K gate) before the H* N jobs.
4. readmain (CPU) -> results/flawed/nphp8/nphp_read_main.json (only if every expected cell exists); E1 per (arm, seed) and pooled -> e1_<arm>_pooled.json.

**Outputs / how to read.**  results/flawed/nphp8/a_<arm>_s<seed>/<arm>_s<seed>_main_<tid>.json (K tids 0 2 4 6 8 10, N tids 16 18 20 22 24 26).  First nphp_gate_s*.json (`gate.passes`; an H* seed below .70 K is UNTESTABLE, pair dropped; fewer than 3 valid -> p1 UNTESTABLE).
Then nphp_read_main.json: `contrasts` keys "p1 PRIMARY ..." (H* - A0, N), "p2" (H* - sL, NI -.10), "pd" (A0 - sL, SUPPORTED iff upper < 0), "pK" (H* - A0 on K, NI -.05); each has label, delta, ci, per_seed_delta and `paired_t` (descriptive); `holm` over {p1, pd}; `seed_sd`; `valid_hstar_seeds`.  E1: e1_pqH_pooled.json `share.H` vs prediction >= .30.
Predictions (nphp_pred.txt): p1 +.12 SUPPORTED 50%; p2 -.06 NI 40%; pd -.15 SUPPORTED 60%; H* class-H share on N2 >= .30 50%.

**What to check.**  (a) `sacct -u $USER -S 2026-11-24 -X --format=JobName,Elapsed,State,AllocTRES | grep nphp_`: GPU-h vs 150.  (b) ident / smoke_pcheck outputs `.claude_tmp/pp/nphp_check_*.out` must end "done"/"4 ok"; if any pre-flight job failed the marker never appears and nothing trains (fix, then resubmit nphp_checks.sh and re-run only the chain links).
(c) disjoint must print TOTAL EQUAL ROWS 0.  (d) FAILED roll jobs: resubmit the same job (finished cells are skipped), then readmain.  Training links: resumable, resubmit nphp_chain.sh with same CFG/SEED/LBL/CHAIN (a link skips when step 6000 exists).
(e) where_pqH_s*.jsonl `head_abs_max` nonzero.  Do not edit any .sh under a queued job.


## 2026-11-25 resubmission (after the disk-full failure)

**What happened.**  /coc/flash10 hit 100% on 2026-11-24 night; 117 nphp jobs died at startup (0:53), the four H* tail-lab links (4 h each, about 16 GPU-h) timed out with their checkpoints lost, and another session then deleted the jax checkpoint dirs.  Of the H* dirs (`pi05_g8v_raah0_l_ae/tl_s42..45`) only empty directories survive, so **all 9 trainings restart from step 0** (no resume possible).  Same registered design, prereg row 8d44ddbd and contrasts untouched.
**Kept (not resubmitted, outputs not touched).**  Pre-flight (cfgcheck, ident, smoke train/export/pcheck, checks_ok marker), prereg, states, disjoint (TOTAL EQUAL ROWS 0), geom; Phase A rollouts for pqsL s42 and s43 (all 12 cells), pqA0 s42 (K and N 16/18/20, cells 22 and 24); E1 for pqsL s42/s43.  These came from the reused pytorch exports (pi05_g8v_raac_l_ae_tl6000, pi05_g8v_raas_l_ae_main6000, ..._s43_6000), which still exist with model.safetensors and were not modified.
**Re-run.**  pqA0 s42 N 22+24+26 (only cell 26 missing; job below), 9 trainings, 9 exports, 4 H* pcheck, 4 readgate, all other rollouts, readmain, E1.  Scripts are new copies in docs/notes/nphp_jobs_r2/ (old ones untouched); pipeline docs/notes/nphp_jobs_r2/nphp_pipeline.sh; ids in `.claude_tmp/pp/nphp_pipeline_ids_r2.txt` (131 lines).  Training keep-period 6000 (final step plus the rolling save only, to save disk).  H* chains on tail-lab (4 GPUs), everything else overcap, full node exclusion list.
**Deviations.**  (1) disk-full failure, lost H* progress and lost checkpoints: restart from scratch, about 16 GPU-h of H* wall time wasted but counted against the cap; (2) keep-period 1000 -> 6000 (disk only, no effect on the trained model); (3) first links no longer wait on checks_ok (marker already exists).
**Budget.**  Spent so far 23.1 GPU-h (cap 150).  Projected: 9 x about 11 h training = 99, plus about 25 h for the 37 remaining rollouts = about 147 total.  Within the cap but with about 3 h margin; the guard in the scripts counts all nphp_* GPU time since 2026-11-24 and skips late rollouts if it trips (then readmain stays unread, k4).  Wall time: 3 training links in parallel per arm group (H* 4 on tail-lab, 5 on overcap, queue-dependent), about 12-24 h to training end, then about 8-10 h of exports and rollouts, so about 22-34 h from submission, queue permitting.

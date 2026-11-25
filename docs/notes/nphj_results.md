# NPHJ results (tag `nphj`; prereg experiments/prereg/nphj_pred.txt; review items 2 and 3 of docs/notes/review_atlas_adv2.md)

## RESUME NOTE (written 2026-11-24 22:45; the run is unattended from here)

**State.**  Prereg row ledger/m3_prereg/m3_prereg__libero_goal__nphj__s0__b530178c.json, commit 9be0fd1f (22:38:03); pipeline submitted 22:40 (82 jobs planned, 80 queued, all dependent, see deviations).  Nothing pushed.
Design: **A0-24k** (pi0_g8v_raac_l_ae, 24,001 steps, seeds 42-43, exp t24k_s<seed>, export tag t24k_s<seed>, pytorch dir pi0_g8v_raac_l_ae_t24k_s<seed>; keep-period 6000 so the 6k/12k/18k checkpoints stay) and **J** (pi0_g8v_raaj_l_ae = pi0-A0 + trainable where-branch MLP 3-64-64-30 in the servo slot, lifelong_ood/policy/raaj_model.py; 6,001 steps, seeds 42-45, exp tj_s<seed>, tag s<seed>_6000).  Comparators = existing results/flawed/nphs7 (pzA0, pzsL, seeds 42-45).  7th set k 880-939 (no new set).
Cap 110 GPU-h (guard in nphj_train.sh / nphj_gated.sh counts nphj_* GPU wall-h since 2026-11-24, starts skip above 110).  Estimate: A0-24k about 2 x 29 h, J about 4 x 7.5 h, rollouts about 20 h, so about 108-112; late rollouts may skip at the cap.
**Pre-training checks passed** (check job 4858929, marker results/flawed/nphj7/checks_ok.marker): tree = A0's 50 tensors + 6 where_head, same 32 frozen; step-0 J output exactly 0 and J == A0 on sample and loss (diff 0); random-weight jax vs torch 7e-7; 30-step smoke; export tensors equal the JAX ones; export parity 4e-11; PI0PytorchZJ = PI0Pytorch + branch.  J training links exit 0 unless the marker exists.
**DAG** (job ids in .claude_tmp/pp/nphj_pipeline_ids.txt; scripts docs/notes/nphj_jobs/, driver nphj_pipeline.sh; everything on overcap):
1. Training chains (afterany): pzA24_42 last link 4859393 (10 links), pzA24_43 4859406 (8 links, see deviations), pzJ_42..45 4859416 / 4859426 / 4859436 / 4859446 (4 links; first link after check job).  A link whose final step exists exits 0.
2. Exports (afterany last link): 4859394, 4859407, 4859417, 4859427, 4859437, 4859447.  J exports also write results/flawed/nphj7/jmag_s<seed>_6000.json (branch magnitude at 5 cm vs servo).
3. Rollouts `nphj_a_<arm>_<seed>_<tids>`: K groups 0+2+4, 6+8+10; readgate_<arm>_<seed> -> results/flawed/nphj7/nphj_gate_<arm>_s<seed>.json (passes = K mean >= .70); N groups 16+18+20, 22+24+26 run only if the gate passed.
4. readmain 4859453 -> results/flawed/nphj7/nphj_read_main.json (only if every expected cell exists, k4); E1 per (arm, seed) and pooled -> e1_<arm>_s<seed>.json, e1_<arm>_pooled.json (share.H).
**Outputs.**  results/flawed/nphj7/a_<arm>_s<seed>/<arm>_s<seed>_main_<tid>.json (arm pzA24 | pzJ) + chan_/where_ logs.
**How to read.**  nphj_gate_*.json first (a failed gate drops that seed; fewer than 3 J seeds -> j1, j2, jK UNTESTABLE).  Then nphj_read_main.json: `contrasts` (u1, u2, j1 PRIMARY, j2, jK), `holm`, `seed_sd`, `descriptive` (paired t), `j_magnitude_5cm_over_servo`.  Compare with nphj_pred.txt.
**What to check.**  (a) `sacct -u $USER -S 2026-11-24 -X --format=JobName,Elapsed,State,AllocTRES | grep nphj_` GPU-h vs 110.  (b) FAILED roll jobs: resubmit the same job (a finished cell is skipped), then readmain.  Short/failed training chains: resubmit `CFG=.. SEED=.. LBL=.. EXP=.. STEPS=.. KEEP=.. NL=.. bash docs/notes/nphj_jobs/nphj_chain.sh`, then the export job (resumable, exp dir is kept).  (c) J: check where_pzJ_s*.jsonl head_abs_max nonzero.
Do not edit any .sh under a queued job.  Nothing touched in paper/.  l90c_config.py edit was add-only (name, AE entry, one branch); CPU cfgcheck (experiments/raaj_cfgcheck.py, job 4857356) passed: all 37 names construct.

**Deviations.**
1. Code-debug jobs ran before the prereg commit: cfgcheck 4857356 (CPU), check jobs 4857361 (identity failed on a .numpy() on a grad tensor; fixed) and 4858929 (passed, wrote the marker).  No rollout or result existed; the prereg row's deviation_notes say so.
2. The configured "cosine" schedule has decay_steps 1,000,000 with decay_lr = peak, so it is constant 5e-5 after warmup; A0-24k therefore changes only --steps (24,001), nothing to stretch.
3. Two sbatch submissions failed transiently ("Zero Bytes were transmitted"): links 1-2 of the pzA24_43 chain were not submitted, so that chain has 8 links (link 3 starts first, no dependency); 8 x about 3,200 steps just covers 24,001, so a short chain needs one resubmission.
4. J output scale = H*'s bound M (.9375) x linear last layer, input / H*'s scale (.2); no tanh or clip (fixed constants of scaling, no fitted form).
5. Seeds of the comparators: all four existing seeds for every contrast (independent seed resampling per arm); matched-seed paired t is descriptive only.

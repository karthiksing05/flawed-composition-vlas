# XB status (pre-registration experiments/prereg/xb_pred.txt, commit 3b1bca90)

Scripts: `experiments/xb_cells.py`, `xb_roll_{mesa,robocasa,robotwin}.py`, `xb_serve_groot15.py`, `xb_predict.py`, `xb_read.py`, `xb_classes.py`; job scripts and the job id table in `docs/notes/xb_jobs/`
(`xb_jobids.tsv`). Logs: `/coc/flash10/ksingara3/.claude_tmp/pp/xb_*`. Outputs: `results/bench/xb/` (cells_*.json, reset/, roll/, rt_seeds/, rt_patched/, pred_*.json, read_main.json).
Ledger rows (written before any cell draw or rollout, job 4875946): `ledger/m3_prereg/m3_prereg__{mesa,bimesa,robocasa365,robotwin2}__xb__s0__3b1bca90.json`.

## Pipeline state
See the last section (updated by hand at each check).  Order followed: ledger -> cells -> policy-free checks (forced pair, placeability, expert-seed filter, reset-state pass) -> partner rollouts ->
predictions (before any held-out rollout; each held-out job refuses to start without `pred_<bench>.json`) -> held-out rollouts -> `xb_read.py`.

## Deviations and clarifications (every one is a choice the file did not fix, or a departure forced by the cluster)
1. **Dependencies.** `sbatch --dependency=afterok:...` is rejected on this cluster ("Job dependency problem") and `afterany` was not honoured (a job with `afterany` on pending jobs started at once).
   Held-out rollout jobs are therefore guarded inside the wrapper (`xb_roll_*.sh` exit 4 unless `results/bench/xb/pred_<bench>.json` exists) and are submitted only after the prediction job has finished.
   One premature `cells_robotwin_final.json` (6 cells, written by a job that ran before its prep jobs) was renamed `INVALID_premature_cells_robotwin_final.json` and regenerated; no rollout had read it (4 RoboTwin jobs submitted with that dependency were cancelled before starting).
2. **Lost jobs.** About 25 jobs that sacct still shows PENDING vanished from squeue; they were resubmitted by `docs/notes/xb_jobs/submit_stage1.sh` (episodes are resumable: an existing episode json is skipped).
3. **RoboCasa placeability check, first run invalid.** My snapshot code called `env.env` on a gym wrapper and every spawn "failed" with an AttributeError; the 0-cell output was renamed `INVALID_bug_cells_robocasa_final.json`
   and the check was rerun (job xb_rc_check2) with the unwrapped env.  Replacement logic is unchanged (3 resets, next draw).
4. **RoboCasa training data for the zero-count cells.** The held-out and partner cells are drawn from the 300-task human data (`pepijn223/robocasa_pretrain_human300_v4` audit), not the atomic-only mirror,
   because the released checkpoints also saw the composite tasks.  Cells are (object category, family) with the object forced by `obj_groups=<category>`.
5. **RoboCasa X-VLA (deksprime/xvla-base-robocasa-human300-ri) and the other models:** see "Models" below.
6. **RoboTwin expert filter.** The expert-solvable seeds per forced pair are found once, policy free (`xb_roll_robotwin.py --mode prep`: scan seeds from 100000, run the scripted expert, keep the first 10 it solves),
   and stored with the episode_info used for the language instruction; policy rollouts replay those seeds without re-running the expert.  The registered rule (same seeds for held-out and partner cells, expert failures skipped) is unchanged.
   Prep covers the first 24 draws per type; the first 20 feasible draws are the cells (`xb_cells.py robotwin_final`).
7. **v2 predictor not applied** (the file says "where the probe exists"; no language-attention probe exists for any xb model).  Both frozen files were hash-checked (6bd3f414, f00ec7c1).
8. **v1 features.** f_obj_targets / f_tgt_holders = training partner counts from the same inventory the cells use (log inside the frozen model); f_tc_conflict = the object has a trained partner receptacle other than the instructed one;
   geometry features use the xy distances at reset between the instructed object and the receptacles present in the scene (MESA: objects whose category contains a receptacle word; RoboCasa: fixtures); RoboTwin has no competing receptacle,
   so f_nearer_conf / f_near_margin / f_instr_rank take v1's impute values.  Where the instructed object or receptacle cannot be matched by name the episode takes impute values (matched share is logged in `pred_<bench>.json`).
   v0 cue vote = `fp_build.vote_score(votes, instruction, 2)` with votes {nearest receptacle: 1} + {each trained partner receptacle: +1}.  b2 score = 1 - base rate.
9. **Failure-class thresholds** (not in the file): lifted = 2 cm; held at the end = within 8 cm of the end effector (RoboTwin: still 2 cm above rest); on a non-instructed receptacle = within 10 cm xy of its centre
   (RoboCasa: inside half the fixture size + 5 cm).  See `experiments/xb_classes.py`.
10. **Python on the login node.** I ran one empty `python3 - <<EOF` heredoc by mistake while editing a file (no script, no memory use); everything else ran under sbatch.
11. **Unwanted smoke episodes.** The first MESA pi0.5 smoke job wrote 4 partner episodes into `roll/mesa/pi05/apple_tray_on`; they are valid partner episodes of the same protocol and are kept.

## Models (state at the first status check, 2026-11-26 ~04:40)
| Benchmark | Model | Server / env | State |
|---|---|---|---|
| xbM MESA | pi0, pi0.5, pi0-FAST | openpi venv + `openpi-mesa/src` on PYTHONPATH | partner rollouts running; pi0.5 partner complete (2,950 episodes) |
| xbM MESA | GR00T N1.6 | mesa-GR00T (`bench/mesa/groot_venv`, flash-attn built) | smoke OK; partner rollouts running |
| xbB BiMESA | pi0.5, pi0, pi0-FAST, GR00T N1.6 | same servers, bimanual configs | pi0.5 partner running clean, others queued; smoke (GR00T) OK |
| xbC RoboCasa365 | pi0, pi0.5 | openpi venv + fork (`bench/robocasa/openpi-robocasa`) + `shim/` that mocks `robocasa.*` for the server | partner rollouts running |
| xbC RoboCasa365 | GR00T N1.5 | `bench/robocasa/groot15_venv` (torch 2.5.1, flash-attn wheel), HTTP server `xb_serve_groot15.py`, client adapter in `xb_roll_robocasa.py` | env imports OK; smoke job queued (candidate: activated by the driver only after an episode exists) |
| xbC RoboCasa365 | X-VLA (community) | not wired | **excluded**: the checkpoint's input state is 8-dimensional and its images are 256/256/224 while the dataset it cites (`pepijn223/robocasa_pretrain_human300_v4`) has a 16-dim state; no documented mapping, no published number to reproduce (k1-type exclusion) |
| xbC | SmolVLA, GR00T N1.6 community, lerobot/pi05_robocasa | - | excluded in the registration |
| xbT RoboTwin | X-VLA (cotrain, ee) | xvla_env + XPolicyLab server | partner rollouts running clean (first episodes successful) |
| xbT RoboTwin | pi0.5 (cotrain, joint) | openpi venv + `Pi_05/openpi/src` + `extra_pylibs` (msgpack-numpy, websockets) | smoke OK (episode succeeded); partner jobs queued |
| xbT RoboTwin | SmolVLA | lerobot 0.4.4 venv + `extra_pylibs` | smoke queued (candidate) |

RoboCasa cell shortfall (policy-free placeability check, 3 resets per object): counter->microwave 2 of 10 held-out cells spawn (canned food, tupperware), counter->sink 5 (rolling pin, whisk, pan, tomato, mug), counter->cabinet 9;
10 partner cells per family.  The candidate lists have no further draws, so the held-out set is 16 cells instead of 30 (n_short logged in `cells_robocasa_final.json`).  Most zero-count candidates are ineligible for the family
(the microwave family samples microwavable food only; the object config requires `graspable` and, for the microwave, `microwavable`), which is why they have zero training count.

## Code hashes at the first status check (first 12 hex of sha256)
- `experiments/xb_cells.py` 2bee859b1d36
- `experiments/xb_roll_mesa.py` 7d0f3c80ca73
- `experiments/xb_roll_robocasa.py` 94b176ac3456
- `experiments/xb_roll_robotwin.py` 794567ff73f8
- `experiments/xb_serve_groot15.py` b6548044a50c
- `experiments/xb_predict.py` c768b166a98c
- `experiments/xb_read.py` e2ca8a9c8991
- `experiments/xb_classes.py` 998b782a6c34
- `experiments/xb_driver.py` 8fd6d291dd27
- `docs/notes/xb_jobs/submit_stage1.sh` 683faf954979
- `docs/notes/xb_jobs/xb_cpu.sh` 7b979d7ba7c4
- `docs/notes/xb_jobs/xb_driver.sh` 7b979d7ba7c4
- `docs/notes/xb_jobs/xb_fix_groot15.sh` 53219fb9da15
- `docs/notes/xb_jobs/xb_lib.sh` c19467af36df
- `docs/notes/xb_jobs/xb_reset_mesa.sh` fee706cf0065
- `docs/notes/xb_jobs/xb_reset_rc.sh` 79c3b4e018c0
- `docs/notes/xb_jobs/xb_roll_mesa.sh` c12a1e91ce00
- `docs/notes/xb_jobs/xb_roll_robocasa.sh` 0ba00e048da2
- `docs/notes/xb_jobs/xb_roll_rt.sh` cc3f362b2bb9
- `docs/notes/xb_jobs/xb_setup_groot15.sh` faac6ba5d727
- `docs/notes/xb_jobs/xb_setup_groot_mesa.sh` 8b86e8268af6
- `docs/notes/xb_jobs/xb_setup_mesa.sh` 69cc621d2e22
- `docs/notes/xb_jobs/xb_setup_rcclient.sh` 21b5e2a80433
- `docs/notes/xb_jobs/xb_setup_rc.sh` 845f2a21d19f
- `docs/notes/xb_jobs/xb_setup_rtextra.sh` 077c439e9cfc
- `docs/notes/xb_jobs/xb_setup_rt.sh` c7394855f95a

## Pipeline driver
`experiments/xb_driver.py` runs as `xb_driver` (CPU sbatch, resubmits itself every 25 minutes, at most 60 iterations): resubmits incomplete rollout chunks, runs `xb_predict.py` for a benchmark once all its active models
have every partner episode (and the policy-free reset pass exists), only then submits that benchmark's held-out jobs (which also refuse to start without `pred_<bench>.json`), and finally runs `xb_classes.py` and `xb_read.py`
(no read while any expected cell is incomplete: k2).  Log: `results/bench/xb/driver.log`.  Active/candidate models: `results/bench/xb/active_models.json`, `candidate_models.json`.
GPU-h guard: sacct elapsed x GPUs of all `xb_*` jobs, no submissions at >= 285, job wrappers skip at >= 296.

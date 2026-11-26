# XR: porting the explicit "where" repair from LIBERO to MESA (pi0 / pi0.5, openpi-mesa)

Written 2026-11-26. Design plus the data-extraction step only; no policy has been trained and nothing was committed.
Data lives outside the repo: `/coc/flash10/ksingara3/robot/bench/mesa/xr_poses/`. Script: `experiments/xr_mesa_poses.py`. Job scripts: `/coc/flash10/ksingara3/robot/bench/mesa/xr_jobs/`.
The running `xb_*` jobs and their files were not touched.

## 0. Verdict

The port is feasible but is not a copy. Three differences from LIBERO decide the design:

1. **MESA's pi0 / pi0.5 act in joint-position space** (`actions_joint_pos`, 7 joint targets + gripper, delta against the joint state for the first 7 dims; eval `controller_type = joint_pos`). Our servo, H* and J write a Cartesian xyz command into action dims 0-2. In MESA those dims are joints 1-3. The where module must therefore map a Cartesian command to joint deltas through the Panda Jacobian at the current joint state (section 5). This is the main new component and the main risk.
2. **State**: MESA's state is 8 numbers (7 joints + jaw width). LIBERO's token overwrote trained state dims 0-3 of pi0; MESA has no spare trained dims, so the token goes into untrained dims 8-11 (section 4). pi0 on MESA therefore behaves like LIBERO's pi0.5 (a newly added pathway), not like LIBERO's pi0 (a natively read one).
3. **pi0.5 does read state in MESA** (`discrete_state_input=True`, the state is printed into the prompt as 256-bin integers), unlike LIBERO's pi0.5 config (`discrete_state_input=False`, state never read, RAA-c added a `state_proj`). So the pi0.5 token can ride in the text with no new parameters, but `max_token_len=60` must grow.

## 1. Do the data store object and receptacle poses? (task 1)

| Source | Per-frame object / receptacle pose? | Evidence |
|---|---|---|
| `albertwilcox/mesa-70-lerobot` (what the released checkpoints trained on) | **No.** Features are 3 camera videos (left/right shoulder, wrist), `actions`, `abs_actions`, `actions_joint_pos`, robot0 joint pos/vel/cos/sin, eef pos/quat, gripper qpos/qvel/jaw width, `robot0_proprio-state` (55), `next.done`, index columns. No object or fixture state, no scene id. | `meta/info.json`, `meta/episodes.jsonl` (6,970 episodes, 1,035,793 frames; 71 task texts) |
| `albertwilcox/mesa-70-hdf5` (source; 71 files, **473 GB** with images, 4.6-11 GB each) | **Yes, by replay.** Per demo: `data/demo_k/states` (flattened MuJoCo state, one row per frame), `actions`, `attrs["model_file"]` (the scene xml), and `data.attrs["env_args"]` (per-demo `env_kwargs`, including the full parsed BDDL problem: objects, fixtures, regions, `demonstration_states`, `goal_state`, `language_instruction`). | `docs/mesa_bench/available_datasets.md`; `mesa/utils/replay.py` replays exactly this way; I read states and attributes of 10 demos x 5 tasks |

So the poses must be replayed from sim states. The hdf5 files are far too large to download (473 GB, and /coc/flash10 filled up on 2026-11-24), but `states`, `actions` and the attributes are small; `h5py` over `HfFileSystem` range requests reads only those (about 25 s per task for 10 demos, 1-2.5 MB per 10 demos).

Facts the replay exposed (and that change the design):

- **Every demo is a different scene instance.** Per-demo `env_kwargs` differ in the receptacle instance (e.g. `bowl_5` vs `bowl_1` vs `bowl_6` inside the single task "put the mushroom in the bowl"), the object instance, the distractor objects, fixtures and table texture; each demo's xml is unique (10 of 10 distinct). Consequence: one env per demo (about 6 s of mujoco xml parsing each), so replay is expensive per episode and cheap per frame.
- **LeRobot is a truncated, filtered copy.** It keeps at most 100 demos per task (6,970 = 70 x 100 - 30; some tasks have 85-99) and cuts each demo at the first success frame (lerobot length = first-success frame, e.g. 141 of 175 hdf5 frames). Demo -> LeRobot episode is recovered by a longest-common-subsequence alignment of lengths within the episodes that share the instruction text (`map` stage; 50/50 matched on the test subset).
- `demonstration_states` is an ordered subtask list (`open`/`grasp`/`on|in`/`close`) per demo, which gives the instructed object, the receptacle (body, or a region site such as `cabinet_834_1_top_region`) and the phase.

## 2. Data extraction (task 2)

`experiments/xr_mesa_poses.py` (stages run in two venvs because the sim env has no `huggingface_hub`):

| Stage | Venv | What | 
|---|---|---|
| `extract` | openpi | hdf5 -> `xr_poses/src/<task>.pkl.gz` (states, scene xml, env args), range reads only | 
| `replay` | mesa (CPU, no rendering) | set each recorded state in the demo's own env; read eef, instructed object, current target; evaluate the subtask predicates with the env's own `_eval_predicate` | 
| `map` | any | hdf5 demo -> LeRobot `episode_index` (LCS on lengths, first-success + off) |
| `vocab` | any | integer ids for the instructed object, receptacle and fine receptacle (`cabinet:top`, `bowl_drainer:left`); `index.json` |
| `table` | any | per-frame arrays in LeRobot global-`index` order, `xr_table.npz`, for the data loader |

Per frame (world frame, metres, aligned with LeRobot `frame_index t` = hdf5 `states[t]`): `eef`, `obj`, `tgt`, `rel_o2t = tgt - obj` (the object-to-receptacle offset), `rel_e2o`, `rel_e2t`, `chan = [rel, role]` (LIBERO's channel: rel = active entity minus end effector, role 1 = object until the hold, 2 = target while held, 0 = none), `role`, `subtask` index, `grasp` (sim `check_grasp`; sampled every 3rd frame, held in between, as is the subtask latch). Per episode (json): instruction, object / receptacle names and normalised categories, receptacle fine key, subtask list, robot base pose, `first_success`, LeRobot episode index. Jaw width and eef are already in the LeRobot parquet.

Held/role phase: role 1 until the current subtask is `on`/`in` (i.e. after the grasp latch), then role 2; `open`/`close` subtasks and the frames after completion get role 0 and rel 0, mirroring LIBERO's unrouted tasks. The latch (not the instantaneous grasp) is used so a dropped object does not flip the role.

### What was run (2026-11-26) and what it cost

Scope: the first 40 demos of every one of the 71 hdf5 task files, not all 100. Replay costs about 6-8 s per demo (a fresh env and a 2.5 s mujoco xml parse each, because every demo is its own scene), so the full 6,970 would be about 13 CPU-h against the 10 CPU-h budget; frames are cheap (about 1 ms).

| Item | Result |
|---|---|
| Subset test (50 demos: 5 tasks x 10) | extract 25 s per task; replay 55-80 s per task (about 6 s per demo); map matched 50 of 50 |
| `extract` (40 demos x 71 files, range reads, one job after HF rate-limited 3 parallel ones with HTTP 429) | about 55 s per task, 1.5 job-h, 442 MB of `src/` |
| `replay` (CPU only, 16 jobs of 2 cores, waves as `src/` appeared) | 71 task files, 2,815 demos, 505,888 frames, 5.0 job-h (about 4 min per task), 30 MB of `mesa70/` |
| `map` + `vocab` + `table` | 2,801 of 2,815 demos matched to LeRobot episodes (11 tasks lost 1-3 demos to a length off-by-one or demos LeRobot dropped); 416,381 frames aligned = 40.2 % of the 1,035,793 LeRobot frames, 2,801 of 6,970 episodes; `xr_table.npz` 16 MB |
| Total | 36 jobs, 6.6 job-h (14.7 CPU-h); no job left running; two failed attempts (HF 429) cost about 0.3 job-h |

Sanity (all 2,815 replayed demos): role 0 / 1 / 2 = .33 / .31 / .37 of frames; the object-to-receptacle distance at the first success frame is p10 .026, p50 .055, p90 .113 m (target body or region-site centre, so a few centimetres off zero is expected); one robot base pose (0.66, 0, 0.912) in every scene; `|chan|` maxima .80 / 1.01 / .57 m and role 2. The eef read from the `grip_site` equals the `robot0_eef_pos` observable (assert in the script). Vocabulary: 47 object categories, 13 receptacle categories (instance variants such as `tray` / `wooden_tray`, `pan` / `chefmate_frypan`, `bowl` / `akita_black_bowl` stay separate; merge them before using the ids as a pairing key), 16 fine receptacles.

Outputs under `/coc/flash10/ksingara3/robot/bench/mesa/xr_poses/`: `mesa70/<task>.d0-40.{npz,json}` (per-frame arrays and per-demo meta), `xr_table.npz` (LeRobot global-index rows: `chan`, `o2t`, `e2o`, `e2t`, `subtask`, `grasp`, `valid`, per-episode ids), `index.json`, `vocab.json`, `map_report.json`, `src/*.n40.pkl.gz` (replay inputs; deletable, 442 MB). Job scripts and logs: `.../mesa/xr_jobs/`, `/coc/flash10/ksingara3/.claude_tmp/pp/xr_*.out`.

To finish the other 60 demos per task (about 8 CPU-h replay plus about 1.7 job-h extract): `extract --tasks all --max-eps 100` (writes `<task>.n100.pkl.gz`), then `replay --tasks all --demos 40:100 --src-sfx .n100` (writes `.d40-100` files), then rerun `map`, `vocab`, `table` (they glob every `mesa70/*.json`). The hdf5 is read by range requests only (473 GB total, never downloaded); keep a single extract job, since HF rate-limits anonymous parallel range reads.

## 3. Mapping of the LIBERO arms onto openpi-mesa

LIBERO names are the `l90c_config.py` configs of the nphs / nphj / nphp runs. "A0" in our repo is the in-network target-as-input route: a state token, servo off, plain data. The plain fine-tune without any token is a separate control (P0), which MESA needs because the released checkpoint is not a fine-tune of the same budget.

| Arm | LIBERO pi0 config | LIBERO pi0.5 config | MESA pi0 (`pi0_mesa`) | MESA pi0.5 (`pi05_mesa`) |
|---|---|---|---|---|
| R: released | (none) | (none) | `albertwilcox/mesa-pi0` as is | `albertwilcox/mesa-pi05` as is (xb already rolls these) |
| P0: plain continued fine-tune | `pi0_libero` recipe, no token | `pi05_g8v_raa_l_ae`-style, no token | `pi0_mesa` data, same steps as the arms, no token | same |
| A0: token as input, servo off (in-network route) | `pi0_g8v_raac_l_ae` | `pi05_g8v_raac_l_ae` (`Pi0ConfigS`, `servo_k = 0`, `RaacLoader`) | `xr_pi0_a0`: state 12-d (section 4), `servo_k = 0` | `xr_pi05_a0`: same 12-d state in the text |
| sL: servo kept (fixed-form where owner) | `pi0_g8v_raas_l_ae` | `pi05_g8v_raas_l_ae` | `xr_pi0_sl`: A0 + joint-space servo (section 5) | `xr_pi05_sl` |
| H*: servo-distilled head, external | `pi0_g8v_raah_l_ae` (`Pi0ConfigZH`, `raah0_head.npz`) | `pi05_g8v_raah0_l_ae` (`raah0p_model.py`) | `xr_pi0_hs`: head refit on MESA (section 5) | `xr_pi05_hs` |
| J: jointly trained where-branch | `pi0_g8v_raaj_l_ae` (`Pi0ConfigZJ`, `RaajLoader`, branch 3-64-64-30, last layer zero) | none (pi0 only in LIBERO) | `xr_pi0_j`: branch 3-64-64-(20x3), zero last layer | `xr_pi05_j` (extension; optional) |

In the paper's terms the contrasts to repeat are: H* - A0 (separate owner vs the same network learning the same token; primary), J - A0, H* - sL (non-inferiority), A0 - sL, plus P0 - R as the control for extra steps. Seeds: 1 first, 3-4 after a go.

## 4. State layout

MESA state = `robot0_joint_pos` (7) + `robot0_gripper_jaw_width` (1) = 8 numbers (`MESADataConfig`, proprio key `robot0_joint_pos+robot0_gripper_jaw_width`), padded to 32 by openpi.

New layout, 12 numbers: `[q1..q7, jaw, rel_x, rel_y, rel_z, role]` with `[rel, role] = chan` from the table. Dims 0-7 are untouched, so the released weights see the same input on those dims.

- **pi0** (continuous `state_proj`, Linear 32 -> 1024): dims 8-11 map to columns that the released run never exercised (the input was zero padding, so their gradient was zero; they keep the values from `pi0_base` pretraining on other embodiments, which are not tuned to a role channel). The token is effectively a newly added pathway. LIBERO's pi0.5 showed such a pathway stays unread without a data dose (share 0), while LIBERO's pi0 read the token at .542 only because it overwrote trained dims. Expect A0 to look like the LIBERO pi0.5 null; the servo-owning arms (sL, H*, J) read the channel inside the module, so they do not depend on the expert reading it. Do **not** overwrite joint dims.
- **pi0.5** (`discrete_state_input = True`): the tokenizer prints the *normalised* state, quantile-scaled to [-1, 1] and binned into 256 bins, into the prompt (`Task: ..., State: 128 140 ...;\nAction: `). Tokenisation runs before padding, so the text carries exactly 8 numbers today; 12 numbers add about 16 tokens and overflow `max_token_len = 60` (the longest instructions, "open the middle drawer of the cabinet and put the mushroom in it and close it", are already close). Set `max_token_len` to 96 in the new configs, otherwise the tail (`Action: `) is silently truncated. The module reads `observation.state[..., 8:12]` (still available in the model input), so no new class is needed for sL / H* / J on pi0.5. The RAA-c `state_proj` route (`raac_model.py`) is an optional second variant.
- **Normalisation**: new asset id `mesa_xr`. Dims 0-7 copy the released `mesa` stats byte for byte. Dims 8-11 get stats from the table (pi0: mean/std, because openpi's pi0 uses z-score; pi0.5: q01/q99). LIBERO forced quantile norm for pi0 (`_quantile_data`) because `servo_norm_*` un-normalises with q01/q99; the MESA module reads whichever stats the config carries, so the un-normalisation is the only change (z: `x * std + mean`).
- **At serving** the client must append `[rel, role]`: the sim computes it from the instructed object and receptacle (parsed problem) with the same predicates as the extraction. This is privileged input, the same caveat as in LIBERO. `experiments/xb_roll_mesa.py` is the model for the client; copy it to `xr_roll_mesa.py`, do not edit it.

## 5. The servo in joint space (new component)

LIBERO: `action_xyz = clip(K * rel, -M, M) * m + expert`, with K = 2.85 per metre and M = 0.9375 in delta-eef units, `m = 1` iff role is 2. MESA's action row k is a joint target, `state + delta_k`, for 20 rows.

Proposal (S1, primary): keep the module Cartesian and map it to joints.

`dq_k = Jpinv(q_t) * v * (k + 1) * dt_eff` per row, `v = clip(K * rel, -vmax, vmax)` (m per step, K and vmax refit from MESA, see below), `m = 1` iff role is 2, gripper and the 7th-joint redundancy left to the expert. `Jpinv` uses the analytic Panda forward kinematics (public DH parameters) evaluated at `q_t = observation.state[..., :7]` un-normalised, plus the fixed robot base pose (stored per episode in the json). It is a few tens of lines in jnp (differentiable, so J's branch trains through it) and torch-free: MESA serves JAX orbax checkpoints, so no PyTorch export is needed.

- **K, vmax, dt_eff**: refit by the same phase-0 gate as LIBERO (`results/flawed/raas_gate.json`: K > 0 and pooled residual variance ratio < 0.9) on the extracted data. Regress the eef displacement of held-phase frames (from `eef` in the table) on `rel_e2t`.
- **H*** (`raah0_model.py`): refit an MLP on `rel -> clipped servo law` with a release point learned from the demonstrations (per fine receptacle: mean `rel_o2t` at the release frame, i.e. `first_success`), exactly as `experiments/raah_fit.py` did for LIBERO; the head stays in Cartesian and goes through the same Jpinv. The release offsets differ by receptacle instance (the demos use different bowls), so fit per category and report the spread.
- **J** (`raaj_model.py`): branch 3 -> 64 -> 64 -> 60 (20 rows x 3 Cartesian), zero last layer, scaled by M, then the same Jpinv. Same inductive bias as H*; only the function of rel is learned.
- Fallback (S2): drop the joint-space model and fine-tune from `pi0_base` on the dataset's 7-d delta-eef `actions` with the OSC controller (eval supports `osc_pose`). This makes the servo identical to LIBERO but gives up the released checkpoints and costs about 30x more (section 7).
- Not covered by the module: the `open` and `close` subtasks (11 of the 17 held-out cells are `open_and_*` or `*_and_close` composites). Role 0 there, so the module is silent; the gain, if any, is on the place phase only. Report per phase.

## 6. Code to copy or adapt (all additive; no openpi file is edited)

From `lifelong_ood/policy/`:

- `raas_model.py` -> `xr_mesa_model.py`: `servo_fields` (constants), `servo_norm_*`. Change: state slice 8:12 (not 0:4), z-score or quantile un-normalisation, Cartesian -> joint through `panda_jpinv(q)`, 20 rows with the (k+1) ramp.
- `raas0_model.py` (`Pi0ZS`, `Pi0ConfigZS`, `PI0PytorchZS`): use unchanged for pi0 (our `Pi0ZS` subclasses `openpi.models.pi0.Pi0`; `src/openpi/models/pi0.py` is byte-identical in openpi-mesa, `pi0_config.py` differs only in the dropped `pytorch_compile_mode`). Drop the PyTorch classes.
- `raah0_model.py` (H*): `head_norm_jax` with the state slice and the joint map; new `xr_head.npz` from the refit.
- `raaj_model.py` (J): `WhereHead` (60 outputs), `Pi0ZJ`, `RaajLoader` (path = released `mesa-pi0/params`; `where_head` keys missing -> init).
- `raac_model.py`: only for the optional continuous-token pi0.5 variant.
- `l90c_config.py` -> new `lifelong_ood/policy/xr_mesa_config.py`: register the configs at runtime on top of `openpi.training.config` (MESADataConfig, `pi0_mesa`/`pi05_mesa` as base), asset id `mesa_xr`, `max_token_len = 96` for pi0.5, weight loader = released checkpoint params.
- `experiments/c_train_nphj.py` / `c_train_pi05.py` -> `experiments/xr_train.py` (name, exp, seed, steps, batch, save/keep interval). `PYTHONPATH` must put `openpi-mesa/src` first (as `xb_roll_*` does); the venv is `/coc/flash10/ksingara3/robot/openpi/.venv`.
- Data: a transform `XRChannel` in the repack stage that looks up `[rel, role]` by LeRobot global `index` in `xr_table.npz` and appends it to the state (no rewrite of the 11.7 GB dataset), plus `episodes=` filtering to the labelled episodes. Normalisation stats: `scripts/compute_norm_stats.py` on the 12-d state.
- Client: `experiments/xb_roll_mesa.py` -> `xr_roll_mesa.py` (append `[rel, role]` to the state, log per-episode as xb does).
- Not needed: `raaj_export.py` / `pi0z_export.py` (PyTorch export; LIBERO served PyTorch, MESA serves JAX), the relabel / Route A scripts (`sc_rv.py`, `raaf_*`; a counterfactual dose is a second stage, only if A0 is the null).

## 7. Training cost

LIBERO reference: batch 32, 6,001 steps, one A40, about 7.5 GPU-h per seed for J (the 4-seed nphs / nphj runs cost 108 GPU-h; frozen-VLM action-expert recipe). MESA released recipe: 50,000 steps at batch 128 (`pi0_mesa`, `pi05_mesa`), full fine-tune from `pi0_base` / `pi05_base`, no EMA.

| Route | Steps x batch | GPU-h per arm per model (A40) | Notes |
|---|---|---|---|
| **From the released checkpoint (recommended)** | 6k x 32 (LIBERO-matched), up to 6k x 64 | about 8 (b32) to 14 (b64); video decode (av1, pyav) may dominate | warm start, only the token pathway, servo-subtracted target and head are new; frozen-VLM recipe fits one A40 |
| From the base | 50k x 128 | about 250-300 | reproduces the released recipe plus the token; about 30x the above, 4 arms x 2 models is about 2,000 GPU-h: not recommended |

First pass, one seed: P0, A0, sL, H*, J (H* and sL share the servo; H* needs the head refit first) on pi0 and four arms on pi0.5, about 9 trainings x 8 = 72 GPU-h; with labels for all 100 demos per task (needs the second replay pass). Three further seeds on the primary contrast (H*, A0, sL on pi0): about 72 GPU-h.

Evaluation cost (pi0.5 smoke, `mesa-composite` `mushroom_tray_on`: 20 episodes in about 4 min, 4 env workers, A40): 17 held-out cells x 50 episodes = 850 episodes, about 3 GPU-h per arm per model; the trained-task gate (K, 17 partner tasks x 20) about 1.2 GPU-h. About 45 GPU-h for the first pass of 9 arms. Total first pass about 120 GPU-h, one seed; about 190 more for 3 extra seeds of the primary contrast. These are estimates from LIBERO timings and the xb smoke runs, not measured on MESA.

## 8. Evaluation on the XB held-out MESA cells

- Cells: `results/bench/xb/cells_mesa.json`: 17 held-out tasks (`mesa-composite`, every slot held out at the fine level against MESA-70), 50 initial states each (0-49, fixed), partners from `mesa-70`. The released checkpoints trained on exactly MESA-70, so the held-out pairs are excluded from every arm's data by construction; our arms train on MESA-70 only (not `mesa-all-train`, which contains 16 of 24 composite slots under other variants, `docs/notes/bench_audit.md`).
- Harness: copy `xb_roll_mesa.py` (`--cell-type heldout|partner`, `controller_type = joint_pos`, replan, step budget, success rule unchanged) to `xr_roll_mesa.py` with the channel. Output under a new `results/bench/xr/`, never `results/bench/xb/`.
- Baselines for free: the xb runs of `mesa-pi0` and `mesa-pi05` (R row) on the same cells and states.
- Pre-register before any held-out rollout (repo practice: ledger row, prediction file, contrasts and non-inferiority margins, failure classes from `xb_classes.py`): primary H* - A0 on held-out success, trained-task gate (K, margin -.05), per-phase split (place-phase failures vs open/close failures), E1 failure classes (G/W/O/H/M).
- Reads: success and subtask completion with per-cell and per-seed intervals; fraction of failures on a non-instructed receptacle (the atlas's wrong-where class).

## 9. Risks

1. **Jacobian servo is unvalidated.** Offline check first: apply `Jpinv(q) * clip(K * rel)` to demo frames and compare with the demo's joint deltas (variance ratio). If it fails the gate (< .9), the where module has no basis in joint space and S2 is the only route.
2. **pi0 token is a new pathway** (section 4); A0 may be a null, which weakens the A0 comparison but not sL / H* / J.
3. **pi0.5 text overflow**: forgetting `max_token_len` silently truncates `Action: `. Check token counts on the longest instruction before training.
4. **The module owns only the place phase.** 11 of 17 held-out cells begin with `open` or end with `close`; their failures may be in the phase the module does not own. Plan a per-phase analysis; a handle-targeting extension is out of scope.
5. **Receptacle instance varies per demo**; the release point differs by instance. H* learns per category; the spread bounds the achievable accuracy (target-body centre vs region site centre differ by 3-8 cm at first success in the 5 test tasks).
6. **Label coverage**: the table labels 2,801 of 6,970 episodes (40 demos per task), training sees only labelled episodes. P0 must train on the same episodes, and R (released, 100 demos) is not a matched control.
7. **Role labels are sim-privileged** (subtask latch with the env's predicates), the same caveat as LIBERO's privileged depth and masks.
8. **Held-out start states are fixed** (50 per cell), so seed variance comes from training and sampling noise only; 1 training seed will not separate arms closer than about .1.
9. **Compute etiquette**: GPU nodes are shared with the xb rollouts; 4-GPU cap on tail-lab; all training via sbatch with explicit `-p`/`-A` and the exclude list; never edit a `.sh` under a running job.
10. **Version drift**: openpi-mesa and the repo's openpi venv share `pi0.py`; confirm `PYTHONPATH` order on the first run (`import openpi; openpi.__file__`).

## 10. Next steps (in order)

1. Finish the replay for demos 40:100 (about 8 CPU-h) so the table covers all 6,970 episodes, or train the first pass on the 2,801 labelled episodes (P0 on the same episodes).
2. Offline gate: fit K, vmax and release points; Jacobian variance ratio (risk 1). Decide S1 vs S2.
3. Implement `xr_mesa_model.py`, configs, `XRChannel`, `xr_train.py`; unit checks as in `raaj_check.py` (step-0 identity: J = 0 and sL with `servo_k = 0` equal the released model on the 8-d inputs).
4. 200-step timing job per model to fix the batch and the GPU-h; then the pre-registration row; then one-seed first pass.

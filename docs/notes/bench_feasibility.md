# Feasibility check: MESA, RoboCasa365, RoboTwin 2.0 for the VLA failure atlas

Date: 2026-11-25/26. Everything lives under `/coc/flash10/ksingara3/robot/bench/` (clones, envs, checkpoints, assets).
All job scripts are in `/coc/flash10/ksingara3/robot/bench/jobs/` (logs: `/coc/flash10/ksingara3/.claude_tmp/pp/bench_*.out`).
Nothing was committed, and `results/` and `paper/` were not touched.

## Verdict

| Benchmark | Verdict | One line |
|---|---|---|
| MESA / BiMESA | **GO** | Installed in 5 min, pi0.5 checkpoint runs 20 episodes in about 4 min per task, and the Composite suite is already a held-out-pairing split (object and receptacle each seen in training, pair never seen). |
| RoboCasa365 | **GO-with-work** | Works end to end with the official pi0.5 checkpoint, but needs a shim for the openpi fork, a fixed client, about 30 GB of object packs, and episodes are slow (long horizons). Composite-Unseen is a task-template split, not an object-by-target split. |
| RoboTwin 2.0 | **GO-with-work** | Works with the official X-VLA cotrain checkpoint (18/20 and 19/20). The install is fragile (pins, nvcc, gcc, ffmpeg). The pairing split needs our own demo regeneration, which is easy because demos are scripted. |

## Smoke results (one released fine-tuned checkpoint, 20 episodes per row)

| Benchmark | Model (checkpoint) | Task | Split / setting | n | Success |
|---|---|---|---|---|---|
| MESA | pi0.5 (`albertwilcox/mesa-pi05`, config `pi05_mesa`) | `mushroom_tray_on` | **mesa-composite (held-out pairing)** | 20 | **16/20 = 0.80** (subtask completion 0.875) |
| MESA | same | `mushroom_bowl_on` | mesa-70 (in-distribution) | 20 | 14/20 = 0.70 |
| RoboCasa365 | pi0.5 (`robocasa/robocasa365_checkpoints` `pi05_pretrain_human300/multitask_learning/75000`) | `CloseFridge` | atomic_seen, pretrain scenes, lightwheel objects only | 20 | 12/20 = 0.60 |
| RoboCasa365 | same | `CuttingToolSelection` (horizon 1200) | composite_unseen, pretrain scenes, objaverse+lightwheel | 20 | 0/20 = 0.00 |
| RoboTwin 2.0 | X-VLA cotrain (`TianxingChen/RoboTwin2.0`, `cotrain_ckpt/X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0.tar.gz`, ckpt-30000) | `place_container_plate` | demo_clean, aloha-agilex, seed 0 | 20 | 18/20 = 0.90 |
| RoboTwin 2.0 | same | `beat_block_hammer` | demo_clean | 20 | 19/20 = 0.95 |

Caveats. All smoke rows are n=20 with one seed, so they confirm the pipeline works and are not benchmark numbers.
The MESA 0.80 on the held-out pair against 0.70 on the in-distribution task is noise at this n.
The RoboCasa numbers are consistent with the published pattern (pi0.5: atomic-seen 39.6 %, composite-unseen 1.2 % averaged over tasks).
RoboCasa smoke used `obj_registries=("lightwheel",)` for CloseFridge because the objaverse packs were not yet downloaded.
RoboTwin eval counts only seeds the scripted expert solves (RoboTwin's default `expert_check`).

## Resources

| | GPU-h (approx.) | Disk |
|---|---|---|
| MESA | 0.15 | 24 GB (venv 1.2 GB, assets 8.9 GB, init states 1.1 GB, pi05 ckpt 12 GB) |
| RoboCasa365 | about 1.4 (includes one preempted and one cancelled run) | 35 GB (venv 4.9 GB, assets 4.2 GB textures/fixtures plus 15 GB objaverse/aigen/lightwheel objects, pi0.5 ckpt params 12 GB) |
| RoboTwin 2.0 | about 0.8 | 38 GB (env 9.8 GB, X-VLA env 5.3 GB, assets 16 GB, X-VLA ckpt 3.3 GB) |
| Shared caches | | `bench/hf` 11 GB (duplicate HF download cache, safe to delete), `bench/pipcache` 7.6 GB, `bench/tmp` 3.1 GB |
| **Total** | **about 2.4 of the 12 GPU-h budget** | **about 120 GB** |

## 1. MESA / BiMESA (Wilcox et al., CoRL 2026)

- **Install: OK.** `bench/mesa/mesa-env` (uv, Python 3.10, mujoco 3.4.0, robosuite 1.5.1), venv at `bench/mesa/venv`.
  `./scripts/setup.sh --no-bimesa` downloaded assets (about 13 GB full; 8.9 GB without BiMESA-only objects) and the fixed eval initial states.
  Job: `jobs/bench_inst_mesa.sh` (its checkpoint step failed because the venv has no `huggingface_hub`; `jobs/bench_dl_mesa.sh` did the download with the openpi venv).
- **Headless: OK** (`MUJOCO_GL=egl`, A40, 4 env workers).
- **Architecture:** a websocket policy server (pi models via the `pairlab/openpi-mesa` fork, GR00T via `pairlab/mesa-GR00T`) plus `scripts/eval_server_parallel.py`.
  I reused the existing `/coc/flash10/ksingara3/robot/openpi/.venv` for the policy server and put `openpi-mesa/src` on `PYTHONPATH`; the dependency pins are identical, so no new JAX env was needed.
  Script: `jobs/bench_smoke_mesa.sh`. Results: `bench/mesa/mesa-env/experiments/{mesa-composite,mesa-70}/smoke/*/statistics/final_summary.json`.
- **Not tested:** BiMESA (needs the 8 GB BiMESA assets and the bimesa checkpoints), GR00T N1.6 (needs the `mesa-GR00T` env; cloned at `bench/mesa/mesa-GR00T`), pi0, pi0-FAST.

### Checkpoint inventory (MESA)

| Model | MESA (single-arm) | BiMESA |
|---|---|---|
| pi0 | `albertwilcox/mesa-pi0` | `albertwilcox/bimesa-pi0` |
| pi0.5 | `albertwilcox/mesa-pi05` (**run**) | `albertwilcox/bimesa-pi05` |
| pi0-FAST | `albertwilcox/mesa-pi0-fast` | `albertwilcox/bimesa-pi0-fast` |
| GR00T N1.6 | `albertwilcox/mesa-gr00t-n1.6` | `albertwilcox/bimesa-gr00t-n1.6` |
| GR00T N1 / N1.5 | none found | none found |
| X-VLA | none found | none found |
| SmolVLA | none found | none found |
| OpenVLA-OFT | none found | none found |

Source: the model table in `docs/mesa_bench/evaluating_trained_policies.md` of the repo and the HF collection `albertwilcox/mesa`. I only fetched the pi0.5 weights.

### Held-out-pairing split (MESA)

Exists, and the data exclude the pairs.
`mesa-composite` has 20 tasks (single-object pick-place, open-then-place, place-then-close, two-object).
I audited it with `bench/mesa/pairing_check.py` (parses the instruction text of each task id against the 70 training tasks):

- 20/20 composite tasks use an object that occurs in MESA-70 and a receptacle that occurs in MESA-70.
- 18/20 have an (object, receptacle) pair that is absent from MESA-70.
- The two exceptions: `open_and_carrot_cabinet_top_region_and_close` (its two component tasks are seen, but the combined task is not) and `fish_garlic_pan_on` (two-object task whose single-object pairs are seen).
- Examples of clean held-out pairs: `mushroom_tray_on`, `apple_plate_on`, `onion_bowl_on`, `carrot_pan_on`, `cheese_basket_contain_region`, `lemon_cutting_board_on`, `cheese_mug_tray_on`.
- `albertwilcox/mesa-all-train-lerobot` (806 tasks, MESA-70 plus MESA-aux) is documented as excluding all held-out eval tasks (v3.0 revision; the v2.1 revision leaks two composite tasks, `cup_bowl_drainer_left_region` and `open_and_jam_slide_cabinet_contain_region`).
  Audit limitation: my parser only understood 70 of the 806 aux task ids, so the "70+aux" inventory is incomplete. A full audit needs the aux task names parsed properly.
- To build our own exclusion for fine-tuning: filter LeRobot episodes by `task_index` (or use `mesa-all-hdf5` `train/` vs `test/`). `openpi-mesa` has the training configs (`pi05_mesa` etc.).
- BiMESA has the same structure (`bimesa-composite`, `bimesa-object`); BiMESA-57 excludes the held-out combos by construction.

Verdict **GO**. Blockers: none. Caveat: the MESA checkpoints were trained by the benchmark authors on MESA-70, so a "held-out pairing" claim for our own models needs our own fine-tune on the aux/70 sets.

## 2. RoboCasa365

- **Install: OK, with work.** `bench/robocasa/venv_sim` (uv, Python 3.11; robosuite master + robocasa `--no-deps` + numpy 2.2.5, mujoco 3.3.1). Job: `jobs/bench_inst_rc.sh`.
  Hit and fixed: `setup_macros` and the asset downloader prompt for input (use `yes y |`); running `python -m robocasa...` from a directory that contains a `robocasa/` folder shadows the package (cwd must be neutral); no `huggingface_hub` in the venv (`jobs/bench_dl_rc.sh` used the openpi venv).
  Assets: `tex tex_generative fixtures_lw objs_lw` (about 4 GB) plus `objs_objaverse objs_aigen` (about 15 GB), `jobs/bench_dl_rc_obj.sh`. Without objaverse, most composite tasks crash at reset (`a cannot be empty` or `Probabilities contain NaN`).
- **Headless: OK** (`MUJOCO_GL=egl`).
- **Policy side:** `robocasa-benchmark/openpi` fork (cloned at `bench/robocasa/openpi-robocasa`) has config `pi05_pretrain_human300`.
  Its `training/config.py` imports `robocasa.*` at module load, which would drag the whole sim stack (and numpy 2) into the JAX env. I avoided touching the shared openpi venv with a `sitecustomize.py` shim (`bench/robocasa/shim/`) that mocks `robocasa.*` for the server only. Serving works.
  The fork's `examples/robocasa/main.py` reads `args.task_soup`, which its `Args` dataclass does not define, so I wrote my own client (`bench/clients/rc_smoke_client2.py`, copied from main.py's `eval_env`). Scripts: `jobs/bench_smoke_rc.sh`, `jobs/bench_smoke_rc2.sh`.
- **Cost:** composite horizons are long (CuttingToolSelection 1200 steps, ArrangeTea 2250, ArrangeBreadBasket 4350); about 1 to 2 min per episode on the shortest, up to 8 min on the longest. 20 episodes of a composite task is 0.5 to 2.5 h.
- **Not tested:** `lerobot/smolvla_robocasa` via LeRobot. The local lerobot 0.4.4 has no `robocasa` env, so it needs a newer lerobot in a new env (the LeRobot doc says `--env.type=robocasa`, three rename-map cameras, MUJOCO_GL=egl). Also not tested: GR00T N1.5 (needs `robocasa-benchmark/Isaac-GR00T`, cloned at `bench/robocasa/gr00t-robocasa`).
- Important: `lerobot/smolvla_robocasa` was trained on `robocasa_target_human_unified` (target split), so composite_unseen is NOT unseen for it. The official `robocasa365_checkpoints` models are trained on `pretrain_human300` and are the right ones for the unseen split.

### Checkpoint inventory (RoboCasa365)

| Model | Repo | Status |
|---|---|---|
| pi0 | `robocasa/robocasa365_checkpoints` -> `pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000`; also `RLinf/RLinf-pi0-robocasa-pretrain-human300` (community) | official, not run |
| pi0.5 | `robocasa/robocasa365_checkpoints` -> `pi05_pretrain_human300/multitask_learning/75000` (**run**; params about 12 GB, optimizer state skipped); also `lerobot/pi05_robocasa`, `lerobot/pi052_robocasa` | official |
| GR00T N1.5 | `robocasa/robocasa365_checkpoints` -> `gr00t_n1-5/{multitask_learning/checkpoint-120000, foundation_model_learning, lifelong_learning, target_fraction}`; `prehj/GR00T-N1.5-robocasa-baseline` (community) | official, not run |
| GR00T N1.6 | `Abhi03/grootn16_robocasa365_multitask_learning` (community only) | not run |
| GR00T N1 | none found for the 365-task suite (only tabletop/other variants) | |
| X-VLA | `deksprime/xvla-base-robocasa-human300-ri` (community) | not run |
| SmolVLA | `lerobot/smolvla_robocasa` (target-split training, 0.5B); community variants (`deksprime/smolvla-robocasa-multitask-ri`) | not run |
| Diffusion Policy | `robocasa/robocasa365_checkpoints` -> `diffusion_policy/...` | official |
| OpenVLA-OFT | none found (`StarVLA/Qwen3-VL-OFT-Robocasa` is a Qwen-VL OFT head, not OpenVLA-OFT) | |

The official table (docs/benchmarking/multitask_learning.md) lists pi0.5 at 39.6 / 7.1 / 1.2 and GR00T N1.5 at 43.0 / 9.6 / 4.4 for atomic-seen / composite-seen / composite-unseen.

### Held-out-pairing split (RoboCasa365)

- The official unseen split is by **task template**: `composite_unseen` is 16 tasks (ArrangeBreadBasket, ArrangeTea, BreadSelection, CategorizeCondiments, CuttingToolSelection, GarnishPancake, GatherTableware, HeatKebabSandwich, MakeIceLemonade, PanTransfer, PortionHotDogs, RecycleBottlesByType, SeparateFreezerRack, WaffleReheat, WashFruitColander, WeighIngredients) whose templates are absent from the 300-task pretraining set. This is a composition split, but not an object-by-target pairing split.
- An object-by-target pairing split would have to be built. The PickPlace atomic tasks (`PickPlaceCounterToCabinet`, `...CounterToStove`, `...DrawerToCounter`, `...SinkToCounter`, `...ToasterToCounter`) sample the object category from `obj_groups` and the target fixture is fixed by the task, so the pair is (object category, source/target fixture). Each human demo's `ep_meta` records the sampled objects, so demos can be filtered by (category, fixture) pair before fine-tuning. The env accepts `obj_groups` / `exclude_obj_groups` for evaluating the held-out pairs.
- Limits: only 100 human demos per task, so after excluding pairs the counts per task are small; the released MimicGen generator (`robocasa-benchmark/mimicgen`) can add demos. Needs a separate pass to confirm that `ep_meta` in the lerobot datasets exposes object categories.

Verdict **GO-with-work**. Blockers: the shim and the client fix (done), about 15 GB of object assets (done), slow composites, and a decision on whether the template split is enough or we build the pairing split.

## 3. RoboTwin 2.0

- **Install: OK, with work.** Conda env `bench/robotwin/env` (Python 3.10, torch 2.4.1+cu121, sapien 3.0.0b1, mplib 0.2.1, pytorch3d 0.7.8 built with CUDA, curobo v0.7.8 built), policy env `bench/robotwin/xvla_env` (X-VLA requirements plus torch cu128). Repo: `bench/robotwin/RoboTwin` (with the `XPolicyLab` submodule). Assets: 16 GB.
  Jobs: `jobs/bench_inst_rt.sh`, `jobs/bench_curobo_rt.sh`, `jobs/bench_ffmpeg_rt.sh`.
  Hit and fixed:
  1. `conda activate` does not take effect in my non-interactive jobs; I set `PATH` to the env's `bin` and call the env python (`python -m pip`) directly.
  2. `conda create -p ... python=3.10` without `pip` picks up the base 3.13 pip and then cannot find torch 2.4.1 or sapien 3.0.0b1.
  3. The system gcc 13.3 is too new for nvcc 12.1 (`nvcc` exit 255 building curobo). Fix: conda-forge `gxx_linux-64=12` plus `NVCC_PREPEND_FLAGS=-ccbin ...`; nvcc 12.1 from conda-forge (`cuda-nvcc=12.1`, no system CUDA exists).
  4. pytorch3d built on a CPU node without `FORCE_CUDA=1` is CPU-only, so it was rebuilt with CUDA.
  5. `ffmpeg` binary is needed by the eval (conda-forge ffmpeg).
  6. `av==14.4.0` in the X-VLA requirements does not build; the unpinned wheel works.
  7. XPolicyLab's shell wrappers use `conda activate` plus `set -u`, which fails with the gcc activation scripts, so I launch the server (`XPolicyLab/setup_policy_server.py`) and the client (`scripts/eval_policy.sh --test_num 20 ...`) directly. Script: `jobs/bench_smoke_rt.sh`.
- **Headless: OK** (SAPIEN, "Render Well", A40). Results: `bench/robotwin/RoboTwin/eval_result/<task>/X_VLA/demo_clean/RoboTwin-cotrain-aloha_agilex-ee-0/*/_result.txt` (with per-episode mp4).
- Each 20-episode task takes about 20 to 25 min.

### Checkpoint inventory (RoboTwin 2.0)

The official cotrain checkpoints are in the HF dataset `TianxingChen/RoboTwin2.0`, folder `cotrain_ckpt/`: AHA_WAM, Abot_M0, EventVLA, FastWAM, GalaxeaVLA, Pi_05, Spatial_Forcing, X_VLA, X_WAM, Xiaomi_Robotics_0, starVLA.

| Model | Repo / path | Notes |
|---|---|---|
| pi0 | none official in `cotrain_ckpt`. Single-task community: `RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle` (+ PPO and "NEW" variants) | XPolicyLab has `Pi_0` and `Pi_0_Fast` adapters, but no released checkpoint found |
| pi0.5 | `TianxingChen/RoboTwin2.0` -> `cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz` (12.4 GB); community: `SidneyXie/pi05_robotwin`, `ShijieSKY/pi05-robotwin`, `Myungkyu/pi05_robotwin_vanilla_b128_60k`, `z-lab/flashvla-pi05-robotwin` | official not run |
| GR00T | N1 / N1.5 / N1.6: none found. Only N1.7: `mjleee/gr00t-n1.7-robotwin-90-k4-recap` (community) and an XPolicyLab `GR00T_N17` adapter | |
| X-VLA | `TianxingChen/RoboTwin2.0` -> `cotrain_ckpt/X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0.tar.gz` (3.2 GB, **run**); note `2toINF/X-VLA-RoboTwin2` on HF is described as a Bridge-V2-trained base and I did not test it | official |
| SmolVLA | `lerobot/smolvla_robotwin` | not run |
| OpenVLA-OFT | per-task only, community: `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-{handover_block,place_empty_cup,place_container_plate,pick_dual_bottles,beat_block_hammer,...}` (+ RL variants) | not run; no multi-task official one found |

### Held-out-pairing split (RoboTwin 2.0)

- There is no official pairing split. RoboTwin tasks are scripted-expert generated (`collect_data.sh <task> <config> <gpu>`), and the task classes sample object identities in `load_actors`, so a pairing split is easy to build by regenerating demos with restricted sampling.
- Best candidates: `place_a2b_left` / `place_a2b_right` (object A and object B are each sampled from a roughly 50-item `object_list` with random model ids, instruction "place A to the left of B"; the pair is (A, B)), `place_object_basket`, `put_object_cabinet`, `place_cans_plasticbox` (two can ids times plasticbox ids {3,5}), `place_container_plate` (container in {bowl 1,2,3,5; cup 1-7}, plate id fixed, so the pairing axis is weak). Pick tasks like `place_a2b_*` where two objects are sampled independently are the cleanest.
- Plan: in a copy of the task class, sample (A, B) from a training set of pairs (every object appears but certain pairs are withheld), collect demos with the expert, fine-tune, then evaluate on the withheld pairs by forcing A and B. The released 100k trajectories are in HF `TianxingChen/RoboTwin2.0/dataset/<task>/...`; whether their per-episode instruction files let us filter by (A, B) must be checked after download (not done).
- Cotrain checkpoints were trained on all 50 tasks with random objects, so none of them can serve as a pairing-held-out model; we would have to fine-tune our own (X-VLA or pi0.5 recipes are provided under `XPolicyLab/policy/*/train.sh`).

Verdict **GO-with-work**. Blockers: fragile install (documented above), long per-task cost, and we must regenerate or filter demos for the pairing split.

## Reproduce

```
sbatch jobs/bench_inst_mesa.sh ; sbatch jobs/bench_dl_mesa.sh ; sbatch jobs/bench_smoke_mesa.sh
sbatch jobs/bench_inst_rc.sh ; sbatch jobs/bench_dl_rc.sh ; sbatch jobs/bench_dl_rc_obj.sh ; sbatch jobs/bench_smoke_rc2.sh
sbatch jobs/bench_inst_rt.sh ; sbatch jobs/bench_curobo_rt.sh ; sbatch jobs/bench_ffmpeg_rt.sh ; sbatch jobs/bench_smoke_rt.sh
```
(The scripts include the fixes listed above; order matters only within one benchmark. `bench_inst_*` scripts were patched in place after failures, so a fresh run of `bench_inst_rt.sh` still needs `bench_curobo_rt.sh` for the gcc 12 + CUDA pytorch3d/curobo step.)

Disk cleanup available: `rm -rf bench/hf bench/pipcache bench/tmp` frees about 22 GB.

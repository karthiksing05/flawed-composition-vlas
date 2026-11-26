# Training-data co-occurrence audits: MESA, RoboCasa365, RoboTwin 2.0

2026-11-26. CPU only, no rollouts. Definition (PLAN_GOAL3): a held-out pairing is an (object, receptacle/target) pair that never
appears together in the data a released checkpoint was trained on, while the object and the receptacle each appear with other partners.
Scripts: `experiments/bench_audit_{mesa,robocasa,robotwin}.py`. Jobs: `/coc/flash10/ksingara3/robot/bench/jobs/bench_audit_*.sh`
(each took under 2 min CPU; about 0.1 CPU-h total). Outputs: `results/bench/<name>/audit/` (the scripts refuse to overwrite).
Downloads: LeRobot `meta/` only, 0.8 GB (MESA) and 0.12 GB (RoboCasa); RoboTwin read two small members per zip by range request (nothing stored).

## 1. MESA / BiMESA

### Checkpoint to training data (model cards)
| Checkpoint | Trained on |
|---|---|
| mesa-pi0, mesa-pi05, mesa-pi0-fast, mesa-gr00t-n1.6 | `albertwilcox/mesa-70-lerobot` (70 tasks, 6,970 episodes, 100 per task) |
| bimesa-pi0, bimesa-pi05, bimesa-pi0-fast, bimesa-gr00t-n1.6 | `albertwilcox/bimesa-57-lerobot` (57 tasks, 6,833 episodes) |

No released checkpoint saw `mesa-all-train-lerobot` (806 tasks, 81,091 episodes), so MESA-70 is the relevant set.

### Result (parsed from the language instructions; pair key = object x receptacle, "fine" separates cabinet drawers top/middle/bottom and dish-drainer left/right)
| Composite suite vs training set | object slots | held-out (fine) | pair seen | object unseen |
|---|---|---|---|---|
| mesa-composite (20 tasks) vs MESA-70 | 24 | **20** (17 coarse) | 4 | 0 |
| mesa-composite vs mesa-all-train (806) | 24 | **8** | 16 | 0 |
| bimesa-composite (15 tasks) vs BiMESA-57 | 18 | **10** | 5 | 3 (carrot, broccoli, fish never appear) |

- **The earlier "aux excludes composite" claim is true only at the task-id level.** The 806-task set contains 16/24 composite
  (object, receptacle) slots under other variants (for example `*_and_close`), so it must not be used to define held-out pairs.
  Against MESA-70 the earlier coarse audit said 18/20 tasks; the fine one says 17/20 tasks have every slot held out.
- Seen-pair exceptions vs MESA-70: `open_and_carrot_cabinet_top_region_and_close` (carrot, cabinet:top), `open_and_potato_microwave_heating_region_and_close`,
  and both slots of `fish_garlic_pan_on`. Coarse-level seen but fine-level held out: cup / dish drainer left, broccoli and mushroom / cabinet middle.
- Clean MESA-70 cells (17 tasks): mushroom_tray, apple_plate, onion_bowl, carrot_pan, cheese_basket, lemon_cutting_board, sponge_drainer_right,
  cup_drainer_left, open_and_broccoli_cabinet_middle, open_and_apple_cabinet_middle, open_and_beer_box, open_and_jam_slide_cabinet,
  open_and_apple_box_and_close, open_and_mushroom_cabinet_middle_and_close, lemon_lime_basket, broccoli_mushroom_cutting_board, cheese_mug_tray.
- Clean BiMESA cells (8 tasks fully held out): apple_plate, mushroom_tray, onion_bowl, lemon_cutting_board, open_and_beer_box, open_and_jam_slide_cabinet,
  open_and_mushroom_slide_cabinet_and_close, lemon_lime_basket. `carrot_pan_on`, `fish_garlic_pan_on` and the broccoli slot of broccoli_mushroom are not
  usable (the object has no training receptacle). Single-arm and bimanual suites share task ids but are different models and data.

### Partner map (per cell, in `cells__*.json`; per-object/receptacle counts in `partner_counts__*.json`)
Each partner task has 100 episodes. MESA-70: an object has 1 to 4 training receptacles (mean 1.6, 44 objects); receptacles have 2 to 10 objects
(basket 6, bowl 4, box 5, cabinet bottom/middle/top 4/4/5, cutting board 6, drainer left/right 2/3, microwave 5, pan 6, plate 4, slide cabinet 7, tray 10).
Example, `mushroom_tray_on`: mushroom's partners are bowl and cabinet:bottom; tray's partners are 10 objects (apple, bottled water, candle, onion, peach, rolling pin, squash, tomato, water bottle, wine).

### How a pairing is forced at eval
The composite suites are fixed BDDL task sets with fixed initial states, so evaluating a listed cell needs nothing extra (`--eval-set-name mesa-composite --task-filter <id>`).
New cells: `scripts/generate_tasks.py pick_and_place --pick-objects <category> --dest-object <category>` builds task variants for any object/destination category pair
(docs: `using_mesa/generating_a_task.md`). Unverified: that generated suites run through `eval_server_parallel.py` without a registered `EVAL_SETS` entry and fixed init states
(`generate_init_states.py` exists).

### Unverified
Instance-level variants (`_ai`, `_ood` ids) share instruction text and are merged into one object name. Distractor objects in the scenes are not audited.

## 2. RoboCasa365

### Checkpoint to training data
| Checkpoint | Data (source) | Status |
|---|---|---|
| pi0, pi0.5 official (`robocasa/robocasa365_checkpoints`) | `pretrain_human300`: 300 tasks, 100 human demos each, pretrain kitchens (docs/benchmarking/multitask_learning.md) | documented |
| GR00T N1.5 official | same for `multitask_learning`; the repo also holds `foundation_model_learning`, `lifelong_learning`, `target_fraction` runs that include target-split data | documented, use `multitask_learning/checkpoint-120000` |
| GR00T N1.6 `Abhi03/grootn16_robocasa365_multitask_learning` | no model card; name suggests pretrain_human300 | **unverified** |
| X-VLA `deksprime/xvla-base-robocasa-human300-ri` | card lists `pepijn223/robocasa_pretrain_human300_v4` (32,043 episodes) | card-cited |
| SmolVLA `lerobot/smolvla_robocasa` | `lerobot/robocasa_target_human_unified` (25,307 episodes, 50 target tasks, target kitchens) | card-cited; composite_unseen is not unseen for it |
| `lerobot/pi05_robocasa` | no card | unverified |

### Co-occurrence (object, source->destination family) from episode instructions
The PickPlace instruction "Pick the <obj> from the <src> and place it in/on the <dst>." gives the cell without downloading `extras/ep_meta.json`.
- Official pretrain-human atomic mirror (`ember-lab-berkeley/robocasa365-pretrain-atomic`, 7,356 episodes): 1,616 parsed PickPlace episodes in 17 families,
  93 object strings, 53 with at least 5 episodes. Families (about 100 demos each): counter->{cabinet, drawer, microwave, sink, blender}, {cabinet, drawer, microwave, sink, rack}->counter,
  dispenser->counter, plate->pan, pan->{bowl, plate}, fridge shelf<->drawer, tray->counter (7 episodes only).
- The 300-task set (`pepijn223/..._v4`) adds composite sub-tasks with the same phrasing (27 families, 3,275 parsed episodes) with fixed-purpose objects (kettle, mug, knife, "brewed coffee").
- **Counts are very sparse:** an object-family cell has 1 to 4 demos on average (max 108 for fixed-object families). Zero-count cells are partly sampling accident, not a designed holdout, so "known partners" are thin.
- Zero-count cells with the object at least 5 episodes elsewhere and the family at least 5 episodes (atomic mirror): **566**, all with graspable categories in the asset registry.
  Per family (realizable): counter->blender 45, counter->drawer 43, counter->microwave 32, counter->sink 20, counter->cabinet 15, cabinet->counter 14, drawer->counter 44,
  microwave->counter 31, sink->counter 21, rack->counter 50, dispenser->counter 52, tray->counter 51, plate->pan 26, pan->bowl 30, pan->plate 31, fridge drawer->shelf 29, shelf->drawer 32.
  So one task family (for example PickPlaceCounterToMicrowave) offers about 15 to 45 held-out objects, 45 at most.
  Partner lists (object's other families, family's other objects) are in `ember-lab-berkeley__robocasa365-pretrain-atomic.json` -> `zero_cells`.
- Not auditable from meta: the SmolVLA target set (`lerobot__robocasa_target_human_unified.json` has task class names but no natural-language instruction; the ep_meta files are in per-task `extras/` that I did not download).

### Can eval set the object and fixture at reset?
Object: yes. `PickPlace*` tasks take `obj_groups` (and `exclude_obj_groups`), forwarded by `gym.make("robocasa/<Task>", obj_groups="<category>", ...)`;
all 198 categories are valid group keys (`object_registry.json`), so a single category can be forced. The task class fixes source and destination fixtures, so the family is the cell's fixture pair.
`obj_registries=("objaverse","lightwheel")` restricts the asset packs (objaverse and aigen packs are on disk now). Unverified: that every zero-count category physically fits and is placeable
in each family (needs a reset test); and that the pretrain-vs-target `obj_instance_split` does not remove the category from the pretrain split.

### Unverified
Instruction objects were matched to registry categories by name (underscores to spaces); composite-task phrasings share the regex and are mixed into the 300-task counts.

## 3. RoboTwin 2.0

### Checkpoint to training data
| Checkpoint | Data | Status |
|---|---|---|
| X-VLA `cotrain_ckpt/X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0` | name says cotrain on aloha_agilex; assumed clean_50 + randomized_500 per task | **assumed** |
| pi0.5 `cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0` | `lerobot_dataset/RoboTwin_lerobot_v30.zip` (4.1 GB) | episode count not checked |
| SmolVLA `lerobot/smolvla_robotwin` | `lerobot/robotwin_unified`, 27.5k episodes = 550 x 50 tasks, matching clean 50 + randomized 500 | card-cited, composition inferred |

The released per-task zips store `scene_info.json` with `{A}`, `{B}` model names per episode. I read it for aloha-agilex clean_50 and randomized_500 (550 episodes per task) without rendering.

### (A, B) co-occurrence, place_a2b_left / place_a2b_right (550 demos each)
- Category level: A has 11 categories, B has 12 (sampler never uses A and B of the same category). Pairs seen 116/132 (left), 120/132 (right), mean 4.7 demos per seen pair.
  Off-diagonal zero cells with A and B each at least 5 episodes: **5** (left): (stapler, mouse), (bread, bell), (phone, bell), (tea-box, bell), (tea-box, toycar); **1** (right): (phone, stapler).
- Instance level (category + model id): A has 52 and B has 58 instances; 488 of 3,016 pairs seen (left), 481 (right). Zero cells, A and B each at least 5 episodes, different categories:
  **1,681 (left), 1,822 (right)**; each instance has 2 to 23 partner instances (mean 9). Partner lists per cell: `place_a2b_*.json` -> `instance.zero_cells`.
- Other A-to-B tasks have no usable pairing axis: place_object_basket (A 9 inst x B 2), place_object_scale (4 x 12), place_object_stand (28 x 5; 12 zero cells), place_cans_plasticbox, place_container_plate (1 x 9), place_bread_*, place_phone_stand, place_mouse_pad, place_fan: all pairs seen except 12 in place_object_stand.
- Category-level cells are mostly chance zeros of a uniform sampler, and the instance-level ones are held-out by a thin margin (about 0.2 demos per random pair), so base rates need the known-partner rollouts of PLAN step 2.

### Can eval force (A, B)?
Not out of the box. Eval seeds start at 100000*(1+seed) (training seeds are small); `_init_task_env_` calls `np.random.seed(seed)` and `load_actors` draws `np.random.choice(object_list)` and model ids, so (A, B) is a function of the seed
but depends on RNG consumed earlier in setup. Two options: (a) a subclass or env-var override of `load_actors` that fixes A and B (3-line patch; not implemented or tested), or
(b) enumerate seeds with the real setup (needs the simulator, no policy). Eval also skips seeds the scripted expert fails (`expert_check`), which can bias pair frequencies.

### Unverified
X-VLA and pi0.5 training composition (clean + randomized, aloha-agilex only, all 50 tasks); that the lerobot v30 episodes match the zip episodes one to one; the 12-category `object_list` filter on zero cells (diagonal only).

## Summary of candidate held-out cells
| Benchmark | Cells | Partner depth | Forcing |
|---|---|---|---|
| MESA (mesa-70 models) | 17 clean composite tasks (20 of 24 object slots) | 1 to 4 receptacles per object, 2 to 10 objects per receptacle, 100 demos each | built in; `generate_tasks.py` for new ones |
| BiMESA | 8 clean tasks (10 of 18 slots) | 1 to 4 receptacles per object | built in |
| RoboCasa365 | 566 (object, family) cells in 17 atomic families, 14 to 52 per family | 1 to 4 demos per partner; object seen in few families | `obj_groups=<category>`; fixtures set by the family |
| RoboTwin 2.0 | 5 + 1 category cells; about 1,700 instance cells per task | 2 to 23 partner instances; 4.7 demos per seen category pair | needs a `load_actors` patch |

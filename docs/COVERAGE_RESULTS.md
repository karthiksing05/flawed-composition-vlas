# Coverage arm: widen demo support with generated demos (PLAN_PARTS §10 item 11)

This is §4.1's labelled generated-data ablation. The user decided it on 2026-11-04 (commit 4b26df5b), after seeing data, and it
is labelled post hoc everywhere. It tests the mechanism claim of PARTS_RESULTS §54: the new-layout shortfall is a failure of
demo support (condition S, Prop. 2), not of the combination rule.

**What the arm can and cannot show.**
- A pass supports the mechanism (condition S). It is not the structural claim. §4.1 says layout generalisation must come
  from frame equivariance, and generated coverage is not that.
- The arm does not change the main arm's GO (PARTS_RESULTS §53-54: Layer-2 GO false). The Layer-3 ladder (docs/LADDER_RESULTS.md)
  is a separate track and is not affected by this arm.
- Every number here is a development read: seeds 10-12 and N2 states 4-23. No confirmatory or sealed read is run.

## 1. Design note (fixed 2026-11-04, before any data generation or code of this arm)

### 1.1 The base arm

**P_min** (PARTS_RESULTS §53). Its flags are `--rank 32 --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize`.
- Checkpoints: `pp_{full,nobottle,noplate}_naf_s{10,11,12}` in /coc/flash10/ksingara3/robot/data/fa1/ckpt.
- Rollouts: results/flawed/pp_cons_roll (S2 `P2naf`, S3 `Pnaf_<cond>_k<k>`).
- Read: results/flawed/pp_cons_min_read.json.
- S1: pp_s1_mix_full_naf_m{0,1,2}.json.

P_min's object term was trained on the human demos with its target term reused from `<cond>_nsz`. Under `--decouple` each role's
term does not depend on the other role's design (PARTS_RESULTS §53, assembly check), and rtclip is never evaluated (§53). So a
build of both roles with P_min's flags on the human demos alone is P_min's function.

### 1.2 Current demo support per atom, against the new layout

All values are from the human demos (fitted rows, never executed). They are taken from results already committed:
pp_rack_demo.json (§49), pp_far_demo.json (§40), §20, and P_min's build reports (pp_train_<cond>_naf_s10.json).

**Object atoms: horizontal hand-to-object offset at the demo's first row** (10th / 50th / 90th percentile; start radius r_s = 99th
percentile over the atom's demos, from P_min's full build).

| object | tasks | demo start offset (m) | r_s (m) | new layout (reported) |
|---|---|---|---|---|
| bowl | K1, K2, K3, K7 | K1 0.102 / 0.115 / 0.134; K1-K3 at most 0.15 (§20) | 0.140 | 0.31 m (§20, median start) |
| cream cheese | K6 | 0.186 / 0.206 / 0.219 | 0.231 | 0.12 m (§20); near starts off the one approach path (§40) |
| bottle | K4, K5 | K4 0.033 / 0.048 / 0.064; K5 0.037 / 0.057 / 0.071 | 0.079 | 0.22 m, range 0.09-0.37 m (§40, K4) |
| plate (object) | K8 | — | 0.279 | not read in GO |

The bottle's demos also contain no level transport phase (§40). The role radius r_role (99th percentile of the not-held hand-to-object
distance over every object row) is 0.225 m.

**Target atoms: carries** (one held segment each; 10th / 50th / 90th / max percentile). d0 is the horizontal object-to-target
distance at the hold's onset. The bearing is the carry's canonical-frame axis relative to the target's own yaw (degrees; min / 10th /
50th / 90th / max). The release is the foot at the demo's last held row: horizontal distance d and height h above the target
(10th / 50th / 90th).

| target | tasks | onset d0 (m) | bearing (deg) | release d (m) | release h (m) |
|---|---|---|---|---|---|
| plate | K1 | 0.118 / 0.142 / 0.161 / 0.171 | −113 / −104 / −98 / −89 / 78 | 0.003 / 0.010 / 0.022 | 0.001 / 0.003 / 0.010 |
| stove | K2 | 0.067 / 0.251 / 0.271 / 0.285 | −161 / 123 / 128 / 133 / 175 | 0.011 / 0.028 / 0.046 | 0.023 / 0.025 / 0.032 |
| cabinet top | K3, K4 | 0.075 / 0.286 / 0.312 / 0.326 | −167 / 115 / 126 / 142 / 172 | 0.039 / 0.066 / 0.097 | −0.002 / 0.001 / 0.009 |
| rack | K5 | 0.121 / 0.133 / 0.152 / 0.167 | −22 / 50 / 60 / 65 / 73 | 0.057 / 0.079 / 0.101 | 0.018 / 0.025 / 0.037 |
| bowl (target) | K6 | 0.125 / 0.142 / 0.156 / 0.164 | 138 / 153 / 161 / 171 / 178 | 0.006 / 0.016 / 0.032 | 0.044 / 0.061 / 0.082 |
| drawer | K7 | 0.152 / 0.176 / 0.210 / 0.292 | 45 / 49 / 56 / 63 / 79 | 0.011 / 0.036 / 0.194 | −0.171 / 0.012 / 0.058 |
| stove front | K8 | 2 carries, 0.052-0.094 | about −166 | 0.007-0.010 | 0.007-0.012 |

**The new-layout ranges the diagnostics report.**
- §49 (rack, K5, P_nsz seed 10, 9 holds): onset d0 0.314 / 0.375 / 0.402 m (10th / 50th / 90th). That is 2-2.5 times the rack
  demos' largest, 0.167 m. Bearings 21 / 26 / 40 / 51 / 54° (min / 10th / 50th / 90th / max), against 50-65° for the demos' 10th-90th.
- §51 (rack, K5, P_rel seed 10): in the new layout the foot reaches rack height 0.15-0.22 m out. The rack demos contain such a
  state only at their release, inside about 0.10 m.
- The new layout's other targets have no reported onset ranges. Its placement region implies them: each movable object is drawn
  uniformly in N2's box ([−0.279, −0.129] to [0.131, 0.210] in table x, y), at least 0.10 m from its own training spot. The
  fixtures (stove, cabinet, rack) are fixed in both layouts.

### 1.3 The widening rule W (one rule, fixed here, applied identically to every atom)

**W.** For each known pick-and-place task K1-K6, generate demos whose initial scenes come from stage 1g's layout sampler
(`fa1g_plan.layout`, unchanged).
- **Placement.** Each of the four movable objects (bowl, cream cheese, bottle, plate) is placed uniformly over N2's layout box.
  The box is the union of the familiar placements (LIBERO's sampler, within 0.027 m of each object's training centre, which lies
  0.08 m or more inside the box) and the new layout's region (the box minus a 0.10 m disc around each object's own centre). The
  new layout's 0.10 m exclusion is not applied. N2's clearance rules are applied: a 0.02 m gap, the fixture footprints and the
  stove front. A 10-step settle check must move no object more than 2 cm.
- **What does not move.** The fixtures (stove, cabinet, rack) are fixed in both layouts, so the sampler leaves them in place.
  The robot's start is LIBERO's, as in both layouts.
- **Seeds.** 8,000,000 + 1000 j + k, with j the task's index in K1-K6 (0-5) and k = 0, 1, 2, ...
  - The planner's noise draws from the same generator, continued.
  - These seeds are disjoint from N2's dev seeds (600000-600023 familiar, 700000-700023 new, branch noise 13,000,000 + 1000 k),
    from the confirmatory bases (9,100,000+, 9,200,000+, 23,000,000+; experiments/pp_conf_states.py), and from every earlier
    generator (900,000+, 950,000+, 960,000+, 970,000+). The generator code asserts this.
- **Count.** 50 kept demos per task: the same count for every task, equal to the 50 human demos per task.
  - Attempts run in seed order up to 600 per task.
  - If any task keeps fewer than 50, every task is cut to the smallest count kept, taking the first demos in seed order.
- **What W does not use.** It reads no evaluation state of any read, and it does not target failing tasks or atoms. It has no
  clearance strata, no grasp-option teachers and no task-specific constant.
  - The planner's keypoints (grasp offset and rotation, carry height, release offset and rotation) are stage 1g's
    a1g_keypoints.json, unchanged. Stage 1g's single rule fitted them from each task's own recorded human demos, as medians. No
    value is set or changed for this arm.
- **K7 and K8 get no generated demos.** The planner's class is pick-and-place. It has no drawer opening and no push. This limits
  the generator; it is not a choice made per atom.
  - The atoms that only K7 and K8 train (the drawer, the stove front, the plate as an object) keep their human support.
  - None of them enters a GO pairing (N1-N6) or an S3 cell.

### 1.4 The generator and its filter

**Generator.** The v2 servoing generator (PARTS_RESULTS §19, commit 6f644e33f): `pp_n4d.plan_episode(..., servo=True)`.
- It is stage 1g's planner. In its close and open phases it keeps servoing to the grasp or release point, instead of freezing.
- Its controller, gains, noise, waypoints, phase durations and the 300-step limit are unchanged.
- Each episode runs as the rollouts do: a fresh env and 10 dummy steps.

**Rows.** Rows are recorded exactly as `pp_n4d2.genfam` records them: the scene before the action, the action and the previous
command. No-op steps are dropped by `fa1_data.noop_mask`. held and off come from `fa1b_common.Held`, stepped over the kept steps.

**Filter (the same for every task).** A demo is kept iff LIBERO's success predicate fires within 300 steps. A layout is accepted
iff it passes the clearance rules and the settle check. Nothing else is filtered.

**Reported.**
- The generator's own success: kept / attempted, per task.
- The placements of the kept demos against the attempted ones, per object. Any non-uniformity the success filter adds is
  reported, not corrected.

**Labels and storage.**
- Generated demos are stored separately from the human demos: DATA/gen/pp_cov_<task>.npz, with a column gen = 1, demo ids 100 + j
  and the task's own index. The generation reports go to results/flawed/pp_cov_gen_<task>.json.
- Builds trained on them are tagged `cov_*`. The human demo files are never modified.
- No recorded human action is executed. The generated actions come from the planner's controller on the live simulator state.

### 1.5 Mixing ratio and training

**Rows.** A build's training rows are P_min's human-demo rows of its tasks, followed by the generated rows of the same tasks.
- Rows are drawn uniformly over the union: pp_p_train.fit, unchanged.
- The ratio is fixed by count: one generated demo per human demo, for each of K1-K6 (50:50). K7 and K8 have 50:0. The resulting
  row share is reported.

**Build.** `pp_p_train.main`, unchanged, run by experiments/pp_cov_train.py.
- pp_cov_train.py replaces the module function `pp_p_train.rows` with a wrapper. The wrapper appends the generated rows before
  every derived field is computed (chunks, canon, rot0, ee0, d0t, dkey).
- P_min's flags, 30,000 steps, batch 512, AdamW 3e-4, EMA 0.999, seeds 10, 11 and 12.
- Both role terms are trained, with no `--reuse`, because the generated rows enter both roles' data. Under `--decouple` each role
  trains as in P_min.

| build | human tasks | generated tasks |
|---|---|---|
| cov_full | K1-K8 | K1-K6 |
| cov_nobottle | K1, K2, K3, K6, K7, K8 | K1, K2, K3, K6 |
| cov_noplate | K2-K8 | K2-K6 |

pp_p_train's `--exclude` removes a held-out task's generated rows with its human rows. So the held-out atom never appears in a
build's data.

**Fitted quantities.** Every quantity a build fits is fitted by its existing rule on the build's training rows, human and generated
alike: the per-part support radii (psupport), the start radii and role radius (rtransport), and the grasp-mode inventory and each
target's m (imode). That is condition S as the architecture states it: a part acts where its training demos are.
- Two constants stay as in P_min, from the human demos: the input centre (the demos' mean end effector) and the home position.
  The centre is a normalisation constant; home is unused without aframe.

### 1.6 S3 creation: the honest variant

Creation runs `pp_p_train.py create`, unchanged, on the cov builds.
- The bottle part (object) is created from the first k human demos of K4, and the plate part (target) from the first k human
  demos of K1, for k = 1, 5 and 10.
- The created part fits its support radius, start radius and m from those k human demos only. That is the creation claim.
- Generated data enter only the builds, so only the training atoms' parts and the role backbones see it. The held-out atom has
  no generated data at any stage. This is the honest variant, and it is the only variant run.

### 1.7 Smoke checks (before any build)

1. **Zero generated rows reproduce P_min's build.** pp_cov_train with zero generated rows gives bitwise the same batch tensors
   as `pp_p_train.rows`. A 300-step build (full, seed 10, P_min's flags) through pp_cov_train gives bitwise the same checkpoint
   as `pp_p_train.py build` with the same arguments.
2. **The generated data follow W** (`pp_cov_gen.py stats`):
   - every seed is in W's range and disjoint from the listed bases;
   - every generated initial placement lies in the box and passes the clearance rules;
   - each task keeps the same count;
   - the placement distribution of attempted and kept demos is compared with uniform (the share within 0.10 m of each object's
     training centre against its area share);
   - the support table of §1.2 is recomputed for the generated rows, beside the human ones.
3. **Merge check.** With generated rows, the batch's human rows equal `pp_p_train.rows`'s, every dkey is unique per episode, and
   held is 0/1.

### 1.8 Read, mapping and prediction

**Read.** The read uses dev states 4-23, N2's seeds and seeds 10-12. It applies pp_l2's frozen estimators through
pp_cons_read.components, unchanged, to P_cov's rollouts and S1 files, and to P_min's.

**Primary verdict (one mapping, fixed here).** The condition-S account is **SUPPORTED** iff all of the following hold for P_cov:
- (a) S1 (restated, m fixed at each of its inventory values) is exact;
- (b) new-layout 3F passes (D3 CI contains 0) on every pairing N1-N6. P_min fails N4 and N5 and passes N1, N2, N3 and N6;
- (c) new-layout OM passes (ratio ≥ 0.8, CI lower bound ≥ 0.6);
- (d) new-layout S3 passes for each atom (bottle, plate): every k in {1, 5, 10} meets ratio ≥ 0.8 with lower bound ≥ 0.6;
- (e) familiar non-regression:
  - every familiar GO item that passes for P_min still passes (OM, 3F per pairing, S3 per atom);
  - pooled familiar known-pairing success (K1-K6), Δ = P_cov − P_min under the paired state bootstrap (2,000 draws, seeds pooled),
    does not have Δ < −0.10 with CI upper bound < 0.

Otherwise it is **NOT SUPPORTED**, and every failing clause is reported.

**Reported beside the verdict, with no verdict of their own.**
- Each component for P_cov against P_min, and P_cov's new-layout values against its own familiar ones: s known, s new, OM, D3 per
  pairing and S3 per cell.
- Competence: success per task in both layouts.
- The generated data's statistics.
- GPU and CPU cost.

**Prediction (fixed here, before any data).**
- (a) S1 is exact. It is structural, and data do not change it.
- New-layout known-pairing success rises from P_min's 0.46 to at least 0.65. Familiar known-pairing success stays within 0.05 of
  P_min's 0.90.
- (b) New-layout 3F passes on all six pairings, N4 and N5 included. K5's rack carries now have far onsets in their own support.
- (c) New-layout OM passes, as for P_min.
- (d) The bottle passes in the new layout. The created bottle part acts only inside its small start radius, and the backbone's
  transport, widened by the training atoms' generated demos, carries the hand there.
  - The plate does not pass at every k. The created target part fits its support from k human demos. New-layout carries start
    beyond that support, and the scale frame maps them into it, as it does in P_min (plate new passed only k = 10 there). This
    clause is the least certain.
- (e) Familiar non-regression holds.
- **Predicted verdict: NOT SUPPORTED, failing on clause (d) for the plate only.** If the plate passes too, the verdict is
  SUPPORTED. The prediction is then wrong on that clause, and that is reported.

### 1.9 Budget and stop

- **GPU.** 15 A40-h for the whole track. The estimate: smoke 0.2, 9 builds about 3.6, 18 creations about 0.3, S1 about 0.2, so
  about 4.3 A40-h.
- **CPU.** Generation, rollouts and reads run on overcap CPU.
- **Stop.** The track stops at the GPU cap. If W's generator keeps too few demos to train on (fewer than 10 per task), the arm
  stops and reports, with no build.

### 1.10 Generation and smoke results (2026-11-04; after the note, before any build)

Files: results/flawed/pp_cov_gen.json, pp_cov_gen_<task>.json and pp_cov_gen_stats.json. Code: experiments/pp_cov_gen.py and
pp_cov_train.py (commits 488e445f and 710168fc).

**The generator's own success under W** (filter: LIBERO's success predicate within 300 steps). Every task reached 50 kept demos
within 120 attempts, so no task was cut.

| task | attempts to 50 kept | layouts accepted | kept / layouts | rows | median steps |
|---|---|---|---|---|---|
| K1 | 72 | 70 | 0.71 | 4,214 | 84 |
| K2 | 61 | 59 | 0.85 | 4,033 | 76 |
| K3 | 63 | 57 | 0.88 | 5,365 | 97 |
| K4 | 116 | 110 | 0.45 | 5,245 | 100 |
| K5 | 77 | 71 | 0.70 | 5,812 | 112 |
| K6 | 107 | 102 | 0.49 | 5,101 | 96 |

**Rule checks** (smoke check 2).
- Every seed is W's and lies outside every listed base.
- Every attempted placement lies in the box.
- Each task keeps 50.
- The generated rows are 29,770 of 81,974 in the full build, a share of 0.36. The generated episodes are shorter than the human
  ones (median 76-112 steps against about 130).

**The success filter's effect on placements.** For each object, the table gives the share of placements within 0.10 m of the
object's own training spot. Uniform placement over the box predicts 0.23.
- Attempted placements: 0.07-0.38.
- Kept placements: 0.04-0.38.
- Within a task, kept and attempted shares differ by at most 0.08 for every object.
- So the filter adds no visible preference for the familiar spots. The clearance rules alone lower the cheese's share (0.07-0.17 attempted).

**Support after W** (generated rows, beside the human ones of §1.2). The table gives 10th / 50th / 90th / max percentiles.

| atom | human | generated |
|---|---|---|
| bowl, start offset (m) | 0.103 / 0.117 / 0.131 / 0.153 | 0.055 / 0.162 / 0.359 / 0.393 |
| cream cheese, start offset | 0.186 / 0.206 / 0.219 / 0.231 | 0.110 / 0.189 / 0.346 / 0.375 |
| bottle, start offset | 0.036 / 0.054 / 0.067 / 0.084 | 0.066 / 0.158 / 0.297 / 0.374 |
| plate, onset d0 (m) | 0.118 / 0.142 / 0.161 / 0.171 | 0.189 / 0.274 / 0.408 / 0.452 |
| stove, onset d0 | 0.067 / 0.251 / 0.271 / 0.285 | 0.097 / 0.284 / 0.373 / 0.435 |
| cabinet top, onset d0 | 0.075 / 0.286 / 0.312 / 0.326 | 0.239 / 0.340 / 0.446 / 0.519 |
| rack, onset d0 | 0.121 / 0.133 / 0.152 / 0.167 | 0.107 / 0.306 / 0.413 / 0.523 |
| bowl (target), onset d0 | 0.125 / 0.142 / 0.156 / 0.164 | 0.127 / 0.192 / 0.331 / 0.388 |
| rack, bearing (deg; min / 10 / 50 / 90 / max) | −22 / 50 / 60 / 65 / 73 | −11 / 23 / 56 / 89 / 99 |
| rack, release d (m; 10 / 50 / 90) | 0.058 / 0.080 / 0.100 | 0.063 / 0.077 / 0.090 |

- The generated support covers the new-layout ranges §20, §40 and §49 report:
  - start offsets up to 0.37-0.39 m (new-layout bowl 0.31, bottle up to 0.37);
  - rack onsets 0.31 / 0.38 / 0.40 m (generated 90th percentile 0.41 m, max 0.52 m);
  - rack bearings 21-54° (generated 10th-90th percentile 23-89°).
- Releases keep the demos' (d, h). The planner releases at the demos' median release offset.

**Smoke checks 1 and 3 pass.**
- With zero generated rows, the batch tensors equal `pp_p_train.rows` bitwise for full, nobottle and noplate. A 300-step build
  through pp_cov_train equals `pp_p_train.py build` bitwise in all 53 arrays.
- **Full-length check (added beyond the note).** pp_cov_train with zero generated rows, the full 30,000 steps, full, seed 10,
  equals P_min's own checkpoint pp_full_naf_s10 bitwise in 52 of its 53 arrays: all 19 object-term arrays, all 19 target-term
  arrays and 14 of the 15 buffers.
  - The only array that differs is the scalar buffer rtclip. P_min's target term was trained with `--rtclip`, which P_min never
    evaluates (PARTS_RESULTS §53).
  - So the zero-generated build is P_min's function exactly, on the code at HEAD f54101dd. That commit is the ladder agent's
    flag-off `--pgate` change.
- With generated rows: every human row's fields equal P_min's bitwise, episode keys are unique, held is 0/1, and the input
  centre is P_min's.
  - The first merge appended the generated rows after the human rows. The human-rows check then failed. `fa1_train.chunks` clips
    its look-ahead at the file's last row, so the last human demo (K8 demo 49) changed its final chunks' padding. The fix puts
    the generated rows first (commit 710168fc), and the check passes. No build had run.
- A 300-step build with generated rows trains.

**Fitted quantities with generated rows** (full build; every quantity by its existing rule, as §1.5 fixes).
- Start radii: bowl 0.381, cheese 0.373, bottle 0.353 m (P_min 0.140 / 0.231 / 0.079). The plate is unchanged at 0.279 m: it has
  no generated demos.
- Role radius: 0.268 m (P_min 0.225).
- Per-part carry radii: plate 0.424, stove 0.384, cabinet top 0.475, rack 0.478, bowl 0.351 m. The drawer and stove front are
  unchanged.
- **The grasp-mode inventory has K = 4** (P_min 3). The BIC rule adds a tight component (κ at its cap, 1000, at −84°) from the
  planner's K5 grasps.
  - Every target's m is unchanged: rack 2, drawer 1, all others 0.
  - The capacity is 2.0 bits (P_min 1.58); the task entropy is 1.06 bits, unchanged.
  - No target emits the fourth value.
  - This is the fixed rule's output on W's rows, not a choice made here. S1 is therefore restated with m fixed at each of
    0, 1, 2 and 3.

## 2. Pre-registration (fixed 2026-11-04, after P_cov's builds and creations, before any S1 run, rollout or read; m3_prereg tag pp_cov)

**Builds done.** cov_full, cov_nobottle and cov_noplate, seeds 10-12, each with k = 1, 5 and 10 creations where it applies. No
rollout or read of them exists. The only outputs seen are the fitted quantities and training losses.

**Primary arm, P_cov, exactly as §1 froze it.** The read is `experiments/pp_cov_read.py` with `--s1 pp_s1_mix_cov_full_m0.json`
through `_m3.json`, `--s2 pp_cov_roll:P2cov` and `--s3 pp_cov_roll:Pcov`, written to results/flawed/pp_cov_read.json. Its verdict
under §1.8's mapping is the arm's verdict. §1.8's prediction is unchanged.

**What the builds' fitted interfaces showed.** §1.5 fits the grasp-mode inventory on all rows. §1.10 looked only at the full build,
where the refit is harmless: every target keeps its m semantics. The held-out builds differ.

| build | P_min's inventory | P_cov's inventory | effect on S3 |
|---|---|---|---|
| nobottle | K = 2, every target m = 0 (0 bits of task entropy) | K = 4, components at −0.5°, −0.9°, −4.3° and −12.3°, two with κ at its cap; plate m = 2, drawer m = 3, others 0 (1.25 bits) | the bottle part is created from K4 (m = 0 only), and N4 (bottle → plate) reads it at m = 2, a value it never trained on |
| noplate | K = 3; rack and drawer have their own m, others 0 | K = 4; same m per target | the plate created at k = 1 labels itself m = 1, the drawer's value. P_min's emits 0. k = 5 and 10 emit 0 |

The near-duplicate components split the planner's tight grasp clusters; they are not distinct grasps. So P_cov's S3 cells on N4
(bottle) and at k = 1 (plate) may fail through the interface, in either layout, for a reason other than demo support. §1.8's clause
(d) or (e) could then read NOT SUPPORTED without testing the account. I did not anticipate this when writing §1.5.

**Secondary arm, P_cov_h, added here, labelled, before any read.** It equals P_cov in every respect except that the grasp-mode
inventory and each target's m are fitted on the build's human rows only. That reproduces P_min's interface in each build.
- The radii stay fitted on all rows.
- Only the object term reads m, so the object term is retrained on the same rows and seeds, and the target term is reused from
  cov_<cond> (`--reuse tgt:cov_<cond>`).
- The creations are rerun with `pp_p_train.py create`, unchanged: bottle from K4, plate from K1, k = 1, 5 and 10.
- The tags are covh_full, covh_nobottle and covh_noplate. S1 uses m fixed at 0, 1 and 2 (`pp_s1_mix_covh_full_m<v>.json`). The
  rollouts are `pp_cov_roll:P2covh` and `pp_cov_roll:Pcovh`. The read is `pp_cov_read.py --out pp_cov_h_read.json`.
- Check before its read: each build's printed inventory equals P_min's (§1.2's source, pp_train_<cond>_naf_s10.json), and each
  build's target term equals cov_<cond>'s bitwise.

**How the two arms relate (fixed here).**
- **P_cov's verdict is the arm's verdict.** P_cov_h never replaces it.
- P_cov_h is read with the same five clauses and reported beside it as secondary, with its own label.
- If the two verdicts differ, the report states which clauses changed, and that the difference lies in the interface refit. The
  primary verdict still stands. A claim that rests on P_cov_h alone goes to the user as post hoc.

**Prediction for P_cov_h (fixed here).** The same as §1.8's for every clause, with the plate still failing clause (d). Compared with
P_cov, its bottle S3 ratio on N4's cells is the same or higher in both layouts.

**The read's states.** Dev only: N2 main states 4-23, N2's seeds, seeds 10-12. No confirmatory or sealed read is run.

## 3. Results (2026-11-04; m3_prereg pp_cov; results/flawed/pp_cov_read.json, pp_cov_h_read.json, pp_cov_check.json; rollouts pp_cov_roll)

### 3.1 Verdict

**The condition-S account is NOT SUPPORTED under the registered mapping.** Clauses (b), (d) and (e) fail for P_cov, the primary arm.
The secondary arm P_cov_h is also NOT SUPPORTED: (b) and (d) fail, and (e) holds. The main arm's GO stays false; this arm cannot
change it.

| clause | P_min (reference) | P_cov (primary) | P_cov_h (secondary) |
|---|---|---|---|
| (a) S1 restated, exact | yes | **yes** (m = 0-3) | yes (m = 0-2) |
| (b) new 3F on all of N1-N6 | fails N4, N5 | **fails** N1, N5, N6 | fails N1, N5, N6 |
| (c) new OM | 1.08 [0.94, 1.29] | **passes** 0.82 [0.73, 0.91] | passes 0.91 [0.80, 1.04] |
| (d) new S3, each atom | fails both | **fails** plate; bottle passes | fails plate; bottle passes |
| (e) familiar non-regression | — | **fails** (3F N4, 3F N6, S3 plate lost) | holds |
| verdict | — | **NOT SUPPORTED** | NOT SUPPORTED |

**What coverage did and did not do.**
- It raised new-layout success on the **known pairings**: 0.46 to 0.64, Δ +0.18 [0.13, 0.24]. The familiar-minus-new gap on those
  pairings fell from 0.44 [0.40, 0.48] to 0.24 [0.15, 0.31].
- K5 succeeds in the new layout for the first time in any arm with tscale: 0 of 60 to 0.60 [0.45, 0.75]. Its arrival given hold
  goes from 0 to 0.82, and its place on arrival from 0 to 0.86. This agrees with §49-51's rack diagnosis, which the intervention tests directly.
- It did not measurably raise success on the **new pairings**: 0.37 to 0.40, Δ +0.03 [−0.06, 0.13]. That gap stays 0.42 [0.30, 0.55],
  against 0.55 [0.45, 0.64] for P_min.
- So the new-layout composition measures did not approach the familiar ones. New OM fell from 1.08 to 0.82 because the known
  pairings rose and the new ones did not.

### 3.2 GO components

**S2**, with P_cov's familiar values beside its new ones.

| measure | P_min familiar | P_cov familiar | P_cov_h familiar | P_min new | P_cov new | P_cov_h new |
|---|---|---|---|---|---|---|
| s known | 0.90 | 0.88 | 0.88 | 0.46 | 0.64 | 0.64 |
| s new | 0.92 | 0.82 | 0.86 | 0.37 | 0.40 | 0.44 |
| OM | 1.09 [1.04, 1.14] | 1.01 [0.94, 1.10] | 1.05 [0.98, 1.13] | 1.08 [0.94, 1.29] | 0.82 [0.73, 0.91] | 0.91 [0.80, 1.04] |

**3F**, D3 = observed − predicted [95% CI]. A pairing fails when the CI excludes 0, shown in bold.

| pairing | P_min familiar | P_cov familiar | P_cov_h familiar | P_min new | P_cov new | P_cov_h new |
|---|---|---|---|---|---|---|
| N1 | −0.03 [−0.08, 0.00] | −0.02 [−0.05, 0.00] | 0.00 [0.00, 0.00] | +0.13 [−0.01, 0.28] | **+0.18 [0.06, 0.30]** | **+0.21 [0.10, 0.33]** |
| N2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | −0.06 [−0.19, 0.04] | −0.02 [−0.14, 0.07] | −0.03 [−0.15, 0.08] |
| N3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.01 [0.00, 0.03] | −0.07 [−0.18, 0.02] | −0.05 [−0.10, 0.00] | +0.02 [0.00, 0.05] |
| N4 | −0.03 [−0.17, 0.08] | **−0.12 [−0.23, −0.02]** | −0.10 [−0.21, 0.00] | **+0.16 [0.04, 0.30]** | −0.03 [−0.13, 0.09] | −0.02 [−0.15, 0.12] |
| N5 | **+0.13 [0.06, 0.21]** | −0.08 [−0.16, 0.00] | −0.07 [−0.17, 0.02] | **+0.28 [0.18, 0.38]** | **−0.12 [−0.21, −0.03]** | **−0.13 [−0.22, −0.05]** |
| N6 | +0.01 [−0.10, 0.12] | **−0.20 [−0.31, −0.08]** | −0.14 [−0.29, 0.01] | +0.11 [−0.02, 0.23] | **−0.23 [−0.37, −0.09]** | **−0.20 [−0.34, −0.04]** |

**Measures that pass only through a CI clause** (PLAN_PARTS §10.13 applies to the ladder only and does not change this mapping;
they are listed so they are visible). Taking the ladder's tolerance of 0.10 as the yardstick for a 3F point estimate:
- P_cov: none.
- P_cov_h: familiar N4 (−0.100, CI upper bound 0.000) and familiar N6 (−0.140 [−0.287, 0.008]). Clause (e) holds for P_cov_h through these two.
- P_min: new N1 (+0.129) and new N6 (+0.107).
- These pass with a CI bound exactly at 0.00, with |D3| under 0.10: P_cov familiar N5 (−0.083) and new N3 (−0.050).

**S3**, transfer ratio [95% CI]; P_cov's and P_cov_h's cells use the same frozen criterion (ratio ≥ 0.8 and lower bound ≥ 0.6, at
every k). Bold cells fail.

| atom, layout | k | P_min | P_cov | P_cov_h |
|---|---|---|---|---|
| bottle, familiar | 1 | 0.91 [0.85, 0.96] | 0.91 [0.88, 0.94] | 0.99 [0.96, 1.03] |
| | 5 | 0.88 [0.80, 0.94] | 0.90 [0.85, 0.95] | 0.97 [0.93, 1.00] |
| | 10 | 0.88 [0.80, 0.95] | 0.91 [0.86, 0.98] | 0.97 [0.92, 1.02] |
| plate, familiar | 1 | 0.91 [0.84, 0.97] | **0.45 [0.34, 0.63]** | 0.81 [0.72, 0.90] |
| | 5 | 0.92 [0.85, 0.99] | 0.83 [0.77, 0.89] | 0.81 [0.71, 0.90] |
| | 10 | 0.93 [0.88, 0.97] | 0.86 [0.78, 0.93] | 0.81 [0.72, 0.90] |
| bottle, new | 1 | 0.82 [0.71, 0.95] | 0.93 [0.83, 1.10] | 0.88 [0.78, 0.97] |
| | 5 | **0.75 [0.61, 0.87]** | 0.98 [0.87, 1.21] | 0.81 [0.72, 0.90] |
| | 10 | **0.74 [0.62, 0.87]** | 0.92 [0.82, 1.06] | 0.86 [0.71, 1.00] |
| plate, new | 1 | **0.79 [0.53, 1.15]** | **0.50 [0.27, 0.90]** | **0.56 [0.37, 0.82]** |
| | 5 | **1.00 [0.52, 2.08]** | **0.82 [0.42, 1.54]** | **0.71 [0.40, 1.27]** |
| | 10 | 1.44 [0.89, 2.78] | **0.80 [0.53, 1.34]** | **0.70 [0.44, 1.03]** |

- The bottle now passes in the new layout at every k, in both arms. P_min passed only k = 1.
- The plate fails in the new layout at every k, in both arms. The cells fail on their lower bounds, and on the ratio as well in most.

**Competence**: success per task, P_min / P_cov / P_cov_h.

| task | familiar | new |
|---|---|---|
| K1 | 1.00 / 0.98 / 0.97 | 0.35 / 0.60 / 0.57 |
| K2 | 1.00 / 0.98 / 0.98 | 0.63 / 0.83 / 0.80 |
| K3 | 1.00 / 0.98 / 0.98 | 0.87 / 0.97 / 0.95 |
| K4 | 0.95 / 0.70 / 0.77 | 0.47 / 0.38 / 0.45 |
| K5 | 0.48 / 0.72 / 0.63 | 0.00 / 0.60 / 0.63 |
| K6 | 0.97 / 0.92 / 0.93 | 0.45 / 0.48 / 0.43 |
| K7 | 0.27 / 0.13 / 0.18 | 0.02 / 0.00 / 0.00 |
| K8 | 0.20 / 0.10 / 0.12 | 0.03 / 0.07 / 0.05 |
| N1 | 0.97 / 0.97 / 0.98 | 0.35 / 0.50 / 0.53 |
| N2 | 1.00 / 0.98 / 0.98 | 0.38 / 0.48 / 0.50 |
| N3 | 1.00 / 0.98 / 0.98 | 0.42 / 0.48 / 0.57 |
| N4 | 0.78 / 0.70 / 0.75 | 0.27 / 0.30 / 0.33 |
| N5 | 0.95 / 0.73 / 0.78 | 0.48 / 0.38 / 0.45 |
| N6 | 0.80 / 0.57 / 0.67 | 0.32 / 0.25 / 0.28 |
| N7 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| N8 | 0.02 / 0.22 / 0.35 | 0.18 / 0.40 / 0.48 |

- Pooled familiar K1-K6: Δ −0.02 [−0.06, 0.02] for P_cov, −0.02 [−0.07, 0.02] for P_cov_h, so no regression on that clause.
- Pooled success hides a trade inside the bottle tasks. K4's familiar success falls by 0.18-0.25, and K5's rises by 0.15-0.23.
- Per-compound CIs are in pp_cov_read.json, under `competence`.

### 3.3 The prediction against the outcome

| fixed in §1.8 | outcome |
|---|---|
| S1 exact | right |
| new known ≥ 0.65 | narrowly wrong: 0.644 (P_cov), 0.639 (P_cov_h) |
| familiar known within 0.05 | right: −0.02 |
| (b) new 3F passes on all six | **wrong**: N1, N5 and N6 fail. N4 does pass |
| (c) new OM passes | right |
| (d) bottle passes, plate does not | right, in both arms |
| (e) familiar non-regression | **wrong for P_cov**, right for P_cov_h |
| verdict NOT SUPPORTED, failing (d) for the plate only | right verdict, **wrong reason**: (b), and (e) in P_cov, fail too |
| §2: P_cov_h's bottle ratio on N4 is the same or higher | right in the familiar layout (0.98-1.00 against 0.78-0.86), **wrong in the new** (0.79-0.87 against 0.88-1.04). Point ratios of per-cell rates, no CI |

### 3.4 The interface confound, measured

§2 added P_cov_h because refitting the inventory on the planner's rows gave created parts an m they never trained on. The read
confirms that this damaged P_cov, in the cell §2 named.
- The plate created at k = 1 emits the drawer's m in P_cov. On familiar N4 it succeeds at 0.08, against 0.62 in P_cov_h and 0.83 in
  P_min. On familiar K1 it succeeds at 0.67, against 0.98 and 1.00.
- P_cov loses clause (e) on familiar 3F N4, 3F N6 and S3 plate. P_cov_h loses none of them. As §2 fixed, this difference lies in the
  interface refit, and P_cov's verdict stands. It does not decide the verdict: (b) and (d) fail in both arms.
- P_cov_h's own familiar S3 plate passes narrowly at each k (0.81, lower bound 0.71-0.72), and its bottle N4 familiar 3F passes at its CI bound.

### 3.5 Where the remaining failures sit (post hoc, descriptive, on the registered rollouts; no verdict)

Per task, new layout, as hold / arrival given hold / place on arrival, P_min → P_cov → P_cov_h.

| task | P_min | P_cov | P_cov_h |
|---|---|---|---|
| K1 (bowl, plate) | 0.93 / 0.82 / 0.46 | 0.98 / 1.00 / 0.61 | 1.00 / 0.97 / 0.59 |
| K4 (bottle, cabinet; m = 0) | 0.50 / 0.90 / 1.00 | 0.42 / 0.88 / 1.00 | 0.52 / 0.87 / 1.00 |
| K5 (bottle, rack; m = 2) | 0.48 / 0.00 / — | 0.85 / 0.82 / 0.86 | 0.85 / 0.90 / 0.83 |
| K6 (cheese, bowl) | 0.52 / 0.94 / 0.93 | 0.58 / 0.91 / 0.91 | 0.58 / 0.94 / 0.79 |
| N1 (cheese, plate) | 0.52 / 0.81 / 0.84 | 0.58 / 0.94 / 0.91 | 0.58 / 0.97 / 0.94 |
| N4 (bottle, plate; m = 0) | 0.50 / 0.93 / 0.57 | 0.42 / 0.92 / 0.78 | 0.52 / 0.90 / 0.71 |
| N5 (bottle, stove; m = 0) | 0.50 / 0.97 / 1.00 | 0.42 / 0.92 / 1.00 | 0.52 / 0.87 / 1.00 |
| N6 (bottle, bowl; m = 0) | 0.50 / 0.90 / 0.70 | 0.42 / 0.92 / 0.65 | 0.52 / 0.87 / 0.63 |

- **Hold moved at m = 2 and not at m = 0, or for the cheese.** The bottle's hold at m = 2 went from 0.48 to 0.85. At m = 0, with its start
  offsets now covered (§1.10), it stayed at 0.42-0.52 against P_min's 0.50. The cheese went from 0.52 to 0.58. Why is not established.
- **N5 fails through the pooled hold.** 3F's h_o pools K4 and K5, but N5 reads m = 0 and holds exactly as K4 does (the object term is
  target-blind at fixed m), while K5 holds at 0.85. With K5 fixed, h_o rises to 0.63-0.68 and over-predicts. On K4's factors alone, N5's
  prediction is 0.35 (P_cov, observed 0.38) and 0.43 (P_cov_h, observed 0.45). So N5's failure says little about composition. It does
  not change clause (b); the frozen estimator stands.
- **N1 and N6 fail at the place, and the place depends on the pair.** The plate's q_t comes from K1 (bowl on plate, 0.59-0.61), while cheese on plate
  places at 0.91-0.94, so N1 exceeds its prediction. Bottle in bowl places at 0.63-0.65, while q_t from K6 (cheese in bowl) is 0.79-0.91, so
  N6 falls short. P_min shows the same two gaps (N1 0.46 against 0.84; N6 0.93 against 0.70). Coverage raised K1's arrival to about 1.00, and its
  place stayed at 0.6. An object-target interaction at the release is one reading. No test here separates it from a remaining support gap.
- **Generated rows cost familiar bottle hold at m = 0**: K4, N4, N5 and N6 go from 0.95 to 0.73 (P_cov) and 0.85 (P_cov_h). K5's rises from 0.70 to 0.95.
- **Known-pairing success in the new layout sits near the generator's own kept rate** (P_cov / P_cov_h / generator): K1 0.60 / 0.57 / 0.71, K2 0.83 / 0.80 / 0.85, K3 0.97 / 0.95 / 0.88,
  K4 0.38 / 0.45 / 0.45, K5 0.60 / 0.63 / 0.70, K6 0.48 / 0.43 / 0.49. The generator's rate is over W's placements, not the new layout's states, and carries no CI, so this comparison is loose. The two tasks
  that the generator completes least often, K4 and K6, are the ones whose hold barely moved.

### 3.6 What this does and does not say about §54's account

- **As registered:** the account is not supported. Widening demo support by one layout-agnostic rule did not bring the new-layout composition measures to the
  familiar ones.
- **What it supports, without a registered verdict:** demo support does limit new-layout competence on the pairings it covers. The gain on known pairings and
  K5's recovery are both large, with CIs excluding 0.
- **What it weakens:** §54's "the new-layout shortfall is a failure of demo support" as the whole account. New pairings did not gain measurably, and new 3F now fails on the
  negative side too, after P_min's positive failures went away. §54's statement should be read with this result beside it.
- **What the arm cannot separate:** the generator is a limited teacher (kept rate 0.45-0.88). A failure at bottle hold under m = 0 may be insufficient support of the right kind, a weak teacher,
  or a limit of the class. The plate's S3 cells have wide CIs (plate new k = 5 spans 0.42-1.54), so their failure on the lower bound is weak evidence.

### 3.7 Construction checks, cost and deviations

**Checks** (results/flawed/pp_cov_check.json).
- 444 rollout files exist, each with 20 episodes, and no episode error. S1 is exact on all seeds at each m.
- P_cov_h's target terms equal cov_<cond>'s bitwise (all 9 builds, 19 arrays each), and its inventories equal P_min's exactly.
- **Shared code moved under the builds.** The ladder's L4 commits (c3515936, 89f623d4) landed on pp_p_train and fa1h_roll at 09:23 and 09:28, while
  cov_full s12, cov_nobottle's creations, and cov_noplate s12 with its creations were still to run or running. Checked at HEAD afterwards: the 300-step P_min build and the 300-step generated build are
  bitwise equal to those built before L4; a bottle creation equals P_min's checkpoint in all 53 arrays; a plate creation equals it in all 52 shared arrays (the older checkpoint has no rrel_p buffer). So
  the code change did not alter these paths. The first plate comparison stopped on that key difference, and pp_cov_check compares shared keys.
- The pre-registered files' hashes match at read time, except docs/PLAN_PARTS.md, which gained the user's item 13 (ladder gate) from the ladder track.
- P_cov's and P_cov_h's rollouts ran at the post-L4 code, and P_min's ran before it. The ladder's own smoke reports flag-off rollouts bitwise on the same host, and reports rollouts differing across hosts
  (LADDER_RESULTS 5.7). I did not repeat that check.

**Cost.** GPU 32,230 s = 8.95 A40-h of the 15 cap (sacct over every attempt, requeues included): P_cov's builds and creations 5.02, P_cov_h's 3.37, smokes and checks 0.56. The note's estimate was 4.3; the overrun is P_cov_h, the
preempted noplate attempts and the full-length check. CPU: 18 generation shards (about 1.8 h), 60 rollout jobs (about 21 h), S1 and reads.

**Deviations.**
- Login-node rule breaks, all read-only:
  - four pipelines across three commands (`| head` three times, `cat | tail` once);
  - one git pathspec glob (`'experiments/pp_*'`) in my first listing;
  - one filename glob (`experiments/n2_*.py`) in a `grep -r`;
  - two uses of `sed -n` on source files.
- P_cov_h and the full-length check were not in the note or the assignment. §2 registered P_cov_h before any read, and P_cov stays primary.
- §1.5's rule was applied as frozen. I had not foreseen its effect on the held-out builds' inventory, and §1.10 inspected only the full build.
- §3.5's per-task phase table and the factor recomputation are post hoc.

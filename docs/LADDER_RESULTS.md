# Layer-3 supervision ladder: results

PLAN_PARTS §5 and §10 item 10 (user, 2026-11-04). The ladder starts from L0 = P_min (PARTS_RESULTS §53). It replaces one
oracle piece per rung and keeps every other piece at the previous rung. The gate applies to the familiar layout only. Every
claim here is scoped as "familiar layout, given Layer-2 GO false in the new layout". New-layout measures are reported at every
rung, with no verdict.

- **L0 = P_min.**
  - Flags: `--rank 32 --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize`.
  - Checkpoints: `pp_{full,nobottle,noplate}_naf_s{10,11,12}`, with the created parts `P_<atom>_k{1,5,10}.npz`.
  - Rollouts: pp_cons_roll (`P2naf`, `Pnaf_<cond>_k<k>`).
  - Read: pp_cons_min_read.json.
- **Gate (§10.10).** Each familiar-layout composition measure stays within 0.10 of L0: S1, OM, 3F per pairing, and S3 per
  atom. A measure that L0 itself fails is not counted (familiar 3F on N5).

## 1. L1 design note: a gripper-width and proprioception event instead of the sim-contact gate (fixed 2026-11-04, before any code of this rung)

**What L1 replaces.** PLAN_PARTS §5 lists the rung as "sim-contact phase gate → gripper width + proprioception event". In
PARTS_RESULTS §54's supervision list, it is the row "contact gate (held) for Φ, and the held labels of demo rows". Its source
is the simulator's contact list. L1 replaces that row and nothing else. Frames, slots, role assignment, sizes, the partition and
the demo-to-task labels stay oracle, as at L0.

### 1.1 Where the contact gate and the held labels enter

The oracle signal is `fa1b_common.Held`, a two-state machine stepped once per simulator step.
- It turns on when the role object touches both fingers (MuJoCo contacts) and has risen `LIFT` = 0.01 m above its height at the
  first step.
- It stays on while the object touches at least one finger, and turns off when it touches neither.

The signal enters in three places.

**Training (experiments/pp_p_train.py).** `rows()` reads `held` from DATA/demos_b.npz. That array is `Held` stepped over each
demo's kept steps, with the recorded simulator states set and their contacts read. Everything below derives from that one array:
- **Φ's gate in the loss.** `PartsFlow.velocity` returns (1 − held)·v_o + held·v_t. So the flow loss trains the object term on
  not-held rows and the target term on held rows. `--decouple`'s role split relies on held being 0/1, which is asserted.
- **The backbone-only term** (`--rtransport`) is weighted by (1 − held).
- **Onset quantities**, each frozen at the hold's onset:
  - `canon_rows`: zrest and cyaw, the canonical place frame;
  - `rot0_rows`: the hand's rotation at the onset;
  - `d0t_rows`: the object-to-target distance at the onset, used by `--tscale`.
- **Fitted radii and inventory.**
  - `part_radii`: object parts from not-held rows, target parts from held rows. The target radii are the scale frame's r_t.
  - `role_radius`: from not-held rows.
  - `grasp_angles`, and through it the m inventory and each target's m: close rows are "not held and the command closes".
  - `release_radii` uses held too, but it is not on P_min.
- **S3 creation.** The created part trains on `rows()` of its k demos, so its loss split and its radius or m come from the same
  held labels.

**Rollout (experiments/fa1h_roll.py `episode`, called by pp_p_roll.py).** `track()` steps `FB.Held` on the simulator's
finger contacts and the object's position each step. The policy reads, from that signal:
- h_on, as `batch_from_scene(held=...)`, which is Φ's gate;
- `FK.Canon`'s zrest and cyaw, stepped on h_on;
- rot0, through `onset_rot`;
- d0t, through `onset_d`.

**Reads (pp_l2r32_read.py, pp_round_read.py, pp_cons_read.py).** The 3F factors use oracle quantities that the rollout logs from
the same h_on:
- h_o is the hold rate, from `held_step`;
- a_o is arrival given hold, from `min_dxy_held` over held steps.

`held_segments`, `held_end` and `onset_canon` are also logged. Success itself is LIBERO's predicate, not the gate.

**S1 (pp_s1_mix.py, pp_f1_oracle.py).** S1 sets held explicitly:
- 0 for the object-term swap test and the grasp-phase frames;
- 1 for the target-term swap test.

The place-phase probe frames take their phase from the demos' oracle label. That choice only picks which states to probe. No
S1 criterion depends on it.

### 1.2 The replacement event and its detector

**The event `ProprioHeld`.** It is a two-state machine of the same form as `Held`, stepped once per step (per kept demo row in
training). It reads only proprioception and the robot's own last command:
- the finger width w = q_left − q_right, from `robot0_gripper_qpos` (demos.npz `grip`);
- the last gripper command c (`prev`: +1 closes, −1 opens);
- the hand's height z, from `robot0_eef_pos`.

It reads no object pose, no contact and no atom identity. Its rules:
- **When c opens:** the event is off, and the close reference is cleared.
- **While c closes:**
  - z_min is the lowest hand height since the close command began.
  - It turns **on** when w ≥ τ_w (the fingers are stopped by something) and z − z_min ≥ LIFT (the hand has risen LIFT since
    closing).
  - It turns **off** when w < τ_w (the fingers closed past the object: it slipped or was missed).

`LIFT` = 0.01 m is the oracle rule's own constant, carried over unchanged. It plays the same role: the oracle asks the object to
rise LIFT, and this rule asks the hand to rise LIFT after closing.

**The threshold τ_w, by one fixed rule.** τ_w is fitted from the build's own training demos, without any label:

  τ_w = ½ · (min of w over the build's demo rows + 1st percentile of w over its close-commanded rows).

- The first term is the narrowest width the fingers reach in the demos, the closed limit as observed.
- The second term is the narrowest width at which closed fingers sit in the demos.
- τ_w is the midpoint between the two.

The rule is the same for every build and every atom. Nothing is set per scene or per task, and no oracle label is used to fit
it.
- A created part does not refit τ_w. The event belongs to the robot, not to an atom, so creation uses its library's τ_w.
- τ_w and LIFT are stored in the checkpoint's metadata (`pgate`), so the rollout uses exactly the build's detector.

**Pre-design look at the data (descriptive, label-free; disclosed).** Before fixing this rule, the demos' widths were tabulated
without the oracle label (`.claude_tmp/pp/L_desc.py`):
- The width on close-commanded rows settles at about 0.004-0.006 m for the bowl's rim, 0.013-0.03 m for the bottle and 0.043 m
  for the cheese. Open is 0.08 m. The narrowest row is 0.0012 m.
- The hand often starts to rise while the fingers are still closing. That is why the event can turn on a few steps before the
  oracle's object lift. This known lag is measured, not tuned away: no finger-speed threshold is added.
- The bowl's rim leaves a small margin above the closed limit. If the detector confuses an empty close with a rim grasp, it
  will show in the agreement, both on demo rows and in rollouts.

### 1.3 What it replaces, and what stays

Under `--pgate`, every use of held listed in §1.1 for training and the policy takes the event instead of the contact label. That
covers:
- Φ's gate;
- the role split of the loss;
- the backbone-only weight;
- the onset quantities zrest, cyaw, rot0 and d0t;
- the radii;
- the close rows of the m inventory;
- creation rows;
- in the rollout, the policy's h_on, its Canon, rot0 and d0t.

Two things stay oracle.
- **The read's measurement instrument.** The rollout still steps the oracle `Held` and logs `held_step`, `min_dxy_held`,
  `held_segments` and `onset_canon` from it. So h_o, a_o and q_t keep their L0 meaning, and L0 and L1 are compared on one
  instrument. The event's own trace is logged next to it: its onset step, its segments, and per-step agreement with the oracle.
- **The probe frames of S1** (pp_f1_oracle.py, unchanged).

**Why S1 stays exact.** Φ is unchanged: v = (1 − c)·v_o + c·v_t, with c now the event. The event reads no atom identity, so:
- swapping the target atom changes neither c nor the object term's inputs;
- swapping the object part changes nothing the target term reads.

S1's tests set c explicitly (0 or 1), and S1 is restated with m fixed at each value of L1's own inventory. So the swap tests and
the grasp-phase components stay exactly 0 by construction, and the S1 run checks this.

**What must be rebuilt.** The held labels split the training rows between the roles. So both role terms change, and neither can
be reused bitwise (`--reuse` does not apply).
- Builds: full, nobottle and noplate, seeds 10-12, both roles trained, `--decouple` kept.
- Creations: the bottle part (from K4) and the plate part (from K1), at k = 1, 5 and 10.

Each creation keeps the asserted guarantee that no parameter outside the created part changes.

**Default off.** Without `--pgate`:
- `rows()` reads demos_b.npz's held, as at L0;
- the checkpoint carries no `pgate`;
- `episode` steps only the oracle signal.

Smoke checks before any build:
- (a) with the flag off, `rows()` equals the committed L0 code's `rows()` bitwise, on every tensor of a build and a creation;
- (b) with the flag off, an L0 checkpoint loads with no detector, and one task file of its rollout, re-run with the new code,
  equals the committed pp_cons_roll file in every episode field except wall-clock time and host;
- (c) the detector's agreement with the oracle on all demo rows, reported per task: row accuracy, the share of held rows, and
  onset and offset lags per demo.

### 1.4 Code and smoke checks (2026-11-04; commits f54101dd and the next; results/flawed/pp_L1_smoke_{rows,agree,roll}.json)

- **Code, behind `--pgate` (default off).**
  - `parts_flow.py`: `ProprioHeld`, `pgate_fit` (the fixed rule) and `pgate_rows`. `PartsFlow.load` sets `pgate` only from a
    checkpoint whose metadata carries it.
  - `pp_p_train.py`: `rows(pgate=...)` and the `--pgate` build and creation.
  - `fa1h_roll.py`: `policy_gate`.
- **(a) Flag off, training rows.** `rows()` equals the committed L0 code's (commit ba4f1e42) bitwise, on every tensor, for the
  full, nobottle and noplate builds and for the k = 10 creation rows of K4 and K1. An L0 checkpoint (pp_full_naf_s10) loads
  with no detector.
- **(b) Flag off, rollout.** Task 0 (K1, familiar) of P2naf seed 10, re-run with the new code, equals the committed
  pp_cons_roll file in all 20 episodes, every field but wall-clock time. The task succeeds in all 20 episodes, so this checks
  the code path, not a varied outcome.
- **(c) Agreement with the oracle on all 52,204 demo rows** (full build's rule, τ_w = 0.00253 m):

| | rows | accuracy | held share: event / oracle | P(event \| oracle) | P(oracle \| event) | onset lag, rows (median) | last held row lag (median) |
|---|---|---|---|---|---|---|---|
| all | 52,204 | 0.968 | 0.311 / 0.313 | 0.946 | 0.951 | −1 | −2 |
| K1 bowl→plate | 4,662 | 0.960 | 0.370 / 0.362 | 0.956 | 0.936 | −2 | −1 |
| K2 bowl→stove | 5,077 | 0.949 | 0.405 / 0.385 | 0.960 | 0.912 | −2 | −2 |
| K3 bowl→cabinet | 5,094 | 0.950 | 0.473 / 0.452 | 0.967 | 0.925 | −3 | −1 |
| K4 bottle→cabinet | 5,389 | 0.961 | 0.456 / 0.480 | 0.935 | 0.984 | 0 | −3 |
| K5 bottle→rack | 8,801 | 0.977 | 0.350 / 0.362 | 0.952 | 0.984 | 0 | −2.5 |
| K6 cheese→bowl | 5,340 | 0.954 | 0.300 / 0.331 | 0.882 | 0.976 | 0 | −4 |
| K7 drawer, bowl | 10,205 | 0.974 | 0.285 / 0.275 | 0.972 | 0.935 | −2 | −1 |
| K8 push plate | 7,636 | 0.995 | 0.000 / 0.005 | 0.0 | — | — | — |

- A negative lag means the event comes earlier than the oracle.
  - For the bowl, the event turns on 2-3 rows early: the hand rises while the fingers are still closing on the rim.
  - The event turns off 1-4 rows early: it ends when the open command is given, while the oracle waits for the fingers to leave
    the object.
  - K8 has no close command, so the event never fires there. The oracle marks 0.5% of its rows (2 demos) as held.
- In two demos the oracle holds and the event never does. In one demo the event holds and the oracle never does.
- **The thresholds per build:**

| build | τ_w (m) | min width | 1st pct of close-commanded width |
|---|---|---|---|
| full | 0.00253 | 0.00123 | 0.00383 |
| nobottle | 0.00261 | 0.00147 | 0.00375 |
| noplate | 0.00250 | 0.00123 | 0.00376 |

- On all rows, the three thresholds give the same agreement to three decimals.
- **Estimator check.** `experiments/pp_ladder_read.py` with L0 as both arms (`--rung L0self`) reproduces pp_cons_min_read.json:
  - every measure's value;
  - L0's pass and fail per item, with familiar N5 failing;
  - Δ = 0 with CI [0, 0] everywhere;
  - verdict HOLDS.

  That run reads only L0 rollouts that were already read in PARTS_RESULTS §53.
- **Build note.** P_min's flags carry no `--rtclip`, so the L1 builds have rtclip = 0. L0's target term came from P_nsz, which
  carried the scalar, but it is never read for any target part that has its own radius (PARTS_RESULTS §53). Under the event, K8
  has no held rows, so the stove-front target part has no radius. No read task uses it.

The protocol and the verdict mapping of the L1 read follow in §2. They are fixed and pre-registered before any read of this
rung.

## 2. L1 protocol and verdict mapping (fixed 2026-11-04, before any L1 rollout, S1 run or read; m3_prereg tag pp_L1)

**Arm L1.** P_min with the event in place of the contact gate: `--rank 32 --psupport --tscale --decouple --imode --rtransport
--bbnom --bbnosize --pgate`.
- Builds: `pp_{full,nobottle,noplate}_L1_s{10,11,12}`, both roles trained, no reuse (§1.3).
- Creations: the bottle from K4 and the plate from K1, k = 1, 5 and 10, `P_<atom>_k<k>.npz`.
- Part size and rank are L0's. No other option changes.

**Data.**
- Dev states 4-23 with N2's noise seeds, seeds 10-12, paired with L0 state for state.
- S2: all 32 tasks, `P2L1` in results/flawed/pp_L1_roll.
- S3: `PL1_<cond>_k<k>` in the same directory, on L0's task lists (nobottle: tids 6, 7, 22-27; noplate: tids 0, 1, 16, 17, 22, 23).
- S1: `pp_s1_mix.py` on pp_full_L1_s10-12, with `--fix-m v` for every value v of the L1 full build's inventory, giving
  `pp_s1_mix_full_L1_m<v>.json`.

**Measures and read.** `experiments/pp_ladder_read.py --rung L1` writes results/flawed/pp_ladder_L1_read.json. Its estimators
are pp_l2r32_read's, asserted equal on both arms.
- **S1** (restated with m fixed): exact for every seed and every value of m.
- **OM** per layout.
- **3F** per layout and new pairing: D3 for N1-N6.
- **S3** per atom and layout: the mean over k = 1, 5, 10 of the transfer ratio.

For each measure, Δ = L1 − L0, with a 95% CI from 2,000 paired draws of the 20 states (the same draw for both arms, seeds pooled,
rng seed 0).

**Counted measures** (familiar layout only, §10.10). These are S1, plus each familiar measure that L0 passes under its own
Layer-2 criterion:
- OM;
- 3F on N1, N2, N3, N4 and N6 (N5 is excluded: L0 fails it, D3 +0.133 [0.058, 0.208]);
- S3 for the bottle and for the plate.

That is 8 measures plus S1.

**Verdict mapping (one rule, frozen here).**
- A counted measure **breaks** iff |Δ| > 0.10 and its CI excludes 0.
  - The 0.10 tolerance is §5's, fixed before any rung.
  - The CI clause keeps state-draw noise from breaking a rung, as pp_cons's rule did. Dev cells vary by about ±0.1.
  - The rule is two-sided, as "within 0.10 of L0" is.
- **L1 holds iff S1 holds and no counted measure breaks.** Otherwise **L1 breaks**, and the broken measures are named.
- The new layout's measures and Δs are reported with no verdict, and so is competence (s known, s new, per task).

**Reported, descriptive.**
- The event against the oracle in the L1 rollouts (`pp_L1_smoke.py rollagree`): step accuracy, onset lag, and episodes held by
  one signal only.
- The L1 inventory: K, bits, and each target's m.
- Build and creation costs.

**Stop rules.**
- **If L1 breaks**, the ladder stops for the user's decision (§10.10). The report names the piece L1 removed and the measures it
  broke.
- **If L1 holds**, L2 (frames from VLA visual tokens) is designed next.
  - Its sources (§5: the S0 KV tools) are checked first.
  - If L2 needs large new infrastructure, or forces a plan-level choice, the options are written up and the ladder pauses.
- **Budget:** this track stops at 15 A40-h of GPU.

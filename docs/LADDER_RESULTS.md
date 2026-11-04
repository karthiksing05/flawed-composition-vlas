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

**Amendment 1 (2026-11-04, after the first build attempt and before any L1 rollout, S1 run or read; m3_prereg tag
pp_L1_amend1).** The arm adds `--rtclip`. Its flags become `--rank 32 --rtclip --psupport --tscale --decouple --imode
--rtransport --bbnom --bbnosize --pgate`.
- **What happened.** The nine builds with the registered flags reached a NaN loss by step 1,000. The cause is a target part with
  no event-held rows: the stove-front part (K8), and the plate in noplate.
  - Such a part's radius falls back to the rtclip scalar, which P_min's flags leave at 0. So s_t = 0, and the target term
    divides by it.
  - The target term is computed on every row before Φ's gate multiplies it by 0, so 0·inf = NaN.
  - Under the contact gate, K8 has 0.5% held rows, so its part had a radius.
- **Why this is no change to L0.** L0's target term is P_nsz's, built with `--rtclip` (PARTS_RESULTS §53: P_min's target term
  carries the scalar, which is never read for a part with its own radius). Adding the flag reproduces L0's actual target
  configuration.
  - The scalar acts only as the fallback radius of a part with no held rows.
  - No read task's target part lacks a radius. The plate gets its own radius at creation.
- The nine builds were cancelled at about 3 minutes. Their GPU time is counted, and they wrote no checkpoint. Nothing else in
  the protocol or the mapping changes.

## 3. L1 results (2026-11-04; m3_prereg pp_L1 + pp_L1_amend1; results/flawed/pp_ladder_L1_read.json, pp_s1_mix_full_L1_m{0,1,2}.json, pp_L1_smoke_rollagree.json; rollouts pp_L1_roll)

**Verdict (pp_L1 mapping): L1 HOLDS.** S1 is exact. No counted familiar measure moves by more than 0.10 from L0. Scoped claim:
in the familiar layout, given Layer-2 GO false in the new layout, P_min composes as well with the sim-contact gate replaced by
a gripper-width + proprioception event as with the gate. All 30 rollout jobs finished with no episode error, and no cell has a
missing episode.

**S1.** With m fixed at 0, 1 and 2 (the L1 full inventory: K = 3, 1.58 bits, the same as L0), both swap tests are exactly 0 on
seeds 10-12, and so is every target-into-grasp component.

**Familiar layout (counted).** Δ = L1 − L0, 2,000 paired state draws, seeds pooled.

| measure | L0 | L1 | Δ [95% CI] | breaks? |
|---|---|---|---|---|
| OM | 1.089 | 1.151 | +0.062 [0.014, 0.115] | no |
| 3F N1 (D3) | −0.033 | +0.017 | +0.050 [−0.017, 0.117] | no |
| 3F N2 | 0.000 | +0.033 | +0.033 [0.000, 0.083] | no |
| 3F N3 | 0.000 | +0.067 | +0.067 [0.017, 0.133] | no |
| 3F N4 | −0.033 | −0.050 | −0.017 [−0.108, 0.092] | no |
| 3F N6 | +0.011 | −0.071 | −0.082 [−0.199, 0.022] | no |
| S3 bottle (mean ratio over k = 1, 5, 10) | 0.888 | 0.918 | +0.029 [0.000, 0.064] | no |
| S3 plate | 0.916 | 0.904 | −0.012 [−0.078, 0.066] | no |
| 3F N5 (not counted: L0 fails it) | +0.133 | +0.100 | −0.033 [−0.083, 0.017] | — |

- OM and 3F N3 move by a CI-supported amount, but by less than 0.10. N6 moves by −0.082, with a CI that contains 0.
- **Under Layer-2's own criteria** (reported, not the gate), L1's familiar 3F fails on N3 (+0.067 [0.017, 0.133]) and on N5
  (+0.100 [0.033, 0.183]). L0 fails on N5 only. Every familiar S3 cell passes, at k = 1, 5 and 10, for both atoms.
- **Competence.** Familiar known-pairing success is 0.87 against 0.90, and new-pairing success 0.88 against 0.92. The largest
  per-task drops are K6 (cheese→bowl, 0.82 against 0.97) and N6 (bottle→bowl, 0.70 against 0.80).

**New layout (reported, no verdict).**

| measure | L0 | L1 | Δ [95% CI] |
|---|---|---|---|
| OM | 1.081 | 1.178 | +0.097 [−0.119, 0.374] |
| 3F N1 | +0.129 | +0.177 | +0.048 [−0.056, 0.143] |
| 3F N2 | −0.063 | +0.001 | +0.064 [−0.085, 0.242] |
| 3F N3 | −0.067 | −0.027 | +0.040 [−0.075, 0.142] |
| 3F N4 | +0.164 | +0.229 | +0.065 [−0.033, 0.175] |
| 3F N5 | +0.276 | +0.247 | −0.029 [−0.092, 0.034] |
| 3F N6 | +0.107 | +0.039 | −0.068 [−0.187, 0.042] |
| S3 bottle | 0.770 | 0.698 | −0.072 [−0.153, 0.006] |
| S3 plate | 1.076 | 1.011 | −0.065 [−0.414, 0.500] |

- Under Layer-2's criteria, new-layout 3F fails on N1, N4 and N5 (L0: N4 and N5). S3 passes 1 of 6 cells, plate k = 1 (L0: 2
  of 6).
- New-layout success: 0.46 against 0.46 (known), 0.36 against 0.37 (new). K5 stays at 0.

**The event in the rollouts** (descriptive; every L1 rollout file, S2 and S3 pooled).

| layout | episodes | step agreement with the oracle | event on, oracle off (steps) | oracle on, event off (steps) | onset lag, median (q10, q90) | episodes where only the event holds | only the oracle |
|---|---|---|---|---|---|---|---|
| familiar | 2,220 | 0.951 | 11,822 | 2,809 | −1 (−4, 0) | 153 | 5 |
| new | 2,220 | 0.971 | 12,011 | 3,818 | −2 (−72, +1) | 549 | 5 |

- The event's errors run one way: it holds when the oracle does not.
  - In the new layout, a quarter of episodes (549) have an event hold and no oracle hold. 462 of them are bottle tasks (K4 93,
    K5 20, N4 157, N5 94, N6 98).
  - None of these episodes succeeds, in either layout (153 familiar, 549 new).
  - **Corrected 2026-11-04.** The first version of this section said these episodes close on something the hand then lifts. The
    data do not establish that. In 321 of the 549 new-layout episodes, no movable object moves more than 3 cm under a close
    command (`bound` is null); in 172 the role object moves, and in 56 another object does. The mechanism is not known.
  - A q10 onset lag of −72 steps means the event's first hold came 72 steps before the oracle's first hold, in episodes that have both.
  - These episodes count as not held for the read's h_o, because the instrument is the oracle.
  - Checked over all 222 L1 rollout files: 4,440 episodes, 0 episode errors.
- The event fires 1.24 segments per familiar episode, against the oracle's 1.00.

**Cost.**
- **GPU: 15,025 GPU-s = 4.17 A40-h** (sacct). That is 9 cancelled first builds at 223 s each (0.56 A40-h), plus 9 builds with
  their creations (3.62 A40-h).
- CPU:
  - 30 rollout jobs, 16-28 minutes each, about 10.5 CPU-job-h;
  - 3 S1 jobs, about 5 minutes each;
  - smoke, read and agreement jobs.

**Deviations.**
- Amendment 1 (`--rtclip`, §2) was made after the first build attempt and before any rollout or read.
- The builds were submitted before the pp_L1 ledger row was written. pp_L1 registers "before any rollout, S1 run or read",
  which holds. The code it hashes is the code the builds ran.
- The descriptive pre-design look at the demos' widths (§1.2) preceded the design note.

## 4. L2: options, and the ladder paused (2026-11-04)

L1 holds, so §10.10's order calls for L2 next: replace the oracle frames with object positions from the VLA's visual tokens.
§5 names the source as "S0 KV tools". **The ladder pauses here for the user's decision.** L2 needs large new infrastructure,
and its design forces plan-level choices. No L2 code was written and no GPU was spent on it.

**What exists.**
- **S0's tools** (`experiments/s0_data.py`, `s0_sandbox.py`). They set demo states, render the agentview and wrist images at
  224-256 px, and give per-token object coverage (`body_cov`, `site_cov`). That coverage comes from the simulator's
  segmentation and camera matrices (`get_camera_intrinsic_matrix` / `extrinsic`).
- **pi0.5's KV-level hooks** (pp_f1, pp_purify, pp_f3t). They read and edit the prefix KV per layer, on rendered frames.

**What L2 would need that does not exist.**
1. **Cameras in the rollout.** Every P rollout runs LIBERO's camera-free ControlEnv (fa1_common.make_env). L2 needs rendered
   frames at every replan, and pi0.5's prefix pass at each one: about 60 calls per episode on 4,440 episodes per rung. That
   moves rollouts from CPU to GPU-backed serving, or needs a cached-feature design.
2. **A token-to-position readout.** Some map from pi0.5 tokens (attention to the object word, or KV grouping) to a pixel set,
   then to a 3-D position.
3. **Lifting from pixels to 3-D.** This needs a camera model and depth. The sim's camera matrices and depth buffer are oracle,
   and §5.3's self-calibration (regress the gripper's pixel location from its end-effector position) is not built.
4. **The rest of the "frames" row** (PARTS_RESULTS §54): the object's yaw, which both terms read; the target sites' positions
   (cook region, rack top, cabinet top, which are sites, not bodies); the canonical place frame's bearing; and the rest height.
   §5 names only "object position".
5. **Training rows from rendered demo frames.** The terms would have to train on the readout's positions, not the simulator's,
   so the demo frames must be rendered and read (S0's renderer can do this).

**Plan-level choices L2 forces.**
- **(a) Supervision of the readout.** A probe from pi0.5 features to position, trained on oracle positions, is supervised
  frames under another name. An unsupervised readout (attention peak or token grouping, with no position labels) is the
  faithful version, but has no tool yet.
- **(b) Depth and calibration.** One option is sim calibration and depth: the camera stays oracle, and only the "which pixels
  are the object" step is removed. The other is §5.3's self-calibration, plus a table-plane or stereo assumption (agentview plus
  wrist) for depth.
- **(c) Scope of "frames".** Position only, with yaw and target sites kept oracle (a partial rung). Or every frame quantity
  (needs yaw from tokens and site localisation).
- **(d) Compute.** Closed-loop pi0.5 per replan makes every rung's rollouts GPU-bound. The rough estimate is 30-60 A40-h for one
  rung's 4,440 episodes at L0's 3 × 20 states. That exceeds the 15 A40-h cap of this track by itself. A cheaper path is the
  readout on a small vision encoder, but that is not "the VLA's visual tokens".

**Options, smallest first.**
1. **L2a "localisation only":**
   - pi0.5 attention or token grouping gives the object's and the target's pixel sets, unsupervised;
   - sim camera matrices and depth lift them to 3-D, with yaw and rest height kept oracle;
   - closed loop on a reduced state set.
   This needs the rollout renderer, the per-replan pi0.5 pass and the readout. It is a partial rung by construction.
2. **L2b, as L2a with self-calibrated cameras** (§5.3) and table-plane depth. It adds the calibration fit, and only the
   physical robot model stays given.
3. **L2c, the full frames row:** L2b plus yaw and site localisation from tokens. This is a new perception module, and closest to
   §5's intent.
4. **Skip to L3/L4 on oracle frames** (slots from pi0.5 features, roles from contact events). §10.10 fixes the order, so this
   changes it. L4 (roles from contact events) could reuse L1's event directly, and needs no camera.

**Open concerns for L1** (no criterion attached):
- The event holds without an oracle hold in 25% of new-layout episodes. The gate is familiar-only, and the read's instrument is
  the oracle, so this does not enter the verdict. A rung whose instrument also became the event would read h_o differently.
- Familiar 3F N3 now fails Layer-2's own criterion (+0.067, CI excludes 0), while staying inside the ladder's tolerance.
- K6 and N6 competence fell by 0.10-0.15. Both place into the bowl. This is an observation of 60 episodes per task, not a test.

## 5. L4 design note: roles induced from contact events instead of the given assignment (fixed 2026-11-04, before any code of this rung)

**Order.** PLAN_PARTS §10.12: L4 runs before L2 and L3, on top of L1, with oracle frames and slots. L0 stays P_min, and the gate
stays §10.10's.

### 5.1 Where the role assignment enters today

`FC.roles(instruction)` is a template parse plus a noun table (`NOUN`). It returns (object entity, target entity).
- **Training.** `pp_p_train.rows()` calls it per known task and sets `oi`, `ti` on every row. From those come:
  - which part each row trains (`opart`, `tpart`);
  - each term's frame, slot pose and size;
  - the onset quantities;
  - `part_radii`, `start_radii`, `role_radius`;
  - the m inventory's grouping by target.
- **Creation.** The same call, on the creation task's rows.
- **Rollout.** `fa1h_roll.episode` calls it on the instruction, and `oi`, `ti` select the parts and frames.
- **`task_entropy_bits`.** It uses it for the reported interface entropy.
- **Reads (instrument, stays).** `n2_tasks.json`'s `object` and `target` fields group the 3F factors and pick the arrival radius.
- **S1.** It sets `oi` and `ti` explicitly.

L4 replaces both things that call does: which role a word plays, and which slot a word refers to. Word-to-slot grounding is not
kept from the noun table. Slot identities and poses stay oracle (L3 is later).

### 5.2 The induction rule

**Fixed, label-free in its fit.** It reads the instructions, L1's event, and the oracle slot positions. It never reads `FC.roles`,
`NOUN` or `n2_tasks`' roles.

1. **Per demo, the sites.** Over a demo's event-held rows, from the first onset to the last held row:
   - grasp site g = the slot that moves farthest between those two rows (3-D);
   - release site r = the slot other than g nearest to g, in 3-D, at the last held row.

   A demo with no event gives nothing.
2. **Per task, one situation.** (g, r) = the most frequent pair over its demos, ties to the first seen. A task with no event gives
   no situation. Tokens are the lowercased, whitespace-split instruction.
3. **Referents, cross-situationally.** For token w, C(w) = the intersection, over the situations whose instruction contains w, of
   {g, r}. If C(w) has one slot, that is ref(w).
   **Explaining away:** for an unresolved w, remove from C(w) any slot that is ref(u) of another token u in every situation
   containing w. Passes are simultaneous, and repeat until nothing changes. A token left with no single slot has no referent.
4. **Role predictiveness.** For a token with a referent, P_g(w) = the share of its situations with g = ref(w), and P_r(w) likewise
   for r. These are the plan's "words that predict the grasp site" and "the release site".
5. **Parsing an instruction.**
   - object = ref(w) of the token with the largest P_g(w) > 0, ties to the earliest token;
   - target = ref(w) of the token with the largest P_r(w) > 0 among referents other than the object, ties to the latest token;
   - if either has no candidate, the instruction has **no induced roles**.

**Library and creation.** A build's lexicon comes from its own training tasks, and its situations go in the checkpoint's metadata.
A creation adds one situation from its k demos, and recomputes the lexicon, so the new atom's words are grounded by the same rule.

**How the rule's form was chosen (disclosed).** I chose steps 3-5 by reasoning by hand about these instructions, with their given
roles in mind. Plain word-to-site prediction fails on paper: "plate" appears only with grasps of the bowl, so it would predict the
bowl as the object on "put the cream cheese on the plate". The nobottle library needs step 4's ranking, because there "put", "the"
and "bowl" all co-occur with the bowl in every task. So the fit uses no labels, but **the design saw them, and agreement on these
16 instructions is weak evidence for the rule's generality**. They share few templates.

### 5.3 What cannot be induced

K8 ("push the plate to the front of the stove") has no close command, so L1's event never fires, and it gives no situation. Its
parse should have no object, since the plate is never a grasp site. FLAWED_COMPOSITION §5.3 lists this gap: contact without a
carry. N8 is the same.
- **Rule: a task with no induced roles trains nothing, and its rollout records a failure with no steps.** Keeping K8 on its given
  roles would leave an oracle input in the rung.
- K8, K7, N7 and N8 are outside every counted measure, as they were at L0.
- So the induced assignment cannot equal the given one on every task, and the "holds trivially" case does not apply if K8 parses
  as predicted.

**Correction (made after the smoke's `roles` run, before any build, rollout or read).** A task now trains iff it has a contact event
(a situation) **and** an induced parse. As first written, only the parse decided. The smoke showed why that is not enough: in the
nobottle library, "the" refers to the bowl, because every remaining event task involves the bowl. So K8 parses there as (bowl, cook
region), with no contact evidence at all, and its push rows would train the bowl's part. A task with no event gives the rule no
evidence, so it trains nothing. This does not change any parse.

### 5.4 Comparison with the given assignment (reported, no criterion)

`experiments/pp_L4_smoke.py roles` compares the induced roles with `FC.roles`, which only that script reads:
- **Per demo:** the share with (g, r) equal to the given pair, per task, and the release margin (second-nearest minus nearest
  slot distance) and grasp margin (largest minus second-largest displacement).
- **Per lexicon:** the parse of all 16 compounds, for the full library, and for nobottle and noplate before and after creation at
  k = 1, 5, 10; with demos 0 to k−1 as the creation uses them.
- **Robustness:**
  - one-demo lexicons: each task's situation from demo j alone, for each demo index j;
  - leave-one-task-out: each known task withheld from the induction, then parsed;
  - token shuffles: 20 random orders of each read instruction;
  - plain prediction without steps 3-4, as the comparator of §5.2.

### 5.5 S1 under induced roles

Φ and the terms are unchanged, and the parse sits upstream of them. S1 has two parts, and L4's S1 holds iff both hold:
- **The model's swap tests and target-into-grasp components**, exactly 0 with m fixed at each inventory value, as at L1, on the L4
  checkpoints.
- **Parse locality, new at this rung.** Over K1-K6 and N1-N6, with the full lexicon: any two instructions that share a given object
  parse to one object, and any two that share a given target parse to one target. So the word carrying one role does not change
  the other role's slot. Without this, a target word could change the object term by a route other than m. The check uses the given
  roles only to choose pairs.

### 5.6 What is rebuilt

- **Object term: retrained.** K8's rows leave the object term's training. They are its only rows for the plate part, and the role
  radius changes.
- **Target term: reused bitwise from L1 (`--reuse tgt:<cond>_L1`), if two conditions hold.**
  - K8 has no event-held rows, so its rows carry zero gate weight in the target term: they give it no gradient. They do affect
    which rows fill L1's batches.
  - The precondition is asserted per build: the induced roles equal L1's on every trained task, and `reuse_role`'s buffer equality
    holds (rtclip, per-part radii). If it fails, the target term is rebuilt too, and that is reported.
- **Creations.** Bottle from K4 and plate from K1, k = 1, 5, 10. Each asserts the old-parameter guarantee as before. Each also
  reports whether adding its situation changes the parse of any library task.

**A plain statement of what to expect.** If the parse equals the given roles on K1-K7 and on every read task, then L4 differs from L1
only by K8's object rows. The rung's read would then be mostly a test of removing those rows and of re-training noise, plus the
induction's own accuracy.

**Default off.** Without `--iroles`, the role source is `FC.roles`, and the rows and checkpoints are L1's. Smoke checks before any
build:
- (a) flag off, `rows()` equals the committed code's (3bb75b8c) bitwise, with and without `--pgate` rows;
- (b) flag off, one varied-outcome rollout file (K5, familiar, 0.48 at L0) re-run on both an L0 and an L1 checkpoint equals the
  committed file;
- (c) the comparison of §5.4.

Protocol and mapping follow in §6, before any L4 rollout, S1 run or read.

### 5.7 Smoke checks and the induction's comparison (2026-11-04; commits c3515936, 89f623d4; results/flawed/pp_L4_smoke_{rows,roles,roll}.json)

**Is the induced assignment equal to the given one on every task? No.** It is exact on 14 of the 16 compounds, including all 12 read
tasks, so the "holds trivially" case does not apply.

| compound | full lexicon's parse | given |
|---|---|---|
| K1-K7, N1-N7 | exact | same |
| K8 | none (no object) | plate, stove front |
| N8 | bowl, stove cook region | differs |

**Per demo.** On K1-K7, all 350 of 350 demos give (g, r) equal to the given pair. K8 has no event in any of its 50 demos.

| task | grasp margin, q10 / median (m) | release margin, q10 / median (m) |
|---|---|---|
| K1 | 0.125 / 0.145 | 0.151 / 0.168 |
| K2 | 0.222 / 0.248 | 0.113 / 0.142 |
| K3 | 0.311 / 0.328 | **0.007 / 0.011** |
| K4 | 0.309 / 0.328 | **0.006 / 0.009** |
| K5 | 0.278 / 0.287 | 0.118 / 0.153 |
| K6 | 0.131 / 0.151 | 0.044 / 0.064 |
| K7 | 0.226 / 0.262 | 0.093 / 0.119 |

The cabinet-top release is decided by under 1 cm in the median demo. Every demo still gets it right, but it is the thin point of the
induction.

**The builds' lexicons, with creation.**
- **Full:** as above.
- **nobottle:** exact on its own tasks K1, K2, K3, K6, K7, and on N1-N3. Before creation it maps "put", "the", "on" and "top" to the
  bowl, and it misreads the bottle tasks (K4, N4, N5, N7 as bowl; N6 as cheese). After creating from K4, at k = 1, 5 or 10, K4 and
  N4-N7 are exact. K5 has no parse, because creating the bottle does not ground "rack".
- **noplate:** exact on its own tasks. K1, N1 and N4 have no parse before creation, and are exact after it, at k = 1, 5 and 10.
- No creation changes the parse of any library task.

**Robustness.**
- **One-demo lexicons:** with each task's situation from demo j alone, all 12 read tasks are exact for all 50 values of j.
- **Leave one known task out** (read tasks exact, of 12):

| withheld | read tasks exact | its own parse | also wrong |
|---|---|---|---|
| K1 | 9 | none | N1, N4 |
| K2 | 9 | none | N2, N5 |
| K3 | 12 | exact | — |
| K4 | 7 | **wrong: bowl as object** | K5, N4, N5, N6 |
| K5 | 11 | none | — |
| K6 | 7 | none | N1, N2, N3, N6 |
| K7 | 12 | wrong on the target slot (cabinet top instead of drawer); K7 is not a read task | — |

  A word first seen in the withheld task cannot be grounded. A task whose two slots are both unexplained stays unresolved, so with
  K4 withheld, K5's "wine", "bottle" and "rack" get no referent. **The failures are not always refusals: withholding K4 gives a
  silent wrong parse.**
- **Token shuffles:** 20 of 20 exact on 11 read tasks, 11 of 20 on N6. So the parse mostly does not use word order. N6 does: "in" is
  grounded to the cheese (it occurs only in K6), and N6 parses correctly only because "wine" comes before "in".
- **The lexicon is not a linguist's.** "in", "of", "and", "open" and "inside" have referents. It parses these instructions; it has
  not learned English prepositions.
- **Plain prediction** (no referents, no explaining away): 6 of 12 read tasks, all six new pairings wrong. This confirms §5.2's
  reasoning on the data.

**Parse locality (§5.5):** holds with no violation.

**Flag off.**
- (a) `rows()` equals the code at 3bb75b8c bitwise, on full, nobottle, noplate and the K4 creation, with oracle held and with the
  event. An L1 checkpoint loads with no lexicon.
- (b) K5 familiar (seed 10), a varied-outcome task, on L0's checkpoint: equal to pp_cons_roll in all 20 episodes.
- (b) on L1's checkpoint, **first run: not equal**. Every episode differed by a small amount at first, 2 of 20 outcomes flipped, and
  the rate was 0.50 against 0.60.
  - Re-run on bmo, the committed file's host: equal in every field of all 20 episodes.
  - The first re-run was on walle. So **the code is unchanged, and rollouts are not bitwise reproducible across hosts**. The first
    episode's values differ at about 1e-8, and the difference grows.
  - L0's file (neo) and its re-run (walle) did agree, so this depends on the host pair, and I have not mapped which.

**What host dependence means for the reads.** L0's and each rung's rollouts ran on whatever overcap gave them. So a state-paired Δ
includes some floating-point divergence, and one rung-versus-rung cell can move by 2 of 20 episodes (0.10) from this alone, on this
one measured task and seed. The paired bootstrap does not model it. L1's read is not re-run, and the rule stays as registered. This
is one more reason to treat a Δ near the tolerance with care.

## 6. L4 protocol and verdict mapping (fixed 2026-11-04, before any L4 rollout, S1 run or read; m3_prereg tag pp_L4)

**Arm L4.** L1's flags plus `--iroles`: `--rank 32 --rtclip --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize
--pgate --iroles`.
- Builds: `pp_{full,nobottle,noplate}_L4_s{10,11,12}`. Object term retrained, target term reused bitwise from `<cond>_L1` (§5.6),
  with the precondition asserted.
- Creations: bottle from K4, plate from K1, k = 1, 5, 10.

**Data.** As in §2: dev states 4-23, seeds 10-12. `P2L4` (32 tasks) and `PL4_<cond>_k<k>` in results/flawed/pp_L4_roll. S1:
`pp_s1_mix_full_L4_m<v>.json` for every value v of the L4 full inventory, plus parse locality from pp_L4_smoke_roles.json.

**Read.** `experiments/pp_ladder_read.py --rung L4 ... --parse-locality pp_L4_smoke_roles.json` -> results/flawed/pp_ladder_L4_read.json.
Measures and estimators as in §2. The reference is L0, as §10.10 fixes. L1's values are shown beside them, with no criterion.

**Counted, familiar layout only:** S1 (§5.5, both parts), OM, 3F on N1, N2, N3, N4, N6, and S3 for the bottle and the plate.

**Verdict mapping (one rule, §2's, unchanged).** A counted measure breaks iff |Δ| > 0.10 and its CI excludes 0. **L4 holds iff S1 holds
and no counted measure breaks.** Otherwise L4 breaks, and the broken measures are named. New layout and competence are reported with
no verdict.

**Scope of a pass, stated before the read.** A pass says that induced roles matched the given ones on these tasks, and the policy did
not lose composition by dropping K8's rows. It does not show the rule generalises beyond these 16 instructions (§5.2, §5.7).

**Stop rules.** If L4 breaks, the ladder stops for the user's decision. If L4 holds, this track stops and reports. L5 and L2/L3 are
not started here. Cap: this track's 15 A40-h, with 10.83 left.

## 7. L4 results (2026-11-04; m3_prereg pp_L4; results/flawed/pp_ladder_L4_read.json, pp_s1_mix_full_L4_m{0,1,2}.json, pp_L4_smoke_rollagree.json; rollouts pp_L4_roll)

**Verdict (pp_L4 mapping): L4 HOLDS.** S1 holds, and no counted familiar measure breaks.

**The pass rests on the CI clause for one measure.** Familiar 3F N4 moves by +0.138, more than 0.10. It does not break only because its
CI, [−0.017, 0.311], contains 0. Read as a plain point-estimate "within 0.10", **N4 would break the rung**. The registered rule is the
one that decides, and it is unchanged, but this is the closest call on the ladder so far. The rule also does not show equivalence:
N4's and N6's CIs reach well past ±0.10 at both L1 and L4.

**Scoped claim.** In the familiar layout, given Layer-2 GO false in the new layout, with oracle frames and slots: roles induced from
contact events, on top of L1, do not move a counted measure past the registered rule. The parse matched the given roles on every read
task, so what this read tests is the removal of K8's rows from the object term and retraining, not a policy running on different
roles. §5.2's limit stands: the rule's form was designed with these instructions in view.

**S1.** With m fixed at 0, 1 and 2 (the L4 full inventory, K = 3), both swap tests and every target-into-grasp component are exactly 0
on seeds 10-12. Parse locality holds with no violation.

**Familiar layout (counted).** Δ = L4 − L0, 2,000 paired state draws, seeds pooled. L1's value is shown for context, with no criterion.

| measure | L0 | L1 | L4 | Δ [95% CI] | breaks? |
|---|---|---|---|---|---|
| OM | 1.089 | 1.151 | 1.166 | +0.077 [0.028, 0.136] | no |
| 3F N1 (D3) | −0.033 | +0.017 | +0.033 | +0.066 [0.000, 0.149] | no |
| 3F N2 | 0.000 | +0.033 | −0.033 | −0.033 [−0.100, 0.000] | no |
| 3F N3 | 0.000 | +0.067 | 0.000 | 0.000 [0.000, 0.000] | no |
| **3F N4** | −0.033 | −0.050 | +0.105 | **+0.138 [−0.017, 0.311]** | no (CI clause) |
| 3F N6 | +0.011 | −0.071 | −0.001 | −0.012 [−0.162, 0.129] | no |
| S3 bottle | 0.888 | 0.918 | 0.919 | +0.031 [−0.005, 0.070] | no |
| S3 plate | 0.916 | 0.904 | 0.944 | +0.028 [−0.028, 0.091] | no |
| 3F N5 (not counted: L0 fails it) | +0.133 | +0.100 | +0.208 | +0.075 [−0.017, 0.175] | — |

- **Under Layer-2's own criteria** (reported, not the gate): familiar 3F fails on N5 only, as at L0. All 6 familiar S3 cells pass.
- **Competence.** Familiar success is 0.87 on known pairings (L0 0.90) and 0.93 on new ones (L0 0.92).
- **K5 falls.** K5 familiar is 0.32, against 0.48 at both L0 and L1. Holds fall too: 35 of 60 at L4, 47 at L1, 42 at L0.
  - Every seed is lower than at L1 (success 6, 3, 10 of 20 against 12, 6, 11).
  - K5 enters N4's, N5's and N6's a_o, so this is a candidate source of N4's and N5's rise in D3. I did not test that.
  - The object term's only input change from L1 is K8's rows leaving. The role radius falls from 0.2251 to 0.1863 m as a result. Whether
    that radius, the missing rows or retraining noise lowers K5 is **not established**.

**New layout (reported, no verdict).**

| measure | L0 | L1 | L4 | Δ (L4 − L0) [95% CI] |
|---|---|---|---|---|
| OM | 1.081 | 1.178 | 1.165 | +0.083 [−0.132, 0.352] |
| 3F N1 | +0.129 | +0.177 | +0.026 | −0.104 [−0.239, 0.022] |
| 3F N2 | −0.063 | +0.001 | −0.044 | +0.019 [−0.089, 0.130] |
| 3F N3 | −0.067 | −0.027 | +0.040 | +0.106 [0.000, 0.233] |
| 3F N4 | +0.164 | +0.229 | +0.171 | +0.007 [−0.128, 0.145] |
| 3F N5 | +0.276 | +0.247 | +0.377 | +0.101 [−0.002, 0.212] |
| 3F N6 | +0.107 | +0.039 | +0.120 | +0.013 [−0.123, 0.158] |
| S3 bottle | 0.770 | 0.698 | 0.840 | +0.070 [−0.029, 0.158] |
| S3 plate | 1.076 | 1.011 | 1.231 | +0.155 [−0.161, 0.746] |

- Under Layer-2's criteria, new-layout 3F fails on N4 and N5. S3 passes 5 of 6 cells; L0 passes 2 and L1 passes 1.
- **Do not read that S3 count as a gain.** L1 and L4 share one target term and the same parse on these tasks, and differ only by the
  retrained object term, yet the count moves from 1 to 5. That shows how noisy these cells are, on their wide CIs.
- New-layout success: 0.43 known, 0.40 new (L0 0.46, 0.37). K5 stays at 0.

**Construction checks.**
- **Target term:** all 19 tensors equal L1's bitwise, on all 9 builds. The object term's 19 all moved.
- **Trained tasks:** K1-K7 in full, K1, K2, K3, K6, K7 in nobottle, K2-K7 in noplate. K8 trains in none.
- **Inventory:** K = 3 for full and noplate, K = 2 for nobottle, the same sizes as L1.
- **Creations:** no creation changed a library task's parse, and each kept the old-parameter guarantee.
- **Rollouts:** 222 files, 4,440 episodes. The only errors are K8's 120 "no induced roles" episodes on the full build, which §5.3 specifies.
  Every parse in the rollouts equals the given roles, except K8 (none) and N8's 120 episodes (bowl, cook region). Neither is counted.
- **Registered files:** all unchanged since pp_L4, except docs/PLAN_PARTS.md, whose §10 heading was reworded by another commit. No
  criterion changed.

**The event in the L4 rollouts** (descriptive; every file, S2 and S3 pooled).

| layout | episodes | step agreement | event on, oracle off (steps) | oracle on, event off (steps) | onset lag, median (q10, q90) | only the event holds | only the oracle |
|---|---|---|---|---|---|---|---|
| familiar | 2,220 | 0.965 | 7,266 | 2,584 | 0 (−4, +1) | 133 | 0 |
| new | 2,220 | 0.971 | 11,121 | 3,326 | −1 (−62, +1) | 520 | 0 |

**Cost.**
- **GPU: 12,404 GPU-s = 3.45 A40-h** (sacct over every attempt, including two jobs pre-empted and requeued, three attempts each).
- **Track total: 27,429 GPU-s = 7.62 A40-h** of the 15.
- CPU: 30 rollout jobs of 15-27 minutes, about 10 CPU-job-h; 3 S1 jobs of about 5 minutes; smoke, check, read and audit jobs.

**Deviations.**
- §5.3's training rule was corrected after the smoke and before any build (a task trains iff it has a contact event and a parse).
- The builds were submitted in the same step as the pp_L4 prereg job, so some started before its row existed. pp_L4 is worded "before any
  L4 rollout, S1 run or read", and that holds: the row was written at 09:36, and rollouts started after 11:00.
- The session restarted twice during this rung. Nothing finished was re-run; the builds' logs and sacct showed them complete.
- pp_L4_smoke's `roll` check records "not equal" for L1's first re-run. That is host dependence, shown in §5.7.
- L1's first report gave a commit id that does not exist ("30f9e0d1"); the commit is a108aa3a. This doc never carried it.

**Open concerns.**
- Familiar 3F N4 at +0.138 (above). If the user reads the gate as a point-estimate tolerance, L4 breaks on N4.
- K5's familiar fall (0.48 to 0.32) and hold fall, unexplained.
- K8, a task that L1's event cannot see, is simply lost at this rung, and N8 runs on a wrong target with no warning.
- Host-dependent rollouts (§5.7) add unmodelled noise to every Δ. L0, L1 and L4 ran on different mixes of hosts.
- The cabinet-top release margin is under 1 cm (§5.7), and a withheld task can produce a silent wrong parse.

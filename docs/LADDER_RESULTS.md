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

The protocol and the verdict mapping of the L1 read follow in §2. They are fixed and pre-registered before any read of this
rung.

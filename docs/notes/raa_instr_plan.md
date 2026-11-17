# RAA place-step instrumentation: plan and predictions (EXPLORATORY diagnostic, no prereg row)

Written 2026-11-17 before any instrumented rollout. Follows `results/flawed/raa_diag.json` (commit 0944a6ef), which used only
existing records. Question: why does a RAA release that ends near the channel target not register success, and does the
flip-back of the role channel to "object" cause the retry loop or follow a failed placement? Nothing here is a registered test;
the numbers will be reported as exploratory.

## What is logged (every control step, both policies)
`experiments/raa_instr.py` wraps the LIBERO env constructor (a pure observer, read-only calls on `sim.data`, no extra render,
no change to any action) and then hands control to `raa.py client` or `rg.py client` unchanged. After every env step it records:
step index, the executed action (7), eef xyz and quaternion, gripper qpos (2), the instructed object's body xyz and quaternion
(wxyz), the target's xyz (body or site), and the success predicate recomputed from its terms plus the simulator's own
`_check_success()` and the step's `done`.

LIBERO `On(obj, tgt)` is `tgt.check_ontop(obj)`:
* body target (plate, bowl; N1, N4, N6): `z_tgt_body <= z_obj_body` AND contact(tgt, obj) AND body-origin xy distance < 0.03 m.
* site target (K5, wine_rack top_region): `site.under(site_pos, site_mat, obj_pos)` (obj point inside the site box in x/y,
  z in [h - 0.005, h + 0.10]) AND contact(rack, obj).
Sub-terms logged separately, plus xy distance and height above target, so a near miss is measured as a margin to each term.
Structural fact (not a prediction): the episode ends at the first step where the full predicate is true, so "flip-back while
the predicate is true" cannot occur in a failed episode. The informative quantity is the margin of each sub-term at and after
a release, and the state at the moment of the re-pick or the flip-back.

RAA role per query comes from the existing server `--chan-diag` jsonl (role target/object per query t, merged by (episode, t)).
rgp release events come from rg.py's own `_rg.json` plus the per-step log.

## Rollouts
Tasks N1 (tid 16), N4 (22), N6 (26), K5 (8), familiar layout. 20 new states each: branches k540-k559 (trials 24-43) of
`n2_branches_rx.json`, states `n2_states_rx`. Policies: RAA (`pi05_g8v_raa_ae_tl6000`, `raa.py serve`) and rgp (`rg.py client`
against `pp_serve_purify.py` frozen pi0.5), same servers, seeds and flags as the section 73 rollouts, new job scripts only.
Action identity: all 20 episodes per task and policy are compared (steps, success) with the existing unlogged record of the
same trial in `results/flawed/raa/raa_rx_<tid>.json` and `results/flawed/raa_rgp_rx/`. If any diverge, the divergence rate is
reported instead of claiming identity (LIBERO/GPU episodes are not exactly reproducible from the init state alone).

## Analysis (`experiments/raa_instr_read.py` -> `results/flawed/raa_instr.json`)
Release = commanded gripper goes from closed (>0) to open (<=0) while the object follows the eef (within 0.10 m and lifted
1 cm above its start). Per task x policy: at release object-to-target xy, height above target, tilt (angle of the body z axis
from world z, minus its tilt at step 0), eef height and speed; then the state after settling (release + 20 steps or the next
close command): TIPPED (tilt change > 30 deg), OFF (upright, xy >= 0.03), SEATED-FAIL (xy < 0.03 but z/contact/under term false),
and the term that failed. Re-pick onset = next close command after a release, and the flip target -> object in the role log;
state at those moments, and fraction where the object was NEAR-VALID (upright, xy <= 0.04, height within 0.03 of the valid z).

## Predictions (before data)
* P1 identity: at least 90% of the logged episodes equal the unlogged record on (steps, success); divergence, if any, is
  nondeterminism and not the logger.
* P2 N1, N4 (plate): most RAA near-releases (xy <= 5 cm) end UPRIGHT and OFF or SEATED-FAIL on the 3 cm xy rule, not tipped
  (tipped < 20%). The gap is release precision against a 3 cm tolerance (outcome ii).
* P3 N6 (bowl): the channel target error exceeds 3 cm in a majority of RAA failures, and the release xy is off by about that
  error (outcome iii).
* P4 K5 (rack): failures split between `under` false (wrong height or x/y) and tipped bottles; no latch signature.
* P5 latch: fewer than 15% of RAA re-picks start from a NEAR-VALID resting object; most start from OFF or TIPPED. So (i) is not
  the main cause, and the flip-back mostly follows a failed placement.
* P6 rgp releases are lower in eef speed and xy offset at release (median xy < 2 cm vs RAA > 3 cm) and release once.

Decision rule for the report: (i) if >= 30% of RAA failed episodes re-pick or hover from a NEAR-VALID resting object;
(ii) if >= 50% of failed releases end tipped or OFF/SEATED-FAIL with rgp releasing tighter; (iii) if the N6 target estimate
error at release exceeds 3 cm in >= 50% of its failures. More than one can hold; counts decide.

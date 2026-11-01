# PLAN_PARTS results

Protocols and results of the adopted plan docs/PLAN_PARTS.md. Each section is pre-registered before its first read
(ledger m3_prereg, scripts under experiments/prereg/). Numbers come from results/flawed/pp_*.json.

## 1. Layer 1 protocol: F1, F2, F3 (fixed 2026-11-01, before any F1 read; m3_prereg tag pp_f123)

Models: pi0.5 (pi05_libero, PyTorch) and GR00T N1.5 (ar0s/groot_libero), the N2 checkpoints. References: prototype A's arms Aq
(seeds 24-26, the clean reference), Hg and Rg (Hb / Rb on gen_g, seeds 0-2, the known-flawed references). SmolVLA and X-VLA are
not run in this pass.

**Frames** (`experiments/pp_f1_frames.py` -> results/flawed/pp_f1_frames.npz; no policy run, no demo executed).
- Grasp frames: N2's main start states 4-13 of each layout, rendered as N2's client sees them after the 10 settle steps. Every
  N2 task of a layout shares them.
- Place frames: for each movable object, the first 10 demos of the lowest-index known task that carries it (bowl K1, bottle K4,
  cheese K6), at the recorded state 5 kept steps after the held flag turns on. The object has just been lifted, before the
  carry heads to the demo's own target. Recorded states are set and rendered, never executed.
- Each frame carries the oracle scene and the projection of every oracle entity point into the 224 model image (checked
  visually on pp_f1_frames_check.png).

**Tasks and pairs.** N2's 12 pick-and-place compounds K1-K6, N1-N6.
- Grasp phase (both layouts): base = the task's instruction. The CROSS partner is the target-changed task (same object). The
  OWN partner is the object-changed task (same target).
- Place phase (familiar layout, the place frames of the task's object): the CROSS partner is the object-changed task. The OWN
  partner is the target-changed task.
- Partner rule: among the 12, the same atom in the kept role and another atom in the changed role; known before new, then
  compound order.
- K5 has no OWN partner in the grasp phase and no CROSS partner in the place phase (the rack takes only the bottle), so it drops
  out of both. That leaves 22 grasp cells (11 compounds x 2 layouts) and 11 place cells.

**Components** (`experiments/pp_f1_mix.py`). The CROSS partner's activations are patched into the base run at one component:
- W: the changed role's instruction words (the token span where the two instructions differ), at the expert-read interface.
  For pi0.5 this is every layer's K and V at those positions; for GR00T, the backbone features the DiT cross-attends.
- V: the changed role's agentview image tokens (Chebyshev radius 1 on the model's patch grid around the entity's projected
  point), at the expert-read interface.
- Rest: every other expert-read position. Descriptive; it is not a role message.
- KV_l (pi0.5, l = 0-17): layer l's K and V at every position. The role span is included when the two spans have equal
  length; otherwise it keeps the base values at that layer.
- Q_l (GR00T, l = 0-11): the changed role's words in the residual stream entering Eagle layer l, propagated through the later
  layers. Equal-length spans only. GR00T's expert reads one feature sequence, not per-layer KV, so this is its per-layer VLM
  family (a deviation from "each VLM layer's KV", forced by the architecture).
- E_j: the action expert's layer-j output, patched at every denoising step from the partner run on the same x_t (pi0.5's 18
  Gemma expert layers, GR00T's 16 DiT blocks).
- Oracle arms (`experiments/pp_f1_oracle.py`). The instruction is the role assignment. Role-derived inputs (held flag, offset,
  zrest, cyaw, rot0) are recomputed by the rollout's rules for each run on the same state. The components are each MLP's
  input and its four hidden layers (H, R: one net; A: the object and target terms). The gate and the frames are the fixed
  combination rule, not components.

**Measure.**
- d = MSE between the patched and the base sample over the first min(10, H) steps x 7 real dims, in the model's normalised
  action rows. It is the mean over 2 noise seeds shared by every variant of a frame.
- Per cell u and component k: m_k(u) = Σ d_k / Σ d_full_own over u's frames. This is the cross-role change relative to the
  own-role swap's change.
- Oracle arms average d over their 3 checkpoints first.
- Pooled m_k is the mean over cells. CIs are 2,000-draw percentile bootstraps (seed 0) over compounds.
- Two directions: tg (target into the grasp chunk) and op (object into the place chunk).

**F1 verdict (per VLA).**
- floor_dir = the largest CI upper bound of A's pooled m over A's components, per direction.
- **F1 holds** iff some component of the VLA, in some direction, has a CI lower bound above floor_dir.
- Location: k*_dir = argmax of the pooled point m over the role-message components {W, V, KV_l} (pi0.5) or {W, V, Q_l}
  (GR00T). Expert layers carry no role message to purify; their profile is reported.

**F2 verdict (per VLA; N2's recorded main rollouts, states 4-23).**
- ρ_g = Spearman over the 22 grasp cells of m_{k*_tg}(u) with 1 − g(u).
- ρ_p = Spearman over the 11 place cells of m_{k*_op}(u) with 1 − p(u). Here p = successes / grasps; a cell with no grasp drops
  out.
- Bootstrap over compounds; degenerate draws are skipped and counted.
- **F2 ties** iff (ρ_g > 0 and its CI excludes 0) or (ρ_p > 0 and its CI excludes 0).
- Read by `experiments/pp_f1_read.py` -> results/flawed/pp_f1_read.json.

**F3 (protocol fixed here; run after the F1 read, with the located k*_op).**
- **Purification.** Purifying component k of a new-pairing run replaces both role messages within k's scope with messages from
  known tasks:
  - A = the known task sharing the object (cheese K6, bottle K4, bowl K1);
  - B = the known task sharing the target (plate K1, stove K2, cabinet K3, bowl K6);
  - object positions take A's values, target positions B's values (positions aligned by the common tokens);
  - W: the object-word and target-word spans at every read layer. V: the object and target image tokens.
  - KV_l: layer l takes B's values at every position (realigned), except the object words, which take A's.
  - Q_l: the object and target words entering Eagle layer l, propagated.
- **Open loop (descriptive only).** On the place frames of N1-N6, compare intact and purified chunks: the cosine between the
  chunk's summed translation and the direction from the end effector to the target. On the grasp frames, the same cosine
  toward the object.
- **Closed loop.**
  - N1-N6, familiar layout (primary) and new layout (secondary), states 4-13: 10 episodes per task and layout.
  - Two arms on the same branch seeds as N2: an intact re-run and the purified run. A new server mode; the client and the
    episode protocol are N2's.
  - Δp = pooled p (Σ successes / Σ grasps over the 6 tasks) of purified minus intact.
  - CI: 2,000-draw bootstrap over tasks and states jointly (seed 0).
  - **F3 ties** iff Δp > 0 with its CI excluding 0.
  - Reported: Δs, Δg, the new layout, and the intact re-run against N2's recorded episodes for the same states.
  - If k*_op is V, closed-loop purification needs the entity pixel positions online. The read then stops and reports instead
    of substituting another component.

**Go / kill (§3 of the plan).**
- **GO to Layer 2** iff, for at least one VLA, F1 holds and F2 or F3 ties.
- **KILL K1** iff no VLA meets that.

**Known weaknesses, stated before the read.**
- A's clean status is by construction only for tg: in the grasp phase A's gate is 0 and its object term never reads the
  target. For op, A's target term locates the held object through the role binding (its foot is the role object's position).
  Under an object-changed instruction it therefore reads another body, and A's op floor is not expected to be 0. The plan's
  "A should be 0" holds for tg only.
- Unequal role spans (cabinet targets, bowl objects) enter W and Rest as insertions. For pi0.5 the keys after the span keep
  their RoPE phases while their slots shift by the length difference (at most 2 tokens).
- Power is low: 11 compounds, 22 and 11 cells. The place frames are familiar-layout demo states only.
- No numeric "noise floor" beyond A's profile is defined by the plan. The floor used here is A's largest upper bound.

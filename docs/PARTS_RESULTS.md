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

## 2. F1 and F2 results (2026-11-01; results/flawed/pp_f1_read.json, pp_f1_pi05.json, pp_f1_groot.json, pp_f1_oracle.json)

Runs: frames (job 4451223), pi0.5 (4451323, 18 min), GR00T (4451324, 7 min), oracle arms (4451325, CPU). 22 grasp cells and 11
place cells per model. CIs are 95% compound bootstraps.

**References** (pooled m: the cross-role patch's change over the own-role swap's change).

| arm | tg, every component | op, object term | op, target term |
|---|---|---|---|
| A (Aq) | 0.000 exactly | 0.000 exactly | 62.1 [19.0, 136.1] |
| H (Hg) | 0.77 [0.44, 1.16] | — | 2.18 [0.63, 4.73] (one net) |
| R (Rg) | 0.24 [0.14, 0.40] | — | 0.82 [0.49, 1.23] (one net) |

- A is exactly 0 for target-into-grasp, by construction.
- A's object-into-place value is large, as stated before the read. A's target term finds the held object through the role
  binding, so an object-changed instruction makes it track another body.
- A's floor is therefore 0.000 for tg and 136.1 for op.
- H and R mix in both directions. In a single fused net every component equals the full swap.

**pi0.5** (selected components; pooled m [95% CI]).

| component | tg | op |
|---|---|---|
| full cross-role swap | 0.53 [0.27, 0.83] | 1.88 [0.48, 3.55] |
| W (target / object words, every layer) | 0.043 [0.008, 0.080] | 0.051 [0.011, 0.101] |
| V (the role's image tokens) | 0.000 [0.000, 0.000] | 0.004 [0.000, 0.012] |
| Rest (every other position) | 0.46 [0.24, 0.70] | 1.66 [0.46, 3.12] |
| KV_14 | **0.29 [0.12, 0.48]** | **0.53 [0.13, 1.15]** |
| KV_16 | 0.013 [0.006, 0.021] | 0.15 [0.03, 0.36] |
| every other KV_l | ≤ 0.007 | ≤ 0.015 |
| expert E_0-E_13 | ≤ 0.041 | ≤ 0.17 |
| expert E_14 / E_17 | 0.42 / 0.53 | 1.10 / 1.88 |

- The expert reads the other role almost entirely from one VLM layer's KV (layer 14), and from positions other than the role
  words and the role's image tokens.
- Inside the expert, the cross-role change enters at layer 14 and grows to the full swap by layer 17.

**GR00T.**

| component | tg | op |
|---|---|---|
| full cross-role swap | 0.23 [0.16, 0.30] | 1.98 [0.70, 3.33] |
| W | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| V | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] |
| Rest | 0.22 [0.15, 0.29] | 1.90 [0.67, 3.22] |
| Q_0-Q_11 (equal-length spans only) | 0.17 [0.08, 0.26] at every l (10 cells) | 1.76-1.94 (2 cells) |
| DiT E_14-E_15 | 0.23 [0.16, 0.30] | 1.98 [0.70, 3.33] |

- In GR00T's backbone features, the role words' and role image tokens' own positions carry none of the cross-role change.
  All of it sits at other positions.
- Patching the words entering any Eagle layer carries about three quarters of it (tg). The same mixing appears at every depth.

**Verdicts against pp_f123.**

| | pi0.5 | GR00T |
|---|---|---|
| F1 | **holds** (tg; e.g. KV_14 lower bound 0.12 > A's 0.000) | **holds** (tg; e.g. Rest lower bound 0.15 > 0.000) |
| located k*_tg / k*_op | KV_14 / KV_14 | Q_3 / Q_6 |
| F2 ρ_g (m vs 1 − g) | −0.08 [−0.39, 0.21] (22 cells) | −0.02 [−0.61, 0.46] (10 cells) |
| F2 ρ_p (m vs 1 − p) | **0.59 [0.01, 0.93]** (11 cells, 1 degenerate draw) | 1.00 [1.00, 1.00] (**2 cells**, 967 degenerate draws) |
| F2 mechanical verdict | ties | ties |

**Defects of the read, reported rather than repaired.**
- **GR00T's F2 "ties" is not interpretable.**
  - The location rule took the argmax over components measured on different cell sets.
  - Q_l exists only where the two role spans have equal length, which leaves 10 grasp cells and 2 place cells.
  - k*_op = Q_6 therefore rests on 2 cells, where any Spearman is ±1.
  - The pre-registered mapping returns "ties". I do not count it as evidence for K1.
- **The F1 criterion is trivial for tg.** A's tg floor is exactly 0, so any nonzero component passes, including the expert's
  first layers at 1e-4 scale. The substantive tg evidence is the size of the large components (pi0.5 KV_14 0.29, GR00T
  Rest 0.22) beside A's exact 0, H's 0.77 and R's 0.24.
- **pi0.5's ρ_p is fragile.**
  - Its lower bound is 0.01.
  - The known pairings all have p-failure 0 (ties) and mostly low KV_14 mixing (K1-K4 ≤ 0.03, K6 0.94). The new pairings
    carry both the failure and most of the mixing (N6 3.22, N1 0.61, N2 0.50).
  - So ρ_p largely says that new pairings both mix more and place worse. It is a known-versus-new contrast, as D1b was.
- **The target-into-grasp mixing does not predict grasp loss** in either model (ρ_g ≈ 0). This matches N2: the grasp loss is
  carried by the layout, not by the compound.

**Go/kill status after F1/F2.**
- pi0.5 meets "F1 holds and F2 ties", so the pre-registered mapping already returns GO.
- F3, the causal test, still runs as registered.

**Cost.** About 0.45 A40-h: frames 0.08, pi0.5 0.30, GR00T 0.12. Smokes are excluded.

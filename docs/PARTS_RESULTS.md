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

## 3. F3 results (2026-11-01; m3_prereg pp_f123 + pp_f3; results/flawed/pp_f3_read.json, pp_f3_open_pi05.json, pp_f3_open_groot.json)

F3 purified the components F1 located, k*_op: pi0.5 KV_14, GR00T Q_6.
- Closed loop: N1-N6, states 4-13, 10 episodes per task and layout, with an intact re-run and the purified run on N2's branch
  seeds.
- The intact re-run reproduced N2's recorded episodes exactly: s and g agree on every episode in both models and layouts.
- No episode errors. Jobs 4451480 (pi0.5) and 4451481 (GR00T).

| model | layout | intact s / g / p | purified s / g / p | Δp [95% CI] | Δs | Δg |
|---|---|---|---|---|---|---|
| pi0.5 (KV_14) | familiar (primary) | 0.48 / 0.95 / 0.51 | 0.08 / 0.18 / 0.45 | **−0.05 [−0.83, 0.48]** | −0.40 [−0.78, −0.03] | −0.77 [−0.97, −0.53] |
| pi0.5 (KV_14) | new | 0.32 / 0.63 / 0.50 | 0.17 / 0.32 / 0.53 | +0.03 [−0.31, 0.40] | −0.15 [−0.35, 0.00] | −0.32 [−0.57, −0.07] |
| GR00T (Q_6) | familiar (primary) | 0.20 / 0.98 / 0.20 | 0.27 / 1.00 / 0.27 | **+0.06 [−0.07, 0.25]** | +0.07 [−0.07, 0.25] | +0.02 [0.00, 0.10] |
| GR00T (Q_6) | new | 0.08 / 0.37 / 0.23 | 0.05 / 0.45 / 0.11 | −0.12 [−0.35, 0.11] | −0.03 [−0.15, 0.07] | +0.08 [−0.10, 0.27] |

**Verdict: F3 does not tie in either model.**

- **pi0.5.**
  - Purifying layer 14's KV destroys the grasp: g falls from 0.95 to 0.18, and N4-N6 (the bottle) lose every grasp.
  - Layer 14 carries the object role as well as the target role. The registered purification gives that layer B's values
    everywhere except the object words, and B is a known task with another object. So it also hands the grasp B's object.
  - The located component is a mixed carrier of both roles, not a target-only message that can be swapped for a pure one.
- **GR00T.** Purifying Q_6 changes little: the familiar p moves from 0.20 to 0.27, with a CI that includes 0. GR00T places the
  bottle on the plate, stove or in the bowl 0 of 30 times either way.
- **Open loop (descriptive).**
  - The cosine between the chunk's translation and the direction to the target, on place frames: pi0.5 0.10 intact against
    0.16 purified; GR00T 0.05 against 0.06.
  - Toward the object on grasp frames: pi0.5 0.33 against 0.29 (familiar) and 0.52 against 0.48 (new); GR00T unchanged.

**Layer 1 go/kill (pp_f123 mapping): GO, through pi0.5 alone, and fragile.**
- pi0.5: F1 holds and F2 ties (ρ_p 0.59 [0.01, 0.93]); F3 does not tie.
- GR00T: F1 holds; F2's "ties" rests on 2 cells (§2) and is not interpretable; F3 does not tie.
- Without that one correlation, whose lower bound is 0.01 and which mostly separates known from new pairings, the mapping would
  return KILL.
- The causal test, which the plan treats as the architectural isolation, failed in both models. K1's "causes" part is not
  supported.

**Deviations.**
- The pi0.5 F3 job was pre-empted on overcap once and requeued. Task files already written were kept. The task in progress
  re-ran from its first episode on the same seeds.
- Mid-run, I read client log lines that print episode outcomes (the criteria were registered).

## 4. Layer 2 protocol: P-build, S1-S4 (fixed 2026-11-01, before any Layer-2 rollout; m3_prereg tag pp_l2)

Layer 2 starts because pp_f123's mapping returned GO on pi0.5 (F1 holds, F2 ties). F3's closed loop was still running when this
protocol was fixed, so this protocol does not depend on F3's outcome.

**Architecture P** (`lifelong_ood/policy/parts_flow.py`).
- **Base.** Stage 1i's arm Ai: the object term in the object's frame and the canonical target term with the rotation since
  onset. Every atom-identity input is removed from both backbones.
- **Parts.** Each atom of a role has a part: a 16-dim slot embedding (a backbone input) and a rank-8 LoRA on the backbone's
  four hidden Linear layers.
- **Sparsity.** A task activates its object's part and its target's part, nothing else.
- **Combination.** Φ = (1 − held)·T_o⁻¹v_o + held·T_c⁻¹v_t, fixed.
- **Not carried over:** occupancy, free space, clearance strata, grasp-option teachers, the object-unit end-effector input
  (1l-1p). The oracle slots, frames and contact gate stay.
- **Build** (`experiments/pp_p_train.py build`).
  - Demos only (demos.npz + demos_b.npz, fitted, never executed). No generated carries; the generated-data ablation arm is not
    run in this pass.
  - Prototype A's recipe (30,000 steps, AdamW 3e-4, batch 512, EMA 0.999), seeds 0-2.
  - Libraries: full (all 8 known tasks), nobottle (without K4, K5) and noplate (without K1).
  - Secondary arm (§9 of the plan): rank-32 parts, nobottle_r32 and noplate_r32.
- **Creation** (`pp_p_train.py create`).
  - The held-out atom's part is re-initialised and trained alone on the first k demos of one task: bottle from K4, plate from
    K1. k ∈ {1, 5, 10, 50}, the build's seed.
  - 4,000 steps, AdamW 3e-4, no decay, batch 256, no EMA.
  - Every other parameter gets a zero gradient. It is asserted bitwise unchanged after creation.
- **Choices this pass makes that the plan leaves open.**
  - The held-out atoms: bottle, because its removal leaves every other object atom trained; plate, the target of a single known
    task.
  - The creation tasks K4 and K1: the lowest-index known task of each atom.
  - The creation recipe.
  - Removing K4, K5 also removes the rack (K5's only target), so the rack is not read in the bottle condition.

**S1 (locality)** (`experiments/pp_s1_mix.py` on the full library, seeds 0-2).
- Swap tests on every F1 frame:
  - the object term's output under a target-atom change;
  - the target term's output when the object's part is replaced by another atom's.
- F1's mixing profile on P (pp_f1_oracle's frames, pairs and d).
- **S1 holds** iff both swap tests are exactly 0 for every seed and every target-into-grasp component is exactly 0 on every frame.
- Object-into-place is reported. P keeps A's foot route, where the role binding locates the held body.

**S2 (multiplicativity on held-out pairings).** Rollouts of the full library (seeds 0-2) on all 32 N2 task-cells, states 4-23,
prototype A's episode (`experiments/pp_p_roll.py` -> results/flawed/pp_s2_roll, name P2). Per layout:
- 3F: every N1-N6 CI of D3 contains 0 (fa1g_read.three_factor, unchanged).
- OM: the object-matched ratio r = mean over N1-N6 of s / mean over N1-N6 of s(the known tasks of the same object; cheese K6,
  bottle K4 and K5). It holds iff r ≥ 0.8 with a state-bootstrap CI lower bound ≥ 0.6.
- **S2 holds** iff 3F and OM hold in both layouts. Pooled success is reported as competence.

**S3 (creation).**
- The created part (bottle: base nobottle; plate: base noplate) is read with the training partner (K4 / K1) and every unseen
  partner (bottle: N4, N5, N6; plate: N1, N4), both layouts, 3 seeds × 20 states (results/flawed/pp_s3_roll).
- Transfer ratio = s_unseen / s_train, with a state-bootstrap CI.
- Old tasks change by exactly 0 (bitwise parameters). Parameters added, demos and GPU-seconds to ready are reported from the
  creation files.
- **Layer-3 GO** (plan §4.3) iff S1 holds, S2 holds, and for both atoms, both layouts and every k ∈ {1, 5, 10}, the ratio is
  ≥ 0.8 with CI lower bound ≥ 0.6.
- The plan's "kill or revise K3 if created parts transfer only to their training partner" has no numeric form in the plan. It is
  judged descriptively from the ratios, and no extra criterion is added.

**S4 (baselines on S3's protocol)** (`experiments/pp_baselines.py`). Same base tasks, same k demos of the same task, same
adaptation recipe:
- HF: holistic (Hb) fine-tuned.
- HL: holistic with a rank-8 per-task LoRA, switched on for every task containing the new atom.
- RF: role-routed (Rb) fine-tuned.
- LIB: a DMPEL/CLARE-style library. Rank-8 LoRA experts on the holistic backbone are mixed by a learned router; one expert per
  base task is trained jointly; for the new atom one expert is added, and it is trained together with the router.
- Reported for each: s_train, s_unseen, the ratio, and P − baseline on unseen partners (state-bootstrap CI).
- Forgetting on the old pick-and-place tasks (familiar layout; rollouts results/flawed/pp_s4_old) for HF, RF, LIB. HL and P
  are exactly 0 by construction.
- S4 carries no verdict.

**Read:** `experiments/pp_l2_read.py` -> results/flawed/pp_l2_read.json.

**Statistics.** Development reads only: 3 seeds × 20 states, 95% CIs, no Holm correction. The S-conf fresh-seed read needs
the user's approval and is not run.

## 5. Layer 2 results (2026-11-01; m3_prereg pp_l2; results/flawed/pp_l2_read.json, pp_s1_mix_full.json, pp_train_*.json, pp_create_*.json, pp_bl_*.json; rollouts pp_s2_roll, pp_s3_roll, pp_s4_old)

**Verdict (pp_l2 mapping): Layer-3 GO = false.**
- S1 holds.
- **S2 fails:** 3F fails on N4 in the familiar layout. The object-matched ratio holds in both layouts.
- **S3's transfer criterion fails** in 9 of 12 cells (atom × layout × k ≤ 10).
- No rollout had an episode error.

**P-build.**
- Size: 1,098,136 parameters per library. The role backbones have 436,536 (object) and 434,072 (target) parameters. A part has
  20,720 (object) or 20,664 (target) parameters, so a part adds about 2.4% of a backbone.
- The rank-32 secondary arm has 82,832 and 82,608 parameters per part.
- A build takes 30,000 steps in about 14 min on an A40.
- Every creation left every other parameter bitwise unchanged (asserted; 48 creations). Parts take 38-68 GPU-s to ready.
- Demo rows at k = 1, 5, 10, 50: 92, 537, 1,066, 5,389 for the bottle (K4) and 90, 444, 904, 4,662 for the plate (K1).

**S1 (holds).**
- Both swap tests are exactly 0.0 for all three seeds.
- Every target-into-grasp component is exactly 0 on every frame.
- Object-into-place: the object term's components are 0. The target term's components reach m = 111.5 through the foot route
  (the role binding locates the held body), as in A (§2).

**S2 (fails).** Full library, 3 seeds × 20 states, all 32 cells.

| layout | s known | s new | object-matched ratio [95% CI] | 3F |
|---|---|---|---|---|
| familiar | 0.80 | 0.72 | 1.05 [1.00, 1.11] (holds) | fails: N4 D3 −0.34 [−0.48, −0.22] (obs 0.30, pred 0.64) |
| new | 0.21 | 0.22 | 0.99 [0.76, 1.42] (holds) | holds (every N1-N6 CI contains 0) |

- Familiar per pairing: N1 0.92, N2 0.97, N3 0.92, N4 0.30, N5 0.67, N6 0.57. Known pairings: K1 0.98, K2 1.00, K3 0.98,
  K4 0.67, K5 0.28, K6 0.90.
- N4 (bottle on plate) is the only pairing far below its factor prediction. The bottle's hold factor (0.75) and arrival (0.87)
  and the plate's place-on-arrival (0.98) predict 0.64.
- Competence: without the generated carries, demos alone do not carry the new layout (s known 0.21, against 0.68 for 1p's Aq
  on generated data). Within that low competence, composition holds: OM 0.99, 3F holds.

**S3 (creation; transfer ratio = s_unseen / s_train, 3 seeds × 20 states, state-bootstrap 95% CI).**

| atom | k | familiar: s_train / s_unseen / ratio | new: s_train / s_unseen / ratio |
|---|---|---|---|
| bottle (from K4) | 1 | 0.98 / 0.78 / 0.80 [0.74, 0.85] | 0.53 / 0.42 / 0.78 [0.69, 0.90] |
| bottle | 5 | 0.80 / 0.64 / 0.80 [0.75, 0.86] | 0.20 / 0.13 / 0.64 [0.45, 0.85] |
| bottle | 10 | 0.88 / 0.68 / 0.77 [0.71, 0.83] | 0.32 / 0.25 / 0.79 [0.65, 1.00] |
| bottle | 50 | 0.92 / 0.71 / 0.78 [0.71, 0.85] | 0.27 / 0.17 / 0.62 [0.48, 0.86] |
| plate (from K1) | 1 | 0.95 / 0.78 / **0.82 [0.73, 0.91]** | 0.20 / 0.23 / 1.17 [0.55, 3.50] |
| plate | 5 | 0.93 / 0.72 / 0.78 [0.70, 0.87] | 0.13 / 0.23 / **1.75 [0.82, 5.17]** |
| plate | 10 | 0.93 / 0.72 / 0.77 [0.69, 0.85] | 0.12 / 0.23 / **1.93 [0.95, 5.67]** |
| plate | 50 | 0.93 / 0.68 / 0.73 [0.65, 0.82] | 0.22 / 0.23 / 1.04 [0.50, 2.70] |

- Bold marks the cells that meet the criterion (ratio ≥ 0.8, lower bound ≥ 0.6). 3 of the 12 k ≤ 10 cells meet it; the
  bottle meets it in none.
- **Every shortfall comes from one unseen pairing, bottle on plate (N4).**
  - Bottle part (familiar, k = 1 / 5 / 10 / 50): N4 0.55 / 0.48 / 0.48 / 0.45, N5 0.98 / 0.80 / 0.88 / 0.95, N6 0.82 / 0.63 /
    0.68 / 0.73.
  - Plate part (familiar): N1 0.97 / 0.98 / 0.98 / 0.97, N4 0.58 / 0.47 / 0.45 / 0.40.
  - N5 and N1 run at the training partner's rate. A created part transfers to most partners at its own rate. The plan's
    "transfers only to its training partner" (kill K3) does not describe this.
- New layout: the training partner itself mostly fails (s_train 0.12-0.53), so the ratios are wide.
- One demo already gives the familiar rate. k changes little; bottle s_train is 0.98 at k = 1 and 0.80-0.92 above it.
- Rank 32 (secondary arm, descriptive): bottle familiar ratios 0.84, 0.83, 0.83, 0.80 (k = 1-50), with lower bounds
  0.75-0.78. Plate familiar 0.88, 0.81, 0.79, 0.81, with lower bounds 0.67-0.79. New layout 0.69-1.90, wide. N4 stays at
  0.43-0.57.
- Old tasks: exactly 0 change (bitwise).

**S4 (baselines; same base tasks, same k demos, same adaptation recipe). Descriptive.**

| method | bottle familiar: s_train (k = 1 / 5 / 10 / 50) | bottle s_unseen | plate familiar: s_train | plate s_unseen | new layout | forgetting on old tasks (familiar) |
|---|---|---|---|---|---|---|
| **P** | 0.98 / 0.80 / 0.88 / 0.92 | 0.78 / 0.64 / 0.68 / 0.71 | 0.95 / 0.93 / 0.93 / 0.93 | 0.78 / 0.72 / 0.72 / 0.68 | s_unseen 0.13-0.42 | exactly 0 |
| HF (holistic fine-tune) | 0.72 / 0.80 / 0.83 / 0.87 | 0.00-0.01 | 0.57 / 0.82 / 0.70 / 0.75 | 0.00 | 0.00 everywhere | 0.82 → 0.00 (bottle), 0.67 → 0.00 (plate), every k |
| HL (per-task LoRA) | 0.48 / 0.77 / 0.68 / 0.68 | 0.00-0.01 | 0.33 / 0.62 / 0.55 / 0.57 | 0.00-0.02 | 0.00 | exactly 0 (off on old tasks) |
| RF (role-routed fine-tune) | 0.95 / 0.92 / 0.88 / 0.98 | 0.00-0.05 | 0.00 / 0.90 / 0.92 / 0.95 | 0.00 | 0.00-0.07 | 0.85 → 0.00, 0.76 → 0.00, every k |
| LIB (DMPEL/CLARE-style router) | 0.65 / 0.60 / 0.67 / 0.75 | 0.00-0.04 | 0.45 / 0.62 / 0.65 / 0.78 | 0.02-0.09 | 0.00 | 0.83 → 0.00; 0.74 → 0.01-0.04 |

- P − each baseline on unseen partners, familiar: +0.59 to +0.78 in every cell (lower bounds 0.52-0.72). New layout: +0.13 to
  +0.42 (lower bounds 0.04-0.30).
- Every baseline learns the training pairing from the k demos. None reaches the unseen pairings: at most 0.09 (LIB, plate,
  k = 50).
- The fine-tuned baselines (HF, RF) and the learned-router library (LIB) lose the old tasks completely at every k. Their 4,000
  steps on one task's demos overwrite the shared weights, or, for LIB, the router.
- The adaptation recipe was fixed before the read and is the same for every method. A replay or smaller-step recipe might
  forget less. That was not tested.
- Trainable parameters per adaptation: P 20,720 / 20,664; HL 31,016; HF 967,224; RF 888,376.
- LIB trains 499,280 parameters: all expert slots are in its optimiser, but only the new expert's rows and the router's 3,024
  parameters receive gradient.
- GPU-s to ready: P 38-68, HF 30-34, HL 38-43, RF 16-17, LIB 37-41.

**Reading.**
- On the N2 scene, with oracle slots, frames and the contact gate, a part created from one demo of one task works with
  partners it never saw, at its own rate, except on one pairing. Old tasks stay exactly unchanged.
- No baseline, the learned-router library included, composes the new atom at all.
- The pre-registered bars are not met. Bottle on plate (N4) runs at 0.30-0.58 wherever it appears (S2 and both S3 atoms). That
  pulls the pooled ratios to 0.73-0.80 in the familiar layout.
- Is N4 a part-competence limit (a tall bottle stood on a low plate under LIBERO's strict On) or a combination failure?
  - S2's 3F says combination: N4's observed 0.30 is below its own factor prediction 0.64.
  - S3's plate part puts the plate's place-on-arrival with the trained bottle part, and gives 0.40-0.58 in N4 too.
  - I report it, and I do not patch it (rule 10).
- Demos alone do not cover the new layout: the competence there is 0.1-0.5. The plan's claim that layout generalization comes
  from frame equivariance does not hold at that level of competence.

**Deviations.**
- `experiments/pp_s1_mix.py` was hashed in pp_l2 and then edited once. Its first run crashed (a tensor needing grad), and
  `torch.no_grad()` was added to its pre-hook sampler. No estimator or criterion changed.
- Smoke artifacts from code checks, never read for outcomes, stay untracked: results/flawed/pp_*smoke*, pp_smoke_roll,
  pp_s1_tmp_full.
- The generated-data ablation arm of P (§4.1 of the plan) was not run.

**Cost.** GPU: builds 5.0 A40-h, creations about 1.9 A40-h. CPU: 246 rollout jobs, 74.6 CPU-job-h, together with S1 and the
oracle F1 run.

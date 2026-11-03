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

## 6. Layer 2 confirmatory protocol: rank-32 parts, fresh seeds (fixed 2026-11-01, before its S1 read and any of its rollouts; m3_prereg tag pp_l2r32)

**This is a post-hoc re-registration.** Rank 32 was chosen after seeing pp_l2's results: the rank-8 primary arm and the
rank-32 secondary arm, both on the development data (seeds 0-2, N2 states 4-23). The user approved this one fresh-seed
confirmatory read (the plan's S-conf, §6 of the plan). The approval covers this read only.

**Frozen before any read.**
- Architecture: P (§4) with rank-32 parts. 82,832 (object) and 82,608 (target) parameters per part.
- Recipes: pp_l2's build recipe (30,000 steps, AdamW 3e-4, batch 512, EMA 0.999) and creation recipe (4,000 steps, AdamW 3e-4,
  batch 256, no EMA, every other parameter asserted bitwise unchanged).
- Held-out atoms: bottle (object, created from K4) and plate (target, created from K1). k ∈ {1, 5, 10, 50}.
- Libraries: full_r32 (S1, S2), nobottle_r32 and noplate_r32 (S3).
- Criteria: pp_l2's S1, S2 and S3 criteria and its Layer-3 GO mapping, unchanged (§4).
- S4: HF, HL, RF and LIB exactly as in pp_l2 (rank 8), descriptive.

**Fresh data.**
- Seeds 10, 11 and 12 for every build and creation. No earlier run used them.
- Start states: 50 per layout from `experiments/pp_conf_states.py` (results/flawed/pp_conf_states/). N2's generator rules are
  unchanged; only the seed bases are new: familiar 9100000 + k, new layout 9200000 + k, episode noise 23000000 + 1000 k
  (N2: 600000, 700000, 13000000). No earlier read used these states.
- 3 seeds × 50 states per cell (the plan's confirmatory size).
- Rollouts: `pp_p_roll.py --phase conf` into results/flawed/pp_s2c_roll (S2), pp_s3c_roll (S3, S4) and pp_s4c_old (forgetting).
- Read: `experiments/pp_l2r32_read.py` -> results/flawed/pp_l2r32_read.json. It runs pp_l2_read's estimators. Its `--check`
  reproduces pp_l2_read.json from the development data, and it was run before registration.

**Statistics.** 95% CIs, as in pp_l2. The plan asks for Holm over at most 6 confirmatory hypotheses. Layer-3 GO is an
intersection-union test: every cell must pass. Such a test needs no multiplicity adjustment, so none is applied.

**Reported whichever way it falls.** If N4 (bottle on plate) stays low, it is reported as competence and not patched.

## 7. Targeted F3 protocol: the target-identity subspace of pi0.5 KV_14 (fixed 2026-11-01, before the subspace fit and any read; m3_prereg tag pp_f3t)

**This is a post-hoc follow-up.** pp_f3 replaced pi0.5's layer-14 KV wholesale with B's, and the grasp collapsed (§3). This
arm replaces only a target-identity subspace and keeps the orthogonal complement.

**Subspace estimator** (`experiments/pp_f3t.py fit`; known tasks only, no rollout, no outcome).
- Contrasts: pairs of known tasks that share the object and differ in the target: bowl K1-K2, K1-K3, K2-K3; bottle K4-K5.
- Frames: the 20 grasp frames of both layouts and the object's own 10 place frames (pp_f1_frames).
- At every aligned valid position, take the difference between the two runs' layer-14 keys and values. Aligned positions are
  every image token, plus every instruction token outside the span where the two instructions differ.
- Keys are compared before RoPE: each key is de-rotated at its own position.
- U_K and U_V are the top 16 right singular vectors of the stacked key and value contrasts. The rank, 16, is fixed here.
- Control: a random 16-dimensional subspace of each (QR of a Gaussian).
- Diagnostic: the share of contrast energy each subspace captures, both in-sample and on odd frames when fit on even frames.

**Targeted purification** of a new-pairing run, at layer 14 only:
- u' = u + P_U(u_S − u) and v' = v + P_V(v_S − v), at aligned positions.
- The key is re-rotated at the base run's own position.
- Arms:
  - **T** (primary): target subspace, source S = B, the known task that shares the target.
  - **R** (control): random subspace, source B.
  - **TA** (open loop only, descriptive): target subspace, source A, the known task that shares the object.
- Why B is the source: every known task that shares the object has a different target. Its target-subspace component would
  install the wrong target. The literal "source shares the object" reading is kept as arm TA, in open loop.

**Reads.**
- Open loop first (`pp_f3t.py open` -> results/flawed/pp_f3t_open.json): pp_f3's cosines and seeds, for every arm.
- Then closed loop: N1-N6, familiar (primary) and new, N2 states 4-13, N2's seeds, 10 episodes per task and layout, arms T
  and R (`experiments/pp_f3t_roll.sh`).
- Intact is pp_f3's intact re-run, which matched N2's recorded episodes exactly.
- Read: `experiments/pp_f3t_read.py` -> results/flawed/pp_f3t_read.json.

**Verdict** (familiar layout; 2,000-draw paired bootstrap over tasks and states):
- place_up: Δp_T > 0 with its CI excluding 0.
- grasp_kept: Δg_T ≥ −0.10 (point estimate). The tolerance is fixed here.
- specific: Δp_T − Δp_R > 0 with its CI excluding 0.
- **F3t ties** iff all three hold. Partial outcomes are labelled: "raises place, not specific", "raises place, costs grasp",
  "does not raise place".

**Scope.** GR00T's Q_3 / Q_6 are not run in this pass.

## 8. Targeted F3 results (2026-11-01; m3_prereg pp_f3t; results/flawed/pp_f3t_read.json, pp_f3t_open.json, pp_f3t_subspace.json; rollouts pp_f3t_roll)

**Verdict (pp_f3t mapping): does not raise place.** place_up, grasp_kept and specific all fail. Targeted purification of
KV_14 does not tie the located component to the place failure.

**Subspace.**
- 62,220 contrast rows. The top-16 key subspace captures 0.59 of the key contrast energy (0.58 on held-out odd frames); a random
  16-dim subspace captures 0.06.
- Values: 0.58 (held out 0.57), random 0.07.
- The spectrum is spread out: the first direction carries 0.17 (keys) and 0.13 (values).

**Open loop** (pooled cosine over N1-N6).

| frames | intact | T | R | TA |
|---|---|---|---|---|
| place, toward the target | 0.10 | 0.15 | 0.10 | −0.01 |
| grasp familiar, toward the object | 0.33 | 0.31 | 0.32 | 0.35 |
| grasp new, toward the object | 0.52 | 0.50 | 0.52 | 0.51 |

- T moves the place chunk toward the target about as much as pp_f3's wholesale purification did (0.10 to 0.16).
- TA (target subspace from the object-sharing task) points the place chunk away from the target: it installs A's target.

**Closed loop** (pi0.5, N1-N6, states 4-13, 10 episodes per task and layout; intact = pp_f3's re-run). Jobs 4454484 (T,
pre-empted and requeued twice; finished task files were kept) and 4454485 (R). No episode errors.

| layout | intact s / g / p | T s / g / p | R s / g / p | Δp_T [95% CI] | Δg_T | Δp_T − Δp_R |
|---|---|---|---|---|---|---|
| familiar (primary) | 0.48 / 0.95 / 0.51 | 0.17 / 0.45 / 0.37 | 0.45 / 0.93 / 0.48 | −0.14 [−0.60, 0.25] | **−0.50 [−0.82, −0.20]** | −0.11 [−0.64, 0.28] |
| new | 0.32 / 0.63 / 0.50 | 0.20 / 0.43 / 0.46 | 0.32 / 0.60 / 0.53 | −0.04 [−0.35, 0.24] | −0.20 [−0.42, −0.02] | −0.07 [−0.39, 0.24] |

- The random control changes almost nothing: Δp_R −0.03 [−0.16, 0.12], Δg_R −0.02 [−0.10, 0.00].
- Per task, familiar (s / g, intact → T):

  | task | intact | T |
  |---|---|---|
  | N1 cheese on plate | 0.1 / 1.0 | 0.4 / 1.0 |
  | N2 cheese on stove | 0.1 / 1.0 | 0.3 / 0.9 |
  | N3 cheese on cabinet | 0.4 / 0.7 | 0.1 / 0.1 |
  | N4 bottle on plate | 0.9 / 1.0 | 0.0 / 0.0 |
  | N5 bottle on stove | 0.9 / 1.0 | 0.2 / 0.2 |
  | N6 bottle in bowl | 0.5 / 1.0 | 0.0 / 0.5 |

**Reading.**
- Sixteen of layer 14's 256 key and 256 value dimensions, chosen to carry the target's identity, are enough to break the
  grasp of the bottle. The located component does not separate into a target part and a grasp part along these directions.
- On the two cheese pairings whose grasp survives (N1, N2), place success rises from 0.1 to 0.3-0.4. That is a per-task
  observation of 10 episodes each, not a pre-registered claim.
- With pp_f3 (§3), neither wholesale nor targeted purification of the located component raises place success. K1's "causes"
  part stays unsupported for pi0.5.

**Deviations.**
- `experiments/pp_serve_purify.py` (hashed in pp_f3) gained the targeted branch after F3 was read. The pp_f3 path is unchanged.
- GR00T was not run.

## 9. Layer 2 confirmatory results, rank 32 (2026-11-01; m3_prereg pp_l2r32; results/flawed/pp_l2r32_read.json, pp_s1_mix_full_r32c.json; rollouts pp_s2c_roll, pp_s3c_roll, pp_s4c_old)

**Verdict (pp_l2r32 mapping, post-hoc re-registration): Layer-3 GO = false.**
- S1 holds.
- **S2 fails in both layouts** (3F). The object-matched ratio holds in both.
- **S3 meets its criterion in 3 of 12 cells**: plate familiar k = 1, 5, 10. It fails for the bottle in every cell, and for the
  plate in the new layout, on the lower bound.
- The fresh-seed read reproduces the development read's pattern. No rollout had an episode error, and no cell has a missing
  episode.

**Build.**
- 1,780,192 parameters per rank-32 library; parts of 82,832 (object) and 82,608 (target) parameters.
- 24 creations, each 35-61 GPU-s to ready. Every one left every other parameter bitwise unchanged.

**S1.**
- Both swap tests are exactly 0 for seeds 10-12.
- Every target-into-grasp component is exactly 0.
- Object-into-place reaches m = 119.8 through the foot route, as in A and pp_l2.

**S2** (full_r32, 3 seeds × 50 fresh states).

| layout | s known | s new | object-matched ratio [95% CI] | 3F: pairings whose D3 CI excludes 0 |
|---|---|---|---|---|
| familiar | 0.79 | 0.76 | 1.08 [1.03, 1.13] (holds) | N4 −0.42 [−0.51, −0.33] (obs 0.28, pred 0.70); N3 −0.03 [−0.07, −0.01]; N5 +0.02 [0.00, 0.03] |
| new | 0.28 | 0.30 | 1.01 [0.89, 1.15] (holds) | N4 +0.06 [0.00, 0.12]; N5 +0.13 [0.08, 0.19]; N6 +0.08 [0.02, 0.14] |

- Familiar per pairing: N1 0.91, N2 0.99, N3 0.95, N4 0.28, N5 0.73, N6 0.67. Known: K1 0.96, K2 0.98, K3 0.91, K4 0.70,
  K5 0.24, K6 0.93.
- With 50 states, the N3 and N5 deviations are small (≤ 0.03) but their CIs exclude 0.
- In the new layout, the bottle pairings succeed more often than their factor prediction. Competence there is low (0.1-0.5).

**S3** (transfer ratio = s_unseen / s_train; 3 seeds × 50 states; state-bootstrap 95% CI).

| atom | k | familiar: s_train / s_unseen / ratio | new: s_train / s_unseen / ratio |
|---|---|---|---|
| bottle (from K4) | 1 | 0.99 / 0.76 / 0.77 [0.74, 0.80] | 0.31 / 0.22 / 0.70 [0.61, 0.79] |
| bottle | 5 | 0.84 / 0.66 / 0.79 [0.75, 0.82] | 0.17 / 0.12 / 0.71 [0.59, 0.91] |
| bottle | 10 | 0.85 / 0.66 / 0.77 [0.74, 0.81] | 0.26 / 0.20 / 0.79 [0.67, 0.93] |
| bottle | 50 | 0.91 / 0.69 / 0.76 [0.73, 0.80] | 0.33 / 0.25 / 0.76 [0.67, 0.90] |
| plate (from K1) | 1 | 0.95 / 0.82 / **0.86 [0.80, 0.94]** | 0.34 / 0.28 / 0.83 [0.60, 1.20] |
| plate | 5 | 0.93 / 0.79 / **0.84 [0.78, 0.91]** | 0.24 / 0.22 / 0.90 [0.57, 1.43] |
| plate | 10 | 0.96 / 0.79 / **0.82 [0.76, 0.88]** | 0.27 / 0.23 / 0.83 [0.53, 1.31] |
| plate | 50 | 0.95 / 0.74 / 0.78 [0.72, 0.83] | 0.34 / 0.25 / 0.72 [0.49, 1.10] |

- Bold marks cells that meet the criterion. The plate's new-layout k = 1 cell misses on its lower bound (0.595 < 0.6).
- **The N4 competence gap remains, reported as competence and not patched.**
  - Bottle part, familiar (k = 1 / 5 / 10 / 50): N4 0.37 / 0.37 / 0.41 / 0.40, N5 0.99 / 0.85 / 0.85 / 0.90, N6 0.93 /
    0.76 / 0.72 / 0.77.
  - Plate part, familiar: N1 0.98 / 0.99 / 1.00 / 0.98, N4 0.65 / 0.59 / 0.57 / 0.50.
  - Without N4, both created parts run at their training partner's rate.
- Old tasks: exactly 0 change (bitwise).

**S4** (baselines as in pp_l2, seeds 10-12, same fresh states). Descriptive.
- Training pairing, familiar (k = 1 / 5 / 10 / 50):
  - bottle: HF 0.78 / 0.81 / 0.89 / 0.91; HL 0.63 / 0.72 / 0.73 / 0.68; RF 0.95 / 0.91 / 0.95 / 0.97; LIB 0.62 / 0.67 / 0.81 /
    0.79;
  - plate: HF 0.40 / 0.89 / 0.86 / 0.85; HL 0.33 / 0.77 / 0.76 / 0.73; RF 0.04 / 0.91 / 0.93 / 0.97; LIB 0.35 / 0.65 / 0.67 /
    0.71.
- Unseen pairings: at most 0.06 in the familiar layout (HF bottle k = 1; LIB plate k = 50) and at most 0.01 in the new layout.
- P − baseline on unseen pairings: familiar +0.64 to +0.82 (lower bounds 0.56-0.77); new +0.12 to +0.28 (lower bounds
  0.07-0.20).
- Forgetting on old tasks:
  - HF: 0.83 → 0.00 (bottle base), 0.68 → 0.00 (plate base);
  - RF: 0.89 → 0.00-0.03, 0.74 → 0.00-0.01;
  - LIB: 0.86 → 0.00, 0.72 → 0.03-0.06;
  - HL and P: exactly 0.

**Reading.**
- On fresh seeds and fresh states, rank-32 parts confirm the development picture. The sparse library composes a created atom
  with unseen partners at close to the training partner's rate, except on bottle on plate. No baseline composes it at all.
- The pre-registered bars are not met. Raising the part rank from 8 to 32 did not close the gap: N4 stays at 0.28 (S2) and
  0.37-0.65 (S3).
- 3F now fails in the new layout too. The bottle pairings there run above the factor prediction. The mapping counts that as a
  failure, whatever its sign.

**Deviations.**
- None in the protocol. `pp_p_roll.py` gained `--phase conf` before registration. The default path is unchanged.

**Cost (§6-§9).**
- GPU, by sacct over every segment including pre-empted ones: 6.4 GPU-h, mostly A40.
  - rank-32 builds 1.3; baseline builds 1.6; creations 1.5;
  - targeted F3 1.6 (T 0.83 over three segments, R 0.58, fit, open loop and smokes 0.14);
  - fresh states 0.51 over three segments (any GPU, for EGL).
- CPU: 223 jobs (222 rollouts and S1), 155.7 CPU-job-h.

## 10. Round 1 design note and protocol: the N4 diagnostic (fixed 2026-11-01, before any of its reads; m3_prereg tag pp_n4d)

**Weakness.** Bottle on plate (N4) runs at 0.28-0.41 in every P configuration (S2, both S3 atoms, ranks 8 and 32). Every other
pairing runs near its factor prediction.

**Question.** Is N4 hard in itself (competence) or only when composed (combination)?

**No demo of N4 exists.** LIBERO-Goal has no bottle-on-plate task, and no holistic checkpoint saw N4. So the arm "trained
directly on N4" needs N4 carries from somewhere. They come from stage 1g's scripted oracle-state planner (`fa1g_plan.py`). The
keypoints are composed from the demos by the relocation rule, with the object rigid in the hand:
- grasp offset, grasp rotation, carry height and release rotation from the bottle's task K4;
- release offset = K4's grasp offset + (K1's release offset − K1's grasp offset). That is, the bottle's hand offset plus
  the plate task's object-to-target offset at release.

This planner is a diagnostic data source only. It is not part of any architecture.

**Arms** (familiar layout, dev states 4-23, N2's seeds; `experiments/pp_n4d.py`).
- **(a) single-task fit.** P's class (PartsFlow, rank 32), trained only on 50 generated N4 carries. The recipe and budget are
  P's (30,000 steps, AdamW 3e-4, batch 512, EMA 0.999). Seeds 0-2.
- **(b) P.** The rank-32 full library pp_full_r32_s10-12, on dev states. These seeds were trained for §6 and never read on dev
  states.
- **Controls.** The same single-task fit on stage 1g's generated K4 and K1 carries, read on K4 and K1. They separate a
  single-task or generated-data advantage from a combination effect.
- **(o) oracle planner.** The composed-keypoint planner itself on N4, and its own keypoints on K4 and K1, from the same dev states.
  Descriptive: it measures how hard the task is physically.

**Measures.**
- Success, plus a per-phase breakdown:
  - hold: the object held;
  - arrival: held within the pair's calibrated radius of the target (a1_calib);
  - place-on-arrival: success given arrival.
- Release-site error: the object's final xy relative to the plate, against the demos' release sites. The reference is the
  median object-minus-plate xy at the first open command after the grasp, over K1's 50 demos (scored from recorded demo
  states only). The final object height above the plate is also reported.
- CIs: 2,000-draw percentile bootstrap over states, seeds pooled.

**Mapping** (pre-registered; Δ = s_a(N4) − s_P(N4)).
- **competence** iff Δ ≤ 0.10.
- **combination** iff Δ > 0.10 and its CI lower bound > 0. The failing phase is the phase where P's rate falls furthest
  below (a)'s.
- Otherwise (Δ > 0.10 with lower bound ≤ 0): **unresolved**.
- Flag, reported alongside and not changing the verdict: if a control's single-task fit beats P on its own task by more than
  0.10 with lower bound > 0, a "combination" verdict may reflect the data source.

**What follows.**
- On combination: the next round targets the failing interface (for example, what the place part conditions on at the handoff).
- On competence: N4 becomes a recorded competence caveat, and the next round moves to new-layout competence.

## 11. Round 1 results: the N4 diagnostic (2026-11-02; m3_prereg pp_n4d; results/flawed/pp_n4d_read.json, pp_n4d_gen.json; rollouts pp_n4d_roll)

**The mechanical verdict is "competence" (Δ = −0.18 [−0.30, −0.07]), but the instrument failed its controls, so the verdict
is not interpretable.** The single-task fit (a) also fails on the two control tasks, where P succeeds.

| arm | task | s [95% CI] | hold | arrival | place-on-arrival | release xy error (m) |
|---|---|---|---|---|---|---|
| (a) single-task fit | N4 | 0.03 [0.00, 0.08] | 0.03 | 1.00 | 1.00 | 0.253 |
| (a) control | K4 | 0.00 [0.00, 0.00] | 0.02 | 0.00 | 0.00 | — |
| (a) control | K1 | 0.00 [0.00, 0.00] | 0.13 | 0.00 | 0.00 | 0.145 |
| (b) P, rank 32 | N4 | 0.22 [0.10, 0.35] | 0.70 | 0.90 | **0.34** | **0.093** |
| (b) P, rank 32 | K4 | 0.63 [0.50, 0.77] | 0.70 | 0.90 | 0.95 | — |
| (b) P, rank 32 | K1 | 0.97 [0.90, 1.00] | 0.97 | 1.00 | 1.00 | 0.011 |
| (o) oracle planner | N4 | 0.65 [0.45, 0.85] | 0.75 | 1.00 | 0.87 | 0.027 |
| (o) oracle planner | K4 | 0.75 [0.55, 0.90] | 0.75 | 1.00 | 1.00 | — |
| (o) oracle planner | K1 | 0.95 [0.85, 1.00] | 1.00 | 1.00 | 0.95 | 0.017 |

- **The single-task fits do not grasp.** Each was trained on 50 generated carries of one task. Their losses converged (0.14-0.16
  at 30,000 steps, against P's 0.12), yet they hold the object in 0-13% of episodes, on the known controls as well. Fifty
  planner carries of one task, without the demos, do not train this class into a working policy.
- So (a) is not a valid "trained directly on N4" reference. The mapping's "competence" rests on an arm that fails everywhere, as
  GR00T's F2 "ties" did (§2). I do not count it.
- **The descriptive evidence points to the place handoff.**
  - The composed-keypoint planner succeeds on N4 0.65 of the time, and on 0.87 of arrivals. N4 is physically feasible.
  - P reaches the plate with the bottle as often as it reaches the cabinet (hold 0.70, arrival 0.90). It then places on only
    0.34 of arrivals, against 0.95 for the bottle on the cabinet and 1.00 for the bowl on the plate.
  - P's final bottle position is 0.093 m from the demos' release site (the bowl in K1: 0.011 m; the planner on N4: 0.027 m).
    The final height above the plate is 0.019 m against the planner's 0.005 m. Both fit a bottle released and tipped over,
    though the episode records no tilt to confirm it.
- The generator kept 50 of 258 N4 attempts (stage 1g kept 50 of 142 for K4 and 50 of 85 for K1).

**Decision.** The mapping's branch rests on an invalid arm. I take the measured weakness instead: P's place-on-arrival on N4
(0.34, against the planner's 0.87). Round 2 targets the place interface. This choice is mine, not the pre-registered mapping's,
and is reported as such.

**Deviation.** On the login node I ran `grep` on two results files (results/flawed/a1g_gen_K4.json, a1g_gen_K1.json) to read
their kept and attempt counts.

**Cost.** 9 single-task fits, 1.7 A40-h. CPU: generation 3.7 h, plus 15 rollout and oracle jobs.

## 12. Round 2 design note and protocol: the held object's geometry at the place interface (fixed 2026-11-02, before any code of this round; m3_prereg tag pp_r2)

**Weakness (measured, §11).** On N4, P's place-on-arrival is 0.34, against 0.95-1.00 on its known placements and 0.87 for
the planner. The released bottle ends 0.093 m from the demo release site.

**Interface diagnosis.**
- Φ hands off at the contact gate. After the handoff, the target term v_t alone moves the held object. It reads the foot's
  position in the target's frame, the last command, the target's size and opening, and the rotation since onset.
- It does not read anything about the held object. Releasing a 0.28 m-tall bottle on a plate needs a different descent and
  release than a 0.05 m bowl: the place outcome depends jointly on the held object's geometry and the target.
- The plate's part saw only the bowl. So the place behaviour it learned is the bowl's, and nothing in its input says that
  the bottle differs.
- Under the composition theory this is a missing interface condition (B, the boundary condition the place part needs from
  the object role). The place term must condition on the held object's physical extent, not on its identity.

**Change (one, structural).**
- The target term's input gains the held object's hand-set size: footprint radius and height, from the same FC.SIZE table
  the object term already reads for its own slot.
- No identity, part or teacher is added, and the combination Φ is unchanged.
- The backbone is shared across targets, and the cabinet's known tasks place both the bowl and the bottle. So the
  size-to-release relation is learnable across atoms and can reach the plate's part.
- Everything else is P at rank 32, the registered arm of §6: recipes, atoms, k.

**Arm.** P_sz (rank 32) at seeds 10-12. Reference: P (rank 32) at seeds 10-12 on the same dev states.

**Reads (dev states 4-23, N2's seeds; pp_l2's estimators and its frozen mapping).**
- S1: the swap tests and the profile, on full_sz.
- S2: full_sz on all 32 cells.
- S3: nobottle_sz and noplate_sz, k ∈ {1, 5, 10, 50}.
- Reference P (rank 32), seeds 10-12, on dev states: S2 on all 32 cells. S3's dev reference is pp_l2's rank-32 secondary arm
  (seeds 0-2).

**Pre-registered round verdict.**
- **improves** iff N4's S2 familiar success with P_sz minus P's is > 0.10, with its state-bootstrap CI lower bound > 0.
- The Layer-2 GO for P_sz uses pp_l2's mapping, unchanged.
- If GO holds on dev, the single confirmatory read follows on fresh seeds (20-22) and fresh states.

## 13. Round 2 results (2026-11-02; m3_prereg pp_r2; results/flawed/pp_r2_read.json; rollouts pp_r2_roll)

**Round verdict: does not improve.** N4's familiar success is 0.30 with P_sz against 0.22 with P: +0.08 [−0.05, 0.22].
**Layer-2 GO = false**: S1 holds, S2 fails, and S3 meets its criterion in 5 of 12 cells.

| | P (rank 32), seeds 10-12 | P_sz (rank 32), seeds 10-12 |
|---|---|---|
| S2 familiar: s known / s new | 0.77 / 0.71 | 0.77 / 0.71 |
| S2 familiar: OM | 1.06 [1.01, 1.11] | 1.02 [0.97, 1.06] |
| S2 familiar: N4 D3 | −0.43 [−0.55, −0.32] | −0.31 [−0.45, −0.18] |
| S2 new: s known / s new | 0.16 / 0.21 | 0.14 / 0.21 |
| S2 new: 3F | fails (N5 +0.04 [0.01, 0.08]) | holds |

- P_sz's familiar 3F also fails on N1, −0.10 [−0.20, −0.02].
- **S3 (P_sz)** meets its criterion in these cells:
  - bottle familiar k = 1, 5, 10: 0.82 [0.76, 0.88], 0.81 [0.75, 0.86], 0.85 [0.80, 0.90];
  - bottle new k = 1: 0.81 [0.72, 0.93];
  - plate familiar k = 1: 0.85 [0.75, 0.96].
- It fails elsewhere: plate familiar k = 5, 10 (0.72, 0.74); the bottle new layout at k = 5, 10 (0.67, 0.59); the plate new
  layout on its lower bounds (0.39-0.53).
- In S3, the bottle part puts N4 at 0.40-0.62 (rank 32 without the size input: 0.37-0.41, §9). N4 stays the lowest pairing
  everywhere.
- Reading the held object's size moves N4 in the right direction, but by less than the registered bar. It is not enough
  to close the place handoff.
- **New-layout competence is lower at these seeds than the development read showed:** s known 0.14-0.16 against 0.21 at
  seeds 0-2 (§5). It now limits S2 and S3 more than N4 does.

**Cost.** P_sz builds and creations: 2.4 A40-h. CPU: 48 rollout jobs, plus S1.

## 14. Round 3 design note and protocol: an approach frame for the object term (fixed 2026-11-02, before any code of this round; m3_prereg tag pp_r3)

**Weakness (measured on dev data; `experiments/pp_phase_diag.py` on P, rank 32, seeds 10-12).**
- In the new layout, P fails before the grasp.
  - Hold: 0.15-0.45 on the known pairings (familiar: 0.70-0.97).
  - In 55-80% of new-layout episodes no object moves at all.
- Once the object is held, arrival (0.67-0.92) and place-on-arrival (0.33-1.00) are close to their familiar values.
- So new-layout competence is lost in the object term, the approach and grasp.

**Diagnosis.**
- The object term works in the object's frame: origin the object, axes the object's yaw. The robot always starts at the
  same home pose.
- In the demos, the object sits within about 1 cm of one spot. So the home-to-object offset, expressed in the object frame,
  is nearly one vector.
- In a new layout, the object is 0.10 m or more from its spot. The start offset then changes in both bearing and distance,
  and the approach starts outside everything the demos cover.
- The target term had the same gap at the carry, and stage 1c closed it with the canonical place frame. Its x-axis is the
  bearing from the held object to the target, frozen at the hold's onset. Arrival in the new layout works today because of
  that frame.

**Change (one, structural; the place frame's principle applied to the approach).**
- The object term's **translation** (its position input and the translational action dims) is expressed in an approach frame:
  - origin: the object;
  - x-axis: the bearing from the robot's home end-effector position to the object.
  - The home position is the demos' mean end-effector position at step 0. It is a constant: every episode starts there, so
    this bearing is the bearing frozen at the approach's onset, the counterpart of cyaw.
- Rotations (the end-effector rotation input and the rotational action dims) stay in the object's yaw frame, because the
  grasp orientation is defined relative to the object.
- Under this frame, a new layout changes only the start distance, not the start bearing.
- No data, teacher or per-scene input is added. Φ, the target term and the parts are unchanged.

**Arm.** P_af (rank 32) at seeds 10-12. Reference: P (rank 32) at seeds 10-12, the same dev rollouts as round 2 (pp_r2_roll).

**Reads.** Dev states 4-23, N2's seeds. S1, S2 (all 32 cells) and S3 (k ∈ {1, 5, 10, 50}), with pp_l2's estimators and its
frozen GO mapping.

**Pre-registered round verdict.** **improves** iff the pooled known-pairing (K1-K6) success in the new layout, P_af minus P, is
> 0.10 with its state-bootstrap CI lower bound > 0.

## 15. Round 3 results (2026-11-02; m3_prereg pp_r3; results/flawed/pp_r3_read.json; rollouts pp_r3_roll)

**Round verdict: does not improve.** New-layout known-pairing success is 0.16 with P_af against 0.16 with P: +0.00
[−0.07, 0.06]. **Layer-2 GO = false**: S1 holds, S2 fails, and S3 meets its criterion in 7 of 12 cells.

- **S2 familiar.**
  - s known 0.79, s new 0.76; OM 1.11 [1.00, 1.25].
  - 3F fails on N4, −0.28 [−0.42, −0.14], and on N5 and N6, whose observed success is above the prediction.
- **S2 new.** s known 0.16, s new 0.19; 3F holds, OM 1.18 [0.76, 1.96]. The OM criterion fails on its lower bound.
- **S3.**
  - Familiar: bottle 0.82-0.86, every cell passes. Plate 0.86 / 0.81 / 0.73 at k = 1 / 5 / 10.
  - New layout: the plate passes at k = 5 and 10 only on wide CIs. The bottle's training pairing almost never succeeds
    there (s_train 0.00-0.10).
  - N4 in S3: 0.53-0.70.
- Turning the approach into a bearing frame does not move new-layout competence. The approach's bearing is not what fails.

**Cost.** P_af builds and creations: 1.7 A40-h. CPU: 36 rollout jobs, plus S1.

**Consecutive principled rounds without improvement: 2 (rounds 2 and 3).**

**Trace diagnostic** (descriptive; `experiments/pp_trace.py`; P and P_af seed 10; K1 and K6, states 4-11, both layouts).
- **P, new layout.** When the object starts far from home (0.31-0.39 m away horizontally, against 0.10-0.22 m in the demos),
  the end effector never moves toward it: the closest approach is the start distance. This held in 6 of 6 such K1 episodes
  and 1 of 1 such K6 episode. Close objects are grasped.
- **P_af, new layout.**
  - The approach frame removes that freeze: every far K1 object is reached and held (8 of 8).
  - But K1 then succeeds only 3 of 8 times. The bowl is lifted 0.10-0.40 m (familiar: about 0.12 m).
  - On K6, 4 of 8 episodes stop 0.18-0.21 m above the cheese and never descend.
- The approach frame acts on the bearing, as designed. The new-layout failures move to the descent and grasp and to the
  carry that follows.

## 16. Round 4 design note and protocol: one consistent approach frame (fixed 2026-11-02, before any code of this round; m3_prereg tag pp_r4)

**Weakness (measured, §15 trace).** With P_af, in the new layout the end effector reaches far objects but then stalls above
the cheese (4 of 8) or lifts the bowl 0.3-0.4 m. Known-pairing new-layout success stays 0.16.

**Diagnosis.**
- P_af expresses the object term's translation in the approach frame and its rotation in the object's yaw frame.
- A new layout rotates the approach bearing. The translations then look like the demos', but the rotations do not. The
  term sees a combination of rotation and position that no demo produced.
- The canonical place frame (stage 1c), which carries the bottle and bowl to new targets, has no such split. Its rotation
  input is expressed in the same bearing frame as its positions.

**Change (one, structural): one frame for the whole object term.**
- Positions, the end-effector rotation input, and both the translational and rotational action dims are all expressed in the
  approach frame (origin the object; x-axis the bearing from home to the object).
- The object's own yaw enters as an input relative to that frame: sin and cos of (object yaw − bearing). In P and P_af this
  slot is always 0.
- The object term is then equivariant to rotating the object's configuration about its vertical axis, as the place term is
  about the target's.
- Nothing else changes: P (rank 32), Φ, the target term, the parts, the data.

**Arm.** P_af2 (rank 32) at seeds 10-12. Reference: P (rank 32) at seeds 10-12 (pp_r2_roll), as in round 3.

**Reads.** Dev states 4-23, N2's seeds. S1, S2, S3 with pp_l2's estimators and frozen GO mapping.

**Pre-registered round verdict.** **improves** iff the pooled known-pairing new-layout success, P_af2 minus P, is > 0.10 with
its state-bootstrap CI lower bound > 0.

**Stop rule.** If this round fails too, three consecutive principled rounds have failed, and the mandate stops here.

## 17. Round 4 results and the stop (2026-11-02; m3_prereg pp_r4; results/flawed/pp_r4_read.json; rollouts pp_r4_roll)

**Round verdict: does not improve. It makes new-layout competence worse.** Known-pairing new-layout success is 0.01 with
P_af2 against 0.16 with P: −0.15 [−0.21, −0.10]. **Layer-2 GO = false.**
- S1 holds.
- **S2 fails.** In the familiar layout 3F fails on N4, −0.64 [−0.75, −0.53], with observed 0.12 against predicted 0.76.
  OM is 0.98 [0.90, 1.06]. In the new layout almost nothing succeeds.
- **S3 passes in all 6 familiar cells.**
  - Bottle 0.88 / 0.86 / 0.85 at k = 1 / 5 / 10, lower bounds 0.80-0.84.
  - Plate 1.02 / 0.94 / 0.89, lower bounds 0.74-0.92.
  - It fails in all 6 new-layout cells: the training pairing succeeds at most 0.05 of the time there.
- Putting the object's rotation into the bearing frame breaks the new layout. Under it the demos' grasp rotations become
  bearing-dependent, so a new bearing presents rotations no demo produced. Of the two frame choices, the split one (round 3)
  is the better.
- In the familiar layout, P_af2 is the best variant so far: s known 0.82, every familiar S3 cell passes, and N4's created
  plate part reaches 0.55-0.88. That is descriptive and carries no verdict.

**Stop: three consecutive principled rounds (2, 3, 4) failed to improve their targeted weakness** (mandate stop rule ii).

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 1 | N4 (bottle on plate) | diagnostic only | pp_n4d | "competence" on an arm that failed its controls: not interpretable | — |
| 2 | N4 place-on-arrival | target term reads the held object's size | pp_r2 | +0.08 [−0.05, 0.22]: does not improve | false |
| 3 | new-layout competence | object-term translation in the approach frame | pp_r3 | +0.00 [−0.07, 0.06]: does not improve | false |
| 4 | new-layout stall after the approach | the whole object term in the approach frame | pp_r4 | −0.15 [−0.21, −0.10]: worse | false |

**Where the program stands.**
- Familiar layout:
  - The sparse part library composes created atoms at their training partner's rate.
  - P_af2 passes every familiar S3 cell.
  - N4 remains below its factor prediction in every variant. It is a place-handoff failure: the planner shows N4 is
    feasible, and P drops the bottle off the release site.
- New layout:
  - Demos alone leave competence at 0.01-0.16 at seeds 10-12.
  - Frame changes moved the failure from "never leaves home" (P) to "stalls above or over-lifts" (P_af). They did not raise
    success.
  - Layout generalization does not come from frame equivariance at this demo coverage. That is the plan's §4.1 claim, and
    it fails here.
- Layer-2 GO was not reached, so Layer 3 did not start.

**Cost of the mandate.**
- GPU: 7.8 A40-h. N4 fits 1.7, P_sz 2.4, P_af 1.7, P_af2 2.0.
- CPU: 146 rollout, oracle, generation, trace and S1 jobs, 48.1 CPU-job-h, plus pre-registration and read jobs.

## 18. Step 1 design note: fixing the N4 instrument (pp_n4d2; fixed 2026-11-02, before any code of this step)

**The defect (§11).** The single-task fits score 0.00 on their own K4 and K1 controls, where P scores 0.63 and 0.97. Their
training losses converged.

**Hypotheses, tested in this order.**
1. **The class, budget or my training path cannot fit one task.** Test: a single-task fit on K1's 50 human demos, through
   `pp_p_train.py build` (P's own row builder) with `--exclude` of every other task. If it fails, the class or budget is at
   fault. If it works, the problem is in the generated data or in my generated-row path.
2. **Coverage, not a bug.** pp_n4d's carries came from stage 1g's relocated-layout sampler, with objects anywhere in the box.
   Fifty such episodes cover the familiar spot thinly, while the dev reads are on the familiar layout. Test: a single-task fit
   on 50 generated carries from **familiar-layout** starts (N2's familiar sampler, new seeds 960000 + 1000 j + k), read on
   the same task.
3. **A bug in the generated-row path.** pp_n4d's train() builds rows outside P's builder. Test: route the generated rows
   through the same function P uses (`pp_p_train.rows` logic over a demos-format dict) and compare fits on identical data.
   If (1) passes and (2) fails, I inspect the path field by field: held, off, prev, chunk boundaries.

**Validation gate (from the user).** The instrument is valid iff a single-task fit on generated K1 (or K4) carries, from
familiar-layout starts, reaches within 0.15 of P on that task (point estimate, dev states 4-23, 3 seeds × 20 states).

**Tilt.** Rollouts for this step record the instruction object's tilt at every step: the angle between its body z-axis and
world z, from the simulator's body quaternion. The rollout wrapper is `experiments/pp_n4d2.py`, and the episode itself is
fa1h_roll.episode, unchanged. Per episode it reports the maximum tilt while held, the final tilt, and whether the object ends
tipped (final tilt > 45°).

**Then pp_n4d2** (pre-registered after the gate passes, before any N4 re-read):
- pp_n4d's arms and mapping, unchanged, with arm (a) and the controls trained on familiar-layout generated carries.
- Tilt separates N4's two failure modes. A failed episode that ends tipped on or near the plate is "tipping". One that ends
  upright but off the plate region is "off-site release".

## 19. Step 1 findings so far, and the second design note (instrument v2; fixed 2026-11-02, before its code)

**Hypothesis 1 is rejected: the class and budget fit one task.** A single-task fit on K1's 50 human demos, through P's own
builder (`pp_p_train.py build --exclude` of every other task, rank 32, seeds 0-2), succeeds on K1 at 1.00, 0.95 and 0.95. P
scores 0.95-1.00 on the same states.

**Hypothesis 2 is rejected: it is not coverage.** Single-task fits on 50 generated carries from **familiar-layout** starts
still score 0.00 on K1 and 0.00 on K4 (P: 0.97 and 0.63). The gate fails.
- The familiar generator kept 50 of 52 K1 attempts, 50 of 76 for K4 and 50 of 83 for N4.

**The cause, from a trace** (`pp_trace.py` on the generated-K1 fit, familiar states 4-9).
- The fit reaches the bowl: closest approach 1.5-2.7 cm horizontally.
- It closes the gripper with the end effector 2.7-3.8 cm above the bowl's origin. P closes at about 1.8 cm.
- It then never descends again, and the bowl is not lifted (lift 0.002-0.061 m).
- The planner explains this. In its close and open phases (2 and 6) it freezes all motion (`a[:6] = 0`) for 12 steps while
  the gripper moves. The policy learns "close and stop". With flow chunks executed 5 steps open loop, the close fires about
  1 cm early and the freeze locks the gripper above the object.
- Human demos keep moving while they close. So the instrument's defect is in the generator's behaviour, not in the row path:
  the fields, chunk boundaries and held labels go through the same functions as P's.

**Fix (instrument only, no architecture involved).**
- In the generator's close and open phases, the planner keeps servoing to the grasp point (or the release point) while the
  gripper moves, instead of freezing.
- Everything else is stage 1g's planner unchanged: the keypoints, controller, gains, noise, waypoints, phase durations and
  acceptance rule.
- Familiar-layout carries for K1, K4 and N4 are regenerated with new seeds (970000 + 1000 j + k), the fits retrained, and the
  gate re-run.
- If the gate still fails, the instrument is declared unrepairable within this design. I report that and do not pre-register
  N4 on it.

## 20. Step 2 diagnostic: which input leaves the training distribution (2026-11-02; `experiments/pp_ood_diag.py`; results/flawed/pp_ood_diag.json)

**Method.**
- For each checkpoint (seed 10) and each known pick-and-place task, the training cloud is the object term's own input
  features on every not-held demo row of that task.
- The probe is the same features at the episode start of dev states 4-23, computed through the rollout's own path.
- Per input group, the table gives the standardized nearest-training-row distance (median over states). Inside the cloud,
  values are about 0.1-0.4.

| model | layout | ee position | ee rotation | gripper / last command / slot | start offset (m) |
|---|---|---|---|---|---|
| P | familiar | 0.06-0.09 | 0.01-0.03 | 0.00 | 0.05-0.20 |
| P | new | **1.7-5.6** | 0.01-0.03 | 0.00 | 0.12-0.31 |
| P_af | new | **1.9-6.8**, except K6 0.09 | 0.01-0.03 | 0.00 | 0.12-0.31 |
| P_af2 | new | 0.1-6.8 | **0.8-9.6** | slot yaw up to 7.1 | 0.12-0.31 |

**Findings.**
- **Only one input leaves the distribution: the end effector's position relative to the object.** Rotation, gripper, last
  command, slot and the Φ gate input (held, which is 0 at every start) all stay inside.
- The demos' horizontal start offsets reach at most 0.15 m (K1-K3), 0.11-0.15 m (bottle) and 0.23 m (cheese). The new
  layout's start offsets are 0.31 m (bowl), 0.22 m (bottle) and 0.12 m (cheese).
- The approach frame (P_af) removes the bearing part of the gap. On K6, where the new start distance is inside the demo
  range, it brings the position input fully inside (1.68 → 0.09). Where the distance itself is larger than any demo's, the
  gap stays.
- P_af2's collapse (round 4) has a measured cause: it pushes the rotation input and the slot yaw out of the distribution.

## 21. Round 5 design note and protocol: radial relocation of the approach state (fixed 2026-11-02, before any code of this round; m3_prereg tag pp_r5)

**Weakness (measured, §20).** In the new layout the object term's position input is out of distribution because the start
distance (up to 0.31 m) exceeds every demo's (at most 0.11-0.23 m by object). The bearing is already handled by P_af.

**Principle.**
- In the far field, the approach is a transport toward the object along the bearing, and moving along that ray is a symmetry
  of the task.
- A state farther out on the ray can therefore be **relocated** to the nearest state the demos cover on the same ray:
  same bearing, same height, horizontal distance clipped to the demos' support radius r. The term acts there.
- Its demo-covered velocity at r points inward. As the end effector closes in, the clip releases, and the term then sees the
  true state.
- The place frame does the same thing at the carry. It removes what the task is symmetric to and keeps what matters.

**Change (one, structural; on P_af).**
- In the object term's input, the horizontal part of the end-effector offset (in the approach frame) is clipped to radius r.
  Height, rotations, the output frame and everything else are P_af's.
- r is a constant fitted once per build, never per scene or task: the smallest, over the build's training objects, of the
  99th percentile of the horizontal end-effector-to-object distance on not-held demo rows. It is stored in the checkpoint.
- No data, teacher or per-scene input is added.

**Arm.** P_rc (rank 32) at seeds 10-12. Reference: P_af (rank 32), seeds 10-12, the round-3 rollouts (pp_r3_roll), which is
this change's base.

**Reads.** Dev states 4-23, N2's seeds. S1, S2 and S3 with pp_l2's estimators and frozen GO mapping.

**Pre-registered round verdict.** **improves** iff the pooled known-pairing new-layout success, P_rc minus P_af, is > 0.10
with its state-bootstrap CI lower bound > 0.

## 22. Step 1 and round 5 results (2026-11-02; m3_prereg pp_n4d2 and pp_r5; results/flawed/pp_n4d2_read.json, pp_r5_read.json)

**Instrument v2: the gate passes.** With the servoing generator (§19), the single-task fits on familiar-layout generated
carries score 1.00 on K1 (P 0.97) and 0.72 on K4 (P 0.63). Both are within 0.15 of P, so the instrument is valid. The
planner itself succeeds at 0.95 (K1), 0.75 (K4) and 0.65 (N4) on the dev states.

**pp_n4d2 verdict, as pre-registered with pp_n4d's mapping: "combination", failing phase place-on-arrival.**
- N4's single-task fit (arm a) succeeds at 0.55 [0.38, 0.72]. P succeeds at 0.22 [0.10, 0.35]. Δ = +0.33 [0.13, 0.53].
- The controls are clean: no control flag.
- By phase, the gap is at the place:
  - hold Δ +0.03;
  - arrival Δ +0.07;
  - place given arrival 0.77 (a) against 0.34 (P), Δ +0.43 [0.20, 0.66].
- Release error in xy: 0.030 m for arm a, 0.093 m for P, 0.027 m for the planner.
- N4 is therefore within the class's competence from carries alone. P's composed parts fail at the place interface, and N4
  becomes a target for a later round at that interface.
- **Tilt separates the failure modes.**
  - P: 29 failures while held, 27 of them end tipped, median maximum tilt while held 95°.
  - Arm a: 11 failures while held, 3 tipped, median maximum tilt 9°.
  - Planner: 2 failures, 1 tipped.
- P's N4 failure is **tipping**: the bottle goes over while it is carried and placed on the plate, not an upright release off
  the site. The plate part learned its carry from the bowl alone, a short, rotation-insensitive object.

**Round 5 verdict: improves.** Known-pairing new-layout success is 0.31 for P_rc against 0.16 for P_af: +0.15 [0.08, 0.21].
Round 5 is the first round that raised new-layout competence. **Layer-2 GO = false.**
- S1 holds.
- **S2 fails, on 3F in both layouts.**
  - Familiar: OM 1.07 [0.97, 1.20], which passes. 3F fails on N4 (observed 0.30 against predicted 0.65, D3 −0.35
    [−0.47, −0.25]) and on N3 (−0.13 [−0.27, −0.02]).
  - New: s known 0.31, s new 0.23, OM 1.17 [0.83, 1.87]. Observed exceeds predicted on N5 (+0.09 [0.03, 0.15]) and N6
    (+0.09 [0.02, 0.19]), where predictions are near 0.
  - New-layout success by known task: K1 0.28, K2 0.55, K3 0.57, K4 0.10, K5 0.00, K6 0.33.
- **S3 passes 4 of 12 cells.**
  - Bottle, familiar: k1 0.82 [0.76, 0.88] and k5 0.87 [0.81, 0.94] pass. k10 0.80 [0.73, 0.86] fails by 0.004.
  - Bottle, new: k1 0.95 [0.67, 2.02] passes. k5 and k10 fail.
  - Plate, familiar: k1 0.82 [0.73, 0.91] passes. k5 0.76 and k10 0.74 fail.
  - Plate, new: all fail.

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| step 1 | single-task fits score 0 (instrument) | servoing generator (instrument only) | pp_n4d2 | gate passes; N4 = combination at place-on-arrival | — |
| 5 | new-layout position input out of distribution (§20) | radial relocation of the approach state | pp_r5 | +0.15 [0.08, 0.21]: improves | false |

## 23. Round 6 diagnostic: where N4's bottle tips (2026-11-02; `experiments/pp_tilt_trace.py`, `pp_carry_rot.py`, `pp_carry_dist.py`)

**Method.** Familiar dev states 4-23, one seed each: P_rc (seed 10) on N4, K4 and K1, and N4's single-task fit (arm a, seed 0)
on N4. Each step records the hand's and the bottle's rotations. From the lift (the object 2 cm above its start), the trace
measures:
- the object's tilt;
- the hand's turn since the lift;
- the object's slip relative to the hand.

**Two failure modes, with different owners.**
- **Tipped at the grasp.** In 7 of P_rc's 20 N4 episodes the bottle is already tilted more than 30° at the lift. The same 7
  states tip identically in K4 under P_rc: the same lift step and the same tilt, because the bottle part acts alone until the
  hold. Arm a tips at the grasp in 6 states. This mode belongs to the bottle part's grasp. It is not the composition.
- **Turned at the place (N4 only).** The plate part, holding the bottle, turns the hand.
  - Over P_rc's lifted episodes that were not tipped at the grasp, the largest hand turn has median 48° (15-170°).
  - Five failures (states 4, 6, 13, 15, 21) tip at the plate after turns of 20-74°. The gripper often opens mid-turn.
  - The same plate part holding the bowl (K1) turns at most 12°, with median about 5°.
  - Arm a on N4 turns at most 20°, with median 7°. P_rc on K4 has median 16°.
- **The demos never turn that far onto a plate.** The median largest hand turn per carry is 7° for K1 (90th percentile 13°)
  and 13° for K4. Only K5 (bottle onto the rack) turns far, at 54°.

**The plate part's input leaves its support.**
- The horizontal object-to-target distance where the plate part's carries start is 0.14 m (median), at most 0.17 m, and 0.16
  m at the 99th percentile of held rows.
- In the familiar scene the bottle starts 0.19-0.28 m from the plate (median 0.25, at K4's hold onsets). Every N4 carry
  therefore begins outside the region where the plate part was trained.
- The target term's offset is the only object-dependent input whose range differs. It reads no hand position and no object
  identity, and its rotation input starts at identity at every hold onset.
- This is round 5's weakness again, now on the target term: a carry starting farther out than any of the part's demos. Inside
  that support (the bowl) the part holds the hand still. Outside it (the bottle) the part turns the hand.

## 24. Round 6 design note and protocol: radial relocation of the carry state (fixed 2026-11-02, before any code of this round; m3_prereg tag pp_r6)

**Weakness (measured, §23).** N4's composed carry starts 0.19-0.28 m from the plate. The plate part's demos start within 0.17
m. Out there the plate part turns the hand 20-74° and tips the bottle at the place.

**Principle.** It is round 5's principle applied to the second term. In the canonical place frame (bearing frozen at the hold's
onset), the far part of a carry is a transport along the bearing toward the target. Moving along that ray is a symmetry of the
task. A held state farther out on the ray is relocated to the nearest state the demos cover on the same ray: same bearing, same
height, horizontal distance clipped to the support radius r_t. The term acts there. Near the target the clip releases, and the
term sees the true state.

**Change (one, structural; on P_rc).**
- In the target term's input, the horizontal part of the held object's offset from the target (canonical frame) is clipped to
  radius r_t. The height, the rotation since the onset, every other input, the output frame and the object term stay P_rc's.
- r_t is one constant per build, fitted by round 5's rule and never per scene or task: the smallest, over the build's training
  targets, of the 99th percentile of the horizontal object-to-target distance on held demo rows. It is stored in the
  checkpoint.
- No data, teacher or per-scene input is added. Created parts inherit the base build's r_t.
- The grasp-time tipping (§23) is the bottle part's. This change does not address it.

**Arm.** P_rt (rank 32), seeds 10-12, builds full, nobottle and noplate as in every round. Reference: P_rc, seeds 10-12, the
round-5 rollouts (pp_r5_roll).

**Reads.** Dev states 4-23, N2's seeds. S1, S2 and S3 with pp_l2's estimators and frozen GO mapping.

**Pre-registered round verdict.** **improves** iff N4's familiar-layout success, P_rt minus P_rc, is > 0.10 with its
state-bootstrap CI lower bound > 0 (metric n4_fam).

## 25. Round 6 results (2026-11-02; m3_prereg pp_r6; results/flawed/pp_r6_read.json; rollouts pp_r6_roll)

**The fitted radius.** The pre-fixed rule gives r_t = 0.0925 m in all three builds. The smallest training target support is
not the plate's (0.16 m); it is set by another target's carries, as the rule allows. r_t was used as fitted. (The object term's
r is 0.124 m in the full and noplate builds and 0.173 m in nobottle.)

**Round verdict: improves.** N4's familiar success is 0.57 for P_rt against 0.30 for P_rc: +0.27 [0.13, 0.42].
**Layer-2 GO = false.**
- S1 holds.
- **S2 fails.**
  - Familiar: s known 0.83, s new 0.81, OM 1.09 [1.02, 1.17], which passes. 3F now holds on N4: observed 0.57 against
    predicted 0.69, D3 −0.13 [−0.28, 0.01].
  - 3F fails on N3 (−0.15 [−0.30, −0.02]) and, sign-agnostic, on N5 (+0.08 [0.02, 0.14]) and N6 (+0.11 [0.03, 0.19]), where
    the observed success exceeds the prediction.
  - New layout: s known 0.34 (P_rc 0.31), s new 0.23, OM 0.92 [0.74, 1.18]. 3F fails on N3, N4 and N6.
- **S3 passes all 6 familiar cells**, plus bottle-new k10.
  - Bottle, familiar: 0.96 / 0.97 / 0.92 at k = 1 / 5 / 10, lower bounds 0.86-0.92.
  - Plate, familiar: 0.88 / 0.88 / 0.86, lower bounds 0.78-0.79.
  - New layout: bottle k10 passes, 1.33 [0.67, 2.67], on near-zero rates (0.03 and 0.04). The other 5 cells fail. Plate-new
    ratios are 0.39-0.50: the created plate part succeeds at 0.15 on N4 in the new layout, while the training pairing K1
    succeeds at 0.48-0.65.
- Known-task success also rises in the familiar layout: K1 1.00, K2 1.00, K4 0.72, K5 0.43 (P_rc 0.95, 0.98, 0.67, 0.33).

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 6 | N4 place tipping: carry starts outside the plate part's support (§23) | radial relocation of the carry state | pp_r6 | +0.27 [0.13, 0.42]: improves | false |

## 26. Round 7 diagnostic: the new layout fails at the grasp (2026-11-02; `experiments/pp_phase_diag.py`, `pp_trace.py`, `pp_close_diag.py`)

**Where the new layout fails now (P_rt, seeds 10-12).** It fails at the grasp, not at the place.
- Bottle tasks (K4, K5, N4-N6): hold 0.18 (familiar 0.85). In 38% of episodes no object is moved at all.
- Cheese tasks (K6, N1-N3): hold 0.48 (familiar 0.98).
- Bowl tasks (K1-K3): hold 0.85, and once held, arrival is 0.65-0.75.
- Given arrival, the place mostly succeeds (0.53-1.00).

**The moment of the close (P_rt seed 10, dev states 4-23).** The probe is the object term's input at its first close command,
or, for an episode that never closes, at its closest approach. The cloud is the not-held demo rows whose chunk starts with a
close.
- **Bottle, new layout: 0 of 20 succeed. The hand descends before it arrives.**
  - The close fires 1.4-18 cm from the bottle horizontally, median 9 cm. In the demos' close rows this distance is at most 7 cm
    (median 1.3 cm).
  - At the close the hand is only 2-16 cm above the bottle's origin. The position input is out of the cloud (nn-z 0.9-9.9).
  - Under the radial relocation (§21), a far state is shown to the term at radius r with its true height. The term then applies
    r's descent rate over the whole extra distance, so the hand comes down too early.
- **Cheese, new layout: 9 of 20 succeed. The hand arrives before it descends.**
  - In all 10 episodes that never close, the hand hovers directly above the cheese (1-4 cm horizontally) but 15-24 cm high.
    That state is out of the cloud (nn-z 1.5-4.1). These are the near starts (0.08-0.14 m).
  - Every grasp that does close does so inside the cloud (nn-z ≤ 0.44) and succeeds.
- **Common cause.** The demos couple the hand's height to its horizontal distance from the object. New-layout starts break that
  coupling in both directions:
  - far starts, relocated, come down too early (bottle);
  - near starts stay too high (cheese).
- **Side finding.** At the close, the hand's rotation in the bottle's frame is far outside the cloud even in familiar successes
  (nn-z up to 156). The bottle is rotationally symmetric, so its body yaw carries no task information, yet the object term
  reads the hand's rotation relative to it.

**Round 7 was not run.** Each candidate change fixes one direction of the coupling only:
- zeroing the term's vertical velocity while the clip is active;
- an approach-progress input in place of (horizontal, height);
- a symmetric-object frame.

Each is either an executor-level output rule or a new input design. That needs the user's decision under the mandate's rule
against executor engineering.

## 27. Round 7 design note and protocol: a horizontal scale frame frozen at the episode's start (fixed 2026-11-02, before any code of this round; m3_prereg tag pp_r7)

(The user's amended mandate of 2026-11-02 continues the rounds. This note supersedes §26's "not run".)

**Weakness (measured, §26).** On far new-layout starts the object term breaks the demos' coupling between height and
horizontal progress. Under round 5's per-step clip, a far state is shown at radius r with its true height. The term applies
r's descent over the whole extra distance, comes down early, and closes 5-18 cm from the bottle. Bottle hold in the new layout
is 0.18.

**Why not a 3D ray (checked first, `experiments/pp_ray_check.py`).** Far-field demo motion is not a straight line to the object
for every object:
- the bottle's commands lie 12-20° from the 3D direction to it;
- the bowl's and cheese's lie 10-31° from the horizontal direction (62-69° from the 3D one).

The coupling of height to progress is object-specific, so a fixed geometric rule (horizontal or 3D) cannot preserve it for
all objects.

**Principle.** The approach is equivariant to a uniform scaling of the horizontal plane about the object. A demo that started
farther out, with the same height profile stretched horizontally, is the same approach. One fixed scale per episode
therefore maps any start onto the support:

s = min(1, r / |horizontal start offset|)

The scale is frozen at the episode start, as the place frame's bearing is frozen at the hold's onset. In that frame the term
sees and acts in scaled horizontal units:
- input: the horizontal hand-object offset × s, and the noisy chunk's horizontal translation × s;
- output: the horizontal translation velocity × 1/s.

Every object then keeps its own coupling of height to progress, because the term never sees an apparent distance beyond r,
and apparent progress advances at the demos' rate.

**Change (one, structural; on P_rt).**
- The object term's per-step horizontal clip (round 5) is replaced by this frame. With s fixed at the start, the apparent
  offset never exceeds r, so the clip would be inert.
- r is fitted by round 5's rule, unchanged. The start offset is the hand's position at the end of the 10 settle steps
  (rollouts) or at step 0 (demo rows), relative to the object.
- The target term, the gate and everything else stay P_rt's.
- Callers that pass no start (S1's grasp frames, which are start states) use the current hand position as the start.
- No data, teacher or per-scene input is added.
- Known limit: commands above 1 are clipped by the executor, as for every arm. If the scaled horizontal command saturates, the
  coupling is only partly restored.

**Arm.** P_hs (rank 32), seeds 10-12, builds full, nobottle and noplate. Reference: P_rt, seeds 10-12 (pp_r6_roll).

**Reads.** Dev states 4-23, N2's seeds. S1, S2 and S3 with pp_l2's estimators and frozen GO mapping.

**Pre-registered round verdict.** **improves** iff the pooled known-pairing new-layout success, P_hs minus P_rt, is > 0.10
with its state-bootstrap CI lower bound > 0 (metric s_known_new).

## 28. Round 7 results (2026-11-03; m3_prereg pp_r7; results/flawed/pp_r7_read.json; rollouts pp_r7_roll)

**Round verdict: does not improve. It makes new-layout competence worse.** Known-pairing new-layout success is 0.23 for P_hs
against 0.34 for P_rt: −0.11 [−0.16, −0.06]. **Layer-2 GO = false.**
- S1 holds. S3 passes all 6 familiar cells and none of the new-layout ones.
- **S2 fails.** Familiar: OM 1.11 [1.03, 1.20]; 3F fails on N3, N4 (−0.11 [−0.21, −0.02]), N5 and N6. New: s known 0.23, OM
  0.92 [0.68, 1.27].
- **The scale frame hurts the bottle in both layouts.**
  - Familiar K4 drops from 0.72 to 0.48, K5 from 0.43 to 0.23. The familiar bottle starts (0.11-0.15 m) lie beyond r = 0.124 m,
    so they are scaled too.
  - New-layout bottle hold is 0.15 (P_rt 0.18), with no object moved in 68% of episodes (P_rt 38%).
- Cheese new-layout hold falls from 0.48 to 0.33, and bowl hold from 0.85 to 0.73.
- **Not established here:** whether the cause is saturation of the scaled horizontal commands (×1/s up to 2.5, clipped at 1 by
  the executor; the known limit in §27) or the loss of round 5's per-step clip. No trace was run.

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 7 | new-layout grasp: far starts descend before arriving (§26) | horizontal scale frame frozen at the episode start | pp_r7 | −0.11 [−0.16, −0.06]: worse | false |

The best arm is still P_rt (round 6). That is one failed round since the last improvement.

## 29. Round 8 diagnostics: round 7's cause, and the cabinet's arrival under the library-wide carry radius (2026-11-03; `experiments/pp_sat_diag.py`, `pp_arrive_diag.py`, `pp_support_diag.py`)

**Round 7's open question: saturation or the scaled grasp?** (`pp_sat_diag.py`, P_hs and P_rt seed 10, K1, K4 and K6, dev states
4-23, both layouts.) Per episode, before the hold: the fraction of executed steps whose raw horizontal command exceeds the
executor's clip, and the hand-object distance where the first close fires.
- **Saturation is rare.** The demos never saturate in the far field (0 of their K1, K4 and K6 rows). Under P_hs the median
  episode saturates on 0 steps for K4 and K6 in both layouts, and on 7% of steps for the bowl in the new layout.
- **The close fires farther from the object.** In the new layout, P_hs closes at a median 0.089 m from the bowl (P_rt 0.055 m),
  and its K1 success there is 0.15 (P_rt 0.35). The scale frame shrinks the hand-object offset by s at the grasp too. A close
  that is accurate in apparent units is off by 1/s in true units, which is up to 2.5 times for far starts.
- **The familiar bottle was not scaled at test time.** Its familiar starts are 0.04-0.06 m from the hand, inside r, so s = 1 in all
  20 states (§28's "0.11-0.15 m" are the bottle demos' starts, not the familiar rollouts'). P_hs's familiar K4 loss is a training
  effect of the frame on the shared backbone, not a test-time one.
- So round 7 failed because the grasp is not equivariant to horizontal scale: the object's size and the gripper's width are
  fixed. Saturation, §27's known limit, is not the main cause.

**The cabinet loses arrival under round 6's carry radius** (`pp_arrive_diag.py` on the S2 rollouts of earlier rounds, seeds
10-12, familiar layout). Arrival given hold, for the three pairings with the cabinet top as target:

| arm | K3 (bowl) | K4 (bottle) | N3 (cheese) | pooled |
|---|---|---|---|---|
| P, rank 32 | 0.97 | 0.90 | 0.91 | — |
| P_af | 0.95 | 0.85 | 0.98 | — |
| P_rc | 0.93 | 0.75 | 0.92 | 0.87 |
| P_rt | 0.78 | 0.73 | 0.81 | 0.78 |

- Every other familiar target arrives at 0.90-1.00 under P_rt.
- The episodes that do not arrive stop 0.13-0.18 m short (median closest approach; radius 0.10 m), and most of them drop the
  object. Cabinet carries start 0.28-0.42 m out and 0.22 m below the cabinet top. They must climb while they travel.
- **This is the 3F failure of the familiar layout.** 3F gives arrival to the object (a_o). A target that loses arrival breaks the
  law in both directions:
  - N3 (cheese on the cabinet) runs below its prediction, −0.15;
  - the bottle's a_o includes K4's cabinet loss, so N5 and N6 (bottle on stove, in bowl) run above theirs, +0.08 and +0.11.

**The cause: one radius for the whole library** (`pp_support_diag.py`, demo rows of the full library). Round 6's rule takes the
minimum over targets of each target's 99th percentile of the held object-to-target distance. Each part's own support is:

| target part | carries (onsets) | 99th percentile (m) | onset median / max (m) |
|---|---|---|---|
| plate | 56 | 0.160 | 0.142 / 0.171 |
| bowl | 52 | 0.162 | 0.142 / 0.164 |
| stove | 57 | 0.273 | 0.251 / 0.285 |
| cabinet top | 129 | 0.311 | 0.286 / 0.326 |
| rack | 55 | 0.173 | 0.133 / 0.167 |
| cabinet drawer | 69 | 0.266 | 0.176 / 0.292 |
| stove front | 2 | 0.0925 | 0.070 / 0.094 |

- The minimum, 0.0925 m, comes from the stove front: 40 held rows from 2 carries of a task outside the N2 pick-and-place set.
- So every carry of every target is relocated to 0.0925 m, including carries that start inside their own part's support. The
  cabinet's familiar carries start at 0.28 m, inside its 0.31 m support.
- Relocation at the true height breaks the coupling of the climb to horizontal progress (§26's mechanism, now on the carry).
- Object parts, same rule (not-held rows): bowl 0.173 m, cheese 0.212 m, bottle 0.124 m. Round 5's r, 0.124 m, is the bottle's,
  so the bowl and cheese are relocated inside their own supports too.

## 30. Round 8 design note and protocol: relocation into each part's own support (fixed 2026-11-03, before any code of this round; m3_prereg tag pp_r8)

**Weakness (measured, §29).** Under P_rt, familiar arrival at the cabinet top falls to 0.73-0.81 (P_rc: 0.75-0.93). The
library-wide carry radius, 0.0925 m, is set by a 2-carry target. It relocates cabinet carries that start inside the cabinet
part's own support.

**Principle (Prop. 2).** A state is relocated into **the part's** support. A state inside the acting part's support needs no
relocation, so the radius is a property of the part, like its LoRA rows. Rounds 5 and 6 took the minimum over all parts. That
relocates in-support states of every part whose support is larger.

**Change (one, structural; on P_rt).**
- Each part carries its own support radius, fitted by rounds 5-6's rule on its own demo rows:
  - object parts: the 99th percentile of the horizontal hand-to-object distance on its not-held rows;
  - target parts: the 99th percentile of the horizontal object-to-target distance on its held rows.
- Each term clips with the radius of the part the row activates. The clip's form and every other input are P_rt's.
- A created part fits its radius from its own k creation demos by the same rule. The radius is stored with the part, and the
  bitwise check of every other parameter and buffer is unchanged.
- An atom without demos in a build falls back to the library-wide radius. No N2 read uses such an atom before its creation.
- No data, teacher or per-scene input is added. The radii are fitted, never set by hand.
- Expected side effects:
  - the bottle's object radius is unchanged (0.124 m is its own);
  - the plate's carry radius rises from 0.0925 m to 0.160 m. N4's carries (0.19-0.28 m) stay relocated, to the edge of the plate
    part's support. Whether N4 keeps round 6's gain is not predicted.

**Arm.** P_pp (rank 32), seeds 10-12, builds full, nobottle and noplate. Reference: P_rt, seeds 10-12 (pp_r6_roll).

**Reads.** Dev states 4-23, N2's seeds. S1, S2 and S3 with pp_l2's estimators and frozen GO mapping.

**Pre-registered round verdict (metric cab_arr_fam).**
- The metric is familiar arrival given hold, pooled over K3, K4 and N3: arrived episodes over held episodes, seeds pooled.
- Δ = P_pp − P_rt, with a 2,000-draw state bootstrap (the same states resampled for both arms).
- **improves** iff the CI's lower bound is > 0. This is the mandate's rule ("the CI excludes 0 upward"). Rounds 2-7 also
  required Δ > 0.10.

## 31. Round 8 results (2026-11-03; m3_prereg pp_r8; results/flawed/pp_r8_read.json; rollouts pp_r8_roll)

**The fitted radii** (full build, by the pre-fixed rule). Object parts: bowl 0.173 m, cheese 0.212 m, bottle 0.124 m. Target
parts: plate 0.160 m, stove 0.273 m, cabinet top 0.311 m, rack 0.173 m, bowl 0.162 m. The created parts fit theirs from their
own demos: bottle (K4) 0.056 / 0.067 / 0.075 / 0.092 m at k = 1 / 5 / 10 / 50; plate (K1) 0.141 / 0.159 / 0.154 / 0.160 m.

**Round verdict: improves.** Familiar cabinet arrival given hold is 0.94 for P_pp against 0.78 for P_rt: +0.16 [0.03, 0.30].
**Layer-2 GO = false.**
- S1 holds.
- **S2 familiar.** s known 0.80, s new 0.79, OM 1.09 [1.00, 1.20], which passes.
  - **3F now holds on N1, N2, N3, N5 and N6.** N3 runs at its prediction (0.97 against 0.97). N5 and N6 run within 0.02 of
    theirs, because the bottle's a_o no longer carries the cabinet's loss.
  - **3F fails only on N4**: observed 0.23 against predicted 0.84, D3 −0.61 [−0.75, −0.47]. The plate's place-on-arrival with the
    bottle is 0.29.
  - Round 6's N4 gain is lost. The plate's carry radius rose from 0.0925 m to its own 0.160 m. N4's carries (onsets 0.26-0.28 m)
    are now relocated to the edge of the plate part's support, not deep inside it. §30 did not predict this outcome.
- **S2 new.** s known 0.36 (P_rt 0.34), s new 0.23, OM 1.04 [0.88, 1.23], which passes. 3F fails only on N6, +0.06 [0.004, 0.14].
  - Bowl hold in the new layout rises to 0.93 (P_rt 0.85). K2 rises to 0.75 and K3 to 0.72.
  - K1 falls to 0.18: its place-on-arrival on the plate is 0.22. The bottle's new-layout hold is 0.13.
- **S3 passes 7 of 12 cells**: all 6 familiar cells, and bottle-new at k = 1, 0.94 [0.89, 1.00], on low rates (0.10 and 0.09).
  - Familiar: bottle 0.91 / 0.91 / 0.91, lower bounds 0.85-0.86; plate 0.89 / 0.89 / 0.91, lower bounds 0.80-0.81.
  - New layout: the bottle fails at k = 5 and 10, where s_train is 0.00-0.02. The plate fails on its lower bounds (0.37-0.59).

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 8 | familiar cabinet arrival lost under the library-wide carry radius (§29) | relocation into each part's own support | pp_r8 | +0.16 [0.03, 0.30]: improves | false |

The failure streak resets to 0. The familiar layout now fails 3F on N4 alone, and so does the plate's place-on-arrival.

## 32. Round 9 diagnostic: N4's bottle touches down before it arrives (2026-11-03; `experiments/pp_carry_profile.py`, `pp_tilt_trace.py`)

**Method.** P_pp and P_rt, seed 10, dev states 4-23: N4 familiar, K1 familiar and K1 new. At every held call, the held object's
foot relative to the target, as the target term reads it before any clip: horizontal distance d and height h. The
**touch-down distance** is d where the foot first returns below 3 cm after the lift. The reference is K1's held demo rows, the
plate part's only training carries.

**The plate part's demo profile.** The foot rises to about 9 cm by d = 0.03-0.10 m (median h 0.087-0.091 m) and comes down
only inside 3 cm of the plate's centre (median h 0.015 m there).

| episode set | carry onset d (m) | touch-down d, median (m) | success |
|---|---|---|---|
| K1 familiar, P_pp | 0.12-0.19 | 0.017 | 19 / 20 |
| K1 familiar, P_rt | 0.13-0.17 | 0.009 | 20 / 20 |
| N4 familiar, P_pp | 0.24-0.34 | 0.060 | 6 / 17 held |
| N4 familiar, P_rt | 0.24-0.28 | 0.045 | 8 / 16 held |

- **The bottle's foot comes down 4-10 cm short of the plate's centre.** At d = 0.03-0.06 m its median height is 0.025 m
  (P_pp) and 0.033 m (P_rt), against the demos' 0.091 m.
  - Every P_pp failure that touches down does so at 0.05-0.10 m. The successes touch down at 0.01-0.05 m.
  - Earlier, at d = 0.10-0.22 m, the bottle is carried higher than the demos: median 0.10-0.17 m (both arms), against 0.01-0.04 m.
- **The tilt trace confirms the mechanism.** N4 tips at d = 0.04-0.07 m with the bottle's origin 1.4-2.5 cm above the plate's, after 15-40°
  of hand turn and 20-30° of slip. The bottle's foot drags on the plate while the hand still travels.
- **This is §26's mechanism on the carry.** Every N4 carry starts beyond the plate part's support (0.16 m). Under radial
  relocation the term sees the support radius with the true height. It runs its onset program, the lift, over the extra
  distance, and its descent then starts too far out. Round 6's smaller radius (0.0925 m) presented the far field as already
  close and hid part of this. Round 8's own radius (0.160 m) shows it fully.
- K1 in the new layout (onsets up to 0.41 m) shows the same thing: P_pp touches down at a median 0.038 m and over-lifts to
  0.18-0.25 m at d = 0.06-0.22 m.

## 33. Round 9 design note and protocol: a carry scale frame frozen at the hold's onset (fixed 2026-11-03, before any code of this round; m3_prereg tag pp_r9)

**Weakness (measured, §32).** N4 carries start beyond the plate part's support. Under radial relocation the bottle's foot
touches down 4-10 cm before the plate's centre, and the bottle tips. N4 familiar success is 0.23 and its 3F D3 is −0.61.

**Principle.** A carry from farther out is the same carry with its horizontal plane stretched about the target: the same height
profile over the same fraction of the way. The place frame (stage 1c) already freezes the carry's bearing at the hold's onset.
This round also freezes its horizontal scale:

s_t = min(1, r_t(part) / d_onset)

Here d_onset is the horizontal object-to-target distance at the hold's onset, and r_t(part) is the activated target part's own
support radius (round 8).

**Why the carry and not the approach (round 7).** Round 7 failed because the grasp is not scale-equivariant: the hand-object
offset at the close was shrunk by s too (§29). At the place, the relevant precision is the release site's, and the plate's
calibrated radius is 0.08 m. Most training carries start inside their own part's support (§29's table), so s_t = 1 on almost
every demo row and the training data is nearly unchanged. Round 7's library-wide 0.124 m scaled a large share of the demos.

**Change (one, structural; on P_pp).**
- In the target term, the radial relocation of the carry (round 6, with round 8's per-part radius) is replaced by this frame:
  - input: the horizontal object-to-target offset × s_t, and the noisy chunk's horizontal translation × s_t, both in the
    canonical frame;
  - output: the horizontal translation velocity × 1/s_t.
- d_onset is captured at the hold's onset, as cyaw and rot0 are. On demo rows it comes from the first row of each hold, by the
  rollout's own rule.
- The object term (round 5's clip with round 8's per-part radius), Φ, the parts and the data are P_pp's.
- No teacher, data or per-scene input is added. s_t uses only the fitted part radius and the onset state.
- **Known limit.** Commands above 1 are clipped by the executor. For far new-layout carries (s_t ≈ 0.4) the scaled horizontal
  command may saturate. N4 familiar (s_t ≈ 0.6) is less exposed.

**Arm.** P_ts (rank 32), seeds 10-12, builds full, nobottle and noplate. Reference: P_pp, seeds 10-12 (pp_r8_roll).

**Reads.** Dev states 4-23, N2's seeds. S1, S2 and S3 with pp_l2's estimators and frozen GO mapping.

**Pre-registered round verdict (metric n4_fam).** Δ = N4's familiar success, P_ts − P_pp, state bootstrap with seeds pooled.
**improves** iff the CI's lower bound is > 0, the mandate's rule as in round 8.

## 34. Round 9 results (2026-11-03; m3_prereg pp_r9; results/flawed/pp_r9_read.json; rollouts pp_r9_roll)

**Round verdict: improves.** N4's familiar success is 0.40 for P_ts against 0.23 for P_pp: +0.17 [0.03, 0.30]. It also clears
rounds 2-7's stricter bar (> 0.10). **Layer-2 GO = false.** No rollout had an episode error.
- **The place recovers.** N4's familiar place-on-arrival is 0.73 (P_pp 0.29, P_rt 0.74). K1 familiar stays at 0.95.
- **The bottle's grasp is worse, although the object term did not change.** Familiar bottle hold is 0.62 (P_pp 0.87). The
  episodes that move no object rise to 0.15-0.17 (P_pp 0.03). K4 familiar falls to 0.53 (P_pp 0.82). See the open point below.
- S1 holds.
- **S2 familiar.** s known 0.76, s new 0.70, OM 1.09 [1.03, 1.16], which passes. 3F holds on N1, N2, N3, N5 and N6. It fails on
  N4: observed 0.40 against predicted 0.52, D3 −0.12 [−0.22, −0.03].
- **S2 new.** s known 0.36, s new 0.25, OM 1.06 [0.86, 1.33], which passes. 3F fails on N4 (+0.13), N5 (+0.14) and N6 (+0.09),
  where the observed rates exceed the near-zero predictions. The bottle's new-layout arrival on the rack is 0.00, which sets
  a_o low.
- **S3 passes all 6 familiar cells** (bottle 0.96 / 0.95 / 0.91, plate 0.94 / 0.90 / 0.94; lower bounds 0.82-0.94) **and none in
  the new layout.**

**Open point: the roles train jointly.** P_ts and P_pp have the same object term: the same inputs, the same radius and the
same seeds. Only the target term differs. The object term's loss has no target-term input either, because Φ gates on held.
Yet the bottle's hold differs (0.62 against 0.87). The one shared element of their training is the optimiser step:
`clip_grad_norm_` runs over both role networks together, so a change in the target term's gradients rescales the object
term's updates. This is a training-time coupling of the roles. The function is unaffected (S1 holds exactly).

**Check** (`experiments/pp_clip_diag.py`; results/flawed/pp_clip_diag.json). Two models that differ only in the target term
(P_pp's and P_ts's) start from the same seed and train on the same batches, on CPU with deterministic algorithms.
- With the joint clip, their object terms agree exactly for 50 steps. After 1,500 steps they differ by up to 6.5e-5 per
  parameter.
- With one clip per role network, they stay identical (difference 0.0) after 1,500 steps.
- The real seed-10 checkpoints (30,000 steps on GPU) have object terms whose difference is 1.40 times the parameter norm. A
  small early difference grows into a different network. GPU non-determinism may add to this; that was not separated.
- **Consequence for every round so far.** A change to one role retrains the other role's term into a different network. So
  per-object factors (the bottle's hold: 0.62-0.87 across rounds 6-9 with the same object-term design) move between arms for
  reasons outside the change. The state bootstrap treats the networks as fixed and does not cover this variance.
- The plan (§4.1) trains each role backbone once. One clip per role, or reusing an unchanged role's trained term, would
  remove this coupling. That is a candidate for round 10. It is not run here.

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 9 | N4: the bottle touches down before it arrives (§32) | carry scale frame frozen at the hold's onset | pp_r9 | +0.17 [0.03, 0.30]: improves | false |

## 35. Round 10 design note and protocol: one training per role (fixed 2026-11-03, before any code of this round; m3_prereg tag pp_r10)

**Weakness (measured, §34).** The roles train together. The joint `clip_grad_norm_` rescales the object term's update by the
target term's gradient, so a change to one role retrains the other role's term into a different network. P_pp and P_ts have
the same object-term design, but their seed-10 object terms differ by 1.40 times the parameter norm, and the bottle's familiar
hold moves from 0.87 to 0.62. Every between-round comparison of a per-object factor carries this variance, and the state
bootstrap does not cover it. A second point, found by reading `pp_p_train.py`: no torch random call precedes the model's
construction, so every build starts from torch's default seed. Seeds 10-12 share their initial parameters, and the seed sets
only the batches and the flow noise.

**Principle (plan §2 and §4.1).** Φ has no learned cross-role term, and each role backbone is trained once. Φ gates on held,
which is 0 or 1 on every row, so the loss already splits exactly into two sums:

L = (1/B) [ Σ over not-held rows of |v_o − u|² + Σ over held rows of |v_t − u|² ]

Each role's gradient therefore depends only on its own rows and its own term. AdamW, weight decay and the EMA act per
parameter. The joint norm clip is the only place where the two roles meet in training. With one clip per role, a role's
training is a function of its own design, its own rows and the seed, and nothing else. A role that an arm does not change then
comes out bitwise unchanged. So it can be trained once and reused, as §4.1 intends.

**Change (one, structural; default off; `pp_p_train.py --decouple`).** "Decoupled training" has three parts. None of them
changes the function class, the inputs, the data, the loss or Φ.
1. One `clip_grad_norm_` per role network, with the same threshold (1.0).
2. Each role network's initial parameters come from its own generator, seeded by the build seed and the role. A role's
   initialisation then depends neither on the other role's architecture nor on the construction order. The build seed now
   also sets the initial parameters.
3. Deterministic kernels: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and cuDNN's
   deterministic mode.

Reuse follows from parts 1-3 and is a saving, not a separate change. `--reuse obj:<tag>` (or `tgt:<tag>`) loads one role's
term from a decoupled build of the same seed: its parameters and its role buffers (radii, home). It then trains only the other
role, on the same batches and noise. The loaded role's configuration must match, which is asserted. The EMA then updates only
the trained parameters. Under parts 1-3 this gives the joint result bitwise. That is checked once below and assumed thereafter.

**What this does not do.** It adds no input, data, teacher or per-scene constant. It does not revisit the devices of rounds 5-9
(aframe, rclip, rtclip, psupport, tscale). Their between-arm comparisons were made under the coupling, so a later round that
removes or merges one of them is now measurable cleanly.

**Arms** (rank 32, seeds 10-12; builds by `submit_round_train.sh`'s convention).
- **P_tdc**: P_ts's flags plus `--decouple`. Builds full, nobottle and noplate, with the S3 creations. This is the new
  baseline.
- **P_pdc**: P_pp's flags plus `--decouple`, build full only. Its object-term design is P_tdc's, and only its target term
  differs (the radial carry clip instead of the carry scale frame).
- **Repeat**: P_tdc full, seed 10, built a second time in a separate job.
- **Reuse check**: P_pdc full, seed 10, built with `--reuse obj:full_tdc`.

**Reads.** Dev states 4-23, N2's seeds. P_tdc: S1, S2 and S3 with pp_l2's estimators and its frozen GO mapping. P_pdc: S2.

**Pre-registered round verdict (metric decouple).**
- **(a) Training decoupled.** For each seed 10-12, every object-term tensor of P_pdc (parameters `obj.*` and the object-role
  buffers `rclip`, `rclip_p`, `home`) equals P_tdc's bitwise.
- **(b) Per-object factors decoupled.** For each seed, both layouts and each object atom (bowl, cheese, bottle), the hold
  flag (held_step set) of every paired S2 episode agrees between P_pdc and P_tdc. So each per-object hold rate differs by
  exactly 0. Hold depends only on the steps before the hold's onset, where Φ uses the object term alone.
- **Verdict.** "decoupled" iff (a) and (b). "decoupled in training only" iff (a) holds and (b) does not; the remaining
  between-arm difference is then from the rollouts, and it is reported. "not decoupled" iff (a) fails.
- The round counts as a success for the failure streak iff the verdict is "decoupled". The improvement rule of rounds 2-9 does
  not apply, because this round targets a coupling, not a competence.

**Reported, with no verdict.**
- (c) Reproducibility: the repeat build against the first, whole state, bitwise.
- (d) Reuse: the reuse build against the jointly trained P_pdc, seed 10, whole state, bitwise.
- (e) Run-to-run variance of the per-object factors under P_tdc (hold; arrival given hold; place on arrival), per object and
  layout: the value per seed and the across-seed range and SD. It is set beside the between-arm range of rounds 6-9 (P_rt,
  P_hs, P_pp, P_ts; seeds 10-12) for the same factors.
- (f) P_tdc's GO read (pp_l2's mapping) and its per-pairing 3F, against P_ts (round 9), descriptively.

**Cost.** About 14 builds and 24 creations: roughly 4-5 A40-h, more if the deterministic kernels are slower.

## 36. Round 10 results (2026-11-03; m3_prereg pp_r10; results/flawed/pp_r10_decouple.json, pp_r10_read.json; rollouts pp_r10_roll)

**Round verdict (pre-registered mapping): "decoupled in training only".** The training part holds exactly. One paired
episode in 1,920 differs at the hold, and a replay shows that a host-to-host simulator difference causes it, not the models.
The round counts as a failure for the streak (streak 1).

- **(a) Training decoupled: holds.** For seeds 10, 11 and 12, all 22 object-term tensors of P_pdc equal P_tdc's bitwise
  (maximum difference 0.0). The two arms' target terms differ.
- **(b) Per-object hold: 1 of 1,920 paired episodes disagrees.** It is K4, new layout, seed 10, state 6: P_pdc holds at step
  106 and succeeds (rollout host oppy); P_tdc never holds (host gideon). The largest per-object hold difference is 0.025
  (the new-layout bottle at seed 10; pp_decouple_check reports 0.01 per object-and-layout cell over all its tasks).
- **The replay** (`experiments/pp_replay_one.py`; both checkpoints in one process on one host, bmo, twice each): both arms hold
  at step 106 with identical pre-hold chunks (the same SHA-256 over every pre-hold call), and both succeed. The disagreement is
  therefore a host difference in the simulator or CPU arithmetic. Fresh environments per episode do not make rollouts
  bit-reproducible across hosts. This applies to every earlier round too: each paired arm is rolled out on whichever hosts
  the scheduler gives.
- **(c) Reproducibility: holds.** The repeat build of P_tdc seed 10 (a separate job, on host conroy rather than protocol)
  equals the first bitwise in all 48 tensors.
- **(d) Reuse: holds.** P_pdc seed 10, built with `--reuse obj:full_tdc` (target term trained alone, on host heistotron),
  equals the jointly trained P_pdc bitwise in all 48 tensors. Later rounds can therefore train only the role they change.
- In the 300-step GPU smoke, two joint-clip builds with different target terms already differed in their object terms by up to
  0.53 per parameter. On GPU the joint clip and non-deterministic kernels together make the object term differ from step 1.

**(e) Run-to-run variance of the per-object factors** (`pp_decouple_check.py var`; per seed 10 / 11 / 12).

| factor | P_tdc (decoupled) | P_pdc (decoupled, same object term) | rounds 6-9 per arm (P_rt / P_hs / P_pp / P_ts) |
|---|---|---|---|
| bottle hold, familiar | 0.65 / 0.70 / 0.70 | 0.65 / 0.70 / 0.70 | 0.80-0.90 / 0.55-0.80 / 0.80-0.95 / 0.50-0.75 |
| bottle hold, new | 0.225 / 0.45 / 0.40 | 0.25 / 0.45 / 0.40 | 0.00-0.35 / 0.15 / 0.10-0.15 / 0.20-0.40 |
| cheese hold, new | 0.50 / 0.40 / 0.45 | identical | 0.45-0.50 / 0.25-0.40 / 0.40-0.45 / 0.35-0.45 |
| bowl hold, new | 0.90 / 0.55 / 0.80 | identical | 0.70-0.95 / 0.70-0.75 / 0.90-0.95 / 0.85-0.95 |

- Between arms that share the object term, every hold factor is identical, except the one host-difference episode. Before
  round 10, P_pp and P_ts, with the same object-term design, differed in familiar bottle hold by up to 0.35 at the same seed.
- **Across seeds the variance stays.** Under P_tdc, familiar bottle hold spans 0.05 and new-layout bottle hold 0.225, and
  new-layout bowl hold spans 0.35. This is seed variance of a network trained on 50 demos per task. Decoupling cannot remove it,
  and the state bootstrap does not cover it either.
- Factors that read the target term (arrival given hold, place on arrival) still differ between P_tdc and P_pdc, as they
  should. In the new layout, rack place-on-arrival is 0.00 under P_tdc at every seed and 0.50-1.00 under P_pdc, on few
  arrivals.
- The first-ever seed-dependent initialisation (part 2 of the change) means P_tdc is a new draw, not a re-run of P_ts.

**(f) P_tdc's GO read: Layer-2 GO = false.** No rollout had an episode error.
- S1 holds (both swap tests 0.0 on all three seeds).
- **S2 familiar.** s known 0.80, s new 0.75, OM 1.08 [1.03, 1.14], which passes. 3F holds on N1, N2, N3, N5 and N6. It fails on
  N4: observed 0.48 against predicted 0.61, D3 −0.13 [−0.19, −0.05]. P_ts: 0.40 against 0.52, −0.12 [−0.22, −0.03].
- **S2 new.** s known 0.33, s new 0.31, OM 1.22 [0.98, 1.54], which passes. 3F fails on N4 (+0.20), N5 (+0.18) and N6 (+0.13),
  where observed success exceeds the near-zero predictions. The bottle's a_o in the new layout pools K4's arrival (0.57) with
  K5's arrival at the rack (0.05).
- **S3 passes 7 of 12 cells**: all 6 familiar cells (bottle 0.92 / 0.89 / 0.93, plate 0.86 / 0.87 / 0.84; lower bounds
  0.78-0.88), and bottle-new at k = 10 (2.33 on rates of 0.02 and 0.04). The other new-layout cells fail.
- N4 familiar success, as a descriptive comparison with P_ts: 0.48 against 0.40, +0.08 [−0.08, 0.23].
- **P_tdc is the new baseline.** GO's blockers are the same as round 9's: familiar 3F on N4, and new-layout 3F on the bottle
  pairings.

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 10 | joint gradient clip couples the roles' training (§34) | one training per role (per-role clip, per-role seeded init, deterministic kernels; reuse) | pp_r10 | (a) exact; 1 of 1,920 holds differs from a host difference: "decoupled in training only" | false |

**Cost.** 3.9 A40-h: P_tdc 3.1 (with creations), P_pdc 0.4, repeat 0.15, reuse 0.25; the smoke run is under 0.02. Under the
deterministic kernels a build takes 8-23 min depending on the host (round 9: 14-15 min).

## 37. Round 11 diagnostic: the bottle part is trained on a target-dependent two-mode grasp, and the rounds stop here (2026-11-03; `experiments/pp_yaw_diag.py`, `pp_grasp_modes.py`, `pp_grasp_yaw_roll.py`)

**Target.** The bottle's grasp is the weakest object factor under P_tdc. Familiar hold is 0.65-0.70, new-layout hold
0.225-0.45, and the bottle pairings carry every remaining GO blocker. The starting point was §26's side finding: at the close,
the hand's rotation in the bottle's frame lies far outside the demo cloud.

**1. It is not a frame artifact** (`pp_yaw_diag.py`, demo rows of K1, K4, K5 and K6; closed-loop probes of P_tdc seed 10).
- In the demos each object's body yaw is nearly constant: circular SD 1.8° (K4), 4.2° (K5), 3.0° (K6) and 6.9° (K1). The
  bottle's body frame is therefore almost the world frame. At the probe the bottle stands at about 180° in every upright
  episode.
- At K4's first close, in 11 of the 20 familiar episodes the hand's rotation lies 1.06-1.69 (of a possible 2.8) from the
  nearest close row in the body frame, and about as far in the world frame. Familiar successes are among them. The approach
  frame does not bring it inside.
- So the hand has really turned, by about 60-80°, before the close. Re-framing the rotation input cannot remove that.

**2. The bottle's demos contain two grasps, one per target** (`pp_grasp_modes.py`; hand yaw modulo 180°, the gripper's
symmetry; 50 demos per task).

| task | hand yaw at the close (circular mean, SD) | turn from the start to the close (median; 10th-90th pct) |
|---|---|---|
| K1-K3 (bowl) | −4.0° to −0.8° (SD 4.1-5.3°) | 3.1-3.7° (0.1-12.4°) |
| K4 (bottle → cabinet top) | 2.1° (SD 3.1°) | 2.5° (0.4-6.3°) |
| **K5 (bottle → rack)** | **−89.0° (SD 13.7°)** | **81.2° (67.7-87.3°)** |
| K6 (cheese) | 1.1° (SD 3.4°) | 1.9° (0.2-5.5°) |

- To place the bottle in the rack, the K5 demos grasp it side-on: the hand turns by about 81° before it closes. K4 grasps it
  head-on, without a turn. The grasp depends on the target.
- The object term is target-blind by construction (S1). The bottle part is trained on both tasks' rows, so it learns a
  two-mode mixture over the hand's yaw.

**3. The closed loop mixes the modes** (`pp_grasp_yaw_roll.py`; P_tdc seed 10, dev states 4-23; the hand's yaw at the
hold's onset, and the largest yaw change between consecutive calls before it).
- **Familiar K4, 13 holds.** 3 are in the K4 mode (−3° to 22°), 9 in the K5 mode (74-106°), and 1 in between (49°). The part
  mostly delivers the rack's grasp, even for the cabinet.
- **Familiar K4, 7 episodes without a hold.** The largest yaw step before the hold has a median of 20° (11-87°). In the held
  episodes it is at most 14.4° (median 10.1°), and 5 of the 7 no-hold episodes exceed that. Hold failures go with the hand
  hopping between modes during the approach.
- **New layout K4: 5 holds,** all in the K5 mode or beyond (68-168°). The 15 no-hold episodes have a median largest step of 22°.
- Before the onset the bottle part acts alone. K5 and N4 give the same onset yaw in every familiar state (to 0.01°). K4
  matches them to within 0.2° on every state that holds, and differs on some no-hold ones. The mode mixture is therefore one
  part's competence, carried into every bottle pairing.
- On N4 (familiar), success does not separate cleanly by mode on these small numbers: 2 of 3 in the K4 mode, 5 of 9 in the K5
  mode.

**Why no round 11 is run (stop condition (d)).** The weakness is measured, but every change that would address it needs a
decision the plan does not settle, or breaks a rule:
- **(i) A target-aware grasp.** The grasp's yaw is a requirement of the target (the rack). Letting the object term read the
  target, or giving the target term the wrist yaw before the hold, is a cross-role term. Φ (§2) and S1 forbid it. Changing Φ
  per action dimension, for example handing the wrist yaw to the target role for the whole episode, is a plan-level decision.
- **(ii) Removing one mode from the data.** Training the bottle part's grasp without K5's rows, or relabelling K5's grasp, is
  per-scene data curation, which the mandate forbids.
- **(iii) Committing to one mode per episode.** Drawing the flow's noise once per episode instead of once per call would stop
  the mode hopping. That is a sampler change. It does not follow from P/F/S/B, frame equivariance or relocation, so it would
  be another device of the kind the user warned against. It would also leave the mode choice random. K4 would still get the
  rack's grasp in most episodes.
- **(iv) Accept it as a part-competence limit of the demos,** as §5.32's option (c) did for the near-rack cheese grasp. The
  frozen 3F would then be reported as failing on N4 (familiar) and on the bottle pairings (new), with this mechanism beside it.

The theory predicts this outcome. Composition makes a pairing as good as its parts. A target-blind part trained on a
target-dependent grasp is a mixture, and the product law then carries the mixture into every partner. That is what N4 to N6
show.

## 38. Round 11 design note and protocol: an explicit grasp-mode message from the target part (fixed 2026-11-03, before any code of this round; m3_prereg tag pp_r11)

**This round follows a plan change made after seeing data.** The user chose option (i) of §37, as PLAN_PARTS §10 item 5
(commit 87e9b02d): one explicit, sparse, declared interface between the roles. The claim becomes composition with a declared
interface of b bits, and b is reported.

**Weakness (measured, §37).** The bottle's demos grasp in two modes, one per target. K4 grasps with the hand unturned (2.1°),
and K5 turns it by about 81° before the close. The target-blind bottle part mixes the two modes. Under P_tdc it delivers the
rack's grasp in 9 of 13 familiar K4 holds and hops between modes during the approach. Familiar bottle hold is 0.65-0.70.

**Principle (§10.5).** A target may need a particular grasp. That requirement is the target's information, so it travels
from the target part to the object part as one discrete message m, the only cross-role channel. Its inventory is induced
from the demos, and each target part's value is fitted from its own demos. Nothing else crosses between the roles.

**Change (one; on P_tdc; default off; `pp_p_train.py --imode`).**
- **Feature.** One number per training demo: the hand's yaw relative to the object's body yaw, doubled (the parallel
  gripper is symmetric under a half turn), at the demo's first close row. A close row is a not-held row whose chunk's first
  gripper command closes. Demos without a close row contribute nothing.
- **Inventory rule (fixed here).**
  - Fit mixtures of von Mises distributions on that doubled angle over all demos of the build, for K = 1, 2, 3 and 4
    components. EM runs 300 iterations from a deterministic farthest-point initialisation that starts at the circular
    mean, with each κ capped at 1,000.
  - Choose K by BIC (3K − 1 parameters). Ties go to the smaller K.
  - Each demo's label is its most responsible component. Components are ordered by weight, so m = 0 is the most common.
- **A target part's value of m** is the most frequent label among its own demos (ties go to the lower value). A target atom
  without demos in the build gets m = 0, and no read uses it before its creation.
- **A created target part** (S3 plate) labels its own k demos with the base build's stored inventory and takes their most
  frequent label. A created object part (S3 bottle) reads m from the existing target parts.
- **The object term reads m** as a one-hot input of K entries, appended to its input. m is fixed per episode by the task's
  target part. The target term, Φ, the gate, the radii and every other input are P_tdc's.
- **Bandwidth.** The capacity is log2 K bits. Also reported: the entropy of m over the build's known tasks, with tasks
  weighted equally.
- No teacher, data, hand label or per-scene constant is added. The inventory and the values are fitted from demo rows.

**Training (decoupled, §35).** The object term gains an input, so it is retrained. The target term is unchanged in function
(m is a fitted value, not a network output), so each build reuses it with `--reuse tgt:<cond>_tdc`, which is bitwise what joint
training would give (§36 (d)). Creations are re-run: the bottle part trains against the new object term, and the plate part
trains as before and also fits its m.

**Arm.** P_imd (rank 32), seeds 10-12, builds full, nobottle and noplate, with the S3 creations. Reference: P_tdc (pp_r10_roll).

**S1, restated per §10.5.**
- **Holds iff,** with m held fixed (every target part's m set to the same value v, for each v in the build's inventory), both
  swap tests are exactly 0 and every target-into-grasp component is exactly 0 on every frame, for every seed.
- Reported beside it: the same profile with m free, which measures what flows through m.

**Reads.** Dev states 4-23, N2's seeds. S1 (restated), S2 and S3 with pp_l2's estimators and its frozen GO mapping, unchanged.

**Pre-registered round verdict (metric bottle_hold_fam).**
- The metric is familiar-layout hold pooled over the bottle's tasks K4, K5, N4, N5 and N6, seeds pooled.
- Δ = P_imd − P_tdc, with a 2,000-draw state bootstrap (the same states resampled for both arms).
- **improves** iff Δ > 0.10 and the CI's lower bound is > 0 (rounds 2-7's rule).

**Reported, with no verdict.**
- **Mode-hopping rate.** The share of familiar K4 and K5 episodes (seeds 10-12) whose largest pre-hold change of the hand's
  yaw (modulo 180°) between consecutive calls exceeds 20°, for P_imd and P_tdc. This uses `pp_grasp_yaw_roll.py`.
- **Mode delivery.** Among the held episodes of those runs, the share whose onset yaw lies nearest the mean of the task's own
  m.
- The inventory, each target's m, and the bandwidth, per build.

**Amendment to the feature (2026-11-03, after the code smoke; before any build, the pre-registration and any read).** The
code smoke (300-step builds) printed the inventory under the body-frame feature. Its largest component sat at 88° and held the
bowl's grasps: the bowl's body yaw in the demos is about 90°, and the bowl is round, so its body yaw is arbitrary. m thus
encoded the object's body yaw, not the target's requirement. It gave the plate and the stove the 88° value, which the
cheese part never sees in training (K6's target is the bowl), so N1 and N2 would read an unseen m. The feature is therefore
replaced by one that does not depend on any object's frame: **the turn of the hand's yaw from the demo's first row to its first
close row**, doubled. This is §37's own measure, which separates K5 (81°) from every other task (1.9-3.7°). Every other part of
the rule is unchanged. This is a change made after seeing the inventory, and it is labelled as such.

## 39. Round 11 results (2026-11-03; m3_prereg pp_r11; results/flawed/pp_r11_read.json, pp_r11_hop.json; rollouts pp_r11_roll)

**The induced interface** (same in all three seeds; pp_train_*_imd_s*.json).
- **Full build: K = 3 by BIC** (BIC 884 / 368 / 316 / 332 for K = 1-4). The components sit at a turn of −0.2° (weight 0.69),
  −11.5° (0.17) and −89.1° (0.14).
- The rack emits m = 2 (the turned grasp). The cabinet drawer emits m = 1. Every other target emits m = 0.
- Bandwidth: a capacity of log2 3 = 1.58 bits, and an entropy over the 8 known tasks of 1.06 bits.
- In the nobottle build (no K4 or K5), K = 2 and every target emits m = 0. That is 0 bits used.
- In the noplate build, K = 3 and the rack emits m = 1 (components ordered by weight).
- The created plate parts fit m = 0 at every k. Their own demos' labels are 0 except a few −12° turns at k ≥ 10.

**Round verdict: improves.** Familiar bottle hold (K4, K5, N4, N5, N6, seeds pooled) is 0.87 for P_imd against 0.68 for P_tdc:
+0.19 [0.003, 0.36]. **Layer-2 GO = false.** No rollout had an episode error.

**Mode hopping and delivery** (`pp_hop_read.py`; familiar, seeds 10-12, 60 episodes per cell).

| | hop rate K4 | hop rate K5 | delivery K4 | delivery K5 |
|---|---|---|---|---|
| P_tdc | 0.12 | 0.13 | 0.37 (of 41 held) | 0.63 (of 41) |
| P_imd | 0.08 | 0.02 | 0.75 (of 52) | 1.00 (of 53) |

The bottle part now delivers the grasp its target asks for. In K5 every hold is in the rack's mode.

- **S1 (restated, §10.5) holds.** With every target's m fixed at 0, 1 or 2, both swap tests are exactly 0 and every
  target-into-grasp component is exactly 0, on all three seeds. With m free, the target-into-grasp profile is non-zero, as
  it must be, so the unrestated S1 fails. That is the 1.58-bit channel.
- **S2 familiar now passes.** s known 0.88, s new 0.88, OM 1.11 [1.02, 1.21]. **3F holds on all six new pairings**, for the
  first time in any round: N1 −0.02, N2 0.00, N3 0.00, **N4 −0.075 [−0.17, 0.00]**, N5 −0.03, N6 −0.03.
  - K4 rises to 0.87 (P_tdc 0.63), N4 to 0.80 (0.48), N5 to 0.85 (0.63) and N6 to 0.75 (0.53).
  - K5 rises to 0.58 (0.28). Its hold is 0.88 and its arrival 1.00; place on arrival at the rack is 0.66.
- **S2 new fails, on N5 only.** s known 0.31, s new 0.22, OM 1.01 [0.85, 1.25], which passes. 3F holds on N1-N4 and N6; it fails
  on N5, +0.07 [0.009, 0.15].
  - **The bottle's new-layout grasp is worse.** Hold is 0.23 (P_tdc 0.37), the first hold comes at a median step of 150-159
    (P_tdc 114), and 20% of episodes move no object.
  - K4 new falls to 0.10 (0.20), and N4-N6 new are 0.05-0.12.
- **S3 passes all 6 familiar cells and none in the new layout.**
  - Familiar: bottle 0.90 / 0.96 / 0.89, plate 0.88 / 0.84 / 0.88; lower bounds 0.75-0.92.
  - New: the bottle's rates are 0.00-0.08. The plate's ratios are 0.60-0.64, with lower bounds 0.25-0.38. Its unseen partner N4 (the
    bottle on the plate) succeeds at 0.02-0.07 in the new layout, while N1 runs at 0.28-0.38.

**What is left for GO.** Everything that fails now sits in the new layout, and almost all of it on the bottle's new-layout
grasp:
- N5's 3F deviation, on near-zero predictions;
- the bottle's S3 cells, at rates near 0;
- the plate's S3 cells, through N4.

| round | weakness | change | prereg | round verdict | GO |
|---|---|---|---|---|---|
| 11 | the bottle part mixes a target-dependent two-mode grasp (§37) | explicit grasp-mode message m from the target part (§10.5), 1.58 bits | pp_r11 | +0.19 [0.003, 0.36]: improves | false |

The failure streak resets to 0. P_imd is the new best arm.

**Cost.** 2.1 A40-h (the object term is retrained, the target terms are reused) plus a 0.06 A40-h smoke run.

# The Flawed Composition of VLAs

Working document, started 2026-10-29. It lays out the problem, how prior work has attacked it, and how we plan to diagnose
and resolve it in VLA architectures. It supersedes `docs/history/2026-10-29-vla-operator-binding/THEORY_FRAMING.md` and
`PLAN_BINDVLA.md`; those remain the record of the evidence cited here.

Citation status: arXiv IDs marked (v) were read or seen in search results on 2026-10-29. Everything else is cited from
memory and is marked (check) until verified against the source.

---

## 0. One-paragraph statement

A VLA is a chain of stages. Perception turns pixels into features, binding maps each phrase to a scene entity, and the
action expert turns what was bound into motion. A new task built from familiar parts is solved only if **every** stage
treats the new combination as a combination: its inputs about each part must be uncontaminated by the other parts, and its
way of combining them must be the one it learned on other combinations. Current VLAs fail this at more than one interface,
and fixing one interface moves the failure to the next. We call the per-interface failure **compositional debt**. We
measure it inside the network with counterfactual interventions, predict it from the training split before training, and
remove it with structure that makes the debt zero by construction, chosen without supervision.

---

## 1. The problem

### 1.1 What "composition" should buy a VLA

Training covers tasks such as "put the bowl on the plate" and "put the mug in the basket". A compositional policy should
also solve "put the mug on the plate": every atom (mug, plate, put-on) was seen, only the combination is new. Keysers et
al.'s MCD split makes this precise (check: ICLR 2020). Hold atom divergence near zero (every test atom appears in training)
and vary compound divergence (how unfamiliar the combinations are). A compositional learner's error stays flat in compound
divergence. A holistic learner's error rises.

### 1.2 What actually happens (our evidence)

All numbers come from this repo; the commits and docs are in `docs/history/2026-10-29-vla-operator-binding/`.

| finding | number | source |
|---|---|---|
| Binding error rises with referring-expression depth | 0.18 / 0.38 / 0.945 at depth 0 / 1 / 2 | E4 |
| Off-the-shelf grounding fails held-out depth 1 | Qwen, Molmo 0.00; pi0.5 no usable grounding | B0 |
| LIBERO's left/right is mirrored relative to the policy image | corr(col, y) = −0.99; VLM depth-1 binding 0.60–0.67 after rescoring | B0 |
| Holistic binding error rises with compound divergence, factorised does not | slope 1.07 [0.65, 1.46] vs 0.21; interaction 0.86 [0.33, 1.37] | B1.3, a55f6c99 |
| A transfer-scored criterion recovers the shape bias; a fit-scored one does not | nouns → appearance, relations → geometry under prequential MDL; B1.2 identity-coding MDL fails all bars | B1.2, B1.3 |
| In distribution, success factorises | success = P(binding) × competence, calibration slope 0.997 | E2b |
| An oracle binding fixes binding errors and leaves motion errors alone | binding error 0.234 → 0.048 | S0 |
| BindVLA's slots bind, including on held-out combinations | 0.98 / 0.89 held-out episodes; 0.73 held-out layouts; 0.72 held-out compounds, flat across compound divergence | main2, 7ffe600c |
| The slots stay bound in closed loop | object slot on the named object 0.81 at start, 0.72 at grasp | 12.4, 6c46ddd2 |
| **Success does not follow binding** | predicted 0.44, observed 0.03; dev success 0.041 vs plain SmolVLA 0.007, +0.034 [−0.003, 0.083] | main2 |
| **About half the failures have the slot on the right object and the gripper elsewhere** | motor 48%, no grasp 18%, drift 16%, start miss 13%, place 4% | 12.4 |
| **Removing scene information from the motor moves the failure** | C1: named-object contact 49% → 58%, habit-object contact 21% → 7%, place failures 4% → 12%, training-task success 0.705 → 0.441 | C1, 5bf63e2c |

The last three rows are the observations this document is built on. Binding was repaired and composition still failed.
The failure moved into the action expert. Taking information away from the expert moved it again.

### 1.3 The general problem

Prior VLA work treats the composition failure as a failure at one place: the referent (ReGuide), or the
instruction-to-trajectory shortcut (ECT). Our data says the failure is a property of the chain:

> A policy composes only if every interface passes role-pure messages, every stage combines them by a rule identified on
> the training combinations, and no route carries compound identity around the stages that are meant to factor it.

Section 3 makes this precise. Sections 4 and 5 give the diagnosis and the fixes.

---

## 2. How prior work has attacked it

Each line of work below fixes one cell of the taxonomy in Section 3.5: one interface, with one lever. The table says which.

### 2.1 Classical and theoretical accounts of compositionality

- **Frege's principle; Fodor & Pylyshyn (check: Cognition 1988).** The meaning of a whole is a function of its parts'
  meanings and their mode of combination. Systematicity: a system that represents aRb can represent bRa. These give the
  homomorphism view that Section 3 adopts.
- **Hupkes et al., "Compositionality decomposed" (check: JAIR 2020).** Five testable properties: systematicity,
  productivity, substitutivity, localism, overgeneralisation. We make localism graded and per-interface.
- **Andreas, Tree Reconstruction Error (check: ICLR 2019).** Measures whether one representation decomposes into parts.
  It is a single-map measure, and it cannot see a downstream stage that ignores a decomposed representation.
- **Wiedemer et al., "Compositional generalization from first principles" (check: NeurIPS 2023).** For a generator
  x = C(φ₁(z₁), …, φ_K(z_K)) with a known composition function C, compositional support in training is sufficient for
  generalisation. **This is the closest formal precedent.** Their condition is our per-stage condition for a single stage
  whose composition rule is given. We add: chains of stages, bypass routes, debt relocation, measurement in trained
  networks, closed loop.
- **Lippl & Stachenfeld, kernel theory of compositional generalisation (check: 2024).** Kernel models generalise only to
  tasks that are additive over conjunctions seen in training. This is a precise statement of why a holistic stage fails on
  a new joint input. It predicts our action-expert failure if the expert behaves like a kernel machine over its joint input.
- **Elmoznino et al., complexity-based theory of compositionality (check: 2024).** Compositionality as compressibility
  of a representation. It relates to our acquisition argument (Section 3.4), but is stated for representations, not for
  multi-stage policies.
- **Okawa et al., multiplicative emergence of compositional abilities (2310.09336, check).** Success on a composite is the
  product of component competences. This is our in-distribution factorisation. We find it breaks out of distribution, and
  explain why.
- **Lake & Baroni, SCAN (check: ICML 2018); Bahdanau et al., "Systematic generalization: what is required and can it be
  learned?" (check: ICLR 2019).** Seq2seq models fail systematic splits. In Bahdanau et al., module layouts given by
  structure generalise where learned layouts do not. This is evidence for the structural lever.
- **Greff et al., "On the binding problem in artificial neural networks" (check: 2020).** Frames segregation,
  representation and composition as the binding problem. Our stage 1 is their binding problem. Our point is that solving
  it is necessary, not sufficient.
- **Our own recoverability paper (`compositionality-theory/`).** Compositionality as role-factorised recovery of fresh
  parts. It covers stage 1. This document extends it to every later stage.

### 2.2 Causal and invariance tools

- **Causal abstraction and interchange interventions (Geiger et al., check: NeurIPS 2021; interchange intervention
  training, check: ICML 2022).** Swap a variable's value between two inputs and check that the output follows the
  high-level model. **This is our measurement tool, and the closest prior art for the per-interface framing.** The novelty
  has to be in what we measure with it (compositional debt in closed-loop policies, predicted from the split), not in the
  tool.
- **ICP (Peters et al., check: 2016), IRM (Arjovsky et al., check: 2019), V-REx (Krueger et al., check: 2021).**
  Invariance across environments as an identification principle. We use layouts as environments. V-REx at λ = 1 did not
  survive our seed check (C5).
- **Independent mechanisms and modular networks (Parascandolo et al., check: 2018; Goyal et al. RIMs, check: 2021).**
  Learned modularity. In general it does not guarantee that the modules factor along the task's roles.

### 2.3 Structure built into robot policies

- **Transporter Networks (check: Zeng et al. 2020).** Pick and place as cross-correlation in image space, exactly
  translation-equivariant. Generalises to new positions by construction.
- **Neural Descriptor Fields (check: Simeonov et al. 2022); Equivariant Diffusion Policy (check: Wang et al. 2024);
  EquiBot (check: Yang et al. 2024); ET-SEED (check: 2024).** SE(3) equivariance or object-frame action generation.
- **What they share with us:** structure that makes a generalisation hold by construction.
- **What they lack:** language roles, a rule for which structure to build, and an account of what the structure costs.

### 2.4 VLA composition and binding failures (the immediate competitors)

| work | claim | interface | lever | what we add |
|---|---|---|---|---|
| ReGuide (2609.38616, v) | Failures on new object–destination pairs come from binding the wrong referent; the skill is intact. Training-free rebinding plus geometric guidance on frozen pi0.5 | language → referent | external structure (grounder + poses) | our closed loop shows the skill is **not** intact once the referent is right |
| ECT (2609.39971, v) | "Instruction-action binding": the instruction recalls a trajectory family, vision only adjusts it. Narrow action support makes grounded and instruction-keyed solutions indistinguishable. Fix: counterfactual demo pairs plus a paired loss. LIBERO-PRO position swap 36 → 59; UR5e 8 → 88 | instruction → action (a bypass) | variation | ECT is the variation lever at one interface; we measure every interface and predict which needs variation |
| LIBERO-OOD / text-latent interpolation (2505.03500, v) | Skills exist and compose when the text latent is interpolated by hand (pi0 9 → 83) | language interface | hand-built message | a hand-built input does not show which stage is indebted |
| ACT-VLA (2607.00351, check) | interpolate, roll out, filter, fine-tune on synthetic compositions | language interface | variation (synthetic) | same |
| Grounded Semantic Re-Binding (2608.02497, v) | paraphrase failures from joint vision–text encoding shifts; decoupled semantics plus a re-initialised action expert | instruction encoding | structure (decoupling) | paraphrase robustness, not new combinations |
| SlotFlow (2609.24155, v; CoRL 2026) | what/where split for flow-matching policies | perception → action | structure | no roles, no composition evaluation |
| ObjDream (2610.20223, v) | object slots plus identity-guided routing in a VLA | perception → VLM | structure | no bottleneck, no composition evaluation |
| POT-VLA (2607.18016, v) | role-indexed 3D object tokens fed to the action expert | binding → action | structure | closest to our role tokens; no factorised combination rule |
| LIBERO-PRO (2510.03827, v), LIBERO-Plus (2510.13626, v), CRAFT (2610.00524, check) | VLAs memorise layouts and largely ignore language | benchmarks | — | these motivate the problem; they do not localise it |

**Summary of the field.** Every published fix acts on one interface: the referent, the instruction shortcut, or the
language latent. Each implicitly assumes the remaining stages compose. None measures where the debt is before choosing a
fix. None predicts from the training split which interfaces will fail.

---

## 3. Theory

### 3.1 Setup

- A task instance is x = (c, z).
  - c = (a₁, …, a_R) is the **compound**: one atom per role r (verb, object, goal, and modifiers at deeper levels).
  - z is the **nuisance**: layout, start state, distractors.
- Training covers a set T of compounds. Test compounds c* ∉ T have all their atoms in T (atom divergence ≈ 0).
- The policy is a directed graph of stages S₁ … S_K. Stage k reads messages and writes a message m_k.
- A **declared interface** splits a message into role parts and a shared part: m_k = (m_k¹, …, m_k^R, m_k⁰). Examples:
  BindVLA's slot tokens are m^r, and the state and global token are m⁰. In a VLA with no declared interface, we impose one
  (Section 4.1).
- Any path from the inputs (o, ℓ) to stage k that avoids its declared input is a **bypass**.

### 3.2 Three conditions for composition at an interface

**(P) Purity.** Each role message depends on its own atom and the nuisance, not on the other atoms:
m_k^r = g_k^r(a_r, z), so m_k^r ⫫ a_{−r} | a_r, z.

**(F) Factored combination.** The next stage combines role messages by a rule with shared per-role components:
S_{k+1}(m) = Φ(h¹(m¹), …, h^R(m^R), m⁰), with Φ from a family identified on T (Section 3.4). A holistic stage computes
an arbitrary function of the joint message. On a new joint value it is extrapolating, even when every m^r is pure.
This is Wiedemer's condition and Lippl & Stachenfeld's conjunction-additivity, applied to one stage.

**(S) Sufficiency.** The role messages carry what the next stage needs about each atom. There exists a Φ meeting (F) that
is correct on all compounds.

**(B) No bypass.** No input to stage k+1 outside its declared messages carries information about c beyond what the role
messages carry.

The policy composes on c* if (P), (F), (S) and (B) hold at every interface, and each stage is competent on the role
values it receives. In that case success on c* is the product of per-role competences: Okawa's multiplicativity, now a
consequence.

### 3.3 Debt

Each condition has a graded violation, measured by interventions (Section 4):

- **Purity debt** δ^P_{k,r}: how much m_k^r moves when other roles' atoms change while a_r and z stay fixed.
- **Combination debt** δ^F_k: the error of stage k+1 when its role messages are interchanged to form a joint message never
  seen in training (role r's message from compound c, role s's from c′), scored against the correct behaviour for the
  recombined compound.
- **Bypass debt** δ^B_k: how much behaviour changes when only the bypass inputs are interchanged between compounds.
- **Insufficiency** shows up as a stage that is incompetent even on in-support joint messages once its bypasses are cut.

**Proposition 1 (decomposition; to be stated and proved).** The gap between predicted success (binding × competence) and
observed success on c* is bounded by a sensitivity-weighted sum of the debts along the chain. The bound itself is standard
error propagation. Its use is that every term can be measured.

**Proposition 2 (relocation; to be stated and proved).** Suppose stage k+1 needs information I about atom a_r, and m_k^r
lacks I. Then any parameter setting that is competent on T must take a proxy for I from another input that is correlated
with a_r on T. That input is another role's message (purity debt), the shared message or a bypass (bypass debt), or the
joint message (combination debt). Removing information without adding a pure route for it relocates the debt; it does not
remove it.
- **Evidence:** C1 removed the expert's scene features. Aim improved, but grasp and place failures rose, and training
  competence fell from 0.705 to 0.441.

### 3.4 Acquisition: why trained VLAs end up indebted

- **Observational equivalence.**
  - On the training support, a factored solution and a holistic one can reach the same imitation loss.
  - In LIBERO, and in most demo datasets, each compound comes with a narrow set of layouts. So compound identity, layout
    and trajectory are confounded.
  - ECT gives this argument for its one interface. Our B1 series shows it for binding: "left" was learned as a remembered
    image location (B1, R² 0.72).
- **Fit versus transfer.**
  - A learner scored by fit on T has no reason to prefer the factored solution.
  - When the confound is strong, the holistic code ("this compound → this trajectory") is often the shorter description.
  - A criterion scored by transfer, such as prequential leave-layout-out MDL, prefers the factored code, because the
    holistic one fails on the held-out layouts.
  - **Evidence:** B1.2's fit-scored MDL put nouns into geometry and transfer collapsed. B1.3's transfer-scored criterion
    routed words correctly and learned the mirror unaided.
- **A support condition per interface (to be derived).** Stage k+1 can be identified as factored only if:
  - T, projected onto that stage's roles, is connected (Wiedemer's compositional support);
  - each atom is seen with enough varied partners and nuisances to break the confound.

  This can be computed from the split alone, before training.

  **Prediction:** the interfaces whose projected support is weakest carry the most debt.

### 3.5 Taxonomy of fixes

| debt | lever | examples | cost |
|---|---|---|---|
| purity | structure: role-local queries, position-free keys, nested part masks | BindVLA binder | features may still leak nonlinearly (MLP probe R² 0.96) |
| purity | variation plus a transfer-scored criterion | B1.3 MDL; layout diversity | needs varied layouts per atom |
| combination | structure: a factored stage (per-role phases, a fixed combination rule) plus frame equivariance for the nuisance | Transporter, NDF and Equivariant Diffusion Policy, for pose | wrong wherever the task violates the factorisation |
| combination | variation on the projected support | ECT counterfactual pairs | demo collection |
| bypass | cut the route (bottleneck) | BindVLA bottleneck | insufficiency, if the cut route carried needed information (C1) |
| bypass | variation on the bypass | ECT | demos |
| insufficiency | enrich the role message | per-object grasp features in the slot | must stay pure |

The theory's job is to say which lever each interface needs, and to predict, before running, what each fix will buy.

---

## 4. Diagnosis: measuring debt in VLA architectures

### 4.1 Declaring the stage graph

| model | interfaces we declare |
|---|---|
| BindVLA (ours) | VLM features → binder; binder slots → bottleneck → action expert; bypasses: state, global token, the nonlinear position content of features |
| SmolVLA, pi0, pi0.5, X-VLA, GR00T N1.5 | prefix (vision + language) KV → action expert; we read role content from the KV entries of the referent's image tokens and the instruction's role words, using the KV-level interventions already built for S0 |

### 4.2 Counterfactual instances

- Use the procedural LIBERO layouts from B1.3 (4 table types, held-out compounds × layouts) and the BDDL generator.
- From them, build matched pairs that change exactly one thing: one role's atom, the nuisance, or the instruction alone.
- Demos are fitted or scored only, never executed.

### 4.3 Measurements

1. **Purity debt.** Interchange other roles' atoms. Measure the change in m^r (normalised distance, and probe accuracy for
   a_{−r} given a_r).
2. **Combination debt.** Build interchanged joint messages. Score the action expert's output against demos of the
   recombined compound, using the dev demos (scored only). Report flow loss and the grasp-site and release-site error.
3. **Bypass debt.** Interchange only the bypass inputs (state, global token, instruction tokens outside the roles). Measure
   the change in output.
4. **Insufficiency.** Cut the bypasses and measure in-support competence.
5. **Decodable versus used.** Report probes and interventions side by side. A probe shows information is present; only an
   intervention shows it is used.

### 4.4 Predictions to pre-register

| ID | prediction | falsified if |
|---|---|---|
| D1a | In main2, combination debt at the motor is high and purity debt at binding is low | combination debt ≤ purity debt |
| D1b | Per-task combination debt predicts per-task held-out failure across the 29 dev tasks | rank correlation CI includes 0 |
| D1c | C1 has lower bypass debt and higher insufficiency than main2 (relocation) | both fall, or neither changes |
| D2 | Across the five archive VLAs, per-task debt predicts per-task OOD failure better than compound divergence alone | no gain over compound divergence |
| D3 | In the B1 binders, the factored code wins beyond a contexts-per-atom threshold, and the threshold is predicted by the MDL arithmetic | no threshold, or a threshold off by more than 2× |
| D0 | Interfaces with the weakest projected support (from the split, before training) carry the highest debt | rank correlation CI includes 0 |

### 4.5 D0/D1 protocol (fixed 2026-10-29, before any D1 read; m3_prereg tag pd01)

No training. Three checkpoints: main2 (results/bindvla/main2, 60k), C1 (results/bindvla/c1main, 20k, the checkpoint of
PLAN_BINDVLA 12.5) and vanilla (results/bindvla/vanilla, 60k). Scripts: `experiments/fc_d0_support.py`, `fc_d1_frames.py`,
`fc_d1_debt.py`, `fc_d1_read.py`.

**D0 (computed from the split alone, before this protocol was fixed; results/flawed/d0_support.json).**
- Atoms and compounds are the split's own (`c_split.steps / compounds`: verb, object type, target with region, scene).
- Per dev task step (v, o, t) in scene s, over the 76 training tasks: the varied-partner counts P_o, P_t (targets seen with o,
  objects seen with t), the layout counts L_o, L_t (scenes in which the atom appears), and whether (v, o), (o, t), (o, s),
  (t, s) occur in training.
- Predicted debt: the components 1 − seen(v,o), 1 − seen(o,t), 1 − seen(o,s), 1 − seen(t,s), −log(1+P_o), −log(1+P_t),
  −log(1+L_o), −log(1+L_t) are averaged over the task's steps, ranked across the 29 dev tasks, and averaged. The resulting
  rank (1 = most debt) is recorded in the pre-registration row. Binder and motor sub-scores are reported, not tested.
- D0 predicted order, most debt first (ties in score keep the split's order): libero_90:44; the caddy book tasks :73, :75,
  :77, :78, :79, :80, :82, :83; :32, :58, :76, :38, :45, :40, :11, :59, :36, :7, :22, :61, :1, :29, :0, :50, :9, :16,
  :60, :14. The ranking mostly follows the part set first and then the CD bins. Its additions over the bin are the partner
  and layout counts.

**Frames.** Each frame is scored at the training-referent events (b1_render.events on the gripper command; b1_bind.transports
drops regrasps): e − 10, e − 5 and e for the first two grasps and releases. An episode with no closed-gripper run (a drawer
pushed shut) is scored at 0.25, 0.5 and 0.75 of its length.
- IN: the held-out episodes of the training tasks (2 per task, never trained on), from the training cache.
- DEV: the 10 demos per dev task that bv_devrefs used, passed through the cache's image pipeline. They are scored only:
  never trained on, never executed.
- In-support set: the 72 training tasks that all three models trained on (C1's 4 held-out-layout tasks are excluded). The
  combination comparator uses its 43 LIBERO-90 cells, which share the demo source and scene families of the dev tasks.

**Predictions and errors.**
- Every prediction is the model's own 10-step Euler sample from 2 fixed noise draws, seeded by the frame's identity, so
  every model and every variant sees the same noise.
- Action error is the MSE over the first 10 chunk steps and the 7 action dims, in the cache's MEAN_STD units.
- The output change of an intervention is the same MSE between the intervened sample and the intact one.

**Interfaces.**
- BindVLA: the role messages are the binder's slot tokens. The bypass is the global token (main2 only) and the state token.
- vanilla: the role messages are the instruction tokens and the agentview connector cells within Chebyshev 1 (8 × 8 grid)
  of the step-0 contact-event referents. The bypass is the other agentview cells, the wrist cells and the state token.

**Estimators.**
- **Purity debt δ^P.** On IN frames, with the image fixed, step 0's goal phrase is replaced by a goal phrase of another training
  task of the same scene, and separately its object phrase is replaced by the object phrase of another transport task of
  the same scene. A frame is scored only when both swaps exist and each replaced phrase occurs once in the instruction (on
  the training data this leaves 345 frames of 29 tasks). For the object message, the ratio is
  Σ‖Δm‖ under the goal swap / Σ‖Δm‖ under the object swap, per task; the goal message is symmetric. δ^P is the mean of the
  two ratios over tasks.
  - Levels: `tok`, the binder's output token (BindVLA only; the primary level for D1a); `kv`, the value states the expert
    reads (16 layers), at the slot position (BindVLA) or averaged over the role phrase's tokens (vanilla).
  - Probe (descriptive): nearest-centroid decoding of the other role's atom from this role's intact message, across held-out
    episodes, within groups of training tasks that share this role's atom.
- **Combination debt δ^F.**
  - E_dev(d) is the mean intact action error on dev task d's frames. E_in is the mean over the 43 in-support LIBERO-90 cells
    of their intact error on held-out episodes.
  - δ^F_rel = (mean_d E_dev − E_in) / E_in, with dev and in-support tasks bootstrapped independently. Per task,
    δ^F(d) = E_dev(d) − E_in.
  - For BindVLA the slot tokens of a dev frame are computed role by role. The interchanged joint message ("object token
    from one compound, goal token from another") is therefore the dev forward itself; the purity check verifies this
    numerically.
  - Secondary scores: grasp-only and release-only frames, the fixed-(t, noise) flow loss, and the summed-position site error.
- **Bypass debt δ^B.** Each IN frame is paired with a frame of another training task of the same scene, at the same phase
  fraction of that task's first held-out episode (fixed hash).
  - Role swap: the partner's instruction on this frame.
  - Bypass swap: the partner's bypass inputs, with the role messages kept.
  - δ^B per task = ΣΔ_byp / (ΣΔ_byp + ΣΔ_role), averaged over tasks.
  - Also reported: the state-only swap (all models) and the global-only swap (main2).
- **Insufficiency.** The bypass is mean-ablated over IN frames (main2: global token; vanilla: non-interface cells; C1 has
  nothing to cut). I is the in-support action error with the bypass cut, averaged over tasks. Also reported: the state
  mean-ablated as well, and I − E_in.
- All CIs are 2,000-draw percentile bootstraps over tasks.

**Predictions and falsifiers.**

| ID | test | holds if | falsified if |
|---|---|---|---|
| D1a | main2: δ^F_rel vs δ^P (tok) | lower CI bound of δ^F_rel > upper CI bound of δ^P | δ^F_rel ≤ δ^P (point estimates); anything else is inconclusive |
| D1b | main2: Spearman ρ(δ^F(d), dev failure(d)) over the 29 tasks; dev failure = 1 − bv_dev success | ρ > 0 and CI excludes 0 | CI includes 0, or ρ < 0 |
| D1c | paired over in-support tasks: dB = δ^B_C1 − δ^B_main2, dI = I_C1 − I_main2 | CI(dB) < 0 and CI(dI) > 0 | both fall (CI(dB) < 0 and CI(dI) < 0), or neither changes (both CIs include 0); anything else is partial |
| D0 | Spearman ρ(D0 predicted-debt score, δ^F_main2(d)) over the 29 tasks | ρ > 0 and CI excludes 0 | CI includes 0, or ρ < 0 |

- Secondary, not tested: the same correlations for C1 and vanilla; failure scored as "neither success nor first contact on
  the named object" (12.4 logs, main2 and C1); D0 sub-scores and the CD bin alone against δ^F.
- **Known weaknesses, stated before the read.**
  - D1a compares two dimensionless ratios of different quantities.
  - For BindVLA the binder output is role-pure by construction (per-role queries and slot MLP), so δ^P(tok) is expected to
    be 0. That part of D1a tests the construction, not a learned property. The kv level, where the frozen text layers mix
    the slots, is the learned one.
  - Dev failure has heavy ties: main2 succeeds on only 3 of the 29 tasks. D1b has little power, and the named-contact
    failure is reported for that reason.
  - E_dev − E_in also contains layout and demo difficulty, not only combination.
- **Validation before the read.**
  - The dev frames went through the cache's pipeline with zero pixel difference on 4 training frames.
  - A `--limit` smoke of `fc_d1_debt.py` (60 IN and 60 DEV frames per model) and of `fc_d1_read.py` checked the code paths.
    Its numbers were written to scratch and not inspected.
  - Two fixes were made before the pre-registration, on training data only: episodes without a gripper event get the
    fixed-fraction frames, and object swaps are restricted to transport tasks (the first swap check produced instructions
    such as "close the black bowl").

### 4.6 D0/D1 results (2026-10-29; ledger fc_d1 tag d01; results/flawed/d1_read.json)

- Frames: 888 IN frames over 76 training tasks and 1,386 DEV frames over 290 demos of the 29 dev tasks.
- Purity frames: 345, from 30 in-support tasks (22 for the goal token).
- Bypass pairs: 63 tasks. Cost: 0.30 A40-h.

**Debt table** (task-bootstrap 95% CIs; action error in normalised units, first 10 steps).

| | main2 | C1 | vanilla |
|---|---|---|---|
| purity δ^P, binder output (tok) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | no binder |
| purity δ^P, expert-read values (kv): object / goal / mean | 0.33 / 0.62 / 0.48 [0.39, 0.57] | 0.80 / 0.57 / 0.69 [0.65, 0.72] | 0.14 / 0.14 / 0.14 [0.13, 0.15] |
| probe: goal atom from the object message, tok / kv (chance 0.32) | 0.45 / 0.57 | 0.44 / 0.65 | – / 0.70 (chance 0.31) |
| probe: object atom from the goal message, tok / kv (chance 0.28) | 0.57 / 0.69 | 0.50 / 0.72 | – / 0.79 |
| combination: E_in (43 cells) → E_dev (29 tasks) | 0.49 → 1.09 | 0.60 → 1.17 | 0.48 → 1.02 |
| combination δ^F_rel | **1.21** [0.55, 2.15] | 0.95 [0.42, 1.72] | 1.13 [0.56, 1.96] |
| δ^F_rel on grasp / release frames | 0.36 [0.13, 0.63] / 0.99 [0.49, 1.61] | 0.39 / 0.83 | 0.48 / 1.12 |
| δ^F_rel, flow loss / site error | 2.20 / 0.81 | 1.25 / 0.50 | 2.17 / 1.02 |
| bypass δ^B (full bypass swap) | 0.58 [0.54, 0.63] | 0.59 [0.54, 0.65] (state only) | 0.65 [0.62, 0.69] |
| bypass, state-only / global-only swap | 0.57 / 0.18 | 0.59 / – | 0.59 / – |
| raw output change: role / bypass / state / global swap | 0.61 / 0.75 / 0.72 / 0.09 | 0.52 / 0.66 / 0.66 / – | 0.53 / 0.91 / 0.72 / – |
| in-support error: intact → bypass cut (I) → cut + state mean | 0.495 → **0.524** → 1.24 | 0.594 → **0.594** → 1.42 | 0.481 → **0.680** → 1.47 |

**Verdicts against the pre-registered falsifiers.**
- **D1a holds, but only as a test of the construction.**
  - Combination debt is δ^F_rel = 1.21 [0.55, 2.15]. Binder-output purity debt is exactly 0, as the per-role binder
    guarantees.
  - At the level the expert reads, after the frozen text layers mix the slots, purity debt is 0.48 [0.39, 0.57]. Against
    that level the comparison is inconclusive: the lower bound 0.55 overlaps 0.57.
  - Over half of the summed excess error comes from three drawer tasks (:22 at 4.7, :7 at 2.5, :11 at 2.2). The median
    per-task excess is 0.24, which is 0.49 relative.
- **D1b holds for main2.**
  - Spearman ρ(δ^F, dev failure) = 0.36 [0.07, 0.58] (95 of the 2,000 draws were degenerate). The secondary
    named-contact failure gives 0.39 [0.03, 0.66].
  - The result is fragile. Failure is tied at 1.0 on 26 of the 29 tasks, so ρ mostly says that main2's three solved tasks
    (:14, :40, :60) have low combination debt: :40 has the lowest (−0.08), and :60 is 24th of 29.
  - The D0 split score predicts the same failure about as well (ρ = 0.37 [0.00, 0.62], secondary). So D1b does not show
    that the measured debt adds information beyond task novelty.
  - Not significant for C1 (0.19 [−0.08, 0.44]) or vanilla (0.13 [−0.25, 0.53]). Their failures are almost all ties.
- **D1c is partial: insufficiency rose, bypass debt did not fall.**
  - dI = I_C1 − I_main2 = +0.070 [0.042, 0.108]. dB = +0.011 [−0.018, 0.041], which includes 0.
  - Removing the global token did not lower the share of output change that comes through the bypass. In main2 the
    global token carries little (global-only share 0.18, raw change 0.09). The state carries the bypass in both models,
    and C1's state share is, if anything, higher (state-only dB = +0.024 [−0.006, 0.053]).
  - So the relocation reading of C1 holds only for its cost (insufficiency), not as a measured drop in bypass debt.
- **D0 is falsified.**
  - ρ(D0 score, δ^F_main2) = −0.05 [−0.40, 0.30].
  - The CD bin alone is no better (−0.11 [−0.48, 0.24]), nor are the binder and motor sub-scores (0.01 and −0.05).
  - Support counted from the split does not rank the measured per-task combination debt. The largest measured debts are
    drawer tasks, which the split ranks as mid-support.

**What this says.**
- Every architecture carries combination debt of about the same size: δ^F_rel 0.95 to 1.21, with overlapping CIs, for
  main2, C1 and vanilla. The bottleneck neither created nor removed it.
- The excess error is larger at release (0.83 to 1.12) than at grasp (0.36 to 0.48). This matches 12.4's finding that the
  gripper reaches the bound object more often than it completes the placement.
- vanilla's kv purity debt (0.14) is lower than BindVLA's (0.48 to 0.69). The kv ratios are not comparable across
  architectures, though. vanilla's own-role swap changes the role words' own embeddings, while BindVLA's slot position
  receives the change only through the binder.
- All three models still decode the other role's atom from a role's message well above chance (probe 0.44 to 0.79 vs about
  0.3). The source is natural co-variation (layout, time), not the interventional dependence that the tok purity rules out.

**Could not be measured, and why.**
- Purity at the binder for vanilla: there are no slots. Only the kv level is measured, at the role words. The referent
  image cells enter vanilla's interface and bypass definitions but not its purity score.
- A KV-level interchange combination test (object-slot KV from one training compound, goal-slot KV from another, on dev
  frames) was not run. δ^F uses the dev joint message itself. For BindVLA this equals the slot-token interchange by
  construction; at the KV level it does not.
- Purity covers only the 30 in-support tasks that have both swaps in their scene. LIBERO-Object (one task per scene) and
  LIBERO-Spatial (one goal) have none.

### 4.7 D2/D1x protocol (fixed 2026-10-29, before any D2 or D1x read; m3_prereg tag pd2)

No training. Scripts: `experiments/fc_d1x.py`, `fc_d2_frames.py`, `fc_d2_support.py`, `fc_d2_debt.py`, `fc_d2_read.py`.
Budget: 10 A40-h.

**D1x: the interchange test at the expert-read level (main2, C1).**
- Compounds C with demos:
  - in-support training tasks (their held-out episodes, the D1 IN frames);
  - dev tasks (the D1 DEV frames; scored only, never executed).
- For each C, two of the 72 in-support training tasks are chosen, same BDDL scene first, then by a fixed hash:
  - A has C's step-0 object phrase and a different goal phrase;
  - B has C's goal phrase and a different object phrase.
- On each of C's frames, four prefixes are scored:
  - true: C's own instruction;
  - A: A's instruction;
  - B: B's instruction;
  - rec: B's key and value states at every layer, with the step-0 object-slot position replaced by A's. The expert then reads
    A's object message and B's other messages, a joint input that no single instruction produces.
- The action error is D1's, with D1's noise draws.
- Agreement (interchange-intervention accuracy): rec's output is nearer to true's than to both A's and B's.
- Estimators:
  - ρ_in = (E_rec − E_true) / E_true over the in-support compounds (task means);
  - ρ_dev, the same over the dev compounds;
  - Δ = ρ_dev − ρ_in, the combination debt at the expert-read level.
- **D1x-a (factored on support)** holds if the upper CI bound of ρ_in is below 0.10. It is falsified if the lower bound is above
  0.10, and inconclusive otherwise.
- **D1x-b**: expert-read combination debt is present if Δ > 0 with a CI that excludes 0, and absent if the CI includes 0.
- Prediction: D1x-a holds and D1x-b is present. On training compounds, recombined in-support messages behave like the true
  compound; on held-out compounds they do not.

**D2: cross-VLA debt.**
- *Models and outcomes.* The archive's policy-alone zero-shot arm (`zs`, 20 episodes per task), on blocks whose per-task
  success is graded:

  | block | file | success distribution: tasks at 0 / at 20 / in between (n) |
  |---|---|---|
  | pi0.5-L90 | g8_read.json | recorded in the pre-registration row (12 sealed tasks of the 66 skipped; n = 54) |
  | pi0.5-LI | g15_read_pi05.json | 6 / 3 / 11 (20) |
  | GR00T N1.5-LI | g18_read_groot_libero.json | 8 / 1 / 11 (20) |
  | X-VLA-LI | g18_read_xvla.json | 14 / 1 / 5 (20) |

  - Excluded: pi0 (14 of 20 tasks at 0) and SmolVLA (one task above 1 success). They are floored.
  - X-VLA is kept at the edge: 5 graded tasks, 4 of them at 11 or more.
  - All four checkpoints were fine-tuned on the same 40 LIBERO tasks.
  - The success distributions were read to choose the blocks; no debt had been measured.
- *Frames.*
  - LI: the 10 rendered start states of practice set a per task (results/lang/frames; never a scoring set).
  - L90: frame 0 of the first 10 IPEC demos, plus their event frames for combination (scored only).
  - Sealed LIBERO-90-C tasks are skipped by the salted hash in split.json; split_sealed.json is never opened.
- *Interface (deviation from 4.1).*
  - The role messages are the expert-read prefix features at the instruction's role phrases:
    - pi0.5: the value states of all 18 layers;
    - X-VLA: the encoder features the denoiser reads;
    - GR00T: the backbone features the DiT cross-attends.
  - The phrase span is where the tokenisation of the phrase-swapped instruction differs from the intact one.
  - The referent image tokens are not declared. The OOD tasks have no demos, so there are no contact-event referents, and
    projecting oracle poses into each model's image tokens was not built within budget.
  - So the bypass is every non-instruction input (both views and the state), and the bypass share measures how much of the
    output's compound dependence enters outside the instruction.
- *Estimators per task.*
  - Purity: D1's ratio, with swaps from the same scene (L90) or the same suite (LI).
  - Bypass share: Σd_byp / (Σd_byp + Σd_role). The partner is another task of the same scene or suite, at the same frame
    index. d is the sample MSE over the first min(10, H) steps, in the model's own normalised rows, with 2 shared seeds.
  - Combination (pi0.5-L90 only): the action error on the event frames, per dim divided by the pooled std of the scored chunks.
  - Insufficiency is not measured in D2. With the image in the bypass, cutting it removes the referent too.
- *Controls (task definitions only, fc_d2_support.py):*
  - u, the fraction of the task's compounds absent from the 40 training tasks (compound divergence);
  - the D0 score with the 40 tasks as training set.
- **KEY test.** The debt is x = the mean of the within-block ranks of purity and bypass.
  - Pooled partial Spearman: ranks within each block, x and failure (1 − zs/20) each residualised on [score, u, block dummies],
    then the correlation of the residuals.
  - CI: 2,000 bootstrap draws over task identities.
  - **Holds** if ρ > 0 and the CI excludes 0. **Falsified** if the CI includes 0, which says that debt adds nothing beyond
    novelty.
  - Reported too: each block; purity alone; bypass alone; pi0.5-L90's combination error; the uncontrolled correlation; and the
    pool of the two most graded blocks (pi0.5-LI, GR00T-LI).
- **Validation before the read.**
  - Smoke runs (`--limit`) of every script on all three models checked the code paths: 2 tasks per block, plus 20 + 20 D1x
    frames.
  - Only key names and role-span token counts were inspected (2 to 5 tokens), never a debt or error value.

### 4.8 D2/D1x results (2026-10-30; ledger fc_d2 tag d2; results/flawed/d2_read.json; 0.50 A40-h)

**D1x: interchange at the expert-read level** (task-bootstrap 95% CIs).

| | main2 | C1 |
|---|---|---|
| compounds with an (A, B) pair: in-support / dev (frames) | 25 / 8 (294 / 453) | 25 / 8 |
| in-support error: true → rec (A alone, B alone) | 0.469 → 0.496 (0.681, 0.727) | 0.529 → 0.560 (0.704, 0.694) |
| dev error: true → rec (A alone, B alone) | 0.648 → 0.647 (0.725, 0.730) | 0.689 → 0.708 (0.813, 0.810) |
| ρ_in | 0.056 [0.007, 0.104] | 0.058 [0.016, 0.111] |
| ρ_dev | −0.000 [−0.060, 0.061] | 0.027 [−0.049, 0.100] |
| Δ = ρ_dev − ρ_in | −0.057 [−0.135, 0.021] | −0.031 [−0.121, 0.052] |
| agreement (IIA), in-support / dev | 0.66 [0.53, 0.78] / 0.76 [0.65, 0.86] | 0.45 [0.34, 0.55] / 0.55 [0.38, 0.72] |

- **D1x-a is inconclusive for both models.** Recombining in-support role messages costs about 6% of action error on training
  compounds, against 45% to 55% for the wrong compound's prefix. The upper CI bounds (0.104, 0.111) sit just above the 0.10 bar.
- **D1x-b: the expert-read combination debt is absent.** Δ's CI includes 0, and the point estimate is negative.
  - On the 8 dev compounds that have in-support parts, a prefix assembled from training compounds' messages does as well as
    the model's own dev prefix.
  - Its outputs also agree with the true compound's more often than on support (main2 0.76).
- **Reading.** The expert combines its read messages close to factorially. D1's dev excess (δ^F_rel ≈ 1) therefore does not
  come from the expert facing a new joint message. It sits where the messages and frames themselves are new:
  - the dev frames' layouts and demo states;
  - the tasks without an in-support (A, B) pair, which are 21 of 29 (part tasks, drawers).
  - The 8 testable dev tasks are the easier ones: dev E_true is 0.65 here, against 1.09 over all 29 tasks in D1.
- **Caveat.** Only the step-0 object position is interchanged. The frozen text layers have already mixed every slot into
  every position, so B's goal-position states still carry B's object. The test is therefore a lower bound on what a fully
  factored read would show.

**D2: cross-VLA debt.**

Success distributions used (policy alone, 20 episodes per task; tasks at 0 / at 1-5 / at 6-14 / at 15-19 / at 20):

| block | n | mean success | histogram |
|---|---|---|---|
| pi0.5-L90 | 52 | 0.31 | 28 / 6 / 4 / 7 / 7 |
| pi0.5-LI | 20 | 0.36 | 6 / 5 / 5 / 1 / 3 |
| GR00T-LI | 20 | 0.28 | 8 / 4 / 5 / 2 / 1 |
| X-VLA-LI | 20 | 0.20 | 14 / 1 / 2 / 2 / 1 |

- pi0.5-L90 has 54 unsealed tasks. Two have no same-scene partner in the block and no valid swaps, so they carry no debt and
  drop out.

Debt per block (means over tasks, 95% CIs):

| block | purity (kv at role words) | bypass share | combination (pi0.5-L90) |
|---|---|---|---|
| pi0.5-L90 | 0.44 [0.39, 0.48] (26 tasks) | 0.74 [0.68, 0.80] | 1.11 [0.97, 1.26] |
| pi0.5-LI | 0.47 [0.44, 0.51] (19) | 0.40 [0.29, 0.51] | – |
| GR00T-LI | 0.42 [0.39, 0.46] (19) | 0.39 [0.25, 0.54] | – |
| X-VLA-LI | 0.34 [0.31, 0.38] (19) | 0.34 [0.20, 0.47] | – |

**KEY test: falsified. Debt adds nothing beyond novelty.**
- Pooled partial Spearman of debt with failure, controlling for the D0 score, u and block: ρ = −0.003 [−0.196, 0.217]
  (112 rows, 72 tasks).
- Without controls: 0.016 [−0.195, 0.221].
- Purity alone: −0.156 [−0.347, 0.136]. Bypass alone: 0.068 [−0.173, 0.286].
- The two most graded blocks pooled: −0.276 [−0.565, 0.114].

Per block (partial ρ of the debt composite):

| block | partial ρ | purity alone | bypass alone |
|---|---|---|---|
| pi0.5-L90 | 0.141 [−0.121, 0.411] | 0.128 [−0.409, 0.564] | 0.049 [−0.209, 0.322] |
| pi0.5-LI | −0.158 [−0.689, 0.403] | −0.458 [−0.807, 0.067] | 0.133 [−0.444, 0.578] |
| GR00T-LI | −0.408 [−0.717, 0.134] | −0.499 [−0.815, 0.082] | −0.049 [−0.579, 0.401] |
| X-VLA-LI | 0.316 [−0.327, 0.771] | 0.326 [−0.191, 0.773] | 0.140 [−0.384, 0.555] |

- pi0.5-L90's combination error alone: 0.072 [−0.205, 0.320].
- No block's CI excludes 0. The point estimates disagree in sign across models.

**What this says.**
- On four model-by-benchmark blocks with graded outcomes, the measured purity, bypass and combination debts do not rank
  per-task OOD failure once task novelty is held fixed. Nor do they rank it without that control.
- With D0 (falsified) and D1b (tie-driven), this leaves no evidence that per-task debt, as estimated here, predicts failure.
- The interface-level picture still holds as a description:
  - every model's role messages move with the other role (purity 0.34 to 0.47);
  - pi0.5 on LIBERO-90's new scenes routes 74% of its compound dependence outside the instruction, against 34% to 40% on
    Li's familiar-scene suites.
- **Deviations.**
  - The D2 interface omits the referent image tokens (4.7).
  - Insufficiency was not measured in D2.
  - pi0 and SmolVLA were excluded as floored.
  - Two pi0.5-L90 tasks carry no debt.

### 4.9 N1 protocol: input novelty against compound novelty (fixed 2026-10-30, before any N1 read; m3_prereg tag pn1)

Question: is held-out failure explained by how far a task's inputs are from training (layout, positions, state, appearance)
rather than by how new its compound is? No training.

Scripts: `experiments/fc_n1_tasklist.py`, `fc_n1_render.py`, `fc_n1_novelty.py`, `fc_n1_read.py`. Budget: 3 A40-h.

**Data.**
- Initial scenes are rendered as the client starts an episode: reset, set the init state, then 10 settle steps of the dummy
  action. No policy is run and no demo is executed.
- Standard LIBERO tasks use the benchmark's init states 0 to 9, the states of the BindVLA dev and parity episodes. They cover
  113 tasks: the 76 LIBERO-90-C training tasks, the 29 dev tasks, the 40 LIBERO tasks, and the 54 unsealed LIBERO-90 tasks of D2.
- Li's 20 tasks use the 10 practice start states of set a (results/robot/fresh_ood_a). Their scoring set e was not used: this is
  a deviation, and the states come from the same generator.
- The sealed LIBERO-90-C tasks were **inferred** from the salted hashes in split.json; split_sealed.json was never opened.
  Their identities are used only to exclude them, and for nothing else.

**Blocks, training sets and outcomes.**

| block | tasks | training set | outcome |
|---|---|---|---|
| bv-main2, bv-vanilla | 29 dev | the 76 LIBERO-90-C training tasks | 1 − closed-loop dev success (c1_read.json) |
| bv-c1 | 29 dev | its 72 trained tasks | same |
| pi05-L90 | 54 | the 40 LIBERO tasks | 1 − zs/20 (g8_read.json) |
| pi05-LI, groot-LI, xvla-LI | 20 | the 40 LIBERO tasks | 1 − zs/20 (g15/g18) |

**Novelty, fixed before any outcome is joined.**
- **Compound novelty.** The D0 score and u (the share of the task's compounds absent from training), the same as before. The
  composite is the mean within-block rank of the two.
- **Input novelty.** Nearest-training distances, from means over the task's 10 states, with positions relative to the robot
  base:
  - (a) referent and goal positions. The instances are the goal atom's arguments; a region maps to its fixture. For each role,
    d_a is the minimum distance to any training task's same role, averaged over the roles present.
  - (b) scene and robot state. d_b is the minimum over training tasks of the Chamfer distance between the object and fixture
    position sets, plus the end-effector distance, plus the gripper-opening distance.
  - (c) visual. The frozen SigLIP features of the agentview first frame (SmolVLM2's tower, mean over tokens and states); d_c is
    the minimum cosine distance.
  - (d) BindVLA only. d_d is the minimum distance of the checkpoint's own step-0 slot-mask moments (μ, σ for the object and
    goal) at the first frame. This one is secondary.
  - **The primary input novelty is the mean within-block rank of (a), (b) and (c).**

**Tests.**
- Method: every variable is ranked within its block, and CIs are 2,000-draw bootstraps over task identities.
- **N1a.** Pooled Spearman of input novelty with failure, with block as stratum.
  - Holds if ρ > 0 and the CI excludes 0; otherwise falsified.
- **N1b (KEY).** Two partial correlations:
  - the input-beyond-compound partial, partial(input | score, u, block);
  - the reverse partial, partial(compound | input, block).
  - The "where, not what" reading **holds** only if the first partial's CI excludes 0 (ρ > 0) **and** the reverse partial's CI
    includes 0.
  - **Falsifier:** the first partial's CI includes 0, meaning input novelty adds nothing beyond compound novelty.
  - If both CIs exclude 0, both kinds of novelty matter.
- **N1c.** For main2 and C1: Spearman of D1's per-task dev excess δ^F(d) with the block's input novelty, over 29 tasks.
  - Holds if ρ > 0 and the CI excludes 0.
- Secondary:
  - each component in N1a and N1b, and d_d in the BindVLA blocks;
  - the pool without the BindVLA blocks;
  - every block on its own;
  - continuous outcomes for main2 and C1, because 26 of the 29 BindVLA dev tasks fail always: (i) the share of episodes with
    neither success nor first contact on the named object, from the 12.4 logs; (ii) the gripper's closest approach to the
    named object over the episode, from the logged rollouts.

---

## 5. Resolution: removing debt in VLA architectures

### 5.1 Stage 1 (binding): already largely paid

- BindVLA's binder: role-local queries, nested part masks, position-free keys, and the label-free contact-event referent
  loss.
- Remaining debt: the nonlinear position content of the frozen features. Lever: layout variation plus transfer-scored
  selection.

### 5.2 Stage 2 (action expert): the factored, frame-equivariant motor

Each constraint targets one condition from Section 3.2:

1. **Phases**, split at the contact events already extracted from proprioception: approach and grasp, then transport and
   place. This gives the combination rule Φ: sequencing.
2. **Role-local reads.** The grasp phase reads only the object slot. The place phase reads only the goal slot and the held
   object. No phase sees the joint instruction or the other role's slot. This targets (F) and (B).
3. **Frame equivariance.** Each phase generates actions in its slot's frame (position from the slot mask) and maps them back
   through the slot pose. Generalisation over the nuisance holds by construction. This targets (F) for z.
4. **Sufficient role messages.** Each slot carries its object's grasp features, learned from the same contact events. This
   targets (S), and is what C1 lacked.

Predictions to pre-register:
- **Multiplicativity.** Held-out success ≈ binding × grasp competence for that object × place competence for that goal,
  each measured on training tasks.
- **A listed failure set.** Tasks that violate the factorisation (a grasp that depends on the goal, articulated objects)
  will fail, and the list is written down before the run.
- **Debt reduction.** D1's measurement on the new motor shows combination and bypass debt near zero.

### 5.3 The unsupervised arm

| component | supervised / privileged version | unsupervised version |
|---|---|---|
| roles | rule parser over LIBERO templates | induced from contact events: words that predict the grasp site are object words; words that predict the release site are goal words |
| phase read-sets | designed | chosen by leave-layout-out prequential MDL over candidate read-sets |
| frames | sim camera calibration | self-calibrated by regressing the gripper's pixel location from its own end-effector position |

**Claim:** the unsupervised arm recovers the designed structure, or we state exactly where it does not.

**Known gap:** tasks with contact but no carry, such as drawers and the stove, may yield only one role.

### 5.4 Baselines

- plain SmolVLA (`vanilla_frozen`);
- BindVLA main2;
- ECT applied to main2 (counterfactual pairs plus the paired loss);
- ReGuide (oracle-pose and perception variants);
- per-task fine-tuning.

All use the same split and the same demo budget.

---

## 6. Experiments, order and cost

| step | content | A40-h | decides |
|---|---|---|---|
| D1 | debt map for main2, C1 and vanilla (no training) | 2–4 | whether debt exists where the theory says it does, and predicts failure |
| D0 | support-graph prediction from the split | < 1 | the pre-training prediction |
| D2 | the same for pi0.5, pi0, SmolVLA, X-VLA and GR00T archive checkpoints | 4–6 | generality across architectures |
| D3 | contexts-per-atom crossover in the B1 binders | 2–3 | the acquisition account |
| N | targeted novelty search: factored, frame-equivariant VLA motors; interchange interventions in policies | 0 | whether Section 5.2 is new |
| R1 | factored motor, hand-specified, on main2's binder | roughly main2's cost, plus 2 seeds | the multiplicativity prediction and the listed failure set |
| R2 | unsupervised arm | roughly R1's cost | the unsupervised claim |
| R3 | baselines (ECT, ReGuide) | to be costed | comparison |

About 20 A40-h of the phase-2 budget remain. D0–D3 fit within it. R1–R3 need a new budget.

---

## 7. Novelty and risks

**Likely ours:**
- the chain view, with purity, combination, bypass and sufficiency debt;
- the relocation result, with C1 as evidence;
- per-interface debt measured by interchange interventions in closed-loop policies, and shown to predict failure;
- the support-graph prediction of debt from the split;
- the factored, frame-equivariant motor selected by transfer-scored MDL, with roles induced from contact events;
- the atlas: depth, compound divergence, the LIBERO mirror.

**Not ours, to be credited:**
- the single-stage support condition (Wiedemer);
- conjunction-additivity (Lippl & Stachenfeld);
- interchange interventions (Geiger);
- the identifiability argument at the instruction interface (ECT);
- equivariant pick and place (Transporter, NDF, Equivariant Diffusion Policy);
- error propagation as a bound.

**Risks:**
- **D1 shows no predictive debt.** That kills the theory cheaply.
- **Interchange interventions on a flow-matching expert are noisy.** Scoring against recombined-compound demos may not
  separate debt from ordinary error. Mitigation: use grasp-site and release-site error, not flow loss alone.
- **Role induction from contact events may fail on articulated tasks.**
- **The factored motor may cost more competence than it buys** (Proposition 2 in reverse, if the factorisation is wrong
  for many tasks).

---

## 8. Open decisions

1. Whether this and the recoverability paper become one theory or companion papers.
2. Budget for R1–R3.
3. Whether to keep LIBERO as the only testbed, or add CALVIN-C (already split) for a second dataset.

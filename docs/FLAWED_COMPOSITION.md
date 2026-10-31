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

### 4.10 N1 results (2026-10-30; ledger fc_n1 tag n1; results/flawed/n1_read.json; 0.53 A40-h)

**Success distributions** (per-task success; "fail" = 0 of the episodes, "succeed" = all of them).

| block | tasks | mean success | always fail | between | always succeed |
|---|---|---|---|---|---|
| bv-main2 | 29 | 0.041 | 26 | 3 | 0 |
| bv-c1 | 29 | 0.024 | 27 | 2 | 0 |
| bv-vanilla | 29 | 0.007 | 27 | 2 | 0 |
| pi05-L90 | 54 | 0.304 | 29 | 18 | 7 |
| pi05-LI | 20 | 0.358 | 6 | 11 | 3 |
| groot-LI | 20 | 0.283 | 8 | 11 | 1 |
| xvla-LI | 20 | 0.200 | 14 | 5 | 1 |

The BindVLA blocks are floored: ranks within them mostly separate 2 or 3 tasks from ties.

**Pooled tests** (201 rows, 7 blocks; task-bootstrap 95% CIs).

| test | ρ [95% CI] | verdict |
|---|---|---|
| N1a: input novelty vs failure | 0.305 [0.153, 0.436] | holds |
| N1b: partial(input \| score, u, block) | 0.288 [0.100, 0.427] | excludes 0 |
| N1b: partial(compound \| input, block) | 0.128 [−0.038, 0.319] | includes 0 |
| N1b verdict | | "where, not what" holds |
| N1c main2: δ^F(d) vs input novelty | −0.190 [−0.591, 0.232] | falsified |
| N1c C1 | −0.129 [−0.487, 0.278] | falsified |

**Per block** (partial ρ [95% CI]).

| block | N1a | input beyond compound | compound beyond input |
|---|---|---|---|
| bv-main2 | 0.21 [−0.11, 0.53] | 0.02 [−0.52, 0.53] | 0.31 [−0.31, 0.71] |
| bv-c1 | 0.06 [−0.17, 0.29] | −0.22 [−0.55, 0.10] | 0.49 [−0.12, 0.74] |
| bv-vanilla | 0.34 [0.11, 0.57] | 0.32 [−0.10, 0.62] | −0.09 [−0.42, 0.26] |
| pi05-L90 | 0.22 [−0.04, 0.46] | −0.02 [−0.36, 0.33] | **0.46 [0.22, 0.67]** |
| pi05-LI | 0.40 [−0.08, 0.71] | 0.33 [−0.13, 0.68] | −0.17 [−0.60, 0.30] |
| groot-LI | 0.55 [0.13, 0.80] | **0.56 [0.09, 0.85]** | 0.17 [−0.36, 0.63] |
| xvla-LI | 0.55 [0.20, 0.76] | **0.63 [0.20, 0.85]** | 0.22 [−0.24, 0.58] |

**Secondary.**
- Components (pooled, N1a / beyond compound):
  - d_a positions: 0.30 [0.14, 0.42] / 0.24 [0.06, 0.40];
  - d_b state: 0.27 [0.09, 0.42] / 0.23 [0.02, 0.39];
  - d_c visual: 0.18 [0.01, 0.33] / 0.16 [−0.03, 0.30].
  - d_d, the BindVLA slot pose (main2 and C1), gives N1a 0.29 [0.09, 0.48].
- Pool without the BindVLA blocks: N1a 0.36 [0.16, 0.53]; input beyond compound 0.38 [0.15, 0.53]; compound beyond input 0.14
  [−0.04, 0.35]. The verdict is the same.
- Continuous BindVLA outcomes (N1a / input beyond compound / compound beyond input):
  - main2 closest approach to the named object (21 tasks): 0.72 [0.43, 0.88] / 0.19 [−0.40, 0.73] / 0.47 [0.01, 0.82];
  - main2 named-contact failure (28 tasks): 0.12 [−0.24, 0.44] / −0.07 [−0.48, 0.37] / 0.37 [−0.06, 0.71];
  - C1 closest approach: −0.05 [−0.55, 0.45] / −0.48 [−0.83, 0.10] / 0.50 [−0.03, 0.82];
  - C1 named-contact failure: **−0.62 [−0.83, −0.29]** / −0.62 [−0.81, −0.36] / 0.16 [−0.36, 0.52]. Novel-input C1 tasks fail
    less often by missing the named object; the sign is the opposite of the prediction.

**Reading.**
- The pooled pre-registered verdict is "where, not what": input novelty predicts failure beyond the D0 score and u, and compound
  novelty adds nothing significant beyond input novelty.
- The pool is heterogeneous, and the verdict comes mostly from Li's OOD suites. GR00T and X-VLA give the clearest input
  effects. Li's three blocks are the same 20 tasks; the task bootstrap resamples them together, so they are not three
  independent replications.
- pi0.5 on LIBERO-90 shows the reverse: compound novelty beyond input 0.46 [0.22, 0.67], with input beyond compound null. One untested
  explanation: Li's suites perturb the layouts of mostly known compounds, while LIBERO-90 tasks are new compounds.
- BindVLA is floored and says little either way. N1c is falsified for both models: D1's dev excess does not track input novelty.
- main2's closest approach follows input novelty (0.72) but loses it once compound novelty is controlled. In that block the two
  measures are confounded.

**Deviations.**
- Li's tasks use set-a practice start states. The scoring set e was not used.
- Sealed LIBERO-90-C identities were inferred from salted hashes and used only for exclusion (0 of the 113 tasks are sealed).
- The overcap render shards were cancelled while they were still pending, and one tail-lab job rendered all 113 tasks.

### 4.11 N2: compound against layout in one scene (protocol fixed 2026-10-30, before any N2 rollout; m3_prereg tag pn2)

Question: N1's split (GR00T / X-VLA failures track input novelty on Li's suites, pi0.5's track compound novelty on LIBERO-90)
is confounded with benchmark, tasks and scenes. N2 crosses the two factors inside one scene, so that only the factor changes.
No training. Scripts: `experiments/fc_n2_tasks.py`, `fc_n2_client.py`, `fc_n2_roll.sh`, `fc_n2_novelty.py`, `fc_n2_read.py`,
`fc_n2_record.py`; pre-registration `experiments/prereg/pn2.py`. Budget: 5 A40-h.

**Scene and factors.**
- The scene is LIBERO-Goal's kitchen table: black bowl, cream cheese, wine bottle and plate (movable); wooden cabinet, flat
  stove and wine rack (fixtures). Both checkpoints were fine-tuned on the same 40 LIBERO tasks, LIBERO-Goal included
  (pi0.5_libero; ar0s/groot_libero, `lifelong_ood/policy/backbone_groot.py`), so "known" means the same thing for both.
- Every N2 BDDL is LIBERO-Goal's `put_the_cream_cheese_in_the_bowl.bddl` with only its language, objects of interest and goal
  replaced. Objects, distractors, fixtures and regions are identical across all 16 tasks.
- **Factor A, compound.** 8 known tasks are LIBERO-Goal's own object-to-target tasks (its prompt, its goal atom): bowl on
  plate, bowl on stove, bowl on cabinet, bottle on cabinet, bottle on rack, cheese in bowl, open the top drawer and put the
  bowl inside, push the plate to the front of the stove. 8 new tasks re-pair the same scene's atoms: cheese on plate, cheese
  on stove, cheese on cabinet, bottle on plate, bottle on stove, bottle in bowl, open the top drawer and put the cream cheese
  inside, push the bowl to the front of the stove. `fc_n2_novelty.py` checks this by `factors.py`'s parse against the 40
  training tasks: a known task's (object type, target, region) slot occurs in some training task; a new task's does not,
  while its object is moved and its target is used in some training task.
- **Factor B, layout.** Familiar: LIBERO's own placement sampler on the scene's training regions (seeds 600000 + k).
  New: the same state k with all four movable objects relocated (seed 700000 + k). Each object is at least 0.10 m from its
  own training position (the mean over the template task's 50 stored init states, whose largest deviation is 0.027 m), inside
  the box spanned by the four training positions widened by 0.08 m. Objects keep 0.02 m clearance from each other and the
  fixtures and stay off the stove-front region. A layout is accepted only if, over 10 settle steps of the dummy action, no
  object moves more than 2 cm in the plane and none drops more than 2 cm beyond its drop from the familiar state. Swaps into another object's training spot are allowed. Fixtures do not move, so the stove,
  cabinet and rack targets keep their training positions.
- Start states 0-3 are the smoke set and 4-23 the main set (20 episodes per task). All 16 tasks of a layout use the same
  states, and every cell and both models use the same noise seeds (the server's branch seed 13000000 + 1000 k).

**Rollouts.** `serve_guided.py` in mode `ours` (the frozen policy) and `libero_client.py` unchanged; `fc_n2_client.py` only
adds the N2 suite to LIBERO's benchmark dictionary. Each episode runs through the client's branch path, which builds a fresh
env and re-seeds it per episode. Prompt = LIBERO's file-name instruction; 300 steps; replan every 5; bind tracking on.
Per episode: s = success; g = success, or the first object moved while the commanded gripper is closed is the task's
object; p = s given g.

**N1's input novelty.** d_a, d_b and d_c (N1's definitions, the 40 LIBERO tasks as training set), from renders of main
states 4-13; composite = mean within-N2 rank. Recorded per task and cell before any rollout (results/flawed/n2_novelty.json):

| cell (compound / layout) | d_a (m) | d_b | d_c | composite |
|---|---|---|---|---|
| known / familiar | 0.0023 | 0.0049 | 0.0002 | 0.26 |
| known / new | 0.0308 | 0.0388 | 0.0012 | 0.76 |
| new / familiar | 0.0023 | 0.0049 | 0.0002 | 0.27 |
| new / new | 0.0333 | 0.0388 | 0.0012 | 0.77 |

The layout factor moves N1's input novelty and the compound factor does not. The new layouts move each object 0.20 to
0.24 m on average from its training position (at least 0.107 m), but d_a rises only to about 0.03 m: N1's d_a takes the
nearest referent of any of the 40 training tasks, and the relocated objects land near other tasks' referents.

The compound check by `factors.py` agrees with the design: each known slot occurs in libero_goal (K1 also in all of
libero_spatial); no new slot occurs in any of the 40 tasks.

**Estimators** (per model, probability scale; s(c, l) = task-cell success, c over compounds).
- Layout effect L = mean_c [s(c, fam) − s(c, new)]. Compound effect C = mean over known c of s̄(c) − mean over new c of s̄(c),
  with s̄ the mean over layouts. Interaction I = layout drop on known − layout drop on new.
- Model × factor contrast DD = (L_GR00T − C_GR00T) − (L_pi0.5 − C_pi0.5).
- CIs: 2,000-draw percentile bootstrap (seed 0), resampling compounds within each status and start states, jointly for both
  models.

**Pre-registered predictions and criteria** (from N1).
- Gate (after the smoke): a model whose known / familiar smoke success is below 0.20 is floored. Then stop and report; no
  main run.
- P1: L_GR00T > 0, CI excluding 0. P2: C_pi0.5 > 0, CI excluding 0. P3: DD > 0, CI excluding 0.
- **Double dissociation** holds if P1, P2 and P3 hold and L_GR00T > C_GR00T and C_pi0.5 > L_pi0.5 (points).
- **Partial**: P3 without P1 or P2, or P1 and P2 without P3. **Reversed**: DD < 0 with its CI excluding 0.
- **Kill (two axes dropped)**: DD's CI includes 0 and, in both models, L and C are both > 0 with CIs excluding 0 while L − C's
  CI includes 0. Both factors then hurt equally in both models.
- **Uninformative**: a model with success ≤ 0.05 in all three non-baseline cells; if both, the experiment is uninformative.
- Otherwise inconclusive.
- Secondary, not tested: g and p; the log-odds scale; the pick-and-place pairs alone (K1-K6, N1-N6); I; per-cell input
  novelty, the manipulation check new − familiar on d_a, d_b, d_c, and the Spearman of the task composite with failure.
- **Known weaknesses, stated before the read.**
  - 8 compounds per status: the compound effect is a between-task contrast with little power. The layout effect is within
    task.
  - New compounds can be physically harder (a bottle standing on a plate, cheese on a stove's burner) under LIBERO's strict
    On. The compound factor carries that along with novelty; g separates the grasp from the placement.
  - The layout factor moves only movable objects; fixture targets stay put.

**N2 results (2026-10-30; ledger fc_n2 tag n2; results/flawed/n2_read.json, n2_smoke.json; 3.59 A40-h)**

- **Smoke gate passed.** Known / familiar smoke success was 0.97 for both models (31 of 32 episodes each), so the main run
  went ahead.
- **Main run.** 640 episodes per model (32 tasks × 20 states), with no episode errors.

**Cell table** (95% CIs; s = success, g = grasped the named object or succeeded, p = success given g).

| model | compound / layout | s | g | p |
|---|---|---|---|---|
| pi0.5 | known / familiar | 0.99 [0.95, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.95, 1.00] |
| pi0.5 | known / new | 0.28 [0.12, 0.45] | 0.47 [0.27, 0.66] | 0.59 [0.35, 0.79] |
| pi0.5 | new / familiar | 0.38 [0.14, 0.64] | 0.79 [0.53, 0.99] | 0.47 [0.21, 0.74] |
| pi0.5 | new / new | 0.28 [0.16, 0.42] | 0.59 [0.41, 0.76] | 0.47 [0.28, 0.72] |
| GR00T | known / familiar | 0.96 [0.89, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.89, 1.00] |
| GR00T | known / new | 0.14 [0.04, 0.28] | 0.37 [0.21, 0.54] | 0.39 [0.14, 0.64] |
| GR00T | new / familiar | 0.13 [0.01, 0.33] | 0.86 [0.63, 1.00] | 0.15 [0.01, 0.37] |
| GR00T | new / new | 0.12 [0.02, 0.26] | 0.36 [0.21, 0.52] | 0.33 [0.07, 0.63] |

**Effects** (success scale; > 0 means the factor hurts).

| | pi0.5 | GR00T |
|---|---|---|
| layout L | 0.40 [0.26, 0.57] | 0.41 [0.31, 0.54] |
| compound C | 0.30 [0.13, 0.46] | 0.43 [0.29, 0.54] |
| L − C | 0.10 [−0.18, 0.41] | −0.01 [−0.20, 0.20] |
| interaction I | 0.62 [0.28, 0.94] | 0.80 [0.56, 1.01] |
| layout on g | 0.37 [0.22, 0.52] | 0.56 [0.43, 0.70] |
| compound on g | 0.04 [−0.15, 0.26] | 0.07 [−0.07, 0.27] |
| layout on p | 0.24 [0.04, 0.45] | 0.23 [0.03, 0.43] |
| compound on p | 0.39 [0.18, 0.58] | 0.60 [0.40, 0.78] |

- DD = (L_GR00T − C_GR00T) − (L_pi0.5 − C_pi0.5) = −0.11 [−0.46, 0.19]. On g it is 0.16 [−0.06, 0.39]; on p,
  −0.22 [−0.64, 0.18].
- The pick-and-place pairs alone (K1-K6, N1-N6) give the same picture:
  - pi0.5: L 0.44 [0.25, 0.63], C 0.25 [0.08, 0.42];
  - GR00T: L 0.43 [0.32, 0.57], C 0.44 [0.28, 0.59];
  - DD −0.20 [−0.58, 0.18].
- The log-odds scale agrees: every L and C is above 0, and DD_logit = −0.72 [−3.00, 1.35].

**Verdict against the pre-registered criteria.**
- P1 holds: GR00T's layout effect is 0.41 [0.31, 0.54].
- P2 holds: pi0.5's compound effect is 0.30 [0.13, 0.46].
- P3 fails: DD's CI includes 0, and its point estimate has the wrong sign.
- So `fc_n2_read.py`'s pre-registered decision order returns **"partial"** (P1 and P2 without P3).
- The kill criterion's conditions are **also all met**. DD's CI includes 0. In both models L and C are both > 0 with CIs
  excluding 0, and L − C's CI includes 0.
  - pn2 does not say which verdict wins when both apply. The code checks "partial" first.
  - Read plainly, P1 and P2 hold only because both factors hurt both models.
- **Substantive verdict: no double dissociation.** Inside one scene, compound novelty and layout novelty each cost pi0.5 and
  GR00T about 0.3 to 0.4 of success, by similar amounts. N1's model-by-axis split does not survive the control, so the
  "two axes, one per model" story is dropped. N1's split was the benchmark confound.

**What the cells show.**
- **The effects are not additive.** Every cell other than known / familiar falls to 0.12 to 0.38, and new / new is no worse
  than the single-novelty cells (I = 0.62 and 0.80). Both policies solve the trained instruction in the trained layout and
  lose most of it under either change.
- **Within each model, the two factors break different parts of the task.** This was not pre-registered as a test.
  - A new compound in the familiar layout leaves the grasp intact: g is 0.79 for pi0.5 and 0.86 for GR00T. Both CIs for the
    compound effect on g include 0. The new compound's cost is in the placement: the compound effect on p is 0.39 and 0.60.
  - A new layout breaks the grasp: the layout effect on g is 0.37 and 0.56. It also costs placement (0.24 and 0.23).
  - This "where → grasp, what → place" split holds in **both** models. It is a within-model dissociation of task phases,
    not a between-model one.
- **Per task.**
  - On new compounds in the familiar layout, both models usually pick the right object and then fail to place it. GR00T
    succeeds 0 of 20 times on bottle on plate, bottle on stove and bottle in bowl, after grasping in 19 to 20 of 20.
  - pi0.5 places the bottle on the plate or stove 17 of 20 times.
  - On cheese on plate and cheese on stove, pi0.5 succeeds more often in the new layout (0.30 and 0.45) than in the familiar
    one (0.10 and 0.20). In the training layout the trained trajectory pulls it toward the familiar target.
  - The drawer pair (K7 / N7) and the push pair (K8 / N8) are near 0 off the baseline cell.
- **N1's input novelty.**
  - The manipulation check holds: the composite rises by 0.50 [0.48, 0.52] from familiar to new layouts, and d_a by
    0.030 [0.022, 0.039] m.
  - Across the 32 tasks, the composite correlates with failure: ρ = 0.52 [0.18, 0.75] for pi0.5 and 0.34 [0.00, 0.65] for
    GR00T.
  - It misses compound novelty entirely: new / familiar has the same composite as known / familiar (0.27 vs 0.26) and fails
    as often as known / new.

**Caveats.**
- The compound effect includes physical difficulty under LIBERO's strict On: GR00T grasps the bottle and never stands it on
  the plate. The g / p split shows where the failure sits, not why.
- 8 compounds per status. The compound effect is a between-task contrast, and its CIs are wide.
- The near-ceiling baseline cell makes every effect on the success scale partly a ceiling effect. The interaction I is large
  for that reason; the log-odds scale gives the same signs.
- New layouts move only the movable objects. N1's d_a registers only about 0.03 m of the 0.20 m mean shift.
- One scene. The result says nothing yet about whether pi0.5's LIBERO-90 compound sensitivity in N1 came from compounds or
  from new scenes.

**Deviations.**
- `fc_n2_read.py` was edited after the pre-registration and before the main read. The smolvla venv has no scipy, so the
  secondary Spearman is computed in numpy as the Pearson correlation of average ranks. No primary estimator changed.
- The ambiguity between "partial" and the kill criterion is reported above, not resolved after the fact.
- The layout generator was relaxed twice before the pre-registration. The bar on distance from other objects' training
  positions was dropped (it allowed almost no layouts), and the settle check was made relative to the familiar state's own
  settle (LIBERO's start states drop about 7 cm). No rollout existed then.

**Cost.** 3.59 A40-h: task generation and renders 0.40 (three failed generator runs included), novelty 0.01, smoke 0.57,
main 2.62.

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

### 5.5 Prototype A, stage 1 (oracle state)

Question: is the motor the reason a correct grasp ends in the wrong place (N2: compound novelty costs p, layout novelty costs
g)? Stage 1 gives every arm privileged perception, so any new-pairing failure that remains belongs to the motor. Pre-registration
`experiments/prereg/pa1.py` (m3_prereg tag pa1). Budget: 10 A40-h.

**Oracle inputs, the same for every arm.**
- Slots: the simulator poses of LIBERO-Goal's 4 movable objects (bowl, cream cheese, wine bottle, plate) and 5 target sites (stove
  cook region, cabinet top side, rack top, top drawer region, stove front region). Each slot carries its position, its yaw, a
  hand-set size (radius, height), an identity one-hot, and (top drawer only) the drawer's opening.
- Roles (object, target) come from a template parse of the prompt (`fa1_common.roles`). This is oracle grounding.
- Proprio: end-effector position and rotation, finger positions, the last gripper command.

**Architecture A** (`lifelong_ood/policy/factored_flow.py`). A rectified-flow policy over 8-step chunks of LIBERO's 7-d delta
actions:
v(a) = (1 − c) · T_o⁻¹ v_o(T_o a | slot_o, proprio in frame_o) + c · T_t⁻¹ v_t(T_t a | slot_t, held object, proprio in frame_t).
- c = 1 while the last gripper command is "close". The object term drives approach and grasp, and the target term drives carry and
  place.
- frame_r is the slot's translation plus yaw. T_r rotates each step's translation and rotation deltas by −yaw_r. The demos confirm
  that LIBERO's delta axes are the world axes: the least-squares map from action to displacement is diagonal, about 0.01 m per unit.
- Each term reads only its own slot. The target term also reads the held object's position and yaw in the target frame and the
  held object's size, masked to 0 while the gripper is open. It never sees the held object's identity.
- The sum is fixed. There is no learned cross-role term.

**Comparison arms**, all on the same data with the same recipe:
- **H** (holistic, the "slots at perception only" arm): one MLP over all 9 slots in the world frame, with role flags, plus the
  instruction's role code (object and target one-hots). Actions are in the world frame.
- **R** (OA-WAM-like): one fused MLP over the two role-routed slots and proprio, all in the end-effector frame. Actions are in the
  end-effector frame.
- **A_noframe**: A with world-frame terms.
- **A_nogate**: A with both terms always on. The sum stays fixed.
- **A_nosum**: A's role-local inputs (both frames, the chunk in both frames) fed to one learned joint MLP. Actions are in the world
  frame.
- Widths are 512 for H, R and A_nosum, and 352 per term for the A variants.

**Data and recipe** (`experiments/fa1_data.py`, `fa1_train.py`).
- The training set is LIBERO-Goal's raw human demos for N2's 8 known tasks, 50 each. No-op steps are dropped (p0's rule), leaving
  52,204 steps.
- Object poses come from setting each recorded simulator state and reading the scene. No demo action is executed.
- Phase labels come from the gripper commands. Of the 50 demos of each pick task (K1-K6), 45 to 50 close the gripper exactly
  once; 42 of the 50 drawer demos (K7) do; every push demo (K8) keeps it open.
- The training start positions vary by only about 2 cm, and yaw does not vary at all.
- Recipe: flow matching (t ~ U(0, 1), 10 Euler steps at inference); AdamW with lr 3e-4, weight decay 1e-4 and cosine decay;
  batch 512; 30,000 steps; seed 0; EMA weights at 0.999. Every arm gets the same budget.
- Development used only held-out demos: the dev models were trained with demos 45-49 of each task held out. Their flow loss was
  tracked, and the dev models were rolled from those demos' start states (`results/flawed/a1_dev`). The final models train on all
  50.
- Dev rollouts (40 episodes per arm; held-out demo starts, not an eval cell): H 26/40, R 25/40, A 22/40. H clears the 0.5 bar, so
  the first recipe was kept, unchanged and identical for every arm.
- Training-curve check (dev models, `results/flawed/a1_train_dev.json`): in every arm the held-out EMA flow loss falls from
  0.17 to 0.20 at step 5,000 and ends within 0.01 of the train loss. At 30,000 steps it is 0.124 (H), 0.120 (R), 0.131 (A), 0.127 (A_noframe), 0.109
  (A_nogate) and 0.130 (A_nosum).
- Parameter counts: H 965,176; R 886,328; A, A_noframe and A_nogate 870,608 each; A_nosum 924,728. The largest is 1.11 times the
  smallest.

**Rollouts** (`experiments/fa1_roll.py`). N2's protocol, in process on CPU:
- N2's BDDLs, start states and noise seed bases (13000000 + 1000 k, plus the query index).
- A fresh env per episode, re-seeded (seed 7); 10 settle steps; 300 steps; replanning every 5 steps; the client's bind track.
  Actions are clipped to [−1, 1].
- The env runs without cameras. `fa1_data.py equiv` showed that this env steps bit-identically to N2's OffScreenRenderEnv: the
  state difference was 0 after 130 steps (`results/flawed/a1_equiv.json`).
- Smoke: states 0-3 on the 8 known / familiar tasks, for every arm. Main: states 4-23 on all 32 tasks (640 episodes per arm).

**Outcomes** (`experiments/fa1_read.py`).
- s and g are N2's.
- r (relaxed placement) is s, or the object's end position within max(RAD, dxy_rest + 0.03) of the target reference with dz in
  [dz_rest − 0.04, dz_rest + 0.06].
  - dz_rest and dxy_rest are calibrated by teleport and settle (`results/flawed/a1_calib.json`). Every new compound except N7 is
    strictly feasible in that check, on all 4 states.
  - The drawer target has no relaxed test.
- p is s given g (strict) and r given g (relaxed).
- Habit view: on new-compound episodes with g = 1 and s = 0, the end position is classified as instructed, habit (a target this
  object reaches in a training task), other target, or elsewhere.

**Pre-registered predictions** (95% CIs from a 2,000-draw bootstrap, seed 0, resampling compounds within status and start states,
jointly across arms).
- **Gate (smoke):** if s_H(known, familiar) < 0.5, stop, fix the recipe on held-out demos, and re-register.
- **A1 (parity):** s_A(known, fam) − s_H(known, fam) ≥ −0.10 (point estimate).
- **A2 (placement):** p_rel,A(new, fam) exceeds both p_rel,H(new, fam) and p_rel,R(new, fam), with both CIs excluding 0.
- **A3 (layout):** L_g = mean_c [g(c, fam) − g(c, new)] over all 16 compounds. L_g(H) − L_g(A) > 0 and L_g(R) − L_g(A) > 0, with
  both CIs excluding 0.
- **A4 (relocation):** for H and for R, p_rel(known, fam) − p_rel(new, fam) > 0, with the CI excluding 0.
- **A5 (descriptive):** per ablation, p_rel,A − p_rel,Y on new / familiar, and L_g(Y) − L_g(A).
- **Verdict:** the first rule that applies.
  1. s_H(known, fam) < 0.5 on the main run → uninformative.
  2. Not A1 → "A1 fails: A is a worse policy".
  3. A2 and A3 → confirmed.
  4. A2 only → "placement confirmed, layout not".
  5. Otherwise → "killed: the factored motor does not fix placement".

  A4 is reported beside the verdict and never changes it. The rules are ordered and exhaustive, so no outcome maps to two verdicts.
- **Listed factorisation violations:** K7 / N7 (the articulated drawer is opened before the place) and K8 / N8 (no gripper closure,
  so A's target term never runs).
- **Secondary:** strict p; the pick-and-place pairs alone (K1-K6, N1-N6); success effects; a per-compound table.
- **Known weaknesses.**
  - The training layout barely varies (2 cm), so every arm extrapolates on the new layouts. A's frames make the inputs
    translation-relative, not in-distribution: the end effector starts at a relative offset never seen in training.
  - 8 compounds per status.

**Results (2026-10-30; ledger fc_a1 tag a1; results/flawed/a1_read.json, a1_smoke.json)**

- **Smoke gate passed.** Known / familiar smoke success (32 episodes per arm) was H 0.53, R 0.53, A 0.375, A_noframe 0.375,
  A_nogate 0.69 and A_nosum 0.53.
- **Main run.** 640 episodes per arm (32 tasks × 20 states), with no episode errors.

**Cell table** (95% CIs; p_s = s given g, strict; p_r = r given g, relaxed).

| arm | compound / layout | s | g | p_s | p_r |
|---|---|---|---|---|---|
| H | known / fam | 0.54 [0.34, 0.75] | 0.75 [0.56, 0.92] | 0.72 [0.52, 0.89] | 0.78 [0.61, 0.92] |
| H | known / new | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| H | new / fam | 0.00 [0.00, 0.00] | 0.12 [0.01, 0.30] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| H | new / new | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.06] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| R | known / fam | 0.56 [0.33, 0.80] | 0.73 [0.49, 0.94] | 0.77 [0.54, 0.93] | 0.83 [0.66, 0.95] |
| R | known / new | 0.01 [0.00, 0.05] | 0.11 [0.02, 0.23] | 0.12 [0.00, 0.40] | 0.12 [0.00, 0.40] |
| R | new / fam | 0.01 [0.00, 0.04] | 0.06 [0.00, 0.17] | 0.10 [0.00, 0.67] | 0.10 [0.00, 0.67] |
| R | new / new | 0.01 [0.00, 0.04] | 0.06 [0.01, 0.13] | 0.11 [0.00, 0.44] | 0.11 [0.00, 0.44] |
| A | known / fam | 0.34 [0.15, 0.56] | 0.62 [0.40, 0.83] | 0.54 [0.27, 0.80] | 0.62 [0.35, 0.85] |
| A | known / new | 0.10 [0.00, 0.26] | 0.31 [0.14, 0.48] | 0.33 [0.00, 0.71] | 0.37 [0.04, 0.75] |
| A | new / fam | 0.10 [0.01, 0.24] | 0.55 [0.36, 0.72] | 0.18 [0.01, 0.40] | 0.22 [0.04, 0.43] |
| A | new / new | 0.08 [0.00, 0.21] | 0.34 [0.20, 0.50] | 0.24 [0.00, 0.51] | 0.25 [0.02, 0.52] |
| A_noframe | known / fam | 0.39 [0.17, 0.63] | 0.69 [0.48, 0.87] | 0.57 [0.27, 0.85] | 0.62 [0.32, 0.89] |
| A_noframe | known / new | 0.04 [0.00, 0.11] | 0.32 [0.17, 0.50] | 0.14 [0.00, 0.30] | 0.14 [0.00, 0.30] |
| A_noframe | new / fam | 0.18 [0.05, 0.36] | 0.66 [0.50, 0.81] | 0.27 [0.07, 0.54] | 0.33 [0.12, 0.61] |
| A_noframe | new / new | 0.04 [0.00, 0.10] | 0.29 [0.12, 0.47] | 0.13 [0.00, 0.34] | 0.13 [0.00, 0.34] |
| A_nogate | known / fam | 0.72 [0.48, 0.92] | 0.80 [0.55, 1.00] | 0.90 [0.77, 0.98] | 0.92 [0.80, 1.00] |
| A_nogate | known / new | 0.04 [0.00, 0.12] | 0.08 [0.01, 0.18] | 0.54 [0.00, 1.00] | 0.54 [0.00, 1.00] |
| A_nogate | new / fam | 0.05 [0.00, 0.17] | 0.06 [0.00, 0.21] | 0.80 [0.00, 1.00] | 0.80 [0.00, 1.00] |
| A_nogate | new / new | 0.03 [0.00, 0.09] | 0.10 [0.03, 0.19] | 0.31 [0.00, 0.72] | 0.31 [0.00, 0.72] |
| A_nosum | known / fam | 0.62 [0.41, 0.82] | 0.78 [0.53, 0.97] | 0.80 [0.65, 0.92] | 0.85 [0.74, 0.94] |
| A_nosum | known / new | 0.07 [0.01, 0.15] | 0.19 [0.07, 0.33] | 0.37 [0.10, 0.64] | 0.37 [0.10, 0.64] |
| A_nosum | new / fam | 0.00 [0.00, 0.00] | 0.09 [0.00, 0.23] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| A_nosum | new / new | 0.00 [0.00, 0.00] | 0.11 [0.01, 0.24] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

**Pre-registered predictions.**
- **A1 (parity) fails.** s_A − s_H on known / familiar is −0.21 [−0.39, −0.03], below the −0.10 bar.
- **A2 (placement) fails.** A's p_r on new / familiar beats H's by 0.22 [0.04, 0.43], but not R's: +0.12 [−0.49, 0.40]. R grasps
  the right object in only 6% of these episodes, so its p rests on few episodes.
- **A3 (layout) fails.** L_g(H) − L_g(A) = 0.16 [−0.02, 0.35] and L_g(R) − L_g(A) = 0.05 [−0.12, 0.23]. The L_g values are H 0.42,
  R 0.32 and A 0.26.
- **A4 (relocation) holds.** With oracle slots, H and R still lose placement on new pairings: p_r falls from known to new by
  0.78 [0.61, 0.92] for H and 0.73 [0.17, 0.94] for R.
- **A5 (descriptive).**
  - Only A_nosum's contrast clears 0: A's p_r on new / familiar exceeds A_nosum's by 0.22 [0.04, 0.44]. With a learned joint
    head, A_nosum reaches 0 placements on new pairings.
  - The frame (A − A_noframe: −0.11 [−0.40, 0.15]) and the gate (−0.58 [−0.90, 0.33]) carry nothing measurable on p_r. No
    ablation carries A3.
  - A_nogate is the strongest known / familiar arm: it exceeds A by 0.38 [0.23, 0.55]. On new pairings, though, it fails at the
    grasp (g 0.06).

**Verdict (pre-registered order): "A1 fails: A is a worse policy".** On known tasks in the training layout, the factored,
gated, frame-local motor solves 0.34 to H's 0.54. Per the pre-registration, the comparison stops here. The two sections below are
descriptive.

**What the cells show (descriptive, not tested).**
- **Oracle perception does not make a holistic head compose (A4).**
  - With ground-truth slots, H and R fall to s ≤ 0.01 in every cell other than known / familiar.
  - Unlike pi0.5 and GR00T in N2, they lose the grasp on new pairings (g 0.12 and 0.06, against 0.79 and 0.86 in N2). In
    0.88 of H's new / familiar episodes, the first object moved with the gripper closed is not the named object, or no object
    moves at all.
- **The role-local object term keeps the grasp.**
  - A grasps the named object in 0.55 of new / familiar episodes, and A_noframe in 0.66. Every arm with a joint head (H, R,
    A_nogate, A_nosum) stays at or below 0.12.
  - That is the clearest effect of the factorisation. The grasp transfers when the term that drives it reads only the object's
    slot.
- **The place after that grasp mostly fails.**
  - A's p_r on new / familiar is 0.22. In the habit view (72 g = 1, s = 0 episodes), 0.76 end elsewhere, 0.11 at one of the
    object's trained targets, 0.08 at another target and 0.04 relaxed-instructed.
  - The failure is not habit capture. The target term, trained with one or two held objects per target, does not carry an
    unseen held object to the target.
- **Layout.**
  - Every arm loses most of its success on the new layout. The best new-layout cell is A's known / new at s 0.10.
  - The translation-and-yaw frame does not make the approach transfer. The end effector starts at relative offsets the 2 cm
    training spread never covered.
- **Where A1 is lost.** On known / familiar, A trails H most on K5 (rack, 0.10 vs 0.40), K7 (drawer, 0.00 vs 0.15) and K8 (push,
  0.00 vs 0.45). K7 and K8 are in the listed factorisation violations. On the pick-and-place pairs alone, A1 is −0.17 [−0.38, 0.03].
- **The listed violations, per compound.**
  - N8 (push the bowl) is A's best new compound (s 0.50), where every other arm scores at most 0.10.
  - N7 is 0 for every arm.
- **Strict p agrees.** A − H on new / familiar is 0.18 [0.01, 0.40].

**Caveats.**
- One seed per arm.
- 8 compounds per status, so the compound contrasts have wide CIs. R's new-pairing p rests on very few grasps.
- The training layout barely varies, so the layout factor tests extrapolation for every arm.
- The relaxed predicate is calibrated by teleport. The drawer has no relaxed test.
- The habit classes are this section's own. N2 had no formal habit test.
- A's held-object features carry geometry only, by design. A variant with identity was not run.

**Deviations.**
- An empty `python3` heredoc (no statements) was run on the login node by mistake. It computed nothing.
- The first pre-registration job failed because section 5.5 lacked its results marker. The marker was added and the job rerun.
  No row was written before that, and no rollout existed.

**Cost.** 0.61 A40-h (GPU only for training: dev 0.32, final 0.28, render-equivalence check 0.005). Rollouts and data ran on CPU
nodes: data and calibration about 0.15 h, dev rollouts 0.4 h, smoke 0.4 h, main 6.9 CPU-job-h (24 jobs of 2 cores).

**Stage-2 plan (vision slots on pi0.5 features; not run).**
1. Do not start stage 2 until a stage-1b closes A1 on oracle state. The factored motor's in-distribution competence, not
   perception, is the current bottleneck.
   - Candidate levers, each tied to a measured weakness:
     - the target term's held-object generalisation, by training it on carry segments re-expressed per held object;
     - layout variation in the start states, needed by every arm;
     - a bounded, measured cross-role residual for the articulated and push tasks.
   - Each lever needs its own pre-registration.
2. If stage 1b passes, replace the oracle slots with BindVLA-style slots read from frozen pi0.5 features.
   - Slot positions come from the slot mask back-projected through depth (frame = the estimated translation; yaw dropped).
   - Identity and shape features come from the slot embedding. Roles first from the same template parse, then induced from
     contact events (5.3).
3. Keep the arms (H, R, A and the ablations), N2's 32 tasks, states, seeds and read. Add binding accuracy per role.
4. Prediction to register: A's success ≈ (stage-1b A success) × (role-binding accuracy), against H's fall.
5. Budget: the pi0.5 feature cache plus training at about 3 A40-h, and rollouts on GPU nodes (rendering) at about 4 A40-h.

### 5.6 What "compositional, with success guaranteed" means; stage 1b (contact gate, grasp offset, 3 seeds)

**Definition.** A policy is *compositional over pairings* on a split if two things hold.
1. A structural guarantee: a proposition, true for every parameter setting, that says which role inputs the action can depend
   on in each phase. From it, success on a new pairing factorises into part competences that were measured on trained tasks.
2. Empirical criteria C1-C4 (below), pre-registered and tested on pick-and-place pairings with oracle state.

The guarantee is conditional. Section 3.2's conditions say what it needs: (F) and (B) come from the architecture, while (S) and
the support condition (3.4) come from the data. The proposition states every assumption that does not follow from the
architecture.

**Setting.** Arm A (stage 1b's Ab). s is the simulator state, o the object role and t the target role. The velocity field is
v = (1 − c) T_o⁻¹ v_o(φ_o) + c T_t⁻¹ v_t(φ_t). The terms read:
- φ_o = the object slot and proprio, in the object frame;
- φ_t = the target slot, proprio and the held-object geometry h(s), in the target frame. h(s) is the held object's position, yaw,
  size and grasp offset. It carries no identity.

The phase c ∈ {0, 1} is the held machine (`experiments/fa1b_common.py`), which reads only the simulator contacts and the height of
the object role o. A chunk is sampled by integrating v from fixed noise. c is an input to that integration, not a function of the
chunk.

**Proposition 3 (role locality and conditional factorisation).**
- (i) *Role locality, for every parameter setting.* If c = 0, the sampled chunk is a function of (φ_o, noise) only. It does not
  change when t is replaced by any t′, or when any slot other than o's is changed. If c = 1, the chunk is a function of
  (φ_t, noise) only. It does not change when o's identity is replaced, provided h is kept, or when any slot other than t's and
  the held object's is changed.
  - *Proof.* Each Euler step adds v(x_k) / 10. With c fixed for the query, v(x_k) is a function of x_k and of the inputs
    listed. So by induction every x_k is, starting from x_0 = noise. ∎
- (ii) *The grasp is target-blind in closed loop.* Take two tasks (o, t) and (o, t′) from the same start state with the same
  noise seeds, where the scene and its physics do not depend on the goal (true for N2's BDDLs). Let τ be the first step at
  which c turns on. By (i) and determinism, the trajectories are identical up to τ, so the law of (τ, s_τ) depends on o only.
  Write g_o = P(τ < ∞ | o).
- (iii) *Factorisation.* Assume (a) a pick-and-place success needs a held phase, and (b) the episode has one held segment. A
  regrasp after a lost hold is counted against the place. Then P(success | o, t) = g_o · q_t(o), where
  q_t(o) = E[P(success | s_τ, t) | τ < ∞, o]. By (i), after τ the actions depend on (o, s) only through φ_t, and in particular
  through h.
- (iv) *Transfer.* q_t(o) = q_t(o′) = p_t holds, so P(success | o, t) = g_o · p_t with g_o measured on any trained task with
  object o and p_t on any trained task with target t, under two further conditions:
  - **(Supp) support:** along the place phase, the law of φ_t under (o, t) stays where the place term was fit for target t. In
    particular, h(o) must lie in the range of held geometries trained with t.
  - **(Suff) sufficiency:** the outcome of the place phase depends on the held object only through h, and not on its mass,
    friction or mesh beyond h.

(i) and (ii) are guarantees of the architecture and are checked numerically below (the swap test, and closed-loop grasp
identity). (iv) is not guaranteed by the architecture. On this training set (Supp) fails for every new pairing:
- each target is placed on with one held object (plate and stove: bowl; rack: bottle; bowl: cheese), except the cabinet (bowl
  and bottle);
- no new pairing's held object lies inside its target's trained range of held geometry. The cheese is lower than both objects
  trained on the cabinet (0.03 m against 0.05 and 0.28).

So the factorisation g_o · p_t is a prediction that the data do not license here. C3 tests whether it holds anyway.

**Empirical criteria** (pre-registered in pa1b; pick-and-place pairs K1-K6 and N1-N6, oracle state, familiar layout; A = Ab,
H = Hb, R = Rb; success s is LIBERO's strict check; values pooled over 3 seeds).
- **C1 parity:** s_A(known) − s_H(known) ≥ −0.10 (point estimate; CI reported).
- **C2 compositional ratio:** s_A(new) / s_A(known) ≥ 0.8 (point), with the CI's lower bound ≥ 0.6.
- **C3 multiplicativity:** for every new pairing (o, t), the 95% CI of s_A(o, t) − g_A(o) · p_A(t) contains 0.
  - g_A(o) is the episode-pooled g over the known pick-and-place tasks with object o.
  - p_A(t) is s given g, pooled over the known pick-and-place tasks with target t.
  - Only start states are resampled for C3.
- **C4 baselines:** s_A(new) − s_H(new) > 0 and s_A(new) − s_R(new) > 0, both CIs excluding 0.
- **Verdict**, the first rule that applies:
  1. s_H(known) < 0.5 → "uninformative";
  2. C1, C2, C3 and C4 all hold → "compositional";
  3. otherwise → "not compositional (fails: …)", listing the failing criteria in order.
- **Descriptive class**, declared before the read: K7 / N7 (the drawer is opened before the place) and K8 / N8 (a push with no
  grasp). These are reported per compound and never enter C1-C4.

**Stage 1b changes** (from 5.5's measured weaknesses; pre-registration `experiments/prereg/pa1b.py`, m3_prereg tag pa1b).
1. **Contact gate.** c is the held machine, not the last gripper command. c turns on when the object role touches both fingers
   and is lifted at least 1 cm above its height at the first step. It stays on while the object touches either finger, and turns
   off when it touches neither.
   - Weakness: the command gate handed control to the target term on every close, including closes on air. A_nogate showed the
     cost: best in distribution, and loses the grasp on new pairings.
   - Principle: (B), since the phase boundary should be the physical event that separates the roles. The command was a bypass
     proxy for it.
   - On the demos (`results/flawed/a1b_held.json`): the machine has 1 held segment in 36-48 of 50 demos per pick task, against
     10-45 for raw two-finger contact. It turns on 11-15 steps (median) after the first close command. On the push task K8 it
     is on for 0.5% of steps, against 49% for raw contact.
2. **Grasp offset.** The target term also reads the held object's position minus the end effector's, captured when the hold
   starts, in the target frame. It still never reads the held object's identity.
   - Principle: (S), since the place term needs the held geometry relative to the gripper.
3. **The same oracle inputs for every arm.** Hb and Rb get the held flag and the offset in their proprio (world and
   end-effector frames).
4. **Seeds.** Hb, Rb and Ab are trained with seeds 0, 1 and 2. The ablations are trained with seed 0 only:
   - Ab_cmd: the stage-1 command gate, with stage 1b's held features;
   - Ab_nooff: no offset;
   - Ab_nosum: a joint MLP over Ab's inputs.
- **Recipe unchanged** (5.5; `experiments/fa1b_train.py`). It was not tuned, so no dev round was run. Training data is stage
  1's 52,204 steps plus the contact labels (`experiments/fa1b_data.py held`; recorded states set and read, no action executed).
- **Rollouts** (`experiments/fa1b_roll.py`): stage 1's protocol, with the held machine stepped after every simulator step.
  - They cover N2's 16 familiar-layout tasks, main states 4-23: 320 episodes per model, 12 models.
  - There is no new-layout run in this round. Layout is a later round (the training layout varies 2 cm).
  - There is no smoke run. The floor rule is applied to the main run.
- **Read**: `experiments/fa1b_read.py`. The bootstrap resamples the 6 known and 6 new compounds within status and the 20 start
  states (2,000 draws, seed 0), jointly for every arm and seed. A state draw applies to every seed, and seeds are not resampled.
- **Swap test** (`experiments/fa1b_swap.py`, training states only, run before the pre-registration): it checks (i) per arm on
  4,000 steps.
  - T1 swaps the target role in the object phase.
  - T2 swaps the object identity, keeping its geometry, in the place phase.
  - T3 moves every non-role slot by 5 cm.
  - A must show 0 change in all three.
- **Predictions**, stated with the support analysis above:
  - (ii) makes A's grasp on a new pairing equal to the grasp on the known tasks with the same object, episode for episode;
  - C1 and C3 are open;
  - C2 is expected to fail, because (Supp) fails.
- Budget: 1 A40-h for this round.

**Results (2026-10-31; ledger fc_a1b tag a1b; results/flawed/a1b_read.json, a1b_swap.json, a1b_diag.json, a1b_diag2.json)**

- 320 episodes per model (16 familiar tasks × 20 states), 12 models, no episode errors. The floor holds (s_Hb(known) = 0.67).
- **Swap test (Proposition 3 (i)), training states, 4,000 steps per test.**
  - Ab changes by exactly 0 on all three tests, and so does Ab_nooff.
  - Hb changes by a mean of 0.80 (max 3.10) on T1, 1.02 (3.37) on T2 and 0.18 (2.16) on T3, against a mean |action| of 0.27-0.31.
  - Rb changes by 1.31 on T1 and 0.46 on T2, and by 0 on T3, since it reads only the role slots.
  - Ab_nosum changes by 1.32 on T1 and 0.81 on T2.
  - Ab_cmd changes by 0.33 on T1 (max 5.74), because its command gate disagrees with contact on some steps.
- **Closed-loop grasp identity (Proposition 3 (ii)).** Groups are (object, start state) pairs where at least one task reaches a
  hold. In those groups, Ab's first held step and the end-effector position there are identical across every familiar task with
  that object: 49 of 49 groups (seed 0), 43 of 43 (seed 1), 42 of 42 (seed 2). For Hb, Rb, Ab_cmd and Ab_nosum the count is 0
  of 47 to 60.

**Cell table, pick-and-place pairs** (95% CIs; 3 seeds pooled for Hb, Rb, Ab; seed 0 for the ablations; "A (1)" is stage 1's A
on the same cells).

| arm | s known | s new | g known | g new | p_s known | p_s new | p_r new |
|---|---|---|---|---|---|---|---|
| Hb | 0.67 [0.49, 0.85] | 0.01 [0.00, 0.03] | 0.88 [0.80, 0.96] | 0.18 [0.06, 0.31] | 0.76 [0.59, 0.90] | 0.03 [0.00, 0.12] | 0.05 [0.00, 0.15] |
| Rb | 0.77 [0.56, 0.95] | 0.01 [0.00, 0.02] | 0.91 [0.80, 0.99] | 0.16 [0.07, 0.26] | 0.84 [0.67, 0.96] | 0.03 [0.00, 0.15] | 0.07 [0.00, 0.23] |
| Ab | 0.72 [0.51, 0.91] | 0.26 [0.09, 0.46] | 0.88 [0.70, 0.99] | 0.69 [0.53, 0.83] | 0.82 [0.67, 0.93] | 0.37 [0.13, 0.64] | 0.48 [0.25, 0.73] |
| Ab_cmd | 0.51 [0.31, 0.71] | 0.17 [0.05, 0.34] | 0.68 [0.53, 0.83] | 0.48 [0.32, 0.63] | 0.74 [0.51, 0.93] | 0.36 [0.09, 0.70] | 0.38 [0.11, 0.71] |
| Ab_nooff | 0.81 [0.62, 0.95] | 0.33 [0.11, 0.57] | 0.90 [0.72, 1.00] | 0.69 [0.46, 0.88] | 0.90 [0.78, 0.98] | 0.47 [0.17, 0.81] | 0.59 [0.30, 0.92] |
| Ab_nosum | 0.79 [0.60, 0.94] | 0.00 [0.00, 0.00] | 0.92 [0.78, 1.00] | 0.07 [0.00, 0.17] | 0.86 [0.73, 0.96] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| A (1) | 0.45 [0.25, 0.67] | 0.05 [0.00, 0.13] | 0.68 [0.49, 0.86] | 0.50 [0.31, 0.68] | 0.66 [0.44, 0.85] | 0.10 [0.00, 0.31] | 0.15 [0.02, 0.40] |

- Per seed, Ab's s(known) is 0.75, 0.72 and 0.68, and its s(new) 0.28, 0.21 and 0.28.

**Criteria.**
- **C1 holds:** s_Ab(known) − s_Hb(known) = +0.04 [−0.10, 0.19]. The contact gate closes stage 1's parity gap (A1 was −0.21).
- **C2 fails:** s_Ab(new) / s_Ab(known) = 0.36 [0.13, 0.69].
- **C3 fails** on 5 of 6 pairings. In the table below, D = s − g_o · p_t, and only N3's CI contains 0.
- **C4 holds:** s_Ab(new) − s_Hb(new) = 0.25 [0.08, 0.45] and s_Ab(new) − s_Rb(new) = 0.25 [0.08, 0.46].

| pairing | g_o (known) | g (observed) | p_t (known) | p (observed) | predicted s | observed s | D |
|---|---|---|---|---|---|---|---|
| N1 cheese → plate | 0.52 | 0.53 | 0.92 | 0.16 | 0.47 | 0.08 | −0.39 [−0.54, −0.25] |
| N2 cheese → stove | 0.52 | 0.55 | 0.97 | 0.55 | 0.50 | 0.30 | −0.20 [−0.33, −0.07] |
| N3 cheese → cabinet | 0.52 | 0.57 | 0.85 | 0.71 | 0.44 | 0.40 | −0.04 [−0.15, 0.07] |
| N4 bottle → plate | 0.87 | 0.87 | 0.92 | 0.13 | 0.79 | 0.12 | −0.68 [−0.79, −0.56] |
| N5 bottle → stove | 0.87 | 0.87 | 0.97 | 0.73 | 0.84 | 0.63 | −0.20 [−0.32, −0.08] |
| N6 bottle → bowl | 0.87 | 0.75 | 0.65 | 0.00 | 0.56 | 0.00 | −0.56 [−0.72, −0.40] |

**Verdict (pre-registered order): "not compositional (fails: C2, C3)".**

**What the cells show (descriptive).**
- **The grasp factor transfers, as Proposition 3 (ii) requires.** Ab's grasp on a new pairing equals its grasp on the known tasks
  with that object: cheese 0.53-0.57 against 0.52, bottle 0.87 against 0.87.
  - The exception is N6 (0.75), even though the held-step identity holds there too. g is the bind track (the first object
    displaced more than 3 cm while the gripper is closed), which is not the hold. Why it differs there was not checked.
- **The whole C3 deficit is in the place factor.** On new pairings p falls below its target's known p_t, most on the movable
  targets (plate 0.13-0.16 against 0.92; bowl 0.00 against 0.65).
- **Where the places fail** (`experiments/fa1b_diag.py`). After a grasp, every failed new-pairing place ends with the object
  released more than 3 cm from the target (93-100% of failures), at a median 8-39 cm. Releases within 3 cm: 0%. The carry ends
  in the wrong place.
- **Why: (Supp) fails in two parts** (`experiments/fa1b_diag2.py`).
  - *Start gap.* At the grasp onset, in the target frame, the end effector lies a median 7-25 cm from the nearest training onset
    for that target (N6, 25 cm, has p = 0). On the known tasks the gap is 1-2 cm.
  - *Offset gap.* The new held object's grasp offset lies 4-9 cm from those trained with the target (under 1 cm on the known
    tasks).
  - Each target was trained from one object's location, so a new pairing starts its carry from a place the target term never
    saw. This is the layout confound of 3.4, met inside one layout.
- **Ablations.**
  - *The contact gate carries parity and the grasp:* Ab − Ab_cmd is +0.21 [0.03, 0.40] on s(known) and +0.21 [0.05, 0.37] on
    g(new).
  - *The offset buys nothing:* Ab − Ab_nooff is −0.09 [−0.21, 0.02] on s(known) and −0.07 [−0.18, 0.03] on s(new). With one
    held object per target, the offset never varies in training.
  - *The joint head again fails to compose:* Ab − Ab_nosum is +0.26 [0.09, 0.46] on s(new) and +0.62 [0.42, 0.79] on g(new).
  - *Against stage 1's A:* +0.27 [0.10, 0.43] on s(known) and +0.21 [0.06, 0.39] on s(new).
- **Descriptive class.**
  - K7 (drawer): Ab grasps the bowl in 0.95-1.00 of episodes and never opens the drawer (s = 0); Hb reaches 0.25-0.30.
  - K8 (push): Ab 0.00-0.05, Hb 0.30-0.45. The plate is never lifted, so the contact gate never hands over, and the object term
    must push alone.
  - N7: 0 for every model.
  - N8: Ab 0 (relaxed 0-0.30). Ab_cmd reaches 0.60, because the command gate lets the target term run.

**Caveats.**
- Seeds are pooled, not resampled.
- Hb and Rb receive the contact flag and offset as inputs. They still fail the grasp on new pairings (g 0.16-0.18).
- No new-layout cells were run.

**Deviations.**
- The swap test was rerun twice before the pre-registration. The first run crashed (targets can be movable objects), and the
  second had a noise-draw bug in T2 (the perturbed sub-batch drew different noise). The registered numbers are the third run.
  No rollout existed then.
- Some job files were written with shell redirection on the login node (sed, printf and cp into the job directory), not with
  the edit tools. No computation ran there.
- Two job waiters overlapped for a few seconds once.

**Cost.** 0.61 A40-h (training: 3 jobs, 37 min). CPU: rollouts 8.45 CPU-job-h (24 jobs); contact labels, swap test, read and
diagnoses about 0.1 h.

### 5.7 Stage 1c: a canonical place frame (protocol fixed 2026-10-31, before any stage-1c rollout; m3_prereg tag pa1c)

**Measured weakness (5.6).** The place factor does not transfer, because (Supp) fails. A new pairing starts its carry 7-25 cm
outside the target term's training onsets (start gap), with a grasp offset 4-9 cm from any trained with that target (offset
gap).

**Principle.** Section 3.5's lever for a nuisance on a factored stage: frame equivariance, so that the stage's inputs are the
same for every value of the nuisance. Here the nuisance is the grasp: which object is held, where it is held, and from which
side of the target it was picked up. Rigid holding makes this exact. A translation delta moves the gripper and the held object
alike, so the place can be written as a motion of the held object relative to the target.

**The change (one, structural; arm Ac, `lifelong_ood/policy/factored_flow.py` ARMS_C).** The target term is moved into a
canonical place frame. The object term and the contact gate are unchanged from Ab.
- **Origin and axes.** The origin is the target. The x-axis is the bearing, frozen at the hold's onset, from the held object's
  foot to the target. The foot is the object's position with its height replaced by height − rest height + table height. The
  rest height is the object's lowest height before the hold began (`experiments/fa1c_common.py`).
- **Inputs.**
  - the foot in this frame;
  - the end effector's tilt (the world-z row of its rotation, which is invariant to rotations about z);
  - finger positions;
  - the last command;
  - the target slot's identity, size and drawer opening.
- **Dropped:** the held object's size, yaw and offset (identity proxies), the end effector's position and yaw, and the target's
  yaw.
- **What it buys, by construction.** The offset gap vanishes, since the foot is the same point on every object. The start gap
  shrinks to the carry distance d, since the bearing is canonical and the foot height at onset is fixed by the lift rule.

**Proposition 3 for Ac.**
- (i) and (ii) hold as before. The object term and the gate are Ab's.
- New in (i): the place chunk is equivariant to rotating the scene about the target's vertical axis. If the bearing, every slot
  except the target, and the end effector rotate together, the chunk rotates with them. This holds for every parameter setting.
- In (iv), (Supp) reduces to one condition: the pairing's carry distance d lies in the range trained for its target.

**Support prediction, made before the rollouts** (`experiments/fa1c_support.py`, results/flawed/a1c_support.json). The onsets
are stage 1b's Ab grasp onsets, and the foot is approximated from the end effector and the object's median training offset. For
each pairing, the distance to the nearest training onset is given in 1b's coordinates (end effector, target frame) and in 1c's
(|d − nearest trained d|):

| pairing | 1b gap | 1c gap | d | trained d for the target (min-max) |
|---|---|---|---|---|
| N1 cheese → plate | 0.069 | 0.010 | 0.18 | 0.12-0.17 |
| N2 cheese → stove | 0.083 | 0.001 | 0.23 | 0.22-0.29 |
| N3 cheese → cabinet | 0.100 | 0.064 | 0.39 | 0.25-0.33 |
| N4 bottle → plate | 0.147 | 0.102 | 0.27 | 0.12-0.17 |
| N5 bottle → stove | 0.137 | 0.001 | 0.26 | 0.22-0.29 |
| N6 bottle → bowl | 0.251 | 0.000 | 0.14 | 0.12-0.16 |

- Prediction: Ac's (Supp) holds for N2, N5 and N6, nearly for N1, and not for N3 or N4, whose carries are longer than any
  trained to that target.
- So C3 is predicted to hold at most on N1, N2, N5 and N6, and only if (Suff) holds there. N6 (bottle in bowl) and N4 (bottle
  on plate) may fail on physics.
- C2 needs most pairings in support and is open.

**Arms.**
- Ac with seeds 0, 1 and 2.
- Ablations, seed 0:
  - Ac_nobear: the foot, in the target's own yaw frame, with the full end-effector rotation;
  - Ac_noft: the bearing frame and tilt, with the end-effector position in place of the foot.
- H and R are stage 1b's Hb and Rb runs (same protocol, states and noise), not retrained. Ac reads the rest height and the
  bearing, which they do not. Both are functions of the object's and target's poses over time, already in their inputs. Ab (1b)
  and stage 1's A are shown beside them.

**Recipe, rollouts, read.**
- The recipe is unchanged and was not tuned (`experiments/fa1c_train.py`).
- Rollouts follow stage 1b's protocol, with the canonical tracker stepped after every simulator step (`experiments/fa1c_roll.py`).
  The read is `experiments/fa1c_read.py`, on the engine `experiments/fa1x_read.py`. With A = Ab, the engine reproduces the
  registered 1b read exactly (worst difference 0.0).
- The criteria C1-C4, the classes, the CIs and the verdict order are 5.6's, with A = Ac.
- The swap test (`experiments/fa1c_swap.py`, training states) adds a fourth check, T4: the place chunk's error after rotating
  the scene about the target. Ac must show 0 on T1-T4.
  - T4 rotates the flow noise with the scene. The sampler is equivariant per sample only then; in distribution it always is,
    because the noise is isotropic.
  - Result before the rollouts (results/flawed/a1c_swap.json): Ac gives 0 on T1-T3 and at most 5e-6 on T4 (float32 round-off).
    Ac_noft gives the same. Ac_nobear gives 0 on T1-T3 and a mean of 0.90 (max 6.07) on T4.
  - An earlier T4 run without the rotated noise gave 0.29 for Ac. It was replaced before the pre-registration.
- Budget: 1 A40-h.

**Results (2026-10-31; ledger fc_a1c tag a1c; results/flawed/a1c_read.json, a1c_diag.json)**

- 320 episodes per model, 5 models, no episode errors.
- Closed-loop grasp identity holds for every Ac model: 50 of 50, 42 of 42 and 41 of 41 groups, and 49 of 49 and 50 of 50 for the
  ablations.

**Cell table, pick-and-place pairs** (Hb and Rb as in 5.6).

| arm | s known | s new | g known | g new | p_s known | p_s new | p_r new |
|---|---|---|---|---|---|---|---|
| Ac | 0.68 [0.42, 0.92] | 0.27 [0.14, 0.41] | 0.87 [0.71, 0.99] | 0.69 [0.52, 0.84] | 0.78 [0.54, 0.95] | 0.39 [0.18, 0.66] | 0.43 [0.25, 0.68] |
| Ac_nobear | 0.67 [0.47, 0.85] | 0.32 [0.11, 0.56] | 0.90 [0.75, 1.00] | 0.74 [0.55, 0.91] | 0.74 [0.55, 0.89] | 0.43 [0.15, 0.72] | 0.47 [0.21, 0.76] |
| Ac_noft | 0.67 [0.44, 0.87] | 0.34 [0.19, 0.53] | 0.92 [0.76, 1.00] | 0.78 [0.57, 0.94] | 0.73 [0.50, 0.91] | 0.44 [0.24, 0.71] | 0.57 [0.34, 0.83] |
| Ab (5.6) | 0.72 [0.51, 0.91] | 0.26 [0.09, 0.46] | 0.88 [0.70, 0.99] | 0.69 [0.53, 0.83] | 0.82 [0.67, 0.93] | 0.37 [0.13, 0.64] | 0.48 [0.25, 0.73] |

- Per seed, Ac's s(known) is 0.71, 0.69 and 0.62, and its s(new) 0.36, 0.21 and 0.23.

**Criteria.**
- **C1 holds:** +0.00 [−0.18, 0.19].
- **C2 fails:** 0.40 [0.19, 0.74].
- **C3 fails** on 4 of 6 pairings (table below). N2 and N3 now have CIs containing 0.
- **C4 holds:** +0.26 [0.13, 0.41] against Hb and +0.26 [0.14, 0.41] against Rb.

| pairing | g_o | g | p_t | p | predicted s | observed s | D | 1c gap |
|---|---|---|---|---|---|---|---|---|
| N1 cheese → plate | 0.53 | 0.53 | 0.97 | 0.56 | 0.52 | 0.30 | −0.22 [−0.35, −0.10] | 0.010 |
| N2 cheese → stove | 0.53 | 0.55 | 0.97 | 0.79 | 0.52 | 0.43 | −0.08 [−0.18, 0.01] | 0.001 |
| N3 cheese → cabinet | 0.53 | 0.52 | 0.84 | 0.68 | 0.45 | 0.35 | −0.10 [−0.21, 0.01] | 0.064 |
| N4 bottle → plate | 0.84 | 0.85 | 0.97 | 0.14 | 0.81 | 0.12 | −0.70 [−0.85, −0.54] | 0.102 |
| N5 bottle → stove | 0.84 | 0.85 | 0.97 | 0.37 | 0.81 | 0.32 | −0.50 [−0.65, −0.34] | 0.001 |
| N6 bottle → bowl | 0.84 | 0.85 | 0.59 | 0.10 | 0.50 | 0.08 | −0.42 [−0.62, −0.23] | 0.000 |

**Verdict (pre-registered order): "not compositional (fails: C2, C3)".**
- The canonical frame does not move the aggregate: Ac − Ab is −0.04 [−0.14, 0.04] on s(known) and +0.01 [−0.16, 0.14] on
  s(new).
- Neither ablation differs from Ac. Ac − Ac_nobear on s(new) is −0.05 [−0.23, 0.11], and Ac − Ac_noft is −0.08 [−0.19, 0.03].

**The support prediction, checked** (descriptive; `experiments/fa1c_diag.py`). In closed loop, Ac's onset gaps match the
prediction: 0.000-0.006 for N1, N2, N5 and N6, 0.065 for N3 and 0.088 for N4. An in-support onset is not enough, though. The
failures split by held object:
- **Cheese (N1-N3): the carry stops short.**
  - Every far release lands short of the target (overshoot 0-9%). The median end position is 0.17-0.38 m back along the bearing,
    near the cheese's start.
  - Placed given a hold is 0.56-0.81, about the same as the cheese's own known task K6 (0.59), which fails the same way (end
    0.12 m back).
  - The cheese loses or ends its hold early on known and new targets alike. That is a property of the held object, and the
    product g_o · p_t has no term for it: p_t was measured with the bowl, a secure hold.
  - In Proposition 3's terms this is (Suff): the place outcome depends on the held object beyond the foot.
- **Bottle (N4-N6): the carry overshoots or never releases.**
  - 27-43% of the failures are still held at the end.
  - Of the far releases, 63-73% end beyond the target (median 0.07-0.12 m along the bearing and 0.06-0.09 m lateral).
  - This happens even where the onset is in support (N5, N6).
- **Descriptive class.**
  - K7: 0 for every model (the bowl is grasped and the drawer never opened).
  - K8: 0.00-0.05.
  - N7: 0.
  - N8: Ac 0.15-0.20 and Ac_noft 0.45, against Ab's 0.00.

**What the result says.** The bearing and foot frame removed the support gap at the onset, as designed. The place term still
fails with an unseen held object. Two of its remaining inputs carry the held object's identity, which `experiments/fa1d_proxy.py`
measured on training states afterwards (5.8):
- the end effector's tilt, because the bottle is grasped tilted;
- the finger positions, because grip width depends on the object.

On a new held object those inputs are off-support. This is condition (B): a bypass of the role message.

**Deviations.** In the stage-1c training jobs, the job files and logs went to `.claude_tmp/pa1b`, not `pa1c`.

**Cost.** 0.40 A40-h (training). CPU: rollouts 3.75 CPU-job-h (10 jobs); support, swap, read and diagnosis about 0.15 h.

### 5.8 Stage 1d: cut the held-object bypass (protocol fixed 2026-10-31, before any stage-1d rollout; m3_prereg tag pa1d)

**Measured weakness (5.7).** With the onset in support, the place term still fails with an unseen held object. The bottle
overshoots or is never released. On training states (`experiments/fa1d_proxy.py`, results/flawed/a1d_proxy.json), two of Ac's
target-term inputs decode the held object (nearest-centroid accuracy on held demo rows, demos 40-49 held out; chance 0.44):
- the end-effector tilt: 0.82;
- the finger positions: 0.81;
- the canonical foot: 0.62. Its trajectory differs by task.
- the last command: 0.42.

On a new held object, the tilt and finger inputs take values never paired with that target. For example, the bottle is grasped
about 28° tilted, and grip widths are 0.002 (bowl), 0.009 (bottle) and 0.021 (cheese).

**Principle.** Condition (B). No input to the place stage may carry the held object's identity beyond what its declared role
message carries, and the declared message is the foot. Proposition 2 names the risk: if the place term needs what those inputs
carried, cutting them moves the debt into insufficiency, and that would show in C1.

**The change (one, structural).** Arm Ad is Ac with the end-effector tilt and the finger positions removed from the target term.
The target term now reads only:
- the foot, in the canonical frame;
- the last command;
- the target slot's identity, size and drawer opening.

The object term, the contact gate and the frame are unchanged.

**Proposition 3 for Ad.**
- (i)-(ii) as before, and (i) gains: in the place phase the chunk does not depend on the end-effector rotation or the finger
  positions.
- (iv) now needs (Supp), which is the carry distance (5.7), and (Suff), which is physics only, because no input of the place
  stage is an identity proxy any more.
- The remaining object-specific term is physical: whether the object stays in the hand during the carry. 5.7 measured it for the
  cheese, on its known task as well.

**Arms.**
- Ad with seeds 0, 1 and 2.
- Ablations, seed 0:
  - Ad_tilt: Ad plus the tilt;
  - Ad_fing: Ad plus the fingers.
- H = Hb and R = Rb are 5.6's runs. Ac (5.7) and Ab (5.6) are shown beside them.
- Trained with `experiments/fa1c_train.py --tag d`, recipe unchanged and not tuned. Rollouts use `experiments/fa1d_roll.py`,
  which is 5.7's protocol plus one logged field: the closest the held object comes to the target while held.
- The read is `experiments/fa1d_read.py`, using the 5.6 / 5.7 engine. C1-C4, the classes, the CIs and the verdict order are
  5.6's, with A = Ad.

**Swap test** (`experiments/fa1d_swap.py`, training states): T1-T4 as in 5.7, plus T5. T5 replaces the end-effector rotation
and finger positions with those of a held row with another object. Ad must show 0 on T1-T3 and T5, and round-off on T4.
- Result before the rollouts (results/flawed/a1d_swap.json): Ad gives 0 on T1-T3 and T5, and at most 7e-6 on T4.
- On T5, Ac changes by a mean of 0.34 (max 2.35), Ad_tilt by 0.24 (2.59) and Ad_fing by 0.17 (2.05).

**Descriptive, never changing the verdict: the three-factor split** of Proposition 3 (iii). For each new pairing, the predicted
s3 is h_o · a_o · q_t, each factor taken from the known tasks:
- h_o = P(a hold is reached | o);
- a_o = P(the held object comes within 3 cm of the target while held | hold, o), measured on o's known tasks;
- q_t = P(success | arrived, t), measured on t's known tasks.

It is reported against the observed s, with a start-state CI. This version moves the carry's object-specific physics into the
object factor, where (Suff) says it belongs.

**Predictions.**
- If the bypass caused the bottle failures, the bottle pairings N4-N6 improve. N4 stays limited by its carry distance, which is
  out of support.
- The cheese pairings stay limited by the cheese's hold (5.7). So two-factor C3 is expected to keep failing where p_t was
  measured with a secure-hold object, while D3 may contain 0.
- C1 is at risk: this is Proposition 2's relocation warning.

**Budget.** 1 A40-h.

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

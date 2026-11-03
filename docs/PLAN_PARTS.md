# Plan: isolate the flaw architecturally, resolve it by sparse structural composition

Adopted by the user 2026-11-01 (decisions in §10). Each layer is pre-registered before its first read.
It supersedes the prototype-A round loop of FLAWED_COMPOSITION.md §5.5-5.32. Those rounds stay on record.

## 0. Why a new plan

**Direction (user, 2026-11-01).**
- Isolate the flaws in VLAs architecturally, then resolve them through structural composition.
- Structural composition should let us create parts and combine them in very sparse, methodical ways.
- Be honest about supervision, and stop converging on hacks.

**What prototype A established (rounds 1-1p). These results stay.**
- **Role locality and target-blind grasping hold exactly, by construction.** The swap tests are 0 in every round, and grasp
  identity is exact in both layouts.
- **Composition held.**
  - Since 1i, new-pairing success follows the three-factor law on most pairings.
  - Within each object, new pairings run at the known rate: object-matched C2 is 0.96-1.12 in 1o and 1p.
  - The familiar layout read "compositional" in 1k, 1l, 1o and 1p.
- **The baselines did not compose even with the same oracle slots** (stage 1, A4).
  - Holistic (H) and role-routed (R) baselines fail on new pairings: relaxed place success drops by 0.78 and 0.73.
  - Retrained on the same generated data (1k), they score 0.00-0.03 on new pairings.
  - So the failure is in how the policy combines information, not in what it perceives. This is the architectural isolation
    we want, but so far it is shown only for small oracle-state policies.

**What went wrong.**
- **Rounds 1k-1p chased a pooled success ratio dominated by one weak part:** the cheese grasp beside the wine rack.
  - Occupancy and free-space inputs, grasp height, clearance-stratified data and grasp-option teachers all target that grasp.
  - That is executor and teacher engineering for one LIBERO scene, not composition.
  - The theory predicts the failure: composition makes a combination as good as its parts. It cannot make a weak part strong.
- **Nothing in prototype A is unsupervised.** The oracle sources:

  | component | source |
  |---|---|
  | object and target slots (identity, pose, size) | simulator state |
  | which entity fills which role | the task definition |
  | frames | simulator poses |
  | phase gate | simulator contact |
  | generated carries | a scripted planner on privileged state, with grasp options picked by simulator trial |
  | the object / target split itself | designed by us |

**Decision for the record.** §5.32's option (c): the near-rack grasp is a part-competence limit of the scripted teacher. The
frozen pooled C2 is reported as failing on the new layout, with the object-matched analysis beside it.

## 1. Claims (each can fail)

| | claim | evidence that would refute it |
|---|---|---|
| **K1 Flaw** | Trained VLAs (pi0.5, GR00T) mix role information densely, at locatable components. That mixing predicts and causes their new-pairing failures: the "where → grasp, what → place" split of N2. | No component carries cross-role mixing beyond noise; or mixing does not predict failure; or purifying the located component does not help |
| **K2 Structure** | A part library with a fixed, sparse combination rule removes the mixing by construction. New-pairing success then equals the product of part competences. | Locality not exact; or success departs from the product beyond the pre-registered tolerance |
| **K3 Creation** | Training one new part from one task's demos, with everything else frozen, solves that part on every partner at once. The cost is per part, not per task, and old tasks are untouched. | The created part works only with the partner it was trained with; or baselines reach the same pairings at equal cost |
| **K4 Unsupervised** | The structure survives replacing each oracle piece with a signal from the VLA itself. | Composition measures degrade beyond tolerance at a rung; we report where |

## 2. Definitions (fixed before any data)

- **Atom:** one filler of one role, such as "cheese" for the object role or "plate" for the target role.
- **Part:** a small module bound to one atom of one role. It reads only its own role's input, in its own frame.
- **Library:** the set of parts, plus one frozen backbone per role shared by that role's parts.
- **Combination rule Φ:** fixed in advance, with no learned cross-role term. Here it is prototype A's gated sum of velocity
  fields:
  v = (1−c)·T_o⁻¹ v_o(·) + c·T_t⁻¹ v_t(·)
- **Sparsity:**
  - a task activates exactly one part per role;
  - a part is at most a fixed parameter size, set in §4.1;
  - every number reported counts the parts used per task and the parameters added per part.
- **Composition measures:** these decide K2 and K3.
  - Locality: swap tests.
  - Multiplicativity: the three-factor law, per pairing.
  - The object-matched ratio.
  - The creation transfer ratio: (created part with unseen partners) / (created part with its training partner).
- **Competence measures:** reported, never deciding a composition claim.
  - Per-part success: hold, arrival, place on arrival.
  - Pooled success.

## 3. Layer 1 — isolate the flaw in real VLAs (K1)

These checkpoints already exist and are scored on the N2 scene: pi0.5 (pi05 LIBERO) and GR00T. SmolVLA and X-VLA come from
D2 if time allows. All interventions run on the prefix KV and expert-read tools from D1/D2 (§4.5-4.8). Demos are scored
only.

**F1 — role-mixing map (open loop, no rollouts).**
- **Matched pairs** from N2's task set:
  - a pair that changes only the target atom: same object, same layout;
  - a pair that changes only the object atom.
- **Patching.** At each component, patch the target-changed run's activations into the base run:
  - the vision tokens of the target, the instruction's target words, each VLM layer's KV, and each action-expert layer;
  - measure the change in the **grasp-phase** action chunk. Reading the target into the grasp is cross-role mixing;
  - symmetrically, patch object-changed activations and measure the **place-phase** chunk.
- **Output:** a mixing profile over components, with task-bootstrap CIs, per model.
- **Also:** the same profile for prototype A's H and R arms, as known-flawed references, and for A, as a known-clean
  reference. A should be 0 by construction.
- **Cost:** about 1-2 A40-h.

**F2 — does mixing predict failure (no new rollouts).**
- Per task: does the located mixing predict new-pairing failure in N2's recorded rollouts?
  - target-into-grasp mixing should predict g loss;
  - object-into-place mixing should predict p loss;
  - rank correlation, task-bootstrap CI.
- Pre-registered with one verdict mapping, before F1 is read.
- **Cost:** about 0.

**F3 — causal test: purify the located component.**
- At the component F1 locates, replace the new-pairing run's role message with a pure one:
  - the message from a known task that shares the same atom in that role;
  - the D1x recombination machinery already does this.
- Read open loop first: grasp-site and release-site error on dev demos (scored only).
- Then a small closed-loop read on N2's new pairings: 10 episodes per task.
- If purifying component k restores placement, the flaw is located at k. That is the architectural isolation.
- **Cost:** about 2-3 A40-h.

**Go / kill:**
- **Go to Layer 2** if F1 finds mixing above the A reference's noise floor at some component, and F2 or F3 ties it to
  failure.
- **Kill K1** if neither ties it. We would then report that VLA new-pairing failure is not a combination flaw, which is
  itself a result against the theory.

## 4. Layer 2 — the structural fix: a sparse part library (K2, K3)

### 4.1 Architecture P (parts)

- **Backbones.** Prototype A's object and target velocity fields become two frozen role backbones, trained once on the
  training atoms.
- **Parts.** Each atom gets a small part module: a low-rank adapter on its role backbone plus its slot embedding.
  - The size cap is fixed before training: rank 8 on the backbone's hidden layers.
  - A task activates its object's part and its target's part, and nothing else.
- **Combination.** Φ as in §2, fixed.
- **Training data: demos only (fitted, never executed).** The generated carries of 1g-1p are dropped from the main arm.
  - Layout generalization must come from frame equivariance, which is structural, not from coverage data.
  - A generated-data arm is kept as a labelled ablation.
- **Removed.** None of the 1l-1p executor devices are carried over: occupancy, free-space, clearance strata, grasp-option
  teachers.
- **Stays (for now).** The oracle slots, frames and contact gate stay until Layer 3 removes them.

### 4.2 Tests

**S1 — locality (structural).**
- Swap tests on P: exactly 0 by construction.
- F1's mixing profile on P: 0.

**S2 — multiplicativity on held-out pairings.**
- Train backbones and parts on the known pairings.
- Test the new pairings with the frozen criteria:
  - the three-factor law per pairing;
  - the object-matched ratio ≥ 0.8, CI lower bound ≥ 0.6.
- Pooled success is reported as competence.

**S3 — creation (the central test of sparse composition).**
- **Held-out atoms.**
  - Hold one object atom out of all training.
  - Then create its part from k demos of **one** task that uses it, k ∈ {1, 5, 10, 50}.
  - Backbones and every other part stay frozen.
- **Read.** The new part with **every** target, in the familiar and new layouts.
  - **Prediction:** success ≈ (new part's hold × arrival) × (each target's place-on-arrival), measured on the training
    partner. Its transfer ratio is ≥ 0.8.
- **Symmetric test:** hold out a target atom and create its part.
- **Cost accounting:** parameters added, demos, GPU-seconds to ready, and the change on all old tasks. The change on old
  tasks must be exactly 0, since nothing else is touched.

**S4 — baselines on S3's protocol.** Each gets the same k demos and the same new atom:
- holistic policy fine-tuned;
- holistic policy with a per-task LoRA;
- role-routed policy (R) fine-tuned;
- a DMPEL/CLARE-style expert-library policy: a learned mixing router, the closest published "compose library experts" foil.

They report success on the trained pairing and on the unseen pairings, and the forgetting on old tasks.

### 4.3 Scope

- **Phase 1.** Start in the N2 scene, where prototype A's code, data and evaluation states exist. That gives a small atom
  set: cheese, bottle and bowl as objects; plate, rack, cabinet and bowl as targets.
- **Phase 2.** If S3 holds, widen to a LIBERO-90-C object × target set with at least 6 objects and 5 targets, so creation
  is tested on several held-out atoms rather than one.

**Go / kill:**
- **Go to Layer 3** if S1 is exact, S2 holds, and S3's transfer ratio is ≥ 0.8 with CI lower bound ≥ 0.6 for k ≤ 10.
- **Kill or revise K3** if created parts transfer only to their training partner. That would mean the part learned a
  pairing, not an atom.

## 5. Layer 3 — remove supervision one rung at a time (K4)

**Protocol.**
- Each rung replaces one oracle piece and keeps every other piece at the previous rung.
- At each rung, re-run S1-S3 at fixed cost.
- The tolerance is fixed before any rung is read: each composition measure stays within 0.10 of rung L0.
- The claim is whatever survives. Where a rung breaks, we report exactly which piece is needed.

| rung | replaces | with | source in this repo |
|---|---|---|---|
| L0 | — | oracle (Layer 2 as built) | prototype A |
| L1 | sim-contact phase gate | gripper width + proprioception event | contact events from proprioception (§5.2) |
| L2 | oracle frames | object position from the VLA's visual tokens (pi0.5 attention / token grouping) | S0 KV tools |
| L3 | oracle slots | object-centric slots grouped from pi0.5 features | stage 2 of prototype A |
| L4 | given role assignment | roles induced from contact events: words predicting the grasp site are object words, words predicting the release site are target words | §5.3 |
| L5 | designed part partition | the partition induced from data: which demos share a part, chosen by leave-layout-out prequential MDL over candidate partitions | §3.4, B1.3 |

L5 is the fully unsupervised claim: the parts and their combination are found from the demos. L1-L4 are partial claims, and
each is reported as such.

## 6. Pre-registration and criteria

- **Each layer has its own pre-registration,** written before its first read, with one verdict mapping. Criteria are fixed
  in this plan and frozen thereafter. There are no post-hoc criterion changes; concerns go in the doc only.
- **Composition claims are decided by composition measures (§2).** Competence is reported separately. Choosing this before
  any Layer-2 data exists is not a post-hoc change.
- **Statistics.**
  - Task-bootstrap CIs and paired episodes.
  - 3 seeds × 20 states per cell in development, 50 states in confirmatory reads.
  - Holm correction over at most 6 confirmatory hypotheses.
- **Every confirmatory claim gets one fresh-seed read with no changes.**

## 7. Order, cost and decisions

| step | content | A40-h | decides |
|---|---|---|---|
| F1 | role-mixing map: pi0.5, GR00T, plus A/H/R references | 1-2 | where mixing lives |
| F2 | mixing → failure, from N2 rollouts | ~0 | whether it predicts |
| F3 | purify the located component | 2-3 | whether it causes (K1) |
| P-build | backbones + part modules from prototype A code | 0.5 | — |
| S1-S2 | locality, multiplicativity on held-out pairings | 0.5-1 | K2 |
| S3-S4 | creation k ∈ {1, 5, 10, 50}, both roles, plus baselines | 2-3 | K3 |
| S-conf | fresh-seed confirmatory read of S2-S3 | 1 | K2/K3 confirmed |
| Phase 2 | LIBERO-90-C atom set, S2-S3 | 4-6 | generality |
| L1-L5 | supervision ladder | 8-15 | K4 |

- **Layers 1-2 in the N2 scene:** about 8-11 A40-h.
- **With phase 2 and the ladder:** about 20-30 A40-h.
- **Budget.** The 6.2 A40-h prototype cap is retired; the user set no cap for this plan (§10).
- **Rollout CPU** stays on tail-lab or overcap, as before.

## 8. Novelty and prior work

**To credit.**
- Factored Diffusion Policies (2605.22596) and composable diffusion (2206.01714): additive score/velocity composition.
- OA-WAM (2605.06481): role-routed heads.
- DMPEL (2506.05985), CLARE (2601.09512) and MergeVLA (2511.18810): expert and adapter libraries with learned routers or
  merges.
- Interchange interventions (Geiger).
- Wiedemer and Lippl & Stachenfeld: the support and additivity conditions.

**Likely ours.**
- The flaw located per component in VLAs by role-mixing interventions, and shown to cause the N2 failures (Layer 1).
- Relocation (Proposition 2), with C1 and rounds 1d-1i as evidence.
- Creation of one atom's part that transfers to every partner under a fixed, sparse rule, with zero change to old tasks and
  multiplicativity verified. The libraries above route or merge learned experts per task. None fixes the combination rule,
  and none tests transfer of a created part to unseen partners.
- The supervision ladder, with the composition measures read at every rung.

## 9. Risks

- **F1 finds mixing everywhere,** at a level comparable to noise. Mitigations:
  - the A reference sets the noise floor;
  - read grasp-site and release-site error, not flow loss alone.
- **Frozen backbones may be too weak for new atoms,** in which case creation fails for capacity reasons. S3 reports k-curves,
  and one larger part size is pre-registered as a secondary arm.
- **The N2 scene has few atoms.** Phase 2 exists for this reason.
- **Rungs L2-L3 depend on pi0.5 features localising objects.** Earlier work found VLAs under-use language, but not vision.
  F1's vision-token profile tells us early.

## 10. Decisions (user, 2026-11-01)

1. **Budget:** no fixed cap; use overcap as needed. Phase 2 and the ladder are committed.
2. **Part size and k grid:** rank-8 adapters; k ∈ {1, 5, 10, 50}. One larger part size stays a secondary arm (§9).
3. **The DMPEL/CLARE-style library baseline is required** in S4.
4. **Phase 2 uses LIBERO-90-C.**

The plan is adopted. Each layer is still pre-registered before its first read (§6).

5. **(User, 2026-11-03) An explicit sparse interface between roles.** PARTS_RESULTS §37 found that a demo's grasp
   depends on its target. The bottle is grasped with the hand turned about 81° for the rack, and unturned for the cabinet.
   The target-blind object part averages the two modes. The user chose to amend §2 as follows. This is a plan-level change
   made after seeing data, and is labelled as such everywhere.
   - **Interface message m.** The target part emits one discrete message m, a "grasp mode" token. It is the only
     cross-role channel. The object part reads m as one extra input. Φ is otherwise unchanged.
   - **Inventory.** The set of values m can take is induced from the training demos alone, for example by clustering
     pre-close hand yaw in the object frame. Its size is chosen by a rule fixed before any read, such as description length
     or BIC. It is not hand-set per scene.
   - **Which value a target emits.** Each target part's value of m is fitted from its own demos. A created part fits its
     value of m from its own k demos.
   - **Bandwidth.** The bandwidth of m, in bits, is a reported composition quantity. The claim becomes: composition with a
     declared interface of b bits, and b is reported.
   - **Locality (S1) is restated.** Swapping the target changes the object term only through m. With m held fixed, the
     swap tests must still be exactly 0. Any change to the object term by a route other than m counts as mixing.
   - **Unchanged.** All other criteria, including the pp_l2 GO mapping, stay frozen.

6. **(User, 2026-11-03) Role-level transport.** This is a plan change made after seeing data, labelled as such everywhere.
   - **Finding.** PARTS_RESULTS §40 showed that the bottle's demos start 3-7 cm from it and contain no level transport, so
     far starts make the bottle part descend early.
   - **Amendment to §2 and Φ's object term.** A part acts only where its own demos have support (condition S). Outside that
     region, the role backbone acts alone. The backbone is trained on every object atom's demos, and it reads the same frame
     and the same message m.
     - The switch between backbone and part is a support test fitted from the part's own demos, for example the start-offset
       radius under §4.1's existing 99th-percentile rule. It is not hand-set per scene.
     - The blend inside the switch is fixed in advance.
     - A created part fits its support region from its own k demos.
   - **The hierarchy this makes explicit.** The role backbone holds the role-general skill (transport to the object). Parts
     hold the atom-specific skill (the grasp).
   - **Unchanged.** There is still no cross-role term beyond m, and every frozen criterion stays as it was.
   - **Removal of earlier devices.** Where this switch subsumes an earlier device (rclip and its relocation of the approach
     state), the round removes that device and measures the removal.

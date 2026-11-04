# A sparse part library for robot policies: Layer 2 report

State of the work on 4 November 2026. Unless marked otherwise, every number describes the current system on development data: seeds 10-12, N2 start states 4-23, 60 episodes per task. Sources are listed in §8.

## Summary

We ask whether a robot policy can solve a new object-target pairing by combining separately learned parts, the way language combines words. We built a library with one small part per object and per target, combined by a fixed rule with one declared message between the two roles. We test it in one LIBERO scene, with simulator state as input.

**The pre-registered gate for this layer is false.** The gate requires composition in a familiar layout and in a new layout, and the current system meets its measures in neither layout in full. It comes close in the familiar layout only, on development data, after 16 post-hoc rounds.

- **Familiar layout.** With the declared message held fixed, swapping a part changes only its own role, exactly. New pairings run at the known pairings' rate (object-matched ratio 1.09 [1.04, 1.14]). The three-factor law holds on five of six new pairings. Parts created from 1, 5 or 10 demos reach partners they never saw in all 6 cells tested, and leave every other parameter bitwise unchanged.
- **New layout.** Known-pairing success falls from 0.90 to 0.46. The law fails on two of six pairings, creation passes 2 of 6 cells, and one task (bottle on the rack) scores 0 of 60.
- **Baselines.** Four baselines reach at most 0.06 on unseen pairings in the familiar layout, on a fresh-state read that compared them with the original rank-32 library. We did not re-run them against the current system.
- **The cause of the new-layout shortfall is only partly known.** Generated demos raise competence on the pairings they show, but a registered test of coverage as the whole explanation returned not supported.

Nothing here is unsupervised. The system receives simulator slots, frames, sizes and role assignments (§6).

## 1. Claims

Layer 1 of this project examined trained vision-language-action policies (pi0.5, GR00T). It located dense mixing of role information, but failed to show that the mixing causes their failures. Layer 2 builds the separation directly and tests two claims:

- **K2, structure.** A part library with a fixed sparse combination rule removes role mixing by construction, and success on new pairings equals the product of part competences.
- **K3, creation.** Training one new part from one task's demos, with everything else frozen, solves that part on every partner at once, and leaves old tasks untouched.

## 2. The system

The library has two role backbones (object, target) and one low-rank part per atom, where an atom is one filler of one role, such as "cheese" or "plate". A task activates exactly one part per role. A fixed gated sum combines the two velocity fields:

v = (1 − c) · T_o⁻¹ v_o + c · T_t⁻¹ v_t

Here c is the contact gate, and T_o and T_t are the object and target frames. The only channel between roles is the declared message m below. Before the hold, the object term alone moves the hand. After it, the target term alone moves the held object.

**The current system, P_min,** adds these to the original design:
- Part rank 32. The plan's primary size was rank 8, and we raised it after the first read. A rank-32 part has about 83,000 parameters, in a library of about 1.78 million (figures from the original rank-32 library).
- One 1.58-bit grasp-mode message m, from the target part to the object part (1.06 bits of task entropy). A demo's grasp depends on its target, so the target-blind object part averaged two grasps without it. This amends the plan's locality test: the object term may change with the target only through m.
- A role backbone that transports the hand outside a part's fitted start radius. It reads neither m nor object size.
- A carry scale frame frozen at the hold's onset, support radii fitted per part, and decoupled training of the two role terms.

The message, the backbone changes and the scale frame came after seeing data, over 16 rounds, and the record labels each as post hoc. A registered leave-one-out then tested five options: it dropped two (rtclip, aframe) and kept three (tscale, bbnom, bbnosize). We did not ablate the other options (psupport, imode, rtransport), the decoupling or the rank.

**Training data** is demos only, 50 per task over 8 known tasks, fitted and never executed.

## 3. Tests and criteria

We fixed these before the first read, apart from the restated locality test above. All are composition measures, and competence is reported separately.

| test | measure | criterion |
|---|---|---|
| S1, locality | swap tests and target-into-grasp mixing, with m fixed | exactly 0 |
| S2, multiplicativity | three-factor law (3F): observed minus predicted, per new pairing | every 95% CI contains 0 |
| S2, object-matched | new pairings' success over known pairings' success, within each object (OM) | ratio ≥ 0.8, lower bound ≥ 0.6 |
| S3, creation | transfer ratio: created part on unseen partners over its training partner, k = 1, 5, 10 demos | ratio ≥ 0.8, lower bound ≥ 0.6 |
| S4, baselines | the same protocol on four baselines | descriptive |

**The three-factor law** predicts s(o, t) = hold(o) × arrival(o) × place(t). We pool an object's factors over its known pairings, and take a target's place rate from its known pairings. The test counts a miss in either direction as a failure.

**The gate.** The next layer requires S1, S2 and S3 to pass, and the plan keeps the new layout inside the claim.

**Scored tasks.** Known pairings K1-K6: bowl on plate, stove and cabinet; bottle on cabinet and rack; cheese in bowl. New pairings: N1 cheese on plate, N2 cheese on stove, N3 cheese on cabinet, N4 bottle on plate, N5 bottle on stove, N6 bottle in bowl. The library also trains on a drawer task and a plate-pushing task, which the gate does not score.

## 4. Results

| measure | familiar layout | new layout |
|---|---|---|
| Known-pairing success | 0.90 | 0.46 |
| New-pairing success | 0.92 | 0.37 |
| OM | 1.09 [1.04, 1.14], passes | 1.08 [0.94, 1.29], passes |
| 3F | passes on 5 of 6 | passes on 4 of 6 |
| S3 | passes 6 of 6 cells | passes 2 of 6 cells |

S1 is exact with m fixed at each of its three values, on every seed. The gate is false.

### Three-factor law

Predicted and observed success, with D3 = observed − predicted. Bold marks a failure of the criterion.

| pairing | familiar: pred / obs | familiar: D3 [95% CI] | new: pred / obs | new: D3 [95% CI] |
|---|---|---|---|---|
| N1 cheese on plate | 1.00 / 0.97 | −0.03 [−0.08, 0.00] | 0.22 / 0.35 | +0.13 [−0.01, 0.28] |
| N2 cheese on stove | 1.00 / 1.00 | 0.00 [0.00, 0.00] | 0.45 / 0.38 | −0.06 [−0.19, 0.04] |
| N3 cheese on cabinet | 1.00 / 1.00 | 0.00 [0.00, 0.00] | 0.48 / 0.42 | −0.07 [−0.18, 0.02] |
| N4 bottle on plate | 0.82 / 0.78 | −0.03 [−0.17, 0.08] | 0.10 / 0.27 | **+0.16 [0.04, 0.30]** |
| N5 bottle on stove | 0.82 / 0.95 | **+0.13 [0.06, 0.21]** | 0.21 / 0.48 | **+0.28 [0.18, 0.38]** |
| N6 bottle in bowl | 0.79 / 0.80 | +0.01 [−0.10, 0.12] | 0.21 / 0.32 | +0.11 [−0.02, 0.23] |

Every failure is an observed rate above its prediction. The bottle's predictions pool its two known pairings, which run under different grasp modes, and the rack task's zero in the new layout lowers its pooled arrival rate (§6).

### Creation

Transfer ratio [95% CI]. Bold marks a cell that meets the criterion. We hold each atom out of training and create its part from k demos of one task. Other parameters stay bitwise unchanged in every creation.

| atom | k | familiar | new |
|---|---|---|---|
| bottle | 1 | **0.91 [0.85, 0.96]** | **0.82 [0.71, 0.95]** |
| bottle | 5 | **0.88 [0.80, 0.94]** | 0.75 [0.61, 0.87] |
| bottle | 10 | **0.88 [0.80, 0.95]** | 0.74 [0.62, 0.87] |
| plate | 1 | **0.91 [0.84, 0.97]** | 0.79 [0.53, 1.15] |
| plate | 5 | **0.92 [0.85, 0.99]** | 1.00 [0.52, 2.08] |
| plate | 10 | **0.93 [0.88, 0.97]** | **1.44 [0.89, 2.78]** |

A ratio says a part transfers at its training partner's rate, and that rate can itself be modest. The created bottle at k = 5 succeeds at 0.72 with its training partner and 0.63 on unseen partners in the familiar layout. The new-layout ratios rest on low success and carry wide intervals. The plate's k = 10 ratio exceeds 1 because its training partner scores only 0.27 there.

### Baselines

Holistic fine-tuning, a per-task LoRA, a role-routed fine-tune and a DMPEL/CLARE-style learned-router library each receive the same k demos and oracle inputs.

| read | compared with | best on unseen pairings |
|---|---|---|
| Seeds 10-12, 50 fresh states | original rank-32 library | 0.06 familiar, 0.01 new |
| Seeds 0-2, N2 states 4-23 | original rank-8 library | 0.09 (router library, k = 50) |

The two fine-tunes and the router library also lose the old tasks on the fresh-state read, falling from 0.68-0.89 to 0.00-0.06. All methods used one adaptation recipe, fixed in advance. We did not test whether a replay or smaller-step recipe forgets less. Forgetting confounds those three methods' scores on unseen pairings. The per-task LoRA does not forget and reaches 0.00-0.02 on development seeds. Prior work has already shown modular policies composing on state input, so this part is largely a replication. No test has yet measured what a created part adds beyond the backbone, or how a scripted servo scores on the same states (docs/ADVERSARIAL_STUDY.md).

## 5. Why the new layout falls short

**Demo coverage explains part of the competence gap.** We assembled a demo-support account from the rounds' diagnostics, then tested it post hoc on a labelled system with 50 generated demos per known task, spread over the new layout's object positions. The registered verdict is not supported.

| success, new layout | P_min | with coverage | change [95% CI] |
|---|---|---|---|
| Known pairings | 0.46 | 0.64 | +0.18 [0.13, 0.24] |
| Bottle on rack | 0.00 | 0.60 | [0.45, 0.75] on the rate |
| New pairings | 0.37 | 0.40 | +0.03 [−0.06, 0.13] |

The gain is pooled and uneven: bottle on cabinet falls from 0.47 to 0.38, and cheese in bowl barely moves. 3F still fails, on a changed set of pairings (N1, N5, N6) that includes shortfalls. This is limited evidence against coverage, because the generator is a weak teacher (it keeps 0.45-0.88 of its attempts) and the plate's creation intervals are wide. Generated rows also cost familiar bottle hold at m = 0, from 0.95 to 0.73, or to 0.85 with the original grasp-mode fit.

**Placing differs by pair.** The law assumes one place rate per target. Three post-hoc registered diagnostics on P_min examined this, on CPU and with no new models. Rates in this part are place given arrival.

- **The target term cannot tell objects apart.** It has no input naming the held object, and swapping the object changes its output by exactly 0.
- **Landing physics inside 2 cm does not separate the objects.** Dropped upright with no hand inside 2 cm of the target, every object succeeds, with one exception (the bottle over the bowl's centre at the lower height). The objects do differ at 3 cm, and the drop says nothing about a tilted or gripped object.
- **Bowl on plate in the new layout: this gap bears on the shortfall, and its cause is unknown.** Cheese on plate exceeds bowl on plate by +0.33 and +0.37 in two traced runs, and by +0.45 [0.11, 0.78] on the 20 episodes where both arrive. Arrival radius, landing physics, tilt at arrival, held rotation and start-state selection do not account for it. Many failing episodes reach the radius, never descend, and end still holding the bowl. Bowl on plate succeeds in 60 of 60 familiar episodes.
- **The bottle in the familiar layout: a separate effect.** Zeroing rotation commands while held raises place for bottle on plate from 0.82 to 0.93 and for bottle in bowl from 0.86 to 0.98, and closes both registered gaps. Bowl and cheese tasks change by at most 0.02. The effects are +0.11 [0.02, 0.22] and +0.13 [0.02, 0.25], uncorrected for multiplicity. This fits the target term applying rotations learned on other objects to the bottle, but no test isolates that mechanism. Zeroing is a diagnostic and no remedy: in the new layout it does not help, and the bottle's point estimates fall (0.57 to 0.50, 0.70 to 0.42).

The other two new-layout place gaps we examined are not established.

**A possible consequence, untested.** The plate's place rate comes from bowl on plate alone, so a weak K1 lowers the N1 and N4 predictions. The bottle's pooled factors also lower N4's. The sizes of these two routes remain unmeasured.

## 6. Limits

**Supervision.** The simulator and task definition supply object and target slots (identity, pose), role assignment, frames, rest height, drawer opening, the contact gate and demo labels. A hand-set table supplies sizes. We designed the object/target split and the one-part-per-atom partition. We fitted support radii from simulator positions, and the m inventory from hand yaw grouped by target label.

**One small scene.** Results cover one LIBERO scene, three scored object atoms and two held-out atoms for creation. The rack does not appear among the bottle's unseen partners in S3. They say nothing yet about a wider atom set.

**Development reads.** P_min has no confirmatory read: the plan requires the user's approval for each, and the gate is false. Seeds 10-12 were fresh for the one confirmatory read so far, and P_min's builds have since reused them with the N2 states. That read used the original rank-32 library with the user's approval. It failed S2 in both layouts and met S3 in 3 of 12 cells, with bottle on plate at 0.28 in the familiar layout. P_min's rate on that pairing is 0.78 on different states, and we cannot say how much of the difference would survive a fresh read.

**Choices made after seeing data.** The rounds tuned P_min on the same development states these numbers come from, so its familiar-layout results may be optimistic.

**Rollouts are not bit-reproducible across hosts.** A re-run of the same policy agreed with the first run on 88-100% of episodes per task, and one gap moved across its CI boundary between runs. The bootstrap does not model this, so CIs that barely exclude 0 deserve caution.

**The estimator.** Each target's place rate rests on one known partner, except the cabinet (K3, K4), which enters only N3. Pooling the bottle's factors across grasp modes also distorts 3F. Conditioning on m (post hoc, no verdict) takes familiar N5 from +0.133 to 0.000, shrinks new N4 and N5 to +0.061 and +0.068, and moves familiar N4 to −0.167 [−0.317, −0.050] and N6 to −0.118 [−0.234, −0.021]. So familiar N4 and N6 may pass the frozen law partly because two errors cancel. The frozen estimator decides every verdict in this report.

## 7. Conclusion

A fixed sparse combination of small parts meets most composition measures in the layout its demos cover: locality is exact with the message fixed, created parts transfer from few demos at their training partner's rate, old tasks stay bitwise unchanged, and the baselines tested reach almost none of the same pairings. This evidence is development data on a system tuned on those data, and it does not confirm K2 or K3. The gate fails, and the new layout is where it fails most: frame-based relocation did not deliver generalization from demos alone. Coverage raises competence on the pairings it shows and leaves new pairings about where they were. Whether the law needs a pair-level term at the place stays open, and bowl on plate in the new layout is the unexplained case. A separate track on removing supervision started on the familiar layout by the user's decision. One rung holds, a second is recorded as breaking on familiar N4, and the track is paused.

## 8. Sources

| content | file |
|---|---|
| Plan, frozen criteria, decisions | docs/PLAN_PARTS.md (§2, §4, §10) |
| Rounds, consolidation, Layer-2 summary | docs/PARTS_RESULTS.md (§5, §9, §52-54) |
| P_min's read | results/flawed/pp_cons_min_read.json |
| Coverage test | docs/COVERAGE_RESULTS.md (§3) |
| Place diagnostics | docs/PLACE_DIAG.md (§3, §5-7) |
| Supervision ladder | docs/LADDER_RESULTS.md |

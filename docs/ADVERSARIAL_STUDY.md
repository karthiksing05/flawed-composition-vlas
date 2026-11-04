# Adversarial study of the claim (PLAN_PARTS §10 item 17)

On 2026-11-04 the user asked to pause, rethink the claim and run an adversarial study. Three independent reviewers each attacked one part of the project with no access to my conclusions: the VLA flaw claim (K1), the part library claims (K2, K3 and the wider inference), and the literature. Reviewers read docs and source only, with no results files and no compute. I checked the charges marked "checked" against the source myself. The literature reviewer read most papers through a summarising tool, many at abstract level only, and most are preprints. I have not re-read those papers.

## 1. Verdict

**K1 as written should be withdrawn.** Its causal clause is unsupported, its predictive clause rests on a confounded correlation, and its mixing measure does not separate a flaw from legitimate behaviour.

**K2 is not an empirical law as stated, and K3 holds only in a weak form.** Much of the positive result follows from construction and oracle inputs, and no experiment has measured how much.

**The inference "VLAs lack this structure" is not licensed.** No experiment connects the library to a VLA, and published work rescues dense VLAs without structure.

**The test I proposed should not run as proposed.** It cannot fail informatively, and it is aimed at the grasp, which is not what fails.

## 2. Charges against K1

| severity | charge | evidence | status |
|---|---|---|---|
| Fatal | Layer 1's GO joins two different directions. F1 clears its floor only on target-into-grasp (floor 0.000). F2 ties only on object-into-place, where the located component's lower bound 0.13 sits far below the floor 136.1. Target-into-grasp mixing does not predict grasp loss (ρ −0.08) | PARTS_RESULTS §2; `pp_f1_read.py` (`any`, `or`) | checked |
| Fatal for object-into-place | Healthy systems score higher than the VLAs: A 62, P 111-120, against 1.88-1.98 | PARTS_RESULTS §2, §9 | checked; A's value is large by design, so the scales may not be comparable, and that itself undermines the measure |
| Serious | Target-into-grasp can be correct behaviour: the bottle's grasp truly depends on its target, and the plan now permits that | PLAN_PARTS §10.5 | checked |
| Serious | No dose-response. Target-into-grasp: pi0.5 0.53, GR00T 0.23, R 0.24, H 0.77. pi0.5 mixes more than GR00T and succeeds more on new pairings (0.38 against 0.13) | PARTS_RESULTS §2; FLAWED_COMPOSITION | checked |
| Serious | Purification results license little: the source task has a different object, the intervention runs at every query including the grasp, the random control is not magnitude-matched, and CIs reach [−0.83, 0.48] | PARTS_RESULTS §3, §8 | partly checked |
| Serious | The F2 tie is a known-versus-new contrast, and m's denominator may be confounded by target under-use | PARTS_RESULTS §2 concedes the first | first part checked |

**Has K1 already failed?** Under the plan's claim table, yes in substance: it is refuted if "purifying the located component does not help", and that holds in both models. The go/kill rule did not fire only because of one correlation with lower bound 0.01. The purification tests had low power, so this does not prove there is no effect. Further purification retries that blame damage would make K1 unfalsifiable in practice.

**What survives.** Both VLAs succeed on known pairings and lose new ones in the familiar layout, mostly at the place step: pi0.5 from 0.99 to 0.38 with g 0.79 and p 0.47; GR00T from 0.96 to 0.13 with g 0.86 and p 0.15. That failure is real. Its cause is open. Layout alone hurts about as much as compound: known pairings in the new layout fall to 0.28 and 0.14 with no new pairing involved.

## 3. Charges against K2, K3 and the wider inference

| severity | charge | evidence | status |
|---|---|---|---|
| Fatal, wider inference | The library shares no vision, language or architecture with any VLA, and the perception rungs never ran | `parts_flow.py`; PLAN_PARTS §5 | checked |
| Fatal, K2 as a law | Hold follows from construction at fixed m, and place is the factor that can fail. It fails once pooling is removed (familiar N4 −0.167, N6 −0.118) and differs by pair | PLACE_DIAG §3.5, §6.1 | checked |
| Serious | 3F passes when the CI contains 0, so noise rewards it. −0.03 fails at 150 episodes and passes at 60. N1-N3 are predicted at 1.00, so they cannot fail upward | PARTS_RESULTS §9; LAYER2_WRITEUP §4 | checked |
| Serious | OM is uninformative: it read 1.08 [1.03, 1.13] with N4 at 0.28 against a prediction of 0.70 | PARTS_RESULTS §9 | checked |
| Serious | K3's contribution from the part itself is unmeasured. No test has run with no created part (k = 0), and the target term has no object input, so "every partner at once" reduces largely to tolerating the held body | `parts_flow.py`; targeted grep | not found in the docs searched |
| Serious | The familiar layout is mostly interpolation, and the post-hoc scale frame mapped N4's one real novelty back into support | PARTS_RESULTS §32-34 | partly checked |
| Serious | No simple-controller floor. A scripted servo has not run on the same evaluation states | COVERAGE_RESULTS §1 | partly checked; the planner is weaker than P_min on the familiar bottle, which cuts against this charge |
| Serious | Forking paths: the rounds selected devices on the same states used for reporting, and the one fresh read failed | PARTS_RESULTS §9, §53 | checked |
| Partly rejected | Baselines as straw men. Forgetting does confound the fine-tuned ones. But the per-task LoRA does not forget and reaches 0.00-0.02, and the frame-input joint head A_nosum reaches 0.00 on new pairings | PARTS_RESULTS §5; FLAWED_COMPOSITION | checked; holistic failure on small oracle policies stands |

## 4. Literature (read by the reviewer, not re-read by me; mostly preprints)

- **Dense policies compose without structure.** Text-latent interpolation lifts π0 from under 15% to 83% on extrapolated tasks (Li, 2505.03500). This repo's earlier work replicated it. Pose steering of frozen VLAs gains up to 56.8 points (Zhang et al., 2609.38616).
- **Rival causes are already named.** After target replacement, 86.5-98.3% of LIBERO-Object trials approached the original target, though the target stayed decodable (Jiang et al., 2610.06235). Removing the training-task object raised success by 20.0% with the network untouched (Fang et al., 2602.17659).
- **Components are already located, with another mechanism.** Injecting at π0.5 expert layers L16/L17 gave the source task's trajectory in 99.6% of 1,968 episodes (Grant, Zhao & Wang, 2603.19233). A rescue by donor patching could therefore replay a memorised program.
- **Interventions can mislead.** Causal tracing does not predict the best layer to edit (Hase et al., 2301.04213). Interventions can move representations off-distribution (Grant et al., 2511.04638) or act through dormant pathways (Makelov et al., 2311.17030).
- **Factorisation does not predict generalization.** Montero et al. (ICLR 2021), Chaabouni et al. (2004.09124), Schott et al. (2107.08221), Bahdanau et al. (CLOSURE, 1912.05783).
- **Modular policies composing on state is known.** Mendez et al. (2207.04136), Zhou et al. (2203.05960), Devin et al. (1609.07088), Mishra et al. (2401.03360). PLAN_PARTS §8's "none fixes the combination rule" holds only for the three libraries it names.
- **Grasps depend on goals.** He et al. (2304.04100), Wang et al. (2304.01290). The project's m amendment rediscovers this.

The reviewer found no paper that tests and refutes role mixing as the cause in VLAs, and no paper doing the restated program in §6. Its search was not exhaustive.

## 5. The test I proposed

The proposal was to purify the located component while preserving grasp-mode information, on the idea that the flaw is excess mixing beyond a small necessary signal. The reviewer's objections, which I accept:

- **It targets the wrong failure.** The compound's effect on the grasp includes 0, and intact familiar grasp is already 0.95 under pi0.5.
- **Grasp mode does not explain the collapse.** With the same source (K1), N4 lost every grasp and N1 kept them (PARTS_RESULTS §8). No N1-N6 target needs a turned grasp. This weakens the lead I raised about the K4-K5 contrast.
- **Every outcome is explainable.** If the grasp survives, that is trivial, because the random control already keeps it. A place gain fits K1, under-use and memorisation alike. A null fits every rival plus "wrong subspace". "A small necessary signal" is also a free dose that can always interpolate back to intact.

## 6. Options for the restated claim (the user's decision)

**A. Rival-cause account (my recommendation).** "On a registered object × target split, new-pairing failure in real VLAs is mostly at the place step. We test which account explains it best: cross-role dependence at located components, scene shortcuts, memorised programs or support coverage." Mixing becomes one registered hypothesis among several, and a null is publishable.

**B. Report Layer 2 alone, with a narrow claim.** The second reviewer's wording: "In one LIBERO scene with simulator state and an oracle contact gate, a fixed gated sum of two frame-local flow terms keeps new pairings near known-pairing success when start states match the demos, and lets a part fitted from one demo reach unseen partners. The factorisation is approximate, breaks at the place, and does not deliver layout generalization. Nothing here concerns VLAs." This is mostly a replication, so its value lies in the per-pairing law and the created-part test.

**C. Keep K1 and run one last decisive test,** with a pre-committed kill and no further retries.

A and C share their first experiment, and B can sit beside either.

## 7. Replacement tests

All would be pre-registered. Costs are rough and unmeasured.

| test | question | design | cost |
|---|---|---|---|
| **V1, phase-switched instruction** | Can the VLA place if it receives a known target's instruction after the hold? | pi0.5 with no training: one known instruction for the object before the hold, one for the target after it. A gain says skills exist and the failure lies in how the instruction binds. No gain points at the visual state or the held object | about 2-3 A40-h |
| **V2, carry-gated intervention** | Is the object contamination in the message what hurts the place? | Act only after proprioceptive close onset, so grasps stay identical. Arms: remove the object contrast only; amplify target content only (under-use); norm-matched random; wrong-target hijack; the same on known tasks; a positive control that contaminates a known task and requires repair. Pre-commit: if the control and hijack pass while removal is null, K1 is dead with no retries | a few A40-h |
| **L1, library floor** | What does a part add? | k = 0 (backbone alone) on S3's states, and a scripted servo on the identical states | CPU only |
| **L2, law power** | Does the law hold as an equivalence? | Re-read 3F with an equivalence margin of ±0.05, with at least two partners per target where possible | needs new builds |

V1 comes first because it is the cheapest direct question about VLAs and its reading does not depend on a purification being clean.

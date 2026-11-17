# Novelty check: predicting held-out-pairing success from a trained-task steerability probe (PLAN_PAPER 2c)

Date 2026-11-16. Web search only, no code. "Abstract only" = I opened the arXiv abstract page, not the full text. Nothing below was read in full. Searches were not exhaustive: arXiv, CoRL/NeurIPS/ICLR listings were reached through the search engine, not browsed directly.

## Closest papers

| id | venue | measures | predicts held-out-pairing success across many policies? |
|---|---|---|---|
| 2512.10675 Evaluating Gemini Robotics Policies in a Veo World Simulator (abstract only) | arXiv | Video-model rollouts of 8 Gemini Robotics checkpoints, 5 tasks, OOD axes = novel objects, backgrounds, distractors; validated on 1600+ real trials | Predicts relative policy performance incl. OOD, but needs closed-loop rollouts in a learned simulator. Axes are not object-target pairings. Single model family. |
| 2511.11520 Scalable Policy Evaluation with Video World Models (abstract only) | arXiv | Action-conditioned video model predicts policy value, correlation with real | Rollout-based (in the world model). No pairing axis stated. Number of policies not stated. |
| 2606.05773 PiL-World (abstract only) | arXiv | Chunk-wise world model, VLA-in-the-loop, 3 real dual-arm tasks | Closed-loop; no OOD stated. |
| 2603.09030 PlayWorld (abstract only) | arXiv | World model from play, policy evaluation | Closed-loop; no OOD stated. |
| 2602.17659 LIBERO-CF, "When Vision Overrides Language" (abstract only) | CoRL 2026 Spotlight | Language grounding vs task success under counterfactual instructions on LIBERO scenes; fix = Counterfactual Action Guidance | No predictor, no correlation analysis in abstract. It is a benchmark plus a fix. Its grounding score is measured on the counterfactual tasks themselves (rollouts on the new instruction), not on trained tasks. |
| 2510.03827 LIBERO-PRO (abstract only) | arXiv | Success under perturbed objects, states, instructions, environments; >90% to 0% | No predictor. Shows trained-task success does not rank perturbed success (only in the sense that everything collapses). Nonsense-instruction invariance is the probe-like finding. |
| 2603.19233 Grant et al., Not All Features Are Created Equal (abstract only) | NeurIPS 2026 poster | Activation injection, SAE concepts, 5 VLAs, 420k rollouts | Interventional and mechanistic. No predictor of held-out success. |
| 2609.13458 STAGE / SAT-Bench (abstract only) | Findings of EMNLP 2026 | "Action sensitivity": rate actions change when only instruction semantics change, scene fixed. OpenVLA 6-8%. | Closest to g_s/p_s as a measurement. Abstract says "cross-policy" but gives no correlation of sensitivity with held-out success, and the page said sensitivity is reported as a diagnostic. Not a prediction. |
| 2609.39971 When Instructions Retrieve Trajectories (abstract only) | arXiv | "Instruction-action binding"; counterfactual swaps on pi0.5 and GR00T-N1.7; fix ECT (LIBERO-PRO position-swap 36 to 59%) | Diagnosis plus training fix, 2 policies. No cross-policy predictor in abstract. |
| 2610.00524 CRAFT, Same Scene Different Task (abstract only) | arXiv | Held-out skill combinations on LIBERO, 3 VLAs, counterfactual instruction pairs used for training | Method, not a probe or predictor. Same task family as ours (object-target recombination). |
| 2602.24143 Robust Skills, Brittle Grounding (abstract only) | arXiv | Held-out object-location pairings; primitive execution vs instruction-conditioned success; recommends decomposed metrics | Shows the dissociation (skill reliable, grounding brittle) on SmolVLA and pi0.5; recommends decomposed metrics but the abstract reports no cross-policy prediction. It is the nearest conceptual precursor to "part-competence". Note: the probe there is measured on held-out settings, not trained tasks. |
| 2607.15714 AC-VLA (abstract only) | arXiv | Compositional OOD gain +28 pts on pi0.5 | Method. No predictor. |
| 2404.01542 Agreement-on-the-Line for foundation models (abstract only) | arXiv, v2 Oct 2024 (NeurIPS 2024 per the search listing) | Label-free OOD accuracy prediction from ensemble agreement, vision and language | Predicts OOD accuracy across models, but classification and needs a shift with a linear ID/OOD relation. No robot policy found. |
| 2310.09336 Okawa et al., Compositional Abilities Emerge Multiplicatively (abstract only) | NeurIPS 2023 | Composed-task performance depends multiplicatively on constituent-task performance (diffusion, synthetic) | Multiplicative form is prior art. Not robotics, not a cross-system predictor, not used as a probe-based forecast. |

Seen only in search snippets, not opened (do not cite as read): MDPI Robotics 15(10):201 "Separating Execution and Target Selection ... VLABench" (403 on fetch; snippet says it separates execution from target selection and finds bounded instruction sensitivity); APT 2606.12366; VLATest 2409.12894 (FSE 2025; opened, abstract only: fuzzing 7 VLAs, unseen objects and instruction mutations hurt; no predictor); Flatness 2606.23641 (opened, abstract only: SAM fine-tuning, no predictor).

Searches that returned nothing: any "agreement/accuracy on the line" applied to robot policies; any paper correlating a trained-task language-sensitivity or counterfactual score with held-out success across a set of VLA systems; any product-of-grasp-and-place-competence forecast in robotics.

## Verdict

Not found (as of these searches):
1. A cross-system rank-correlation validation of a probe-based forecast of held-out-pairing success. Closest are world-model evaluators (Veo 2512.10675 and kin), which predict relative policy performance and OOD axes but require closed-loop simulated rollouts of the held-out condition; the probe here needs none and runs on trained tasks.
2. Using counterfactual-instruction sensitivity measured on trained tasks as a predictor. STAGE (2609.13458) measures action sensitivity but as a diagnostic, and its own page states one-step sensitivity is not a success metric.
3. The factorised g_s x p_s with phase split (grasp before hold, place during carry), scored by action-chunk endpoint displacement toward the swapped entity, and applied through any front end or wrapper.

Not novel:
- The multiplicative part-competence form (Okawa et al. 2310.09336). Cite it as the source; our contribution is using it as a zero-fit forecaster.
- The diagnosis that VLAs follow trained cues over the instruction (LIBERO-PRO, LIBERO-CF, 2609.39971, 2602.24143, CRAFT, AC-VLA), and nonsense-instruction invariance.
- The intervention/steering framing (Grant et al.).

## Narrowest surviving claim

"On a fixed object-target pick-place task family, a two-number open-loop counterfactual-instruction probe taken on trained tasks, combined multiplicatively with no fitting, rank-orders ~15-20 VLA systems (including wrapped systems) by held-out-pairing success, and beats trained-task success and a nonsense-word sensitivity score." The claim lives or dies on T1 (rho >= .6) and the baselines. Do not claim generality beyond the family, and do not claim the product form is novel.

## What a reviewer would cite against it

- Veo/world-model evaluators (2512.10675, 2511.11520, PiL-World, PlayWorld): "policy ranking without real robots already exists, including OOD." Answer: they need the held-out condition simulated and a trained video model; ours is a cheap trained-task probe. Compare cost and, if possible, accuracy.
- STAGE 2609.13458 and LIBERO-PRO: "action sensitivity to instruction changes is already a known diagnostic." Answer: diagnostics, not validated forecasts; the novelty is the validation across systems.
- Okawa et al. for the product form; 2602.24143 for decomposing primitive execution from instruction-conditioned success.
- Agreement-on-the-line (2404.01542): "label-free OOD prediction is established; where is the comparison?" Weak objection (classification, ensembles) but a reviewer may ask for an agreement-based baseline across our policies.
- Statistical: 15-20 systems from ~5 families are not independent (retrains of one backbone share failures); rho can be driven by between-family differences. T2 (leave-family-out) is the answer and should be stated as primary; also report within-family rho.
- Circularity: the policies include wrappers whose behaviour the probe inherits by design (ReGuide, Hp). A reviewer may say the probe measures the intervention, not a predictive property. Include wrapper-free systems as the headline subset.
- A multi-object probe in a scene with two candidate objects can confound steerability with plain detection of the swapped object; include a probe-validity control (e.g. swap to an absent object word).
- Recheck before submission: CoRL 2026 / ICLR 2027 / NeurIPS 2026 OpenReview listings were not browsed directly, so a concurrent submission may exist.

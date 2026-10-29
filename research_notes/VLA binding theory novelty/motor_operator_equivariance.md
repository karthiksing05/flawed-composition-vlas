# Motor operators and pose equivariance in learned robot policies: prior work on how they are measured (novelty check for C2)

Scope: claim C2. On our pi0.5 (LIBERO-90 split), counterfactual displacement of the named object moves the chunk endpoint toward it in 0.864 of observations (chance 0.20), with gain 0.13-0.25 for all 19 object types. Displacing the target moves the release point (139/139, leave-type-out R^2 0.81). On public pi05_libero, leave-scene-out transfer is 0.62 in the object frame and -0.07 in the world frame.
Searches were run 2026-10-26 (about 17 search and fetch calls). Every arXiv ID below was opened (abstract or HTML). Where only part of a paper was read, the note says so.

## Q1. Has anyone estimated action Jacobians or sensitivities with respect to object pose, or run counterfactual-displacement or equivariance analyses on VLAs or diffusion policies?

### Takeaway
I found no paper that measures a per-observation displacement gain (d endpoint / d object pose) for a VLA, or one that checks whether such a gain is shared across object types. The closest measurements fall into three groups:
- an image-rotation equivariance-error metric on GR00T N1.5 (EquiVLA);
- success-rate or behavioural counts under position swaps and shifts on pi0.5 (2609.39971, 2603.19233);
- linear probes that decode object position from OpenVLA activations.

C2's measurement (small physical displacement in simulation, a continuous action-endpoint response, per-type gain, and a frame comparison) looks novel as a methodology. It still has to be positioned against these works.

### Cited Findings
- **EquiVLA, arXiv 2606.19784 (Ha, Nguyen, Ho et al., June 2026).**
  - The paper defines an equivariance error as the mean ||f(g·o) − g·f(o)|| over observations and group elements.
  - It evaluates this on "M=500 LIBERO observations across all g∈C8." The transform "rotates the top-down image and end-effector pose while keeping the wrist image fixed."
  - The non-equivariant baseline GR00T N1.5 scores ε_eq = 7.754 ± 3.572. EquiActor scores 0.837 ± 0.738, and full EquiVLA scores 0.284 ± 0.134 (27.3x lower).
  - It covers rotation only, with no translation. The transform is a synthetic image rotation, not a physical displacement of an object. — [EquiVLA](https://arxiv.org/html/2606.19784)
- **"When Instructions Retrieve Trajectories," arXiv 2609.39971 (Chen, Hou, ..., Sun, Lee; 30 Sep 2026). Models: fine-tuned pi0.5 and GR00T-N1.7.**
  - On LIBERO-PRO Swap, pi0.5 success is 46.6 / 18.2 / 34.2 / 9.4% (Spatial / Object / Goal / Long). GR00T-N1.7 scores 1.2 / 0.0 / 2.2 / 0.2%. In-distribution success is above 90%.
  - In 381 pi0.5 rollouts, 69% were "retrieval-like", 27% followed the changed requirement, and 4% collapsed.
  - Key result for C2: "A 10 cm target shift was followed visually." In all nine combined swap-and-shift probes, the policy tracked the displaced object in the source-bound slot, not the instructed object.
  - The authors conclude "visual feedback adjusts execution, but the instruction selects the trajectory family" ("instruction-action binding"). They report not finding simple fixed-coordinate replay.
  - Interventions: replacing the prefix-KV with the target task's state moved pi0.5 toward the target in 109/200 rollouts, against 0/200 for controls. Their "Equivariant Counterfactual Training" raised UR5e unseen-position Swap success from 3/40 to 35/40.
  - (Only the first ~100k characters of the HTML were read.) — [2609.39971](https://arxiv.org/html/2609.39971)
- **Linear probes and SAE features.**
  - A 2026 SAE paper on VLA residual streams reports probes decoding object positions and actions from OpenVLA activations at >90% accuracy for most layers. It also reports motion-primitive-like features in action-expert layers on LIBERO and DROID. (Seen at snippet level only; full text not read.) — [2603.19183](https://arxiv.org/html/2603.19183v1)
  - Häon et al. inject semantic FFN directions (e.g. speed, direction) into OpenVLA on LIBERO and pi0 on a UR5. — [2509.00328](https://arxiv.org/pdf/2509.00328)
  - 2603.05487 reports decoding spatial relations between objects and movement targets from OpenVLA with linear probes. (Snippet only.) — [2603.05487](https://arxiv.org/pdf/2603.05487)
- **Diffusion-policy Jacobians.** Contractive Diffusion Policies (arXiv 2601.01003) penalises the score Jacobian with respect to actions/noise. That Jacobian is not taken with respect to object pose. (Seen via search snippet.) — [2601.01003](https://arxiv.org/pdf/2601.01003)
- **No hit.** I found no paper computing ∂action/∂object-pose for a diffusion or flow policy. — (search result, no source)

### Inferences
- EquiVLA is the closest metric-level precedent: equivariance error of a non-equivariant pretrained VLA on LIBERO. C2 differs from it in three ways:
  - it uses translations of a single bound object in simulation, not a rotation of the whole image;
  - it reports a direction-hit rate and a gain, not an error norm;
  - it compares frames and checks sharing across object types.
- C2 should cite EquiVLA and possibly report ε_eq-style numbers for comparability.
- 2609.39971 partly anticipates the approach result qualitatively: pi0.5 visually follows a 10 cm target shift. It is consistent with C2, not contradictory. It also shows the operator is bound to the "source-bound slot", which supports C2's "filler is bound to a role" reading.

### Gaps
- I could not confirm whether 2609.39971 quantifies endpoint displacement (gain). The section after 100k characters was not read.
- I did not check the full 2603.19183 text for position-probe methodology.

## Q2. What do the robustness benchmarks find about whether VLAs track object position or memorise trajectories? Do they contradict the object-frame finding?

### Takeaway
LIBERO-PRO and LIBERO-Plus report success collapse under position changes and read it as memorisation or "positional bias." Both measure task success under large or out-of-distribution changes, not the local action response. Neither tests frames. This is not a direct contradiction of C2: a policy can respond equivariantly to ±2 cm and still fail at task level under large shifts. The narrative tension still has to be addressed explicitly.

### Cited Findings
- **LIBERO-PRO, arXiv 2510.03827 (Zhou, Xu, Tie, Chen, Zhang, Chu, Zhou, Sun; v1 4 Oct 2025, v2 25 May 2026).**
  - It perturbs objects, initial states, instructions, and environments.
  - Models above 90% on LIBERO drop to 0.0% in the generalized setting, which the authors attribute to "rote memorization of action sequences and environment layouts."
  - Models kept grasping when the target was replaced with irrelevant items. Outputs were unchanged under corrupted instructions. — [LIBERO-PRO](https://arxiv.org/abs/2510.03827)
- **LIBERO-Plus, arXiv 2510.13626 (Fei, Wang, ..., Qiu; Oct 2025).**
  - It applies seven perturbation axes. Under camera and robot-init changes, success falls "from 95% to below 30%."
  - The layout axis = confounders + target placement/pose. Success "decreases only marginally when confounding objects are added," while "when the target objects' placement is altered, the performance of the models drops significantly."
  - Finding 5: "Current VLAs exhibit positional bias rather than genuine semantic understanding of objects." The models "may have merely learned the positional information of the target objects." This rests on the qualitative Fig. 1, with no quantitative split.
  - Table 2 layout scores: OpenVLA 28.5, UniVLA 31.9, WorldVLA 38.0, NORA 62.1, pi0-Fast 68.8, pi0 68.9, OpenVLA-OFT 74.2, RIPT-VLA 74.2. — [LIBERO-Plus](https://arxiv.org/html/2510.13626v1)
- **2603.19233** reports that on its UR10e, "about a centimeter of displacement already makes the grasp fail." It also cites Zhou et al. 2025 (LIBERO-PRO) for small position perturbations dropping OpenVLA and pi0 to 0.00. — [2603.19233](https://arxiv.org/html/2603.19233)
- **2609.39971** directly argues for a middle position: pi0.5 follows a 10 cm target shift visually, but trajectory families are selected by the instruction. They "do not find simple fixed-coordinate replay." — [2609.39971](https://arxiv.org/html/2609.39971)

### Inferences
- The "positional bias" in LIBERO-Plus is ambiguous. It could mean bound to absolute positions (contradicting C2) or bound to a role or slot (compatible with C2). The paper's evidence is success rates only, so it cannot distinguish the two.
- C2's leave-scene-out object-frame transfer (0.62 vs −0.07 world frame) gives a direct test that these benchmarks lack. It should be framed as reconciling them: local equivariance can hold while task success still fails out of distribution.
- C2's gain of 0.13-0.25 (below 1) is itself evidence of partial tracking. It is consistent with policies that follow small shifts but undershoot. This fits the benchmark collapse under large shifts. The paper should say so explicitly.
- Risk: a reviewer could say ±2 cm is inside the training init distribution, so the effect is interpolation. LIBERO-PRO and LIBERO-Plus test beyond it.

### Gaps
- I did not obtain LIBERO-Plus numbers for the confounder and target-shift sub-conditions separately (the text has none).
- I did not obtain LIBERO-PRO position-specific numbers from the full text.

## Q3. What exactly does arXiv 2603.19233 test with "bound to absolute workspace coordinates," and does any follow-up test frames?

### Takeaway
2603.19233 tests where an injected program goes. Its claim covers activation-injected programs, not the policy's normal response to object displacement. The paper itself says the uninjected pi0.5 reaches the displaced object. It also admits the coordinate-frame control (rigid world transform) was not run. C2 therefore does not contradict it. C2's object-vs-world-frame test is exactly the control 2603.19233 lists as missing.

### Cited Findings
- **Paper:** "Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models," Bryce Grant, Xijia Zhao, Peng Wang. v1 19 Mar 2026, v4 20 Oct 2026. — [abs](https://arxiv.org/abs/2603.19233)
- **Models:** pi0.5 (3B flow), OpenVLA-OFT (7B), X-VLA (1B), SmolVLA (450M), GR00T N1.5 (3B DiT), and ACT (80M, a language-free control). — [HTML](https://arxiv.org/html/2603.19233)
- **Simulation result.**
  - Injecting task A's action-expert activations (layers 16-17) while the policy views scene B makes pi0.5 execute A's trajectory in 99.6% of 1,968 episodes.
  - Override rates: OFT 96.4%, X-VLA 99.8%, SmolVLA 52.1%, GR00T 57.0%, ACT 0.0%.
  - §4.3 states that "its motor program is bound to absolute workspace coordinates." Under injection, 68-80% of episodes displace an object by more than 5 mm, but that object belongs to the destination scene: the arm reaches where the source objects would have been.
  - DTW source-dominance is 86-91%. A norm-matched random direction gives 22% at n=50 and chance at n=200. — [HTML](https://arxiv.org/html/2603.19233)
- **Physical result (§4.7 and App. O).**
  - Setup: pi0.5 fine-tuned on a UR10e, seven tasks, baseline 76.2%.
  - Directed injection redirects the task in 26/26 and 18/26 rollouts, against no-injection floors of 0/24 and 0/25.
  - With the source object moved 10-15 cm, "the arm drove to the recorded coordinates every time and closed on empty air" (50/50).
  - Key line: "Without injection and with the source prompt, the policy reaches the displaced object." This is qualitative, with no trial count given. — [HTML](https://arxiv.org/html/2603.19233)
- **Frames.** Appendix B ("Coordinate-Frame Disambiguation") says: "A rigid transform of the whole world, exact in simulation, would separate the two, and we have not run it." The NeurIPS checklist lists "the coordinate-frame control not run." Appendix B also makes this prediction: "Moving an object out of its training-distribution position should break execution with the instruction unchanged." — [HTML](https://arxiv.org/html/2603.19233)
- **Follow-up.** 2609.39971 reports that pi0.5 follows a 10 cm target shift and "do[es] not find simple fixed-coordinate replay." This is the closest follow-up, but it does not run an explicit frame test either. — [2609.39971](https://arxiv.org/html/2609.39971)

### Inferences
- 2603.19233's claim is narrower than its headline. Injected expert activations carry a world-frame trajectory, and the uninjected policy tracks the object. One reconciliation fits both results: expert-layer activations at L16-17 encode the motor program in world coordinates, after binding to the object has already been resolved upstream (VLM or prefix). C2 measures the end-to-end operator. Both can be true at once.
- C2's leave-scene-out frame comparison provides the frame control 2603.19233 says is missing. That is a concrete novelty point. Cite their Appendix B sentence directly.

### Gaps
- I found no paper that ran the rigid-world-transform control as of Oct 2026.
- 2603.19233 gives no trial count for the uninjected displaced-object condition.

## Q4. Is there work measuring emergent equivariance in non-equivariant pretrained VLAs (as opposed to building equivariant policies such as EquiDiff or EquiBot)?

### Takeaway
I found only one paper: EquiVLA measures the equivariance error of plain GR00T N1.5, and uses it as a baseline to motivate imposing SO(2) equivariance. Every other work in this group builds equivariance in by design. I found nothing measuring emergent translation equivariance, or object-frame equivariance, in a pretrained VLA.

### Cited Findings
- **EquiVLA (2606.19784):** GR00T N1.5 ε_eq = 7.754 ± 3.572 under C8 image rotations on 500 LIBERO observations. It imposes SO(2) equivariance on both the VLM visual backbone and the flow head. — [EquiVLA](https://arxiv.org/html/2606.19784)
- **Built-in equivariance in VLAs (seen via search listing; abstracts not opened):**
  - SAVLA (arXiv 2609.16641) adds a symmetry-preserving action head on a frozen non-equivariant backbone. — [SAVLA](https://arxiv.org/pdf/2609.16641)
  - "Toward Embodiment Equivariant VLA Policy" (arXiv 2509.14630). — [2509.14630](https://arxiv.org/pdf/2509.14630)
  - EfficientFlow (arXiv 2512.02020). — [2512.02020](https://arxiv.org/pdf/2512.02020)
  - EquivAct (arXiv 2310.16050). — [2310.16050](https://arxiv.org/pdf/2310.16050)
- **Emergent equivariance outside robotics:** Gerken & Kessel (arXiv 2403.03103) show that deep ensembles trained with full augmentation become equivariant collectively, while individual members do not. — [2403.03103](https://arxiv.org/html/2403.03103v1)
- **Object-centric action frames:** MCF-Proto (arXiv 2605.11809, Yang et al., 12 May 2026) learns a motion-centric SO(3) action frame and composes actions from prototypes in that frame. Its abstract reports better robustness "especially under geometric perturbations." The frame is aligned to end-effector motion, not to the object, and the paper does not analyse whether pretrained VLAs encode object-relative frames. — [2605.11809](https://arxiv.org/abs/2605.11809)

### Inferences
- C2's object-frame vs world-frame result on public pi05_libero appears to be the first measurement of emergent object-frame (translation) structure in a non-equivariant VLA. Two prior items must be cited and differentiated: EquiVLA's rotation-error metric, and MCF-Proto's design argument that world-frame heads are suboptimal.
- MCF-Proto ("prototype-based actions in a local frame") is the closest design analog to "compositional pose-parameterised operators." Its abstract gives no measurement of pretrained models.

### Gaps
- EquiDiff and EquiBot were not individually re-verified in this pass. They are by-design equivariant and therefore not competitors on the "emergent" claim.
- Abstracts of SAVLA, 2509.14630, and EfficientFlow were not opened.

## Q5. Is there work extracting motor primitives or skill operators post hoc from generalist policies, and does it show sharing across object types?

### Takeaway
Post-hoc primitive or skill work exists:
- behavioural-mode discovery for RL fine-tuning;
- SAE motion-primitive features in VLA action experts;
- latent-action models whose retrieved primitives span objects.

None of it shows a pose-parameterised operator with a quantified, shared gain across object types. The cross-object sharing claims are qualitative (retrieval examples).

### Cited Findings
- **Behavioral Mode Discovery, arXiv 2605.11387 (Longhini, Emukpere, Renders, Kim; 12 May 2026; listed as ICML 2026).**
  - It discovers latent behavioural modes in a pretrained generative (e.g. diffusion) policy without supervision, and uses mutual information as an intrinsic reward so that RL fine-tuning keeps multimodality.
  - The abstract mentions no cross-object sharing, pose, or equivariance analysis. It is a fine-tuning regulariser, not an operator analysis. — [2605.11387](https://arxiv.org/abs/2605.11387)
- **SAE features (2603.19183):** reports motion-primitive-like features in action-expert layers on LIBERO and DROID that steer behaviour predictably. (Snippet-level; sharing across objects not verified.) — [2603.19183](https://arxiv.org/html/2603.19183v1)
- **WALA (arXiv 2607.11397):** its latent action model retrieves similar manipulation primitives across objects, layouts, and backgrounds (appendix Fig. 9 retrieval examples). This is qualitative and is a separately trained latent model, not the VLA's own operators. (Snippet-level.) — [WALA](https://arxiv.org/html/2607.11397v2)
- **LAPS (arXiv 2511.21428, CVPR 2026):** segments primitives from a motion tokenizer's "Latent Action Energy" on industrial human video. — [2511.21428](https://arxiv.org/html/2511.21428v2)
- **2603.19233** shows injected expert activations carry a whole task's motor program: a transferable "how," but in world coordinates. — [HTML](https://arxiv.org/html/2603.19233)

### Inferences
- Novelty for C2 holds on two points:
  - a quantitative per-type gain (0.13-0.25, similar across 19 types) as evidence of filler-shared operators;
  - a release operator with leave-type-out R² 0.81.
- I found no prior work that reports operator parameters (gain, R²) shared across object categories.
- The approach gain is well below 1. Reviewers may read the operator as an "attraction" or bias, not an equivariant map, so the paper should define equivariance up to gain (a partial or contracted equivariance).

### Gaps
- Full texts of BMD, WALA, and 2603.19183 were not read for cross-object analyses.
- Searches for OpenReview/CoRL 2026 operator-extraction work turned up nothing beyond these. Coverage of non-arXiv venues is limited.

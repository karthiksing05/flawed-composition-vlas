# Prior theory colliding with "compositional operators + holistic exemplar-like binding" in VLAs

Scope: novelty check for the framing pi(a|o,l) = sum_b P(b|o,l) O(a|o,p_b), with O compositional/shared/pose-equivariant and P holistic, keyed to the nearest training (scene, instruction). Predictions: (a) the wrong binding equals the most frequent training binding in that scene; (b) error rises with relational depth (type < instance < relational sub-region); (c) external binding fixes binding errors, not motion errors. C5: relational sub-region binding errors as a named failure class.

Verification note: each arXiv entry below was checked against its arXiv abstract page in this session (WebFetch), unless it is marked [UNVERIFIED THIS SESSION]. Abstract pages were summarised by a fetch model and limited verbatim quoting, so short quoted phrases are exact, while longer descriptions are faithful paraphrases. Full texts were NOT read, so anything a paper says only in its body (failure taxonomies, drawer tasks, theory sections) remains unchecked.

## Q1. Does any VLA / robot-learning paper frame compositional failures as variable or role-filler binding? What theory do they give?

### Takeaway
The empirical split (manipulation skill intact, target grounding wrong) is now well documented. Three 2026 preprints come closest: ReGuide ("Correcting WHERE, Preserving HOW"), "Robust Skills, Brittle Grounding", and CAG/LIBERO-CF. CAG also reports that the wrong choice tends to be the frequently seen object. None of the abstracts uses the word "binding", gives a mixture-over-bindings formalism, or states the exemplar-vs-compositional asymmetry as a theory. All of them describe it as "visual shortcuts" or "spurious correlations". The descriptive split is therefore not novel. The formal factorization and the three graded predictions appear to be.

### Cited Findings
- **ReGuide: the highest collision risk.** Zhang, Liu, Li, Song, Hariri, Ganguly, Yang, Ye, Grant, Chaudhary, Yin, "Correcting WHERE, Preserving HOW: Compositional Generalization for Vision-Language-Action Models via Referential Guidance", arXiv 2609.38616 (cs.RO, submitted 29 Sep 2026).
  - Claim: VLAs "fail at global grounding while retaining local manipulation skills" and "exploit visual shortcuts, associating actions with task-irrelevant visual features".
  - Method: ReGuide, a training-free wrapper. It uses object poses from a grounding module to steer the end-effector into "demonstration-supported configurations" of the instructed referent, and a frozen policy then executes.
  - What the abstract leaves out: no formal theory, and no claim that the wrong referent is the dominant training referent. The WHERE/HOW split is presented as an observation.
  - Why it collides: this is effectively prediction (c) as a method (supply the binding externally, keep the motor policy). It is a partial overlap, because ReGuide also steers the end-effector. — [arXiv abs](https://arxiv.org/abs/2609.38616)
- **Robust Skills, Brittle Grounding.** Emukpere, Deffayet, Renders, "Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking", arXiv 2602.24143 (27 Feb 2026).
  - Setup: SmolVLA and pi0.5, tested on held-out object-location pairings.
  - Findings: the primitive stays reliable while instruction-conditioned success collapses. They conclude that "manipulation skill acquisition is decoupled from instruction following" and propose decomposed metrics.
  - Numbers (from the search snippet only, not checked on the abstract page): 44% to 0% instruction-conditioned success on held-out pairings; reach 100% to 2%; grasp-anything 57% to 31%. — [arXiv abs](https://arxiv.org/abs/2602.24143)
- **CAG / LIBERO-CF.** Fang, Feng, Jing, Liu, Yang, Wei, Szafir, Ding, "When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs", arXiv 2602.17659 (v1 19 Feb 2026, v2 15 Jul 2026).
  - Claim: when instructions lack scene-specific supervision, VLAs show "counterfactual failures". They repeat well-learned behaviors and pick frequently seen objects regardless of the instruction.
  - Benchmark: LIBERO-CF assigns alternative instructions to familiar LIBERO layouts.
  - Method: CAG, a dual-branch method with a language-unconditioned Vision-Action module. Reported gains are +9.7% language-following accuracy for pi0.5 training-free and +15.5% with the VA module.
  - Search snippet only: removing the training-task object from the scene improves counterfactual performance.
  - Relevance: this is the closest prior evidence for prediction (a), but stated qualitatively as "frequently seen objects". It is not measured per scene as "the most frequent training binding". — [arXiv abs](https://arxiv.org/abs/2602.17659); [project page](https://vla-va.github.io/)
- **CRAFT.** Yang, Na, Cho, Yoon, "Same Scene, Different Task: Skill Alignment for Compositional Generalization in VLAs", arXiv 2610.00524 (30 Sep 2026).
  - Theory: the visual scene serves as "a proxy for the instruction", so the policy runs the combination it learned for similar-looking scenes.
  - Method: build pairs with the same observation and an undemonstrated instruction, and transfer supervision from skill representations.
  - The abstract does not use the term binding.
  - Relevance: "scene as proxy for instruction" is close to the "binding keyed on nearest training scene" half of the theory. — [arXiv abs](https://arxiv.org/abs/2610.00524)
- **AC-VLA.** Peng, Peng, Lu, Lian, Yu, Wang, "AC-VLA: Robust Out-of-Distribution Action Execution via Compositional Learning", arXiv 2607.15714 (17 Jul 2026).
  - Two failure modes: "trajectory overfitting" (memorizing holistic trajectory patterns instead of sub-skill semantics) and a wrist-view perceptual shortcut.
  - About +28% absolute on LIBERO-OOD with pi0.5.
  - Relevance: this says the motor side is holistic, which partly contradicts "O is compositional". — [arXiv abs](https://arxiv.org/abs/2607.15714)
- **VLAs are Confined yet Capable.** Quanyi Li, "VLAs are Confined yet Capable of Generalizing to Novel Instructions", arXiv 2505.03500 (v1 6 May 2025, v5 1 May 2026).
  - Diagnosis: "spatial overfitting", in which VLAs map object names to locations seen in demonstrations.
  - Results: on libero-ood (20 tasks) all SOTA VLAs score under 15%, and pi0 with text-latent interpolation reaches 83%.
  - Relevance: this supports binding-as-memorized-location and shows that the latent-level recombination of behaviors works. — [arXiv abs](https://arxiv.org/abs/2505.03500)
- **LIBERO-PRO.** arXiv 2510.03827: LIBERO accuracies above 90% largely reflect rote memorization of fixed train-set mappings. — [arXiv](https://arxiv.org/pdf/2510.03827) (search snippet; abstract not opened)
- **LIBERO-Plus.** arXiv 2510.13626: OpenVLA-OFT on LIBERO-Object is largely unchanged with a blank instruction, which the authors read as degenerating into a Vision-Action model; the long suite degrades. — [arXiv](https://arxiv.org/pdf/2510.13626) (search snippet; abstract not opened)
- **Multilingual LIBERO study.** arXiv 2606.11906: OpenVLA-OFT on LIBERO-Goal falls from 97.9% in English to about 10% in several other languages. — [arXiv](https://arxiv.org/pdf/2606.11906) (search snippet only)
- **CAST.** arXiv 2508.13446: standard VLAs ignore language until distractors make the task non-inferable from images, and then oscillate between modes; counterfactual relabeling roughly doubles success. — [arXiv](https://arxiv.org/pdf/2508.13446) (search snippet only)
- **CofactVLA.** arXiv 2608.04396: a deconfounding approach with counterfactual intervention. — [arXiv](https://arxiv.org/pdf/2608.04396) (title only; content not checked)

### Inferences
- "Binding is wrong, motion is right" is established empirically by ReGuide, Robust Skills/Brittle Grounding and CAG. The paper must cite these, and must not claim the split itself as new.
- The candidate novelty is a theory of *why*, with graded and falsifiable predictions:
  - the formal mixture over bindings;
  - the asymmetry (compositional O versus exemplar P);
  - per-scene frequency prediction (a);
  - the relational-depth ordering (b).
  None of the checked abstracts gives these.
- CAG's "frequently seen objects" is a qualitative precursor of (a). The paper should test (a) quantitatively and credit CAG.
- ReGuide is a precursor of (c) as an intervention. The distinct claim in (c) is that external binding fixes binding errors but *not* motion errors, which is a dissociation test ReGuide does not frame.
- AC-VLA ("trajectory overfitting") is a potential contradiction: it says the trajectories are holistic too. The paper needs evidence that O is compositional, for example pose equivariance.

### Gaps
- Full texts of ReGuide, CAG and CRAFT were not read, so a body-level "binding" framing or a frequency analysis cannot be ruled out. ReGuide is the most urgent full-text check.
- The LIBERO-PRO, LIBERO-Plus, CAST and multilingual claims come from search snippets, not abstract pages.

## Q2. Is there exemplar- or memorization-based theory of robot-policy generalization?

### Takeaway
Yes, for whole policies. He et al. argue that diffusion policies behave like an action lookup table keyed on the nearest training image. "Much Ado About Noising" is a counterpoint: nearest-neighbour policies fail on precise contact-rich tasks. The checked work does not split the policy into an exemplar binding part and a compositional motor part. That asymmetry is the candidate novelty, and the lookup-table paper is the main prior to position against.

### Cited Findings
- **Demystifying Diffusion Policies (lookup table).** Chengyang He, Xu Liu, Gadiel Sznaier Camps, Guillaume Sartoretti, Mac Schwager, "Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives", arXiv 2505.05787 (9 May 2025).
  - Claim: "diffusion policies essentially memorize an action lookup table", and this is beneficial in the sparse-data regime. At runtime the policy finds the closest training image in latent space and recalls its action sequence.
  - Even on OOD images (cats and dogs) it outputs training action sequences.
  - Alternative: ALT, which uses a contrastive encoder as a hash. It matches diffusion policy performance with about 0.34% of the inference time and 0.85% of the memory, and flags OOD inputs.
  - Venue: listed as an ICLR 2026 poster ("Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative"), although the arXiv page lists no venue. — [arXiv abs](https://arxiv.org/abs/2505.05787); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009689)
- **Much Ado About Noising.** arXiv 2512.01809: a nearest-neighbour policy (VINN) gets only 12% on state-based Tool-Hang, so lookup tables are insufficient to explain generative control policies on precise contact-rich tasks. — [arXiv](https://arxiv.org/pdf/2512.01809) (search snippet only)
- **Closed-form diffusion policies.** arXiv 2606.01238: training-free imitation with closed-form diffusion policies, i.e. an explicit memorizing denoiser. — [arXiv](https://arxiv.org/pdf/2606.01238) (title only)
- **Rule versus exemplar generalization in networks.** Dasgupta, Grant, Griffiths, "Distinguishing rule- and exemplar-based generalization in learning systems", ICML 2022, arXiv 2110.04328.
  - Method: cognitive-psychology protocols that separate feature-level bias from exemplar-vs-rule bias.
  - Finding: "standard neural network models are feature-biased and exemplar-based."
  - Relevance: the direct cognitive-science prior for calling P "exemplar-like", though it was not applied to robots. — [arXiv abs](https://arxiv.org/abs/2110.04328)

### Inferences
- He et al. predict lookup over *whole action sequences*. The proposed theory predicts lookup only over *bindings*, with actions re-generated equivariantly at the bound entity's pose. These two make distinguishable predictions: move the training-binding object to a new pose. ALT replays the old trajectory, while the proposed theory reaches the new pose of the wrong object.
- This contrast is the cleanest way to claim novelty over the lookup-table account.

### Gaps
- I did not find a paper applying exemplar models (GCM/ALCOVE-style) to VLAs specifically.
- The search was limited to about 15 calls, so a VLA-specific nearest-neighbour analysis cannot be excluded.

## Q3. What does the neural binding-problem literature say, and has it been applied to embodied policies?

### Takeaway
Binding is a mature framing for VLMs and text-to-image models (binding IDs, attention-map binding losses, the cognitive-science interference account). The checked sources do not apply it to embodied action policies. Applying it to VLAs, with role = object/goal region and filler = scene entity, appears to be open.

### Cited Findings
- **Binding problem in VLMs.** Campbell, Rane, Giallanza, De Sabbata, Ghods, Joshi, Ku, Frankland, Griffiths, Cohen, Webb, "Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem", arXiv 2411.00238 (v1 31 Oct 2024, v2 16 Apr 2025).
  - Claim: multi-object failures of VLMs and text-to-image models (counting, localization, visual analogy) stem from the binding problem. Shared representational resources cause interference, "necessitating the use of serial processing", and the failures resemble the limits of rapid feedforward processing in the brain.
  - The abstract does not mention robots or embodied policies. — [arXiv abs](https://arxiv.org/abs/2411.00238)
- **Binding IDs in VLMs.** Saravanan, Tapaswi, Gandhi, "Investigating Mechanisms for In-Context Vision Language Binding", CVPR 2025 Workshops (MIV), arXiv 2505.22200: VLMs give an object's image tokens and its text references a shared binding identifier. — [CVF](https://openaccess.thecvf.com/content/CVPR2025W/MIV/html/Saravanan_Investigating_Mechanisms_for_In-Context_Vision_Language_Binding_CVPRW_2025_paper.html) (search result; abstract not opened)
- **Binding IDs in LMs.** Feng & Steinhardt (2024) propose binding-ID vectors attached to entities and attributes in Pythia and LLaMA. Dai, Heinzerling, Inui (EMNLP 2024) localize binding information in entity activations. [UNVERIFIED THIS SESSION: known via the search summary; arXiv ID for Feng & Steinhardt not checked]
- **Linguistic binding in text-to-image.** Rassin et al., "Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment" (SynGen), NeurIPS 2023, arXiv 2306.08877.
  - Problem: attributes bind to the wrong entities ("a pink sunflower and a yellow flamingo").
  - Method: a syntax-guided cross-attention overlap loss at inference, with no retraining. — [arXiv](https://arxiv.org/html/2306.08877v3) (search result)
- **Attend-and-Excite.** Chefer et al. 2023: names "catastrophic neglect" and "attribute binding" failures in Stable Diffusion. — [summary page](https://deepai.org/publication/attend-and-excite-attention-based-semantic-guidance-for-text-to-image-diffusion-models) (secondary source; arXiv 2301.13826 [UNVERIFIED THIS SESSION])
- **Object-attribute binding evaluation.** Trusca, Nuyts et al., arXiv 2404.13766: CLIP score fails to check attribute-object binding in multi-object prompts. — [arXiv](https://arxiv.org/pdf/2404.13766) (search result)
- **Pointing removes binding errors.** "Binding Visual Features Point by Point", arXiv 2605.25427: learning to point via text reportedly removes binding errors in Qwen2.5-VL, Gemma-3 and LLaVA. — [arXiv](https://arxiv.org/html/2605.25427) (search snippet only)
- **Classics, not re-checked this session:**
  - Greff, van Steenkiste, Schmidhuber, "On the Binding Problem in Artificial Neural Networks", arXiv 2012.05208 (2020) [UNVERIFIED THIS SESSION]
  - Yuksekgonul et al., ARO / "When and why vision-language models behave like bags-of-words", ICLR 2023, arXiv 2210.01936 [UNVERIFIED THIS SESSION]
  - Thrush et al., Winoground, CVPR 2022 [UNVERIFIED THIS SESSION]

### Inferences
- Interventions that supply the binding externally exist in VLMs (pointing) and in text-to-image models (SynGen's attention alignment). They are analogues of prediction (c) in non-embodied models, and ReGuide is the embodied analogue.
- Campbell et al.'s interference/serial-processing account predicts errors that grow with the number of similar entities. The proposed theory predicts errors that grow with relational depth and track training frequency. The paper can present these as competing accounts.

### Gaps
- No search turned up "binding problem" applied to a VLA or other action policy. Absence from about 15 searches is weak evidence, not proof.

## Q4. Is there work on relational depth of referring expressions, or on sub-region errors (drawer vs top, shelf level), in VLAs or robot grounding?

### Takeaway
Spatial-grounding benchmarks exist: Spa-Bench (Oct 2026), RoboRefer/RefSpatial (multi-step spatial referring for VLMs, NeurIPS 2025), and VLA-Arena. RefSpatial grades reasoning by number of steps. No checked abstract names "relational sub-region" errors (wrong drawer level, cabinet top vs drawer, shelf level) as a failure class, or orders type < instance < relational. C5 appears unclaimed, apart from anecdotal mentions such as SpatialActor's "opens the wrong drawer".

### Cited Findings
- **Spa-Bench.** Tien-Smith, Pidden, Potamias, "Spa-Bench: A Real-Robot Benchmark for Spatial Grounding in Vision-Language-Action Models", arXiv 2610.12696 (8 Oct 2026).
  - Tasks: six tabletop tasks (object state, relative size, referential description, relational placement, ordinal position, counting).
  - Results: success falls from 67.5-73.3% in distribution to 40.7-52.0% out of distribution.
  - Conclusion: policies can execute the manipulation but lack robust spatial grounding.
  - The abstract has no drawer or shelf sub-region class and no depth hierarchy. — [arXiv abs](https://arxiv.org/abs/2610.12696)
- **RoboRefer / RefSpatial.** Zhou et al., "RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics", NeurIPS 2025, arXiv 2506.04308.
  - RefSpatial: about 20M QA pairs, 31 spatial relations, reasoning chains of up to 5 steps (NeurIPS version numbers; the arXiv HTML renders them garbled).
  - RefSpatial-Bench targets multi-step spatial referring.
  - RFT beats Gemini-2.5-Pro by 12.4 points on RefSpatial-Bench (NeurIPS version; the arXiv v3/v4 HTML says 17.4).
  - Relevance: graded reasoning steps is a VLM-side analogue of prediction (b). It concerns a pointing VLM, not an end-to-end VLA's binding. — [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/29416b66c2149872b9d1415a3fd2c5e0-Abstract-Conference.html); [arXiv](https://arxiv.org/html/2506.04308v4)
- **SpatialActor.** arXiv 2511.09555 lists a failure where the Open Drawer task opens the wrong drawer, alongside instruction misunderstanding and semantic grounding confusion. — [arXiv](https://arxiv.org/pdf/2511.09555) (search snippet only)
- **VLA-Arena.** arXiv 2512.22539 describes a failure where the policy grasps "a visually prominent distractor instead of the text-specified target". — [arXiv](https://arxiv.org/pdf/2512.22539) (search snippet only)
- **VIPA-VLA.** arXiv 2512.13080 attributes baseline failures to inaccurate spatial grounding, such as reaching toward incorrect regions. — [arXiv](https://arxiv.org/pdf/2512.13080) (search snippet only)

### Inferences
- Prediction (b) has neighbours: RefSpatial's step counts, and Spa-Bench's ordinal/relational categories. No source states the graded type < instance < sub-region ordering for VLAs.
- C5, a named class of relational sub-region binding errors, looks novel. SpatialActor's wrong-drawer example should be cited as anecdotal precedent.

### Gaps
- Spa-Bench's full text may define failure classes (a search summary mentioned target-selection and spatial-outcome errors, but the abstract does not), so it needs checking.
- I did not search LIBERO-Goal-specific analyses of "top drawer" versus "cabinet top" errors.

## Q5. Is there a mixture-over-bindings or latent-assignment model of VLA or diffusion-policy behaviour?

### Takeaway
None was found. The closest are generic latent-plan/mixture imitation models (MCIL/HULC latent plans, Gaussian-mixture action heads) and CAST's observation of mode oscillation. None of these defines latent variables as role-to-entity assignments.

### Cited Findings
- **Latent plans.** HULC and MCIL learn latent goal-conditioned plans (a CVAE plan space) for CALVIN-style language-conditioned imitation. Their latents are plans, not bindings. — [HULC arXiv 2204.06252](https://arxiv.org/pdf/2204.06252); [MCIL RSS 2021](https://www.roboticsproceedings.org/rss17/p047.pdf) (search results)
- **Continual imitation grounding.** arXiv 2610.00542 uses a Gaussian-mixture action head and finds that continual-learning performance does not guarantee language grounding. — [arXiv](https://arxiv.org/html/2610.00542) (search snippet only)
- **Mode oscillation.** CAST (arXiv 2508.13446) reports that VLAs oscillate between modes when several plausible behaviors fit the image. This is consistent with a mixture over bindings, but it is not modelled as one. — [arXiv](https://arxiv.org/pdf/2508.13446) (search snippet only)

### Inferences
- The explicit mixture pi = sum_b P(b|o,l) O(a|o,p_b) appears novel as a model of VLA behaviour. Its testable signature is multimodal rollouts that reach different scene entities with the same motor primitive, and CAST's oscillation is suggestive evidence for it.

### Gaps
- Object-centric or slot-based policy papers (for example slot-attention policies) that might implicitly assign slots to roles were not searched.

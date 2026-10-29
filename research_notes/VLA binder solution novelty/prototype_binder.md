# Novelty check: unsupervised prototype-based binder for robot instructions (contact-event referents + DINO/VLA-patch body/part regions + two-channel Gaussian prototypes + MDL channel assignment + PoE binding inside a parse parent, plugged into a frozen VLA)

Provenance note. Items marked [fetched] were read this session (abstract or full text). Items marked [known] are standard references cited from prior knowledge with their arXiv/venue IDs. Their content was not re-read this session, so a writer should spot-check them before quoting specifics. Search budget was about 16 calls, so the coverage is broad but shallow.

## Q1. Cross-situational word learning, and contact events in robot demos as the supervision for word-referent learning

### Takeaway
Cross-situational learning is old and well covered, including on robots and including unsupervised assignment of words to perceptual channels. I found no paper that uses gripper grasp and release events in robot demonstrations as the referent selector for *learning a lexicon*. The closest modern work, SPARC (2606.13497, Oct 2026) and Robo2VLM (2505.15517), already segments demos by gripper phase to find the manipulated object and the goal. Both do it to *auto-annotate* data for VLM/VLA training, and both get object names from an LLM parse plus an open-vocabulary detector, so neither learns word-referent mappings. The contact-event idea is therefore partly prior art as a signal and novel as a word-learning supervisor.

### Cited Findings
- Classic CSL. Siskind 1996 (Cognition 61); Yu & Smith 2007 (Psych. Sci.); Frank, Goodman & Tenenbaum 2009 (Psych. Sci., Bayesian intention-based word learning); Fazly, Alishahi & Stevenson 2010 (Cognitive Science, incremental probabilistic CSL). [known] Frank et al. also appears as background in [the Taniguchi Advanced Robotics line](https://arxiv.org/pdf/2106.08574).
- Roy & Pentland CELL (Cognitive Science 26(1):113-146, 2002) segments speech, forms visual categories and associates words with them, using cross-modal mutual information. Its scale was small (about 20 words, 7 visual categories). [fetched snippet] — [MIT Media Lab](https://www.media.mit.edu/publications/learning-words-from-sights-and-sounds-a-computational-model-2); [survey 2104.13225](https://arxiv.org/pdf/2104.13225)
- **Taniguchi, Taniguchi & Cangelosi 2017, "Cross-Situational Learning with Bayesian Generative Models for Multimodal Category and Word Learning in Robots" (Front. Neurorobotics 11:66)** is the closest prior work on channel assignment. It keeps a GMM per sensory channel (action, position, object, color) with a stick-breaking prior. Each word is tied to a category in exactly one channel, and the model infers which. Mutual-exclusivity constraints allow one channel per word and each channel at most once per sentence. Training is unsupervised Gibbs sampling. Object features come from CaffeNet+PCA, and the action features include hand position relative to the object, finger bend, joint angles and tactile data, with grasp/touch/reach categories. It runs on the iCub simulator and a real iCub (14-16 words, 14 objects). Word-to-channel assignment accuracy is about 0.99. Positions are absolute table coordinates, not relative to a parent. The model is bag-of-words with no grammar, and grammar is listed as future work. [fetched] — [Frontiers](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2017.00066/full)
- Roesler et al. ground words cross-situationally and without supervision on robots (HSR), online with no training phase. They cluster percepts and link them to words via co-occurrence, using object geometry plus robot joint kinematics during actions. [fetched snippet] — [arXiv 2007.04304](https://arxiv.org/pdf/2007.04304); [Springer 2019](https://link.springer.com/chapter/10.1007/978-3-030-36150-1_19)
- ReSCAM / ReSCAM+O (Advanced Robotics 2021) learn relative spatial concepts ("front", "right") from spoken utterances with no prior word knowledge. They estimate which object is the reference and learn object words jointly. [fetched snippet] — [arXiv 2106.08574](https://arxiv.org/pdf/2106.08574)
- Pepper robot grounded word learning pairs human-shown take/put/push actions with speech, learning word-object and word-action maps in real time. [snippet] — [ACM 10.1145/3267851.3267903](https://dl.acm.org/doi/10.1145/3267851.3267903)
- Robotics grounded-semantics lines [known]: Matuszek et al. 2012, joint model of language and perception for grounded attribute learning ([arXiv 1206.6423](https://arxiv.org/abs/1206.6423); not confirmed by search this session); Tellex/Kollar G3 Generalized Grounding Graphs ([arXiv 1712.01097](https://arxiv.org/pdf/1712.01097)); Thomason et al. 2016 "I Spy" multi-modal grounded semantics (IJCAI).
- Modern developmental analogue. Vong, Wang, Orhan & Lake 2024 (Science, doi 10.1126/science.adi1374) learn word-referent maps contrastively from 61 h of one child's headcam. [snippet] — [Gwern PDF](https://Gwern.net/doc/ai/nn/cnn/2024-vong.pdf); robustness follow-up [arXiv 2507.14749](https://arxiv.org/html/2507.14749v2)
- **SPARC (Blank, ..., Reuss, Lioutikov; arXiv 2606.13497 v2, 14 Oct 2026)** is the closest match for the contact-event signal.
  - It extracts closed-gripper intervals from the gripper signal and treats the adjacent open intervals as approach and release.
  - Phase-aligned object motion plus 3D gripper proximity pick the manipulated object.
  - The goal is chosen by counting the object's tracked points inside candidate target boxes on late frames.
  - Qwen3.6-30B parses the instruction, and LLMDet, SAM2.1, RobotSeg, AllTracker and MoGe-2 handle perception.
  - The output is a 511K VQA set and reasoning VLAs.
  - It does not learn word-referent mappings. Language only names objects for open-vocabulary detection.
  [fetched] — [arXiv 2606.13497](https://arxiv.org/html/2606.13497)
- Robo2VLM (Chen, Xie, Ma, Goldberg; arXiv 2505.15517, NeurIPS 2025) segments Open-X trajectories into manipulation phases from end-effector pose, gripper aperture and force-torque. It generates 684K VQA questions. SPARC cites it as the basis for its grasp-phase extraction. [snippet] — [arXiv 2505.15517](https://arxiv.org/pdf/2505.15517)
- Egocentric analogue: hand contact as weak supervision for nouns. WISH (arXiv 2509.26004, 2025) learns in-hand object segmentation from narrations only, aligning nouns to candidates "in contact with the hand", and reaches more than 50% of fully supervised performance on EPIC-Kitchens/Ego4D. Shen et al. CVPR 2024 segment referred objects from narrated egocentric video. Helping Hands (2308.07918) uses hand-object contact detections. Fathi et al. CVPR 2011 used MIL with object names per activity. [snippet] — [arXiv 2509.26004](https://arxiv.org/html/2509.26004v1); [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Learning_to_Segment_Referred_Objects_from_Narrated_Egocentric_Videos_CVPR_2024_paper.pdf); [arXiv 2308.07918](https://arxiv.org/pdf/2308.07918); [Fathi 2011](https://ai.stanford.edu/~alireza/publication/CVPR11.pdf)
- GraspVLA (2505.03233) supervises target bounding boxes and the pre-closure grasp pose. Its boxes come from synthetic/web labels, not learned from contact. [snippet] — [arXiv 2505.03233](https://arxiv.org/html/2505.03233)

### Inferences
- **Verdict, contact-event referents: PARTIAL.** Gripper-phase segmentation to find the manipulated object and the goal is published, and recent (SPARC 2026, Robo2VLM 2025). Hand contact as noun supervision is published for egocentric video (WISH 2025). Using these events as the *only* word-referent supervisor, with no detector and no LLM naming, appears unclaimed. The same holds for separate object and goal roles both learned cross-situationally.
- Strongest objection: "SPARC already localizes object and target from grasp/release; you replaced its LLM+detector with co-occurrence counting, which is Siskind/Fazly. The contact heuristic is engineering, and LIBERO language is templated, so CSL is trivial there." Pre-empt it with ambiguity curves (number of distractors, synonym rate) and a comparison against a SPARC-style detector pipeline.
- Rename risk: the object-at-grasp, goal-at-release role split is close to SPARC's "initial box / target box". Cite SPARC prominently.

### Gaps
- I did not search ACL/CogSci 2023-2026 for "contact" or "grasp" as cross-situational cues specifically. A paper in developmental robotics venues (ICDL, Frontiers) may exist.
- The SPARC full related-work table was only partly read (first 100K chars).

## Q2. Unsupervised discovery of word categories via perceptual-channel consistency, and MDL for lexicon learning

### Takeaway
Choosing the channel in which each word is tightest is directly prior art in Taniguchi et al. 2017, which does it with Bayesian per-channel GMMs and Gibbs sampling. MDL lexicon learning is prior art in de Marcken 1996 and Goldwater et al. 2009, though those are text-only segmentation models. The defensible new parts are narrower:
- a *relative-to-parent geometry* channel, so part and relation words land there;
- MDL used as the channel-selection criterion over frozen foundation features;
- function words treated explicitly as words that fit no channel.

### Cited Findings
- Taniguchi et al. 2017 infer each word's channel among action/position/object/color, unsupervised, on a real iCub. [fetched] — [Frontiers](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2017.00066/full)
- ReSCAM+O jointly learns relative spatial words, reference objects and object words. [snippet] — [arXiv 2106.08574](https://arxiv.org/pdf/2106.08574)
- SpCoA++/SpCoSLAM learn place words from location plus speech. [snippet] — [arXiv 1602.01208](https://arxiv.org/pdf/1602.01208)
- Kartmann & Asfour 2023 learn spatial relation models ("right of", "between") as cylindrical distributions from a few demos, incrementally, on a humanoid. This is supervised by relation label. [snippet] — [arXiv 2305.09551](https://arxiv.org/pdf/2305.09551)
- Spranger / Steels grounded naming-game spatial lexicons work on continuous perceptual spaces with robots. [snippet] — [ResearchGate](https://www.researchgate.net/publication/261041195_Grounded_Lexicon_Acquisition_-_Case_Studies_in_Spatial_Language)
- de Marcken 1996 PhD thesis, "Unsupervised Language Acquisition" (MDL lexicon), arXiv cmp-lg/9611002. Goldwater, Griffiths & Johnson 2009, Cognition (Bayesian word segmentation). [known, not re-verified]
- Roy & Pentland CELL scores word-category pairs by cross-modal mutual information, a close cousin of an MDL criterion. [snippet] — [MIT](https://www.media.mit.edu/publications/learning-words-from-sights-and-sounds-a-computational-model-2)

### Inferences
- **Verdict, MDL channel assignment: PRIOR ART in concept, PARTIAL in specifics.** Taniguchi 2017 is the paper a reviewer will cite. Its channels are absolute position, color, object CNN features and action, and it has no part/relative-geometry channel and no parse.
- Strongest objection: "This is Taniguchi et al. 2017 (and Roy 2002) with DINOv2 features; MDL vs Bayesian GMM-with-ME is a cosmetic swap." The rebuttal has to come from the relative-to-parent channel, the explicit parse-scoped evaluation, and scale (a real VLA benchmark rather than 14 words).

### Gaps
- I found no paper that uses an MDL/description-length criterion to choose a word's *perceptual channel* specifically, as opposed to Bayesian posterior or mutual information. Absence is weak evidence given the budget.

## Q3. Neuro-symbolic / modular grounding, part-level grounding, PoE concept composition, prototypes

### Takeaway
Parse-driven, modular, relation-aware grounding (NMN, CMN, MAttNet, NS-CL) and product-of-experts concept composition (Du et al.) are mature. All are supervised by answers or boxes, or are generative rather than referential. The proposed binder is closest to NS-CL in spirit, with these differences:
- concepts as Gaussians in frozen-feature channels instead of learned embeddings;
- supervision from contact, not QA;
- "A of B" resolved by recursing into B's parts.

Recursive part-within-parent resolution is the least covered piece.

### Cited Findings
- [known] Neural Module Networks ([1511.02799](https://arxiv.org/abs/1511.02799)); NS-VQA ([1810.02338](https://arxiv.org/abs/1810.02338)); NS-CL, Mao et al. 2019 ([1904.12584](https://arxiv.org/abs/1904.12584)), which learns visual concepts as embeddings from QA, with no annotation of concepts, via a program executor; MAttNet ([1801.08186](https://arxiv.org/abs/1801.08186)), with subject/location/relation modules and box supervision; FALCON, few-shot concept learning in NS-CL style ([2203.16639](https://arxiv.org/abs/2203.16639)).
- Compositional Modular Networks (Hu et al. 2017, [1611.09978](https://arxiv.org/abs/1611.09978)) parse expressions into entities and relationships with pairwise relation modules. [snippet] — [RSS18 grounding paper](https://www.roboticsproceedings.org/rss14/p28.pdf); [graph-structured REF 2004.08814](https://arxiv.org/pdf/2004.08814)
- Part-aware weakly supervised REG exists (Springer 2023). [snippet] — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-53311-2_36)
- PoE: products treat experts as soft constraints, while mixtures treat them as prototypes. [snippet] — [Scholarpedia](http://www.scholarpedia.org/article/Product_of_experts). Du et al., Compositional Visual Generation with EBMs ([2004.06030](https://arxiv.org/abs/2004.06030)) [known]; Composable Diffusion ([2206.01714](https://arxiv.org/abs/2206.01714)) [known]; Product of Experts for Visual Generation ([2506.08894](https://arxiv.org/html/2506.08894v1)) [snippet]. All of these are generative.
- [known] Part grounding: PartSLIP ([2212.01558](https://arxiv.org/abs/2212.01558)), PartGLEE ([2407.16696](https://arxiv.org/abs/2407.16696)). Prototypes: ProtoPNet ([1806.10574](https://arxiv.org/abs/1806.10574)), Prototypical Networks ([1703.05175](https://arxiv.org/abs/1703.05175)).

### Inferences
- **Verdict, PoE over content words: PRIOR ART (as composition), PARTIAL for referential binding.** A product of per-word Gaussians over regions is a log-linear AND of attribute classifiers, which is classic (Matuszek 2012 and G3 factor graphs multiply per-word factors).
- **Verdict, parse-scoped recursion ("A of B" = resolve B, then A among B's parts in parent-relative coordinates): PARTIAL to NOVEL.** NS-CL/NMN do relational filtering over objects. Part-referring datasets exist, but I found no unsupervised system that recurses into a discovered part hierarchy with parent-relative geometry.
- Strongest objection: "This is NS-CL/G3 with Gaussians instead of MLPs, on templated LIBERO language; the parse is given or trivial."

### Gaps
- I did not search the 2024-2026 part-referring-expression datasets (e.g., RefCOCO-part variants) or check whether any are unsupervised.

## Q4. Robotics binders (CLIPort, VIMA, Grounded-SAM pipelines, RoboPoint, RoboRefer, SoFar, MOKA, KITE, ConceptGraphs, HOV-SG, Code-as-Policies, VoxPoser): which are unsupervised, tiny, prototype-based, role-factorised?

### Takeaway
None of these are unsupervised in the lexical sense. Each relies on a pretrained VLM/LLM/CLIP for word meaning, and most are large. Some are role-factorised, separating the pick target from the place target or a keypoint from a waypoint, but always via a pretrained open-vocabulary model or supervised data. A few-KB, demo-learned, prototype lexicon attached to a frozen VLA has no direct counterpart I could find. The 2026 frozen-VLA wrappers (Harness VLA, IGAR) do grounding through planners or attention recalibration, not learned lexicons.

### Cited Findings
- [known] CLIPort ([2109.12098](https://arxiv.org/abs/2109.12098)): CLIP semantic stream with pick/place affordance heads (role-factorised, supervised by demos, large). VIMA ([2210.03094](https://arxiv.org/abs/2210.03094)): multimodal prompts with object tokens. KITE ([2306.16605](https://arxiv.org/abs/2306.16605)): keypoint grounding, supervised. MOKA ([2403.03174](https://arxiv.org/abs/2403.03174)): VLM mark-based affordances for grasp point and target point (role-factorised, zero-shot via GPT-4V). RoboPoint ([2406.10721](https://arxiv.org/abs/2406.10721)), RoboRefer ([2506.04308](https://arxiv.org/abs/2506.04308)), SoFar ([2502.13143](https://arxiv.org/abs/2502.13143)): VLMs fine-tuned on large synthetic spatial-referring data. Grounded SAM ([2401.14159](https://arxiv.org/abs/2401.14159)). ConceptGraphs ([2309.16650](https://arxiv.org/abs/2309.16650)) and HOV-SG ([2403.17846](https://arxiv.org/abs/2403.17846)): open-vocabulary 3D scene graphs queried by LLM, with no learned lexicon. Code-as-Policies ([2209.07753](https://arxiv.org/abs/2209.07753)) and VoxPoser ([2307.05973](https://arxiv.org/abs/2307.05973)): LLM-programmatic, zero-shot via LLM+VLM.
- Harness VLA (2607.08448) wraps a frozen VLA as a contact-rich primitive. A planner does target grounding, transport and release, giving 82.4% vs 50.0% for the direct frozen policy on perturbed LIBERO-Pro. [snippet] — [arXiv 2607.08448](https://arxiv.org/html/2607.08448)
- IGAR (2603.06001) is training-free attention recalibration toward instruction tokens in pi0/pi0.5/OpenVLA-OFT, tested on 30 LIBERO tasks. [snippet] — [arXiv 2603.06001](https://arxiv.org/html/2603.06001)
- GSR, "Grounded Semantic Re-Binding" (Yin & Zhang, 2608.02497, Aug 2026), fuses independently extracted task semantics with native visual features and retrains an action expert from scratch on canonical demos. The abstract gives no word-referent mechanism. [fetched abstract] — [arXiv 2608.02497](https://arxiv.org/abs/2608.02497)
- LiLo-VLA (2602.21531) composes object-centric VLA skills. ProGAL-VLA (2604.09824) is grounded-alignment reasoning. [snippet] — [2602.21531](https://arxiv.org/html/2602.21531v1); [2604.09824](https://arxiv.org/html/2604.09824)
- SPARC's annotations train reasoning VLAs that beat detection-annotated policies in clutter. [fetched] — [arXiv 2606.13497](https://arxiv.org/html/2606.13497)

### Inferences
- **Verdict, a tiny unsupervised prototype binder plugged into a frozen VLA: NOVEL as a combination.** The closest are GSR (re-binding, but it retrains the action expert, and its mechanism is not in the abstract), Harness VLA (frozen VLA plus external grounding by planner) and SPARC (contact-derived roles, but annotation for training).
- Strongest objection: "Why not Grounded-SAM / a VLM pointer (RoboPoint, MOKA)? They are zero-shot and open-vocabulary." The paper must show either that the binder handles words the VLM does not resolve in this scene (in-domain ad hoc names, parts) or the parameter, compute and no-annotation advantage. It also needs a head-to-head against a Grounded-SAM binder.

### Gaps
- GSR's full method was not read. It is the most likely "same idea" collision in 2026 and should be read in full.
- I did not check whether OA-WAM or other LIBERO-Plus object-binding methods use demos or prototypes.

## Q5. Unsupervised body-then-part discovery from self-supervised features

### Takeaway
Clustering DINO/ViT patch features into objects and parts is well established: Amir et al. 2112.05814, Deep Spectral Methods, Leopart, PARTICLE. Hierarchical whole-to-part trees from SSL ViTs also exist (CAST, DependencyViT). Clustering a *VLA's own vision tower* (SigLIP inside pi0 etc.) for this is less common but is a feature swap.

### Cited Findings
- Amir et al., "Deep ViT Features as Dense Visual Descriptors" (2112.05814), do part co-segmentation from DINO features and get temporally consistent parts in video. They are competitive with supervised methods on CUB parts. [snippet] — [arXiv 2112.05814](https://arxiv.org/pdf/2112.05814)
- PARTICLE (2309.13822) runs k-means on DINO last-layer key features per image to get parts. [snippet] — [arXiv 2309.13822](https://arxiv.org/pdf/2309.13822)
- Choudhury et al. NeurIPS 2021 argue that clustering pre-computed features as a proxy for parts is unlikely to find meaningful parts. [snippet] — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/ec8ce6abb3e952a85b8551ba726a1227-Paper.pdf). A 2025 fine-grained GCD paper (2507.06928) reports that single-image DINO patch clustering is "highly unreliable". [snippet] — [arXiv 2507.06928](https://arxiv.org/pdf/2507.06928)
- Leopart over-clusters dense patches, then merges parts via community detection. [snippet] — [arXiv 2204.13101](https://www.arxiv.org/pdf/2204.13101)
- CAST (2210.00314, ICLR 2024) builds hierarchical segmentation into a ViT from unlabeled ImageNet, beating SAM by 8% mIoU on PartImageNet; its ViT baseline uses fine-to-coarse K-means. DependencyViT (2304.03282) gets a dependency tree from reversed attention for unsupervised parts. PDiscoFormer (2407.04538) does part discovery on SSL ViTs. [snippet] — [2210.00314](https://arxiv.org/pdf/2210.00314); [2304.03282](https://arxiv.org/pdf/2304.03282); [2407.04538](https://arxiv.org/pdf/2407.04538v2)

### Inferences
- **Verdict, body-to-part regions: PRIOR ART.** Present it as a borrowed component, citing Amir et al. and CAST. Expect the objection that single-image DINO part clustering is unreliable (Choudhury 2021; 2507.06928), and add a part-quality check on the target scenes.

### Gaps
- I did not verify whether anyone clusters VLA vision-tower (SigLIP/PaliGemma) patch features into objects/parts for grounding. This is likely only partially covered by "VLA attention" interpretability papers.

## Overall combination verdict (synthesis; inference)
- Each component on its own:

  | Component | Verdict | Closest prior work |
  |---|---|---|
  | Contact events | PARTIAL | SPARC/Robo2VLM as the signal; WISH for hand contact as noun supervision |
  | Region discovery | PRIOR ART | 2112.05814, CAST |
  | Per-channel Gaussian word models with unsupervised channel assignment | PRIOR ART | Taniguchi 2017 |
  | PoE AND-composition | PRIOR ART | G3, Matuszek |
  | Parse-scoped recursion into parts with parent-relative geometry | PARTIAL/NOVEL | none found |
  | Few-KB plug-in to a frozen modern VLA | NOVEL as a setting | none found |

- **The combination looks NOVEL:** contact-supervised, annotation-free lexicon, part hierarchy, MDL channel split, attached to a frozen flow VLA. I found no paper doing it through Oct 2026. Main collision risks: SPARC (Oct 2026), GSR (Aug 2026), and the Taniguchi lab line.
- **Strongest reviewer objection:** "This is 2010-era grounded language learning (Roy 2002, Taniguchi 2017, Matuszek 2012, G3) with DINOv2 features and SPARC's gripper-phase trick; on templated LIBERO instructions, cross-situational learning is trivially solvable, and a Grounded-SAM/VLM pointer is a stronger zero-shot binder."
- Evidence that would answer it:
  - non-templated or paraphrased language and distractor-heavy scenes;
  - part words that open-vocabulary detectors miss;
  - a sample-efficiency curve of words learned per demo;
  - head-to-head against Grounded-SAM and RoboPoint binders, and against a SPARC-style pipeline;
  - an ablation of MDL against Taniguchi-style Bayesian channel assignment.

# P4 novelty check: compositional binding as invariant prediction across layouts (ICP/IRM), identifiability, transfer-scored MDL, shape bias, multiplicativity

Provenance tags: [full text] = read from arXiv HTML this session (in some cases only the first ~100k characters, noted). [abstract/snippet] = abstract page or search snippet only. [not fetched; prior knowledge] = cited from memory of the paper; verify before you cite it in a paper. Date of check: 2026-10-27. I made about 17 tool calls, so coverage is targeted and not exhaustive.

## Q1. Has anyone formally linked compositional/systematic generalisation to invariance / ICP / IRM / causal invariance?

### Takeaway
The closest compositional-generalisation theory papers (Wiedemer et al. 2307.05596, Mahajan et al. CRM 2410.06303) never frame components as invariant predictors across environments, and Wiedemer et al. never state the observational-equivalence (non-identifiability) argument. So the specific mapping "an atom expert is compositional iff its referent likelihood is invariant across layouts, with layouts as environments" looks unclaimed. The mathematical content is a different matter. "A feature perfectly correlated with the label in all training environments cannot be separated from the causal feature" is the textbook IRM/ICP point: Arjovsky's cow/camel background example, plus the environment-diversity conditions in ICP. A reviewer will call P4 "IRM restated with layouts as backgrounds" unless the contribution is located somewhere else: the compositional/referring-expression instantiation, the transfer-scored criterion, and the predictions.

### Cited Findings
- **Wiedemer et al. 2023, "Compositional Generalization from First Principles" (2307.05596) [full text, HTML]:**
  - Definition 2, *compositional support*: the train and test marginal supports match for every component.
  - Definition 3, *sufficient support*: for each component value there are training points whose summed Jacobian of the composition function C has full rank.
  - Theorem 1: if the shared composition function C is known and both conditions hold, fitting on P implies fitting on Q.
  - The paper says "compositional generalization requires compositional support". It notes that "the compositional representation of a function f is also not unique", and that the theorem holds "if there exists at least one [representation] for which the assumptions are fulfilled".
  - None of the terms invariance, ICP, IRM, environments or causal representation learning appear. The paper never states that alternative decompositions fit training data equally well when the training data do not vary.
  - [Source](https://arxiv.org/html/2307.05596)
- **Mahajan, Pezeshki, Arnal, Mitliagkas, Ahuja, Vincent, "Compositional Risk Minimization" (2410.06303, ICML 2025) [full text, first ~100k of 179k chars]:**
  - Setup: attributes z = (z1..zm) define groups, and p(x|z) ∝ exp(-Σ_i E_i(x, z_i)) (an additive energy distribution).
  - Theorem 1: a learned AED that matches on the training groups extrapolates to the discrete affine hull of the training one-hots.
  - Appendix D.6 shows that extrapolating beyond the affine hull "is indeed necessary". This is a support/necessity result.
  - IRM appears only as a baseline and in related work, with the comment that such guarantees "require a large diversity in terms of the environments". The paper never treats groups as environments. Its invariance is the modelling assumption that p(x|z) is invariant from train to test, and it is not used as a learning criterion. The paper also invokes the independent-mechanisms principle (Janzing & Schölkopf).
  - [Source](https://arxiv.org/html/2410.06303v3)
- **Wiedemer, Brady et al., "Provable Compositional Generalization for Object-Centric Learning" (2310.05327, ICLR 2024 oral) [abstract/snippet]:** structural assumptions on the decoder plus encoder-decoder consistency give provable compositional generalisation. The abstract does not mention invariance or environments. [Source](https://arxiv.org/abs/2310.05327)
- **Lippl & Stachenfeld, "When does compositional structure yield compositional generalization? A kernel theory" (2405.16391, ICLR 2025) [abstract]:**
  - Kernel models with compositional representations are restricted to "conjunction-wise additivity": they assign values to the component combinations seen in training and sum them.
  - The paper names two failure modes caused by training-data biases, "memorization leak and shortcut bias".
  - The abstract does not mention invariance. "Shortcut bias" is the nearest neighbour to the layout-shortcut claim.
  - [Source](https://arxiv.org/abs/2405.16391)
- **Arjovsky et al., IRM (1907.02893) [abstract/snippet + not fetched; prior knowledge]:** learns nonlinear invariant causal predictors from multiple training environments for OOD generalisation. Its motivating example is a classifier that uses the background (grass versus sand) to tell cows from camels. This is structurally the same as "a layout code predicts the referent". [Source](https://arxiv.org/abs/1907.02893)
- **Search for a direct link:** a targeted search for "compositional generalization" + IRM / causal invariance / environments returned no 2024-2025 paper that frames compositional generalisation as invariance across environments. The closest hits were CRM and generic causal-identifiability papers, such as "Learning Causal Representations from General Environments: Identifiability and Intrinsic Ambiguity" (2311.12267). [Source](https://arxiv.org/pdf/2311.12267); [Source](https://arxiv.org/pdf/2410.06303)
- **ICP for Block MDPs (Zhang et al., ICML 2020) [snippet]:** "applying invariant causal prediction methods in the multi-environment setting will yield the minimal causal set of variables when the training environment interventions satisfy certain conditions necessary for the identifiability of the causal variables". This is the standard statement that ICP identifiability depends on how diverse the environments are. [Source](http://proceedings.mlr.press/v119/zhang20t/zhang20t.pdf)
- **Grounding-specific shortcut evidence:**
  - Cirik, Morency, Berg-Kirkpatrick (NAACL 2018, 1805.11818) [snippet]: an image-only model reaches 71.2% top-2 precision on referring-expression recognition without seeing the expression, so the dataset biases are exploitable. [Source](https://www.arxiv.org/pdf/1805.11818)
  - Liu et al., "Deconfounded Visual Grounding" (2112.15324) [snippet]: grounding models learn "spurious correlations between certain language pattern and object location". This is the layout/location confound in grounding, handled with a causal deconfounder. [Source](https://arxiv.org/pdf/2112.15324)
- **Not fetched; prior knowledge, verify before citing:**
  - Peters, Bühlmann, Meinshausen, ICP (1501.01332): ICP's confidence guarantee holds without further assumptions, but *identification* of the causal parents needs sufficiently many and sufficiently different interventions. [Source](https://arxiv.org/abs/1501.01332)
  - Rosenfeld, Ravikumar, Risteski, "The Risks of Invariant Risk Minimization" (2010.05761): linear IRM needs more environments than the number of spurious dimensions, and it can fail in the nonlinear regime. [Source](https://arxiv.org/abs/2010.05761)
  - Schölkopf et al., "Toward Causal Representation Learning" (2102.11107): the independent mechanisms / sparse mechanism shift framing. [Source](https://arxiv.org/abs/2102.11107)
  - Lachapelle et al., "Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation" (2307.02598): additivity gives identifiability and extrapolation to the Cartesian product of the training supports. [Source](https://arxiv.org/abs/2307.02598)
  - Brady et al., "Provably Learning Object-Centric Representations" (2305.14229): identifiability under "compositionality + irreducibility" decoder assumptions. [Source](https://arxiv.org/abs/2305.14229)
  - Geirhos et al., "Shortcut Learning in Deep Neural Networks" (2004.07780): shortcuts are decision rules that do well on the i.i.d. test set but fail under distribution shift. [Source](https://arxiv.org/abs/2004.07780)

### Inferences
Verdicts per sub-claim:
- **"Compositional iff invariant across layouts (layouts = environments)": PARTIAL.**
  - Prior art: the abstract equivalence "causal or robust predictor = invariant across environments" (ICP/IRM).
  - Novel as far as I found: casting each *atom/word expert's referent likelihood* as the invariant mechanism, and *scene layouts* as the environments. No compositional-generalisation theory paper I read uses environments. CRM uses "invariance" only as a fixed modelling assumption.
- **Identifiability ("when a training layout determines the referent, a layout code and an appearance code are observationally equivalent, so any training-fit criterion cannot distinguish them"): PRIOR ART in substance.**
  - This is the perfect-spurious-correlation non-identifiability that motivates IRM.
  - It is also the "compositional support" necessity in Wiedemer et al.: their Definition 2 is a support condition, and they explicitly note that representations are non-unique.
  - What may be new: stating it as an observational-equivalence theorem *for binding codes* (layout vs appearance) and tying it to the scoring criterion. Wiedemer and CRM never state the equivalence explicitly, and Wiedemer does not frame it as non-identifiability.
- **"Identified only by variation across contexts plus a transfer-scored criterion": PARTIAL / likely NOVEL as a combination.**
  - The variation half is the standard ICP environment-diversity condition, and the Wiedemer compositional/sufficient-support condition.
  - The transfer-scored criterion is related to Bengio et al. 2019, a meta-transfer objective that uses adaptation speed under distribution change to identify causal structure (1901.10912; not fetched; prior knowledge, verify). It is also related to prequential MDL for causal structure (Bornschein et al., see Q2). I found no work that applies it to compositional or word-meaning channels.

Strongest reviewer objections:
1. "This is IRM's cow/camel argument with layouts as backgrounds. The identifiability claim is the standard support / environment-diversity condition (Wiedemer Def. 2; ICP intervention diversity; Rosenfeld's environment-count bound)."
2. "IRM itself fails in practice (Rosenfeld 2010.05761; Gulrajani & Lopez-Paz 'In Search of Lost Domain Generalization' 2007.01434, not fetched). Why would an IRM framing help?" The answer has to be that P4 uses invariance as a *diagnostic/selection criterion* through transfer-scored MDL, and not as an IRM penalty.
3. "If you need varied layouts anyway, the theory adds nothing beyond 'train with augmentation'." The defence is the testable predictions: failure exactly when training layouts determine referents, and recovery as soon as the support is varied.

### Gaps
- I did not read the full text of Lachapelle 2307.02598, Brady 2305.14229, Schölkopf 2102.11107, Jarvis et al. (compositional structure), or the full IRM/ICP papers this session. My statements about them are from prior knowledge.
- I could not run a full-text search of "systematic generalization" + "invariant" across 2025-2026 arXiv and OpenReview. A 2025-2026 paper making exactly this link could exist and I did not find it.
- I did not locate any "Jarvis et al." compositional-structure paper; the identity of that reference was not determined.
- CRM appendix A (further related work) was not read; it was truncated after 100k characters.

## Q2. Prequential / online-code MDL for model or structure selection, and its use for compositional structure or word-meaning channels

### Takeaway
Prequential MDL is established for probing (Voita & Titov) and for neural and causal structure selection (Blier & Ollivier; Bornschein et al.). Elmoznino, Jiralerspong, Bengio & Lajoie (2410.14817) already *define* compositionality via Kolmogorov complexity and *estimate* it with prequential coding. That paper is the most dangerous prior art for the MDL leg. It scores compression on the data the system was given, however, and does not use leave-context-out or transfer scoring to separate a layout code from an appearance code. I found no prior use of leave-context-out prequential MDL to select word-meaning channels.

### Cited Findings
- **Elmoznino, Jiralerspong, Bengio, Lajoie, "A Complexity-Based Theory of Compositionality" (2410.14817, v5 June 2025) [full text, first ~100k of 188k chars]:**
  - Defines compositionality as C(Z) = K(Z)/K(Z|W), with K(Z) = K(p_w) + K(W|p_w) + K(f) + K(Z|W,f).
  - Compositional representations are those "that can be significantly compressed as a function of constituent parts".
  - The parts and semantics are the components of the shortest program, not free choices.
  - Uses "a compression method called prequential coding" to estimate K(Z|W^L) for emergent and natural languages.
  - Links systematicity to f generalising to novel sentences, and says "Equivariances (and invariances) are symmetries—sources of structure that decreases the complexity of a function".
  - In the part I read, I found no treatment of variation across training contexts or of a leave-context-out criterion.
  - [Source](https://arxiv.org/html/2410.14817)
- **Bornschein et al., "Prequential MDL for Causal Structure Learning with Neural Networks" (2107.05481) [abstract/snippet]:** uses prequential MDL to score Bayesian-network structures with flexible neural conditionals. [Source](https://arxiv.org/abs/2107.05481)
- **Grünwald, "The MDL Principle", ch. 17 [snippet]:** prequential coding estimates parameters from the data prefix and encodes each new point with its conditional probability. [Source](https://homepages.cwi.nl/~pdg/book/chapter17.pdf)
- **Not fetched; prior knowledge, verify:**
  - Blier & Ollivier, "The Description Length of Deep Learning Models" (1802.07044): the prequential/online code gives much shorter codelengths than variational codes. [Source](https://arxiv.org/abs/1802.07044)
  - Voita & Titov, "Information-Theoretic Probing with Minimum Description Length" (2003.12298): online-code MDL probing, which separates a representation's information from probe capacity. [Source](https://arxiv.org/abs/2003.12298)
- **Search result:** a search for prequential MDL to select compositional structure or word meaning returned no such application, beyond Elmoznino et al. and the general MDL literature. [Source](https://arxiv.org/abs/2107.05481)

### Inferences
Verdicts per sub-claim:
- **"Prequential MDL as a model/structure selection criterion": PRIOR ART** (Blier & Ollivier; Voita & Titov; Bornschein et al.).
- **"MDL/compression defines compositionality": PRIOR ART** (Elmoznino et al. 2410.14817).
- **"*Leave-context-out* (transfer-scored) prequential MDL as the criterion that identifies compositional binding where training-fit criteria provably cannot": LIKELY NOVEL.**
  - The argument goes as follows. Standard prequential MDL on i.i.d.-ordered data from training layouts inherits the same observational equivalence: both codes compress the training layouts equally well.
  - Ordering the code by context, so that each new layout is encoded by a model fitted on the other layouts, makes the codelength penalise the layout code.
  - Elmoznino et al.'s K(Z) on fixed data does not do this. I did not see it in the part of their paper I read.

Reviewer objections:
- "Leave-context-out MDL is cross-validation across environments, i.e. leave-one-environment-out model selection (Gulrajani & Lopez-Paz's training-domain / oracle selection, not fetched)."
- "Elmoznino et al. already define compositionality via prequential complexity."
- The defence for both: P4's claim is not that MDL is new. It is that *ordering by context* is necessary, and that it is sufficient under compositional support.

### Gaps
- I did not read the second half of Elmoznino et al. (which may contain generalisation experiments) or Bornschein et al.'s 2022 "Sequential learning of neural networks for prequential MDL" (2210.07931).
- I did not check MDL work in emergent communication (e.g. Kharitonov / Chaabouni compositionality-and-generalisation papers) in detail.

## Q3. Computational models of the shape bias and learned inductive biases, and whether any connect the shape bias to invariance across contexts or identifiability

### Takeaway
The classical hierarchical-Bayesian account (Kemp, Perfors & Tenenbaum 2007) already explains the shape bias as a learned overhypothesis about *within-category feature variability*: shape is homogeneous within a category while colour and texture vary. That is a variability/invariance statistic in all but name. I found no paper that phrases this as ICP/IRM-style invariance or as an identifiability result. The cognitive parallel is fair and well grounded. "Shape bias = invariance learned across exemplar contexts" is a re-description of Kemp et al., so it should be presented as a parallel and not as a new cognitive claim.

### Cited Findings
- **Kemp, Perfors & Tenenbaum 2007, "Learning overhypotheses with hierarchical Bayesian models", *Developmental Science* [abstract/snippet; the PDF could not be text-extracted this session]:**
  - "Inductive learning is impossible without overhypotheses".
  - Hierarchical Bayesian models explain how such overhypotheses are acquired, including "overhypotheses about feature variability (e.g. the shape bias in word learning)".
  - [Source](https://sites.socsci.uci.edu/~lpearl/courses/readings/KempPerforsTenenbaum2007_LearningOverhypotheses.pdf)
  - Prior knowledge, verify: the model is Dirichlet-multinomial. Per-dimension α controls how homogeneous features are within a category and β gives the feature distribution. Low α learned for shape gives the shape bias.
- **Follow-up (Xu and colleagues) [snippet]:** preschoolers rapidly form overhypotheses about the role of *arbitrary* features, not just shape, in how words refer to categories. [Source](https://sites.socsci.uci.edu/~lpearl/courses/readings/KempPerforsTenenbaum2007_LearningOverhypotheses.pdf)
  - Note: this snippet came from the search result for the Kemp page; verify the exact citation.
- **Feinman & Lake, "Learning Inductive Biases with Simple Neural Networks" (1802.02745, CogSci 2018) [abstract]:** networks develop a shape bias after "as few as 3 examples of 4 object categories", using controlled synthetic data. The abstract does not say whether non-shape variation is required. [Source](https://arxiv.org/abs/1802.02745)
- **Ritter et al., "Cognitive Psychology for Deep Neural Networks: A Shape Bias Case Study" (1706.08606) [snippet]:** shape-bias magnitude "varies greatly among architecturally identical, but differently seeded models", and it fluctuates over training at similar accuracy. That is consistent with shape and non-shape solutions being near-equivalent on training data. [Source](https://arxiv.org/pdf/1706.08606)
- **Portelance et al. 2021, "The Emergence of the Shape Bias Results from Communicative Efficiency" [snippet]:** the shape bias is strongest where referents are assumed to be artifacts, so the bias depends on context. [Source](https://nlp.stanford.edu/pubs/portelance2021shape.pdf)
- **In-context VLM inductive biases (2502.01530) [snippet]:** VLMs generally prefer shape over colour, and more so when the examples are visual. [Source](https://arxiv.org/html/2502.01530)
- **Shape-bias meta-analysis [snippet]:** variability across studies is driven by procedure and possible publication bias, more than by theoretical moderators. [Source](https://www.researchgate.net/publication/407543492_Examining_the_Robustness_and_Generalizability_of_the_Shape_Bias_A_Meta-Analysis)
- **Not fetched; verify:**
  - Landau, Smith & Jones 1988 (original shape bias).
  - Smith, Jones, Landau, Gershkoff-Stowe & Samuelson 2002 (*Psych Science*): training on 4 novel categories that are shape-consistent but vary in colour and material accelerates noun vocabulary growth.
  - Colunga & Smith 2005 (*Psych Review*): connectionist account of learned shape/material biases from the statistics of the noun lexicon.
  - Geirhos et al. 2019 (1811.12231): ImageNet CNNs are texture-biased.

### Inferences
- **Verdict on "shape bias parallels learning invariance across contexts": PARTIAL / prior art in substance.**
  - The variability overhypothesis of Kemp et al. and the shape-consistent / colour-varying training of Smith et al. 2002 already show the mechanism.
  - What is new is the explicit identifiability reading. If colour were held constant within each training category, shape and colour would be observationally equivalent cues. The overhypothesis is learnable only because non-shape features vary. This is the P4 identifiability theorem in miniature, and it gives a sharp developmental prediction: a matched-colour training set should yield no shape bias.
  - Kemp et al.'s model should already predict this. Checking whether they state it needs their full text.
- Ritter et al.'s seed variance is useful supporting evidence for "observational equivalence". Many solutions fit the training data equally well, and which bias appears is under-determined.
- Reviewer objection: "The shape bias is already explained as a learned second-order generalisation over feature variability (Smith 2002; Kemp 2007). Calling it invariance adds terminology, not explanation." The defence: P4 contributes the *negative* half, i.e. the impossibility without variation, and the *transfer-scored* criterion.

### Gaps
- The Kemp et al. full text was not read: the PDF was binary, pdftotext was unavailable on the compute node, and HTML access was not tried. I did not verify whether they explicitly analyse the constant-colour (confounded) training regime.
- I did not read Colunga & Smith 2005 or Smith et al. 2002.
- I found no paper that links the shape bias to ICP/IRM or identifiability. The absence is from limited searching, not proof.

## Q4. Multiplicativity of component skills, and whether a referring-depth version exists

### Takeaway
Multiplicative dependence of composite success on component success is established qualitatively: Okawa et al. 2310.09336 ("multiplicative reliance on the performance of constituent tasks"), and Arora & Goyal's skill-mix theory, which is from prior knowledge. Error compounding with compositional depth is shown for multi-step reasoning (Dziri et al. "Faith and Fate", from prior knowledge). I found no referring-expression-depth version, i.e. binding error as the product of per-atom binding errors across nesting depth, and no factorisation success = P(binding) x competence.

### Cited Findings
- **Okawa, Lubana, Dick, Tanaka, "Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task" (2310.09336, NeurIPS 2023) [abstract + snippet]:**
  - Sudden emergence of compositional ability due to "multiplicative reliance on the performance of constituent tasks".
  - The order of emergence is governed by the data-generating process, and composing lower-frequency concepts needs more optimisation steps.
  - The abstract gives no explicit product formula and no depth dependence.
  - [Source](https://arxiv.org/abs/2310.09336); [Source](https://arxiv.org/pdf/2310.09336)
- **Ramesh et al., "Compositional Capabilities of Autoregressive Transformers" (2311.12997) [snippet]:** cites Okawa for components being learned before their compositions. [Source](https://arxiv.org/pdf/2311.12997)
- **Related [snippet]:** "Can Models Learn Skill Composition from Examples?" (2409.19808, skill-mix follow-up). [Source](https://arxiv.org/pdf/2409.19808)
- **CoT Referring (2510.06243) [snippet]:** defines a hop-level complexity measure for compositional grounding. This is the closest existing "referring depth" axis, but it is not a multiplicative theory. [Source](https://arxiv.org/pdf/2510.06243)
- **Not fetched; prior knowledge, verify:**
  - Arora & Goyal, "A Theory for Emergence of Complex Skills and Learning in LLMs" (2307.15936): a random bipartite skill-text graph, where competence on k-tuples of skills follows from per-skill competence. [Source](https://arxiv.org/abs/2307.15936)
  - Dziri et al., "Faith and Fate: Limits of Transformers on Compositionality" (2305.18654): errors propagate through multi-step computation graphs, and they argue that success probability decays roughly exponentially with depth. [Source](https://arxiv.org/abs/2305.18654)

### Inferences
Verdicts per sub-claim:
- **"Factorised binding error ~ product of atom errors across referring depth": PARTIAL.**
  - The multiplicative principle is prior art (Okawa; Arora & Goyal; Dziri depth compounding).
  - The referring-depth / binding instantiation appears new as far as I found.
  - It is only interesting if it is a *falsifiable quantitative prediction*: per-atom binding accuracies measured at depth 1 predict depth-k success with no free parameters.
- **"Success = P(binding) x competence": NOVEL as a stated decomposition for VLAs, as far as I found.** A reviewer will call it an independence assumption and ask for evidence that binding and motor competence are independent: test the product against the joint across tasks.
- Strongest objection: "The product of independent probabilities is trivially true under independence. The content is only in whether independence holds."

### Gaps
- I did not read the full text of Okawa et al. (I could not confirm a formal product model in the body), Arora & Goyal, or Dziri et al.
- I did not search for multiplicative error models in classical referring-expression or semantic-parsing literature (e.g. CLEVR-Ref+ depth analyses, 1901.00850, not checked).

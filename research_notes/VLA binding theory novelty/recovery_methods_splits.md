# Test-time recovery of VLA binding errors, and held-out-at-the-weights compositional splits (novelty check for C4)

Verification note: every arXiv ID below was fetched from its arxiv.org abstract page (or HTML full text where stated) on 2026-10-26, unless it is marked "[snippet only]" or "[UNVERIFIED]". The arxiv abstract pages were summarized by the fetch tool, so the wording here paraphrases the abstracts. Numbers are copied as the tool reported them. Quote-level wording should be re-checked against the PDF before it goes into the paper.

## Q1. Which papers try activation steering, attention steering, CFG or guidance on VLAs to fix grounding, and what do they report on compositional or recombination tasks?

### Takeaway
Several training-free, policy-internal grounding fixes exist: CAG, IGAR attention recalibration, TAG object-erased CFG, Häon et al. FFN steering, and SAE steering. All of them report POSITIVE results, but on counterfactual or contradictory instructions, clutter, or behavior steering. None of them evaluates held-out object x goal recombinations on a policy fine-tuned without those combinations. The only measurements of an internal handle on true recombinations come from CAG run as a baseline in ReGuide and CRAFT. There CAG gives small or inconsistent gains: on public pi0.5, Comp. Avg. goes from 39.0 to 41.4 while Orig. drops from 94.0 to 81.4. In CRAFT, CAG-VA reaches 66.5% on Pick-Place pi0.5 but 1-7% for pi0 and GR00T. This partially supports C4 for one handle (CFG) only.

### Cited Findings
- **CAG, arXiv:2602.17659** (Fang, Feng, Jing, Liu, Yang, Wei, Szafir, Ding; v1 19 Feb 2026, v2 15 Jul 2026), "When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs". Introduces the LIBERO-CF counterfactual benchmark and Counterfactual Action Guidance, a dual-branch method that contrasts the VLA with a language-unconditioned vision-action branch. Training-free CAG improves pi0.5 by 9.7% in language-following accuracy and 3.6% in task success on under-observed tasks. With the trained VA module the gains are 15.5% and 8.5%. On a real robot, counterfactual failures drop by 9.4% and success rises by 17.2% on average. It reports a positive result on counterfactual instructions in seen layouts, and does not test held-out-at-the-weights combinations. — [arXiv 2602.17659](https://arxiv.org/abs/2602.17659)
- **CAG measured on recombinations (inside ReGuide, arXiv:2609.38616, Table 1).** The baseline is public pi0.5: Orig. 94.0, Comp. Avg. 39.0. With CAG: Orig. 81.4; Target Region 53.0; Target Object 35.0; Unseen Object 44.0; Task Scene 33.5; Comp. Avg. 41.4. ReGuide's text says CAG "only" raises Comp. Avg. from 39.0 to 41.4 and lowers Orig. Its Table 17 sweeps CAG's guidance weight. — [arXiv 2609.38616 HTML](https://arxiv.org/html/2609.38616)
- **CAG-VA measured on held-out combinations (inside CRAFT, arXiv:2610.00524).** Undemonstrated success, with guidance strength tuned per model and benchmark: Pick-Place pi0.5 66.5, pi0 6.8, GR00T 4.9; Pick-Place-Press pi0.5 21.8, pi0 1.1, GR00T 2.1. The FT Full baselines on Pick-Place are 8.4, 2.8 and 1.1. CRAFT says CAG-VA "improves generalization in several settings, but its gains vary substantially across models and benchmarks." — [arXiv 2610.00524 HTML](https://arxiv.org/html/2610.00524)
- **IGAR, arXiv:2603.06001** (N. Zhang, B. Zhu, S. Zhou, J. Chen; v1 6 Mar 2026, v2 2 Jul 2026), "Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration". It identifies "linguistic blindness" and builds ICBench from LIBERO with contradictory instructions. Tested on Pi0, Pi0.5 and OpenVLA-OFT, the method finds attention-sink tokens, selects grounding heads, and shifts attention toward under-weighted instruction tokens. On 30 LIBERO tasks it "substantially reduces erroneous execution under contradictory instructions while preserving baseline performance", and it was validated on a Franka arm. The abstract gives no numbers. This is the closest prior work to C4's "early attention steering", but its target is refusing impossible instructions. It does not test rebinding on held-out combinations. — [arXiv 2603.06001](https://arxiv.org/abs/2603.06001)
- **TAG, arXiv:2603.24584** (Zhou, Zhan, Zhai, Lyu, Liu, Wang, Lin, Wang; 25 Mar 2026). This is CFG-inspired guidance that uses an object-erased observation as the null branch. It targets instance-level grounding errors in clutter (near-miss and wrong-object grasps) and reports gains on LIBERO, LIBERO-Plus and VLABench. The abstract gives no numbers or backbones and does not describe a held-out-combination protocol. It is close to C4's "contrastive region guidance" (region-erased contrast), but its setting is clutter. — [arXiv 2603.24584](https://arxiv.org/abs/2603.24584)
- **Häon et al., arXiv:2509.00328** (Häon, Stocking, Chuang, Tomlin; 30 Aug 2025; CoRL 2025, PMLR 305). The method projects FFN activations onto the token-embedding basis to find sparse semantic directions (e.g. speed, direction) and injects them at inference on Pi0 and OpenVLA in LIBERO and on a UR5. The abstract reports no metrics. It steers behavior, and grounding and composition are not addressed. — [arXiv 2509.00328](https://arxiv.org/abs/2509.00328); [PMLR](https://proceedings.mlr.press/v305/haon25a.html). [UNVERIFIED] that vla-mech-interp.github.io is this paper's project page; I did not fetch it.
- **SAE steering, arXiv:2603.19183** (Swann, McGranahan, Buurmeijer, Kennedy III, Schwager; v1 19 Mar 2026, v2 6 Jun 2026), "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models". SAE features for motion primitives and semantic concepts are causally steerable on LIBERO and DROID hardware. Amplifying general features gives matching behavior, and ablating them degrades performance. Steering can reach behaviors that prompting cannot. The abstract gives no numbers and no compositional evaluation. — [arXiv 2603.19183](https://arxiv.org/abs/2603.19183)
- **Related, [snippet only]:** "Not All Features Are Created Equal: A Mechanistic Study of VLA Models" — [arXiv 2603.19233](https://arxiv.org/pdf/2603.19233). "Observing and Controlling Features in VLA Models" (Buurmeijer, Amo Alonso, Swann, Pavone), which introduces feature observability and controllability — [arXiv 2603.05487](https://arxiv.org/html/2603.05487). I did not fetch either abstract.
- **Other flow-sampler guidance, [snippet only]:**
  - ACG (Action Coherence Guidance) for flow VLAs uses language-only CFG as the standard recipe — [arXiv 2510.22201](https://arxiv.org/pdf/2510.22201).
  - A FLOWER-based study trains with language dropout and then applies CFG, v_cfg = v(∅) + s[v(L) − v(∅)] — [arXiv 2606.06461](https://arxiv.org/pdf/2606.06461).
  - OmniGuide uses an energy-based guidance field from external models/VLMs — [arXiv 2603.10052](https://arxiv.org/html/2603.10052).
  - AFIL uses negative guidance from a failure-trajectory generator — [arXiv 2605.08434](https://arxiv.org/pdf/2605.08434).
- **Guided Action Flow, arXiv:2607.02092** (Yang et al.; v1 2 Jul 2026, v4 6 Oct 2026). A learned critic's action gradient steers reverse-time flow sampling of a frozen 0.45B VLA, with a 2.735M-parameter critic. Real-robot success goes from 60.0% to 82.5% on six nominal tasks, and from 34.2% to 49.2% under altered lighting and distractors. This is the closest analogue to C4's "x0-projection guidance in the flow sampler", and it is positive, but on nominal and perturbation shifts, not recombination. — [arXiv 2607.02092](https://arxiv.org/abs/2607.02092)
- **RL²-VLA, arXiv:2607.26991** (Tan et al.; 29-30 Jul 2026). An offline RL policy on action-expert latents composes its flow velocity with the frozen VLA. The steering switches on only when failure is predicted, and it gives up to +17.3% out-of-domain on SIMPLER and PolaRiS. The authors note that always-on steering perturbs accurate actions. — [arXiv 2607.26991](https://arxiv.org/abs/2607.26991)
- **Attention steering in a driving VLA, arXiv:2608.17095** (Nagendra Prasad, Ullrich, Graichen; 17 Aug 2026; ECCV 2026 per the listing). A bounded attention bias on detected-actor visual tokens in Alpamayo-R1 gives a dose-dependent trajectory change: mean about 17 cm; about 2.0 cm when only the first 8 layers are hooked versus about 67.6 cm for all 36. The reasoning text never changes, which the authors call "verified exposure, not robustness". This is relevant to C4's "early attention steering fails": it independently finds that early-layer attention bias has little action effect. Its domain is driving, not manipulation. — [arXiv 2608.17095](https://arxiv.org/abs/2608.17095)

### Inferences
- No found paper tests a long list of internal handles on a held-out-at-the-weights split. C4's breadth (nine handles) and its testbed (own pi0.5 trained on a LIBERO-90 split) look novel.
- Existing positive results for IGAR, TAG, SAE and FFN steering come from different failure types: contradictory instructions, clutter, behavior modulation. They therefore do not directly contradict C4. A reviewer will still cite them, so C4 should run IGAR-style and TAG-style variants (we have analogues: early attention steering and contrastive region guidance) and name them as such.
- CAG is the one handle with prior recombination numbers. ReGuide's +2.4-point Comp. Avg. on public pi0.5 is consistent with our CAG failure. CRAFT's 66.5% for CAG-VA on Pick-Place pi0.5 is a partial counterexample: in a 2-attribute colour-matching split, CFG-type guidance recovered much of the gap for pi0.5. C4 must explain this. Possible explanations are a different split, the trained VA branch, or per-benchmark tuning of the strength.
- ReGuide's introduction argues that steering methods which amplify the policy's own language-dependent distribution "cannot carry the end-effector to a referent it never considered". This is a prior qualitative statement of C4's thesis, so C4 is the systematic empirical version of an argument ReGuide has already made.

### Gaps
- I could not confirm TAG's backbones or whether its LIBERO-Plus results include recombinations; this needs the full text.
- I could not confirm whether IGAR reports any held-out-combination results, since only the abstract was read.
- I did not fetch vla-mech-interp.github.io.

## Q2. Which test-time sample-selection or verifier methods exist for VLAs, and do any report failure on spatial or region binding?

### Takeaway
Verifier and reranking methods (V-GPS, RoboMonkey, CoVer, VERITAS, frozen-feature probes) all report positive gains, mostly on OOD perturbations, SIMPLER, or nominal tasks. I found no verifier paper that reports results on held-out object x goal recombinations or that reports failure on region binding. The probe study (2605.28527) says its gains are "not universal", which is the nearest negative statement.

### Cited Findings
- **V-GPS, arXiv:2410.13816** (Nakamoto, Mees, Kumar, Levine; CoRL 2024). An offline-RL value function reranks the actions of generalist policies without needing weights. It gives consistent gains for five policies across 12 tasks on multiple platforms. — [arXiv 2410.13816](https://arxiv.org/abs/2410.13816)
- **RoboMonkey, arXiv:2506.17811** (Kwok, Agia, Sinha, Foutter, Li, Stoica, Mirhoseini, Pavone; v1 21 Jun 2025). It samples actions, applies Gaussian perturbation and majority voting, and selects with a VLM verifier. Action error follows an "exponentiated power law" in the number of samples. Gains: +25% absolute OOD, +9% in-distribution, and +7% when fine-tuning both the VLA and the verifier for new setups. The arXiv page lists no venue. — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811)
- **CoVer / CoVer-VLA, arXiv:2602.12281** (Kwok, Zhang, Xu, Liu, Mirhoseini, Finn, Pavone; Feb 2026). A contrastive verifier is combined with VLM instruction rephrasings and selects over instructions and action chunks. Compared with scaling pre-training on the same data: +22% in-distribution and +13% OOD on SIMPLER, +45% in the real world, and +14% task progress and +9% success on PolaRiS. — [arXiv 2602.12281](https://arxiv.org/abs/2602.12281)
- **VERITAS, arXiv:2606.18247** (Mingtong Zhang, Dhruv Shah; 16 Jun 2026), "Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement". A gradient-free visual verifier scores candidates from a pre-trained policy, and the verified rollouts are then used for offline fine-tuning, which is comparable to expert demonstrations. The abstract gives no numbers. — [arXiv 2606.18247](https://arxiv.org/abs/2606.18247)
- **Frozen-VLA value probes, arXiv:2605.28527** (J. Zhang et al., PKU/CAU; 27 May 2026). Linear probes on frozen OpenVLA, Pi0.5, DINOv2 and CLIP features predict Monte-Carlo success on LIBERO-Goal. Pi0.5 probes reach about 92% pairwise ordering accuracy. Used as a test-time selector over Pi0.5 action prefixes, success on push-plate rises from 26.7% to 44.3%, with a second positive case on wine-rack. The authors say the gains are not universal and cost extra compute. This is the nearest prior to C4's "first-chunk learned verifiers". It is positive on in-distribution LIBERO-Goal and not tested on recombinations. — [arXiv 2605.28527](https://arxiv.org/abs/2605.28527)
- ReGuide cites RoboMonkey and IGAR ("attention recalibration") only in related work and runs no verifier baseline. — [arXiv 2609.38616 HTML](https://arxiv.org/html/2609.38616)

### Inferences
- A verifier can only select among sampled candidates. If the policy never samples the correct referent, selection cannot recover it, which matches ReGuide's "referent it never considered" argument. C4's first-chunk verifier failure therefore fits a coverage explanation. I found no paper that measures this coverage limit on binding errors, so it is a novel measurable point for C4: report the oracle best-of-N on our split.

### Gaps
- I did not check the full texts of RoboMonkey, CoVer and VERITAS for per-category failure analysis (e.g. wrong-object versus wrong-location errors).

## Q3. Which benchmarks or splits hold out object x goal or object x scene combinations from VLA training, versus testing public checkpoints on recombinations?

### Takeaway
The only verified split that is held out at the weights, uses LIBERO-derived tasks, and trains pi0.5 itself is CRAFT (arXiv:2610.00524, 30 Sep 2026). CRAFT fine-tunes pi0, pi0.5 and GR00T N1.7 only on the demonstrated combinations of the Pick-Place and Pick-Place-Press LIBERO adaptations. Its Pick-Place split (4 demonstrated and 12 held-out cube x plate colour combinations) is a direct precedent for C4's testbed design. ReGuide (2609.38616) is the main recombination benchmark on LIBERO with external grounding. It uses PUBLIC openpi LIBERO checkpoints trained on the four LIBERO suites and does not state that it excluded its compositional cells. Its "unseen objects" come from LIBERO-90. Our LIBERO-90 split with our own pi0.5 is distinct from both, but CRAFT already establishes the "train pi0.5 from base on demonstrated combinations only" design.

### Cited Findings
- **CRAFT, arXiv:2610.00524** (Yang, Na, Cho, Yoon; 30 Sep 2026), "Same Scene, Different Task: Skill Alignment for Compositional Generalization in VLAs".
  - Split design: Pick-Place has 16 cube x plate colour combinations; 4 same-colour ones are demonstrated (50 demos each) and 12 are held out. Pick-Place-Press has 8 combinations; 2 are demonstrated and 6 held out. Both are adapted from LIBERO.
  - Fine-tuning uses only the demonstrated combinations. The starting checkpoint is not clearly stated.
  - A motivating study builds 5 object-target combinations on LIBERO-Goal.
  - Results (demonstrated / undemonstrated):

    | Benchmark | Backbone | CRAFT | FT Full |
    |---|---|---|---|
    | Pick-Place | pi0.5 | 99.0 / 84.2 | 100.0 / 8.4 |
    | Pick-Place | pi0 | 94.0 / 64.3 | 100.0 / 2.8 |
    | Pick-Place | GR00T | 99.0 / 72.1 | 98.8 / 1.1 |
    | Pick-Place-Press | pi0.5 | 99.7 / 62.6 | 100.0 / 0.7 |

  - Real robot (pi0.5): CRAFT 43/60 undemonstrated versus 9/60 for FT Full.
  - Baselines include CAG-VA and "Skill Composition". The latter scores 25.7, 12.7 and 13.1 undemonstrated on Pick-Place.
  - The full text was read only to character 100k. — [arXiv 2610.00524 HTML](https://arxiv.org/html/2610.00524)
- **ReGuide, arXiv:2609.38616** (Y. Zhang, Liu, Li, Song, Hariri, Ganguly, Yang, Ye, Grant, Chaudhary, Yin; 29 Sep 2026).
  - Benchmark: LIBERO with 4 factor settings (Target Region, Target Object, Unseen Object, Task Scene) x 4 tasks, giving 16 cells with 50 paired initial states each, plus 10 original tasks: 1,300 trials per method.
  - Checkpoints: official openpi LIBERO checkpoints for pi0 and pi0.5, official ones for OpenVLA-OFT and X-VLA, and an author-fine-tuned GR00T N1.7 on the four suites (1,693 demos, 80k steps).
  - Bare Comp. Avg.: OpenVLA-OFT 14.0, pi0 8.5, pi0.5 39.0, X-VLA 20.6, GR00T N1.7 13.0.
  - With ReGuide: OpenVLA-OFT 70.8, pi0.5 94.5 (756/800; 87.5% with perception grounding), GR00T 69.1. That is +55.5 to +56.8 points, with Orig. changing by at most 0.8.
  - Real robot (pi0.5 fine-tuned on 250 scripted demos): composition success from 15.0% to 90.0%.
  - Its limitations say the main simulation results use oracle grounding, and that compositions which change several factors at once are not evaluated. — [arXiv 2609.38616](https://arxiv.org/abs/2609.38616); [HTML](https://arxiv.org/html/2609.38616)
- **"Robust Skills, Brittle Grounding", arXiv:2602.24143** (Emukpere, Deffayet, Renders; 27 Feb 2026). A controlled multi-object picking study with held-out object-location pairings on SmolVLA and pi0.5. Primitive execution stays more reliable than instruction-conditioned success, so skill appears decoupled from instruction following. It proposes task ladders and decomposed metrics, and tests no test-time fix. The training protocol is not stated in the abstract. This is a diagnostic precedent for "binding fails while motor skill is intact". — [arXiv 2602.24143](https://arxiv.org/abs/2602.24143)
- **ACG-Bench, arXiv:2610.06184** (Zaibin Zhang et al.; 5 Oct 2026). Dual-arm arm-wise compositions: 23 task-condition pairs, 6 in-domain and 17 unseen, on a pi0.5 backbone. Simulation generalization success: AE-VLA 21.53%, Single pi0.5 2.94%, MA-VLA 3.06%, two independent pi0.5 5.53%. Real SO101: 39.00% versus 10.00%. These are held-out compositions; the training-split details were not confirmed. — [arXiv 2610.06184](https://arxiv.org/abs/2610.06184)
- **VLA-GAUGE, arXiv:2610.16567** (Xiao et al.; 12 Oct 2026). 12 VLAs, paired in-distribution and OOD suites, and combinatorial perturbation screening. Mean zero-shot score 22.3% and mean R-Score 9.3. Diagnosis-driven fine-tuning gives pi0.5 +72.4 zero-shot and +31.8 robustness, versus +10.9 for LIBERO-Plus fine-tuning. The split construction was not confirmed. — [arXiv 2610.16567](https://arxiv.org/abs/2610.16567). [snippet only] A search snippet said compositional capability averages 2.8% OOD with 9 of 12 models at zero; this is not in the abstract and is unverified.
- **AGNOSTOS, arXiv:2505.15660** (Jiaming Zhou et al.; v3 19 Oct 2025). 23 unseen manipulation tasks at two difficulty levels for zero-shot cross-task tests. The abstract does not say whether models were trained excluding these tasks. Its method X-ICM uses in-context LLM demos. These are unseen tasks rather than controlled recombinations. — [arXiv 2505.15660](https://arxiv.org/abs/2505.15660)
- **GemBench, arXiv:2410.01345** (Garcia, Chen, Schmid; ICRA 2025). 7 action primitives and 4 generalization levels (novel placements, rigid objects, articulated objects, long-horizon). 3D-LOTUS is weak on novel tasks and 3D-LOTUS++ fixes it with LLM planning plus VLM grounding, i.e. external grounding helps. The level definitions were not confirmed from the abstract. — [arXiv 2410.01345](https://arxiv.org/abs/2410.01345)
- **[snippet only]:** Colosseum V2 — [arXiv 2605.27759](https://arxiv.org/pdf/2605.27759); "Towards Generalizable Robotic Data Flywheel: High-Dimensional Factorization and Composition" — [arXiv 2603.25583](https://arxiv.org/pdf/2603.25583). Not checked.

### Inferences
- The testbed design, own pi0.5 trained only on demonstrated combinations and evaluated on held-out ones, is NOT novel per se: CRAFT does exactly this for pi0, pi0.5 and GR00T on LIBERO-derived splits. What remains distinct for C4:
  - the split is built on LIBERO-90, over object x goal x scene, rather than colour cube x plate;
  - C4 tests internal handles, whereas CRAFT changes training;
  - C4 runs external grounding (ReGuide) on the held-out-at-the-weights policy.
- ReGuide's results are on public checkpoints whose training may contain some cells' elements (the cells "recombine elements of training tasks"). Its "binding is recoverable externally" result was therefore never shown on a policy where the combination is unseen at the weights. If C4 shows ReGuide fixes errors on OUR held-out split, that adds a result.
- Prior work agrees with the C4 premise that motor skill survives while target binding fails: ReGuide's "fail global grounding, keep local manipulation skills" and Emukpere et al.'s decoupling finding.

### Gaps
- RoboCasa365, CompoSuite (I recall Mendez et al., ICLR 2022, arXiv 2207.04136 [UNVERIFIED], a held-out robot x object x obstacle x objective RL benchmark, not VLA), "LIBERO-OOD" and the Li et al. OOD suites were not verified because the tool-call budget ran out.
- LIBERO-Plus and LIBERO-PRO, which perturb public checkpoints, were also not verified.
- CRAFT's starting checkpoint (pi0.5 base versus LIBERO-fine-tuned) is not stated in the part I read.

## Q4. Is there a systematic negative result, "internal handles cannot fix VLA binding errors", anywhere?

### Takeaway
No. I found no paper that systematically tests several policy-internal handles and reports that they fail on VLA binding or recombination errors. The closest pieces are listed below. A systematic negative result like C4 appears unclaimed as of 2026-10-26, within the limits of the searches run.

### Cited Findings
- ReGuide states qualitatively that steering methods amplifying the language-dependent part of the policy distribution "cannot carry the end-effector to a referent it never considered". It also says language-side steering uses hand-set scales and is always on. Its one measured internal handle (CAG) gives +2.4 Comp. Avg. at an Orig. cost of 12.6 points. It reports no ablation of attention, activation steering or verifiers. — [arXiv 2609.38616 HTML](https://arxiv.org/html/2609.38616)
- CRAFT finds that CAG-VA gains "vary substantially across models and benchmarks", ranging from 1.1% to 66.5% undemonstrated. — [arXiv 2610.00524 HTML](https://arxiv.org/html/2610.00524)
- RL²-VLA notes that always-on test-time steering perturbs accurate actions and that sampled candidates share correlated failure modes. — [arXiv 2607.26991](https://arxiv.org/abs/2607.26991)
- The 2605.28527 probe-based selector's gains are "not universal". — [arXiv 2605.28527](https://arxiv.org/abs/2605.28527)
- Early-layer attention steering has little action effect in a driving VLA (about 2.0 cm versus about 67.6 cm for all layers). — [arXiv 2608.17095](https://arxiv.org/abs/2608.17095)
- "Legible Failures: Detecting and Repairing In-Context Binding Errors" is about LLMs, not VLAs. It discusses negative steering results by Roy et al. and Basu et al. on binding — [arXiv 2609.11216](https://arxiv.org/pdf/2609.11216) [snippet only; not fetched].

### Inferences
- C4's novelty lies in being systematic: nine internal handles, including an oracle-retrieval upper bound that closes 0.21 of the gap, measured on a held-out-at-the-weights split, plus a positive external control. The pieces exist only as scattered single-handle observations (ReGuide's CAG row, CRAFT's variable CAG-VA, and the coverage argument).
- The main risks to the claim:
  - CRAFT's CAG-VA at 66.5% on pi0.5 Pick-Place is a counterexample in a simpler split.
  - IGAR and TAG are positive internal-attention and contrastive-guidance methods that reviewers will ask us to reproduce faithfully on our split. Our "early attention steering" and "contrastive region guidance" must be close variants, or the named methods must be run.

### Gaps
- The search was limited to about 15 tool calls. Workshop papers and GitHub READMEs, which may hold an unpublished negative result, were not searched exhaustively.
- I found no OpenReview (ICLR 2027 submissions) check.

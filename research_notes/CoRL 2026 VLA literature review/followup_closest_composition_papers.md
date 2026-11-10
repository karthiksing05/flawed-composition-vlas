# Closest 2026 papers on VLA compositional failure (full-text follow-up)

Reading depth: each arXiv HTML page was fetched through a summarizing fetch tool, not read raw. Per-paper coverage is stated in "Gaps". Everything below is as reported by that tool; numbers are not independently re-verified. Fetch date 2026-11-10.

## Q1. Does any paper show that the number of training partners of an object or target predicts success on new pairings?

### Takeaway
Two papers vary coverage directly. Wang et al. (2607.29687) shows OOD success rising with the number of seen instruction pairs per held-out task. Qi et al. (2607.21582) shows that where the diagonal and extra columns are placed matters more than raw budget. None reports "object seen with 1 vs 2 targets" on pi0.5 or GR00T with frozen backbones at the reader's granularity. CRAFT (the closest paper on setup) has exactly one partner per object and does not vary it.

### Cited Findings
- Wang et al. (Berkeley, arXiv 2607.29687v2, 15 Oct 2026, no venue): "OOD success rises with the number of seen instruction pairs per held-out task (Figure 3)"; saturation around B=8 (PP) and B=16 (PPP, Fig 2). Flow-matching OOD SR at B=4/8/16 = 0.83% / 23.93% / 82.29%, ID loss flat (~0.0065). Orthogonal 16-task set gets 78.2% OOD vs 78.6% for full-task training at a quarter of the tasks (Sec 5.2, Fig 4). Own transformer flow policy with frozen DINOv2, robomimic, not a pretrained VLA. — [arXiv HTML](https://arxiv.org/html/2607.29687)
- Wang et al. 2S-PP: broader (o1,c2) pair coverage (Cov12/9/6/4 on 36 tasks; Cov16/12/9/6 on 48 tasks) improves OOD success and reduces violations (Figs 8, 15). Causes named: "sparse coverage causes instruction-steering failure, not missing skills"; spurious correlations from insufficient pair coverage. — [arXiv HTML](https://arxiv.org/html/2607.29687)
- Wang et al.: 0.4% -> 54.7% OOD came from pretraining on a 4-task diagonal set ((0,0,0),(1,1,1),(2,2,2),(3,3,3)) then finetuning with ONE demonstration per task of the full set (Sec 5.3, Fig 6); ID success dropped after finetuning. — [arXiv HTML](https://arxiv.org/html/2607.29687)
- Qi et al. (NVIDIA/Northeastern, arXiv 2607.21582v1, 23 Jul 2026, no venue): ManiSkill, 5 factors (verb, color, object, size, spatial), 6,480-instruction space. Training = diagonal of each factor-pair grid (each value paired with exactly one value of the other factor); off-diagonal cells held out. Strategy "V" adds the leftmost column (fixes the under-grounded factor, varies the dominant one). Real robot GR00T-N1.7 f=12 (Table 3): L 22.9/37.5/35.4/37.5, Random 37.5/47.9/50.0/47.9, V 54.2/64.6/62.5/66.7 (columns Bun100, Bun200, Pizza100, Cup100). Sim: V at n=400 beats baselines at n=800 (All Factor). — [arXiv HTML](https://arxiv.org/html/2607.21582)
- Qi et al. factor dominance: consensus hierarchy color >= object >= spatial >= verb >= size (Table 2); (color, object) FDR 57.3% pi0.5, 42.1% pi0 (Table 1). Judge Gemini-2.5-Flash agreed with humans on 88.7% of 80 rollouts (Table 5). No dedicated ablation on number of training combinations. — [arXiv HTML](https://arxiv.org/html/2607.21582)
- CRAFT (Yang, Na, Cho, Yoon, KAIST, arXiv 2610.00524v1, 30 Sep 2026, no venue): Pick-Place 4 colors per operation, 16 combinations, only the 4 same-color ones demonstrated (50 demos each), so every cube color has exactly ONE plate partner; 12 held out. Pick-Place-Press: 2 of 8 demonstrated. Appendix D varies query count/allocation only, not number of training combinations. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- Robust Skills, Brittle Grounding (Emukpere, Deffayet, Renders, Naver Labs, arXiv 2602.24143v1, 27 Feb 2026): for each object 3 of 5 regions in training, 2 held out; SmolVLA small jitter Table 3: in-distribution success 44, held-out 0 (grasp-any 57 vs 31, reach 100 vs 2). Data scaling 10k->100k gives no gain (SmolVLA full-random); no data-diversity ablation. Task is grasp only, not pick-place. — [arXiv HTML](https://arxiv.org/html/2602.24143)

### Inferences
- Nobody found measures "object had 1 vs 2 targets" on frozen pi0.5/GR00T; the nearest is Wang's seen-pair-count curve (small custom policy) and Qi's allocation study. The reader's finding remains novel in the VLA setting, with these as supporting prior art (verify the exact Fig 3 axes before citing).
- CRAFT's setup is the 1-partner extreme of the reader's axis; the reader's 1-vs-2 comparison would extend it.

### Gaps
- Wang Fig 3 values were not given numerically by the fetch tool. Qi appendix tail (last ~11k chars) unread.

## Q2. Does any paper hide/mask the non-active object's identity, or route by phase?

### Takeaway
No paper abstracts the non-active entities to silhouettes. AC-VLA masks wrist-view tokens during place phases (phase-gated, not object-level). ReGuide and Wang's stage-wise policies are phase-aware; Wang explicitly says stage-wise policies that cannot see inactive instructions hit an information bottleneck.

### Cited Findings
- AC-VLA (Peng, Peng, Lu, Lian, Yu, Wang; arXiv 2607.15714v1, 17 Jul 2026): during training only, wrist-view tokens masked in attention during "place" segments (found via LLM instruction decomposition + proprioceptive cut at gripper-state changes); wrist retained during approach/grasp. Not object-level masking. Abstract says "closed-gripper phases", Sec 3.3 says "place" (inconsistent). — [arXiv HTML](https://arxiv.org/html/2607.15714)
- ReGuide (Zhang et al., arXiv 2609.38616v1, 29 Sep 2026): phase-dependent criteria (approach/carry vs place), but candidate referents include all movable entities; no masking. — [arXiv HTML](https://arxiv.org/html/2609.38616v1)
- Wang et al.: ground-truth stage indicator / stage-wise policy does well on independent-subtask PPP but fails on 2S-PP where the first placement depends on a later instruction; stage-wise policies "cannot see inactive instructions". Omitting container instruction in PP: 71.05 -> 49.95. — [arXiv HTML](https://arxiv.org/html/2607.29687)
- CRAFT: L_skill/L_cf operate on skill queries with labels from a VLM segmenting demos at operation transitions; no entity hiding. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- Grant et al. (2603.19233v6): injected activations carry motor program bound to absolute workspace coordinates; see Q6. — [arXiv HTML](https://arxiv.org/html/2603.19233)

### Inferences
- The reader's "silhouette every non-active entity + prompt switch on grasp event" is not anticipated by any paper read; closest are AC-VLA (phase gating of wrist view) and Wang's stage-wise caveat (which warns that hiding the other instruction hurts when later constraints depend on it; the reader's tasks do not seem to have this dependency).

### Gaps
- ReGuide appendix B (second 70k chars) fetch failed; full text of AC-VLA ablation not extracted beyond Table 3 summary.

## Q3. Which report results on LIBERO-Goal-derived held-out pairings comparable to the reader's suite?

### Takeaway
ACT-VLA and AC-VLA use LIBERO-Goal-OOD (re-pairing objects with targets from other tasks), same pi0.5 baseline (46.6 Goal-OOD). ReGuide uses a LIBERO-derived 16-cell grid. CRAFT's Appendix A uses LIBERO-Goal for a motivating test. None gives held-out lists or partner counts in the text read.

### Cited Findings
- ACT-VLA (Peng, Lu, Peng; arXiv 2607.00351v1, 1 Jul 2026): LIBERO-Spatial-OOD / Goal-OOD pair the object from one task with the target of another, following protocol of ref [12] (text-latent interpolation prior work); specific held-out combos and partner counts not specified. 3 seeds x 50 episodes, 30k steps. Table I pi0.5: Spatial-OOD 35.5 -> 88.2, Goal-OOD 46.6 -> 95.6, six-suite AVG 78.6 -> 95.6. Baselines Spatial/Goal OOD: UniVLA 11.0/32.0, OpenVLA-OFT 0.0/1.0, GR00T-N1 17.9/24.1. Table II pi0: 2.0/16.0 original-only, 81.0/85.0 inference-time TLI, 85.9/90.5 synthetic training. Method: offline augmentation; roll out text-latent interpolation between two task latents, keep successful rollouts (~30-40 per task, ~300-400 per suite), retrain with plain BC. In-dist values in Table I identical to baseline row (unexplained). Caveat: rollouts come from the model itself via latent interpolation (relies on backbone). — [arXiv HTML](https://arxiv.org/html/2607.00351v1)
- AC-VLA Table 2: LIBERO-Spatial-OOD and Goal-OOD each 20 novel compositional tasks; pi0.5 baseline 35.5 / 46.6 -> AC-VLA 64.2 / 73.3, AVG 78.6 -> 87.3; GR00T-N1 version 36.4 / 44.0 / 77.3. Table 4 real world: pi0.5 OOD 35.0 -> 82.5 (ID 93.7 -> 88.7). Held-out list not given. Code "will be made publicly available" at https://ac-vla.github.io/ (no release confirmed). — [arXiv HTML](https://arxiv.org/html/2607.15714)
- ReGuide: LIBERO, 4 axes (Target Region, Target Object, Unseen Object, Task Scene) x 4 settings x 4 tasks = 16 cells, 50 shared initial states, 1300 paired trials/method. Table 1: bare pi0.5 composition average 39.0% -> ReGuide 94.5%; OpenVLA-OFT / GR00T N1.7 ReGuide 70.8 / 69.1; gain 55.5-56.8 pts, Orig. changes <= 0.8. Real robot (Table 2): 15.0 -> 90.0 compositional, 86.0 -> 92.0 original. Table 3 naive transport 68.5; perception-based grounding 87.5 (7.0 below oracle). Training-free wrapper (needs object poses from a grounding module, demo statistics, candidate referent monitoring, bounded transport to demonstrated pre-contact set, hand back to frozen policy). Cause: wrong referent binding via habitual/visual shortcuts. Splits/partner counts not extracted. — [arXiv HTML](https://arxiv.org/html/2609.38616v1)
- CRAFT Appendix A (LIBERO-Goal): 6 pick-place tasks as demos, 5 hand-chosen undemonstrated combos; released checkpoints Table A.3 mean undemonstrated: pi0 2.9, pi0.5 50.9, pi0-FAST 19.5, GR00T N1.7 2.0 (demonstrated 88.8-97.4). — [arXiv HTML](https://arxiv.org/html/2610.00524)
- LIBERO-CF (Fang et al., UNC, arXiv 2602.17659v3, stamped 04 Nov 2026, no venue stated): counterfactual instructions on LIBERO scenes; suites CF-Spatial 15, CF-Object 10, CF-Long 10, CF-OOD 15 tasks. Table 3 averages (Faithful/Biased %): pi0.5 success 13.2/60.9 baseline, TF-CAG 16.8/43.5, VA-CAG 21.7/36.2; pi0 9.6/45.0 -> 10.4/25.8 (VA); OpenVLA-OFT 0.4/78.6 -> 2.1/41.0. Grounding pi0.5 30.8/65.6 -> 46.3/52.2. Table 1: vision-only vs vision+language LIBERO: OFT 97.1 vs 69.1, pi0 94.2 vs 18.5, pi0.5 96.9 vs 51.7. Method CAG: a_CAG = a_uncond + w(a_cond - a_uncond), w=1.5 (pi0, pi0.5, X-VLA), 3.0 (OFT); TF = same VLA without language; VA = separately finetuned vision-action model. Table 7 X-VLA success 13.8/39.3 -> 19.5/42.6 (TF). Causes: dataset bias (scene-specific demos cover small subset of feasible tasks), modality imbalance, training-task objects as visual attractors. Project https://vla-cf.github.io/; no code link found. — [arXiv HTML](https://arxiv.org/html/2602.17659)

### Inferences
- The reader's suite is directly comparable in spirit with ACT-VLA/AC-VLA Goal-OOD (same pi0.5 baseline 46.6) only if the held-out re-pairing list is matched; it is not stated in the text read, so check the project page/appendix or released data.

### Gaps
- LIBERO-CF baseline pi0.5 grounding vs success per-suite breakdown not extracted. Held-out lists for ACT-VLA, AC-VLA, ReGuide not found.

## Q4. Failure-mode analyses (completing a demonstrated task sharing object or target)

### Takeaway
CRAFT Table A.4 is the closest: failures overwhelmingly complete a demonstrated task sharing the object or target; MESA reports a language-following rate on distractor completions.

### Cited Findings
- CRAFT Table A.4 (LIBERO-Goal, % of rollouts; instructed / shares object / shares target / other / none): pi0 2.9/9.6/48.8/0.5/38.1; pi0.5 50.9/38.4/2.8/0.0/7.9; pi0-FAST 19.5/41.9/4.1/0.8/33.7; GR00T N1.7 2.0/39.3/25.5/1.9/31.3. pi0.5 with empty instruction still performs demonstrated behaviors. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- CRAFT cause: "vision shortcut" (observation used as stand-in for instruction); authors say evidence is behavioral and does not exclude other factors. High demo success does not imply recombination. Adaptation method (Full/LoRA/Frozen) alone does not close the gap. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- CRAFT Table 1 (success %, demonstrated / undemonstrated), Pick-Place pi0.5: FT Full 100.0/8.4; LoRA 100.0/46.3; Frozen 99.0/21.1; CAG-VA 96.3/66.5; Skill Composition 100.0/25.7; CRAFT 99.0/84.2. pi0: Full 100/2.8, CRAFT 94.0/64.3 (LoRA 1.0, Frozen 3.0 undemo). GR00T N1.7: Full 98.8/1.1, CRAFT 99.0/72.1. Pick-Place-Press pi0.5: Full 100/0.7, CAG-VA 92.7/21.8, Skill Comp 99.7/5.8, CRAFT 99.7/62.6; pi0 CRAFT 40.7; GR00T 31.9; LoRA 15.4 (Table D.1). Real Piper arm (Table 3): 19/20 demo both; undemo FT 9/60 vs CRAFT 43/60. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- Why LoRA (46.3) > Full (8.4)? The paper reports it but, in the text read, gives no explanation; plausible reading is only that full FT overfits shortcuts more (inference, not stated). Ablation Table 2 (Pick-Place): FM loss alone ~1.9% (pi0.5) / 3.8% (GR00T) undemo; L_skill alone no clear gain; L_preserve alone none; L_change alone 14.7% / 12.1%; full 84.2 / 72.1. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- "Skill alignment" = CRAFT trains the skill representation of a counterfactual instruction to reflect the currently required skill, using demonstrated executions of that skill as supervision. Method: learnable skill queries (attend image+text -> z_skill) and state queries (image only -> z_state); action expert conditioned on both; contrastive skill loss via flow-matching velocity predictions with state fixed and skill swapped (positive = other execution of same skill, 3 negatives = other entities of same operation, tau 0.1); counterfactual L2 velocity loss (change / preserve cases); L_total = L_fm + 0.03 L_skill + 1.0 L_cf, ramped over 3k steps; 20k steps, LoRA rank 16, 64 queries (pi0.5 48 skill/16 state; GR00T 32/32); pi-family uses layer-wise K/V at query positions, GR00T final hidden states; skill labels by a VLM segmenting demos at operation transitions, training only. Limitations: fixed operation order; all constituent skills must be demonstrated. Code/benchmarks/data "will be released"; project site taegeunyang.github.io/craft/; no repo confirmed. — [arXiv HTML](https://arxiv.org/html/2610.00524)
- MESA / BiMESA (Wilcox, Chang, Nguyen, Chakraborty, Collins, Saxena, Joffe, Karamcheti, Garg; Georgia Tech; CoRL 2026): MESA-Composite = unseen composite tasks of in-distribution subtasks; data via "subtask stitching". Success % MESA-Composite / BiMESA-Composite: DP 0.3/5.7, PG-Bin -/14.4, PG-FM 17.7/17.9, GR00T-N1.6 15.0/22.1, pi0 27.5/18.4, FAST 28.9/28.8, pi0.5 46.7/37.5. Language-following rate: DP 6.4/14.5, PG-Bin -/65.1, PG-FM 65.1/66.7, GR00T 56.2/71.2, pi0 65.5/42.3, FAST 66.3/66.7, pi0.5 81.5/80.1 (follow = a distractor predicate completes after failure). MESA-70: 7,000 demos/70 tasks; BiMESA-57: 6,833 demos/57 tasks. The project page gives no seen-composite split and no partner counts. Code https://github.com/pairlab/mesa-env; weights/data https://huggingface.co/collections/albertwilcox/mesa; arXiv listed as 2611.05567 (not fetched; unverified); OpenReview https://openreview.net/forum?id=Br2rXixvyN. — [project page](https://www.pair.toronto.edu/MESA/)
- Brittle Grounding cause: failures are in instruction-conditioned target selection, not control or data scale; policies rely on region-level positional cues. — [arXiv HTML](https://arxiv.org/html/2602.24143)

### Inferences
- CRAFT A.4 differs by model: pi0 mostly finishes a task sharing the TARGET (48.8%), pi0.5/pi0-FAST/GR00T the OBJECT (38-42%) -- this is on the reader's axis but pairs it with fine-tuned-from-release evaluation on LIBERO-Goal, not a frozen model on grasp-correct-then-place-wrong decomposition. The reader's 95-99% grasp-correct then place-fail split is not reported by any paper read.

### Gaps
- MESA full paper (arXiv/OpenReview) not read; seen-vs-unseen composite numbers absent from the page. CRAFT Table 1 caption details of "Skill Composition" baseline not read.

## Q5. What do the interpretability papers say about where object, target and skill live?

### Takeaway
Neither paper separates object, target and skill. Grant et al. find injected pi0.5 activations transfer a motor program bound to workspace coordinates; Singh et al. show primitive identity dominates object identity only when trained with primitive prompts. Singh's results carry a data-integrity caveat.

### Cited Findings
- Grant, Zhao, Wang (CWRU, arXiv 2603.19233v6, 02 Nov 2026, preprint; code github.com/CWRU-AISM/action-atlas; project cwru-aism.github.io/vla-interp-page/): injecting source-episode activations at pi0.5 action-expert layers 16-17 across 50 action-token positions yields source-task motion in 99.6% of cross-task episodes (n=1,968), destination success 2.6% (Table B). VLM pathway selects goal, expert generates coordinate-bound motor program; PaliGemma goal classification 76.4% vs expert 62.6%. Language probed >99% for prompt category but mainly used when scene ambiguous. Table A pi0.5 LIBERO baseline Goal/Object/Spatial/Long 93.2/87.6/92.8/86.0; null prompt 6.0/22.0/62.0/56.6 (Goal near 0 without language; Spatial 62). Section heading says object identity does not predict reach target, body text not read. — [arXiv HTML](https://arxiv.org/html/2603.19233)
- Singh, Happi, Relan, Nair, Baradwaj (arXiv 2605.30695v1, 29 May 2026, no venue; code "upon publication"): LoRA OpenVLA and pi0.5 on REASSEMBLE and LIBERO-Long; flat vs primitive-segmented episodes with primitive prompts. Few-shot m=3: pi0.5 primitive 0.66/0.74 vs flat 0.31/0.45 (REASSEMBLE/LIBERO-Long); OpenVLA 0.62/0.71 vs 0.34/0.42. Removing 4-D probe-defined primitive subspace at layer 24: 0.66->0.34 (pi0.5), 0.62->0.30 (OpenVLA); random subspace 0.64/0.60. Cosine: same-primitive cross-object 0.71 vs same-object cross-primitive 0.31 (primitive-trained), reversed in flat (0.43 vs 0.58). Figures 2 and 3 are captioned "illustrative pending verification". No target probing. — [arXiv HTML](https://arxiv.org/html/2605.30695)

### Inferences
- Singh's "prompts switched per primitive" is similar in spirit to the reader's grasp-event prompt switch but is training-time and LoRA-trained, not a frozen-model edit; treat numbers cautiously given the caveat.

### Gaps
- Grant paper: 160k of 260k chars unread, including the object-identity section.

## Q6. Existence of ReGuide, ACT-VLA; AC-VLA, PRTI/RoboCombine, LangForce

### Takeaway
All resolved. ReGuide ID 2609.38616 is correct (title "Correcting WHERE, Preserving HOW", first author Yanyan Zhang). ACT-VLA ID 2607.00351 correct, but first author is Kai Peng (with Jie Lu, Xiaojiang Peng), not just "Peng".

### Cited Findings
- AC-VLA method (two sentences): an LLM splits instructions into sub-tasks and a proprioceptive aligner cuts demos at gripper changes to build a sub-task dataset mixed with full demos; wrist tokens are masked in training during place segments. Numbers under Q3. — [arXiv HTML](https://arxiv.org/html/2607.15714)
- RoboCombine / PRTI: paper "Exploring the Limits of Combinatorial Generalization in Embodied Manipulation" (Zhang, Yang, Guo, Yin, Zhang; SJTU AutoLab; arXiv 2610.15425v1, 11 Oct 2026); code https://github.com/AutoLab-SAI-SJTU/RoboCombine. Train 50 atomic RoboTwin skills only; test 19 unseen two-stage compositions x 20 rollouts. PRTI post-trains atomic policies on a 1:1 mix of canonical and late-trajectory start states. Full-instruction training ordered success: seen Set A 40.0, partial Set B 15.0, disjoint Set C 4.4 (B+C avg 7.7). Oracle timed HSE: base 2.63 / random perturbation 4.21 / robot reset oracle 9.47 / PRTI 18.16 ordered success. Real: Full-Inst 0.16, Oracle-HSE 0.75, +PRTI 0.80 mean progress. — [arXiv HTML](https://arxiv.org/html/2610.15425)
- Conflict: NeurIPS 2026 poster (title "Exploring the Limits of Compositional Generalization in VLA Manipulation") abstract reportedly gives PRTI 2.11% -> 26.32% ordered success, 2.63% -> 60.53% transition conversion; arXiv v1 gives 18.16%. Differing versions/setups unresolved. — [NeurIPS poster](https://neurips.cc/virtual/2026/poster/139255) (via search snippet only; not fetched) vs [arXiv HTML](https://arxiv.org/html/2610.15425)
- LangForce (Lian et al., arXiv 2601.15197v5, 31 Jan 2026; ICML 2026 poster per search result https://icml.cc/virtual/2026/poster/65457; code https://github.com/ZGC-EmbodyAI/LangForce): vision-only prior vs vision+language posterior with learnable Latent Action Queries, trained to maximize log-likelihood ratio; only posterior branch used at inference. SimplerEnv WidowX avg 55.2 -> 66.5 (Carrot 50.0->63.8, Eggplant 54.2->79.2); LIBERO avg 96.5 -> 98.4 (Goal 97.4 -> 99.4); RoboCasa 47.8 -> 52.6. Simulation only; no held-out pairing test. — [arXiv HTML](https://arxiv.org/html/2601.15197v5)

### Inferences
- LangForce and LIBERO-CF's CAG are both prior/posterior (vision-only vs language) contrasts; neither hides entities.

### Gaps
- RoboCombine is skill-sequence recombination, not object-target pairing; relevance low. NeurIPS poster page not fetched.

## Runnable code summary (Key question 4)

### Takeaway
Confirmed links: MESA (github.com/pairlab/mesa-env), RoboCombine (github.com/AutoLab-SAI-SJTU/RoboCombine), LangForce (github.com/ZGC-EmbodyAI/LangForce), Grant et al. (github.com/CWRU-AISM/action-atlas). All others: promised or none seen.

### Cited Findings
- Promised, not confirmed: CRAFT ([HTML](https://arxiv.org/html/2610.00524)), AC-VLA ([HTML](https://arxiv.org/html/2607.15714)), Singh et al. ([HTML](https://arxiv.org/html/2605.30695)). No link found: ReGuide, ACT-VLA, LIBERO-CF (project page only), Wang (project page claims code/dataset), Qi, Emukpere. Links above were as reported by the fetch tool; repo contents not opened.

### Gaps
- Whether repos actually contain runnable code or checkpoints was not checked (no GitHub fetch).

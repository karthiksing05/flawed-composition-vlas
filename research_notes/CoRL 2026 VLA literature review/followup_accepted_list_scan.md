# CoRL 2026 accepted-list scan (VLA and compositional generalization)

Scan date 2026-11-10. Only the third-party portal loaded. All entries below are from it unless stated. Mark every Oral/Spotlight designation as UNVERIFIED against the official site.

## Which sources loaded, and how much of the 687 papers was seen

### Takeaway
Only https://hongsong-wang.github.io/CoRL2026 loaded (683,934 characters). It contains about 300 entries (IDs 1 to 300, the last one cut off), not 687. I saw IDs 1-299 in full, plus ID 300 header only. The official list, OpenReview and awards page did not load.

### Cited Findings
- Official accepted-papers page returned only navigation and footer, no list (JS-rendered or stripped) — [2026.corl.org/program/accepted-papers](https://2026.corl.org/program/accepted-papers)
- Awards page https://2026.corl.org/awards returned HTTP 404.
- OpenReview API redirected to a challenge page; the OpenReview group page was blocked by the fetch tool's domain check — [api2.openreview.net](https://api2.openreview.net/notes?content.venueid=robot-learning.org/CoRL/2026/Conference&limit=5), [group page](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Conference)
- Portal format: each entry has PaperID, Oral/Spotlight/Poster mark, OpenReview forum ID, arXiv, authors, abstract. The portal is not the official site, and its numbering and marks are unverified — [hongsong-wang.github.io/CoRL2026](https://hongsong-wang.github.io/CoRL2026)
- Portal marks: 21 Oral (IDs 1-21), Spotlight IDs 22-93 (72 spotlights), Poster from ID 94 on. In IDs 1-300 that is 21 orals + 72 spotlights + ~207 posters. The portal order looks sorted by designation, so orals and spotlights are probably complete only if the portal is a complete copy of the designations; I could not verify this.
- 687 accepted from 2,094 active submissions (~32.8%) per the official @corl_conf post as relayed by Digg; at that time chairs were still "selecting the oral presentations" — [Digg](https://digg.com/ai/ozl3pl3c)
- Author instructions: all accepted papers get posters, selected ones are invited for an oral spotlight presentation — [CoRL instructions](https://2026.corl.org/contributions/instruction-for-authors)
- Paper links: OpenReview URL form is https://openreview.net/forum?id=<forum ID>; arXiv is https://arxiv.org/abs/<id>. Forum IDs and arXiv IDs are in the tables below. Several arXiv IDs look unusual for the date (e.g. 2610.x, 2609.x, later than today's date of 2026-11-10 is not the case, but 2607.29687 has an implausibly high serial); treat arXiv IDs as UNVERIFIED.

### Gaps
- Papers ~301-687 not seen. No source found for them.
- Authors: the portal gave first author only for most entries; full author lists are not recorded here.
- Abstract summaries were produced by the fetch tool's summarizer from the portal text, not read verbatim by me. Treat them as paraphrases to be spot-checked.

## Complete oral and spotlight lists (as marked on the portal; UNVERIFIED against official site)

### Takeaway
21 orals and 72 spotlights are marked in IDs 1-93. Forum ID is listed so the link is https://openreview.net/forum?id=ID.

### Cited Findings (source for all rows: [portal](https://hongsong-wang.github.io/CoRL2026))

Orals (ID, title, first author, forum ID, arXiv):
- 1 Large-Scale Simulation Enables Zero-Shot Manipulation, Abhay Deshpande, c4Z0Y8gvnI, 2603.16861
- 2 OPENDEXGRASP: Open-vocabulary Task-Oriented Dexterous Grasping, Jiyao Zhang, wLgEYxjqid
- 3 RoboTTT: Context Scaling for Robot Policies, Yunfan Jiang, NHE9q2cz1n, 2607.15275
- 4 World Action Models are Zero-shot Policies (DreamZero), Seonghyeon Ye, Bshs9HDDiu, 2602.15922
- 5 Flow Policy Gradients for Robot Control, Brent Yi, acslhrig33
- 6 Memory-augmented Action-conditioned World Model for Robot Manipulation, Zirui Zheng, f9grq4Ncx9
- 7 Being-H0.5, Hao Luo, cfzFrOJmKA, 2601.12993
- 8 What Are We Actually Benchmarking in Robot Manipulation?, Tianchong Jiang, RdvH11pAfJ, 2606.04233
- 9 RL Token: Bootstrapping Online RL with VLAs, Charles Xu, RYXo2ktK1M, 2604.23073
- 10 PACE: Temporal-Role Tactile Conditioning for Chunk-Based Insertion Policies, Zhiyuan Chen, AIInkHZlvB
- 11 Closing the Representational Gap for VLAs in Dynamic Settings, Zhenyang Feng, FVZKpmPHV8
- 12 EgoScale, Ruijie Zheng, WF4X17Al2r, 2602.16710
- 13 SuperMimic, Qiayuan Liao, psNqjklVOK
- 14 pi0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities, Bo Ai, 5XKFwWrW0v, 2604.15483
- 15 Play2Perfect, Tyler Ga Wei Lum, RRl1xXZLdM
- 16 Long-Horizon Multi-Object Rearrangement via Distilled Tree Search, Kallol Saha, hWuk4y2AeR
- 17 GRAIL (humanoid loco-manipulation from 3D assets and video priors), Tianyi Xie, 2dEfANeQL7
- 18 Learning External Force Sensing for Commodity Robot Arms, Steven Oh, 03zMMFkTXk
- 19 Diversity You Can Actually Measure (dataset diversity metric), Sreevardhan Sirigiri, TscNG5Wcwn
- 20 RoboNaldo, Yichao Zhong, WBGnJHRIso
- 21 Driving, Fast or Slow? Neuro-Symbolic Guidance, Simon Kohaut, fHzDEYENPF

Spotlights, IDs 22-40 (from first chunk):
- 22 TARC, Arnav Sukhija; 23 Chunk-wise Model-based Policy Optimization, Weixi Song; 24 RAM: Reachability Across Morphologies, Tim Walter; 25 Playful Agentic Robot Learning, Junyi Zhang; 26 Control-Ready Uncertainty for Trajectory Diffusion, Zhiwei Xue; 27 Semantic Anchoring for Robotic Action Representations, Yuan Xu; 28 Visuo-Tactile Policy Learning for Submerged Manipulation, Bowen Jiang; 29 Learning Human Intention for Robotic Manipulation, Chengyang Li; 30 Scalable Deep Unfolding of Conic Optimizers, Alex Oshin; 31 LadderMan, Siheng Zhao; 32 PHUMA, Kyungmin Lee; 33 MolmoAct2, Haoquan Fang (Q6xWLHyg6x, 2605.02881); 34 MoRE: Mixture-of-Experts for Multi-Physics Robotic World Models, Ziyu Zhu; 35 BitVLA, Hongyu Wang (2506.07530); 36 Diagnosing Compositional Generalization in Sequential Robot Tasks, Yixiao Wang (gQFYIM5dJy); 37 Visible Touch, Metin Alp Dogan; 38 Emergent Control via Prior-Based Free Energy Minimization, Manish Kolachalam; 39 SidewalkBench, Zhizheng Liu; 40 Targeting World Models to Compromise Robot Learning Pipelines, Ethan Rathbun.
Spotlights, IDs 41-93 (first author in brackets where given; forum IDs in the portal): 41 Diffusion ReRoll [Seonsoo Kim]; 42 Physics-Guided Residual Dynamics for Deformable Simulation [Shivansh Patel]; 43 Robust Cloth Manipulation via Simulator-in-the-Loop Refinement [Xin Liu]; 44 WireCraft [Chongyu Zhu]; 45 Real-to-Sim-to-Real Shared Autonomy Teleoperation [Shuo Sha]; 46 FingerEye [Zhixuan Xu]; 47 Combining Constrained Sampling and RL [Marc Toussaint]; 48 Sequential Asymmetric Imitation [Ranpeng Qiu]; 49 Articulated 3D Scene Graphs [Martin Buchner]; 50 RoboKit [Jianglong Ye]; 51 OpenHLM [Yingdong Hu]; 52 OSCAR [Hshmat Sahak]; 53 MEM [Marcel Torne]; 54 Ego2Robot [Ye Wang]; 55 Shape Your Body [Nico Bohlinger]; 56 KPGrasp [Yuansen Huang]; 57 SCoCaT [Aman Arora]; 58 Language Grounded Bayesian Active Learning [Huao Li]; 59 Loco-Manipulation from SMPC Demonstrations [Martin Schuck]; 60 Region Arrival to Instance-Level Grounding in VLN [Xiangyu Shi]; 61 MosaicIMU [Junye Zou]; 62 PRTS [Yang Zhang]; 63 Jetson-PI [Zebin Yang]; 64 Systematic Study of Zero-Shot Cross-Embodiment Transfer in VLAs [Mi Yan]; 65 FlowDAgger [Michael Murray]; 66 BiDexGrasp [Mu Lin]; 67 Constraint-Decomposed Impedance Control [Kyungseo Park]; 68 TAM [Dongwon Son]; 69 Leave It Better Than Found [Mario Gomez Andreu]; 70 Noise-Invariant Spatial Encoding for 3D Policies [Zhiyuan Yang]; 71 Kamino [Vassilios Tsounis]; 72 GaP [Kaiyuan Chen]; 73 Meta-Ctrl [Gwen Yidou-Weng]; 74 ResSafe [Gechen Qu]; 75 Humanoid Navigation with Whole-Body VLA (TANGO) [Anqi Li]; 76 CMP [Ziyang Cheng]; 77 TRIAD Planner [Jiyun Kim]; 78 When Vision Overrides Language [Yu Fang] (HmdwOezjlQ, 2602.17659); 79 WM-Craftnet [Jie Yin]; 80 SERF [Sunghwan Kim]; 81 VisionSIM [Sacha Jungerman]; 82 TactileStep [Zizhuo Wang]; 83 Anchor3R [Peilin Tao]; 84 EmbodiSteer [Shihefeng Wang]; 85 Robustness without Wrinkles [Wei-Chen Li]; 86 Never Too Late for Force (LIFT) [Yi Wang]; 87 DeTac [Zhirui Zhang]; 88 REACT [Houlong Xiong]; 89 FedGuide [Zhilin He]; 90 ATOM-Bench [Zenan Wu]; 91 Collaborative Planning for UAV-UGV Teams [Zihao Deng]; 92 Learning What to Say to Your VLA [Hyun Joe Jeong, Gokul Swamy, Andrea Bajcsy]; 93 Demystifying When and Why VLAs Fail (FACT) [Carlota Pares Morlans].
- Chunk notes: paper 40's abstract and paper 83/128/174/300 boundaries were cut at chunk edges; entries 41-93 first author only.

### Gaps
- Full author lists for most oral/spotlight papers. Official confirmation that the marks are final.

## Catalogue of relevant papers (one sentence each, from portal abstracts via summarizer; source for all: [portal](https://hongsong-wang.github.io/CoRL2026), link = OpenReview forum ID given)

### Takeaway
Roughly 100 of the ~300 seen entries touch VLAs or related themes. New compositional/binding-relevant finds beyond your known list: ATOM-Bench (90), Compositional Diffusion Policy for VLA Agents (111), CoStream (219), AC latent actions (208), Relational Spatial Alignment (123), CounterAlign (122), Trajectory-Level Redirection Attacks (126), Learning What to Say (92), FACT (93), Info-VLA (124), Memory Anchors (114), Systematic cross-embodiment study (64), IMBench (176).

### Cited Findings

#### A. Compositional / systematic generalization, binding, language shortcuts
- 36 Diagnosing Compositional Generalization in Sequential Robot Tasks (Spotlight, Yixiao Wang, gQFYIM5dJy; known) — structured subsets of instruction tuples, as small as a quarter of the task space, recover strong OOD performance.
- 78 When Vision Overrides Language (Spotlight, Yu Fang, HmdwOezjlQ; known) — VLAs often ignore language for visual shortcuts; counterfactual action guidance mitigates this.
- 90 ATOM-Bench (Spotlight, Zenan Wu, DpOVq6nE2d; NEW) — real-world benchmark separating atomic motor skills from compositional generalization; strong atomic skills do not reliably transfer to new compositions.
- 111 Compositional Diffusion Policy for VLA Agents (Poster, Yuxuan Xu, Y55iJjxMsh; NEW) — combines semantic and physical energy terms in a diffusion policy for compositional generalization and feasibility.
- 219 CoStream (Poster, Haonan Chen, WeNTfF2Rks, 2606.26423; NEW) — composes semantic, predictive and reactive behaviors via one fusion operator for contact-rich assembly, with keypoint forecasting in imagined video.
- 208 Learning Additively Compositional Latent Actions (Poster, Hangxing Wei, 03qRtmw4qx, 2604.03340; NEW) — additive-composition constraint on latent action models for more meaningful, transferable latents.
- 14 pi0.7 (Oral; known) — steered by multimodal prompts of strategy and subgoals, enabling compositional task generalization.
- 8 What Are We Actually Benchmarking (Oral; known) — audits LIBERO, CALVIN etc.; scores often fail validity checks such as shortcut solvability.
- 122 CounterAlign (Poster, Haru Kondoh, AY4totylcx, 2608.21740) — counterfactual instruction-action pairs from demos train an instruction-grounded reward for offline RL on VLAs.
- 92 Learning What to Say to Your VLA (Spotlight, Jeong/Swamy/Bajcsy, z47bcul4Zk) — searches for instructions that improve a frozen VLA, with a conformalized head to avoid harmful steering.
- 126 Trajectory-Level Redirection Attacks on VLAs (Poster, Gokul Puthumanaillam, e9ZYAwJqFR) — near-benign prompt perturbations redirect VLA outcome while seeming to preserve the instruction.
- 244 Flexible Language-Guided Representation Editing (Poster, Audrey Lee, OJkazbkLtE) — refines task-feature representations from natural-language corrections.
- 171 CLAP (Poster, Yuri Ishitoya, GeENMbQXhi) — prepends action descriptions to action tokens to convert a VLM into a VLA.
- 58 Language Grounded Bayesian Active Learning (Spotlight, Huao Li, El4EDBhqCG) — clarifying ambiguous instructions as active learning over formal task specs.
- 241 Steering AR VLA via Action Token Intervention (Poster, Jason Chan, 3KzmpJQUpy) — injects user inputs into action tokens at inference.
- 240 VLS (Poster, Shuo Liu, RmP7m64fQ6, 2602.03973) — VLM-synthesized rewards over 3D keypoints steer frozen generative policies.
- 282 SynthICL (Poster, Cheng Qian, 6pHgTk73zs) — in-context imitation from synthetic RGB data generalizes to 16 unseen real tasks from one demo.
- 1 Large-Scale Simulation Enables Zero-Shot Manipulation (Oral) — procedurally generated synthetic data alone gives zero-shot real manipulation policies.

#### B. Object-centric, slot, keypoint, mask-conditioned policies
- 117 SlotFlow (Poster, Jijie Li, dKRgZYcdRV, 2609.24155; known) — separates "what" and "where" for object-centric flow-matching policies robust to distractors and spatial shifts.
- 281 Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement (Poster, Kinam Kim, tAXtXe4Oqd) — object-pose residual policy refines frozen VLA actions, real Franka success 42% to 76%.
- 253 ObstaDiff (Poster, Jiawen Wang, NnFdATGR3L) — structured target/obstacle/background representation for diffusion policies.
- 157 Phase-Conditioned Affordance Fields (Poster, Xinzhe Chen, zhiG7XB56D) — per-point affordance field anchoring SE(3)-equivariant policies.
- 254 Robot-WATCHDOG (Poster, Quentin Rolland, 1yL1d39r7K) — object-centric spatio-temporal graphs for failure detection.
- 49 Articulated 3D Scene Graphs (Spotlight); 232 Functional-SLAM; 228 PROSE — scene-graph representations (perception, not VLA).
- 56 KPGrasp (Spotlight) — grasps as object-frame hand keypoints with flow matching.

#### C. VLAs and generalist policies, architecture changes adding structure
- 3 RoboTTT (Oral; known) — test-time-training memory scales context to 8K timesteps, enabling one-shot imitation from human video.
- 4 DreamZero / World Action Models are Zero-shot Policies (Oral; known) — jointly predicts video and actions from a video-diffusion backbone, generalizes better than VLAs to new tasks.
- 7 Being-H0.5 (Oral; known) — shared action space plus Mixture-of-Flow separating shared primitives from embodiment experts.
- 33 MolmoAct2 (Spotlight; known) — action-reasoning VLA grafting a flow-matching action expert onto a discrete-token VLM.
- 53 MEM (Spotlight, Marcel Torne, 5TiiTY1fK3, 2603.03596; known) — compressed video memory plus text memory for tasks up to 15 minutes.
- 27 Semantic Anchoring for Robotic Action Representations (Spotlight) — anchors action representations to a semantic manifold to preserve pretrained structure.
- 11 Closing the Representational Gap for VLAs in Dynamic Settings (Oral) — adds motion summaries and proprioceptive history to resolve motion ambiguity.
- 9 RL Token (Oral) — compact RL token lets small actor-critic refine a VLA online.
- 37 Visible Touch (Spotlight) — renders contact into the image so VLAs use touch without architecture change.
- 12 EgoScale (Oral) — VLA trained on ~20k hours egocentric human video transfers to dexterous hands.
- 35 BitVLA (Spotlight) — native 1-bit VLA, 11x less memory.
- 25 Playful Agentic Robot Learning (Spotlight) — agentic play builds a persistent code-skill library.
- 29 Learning Human Intention (Spotlight) — gaze as intention with chain-of-thought reasoning for cross-embodiment transfer.
- 62 PRTS (Spotlight) — VLA pretraining with goal-conditioned contrastive RL.
- 64 Systematic Study of Zero-Shot Cross-Embodiment Transfer in VLAs (Spotlight, Mi Yan, 4RLjGPEfpH) — benchmark with 14 held-out embodiments isolating transfer factors.
- 80 SERF (Spotlight) — map tokens into a VLA for long-horizon mobile manipulation.
- 51 OpenHLM (Spotlight) — empirical recipe for a humanoid whole-body VLA.
- 86 LIFT (Spotlight) — reactive force-injection action expert added to a VLA.
- 93 FACT (Spotlight, z? BV0T1o1ZIL) — diagnoses two VLA failure modes in contact-rich tasks (flow-matching training mismatch, force-signal structure).
- 88 REACT (Spotlight); 63 Jetson-PI (Spotlight); 170 Barrier-Constrained Guidance; 236 DiscreteRTC; 238 DAM-VLA; 101 Real-Time AR Policies — inference/async execution for VLAs.
- 99 T-Rex (Poster, KdaClHHEUt) — tactile dataset plus mix-of-transformer VLA.
- 110 Coarse-to-Control (Poster) — VLA predicts coarse action tokens as a plan, then executable actions.
- 123 Relational Spatial Alignment for VLA (Poster, P1R9iBKyv4) — aligns pairwise relations to add 3D awareness without rank collapse.
- 124 Info-VLA (Poster, H5QWMBcNSU) — preserves cross-modal alignment during continual VLA adaptation.
- 203 GeomVLA; 204 SAGE-3D (semantic 3D reasoning tokens, FiLM-conditioned decoding); 144 MotionVLA; 205 TBD-VLA; 206 HARP-VLA; 207 FIBERTUNE; 262 FR-VLA (VLM module estimating when force matters); 273 PLaW-VLA (predictive latent world modeling in Mixture-of-Transformers); 141 NAC and 142 ActionCodec (action tokenizers); 143 Unified Motion-Action Modeling; 251 UniFlow; 248 HuRo; 211 CHORUS (one VLA, multi-robot collaboration); 145, 153, 214 (VLA post-training/RL); 136 TUTOR; 276 EMMA; 288 TDC (VLA outputs task frames and gains).
- Structure added to VLAs specifically: memory (53, 3, 144, 147 Native Memory Compression, 146 Chameleon, 209 TRACE, 210 Workspace Models, 94 Gated Memory Policy, 113 MessyMem); phases/roles (10 PACE temporal-role tactile conditioning, 157 phase-conditioned affordances); reasoning (33, 29, 110, 201 DriveMA meta-actions); objects/3D (203, 204, 281); experts (7, 99, 273).

#### D. Modular, skill-library, adapter, MoE
- 95 Energy-Based Policy Orchestration (Poster, NZrwlgnR4X) — runtime policy-goal selection via per-policy energy models; new policies added without retraining.
- 96 Proxy Policy Steering (Poster, tfC6LMD02z) — residual from lightweight proxy policies steers a frozen foundation model without eroding base skills.
- 72 GaP (Spotlight) — computation graphs from a modular skill library; 25 code-skill library; 296 Conquer (Poster) — semantic skill library, little forgetting; 264 SkillX; 34 MoRE (MoE world models); 7 Being-H0.5 (Mixture-of-Flow); 61 MosaicIMU (MoE); 158 LCLA and 166 PathLift (adapters); 231 G2G (32M bridging params); 168 Automating Design of Embodied Agent Architectures; 269 FT-WBC.

#### E. Hierarchical / dual-system
- 112 LoHo-Manip (Poster, n31eEof0k3) — task-management VLM issuing receding-horizon plans with a short-horizon VLA executor.
- 188 VISTA scaling world model (Poster, hCG12ifteJ, 2602.10983) — world model proposes goal-image subtasks for a low-level VLA.
- 116 Procedural Memory for Proactive Assistance (Poster) — VLM reading FSM memory plus VLA executor.
- 218 VoLo (Poster) — planning/monitoring orchestrator; 133 Hierarchical Planning with Latent World Models; 184 HLO-AMP; 266 EVOLVE; 234 EvoPlan.

#### F. Interpretability / mechanistic analysis
- 108 Mechanistic Interpretability for End-to-End Self-Driving (Poster, bqXyS2gKba) — sparse autoencoders on a driving planner find latents tied to intent, speed, braking; steering them changes outputs.
- 72 GaP and 235 LEMCA produce interpretable programs. No other policy-interpretability paper seen in IDs 1-300. 93 FACT is a diagnosis paper, not mechanistic.

#### G. Continual / lifelong
- 114 Memory Anchors for Continual Robot Learning (Poster, dSbtHPqFj9, 2608.26545) — identifies a small set of old-task experiences anchoring replay; large forgetting reduction on LIBERO.
- 124 Info-VLA (continual VLA adaptation); 296 Conquer (skill library); 65 FlowDAgger (adaptation preserving priors).

#### H. Benchmarks for generalization
- 8 (Oral), 90 ATOM-Bench, 64 cross-embodiment, 176 IMBench (35 tasks, physical reasoning to constrained plans), 177 stochastic real-world challenges, 174 HARP-Bench, 255 RMMBench, 278 H2RBench, 256 AnyRank, 44 WireCraft, 39 SidewalkBench, 265 HumanoidMimicGen, 19 dataset diversity metric (Oral), 169 physical danger benchmark.

#### I. World-action models and world models
- 4 DreamZero (Oral; known); 186 WAM-STEP (Poster, 1u6oriJJcU) — turns the training-only visual branch into an imagination module for an action-only policy; 187 CAM Causal WAMs (Poster, 2608.10232) — learns from failed rollouts; 65 FlowDAgger adapts WAMs; 298 GeoWorldAD; 148 Large Video Planner; 190 mu0; 274 AnyWorld; 277 AIM; 6 memory-augmented world model (Oral); 40 world-model poisoning (Spotlight).

### Inferences
- None of the new finds I saw is a direct duplicate of MESA/BiMESA, MAST-VLA or "binding" work. I did not see those names in IDs 1-300; they may be in the unseen ~387 papers, or not in this portal.
- ATOM-Bench, Compositional Diffusion Policy for VLA Agents, CoStream and AC latent actions are the nearest additional compositional-generalization papers.

### Gaps
- MESA/BiMESA, MAST-VLA and DreamZero-related items: MESA/BiMESA and MAST-VLA not found in the seen portion (unverified whether accepted).
- Poster entries mostly lack full authors; verify abstracts directly for any entry cited in the paper.

## Rough composition of the conference (from IDs 1-299 only; my own hand classification, +/-30%)

### Takeaway
VLA/generalist-policy work is the largest single theme among seen entries, followed by humanoid/legged control and tactile/dexterous manipulation. The sample is the portal's order (designation-sorted), so poster-heavy IDs 94-300 dominate and orals/spotlights are overrepresented relative to the full 687.

### Cited Findings (all derived from the portal listings above)
- VLA and generalist policies (incl. tokenizers, inference, post-training): ~55
- World models and world-action models: ~25
- Humanoid, legged and whole-body control/loco-manipulation: ~45
- Tactile, force, dexterous and grasping: ~40
- Navigation (incl. VLN) and autonomous driving: ~35
- Memory, long-horizon, planning, hierarchical: ~20
- Data, teleoperation, human video, sim: ~25
- RL / human-in-the-loop fine-tuning: ~15
- Benchmarks and evaluation: ~15
- Perception, SLAM, localization, safety, multi-robot, other: ~25
- Counts overlap; they are not an exact partition.

### Gaps
- No full-list count possible without the other ~387 papers.

## Awards

### Takeaway
No best-paper award or finalist announcement found.

### Cited Findings
- Awards page returned 404 — https://2026.corl.org/awards
- Search found no 2026 finalists; official notification date was September 4, 2026 — [reviewer instructions](https://www.corl.org/contributions/instruction-for-reviewers)
- A workshop (Sim-to-Real-to-Field) has its own $1,000 best paper award, which is not the main-conference award — [search result](https://2026.corl.org/contributions)

### Gaps
- Main-conference award status unknown; check https://2026.corl.org/ directly or the program page.

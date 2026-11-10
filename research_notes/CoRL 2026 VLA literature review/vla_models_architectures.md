# CoRL 2026 and 2026 VLA models and architectures (excluding NVIDIA GR00T)

Scope and caveats: ~17 tool calls, partial coverage. The alphaxiv oral/spotlight pages were truncated (about 20 papers each), so the CoRL 2026 list is incomplete. Many details below come from search-engine summaries or the awesome-VLA list (a secondary aggregator) and are marked "unverified". Venue labels: CoRL-O = CoRL 2026 oral, CoRL-S = CoRL 2026 spotlight (per alphaxiv listing), arXiv = arXiv/blog release with no CoRL listing found.

## Which new VLAs / generalist policies appeared at CoRL 2026, and which are most discussed?

### Takeaway
CoRL 2026 orals and spotlights include pi0.7, MEM, RL Token (Physical Intelligence), MolmoAct2, Being-H0.5, BitVLA, MolmoBot, and several VLA-diagnosis papers. Outside CoRL, the 2026 releases the field treats as SOTA are Gemini Robotics 2.5 (closed), Figure Helix 02 (closed), Qwen-VLA, LingBot-VLA 2.0, Xiaomi-Robotics-1 and G0.5. Only pi0.7, MolmoAct2 and Being-H0.5 have been verified in detail below.

### Cited Findings
CoRL 2026 orals (alphaxiv list, partial; [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)):
- pi0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities (Physical Intelligence) is a CoRL-O — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- RL Token: Bootstrapping Online RL with VLAs (Physical Intelligence; Xu, Springenberg, Levine) is a CoRL-O — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- Being-H0.5 (Luo, Wang, Lu) is a CoRL-O — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- MolmoBot: Large-Scale Simulation Enables Zero-Shot Manipulation (AI2/UW; Deshpande, Guru, Krishna) is a CoRL-O; policies trained only on synthetic data transfer zero-shot to real robots — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- RoboTTT: Context Scaling for Robot Policies (NVIDIA, Stanford; Jiang, Chebotar, Fei-Fei) is a CoRL-O; long-context policy imitating from a single human video — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- EgoScale (Zheng, Niu, Fan): egocentric human data pretraining for dexterous manipulation, one-demo adaptation; CoRL-O — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- World Action Models are Zero-shot Policies (NVIDIA; Ye, Ge, Du) is a CoRL-O; joint future video + action prediction. Not GR00T by title, but NVIDIA, so flag to the other researcher — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- "What Are We Actually Benchmarking in Robot Manipulation?" (TTIC/UChicago; Jiang, Tan, Wheeler) is a CoRL-O that audits LIBERO and CALVIN and proposes diagnostics. Directly relevant to benchmark validity — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- Other VLA-adjacent orals: TEMPO (temporal context for dynamic manipulation), Play2Perfect, PACE (tactile), OPENDEXGRASP, Memory-augmented Action-conditioned World Model — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)

CoRL 2026 spotlights ([alphaxiv spotlight](https://www.alphaxiv.org/conferences/corl/2026/spotlight)):
- MolmoAct2: Action Reasoning Models for Real-World Deployment (Fang, Duan, Krishna) — CoRL-S
- MEM: Multi-Scale Embodied Memory for VLAs (Torne, Pertsch, Finn) — CoRL-S
- BitVLA: 1-bit VLA models (Wang, Xiong, Wang), fully 1-bit, matches a full-precision baseline per listing — CoRL-S
- When Vision Overrides Language: counterfactual failures in VLAs (Fang, Feng, Ding) — CoRL-S
- Demystifying When and Why VLAs Fail in Contact-Rich Tasks (Parés Morlans, Kuhn, Bohg); proposed FACT reaches 66% vs 41% best baseline (listing's figure, unverified) — CoRL-S
- ZETA: controlled study of zero-shot cross-embodiment VLA transfer; local end-effector coordinates help — CoRL-S
- PRTS (VLA pretraining as goal-conditioned RL), Semantic Anchoring for Robotic Action Representations, Never Too Late for Force (reactive force injection), FlowDAgger (Murray, Chen, Mees), Ego2Robot, Learning Human Intention (gaze), OpenHLM and Humanoid Navigation with Whole-Body VLA (Li, Chen, Shah) — CoRL-S
- Other 2026 CoRL: DILL (Domain-Invariant Latent Lookahead) is a CoRL 2026 paper, code "coming soon" — [GitHub](https://github.com/DILL-VLA/DILL)

Note: LIBERO-CF / language papers were not all independently confirmed for CoRL acceptance beyond the alphaxiv listing.

### Inferences
- PI has three CoRL-featured papers (pi0.7, MEM, RL Token), so its stack is the de facto reference that others compare against (most 2026 releases below benchmark against pi0.5).

### Gaps
- Full accepted list (OpenReview/corl.org) not retrieved; citation counts not found. OpenVLA-OFT/SmolVLA/Octo/CogACT/UniVLA/FLOWER/WALL-OSS 2026 successors not searched individually.

## Architectures: what each model does (grouped by idea)

### Takeaway
Dominant 2026 recipe: pretrained VLM backbone + flow-matching/DiT action expert, with the VLM often trained by discrete tokens (FAST-style) and gradients insulated; added ideas are memory, subgoals/metadata conditioning, latent world models, and reasoning tokens.

### Cited Findings

**A. Knowledge-insulated VLM + flow expert, with rich context conditioning (Physical Intelligence)**
- pi0.7 (arXiv 2604.15483, submitted 16 Apr 2026, v2 24 Apr; Physical Intelligence, 88 authors incl. Levine, Finn, Pertsch, Hausman; CoRL-O). Links: [arXiv](https://arxiv.org/abs/2604.15483), [PDF](https://www.pi.website/download/pi07.pdf), project pi.website/pi07 — [arXiv abs](https://arxiv.org/abs/2604.15483)
- About 5B params: 4B VLM initialised from Gemma3 4B (400M vision encoder) + 860M flow-matching action expert, 50-step chunks, adaptive RMSNorm timestep injection. Knowledge insulation: backbone supervised with FAST tokens by cross-entropy, expert gradients do not flow into backbone. MEM-style video encoder, up to 6 history frames at 1 s stride, 448x448 images. Real-time chunking with simulated delay 0-12 steps. 5 denoising steps, up to 240 ms latency — [arXiv HTML](https://arxiv.org/html/2604.15483)
- Context conditioning: metadata (overall speed binned, quality 1-5, mistake label), subgoal images in 25% of examples, control mode, CFG on metadata (beta 1.3/1.7/2.2). At test time speed = 15th-percentile episode length, quality 5, mistake false — [arXiv HTML](https://arxiv.org/html/2604.15483)
- Subgoal world model initialised from BAGEL (14B MoT), generates multi-view subgoal images, refreshed per subtask or every 4 s — [arXiv HTML](https://arxiv.org/html/2604.15483)
- Data: multi-robot demos, autonomous eval data incl. failures and interventions (incl. pi0.6* RL runs), open robot datasets, egocentric human video, web data (VQA, captioning, localisation) — [arXiv HTML](https://arxiv.org/html/2604.15483)
- Numbers (text only): zero-shot shirt folding on UR5e, humans 90.9% progress / 80.6% success vs pi0.7 85.6% / 80%; unseen tasks roughly 60-80%, seen often >90%. Most comparisons (vs pi0.5, pi0.6, RL specialists, 14-scenario instruction following) appear only in figures, not extracted — [arXiv HTML](https://arxiv.org/html/2604.15483). Claims espresso/laundry/box generalist matches RL-finetuned specialists — [arXiv abs](https://arxiv.org/abs/2604.15483)
- Weights: not stated in paper text read; no release found. Pi API beta (13 Oct 2026) offers fine-tuning and hosted inference — [pi.website blog](https://www.pi.website/blog). Treat as not open (unverified).
- TechCrunch frames it as compositional generalization, and notes the team says benchmarks are lacking — [TechCrunch](https://techcrunch.com/2026/04/16/physical-intelligence-a-hot-robotics-startup-says-its-new-robot-brain-can-figure-out-tasks-it-was-never-taught/)
- Third-party explainer: air-fryer zero-shot instruction gave only partial progress; coaching improved it; paper claims "early signs" of compositional generalization, not unrestricted language control — [Physical AI Lab](https://physicalailab.io/en/physical-intelligence-pi07-steerable-vla/)
- pi0.6* ("A VLA That Learns From Experience", arXiv 2511.14759, Nov 2025) and pi0.5 (arXiv 2504.16054) are predecessors — as cited in [HarnessPAI refs](https://arxiv.org/pdf/2609.29166), [UniSteer refs](https://arxiv.org/pdf/2605.10821)

**B. Memory and long horizon**
- MEM (arXiv 2603.03596, 4 Mar 2026; Torne, Pertsch, Walke, Levine, Finn, Driess et al., 17 authors; CoRL-S): video-based short-term memory via a video encoder plus text-based long-term memory; tasks up to 15 minutes (kitchen cleaning, grilled cheese); in-context strategy adaptation. Abstract gives no success numbers — [arXiv](https://arxiv.org/abs/2603.03596). Built on pi0.6; one secondary source says 6 frames in pretraining, up to 18 frames / 54 s at inference (unverified) — [search summary](https://arxiv.org/html/2603.03596). PI blog 3 Mar 2026 — [pi.website blog](https://www.pi.website/blog)
- G0.5 (Galaxea, arXiv 2608.11739): single autoregressive decoder emitting reasoning + action tokens, learnable cross-embodiment action tokenizer, visual memory; LIBERO 98.9%; real-world fine-tuning 76.7% vs 53.3% for pi0.5 (all via aggregator, unverified) — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)
- MiniCPM-Robot (OpenBMB): 1.5B on-device, streaming context, 4x visual-token compression; claims beating pi0.5 and Qwen-VLA on LIBERO/CALVIN/RoboTwin2 (no numbers; unverified) — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)

**C. Reasoning tokens + discrete tokenizer + flow expert (open)**
- MolmoAct2 (arXiv 2605.02881, submitted 4 May 2026; Fang, Duan, Krishna, Farhadi, Fox et al., AI2; CoRL-S): flow-matching action expert grafted onto a discrete-token VLM via per-layer KV-cache conditioning; MolmoER spatial/embodied-reasoning VLM (3.3M samples, specialize-then-rehearse); OpenFAST open tokenizer (millions of trajectories, 5 embodiments); MolmoThink adaptive-depth reasoning re-predicting depth tokens only for changed regions; MolmoAct2-BimanualYAM 720 h plus filtered DROID and SO100/101 subsets. Claims beats Pi-05 across 7 sim+real benchmarks and MolmoER beats GPT-5 and Gemini Robotics ER-1.5 on 13 embodied-reasoning benchmarks (abstract, no numbers). Weights, training code, and full data stated released; project page https://allenai.org/blog/molmoact2 — [arXiv](https://arxiv.org/abs/2605.02881). Best academic fine-tuning candidate (open, SO100-scale hardware), but VRAM needs unverified.

**D. Human/egocentric data, unified action space, cross-embodiment**
- Being-H0.5 (arXiv 2601.12993, 19 Jan 2026; BeingBeyond, Zongqing Lu; CoRL-O): Mixture-of-Transformers with a Mixture-of-Flow action framework (shared motor primitives plus embodiment-specific experts); Unified Action Space with semantically aligned slots; UniHand-2.0 corpus 35,000+ h / 120B tokens (16k h egocentric human video, 14k h robot, 5k h VL); LIBERO 98.9%, RoboCasa 53.9% with RGB-only; one checkpoint on 5 real embodiments (PND Adam-U, Franka+Inspire, Unitree G1, BeingBeyond D1, SO-101) — [arXiv](https://arxiv.org/abs/2601.12993) (numbers from search summary; verify tables)
- Being-H0.7 (arXiv 2605.00078): latent world-action model, deployable prior branch + training-only posterior branch; "SOTA or comparable on six sim benchmarks" — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action) (unverified)
- EgoScale, Ego2Robot, Psi0 (arXiv 2603.12263, RSS 2026: Qwen3-VL planner + ~500M DiT expert + RL lower-body tracker, ~80 trajectories to new skills), and ZETA (CoRL-S) all push human video or cross-embodiment transfer — [alphaxiv](https://www.alphaxiv.org/conferences/corl/2026/oral), [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)
- Qwen-VLA (arXiv 2605.30280, 28 May 2026; Alibaba Qwen, 40 authors; arXiv, no CoRL listing found): Qwen VLM + DiT action decoder, embodiment given by text prompt, one model for manipulation, navigation, trajectory prediction. LIBERO 97.9%, SimplerEnv-WidowX 73.7%, RoboTwin Easy/Hard 86.1/87.2%, R2R-OSR 69.0, RxR-SR 59.6, real ALOHA OOD avg 76.9%, DOMINO 26.6%. No code repo linked on arXiv page (awesome list says Apache-2.0, unverified) — [arXiv](https://arxiv.org/abs/2605.30280)
- RDT2 (arXiv 2602.03310, Tsinghua): 7B diffusion transformer on UMI data, zero-shot to unseen embodiments (unverified) — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)
- LingBot-VLA (arXiv 2601.18692; ~20,000 h dual-arm, 9 configs) and LingBot-VLA 2.0 (arXiv 2607.06403; 6B, Qwen3-VL backbone, token-level MoE action expert, 55-D canonical action space, ~60K h, 20 configs, Apache-2.0, beats pi0.5 on long-horizon mobile manipulation) — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action) (unverified)
- Xiaomi-Robotics-1 (arXiv 2607.15330): Mixture-of-Transformers, Qwen3-VL + DiT flow head, 100K+ h UMI pretraining; RoboCasa365 57.6% — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action) (unverified)
- Hy-Embodied-0.5-VLA (Tencent, arXiv 2606.14409): MoT, 2B edge and 32B-activated variants, RoboTwin 2.0 ~90.9% — same source (unverified)
- ABot-M0 (arXiv 2602.11236): DiT expert on Qwen3-VL, 6M+ trajectories / 9,500 h, 20+ embodiments — same source (unverified)
- Dexora (arXiv 2605.18722): 36-DoF dual-arm dual-hand; 66.7% vs GR00T N1 51.7% — same source (unverified)
- PointACT (arXiv 2605.21414, RSS 2026): Qwen2.5-VL + point-action expert on point clouds; LIBERO avg 96.0%, RLBench-10 82.3% — same source (unverified)

**E. Latent world models / video co-training**
- World Action Models are Zero-shot Policies (NVIDIA, CoRL-O): joint future video and action prediction — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- VLA-JEPA (arXiv 2602.10098, latent world model), DILL (CoRL, predicts task-relevant future representations, suppresses irrelevant visual cues), tau0-VLA (world-model-guided test-time compute), Magic-W0 (arXiv 2609.39870, 3.4B world-action model; LIBERO 99.1%, aggregator claim) — [arXiv VLA-JEPA](https://arxiv.org/pdf/2602.10098), [DILL](https://github.com/DILL-VLA/DILL), [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)
- Orca (arXiv 2606.30534) and FoMoVLA (arXiv 2607.14739, visual foresight + motion guidance) surfaced but were not read — [search result](https://arxiv.org/pdf/2606.30534)

**F. Efficiency / small VLAs**
- BitVLA (CoRL-S): fully ternary 1-bit VLA on BitNet b1.58 2B4T — [alphaxiv spotlight](https://www.alphaxiv.org/conferences/corl/2026/spotlight)
- MiniCPM-Robot above; FASTER (real-time flow-VLA scheduling) — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)
- No SmolVLA successor or new LeRobot VLA found in one search — [roboticscenter.ai summary search](https://www.roboticscenter.ai/learn/vla-models-comparison)

**G. Hierarchical / closed commercial systems**
- Gemini Robotics 2.5 (DeepMind blog dated 10 Nov 2026, i.e. today): VLA with inference-time reasoning guidance; 200 tasks; success 30-100% above baselines, 20-30% faster; Apptronik Apollo 2 at Mercedes-Benz kitting 96%; rolling out to partners only. A separate ER model tracked workflow — [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/). The fetch summary also mentioned a "Gemini 4 Argon" jailbreak figure that I could not reconcile with other sources; treat as unverified. Note: an earlier iottechnews headline "Gemini Robotics 2.5 for factory humanoids" exists — [iottechnews](https://iottechnews.com/news/google-unveils-gemini-robotics-2-5-for-factory-humanoids/)
- Gemini Robotics ER 2 (30 Jul 2026, via Gemini API; ER 1.6 preview 14 Apr 2026) — [Google blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/), [API docs](https://ai.google.dev/gemini-api/docs/robotics-overview)
- Figure Helix 02 (27 Jan 2026): System 0 (10M-param 1 kHz whole-body controller) + System 1 (200 Hz) + System 2; closed. Helix 2.5 (17 Sep 2026) closed, claims zero-shot in 30 unseen homes (aggregator, unverified) — [Figure](https://figure.ai/news/helix-02), [roboticscenter.ai](https://www.roboticscenter.ai/institutions/models/figure-ai)
- GR00T N1.7 deliberately excluded.

### Inferences
- Architectural trends visible: (1) VLM + flow/DiT expert is near-universal, with MolmoAct2 explicitly hybridising discrete tokens (OpenFAST) with flow, and pi0.7 using FAST tokens only to train the backbone; (2) knowledge insulation is now standard in PI's line; (3) memory (MEM, G0.5) and conditioning on metadata/subgoals (pi0.7) instead of pure language; (4) latent world-model branches (Being-H0.7, VLA-JEPA, DILL, WAM); (5) human egocentric video as a main data scale axis (Being-H0.5, EgoScale, Psi0); (6) text-prompted embodiment (Qwen-VLA) and unified action spaces (Being-H, LingBot 55-D); (7) real-time chunking in training (pi0.7).

### Gaps
- Not retrieved: Octo/OpenVLA-OFT/CogACT/UniVLA/FLOWER/WALL-OSS 2026 updates, X-VLA, pi0.6 details, Helix primary details, per-paper arXiv for BitVLA, PRTS, ZETA.

## Open checkpoints and codebases; fine-tuning on LIBERO with A40s

### Takeaway
Confirmed-open: MolmoAct2 (weights, code, data stated in abstract). Claimed-open via aggregator (unverified): LingBot-VLA 2.0, Xiaomi-Robotics-1, Hy-Embodied-0.5-VLA, MiniCPM-Robot, RDT2, Dexora, Qwen-VLA. Closed: pi0.7 (API only), Gemini Robotics 2.5, Helix.

### Cited Findings
- MolmoAct2 weights/code/data released — [arXiv](https://arxiv.org/abs/2605.02881); URL for repo/HF not found
- MiniCPM-Robot repo — [GitHub](https://github.com/OpenBMB/MiniCPM-Robot); small (1.5B), most plausible for 48 GB A40 fine-tuning (my inference)
- Licenses/links for others from [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action) only
- BitVLA code not checked.

### Inferences
- A40 (48 GB) fine-tuning is plausible for the 1.5B to ~7B open models with LoRA or small batch; pi0.7 is not available to fine-tune except through the paid Pi API.

### Gaps
- VRAM requirements and actual HF checkpoint links not verified for any model.

## Reported benchmark numbers (LIBERO, LIBERO-Plus/PRO, SimplerEnv, RoboCasa, CALVIN, real)

### Takeaway
LIBERO is saturated near 98-99% for several 2026 models (self-reported); robustness benchmarks show far larger gaps.

### Cited Findings
- LIBERO: Magic-W0 99.1% (aggregator), Being-H0.5 98.9% ([arXiv](https://arxiv.org/abs/2601.12993)), G0.5 98.9%, Qwen-VLA 97.9% ([arXiv](https://arxiv.org/abs/2605.30280)), PointACT avg 96.0% — [awesome-VLA](https://github.com/ai4s-research/awesome-vision-language-action)
- RoboCasa: Being-H0.5 53.9% (search summary), Xiaomi-Robotics-1 RoboCasa365 57.6% (aggregator). Different RoboCasa versions, not comparable.
- SimplerEnv-WidowX: Qwen-VLA 73.7% — [arXiv](https://arxiv.org/abs/2605.30280)
- RoboTwin 2.0: Hy-Embodied ~90.9%; Qwen-VLA 86.1/87.2% (Easy/Hard)
- CALVIN: no concrete 2026 numbers found.
- LIBERO-Plus (CVPR 2026, originally arXiv 2510.13626): perturbations across 7 dimensions; performance drops from 95% to below 30% under modest perturbations; camera viewpoint and robot initial state are the most damaging; models largely ignore language — [arXiv](https://arxiv.org/pdf/2510.13626), [CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)
- LIBERO-PRO: perturbs object, position, instruction, environment; reported that models produce identical trajectories with meaningless text (second-hand via arXiv 2603.00592) — [arXiv 2603.00592](https://arxiv.org/html/2603.00592v1). No own numbers found.
- CoRL-O "What Are We Actually Benchmarking" audits LIBERO/CALVIN — [alphaxiv oral](https://www.alphaxiv.org/conferences/corl/2026/oral); contents not read.
- Real robot: pi0.7 above; G0.5 76.7 vs 53.3 (pi0.5); Qwen-VLA ALOHA OOD 76.9%.

### Inferences
- Self-reported LIBERO figures use differing protocols (per-suite averages, fine-tuning regimes); not comparable without rerunning.

### Gaps
- No LIBERO-Plus/PRO scores for 2026 models; CALVIN none; SimplerEnv only one model.

## Language following and generalization claims

### Takeaway
Evidence suggests current VLAs under-use language; 2026 papers address it with counterfactual benchmarks (LIBERO-CF) while pi0.7 makes the strongest compositional-generalization claim, with company-run evidence.

### Cited Findings
- When Vision Overrides Language (arXiv 2602.17659, 19 Feb 2026; Fang, Ding, Szafir et al.; CoRL-S): LIBERO-CF counterfactual benchmark; all VLAs tested fail, OpenVLA-OFT worst (4.7% grounding, 0.4% success); Counterfactual Action Guidance (training-free dual branch with language-unconditioned vision-action prior) gives pi0.5 +9.7% language-following accuracy, +3.6% success (training-free) and +15.5% / +8.5% with a trained prior; real-world +17.2% avg success — [arXiv](https://arxiv.org/abs/2602.17659) (figures from search excerpts, unverified)
- LIBERO-Plus finding that models ignore language — [arXiv 2510.13626](https://arxiv.org/pdf/2510.13626)
- pi0.7: 14-scenario instruction-following, referential instruction, and dataset-bias experiments exist but numbers only in figures; unseen-task success ~60-80% — [arXiv HTML](https://arxiv.org/html/2604.15483)
- Qwen-VLA: OOD generalisation over scene, lighting, object, embodiment — [arXiv](https://arxiv.org/abs/2605.30280)
- ZETA, Semantic Anchoring (keeps action representations aligned with pretrained semantics) are CoRL-S papers on transfer/generalisation — [alphaxiv spotlight](https://www.alphaxiv.org/conferences/corl/2026/spotlight)

### Inferences
- For a compositional-generalization study on LIBERO, LIBERO-CF, LIBERO-Plus, and LIBERO-PRO are the relevant stress tests; pi0.7's data-diversity and metadata ablations are the closest published mechanism evidence.

### Gaps
- pi0.7 figure numbers (language following vs pi0.5/pi0.6) require reading the PDF figures. The tool-results copy of the PDF is at /nethome/ksingara3/.claude/projects/-coc-flash10-ksingara3/519e1945-9c29-4a3d-8901-9dabcdd417f1/tool-results/webfetch-1794344902353-6ry0fc.pdf (binary, not parsed).

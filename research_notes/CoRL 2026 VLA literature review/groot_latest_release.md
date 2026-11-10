# Newest NVIDIA Isaac GR00T release (as of 2026-11-10) and the GR00T line

Method note: findings come from WebSearch/WebFetch summaries (a small model summarised each page; exact wording not verified). The GR00T 2 whitepaper PDF exceeded the fetch size limit and was NOT read, so everything on GR00T 2 internals comes from the project page summary. Anything not stated in a source is listed under Gaps.

## 1. What is the newest release called, when and where was it announced, and where are the links?

### Takeaway
The newest release is literally named "GR00T 2" (not "N2"), announced 2026-11-09 at CoRL 2026 (Austin, TX) alongside Isaac Lab 3.0. It is a "world action model" (WAM) built on NVIDIA Cosmos / DreamZero-style joint video-action diffusion, not a dual-system VLA. "GR00T N2" was the name used for a GTC 2026 (March) preview; whether GR00T 2 is the same model is not stated by any source I read. Code is "Available in December" (per project page); weights availability and license are not stated.

### Cited Findings
- NVIDIA announced GR00T 2 on Nov 9, 2026 at CoRL in Austin, called an "updated open frontier world action model", together with Isaac Lab 3.0 — [Robot Report](https://www.therobotreport.com/nvidia-launches-gr00t-2-isaac-lab-3-0-corl/); [NVIDIA blog (via fetch)](https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/)
- Primary page: GEAR project page, whitepaper dated November 2026, @techreport (not arXiv) — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/); whitepaper PDF https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf (unread, too large)
- The project page says code is "Available in December"; it gives no GitHub or Hugging Face model link, and no weights license — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- Press coverage calls it "open" but none states license, release date or download source — [NVIDIA blog](https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/); [dev.to](https://dev.to/paullovelace/nvidia-just-launched-gr00t-2-and-isaac-lab-30-one-demo-to-teach-a-robot-a-new-skill-cen)
- Released dataset: Human-1M-100K — https://huggingface.co/datasets/nvidia/PhysicalAI-Human-1M-100K ; explorer https://huggingface.co/spaces/nvidia/PhysicalAI-Human-1M-Explorer — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- GTC 2026 (March): Jensen Huang previewed "GR00T N2", "based on DreamZero research", new world-action-model architecture, "slated to be available by the end of the year"; GR00T N1.7 called current, "commercially viable" — [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-expands-open-model-families-to-power-the-next-wave-of-agentic-physical-and-healthcare-ai)
- Third-party explainer dated Nov 6, 2026 said no N2 checkpoint or branch existed yet — [physicalailab.io](https://physicalailab.io/en/nvidia-groot-n2-vs-n17/)
- Version timeline: N1 paper arXiv 2503.14734 (March 2025); N1.5; N1.6 released 2025-12-15 (research page; HF nvidia/GR00T-N1.6-3B); N1.7 GA (repo main; HF nvidia/GR00T-N1.7-3B; exact date not found) — [GitHub](https://github.com/NVIDIA/Isaac-GR00T); [N1.6 page](https://research.nvidia.com/labs/gear/gr00t-n1_6/)

### Inferences
- "GR00T 2" very likely is the shipped form of the "N2" preview (both are WAMs claimed to build on DreamZero/Cosmos), but no source I read confirms the identity. Treat as unverified.

### Gaps
- Official GR00T 2 paper/arXiv ID, HF model card, GitHub repo, license, and exact weight release: not found. N1.7 release date not found. The GR00T 2 Nov 6 vs Nov 9 status: the physicalailab page (Nov 6) predates the announcement.

## 2. Architecture changes N1 -> N1.5 -> N1.6 -> N1.7 -> GR00T 2

### Takeaway
N1-N1.7 are dual-system VLAs (VLM + flow-matching DiT action head). GR00T 2 abandons that: a single diffusion transformer jointly denoises future video and actions (world-action model). Per-version effect on language following is quantified only for N1 -> N1.5.

### Cited Findings
- N1.5: VLM (Eagle 2.5) frozen during pretraining and finetuning; FLARE loss coefficient 0.2; language following on real GR-1 46.6% (N1) -> 93.3% (N1.5), overall success 43.3% -> 83.0% (one third-party page attributes the same numbers to Unitree G1; NVIDIA page is the better-supported GR-1 attribution) — [N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5) (via search summary)
- N1.6 (2025-12-15): internal Cosmos-2B VLM variant, native aspect ratio; DiT depth 16 -> 32 layers; post-VLM 4-layer adapter removed; state-relative action chunks for most embodiments; top 4 VLM layers unfrozen in pretraining; added several thousand hours of YAM, AgiBot Genie-1, sim Galaxea R1 Pro, Unitree G1 data; 300K steps, batch 16,384; ~3B params; page gives no numeric benchmarks, only "outperforms N1.5" — [N1.6 page](https://research.nvidia.com/labs/gear/gr00t-n1_6/); params from [aiwiki](https://aiwiki.ai/wiki/groot_n1_6/raw) (secondary)
- N1.7: VLM replaced by Cosmos-Reason2-2B (Qwen3-VL architecture); relative end-effector action space shared by robot and human data; +20K h EgoScale human video; action horizon 16 -> 40; state/action dims 29 -> 132; DiT 16 layers (down from 32); ~3B (HF name 3B); TensorRT ~40 Hz (RTX PRO 6000 Blackwell) and ~28 Hz (Jetson AGX Thor) batch-1 LIBERO; code Apache 2.0, weights NVIDIA Open Model License; "comparable performance to N1.6" — [GitHub README](https://github.com/NVIDIA/Isaac-GR00T)
- N1.7 LIBERO avg 96.5% (Spatial 95, Object 100, Long 93) vs N1.5 87% — [HF LeRobot blog](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot) (via search summary); a Perceptron comparison cites 97.0% for N1.7 and 96.9% for pi0.5 — [ai2.work](https://ai2.work/blog/perceptron-isaac-0-5-edges-out-nvidia-gr00t-and-pi-0-5-on-libero) (secondary, discrepancy unresolved)
- GR00T 2: one DiT denoises future video and robot actions jointly; inputs recent video, instruction, robot state; executes an action chunk then repeats; small per-robot heads convert plan to motor commands; video 320x224 over head and both wrist cameras — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- GR00T 2 Turbo: smaller joint video-action transformer (2B), higher-compression video encoder, same action horizon; 124 ms/prediction on Jetson Thor, 38 ms on RTX 5090; plans 1.6 s chunks about once per second; SONIC whole-body controller at 50 Hz on G1; optimisations cut median latency 66-86% vs eager BF16 — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- Agentic layer: AI agents use GR00T 2's API/tools to chain skills and recover from errors; "visual steering" (marks on images) as a planner interface — [Robot Report](https://www.therobotreport.com/nvidia-launches-gr00t-2-isaac-lab-3-0-corl/); [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- Predecessor research (DreamZero, arXiv 2602.15922): WAM on a pretrained video diffusion backbone, 14B autoregressive, ~7 Hz (~150 ms/chunk) after 38x speedup; AgiBot seen tasks 62.2% task progress vs best pretrained VLA 27.4%; unseen tasks 39.5%; DROID unseen verbs 49% vs 25-32% VLAs; video-only human/robot demos give >42% gain on unseen tasks from 10-20 min — [DreamZero](https://dreamzero0.github.io)

### Inferences
- The N1 "System 2 / System 1" split with a (mostly) frozen VLM no longer applies to GR00T 2 as described; communication is within one transformer over joint video+action tokens. The planner/agent role moved to an external agent plus visual marks.

### Gaps
- GR00T 2 (full) parameter count, backbone initialisation (Cosmos variant), frozen/unfrozen status, arm action representation: not stated on page; whitepaper unread. Per-change ablations for N1.6, N1.7 on language following: none found.

## 3. Training data, embodiments, parameter counts

### Takeaway
GR00T 2 scales human egocentric video (402,730 h, 100K public), UMI demos and teleop; the 2B figure is for Turbo only.

### Cited Findings
- GR00T 2 data: 402,730 h egocentric human video (Human-1M; 100,000 h public as Human-1M-100K), 10,184 h handheld UMI gripper demos over 2,897 tasks, 23,500 h robot teleoperation — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- Embodiments: YAM and Franka (parallel grippers); Sharpa and Wuji dexterous hands (Dexter = bimanual, two 22-DoF Sharpa hands); Unitree G1 (Turbo, via SONIC); Tianji with XHAND1 (DexUMI) — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- Other press: "about 2B parameters" is attributed to GR00T 2 by a search summary, but the project page ties 2B to Turbo; treat as conflated — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- Companion data/tools: EgoHand (HF nvidia/EgoHand-v1), Video-to-Data workflow, Isaac Lab 3.0 (Newton/PhysX, RL post-training library; up to 2.5x throughput vs Isaac Lab 2.3), Isaac Lab-Arena 0.4 preview, DexBench (RLWRLD, 15 tasks), VABAR (UC Berkeley, 6 tasks), SGS, AlphaNIST (>98% vs 81%, ~39x less GPU time) — [Robot Report](https://www.therobotreport.com/nvidia-launches-gr00t-2-isaac-lab-3-0-corl/)
- Cosmos 3 "expected to come soon" as of GTC; GR00T-Dreams/DreamGen not mentioned in GR00T 2 coverage — [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-expands-open-model-families-to-power-the-next-wave-of-agentic-physical-and-healthcare-ai)

### Gaps
- Whether neural trajectories / DreamGen data are used in GR00T 2; synthetic-data share.

## 4. NVIDIA's generalization claims and numbers

### Takeaway
Headline: few-shot adaptation on 50 unseen tasks (Dexter): 40% zero-shot, 71% 1-shot, 95% 5-shot. No explicit compositional (new object-target pairing) test appears; evaluation is task-novelty and human-to-robot transfer.

### Cited Findings (all NVIDIA-reported, from [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/) unless noted)
- 50 unseen tasks: 0-shot 40% (press: 40/50 tasks succeed at least once; note the page lists 40% while press says "completed 40 of 50 tasks at least once" — possible metric mix-up, flag), 1-shot 71%, 5-shot 95%; 5-shot used <3 min robot data and 30-min LoRA on one RTX 5090; 37/50 tasks succeed every trial
- Human-data scaling (72 Dexter tasks): video-rollout progress 64% -> 77%, real progress 50% -> 70% from 0% to 100% of human data
- Human -> robot without teleop (Dexter): filter-paper 65%; all four chemistry-lab tasks succeed (20 trials each)
- UMI -> YAM truck assembly 60%; GB300 memory seating 85%; toy truck 65%
- Real-world RL: three contact-rich tasks 40-76% -> 88-96%; cable insertion 19/20 (RL) vs 12/20 (SFT); chargers 20/20 vs 2/20
- Agent control (YAM, 100 trials): toy plane 98, tractor 97, car 97; without agent 71-84; SFT alone 23-69
- Visual steering: YAM short-horizon 40% -> 92%, Dexter 30% -> 70%; long-horizon trash sorting 17/20 with marks vs 0 without; board disassembly 3/10
- G1 Turbo: popcorn 48/48; table clearing 93%; sim->real OOD 6.7% -> 63.3% with sim data, GR00T 1.7 = 40.0%
- Sim fidelity (14 policies, 25-task YAM): success Pearson 0.92 / Spearman 0.89
- Leaderboards: top three in VLA and WAM categories on RoboCasa365, RoboLab, RoboDojo, MolmoSpaces — [Robot Report](https://www.therobotreport.com/nvidia-launches-gr00t-2-isaac-lab-3-0-corl/); GTC claims: >2x success of leading VLAs on new tasks in new environments, #1 on MolmoSpaces and RoboArena — [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-expands-open-model-families-to-power-the-next-wave-of-agentic-physical-and-healthcare-ai)
- N1.5 language following 46.6% -> 93.3% (see Q2)

### Gaps
- No LIBERO, SimplerEnv, or RoboCasa numeric scores for GR00T 2 found; no language-following metric for GR00T 2; leaderboard ranks lack numbers; N1.6 benchmark numbers not on its page.

## 5. Independent evaluations and criticism

### Takeaway
I found no independent evaluation of GR00T 2, and none of N1.x on LIBERO-Plus/PRO/RoboArena with numbers.

### Cited Findings
- LIBERO-Plus (CVPR 2026, arXiv 2510.13626): VLAs drop from ~95% to <30% under modest perturbations, most sensitive to camera viewpoint and initial state, largely insensitive to language; GR00T scores not confirmed in what I saw — [CVPR page](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)
- Commentary says LIBERO is saturated and LIBERO-PRO argues standard configs inflate results — [ai2.work](https://ai2.work/blog/perceptron-isaac-0-5-edges-out-nvidia-gr00t-and-pi-0-5-on-libero) (secondary)
- Press articles on GR00T 2 are repackaged NVIDIA claims; one states no hands-on testing — [dev.to](https://dev.to/paullovelace/nvidia-just-launched-gr00t-2-and-isaac-lab-30-one-demo-to-teach-a-robot-a-new-skill-cen)

### Gaps
- CoRL 2026 third-party papers using GR00T 2 as a baseline; failure analysis; RoboArena numbers; LIBERO-PRO tables.

## 6. Comparison with pi0.5 / later PI models, Gemini Robotics, Helix, open VLAs

### Takeaway
GR00T 2's page does not benchmark against external models. Only secondary numbers exist: N1.7 vs pi0.5 on LIBERO.

### Cited Findings
- GR00T 2 page compares only to its own ablations, Turbo, and GR00T 1.7 — [GR00T 2 page](https://research.nvidia.com/labs/gear/gr00t-2/)
- LIBERO avg: N1.7 97.0% vs pi0.5 96.9% vs Perceptron Isaac 0.5 97.2% (vendor-reported, secondary) — [ai2.work](https://ai2.work/blog/perceptron-isaac-0-5-edges-out-nvidia-gr00t-and-pi-0-5-on-libero)
- DreamZero vs pretrained VLAs on AgiBot/DROID (numbers in Q2) — [DreamZero](https://dreamzero0.github.io)
- NVIDIA claim ">2x leading VLAs" — [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-expands-open-model-families-to-power-the-next-wave-of-agentic-physical-and-healthcare-ai)

### Gaps
- No found comparison with Gemini Robotics, Figure Helix, or later PI models.

# CoRL 2026 learning paradigms beyond VLA architecture: world models, RL, data, adaptation

Scope note: ~17 search/fetch calls. Mostly abstract-level facts. CoRL 2026 was held 9-12 Nov 2026 in Austin, TX; 2,094 active submissions, 687 accepted (~33%) ([Chalvatzaki on X](https://x.com/GeorgiaChal/status/2095935964786954746); [search summary of workshop pages](https://www.corl.org/program/workshops)). Venue labels: "CoRL26 oral/spotlight" = listed on alphaxiv CoRL 2026 oral/spotlight pages ([oral](https://www.alphaxiv.org/conferences/corl/2026/oral), [spotlight](https://www.alphaxiv.org/conferences/corl/2026/spotlight)); "CoRL26 (listed)" = on the third-party list [hongsong-wang.github.io/CoRL2026](https://hongsong-wang.github.io/CoRL2026/) (only first ~40 of 687 read); "arXiv only" = venue NOT confirmed. Oral/spotlight pages were truncated (20 each visible), so the lists are partial. No CoRL 2026 main-conference best paper winner was found (the conference was still running); only workshop award announcements are planned. alphaxiv summaries are paraphrases, not verified against papers.

## Q1. Which CoRL 2026 papers were award winners, orals, or widely discussed?

### Takeaway
No main-track award winner found as of 2026-11-10 (awards likely announced at/after the event). Visible orals include DreamZero, pi0.7, RoboTTT, EgoScale, RL Token, MolmoBot, Being-H0.5, GRAIL, Flow Policy Gradients.

### Cited Findings
- Visible orals (partial list, 20 shown): SuperMimic (2610.28480); World Action Models are Zero-shot Policies (2602.15922, NVIDIA); pi0.7 (2604.15483, Physical Intelligence); RoboTTT (2607.15275); EgoScale (2602.16710); RL Token (2604.23073); Flow Policy Gradients for Robot Control (2602.02481); Being-H0.5 (2601.12993); GRAIL (2606.05160); MolmoBot (2603.16861); FACTR 2 (2606.12406); Memory-augmented Action-conditioned World Model (2606.18960); RoboNaldo; OpenDexGrasp; What Are We Actually Benchmarking in Robot Manipulation? (2606.04233); PACE; Play2Perfect (2606.26428); DISCO long-horizon rearrangement; TEMPO; TraCS — [alphaxiv orals](https://www.alphaxiv.org/conferences/corl/2026/oral)
- Visible spotlights (20 shown): MolmoAct2 (2605.02881); Ego2Robot (2608.02580); MEM multi-scale embodied memory (2603.03596); Playful Agentic Robot Learning (2606.19419); GaP (2607.05369); FlowDAgger (2607.08877); OpenHLM; ZETA cross-embodiment study (2609.02546); WM-Craftnet; Semantic Anchoring (2607.13597); ResSafe; Demystifying VLA failures in contact-rich tasks (2608.01402); PRTS (2604.27472); Never Too Late for Force (2607.14236); When Vision Overrides Language (2602.17659); PHUMA (2510.26236); BitVLA — [alphaxiv spotlights](https://www.alphaxiv.org/conferences/corl/2026/spotlight)
- Workshop best-paper awards exist (Scaling H2R workshop, NVIDIA-sponsored; LEAP workshop) but no winners found — [Scaling H2R](https://scaling-h2r-corl.github.io/); [LEAP](https://leap-workshop.github.io/)
- Relevant CoRL 2026 workshop: "Compositional Generalization for Real-World Robot Learning" — [site](https://compogen-robotics.github.io/corl2026/)
- RAI Institute roundup of its CoRL26 papers (HLO-AMP, SMPC sparse offline-to-online RL, ExpertGen, NovaPlan, workshop: Eucalyptus UMI, Koala Gripper, Sceniris, BRIDGE) — [RAI blog](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/)

### Inferences
- Orals skew toward NVIDIA (GEAR), Physical Intelligence, AI2, Berkeley/Stanford.

### Gaps
- Official best-paper/award list; full oral/spotlight lists; OpenReview decisions not read.

## Q2. Group 1: world models / video generation for policy learning (evidence on generalization)

### Takeaway
Joint video+action "world action models" (WAMs) are the main 2026 story; DreamZero (CoRL26 oral) reports >2x generalization to new tasks/environments over SOTA VLAs and video-only cross-embodiment gains. Evidence for new *objects/combinations* specifically is thin in what I could verify.

### Cited Findings
- **DreamZero / World Action Models are Zero-shot Policies** (Seonghyeon Ye, Yunhao Ge, ... Joel Jang; NVIDIA; arXiv 2602.15922, 17 Feb 2026; CoRL26 oral). 14B autoregressive video-diffusion backbone predicts future frames and actions jointly. ">2x improvement in generalization to new tasks and environments" vs SOTA VLAs (real robot); video-only demos from other robots/humans give >42% relative gain on unseen tasks from 10-20 min of data; adapts to new embodiment with 30 min play data; 7 Hz closed loop — [arXiv](https://arxiv.org/abs/2602.15922). No separate unseen-object number seen.
- **GRAIL: Generating Humanoid Loco-Manipulation from World Models** (Tianyi Xie, Haotian Zhang, Linxi Fan; arXiv 2606.05160; CoRL26 oral). Generated interactions grounded in 3D scenes as virtual demos; 84% and 90% real-world success reported on third-party list — [oral list](https://www.alphaxiv.org/conferences/corl/2026/oral); [list](https://hongsong-wang.github.io/CoRL2026/)
- **SuperMimic** (Liao, Shao, Abbeel; 2610.28480; CoRL26 oral): differentiable learned world model for humanoid motion tracking; ">1000x fewer GPU-hours" (list claim, unverified) — [oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- **Memory-augmented Action-conditioned World Model** (Zheng, Yu, Peng; 2606.18960; CoRL26 oral): geometry-aware memory retains occluded objects in long rollouts — [oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- **NovaPlan** (CoRL26 per RAI): VLM verifies video-generation rollouts for zero-shot long-horizon planning; no numbers — [RAI](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/)
- Other CoRL26 (listed) world-model items, no numbers: MoRE (multi-physics MoE world model), Ctrl-Mem, Chunk-wise Model-based Policy Optimization (2610.14564), "Targeting World Models to Compromise Robot Learning Pipelines" (2606.09499); also LaWAM, ImageWAM (summarized by search tool, unverified) — [list](https://hongsong-wang.github.io/CoRL2026/); [alphaxiv](https://www.alphaxiv.org/conferences/corl/2026)
- arXiv-only (venue unconfirmed): RISE self-improving policy with compositional world model (2602.11075); WorldScape Policy 2.0 (2607.18840); Flex-pi (2608.10860); Rolling-WAM (2609.30247); WAM survey (2609.16074); tutorial (2607.00836); VLA-RFT world-simulator RL (2510.00406) — [search results](https://www.alphaxiv.org/conferences/corl/2026)
- Ctrl-World (controllable multi-view world model for policy eval/improvement) is ICLR 2026, not CoRL — per search summary, unverified.

### Inferences
- DreamZero's ">2x on new tasks/environments" is the strongest world-model-generalization claim seen; it is an abstract claim without per-axis breakdown.

### Gaps
- Per-axis (object/instruction/combination) breakdowns; world-model co-training ablations; policy-evaluation-in-world-model papers at CoRL26 not located.

## Q3. Group 2: RL on top of pretrained policies; robot-hours

### Takeaway
RL Token (PI, oral) and Learning While Deploying (AgiBot, 95% avg success) are the headline real-world RL-on-VLA results; robot-hours mostly not extracted.

### Cited Findings
- **RL Token** (Charles Xu, Springenberg, Equi, Amin, Esmail, Levine, Ke; Physical Intelligence; arXiv 2604.23073; CoRL26 oral). Exposes a compact "RL token" readout of the VLA; small actor-critic on it, anchored to VLA. Four real tasks (screw, zip tie, charger, Ethernet insertion): up to 3x speed on hardest segment; success rises "within minutes to a few hours of practice"; exceeds teleop speed on some tasks — [arXiv](https://arxiv.org/abs/2604.23073). Exact hours/success % not seen.
- **Learning While Deploying (LWD)** (Yi Wang + 15; arXiv 2605.00416, 1 May 2026; CoRL26 listing unconfirmed—appeared on hongsong-wang CoRL page via search summary). Fleet-scale offline-to-online RL on a pi0.5 flow VLA with DIVL value learning and QAM extraction; single policy reaches 95% average success over 8 manipulation tasks (per review) — [arXiv](https://arxiv.org/pdf/2605.00416); [project](https://finch.agibot.com/research/lwd)
- **RL-Co / RLinf-Co** (arXiv 2602.12628): sim-real co-training then RL in sim with real supervised loss; +24% real success on OpenVLA, +20% on pi0.5 over 4 tabletop tasks (search summary; CoRL26 venue per search summary, unverified) — [arXiv](https://arxiv.org/html/2602.12628v4)
- **Flow Policy Gradients for Robot Control** (Yi, Choi, Abbeel; 2602.02481; CoRL26 oral) — likelihood-free policy gradient for flow policies, sim-to-humanoid — [oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- **Play2Perfect** (Lum, Kedia, Bohg; 2606.26428; oral): list reports 33x sample efficiency, 60% on tight insertions — [list](https://hongsong-wang.github.io/CoRL2026/)
- **PRTS** (2604.27472; spotlight): VLA pretraining as goal-conditioned RL — [spotlight](https://www.alphaxiv.org/conferences/corl/2026/spotlight)
- **ResSafe** (2609.15988; spotlight): residual RL safety filter for humanoids — same.
- **ExpertGen** (RAI, CoRL26): RL optimizes diffusion-policy initial noise; up to 85% long-horizon, 91.1% industrial assembly — [RAI](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/)
- **FlowDAgger** (Murray, Chen, Mees; 2607.08877; spotlight): human corrections adapt frozen generative policy in latent space — spotlight list.
- arXiv-only: Real-Time EXPO-FT (2609.18207); FlowPRO (reward-free offline RL for flow VLAs); LifeLong-RFT (RSS 2026) — [search](https://arxiv.org/abs/2609.18207)

### Gaps
- Robot-hour totals for LWD/RL Token; reward-model papers not found.

## Q4. Group 3: data scaling and diversity (incl. combinatorial coverage)

### Takeaway
Egocentric human data at scale (EgoScale, HumanScale) and big synthetic sim data (MolmoBot) are the 2026 pillars. The closest study on combinatorial coverage is "Scale Up Strategically" (Wong lab/NVIDIA, arXiv only, instruction-factor level) plus the CoRL26 "Diagnosing Compositional Generalization" paper.

### Cited Findings
- **EgoScale** (Ruijie Zheng, Dantong Niu, ... Linxi Fan; arXiv 2602.16710, 18 Feb 2026; CoRL26 oral): 20,854 h action-labeled egocentric video (>20x prior); log-linear scaling of validation loss with human data; +54% avg success vs no pretraining on 22-DoF hand; one-shot adaptation claimed, no figure seen — [arXiv](https://arxiv.org/abs/2602.16710)
- **MolmoBot** (Deshpande, Guru, Krishna; 2603.16861; oral): 1.7M simulated trajectories; zero-shot real 79.2% vs 39.2% for pi0.5 (list claim) — [list](https://hongsong-wang.github.io/CoRL2026/)
- **Being-H0.5** (Luo, Wang, Lu; 2601.12993; oral): 35,000 h human + 30 robot embodiments; SOTA on LIBERO and RoboCasa (claimed) — oral/list.
- **Ego2Robot** (2608.02580; spotlight): egocentric video to robot training data; **PHUMA** (2510.26236; 73 h humanoid locomotion, spotlight); **FACTR 2** (2606.12406, >14% task progress); **OpenHLM** (2606.22174); **ZETA** zero-shot cross-embodiment study (2609.02546); **What Are We Actually Benchmarking** (2606.04233): LIBERO and CALVIN fail more validity diagnostics than RoboCasa and RoboTwin 2.0 — oral/spotlight lists.
- Hardware (RAI workshop papers): Eucalyptus force-aware UMI; Koala Gripper; BRIDGE (handheld + few teleop, up to +36.7%) — [RAI](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/)
- **HumanScale** (Ma et al., PKU/MIT, arXiv 2606.20521, 18 Jun 2026; venue not stated): at matched data, egocentric pretraining gives 24% lower val loss, +52.5% in-distribution and +90% OOD real success vs real-robot data — [arXiv](https://arxiv.org/abs/2606.20521). Also HumanNet (~1M h, 2605.06747), EgoVerse (2604.07607), HumanEgo (2605.24934), ACE-Ego-0 (2606.17200) — arXiv only, [search](https://arxiv.org/pdf/2606.17200)
- **Scale Up Strategically: Compositional Generalization via Bias-Aware Evaluation and Data Collection** (Yu Qi, Zhang Ye, ... Jonathan Tremblay, Lawson L. S. Wong; arXiv 2607.21582, 23 Jul 2026; arXiv page lists no venue, a search result tagged it CoRL26 submission). Defines instruction factor bias (Factor Dominance Rate/Hierarchy); across six foundation policies color dominates, verb and size least grounded; shifting a fixed demo budget toward under-grounded factors beats baselines in sim and real "using half the demonstrations" — [arXiv](https://arxiv.org/abs/2607.21582). It is about instruction-factor budget allocation, not explicitly object-skill combination coverage.
- **Diagnosing Compositional Generalization in Sequential Robot Tasks** (arXiv 2607.29687; CoRL26 listed): OOD success 0.4% to 54.7% (what produced the rise not read); includes one-demonstration finetuning — [list](https://hongsong-wang.github.io/CoRL2026/); [arXiv](https://arxiv.org/pdf/2607.29687)
- Pre-2026 precursor: Gao et al., Efficient Data Collection via Compositional Generalization, 2403.05110 — [link](https://arxiv.org/pdf/2403.05110)
- Related: ATOM-Bench (atomic skill vs composition failures) cited in search summary, unverified.

### Gaps
- No study found directly varying number of object-skill/object-target combinations covered; 2607.29687 and 2607.21582 need full reading. Axis ranking (objects vs scenes vs tasks vs embodiments) not established.

## Q5. Group 4: few-shot, in-context, continual adaptation

### Takeaway
Long-context/test-time-training (RoboTTT) and human-video pretraining (EgoScale) claim one-shot adaptation without published numbers in abstracts. Pretrained VLAs forget little under replay in LIBERO, but real-world and perturbation tests are harsher.

### Cited Findings
- **RoboTTT** (Yunfan Jiang, Chebotar, Zheng, ... Fei-Fei, Zhu, Fan; arXiv 2607.15275, 16 Jul 2026; CoRL26 oral): fast-weight test-time training in VLA, 8K-step visuomotor context; +87% vs single-step context baseline; 8K beats 1K-pretrained by 62%; completes a 5-min 10-stage assembly no baseline completes; one-shot imitation from a single human video claimed, no number — [arXiv](https://arxiv.org/abs/2607.15275)
- **pi0.7** (Bo Ai, Amin, Aniceto ...; PI; 2604.15483; oral): steerable generalist, claims new tasks without task-specific training — [oral](https://www.alphaxiv.org/conferences/corl/2026/oral)
- **FlowDAgger** (spotlight, 2607.08877); **Semantic Anchoring** (2607.13597; +18.7% in-dist, +21.5% OOD per list); **When Vision Overrides Language** (2602.17659; inference-time guidance) — spotlight/list.
- **MEM** (Torne, Pertsch, Finn; 2603.03596; spotlight): ~15-min memory — spotlight list.
- **Playful Agentic Robot Learning** (2606.19419; +20.6 / +17.0 over CaP-Agent0) — list.
- LeCAR lab FADA (few-shot, finetune inverse-dynamics only) and WANDA (one real demo to synthetic data) listed as CoRL26 — [LeCAR](https://lecar-lab.github.io/publications.html) (unverified details)
- arXiv-only: RoboICL (2609.34261; +20-27 progress points over zero-shot GPT-6 Astra, one demo/task, 30 RoboDojo tasks); counter-result 2609.24170 (one-shot demos did not beat zero-shot); Reflective VLA (2606.25215); VLBiMan++ (2609.14310); AC-VLA (2607.15714) — metadata looked odd, verify.
- **Continual**: "Pretrained VLAs are Surprisingly Resistant to Forgetting" (Liu, Kim, Liu, Liu, Zhu; arXiv 2603.03818, Mar 2026; no venue): pi0 and GR00T N1.5 forget far less than small BC policies; experience replay sometimes zero forgetting with small buffers; forgotten skills recover quickly — [arXiv](https://arxiv.org/abs/2603.03818). CRL-VLA (2602.03445; forgetting F=0.03 in one table) — [alphaxiv](https://www.alphaxiv.org/abs/2602.03445). 2605.26820: real-world 10-task benchmark, naive sequential finetuning forgets catastrophically — [arXiv](https://arxiv.org/html/2605.26820v2). LIBERO-PRO: models >94% on LIBERO collapse to ~0% under pose/object swaps (search summary). LifeLong-RFT (RSS 2026) — [RSS](https://www.roboticsproceedings.org/rss22/p086.pdf). None of these confirmed as CoRL 2026.

### Gaps
- Verified few-shot numbers for VLA fine-tuning vs in-context vs retrieval; LIBERO continual SOTA table; LoRA/adapter/merging papers not found.

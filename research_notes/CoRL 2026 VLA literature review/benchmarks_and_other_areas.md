# CoRL 2026: evaluation/benchmarks/reproducibility (Part A) and non-VLA research areas (Part B)

Research depth note: about 14 tool calls. Most sources are abstracts, aggregator pages (alphaXiv) and workshop pages. No official CoRL 2026 accepted-paper list was fetched; alphaXiv says 687 accepted papers (593 indexed there). Numbers marked (secondary) came from search-engine summaries of the source, not from reading the paper's tables.

## Which benchmarks did CoRL 2026 papers most often report, and which are considered saturated or misleading?

### Takeaway
The oral "What Are We Actually Benchmarking in Robot Manipulation?" (CoRL 2026 oral) audits LIBERO, CALVIN, SimplerEnv, RoboCasa and RoboTwin 2.0 and finds LIBERO and CALVIN fail multiple diagnostics (saturated, shortcut-solvable). RoboCasa and RoboTwin 2.0 pass more. No systematic count of benchmark usage across CoRL 2026 papers was found; only anecdotal usage (e.g. Being-H0.5 reports LIBERO and RoboCasa).

### Cited Findings
- Audit paper is a CoRL 2026 oral: Tianchong Jiang, Xiangshan Tan, Samuel Wheeler (TTIC, UChicago) et al.; abstract: LIBERO and CALVIN fail multiple diagnostics; a 0.09B probe with no language encoder scores at or near reported LIBERO SOTA — [alphaXiv CoRL 2026 orals](https://www.alphaxiv.org/conferences/corl/2026/oral); arXiv 2606.04233 (posted 2 Jun 2026), authors incl. Luzhe Sun, Tewodros Ayalew, Matthew Walter — [arXiv](https://arxiv.org/pdf/2606.04233); artifacts [project page](https://ripl.github.io/manipulation_benchmark_audit/) (listed "coming soon" at search time).
- Four diagnostics: shortcut solvability, statistical significance, creeping overfitting (distribution vs sample), data-source dependence — [arXiv HTML](https://arxiv.org/html/2606.04233).
- LIBERO: ~0.09B DINO+MLP probe scores 99.0 Spatial / 100.0 Object / 98.8 Goal / 92.4 Long vs best reported 99.8 / 100.0 / 99.5 / 98.8; only 19.8% of SOTA claims provably significant; redrawing initial states moves success <1 point; 79 arXiv papers reported LIBERO in March 2026 alone — [arXiv HTML](https://arxiv.org/html/2606.04233) and [arXiv search summary](https://arxiv.org/pdf/2606.04233).
- CALVIN: probe gets 3.12 (D->D), 3.24 (ABC->D), 3.87 (ABCD->D) average tasks (of 5) vs best ~4.80 ABCD->D; resampling block poses within the training range drops ABC->D ATC for X-VLA 4.17 -> 3.14 (-1.03), GR-1 -0.75, RoboFlamingo -0.50 — [arXiv HTML](https://arxiv.org/html/2606.04233).
- SimplerEnv: probe 0.0% on WidowX vs 95.8% X-VLA (no shortcut), but only 19.7% of SOTA claims provably significant; stack-on-support-block variant drops X-VLA-WidowX 28.5 points, Dexbotic 17.4; reversing the language instruction drops CogACT-Base 11.1; small DINOv2-S policies on 120 scripted sim demos/task reach 94.8% (91/96) vs X-VLA 95.8% (92/96) (data-source dependence) — [arXiv HTML](https://arxiv.org/html/2606.04233).
- RoboCasa: probe 18.8% vs GR00T-N1 49.6% and best reported 79.2%; 53.3% of SOTA claims provably significant. RoboTwin 2.0: probe 60.4% clean / 59.4% randomized vs X-VLA 72.8%, best ~96%; 73.7% of SOTA claims significant (highest of five) — [arXiv HTML](https://arxiv.org/html/2606.04233).
- Being-H0.5 (CoRL 2026 oral) reports SOTA on LIBERO and RoboCasa plus five real robots, 35,000-hour corpus, 30 robot embodiments — [alphaXiv orals](https://www.alphaxiv.org/conferences/corl/2026/oral) (benchmark numbers not given on page; unverified).
- RoboCasa365 (ICLR 2026, not CoRL): 365 tasks, 2500 kitchens, 600+ h human demos, 1600 h synthetic; aggregator leaderboard (sampled 2026-05-27, unverified): RLDX-1 33.2%, GR00T N1.5 23.9, GR00T N1.6 21.9, GigaWorld-Policy 0.1 20.7, pi0.5 16.9, pi0 14.8, Diffusion Policy 6.1; Composite-Unseen best 5.4% — [arXiv](https://arxiv.org/pdf/2603.04356), [benchmarklist](https://benchmarklist.com/benchmarks/robocasa365/).
- RoboDojo (arXiv 2607.04434; CoRL link not established): 42 sim tasks on five axes (Generalization, Memory, Long-Horizon, Precision, Open), 18 real tasks on ARX X5/Piper/Piper X, 30 policies on leaderboard; lowest scores 8.8% sim / 12.8% real, human experts 76.03% sim (secondary); sim/real leaderboard alignment only partial — [arXiv](https://arxiv.org/abs/2607.04434), [leaderboard](https://robodojo-benchmark.com), [code](https://github.com/RoboDojo-Benchmark/RoboDojo). Author list conflicts between sources.
- CoRL 2026 workshop (12 Nov 2026, Austin) "From GPU-Accelerated Simulation to Scalable and Generalizable Real Robot Policy Learning" runs a 3-track challenge suite: BEHAVIOR long-horizon household (1,000 activities, 50 scenes, 10,000 objects, Isaac Sim, 20,000 demos), RoboEval diagnostic bimanual (8 MuJoCo tasks; score = outcome 40%, safety/stability 30%, efficiency 20%, bimanual coordination 10%; 3,000+ VR demos as LeRobot dataset; HF leaderboard), RoboChess sim-to-real — [workshop page](https://corl2026-realworld-robot-learning.github.io/).
- AIRoA CoRL 2026 workshop "The UMI Arena": shared real-robot platform with public leaderboard; submission deadline late Oct 2026 (passed) — [AIRoA](https://airoa.org/updates/20260821/649/) (details from search snippet only).
- RoboArena (CoRL 2025, not 2026): distributed real-world pairwise evaluation on DROID platform, live benchmark through Dec 2026 — [RoboArena](https://robo-arena.github.io/) (snippet only).
- XPolicyLab (arXiv 2608.09892): unified standard/open ecosystem for policy evaluation and deployment — [arXiv](https://arxiv.org/pdf/2608.09892) (title only; unread).

### Inferences
- LIBERO and CALVIN should be treated as saturated/shortcut-solvable for ranking claims; RoboCasa(365), RoboTwin 2.0, BEHAVIOR and RoboEval are where headroom remains.

### Gaps
- No frequency count of benchmarks across the 687 CoRL 2026 papers; official accepted list and OpenReview not accessed.
- No fetch of RoboArena 2026 results, SimplerEnv 2026 updates, ManiSkill, or LIBERO-PRO beyond the abstract summary below.

## Which benchmark best isolates compositional generalization, and what do leading VLAs score?

### Takeaway
No single canonical benchmark; candidates are RoboCasa365 "Composite-Unseen" split, "Robust Skills, Brittle Grounding" (held-out object-location pairings), RoboCombine and ACG-Bench. All show near-collapse on unseen combinations even when parts are individually seen. Not CoRL 2026 papers as far as verified.

### Cited Findings
- "Robust Skills, Brittle Grounding" (arXiv 2602.24143, Feb 2026; venue not stated): custom ManiSkill GraspObject-v1, 5 YCB objects, Franka. Every object and location seen separately in training, pairings held out. SmolVLA seen pairings 44% success vs held-out 0% (reach 100% -> 2%); jitter ladder 90/37/41/2% (small/medium/large/full random); pi0.5 large 10%, full random 6%; 100k demos gave no gain under full randomization; one object raises success, so failure is target selection — [arXiv HTML](https://arxiv.org/html/2602.24143).
- RoboCombine (NeurIPS 2026 poster, not CoRL): one training setup 40.0% ordered success on covered combos vs 7.7% on recombination (secondary) — [NeurIPS](https://neurips.cc/virtual/2026/poster/139255).
- ACG-Bench / AE-VLA (arXiv 2610.06184): dual-arm, 23 task-condition pairs, 17 unseen compositions; proposed model 21.53% vs 2.94% single pi0.5 (secondary) — [OpenTrain](https://www.opentrain.ai/papers/arm-wise-compositional-generalization-in-dual-arm-vision-language-action-models--arxiv-2610.06184/).
- RoboHiMan (arXiv 2510.13149, Oct 2025): pi0 and pi0.5 moderate on seen long-horizon tasks, near-complete failure on unseen — [arXiv](https://arxiv.org/pdf/2510.13149) (secondary).
- SkillVLA (arXiv 2603.03836): 51% on unseen skill combinations where baselines near zero (secondary) — [arXiv](https://arxiv.org/html/2603.03836).
- LIBERO-Plus (CVPR 2026, not CoRL): pairwise perturbations show consistently negative interaction (combined shifts harder than individual effects), analysed only for OpenVLA-OFT — [arXiv v3](https://arxiv.org/html/2510.13626v3).
- RoboCasa365 Composite-Unseen: best 5.4% (aggregator, unverified) — [benchmarklist](https://benchmarklist.com/benchmarks/robocasa365/).
- AC-VLA (arXiv 2607.15714) uses LIBERO-OOD with 20 novel compositional tasks per suite — [Emergent Mind](https://www.emergentmind.com/papers/2607.15714) (snippet only).

### Inferences
- The held-out object-location pairing design (2602.24143) most cleanly isolates composition of seen parts (nothing else changes), but it is a small custom ManiSkill task and covers only SmolVLA and pi0.5. RoboCasa365 Composite-Unseen is the larger, broader option.

### Gaps
- No paper found that reports a cross-model table (OpenVLA-OFT, pi0.5, GR00T, X-VLA) on one compositional split; RoboCasa365 Composite-Unseen numbers are unverified aggregator values.

## What do 2026 papers say about the gap between 95%+ simulation scores and robustness under small changes?

### Takeaway
Large and consistent: LIBERO-Plus reports collapse from ~95% to under 30% for some models under modest perturbations, and LIBERO-PRO reports 0.0% under generalized settings; the CoRL 2026 audit shows models ignore language and that simple probes match SOTA.

### Cited Findings
- LIBERO-Plus (CVPR 2026): 10,030 tasks, 7 perturbation dimensions. Original/Camera/Robot-init/Language/Light/Background/Noise/Layout (%): OpenVLA-OFT 97.1/59.7/37.2/81.5/85.8/92.4/76.7/77.1; pi0 94.2/15.8/6.6/61.0/79.6/78.5/79.4/70.4; pi0-fast 85.5/66.4/24.8/63.3/73.0/67.7/75.8/70.3; UniVLA 95.2/4.3/50.3/71.8/59.1/80.0/25.3/34.3; WorldVLA 79.1/0.3/30.2/44.2/29.4/14.5/12.2/39.4; RIPT-VLA 97.5/58.3/36.7/80.1/87.9/90.4/73.8/76.5 — [arXiv v3](https://arxiv.org/html/2510.13626v3); CVPR listing [here](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html).
- Camera viewpoint and robot initial state most damaging; language drops small because models ignore language; fine-tuning OpenVLA-OFT on generated perturbation data reaches 79.5% total (possible train/test perturbation overlap; baseline-solved tasks were removed) — [arXiv v3](https://arxiv.org/html/2510.13626v3).
- LIBERO-PRO (arXiv 2510.03827, Zhou et al., Oct 2025): models >90% on standard LIBERO collapse to 0.0% under perturbed objects/initial states/instructions/environments (abstract via third-party site) — [mirror](https://arxiv.deeppaper.ai/papers/2510.03827v1).
- CALVIN pose resampling and SimplerEnv stack/language variants cause large drops (see first section) — [arXiv HTML](https://arxiv.org/html/2606.04233).
- RoboDojo: competitive-in-sim models underperform dramatically on real hardware (secondary) — [arXiv](https://arxiv.org/abs/2607.04434).
- MolmoBot (CoRL 2026 oral, AI2/UW): 1.7M synthetic trajectories, zero-shot sim-to-real, 79.2% real tabletop pick-and-place over 4 settings vs 39.2% for pi0.5 — [alphaXiv orals](https://www.alphaxiv.org/conferences/corl/2026/oral).

### Inferences
- The robustness axes that break models most (camera pose, initial state, target position) differ from those LIBERO varies, so a 95%+ score says little about them.

### Gaps
- LIBERO-PRO numbers are from a mirror abstract; not checked against the paper. Neither LIBERO-Plus nor LIBERO-PRO is a CoRL 2026 paper.

## Which evaluation tools are open and runnable by an academic lab with A40 GPUs and no real robot?

### Takeaway
LIBERO-Plus, RoboCasa365, RoboDojo (sim half), BEHAVIOR and RoboEval (MuJoCo, HF leaderboard) are open sim benchmarks; the audit's reference implementations are released. World-model evaluators are heavy (RoboWorld ~100 H100 GPU-hours for 8 policies).

### Cited Findings
- RoboDojo code and leaderboard: [GitHub](https://github.com/RoboDojo-Benchmark/RoboDojo), [site](https://robodojo-benchmark.com).
- RoboEval challenge portal: [HF space](https://huggingface.co/spaces/helen9975/roboeval-challenge), [challenge](https://robo-eval.github.io/challenge.html); BEHAVIOR: [behavior.stanford.edu](https://behavior.stanford.edu); both listed on the [workshop page](https://corl2026-realworld-robot-learning.github.io/).
- Audit reference implementations: [project page](https://ripl.github.io/manipulation_benchmark_audit/).
- Custom ManiSkill task ladder from 2602.24143 — [arXiv](https://arxiv.org/html/2602.24143).
- World-model evaluators: WorldEval (0.79 Pearson with real, secondary) [arXiv](https://arxiv.org/abs/2505.19017); Mem-World (arXiv 2606.18960; pi0 and pi0.5, r=0.97 over five tasks; likely the CoRL 2026 oral "Memory-augmented Action-conditioned World Model", unconfirmed link) [arXiv](https://arxiv.org/pdf/2606.18960); RoboWorld (arXiv 2607.01060; replays RoboArena episodes, VLM judge, ~100 H100 GPU-hours for 8 policies) [arXiv](https://arxiv.org/pdf/2607.01060).
- Real-to-sim evaluation with imperfect simulators: [arXiv 2510.04354](https://arxiv.org/pdf/2510.04354); scalable benchmarking via real-to-sim, ICLR 2026: [paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/2aa3da3c1463ee2cdaaee94be4f8ba3f-Paper-Conference.pdf) (titles only).

### Inferences
- A40-friendly (no real robot): LIBERO-Plus, RoboCasa365, RoboEval, ManiSkill custom ladders. GPU memory/time for each was not checked.

### Gaps
- No verified hardware requirements; licenses and install status not checked. SimplerEnv, ManiSkill and CALVIN 2026 updates not researched.

## Statistical practice and evaluation pitfalls (2026)

### Takeaway
The CoRL 2026 audit is the main 2026 statistics paper: aggregate success rates rarely prove significance, and it prescribes per-instance reporting and paired tests.

### Cited Findings
- Recommendations: report per-instance outcomes; paired tests for shared instances, stratified two-sample otherwise; state minimum meaningful gap; size test sets to claimed gains; Wilson/Newcombe intervals; treat closed-loop nondeterminism as uncertainty — [arXiv HTML](https://arxiv.org/html/2606.04233).
- Proportion of SOTA claims provably significant: LIBERO 19.8%, SimplerEnv 19.7%, RoboCasa 53.3%, RoboTwin 2.0 73.7% — same source.
- Controller choice substantially changes success of UMI-style policies (workshop paper) — [RAI Institute round-up](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/).
- FailBench (arXiv 2609.03611): reliability of VLMs as task-success judges (title only) — [arXiv](https://arxiv.org/pdf/2609.03611). Relevant because world-model evaluators use VLM judges.
- Why Does Action Chunking Improve BC? (CoRL 2026, Lazzati, Stachowicz, Levine) — [alphaXiv](https://www.alphaxiv.org/conferences/corl/2026).

### Gaps
- No other 2026 statistics-specific paper found.

## Standout CoRL 2026 results by area (Part B)

### Takeaway
Papers below are verified only by title/abstract on alphaXiv or workshop pages. Per-area depth (4-8 papers each) was not reached for navigation, safety, hardware, and bimanual; coverage is partial.

### Cited Findings
Humanoids / whole-body control
- SuperMimic (oral), Qiayuan Liao, Yiyang Shao, Pieter Abbeel: learned dynamics model matches large motion libraries with >1,000x fewer GPU-hours — [alphaXiv](https://www.alphaxiv.org/conferences/corl/2026/oral).
- GRAIL (oral), Tianyi Xie, Haotian Zhang, Linxi Fan: humanoid loco-manipulation from world-model-generated virtual demos; real pickup and stair climbing — same.
- Flow Policy Gradients for Robot Control (oral), Brent Yi, Hongsuk Choi, Pieter Abbeel: likelihood-free flow policy RL; transfers to physical humanoids — same.
- RoboNaldo (oral), Yichao Zhong, Yidan Lu, Hongyang Li: humanoid soccer shooting; sim error -50%, shot velocity 2.96x; Unitree G1 target error 0.73 m (free kick), 0.86 m (moving ball) at 3 m, ball speed 13.10 m/s — same.
- HLO-AMP (hierarchical loco-manipulation, G1 box pick-and-place) and SMPC-demos sparse offline-to-online RL (Spot arm and G1) — [RAI Institute](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/) (venue as presented there: publications; authors not listed).
- HuG (whole-body affordance-aware grasping), Humanoid Badminton listed as CoRL 2026 — [search summary via awesome list](https://github.com/YanjieZe/awesome-humanoid-robot-learning) (unverified).

Dexterous / bimanual
- EgoScale (oral), Ruijie Zheng, Dantong Niu, Linxi Fan: diverse egocentric human data cuts robot data needs; one-demo adaptation — [alphaXiv](https://www.alphaxiv.org/conferences/corl/2026/oral).
- Play2Perfect (oral), Tyler Lum, Kushal Kedia, Jeannette Bohg: dexterous play pretraining enables insertion, screwing, multi-part assembly without demos — same.
- OPENDEXGRASP (oral), Jiyao Zhang, Junhan Wang, Hao Dong — same.
- Do as I Do, Bhawna Paliwal, Haritheja Etukuru, Pieter Abbeel; T-Rex, Dantong Niu, Zhuoyang Liu, Li Fei-Fei — [alphaXiv](https://www.alphaxiv.org/conferences/corl/2026). BiDexTool workshop lists 15 accepted papers: [site](https://bimanual-tool.github.io/).
- Touch2Trace [arXiv 2609.15921](https://arxiv.org/pdf/2609.15921) and Cartesian Hand [arXiv 2609.25696](https://arxiv.org/pdf/2609.25696) appeared in search; CoRL status unconfirmed.

Tactile / force
- FACTR 2 (oral), Steven Oh, Jason Liu, Ruslan Salakhutdinov: NEXT estimates external joint torques without force sensors (trains ~1 min from 10 min free-motion data); FIRST gains >14% task progress over prior force-aware policies — [alphaXiv orals](https://www.alphaxiv.org/conferences/corl/2026/oral).
- PACE (oral), Zhiyuan Chen et al.: separates steady contact from sudden collisions in chunked insertion policies — same.
- T-Rex (above); SlipSense, FR-VLA (He et al.) listed as CoRL 2026 (search summary, unverified).

Mobile manipulation / navigation
- Hierarchical Planning with Latent World Models, Wancong Zhang, Basile Terver, Yann LeCun (Meta, NYU) — [alphaXiv](https://www.alphaxiv.org/conferences/corl/2026).
- A Balanced Data Diet, Octi Zhang, Mateo Guaman Castro, Abhishek Gupta (terrain navigation, assembly) — same.
- Long-Horizon Multi-Object Rearrangement via Distilled Tree Search, Kallol Saha, Yining Hong, Yejin Choi (oral) — orals page.
- "Cracking Mobile Manipulation" (arXiv 2610.19600): failures split into task planning, navigation/base positioning, local manipulation; CoRL status unconfirmed — [arXiv](https://arxiv.org/html/2610.19600).

Sim-to-real
- MolmoBot (oral), Abhay Deshpande, Maya Guru, Ranjay Krishna: 79.2% real vs 39.2% pi0.5 — orals page.
- ExpertGen (RAI): up to 85% long-horizon, 91.1% industrial assembly in sim, distilled to hardware — [RAI](https://rai-inst.com/resources/blog/corl-2026-publication-and-workshop-round-up/).

Safety / failure / memory
- RoboTTT (oral): long-context policies recover from mistakes — orals page. MEM (spotlight), Marcel Torne, Karl Pertsch, Chelsea Finn: multi-scale memory up to 15 min. ENPIRE, Wenli Xiao et al. Memory Anchors for Continual Robot Learning, Max Du, Zhanyi Sun, Shuran Song — [alphaXiv](https://www.alphaxiv.org/conferences/corl/2026).
- ProTracer (arXiv 2609.21369) proprioception-guided failure diagnosis — [arXiv](https://arxiv.org/pdf/2609.21369); CoRL status unconfirmed.

Hardware / low-cost
- MolmoAct2 (spotlight), Haoquan Fang, Jiafei Duan, Ranjay Krishna: open policies for low-to-medium-cost platforms (825 GitHub stars at fetch) — alphaXiv.
- Eucalyptus (multimodal UMI), Koala Gripper (co-designed gripper/data capture) — RAI workshop papers, link above.

Other CoRL 2026 headline VLA-adjacent papers: World Action Models are Zero-shot Policies (NVIDIA, oral), pi0.7 (Physical Intelligence, oral), RL Token (Xu, Springenberg, Levine, oral), Being-H0.5 — orals page.

### Inferences
- Visible theme: humanoid work is dominated by RL-from-sim and world-model-generated data; manipulation by human-video (egocentric) data and world-action models.

### Gaps
- Four-to-eight papers per area with numbers not reached for bimanual, navigation, safety, hardware; ranking by "most discussed" is only proxied by alphaXiv engagement counts (unlabeled). No trip reports, X/Twitter threads, or blog reviews were found. The official accepted-paper list and OpenReview were not fetched.

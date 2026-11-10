# Follow-up: identity hiding, phase-switched conditioning, structured interfaces (2026 robot learning)

Research date 2026-11-10. All content below was obtained via WebFetch of the cited URL. "Read" levels are stated per paper. Venue is "not stated" wherever the page gave none (the CoRL 2026 label for SlotFlow in the brief was NOT confirmed by the page).

## Key question 1: Has anyone hidden non-active object identity from a VLA's motor pathway and measured compositional generalization?

### Takeaway
No paper found hides non-active identity while leaving a flat silhouette/geometry, and none measures it with a role/phase split. The nearest are distractor removal (CGVD inpainting, LiLo-VLA masking, OBEYED-VLA, TAG) and the S2 learned token-budget gate. LiLo-VLA is the closest on composition: masking non-target objects plus unseen skill orderings (unseen-order tests, 69% vs pi0.5 28%), but its ablation of masking is confounded with other modules.

### Cited Findings

**SlotFlow (arXiv 2609.24155v2)**: full text read (all tables incl. appendix; figures not viewed). Title "Object-Centric Conditioning for Visuomotor Flow Matching". Authors Jijie Li, Xu Yang, Junhong Zou, Chunhai Zhao, Chaoyang Zhao, Zhen Lei, Xiangyu Zhu (CASIA/UCAS). Dated 29 Sep 2026; venue NOT stated on page (CoRL claim unverified). — [arXiv HTML](https://arxiv.org/html/2609.24155)
- Mechanism: Cascaded Foveated Module, frozen DINOv3-S, 2 slots; coarse stage locates object, fine stage processes a crop. "What" = semantic slot feature; "where" = 2D image-plane centroid of fine mask as positional embedding. Policy (A2A flow matching, starts from encoded action history, 6 steps) receives global image features + proprio + (slot + geometric embedding). Global image features are still passed, so non-target identity is NOT hidden from the policy. — [arXiv HTML](https://arxiv.org/html/2609.24155)
- Table 1 Close Box (L0/L1/L2/L3/Pos): SlotFlow 100/50/52/52/80; A2A 100/12/18/10/70; VITA 100/4/2/4/74; FM-DiT 98/2/2/2/60. Real UR3 Close Drawer (20 trials; ID/tablecloth shift/pos shift): SlotFlow 100/85/75 vs A2A 100/55/40. Pick Cube (A4): SlotFlow 76/20/26/24/38 vs A2A 64/12/16/16/26. Push Cube (A5): SlotFlow 94/72/74/74/86 vs A2A 92/72/72/74/80. Single training seed (42); 50 episodes per setting; variance across trainings not characterized. (Meaning of Levels 1-3 not extracted; unverified.) — [arXiv HTML](https://arxiv.org/html/2609.24155)
- Ablation Table 3 Close Box (L1/L2/L3/Pos): ResNet18 A2A 12/18/10/70; +semantic only 28/28/20/66; +semantic+spatial 50/52/52/80; Table A2: mask-pooled DINO 4/2/4/54, spatial-only 22/22/18/64, no-crop 26/24/30/74. Both "what" and "where" needed. — [arXiv HTML](https://arxiv.org/html/2609.24155)
- Code: github.com/Li-Jijie/Object-Centric-Visuomotor-Flow linked; abstract says "will be released". — [arXiv HTML](https://arxiv.org/html/2609.24155)

**S2 "See Less, Specify More" (arXiv 2606.02735v2)**: full text read (figures not viewed). Authors Yueh-Hua (Kris) Wu, Tatsuya Matsushima, Kei Ota; May 2026, v2 8 Jun 2026; venue not stated; no code link found. — [arXiv HTML](https://arxiv.org/html/2606.02735)
- Visual evidence budget: per camera view (base, wrist) lightweight gate heads predict soft keep values per image token, conditioned on token + pooled language summary; nonzero floor; budget loss pushes mean gate to rho=0.2 default. No masks/boxes/VLM labels: gates trained only by control loss (gated action loss + ungated loss + annealed temperature). Retained evidence may be contact surface/target region/clearance cue. — [arXiv HTML](https://arxiv.org/html/2606.02735)
- LIBERO-PRO Table 1 (Obj/Pos/Sem/Task), pi0.5 vs S2 (Kimi K2.5 planner): libero_10 66.0/6.0/91.0/17.0 vs 68.0/18.5/94.5/36.0; libero_goal 90.0/29.0/95.0/17.0 vs 87.0/49.5/97.0/43.0; libero_object 89.0/19.0/95.0/10.0 vs 90.0/62.5/100.0/29.5; libero_spatial 99.0/53.0/97.0/55.0 vs 99.0/55.0/90.5/54.5. — [arXiv HTML](https://arxiv.org/html/2606.02735)
- Real robot Table 2 (subtask success): TX-G2 pi0.5 Cutlery 5.0, Bowl 47.5, Clothes 83.3, Dish 60.0 vs S2 15.0/67.5/96.7/90.0; HSR Coffee 83.3, Bottles 45.8, Box 41.7, Mug 66.7 vs S2 100/87.5/91.7/83.3. Mean over 8 tasks 54.2% to 79.0%. — [arXiv HTML](https://arxiv.org/html/2606.02735)
- Ablation Table 9 (LIBERO-PRO mean): libero_goal Original 57.8, Refined-only 37.8, Original+VEB 62.5, Refined+VEB 35.0, full S2 70.1; libero_object 53.3, 70.4, 61.8, 71.9, full 73.1. — [arXiv HTML](https://arxiv.org/html/2606.02735)
- Relevance: a learned gate that sees less, but token-level, not identity-vs-geometry; no role/phase split.

**CGVD (arXiv 2603.10340)**: full text read. Song, Kodagoda, Carmichael (UTS), Thiyagarajan (WSU); 11 Mar 2026. Training-free: parse instruction into safe set (target, anchor, robot) and distractors, segment with SAM 3, LaMa-inpaint distractors once on first frame, cache clean scene. SimplerEnv WidowX, pi0 and GR00T, 2 tasks. Table II (pi0, spoon on towel, 18 semantic distractors): baseline 43.0 vs CGVD 77.5; mean-color fill 56.5; no target refinement 65.0; no robot mask 73.0. Latency 317 ms/step to 421 ms; init 4,914 ms. Underperforms on carrot-on-plate at some densities (clutter acts as context). No new object/layout generalization test in body; no code link. — [arXiv HTML](https://arxiv.org/html/2603.10340)

**TAG (arXiv 2603.24584v1)**: full text read. Zhou, Zhan, Zhai, Lyu, Liu, Wang, Lin, Wang (Sun Yat-sen), 25 Mar 2026. Classifier-free-guidance-style: run policy on original and a target-erased baseline, extrapolate (w~1.25). LIBERO (Spatial/Object/Goal/Long/Avg): pi0.5 95.4/98.4/97.2/89.6/95.2; +TAG-bg 97.8/99.2/97.4/97.0/97.9; +TAG-black 97.6/99.8/97.4/95.8/97.7. Per-frame gray/black masking or inpainting at inference collapses to ~63.65-64.50 avg (Long 12.6-13.8) vs static background 97.85, attributed to temporal inconsistency. LIBERO-Plus 81.4 to 86.1/87.2; VLABench 29.40 to 55.41. No code link. Direct evidence that naive per-frame masking of a pi0.5 hurts, a risk for the planned gate (the plan's silhouettes would need to be consistent frame to frame). — [arXiv HTML](https://arxiv.org/html/2603.24584)

**LiLo-VLA (arXiv 2602.21531v1)**: read as summarised by fetch (full text, tables). Yang, Cheng, Fang, Bharadhwaj, Ding, Bertasius, Szafir; 25 Feb 2026. Reaching module (MPLib planner) + interaction module (OpenVLA-OFT, wrist camera only, non-target objects masked, random erasing in training) + recovery. Table I overall SR/AP: pi0.5 28/31; OpenVLA-OFT 2/4; w/o Masking 48/64; w/o Recovery 8/33; LiLo-VLA 69/86 (Suite 1 82/89; Suite 2 44/79). pi0.5 83% on original Suite 1 sequences but 0% on reordered; LiLo-VLA 85% on reordered. Skills from segmented LIBERO-90 atomic demos; never trained on the test sequences. Real Franka 85% over 8 tasks. Masking ablation (random erasing removed) 69 to 48. Project page https://yy-gx.github.io/LiLo-VLA/ ; no code link seen. Closest paper on "mask non-target + compose unseen orderings + reach/interact (phase-like) split"; but the phase split is by module, not a conditioning switch in one VLA. — [arXiv HTML](https://arxiv.org/html/2602.21531)

**OBEYED-VLA (arXiv 2512.22519v2, rev. 24 Apr 2026)**: read to references (appendices A-B not seen). Vo, Hanyu, Ikebe, Pham, Chung, Vu, Minh, Nguyen, Gunderman, Rainwater, Le. YOLO11-Seg masks + Qwen3-VL set-of-mark selects task-relevant masks, matched across base/wrist views; Depth Anything masked depth; pi0 / pi0-FAST LoRA fine-tuned on ~2,000 clean single-object demos. Table II (pi0): full 85/73/78 (4 distractors seen / spatial / 4 distractors unseen), baseline pi0 5/20/11, no 2-stage 69/43/69, OC-only RGB 71/37/57. Chart-only: stays >90% with 1 distractor, ~80% with 7; pi0.5 near zero under colored backgrounds. Runtime 0.78 s (pi0) per cycle. Project page https://uark-aicv.github.io/OBEYED_VLA; no code. Real-robot UR10e, single-object training, so it tests held-out distractors/objects. — [arXiv HTML](https://arxiv.org/html/2512.22519)

**VP-VLA (arXiv 2603.22003)**: abstract + prose read only (via alphaxiv; tables/appendix not seen). Wang, Chen, Liu, Ye, Chen, Lu, Liu, Yu, Jia (HKUST/CUHK/SmartMore), 9 May 2026, code github.com/JIA-Lab-research/VP-VLA. System 2 planner runs only at transition events such as a gripper state change; VLM + SAM3 produce a crosshair (pick) or box (place) drawn on the image; System 1 gets original image + prompted image + instruction; auxiliary grounding loss at key frames. Numbers: RoboCasa-GR1 53.8% (+5.0), tray-to-plate novel split 66.0 vs 56.0; SimplerEnv 58.3% (+8.3); real waste sorting novel objects 85% vs 63.3 (baseline 80% in-domain); unseen egg colors 75.0 vs 29.2. No interface-type ablation on the page. This is the closest found for "switched by gripper event" (at inference; planner re-fires on gripper change). — [alphaxiv](https://www.alphaxiv.org/abs/2603.22003)

**Others seen in search snippets only (unread, unverified details):** DRAGON-VLA/"Distraction-Robust VLA" (SSRN 7571503, appears to be OBEYED-VLA's later version); ST-VLA (2603.13788; argues hard masks cause jitter); HSC-VLA (2603.07484); BYO-VLA (OpenReview); MaskLAM (2602.02259); LIBERO-Plus (2510.13626) reports models ignore distractors but fail when the target is displaced. Search source: web search results, URLs above. "Robust Skills, Brittle Grounding" (arXiv 2602.24143) fetch timed out; not read.

### Inferences
- No found paper replaces non-active entities by identity-free silhouettes; the planned edit appears novel in that detail, but "hide distractors" is crowded (CGVD, TAG, LiLo-VLA, OBEYED).
- Evidence of benefit: large under distractors (CGVD +34.5 pts; OBEYED 5 to 85), but compositional benefit is only indirectly shown (LiLo-VLA reorders).
- TAG shows per-frame masking/inpainting on pi0.5 can destroy long-horizon performance; silhouette gating needs temporally stable masks.

### Gaps
- No controlled test hiding identity (not geometry) of non-active entities; SlotFlow Level 1-3 definitions not extracted; DRAGON-VLA, ST-VLA, HSC-VLA not read in full; "Robust Skills, Brittle Grounding" unread.

## Key question 2: Has anyone switched a VLA's conditioning by task phase or grasp event during training?

### Takeaway
Switching by gripper event exists at inference (VP-VLA planner re-fires on gripper change; ForeTime-VLA derives phase from gripper transitions; ActionGround state machine). Phase-conditioned adapters exist (PhaseLoRA) but with weak proxy descriptors, no held-out-composition test.

### Cited Findings
- VP-VLA planner runs at transition events such as a gripper state change; prompts switch from crosshair (pick) to box (place); grounding loss only at key frames. — [alphaxiv](https://www.alphaxiv.org/abs/2603.22003)
- PhaseLoRA (arXiv 2608.15285v1, Guo, Wu, Duan, Ding, Han, Tsinghua, 15 Aug 2026; essentially full text read): per-timestep B_t = B0 + P*B_P + E*B_E + (P*E)*B_PE on action-expert LoRA; GRU router predicts fine-control tendency P and event intensity E from weak action-statistic targets (motion magnitude, jerk), "proxies, not true phase labels". LIBERO avg (Spatial/Object/Goal/10/Avg): LoRA 38.3, High-rank LoRA 56.7, DoRA 58.4, LoRA-MoE 58.3, LoRA-SP 59.2, PhaseLoRA 68.9 (85.3/88.0/64.0/38.3). Real Piper 4 tasks avg 50.8 to 69.2. Ablation LIBERO-Spatial: random descriptors 66.3, scalar gating 73.7, no interaction term 82.3, full 85.3. No unseen-combination evaluation. Code promised at github.com/Grinffin/PhaseLoRA, not yet available. (These low LIBERO numbers imply a small-data/low-budget regime; not comparable to full fine-tunes.) — [arXiv HTML](https://arxiv.org/html/2608.15285v1)
- Snippet-only (unread): ForeTime-VLA 2608.20735 phase target from gripper transitions (approach, grasp/release, transport, place/retract); ActionGround 2609.33256 training-free phase state machine; CAC-VLA 2607.04816 (context-gated action conditioning); EventVLA 2606.20092. — [search result](https://arxiv.org/html/2608.20735), [ActionGround](https://arxiv.org/html/2609.33256v1)

### Inferences
- A grasp-event-switched conditioning trained end to end on pi0.5 with role-specific visibility was not found.

### Gaps
- ForeTime-VLA, ActionGround, CAC-VLA not opened.

## Key question 3: Structured vs latent slow-fast interfaces

### Takeaway
No controlled head-to-head of latent vs text vs points vs images was found. Evidence is per-paper: InternVLA-N1 ablation (pixel + latent goals complementary), VP-VLA (visual prompt interface beats baseline). pi0.7 uses subgoal images + text (secondary source only).

### Cited Findings
- InternVLA-N1 ("Ground Slow, Move Fast"): removing pixel goals drops SR 2.1 points, removing latent goals 3.4 points (navigation, not manipulation; from a search-result summary, paper not opened). — [arXiv HTML](https://arxiv.org/html/2512.08186v1)
- Critiques that latent interfaces (OpenHelix, GR00T N1 static embeddings) are opaque/lagging; this is the search summary's characterisation. — [search summary of 2512.20188 etc.](https://arxiv.org/pdf/2512.20188)
- pi0.7 (arXiv 2604.15483, Physical Intelligence): per secondary summaries, a high-level policy gives subtask text and a world model generates subgoal images, refreshed on intent change or interval; trained on real + generated subgoals. Not verified from the original paper. — [hyper.ai summary](https://hyper.ai/en/papers/pi07)
- Harness VLA/CaP style systems use code/primitive interfaces (see Q5).

### Gaps
- Helix 02, GR00T N-series, Gemini Robotics ER+VLA interface details: not found/not read. No controlled interface comparison found.

## Key question 4: Adapter libraries and composing parts

### Takeaway
Neither MAST-VLA nor CORAL composes adapters in unseen combinations.

### Cited Findings
- MAST-VLA (arXiv 2610.32312v1, 23 Oct 2026; full text read, figures not viewed; Yang, Xu, Zhang, Tan, Zhang, Gong, Xie; venue not stated; no code link): skill picked from instruction among 7 coarse categories, indexes 32 learnable skill-query tokens + per-layer routing bias; LoRA-MoE with 6 experts, top-2, rank 64, alpha 128; skill bias shifts routing, no hard assignment. Does NOT evaluate unseen skill combinations. CALVIN ABC to D: MAST-VLA success@5 84.7, avg len 4.65 vs VLA-Adapter-Pro 80.0/4.50, FLOWER 77.5/4.54; random non-matching skill condition at test 4.55. Real robot 160 trials: S@1 98.8, S@2 92.5, S@3 70.6, avg len 2.62 vs pi0.5 2.41, pi0 2.18. — [arXiv HTML](https://arxiv.org/html/2610.32312)
- CORAL (arXiv 2603.09298v1, 10 Mar 2026; full text read, figures not viewed; Luo, Chen, Liang, Li, Frontier Robotics): one LoRA expert per task keyed by language instruction (k = R(T)), swap by unmerge/merge into frozen base; no composition. LIBERO avg: CORAL_pi0.5 98.4 vs pi0.5 96.9; CORAL_SimVLA 99.3 vs 98.6 (largest gain Long +3.4 for pi0.5); WidowX 97.9 vs 95.8; Google Robot 84.9 vs 77.0. Rank-16 expert ~26 MB (~100x smaller than checkpoint); swap <100 ms. Sequential full fine-tuning drops earlier tasks to 0%. Text inconsistency on training steps (1-5 epochs vs 50 steps). Project page https://frontierrobo.github.io/CORAL; no code. — [arXiv HTML](https://arxiv.org/html/2603.09298)

### Inferences
- A keyed (role, object) adapter library that is composed across unseen pairs would go beyond both; neither tests it.

### Gaps
- No adapter-composition-on-unseen-combos paper found in this pass.

## Key question 5: Code/agent systems (items 5, 6, 7)

### Cited Findings
- Harness VLA (arXiv 2607.08448v5, 24 Sep 2026; first ~100k of 204k chars read plus 100k-200k; last 4k not read; Appendix tables partly cut). Authors Yixian Zhang, Huanming Zhang, ... Chao Yu (Tsinghua et al.). Frozen pi0.5 (RLinf pi05_libero130_fullshot) wrapped by Codex or Claude Code planner; 6 analytic primitives + vla_act (+2 mobile); planner sees task text, agent-view RGB, metric depth, proprio, retrieved Task-Specific and Global Memory; bootstrap on one reference seed then deploy. LIBERO-Pro (T = instruction redirection, S = position swap; cells Spat-T/Spat-S/Obj-T/Obj-S/Goal-T/Goal-S/L10-T/L10-S/Overall): pi-RLinf 42/59/71/78/45/42/49/14/50.0; Harness (Codex) 81/69/94/91/75/66/52/49/72.1; Harness (CC) 94/80/88/90/87/87/71/62/82.4; Cap-X 14/12/18/22/17/26 (18.2*); RATS 31/29/63/61/36/43 (43.8*). Zero-shot Goal (no memory): CC Pos-S 31.0, Task-T 79.0; Cap-X 25.6/16.8. Latency Table 7: Codex high reasoning 78.50% success, 529.23 s planner time; no reasoning 62.50%, 344.17 s; Flash Mode (max) 72.63% (581/800), 56.17 s tool time (retrospective best-card on selection seeds, possibly optimistic). GPT-6 Astra planner low effort 92.63% (741/800, Table 21). No dollar/token cost found in the read portion. Code https://github.com/RLinf/RPent ; site https://harnessvla.github.io/. — [arXiv HTML](https://arxiv.org/html/2607.08448)
- CaP-X / CaP-Agent0 (arXiv 2603.22435v2, 2 Jul 2026; Fu, Yu, ... Zhu, Goldberg, Fan; NVIDIA/Berkeley/Stanford/CMU). Read chars 0-200k of 207k. Table 2 LIBERO-PRO (Pos/Task): libero-object OpenVLA 0/0, pi0 0/0, pi0.5 0.17/0.01, CaP-Agent0 0.22/0.18; libero-goal pi0.5 0.38/0.00 vs CaP-Agent0 0.26/0.17; libero-spatial pi0.5 0.20/0.01 vs CaP-Agent0 0.12/0.14. Appendix K.2 cost analysis NOT found in text read (listed in TOC; the unread last 7k chars may hold it). Project page https://capgym.github.io. — [arXiv HTML](https://arxiv.org/html/2603.22435)
- Item 7 existence checks:
  - RHO exists: arXiv 2606.16458, Elmaaroufi, Svegliato, Kalade, Schelle, Seshia, Zaharia, 15 Jun 2026; LIBERO-PRO: OpenVLA 0.0, pi0.5 12.83 avg, RHO 45.0 (~2.5x strongest multi-turn agent); Robosuite 70.0 (prior 68.29); O3DE held-out 23.5 to 44.3. Only abstract page read. Project https://rho-robotics.github.io/. Note search snippet gave 45% vs 70% figures that were not reconciled. — [arXiv abs](https://arxiv.org/abs/2606.16458)
  - URAI exists: arXiv 2609.39018 "Make Code as Policy Great Again: Frontier Agents Write, Call, and Evolve Robot Tools", Ge et al., 30 Sep 2026; five RoboDojo tasks 18.0 to 53.0%; 7 real AgileX tasks; no LIBERO numbers on the abstract page (only abstract read). Code github.com/HexaFuture/URAI. — [roboticscenter summary](https://www.roboticscenter.ai/research/papers/make-code-as-policy-great-again-frontier-agents-write-call-and-evolve-robot-tools-2609)
  - "Agent as Policy": found as arXiv 2609.12541, Jia, Lin, Zhang, Zhang, Liu, Jiang, 11 Sep 2026 (v3 28 Sep). No LIBERO numbers (abstract only). The URL agent-as-policy-2026.github.io was NOT found; no project page listed. — [arXiv abs](https://arxiv.org/abs/2609.12541)
  - "GPT-GEAR": name NOT found. arXiv 2610.28259 exists but is a different title: "Large Multimodal Models for Direct Robot Control: GPT-6 as a Beginning" (Zhou et al., 20 Oct 2026); only LIBERO-GeoMotion 11.1% zero-shot reported in abstract page. Possibly the brief's name is wrong; unverified link. — [arXiv abs](https://arxiv.org/abs/2610.28259)
  - EMERGE-Policy exists: arXiv 2608.29896, "A Robot Mind Emerges Beyond a Single Policy", graph-structured multi-agent system; LIBERO-Plus 93.9% vs 82.2% raw Cosmos Policy backend, 85.1% without Imagination Skill, 99.2% LIBERO (secondary). Snippet-only. Dates differ across indexes (30 Aug vs 9 Sep). — [arXiv PDF](https://arxiv.org/pdf/2608.29896)
  - FAEA exists: arXiv 2601.20334 "Demonstration-Free Robotic Control via LLM Agents", Tsui, Fang, Hwu, 28 Jan 2026; Claude Agent SDK; with privileged state 84.9% LIBERO, 85.7% ManiSkill3, 96% MetaWorld; +1 round human feedback 88.2% LIBERO; code github.com/robiemusketeer/faea-sim. Snippet-level. — [arXiv HTML](https://arxiv.org/html/2601.20334)
- Related: NVIDIA tech report Oct 14, 2026 on GPT-6 Astra as robot policy: Agentic Policy 74.8% vs Direct Policy EEF 66.3% and Orchestrated 66.5% (snippet only). — [NVIDIA](https://research.nvidia.com/labs/srl/projects/astranomy/)

### Gaps
- CaP-X cost appendix and per-task LIBERO-PRO (App. J/K) not seen; Harness VLA dollar cost not found; RHO, URAI, AGP, EMERGE, FAEA LIBERO-family tables not read in full.

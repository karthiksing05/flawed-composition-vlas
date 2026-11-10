# Structured and hierarchical architectures around VLAs (2026 scan, 2026-11-10)

Scope note: about 12 tool calls. Coverage is partial. I could not reach the official CoRL 2026 accepted list. Most fetches returned abstract pages only, so many numbers are missing. Several arXiv IDs (26xx) are post-training-data and were seen only through search/fetch tools.

## Q1. CoRL 2026 object-centric / role-structured VLAs; gains on new objects, combinations, layouts

### Takeaway
Only one confirmed CoRL 2026 object-centric policy turned up (SlotFlow, a flow policy, not a VLA, no numbers seen). No paper found that gives separate role pathways (object/target/verb) inside a VLA. Slot/mask work exists but mostly outside confirmed CoRL 2026.

### Cited Findings
- SlotFlow, "Object-Centric Conditioning for Visuomotor Flow Matching" (Li, Yang, Zou, Zhao, Zhao, Lei, Zhu; arXiv 2609.24155, submitted 21 Sep 2026, v2 29 Sep). Accepted to CoRL 2026. Splits observations into semantic "what" features and image-plane "where" cues, built on Action-to-Action flow matching. Claims robustness to visual distractors and severe spatial perturbations. No numbers and no code link on the abstract page. — [arXiv](https://arxiv.org/abs/2609.24155)
- S2 "See Less, Specify More" (Wu, Matsushima, Ota; arXiv 2606.02735, 1 Jun 2026). Not confirmed as CoRL. The executor gets a "visual evidence budget" and needs no mask annotation, and instructions are relabeled at subtask level. Real robot: mean subtask success 54.2% to 79.0% over pi0.5 (8 tasks, TX-G2 and HSR). A search summary says it gains on LIBERO-PRO over OpenVLA-OFT, X-VLA, VLA-Adapter, pi0.5, especially goal/position swaps and object distractors; I did not see those numbers. Project page https://s2.airoa.io; no code listed. — [arXiv](https://arxiv.org/abs/2606.02735), [emergentmind](https://www.emergentmind.com/papers/2606.02735)
- SAM3D-Guided Object-Centric Representation Alignment for VLAs (arXiv 2607.25912). SAM2 masks per training frame; aligns the VLA to a frozen SAM3D teacher; 99.1% average on LIBERO (standard, not PRO). Venue not confirmed. Only search-snippet level. — [PDF](https://arxiv.org/pdf/2607.25912)
- Slot representations for visuomotor imitation (arXiv 2607.09825, Chapin, Chen among authors): frozen slot encoder beats dense DINO baseline by about 22 points. Search snippet only; unverified. — [PDF listing via search](https://arxiv.org/pdf/2607.09825)
- 2025 reference points: Oat-VLA (object-agent-centric tokenization, CoRL 2025, at least 2x faster convergence than OpenVLA on LIBERO), SlotVLA (arXiv 2511.06754, slots plus relation tokens), Embodied-SlotSSM (arXiv 2511.11478, LIBERO-Mem). Snippet-level only. — [SlotVLA](https://arxiv.org/pdf/2511.06754v1), [Embodied-SlotSSM](https://arxiv.org/html/2511.11478v3)
- BayesVLA (arXiv 2512.11218, Dec 2025, reference): splits a vision-action prior from a language-conditioned likelihood; motivated by VLA bias to seen configurations under LIBERO-PRO. — [PDF](https://arxiv.org/pdf/2512.11218)
- LIBERO-PRO (Zhou et al., arXiv 2510.03827; v2 May 2026): models above 90% on LIBERO fall to about 0 under object/position/instruction/environment perturbations. — [arXiv](https://arxiv.org/abs/2510.03827)

### Inferences
- The role-pathway idea (identity of filler seen only by its own pathway) appears unoccupied in what I found; SlotFlow's what/where split is the nearest, but it is not per role.

### Gaps
- Official CoRL 2026 list not accessed; SlotFlow numbers, S2 LIBERO-PRO numbers, SAM3D results on perturbation not read. No paper found that deliberately hides identity from the motor policy and reports generalization (S2's evidence budget is the closest, and it hides visual evidence, not identity).

## Q2. Skill-library, adapter-library, MoE, composition, continual learning

### Takeaway
LoRA-expert libraries routed by language are common and claim no forgetting by construction; I found no reported numbers for composing skills never seen together.

### Cited Findings
- MAST-VLA (Yang, Xu, Zhang, Tan, Zhang, Gong, Xie; arXiv 2610.32312, 23 Oct 2026). Accepted to CoRL 2026. Skill-conditioned LoRA-MoE adapters in the LLM backbone with skill-query tokens and routing biases, plus an affordance-heatmap-derived patch-level trajectory prior for the action head. CALVIN ABC to D: Avg. Len 4.65 (claimed SOTA). No LIBERO, no unseen-skill-combination numbers, no code link on page. — [arXiv](https://arxiv.org/abs/2610.32312)
- CORAL (Luo, Chen, Liang, Li; arXiv 2603.09298, 10 Mar 2026; arXiv preprint). Frozen VLA plus one LoRA expert per task; a manager routes by language instruction; claims no catastrophic forgetting by parameter isolation and "substantially outperforms joint training" on LIBERO, WidowX, Google Robot. No numbers or code on page; site frontierrobo.github.io/CORAL. — [arXiv](https://arxiv.org/abs/2603.09298)
- 2025 references: DMPEL (arXiv 2506.05985, growing low-rank expert library, LIBERO lifelong); PSEC (arXiv 2502.05932, ICLR 2025, LoRA skill library composed in parameter space, D4RL/DMC not VLA). — [DMPEL](https://arxiv.org/pdf/2506.05985v2), [PSEC](https://arxiv.org/abs/2502.05932)

### Inferences
- CORAL routes by whole-instruction text, so each adapter is per task, not per object or target; a per-object keyed adapter with composition is not shown here.

### Gaps
- No per-adapter composition results (unseen pairings) verified. Model merging and policy-composition work (e.g. diffusion composition) not searched.

## Q3. Slow/fast interfaces

### Takeaway
Latent-embedding interfaces dominate the dual-system papers I found; points/affordances are used by a few. I found no controlled evidence that structured interfaces generalize better than latent ones.

### Cited Findings
- DIAL (Mar 2026): System 2 predicts latent future ViT features; System 1 is a latent inverse-dynamics decoder. — [emergentmind](https://www.emergentmind.com/papers/2603.29844)
- InternVLA-M1 (2025 ref): latent planning tokens from a spatially grounded VLM condition a diffusion action expert. — [arXiv](https://arxiv.org/html/2510.13778v1)
- Fast-in-Slow (NeurIPS 2025 ref): System 1 embedded in the VLM by shared parameters, 117.7 Hz, chunk size 8. — [NeurIPS](https://neurips.cc/virtual/2025/poster/119931)
- DualVLN (navigation; arXiv 2512.08186): latent query embeddings plus pixel goals; S2 at 2 Hz, controller 200 Hz. — [PDF](https://arxiv.org/pdf/2512.08186v1)
- Affordance/point interfaces: Afford-VLA (arXiv 2605.24203, affordance query tokens decode masks and condition action), AffordanceVLA (arXiv 2606.06155), AffordanceWAM (arXiv 2609.22332); venues and numbers unread. A third-party CoRL 2026 mirror lists H-VLA (key-action vs motion decoupling) and RoboMAP (affordance heatmaps as interface), acceptance unverified. — [Afford-VLA](https://arxiv.org/abs/2605.24203), [mirror](https://hongsong-wang.github.io/CoRL2026/)
- Typed function calls: Harness VLA (see Q4) uses a typed primitive vocabulary with `vla_act` as one call. — [HF](https://huggingface.co/papers/2607.08448)
- Unverified: a blog claims GR00T N1.7 timings (S2 about 100 ms, S1 8-20 ms). — [dev.to](https://dev.to/codebyankita/vision-language-action-models-when-llms-learn-to-use-their-hands-38am)

### Gaps
- No head-to-head latent vs structured interface study found; no 2026 survey fetched.

## Q4. Frontier-model agent / code-as-policy stacks vs VLAs

### Takeaway
CaP-X and Harness VLA exist. Code agents beat VLAs on instruction perturbations but are not clearly better on position perturbations; wrapping a VLA as a callable primitive gives the largest reported LIBERO-Pro gains. RHO, Agent as Policy, URAI, GPT-GEAR, EMERGE-Policy: not found.

### Cited Findings
- CaP-X (Fu, Yu, El-Refai, ... Fei-Fei Li, Guanya Shi, Jiajun Wu, Yuke Zhu, Ken Goldberg, Linxi Fan; arXiv 2603.22435, 23 Mar 2026, v2 2 Jul 2026; no venue listed). CaP-Gym, CaP-Bench (12 frontier models), CaP-Agent0 (training-free: multi-turn, visual differencing, skill synthesis, ensembled code), CaP-RL. — [arXiv](https://arxiv.org/abs/2603.22435)
- CaP-Agent0 LIBERO-PRO (Table 2, success, Pos / Task): Object 0.22/0.18; Goal 0.26/0.17; Spatial 0.12/0.14. pi0.5: 0.17/0.01; 0.38/0.00; 0.20/0.01. OpenVLA and pi0: 0.00 everywhere. Agent wins on instruction perturbations; loses to pi0.5 on goal and spatial position. Cost in Appendix K.2, not read. — [HTML](https://arxiv.org/html/2603.22435)
- CaP-X failures noted on BEHAVIOR: limited field of view, occlusion, knocked-over objects. — same HTML.
- CaP-X code: github.com/capgym/cap-x, MIT, includes LIBERO-PRO tasks (130+ LIBERO tasks), separate venv (Python 3.12, `uv sync --extra libero --extra contactgraspnet`); no scores in README. Project page https://capgym.github.io. — [GitHub](https://github.com/capgym/cap-x)
- Harness VLA (Zhang Yixian, Zhang Huanming, ... Wenbo Ding, Chao Yu; Tsinghua et al.; arXiv 2607.08448, about Jul 2026). LLM planner composes fixed analytic primitives (grounding, transport, release) and calls a frozen VLA as `vla_act` (prompt, max chunks, stop predicate) for contact-rich phases; memory of successful traces. LIBERO-Pro aggregate: 82.4% (Claude Code), 72.1% (Codex), vs RATS 43.8%, RLinf (pi0.5-SFT used alone) 50.0%. Zero-shot Goal vs Cap-X: position-swap 31.0 vs 25.6; instruction-redirection 79.0 vs 16.8. Also RoboCasa365 +25.4 pts, RoboTwin 58.4%. No code in page; site https://harnessvla.github.io/. Only first 100k chars read. — [HF](https://huggingface.co/papers/2607.08448)
- Act-Observe-Rewrite (arXiv 2603.04466, Mar 2026): Claude Code writes controller code between trials; 100% on two tasks, 91% on a third. ALRM (arXiv 2601.19510): CaP and Tool-as-Policy modes, 56-task benchmark. — [AOR](https://arxiv.org/abs/2603.04466), [ALRM](https://arxiv.org/abs/2601.19510v2)
- LIBERO-Plus (arXiv 2510.13626, CVPR 2026): seven perturbation dimensions, 10,030 tasks; some models fall from 95% to below 30%. CoTinyVLA reports 90.8/87.3/86.6/80.7 on its four suites. I found no frontier-agent score on LIBERO-Plus. — [CVPR](https://cvpr.thecvf.com/virtual/2026/poster/38735), [CoTinyVLA](https://www.alphaxiv.org/abs/2607.25487.md)

### Inferences
- Binding in code stacks is handled by the planner naming entities in code or prompts (and perception returns coordinates), so identity never enters the motor policy; Harness VLA still passes a task prompt to the VLA, so identity reaches it.

### Gaps
- Not found: RHO, "Agent as Policy", URAI, GPT-GEAR, EMERGE-Policy (searched once, combined query; each should be searched separately). Runtime and token cost not obtained. Harness VLA code availability unverified.

## Which have open code that runs on LIBERO
- Confirmed: CaP-X (MIT, LIBERO-PRO) — [GitHub](https://github.com/capgym/cap-x).
- No code link seen: SlotFlow, S2, MAST-VLA, CORAL, Harness VLA (project pages only).

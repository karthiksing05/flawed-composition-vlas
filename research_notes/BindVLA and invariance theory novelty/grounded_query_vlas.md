# Grounded-query / mask / slot VLAs: novelty check for BindVLA (through Oct 2026)

Reference design, BindVLA (SmolVLA base):
(1) one query per instruction role (object, goal), built ONLY from that role's phrase tokens plus a role embedding, with no cross-role attention;
(2) each role query cross-attends to visual tokens, giving a soft mask and a slot feature, with two-hop "A of B" resolution;
(3) position-free matching keys, with position entering only through a relative-geometry channel;
(4) a hard bottleneck: the action expert sees only the role slots, the slot object-frame pose, proprioception and one low-res context token;
(5) a V-REx/IRM penalty across scene layouts.
Training uses flow matching with no binding supervision. Evaluation uses MCD splits.

Provenance tags:
- [full text] = the arXiv HTML was read end to end (method, experiments, appendix where one exists) through a fetch-and-summarise tool.
- [abstract/snippet] = only the abstract page or a search snippet was read.
- [prior knowledge, not re-read] = recalled from training data, with the URL given for checking.

All arXiv IDs below were checked against arxiv.org this session except where marked. Correction: GROOT is **2310.14386**, not 2310.xxxxx/2310.12945.

## Q1. What does each paper do vs BindVLA (1)-(5), is binding supervised, which splits are reported, and is there a collision?

### Takeaway
No paper found combines (1)-(5). Four papers each collide with one piece, and BindVLA must cite and differentiate against all four:
- **RoboGround**: separate object and placement query sets, but the masks come from a supervised external VLM and there is no bottleneck.
- **POT-VLA (2607.18016)**: role-typed slots (TARGET/DESTINATION) with learned role embeddings and base-frame 3D pose fed to a GR00T DiT, but grounding comes from external SAM3 and there is no bottleneck.
- **Pelican-VLA 0.5**: a hard bottleneck between the action expert and the scene, with flow matching and no grounding supervision, but through 32 generic role-agnostic tokens.
- **OTTER**: unsupervised text-token-to-patch cosine selection, frozen, with a position-free similarity and the policy seeing only pooled selected features. It works per token, not per role, and position is absolute in the values.

I found no VLA paper using V-REx/IRM (5), two-hop "A of B" resolution, or MCD/compound-divergence splits.

### Cited Findings

**RoboGround, arXiv 2504.21530 (Huang et al., ZJU/Shanghai AI Lab, 30 Apr 2025) [full text]** — [arXiv](https://arxiv.org/html/2504.21530)
- Mechanism: a GLaMM-based grounded VLM emits a `<SEG>` token per target, decoded SAM-style into a target-object mask M_o and a placement mask M_p.
- The masks are channel-concatenated with the image and passed through a ViTMAE encoder.
- A "Grounded Perceiver" then uses three learnable query sets: global Q_g, object Q_o and placement Q_p, 9 each (27 total, per the appendix).
- vs (1): there are role-specific query sets, but they are learnable vectors, not built from the role's phrase tokens. The text does not say whether query-to-query (cross-role) attention is masked; all queries share one 27x223 attention matrix.
- vs (2): masks come from the external VLM. The perceiver's mask use is described ambiguously: attention in masked regions is "replaced with the highest attention value in the current matrix", which reads as boosting the masked region, not restricting to it.
- vs (3)/(5): no position-free keys, no relative geometry, no invariance penalty.
- vs (4): no bottleneck. The policy also gets the global CLS, global queries, the CLIP text token Z_t and state.
- Binding is SUPERVISED. GLaMM is LoRA-fine-tuned on 112K simulated QA pairs plus 277K grounded-conversation pairs, with BCE+DICE mask loss.
- Policy: a GR-1 re-implementation trained from scratch with Smooth-L1 plus gripper BCE, not flow matching.
- Splits: the instruction types are Easy / Appearance / Spatial / Common-sense, plus zero-shot unseen-instance and unseen-class splits (30 held-out classes, 597 objects). There is no compositional split.
- Numbers: on Appearance/Spatial/Common-sense, GR-1 scores 13.8 / 16.3 / 11.5 and RoboGround 30.5 / 33.5 / 30.0. Unseen-class Spatial goes 6.3 -> 14.8 with masks.
- Ablation: GT masks beat predicted masks (Easy 68 vs 64). Full masks beat a bbox channel, which beats points.
- **Collision: PARTIAL on (1)**, the role-split object/placement queries. That is the main prior BindVLA must contrast with. BindVLA's queries are phrase-built and unsupervised, and BindVLA adds a bottleneck.

**SlotVLA, arXiv 2511.06754 (Hanyu et al., v4 stamped 01 Oct 2026) [full text; no appendix on the page]** — [arXiv](https://arxiv.org/html/2511.06754)
- Mechanism: Locatello slot attention over patch features, which is NOT language-conditioned.
- A task-aware filter (bidirectional slot<->language cross-attention, then a relevance score) keeps the top-k slots (N_S=4).
- Relation tokens come from learnable queries that cross-attend first to the dense patches V_t, then to the slots.
- The action decoder gets [slots; relation tokens; language; proprio].
- vs (4): there is no pure bottleneck, because dense visual information leaks through the relation tokens (Eq. 7).
- vs (1): no per-role query sets.
- vs (3): no positional or relative-geometry treatment described.
- vs (5): no invariance penalty.
- Binding is SUPERVISED. Stage 1 uses Hungarian matching with box loss, mask BCE, objectness, a contrastive tracking loss and a relevance BCE whose labels come from the objects named in the task.
- Action side: OpenVLA with LoRA, discretised action tokens (CE).
- Splits: LIBERO+ (new annotations) Goal/Spatial/Object/Long, trained and tested on the same tasks. No held-out generalisation split, despite the abstract's claims.
- Numbers: ORC vs OpenVLA is L-Goal 0.86 vs 0.77, L-Spatial 0.60 vs 0.72, L-Long 0.31 vs 0.56.
- **Collision: LOW.** It is language-selected slots, but supervised, not role-structured, with no bottleneck and no OOD split.

**Pelican-VLA 0.5, arXiv 2607.06655 (X-Humanoid, 9 Jul 2026) [full text; no appendix]** — [arXiv](https://arxiv.org/html/2607.06655)
- Mechanism: a single Qwen3-VL-4B stream holding the prefix VL tokens, Cosmos latents, K=32 learnable "BotTokens" and a suffix (state plus 50 noisy action tokens).
- In bottleneck mode, all suffix->prefix/middle attention is masked (Eq. 9), so the action tokens read the scene ONLY via the BotTokens.
- The mask is curriculum-applied with p_t = min(1, t/10k).
- vs (4): **direct collision with the "hard bottleneck on the action expert" idea.** The BotTokens are generic and shared, though: not role- or phrase-derived, and they carry no masks, slots or poses.
- Language enters only via the prefix, plus an InfoNCE loss between trajectory and instruction.
- vs (3)/(5): positional encoding is not discussed. The only regulariser is an orthogonality penalty on the BotTokens (0.01). No invariance penalty.
- No grounding supervision: "no object labels, segmentation masks, or attention supervision".
- Objective: flow matching (u = eps - a, 10 Euler steps).
- Splits: RoboTwin Clean/Randomized (91.4/91.0) and qualitative zero-shot RoboTwin 2.0. No compositional split.
- Evidence: target-object attention IoU rises 0.054 -> 0.124 over training. Ablations are qualitative attention maps only; there is no success-rate ablation of the bottleneck.
- **Collision: MODERATE on (4).** It is the closest prior for "unsupervised flow-matching VLA with a hard scene bottleneck". BindVLA must show that role-structured slots, not merely a narrow channel, drive binding. That calls for a matched BotToken-style ablation with the same K.

**STORM, arXiv 2601.20381 (Chapin, Dellandrea, Chen, Ecole Centrale de Lyon; v3 25 Jun 2026) [full text; no appendix]** — [arXiv](https://arxiv.org/html/2601.20381)
- This is not a VLA. It is a slot adapter (K=7) on frozen DINOv2, feeding a transformer imitation policy with a GMM head.
- vs (1): slots are **initialised from CLIP text embeddings of nouns parsed from the instruction (spaCy)**, with "robot arm" always appended. This is the nearest thing to "queries built from phrase tokens", but it uses nouns, not roles, has no role embedding, and slots still compete within slot attention.
- vs (3): each slot is given its mask's **centre of mass concatenated to the slot**. That is absolute position, the opposite of (3). Center encoding beats mask, bbox and no-position in its Table IV.
- vs (4): the policy also gets the task embedding, proprio and an ACT token. No hard role bottleneck.
- Binding is weakly supervised: DINOSAUR reconstruction plus a PatchCLIP contrastive loss to text on VG-COCO, plus an entropy usage penalty. No masks or boxes.
- Splits: MetaWorld/LIBERO/real, ID vs OOD (distractors, lighting, textures). Numbers: LIBERO 89.6/83.1 vs DINOv2-frozen 78.9/70.3.
- The paper has arithmetic inconsistencies, e.g. a claimed +19.0 OOD gain where the table gives 9.9.
- **Collision: LOW-MODERATE on (1)/(2)**, the language-initialised object slots. Distinguish it on roles, two-hop resolution, the bottleneck and position-free matching.

**OTTER, arXiv 2503.03734 (Huang, Fu, Malik, Goldberg, Abbeel et al.; v4 30 Dec 2025) [full text incl. Appendices A-D]** — [arXiv](https://arxiv.org/html/2503.03734)
- Selection mechanism:
  - It uses frozen CLIP features: the last vision attention-block output (ClearCLIP-style) and per-token text features.
  - For each text token, cosine similarity to the patches goes through softmax(sim/tau), and the patch values (feature + 2D sin-cos PE) are averaged with those weights. This gives one soft mask and one feature per text token.
  - tau is the only learnable parameter in the selection step.
- vs (3): the matching scores are position-free (PE is added to the values, not the keys), but position is ABSOLUTE in the values, not relative.
- vs (4): there is a bottleneck. The per-token features are pooled by 4 learnable cross-attention queries and concatenated with pooled text and proprio into one token per timestep, so raw patches never reach the policy. It is not role-structured.
- vs (1): selection is per text token, not per role phrase, and there is no two-hop resolution.
- Binding is UNSUPERVISED, learned implicitly through the action loss.
- Policy: a small causal transformer (4-8 layers). The paper does not state the loss; it is not flow matching.
- Splits: LIBERO 10 unseen tasks (changed instructions, objects and colours from LIBERO-90) plus real unseen objects and object-placement combinations. Numbers: LIBERO unseen 61% vs OpenVLA 48%; real multi-primitive unseen 67-77% vs fine-tuned pi0 35%.
- Ablation: fine-tuning CLIP collapses unseen success to 15%; standard xattn fusion gets 2% unseen.
- **Collision: MODERATE on (2)+(3)+(4) jointly.** It is the closest prior for "language-keyed, position-free soft selection with the policy seeing only selected features". BindVLA must cite it and differentiate on per-role queries, no cross-role mixing, relative geometry instead of absolute PE, a learned (not frozen-CLIP) matcher inside a flow VLA, two-hop resolution and the invariance penalty. OTTER is also a natural baseline.

**ObjectVLA, arXiv 2502.19250 (Zhu et al., Midea/ECNU, Feb 2025) [full text; appendix thin]** — [arXiv](https://arxiv.org/html/2502.19250)
- Mechanism: built on DiVLA (Qwen2-VL-2B, diffusion head). It co-trains on 2,000 image-text box-detection pairs and makes the model generate the target bbox as text tokens before acting.
- There are no queries, masks, slots or bottleneck. The box-reasoning module's details are deferred to DiVLA.
- Binding is SUPERVISED: DINO-X boxes, manually corrected.
- Splits: 100 OOD objects, with "move to" success 64% OOD vs 19% without box reasoning. Bin-picking OOD 87/150 vs OpenVLA 17/150. Smartphone-photo fine-tuning of new objects reaches 80-90%.
- **Collision: NONE architecturally.** It is relevant only as "explicit target localisation helps OOD objects".

**InSpire, arXiv 2505.13888 (Zhang et al., UESTC; v3 29 Sep 2025) [full text; no appendix visible]** — [arXiv](https://arxiv.org/html/2505.13888)
- Mechanism: the VLA first answers a text VQA question, "In which direction is the [object] relative to the robot?", with the object noun from NLTK. The question and answer are then fed back as tokens before the action is predicted. No new modules, queries, masks, slots or bottleneck.
- Binding is SUPERVISED: a rule-based label from the EE-to-object vector's dominant axis.
- vs (3): this is weakly related, as relative EE-object direction expressed as a text token.
- Splits: train LIBERO-90 and test the other LIBERO suites as "unseen". miniVLA-VQ unseen goes 3.6 -> 13.6, and pi0-FAST unseen goes 5.7 -> 8.5. Also CALVIN ABC->D and real 5 unseen tasks.
- **Collision: NONE.**

**POT-VLA, "Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation", arXiv 2607.18016 (Ren et al., v2 26 Jul 2026) [full text]** — [arXiv](https://arxiv.org/html/2607.18016)
- Mechanism:
  - A VLM planner or human task file emits typed subtasks with ROLES (TARGET, DESTINATION, SUPPORT, HANDOVER_PARTNER) and grounding queries.
  - SAM3 segments each role's query region. The depth pixels are back-projected to a base-frame 3D centroid and extent, plus relation features (EE-object offset, displacement, support, partner distance).
  - Result: a K=8 x F=33 slot tensor, MLP-projected with **learned role and context embeddings**, inserted into the GR00T-N1.7 DiT self-attention as [state, object, action].
- vs (1): **role-typed slots with role embeddings collide conceptually.** But the role-to-phrase binding is done by an external planner plus SAM3, not by learned phrase-built queries.
- vs (3): relative geometry is present (EE-object offsets in the base frame).
- vs (4): no bottleneck; the image tokens still reach the DiT via the VLM.
- vs (5): no invariance penalty.
- Grounding is external and frozen (SAM3), with no auxiliary loss ("no auxiliary object-memory or predicate loss").
- Splits: real Unitree G1, 10 trials per cell. Novel instances, shifted layouts, distractors and perturbations each go from direct to POT as 6->9, 5->9, 8->9 and 4->8 (out of 10). Matched comparison: 39/80 -> 71/80.
- The baseline is confounded (a different checkpoint).
- **Collision: MODERATE on (1) role slots plus the pose/relative-geometry part of (3)/(4).** Differentiate on learned, unsupervised binding from phrase tokens, the hard bottleneck, the invariance penalty and controlled MCD splits.

**ControlVLA, arXiv 2506.16211 [full text, via fetch]** — [arXiv](https://arxiv.org/html/2506.16211)
- Mechanism: GroundingDINO, prompted by the instruction, then SAM2 tracking. Object tokens are z_i = [sinusoidal PE of the mask's mean coordinates; a CNN over the mask]. They are injected through zero-initialised K/V cross-attention into a DROID-pretrained DiT policy, and via an "object expert" with zero-conv layers into pi0.
- Objects are generic, with no roles and no bottleneck. Grounding is external.
- Numbers: 76.7% overall vs Diffusion Policy 20.8% (10-20 demos). Unseen objects 70% on OrganizeToy.
- **Collision: LOW.** It is an object-token injection prior that uses absolute position.

**Adjacent hits [abstract/snippet]**
- AnySlot 2604.10432 (Apr 2026) converts language into a rendered spatial marker at the placement slot, then runs a goal-conditioned VLA. It introduces SlotBench. Here "slot" means a placement location, not slot attention. — [arXiv](https://arxiv.org/abs/2604.10432)
- FocusVLA 2603.28740 (Mar 2026) uses "Modality Cascaded Attention" to remove shortcut pathways plus "Focus Attention" patch selection. Its details are not visible from the abstract. A possible weak bottleneck overlap is worth a full-text check. — [arXiv](https://arxiv.org/abs/2603.28740)
- TAG 2603.24584 does target-agnostic guidance for object-centric inference under distractors, at inference time. — [search snippet](https://arxiv.org/html/2603.24584)
- CTRL-O 2503.21747 makes slots controllable by language (category or referring expression) without mask supervision. It is a representation method, not a policy, but it is prior art for the language-queried slot mechanism in (2). — [search snippet](https://arxiv.org/html/2503.21747v1)
- Object-Centric World Model 2503.06170 combines slot attention with an action decoder and masks slots in the decoder (per the appendix, according to the snippet). — [search snippet](https://arxiv.org/html/2503.06170v1)
- "What Matters, When?" 2609.05376 (Sep 2026, ECCV workshop, non-archival) frames ACT failures as conditional visual grounding. Its interventions are distractor augmentation, phase-dependent attention regularisation and visual prompting. It supports BindVLA's motivation; no role queries are mentioned. — [arXiv](https://arxiv.org/abs/2609.05376)
- DILL 2609.37165 targets spurious correlations in VLAs via domain-invariant latent lookahead, not a risk penalty. It is the nearest "invariance in VLAs" hit and is relevant to (5). — [arXiv](https://arxiv.org/abs/2609.37165v1)
- PointACT 2605.21414 uses action tokens as bottleneck queries over point tokens (windowed attention). — [search snippet](https://arxiv.org/pdf/2605.21414v1)
- SmolVLA 2506.01844 itself: its action expert cross-attends to VLM features (BindVLA's base). — [arXiv](https://arxiv.org/pdf/2506.01844)

**Classic object-centric / two-stream / relative-position designs**
- VIOLA, arXiv 2210.11339 (Zhu, Joshi, Stone, Zhu, CoRL 2022): object proposals from a pretrained model become per-object tokens for a transformer policy, with no language. **ID verified [abstract]**; mechanism details are [prior knowledge, not re-read]. — [arXiv](https://arxiv.org/abs/2210.11339)
- GROOT, arXiv **2310.14386** (Zhu, Jiang, Stone, Zhu, CoRL 2023): object-centric 3D point-cloud representations from segmentation, plus a segmentation-correspondence model for new instances, with a transformer policy. It is robust to background and viewpoint changes, and is single-task with no language. **ID verified [abstract/snippet]**. — [arXiv](https://arxiv.org/pdf/2310.14386); [PMLR](https://proceedings.mlr.press/v229/zhu23b.html)
- PolarNet, arXiv 2309.15596 (Chen, Garcia, Schmid, Laptev): point cloud plus language through a multimodal transformer on RLBench. No object slots or role queries. **ID verified [abstract]**. — [arXiv](https://arxiv.org/abs/2309.15596)
- VIMA, arXiv 2210.03094 (Jiang et al., ICML 2023): multimodal prompts and a "four-level evaluation protocol for systematic generalization" (verified [abstract]). It uses detector-derived object tokens (bbox plus crop), and its action decoder cross-attends to the prompt. Its levels L1-L4 cover placement, combinatorial, novel-task and novel-task-unseen generalisation; those level names are [prior knowledge, not re-read]. VIMA's combinatorial level is the closest prior to an MCD-style split, but it is not compound-divergence controlled. — [arXiv](https://arxiv.org/abs/2210.03094)
- CLIPort 2109.12098 (semantic CLIP stream x spatial Transporter stream), PerAct 2209.05451 (voxel Perceiver), Act3D 2306.17817 and 3D Diffuser Actor 2402.10885 (relative 3D rotary attention): all [prior knowledge, not re-read; IDs not re-verified this session]. Act3D and 3D Diffuser Actor are the main prior for "position only through relative geometry" in (3). They apply relative 3D position inside attention keys and queries, the opposite of BindVLA's position-free matching keys. Cite them as the relative-position lineage.

### Inferences
- The defensible novelty is the conjunction, not any single part:
  - phrase-built, per-role, cross-role-isolated queries (vs RoboGround's learnable role queries and POT-VLA's external role grounding);
  - unsupervised soft-mask binding inside a flow-matching VLA (OTTER is unsupervised but frozen-CLIP and per-token; RoboGround, SlotVLA and ObjectVLA are supervised);
  - a hard bottleneck that is role-structured (vs Pelican's generic BotTokens and OTTER's pooled per-token features);
  - position-free keys with relative-geometry-only position (vs OTTER, STORM and ControlVLA using absolute PE; Act3D/3DDA put relative position inside the keys);
  - V-REx across layouts, plus MCD splits.
- Reviewers will most likely raise Pelican-VLA 0.5 ("bottlenecks already make attention focus") and OTTER ("text-keyed patch selection already generalises"). The proposed theory-predicted ablations should include:
  - a generic K-token bottleneck (Pelican-style) at matched capacity;
  - OTTER-style per-token cosine selection;
  - RoboGround-style learnable role queries without phrase construction.
- POT-VLA shows role-typed slots plus base-frame pose plus relation features already help layout/instance OOD on a real humanoid. BindVLA's claim must be that this can be learned without external segmentation or a planner.
- The two-hop "A of B" resolution and V-REx/IRM across layouts had no hits in any VLA paper. They are the least contested pieces, subject to the gaps below.

### Gaps
- FocusVLA 2603.28740, TAG 2603.24584, AnySlot 2604.10432, CTRL-O 2503.21747, Object-Centric World Model 2503.06170 and DILL 2609.37165 were read only at abstract or snippet level. FocusVLA's "Focus Attention" and CTRL-O's language-conditioned slots deserve a full-text check for overlap with (2)/(4).
- VIMA, CLIPort, PerAct, Act3D, 3D Diffuser Actor, VIOLA and GROOT internals were not re-read this session. The VIMA level definitions and the Act3D/3DDA relative-position details come from prior knowledge.
- Several fetched pages may be summarised imperfectly (the reading went through a summarising tool). Specifically:
  - RoboGround's perceiver mask operation is ambiguous in the source itself.
  - SlotVLA's Table III was hard to parse.
  - Pelican's "22k warmup" and its 6000-hour vs 2400-hour figures are not reconciled in the source.
- No search found "role queries" or "target and placement queries" in other VLA papers. Search coverage of the Aug-Oct 2026 arXiv is limited by the search engine, so a manual arXiv listing search for "role query", "object-centric bottleneck" and "slot VLA" over Sep-Oct 2026 is advised.
- No VLA paper reporting compound-divergence (MCD) splits was found. Absence is not proven.

# Novelty check: RAA (role-addressed action), 2026-11-16

Method and limits. About 10 web searches and 12 arXiv abstract-page fetches. Every paper below was read at abstract level only ("abstract only"); the fetch tool returns a small-model summary of the page, so wording is paraphrase. No full texts opened. Papers already in docs/NOVELTY.md (ReGuide, LiLo-VLA, AC-VLA, CRAFT, ACT-VLA, Harness VLA, VP-VLA, Grant et al., MEM, TAG, Brittle Grounding, LIBERO-PRO/CF/Plus) are not re-read; I reuse that file's descriptions. Search was not exhaustive (no RoboPoint/MOKA/PointVLA/TAX-Pose hits beyond TAX-Pose; those are cited from memory and marked so). A "not stated" below means the abstract is silent, not that the paper lacks it.

## Q1. Actions in an object/target-relative frame

| Paper | What it does (abstract only) | Pre-empts? |
|---|---|---|
| TAX-Pose, Pan et al., CoRL 2022, arXiv 2211.09325 | Learns a task-specific cross-pose between two objects (action object vs anchor) from demos; planner executes it. Handles moved anchor. | Pre-empts "target-relative placement" as a principle. Not a VLA, no language roles, no pairing hold-out. |
| SPOT, 2411.00965 (arXiv, v2 2025) | Task = SE(3) pose trajectory of the object relative to its target; diffusion policy follows. | Pre-empts object-relative parameterisation in imitation. No language, no held-out pairings stated. |
| FOCI Policy, 2609.08743, CoRL 2026 (accepted) | Skills as relative SE(3) motion between task-relevant objects, extracted from short interaction segments; claims invariance to scene configuration. RLBench, COLOSSEUM, real. | Closest on Q1. Pre-empts "predict motion in an object-relative frame, from demo segments". Abstract silent on language, role assignment, VLA backbone, held-out pairings. Must be read in full before submission. |
| LiLo-VLA 2602.21531 (in NOVELTY.md) | Each action is a primitive plus a reference object defining the local frame; wrist-only object-centric VLA. | Pre-empts object-frame actions inside a VLA-like interaction module, with non-target masking. |
| OC-VLA 2508.13103 | Actions in the third-person camera frame. | Not an object frame; irrelevant except as "alternative frame" prior. |
| SeeR-VLA 2607.11498 | Pointmaps recentred on the end-effector as observation input (pi0.5 backbone). | Observation frame only; abstract does not say action frame. Not pre-empting. |
| CamVLA 2607.05396 | Camera-local action frame. | Not object frame. |
| ControlVLA 2506.16211, CoRL 2025 | Injects object-centric representations into a pretrained VLA via zero-init ControlNet-style layers; few-shot. | Object-centric conditioning, not an object-frame output, no role pointer. Adjacent only. |

Verdict Q1: object-relative action frames are prior art (TAX-Pose, SPOT, FOCI, LiLo-VLA). Doing it with a role-conditioned entity chosen by instruction inside a VLA expert is not covered by these abstracts.

## Q2. Pointer/slot binding of instruction roles to object tokens inside a VLA

- OA-WAM, 2605.06481 (arXiv preprint, May 2026). Object slots (persistent address vector + content vector) fused with text tokens; world head plus flow-matching action head; LIBERO 97.8, SimplerEnv 79.3, LIBERO-Plus gains, slot-swap causal test. Abstract does not say: explicit role pointer, action-frame change, restriction of the action head to the pointed slot, or held-out pairings. Its address-only keys "separate which object to act on from what it is". Closest prior for entity tokens plus identity/address split. Pre-empts "entity-token VLA with addresses"; does not pre-empt the pointer + role-frame + restricted-input combination as far as the abstract shows.
- POT-VLA, 2607.18016 (arXiv, Jul 2026). Role-indexed persistent 3D object records from RGB-D feed a whole-body action expert (GR00T N1.7 base, humanoid, 39/80 to 71/80). Abstract does not say how roles are bound, whether the head is restricted, or any held-out pairing eval. Pre-empts "role-indexed object tokens into an action expert, with persistence". Uses RGB-D tracking, not learned segments from VLM patches (not stated).
- SAM3D-guided object-centric alignment, 2607.25912. Training-time alignment of pi0 features to SAM3D object features. No pointer, no restriction. Not pre-empting.
- Encoded but Not in Control, 2610.06235. Diagnostic: instruction target is linearly decodable but not used; target-token attention stays on the scene-implied object. No fix in abstract. Strong motivation for RAA's diagnosis; not a competitor.
- Action QFormer 2607.14635: action supervision sharpens instruction-to-visual attention. Soft attention, no hard restriction (from search snippet only; not opened).
- ECoT 2407.08693 (CoRL 2024): policy writes object boxes and gripper position as reasoning text before actions. Pre-empts "ground the referent then act", but grounding is text tokens, the action head still sees the full scene (from search result, not opened this session).
- GTA-VLA 2605.13632 / FSD: points/boxes/traces as spatial CoT. Same category as ECoT (snippet only).
- RoboPoint, MOKA, PointVLA: not surfaced by search; known from memory as pointing/marks/point-cloud methods; not verified here.
- Harness VLA 2607.08448, ReGuide 2609.38616 (in NOVELTY.md): binding outside the VLA, symbolic or grounding module, action head not retrained with a pointer.

Verdict Q2: entity/slot tokens (OA-WAM, POT-VLA), grounding-then-act (ECoT, ReGuide) and "binding as an address" (OA-WAM abstract) exist. A learned instruction-conditioned two-role pointer whose output gates what the action expert sees is not stated in any abstract I opened.

## Q3. Self-supervised role labels from gripper contact events

Searched twice. Nothing found for "grasped entity at gripper close, target under held object at release" as a supervision signal for a language-role binding head. Nearest: Grasp2Vec (1811.06964, CoRL 2018) uses the object that ends up in the gripper as a self-supervised label linking pre-grasp scene to object embedding (no language roles, no place target); FOCI extracts interaction segments from demos (segmentation, not binding labels); VP-VLA and AC-VLA switch on gripper state (phase cue, not labels). Gripper-event segmentation of demos is a standard preprocessing trick, so the weak form is likely folk prior art even if I found no paper. Verdict: no direct pre-emption found; novelty is thin and rests on my search being incomplete. Grasp2Vec must be cited.

## Q4. Bottlenecks hiding non-active identity from the action head

- AC-VLA 2607.15714, LiLo-VLA 2602.21531 (in NOVELTY.md): masking during closed-gripper phase / non-target masking.
- TAG 2603.24584: runs the policy with the target erased and uses the difference as guidance (inference-time, not a bottleneck).
- See Less, Specify More (S2), 2606.02735 (arXiv, Jun 2026): trains the executor under an explicit "visual evidence budget" so it acts from task-sufficient evidence, no masks needed; pi0.5, real robots only (54.2 to 79.0). Abstract does not say identity is hidden. Closest to a generic information-bottleneck prior; weakens any broad "restrict what the action head sees" claim.
- OA-WAM address-only keys: identity separated from address inside attention (abstract only).
- MaskVLA, CGVD, OBEYED-VLA, Cloak (in NOVELTY.md): distractor masking/inpainting.

Verdict Q4: "show the action head only the relevant object" is prior art in several forms. The specific form (other entities as occupancy geometry without identity, active role switched by gripper) is the same idea as AC-VLA + LiLo-VLA with a learned pointer; only the geometry-without-identity detail is not in any abstract, and NOVELTY.md already tests silhouettes vs grey.

## Q5. 2026 papers combining two or more of Q1-Q4 with held-out pairing evaluation

None found that combines two or more AND reports held-out object-target pairings.
- Brittle Grounding 2602.24143: held-out object-region pairings, diagnosis only.
- CRAFT 2610.00524 and ACT-VLA 2607.00351: held-out pairings, data-side fixes, no architecture.
- LiLo-VLA: Q1 + Q4 (object frame + masking), evaluated on unseen long-horizon orderings, not pairings.
- OA-WAM and POT-VLA: Q2 (+ persistence), evals are LIBERO/SimplerEnv/LIBERO-Plus and real humanoid; no held-out pairing in the abstracts.
- FOCI: Q1 only, CoRL 2026.
- GloVLA 2609.06256: geometric transport, local VLA for interaction, LIBERO-Challenge clutter/perturbation; no held-out pairing in the abstract.
- Anchor-Align VLA 2607.13429: reports LIBERO-PRO object-swap numbers; fine-tuning regulariser, not architecture (snippet only).
Caveat: coverage of 2026 arXiv and CoRL 2026 is incomplete; ICLR 2027 submissions (OpenReview) were not searched.

## Verdict

Prior art (do not claim): object-relative action frames (TAX-Pose, SPOT, FOCI, LiLo-VLA); entity/slot tokens with identity-address split in a VLA action expert (OA-WAM, POT-VLA); ground-then-act (ECoT, ReGuide, Harness); masking or restricting vision to the active object by gripper phase (AC-VLA, LiLo-VLA, VP-VLA, TAG, S2); persistent role memory (POT-VLA role-indexed records, MEM, mu-VLA 2606.12497 recurrent memory, own F6 repaint).

Possibly novel (not seen in any abstract opened): (a) an instruction-conditioned two-role pointer trained from the demos' own contact events with no extra labels; (b) the pointer hard-gates a VLA action expert so its input is the active entity plus identity-free geometry; (c) a stated by-construction reduction of new pairings to trained inputs, tested on held-out pairings (CRAFT/LIBERO-PRO splits) against a plain retrain at matched data.

Adversarial reading: (a) is the weakest, since contact-event segmentation is routine and Grasp2Vec/FOCI sit next to it. (b) is a union of OA-WAM + FOCI/LiLo-VLA + AC-VLA; a reviewer can call it an assembly. OA-WAM and POT-VLA could pre-empt more than their abstracts show, and FOCI (CoRL 2026) is the closest object-frame paper; none was read in full.

Narrowest claim that survives: "In a VLA, the instruction-to-entity binding is learned from the demonstrations' own gripper-contact events and made a hard bottleneck, so the action expert sees only the active role's entity (frame plus pose/extent) and identity-free occupancy of the rest; we test whether this removes held-out object-target pairing failures against a matched-data plain retrain, ReGuide and CRAFT." The contribution is the controlled test of the combination on held-out pairings, not any single component. Do not claim object-frame actions, entity tokens, masking, or role-indexed persistence as new.

Before drafting: read in full FOCI (2609.08743), OA-WAM (2605.06481), POT-VLA (2607.18016), S2 (2606.02735), LiLo-VLA Sec. on reference frame; search OpenReview ICLR 2027 and CoRL 2026 lists for "role", "slot", "pointer" VLA; verify RoboPoint/MOKA/PointVLA.

Sources (abstract pages): arxiv.org/abs/2605.06481, 2607.18016, 2607.25912, 2610.06235, 2606.02735, 2609.06256, 2607.11498, 2603.24584, 2609.08743, 2411.00965, 2606.08775; TAX-Pose arxiv.org/abs/2211.09325 and ControlVLA arxiv.org/abs/2506.16211 via search results; Grasp2Vec arxiv.org/abs/1811.06964 via search result.


## Full-text reads (2026-11-16)

Method and limits. arXiv HTML fetched through a summarising fetch tool (small model, paraphrase; page truncated at 100k chars, second chunk read for OA-WAM and FOCI). POT-VLA read from HTML only in two passes; its PDF could not be text-extracted (no pdftotext). S2 page read once (no appendix). "Not stated" = not in the text I received. I did not get the OpenReview ICLR 2027 listing (submissions mostly not public; searches returned nothing); CoRL 2026 and arXiv coverage is by web search only. Queries contained no project information.

### FOCI Policy, 2609.08743 (CoRL 2026)
- Architecture: PointNet++ encoders on source-object and target-object point clouds, CLIP-ViT32 on the instruction; random token dropout, cross-attention source<->target, concat with language embedding, one transformer layer -> context H. Stage 1 GMM predicts terminal relative pose; stage 2 decodes waypoints. Head sees only the two object clouds plus language (not the scene). Action frame: relative SE(3) between source and target, T_des = T_ofs * T_rel * T_tgt (T_ofs = identity for gripper-object, gripper-to-grasped-object for object-object).
- Binding: source/target names come from a one-time per-task semantic specification (names + phase descriptions); objects located with Grounded SAM + FoundationPose. No learned instruction-to-role binding. Interaction segments by change-point detection on robot state / gripper open-close (the paper itself lists "identifiable contact event" as a difficulty axis). So gripper events are used for segmentation, not for training a language-role binder.
- Eval: RLBench 8 tasks (1 demo; avg ~74), COLOSSEUM 6 tasks (58.9 clean, 25.3 all perturbations; BridgeVLA/ActiveVLA 100 demos: 96.5/20.1, 96.9/21.4), RLBench-18 (43.7 overall; 79.6 easy-to-ground, 7.8 hard; 88.2 with GT poses), real 53-80% vs replay 7-27%. Ablations: two-stage 69.6 vs one-stage 59.5; no interaction segments 38.6. No held-out object-target pairing test (one qualitative cup-to-cup transfer), no matched-data plain retrain baseline.
- Reading for RAA: pre-empts object-relative frame + two-role restricted input + gripper-event segmentation, but it is not a VLA, roles are hand-specified per task, and there is no pairing hold-out.

### OA-WAM, 2605.06481
- Architecture: 7B Chameleon-style trunk; N+1 slots (robot + N objects), each 320-d = 32-d identity address + 256-d content + frame-time + role embedding (3 classes: robot/object/padding). Cross-slot attention keys read only the address subspace; a hook resets the address slice of the residual stream each layer. Action head = flow matching on a learned [act_q] token (16-step chunk, delta-space actions, residual end-effector trajectory).
- Binding: addresses from language label + initial DINOv3 feature; Qwen3-VL noun phrases prompt SAM 3, relation graph disambiguates. 4 soft role queries (target, reference, tool, distractor). Supervision: weak role KL loss (weight 0.05) to target/reference labels from "Path T1 plus language heuristics", annealed to 0 after step 50k; compose loss (distractor permutation/insertion invariance). Not contact events.
- Restriction: soft (address-keyed attention), not a hard gate to one object.
- Relative position: an SE(3) relative-geometry bias (6-scalar relative transform, translation + axis-angle) is added to slot-to-slot attention logits, on by default (~2K params/layer; removal cost ~2% on LP camera). This is a relative-pose channel between slots including the robot slot, but in the attention logits, not a role-conditioned state token.
- Eval: LIBERO 97.8, SimplerEnv WidowX 79.3, LIBERO-Plus (camera 80.5, robot init 89.6, Geo avg 84.3, 7-axis avg 83.9 vs pi0.5 85.7). Ablation V2 (same slots/data/trunk, no key mask/reset) is the matched control: LP camera 60.5 -> ~80.5, LP avg 76.2 -> 83.9, LIBERO +2.4; swap-binding cosine 0.06 vs 0.87. No held-out object-target pairing evaluation; no plain-retrain-at-matched-data comparison beyond V2.
- Reading: pre-empts entity slots + identity/address split + weak role supervision + relative SE(3) geometry in the head. Does not pre-empt hard restriction, contact-event labels, or pairing hold-out.

### POT-VLA, 2607.18016
- Architecture: GR00T-N1.7 action expert (Unitree G1). Role-indexed 3D object records (8 slots x 33 features: visibility, confidence, 2D box, 3D centroid, 3D extent, optional orientation, relations) inserted into the action head's self-attention sequence; the head also cross-attends to VLM hidden states (language + image), so it is NOT restricted to the bound object. Roles: TARGET, DESTINATION, SUPPORT, HANDOVER_PARTNER, REFERENCE, PADDING, from a typed-subtask plan (VLM planner or human-authored task file); how instruction phrases map to roles is not explained. Tracking by SAM3 + RGB-D.
- KEY: Table 3 relational features include end-effector-object offset, target-destination displacement, object-support relation, partner distance; metric fields stated in the robot base frame (relation features presumably too, not explicitly confirmed). This is a relative-position-to-referent channel in a VLA action head, role-indexed.
- Supervision: action loss only; no auxiliary object/predicate loss. No contact-event labels.
- Eval: 8 real task families 71/80 vs 39/80 direct baseline (same backbone, embodiment, runtime; no object tokens); ablation direct 15/40, verifier-only 22/40, tokens-only 31/40, full 34/40. Generalization (10 trials each): novel instances 9/10 vs 6/10, mid-execution perturbation 8/10 vs 4/10. No held-out pairing test; matched baseline is the same-data direct policy (not a "retrain" in the pairing sense).
- Reading: the most damaging pre-emption of the RAA state channel (role-indexed EE-to-object offset fed to an action expert). Caveat: unverified feature list; mixed with 2D/3D identity-free fields, includes extent and an identity-bearing visual stream, so no pairing claim.

### S2 (See Less, Specify More), 2606.02735
- Architecture: pi0.5 executor; lightweight learned gates produce soft masks over image tokens in base and wrist views; budget regulariser pushes mean retained evidence to rho = 0.2; loss = ungated + gated + budget. No mask/box labels. Language side: relabeled refined trajectory + ordered subtask instructions with original task instruction kept.
- No object-to-role binding, no object-relative frame, no state channel. Action frame unchanged.
- Eval: LIBERO-PRO strongest on object and goal perturbation axes; real TX-G2 and HSR, 8 tasks, subtask success 54.2 -> 79.0 (VLA-Adapter 0%); CALVIN ABC->D 3.95 vs 3.92 (language only). Cluttered unseen distractors 3/3 rollouts. No held-out pairing protocol, no matched-data retrain stated (baseline is pi0.5 fine-tuned).
- Reading: supports "restrict what the head sees" as prior art (soft, learned, unsupervised). Its masks are task-sufficient evidence, not role-addressed.

### Additional papers found this pass (new, abstract or one full read)
- w2VLA, "Decoupling the Declarative from the Procedural in VLA Models", arXiv 2606.21496 (Jun 2026, full HTML read). Splits the instruction into skill string and object string, encoded separately. "Where" = softmax heatmap of VLM patch embeddings against the object text embedding (temperature 0.05), flattened, MLP-projected, injected by FiLM into the proprioceptive state tokens; "what" = skill. Action expert also cross-attends to summarised VFM patches (patch masking p = 0.5), so not restricted. Binding comes from text-patch similarity, no labels. The target location is NOT stated as relative to the gripper (frame unstated). Eval is the one that matters for RAA: real SO-101, four 2-pair scenarios, 16 demos per (skill, object) pair, "transfer" = skill on the other object; avg seen / transfer: OTTER 93.8/30.6, pi0.5 95.8/38.2, w2VLA 95.1/91.7. Reversing where->what drops transfer 94.4 -> 55.6. Unseen target objects: -8.3. This is a held-out skill-object pairing test with a plain pi0.5 baseline, architecture-side, in the robot state token. It pre-empts the "hold-out pairing + architectural where/what split" framing, and is the single most dangerous paper for the RAA headline. Small scale (one embodiment, 8 pairs).
- ICT / Interaction-Centric Tokens, arXiv 2609.18243 (abstract only): end-effector pose trajectories relative to manipulated objects as tokens, jointly denoised with actions in flow matching; LIBERO +0.80, RoboTwin 2.0 +3.59, real +6.80, real OOD +7.45. Relative-to-referent signal as a supervised target rather than input; no pairing hold-out in the abstract. Needs a full read.
- "When Instructions Retrieve Trajectories", arXiv 2609.39971 (abstract only): ECT, a counterfactual data+loss fix; LIBERO-PRO position swap 36 -> 59, real UR5e unseen positions 8 -> 88. Data-side, no relative channel (as far as the abstract shows). Same category as CRAFT/ACT-VLA.
- LiLo-VLA (already listed): converts proprioception to a relative frame, approach pose relative to the reference object; another relative-pose-in-VLA precedent. SlotVLA 2511.06754, OAT-VLA 2509.23655 (object/agent tokens), PosA-VLA 2512.03724 (EE pose attention anchor), CorridorVLA (EE displacement field in head): abstract-level hits only, not opened; none claims role-addressed relative position plus pairing hold-out.
- Arm-wise compositional generalization 2610.06184 and MA-VLA 2608.25864: role (arm) binding in dual-arm VLAs, augmentation based; different role notion.
- OpenReview ICLR 2027: nothing retrievable. CoRL 2026: only FOCI and the above surfaced.

### Papers adding a relative-position-to-referent channel to a VLA's action head
1. POT-VLA: yes, explicit (EE-to-object offset, target-to-destination displacement; role-indexed tokens in action-head self-attention; base frame).
2. OA-WAM: yes, in attention (slot-to-slot SE(3) relative bias, includes robot slot), default on.
3. LiLo-VLA: relative proprioception frame wrt reference object (already in NOVELTY.md).
4. ICT 2609.18243: relative EE-object poses as jointly denoised tokens (output side).
5. FOCI: relative SE(3) is the entire output, not a VLA.
Not found: a paper whose single added channel is "active role entity position minus gripper position" with the role chosen by instruction and trained on held-out pairings against a plain retrain. Absence is weak evidence; I could not search OpenReview.

### Updated verdict on RAA-min (PLAN_PAPER 2b)
Components:
- R2 role views (head restricted to active-role entity): pre-empted in kind. FOCI (two object clouds only), AC-VLA/LiLo-VLA (masking), S2 (learned budget), OA-WAM (address-only keys, soft). What survives: hard restriction of a pretrained VLA head to the active role's entity, as a controlled variable. Not novel as an idea.
- One state channel = active role entity position minus gripper position: pre-empted. POT-VLA has the EE-object offset and target-destination displacement per role in an action head; OA-WAM has relative SE(3) between robot and object slots; LiLo-VLA relative frame; w2VLA puts target location into the state tokens (frame unstated). Claiming the channel itself as new will fail review.
- Roles from contact events: not pre-empted but weak. FOCI uses gripper open/close and change-point detection for segmentation; OA-WAM uses language-derived role labels (annealed away); POT-VLA uses a planner file. No paper uses grasp/release events to supervise an instruction-to-role binder. Survives as a design choice (label-free), but contact segmentation is routine and Grasp2Vec/FOCI are neighbours.
- Held-out pairings vs plain retrain at matched data: partly pre-empted. w2VLA (2606.21496) already shows skill-object transfer 91.7 vs 38.2 (pi0.5) on unseen pairings of seen skills and objects, with an architectural split. CRAFT/ACT-VLA/ReGuide/ECT cover the data/grounding side. OA-WAM and POT-VLA do not test pairings, so they do not pre-empt this part. What survives is the evaluation on standard benchmark splits (LIBERO-PRO/CRAFT) with a stronger, role-indexed mechanism, if the effect holds up.

Narrowest surviving claim: "A single hard-restricted role view plus a role-conditioned entity-minus-gripper offset, with roles bound by gripper-contact events and no entity labels, closes held-out object-target pairing failures of a VLA on [benchmark] at matched data, and we isolate which of (restriction, offset, contact-event binding) is responsible." Novelty is the ablation structure and the contact-event label source, not any component. A reviewer can legitimately say: POT-VLA + w2VLA + FOCI.

Adversarial points to deal with in the paper: (1) w2VLA must be a baseline or at least a cited comparison, since it is the same test (held-out skill-object pairs) with a large gap; if RAA-min does not beat or match 91.7 transfer in a comparable setting, the headline is gone. (2) POT-VLA's offsets are a direct ablation target: show RAA's gain is not just the offset by running offset-only (no restriction) and restriction-only. (3) OA-WAM's V2 vs V0 shows address-restriction gains mostly on robustness (camera, init) not on LIBERO in-distribution; expect reviewers to ask whether restriction helps pairings at all. (4) Because OA-WAM and POT-VLA were read from fetch-tool summaries with truncated pages, re-read their PDFs (OA-WAM appendix on action frame and relative bias; POT-VLA Table 3) before citing exact details. (5) Not yet read: ICT 2609.18243, SlotVLA, OAT-VLA, PosA-VLA, CorridorVLA, 2609.39971 in full; ICLR 2027 OpenReview list unsearched.

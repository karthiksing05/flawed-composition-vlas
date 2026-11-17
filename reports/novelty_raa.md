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

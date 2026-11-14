# A role-addressed structure for VLAs: proposal and vetting record (draft, 2026-11-12)

This document joins the failure account (docs/ARCH_FAILURES.md) to the structural fixes tested so far. Every claim cites a registered read. Sections marked PENDING wait on reads that are queued.

## 1. What the failures say about the architecture

| failure | where it breaks | what the architecture lacks |
|---|---|---|
| F1: the held object picks the destination (4 of 5 models; in a second scene, 0 of 720 new-pairing episodes on pi0.5, GR00T and pi0, cross-chunk excess +1.00 on pi0.5, and the swapped two-step task ends in the trained arrangement on .83-.98 of episodes; SR §50) | hold and release, after a correct grasp | the place reads the held object's identity, which the task does not need |
| F1m: the held object picks the motor program (N4, N6; SR §51, m3_prereg n46) | transport, after a correct grasp | the held wine bottle starts K4's high carry toward the cabinet (pi0.5 succeeds .04 after a high carry, .66 after a low one; GR00T carries high in 556 of 582 grasps). With the carry removed, frozen pi0.5 places 26 of 26 (PLACE PRESENT), so the place skill exists; GR00T N6 places 0 of 14 (data gap as measured) |
| F2: the habitual object is grasped (pi0, SmolVLA) | grasp | the instruction's object role loses to the scene's habit |
| F4: a new pairing cannot be written as parts | parse | no role vocabulary: retrieval returns trained wholes |
| F7: relations are ignored (GENERAL, SR §45.1: 3 of 4 model-family cells; the relational choice equals the ambiguous choice on 92-100% of states) | referent choice | no step that turns a relation into one entity |
| F3: new layouts fail at the grasp | grasp | reachability as measured: a planner with exact positions fails the same states; not architectural |
| F6: memory (PARTIAL on both pi0.5 and GR00T, SR §45.5-45.6, 140 states pooled): one view of the plate, then a physical cover, costs .38 (pi0.5) and .60 (GR00T), and success stays within .05 of the never-seen control; the equivalence bound misses .10 by .058 / .017 | transport and release toward a target that is out of view | no state that holds an entity's position once it leaves view; a memory slot that repaints the last seen target restores it (part 5) |

**No large effect found:** action chunk length (the intervals still allow +.15 on pi0.5 and +.23 on GR00T) and release timing (pi0.5's lower bound sits at 0). The partner-count law is not reproduced on two frozen VLAs.

**Scope of every claim here:** one LIBERO-Goal scene, one seed per retrained rung, 20-60 fresh states per confirmatory read. F1's behaviour has prior reports (CRAFT Table A.4; Failing Forward Table S9); what is new is the excess over trained partners, the hand-off stage and the five-model replication (docs/NOVELTY.md). Our records test consequences, not mechanisms.

## 2. The proposal

A VLA keeps its perception and its motor prior. Five structural parts route information between them, and none of them adds information.

1. **Role sequencer.** A proprioceptive hold event splits each task into an object phase and a target phase.
2. **Role phrases.** The prompt names only the active role.
3. **Role views.** Every non-active movable object is painted as a silhouette in both cameras, from a learned segmenter. Identity is hidden; shape and geometry stay.
4. **Referent resolver.** A relation in the instruction is resolved to one entity mask before the policy sees it; the other candidate is painted as a role view. Vetted on 2 relation families (SR §45.2). It covers fixed templates only and reads a parsed scene, so it is a routing fix, not a language-understanding result.
5. **Entity memory slot.** The segmenter's mask of the target and its depth-projected position are kept from the last frame where the target was seen. While it is hidden, the slot repaints it into both cameras at its remembered position, with occlusion by the gripper. Vetted on one occluder task (SR §45.6).

The same principles in their strongest form give the **hand-over variant**: the VLA acts until the hold event, then an object-blind parts policy places, reading learned perception.

## 3. Vetting so far

| part | test | result | label |
|---|---|---|---|
| role phrases | R1 − R0, matched retrain (SR §37.1) | grasp .41 → .93; success +.20 [+.025, +.433] | NO CLEAR DIFFERENCE |
| role views (simulator masks) | R2 − R0 (SR §37.2) | .025 → .425, +.40 [+.17, +.63] | STRUCTURE WINS |
| role views (learned masks) | r2l − r0 (SR §43.1) | +.39 [+.12, +.68] | STRUCTURE WINS |
| silhouettes vs boxes | R2b − R2 (SR §37.2) | −.275 [−.46, −.11] | boxes lose: shape must stay |
| silhouettes vs AC-VLA whole-wrist mask | R2 − R2w (SR §37.2) | +.18 [−.09, +.48] | NO CLEAR DIFFERENCE |
| role vocabulary for parsing | Q2-N (SR §42.1) | role rungs parse 5/6 new-pairing demos; plain cannot | PARSES NEW |
| hand-over, learned perception | HP − intact (SR §44.1) | GR00T .17 → .84; pi0.5 .50 → .85 [+.017, +.675] | GR00T BEATS THE VLA; pi0.5 NOT SHOWN |
| parts policy alone, learned masks | PU-L (SR §40.1) | .881 on new pairings | HOLDS against simulator masks |
| role views at test time on FROZEN VLAs (no training) | TV − I, new pairings (SR §46) | pi0.5 .50 → .725 [−.08, +.53]; GR00T .167 → .40 [+.025, +.47]; cross-chunk excess pi0.5 +.325 → +.054 | GR00T TV WINS; pi0.5 NO CLEAR DIFFERENCE. **GR00T confirmed on 60 new fresh states (SR §49, m3_prereg tv_g): .219 → .406, +.186 [+.033, +.361] TV WINS; trained tasks −.02 [−.07, +.02]; vs wrist grey +.236 [−.017, +.533] NOT SEPARATED (wrist grey costs .34 on trained tasks)** |
| role views + role phrases on frozen pi0.5 | TV+P − I (SR §46, descriptive, unregistered) | .50 → .925 [+.13, +.72]; trained tasks −.01; cross-chunk break gone | **CONFIRMED on 20 fresh states (SR §47, m3_prereg tv_fresh): .450 → .817, +.367 [+.092, +.634]; trained tasks −.010 [−.08, +.04] NO COST; cross-chunk excess +.450 → −.008.** On GR00T the phrases hurt (+.00 new, −.31 trained; not run fresh) |
| role views vs AC-VLA wrist mask on frozen VLAs | TV − W (SR §46) | pi0.5 +.18 [−.05, +.45]; GR00T +.24 [−.03, +.55] | not separated at an unmatched prompt; **SEPARATED at a matched prompt on pi0.5 (next row)** |
| attribution of the TV+P gain on frozen pi0.5, fresh states | TV+P − P and TV+P − W+P (SR §48, m3_prereg tv_attr) | P (phrases only) .475, W+P (wrist grey + phrases) .375, TV .700, TV+P .817; TV+P − P +.342 [+.050, +.667]; TV+P − W+P +.442 [+.208, +.675]; W+P costs .23 on trained tasks | **VIEWS ADD; SEPARATED from a whole-wrist grey proxy at a matched prompt.** The silhouettes carry the gain; phrases alone add +.025 [−.15, +.19]. W is a test-time proxy for AC-VLA, not its trained method |
| referent resolver + role view | F7R − relational (SR §45.2) | pi0.5 SP .40 → 1.00, NG .20 → .775; GR00T SP .30 → .85; GR00T NG capped by its own unique-name control (.50) | FIXED on 3 of 4 cells; resolver 160/160. On pi0, X-VLA, SmolVLA 0 of 6 FIXED, capped by each model's named-referent ceiling (SR §45.3) |
| role views in a SECOND scene (LIBERO-10 mugs × plates; simulator masks, the learned segmenter does not transfer) | tv / tvp − I, 60 states (SR §50, m3_prereg scene2) | pi0.5 tvp .000 [.000, .000], tv .000; GR00T tv .000; trained-half cost pi0.5 tvp −.117; on pi0.5 views move the first grasp to the named mug on about half of N2, but the mug still goes to its trained plate | **NOT REPLICATED (pi0.5 REFUTED)** |
| entity memory slot on frozen VLAs | M − b, plate seen once then covered, 120 new states (SR §45.6, m3_prereg f6o2) | pi0.5 .58 → .84, +.26 [+.15, +.35]; GR00T .14 → .28, +.14 [+.04, +.24]; recovers 67% / 24% of the cost of covering; overlay IoU median .987 | MEMORY RESTORED on pi0.5; on GR00T at the rule's boundary (M's lower bound equals b's upper bound, .208) and only 24% recovered, so treat it as weak |

## 4. What the vetting does not yet support

- **Resolved for pi0.5 by the frozen front end.** The retrained R2 (.425) trails frozen pi0.5 (.50), but the same structure applied at test time to frozen pi0.5 (role sequencer + role views + role phrases, no training) beats intact pi0.5 on new pairings, confirmed on fresh states (+.367, SR §47) at no trained-task cost. On GR00T, role views alone gain +.233 on main states (TV WINS, §46) and +.186 [+.033, +.361] on 60 new fresh states (TV WINS, §49) at no trained-task cost; the phrases hurt GR00T, so its front end is views only. The gain sits in N2, N3 and N5. N4 and N6 stay near 0 on GR00T and near intact on pi0.5 under every arm: the front end fires and paints correctly there (IoU .97-.99), but the held bottle still selects K4's high carry (F1m, SR §51). In the target phase the role views already paint the held bottle as a silhouette (tv_frozen's rule), so the carry is selected without the bottle's colour and texture: from its shape, the gripper's opening, or the arm's state. Hiding identity does not reach this route; which cue carries it is untested. GR00T's gain is half of pi0.5's, and its cross-chunk excess falls only from +.44 to +.25.
- **Separation from AC-VLA-style masking holds only at test time on pi0.5.** With matched phrases, silhouettes beat whole-wrist grey by +.442 (SR §48), and wrist grey with phrases also breaks the hold detector (it fires more than once in 65% of episodes). The retrained comparison R2 − R2w is still NO CLEAR DIFFERENCE, and the AC-VLA proxy is not its trained method. On GR00T without phrases, views beat wrist grey by +.236 but the lower bound misses 0 by .017 (NOT SEPARATED, SR §49); wrist grey there costs .34 on trained tasks, views .02.
- **Most fixes have prior art; the diagnosis and two narrow tests are new.** The hand-over variant and the frozen test-time front end overlap ReGuide, LiLo-VLA, Harness VLA and AC-VLA (no priority claim), while the stage-resolved five-model atlas, silhouettes against whole-wrist grey at a matched prompt on pi0.5, and the frozen-VLA memory slot have no direct prior; see docs/NOVELTY.md.
- **Everything rests on one scene,** with one seed per retrained rung. The failure atlas replicates on 20 fresh states (F1 on 4 of 4 models, F2 on 2 of 2; ARCH_FAILURES "Fresh-state replication"). The frozen front end is confirmed on fresh states (§47, §49); the retrained rungs (R0-R2, HP) are not re-run. **The second scene splits the story (SR §50):** the failure replicates at full strength (0 of 720; cross-chunk excess +1.00 on pi0.5), and the fix does not replicate (role views 0 on both models). There, two mugs share one shape, so hiding identity leaves only position and the trained two-step sequence, and the policies rebuild the trained end arrangement. Identity-hiding views fix F1 only where identity is the cue that routes the place; where the cue is the scene's trained arrangement, no test-time view tested here reaches it. That makes the failure deeper than the fix, which supports the failure account and limits the proposal. We have not run ReGuide on our states (running, tag reguide).
- **F7R fixes only where the policy can act on a named referent.** On pi0, X-VLA and SmolVLA it is FIXED on 0 of 6 cells (SR §45.3): the resolver is exact (240/240), but the models fail the named plate bowl outright (0/20 each), so the front end reaches each model's own ceiling and stops. Two of the four parts (role views, referent resolver) remove routing failures; neither removes a scene habit the policy holds when the referent is named. Painting every distractor, the proposal's own role-view rule, does not lift that ceiling either (0 of 7 cells; SR §45.4): the policies stop grasping the cheese and grasp nothing instead. The referent resolver covers fixed templates only.
- **The memory slot is vetted on one task and is partial on GR00T.** M still trails the never-covered control d (pi0.5 −.125, GR00T −.46). The plate leaks back into view more often under M (45-49% of later queries) than under b (34-43%), so late views may carry part of the gain. The occluder rule changed twice after policy-free checks, before any rollout.
- **New layouts are unsolved by every system.**

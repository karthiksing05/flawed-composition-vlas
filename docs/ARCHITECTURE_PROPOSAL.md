# A role-addressed structure for VLAs: proposal and vetting record (draft, 2026-11-12)

This document joins the failure account (docs/ARCH_FAILURES.md) to the structural fixes tested so far. Every claim cites a registered read. Sections marked PENDING wait on reads that are queued.

## 1. What the failures say about the architecture

| failure | where it breaks | what the architecture lacks |
|---|---|---|
| F1: the held object picks the destination (4 of 5 models) | hold and release, after a correct grasp | the place reads the held object's identity, which the task does not need |
| F2: the habitual object is grasped (pi0, SmolVLA) | grasp | the instruction's object role loses to the scene's habit |
| F4: a new pairing cannot be written as parts | parse | no role vocabulary: retrieval returns trained wholes |
| F7: relations are ignored (GENERAL, SR §45.1: 3 of 4 model-family cells; the relational choice equals the ambiguous choice on 92-100% of states) | referent choice | no step that turns a relation into one entity |
| F3: new layouts fail at the grasp | grasp | reachability as measured: a planner with exact positions fails the same states; not architectural |
| F6: memory (PARTIAL on both pi0.5 and GR00T, SR §45.5): one view of the plate, then a physical cover, costs .50-.60, and success equals the never-seen control within .05; the equivalence bound misses at n = 20 | transport and release toward a target that is out of view | no state that holds an entity's position once it leaves view; untested fix: a role-view slot keeps the last seen target mask |

**Ruled out as causes:** action chunk length, release timing, and the partner-count law.

## 2. The proposal

A VLA keeps its perception and its motor prior. Four structural parts route information between them, and none of them adds information.

1. **Role sequencer.** A proprioceptive hold event splits each task into an object phase and a target phase.
2. **Role phrases.** The prompt names only the active role.
3. **Role views.** Every non-active movable object is painted as a silhouette in both cameras, from a learned segmenter. Identity is hidden; shape and geometry stay.
4. **Referent resolver.** A relation in the instruction is resolved to one entity mask before the policy sees it; the other candidate is painted as a role view. Vetted on 2 relation families (SR §45.2). It covers fixed templates only.

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
| referent resolver + role view | F7R − relational (SR §45.2) | pi0.5 SP .40 → 1.00, NG .20 → .775; GR00T SP .30 → .85; GR00T NG capped by its own unique-name control (.50) | FIXED on 3 of 4 cells; resolver 160/160. On pi0, X-VLA, SmolVLA 0 of 6 FIXED, capped by each model's named-referent ceiling (SR §45.3) |

## 4. What the vetting does not yet support

- **R2 does not beat frozen pi0.5** (.425 against .50). It was trained on 8 tasks with a frozen VLM.
- **R2 is not separated from AC-VLA-style masking.**
- **The hand-over variant overlaps ReGuide and LiLo-VLA** (reports/novelty_check_failure_atlas.md).
- **Everything rests on one scene,** with one seed per rung. The failure atlas replicates on 20 fresh states (F1 on 4 of 4 models, F2 on 2 of 2; ARCH_FAILURES "Fresh-state replication"); the fix rungs have not been re-run on fresh states.
- **F7R fixes only where the policy can act on a named referent.** On pi0, X-VLA and SmolVLA it is FIXED on 0 of 6 cells (SR §45.3): the resolver is exact (240/240), but the models fail the named plate bowl outright (0/20 each), so the front end reaches each model's own ceiling and stops. Two of the four parts (role views, referent resolver) remove routing failures; neither removes a scene habit the policy holds when the referent is named. Painting every distractor, the proposal's own role-view rule, does not lift that ceiling either (0 of 7 cells; SR §45.4): the policies stop grasping the cheese and grasp nothing instead. The referent resolver covers fixed templates only.
- **New layouts are unsolved by every system.**

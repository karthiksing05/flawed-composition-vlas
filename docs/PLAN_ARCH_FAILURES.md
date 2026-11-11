# Architecture-rooted failures of current VLAs: plan (draft for the user, 2026-11-11)

The user, 2026-11-11: "isolating problems with existing VLAs ... a broad account of unsolvable failures ... identifying problems with existing VLAs that are deeply rooted in the architecture (with recency bias to CoRL)". Fixes are a bonus.

## 1. What counts as architecture-rooted

A failure is architecture-rooted when all three hold.

1. **The information is there.** The input holds everything the task needs, and a simple reference system (scripted servo, oracle parts policy) succeeds on the same states.
2. **It persists inside the architecture.** It appears across current VLAs and survives more data, longer training, a different recipe, or prompting. "Unsolvable" means unsolvable by those levers.
3. **A structural change removes it.** The change adds no information: routing, masking, a hand-over. This proves the information was usable and the architecture did not use it.

A failure that fails (1) belongs to the task or the robot. A failure that fails (2) belongs to the training recipe. Both are reported, and neither is called architectural.

## 2. The classes, with the record so far

Status: **E** established here, **P** partial, **O** open. "CoRL" lists independent 2026 support (reports/CoRL 2026 VLA literature review.md).

| | failure | architectural locus | evidence here | status | CoRL / 2026 |
|---|---|---|---|---|---|
| F1 | **The held object picks the destination.** On new pairings both VLAs grasp right, then end the hold early or release badly | one token stream; the action expert attends to everything, and the carried object is visible in every carry frame | atlas: HANDOFF +.31/+.39, RELEASE +.16/+.38 (BREAK_ATLAS §7). Gate: NOT TIMING. Hand-over: .50→.96, .17→.93 (HAND_OVER §6.1). Retrained R0 .025, role views R2 .425 (SR §37.1) | **E** (1, 3); (2) only for our recipe | CRAFT share-object 38%, LIBERO-CF, Brittle Grounding (100k demos give no gain), Grant et al. |
| F2 | **The scene picks the object.** Narrowly trained, the VLA grasps a habitual object | vision dominates language in the joint prefix | R0 grasps the right object .41 on new pairings; role phrases raise it to .93 (SR §37.1) | **P**: shown on our retrain; frozen pi0.5/GR00T grasp right .95-.99 | LIBERO-CF, LangForce "information collapse" |
| F3 | **Motor programs bound to absolute position.** In new layouts every system fails at the grasp, by start region | the expert's motion is tied to workspace coordinates (Grant et al.) | new layout ≈ .30 for VLAs; but servo and P_min fail on the same states (BREAK_ATLAS §8.2) | **O**: fails criterion (1) as measured; needs a test where the reference succeeds | Grant et al. (absolute coordinates); MESA Spatial |
| F4 | **No addressable parts.** A new pairing cannot be expressed as a combination | no part store; retrieval returns trained wholes | Q2-N: plain r0 parses 0/6 new-pairing demos; role rungs 5/6 (SR §42.1) | **P** | ReGuide's habit diagnosis; ACT-VLA needs new data |
| F5 | **Chunking and open loop are not the cause** (negative) | action chunks | T1: NO GAIN at 2-10 actions (SR §32.1) | **E**, as a ruled-out cause | |
| F6 | **No memory.** Tasks needing state that is no longer visible fail | a Markov policy over a short window | none here | **O** | PI's MEM (CoRL 2026); memory benchmarks |
| F7 | **Relational and negated language** ("not the bowl", "the left one") is ignored | language read as a bag of cues by the action path | none here; LIBERO-PRO's language perturbations are on disk | **O** | LIBERO-Plus, LangForce |

**Corrected 2026-11-11 by the consolidation (docs/ARCH_FAILURES.md):**
- **F1 is partial, not established.** Its criterion (3) holds only with privileged inputs: the hand-over reads oracle positions, and R2 reads simulator masks. R2 − R1 does not isolate identity hiding (+.200 [−.058, +.458]). Criterion (2) on more data is untested.
- **F2 is recipe-rooted as measured.** Frozen pi0.5 and GR00T grasp right at .95-.99.
- **F3 is a layout failure that fails criterion (1).** GR00T's new-layout success is .08, not .30.
- **F4's "0/6" is zero by construction.** No frozen VLA was parsed, and the role rungs add vocabulary as well as structure.
- **F5 rules out large effects only.**
- **LIBERO-CF supports F2, not F1.** Brittle Grounding's failure sits at the reach.

## 3. Models

- pi0.5 (openpi, frozen `pi05_libero`) and GR00T N1.x (ar0s), already served.
- pi0 (openpi).
- X-VLA (robot/ext/X-VLA, CoRL-era; serving to check).
- FLOWER on CALVIN, for a second benchmark.
- GR00T 2 when its code ships (December), as the world-action-model line.

A class counts as general when it holds on at least three of these.

## 4. Work, cheapest decisive first

1. **Consolidate (CPU).** One atlas document, docs/ARCH_FAILURES.md, holding §2's table with every number traced to its ledger row, plus each class's three criteria marked met, unmet or untested.
2. **F1 and F2 across models.** Rerun the atlas harness (N1-N6, K1-K6, both layouts, N2 main states) on pi0 and X-VLA. That is about 7 A40-h per model at the atlas's rate.
3. **F1, criterion (2).** Does more data fix it inside the architecture? Retrain R0 on 8 tasks, then on more LIBERO tasks sharing the scene, and read whether new-pairing success moves. Brittle Grounding predicts it does not.
4. **F3, criterion (1).** Find layouts where the scripted servo succeeds and the VLAs fail. Shift the whole scene rigidly, with the camera fixed, so reachability is unchanged. If VLAs still fail, position binding is architectural.
5. **F6 and F7.** One diagnostic each on LIBERO:
   - F6, memory: occlude the target after the instruction, or "put it back where it was".
   - F7, language: LIBERO-PRO's negation and spatial-relation variants.
   Each runs against a reference that succeeds.
6. **Bonus fixes,** already running: role views (SR §37), the AC-VLA control (§41), the hand-over without simulator state (§44).

## 5. The user's decisions (2026-11-11)

- F6 (memory) and F7 (relational and negated language) are in scope.
- "Let's run with whatever exists right now, and iterate on GR00T 2 when it drops."

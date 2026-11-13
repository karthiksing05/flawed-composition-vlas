# Architecture-rooted failures of current VLAs: the consolidated record (2026-11-11)

This document implements step 1 of `docs/PLAN_ARCH_FAILURES.md`. It restates that plan's classes F1-F7 and traces every number to the document and section that holds it. It runs no experiment and changes no verdict. The user's direction (2026-11-11) is to isolate problems with existing VLAs that sit deep in the architecture, with a recency bias to CoRL 2026.

Section marks: BA = BREAK_ATLAS, HG = HOLD_GATE, HO = HAND_OVER, RT = RIVAL_TESTS, SR = SCAFFOLD_RESULTS, TC = THEORY_COMPOSITION, LR = `reports/CoRL 2026 VLA literature review.md`, PAF = PLAN_ARCH_FAILURES. All files sit in `docs/` except LR. `docs/PAPER_STYLE.md` does not exist, so the prose follows the rules given for this task.

## Update 2026-11-12: the atlas on five VLAs (m3_prereg pp_atlas_more; results/flawed/pp_atlas_more_read.json)

BREAK_ATLAS's registered estimator and mapping, unchanged, were run on three more frozen LIBERO-trained VLAs: pi0 (`pi0_libero`), X-VLA (`xvla-libero`) and SmolVLA (`smolvla_libero`). The read reproduces the registered pi0.5 and GR00T numbers exactly. Familiar layout, N1-N6 against their known partners:

| model | within-chunk E (grasp side) | cross-chunk E (hold and release) | verdict |
|---|---|---|---|
| pi0.5 | +.175 [.05, .33] | +.325 [.13, .53] | BOTH |
| GR00T N1 | +.392 [.15, .66] | +.429 [.19, .67] | BOTH |
| pi0 | +.708 [.47, .93] | +.254 [.05, .48] | BOTH |
| X-VLA | +.450 [.13, .81] | +.392 [.06, .73] | BOTH |
| SmolVLA | +.792 [.63, .93] | +.083 [−.03, .27] | WITHIN-CHUNK |

The new layout reads NEITHER on all five.

**F1, criterion (2): F1 GENERALISES.** Its cross-chunk break clears the bar on pi0 and X-VLA, so on 4 of 5 models. Every registered prediction held.

**F2 changes status.** The grasp-side excess is large on pi0 (GRASP E +.64) and on SmolVLA (+.78): frozen VLAs trained on 40 LIBERO tasks grasp the habitual object on new pairings. F2 now meets criterion (2) on 2 of 5 models. pi0.5 and GR00T show the place-side failure instead. Which failure a VLA shows varies by model; that it fails on new pairings in the familiar layout does not.

**Fixes (SR §37.2).** Role views raise a matched VLA on new pairings, R2 − R0 = +.40 [+.17, +.63]. Silhouettes beat boxes (−.275). They do not clearly beat an AC-VLA-style whole-wrist mask (+.18 [−.09, +.48]).

## Fresh-state replication (2026-11-12)

(m3_prereg pp_atlas_fresh, registered before any fresh rollout; results/flawed/pp_atlas_fresh_read.json)

The five-model atlas was re-run on 20 new familiar-layout start states, k = 36-55, drawn by the same sampler rule. The rule reproduced stored states 0, 1, 23, 24 and 35 exactly. The models, tasks (K1-K4, K6, N1-N6), estimator, partners and labels are unchanged. Familiar layout, fresh states:

| model | within-chunk E | cross-chunk E | GRASP E | verdict | development verdict |
|---|---|---|---|---|---|
| pi0.5 | +.075 [.00, .18] | +.450 [.28, .61] | +.000 [.00, .00] | CROSS-CHUNK | BOTH |
| GR00T N1 | +.308 [.08, .57] | +.500 [.26, .74] | +.017 [.00, .07] | BOTH | BOTH |
| pi0 | +.733 [.53, .92] | +.225 [.05, .43] | +.650 [.39, .88] | BOTH | BOTH |
| X-VLA | +.492 [.17, .83] | +.379 [.04, .73] | +.192 [.00, .53] | BOTH | BOTH |
| SmolVLA | +.862 [.73, .96] | +.054 [−.00, .14] | +.833 [.67, .96] | WITHIN-CHUNK | WITHIN-CHUNK |

**F1 REPLICATES.** The cross-chunk break clears the bar (CI lower bound > 0 and E > .10) on all four models where it held on the development states: pi0.5, GR00T, pi0 and X-VLA. The bar was 3 of 4.

**F2 REPLICATES.** The GRASP excess clears the same bar on both pi0 (+.650) and SmolVLA (+.833).

**Verdicts match on 4 of 5 models.** pi0.5 moves from BOTH to CROSS-CHUNK: its within-chunk E falls from +.175 to +.075, below the .10 bar. The registered prediction (both replicate, at least 4 of 5 verdicts match) held.

## 0. The three criteria

A failure is architecture-rooted when all three hold (PAF §1).

1. **The information is there.** The input holds what the task needs, and a simple reference system succeeds on the same states.
2. **It persists inside the architecture.** It appears across current VLAs and survives more data, longer training, a different recipe, or prompting.
3. **A structural change removes it.** The change adds no information: routing, masking, a hand-over.

A failure that misses criterion 1 belongs to the task or the robot. A failure that misses criterion 2 belongs to the training recipe.

Marks: **MET**, **UNMET**, **UNTESTED**. Where a criterion is met on part of its scope, the mark names the part.

## 1. Summary table

"PAF status" is the status PAF §2 assigns (E established, P partial, O open). "This document" is the status the cited records support.

| class | claim | C1 information there | C2 persists | C3 structure removes it | PAF status | this document |
|---|---|---|---|---|---|---|
| F1 | The held object picks the destination | MET | MET on two frozen VLAs and on chunk length; UNTESTED for more data | MET in the familiar layout, with privileged inputs | E (1, 3) | Partial: C1 and C3 met, C2 open on data |
| F2 | The scene picks the object | MET | UNMET: frozen pi0.5 and GR00T grasp right at .95-.99 | MET on our retrain (R0 .408 to R1 .925), no CI in SR §37.1 | P | Recipe-rooted as measured |
| F3 | Motor programs bind to absolute position | UNMET: servo and P_min fail on the same states | UNTESTED | UNTESTED | O | Open, and fails C1 as measured |
| F4 | No addressable parts | MET (flow-loss retrieval over role phrases 6 of 6 on r1) | UNTESTED on any frozen VLA | UNTESTED: the role rungs add the phrase vocabulary | P | Open; thinner than PAF states |
| F5 | Chunking is not the cause | n/a | n/a | n/a | E (ruled out) | Ruled out for large effects (SR §32.1) |
| F6 | No memory | UNTESTED | UNTESTED | UNTESTED | O | Open, no local evidence |
| F7 | Relational and negated language is ignored | UNTESTED | UNTESTED | UNTESTED | O | Open, no local evidence |

No class meets all three criteria. F1 comes closest.

**Update 2026-11-12 (supersedes the rows above where they differ).**

| class | C1 information there | C2 persists | C3 structure removes it | status |
|---|---|---|---|---|
| F1 | MET | MET across models: cross-chunk break on 4 of 5 VLAs, replicated on fresh states (4 of 4); UNTESTED for more data | MET in the familiar layout (R2, HP; SR §37.2, §44.1) | All three met within one scene; data scaling open |
| F2 | MET | MET on pi0 and SmolVLA (GRASP E +.65 / +.83 fresh), UNMET on pi0.5 and GR00T | MET on our retrain (R1) | Model-dependent: architecture-rooted on two of five VLAs |
| F6 | UNTESTED: hiding the plate breaks both VLAs with no memory needed (SR §45.1) | — | — | Open; needs a physical-occluder design |
| F7 | MET: unique-name controls .825-1.00, references .93-1.00 (SR §45.1) | MET on pi0.5 (both families) and GR00T (SP): the relational choice equals the ambiguous choice on 92-100% of states | MET: F7R (symbolic resolver + learned-segmenter role view, no new information) FIXED on 3 of 4 cells, resolver 160/160 (SR §45.2) | All three met on 3 of 4 cells; GR00T NG capped by its own unique-name control |
| F7P | negation (LIBERO-PRO): trained .95, negated 0/20 on both | — | UNTESTED | Descriptive |

F7 is the cleanest architecture-rooted failure in the record: the same frozen VLAs that succeed when the referent is named fail when it is described, and a structural front end that adds no information removes the failure on pi0.5 and GR00T.

**Extension to pi0, X-VLA, SmolVLA (SR §45.3, m3_prereg fx_more).**
- **F7 EXTENDS** (RELATION IGNORED on 4 of 6 cells; X-VLA SP and SmolVLA SP are NO POWER, U .10 / .05). The relational choice equals the ambiguous choice on 75-100% of states. F7 now holds on 5 of 5 VLAs on at least one family.
- **F7R DOES NOT EXTEND** (0 of 6 FIXED). The resolver is exact (240/240), so the limit sits in the policies:
  - On NG every model, given the plate bowl alone by name, succeeds on 0 of 20 (pi0 and SmolVLA grasp the cream cheese, as GR00T did). F7R reaches that ceiling (.50) on all three, and the name-only prompt already gets there.
  - pi0 SP is the one cell where the role view helps (F7R − name +.225), PARTIAL only on C's lower bound (.725 against .75).
- **Reading.** F7R fixes the relation step wherever the policy can act on a named referent. Where it cannot, the remaining failure is the scene habit (F2: the plate bowl's habitual partner, or the cheese, wins), which neither naming nor painting the distractor removes. F7's criterion 3 is MET on 3 of 10 cells and capped by F2 on the rest.
- **Painting every distractor does not lift the cap (SR §45.4, m3_prereg fx_ra).** f7rA is FIXED on 1 of 7 cells (pi0.5 NG, as before) and lifts the named-referent ceiling on 0 of 7. Painting the cream cheese stops the cheese grasps (pi0 14 → 3, SmolVLA 7 → 0), but the policies then grasp nothing: a bowl resting on the plate is picked up on 0/20 by pi0 and SmolVLA even when it is the only unpainted movable object. On GR00T SP paint-all costs −.175 [−.30, −.05] against F7R.
- **Reading.** The residual is a motor habit tied to the bowl's resting support: in LIBERO-Goal's training tasks no bowl is ever picked off the plate. It is plausibly a data gap, so it does not meet criterion 2 until a model trained with such demos is tested. Role views and the resolver cannot reach it, because the information already reaches the policy.

## 2. Ruled-out causes

### 2.1 Chunking (F5; SR §32.1, results/flawed/sc_t1_read.json, rollouts sc_t1, m3_prereg sc_t1)

Success on new pairings, familiar layout, against the frozen read at 5 actions per query:

| model | 2 per query | 10 per query |
|---|---|---|
| pi0.5 (.500 at 5) | .450, difference −.050 [−.250, +.150] | .458, difference −.042 [−.200, +.100] |
| GR00T (.167 at 5) | .158, difference −.008 [−.117, +.100] | .250, difference +.083 [−.025, +.233] |

All four settings read NO CLEAR CHANGE, and the overall verdict is NO GAIN FROM CHUNKING. The right object is grasped at .95-.99 in every setting. The intervals allow an effect up to about +.15 for pi0.5 and +.23 for GR00T, so the test rules out a large gain and leaves a modest one open. Replanning at 1 action stayed out for cost.

### 2.2 The partner-count law (TC §2; SR §35.1, results/flawed/sc_e0_read.json, ledger row sc_e0 at c51e5412, m3_prereg sc_e0)

TC §2 proposed that a VLA composes when training broke the tie between object and target, so the bottle pairings (two training targets) should beat the cheese pairings (one). E0 returned REFUTED. GR00T reverses the order at every setting:

| arm | cheese | bottle | gap [95% CI] | label |
|---|---|---|---|---|
| GR00T, 5 per query | .333 | .000 | −.333 [−.433, −.250] | REVERSED |
| GR00T, 2 | .317 | .000 | −.317 [−.433, −.217] | REVERSED |
| GR00T, 10 | .483 | .017 | −.467 [−.567, −.383] | REVERSED |
| GR00T, pooled | .378 | .006 | −.372 [−.428, −.317] | REVERSED |
| pi0.5, 2 | .367 | .533 | +.167 [.000, +.333] | UNCLEAR |
| pi0.5, 10 | .300 | .617 | +.317 [+.167, +.467] | HOLDS |
| pi0.5, 5 (seen before; descriptive) | .300 | .700 | +.400 [+.233, +.583] | |

TC §2 withdraws the sentence "a compositional architecture should need one partner where the VLA needs two or more". SR §35.1 withdraws it as a law about objects. Two descriptive observations remain. N3 (cheese on the cabinet, the one target with two holders in training) beats the mean of N1 and N2 in all six rows, by +.43 to +.60; SR §35.1 flags this as an observation with one old row. The bottle on the plate and the stove (.85-.90 on pi0.5, targets with one holder) breaks any count rule that fits both models. The confound that the cheese may differ from the bottle in shape stays open, and E2 (a second partner for the cheese) would separate them. LR's open question "partner-count law on pretrained VLAs" matches this record: Wang et al. (arXiv 2607.29687) show a coverage law only on a small custom policy, and E0 is a pretrained-VLA point that fails to show it.

### 2.3 Release timing alone (HG §5, results/flawed/pp_gate_read.json, m3_prereg pp_gate)

The gate forces the gripper shut while the object is held and farther than 0.05 m from the target. Familiar layout, N1-N6:

| model | intact | G (gate) | GA (gate and known target prompt) | G − intact | GA − G | known tasks, intact to G |
|---|---|---|---|---|---|---|
| pi0.5 | .50 | .63 | .68 | +.13 [.00, +.35] | +.05 [−.09, +.21] | 1.00 to 1.00 |
| GR00T | .17 | .23 | .28 | +.06 [−.02, +.15] | +.05 [−.27, +.33] | .98 to .98 |

The registered verdict is NOT TIMING for both VLAs. For pi0.5 the point estimate clears 0.10 and the CI lower bound sits exactly at 0, which the rule does not count as above 0; HG §5 calls the miss of the author's TIMING prediction narrow. The share of failures that end with the object still held and more than 0.05 m from the target rises from .03 to .28 on pi0.5 and from .29 to .66 on GR00T (HG §5.1). The new layout does not move (pi0.5 .30, .32, .30; GR00T .08, .09, .08).

Two follow-up reads (HO §6.3, results/flawed/pp_gate_reads2.json, m3_prereg pp_gate_reads) show that the gated policy stops short or wanders. pi0.5 STALLS (21 of 33 failures under G, median net motion 0.011 m, final `d` 0.20 m, none ever within 3 cm). GR00T KEEPS MOVING (17 of 79 stall, median net motion 0.060 m, final `d` 0.14 m, 8% ever within 3 cm). The author predicted STALLS on both, so the GR00T half of the prediction failed.

HG §5.2 reads the result post hoc: opening early is a symptom, failures move from the cross-chunk column to the within-chunk column, and the VLA has no reliable way to finish this object at this target.

### 2.4 Other readings withdrawn or corrected

| reading | where | correction |
|---|---|---|
| "The cheese arrives and misses by centimetres" and cross-chunk composition "mostly intact" on the cheese | RT §4.1 | Withdrawn by BA §6.3. Most N1 and N2 failures end their hold well short of the target, and a final position near the target does not show a carry. |
| HANDOFF means "held and never went to the target" | BA §2 | BA §6.3 Check A: 18 of pi0.5's 37 HANDOFF episodes (0.49) end within 0.10 m of the target (AMBIGUOUS), and 2 of GR00T's 47 (0.04) do (LABEL HOLDS). The accurate reading is "the hold ends before the object is carried to the target". |
| pi0.5's cross-chunk excess is a real drop | BA §8.3 Read C | MIXED on both models: 12 of 18 pi0.5 episodes are missed carries (N2 median carried share 0.89; N1 0.04 and N3 0.10). Part of pi0.5's cross-chunk E is the hold signal losing a carry. GR00T has only 2 such episodes. |
| E is a pure interaction | BA §6.4 | If a stage's rate is additive in object and target, E equals half the object difference plus half the target difference. The control also mismatches objects at GRASP. The VLA verdicts show no visible effect. |
| pi0.5 "0.99 to 0.38, g 0.79, p 0.47" | RT §3.1 | That figure is N2's "new" cell over N1-N8, which includes drawer and push tasks. On the six pick-and-place pairings, intact is s .50, g .97, p .52. |
| Additive offset account of the miss | RT §5.4 | Call 1 returns no (β1 = 0.26 [0.19, 0.33]). Call 2 could not test its target because |q| = 0.009 m, below the 0.02 m floor; the author names this a design error. UNRESOLVED. |
| The new layout's grasp failures come from a state-locked ceiling | BA §6.3, §8.3 | Check B NOT STATE-LOCKED (|L| = 2, |C| = 2). Read D NOT LOCKED (cheese |L| = 8, |C| = 7; bottle |L| = 6, |C| = 4). |
| The servo's new-layout grasp failures come from its 40-step timer | HO §7.4, §8.3 | Read G returns BLOCK: hold rate 0 of 10 states for the cheese and 0 of 10 for the bottle at 120 steps as at 40. The block holds for the demo's fixed top-down grasp and not necessarily for every approach. |
| U's new-layout success is .381 | TC §2 table | SR §38.1 gives .378 (oracle) on new pairings, new layout. The two documents disagree by .003. |
| A lack of position information explains the failure | HO §9.1 | In the familiar layout every target and object position lies within 0.02 m of a constant (95th percentile radii .000-.019 m). The VLAs place these targets at .98-1.00 on known tasks. |

## 3. The classes

### F1. The held object picks the destination

**Claim.** On new pairings both frozen VLAs grasp the right object, then end the hold before the object reaches the target or release it badly.

**Architectural locus.** One token stream, with an action expert that attends to everything. The carried object is visible in both cameras in every carry frame (TC §1, P1). This is a mechanism proposed by the theory. The records below test its consequences and have no direct probe of the attention path. Grant et al. (arXiv 2603.19233) give the closest mechanistic evidence in LR.

**Criterion 1, information is there: MET (familiar layout).**
- Known tasks run at .98-1.00 under both VLAs (HG §5, HO §6.1).
- P_min alone on seed 10 scores .92 on N1-N6 familiar (HO §6.1). U scores .894 on new pairings from simulator poses and .883 from camera-perceived poses (SR §38.1, results/flawed/sc_pu_read.json, ledger row sc_pu). U scores .881 from a learned segmenter (SR §40.1, results/flawed/sc_seg_pu_read.json, ledger row sc_seg_pu).
- Positions need no tenfold perception in the familiar layout: the 95th percentile radius of every target and object is at most .019 m (HO §9.1, results/flawed/pp_read_h.json).
- The new layout fails this criterion for any pairing claim (see F3).

**Criterion 2, persists: MET on models and on chunk length; UNTESTED on more data.**
- New pairings, familiar layout: frozen pi0.5 .50, frozen GR00T .17 (HG §5; BA §5 verdict BOTH). Atlas excess over known partners: pi0.5 within-chunk +0.18 [+0.05, +0.33], cross-chunk +0.33 [+0.13, +0.53]; GR00T within-chunk +0.39 [+0.15, +0.66], cross-chunk +0.43 [+0.19, +0.67] (BA §5, results/flawed/pp_atlas_read.json, m3_prereg pp_atlas). Stage level: HANDOFF +0.31 (pi0.5), +0.39 (GR00T); RELEASE +0.16, +0.38 (BA §7). BA reports no CI at stage level.
- Our retrain (frozen VLM, trainable action expert, 8 known tasks, one seed) reaches .025 on new pairings (SR §37.1, results/flawed/sc_t37_read_r012.json, m3_prereg sc_t37). Fit gate passes (R0 .85 on known K1-K6 familiar, threshold .80).
- Chunk length: no gain (SR §32.1, §2.1 above).
- Prompting: V1 arm A raises p by +0.03 [−0.16, +0.23] (RT §3.1, results/flawed/pp_v1_read.json, m3_prereg pp_rival1). The interval contains a gain above 0.10, so this shows no recovery and does not establish zero. Arm S (full phase switch) changes success by +0.03 [−0.27, +0.30].
- Release timing: NOT TIMING (§2.3).
- More data: UNTESTED. PAF §4.3 plans it. The only external datum is Brittle Grounding (arXiv 2602.24143, LR): no gain from 10k to 100k demonstrations, on SmolVLA's grasp-phase failure.
- PAF §2 writes "(2) only for our recipe". The record is broader: two frozen VLAs from other recipes also fail. It is narrower on two points: no pi0 or X-VLA run exists here, and the data lever has not run.

**Criterion 3, structure removes it: MET in the familiar layout, with privileged inputs.**
- Hand-over of the place chunk to P_min at the first hold (HO §6.1, results/flawed/pp_hand_read.json, m3_prereg pp_hand): pi0.5 .50 to .96, difference +.46 [+.18, +.73]; GR00T .17 to .93, +.76 [+.53, +.93]. Excess after the hand-over: within-chunk pi0.5 +.02 [−.04, +.10], GR00T +.04 [−.05, +.16]; cross-chunk pi0.5 +.01 [.00, +.05], GR00T +.01 [.00, +.05]. Known tasks 1.00 to .98 (pi0.5) and .98 to .98 (GR00T). H sits level with P_min alone.
- Replication on part seeds 11 and 12: .94 and .93 (HO §8.1, results/flawed/pp_hand2_read.json, m3_prereg pp_hand2). Without simulator contact (H1): .88 and .92; H1 − H is −.08 [−.18, .00] on pi0.5 and −.01 [−.10, +.08] on GR00T.
- Role views (R2), a one-line change to what the place sees: R2 − R0 on N1-N6 familiar is +.400 [+.167, +.625], verdict STRUCTURE WINS (SR §37.1). Success .025 (R0), .225 (R1), .425 (R2). The R2 − R1 difference is +.200 [−.058, +.458], NO CLEAR DIFFERENCE, so the identity-hiding step alone does not clear the margin. Placing given a right grasp: .061, .243, .436. All three costs on known K1-K6 read COST under the registered rule (R2 − R0 −.092 [−.308, +.117]).
- Privilege caveats the documents state: H reads oracle object and target positions, rest height, sizes and roles (HO §8.4); R2 reads simulator instance masks at test (SR §37 limits). The perception-only runs (SR §38.1, §40.1) test U, and have not tested the VLA hand-over with perception. The HP hybrid (SR §44) and the learned-mask R2 (SR §43) have no result yet.
- H replaces the VLA's place chunk and leaves open whether that chunk can be repaired (HO §5, §10).

**Evidence on what carries the failure.**
- V1 returns HELD-OBJECT BINDING: a known instruction naming another target steers the held object (Δsteer +0.67 [+0.45, +0.86]) and recovers nothing (RT §3.1). W arm success is 0.00 at g 0.96.
- RT §5.4 records a post hoc object × target sign reversal in where held objects end. It has no CI and no registered test.

**Thin or conflicting spots.**
- BA §6.3 and Read C show that part of pi0.5's HANDOFF excess comes from the hold signal losing a carry.
- The mechanism (identity read by the place) rests on R2 − R0, which is not separated from R1 − R0 (+.200 [+.025, +.433], NO CLEAR DIFFERENCE) at the registered margin. The whole-wrist-mask control (SR §41) and the box-shape rung R2b have no result.
- One scene, 20 development states, one seed per rung, and every test was post hoc or development (BA, HO and RT headers; SR §37 limits).

**External 2026 support (LR).**
- CRAFT (arXiv 2610.00524): held-out combinations after plain full fine-tuning reach 8.4% (pi0.5), 2.8% (pi0) and 1.1% (GR00T N1.7). Table A.4 shows pi0.5 completing a task that shares the instructed object in 38.4% of rollouts, and GR00T N1.7 in 39.3%.
- Brittle Grounding (arXiv 2602.24143): SmolVLA 44% on seen pairings and 0% on held-out; the failure appears at reach and grasp, so it sits at a different phase than ours.
- Grant et al. (arXiv 2603.19233): activation injection at pi0.5 action-expert layers 16-17 reproduces the source task's motion in 99.6% of 1,968 cross-task episodes.
- AC-VLA (arXiv 2607.15714): masks the wrist view during place; pi0.5 Goal-OOD 46.6 to 73.3. It hides identity and geometry together.
- ReGuide (arXiv 2609.38616): frozen pi0.5 39.0 to 94.5, GR00T N1.7 69.1.
- MESA (project page; LR gives no arXiv id): pi0.5 Composite 46.7 with language followed on 81.5% of composite episodes.
- LR states that no paper reports the phase decomposition (grasp right on nearly every rollout, placement failing).

**Missing test.** PAF §4.3: retrain R0 on 8 tasks, then on more LIBERO tasks that share the scene, and read whether new-pairing success moves. Brittle Grounding predicts it stays flat. This test decides criterion 2.

### F2. The scene picks the object

**Claim.** A VLA trained on a narrow task set grasps a habitual object when the instruction names another.

**Architectural locus.** Vision outweighs language in the joint prefix.

**Criterion 1, information is there: MET.** The instruction names the object. Frozen pi0.5 and GR00T grasp the named object at .95-.99 (SR §32.1). U reaches .894 on new pairings (SR §38.1).

**Criterion 2, persists: UNMET.**
- Our retrain R0 grasps the right object on .408 of new pairings (SR §37.1). Frozen pi0.5 and GR00T grasp it at .950-.992 across all settings (SR §32.1: pi0.5 .950, .967, .967; GR00T .975, .992, .992).
- Under PAF §1, a failure absent in the frozen published VLAs belongs to the training recipe. R0 trains a frozen VLM and an action expert on 8 tasks for 6,000 steps with one seed.
- The frozen models train on 40 LIBERO tasks (SR §37.1, "Against frozen pi0.5").

**Criterion 3, structure removes it: MET on our retrain, without a CI in the sections read.**
- Role phrases (R1) raise the right-object grasp from .408 to .925, and role views (R2) to .975 (SR §37.1, results/flawed/sc_t37_read_r012.json). SR §37.1 reports no CI on the grasp rate. R1 − R0 on success reads NO CLEAR DIFFERENCE (+.200 [+.025, +.433]).
- Role phrases supply the roles in a new form, so a reader may count them as added structure in the prompt.

**External 2026 support (LR).**
- LIBERO-CF (arXiv 2602.17659; CoRL spotlight): pi0.5 completes 13.2% of faithful counterfactual tasks and 60.9% of its episodes complete the biased habitual task. It scores 96.9% on standard LIBERO and 51.7% in the vision-only condition.
- LangForce (arXiv 2601.15197, ICML 2026): "information collapse" between instruction and action; SimplerEnv-WidowX 55.2 to 66.5 with its fix.
- Grant et al. (arXiv 2603.19233): with a null prompt pi0.5 scores near zero on LIBERO-Goal and 62% on LIBERO-Spatial; four of five architectures weight the instruction below vision when the scene identifies the goal.
- ReGuide (arXiv 2609.38616) diagnoses wrong-referent binding by habit.

**Missing test.** A matched-recipe data ladder: retrain R0 on 8, then on more LIBERO tasks, and read the right-object grasp rate on new pairings. If the rate stays near .41 at 40 tasks, criterion 2 passes and F2 stands. Frozen pi0.5 at .97 suggests it will not.

### F3. Motor programs bound to absolute position

**Claim.** In new layouts every system fails at the grasp, by start region.

**Architectural locus.** The action expert's motion tied to workspace coordinates (Grant et al., arXiv 2603.19233).

**Criterion 1, information is there: UNMET as measured.**
- New-layout success on N1-N6: pi0.5 .30, GR00T .08 (HG §5). PAF §2 writes "new layout ≈ .30 for VLAs", which holds for pi0.5 only.
- The scripted servo reaches new/new .31 and P_min .37 (RT §3.3, results/flawed/pp_l1_read.json, m3_prereg pp_rival1). Both grasp the cheese and bottle on about half the episodes (BA §6.2, §7). U reaches .378 (SR §38.1) and the learned-mask U .375 (SR §40.1). Every system sits near .30-.42 except GR00T.
- Read D finds all four systems failing on the same states: VLA grasp failure on `L` against `C` is cheese pi0.5 0.72 / 0.18, GR00T 0.91 / 0.36; bottle pi0.5 0.58 / 0.13, GR00T 0.88 / 0.44 (BA §8.3, results/flawed/pp_atlas_reads2.json, m3_prereg pp_atlas3).
- Read G finds a physical block for the demo's fixed top-down grasp: hold rate 0 of 10 for the cheese and 0 of 10 for the bottle (HO §8.3, results/flawed/pp_read_g.json). On 7 of 10 cheese states the hand touches the wine rack 0.09-0.12 m short.
- The VLAs sometimes succeed on `L` (pi0.5 .16 cheese, .08 bottle; GR00T .03 cheese, .00 bottle). So the states are not impossible for every approach, and a reference that succeeds on them has not run.
- The atlas finds no composition break in the new layout (pi0.5 within-chunk +0.05 [−0.13, +0.21], cross-chunk −0.00 [−0.15, +0.14]; GR00T +0.04 [−0.16, +0.24] and +0.04 [−0.09, +0.18]; verdict NEITHER, BA §5). The layout failure is separate from the pairing failure.

**Criterion 2, persists: UNTESTED.** Both VLAs fail the new layout, and 2 and 10 actions per query leave it unchanged (SR §32.1: pi0.5 .183 and .308 against .300; GR00T .067 and .100 against .083). The data, recipe and prompting levers have not run on the layout failure, and every rung of the retrain ladder sits near 0 there (SR §37.1).

**Criterion 3, structure removes it: UNTESTED.** The hand-over lifts new-layout success (pi0.5 .30 to .51, +.21 [+.05, +.40]; GR00T .08 to .31, +.23 [+.10, +.37]; HO §6.1), and this works on the place after a grasp; the VLAs hand over in .63 and .39 of episodes. The grasp failure itself has no structural fix tested. HO §8.3 states that getting past the block needs an obstacle-aware approach, which the standing rule bars from executors.

**Open spatial reading, post hoc.** Cheese `L` states all start at y below −0.07 m and `C` states at y above 0. Bottle `L` states all start at x above 0 and `C` states at x below 0 (BA §8.3). No registered test covers this. The shift in the new layout also moves objects by about 0.2 m of spread (HO §9.1: 95th percentile radius .237 plate, .289 bowl, .250 cheese, .181 bottle), so layout and reachability change together.

**External 2026 support (LR).**
- Grant et al. (arXiv 2603.19233): the injected motion is bound to absolute workspace coordinates.
- LIBERO-Plus (arXiv 2510.13626, CVPR 2026): pi0 falls from 94.2% to 15.8% under camera changes and to 6.6% under robot-initial-state changes.
- LIBERO-PRO (arXiv 2510.03827): models above 90% collapse to 0.0% under perturbed objects, initial states and instructions.
- PAF §2 lists MESA Spatial. In LR's MESA table the Spatial column is 68.6 for pi0.5 (ID 62.8) and 23.2 for GR00T-N1.6 (ID 25.0), so it shows little drop and gives weak support.

**Missing test.** PAF §4.4: shift the whole scene rigidly with the camera fixed, so reachability is unchanged, on states where the scripted servo succeeds. If the VLAs still fail while the servo succeeds, position binding is architectural.

### F4. No addressable parts

**Claim.** A VLA stores trained wholes and cannot express a new pairing as a combination of known parts.

**Architectural locus.** No part store: instruction-conditioned retrieval returns trained wholes.

**Criterion 1, information is there: MET.**
- One demo of each new pairing (N1-N6, familiar layout, recorded by U from start state 24) contains the pairing, and all six succeeded on the first attempt (SR §42).
- Flow-loss retrieval over role phrases parses 6 of 6 on r1 and 5 of 6 on r2 (SR §42.1, results/flawed/sc_q2/<rung>/demo_new.json, m3_prereg sc_q2n).

**Criterion 2, persists: UNTESTED on any frozen VLA.**
- The only model that fails to express the answer is our retrain r0: operator joint 0 of 6, object matched 2 of 6, target matched 4 of 6, flow loss joint 0 of 6 (SR §42.1).
- The joint count of 0 holds by construction, since r0's candidates are trained whole instructions (SR §42). PAF §2's phrase "plain r0 parses 0/6 new-pairing demos" reports a count that no outcome could change. The informative figures are the object and target matches.
- r0 parses known demos perfectly (joint 1.000, SR §39.1), so it retrieves wholes well.
- No frozen pi0.5, GR00T, pi0 or X-VLA has been parsed.

**Criterion 3, structure removes it: UNTESTED.**
- On r1 the operator reaches 5 of 6 (miss: N6 target "plate" for "bowl"), and on r2 5 of 6 (miss: N4 target "bowl" for "plate", the same miss under flow loss). At setting 0 r1 gets 4 of 6 and r2 gets 3 of 6. R2b is pending.
- The role rungs differ from r0 in candidates and training. The role vocabulary adds the part phrases themselves, so no result separates added information from added structure.
- The operator and retrieval do equally well on every rung: picked − flow loss is .000 on r0, −.100 [−.167, −.033] on r1, .000 on r2 (SR §39.1, results/flawed/sc_q2/<rung>/read.json, m3_prereg sc_q2). SR §42.1 states that the parts in the role vocabulary make the pairing parseable and that the operator is one way to read them.
- The prediction that cheese demos would parse worse than bottle demos was wrong: all three cheese demos parse on r1 and r2, and each miss is a bottle target.
- Six demos, one state each, parsing only, with no closed-loop composition from the parse (SR §42 limits).
- BindVLA, explicit binding slots on the same motor path, reaches .03 against a predicted .44 (TC §2 cites PS §1; that section was outside this read).

**External 2026 support (LR).**
- ReGuide (arXiv 2609.38616) diagnoses habit binding.
- ACT-VLA (arXiv 2607.00351): pi0.5 Goal-OOD 46.6 to 95.6 through new data and retraining.
- CORAL (arXiv 2603.09298) keys one adapter per whole task and composes none; MAST-VLA (arXiv 2610.32312) routes by seven coarse skill categories; PhaseLoRA (arXiv 2608.15285) keys by motion statistics. None tests unseen pairings.
- Dr. VLA (arXiv 2603.19183): most SAE features in VLAs encode memorized demonstration sequences.

**Missing test.** Run `sc_q2_demo.py` on a frozen pi0.5 and a frozen GR00T over the six new-pairing demos, with the same candidate list the role rungs receive (the part phrases composed from known instructions). If the frozen models parse 0 to 2 of 6 and the role rungs parse 5, criterion 2 passes. The paired r0 control (whole-instruction candidates only) would separate vocabulary from architecture.

### F5. Chunking and open loop are not the cause

**Claim.** Query length does not move the failure.

**Architectural locus.** Action chunks.

**Criteria.** Not applicable. The class reports a ruled-out cause. The record sits in §2.1: NO GAIN FROM CHUNKING at 2 and 10 actions per query on both frozen VLAs (SR §32.1). The intervals allow up to about +.15 (pi0.5) and +.23 (GR00T), and replanning at 1 did not run.

**External 2026 support.** LR reports none. pi0.7 (arXiv 2604.15483) adds real-time chunking, and LR gives no ablation of it.

**Missing test.** Replanning at 1 action per query, which SR §32 left out for cost. It would close the lower end of the chunk-length range.

### F6. No memory

**Claim.** Tasks that need state no longer visible fail.

**Architectural locus.** A Markov policy over a short window.

**Criteria.** C1 UNTESTED, C2 UNTESTED, C3 UNTESTED. No document in this read tests memory. PAF §2 states "none here".

**External 2026 support (LR).** MEM (arXiv 2603.03596, spotlight) gives pi0.6 video short-term memory and text long-term memory for tasks up to 15 minutes. pi0.7 (arXiv 2604.15483) adds a video encoder with up to six history frames. RoboTTT (arXiv 2607.15275) applies fast-weight test-time training over an 8K-step context and reports +87% over a single-step-context baseline. A memory-augmented action-conditioned world model (arXiv 2606.18960) keeps occluded objects across long rollouts. These papers add memory to a policy. LR reports no failure rate of a memoryless VLA by memory demand, so they motivate the class and leave it unmeasured.

**Missing test.** PAF §4.5: occlude the target after the instruction (or ask the VLA to put the object back where it was) on LIBERO, against a reference that succeeds on the same states.

### F7. Relational and negated language is ignored

**Claim.** "Not the bowl" and "the left one" do not change what the action path does.

**Architectural locus.** Language read as a bag of cues by the action path.

**Criteria.** C1 UNTESTED, C2 UNTESTED, C3 UNTESTED. PAF §2 says LIBERO-PRO's language perturbations are on disk. No document in this read reports a run, and I did not read the disk files.

**Local evidence that bears on it.** V1 arm W changed the target by naming another known task and moved the held object toward that target (Δsteer +0.67 [+0.45, +0.86], RT §3.1). Language about a target noun steers the place. That result concerns plain target nouns and has no negation or relation in it.

**External 2026 support (LR).**
- LIBERO-Plus (arXiv 2510.13626): language perturbations cause small drops because models ignore language.
- LIBERO-PRO (arXiv 2510.03827): outputs do not change under nonsense instructions.
- LIBERO-CF (arXiv 2602.17659): 51.7% success in the vision-only condition on standard LIBERO.
- LangForce (arXiv 2601.15197): instruction-action information collapse.
- S2 (arXiv 2606.02735): LIBERO-PRO Goal-Pos 29.0 to 49.5 with a learned token gate.
- These results show that language is under-used in general. They do not isolate negation or spatial relations, so the support is indirect.

**Missing test.** PAF §4.5: LIBERO-PRO's negation and spatial-relation variants on frozen pi0.5 and GR00T, scored against a reference that succeeds on the same states, with a control prompt of equal length.

## 4. Where the record is thinner than PAF §2, or claims a status the documents do not support

1. **F1 status E.** PAF marks criteria (1) and (3) established. Criterion 3 holds with privileged inputs (HO §8.4, SR §37 limits), and the perception-only runs test U, not the VLA hand-over. SR §43 and §44 have no result. The identity-hiding step (R2 − R1) reads NO CLEAR DIFFERENCE (+.200 [−.058, +.458]). Criterion 2 on more data has not run.
2. **F1 numbers.** The stage-level excesses (HANDOFF +.31/+.39, RELEASE +.16/+.38) have no CI in BA §7. The CIs sit at group level (BA §5). Part of pi0.5's HANDOFF excess is a hold-signal artefact (BA §8.3 Read C: 12 of 18 missed carries).
3. **F1 "(2) only for our recipe".** This understates the record on models (two frozen VLAs) and overstates it on levers, since the data lever is untested and the prompting CI contains a gain above 0.10.
4. **F1 citations.** PAF lists LIBERO-CF under F1. LR places it with the vision-overrides-language failure (F2). PAF writes "Brittle Grounding (100k demos give no gain)" as F1 support. LR attributes the failure to reach and grasp, which differs from F1's phase, and the 10k-to-100k statement concerns SmolVLA. PAF lists only pi0.5's 38% share-object figure and omits GR00T N1.7's 39.3%.
5. **F2 status P.** Under PAF §1 the class fails criterion 2, because frozen pi0.5 and GR00T grasp right at .95-.99. It reads as recipe-rooted as measured. The R1 grasp gain (.408 to .925) has no CI in SR §37.1.
6. **F3 numbers.** PAF writes "new layout ≈ .30 for VLAs". The figure is .30 for pi0.5 and .08 for GR00T (HG §5). PAF's list of MESA Spatial as support is weak: pi0.5 Spatial 68.6 sits above its ID 62.8.
7. **F3 and F1 layouts.** The atlas finds no pairing effect in the new layout (verdict NEITHER). F3 therefore concerns layout, and the 20 development states give wide CIs (up to about +0.2 allowed, BA §7).
8. **F4 status P.** "Plain r0 parses 0/6" holds by construction (SR §42). No frozen VLA has been parsed, so criterion 2 is untested. The role rungs add the part vocabulary, so criterion 3 mixes added structure with added information. R2b is pending. The finding rests on six demos with one state each.
9. **F5.** Marked E as a ruled-out cause. That holds only for large effects: the intervals allow +.15 and +.23, and replanning at 1 did not run.
10. **F6 and F7.** PAF marks them O, consistent with the record. The F7 claim "bag of cues" has no support in LR beyond "language is under-used"; V1 arm W shows language can steer a place.
11. **Models.** PAF §3 asks for at least three models for generality. This document rests on two frozen VLAs (pi0.5, GR00T) and our retrain. pi0 and X-VLA have not run. External support covers pi0, pi0.5, GR00T N1.7 and others (CRAFT, MESA).
12. **Scope of all local evidence.** One scene, 20 development states (N2 main states 4-23), one seed per trained rung, post hoc development reads (BA header, HO header, RT preamble). HO §10 lists a fresh-seed confirmatory read as needing the user's approval.
13. **Pending or unrun.** R2b (SR §37.1, §39.1, §42.1), R2w and R2w0 (SR §41), R2 with learned masks (SR §43), HP (SR §44), and the 7 A40-h per-model reruns on pi0 and X-VLA (PAF §4.2).
14. **LR coverage.** About 300 of 687 CoRL papers were seen, most numbers come from abstracts or a summarizing fetch tool, and several of the closest papers (CRAFT, AC-VLA, ACT-VLA, ReGuide, Brittle Grounding) state no venue (LR, "What could not be verified"). LR notes that some arXiv ids carry unusually high serial numbers, so the ids here need a check before citation.

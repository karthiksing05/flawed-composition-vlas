# Plan: composition as a scaffold for generalization

Draft for the user's review, 2026-11-05. Nothing here has run. It follows three independent reviews of the record (FLAWED_COMPOSITION; PLAN_PARTS, PARTS_RESULTS, LADDER_RESULTS, COVERAGE_RESULTS; ADVERSARIAL_STUDY, RIVAL_TESTS, PLACE_DIAG, BREAK_ATLAS, HOLD_GATE, HAND_OVER). I checked the main claims against the sources. Section marks cite those files.

## 0. Thesis

**What a policy stores decides what it can generalize to.** A policy that stores tasks entangled in its weights must interpolate on a new combination. A policy that stores typed parts and fixed rules reaches a new combination by lookup. So composition, encoded in the structure of memory, scaffolds generalization in several senses at once: new pairings, new parts, new sequences and new layouts.

**The argument rests on removal.** With the same VLM, demos and parameter budget, generalization should fall as each piece of structure comes out. This is a causal claim about structure. It does not depend on measuring factorization in a representation, which the literature finds does not predict generalization (ADVERSARIAL_STUDY §4).

**What this plan does not claim.** "VLAs are the wrong architecture" is not licensed. Every VLA arm so far is frozen, no structural retrain or fine-tune of a VLA exists, a few test-time interventions failed with wide intervals, and published work rescues dense VLAs without structure (ADVERSARIAL_STUDY §1, §4). The removal ladder in §6 must include a retrained VLA, and the thesis has to win that comparison.

## 1. Senses of composition, by timescale

Layout enters as its own sense. The record suggests that scene novelty lives mostly at long timescales and object identity at short ones (§3).

| sense | timescale | question | evidence | status |
|---|---|---|---|---|
| Motor field | one action | How do active components combine? | Only hard switching has worked. Always-on summing gets 0.72 on known tasks and g 0.06 on new pairings (FC §5.5) | Measured on our own policy only |
| Object-to-motor fit | one chunk | Does the motion fit the bound object? | Sharing one place term across targets hurt (K1 0.97→0.73; FC §5.9). Restoring target identity helped (FC §5.16). A known place instruction steers the held object by +.67 and recovers the place by only +.03 (RT §3.1) | Thin on real VLAs |
| Role-filler | one chunk to one skill | Do known objects and targets combine? | VLAs lose the place on new pairings (BA §7). The part policy holds near known rates in the familiar layout (PR §54) | Strongest, with caveats (§4) |
| Referential binding | one look | Does a phrase pick the right thing? | Slots bind at 0.72-0.73 on held-out compounds and layouts, flat across compound divergence; success stays at 0.03 against a predicted 0.44 (FC §1.2) | Binding alone does not deliver success |
| Sequential | across skills | Does one skill lead into the next? | The event gate beats the command gate by +.21 (FC §5.6). The hold event replaces contact (LR §3; HO §8.1). P_min scores the drawer K7 .27, N7 .00 and push K8 .20, N8 .02 in the familiar layout (CR competence table) | **Two phases only.** Actions per query never varied |
| Scene | episode | Does a known skill survive a new arrangement? | P_min 0.90→0.46 on known tasks (PR §53-54). VLAs reach .28 and .14 on known pairings (AS §2) | Causes partly known (§3) |
| Lifetime | across tasks | Does a new part reach every partner, at no cost to the old? | S3 passes six of six familiar cells, two of six new (PR §53). Old tasks bitwise unchanged (PR §54) | On simulator state only |

## 2. How the earlier experiments scaffold this plan

| work | what it establishes | what the plan takes from it | caveat that carries over |
|---|---|---|---|
| BindVLA, E4, B0-B1 | Binding can stay flat across compound divergence, and success does not follow | Binding is one store among several. The motor needs its own structure | Never joined to a factored motor |
| Debt theory, D0-D2, N1 | Purity, combination, bypass as named conditions | Keep them as design rules for the stores | D0 and D2's predictive claims are falsified (FC §4), and D1x found VLA expert recombination near-factorial on easier tasks (−.057 [−.135, .021]). Unreconciled |
| Prototype A, rounds 1-1p | Joint heads fail on the same oracle inputs (Ab − Ab_nosum +.26 on success, +.62 on grasp; FC §5.6). Locality exact | The cleanest within-policy evidence that structure causes composition. It seeds the removal ladder | Small oracle policies. Rounds 1g-1n trained on data missing 12,114 dwell rows (FC §5.27), so only 1o and 1p are clean |
| Layer 1, F1-F3 | Mixing is locatable, its causal role is unsupported | Use VLM features from before language fusion, since pi0.5's mixing sits in KV 14 | My inference; no test isolates it |
| Layer 2, P_min | Sparse parts with a fixed rule compose in the familiar layout. A one-demo part costs 35-68 GPU-s. Baselines reach at most 0.09 on unseen pairings | P_min is the starting part store | GO false for every arm. Sixteen post hoc rounds on reporting states. No confirmatory read. No pixels, no language |
| Ladder L1, L4 | Hold event from proprioception works. A cross-situational rule parses roles exactly on 12 read tasks | Event sequencer and the lexicon's first version | Lexicon rule designed with these instructions in view. Withholding K4 gives a silent wrong parse (LR §5). Event holds without a real hold in 25% of new-layout episodes |
| Coverage test | More demos raise known tasks (+.18; K5 0→.60). New pairings gain +.03 [−.06, +.13] | More data alone will not deliver the layout claim. It also costs familiar K4 .18-.25 | Teacher kept only .45-.88, so a limit of the class stays possible |
| Place diagnostic | The target term cannot tell objects apart. Familiar gaps N4 − K1 and N6 − K6 read RELEASE-STATE (PD §3.2). New-layout verdicts are NOISE or UNMEASURED | Look first at the state the object arrives in. A pair-keyed place store has no support yet (§3.1) | All three of my predictions failed. Cause not established (PD §3.1) |
| Break atlas | Familiar: failures come after the first hold. New: fail alike at the grasp | The target for what the sequencer and parts must fix | E's control is flawed (BA §6.4). HANDOFF overcounts on pi0.5. Stage groups are behaviour, unrelated to action chunks |
| Hold gate, hand-over | A VLA's pick composes with an external place part (.50→.96, .17→.93). Holding shut does not fix it | Skill boundaries are real seams. A VLM-backed skill can feed a part | Replacement, on oracle positions, one development pass |
| Reads E, G, H | New-layout grasps fail by not reaching. Servo blocked 0/10 on a fixed approach | Separates a physical approach problem from a composition problem | Blocks one approach only; VLAs sometimes pass |
| Locator rounds | Moved cutouts remove most object confusion. Demo events label objects with no simulator position | Entity store: an object is what changed when the robot moved it. Pasting is a way to train entity keys | Camera matrices oracle. 20 states shaped it. No VLM features yet |

## 3. What the record changes in the design

**3.1 Parts may not be purely (role, atom), and only one case is firm.** The bottle's grasp depends on its target: K5 turns about 81°, every other task 1.9-3.7° (PR §37). That case stands. The place evidence is weaker. An early P lost N4 at the place interface against a single-task fit (place on arrival .34 against .77, PR §22, instrument v2), and the device tscale later bought +.23 of it back (PR §53). P_min's only familiar 3F failure is N5, which runs above its prediction. The registered place diagnostic reads the remaining familiar gaps as RELEASE-STATE, which points at the arriving object and away from a pair-keyed place term. So the design adds a **sparse exception store**, keyed by a pair and created only when a measured gap justifies it, starting from the one case that needs it. Its size is counted, and its growth is itself a test: if exceptions grow with pairs, the thesis fails.

**3.2 "Small part" hides a large shared store.** A rank-8 part adds about 2.4% of a backbone (PR §5) and depends on it. Transport moved into the backbone (rtransport, round 13, bottle hold in the new layout +.28 [.12, .44]; PR §41-42). So the store has shared role backbones as a first-class type, and every byte count includes them. One competing fact: a single-task fit on servoing carries matches P's class on K1 and K4 (1.00 and .72 against .97 and .63), so the class fits a task alone. The backbone's value lies in sharing, and only the removal ladder measures that.

**3.3 New layouts split by timescale.** This is the plan's main hypothesis about layouts, and it is untested.

- Most new-layout gaps PR §54 lists concern getting there: no far rows, stalls, carries starting beyond support.
- Failures near fixtures are stark on an earlier policy: Ao1 succeeds in 0 of 54 episodes with the cheese within 0.10 m of the rack, and 41 of 66 elsewhere (FC §5.29). P_min has not been stratified this way.
- Once the hand holds the object, parts do well: success when handed over .80 and .79 in the new layout (HO §6.1). That figure is conditional on the VLA's grasp succeeding, so it selects easier states.
- Fixture targets hold up: P_min K3 scores .87 in the new layout (CR competence table).

**Hypothesis.** Contact-local skills are close to layout-invariant in their own object frame. Layout novelty sits in the long-range transit and the approach. So the store separates a **short, atom-specific contact part** from a **shared transit field**, which moves between pre-poses and treats other entities as obstacles. The approach frame alone did not help (+.00 [−.07, +.06], PR §14-15), which I read as one term carrying both timescales. That reading is untested.

**Evidence against the hypothesis, already in the record.**
- Place on arrival falls in the new layout on P_min: K1 1.00→.51, N4 .84→.59, N6 .86→.70, N1 .98→.84, with K6 nearly level at .98→.93 (PD §3.3). K1's failures include a bowl still held at the end, and one at 175° of tilt. These episodes arrived, so a transit field alone will not repair them. They may reflect the state the object arrives in, which T3 can separate by starting from correct pre-poses.
- Grasp itself falls: new-layout failures cluster at the grasp on both VLAs, and a scripted servo halves too (.62→.31, PR §54).
- More near-rack data did not help. Stage 1p, on clean data, doubled near-rack carries and stayed at 1 of 27 on K6 and on N1; the registered falsification fired (FC §5.30). Only 5 of 10 near layouts gave the generator any grasp option that held (FC §5.29). So part of this gap may be a physical limit or a limit of that generator's options, and the record does not separate them.

I hold this hypothesis at less than even odds. T0 and T3 can kill it for little compute, which is why they come first.

**3.4 Obstacles become a composition question.** The transit field composes with avoidance terms built from entities in the entity store: each entity in view contributes one term, and the same rule applies in any scene. The standing rule bars occupancy and free-space inputs, and 1l-1m supports caution: a global scene input keyed on far entities (FC §5.23). But those rounds trained on the defective data, and no ablation was rerun after the fix. **This needs the user's decision** (§8).

**3.5 Sequencing has almost no positive evidence.** Push never hands over, because no event fires (every K8 demo keeps the gripper open; FC §5.5). P_min scores K7 .27, K8 .20, N7 .00, N8 .02 in the familiar layout. Schemas must first show they can do a three-phase task. Actions per query becomes per-skill only if the timescale test supports it.

**3.6 The VLM must not become a bypass.** Language reaches parts only through the lexicon, and each part sees only its own entity. Proposition 2's warning still applies: a part denied what it needs takes a proxy (C1 fell .705→.441). Each part therefore gets its own pure route to what it needs, such as local visual features around its entity.

**3.7 The lexicon must be able to abstain.** A silent wrong parse is worse than a refusal.

## 4. Discipline the record demands

- **Fresh data.** N2 states 4-23 have shaped sixteen rounds and the locator. Development moves to new states and, for the layout claim, scenes where fixtures move as well. Confirmatory reads and sealed sets stay with the user's approval.
- **New scenes.** LIBERO-Goal is one kitchen with fixed fixtures, so a layout claim from it alone will not stand.
- **Every piece earns its place by removal.** psupport, imode, rtransport and rank 32 were never ablated (PR §53). P_min's part contribution needs a k = 0 control (AS §3).
- **An equivalence test.** 3F passes when a CI contains 0, so noise rewards it. Use a margin of ±.05 with enough partners per target (AS §7).
- **Audit data pipelines** before reading results. The dwell bug hid for eight rounds.
- **No diagnose-and-patch rounds.** Diagnoses repeatedly failed their own rounds: the K5 rise, the cheese hold rise, the height account (FC §5.26-5.27), the coverage account (FC §5.30). Each new mechanism needs a named weakness measured beforehand, one principle and a removal control.
- **Do not reuse flawed measures:** E's control, the .99→.38 baseline that pools N7 and N8, F1's target-into-grasp floor, "cheese misses by centimetres".

## 5. Architecture

| store or component | contents | grows with | starts from |
|---|---|---|---|
| VLM backbone | frozen encoder; pre-fusion features | never | pi0.5's vision tower |
| Lexicon | word → role and atom, with abstention | new words | L4's rule |
| Entity store | atom → visual key, frame, size | new objects | moved cutouts; centre labels |
| Contact parts | (role, atom) → short flow field in the object's frame | new atoms | P_min's terms, cut to the contact phase |
| Shared stores | role backbones; one transit field | never, once built | P_min's backbones, rebuilt |
| Exceptions | (object, target) → small correction, on measured need | should stay sparse | m, the 1.58-bit message |
| Schemas | verb → skills, slots, event transitions, actions per query | new verbs | pick-place; then the drawer |

No task is stored. A task is a path through these stores.

## 6. Claims, each can fail

| | claim | refuted by |
|---|---|---|
| **C1 Causation** | Generalization falls at each removal: full → learned combination → one holistic head → retrained VLA | A retrained VLA matches the full system on new pairings at equal demos |
| **C2 Layout** | Contact-local success is close to layout-invariant, and composing transit with entities recovers new layouts | Contact parts fail from correct pre-poses in new layouts |
| **C3 Sequence** | Schemas with events reach longer chains at the product of skill competences | The drawer task stays near 0, or chains fall below the product beyond the margin |
| **C4 Lifetime** | A new atom costs one part, works with all partners, leaves old tasks exact, and exceptions stay sparse | Exceptions grow with pairs, or a k = 0 control does as well |
| **C5 Performance** | On known tasks the full system stays near VLA levels (.98-1.00 familiar), from pixels | It trails on known tasks beyond a registered margin |

## 7. Tests, cheapest decisive first

Each gets its own registration when it starts, with a single verdict mapping. Costs are my estimates.

| | question | design | cost | what it decides |
|---|---|---|---|---|
| **T0** | Do VLA failures line up with query boundaries? How does the new layout's loss split between far and near? | Existing tracks only: gripper flips and first divergence by step within a query; loss by stage and fixture clearance | CPU | Whether T1 is worth its cost; first read on C2 |
| **T1** | Your timescale test: where does sequential composition break as actions per query changes? | Frozen VLAs, N1-N6 and partners, replanning below and above today's 5. Failure rate relative to chunk and skill boundaries | about 7 A40-h per setting | If failures follow chunks, a VLA with variable chunking may fix them, which weakens the case for schemas |
| **T2** | Do pre-fusion VLM features locate entities? | Offline: entity keys from one demo's cutout against the small locator, on new states. One token covers about 0.06 m (my estimate), so test refinement inside a token | 1-2 A40-h | Whether the entity store can sit on the VLM |
| **T3** | Is contact-local behaviour layout-invariant? | Oracle positions: start the hand at each part's pre-pose, by simulator state, in both layouts, stratified by fixture clearance | CPU | **C2's hinge.** If it fails, the layout hypothesis dies early |
| **T4** | Clean rebuild | Contact parts, shared backbones and transit, with removal controls including k = 0 and the four unablated devices. Pipeline audit first | about 5 A40-h | The real starting store |
| **T5** | Obstacles by composition | Entity-based avoidance terms in transit, on near-fixture states, once the user decides | about 5 A40-h | C2 |
| **T6** | Sequences | Drawer task, then two-skill chains; schemas with events | about 5 A40-h | C3 |
| **T7** | From pixels | Full stack on VLM features, six pairings and known tasks | 10-30 A40-h | C5 |
| **T8** | Removal ladder | Same VLM and demos, including a VLA fine-tuned on these demos and an ACT-VLA-style arm | 30-60 A40-h | C1 |
| **T9** | Lifelong stream | Atoms arrive one at a time across layouts; bytes, reuse, forgetting, exception growth | about 10 A40-h | C4 |
| **T10** | Confirmatory | Fresh states and scene, once frozen | needs the user's approval | The claim |

**Stop rule.** If T3 fails, the layout claim has no mechanism, and I bring that back before any further layout work. If T2 fails, the entity store stays on a small encoder, which weakens the tie to the VLM. If T8's retrained VLA matches the full system, the thesis as written fails, and we report it.

## 8. Decisions for the user

1. **Entity-based avoidance terms in the transit field.** The standing rule bars occupancy and free-space inputs. These terms are entity-keyed and use one rule in every scene, but they are close to what the rule bars. The layout claim likely needs them or something like them. The evidence behind the rule came partly from defective data.
2. **Ownership of T1.** You said this experiment is running. If it is, I build on yours.
3. **Scenes.** Which scenes count for the layout claim, including authoring BDDL scenes with moved fixtures.

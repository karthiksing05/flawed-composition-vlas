# A theory of composition for VLAs, and the smallest edit that tests it

Draft for the user's review, 2026-11-10. Reasoning from the existing record; nothing new has run. The user's direction, in their words: "creating a theory of composition", "edit the VLA simply, while explaining where and how prior versions of it fail, to produce a compositionality-first architecture that brings together the best of both worlds", built from "concrete physical parts which can be combined together in novel ways", where physical means "structurally important parts". `SYS1_SYS2.md` is inspiration only and `docs/PLAN_S12.md` is not the experiment.

Section marks: RT = RIVAL_TESTS, SR = SCAFFOLD_RESULTS, PS = PLAN_SCAFFOLD, FC = FLAWED_COMPOSITION.

## 1. The theory

A task is a short program: a verb, an object filler and a target filler, run in phases that an event separates (reach and grasp, then carry and place). A policy **composes** when its success on an unseen pair equals the product of its parts' competences, s(o, t) = g(o) · p(t). The repo has used this criterion since FC (3F).

**P1. Shortcut.** Training on a set of pairs cannot tell a policy to learn parts. If a filler has one partner in the data, the partner's identity predicts everything the phase needs, and the loss is minimised equally well by a policy that stores the pair. The held object sits in two cameras in every carry frame, so it is the easier cue.

**P2. Purity.** If a phase can read the identity of its own filler only, and reads everything else as geometry, the pair solution cannot be represented. One partner per filler then suffices, and a new pair is a lookup.

**P3. Fixed rule.** Parts join through an operator that no pair trained: an event switch, a route. A learned combiner brings P1 back.

**P4. Sparse exceptions.** Real interactions between fillers exist: the bottle's grasp turns 81° for one target (PS §3.1), and the release depends on what the hand holds. They pass through a narrow, counted channel. If exceptions grow with pairs, the theory fails for the domain.

**P5. Addressing.** Identity selects which part runs. It is an address, and the motor computation never takes it as an input.

**In one line: "what" is private to a role, "where" is shared, and identity is an address.** U follows all five on simulator state. A VLA follows none by construction: one token stream, full attention from every action token, one jointly trained map.

## 2. Where prior versions fail, by principle

| system | binds phrases | own-role identity only (P2) | fixed rule (P3) | addressable parts (P5) | new pairings, familiar |
|---|---|---|---|---|---|
| frozen pi0.5 | implicitly | no | no | no | .50; grasp .97, place .52 (RT §3.1) |
| frozen GR00T, two systems joined by a latent | implicitly | no | no | no | .167; grasp-right .99 (SR §32.1) |
| BindVLA: explicit binding slots, same motor | yes, .72-.73 | no | no | no | .03 against a predicted .44 (PS §1) |
| V1 arm S: role-phase prompts at inference | yes | language only | event | no | +.03 [−.27, +.30] over intact (RT §3.1) |
| hand-over: VLA's pick, object-blind place part | yes | yes, at the place | event | the place | .50→.96, .17→.93 (PS §2) |
| U: typed parts on simulator state | lexicon | yes | event | yes | .894; no pixels; new layout .381 |

**What the table says.**
- Binding alone does not help (BindVLA). The motor path stays impure.
- Purifying language alone does not help (V1 arm S). V1's verdict was HELD-OBJECT BINDING: a known place instruction steers the carry (+.67) and does not recover the place (+.03).
- Hiding the held object's identity from the place does help (hand-over, U).

So the missing piece is P2 on the visual side, during the carry and the place.

**Partner counts in the training data (K1-K6).** Cheese: one target (bowl). Bottle: two (rack, cabinet). Bowl: three. Plate, stove, rack and bowl-as-target: one holder each. Cabinet: two (bowl, bottle).

| pairing | object's targets in training | target's holders in training | frozen pi0.5, intact (RT §3.1) |
|---|---|---|---|
| N1 cheese → plate | 1 | 1 | .10 |
| N2 cheese → stove | 1 | 1 | .20 |
| N3 cheese → cabinet | 1 | 2 | .60 |
| N4 bottle → plate | 2 | 1 | .85 |
| N5 bottle → stove | 2 | 1 | .85 |
| N6 bottle → bowl | 2 | 1 | .40 |

This pattern is post hoc, on six tasks at 20 states each, and N6 does not fit cleanly. It is what P1 predicts: the VLA composes where the data broke the correlation and stores the pair where they did not. **So the VLA's architecture is not wrong everywhere. It composes only what the data decorrelate, and a compositional architecture should need one partner where the VLA needs two or more.** That sentence is the claim I would put at the centre of the paper, and the cheese pairings are its diagnostic cells.

**A dissent from my own registration.** T16 (SR §33) trains role prompts into pi0.5 and I predicted STRUCTURE WINS near .75. T16 closes the language channel only; the held object stays visible, and every "target: the plate" frame shows the bowl in the hand. Under this theory T16 gains little on the cheese pairings. The registered prediction stands in the ledger; the read now tests the theory against it.

## 3. The edit: a role channel between perception and motor

**The structural part.** A gate that, in each phase, passes identity for the active filler and geometry for everything else. Everything else in pi0.5 stays: weights, VLM, flow action expert, training recipe.

**First implementation, at the input, with no new weights.**
1. **Sequencer.** `ProprioHeld` picks the phase, as in T16.
2. **Role phrase.** The prompt names the active role only, as in T16.
3. **Role view.** In both cameras, every nameable entity other than the active filler becomes a flat silhouette of one fixed colour. During the place the hand's object keeps its shape and loses its identity; the target stays intact.

Masks come from the simulator in the supervised stage and from the locator's moved cutouts (PS §2) in the unsupervised stage, in the order the user set.

**Why at the pixels.** The prefix mixes tokens by self-attention, and F1-F3 placed pi0.5's mixing mid-stack (PS §2). A mask applied after that point passes tokens already contaminated. A gate before the encoder is the one place where purity holds without changing the interface the pretrained expert expects. A token-level form (drop or split prefix tokens through `input_mask` in `openpi/models/pi0.py`) is the follow-up once the pixel form shows the effect.

**The ladder: one structural part per rung, each removable.**

| rung | adds | principle | theory's prediction on the cheese pairings |
|---|---|---|---|
| R0 | nothing: plain retrained VLA (T8b) | none | stays near .10-.20 |
| R1 | sequencer and role phrases (T16) | P3, P2 for language | small gain |
| R2 | role views | P2 for vision | rises toward the bottle's .85 |
| R3 | parts store: one keyed adapter per (role, atom) on the action expert, selected by the lexicon | P5 | holds R2; a new atom costs one adapter and leaves old tasks exact |
| R4 | geometry in the active entity's frame | layout | uncertain; U has this and still reads .381 |

**Best of both worlds.** The VLM keeps perception and language from pixels, and the pretrained action expert keeps the motor prior that gives .99 on known tasks. The structure adds what U showed works: role-private identity, an event rule and addressable parts.

## 4. Tests, cheapest first

| | question | design | cost |
|---|---|---|---|
| **E0** | Does partner count order new-pairing success across every arm already run? | Registered read of existing tracks: pi0.5 and GR00T, intact, A, S; grasp and place by pairing | CPU |
| **E1** | Does the ladder behave as §3 predicts? | R0, R1, R2 under one single-GPU LoRA recipe (`pi05_l90c_lora`'s settings on the eight known tasks), so the queue for four A40s stops blocking. T8b and T16 stay queued as the full fine-tune forms of R0 and R1 | about 1 A40 for 6-8 h per rung |
| **E2** | Is P1 right about the data? | Give the cheese a second partner in training and nothing else. P1 predicts the plain VLA then composes on the cheese's other targets | one retrain; needs demos of one new task, which the demo rules leave to the user |
| **E3** | Do parts make it lifelong? | R3 and T9's stream | later |

**What would refute the theory.** R2 leaves the cheese pairings where R1 left them. Or E2's second partner does not lift the cheese's other targets. Or R0, the plain retrained VLA, already composes on the cheese.

## 5. Limits

- Six pairings over three objects and five targets. The partner-count law needs more atoms than this suite has.
- Simulator masks are privileged inputs at test time. The supervised stage is a ceiling, and the claim needs the unsupervised stage.
- A silhouette keeps shape, and the gripper's opening still tells the objects apart. P4 treats both as geometry; if the place fails on them, the exception channel is wider than I assume.
- New layouts fail at the grasp for every arm, and this edit does not address that.
- LoRA rungs and full fine-tunes may differ. Each comparison stays inside one recipe.
- States 4-23 have shaped every rule here. Confirmation needs fresh states.

# Plan: System 2 builds System 1, tested on composition

Draft for the user's review, 2026-11-10. Nothing here has run. It joins two sources: the user's bet in `SYS1_SYS2.md` (version 5) and this repo's record on how VLAs fail to compose (FLAWED_COMPOSITION, PLAN_SCAFFOLD, SCAFFOLD_RESULTS). Section marks cite those files. Results that `SYS1_SYS2.md` quotes from the literature are that file's citations; I have not re-checked them.

## 0. What the user set

2026-11-09, in their words:

- "creating a System 1, System 2 style of abstraction"; "abstract System 1 as an MCP over the robotic arm and alter it over time with respect to Opus wrapped in a harness".
- The split separates "what is deterministic and discrete from what is non-deterministic and higher-level reasoning".
- "System 1 and System 2 should each handle a percentage of the work, and both systems should work together so that this happens for as many few-shot tasks as possible."
- The paper is "a culmination of compositionality and how different VLAs break, what VLA failure is, and how it comes from binding. My proposition is that the architecture itself is completely wrong."
- It ends as "a discussion of all the problems with VLAs, as well as how an abstracted, structured, architecturally structural system can fix it, and how it involves the tenets of compositionality".

`SYS1_SYS2.md` adds the mechanism: a frontier model in a harness builds System 1 (S1) from a hardware spec, a simulator and optional demos; S1 runs with no frontier calls on routine work; a gate in S1 escalates; update rounds patch S1 behind regression tests.

## 1. Where the repo stands against this goal

**What the record already supplies: the paper's first half.**

| finding | numbers | source |
|---|---|---|
| Frozen VLAs grasp the right object on new pairings and then fail | grasp-right .95-.99; success .50 (pi0.5), .167 (GR00T); known tasks .99, .96 | SR §32.1 |
| Query length does not move the failure | all four settings NO CLEAR CHANGE | SR §32.1 |
| Binding a phrase to the right thing does not deliver success | slots bind at .72-.73; success .03 against a predicted .44 | PS §1 |
| A typed seam at the hold event repairs the VLA | VLA pick + external place part: .50→.96, .17→.93 | PS §2 (hold gate, hand-over) |
| Typed parts with fixed rules compose | U .894 on new pairings, familiar; verb slot N8 .033→.833; push K8 .867 | SR §24-25, §30 |
| A new part is cheap and leaves old tasks exact | 35-68 GPU-s per one-demo part; old tasks bitwise unchanged; N4 from two created parts .82 | PS §2, SR §20.3 |
| New layouts fail for every arm | pi0.5 .30, GR00T .083, U .381 on new pairings | SR §30, §32.1 |
| Longer chains stay below the product of their skills | .567 against .817 | SR §29.1 |

SR = SCAFFOLD_RESULTS, PS = PLAN_SCAFFOLD.

**What the record lacks.**

1. **No VLA has been trained here with or without structure.** T8b (plain retrained pi0.5) and T16 (role prompts inside pi0.5) have not trained since 2026-11-06. PS §0 still reads: "VLAs are the wrong architecture is not licensed." The user's proposition needs these two reads, and it needs to survive them.
2. **U reads oracle positions.** Nothing runs from pixels.
3. **New layouts and chains have no mechanism that works.** Sixteen rounds of learned parts did not fix them. E, G and H located the layout loss in the approach: grasps fail by not reaching (PS §2).
4. **States 4-23 have shaped every rule.** A paper needs fresh states. The user allowed them on 2026-11-06.
5. **Unsupervised part discovery over-splits** (SR §9.2), and the lexicon rule breaks on one withheld task (PS §2).

Items 3 and 5 are where a System 2 can pay: geometry and language are what a frontier model and written control code handle well.

## 2. The thesis, sharpened

**Binding fails at a latent interface, and a typed interface makes binding a lookup.**

- A VLA passes "which object, which target, which verb" from language to motor through one entangled vector. The record shows the result: the carry stays bound to the held object's training target.
- GR00T is itself a two-system model, a VLM feeding an action head through a latent. It scores .167 on new pairings. So two systems alone do not fix binding. The interface between them decides.
- In an S1 built from typed routines, `place(target=plate)` takes an entity. A new pairing holds nothing new at the motor level. `SYS1_SYS2.md` §11.4 already warns against a latent interface for this reason.

So the paper's claim reads: **the fault sits in the architecture's interface, and a frontier model can build the system that replaces it.** This is narrower than "everything about VLAs is wrong", and the record can support it.

**The tenets, each as a test of S1's task programs** (Hupkes et al.; the framing of the earlier plan's §6):

| tenet | test on the built S1 | cells that exist |
|---|---|---|
| Systematicity | new object × target and verb × object pairings | N1-N8 |
| Productivity | chains longer than any demo | T6, T14 chains |
| Substitutivity | paraphrased instructions give the same program | to build |
| Localism | a patch to one routine changes only the tasks that call it | regression suite |
| Overgeneralisation | impossible or out-of-family instructions get refused | to build |
| Lifelong | a new atom costs one artifact and leaves old tasks exact | T9's stream |

## 3. How the repo maps onto the bet

| `SYS1_SYS2.md` | this repo |
|---|---|
| B0, the VLA | frozen `pi05_libero` and GR00T reads; T8b adds a VLA retrained on equal demos; T16 adds a VLA with role structure |
| Build mode 5, Fit | event thresholds (`ProprioHeld`, `HandMoved`), stop radius, verb modes read from demos |
| Build mode 6, Tabulate | the lexicon: word → role and atom |
| Build mode 7, Train | one-demo parts in `lifelong_ood/policy/parts_flow.py` |
| Arm H, hand-built S1 | U with its rules. A frontier model wrote it across these sessions under a person's direction, so it is a reference arm and no evidence for the bet |
| T3, robustness to change | the N2 suite: LIBERO-Goal's tasks with held-out pairings and a second layout. It adds composition as a named axis, which LIBERO-PRO lacks |
| Leakage rules | this session has read N2's states and BDDL predicates, so it cannot serve as S2. S2 runs as a separate API-called model in a sandbox that holds neither |

## 4. Routes not yet explored, ranked

1. **Written control for the approach, fitted parts for the contact (my recommendation).** Every arm fails new layouts at the reach. Code that solves IK and picks an approach by geometry does not depend on where the demos happened. S2 writes reach and transport; the one-demo parts keep the contact phase, where demos carry what code cannot guess. S2 declares a build mode per component, so the ledger shows each system's share.
2. **S2 as the parser and binder.** Replace the rarest-token lexicon with a tabulated parser that S2 writes and that can refuse. This targets the silent wrong parse and the second clauses that T16 could not route.
3. **Chains as programs with outcome checks.** T14-T15b broke at my hand-written transitions. Task programs with S1 outcome checks and local recovery are the bet's native answer.
4. **The gate as the measure of shared work.** Report success against the escalation rate ρ, per tenet. This is the user's "each handle a percentage of the work".
5. **A VLA as one S1 routine.** The hand-over result suggests a frozen VLA's pick can serve as a callable skill behind a typed seam. It would show the bet absorbing a VLA where the VLA is strong.

Route 1 is the one I would build first, because it attacks the one failure nothing in the record has moved, and because it produces the share-of-work measurement the user asked for.

## 5. Tests, cheapest decisive first

Each gets its own registration before it runs. Arms and names follow `SYS1_SYS2.md` §6.

| | question | design | needs |
|---|---|---|---|
| **P0** | Does the harness hold? | Sandbox server over LIBERO with a trivial S1, base tests, version store, ledger. No scoring code and no evaluation states inside | no key |
| **P1** | Can S2 build IK and control from the spec? | `SYS1_SYS2.md` T1, unchanged | key |
| **P2** | Can S2 build skills and task programs for K1-K8? | Build on training layouts only, with and without demos (K0, K1). Base tests: success on K1-K8 training layouts | key |
| **P3** | Does the built S1 compose? | K1 and K1-off against B0, the retrained VLA, U and the model as policy (L), on N1-N8 in both layouts and on chains, on fresh states. Verdicts per tenet | key; T8b trained |
| **P4** | How is the work shared? | Sweep the gate; success against ρ; frontier tokens and wall-clock per episode | key |
| **P5** | Does it improve? | Five update rounds on training layouts; `SYS1_SYS2.md` T6 | key |
| **P6** | Parity and precision on stock LIBERO | `SYS1_SYS2.md` T2 and T5 | key |

**Stop rules.** If P1 fails, IK gets provided and the bet narrows, as the user's scorecard says. If the retrained VLA matches the built S1 on new pairings, the proposition fails as written and we report it. If K1-off matches K1, S2's runtime share is zero, and the claim becomes "S2 builds, S1 runs".

## 6. Risks I can name now

- **Prior art is close.** `SYS1_SYS2.md` §10 lists RHO, CaP-Agent0, URAI and Agent as Policy. Its §10.3 names what remains. This plan adds one item to that list: a compositional account of why the VLA fails and a factorial test of whether the built system composes.
- **Perception.** P0-P3 can start from simulator poses, as U does, and must then move to rendered depth and segmentation. The user's order holds: supervised first, then the labels come off.
- **A closed model as S2.** Pin the model ID and effort, hash prompts, log and replay every transcript.
- **Cost.** Build and escalation tokens need a cap per stage, set by the user.
- **Two benchmarks.** N2 answers the composition question; stock LIBERO answers parity. Running both doubles the evaluation.

## 7. Decisions for the user

1. **The key.** The path of an env file that jobs may source, the model ID to pin, and a spending cap. I will not open the file.
2. **Existing code.** Whether this week's harness exists as code beyond `SYS1_SYS2.md`, and where.
3. **Where the build lives.** A new repo `s2-builds-s1/` as `SYS1_SYS2.md` §9.3 lays out, or a folder in this repo that reuses its N2 suite and readers. I recommend the folder, because P3 needs this repo's harness.
4. **The queued VLA arms.** T8b is the baseline the proposition must beat, and T16 is its fairest rival. I recommend keeping both and dropping nothing.

# Adversarial ICML 2027 review: "Routed by Cues: A Failure Atlas of VLAs on New Pairings" (paper/icml2027, build of 2026-11-23)

Reviewer role: hostile but fair, robot learning / VLA analysis. Written 2026-11-23.
Inputs: paper/icml2027/main.tex, parts/matrix.tex, parts/appendix.tex (all read in full), main.log (33 pages; the main text ends on page 8), paper/DRAFT.md (headings and the tvpb passages), paper/icml2027/TODO.md, docs/notes/novelty_review_adv.md (the earlier 3/10 review), and the GPU-hour tables in docs/notes/plan_oracle_pi0.md and docs/notes/raaf_plan.md. I could not render the PDF (no poppler on the login node), so I give locations as source line numbers. I read web sources at abstract level only. I ran no code and opened no results files.

---

## 0. Verdict in one paragraph

This version is a real improvement in honesty and organisation. It concedes the where/how framing to ReGuide, labels the module-removal test as separability in a system the authors built, moves the failed predictor to the appendix, runs ReGuide with the resolver added, and answers the earlier review's main threat (O4: "the token is ignored because the pathway is new") with an oracle arm and a pi0 arm. It also turns the paper's thesis into something stronger than before: that the failures need an explicit module, because "repairs inside the network fall short". The evidence does not carry that claim. On pi0.5 the best in-network route is not significantly worse than the explicit owner (A −.100 [−.261, +.050]). An input-only front end on the *frozen* network (tvpb, .938) reaches the same ceiling as ReGuide and W5 on scene 1. Published in-network training fixes (CRAFT, ACT-VLA) reach the same range on similar splits. The one significant contrast, pi0 servo-off against servo-kept (−.256), comes from one seed, development states and a modified pi0. It compares a network with the same network plus a hand-set feedback controller on privileged geometry, and it never shows whether the failures are routing errors or centimetre misses. Three of the five atlas rows have "not tested" in both in-network columns. What the paper supports is a well-measured atlas plus a comparative result: in this recipe, a hand-set controller on privileged geometry beats letting a 6k-step expert fine-tune learn to use the same geometry. That is not necessity.

**Score: 4 / 10 (weak reject), confidence 4 / 5.** Previous review: 3 / 10. Section 4 says what changed.

---

## 1. Did this version answer the earlier review (novelty_review_adv.md)?

| Earlier objection | Status now | Comment |
|---|---|---|
| O1: known diagnosis; the method loses to ReGuide, which owns the where/how split | **Mostly answered (rewrite).** Intro line 61 and Related line 89 concede ReGuide; the method is no longer a contribution; Jiang (Encoded but Not in Control), Wulff & Cangelosi, de Haan, RoboMME and LIBERO-Mem are cited | The ReGuide comparison is still sold as "convergence" (abstract, line 212), and line 212 says the two systems "share nothing but the explicit where". They share the perception stack (depth, calibration, segmenter; line 230). See §6 below. |
| O2: the removal test is circular, the cross terms are constants, F6 is design-induced, the patching is null | **Answered in the appendix, not in the abstract.** L.2 states all three caveats; line 160 summarises them | The abstract (line 52), contribution 2 (line 75) and the conclusion (line 240) still say removal returns "exactly its failure", and the abstract presents that as support for the architecture thesis. The where-module, the module the necessity claim is about, is not in the removal test (matrix row F1, last column). |
| O3: one seed, development-state selection, multiplicity | **Partly answered.** RAA-sL has a second seed (App. J); seed noise is stated (line 234) | No Holm correction (line 234 says none). The new headline contrasts (pi0 −.256, the route grid, the oracle) are all single-seed and on development states. |
| O4: "the token is ignored" may be an optimisation artefact; test on pi0 | **Answered, and the answer overturned the earlier best claim.** The oracle shows the new layer can be learned (φ .77); pi0 reads the token at .542 with no dose | The BC-shortcut result, which the earlier review called the paper's best asset, now holds only for a newly added pathway. The paper replaced it with "read but not in control" (§2.3 below), which is weaker evidence for a stronger claim. |
| O5: privileged inputs; baseline fairness | **Partly answered.** C3 rephrased (App. A); rgpr run (App. L.7); the missing escort is stated | Still no arm on the CRAFT, ACT-VLA or AC-VLA splits (line 234). This matters more now, because those are in-network fixes and the paper's thesis says in-network fixes fall short. |
| Cut the predictor and the W chronology; trim to 8 pages | **Done.** The main text ends on page 8; the predictor is App. G; W1–W6 is a table | |

---

## 2. Prior work: still overlapping or still missing (2025–2026)

Already cited and handled adequately: ReGuide, CRAFT, LIBERO-PRO/CF/Plus, Emukpere et al. (Robust Skills, Brittle Grounding), Encoded but Not in Control, Grant et al., Wulff & Cangelosi, ACT-VLA, AC-VLA, LiLo-VLA, w2VLA, OA-WAM, POT-VLA, FOCI, RoboMME, LIBERO-Mem, the LIBERO shortcut audit, de Haan et al.

**Missing, in order of how much they hurt:**

1. **Schmalz & Gupta, "Partial Composition, Not Blind Substitution: Decomposing a Grounding Failure in VLA Models"** (NeurIPS 2026 RoboPAD workshop). LIBERO-finetuned SmolVLA on untrained object–place combinations. Wrong destinations go to places the object was previously paired with, 10–11 points above chance. That is the paper's F1 and its known-partner logic. Two further points bear on this paper. (i) A 130-site causal patching sweep *appeared* to localise the failure, but the same sweep on a no-failure contrast gave the same profile, so they warn about localisation without such a control. Section 5 and App. D should be read against this. (ii) They argue the failure is partial composition, not vision overriding language. That is a direct counter-framing to "Routed by Cues", and the title should be defended against it.
2. **Gerigk et al., "When Listening Becomes Easier: Scrubbing Visual Cues for Shortcut-Free VLAs"**, arXiv 2610.10912 (Oct 2026). Visual shortcuts in VLAs; backbone-dependent susceptibility; shortcuts sit in early-layer action representations, and later layers differ in how much language corrects them. They propose a rollout-free "action margin" metric. Their fix, task scrubbing, is domain-adversarial *training inside the network*, and it "often eliminates visual shortcut learning". This is a direct counterexample to "repairs inside the network fall short" for cue-routed failures. It must be cited, and either tested or scoped out.
3. **CofactVLA**, arXiv 2608.04396 (Aug 2026). Names the "vision-override" failure (spurious salient objects or familiar layouts), and fixes it inside pi0.5 with counterfactual language-masked guidance plus KV-space confounder removal. That is another in-network repair of cue routing, this time at inference time.
4. **ThinkProprio**, arXiv 2602.06575, and **ReViP**, arXiv 2601.16667. Both study how the state entry point and encoding affect state use in pi0 versus pi0.5 (pi0.5 tokenises state as text; pi0 projects it to the action head). This bears directly on the "newly added pathway" explanation of the pi0.5 null and on the pi0 arms' deviation, and neither is cited.
5. **BeTTER, "Unmasking the Illusion of Embodied Reasoning in VLAs"**, arXiv 2604.18000. A failure taxonomy with controlled interventions (layout shifts, kinematic isolation) and an explicit *architectural* attribution (capacity compression, myopic downsampling). It is the closest prior "failure atlas rooted in the architecture".
6. **ROBOCOMBINE** (Zhang et al., NeurIPS 2026, "Exploring the Limits of Compositional Generalization in Embodied Manipulation"). Compositional recombination limits, failures at hand-off states, and a modular diagnostic interface (planner, trigger, policy). It is adjacent rather than overlapping, but it belongs in Related work.
7. **VLA-Trace** (arXiv 2605.30117) and **Embodied Interpretability / Interventional Significance Score** (ICML 2026): causal or interventional VLA diagnosis pipelines. Cite them as methodology neighbours of the cue test.
8. General failure taxonomies: arXiv 2511.11298 (symbol-grounding failure, among others) and arXiv 2606.08881 (semantic versus execution decomposition). One sentence is enough.
9. Shortcut literature beyond de Haan: Wen et al. 2020 (copycat) and Geirhos et al. 2020. The previous review asked for both, and neither is in the bib.

**Verify:** line 87 says "CRAFT reports 8.4%, 2.8% and 1.1% on held-out pairings for pi0.5, pi0 and GR00T". A search snippet of CRAFT's full text reports 2.0–50.9% mean on undemonstrated combinations across four models. Check which table and which subset the three numbers come from, and say so.

---

## 3. Claim-by-claim judgement

### 3.1 (a) "Necessity of an explicit owner" — under-evidenced, and contradicted in places by the paper's own numbers

The paper hedges the claim ("an empirical claim over the tests we ran", line 76), but its title-level message, the abstract ("Repairs inside the network fall short"), the conclusion and the authors' stated goal all rest on it. Each piece of evidence, read against the strongest counter-reading:

1. **pi0.5, the dose-by-servo grid.** The best in-network route on behaviour is A (relabel dose, servo off): .778 against RAA-sL .878, −.100 [−.261, +.050]. The interval includes 0. A0 (no dose, no token read) is −.192 [−.406, +.019], also including 0. Only B (−.233) and A+0 (−.275) are significantly below. Line 191 nonetheless says "no servo-free route recovers RAA-sL's release aim", and the matrix's F1 "more data" cell cites A+0 .603 and omits A .778 (matrix.tex line 12). That is selective, and a reviewer will find it in Table E1.
2. **Frozen pi0.5 with an input-only front end.** tvpb lifts frozen pi0.5 from .454 to .938 on scene-1 held-out pairings (App. B, "Input-only"), EQUIVALENT to ReGuide (.929) and above W5 (.892). No where-owner is involved: the same frozen network places correctly once the cue is removed from its input. On scene 1, then, F1 is not a *missing function*. The function exists and is routed by a cue, which is what the title says and what the matrix's "missing function: where the object goes" contradicts. tvpb appears only in the teaser caption and is never defined in the main text. A reader who finds it in App. B will ask why it is absent from the repair matrix.
3. **Published in-network fixes.** CRAFT (skill-alignment training), ACT-VLA (95.6 on Goal-OOD) and AC-VLA (73.3) reach the same range on LIBERO-Goal-style recombination splits, and so do Gerigk et al. (task scrubbing) and CofactVLA on cue-shortcut failures, all without a hand-set servo or privileged depth. Line 234 concedes that "our numbers are not directly comparable". For a necessity claim, that concession is fatal unless the claim is scoped to the authors' own recipe: a 6,001-step action-expert fine-tune with the VLM frozen, on 52k frames, with one seed.
4. **pi0, "read but not in control".** This is the only clean, significant support (§3.3).
5. **The matrix's empty cells.** F1m, F2 and F7 have "not tested" in both in-network columns (§3.2).

**What the evidence supports:** within this recipe and on development states, adding a hand-set closed-loop servo on privileged geometry beats letting the expert learn to use the same geometry (pi0: −.256; pi0.5: significant only against B and A+0). Explicit modules are *sufficient* for F1, F6 and F7 with privileged perception. That is sufficiency plus one comparative result, not necessity.

**Minimum additional evidence for a defensible necessity claim** (detail and costs in §5):
- (i) **A learned, separable where-owner.** A small head outside the expert, trained by BC on the same token, data and steps, that outputs the translation residual the hand-set servo outputs now. If the learned separate head matches the servo and the in-expert token routes do not, the claim becomes "an architecturally separate owner beats the distributed one at matched learning". That is the claim the authors want. Without it, "explicit owner" means "hand-coded controller", and nobody doubts that a P-controller on exact offsets hits a 3 cm tolerance better than a 6k-step fine-tune.
- (ii) **A failure-mode split of the in-network failures.** For A, A0 and pi0-A0: wrong target versus right target with a centimetre miss. If the failures are centimetre misses, the paper has shown a *precision* deficit, not missing ownership of "where the object goes".
- (iii) **The pi0 contrast on a held-out state set with a second seed** (§3.3).
- (iv) **Restrict or test the scope.** Either run the CRAFT split (eval-only), or state that necessity holds only against matched-data expert fine-tunes of this recipe, and discuss the training fixes in items 2–3 of §2 explicitly as counterexamples outside that scope.

### 3.2 (b) Do the "not tested" cells undermine the repair matrix? Yes, for the necessity reading, though not for the atlas

- **F2 and F1m:** both in-network columns are untested, and the explicit-module column is ReGuide, an external wrapper and not one of the paper's modules. Removal is untested. These rows support "an external where-correction helps", which ReGuide already published. They say nothing about necessity.
- **F7:** both in-network columns are untested. The cells are also worse than untested, because the paper's own rgpr experiment (App. L.7) shows that the F7 repair runs *through the network's input*. rgpr (resolver, unchanged VLA input) gives .681. rc (resolver plus painted view plus unique-name prompt into the same frozen VLA) gives .963. The +.281 comes from telling the frozen network which entity, after which the network does the where. That is a "target as an input" repair that works, and it belongs in that column.
- **F6:** the "target as an input" cell reports *servo* arms (RAA-sL .017, pi0-sL .000). Those are explicit-module arms, not input repairs. The genuine in-network arms do better: pi0-A0 .200, pi0.5 A0 .383, frozen pi0.5 .517. The paper's own diagnosis (App. B, L.2) says frozen pi0.5's F6 is placement precision, with 44 of 50 failures releasing within 3 cm, and "says nothing about memory in frozen pi0.5". Yet the row lists pi0.5 (cost .38) under "missing function: memory of the target". The row contradicts its own appendix.
- **Moved plate:** frozen pi0.5 succeeds .975, so frozen VLAs show no failure here. The failure is induced by the authors' narrow retrain R2. The "explicit module" cell (ReGuide .850 and .925) reports the moved-*bowl* cells (App. F), a different failure (frozen pi0.5 .375/.400) that has no atlas row.
- **"Exact target given, still fails":** a row of analysis results, not a failure. It is honest that it is "not a removal".

Net: of seven rows, in-network repair is tested against the explicit module on one failure (F1). The matrix reads as a uniform verdict, "inside falls short, modules repair", and it is not one. **Minimum fix (rewrite, 0 GPU-h):** keep the matrix, but (1) put A .778 and tvpb .938 in the F1 row; (2) move rc .963 into F7's input column with the reading "works once the referent is named"; (3) put pi0-A0 .200, A0 .383 and frozen .517 into F6's input column and drop pi0.5 from F6's "missing function: memory"; (4) split "moved plate" from "moved bowl"; (5) shade "not tested" cells so the reader sees the coverage. Then rewrite the claim as "where we could test it (F1), a hand-set where-controller beats in-network uptake of the same geometry on pi0; on pi0.5 the difference is not significant for the best route".

### 3.3 (c) Is pi0 "read but not in control" convincing? Not yet

Strengths: the rule was registered before the rollouts; the fit gate passed; K is non-inferior; the interval excludes 0 (−.256 [−.386, −.128]); and the port and parameter tree were checked against openpi.

Problems, from most to least serious:
1. **The contrast does not test "control".** pi0-A0 and pi0-sL differ only by the hand-set servo, a closed-loop translation residual computed from the same token. The contrast measures "network" against "network plus P-controller", and adding a controller that knows the exact offset is expected to improve centimetre-scale placement. "Not in control" would need the failures to be routing errors, with the arm going to the wrong target despite reading the right one. The paper never classifies pi0-A0's failures. Its own RAA analysis (47 of 51 failures were the right object to the right target, missing by 3.7–7.9 cm) suggests these will be precision misses too. If so, the honest label is "read but imprecise", which is far weaker.
2. **No "not read" baseline.** The registered rule compares servo-off with servo-kept. It never compares pi0 *reading* the token against pi0 *without* the token. The only such comparison is cross-backbone, and it points the wrong way: pi0-A0 (reads, .542) scores .506 on N, *below* pi0.5 A0 (does not read, −.001) at .686 (−.181 [−.344, −.008]).
3. **The deviation is larger than the text suggests.** The token *replaces* state dimensions 0–3, which pi0_libero trained on end-effector position and one rotation component (App. K, line 507: "pad columns exactly 0"). The arms therefore lose proprioception, and they also drop delta actions and change normalisation. The negative share of the servo-kept arm (−.150) is attributed to "leftover dependence on those columns", which means the network still treats the token partly as EE position. A pi0 that has lost its EE state could plausibly be imprecise for that reason alone, and the servo would repair it.
4. **Jiang et al. is mis-cited as an extension.** Jiang's finding is *encoded but not causally used*. Here the token *is* causally used (patching share .542; .755 on N). "Extends the finding of Jiang from internal representations to an explicit input" (lines 76 and 202) describes a different phenomenon. Rephrase it, or drop it.
5. **Single seed, development states, and a home advantage for the servo.** The servo constants (gain, clip) and the W designs came from diagnoses on these development states (line 234). The servo-kept arm is therefore evaluated where its controller was tuned, which favours it. The seed spread is unknown for pi0. For RAA-sL the seed gap was .056 on N, but per-task seed swings reach .48 (K5 .683 against .200), and the pi0 deficit sits in N4 and N6, the bottle pairings where the RAA-sL seeds also differed most (.883 against .772).
6. The development N on the capture subset (.400) and on the rollouts (.506) are consistent. Fine.

**Minimum to make it convincing:** (i) a failure-mode split of the pi0-A0 failures (0 GPU-h if rollouts logged end-effector and object poses; otherwise about 3 GPU-h to re-roll N4 and N6); (ii) both pi0 arms on a held-out state set, eval-only (about 10–12 GPU-h); (iii) a second seed of pi0-A0, plus a pi0 no-token control trained on the same recipe (about 2 × 10 GPU-h training and 2 × 6 GPU-h rollouts, about 32 GPU-h); (iv) ideally, both arms with the token appended in *pad* dimensions and EE state kept (about 2 × 16 GPU-h). If (i) shows wrong-target failures and (ii) holds, the claim survives.

### 3.4 (d) Single seeds, development-state selection, privileged inputs

- **Seeds.** Every trained policy has one seed except RAA-sL. The paper's own seed gap (.056 on N, .081 on K, line 234) is larger than several contrasts it interprets: W5 against RAA-sL +.017, A0 against A on F6, and RAA-sL against ReGuide −.019. Bootstrap intervals over tasks and states do not include seed variance, and the paper should say so next to every single-seed contrast, not only in Limitations.
- **Development states.** The atlas (except F1 and F2), all routes, the oracle and both pi0 arms are development reads. Only the W5/ReGuide comparison and the removal test are on held-out sets. Line 234 says "the held-out sets are free of that selection". That holds for set 1. W6 was designed from diagnoses of W5's failures, and the pooled analysis uses two sets seen before registration (App. L.5 says so). Write "free of the W1–W5 selection" and nothing broader.
- **Privileged inputs.** Within the pi0 contrast, information is matched: both arms receive the same token from the same estimator. That is fair. Against frozen RGB VLAs and against CRAFT or ACT-VLA, it is not: the explicit owner needs rendered depth, calibration and a segmenter, scene 2 uses a simulator channel with error 0, and F7 uses a template parser on a parsed scene. A necessity claim about VLA *architecture* that relies on inputs the VLA never receives conflates "missing information" with "missing ownership". The oracle arm partly addresses this, since a network with the exact token learns to follow it, but it also shows the network can use the information.
- **Multiplicity.** The paper runs dozens of registered tests with no correction, and states it. Holm-correct the 6–8 contrasts that the abstract uses. Most survive (the pi0 −.256 does). Doing it removes an easy objection.

### 3.5 Other claims

- **The atlas (contribution 1).** This is the most solid part. E over known partners, the same-state reference planner, five models and the scene-2 majority rule (.912 of 445, registered) are good. Problems: (i) "a causal cue test routes each one" (abstract) is true for F1, F1m and the scene-2 F1. F6's "cue" is the manipulation itself, F7's "cue" is ambiguity, and F2's cue test is weak (role views +.183 [+.021, +.412] on pi0, nothing on SmolVLA). (ii) Scene-2 F1 fails the atlas's own condition 2: the planner pools .65 [.25, 1.00] and scores .333 on N1 (App. A), against the .80 rule. (iii) F7 "all five" (matrix line 16) counts pi0, X-VLA and SmolVLA, which fail even the unambiguous named-bowl control (0 of 20, App. B), so F7 is untestable on them, not present. (iv) The "every model" in scene 2 includes GR00T, which is below the testability floor (.508), while X-VLA and SmolVLA have no scene-2 entry in Table B1.
- **"The functions are spread through that path" (Section 5 heading, line 140).** The registered *distributed* criterion (participation ratio ≥ .2) failed (.155), so the effect is *less* distributed than the criterion required. Meanwhile the residual-stream patch shows a cliff at layer 14 (1.21 to .10), a localised read site. The heading asserts the opposite of the registered outcome. Rewrite it as "the effect does not localise to a single write or band, and the expert reads the prefix through layer-14 keys and values". Also add the no-failure control contrast that Schmalz & Gupta show is needed: the K set is a partial control and should be framed as one.
- **"The servo takes over the function" (line 214).** On pi0.5 the takeover evidence is A+ against A+0 (.124 to −.007) and the oracle frames. The two RAA-sL seeds (−.007, −.002) are *not* evidence of takeover, because the no-servo, no-dose model A0 also reads −.001. Remove "both seeds of the main retrain" from the takeover argument.
- **W5 against ReGuide "converge".** The registered non-inferiority failed twice, the lower bounds are −.114 and −.147, and the pooled analysis uses sets seen before registration. "Converge at .89 to .92" is also numerically wrong: on set 2, W5 is .931 and ReGuide .883 (Table C2). Write "land within a few points of each other; non-inferiority not shown". W5 against RAA-sL is +.017 on N (not significant on any set), so the convergence is RAA-sL's servo, not W5's release-point design.

---

## 4. Score and what changed

**4 / 10 (weak reject), confidence 4 / 5.**

Up from 3 because:
- The paper is coherent, fits the page limit, concedes prior art and labels its caveats. The process is unusually rigorous: registration, held-out sets and a frame-selection rule fixed before looking.
- The atlas with E, the planner check and the scene-2 majority rule is a genuine, reusable diagnostic.
- The oracle arm cleanly kills the "optimisation artefact" reading. That is good science even though it cost the authors their best claim.
- rgpr, the second seed and the pi0 arm were all run as the earlier review asked.

Not higher because:
- The headline has shifted to necessity, and the evidence for it is one single-seed, development-state, modified-pi0 contrast against a hand-coded controller.
- The paper's own tvpb (.938) and route A (.778, not significantly different) cut against it, and published in-network fixes cut against it at the field level.
- The matrix is mostly empty where necessity would be tested, and it reports cells selectively (A+0 rather than A; servo arms in the F6 input column).

The earlier review's route to 6 ("if the BC-shortcut replicates on pi0") is closed: it did not replicate. With the rewrites in §5 (O1, O2, O5) plus the cheap experiments (O3 eval-only, failure-mode split), I would score it **5**. With a learned separable owner that matches the servo while in-expert routes do not, plus a second pi0 seed, **6**. Below that, the realistic home is CoRL 2027 or RA-L as an analysis paper.

---

## 5. Ranked objections, minimum fixes and value per GPU-hour

GPU-h on A40, from the authors' own tables: training about 9–10 GPU-h per arm (single chain); development rollouts about 6 GPU-h per arm (Phase A about 2, Phase B about 4); a capture plus probe about 1–2 GPU-h per arm.

**O1 (most likely, decisive). "The necessity claim is not supported. Your own best in-network route is not significantly worse (A −.100, CI includes 0); an input-only front end on the frozen VLA reaches the ceiling (tvpb .938); published in-network training fixes (CRAFT, ACT-VLA, task scrubbing, CofactVLA) reach the same range; three matrix rows are untested."**
- Minimum fix, rewrite (0 GPU-h, highest value per GPU-h):
  - Scope the claim to "in matched-data expert fine-tunes of a frozen-VLM VLA, a hand-set where-controller on the same geometry beats in-network uptake (significantly on pi0, not significantly for the best pi0.5 route)".
  - Put A .778 and tvpb .938 in the matrix, and fix the F6, F7 and moved-plate cells as in §3.2.
  - Discuss CRAFT, ACT-VLA, Gerigk et al. and CofactVLA as in-network repairs outside the scope.
  - Change "Repairs inside the network fall short" in the abstract to the scoped form.
- Minimum fix, experiment (about 25–30 GPU-h, highest value per GPU-h among experiments): **a learned, separable where-head.** Same token, data and steps; a small MLP outside the expert that outputs the translation residual (trained by BC on the residual the demonstrations imply), against the hand-set servo and against A. One training run (about 10) plus development rollouts (about 6) plus one held-out set eval-only for it and the servo arm (about 8–12). If it matches the servo, "explicit owner" becomes an architectural claim instead of a claim about hand-coded control. This single experiment is what moves the paper from 4–5 to 6.

**O2. "'Read but not in control' is one seed, development states, a pi0 with its EE state overwritten and no delta actions, and no breakdown of whether the failures are wrong-target or centimetre misses. It contrasts a network with network-plus-P-controller, and it extends Jiang in name only."**
- Fixes, cheapest first:
  - (a) Failure-mode split of the pi0-A0 and pi0-sL failures, and of A and A0 on pi0.5: wrong target, wrong release height, or xy miss in cm. 0 GPU-h if poses are logged, otherwise about 3 GPU-h for N4/N6 re-rolls.
  - (b) Both pi0 arms on a held-out state set, eval-only, about 10–12 GPU-h. The sealed fifth set is the natural place, with a registration.
  - (c) A second seed of pi0-A0 and pi0-sL plus a no-token pi0 control, about 45 GPU-h. If the budget is short, run seed 2 of pi0-A0 and the no-token control only, about 32 GPU-h.
  - (d) Optional: arms with the token in pad dimensions and EE state kept, about 32 GPU-h.
  - Rewrite (0 GPU-h): fix the Jiang sentence; say "development states, one seed" in the abstract sentence that quotes .542 and −.256.
- Value: (a) + (b) are the best value per GPU-h after O1's rewrite, because they decide whether the paper's most-cited number survives at all.

**O3. "The removal test is still circular and is still in the abstract as evidence ('returns exactly its failure'); the where-module is not in it; the memory half measures a design-induced failure."**
- Fix: rewrite (0 GPU-h). Remove "exactly"; say "in the system we built"; move the removal result out of the abstract's causal chain or present it as an engineering property.
- Optional experiment that would make it bite: remove the servo from W5 at serving time on a held-out set (eval-only, about 6 GPU-h). That gives the where-module a removal cell, and the matrix's F1 "not in the removal test" disappears. Expected outcome: back to roughly RAA levels. Cheap and on point.

**O4. "Missing related work: Schmalz & Gupta (the same failure on SmolVLA; a patching-artefact control; 'partial composition, not substitution'), Gerigk et al. (cue shortcuts fixed in-network), CofactVLA, ThinkProprio/ReViP (state entry point), BeTTER (an architecture-rooted failure taxonomy)."**
- Fix: rewrite (0 GPU-h). Six to eight sentences in Related work. Answer Schmalz & Gupta's counter-framing directly: the scene-2 majority rule and the E measure are the evidence for routing over partial composition. Frame the K set in Section 5 as the no-failure control for the patching profile.

**O5. "Single seeds, development states, no multiplicity correction, privileged depth: LIBERO-only, two scenes."**
- Fix: Holm-correct the abstract's contrasts (0 GPU-h). Put "(one seed; development states)" beside every single-seed number in the main text. One sentence on the LIBERO audit, presenting the paired, registered statistics as the answer to it.
- Do not spend GPU on more W variants or more atlas models. They do not move the score.

**Priority list for a finite budget (deadline Jan 22, 2027):**
1. All rewrites: O1 rewrite, O2 rewrite, O3, O4, O5 statistics. 0 GPU-h. Score 4 to about 5.
2. O2(a) failure-mode split, 0–3 GPU-h.
3. O2(b) held-out eval of the pi0 arms, about 11 GPU-h.
4. O3 servo-removal eval on a held-out set, about 6 GPU-h. Items 2–4 together cost about 20 GPU-h and make the necessity section defensible at its scoped strength.
5. O1 learned separable where-head, about 25–30 GPU-h. This is the experiment that could move the score from 5 to 6.
6. O2(c) second pi0 seed plus a no-token control, about 32–45 GPU-h.

Total for items 1–6: about 85–95 GPU-h, which fits easily before Jan 22 at 4 concurrent GPUs. Items 1–5 (about 50 GPU-h) carry most of the gain.

---

## 6. Text problems (locations are source lines)

**Overclaiming**
1. main.tex 52 (abstract), 75, 240: "removing a module returns exactly its failure on three held-out state sets". Set 2 ran W6, not the W5 policy (Table removal); the resolver cross terms hold by construction; the where-module is not tested. Drop "exactly" and say "in the system we built".
2. main.tex 52, 212, 240: "converge at .89 to .92". Table C2 gives W5 .892/.931/.894 and ReGuide .914/.883/.917, so the range is .883–.931. Non-inferiority was not shown on either registered set. "Converge" overstates both.
3. main.tex 212: "Two systems that share nothing but the explicit where". They share the perception estimator (depth, calibration, segmenter; line 230) and the pi0.5 lineage.
4. main.tex 52: "Repairs inside the network fall short". This is true for B and A+0, not shown for A (−.100 [−.261, +.050]) and contradicted by tvpb (.938, an input-only fix to the frozen network).
5. main.tex 52: "a causal cue test routes each one". There is no cue edit for F6 or F7, and F2's test is weak (see §3.5).
6. main.tex 76 and 202: "extends the finding of Jiang et al. from internal representations to an explicit input". Jiang shows encoded but unused; here the token is used (.542). These are different phenomena.
7. main.tex 191: "no servo-free route recovers RAA-sL's release aim". The A and A0 intervals include 0. Also, "release aim" is not what the success numbers measure.
8. main.tex 214: "both seeds of the main retrain (−.007 and −.002) show the same takeover". A0, with no servo, also reads −.001, so these values cannot show takeover.
9. main.tex 140 (heading "The functions are spread through that path"). The registered distributed criterion failed (PR .155 < .2), and layer-14 KV is a localised read site (App. D).
10. main.tex 234: "the held-out sets are free of that selection". W6 was designed after set-1 diagnoses, and two of the three pooled sets were seen before registration (App. L.5).
11. matrix.tex 16: F7 "all five, at least one family". pi0, X-VLA and SmolVLA fail the named-bowl control (0 of 20, App. B), so F7 is untestable on them.
12. matrix.tex 15: F6 lists pi0.5 under "missing function: memory", which contradicts App. L.2 ("says nothing about memory in frozen pi0.5") and App. B (44 of 50 failures release within 3 cm).

**Inconsistent or confusing numbers**
13. main.tex 180 and appendix 235: "the servo alone carries .878", while servo plus token together give .871. The part exceeds the whole; explain the non-additivity. Table J1 (appendix 399) gives seed-1 in.servo **.920**, not .878. Reconcile the two. That .878 also equals RAA-sL's N success is a coincidence that invites a copy error, so check it.
14. main.tex 68, 118 and appendix 64/76: π0.5 baseline success on scene-1 N is .450 (role-view test) in one place and .454 (tvpb, ReGuide) in another, and E for the same model is also +.450. Readers will conflate E and success. State the state set for each, and use one baseline.
15. main.tex 173: "The trained tasks stay level (+.086 [+.011, +.172])". The interval excludes 0, so RAA *improves* trained tasks (R2 .753 to RAA .839).
16. main.tex 175: ".878 (RAA-sL)" is attributed to "a fixed servo term", but RAA-sL also changes the initialisation, and RAA-s to RAA-sL is NOT SUPPORTED (+.161 [−.017, +.311]). App. L.1 says part of the gain is from the initialisation; say so in the main text.
17. main.tex 156 (Figure removal caption): it says "development states (RAA-L, RAA-sL) and two held-out sets (RAA-sL-W5)", then reports three held-out sets. It also gives "+.41" and "+.37" without intervals, while Table removal has +.408 and +.367 with intervals (TODO.md flags this as unchecked).
18. main.tex 97, 171 and App. A, Fig A1: task counts conflict. Scene 1 has "five trained tasks (K1–K4, K6)"; the ladder trains on "LIBERO-Goal's eight known tasks"; the pools are K1–K6 (with K5, the rack, in Table C2); Fig A1 shows a "K1–K8 / N1–N8" grid. Define K5 and say which tasks are trained in each experiment.
19. main.tex 208: "it took six variants, W1 to W6". W5 is the final design and W6 came after it and lost, so five variants led to the design.
20. main.tex 180: "Four data routes test it", then five (A, A0, B, A+, A+0). Say "three routes, a comparator and a post hoc arm".
21. main.tex 191: "the fifth held-out state set stays sealed". The main text mentions only three held-out sets; the reader does not know a fourth or fifth exists.
22. main.tex 128 and appendix 13: "every model scores 0 of 720" in scene 2. GR00T is below the testability floor (.508), and X-VLA and SmolVLA have no scene-2 entries in Table B1. Say which models.
23. main.tex 99 and appendix 17: the atlas entry condition requires the planner to succeed on the same states (threshold .80 in App. A). Scene-2 F1 has the planner at .65 [.25, 1.00] and .333 on N1, so it fails the paper's own entry rule, yet it is in the atlas.
24. matrix.tex 17: the moved-plate row's "explicit module: ReGuide .850 and .925" are moved-*bowl* cells (App. F). On the moved plate, frozen pi0.5 succeeds .975, so the row's failure belongs to R2, not to a VLA.
25. matrix.tex 12: the F1 "more data" cell cites A+0 .603 and omits A .778. The F1 row also omits tvpb .938.
26. matrix.tex 15: the F6 "target as an input" cell reports servo arms, not input repairs, and omits pi0-A0 .200, A0 .383 and frozen pi0.5 .517.
27. main.tex 216: W5's covered-plate .817 is compared against RAA-sL .050, a design-induced failure. The relevant comparator is frozen pi0.5 (.500 on set 3; on set 1 W5 .650 *ties* frozen .667, App. C).
28. main.tex 52: "learned where-share −.007" is cited for "a new input pathway stays unread unless the data force it". −.007 is the *servo-kept* RAA-sL value. The no-dose, servo-free value is A0's −.001. Use A0.
29. main.tex 59: "VLAs reach .98 to 1.00 success on the tasks they trained on". That holds for pi0.5 and GR00T in scene 1 only (App. B); pi0 scores .81 on the scene-2 trained halves, and the RAA/pi0 arms score .81–.94.

**Undefined or unclear terms in the main text**
30. tvpb (line 68, teaser caption), rgp, RAA-c and RAA-s (matrix), RAA-sL-W5 (line 151), SP/NG (line 124 "one family … the negated one"), F6_b (line 216), "role views", "role phrases", "front end", "hand-over" and K5 are never defined in the main text. Add a one-line glossary or define each on first use.
31. main.tex 128: "The views that repair scene 1 give .000 here, so an input edit removes only cues that the task does not need" is a non sequitur as written. Say that in scene 2 the cue is position, which the grasp needs, so painting it out removes needed information (the wording of App. L.3).
32. main.tex 99: "A failure that misses the second condition belongs to the task". Name the threshold (.80) in the main text.
33. main.tex 101: the pass rule in the main text omits the prefix-share condition that App. E says "fails every servo-free route by construction". Either state it or say it was not binding.
34. main.tex 232: "the .256 gap is a single sample" is honest, but the abstract and conclusion quote .256 without "development states, one seed". Make them agree.
35. main.tex 87: verify the CRAFT numbers (8.4/2.8/1.1) against the table you take them from (see §2).
36. Bib: add Schmalz & Gupta, Gerigk et al., CofactVLA, ThinkProprio, ReViP, BeTTER, ROBOCOMBINE, Wen et al. 2020 and Geirhos et al. 2020 (TODO.md also lists unconfirmed short names for POT-VLA and w2VLA).

---

## Sources (web, read for this review; abstract level)
- Schmalz & Gupta, Partial Composition, Not Blind Substitution (NeurIPS 2026 workshop): https://neurips.cc/virtual/2026/162908
- Gerigk et al., Scrubbing Visual Cues for Shortcut-Free VLAs: https://arxiv.org/abs/2610.10912
- CofactVLA: https://www.alphaxiv.org/abs/2608.04396
- ROBOCOMBINE / Exploring the Limits of Compositional Generalization in Embodied Manipulation (NeurIPS 2026): https://neurips.cc/virtual/2026/poster/139255
- BeTTER, Unmasking the Illusion of Embodied Reasoning in VLAs: https://arxiv.org/abs/2604.18000
- ThinkProprio: https://arxiv.org/html/2602.06575 ; ReViP: https://arxiv.org/abs/2601.16667v1
- VLA-Trace: https://arxiv.org/html/2605.30117 ; Embodied Interpretability (ICML 2026): https://icml.cc/virtual/2026/poster/63997
- Failure taxonomies: https://arxiv.org/html/2511.11298v1 , https://arxiv.org/html/2606.08881
- CRAFT: https://arxiv.org/pdf/2610.00524 ; ReGuide: https://arxiv.org/pdf/2609.38616 ; Robust Skills, Brittle Grounding: https://arxiv.org/abs/2602.24143 ; LiLo-VLA: https://arxiv.org/html/2602.21531v1

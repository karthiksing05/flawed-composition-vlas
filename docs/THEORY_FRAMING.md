# Theory framing: composition in VLAs is limited by binding (brainstorm, 2026-10-28)

Status: a brainstorm for the user. Every number is a cited past result. Nothing here is a claim until the listed test runs.

## The thesis in one sentence

A VLA's motor operators compose, but its bindings do not. Systematic binding needs atom experts that are invariant across
contexts, and such experts are learnable only when the structure is selected for transfer across contexts, not for fit within
them.

## Five pillars

**P1. Decomposition (measured).**
- pi(a | o, l) = sum_b P(b | o, l) O(a | o, p_b), and success = P(correct b) x competence.
- Evidence: E2b (calibration slope 0.997); S0 (an oracle binding cuts binding error 0.234 -> 0.048 and leaves motion errors);
  ReGuide's beta x gamma.
- Contribution: the factorisation used as a PREDICTION. Success on unseen compounds is predicted from separately measured
  binding and competence.

**P2. Binding is where systematicity fails (Fodor & Pylyshyn, localised).**
- Systematicity needs constituent structure. In a VLA it exists in the operators (shared, causal, pose-parameterised: D4, D5)
  and is missing in the map from phrase constituents to scene entities.
  - The goal role is scene-exemplar-keyed (E1: +0.48).
  - Binding is committed by the grasp (R7).
- This is the classic binding problem (Treisman; Smolensky; Hummel & Holyoak), found inside a modern robot policy.

**P3. Depth = the number of binding operations (measured).**
- A referring expression of depth d needs d + 1 atom resolutions (type, then instance, then relation or part).
- Factorised account: P(correct) ~ product over atoms of (1 - eps_atom), so error compounds with depth. E4: 0.18 / 0.38 / 0.945.
- Holistic account: error depends on the distance to the nearest training exemplar (MCD).
- The two accounts make different predictions on a compound-divergence (CD) x depth grid. B1.3 builds that grid.

**P4. Compositional binding = invariant prediction across contexts (the learnability claim; new).**
- T2's atom-invariance assumption is invariant causal prediction (Peters et al.; IRM), with layouts as the environments.
- An atom expert is compositional iff its referent likelihood is the same in every layout.
- **Identifiability:** when one training layout determines the referent, a layout code and an appearance code are
  observationally equivalent, and no criterion that scores fit on training layouts can tell them apart.
  - Compositional structure is identified only by variation across contexts (compositional support, as in Wiedemer et al.),
    and only by a criterion that scores transfer: prequential / leave-layout-out MDL.
- **The B1 series is an empirical demonstration:** each fit-based criterion found the shortcut its features allowed.
  - B1: absolute position in the ViT features, so "left" became a remembered location.
  - B1.1: types bound, relations not.
  - B1.2: identity-coding MDL put nouns into a layout code, and transfer fell 0.86 -> 0.30.
- **Cognitive parallel: the shape bias.** Children learn that nouns generalise by shape and not by location (Landau, Smith &
  Jones 1988). Training on varied exemplars induces the bias (Smith et al. 2002). B1.2 lacks a shape bias, and prequential MDL
  over varied layouts is the mechanism that should induce one. Prediction: nouns and colours go to appearance and lateral /
  part words go to relative geometry ONLY when the selection criterion scores held-out layouts.
- **Invariance predicts transfer, word by word:** the words whose invariance test is rejected are exactly the ones that fail to
  transfer (B1: front, left, right; B1.1: cabinet).

**P5. The crutch (measured on the oracle, open for deployment).**
- A small, external, invariant binder supplies bindings. A once-trained adapter writes them where the policy reads binding (deep
  image K/V, R3c / S0).
- Bound (T4): success >= (1 - sum eps) x kappa_adapter x competence.
- The VLA keeps every ingredient, and the crutch supplies only the invariant part it fails to learn by fine-tuning.

## Falsifiable predictions (each a test we can run)

| # | prediction | test | status |
|---|---|---|---|
| 1 | Holistic binding error rises with CD at fixed atom divergence; factorised stays flat | B1.3 CD x depth grid, interaction CI | running |
| 2 | Factorised error ~ product of atom errors across depth (multiplicativity, Okawa et al.) | B1.3 per-depth vs atom budget | running |
| 3 | A word transfers iff its expert is invariant across layouts | per-word invariance vs per-word held-out hit | partly seen (B1, B1.1) |
| 4 | Transfer-scored (prequential) MDL recovers the shape bias: nouns -> appearance, relations -> geometry | B1.3 channel table | running |
| 5 | Identifiability fails without layout variation: with one layout per word, every criterion picks a layout code | ablate layout diversity in the generated pool | to add |
| 6 | Success on unseen compounds = binding x competence, predicted before rollout | closed loop with the B1.3 binder + S0 adapter | after B1.3 |
| 7 | The same pattern holds in other VLAs (operators compose, binding does not) | E2b / E4 on pi0, SmolVLA, GR00T | needs budget |

## Possible titles

- "Compositional Operators, Holistic Binding: Why VLAs Fail to Compose and How Invariance Fixes It"
- "Systematicity Lives in the Binding"
- "Learning to Bind by Transfer, Not by Fit"

# Plan BINDER: binding is where VLA systematicity fails, and how to make it systematic (2026-10-27)

Follows PLAN_BIND (theory), PLAN_SOLVE (S0-S4). The user's goals:
- **(A)** show that binding at depth is a failure mode of existing VLAs AND existing fixes;
- **(B)** build, from the ground up, a solution that resolves it across several VLAs and datasets, with a provable accuracy
  guarantee, inside a true compositional theory.

## 0. Where we stand

- **Proven:** success = P(correct binding) x competence (E2b, slope 0.997). Binding error rises with referring-expression depth:
  type 0.18, instance 0.38, sub-region 0.945 (E4). The goal role is scene-exemplar-keyed (E1).
- **Repair:** a correct binding delivered by a once-trained adapter cuts binding error by 80% (S0).
- **Gap:** every deployable binder fails at depth 1-2 on held-out cells (S1, S1b, S1c, GroundingDINO, RoboPoint, S3a gate, S4).

## 1. Theory (what is claimed and what can be proved)

A task binds roles i = 1..k. Role i's referring expression is a chain of atoms (body type, instance attribute, part, relation).

- **T1 (holistic binding is not systematic).**
  - An exemplar binder answers a test compound with the binding of its nearest training compound, P(b | k(o, l)).
  - Its error on a held-out compound is at least the probability that the nearest training compound binds the role differently.
  - That probability rises with compound divergence (CD) at fixed atom divergence.
  - E1 measures this for the policy; S1's B1b shows it for a learned holistic binder (seen 1.00, held-out 0.38).
- **T2 (factorised binding is systematic).**
  - Factorised binder: each atom is resolved by a primitive g_a, applied in the order the parse gives. Each primitive sees only
    its own atom and its input region.
  - Invariance assumption (testable): a primitive's error given its atom does not depend on the other atoms of the compound.
  - Then, by a union bound, P(binding error | c) <= sum over atoms a in c of eps(a), for ANY compound c of seen atoms. The bound
    does not depend on CD.
  - Prediction: holistic error rises with CD and factorised error stays flat. We test the invariance assumption directly by
    measuring eps(a) conditioned on the co-occurring atoms.
- **T3 (calibrated guarantee).**
  - Split-conformal calibration of each primitive's score on held-out calibration cells gives a binding set that contains the
    true binding with probability >= 1 - alpha, under exchangeability of cells.
  - Act when the set is a singleton; otherwise abstain or ask.
  - This is the provable accuracy statement. Its price, the abstention rate, is reported.
- **T4 (end to end).**
  - success >= (1 - sum eps) x kappa_adapter x competence.
  - kappa_adapter = the adapter's fidelity with a correct binding (S0: residual binding error 0.048).
  - competence = the policy's operator success with a correct binding (E2b).
  - Every term is measured separately, so the bound is checked against data.

## 2. Part A: the failure atlas across models and existing fixes

**Benchmark: BIND-depth.** Units are stratified by referring-expression depth (0 / 1 / 2) x CD bin x role.
- **Datasets:**
  - LIBERO-90-C (own split, never-seen guaranteed);
  - authored LIBERO recombinations with depth-2 phrases;
  - CALVIN-C (type / colour level: the depth-0 control);
  - one public suite as a reference: LIBERO-PRO / LIBERO-Plus, with contamination noted.
- **VLAs:** pi0.5 (own), and pi0, SmolVLA, X-VLA, GR00T N1.5. Public checkpoints are reported as contaminated references unless
  retrained on the split.
- **Existing fixes, run on the same units:**
  - ReGuide (perception);
  - ECT / CRAFT-style counterfactual fine-tuning;
  - pointing pipelines (RoboPoint, RoboRefer);
  - CAG-VA;
  - the one-demo token (ceiling).
- **Claim A:** every model and every fix degrades with depth, and the holistic ones with CD (T1).

## 3. Part B: the solution, built up in stages, each with a gate

| id | what | gate (pre-registered) | A40-h |
|---|---|---|---|
| **B0** | Does the VLA's own pretrained backbone still bind? Offline screen on the S1 units: PaliGemma-mix `detect` / `segment`, the pi05_base backbone with the same prompts, Qwen2.5-VL and Molmo pointing | held-out depth-1 hit >= 0.70, depth-2 seen-body hit >= 0.70, depth-0 >= GroundingDINO (S1c's bars) | about 1 |
| B1 | Factorised neuro-symbolic binder: body (detector or B0 winner) x instance head x part head x relation in the body frame. Supervision from sim-generated layouts (labels free, no demos) under an MCD split over the binder's atoms | the same bars, plus binder error flat in CD (slope CI includes 0) while B1b's rises; invariance test of T2 | 2-3 |
| B2 | Conformal calibration (T3) | empirical coverage >= 1 - alpha on held-out cells; abstention rate reported | CPU |
| B3 | Closed loop: binder + S0 adapter on pi05_l90c | closes >= 0.5 of the zs-to-oracle gap; training-cell damage >= -0.05; T4 bound holds | 3-5 |
| B4 | Second and third VLA (adapter recipe ported, trained once each) and CALVIN-C | same direction of effect | about 30-60 (needs approval) |

- **B0 framing:** if the base VLM binds where the fine-tuned VLA does not, the crutch claim becomes concrete. The VLA holds the
  ingredient, fine-tuning lost the binding, and the crutch restores it from the model's own pretrained backbone.
- B1 is the compositional core whatever B0 finds: B0 may still fail at sub-regions.

## 4. Rules (unchanged)

- SLURM only (overcap / tail-lab, explicit -p/-A, exclude list).
- Pre-register before every read. Audits stay at 0 / 0. Never open split_sealed.json.
- Downloaded code and weights go in their own directory and run with python -I.
- The confirmatory read, the sealed set, and any budget beyond B0 need the user's approval.

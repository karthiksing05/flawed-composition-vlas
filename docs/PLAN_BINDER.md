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

## 3.1 B1 revised (user, 2026-10-27): unsupervised structure, prototypes, few parameters

The binder is a product of experts over atom prototypes. Its structure is discovered without labels.

- **Referents from the demos, not from sim labels.**
  - On every training demo, the object role's referent is the region under the gripper at the grasp, and the goal role's referent
    is the region under the gripper at the release (the existing contact-event segmentation).
  - This is cross-situational word learning (Siskind; Frank, Goodman & Tenenbaum): across tasks, the words that co-occur with a
    referent are bound to it. It needs no annotation, so it carries to CALVIN and to real data.
- **Candidate regions without labels.**
  - Bodies and parts are clusters of the policy's own vision-tower patch features (or DINOv2 as a comparison), split
    recursively inside each body.
  - Each region gets two feature channels: appearance (pooled patch features) and geometry (position, height and extent relative
    to its parent body).
- **Atoms are prototypes.**
  - Each word w gets a prototype (mean and diagonal variance) in BOTH channels, fitted from its co-occurring referents.
  - **Structure discovery by minimum description length:** each word is assigned to the channel where its prototype is
    tightest.
    - Appearance-tight words are type and instance words ("ketchup", "black bowl").
    - Geometry-tight words are relation and part words ("top", "middle", "front").
    - Words tight in neither are function words, and carry no binding.
  - The composition order follows the parse's head structure ("A of B": resolve B, then A inside B), checked by likelihood on
    the training cells.
- **Binding is a product of experts.**
  - score(region | phrase) = sum over the phrase's content words of log N(region features in w's channel; prototype_w),
    evaluated inside the parent the parse gives.
  - Each expert sees one atom and one channel, so T2's invariance assumption holds by construction. The conformal set (T3) is
    computed on the same score.
- **Size:** about two vectors per word and no trained network; a few KB. The vision tower is shared and frozen.
- **Risks:**
  - Words that always co-occur in training cannot be separated (a confound). That is exactly where held-out combinations test
    it, so this is reported per word.
  - Fine instance discrimination (ketchup vs tomato sauce) may need sharper appearance features than the vision tower gives.
  - Part clusters may not align with drawers or shelf levels. Report part-proposal recall against sim masks, used for scoring
    only.

## 4. Rules (unchanged)

- SLURM only (overcap / tail-lab, explicit -p/-A, exclude list).
- Pre-register before every read. Audits stay at 0 / 0. Never open split_sealed.json.
- Downloaded code and weights go in their own directory and run with python -I.
- The confirmatory read, the sealed set, and any budget beyond B0 need the user's approval.

## 5. B0 pre-registration (2026-10-27; ledger m3_prereg tag pb0; written after a smoke on 6 TRAINING units, before any dev unit)

**Units and metric.** S1's 240 dev units (results/bind/s1/labels.json), the frames S1, S1b, S1c and S3b scored. S1's hit
(s1_read.hit: the peak token lies in the target and is covered by it at least as much as by every competitor), the two-level
task-cluster bootstrap (d4_read.rate). Code: experiments/b0_vlm.py (predictions, isolated venv in /coc/flash10/ksingara3/b0_ext,
`python -I`), experiments/b0_read.py (read, chunks env).

**Arms.**
- **pi05_base backbone** (primary): the PaliGemma weights inside openpi's pi05_base (the init of pi05_l90c), in the HF PaliGemma
  class. Prompts `detect {phrase}` (primary) and `segment {phrase}`.
  - The smoke found fluent English captions and yes-answers, and no `<loc>` token, under every prompt. This is recorded as found.
    No prompt or decoding was changed.
- **Qwen2.5-VL-7B-Instruct** (primary): `Locate "{phrase}" in the image and output its bbox coordinates in JSON format.` on a 2x
  upsample (448).
- **Molmo-7B-D-0924** (primary): `Point to "{phrase}".`
- **pi05_l90c@5000 backbone** (supplementary, detect only): the fine-tuned policy's own backbone.
- **Not run: gated, no local weights.** PaliGemma mix (`google/paligemma-3b-mix-224`, not the policy's base) and PaliGemma pt-224
  (`google/paligemma-3b-pt-224`, the policy's base).
- **Variants.** {phrase} is the role phrase exactly as S1's parse gives it. Every arm also runs a context variant with the full
  instruction (PaliGemma: `detect {phrase} (from: {lang})`; Qwen / Molmo: the prefix `The robot is told: "{lang}". `).

**Conversion to S1's format.**
- A box gives its centre token as the peak, and the tokens covered >= 0.5 as the mask (B1a's rule).
- Points use S3b's rule.
- A segment output is decoded by big_vision's VQ-VAE, with its maximum-coverage token as the peak.
- An unparseable answer is a miss.

**Bars (S1c's, point estimates, per arm x config).**
- (i) held-out depth-1 hit >= 0.70 (15 units, 3 tasks);
- (ii) hit on all depth-2 units with a seen body type >= 0.70 (20 units, 4 tasks);
- (iii) depth-0 hit >= GroundingDINO alone (0.786).

**Verdict.** B0 PASSES iff at least one primary config (phrase variant; detect for the PaliGemma family) passes all three.
- Secondary configs (segment, context) and the supplementary arm are scored against the same bars and reported.
- Every arm is reported per depth x {seen, held-out} x role, and the unseen-caddy units are reported separately.

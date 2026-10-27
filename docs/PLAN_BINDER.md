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

## 6. B0 result (read 2026-10-27; ledger b0_vlm tag read; results/bind/b0.json; commit 800ab33d): FAIL

No primary config passes all three bars. Hit, two-level task-cluster bootstrap, point estimates (CIs in b0.json):

| arm / config | (i) held-out d1 (15 units, 3 tasks) | (ii) d2 seen-body (20, 4) | (iii) d0 (140, 23) vs GroundingDINO 0.786 | caddy (45, 9) | parse | pass |
|---|---|---|---|---|---|---|
| pi05_base detect (primary) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| Qwen2.5-VL-7B bbox (primary) | 0.00 | 0.40 | 0.879 | 0.00 | 1.00 | no |
| Molmo-7B-D point (primary) | 0.00 | 0.00 | 0.779 | 0.067 | 0.91 | no |
| Qwen bbox + instruction (secondary) | 0.00 | 0.65 | 0.836 | 0.022 | 1.00 | no |
| Molmo point + instruction (secondary) | 0.00 | 0.60 | 0.886 | 0.111 | 0.97 | no |
| pi05_base segment / context variants (secondary) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| pi05_l90c@5000 detect (supplementary) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| S1c (reference) | 0.40 | 0.65 | 0.879 | 0.05 | - | no |

- **The policy's own backbone emits no grounding output.** pi05_base answers `detect` / `segment` with fluent English captions
  and never a `<loc>` / `<seg>` token; pi05_l90c@5000 emits byte noise. The crutch framing (the pretrained backbone still binds)
  cannot be tested with the weights on disk: the PaliGemma pt-224 / mix checkpoints that do emit `<loc>` are gated and absent.
- **General VLMs bind types, not instances or sub-regions.** Qwen / Molmo match or beat GroundingDINO at depth 0 (0.88 / 0.78;
  held-out depth 0 0.90 for both), and seen depth 2 reaches 0.53 (Qwen, phrase) vs S1c's 0.87.

### 6.1 Left / right convention (supplementary, not pre-registered; experiments/b0_lr_check.py, b0_mirror.py)

Every VLM scored 0.00 on held-out depth 1. The check reads S1's rendered frames (the policy orientation, agentview[::-1, ::-1])
and LIBERO's BDDL files.

- **World to image.** Over 3782 body observations, the token-centroid column falls with world y (corr -0.99) and the row rises
  with world x (corr 0.74). The robot sits at -x facing +x, so the robot's left (+y) is on the IMAGE LEFT and +x (away from the
  robot) is nearer the image bottom.
- **LIBERO's words.** The BDDL files place the "left" instance at -y and "front" at +x (e.g. LIVING_ROOM_SCENE4: left bowl
  y in [-0.175, -0.125], right bowl y in [0.025, 0.075]; KITCHEN_SCENE2: front bowl x in [0.075, 0.125]). In every training frame
  with a lateral word the named instance is the IMAGE-OPPOSITE one ("left bowl" 16 / 16, "left plate" 32 / 32, "book on the
  right" 24 / 24, "right bowl" / "right plate" all) and the dev units agree (:60, :38). "front / back" agree with the image
  (front = lower in the image) in 96 / 96 training frames.
- **Reading.** LIBERO's left / right is neither the robot's frame (robot left = +y) nor the image the policy sees. It is the frame of
  a viewer at the agentview camera (at +x, facing the robot), and the policy's [::-1, ::-1] image is that view MIRRORED
  left-right (our inference: the MuJoCo render is flipped only vertically, so a 180-degree rotation adds a horizontal mirror). The caddy compartments follow the same rule: "left compartment" is the
  image-right one.
- **Where the VLMs looked.** On :60 ("black bowl on the left") and :38 ("right moka pot") 38 of the 40 Qwen / Molmo peaks (4 configs x 10 frames) lie on the
  instance that is on the named side OF THE IMAGE, the other instance; the remaining two (Qwen, :60 frame 4) fall on bare table. On :16's goal ("the black bowl in the
  middle") every peak lies on the laterally central bowl; LIBERO's "middle" is the middle in depth (x). Mirroring does not change
  that unit.
- **Mirrored rescore** (committed predictions, same hit and bootstrap; the target of a left / right instance phrase becomes the
  instance at that image side; nothing else changes; results/bind/b0_mirror.json):

| config | held-out d1, pre-registered | held-out d1, mirrored | left / right units only (10, 2 tasks) | caddy left / right (25, 5; swap) |
|---|---|---|---|---|
| Qwen bbox phrase | 0.00 | 0.60 | 0.90 | 1.00 |
| Qwen bbox + instruction | 0.00 | 0.60 | 0.90 | 0.80 |
| Molmo point phrase | 0.00 | 0.667 | 1.00 | 0.64 |
| Molmo point + instruction | 0.00 | 0.667 | 1.00 | 0.44 |

- Under the mirrored reading bar (i) still fails for every config (0.60-0.667 < 0.70), because :16's "middle" stays 0. The
  pre-registered verdict (FAIL) stands and is not replaced.
- **What this changes.** The VLMs' held-out depth-1 zero is a frame-convention mismatch, not a grounding failure: they read
  left / right correctly in the image they are shown. "middle" is a genuine ambiguity (lateral vs depth). Any binder that is not
  fitted on LIBERO's own language (zero-shot pointers, RoboPoint, GroundingDINO) is scored against a mirrored convention on lateral
  words; binders fitted on the training cells (B1b, S1b, S1c, B1) learn the convention from data. The S1 / S3b numbers for
  zero-shot pointers on lateral words carry the same caveat.

**Cost.** B0 used 0.99 A40-h (sacct: b0smoke 4266776 1 m 35 s, 4266869 3 m 18 s; b0dev 4266999 15 m 57 s, 4267000 38 m 33 s; plus a
CPU setup job). The convention check and rescore were CPU (under 1 min).

## 7. B1 pre-registration (2026-10-27; ledger m3_prereg tag pb1; written after smokes on TRAINING units only, before any dev unit is predicted)

The unsupervised prototype binder of section 3.1, with the novelty review's correction: factorising the scorer does not make the
errors independent (the experts share the region proposals and the parent), so region proposal, role assignment and parent
resolution are atoms of the T2 budget, and atom invariance is TESTED.

**Code.** experiments/b1_acts.py, b1_render.py, b1_demos.py (referents), b1_feats.py (features), b1_bind.py (binder),
b1_read.py (read). No simulator label is read before the read; sim masks are used only for scoring and diagnostics.

**Referents (no annotation).**
- 59 training cells with a matched demo (61 minus libero_10:1 / :7, "both X and Y", as S1); every recorded demo of each cell
  (LeRobot ports on disk, offline): 2650 demos, of which 2164 have a contact event for every step -> 4195 referents.
- Grasp / release = a closed gripper-command run of >= 10 frames and its end; a release within 5 cm of its grasp is a regrasp and
  is dropped. The object role's referent is the agentview token under the gripper at the grasp, the goal role's at the release:
  the proprioceptive eef position, projected into the demo's frame 0 with the task's agentview K / E.
- Check: on the 61 matched demos the dataset-projected event tokens equal the simulator-rendered ones in 48 cells; in the other 13 one
  event differs, mostly by one token (a neighbouring column) or one grid row (16 tokens). Referent purity against sim masks is
  reported by the read as a diagnostic.

**Regions.** PCA (32 dims, whitened; fitted on training frames only) of the patch features; Ward clustering of the 256 tokens under
grid connectivity -> a merge tree; cut at K clusters = bodies, nodes inside a body = parts; candidates = nodes of 2-128 tokens.
Appearance channel = node mean feature; geometry channel = centroid / extent relative to the parent (the node's body, or B for
"A of B" under head composition) + position within the phrase's instance set (non-overlapping candidates within DELTA nats).

**Atoms.** A diagonal-Gaussian prototype per word in both channels (hard EM over which node at the referent token is the referent,
3 rounds; B words fitted on the resolved B). MDL (as 3.1): the word goes to the channel whose prototype saves the most nats on its
referents' features against the candidate background, minus d log n; none positive = function word.

**Binding.** Product of experts: the sum over the phrase's content words of the log-likelihood ratio (prototype vs candidate
background) in the word's channel; "A of B" under head composition resolves B first. Peak = the chosen node's member token nearest
its centroid. Unknown words carry no evidence; no evidence = no prediction (a miss).

**Selection (training demos only).** K in {8, 16, 32} x composition {head, flat} x variance floor {0, 0.05, 0.2} (x the candidate
background variance), DELTA = 4: the highest held-out-cell referent log-likelihood (a fixed hash holds out a third of the cells).
Chosen: siglip K 8 flat floor 0.2; dino K 16 flat floor 0.2. Head composition lost to flat at every (K, floor) for both arms.
The chosen held-out-cell referent log-likelihood (-17.0 siglip, -12.6 dino nats) is BELOW the uniform code over about 250
candidates (about -5.5): the prototypes are overconfident on cells they were not fitted on.

**Smokes and design changes before registration (all on TRAINING units; disclosed).**
- One demo per cell (107 referents) gave training-unit hit 0.53 object / 0.20 goal; all recorded demos were used instead.
- Five scoring / MDL variants were smoked on 800 sampled S1 training units (siglip; object / goal hit, demo phrases + other phrases):
  - A, word evidence against the candidate background, no role expert, MDL on the referent's features: 0.711 / 0.499 (CHOSEN, the
    3.1 design as written);
  - B, against the role's referent code + a role expert: 0.591 / 0.510;
  - C, B with MDL on the referent's identity among the frame's candidates: 0.477 / 0.507;
  - D, A with identity MDL: about 0.1;
  - conditional MDL sweeps (a word coded given its phrase's other words; product or best-code context): 0.56 / 0.44-0.52.
  - The freeze rule was the mean training-unit hit. Variant A puts EVERY word in the appearance channel (function words
    included). This is recorded as found: on LIBERO every word's referents come from few scenes, so appearance always codes them
    more cheaply than geometry. Geometry words ("left", "top", "middle") therefore have no geometry expert, and the read is
    expected to fail on held-out depth 1.
- Role assignment has no expert in variant A; its error is still measured as an atom (below).

**Units, metric, bars, verdict** (S1c's / B0's, unchanged). S1's 240 dev units, s1_read.hit, task-cluster bootstrap.
(i) held-out depth-1 hit >= 0.70 (15 units, 3 tasks); (ii) depth-2 seen-body hit >= 0.70 (20, 4); (iii) depth-0 hit >= GroundingDINO
0.786 (140, 23). B1 PASSES iff the PRIMARY arm (siglip: pi05_l90c's own image tokens) passes all three; dino (DINOv2-small) is the
comparison arm, scored against the same bars.

**Reported with the verdict.**
- B1 vs B1b (S1), S1c, GroundingDINO, Qwen2.5-VL, Molmo per depth x split x role; the caddy units.
- Part-proposal recall vs sim masks (best candidate IoU >= 0.5; any candidate whose peak hits) per target kind.
- The MDL channel and savings of every word; confounds (content words whose training role phrases coincide, and pairs with
  Jaccard >= 0.8).
- Error vs compound divergence (split.json's per-task CD; the 11 recombination dev tasks): OLS slope with a 2000-draw
  task-bootstrap CI for B1 and B1b. T1/T2 prediction: B1's slope CI includes 0, B1b's is positive.
- T2 budget per depth: atom errors and the union bound vs the observed error. Atoms (scoring only):
  - region proposal: no candidate's peak hits the target;
  - role assignment: the peak lies on the other role's target;
  - parent resolution (head composition only): the resolved B is off the target's reference body;
  - each content word: an appearance word errs if its expert's best node is off the target's type; a geometry word errs if,
    among nodes on the target's type, its best node misses the target.
- Atom invariance: per atom, its error in each co-occurring context (the role phrase minus the atom); permutation test (1000
  draws) of the between-context variance. Invariance predicts no context effect.
- Power: (i) 3 tasks, (ii) 4 tasks; the frames within a task are near-duplicates. The invariance and CD tests have few tasks and
  are descriptive.

**Size.** Lexicon: 55 content words x (32-dim mean + variance) = 3520 floats = 14.1 KB fp32 (the stored count, 3664, also holds the
two unused role codes); the PCA basis is 67584 floats (2048 x 32 + 2048) = 270 KB fp32 for siglip, 12672 for dino. No trained
network.

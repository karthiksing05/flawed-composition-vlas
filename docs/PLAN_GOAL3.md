# Plan: atlas + failure predictor + architectural repair, across benchmarks and models

Set 2026-11-25 by the user's goal: establish a failure atlas, design a failure predictor
for provided tasks, and repair VLAs with an architectural adaptation, across many
benchmarks and many VLAs, ending in a well-patched paper (ICML 2027, Jan 22).

## Three pillars

**A. Atlas (exists on LIBERO).** Extend the frozen-model protocol to three benchmarks
(candidates: MESA/BiMESA Composite, RoboCasa365, RoboTwin 2.0; VLABench backup), chosen by
docs/notes/bench_feasibility.md. Per benchmark: a held-out-pairing set (known object +
known target never together in the benchmark's training data), known-partner references,
excess failure E, failure classes by the E1 classifier (G grasp, W wrong target, O
overshoot/miss, H hold, M no-move...), one cue test per class.

**B. Failure predictor (new).** Input: a task specification and the benchmark's training
demo index only, no rollouts. Output: P(fail) and the predicted class for each
(task, model) cell.
- Features, all computable before any rollout:
  1. pairing novelty: is (object, target) in training; partner counts of object and target;
  2. trained-cue conflict: does the object's trained partner sit in the scene as a distractor;
  3. geometric cue conflict: is a non-instructed receptacle nearer the object/gripper, or more
     salient (largest projected area in the policy's main camera);
  4. language form: relational / negated / ordinal reference (F7); target occluded after first
     view (F6);
  5. model-agnostic base rate: the model's success on the known partners (the E reference).
- Predictor v0 = the registered cue-vote rule (place goes to the receptacle backed by most of
  {nearer, salient, trained}); predicted failure when that receptacle is not the instructed one.
- Predictor v1 = logistic regression on features 1-5, fit on LIBERO atlas episodes only
  (both scenes, all five VLAs), then FROZEN with its hash recorded.
- Baselines: (b1) novelty only; (b2) known-partner success only; (b3) a VLM asked to predict
  (cheap prompt, frozen).
- Confirmatory test: pre-registered AUC of v0 and v1 on each new benchmark, per model, against
  b1/b2; the claim holds if v1 beats b2 by a registered margin on at least 2 of 3 benchmarks.
  Class prediction scored by top-1 agreement with the E1 classifier.

**C. Architectural repair (exists on LIBERO: W5 where-module, H* head, servo).** Port the
explicit where-component (object/target positions -> separate module that owns the place
target, fused with the frozen or lightly fine-tuned VLA) to each new benchmark, on at least
two models per benchmark (one pi-family, one non-pi: GR00T or X-VLA or SmolVLA). Compare
against the matched in-network route (same data, target as a learned input) and the
plain fine-tune. Privileged sim poses are allowed but labelled; a perceived-pose variant
where a cheap detector exists.

## Order
1. Feasibility (running) -> pick 3 benchmarks, checkpoint inventory.
2. Predictor dev on LIBERO from existing atlas results (CPU) -> freeze v0/v1.
3. Pre-register per-benchmark held-out pairings + predictor predictions (ledger/m3_prereg).
4. Frozen atlas rollouts on the new benchmarks with released checkpoints (many models);
   own fine-tunes with excluded pairings only where no released checkpoint has a held-out
   pairing by construction.
5. Repair runs on one benchmark first, then the others.
6. Fold into paper: atlas across benchmarks, predictor section, repair matrix across
   benchmarks; related work vs MESA (2611.05567) and Robust Skills, Brittle Grounding
   (2602.24143); adversarial review; rebuild; push.

## Rules carried over
Pre-register before reads; never overwrite results; Holm within families; hierarchical
bootstrap (seeds then states, tasks fixed); compute via sbatch on overcap/tail-lab only.

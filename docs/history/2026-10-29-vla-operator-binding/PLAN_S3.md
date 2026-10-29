# PLAN_SOLVE S3: confidence gate (S3a) and an off-the-shelf relational grounder (S3b) (S3a pre-registered 2026-10-26)

Diagnosis from S2: the adapter works with oracle masks. The deployable S1b masks are often wrong or one token (object mask mostly
inside the target in 0.58 of dev units), and wrong masks damage training cells (-0.18).

EXPLORATORY. There is no confirmatory read and no sealed set. Ledger: m3_prereg tags `ps3a` and `ps3b`; read rows `s3a_gate` and
`s3b_grounder`.

## S3a. Confidence gate

**Features.** experiments/s3_gate.py defines them, and s2_serve.Binder reports them per role (additive change; the masks are
unchanged):
- the head's candidate margin, top-1 minus top-2 softmax probability;
- the top-1 probability;
- the GroundingDINO score of the chosen box;
- the mask mass;
- the maximum crop sigmoid;
- the candidate count.

**Score family** (pre-declared):
- `margin`: min over roles of the margin;
- `det`: min over roles of the detector score;
- `joint`: min over roles of margin x det, with a one-token mask scoring 0.

**Gate.** The gate is unit-level. Both role masks are kept, or both are zeroed, and zero masks are exactly zs (S0's parity check
was 24/24).

**Data (one GPU pass, no rollout).** s2_sandbox `--conf-only` sends the first call's frame of every dev and train unit to the S1b
and S1c servers. It stores the features and each mask's quality against the live oracle masks. The frame is the same for every
seed, so these rows reproduce S2's first-call masks; the read checks this.

**Selection (training cells only).** experiments/s3a_read.py does this.
1. For each score and each threshold, the gated train success is computed per unit: S2's deploy runs where the gate is on, S0's
   zs runs where it is off (seeds 0-3).
2. Admissible gates have damage >= -0.02 (point estimate).
3. The chosen gate has the largest train coverage. Ties go to the order margin, det, joint, then the larger threshold.
4. If only the always-off gate is admissible, there is no gate and S3a stops.

**Offline dev effect** (the chosen gate, decided per unit at the first frame):
- coverage;
- mask quality of gated-on vs gated-off units;
- dev rates recombined from existing paired runs (on: S2 deploy, off: S0 zs; seeds 0-3), diffs vs zs, closure vs oracle.

**Closed-loop check.** It runs only if dev coverage >= 0.10 AND the gated-on dev object masks lie mostly inside the target in
>= 0.80 of units.
- Adapter + S1b with the chosen gate applied per call.
- 120 dev + 50 train units at **seeds 4-5**, compared with S0's zs and oracle runs at seeds 4-5.
- Deviation from the coordinator's seeds 0-1, disclosed: seeds 0-3 chose the threshold, so seeds 4-5 are held out from the
  selection. S0's runs at seeds 4-5 give paired comparators.

**Bars** (point estimates; PASS iff both):
- **B1**: training-cell damage, success(gated) - success(zs), >= -0.05.
- **B2**: dev binding error(gated) <= zs.

The bars are read on the closed loop if it runs. Otherwise they are read on the offline recombination, where B1 is in-sample and
flagged as such.

S1c: the read reports first-frame mask quality only, since no S1c rollouts exist.

## S3a result (read 2026-10-26; ledger s3a_gate tag read; results/bind/s3a.json): the bar passes VACUOUSLY; no useful gate exists

The confidence pass reproduces S2's first-call masks on 120 of 120 dev units.

**Selection** (training cells, 50 units):
- Only 16 of 133 (score, threshold) pairs keep damage >= -0.02.
- The chosen gate is `det` >= 0.556 (both roles' detector scores). It turns on for 12% of train units, with damage -0.010
  [-0.030, 0.000].
- For every score, damage grows steadily with coverage.

| train coverage | margin | det | joint |
|---|---|---|---|
| about 0.24 | -0.065 | -0.045 | -0.045 |
| about 0.48 | -0.075 | -0.095 | -0.100 |
| 1.0 (= S2) | -0.18 | -0.18 | -0.18 |

None of the three scores separates harmful masks from harmless ones.

**Offline dev effect.**
- The gate turns on for 5 of 120 dev units, all in 1 task (coverage 0.04).
- On those 5 units the object masks are good (all mostly inside the target, IoU 0.64), but the goal masks are entirely outside
  the target.
- Recombined dev = zs exactly: success 0.277, binding error 0.229; closure 0.

**Closed loop**: not run. The pre-declared condition failed (dev coverage 0.04 < 0.10).

**Bars**, read offline:
- B1 damage -0.010 (in-sample): pass.
- B2 binding error = zs: pass.
- These are VACUOUS. The gate is almost always off, so the arm is zs.

**S1c first-frame masks** (dev; quality only):

| role | mostly inside the target | one token |
|---|---|---|
| object | 0.74 (S1b 0.575) | 0.24 |
| goal | 0.56 (S1b 0.62) | 0.37 |

S1c has better object masks than S1b but more degenerate masks. It does not reach the gated-on quality the closed loop required.

**Reading**: these binder confidences (candidate margin, detector score, and their product) are not calibrated to mask
correctness on these scenes. A gate built from them can only buy safety by switching the adapter off.

## S3b. Off-the-shelf relational grounder (offline mask quality; pre-registered as ps3b before any dev prediction)

The probe found every candidate obtainable (RoboPoint-13B, RoboRefer-2B / 8B). RoboPoint is used because it has the simpler
stack, a LLaVA fork.

- **Model**: wentao-yuan/robopoint-v1-vicuna-v1.5-13b, code github wentaoyuan/RoboPoint @ 206cb1e.
- **Isolation**: code, venv and weights live in /coc/flash10/ksingara3/s3b_ext (not the repo). The script runs with `python -I`.
- **Query**: object: "Locate several points on the {phrase}."; goal: "Locate several points {phrase}." (with "on the " prepended when
  the phrase has no preposition). RoboPoint's own answer-format suffix follows. Decoding is greedy.
- **Map**: points go onto S1's 16 x 16 tokens. The peak is the token with the most points; the mask is every token with a point;
  no point is a miss.
- **Units**: S1's 240 dev units, the frames S1, S1b and S1c scored. The read (experiments/s3b_read.py) uses S1's hit metric
  and reports RoboPoint vs S1b, S1c and B1a by depth x seen / held-out / held-out body-seen.
- **No pass bar**: this is a comparison only.
- **Smoke**: 4 TRAINING units only, so no dev unit is seen before ps3b.

## S3b result (read 2026-10-26; ledger s3b_grounder tag read; results/bind/s3b.json): RoboPoint is worse than our binders

S1's hit metric on the same 240 dev units. RoboPoint's hit carries its task-bootstrap CI.

| cell | n (tasks) | RoboPoint-13B | S1b | S1c | GroundingDINO (B1a) |
|---|---|---|---|---|---|
| depth 0, all | 140 (23) | 0.743 [0.619, 0.872] | 0.821 | 0.879 | 0.786 |
| depth 0, held-out | 50 (9) | 0.74 | 0.82 | 0.90 | 0.90 |
| depth 1, all | 40 (7) | 0.15 [0.00, 0.35] | 0.45 | 0.575 | 0.125 |
| depth 1, held-out | 15 (3) | 0.00 | 0.27 | 0.40 | 0.00 |
| depth 2, seen | 15 (3) | 0.00 | 0.67 | 0.87 | 0.00 |
| depth 2, held-out | 45 (9) | 0.18 [0.00, 0.42] | 0.18 | 0.04 | 0.11 |
| depth 2, held-out, unseen body (caddy) | 40 (8) | 0.20 | 0.125 | 0.05 | 0.125 |

RoboPoint answered every unit with at least one point. Its point masks have IoU 0.26 at depth 0.

Reading:
- RoboPoint, an off-the-shelf relational pointer trained for spatial references, is below S1b, S1c and GroundingDINO at depth 0.
- At depth 1 it fails on held-out cells, and at depth 2 it fails on seen ones.
- Its only edge is the unseen caddy body (0.20).
- It does not supply the instance and sub-region binding the adapter needs on these LIBERO frames.

Disclosures:
- Loading the checkpoint warned that its vision-tower weights were unused. This is LLaVA's builder, which loads CLIP-L/336
  separately.
- The smoke used 4 TRAINING units.
- The code, venv (5 GB) and weights (25 GB) are in /coc/flash10/ksingara3/s3b_ext.
- The alternative model, RoboRefer, was not tried.

## S3 cost

- S3a confidence pass: 4 jobs, about 0.26 A40-h.
- S3b RoboPoint: smoke plus dev, about 0.72 A40-h.
- Setup, probe and reads ran on CPU.
- About 1.0 A40-h in total. About 4.7 A40-h of the approved 20 remain.

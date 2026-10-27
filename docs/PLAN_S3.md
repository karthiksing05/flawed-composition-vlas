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

## S3b. Off-the-shelf relational grounder (offline mask quality)

To be pre-registered as ps3b before any score is computed. RoboPoint or RoboRefer, point to mask, scored with S1's hit metric by
depth and seen / held-out. The setup cap is about 1 A40-h; if neither model is obtainable, S3b is reported and skipped.

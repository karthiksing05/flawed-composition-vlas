"""PRE-REGISTRATION ROW of PU (experiments/sc_pu.py): U from perceived poses (masks + depth + calibration) vs the oracle arm, on
sc_t4's U combos (seeds 10-12 x K1-K6, N1-N6 x familiar, new; N2 main states 4-23).  BEFORE any non-smoke sc_pu rollout.
The placeholders MAPPING, PREDICTION and SEEN must be replaced by the orchestrator; the script refuses to run while any remains.

    python experiments/prereg/sc_pu.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

MAPPING = ("experiments/sc_pu_read.py. Four cells: known (K1-K6) / new pairings (N1-N6) x familiar / new layout; seeds 10-12; N2 main "
           "states. Per cell, percep - oracle success, bootstrapped over tasks and states jointly (2,000 draws, rng 0). Sanity first: "
           "oracle episodes must agree with sc_t4's U rollouts on >= .90 of episodes overall, else INVALID. Then COSTS if any cell's CI "
           "upper bound < -.10; HOLDS if every cell's CI lower bound >= -.10; UNRESOLVED otherwise. Per-task rows are descriptive only.")
PREDICTION = ("HOLDS. Point differences: known|familiar -.03, new_pairings|familiar -.05, known|new -.03, new_pairings|new -.03. The "
              "largest per-task costs on the wine-bottle tasks (N4, N5), where the check's held-bottle error is largest (p90 2.7 cm, "
              "rigid). Second most likely: UNRESOLVED from a bottle cell or from the new layout, whose objects sit outside the "
              "calibration demos' positions. Oracle-vs-T4 agreement >= .98.")
SEEN = ("Before this row: calib on demos 0-24 of K1-K6 (results/flawed/sc_pu_calib.json) and check on demos 25-49 (sc_pu_check.json; "
        "position errors median .03-.49 cm, p90 up to 2.1 cm during the hold, bottle post-release p90 6.9 cm). Smoke: seed 10, smoke "
        "states 0-1, K1 familiar and N4 familiar, arms ref / oracle / percep: all 12 episodes succeeded; oracle equals ref step for step; "
        "percep took 66-104 steps. One smoke recording (N4 familiar, smoke state 0). No main-state sc_pu rollout. The orchestrator has "
        "read sc_t4's U results on these states (SCAFFOLD_RESULTS 11: new pairings familiar .894, new layout .381) and nothing else of PU.")

for name, v in (("MAPPING", MAPPING), ("PREDICTION", PREDICTION), ("SEEN", SEEN)):
    assert v != name, "placeholder %s not filled in" % name

rows = ledger.load("m3_prereg")
od = os.path.join(ROOT, "results", "flawed", "sc_pu")
assert not os.path.isdir(od) or not os.listdir(od), "non-smoke sc_pu results exist"
assert not any(r.get("tag") == "sc_pu" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_pu", seed=0, command="pre-registration: PU (experiments/sc_pu.py)",
                    metrics={"status": "PRE-REGISTERED PU",
                             "mapping": MAPPING,
                             "prediction": PREDICTION,
                             "seen": SEEN,
                             "arms": "oracle (sc_pu's path, simulator poses; the sanity arm) and percep (movable objects from agentview masks + depth, "
                                     "fallbacks wrist / last / prior; sites from fixture masks; drawer simulator-read, unused in K1-K6/N1-N6)",
                             "states": "N2 main states 4-23 (n2_branches_main.json); smoke states 0-1 read before this row",
                             "calib": "demos 0-24 of K1-K6 (results/flawed/sc_pu_calib.json); check demos 25-49 (results/flawed/sc_pu_check.json)",
                             "files_sha256": {p: fsha(p) for p in ["experiments/sc_pu.py", "experiments/sc_pu_read.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py",
                                                                   "experiments/sc_rv.py", "results/flawed/sc_pu_calib.json",
                                                                   "results/flawed/sc_pu_check.json"]},
                             "written_before": "any non-smoke sc_pu rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_pu.py"))

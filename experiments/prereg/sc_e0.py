"""PRE-REGISTRATION ROW of E0, the partner-count read (docs/THEORY_COMPOSITION.md 2).  BEFORE the read.

    python experiments/prereg/sc_e0.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_e0_read.json")), "read exists"
assert not any(r.get("tag") == "sc_e0" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_e0", seed=0, command="pre-registration: E0 partner-count read (docs/THEORY_COMPOSITION.md 2)",
                    metrics={"status": "PRE-REGISTERED E0",
                             "labelled": "development read on N2 main states 4-23, existing rollouts, familiar layout; frozen GR00T (n2_roll/groot_main, sc_t1/groot_r2, groot_r10) and pi0.5 at 2 and 10 actions per query (sc_t1/pi05_r2, pi05_r10); one seed per arm",
                             "mapping": "Groups by task object in n2_tasks.json: cream cheese (N1-N3) vs wine bottle (N4-N6). Per arm, gap = mean success(bottle) - mean success(cheese); CI by bootstrap over states (2000 draws, seed 0), states resampled jointly across tasks, tasks fixed. HOLDS iff CI lower > 0; REVERSED iff CI upper < 0; else UNCLEAR. Overall: SUPPORTED iff GR00T's pooled arm (the three GR00T files averaged) HOLDS and no registered arm is REVERSED; REFUTED iff GR00T's pooled arm is REVERSED; else UNRESOLVED",
                             "prediction": "SUPPORTED: every arm HOLDS; GR00T's gap smaller in absolute size than pi0.5's because its level is near the floor",
                             "seen_before_registration": "pi0.5 intact (n2_roll/pi05_main) per-task values were seen before registration (RIVAL_TESTS 3.1) and are descriptive only; GR00T's and the r2/r10 per-task values were not seen",
                             "also_reported_no_verdict": "per-task success, grasped-right and place-given-grasp per task and arm, both layouts, N3 minus mean(N1,N2) per arm",
                             "states": "dev only: N2 main states 4-23",
                             "files_sha256": {p: fsha(p) for p in ["docs/THEORY_COMPOSITION.md", "docs/SCAFFOLD_RESULTS.md", "experiments/sc_e0.py", "experiments/sc_t1.py"]},
                             "written_before": "any E0 read"}, bytes=Z, config={}, script="experiments/prereg/sc_e0.py"))

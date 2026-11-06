"""PRE-REGISTRATION ROW of T6 (docs/SCAFFOLD_RESULTS.md 17).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t6.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t6_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t6" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t6", seed=0, command="pre-registration: T6 (docs/SCAFFOLD_RESULTS.md 17)",
                    metrics={"status": "PRE-REGISTERED T6",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "familiar: chain success minus the product of the two skills' U outcomes on matching seed and state: BREAKS iff the CI lies wholly below -.10; HOLDS iff the lower bound >= -.10; else UNRESOLVED",
                             "prediction": "UNRESOLVED; first skill holds, second loses something to moved objects",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 on C1 and C5 read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t6.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any non-smoke T6 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t6.py"))

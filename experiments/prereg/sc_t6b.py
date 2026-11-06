"""PRE-REGISTRATION ROW of T6b (docs/SCAFFOLD_RESULTS.md 22).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t6b.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t6b_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t6b" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t6b", seed=0, command="pre-registration: T6b (docs/SCAFFOLD_RESULTS.md 22)",
                    metrics={"status": "PRE-REGISTERED T6b",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work and T6's post hoc breakdown; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "each arm R and D, familiar: chain success minus the product of the two skills' U outcomes on matching seed and state: BREAKS iff the CI lies wholly below -.10; HOLDS iff the lower bound >= -.10; else UNRESOLVED. T6's verdict stays on the record",
                             "prediction": "R HOLDS, point estimate near -.05; D BREAKS",
                             "states": "dev only: N2 main states 4-23; smoke states 0-2 of chains, seed 10, familiar, shaped the elbow rule and were read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t6b.py", "experiments/sc_t6.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/scaffold/stores.py"]},
                             "written_before": "any non-smoke T6b rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t6b.py"))

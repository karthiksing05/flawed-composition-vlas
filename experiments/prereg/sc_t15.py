"""PRE-REGISTRATION ROW of T15 (docs/SCAFFOLD_RESULTS.md 28).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t15.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t15_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t15" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t15", seed=0, command="pre-registration: T15 (docs/SCAFFOLD_RESULTS.md 28)",
                    metrics={"status": "PRE-REGISTERED T15",
                             "labelled": "development read on N2 main states 4-23, the same chains and states T14 just read, so this repeats a read after a fix designed on that failure; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "per arm, six chains M1-M6, familiar: chain minus product. BREAKS iff CI upper < -.10; HOLDS iff CI lower >= -.10; else UNRESOLVED. R carries the claim, G is the ceiling",
                             "prediction": "R UNRESOLVED, chains near .65, product .82, gain near +.25 over T14; G within .05 of R; D1, D2 near .65; M3 stays low",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row: R and G each M1 1/2, D1 1/2 on seed 10",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t15.py", "experiments/sc_t14.py", "lifelong_ood/policy/parts_flow.py", "results/flawed/sc_t12_modes.json"]},
                             "written_before": "any non-smoke T15 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t15.py"))

"""PRE-REGISTRATION ROW of T2 (docs/SCAFFOLD_RESULTS.md 19).  BEFORE the non-smoke read.

    python experiments/prereg/sc_t2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t2_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t2" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t2", seed=0, command="pre-registration: T2 (docs/SCAFFOLD_RESULTS.md 19)",
                    metrics={"status": "PRE-REGISTERED T2",
                             "labelled": "development read on N2 states 4-23, which have shaped the locator; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "median over 3 repeats of each head's p95 horizontal error, k = 1, shift-corrected: WITHIN BUDGET iff every head <= .02 m in both layouts; BEATS A TABLE, OVER BUDGET iff every head in the new layout <= half a constant's error; else NO BETTER THAN A TABLE",
                             "prediction": "NO BETTER THAN A TABLE, with medians well below the p95",
                             "states": "dev only; the smoke printed no error",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t2.py", "experiments/pp_loc.py"]},
                             "written_before": "the non-smoke T2 read"}, bytes=Z, config={}, script="experiments/prereg/sc_t2.py"))

"""PRE-REGISTRATION ROW of T14 (docs/SCAFFOLD_RESULTS.md 27).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t14.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t14_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t14" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t14", seed=0, command="pre-registration: T14 (docs/SCAFFOLD_RESULTS.md 27)",
                    metrics={"status": "PRE-REGISTERED T14",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the open clause's end rule is new; the user asked to keep iterating overnight and has not reviewed these rules",
                             "mapping": "six chains M1-M6, familiar: chain minus product of each skill alone. BREAKS iff CI upper < -.10; HOLDS iff CI lower >= -.10; else UNRESOLVED. D1, D2 descriptive",
                             "prediction": "UNRESOLVED; chains near .55, product near .75; D1, D2 near .40",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row: M1 1/2, M3 1/2, D1 1/2 on seed 10",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t14.py", "lifelong_ood/policy/parts_flow.py", "results/flawed/sc_t12_modes.json"]},
                             "written_before": "any non-smoke T14 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t14.py"))

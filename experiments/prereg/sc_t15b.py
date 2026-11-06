"""PRE-REGISTRATION ROW of T15b (docs/SCAFFOLD_RESULTS.md 29).  BEFORE any rollout.

    python experiments/prereg/sc_t15b.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t15b_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t15b" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t15b", seed=0, command="pre-registration: T15b (docs/SCAFFOLD_RESULTS.md 29)",
                    metrics={"status": "PRE-REGISTERED T15b",
                             "labelled": "development read on N2 main states 4-23, the third read of these chains and states, designed on the last failure; closes the block on these chains whatever the label; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "six chains M1-M6, familiar: chain minus product. BREAKS iff CI upper < -.10; HOLDS iff CI lower >= -.10; else UNRESOLVED",
                             "prediction": "UNRESOLVED, chains near .60; first goal at transition on M1, M2, D1, D2 rises from about .54 to about .75; M3 near 0",
                             "states": "dev only: N2 main states 4-23; no smoke",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t15b.py", "experiments/sc_t15.py", "experiments/sc_t14.py", "results/flawed/sc_t12_modes.json"]},
                             "written_before": "any T15b rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t15b.py"))

"""PRE-REGISTRATION ROW of T5b (docs/SCAFFOLD_RESULTS.md 14).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t5b.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t5b_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t5b" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t5b", seed=0, command="pre-registration: T5b (docs/SCAFFOLD_RESULTS.md 14)",
                    metrics={"status": "PRE-REGISTERED T5b",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the user asked for iteration and has not reviewed this rule",
                             "mapping": "Ur minus U, first match: HARMS iff either familiar CI lies wholly below -.10; PAYS iff both familiar lower bounds >= -.10 and in either new-layout cell the difference >= +.10 with CI lower > 0; else NOT YET",
                             "prediction": "NOT YET, positive on bowl and bottle",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 on four combos read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t5b.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any non-smoke T5b rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t5b.py"))

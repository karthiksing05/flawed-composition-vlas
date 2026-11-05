"""PRE-REGISTRATION ROW of T3b (docs/SCAFFOLD_RESULTS.md 5).  BEFORE any T3b run on dev states.

    python experiments/prereg/sc_t3b.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t3b_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t3b" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t3b", seed=0, command="pre-registration: T3b, grasp owner x carry owner (docs/SCAFFOLD_RESULTS.md 5)",
                    metrics={"status": "PRE-REGISTERED T3b",
                             "labelled": "post hoc follow-up to T3's UNRESOLVED read; development read on N2 main states 4-23, which have shaped earlier work; the user has not reviewed these rules",
                             "mapping": "effect iff CI lower bound above 0 and estimate > 0.10; on success given an owned hold, new layout, pooled over 4 object groups; G = SP - PP, C = PS - PP. First match: INSTRUMENT FAILS iff PP disagrees with T3's A on more than 5% of episodes; BOTH iff both effects; CARRY OWNS IT iff only C; GRASP OWNS IT iff only G; else NEITHER",
                             "prediction": "CARRY OWNS IT, C about +.25, G within the margin",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t3b.py", "experiments/sc_t3.py"]},
                             "written_before": "any T3b run outside the smoke"}, bytes=Z, config={}, script="experiments/prereg/sc_t3b.py"))

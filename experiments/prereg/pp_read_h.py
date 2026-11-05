"""PRE-REGISTRATION ROW of read H (docs/HAND_OVER.md 9).  BEFORE the read.

    python experiments/prereg/pp_read_h.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_read_h"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_read_h.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/HAND_OVER.md", "experiments/pp_read_h.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: read H (docs/HAND_OVER.md 9)",
                    metrics={"status": "PRE-REGISTERED read H",
                             "labelled": "post hoc; development read; no training; no rollouts; changes no earlier verdict; not reviewed by the user, who is away and asked to keep iterating",
                             "mapping": "per layout, separately for targets and objects: A TABLE SUFFICES iff every entity's 95th percentile distance from its mean horizontal first-tracked position is within 0.02 m, otherwise NEEDS PERCEPTION",
                             "prediction": "NEEDS PERCEPTION for objects and targets in both layouts",
                             "states": "dev only: N2 main states 4-23; existing atlas and hand-over tracks",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_read_h.py"))

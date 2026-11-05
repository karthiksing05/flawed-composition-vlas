"""PRE-REGISTRATION ROW of the locator's diagnostic (docs/LOCATOR.md 8).  BEFORE the read, and it runs the read's job only if it succeeds.

    python experiments/prereg/pp_loc_diag.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_loc_diag"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_loc_diag.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/LOCATOR.md", "experiments/pp_loc_diag.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: locator diagnostic (docs/LOCATOR.md 8)",
                    metrics={"status": "PRE-REGISTERED locator diagnostic",
                             "labelled": "post hoc; development read; no training; changes no verdict in LOCATOR 7",
                             "mapping": "per arm and head, new layout: FEW STATES iff every seed has at most 3 of 20 states with horizontal error above 0.05 m after removing the head's mean label shift and its median error is at most 0.02 m; otherwise BROAD",
                             "prediction": "FEW STATES on all of A's heads and on C's heads except the bottle; bad states repeat across seeds and come from occlusion",
                             "states": "dev only: N2 main states 4-23, new layout",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_loc_diag.py"))

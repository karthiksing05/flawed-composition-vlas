"""PRE-REGISTRATION ROW of the V1 arrival read (docs/RIVAL_TESTS.md 4).  After V1's verdict and BEFORE this read.

    python experiments/prereg/pp_rival2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_rival2"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_v1_arrive.json"))
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: V1 arrival read (docs/RIVAL_TESTS.md 4)",
                    metrics={"status": "PRE-REGISTERED V1 arrival read",
                             "labelled": "post hoc; development read of V1's existing episodes; changes no earlier verdict",
                             "mapping": "intact, familiar, pooled N1-N6, g = 1 failures, share within 0.10 m of the true target: CI lower > 0.5 ARRIVES, FAILS TO SEAT; CI upper < 0.5 DOES NOT ARRIVE; else MIXED",
                             "prediction": "MIXED, the bottle's failures arriving and the cheese's not",
                             "files_sha256": {p: fsha(p) for p in ("docs/RIVAL_TESTS.md", "experiments/pp_v1_arrive.py")},
                             "written_before": "this read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_rival2.py"))

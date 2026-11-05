"""PRE-REGISTRATION ROW of the centre labels (docs/LOCATOR.md 14).  BEFORE any non-smoke training.

    python experiments/prereg/pp_loc4.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_loc4"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_loc4.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/LOCATOR.md", "experiments/pp_loc4.py", "experiments/pp_loc3.py", "experiments/pp_loc2.py", "experiments/pp_loc.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0, command="pre-registration: centre labels (docs/LOCATOR.md 14)",
                    metrics={"status": "PRE-REGISTERED centre labels",
                             "labelled": "post hoc; development read; no rollouts; the 20 states shaped this design; changes no earlier verdict; the user asked to keep iterating, details not reviewed",
                             "mapping": "on arm CC, LOCATOR 5's mapping, first match: WITHIN BUDGET iff every head has e <= 0.02 m in both layouts; BEATS A TABLE, OVER BUDGET iff every head has e <= T/2 in the new layout; otherwise NO BETTER THAN A TABLE",
                             "prediction": "BEATS A TABLE, OVER BUDGET; the plate under 0.05 m; raw error below CP's on the plate and the bowl; the familiar bottle near 0.02 m",
                             "states": "dev only: N2 main states 4-23, never in training; training on K1-K6 demos 0-39, familiar layout",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any non-smoke training"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={}, script="experiments/prereg/pp_loc4.py"))

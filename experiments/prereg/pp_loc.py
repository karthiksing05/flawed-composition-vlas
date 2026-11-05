"""PRE-REGISTRATION ROW of the demo-labelled locator (docs/LOCATOR.md).  BEFORE any training or read.

    python experiments/prereg/pp_loc.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_loc"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_loc.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/LOCATOR.md", "experiments/pp_loc.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: demo-labelled locator (docs/LOCATOR.md)",
                    metrics={"status": "PRE-REGISTERED locator read",
                             "labelled": "post hoc; development read; no rollouts; changes no earlier verdict; route chosen by the user, details not reviewed",
                             "mapping": "per arm C and A, e = 95th percentile over 20 states of horizontal error after removing the head's mean label shift, median over 3 seeds; T = the same for the mean training label; first match: WITHIN BUDGET iff every head has e <= 0.02 m in both layouts; BEATS A TABLE, OVER BUDGET iff every head has e <= T/2 in the new layout; otherwise NO BETTER THAN A TABLE",
                             "prediction": "both arms BEATS A TABLE, OVER BUDGET; familiar e near 0.02 m, new e near 0.05 m; grasp displacement under 0.02 m familiar, over it new",
                             "oracle_kept": "simulator camera matrices, in both arms; rest height, sizes, yaw and roles untouched",
                             "states": "dev only: N2 main states 4-23, never in training; training on K1-K6 demos 0-39, familiar layout",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any training or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_loc.py"))

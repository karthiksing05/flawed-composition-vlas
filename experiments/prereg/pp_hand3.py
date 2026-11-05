"""PRE-REGISTRATION ROW of the closed-loop locator (docs/LOCATOR.md 10).  BEFORE any rollout outside the smoke.

    python experiments/prereg/pp_hand3.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_hand3"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_hand3_read.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/LOCATOR.md", "experiments/pp_hand3_client.py", "experiments/pp_hand3_roll.sh", "experiments/pp_hand3_read.py", "experiments/pp_loc.py", "experiments/pp_loc2.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: closed-loop locator (docs/LOCATOR.md 10)",
                    metrics={"status": "PRE-REGISTERED closed-loop locator",
                             "labelled": "post hoc; development read; no training; the 20 states shaped the locator; changes no earlier verdict; the test approved by the user, details not reviewed",
                             "mapping": "per VLA and layout, N2 N3 N5 pooled over locator seeds: RAISES iff s(LS)-s(intact) has CI lower bound above 0 and exceeds 0.10; PERCEPTION HOLDS iff RAISES and s(H1)-s(LS) <= 0.10; PERCEPTION COSTS iff RAISES and s(H1)-s(LS) > 0.10; NO FIX iff not RAISES. KNOWN COST qualifies the verdict iff K4 and K6 lose more than 0.05 against H1",
                             "prediction": "PERCEPTION HOLDS on all four; the bottle in the new layout does worst",
                             "oracle_kept": "object height and rest height, yaw and size, target positions, role assignment, camera matrices",
                             "states": "dev only: N2 main states 4-23",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any rollout outside the smoke, which uses smoke states 0-3"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_hand3.py"))

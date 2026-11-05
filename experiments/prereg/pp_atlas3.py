"""PRE-REGISTRATION ROW of reads C and D on the break atlas (docs/BREAK_ATLAS.md 8).  BEFORE these reads.

    python experiments/prereg/pp_atlas3.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_atlas3"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_atlas_reads2.json"))
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: reads C and D on the break atlas (docs/BREAK_ATLAS.md 8)",
                    metrics={"status": "PRE-REGISTERED reads C and D",
                             "labelled": "post hoc; development read of recorded tracks; changes no registered verdict",
                             "C_mapping": "VLA HANDOFF episodes ending within 0.10 m, familiar, per model; an episode is a missed carry iff over half its object travel after the last held step is lifted >= 0.01 m, gripper commanded closed, within the median held gap + 0.02 m. "
                                          "Share > 2/3 MISSED CARRY; < 1/3 REAL DROP; else MIXED",
                             "D_mapping": "new layout, per object (four tasks each), per state, grasp-failure rate for servo and P_min; L both >= 0.9, C both <= 0.1; OBJECT-STATE LOCKED iff for both objects |L| >= 6 and |L| + |C| >= 16; else NOT LOCKED",
                             "prediction": "C REAL DROP on pi0.5; D OBJECT-STATE LOCKED",
                             "files_sha256": {p: fsha(p) for p in ("docs/BREAK_ATLAS.md", "experiments/pp_atlas.py", "experiments/pp_atlas_reads2.py")},
                             "written_before": "these reads"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_atlas3.py"))

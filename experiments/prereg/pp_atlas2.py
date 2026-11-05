"""PRE-REGISTRATION ROW of the checks on the break atlas (docs/BREAK_ATLAS.md 6).  After the atlas read and BEFORE these reads.

    python experiments/prereg/pp_atlas2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_atlas2"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_atlas_checks.json"))
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: checks on the break atlas (docs/BREAK_ATLAS.md 6)",
                    metrics={"status": "PRE-REGISTERED checks A and B",
                             "labelled": "post hoc; development read of recorded tracks; changes no registered verdict",
                             "A_mapping": "VLA HANDOFF episodes, familiar, per model, share ending within 0.10 m of the target: > 0.5 LABEL FAILS; < 0.25 LABEL HOLDS; else AMBIGUOUS",
                             "B_mapping": "new layout, eight cheese and bottle tasks, per state GRASP rate for servo and P_min; L both >= 0.9, C both <= 0.1; STATE-LOCKED iff |L| >= 8 and |L| + |C| >= 18; else NOT STATE-LOCKED",
                             "prediction": "A LABEL FAILS for pi0.5, none for GR00T; B STATE-LOCKED",
                             "files_sha256": {p: fsha(p) for p in ("docs/BREAK_ATLAS.md", "experiments/pp_atlas.py", "experiments/pp_atlas_checks.py")},
                             "written_before": "these reads"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_atlas2.py"))

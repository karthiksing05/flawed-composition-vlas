"""PRE-REGISTRATION ROW of the break atlas (docs/BREAK_ATLAS.md; PLAN_PARTS 10.19).  After the instruments' code and BEFORE any main run or read.

    python experiments/prereg/pp_atlas.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_atlas"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_atlas", "pp_atlas_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/BREAK_ATLAS.md", "docs/PLAN_PARTS.md", "experiments/pp_atlas.py", "experiments/pp_atlas_client.py", "experiments/pp_atlas_roll.sh", "experiments/pp_l1_floor.py",
         "experiments/libero_client.py", "experiments/fc_n2_client.py", "experiments/pp_serve_purify.py", "experiments/pp_n4d.py", "experiments/fa1h_roll.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: the break atlas (docs/BREAK_ATLAS.md; PLAN_PARTS 10.19)",
                    metrics={"status": "PRE-REGISTERED the break atlas",
                             "labelled": "post hoc; development read; no training; changes no earlier verdict",
                             "stage_rule": "failures, first match: GRASP never held; HANDOFF held, dmin > 0.10, d shrank < half; TRANSPORT held, dmin > 0.10, shrank >= half; "
                                           "RELEASE dmin <= 0.10 and still held at end or last held > 0.03; PHYSICS dmin <= 0.10 and last held <= 0.03",
                             "groups": "within-chunk = GRASP + RELEASE; cross-chunk = HANDOFF + TRANSPORT",
                             "mapping": "per model (pi05, groot) and layout: E = new pairing's rate minus the mean of its object and target partners; a group BREAKS iff CI lower > 0 and E > 0.10. "
                                        "within only WITHIN-CHUNK; cross only CROSS-CHUNK; both BOTH; neither NEITHER. P_min and servo reported with no verdict",
                             "prediction": "WITHIN-CHUNK for both models in the familiar layout, carried by RELEASE; NEITHER in the new layout",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read; demos never executed",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any main atlas run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_atlas.py"))

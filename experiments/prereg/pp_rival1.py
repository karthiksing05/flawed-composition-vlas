"""PRE-REGISTRATION ROW of V1 and L1 (PLAN_PARTS 10.18; docs/RIVAL_TESTS.md).  After the instruments' code and BEFORE any run or read.

    python experiments/prereg/pp_rival1.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_rival1"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_v1_roll", "pp_v1_read.json", "pp_l1", "pp_l1_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


FILES = ["docs/RIVAL_TESTS.md", "docs/PLAN_PARTS.md", "experiments/pp_v1_client.py", "experiments/pp_v1_roll.sh", "experiments/pp_v1_read.py", "experiments/pp_l1_floor.py",
         "experiments/libero_client.py", "experiments/fc_n2_client.py", "experiments/pp_serve_purify.py", "experiments/pp_n4d.py", "experiments/fa1h_roll.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: V1 and L1 (docs/RIVAL_TESTS.md; PLAN_PARTS 10.18)",
                    metrics={"status": "PRE-REGISTERED V1 and L1 (docs/RIVAL_TESTS.md)",
                             "labelled": "post hoc; development reads; no training; changes no earlier verdict",
                             "V1_mapping": "familiar, pooled N1-N6: RECOVERS iff delta_p_A CI lower > 0 and > 0.10; STEERS iff delta_steer CI lower > 0 and > 0.10. yes/yes INSTRUCTION BINDING; "
                                           "no/yes HELD-OBJECT BINDING; no/no NOT THE INSTRUCTION; yes/no RECOVERS WITHOUT STEERING",
                             "L1a_mapping": "per atom and layout: k1 - k0 success CI lower > 0 and > 0.10 -> PART ADDS; else BACKBONE SUFFICES",
                             "L1b_mapping": "per group and layout: P_min - servo CI lower > 0 and > 0.10 -> P_MIN BEATS SERVO; CI upper < 0 and < -0.10 -> SERVO BEATS P_MIN; else TIED",
                             "prediction": "V1 INSTRUCTION BINDING; L1a PART ADDS for both atoms in the familiar layout; L1b SERVO BEATS P_MIN in the new layout, TIED in the familiar",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read; demos never executed",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any V1 or L1 run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_rival1.py"))

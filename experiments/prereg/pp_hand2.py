"""PRE-REGISTRATION ROW of the second hand-over round and Read G (docs/HAND_OVER.md 7).  After the instruments' smoke and BEFORE any main run or read.

    python experiments/prereg/pp_hand2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_hand2"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_hand2", "pp_hand2_read.json", "pp_read_g.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/HAND_OVER.md", "experiments/pp_atlas.py", "experiments/pp_hand2_client.py", "experiments/pp_hand2_roll.sh", "experiments/pp_hand2_read.py", "experiments/pp_read_g.py", "experiments/libero_client.py", "experiments/fc_n2_client.py", "experiments/pp_serve_purify.py", "experiments/fa1h_roll.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: second hand-over round and Read G (docs/HAND_OVER.md 7)",
                    metrics={"status": "PRE-REGISTERED the second hand-over round and Read G",
                             "labelled": "post hoc; development read; no training; changes no earlier verdict; not reviewed by the user, who is away and asked to keep iterating",
                             "mapping": "7.1 per VLA, familiar, seeds 11 and 12 pooled: REPLICATES iff the pp_hand mapping returns FIXED. "
                                        "7.2 per VLA, familiar: CONTACT NOT NEEDED iff the pp_hand mapping on H1 returns FIXED and s(Hs10) - s(H1) <= 0.10, else CONTACT MATTERS. "
                                        "7.3 pi0.5, familiar, N1-N6, per factor T and O: the budget is the largest radius, going up from 0.01 m with no skipped radius, whose s stays within 0.10 of s(Hs10); none reads BELOW 0.01 M. "
                                        "7.4 per object on the 120-step hold rate over the servo's GRASP states: NOT A BLOCK above 0.5, BLOCK below 0.2, otherwise MIXED",
                             "prediction": "7.1 REPLICATES on both; 7.2 CONTACT NOT NEEDED on both; 7.3 0.02 m on both factors; 7.4 NOT A BLOCK on both",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read; demos never executed",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any main run of this round or Read G"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_hand2.py"))

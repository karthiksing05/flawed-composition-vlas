"""PRE-REGISTRATION ROW of the hand-over (docs/HAND_OVER.md 2-3).  After the instruments' smoke and BEFORE any main run or read.

    python experiments/prereg/pp_hand.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_hand"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_hand", "pp_hand_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/HAND_OVER.md", "experiments/pp_atlas.py", "experiments/pp_hand_client.py", "experiments/pp_hand_roll.sh", "experiments/pp_hand_read.py", "experiments/libero_client.py", "experiments/fc_n2_client.py", "experiments/pp_serve_purify.py", "experiments/fa1h_roll.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: the hand-over (docs/HAND_OVER.md)",
                    metrics={"status": "PRE-REGISTERED the hand-over",
                             "labelled": "post hoc; development read; no training; oracle contact and oracle positions; changes no earlier verdict; not reviewed by the user, who is away and asked to keep iterating",
                             "mapping": "per VLA, familiar, pooled N1-N6, success: PART RAISES iff s(H) - s(intact) CI lower > 0 and > 0.10; a break remains under H iff E (atlas rule, partners under H) has CI lower > 0 and > 0.10. "
                                        "no PART RAISES: NO FIX. yes and none remain: FIXED. yes and one or both remain: RAISED, WITHIN STILL BREAKS; RAISED, CROSS STILL BREAKS; RAISED, BOTH STILL BREAK. Known tasks losing > 0.05 under H: KNOWN COST",
                             "prediction": "FIXED on both VLAs, familiar s(H) about .85; new layout about .55 on pi0.5, .35 on GR00T",
                             "part": "pp_full_naf_s10/P.npz, one seed, frozen",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read; demos never executed",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any main hand-over run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_hand.py"))

"""PRE-REGISTRATION ROW of reads E and F (docs/HAND_OVER.md 4).  BEFORE either read.

    python experiments/prereg/pp_gate_reads.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_gate_reads"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_gate_reads2.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/HAND_OVER.md", "experiments/pp_gate_reads2.py", "experiments/pp_atlas.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: reads E and F on existing tracks (docs/HAND_OVER.md 4)",
                    metrics={"status": "PRE-REGISTERED reads E and F",
                             "labelled": "post hoc; development read; no training; no rollouts; changes no earlier verdict; not reviewed by the user, who is away and asked to keep iterating",
                             "mapping": "E per system on N1-N6 new layout GRASP failures: first matching of PUSHED, NEVER REACHED, NO CLOSE, CLOSED NO HOLD; verdict is the class above one half, else MIXED. "
                                        "F per VLA on arm G, failed episodes still held at the end with d > 0.05: STALLS iff the share with net horizontal motion under 0.02 m over the last 50 tracked steps is above one half, else KEEPS MOVING",
                             "prediction": "E: PUSHED on the servo, NEVER REACHED on the VLAs. F: STALLS on both",
                             "states": "dev only: N2 main states 4-23; existing atlas and gate tracks",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "either read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_gate_reads.py"))

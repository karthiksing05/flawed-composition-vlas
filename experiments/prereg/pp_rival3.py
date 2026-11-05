"""PRE-REGISTRATION ROW of the V1 offset read (docs/RIVAL_TESTS.md 5).  BEFORE this read; only distances had been read from these files.

    python experiments/prereg/pp_rival3.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_rival3"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_v1_offset.json"))
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: V1 offset read (docs/RIVAL_TESTS.md 5)",
                    metrics={"status": "PRE-REGISTERED V1 offset read",
                             "labelled": "post hoc; development read of V1's existing episodes; changes no earlier verdict",
                             "mapping": "beta1 (arm A cheese fit) and beta2 (arm W bottle minus cheese contrast): yes iff CI lower > 0.5; no iff CI upper < 0.5; else unclear, also if a vector < 0.02 m or a cell < 5 episodes. "
                                        "yes/yes OFFSET CARRIES; no/no NOT THE OFFSET; anything else UNRESOLVED",
                             "prediction": "UNRESOLVED: call 1 yes, call 2 unclear",
                             "files_sha256": {p: fsha(p) for p in ("docs/RIVAL_TESTS.md", "experiments/pp_v1_offset.py")},
                             "written_before": "any miss direction was computed"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_rival3.py"))

"""PRE-REGISTRATION ROW of T1 (docs/SCAFFOLD_RESULTS.md 32).  BEFORE any rollout.

    python experiments/prereg/sc_t1.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t1_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t1" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t1", seed=0, command="pre-registration: T1 (docs/SCAFFOLD_RESULTS.md 32)",
                    metrics={"status": "PRE-REGISTERED T1",
                             "labelled": "development read on N2 main states 4-23; frozen VLAs; the user asked for this test on 2026-11-06",
                             "mapping": "per model and setting (2, 10 actions per query), new pairings N1-N6, familiar, against the frozen read at 5: GAIN iff CI lower > +.10; LOSS iff CI upper < -.10; else NO CLEAR CHANGE. Overall: CHUNK-SENSITIVE iff any GAIN; else NO GAIN FROM CHUNKING",
                             "prediction": "NO GAIN FROM CHUNKING; all four settings NO CLEAR CHANGE",
                             "states": "dev only: N2 main states 4-23; no smoke",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t1.py", "experiments/fc_n2_client.py", "experiments/libero_client.py"]},
                             "written_before": "any T1 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t1.py"))

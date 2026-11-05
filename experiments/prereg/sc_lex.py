"""PRE-REGISTRATION ROW of S1 (docs/SCAFFOLD_RESULTS.md 3).  BEFORE its read.

    python experiments/prereg/sc_lex.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_lex.json")), "read exists"
assert not any(r.get("tag") == "sc_lex" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_lex", seed=0, command="pre-registration: S1, the lexicon's refusal (docs/SCAFFOLD_RESULTS.md 3)",
                    metrics={"status": "PRE-REGISTERED S1",
                             "labelled": "development read, no rollouts; these instructions shaped L4's rule; the user has not reviewed these rules",
                             "mapping": "first match: OVER-REFUSES iff the full lexicon is exact on fewer than 12 read tasks; SILENT WRONG iff any leave-one-out or one-demo lexicon gives a wrong parse on a read task; else REFUSES SAFELY",
                             "prediction": "SILENT WRONG survives somewhere, because a token grounded to the wrong slot passes both refusal checks",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_lex.py", "lifelong_ood/scaffold/stores.py"]},
                             "written_before": "S1's read"}, bytes=Z, config={}, script="experiments/prereg/sc_lex.py"))

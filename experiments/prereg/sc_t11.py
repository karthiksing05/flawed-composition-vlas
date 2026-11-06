"""PRE-REGISTRATION ROW of T11 (docs/SCAFFOLD_RESULTS.md 24).  BEFORE any non-smoke build.

    python experiments/prereg/sc_t11.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t11_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t11" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t11", seed=0, command="pre-registration: T11 (docs/SCAFFOLD_RESULTS.md 24)",
                    metrics={"status": "PRE-REGISTERED T11",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the demos' event and lexicon rules were designed after reading their effect on the demos; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "push: K8 familiar over states: PUSH WORKS iff mean >= .50 and CI lower >= .25; PUSH PARTIAL iff CI lower > .05; else NO PUSH. Cost: Um - U, familiar, known K1-K6 and new pairings N1-N6: NO COST iff both CI lower >= -.10; else COST",
                             "prediction": "PUSH PARTIAL, K8 near .35; NO COST; N8 near 0",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t11.py", "experiments/pp_p_train.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any non-smoke T11 build"}, bytes=Z, config={}, script="experiments/prereg/sc_t11.py"))

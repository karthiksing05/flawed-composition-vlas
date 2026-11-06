"""PRE-REGISTRATION ROW of T12 (docs/SCAFFOLD_RESULTS.md 25).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t12.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t12_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t12" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t12", seed=0, command="pre-registration: T12 (docs/SCAFFOLD_RESULTS.md 25)",
                    metrics={"status": "PRE-REGISTERED T12",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the rule was designed after reading T11's N8 failure and revised after a smoke; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "N8 familiar over states: VERB COMPOSES iff mean >= .50 and CI lower >= .25; VERB PARTIAL iff CI lower > .05; else VERB DOES NOT COMPOSE. K8 familiar V - Um: K8 HOLDS iff CI lower >= -.10; else K8 HURT",
                             "prediction": "VERB COMPOSES, N8 near .70 familiar; K8 HOLDS, delta near 0; N8 near .20 on the new layout",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row: N8 2/2, K8 2/2 on seed 10",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t12.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py", "results/flawed/sc_t12_modes.json"]},
                             "written_before": "any non-smoke T12 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t12.py"))

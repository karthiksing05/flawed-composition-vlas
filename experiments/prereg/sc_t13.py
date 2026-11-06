"""PRE-REGISTRATION ROW of T13 (docs/SCAFFOLD_RESULTS.md 26).  BEFORE any non-smoke rollout.

    python experiments/prereg/sc_t13.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t13_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t13" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t13", seed=0, command="pre-registration: T13 (docs/SCAFFOLD_RESULTS.md 26)",
                    metrics={"status": "PRE-REGISTERED T13",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; two new instructions; the word rule changed after working the cheese parse by hand; the user asked to keep iterating overnight and has not reviewed this rule",
                             "mapping": "per task, familiar over states: VERB COMPOSES iff mean >= .50 and CI lower >= .25; VERB PARTIAL iff CI lower > .05; else VERB DOES NOT COMPOSE. Reach: GENERAL iff both tasks at least partial; SOME OBJECTS iff one; BOWL ONLY iff neither",
                             "prediction": "P1 cheese VERB COMPOSES near .70; P2 bottle VERB DOES NOT COMPOSE near .05; reach SOME OBJECTS",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row: P1 2/2, P2 0/2 on seed 10",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t13.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py", "results/flawed/sc_t12_modes.json", "results/flawed/sc_t13_bddl/P1.bddl", "results/flawed/sc_t13_bddl/P2.bddl"]},
                             "written_before": "any non-smoke T13 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t13.py"))

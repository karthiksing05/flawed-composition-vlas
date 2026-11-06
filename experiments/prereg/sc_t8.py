"""PRE-REGISTRATION ROW of T8a and T9 (docs/SCAFFOLD_RESULTS.md 16).  BEFORE any non-smoke build or rollout.

    python experiments/prereg/sc_t8.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t8_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t8" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t8", seed=0, command="pre-registration: T8a and T9 (docs/SCAFFOLD_RESULTS.md 16)",
                    metrics={"status": "PRE-REGISTERED T8a T9",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the user asked to keep iterating overnight and has not reviewed these rules",
                             "mapping_ladder": "per arm, U minus arm, first match: UNDERFIT iff known familiar difference > +.10 with CI lower > 0; FALLS iff new pairings familiar CI lower > +.10; else NO FALL. Overall C1 RUNGS HOLD iff RF and HF both FALLS; else C1 RUNGS FAIL. Ulc reported separately",
                             "mapping_stream": "final store minus full U on K1 K4 N1 N4 N5 N6 familiar: STREAM HOLDS iff lower bound >= -.10 in both orders; else NOT YET",
                             "mapping_old": "final store minus base on K2 K3 K6 N2 N3 familiar: NOT FORGOTTEN iff lower bound >= -.10 in both orders; else FORGETS",
                             "prediction": "C1 RUNGS HOLD with Ulc NO FALL; NOT YET; NOT FORGOTTEN",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 with 50-step builds read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t8.py", "experiments/pp_p_train.py", "experiments/pp_baselines.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any non-smoke build or rollout of T8a or T9"}, bytes=Z, config={}, script="experiments/prereg/sc_t8.py"))

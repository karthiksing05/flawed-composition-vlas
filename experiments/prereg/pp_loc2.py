"""PRE-REGISTRATION ROW of the locator's second round (docs/LOCATOR.md 9).  BEFORE any of its training or its read.

    python experiments/prereg/pp_loc2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_loc2"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_loc2.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/LOCATOR.md", "experiments/pp_loc2.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: locator round two (docs/LOCATOR.md 9)",
                    metrics={"status": "PRE-REGISTERED locator round two",
                             "labelled": "post hoc; development read; no rollouts; the 20 states shaped this design, so a pass is not evidence on unseen states; changes no earlier verdict; the two mechanisms approved by the user, details not reviewed",
                             "mapping": "on arm CPS alone: LOCATOR 5's mapping unchanged, first match: WITHIN BUDGET iff every head has e <= 0.02 m in both layouts; BEATS A TABLE, OVER BUDGET iff every head has e <= T/2 in the new layout; otherwise NO BETTER THAN A TABLE. CP, CS and AP are controls and take labels only as a report",
                             "prediction": "CPS BEATS A TABLE, OVER BUDGET, with most of the gain from P; AP also BEATS A TABLE",
                             "oracle_kept": "simulator camera matrices in every arm",
                             "states": "dev only: N2 main states 4-23, never in training; training on K1-K6 demos 0-39, familiar layout",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any of this round's training or its read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_loc2.py"))

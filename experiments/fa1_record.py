"""Ledger row fc_a1 for the prototype-A stage-1 read (FLAWED_COMPOSITION 5.5): the summary of results/flawed/a1_read.json, written
after experiments/fa1_read.py.  CPU.

    python experiments/fa1_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1"]
assert len(pre) == 1, "the pa1 pre-registration row must exist"
print(ledger.record("fc_a1", bench="libero_goal", tag="a1", seed=0, command="python experiments/fa1_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], **{k: v for k, v in r.items() if k not in ("per_compound", "secondary")}},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1_read.json", "smoke": "results/flawed/a1_smoke.json",
                               "calib": "results/flawed/a1_calib.json", "train": "results/flawed/a1_train_final.json"},
                    script="experiments/fa1_record.py"))

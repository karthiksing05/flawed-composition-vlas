"""Ledger row fc_a1c for the prototype-A stage-1c read (FLAWED_COMPOSITION 5.7): the summary of results/flawed/a1c_read.json, written
after experiments/fa1c_read.py.  CPU.

    python experiments/fa1c_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1c_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1c"]
assert len(pre) == 1, "the pa1c pre-registration row must exist"
print(ledger.record("fc_a1c", bench="libero_goal", tag="a1c", seed=0, command="python experiments/fa1c_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], **{k: v for k, v in r.items() if k not in ("per_compound",)}},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1c_read.json", "swap": "results/flawed/a1c_swap.json",
                               "support": "results/flawed/a1c_support.json"},
                    script="experiments/fa1c_record.py"))

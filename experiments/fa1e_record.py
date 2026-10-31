"""Ledger row fc_a1e for the prototype-A stage-1e read (FLAWED_COMPOSITION 5.9): the summary of results/flawed/a1e_read.json, written
after experiments/fa1e_read.py.  CPU.

    python experiments/fa1e_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1e_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1e"]
assert len(pre) == 1, "the pa1e pre-registration row must exist"
print(ledger.record("fc_a1e", bench="libero_goal", tag="a1e", seed=0, command="python experiments/fa1e_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], **{k: v for k, v in r.items() if k not in ("per_compound",)}},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1e_read.json", "swap": "results/flawed/a1e_swap.json"},
                    script="experiments/fa1e_record.py"))

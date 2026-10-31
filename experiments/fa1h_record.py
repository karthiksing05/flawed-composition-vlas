"""Ledger row fc_a1h for the prototype-A stage-1h read (FLAWED_COMPOSITION 5.14): the summary of results/flawed/a1h_read.json.  CPU.

    python experiments/fa1h_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1h_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1h"]
assert len(pre) == 1, "the pa1h pre-registration row must exist"
summ = {lay: {k: v for k, v in x.items() if k in ("verdict", "cells", "contrasts", "three_factor", "grasp_identity", "errors", "K5_K6")}
        for lay, x in r["layouts"].items()}
print(ledger.record("fc_a1h", bench="libero_goal", tag="a1h", seed=0, command="python experiments/fa1h_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "verdict": r["verdict"], "layouts": summ},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1h_read.json", "swap": "results/flawed/a1h_swap.json",
                               "probe": "results/flawed/a1h_probe.json"},
                    script="experiments/fa1h_record.py"))

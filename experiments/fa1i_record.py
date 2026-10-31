"""Ledger row fc_a1i for the prototype-A stage-1i read (FLAWED_COMPOSITION 5.16): the summary of results/flawed/a1i_read.json.  CPU.

    python experiments/fa1i_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1i_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1i"]
assert len(pre) == 1, "the pa1i pre-registration row must exist"
summ = {lay: {k: v for k, v in x.items() if k in ("verdict", "cells", "contrasts", "three_factor", "grasp_identity", "errors", "K5_K6")}
        for lay, x in r["layouts"].items()}
print(ledger.record("fc_a1i", bench="libero_goal", tag="a1i", seed=0, command="python experiments/fa1i_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "verdict": r["verdict"], "layouts": summ},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1i_read.json", "swap": "results/flawed/a1i_swap.json",
                               "diagnostic": "results/flawed/a1i_diag.json"},
                    script="experiments/fa1i_record.py"))

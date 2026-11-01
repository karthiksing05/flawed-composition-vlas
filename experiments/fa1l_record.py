"""Ledger row fc_a1l for the prototype-A stage-1l read (FLAWED_COMPOSITION 5.21).  CPU.

    python experiments/fa1l_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1l_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1l"]
assert len(pre) == 1, "the pa1l pre-registration row must exist"
summ = {lay: {k: v for k, v in x.items() if k in ("verdict", "cells", "contrasts", "three_factor", "grasp_identity", "errors", "K5_K6",
                                                  "post_hoc_object_matched_C2_descriptive", "descriptive_same_data_baselines")}
        for lay, x in r["layouts"].items()}
print(ledger.record("fc_a1l", bench="libero_goal", tag="a1l", seed=9, command="python experiments/fa1l_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "verdict": r["verdict"], "layouts": summ},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1l_read.json", "swap": "results/flawed/a1l_swap.json",
                               "data": "results/flawed/a1l_gen.json", "diagnostic": "results/flawed/a1l_diag.json"},
                    script="experiments/fa1l_record.py"))

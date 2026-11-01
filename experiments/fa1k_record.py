"""Ledger row fc_a1k for the prototype-A stage-1k read (FLAWED_COMPOSITION 5.19).  CPU.

    python experiments/fa1k_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1k_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1k"]
assert len(pre) == 1, "the pa1k pre-registration row must exist"
summ = {lay: {k: v for k, v in x.items() if k in ("verdict", "cells", "contrasts", "three_factor", "grasp_identity", "errors", "K5_K6",
                                                  "post_hoc_object_matched_C2_descriptive", "descriptive_same_data_baselines")}
        for lay, x in r["layouts"].items()}
print(ledger.record("fc_a1k", bench="libero_goal", tag="a1k", seed=6, command="python experiments/fa1k_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "verdict": r["verdict"], "layouts": summ},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1k_read.json", "swap": "results/flawed/a1k_swap.json",
                               "data": "results/flawed/a1k_gen.json", "diagnostic": "results/flawed/a1k_diag.json"},
                    script="experiments/fa1k_record.py"))

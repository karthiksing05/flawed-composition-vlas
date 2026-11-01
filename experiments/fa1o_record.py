"""Ledger row fc_a1o for the prototype-A stage-1o read (FLAWED_COMPOSITION 5.28).  CPU.

    python experiments/fa1o_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1o_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1o"]
assert len(pre) == 1, "the pa1o pre-registration row must exist"
summ = {lay: {k: v for k, v in x.items() if k in ("verdict", "cells", "contrasts", "three_factor", "grasp_identity", "errors", "K5_K6",
                                                  "post_hoc_object_matched_C2_descriptive", "hold_factor_descriptive")}
        for lay, x in r["layouts"].items()}
print(ledger.record("fc_a1o", bench="libero_goal", tag="a1o", seed=21, command="python experiments/fa1o_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "verdict": r["verdict"], "layouts": summ},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1o_read.json", "swap": "results/flawed/a1o_swap.json",
                               "sensitivity": "results/flawed/a1o_sens.json", "identity_probe": "results/flawed/a1o_idprobe.json",
                               "merge": "results/flawed/a1o_merge.json", "diagnostic": "results/flawed/a1o_close.json"},
                    script="experiments/fa1o_record.py"))

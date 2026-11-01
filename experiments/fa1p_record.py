"""Ledger row fc_a1p for the prototype-A stage-1p read (FLAWED_COMPOSITION 5.30).  CPU.

    python experiments/fa1p_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "a1p_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pa1p"]
assert len(pre) == 1, "the pa1p pre-registration row must exist"
summ = {lay: {k: v for k, v in x.items() if k in ("verdict", "cells", "contrasts", "three_factor", "grasp_identity", "errors", "K5_K6",
                                                  "post_hoc_object_matched_C2_descriptive", "hold_factor_descriptive")}
        for lay, x in r["layouts"].items()}
print(ledger.record("fc_a1p", bench="libero_goal", tag="a1p", seed=24, command="python experiments/fa1p_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "verdict": r["verdict"], "layouts": summ},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/a1p_read.json", "swap": "results/flawed/a1p_swap.json",
                               "sensitivity": "results/flawed/a1p_sens.json", "identity_probe": "results/flawed/a1p_idprobe.json",
                               "generation": "results/flawed/a1p_sgen.json", "near_rack": "results/flawed/a1p_cover.json"},
                    script="experiments/fa1p_record.py"))

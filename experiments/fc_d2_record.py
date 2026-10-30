"""Ledger row fc_d2 for the D1x / D2 read (FLAWED_COMPOSITION 4.8): the summary of results/flawed/d2_read.json, written after
experiments/fc_d2_read.py.  CPU.

    python experiments/fc_d2_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "d2_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pd2"]
assert len(pre) == 1, "the pd2 pre-registration row must exist"
print(ledger.record("fc_d2", bench="libero_90_c", tag="d2", seed=0, command="python experiments/fc_d2_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"],
                             "D1x": {m: {k: v for k, v in x.items() if k != "per_task"} for m, x in r["D1x"].items()},
                             "D2": {k: v for k, v in r["D2"].items() if k != "rows"}},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/d2_read.json",
                               "records": [f"results/flawed/{f}" for f in ("d1x_main2.json", "d1x_c1.json", "d2_pi05.json", "d2_xvla.json", "d2_groot.json")]},
                    script="experiments/fc_d2_record.py"))

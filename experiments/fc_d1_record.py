"""Ledger row fc_d1 for the D0 / D1 read (FLAWED_COMPOSITION 4.6): the summary of results/flawed/d1_read.json, written after
experiments/fc_d1_read.py.  CPU.

    python experiments/fc_d1_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "d1_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pd01"]
assert len(pre) == 1, "the pd01 pre-registration row must exist"
strip = lambda d: {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_task"})   # noqa: E731
                   for k, v in d.items()}
print(ledger.record("fc_d1", bench="libero_90_c", tag="d01", seed=0, command="python experiments/fc_d1_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], "task_sets": r["task_sets"], "D1a": r["D1a"], "D1b": r["D1b"],
                             "D1c": r["D1c"], "D0": r["D0"], "table": {m: strip(v) for m, v in r["table"].items()},
                             "per_task_delta_F": r["per_task_delta_F"]},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/d1_read.json", "records": [f"results/flawed/d1_{m}.json" for m in ("main2", "c1", "vanilla")]},
                    script="experiments/fc_d1_record.py"))

"""Ledger row fc_n3 for the N3 read (FLAWED_COMPOSITION 4.12): the summary of results/flawed/n3_read.json, written after
experiments/fc_n3_read.py.  CPU.

    python experiments/fc_n3_record.py
"""
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

r = json.load(open(os.path.join(ROOT, "results", "flawed", "n3_read.json"), encoding="utf-8"))
pre = [x for x in ledger.load("m3_prereg") if x.get("tag") == "pn3"]
assert len(pre) == 1, "the pn3 pre-registration row must exist"
print(ledger.record("fc_n3", bench="libero_spatial", tag="n3", seed=0, command="python experiments/fc_n3_read.py",
                    metrics={"prereg_run_id": pre[0]["run_id"], **{k: v for k, v in r.items() if k != "per_pair"}},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    artifacts={"read": "results/flawed/n3_read.json", "smoke": "results/flawed/n3_smoke.json",
                               "novelty": "results/flawed/n3_novelty.json", "tasks": "results/flawed/n3_tasks.json"},
                    script="experiments/fc_n3_record.py"))

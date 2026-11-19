"""PRE-REGISTRATION ROW of MT (SR 81; docs/notes/mt_plan.md): the moved-target test on frozen pi0.5 (I), R2, RAA and rgp, no training.  BEFORE any main rollout.
Predictions: experiments/prereg/mt_pred.txt.

    python experiments/prereg/mt.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "mt" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "mt_roll")), "mt main rollouts exist"
assert not os.path.exists(os.path.join(FL, "mt_read.json")), "mt read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/mt/mt_tasks.json", "results/flawed/mt/states/n2_goal_0.npy", "results/flawed/mt/states/n2_goal_2.npy"]
STATES += ["results/flawed/mt/mt_br_%s_main.json" % c for c in ("K1_in", "T_near", "T_far", "O1_near", "O1_far", "K2_in", "O2_near", "O2_far")]
CODE = ["experiments/mt_tasks.py", "experiments/mt_client.py", "experiments/mt_read.py", "experiments/prereg/mt.py", "experiments/prereg/mt_pred.txt",
        "docs/notes/mt_plan.md", "experiments/raa.py", "experiments/rg.py", "experiments/sc_rv.py", "experiments/pp_atlas.py", "experiments/fc_n2_client.py",
        "experiments/libero_client.py", "experiments/pp_serve_purify.py", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json", PP + "mt_roll.sh"] + STATES
PRED = open(os.path.join(ROOT, "experiments", "prereg", "mt_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-18.  Known when written: SR 70 (I on F6_a .967), SR 73, the build's render check, and a 4-state smoke per arm per cell (jobs 4795016-4795019, read 4795141, outside results/) run "
        "for plumbing whose n=4 outcomes the read's table printed (summary in mt_pred.txt; the predictions were written before it and are unchanged).  mt_read selftest passed.  No code, state, arm, mapping or prediction changed after this row.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="mt", seed=0,
                    command="pre-registration: MT (SR 81), moved-target test, arms I / r2 / raa / rgp, states results/flawed/mt (docs/notes/mt_plan.md)",
                    metrics={"status": "PRE-REGISTERED MT",
                             "arms": "I = frozen pi0.5 (pp_serve_purify intact), r2 = pi05_g8v_rview2_ae_tl6000, raa = pi05_g8v_raa_ae_tl6000, rgp = ReGuide-perc on I; no training",
                             "cells": "K1_in T_near T_far O1_near O1_far K2_in O2_near O2_far, 40 main states each",
                             "mapping": "experiments/mt_read.py docstring (endpoint attribution, M1, M2, M3, MK)",
                             "prediction": PRED, "seen": SEEN, "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main rollout of MT"},
                    bytes=Z, config={}, script="experiments/prereg/mt.py"))

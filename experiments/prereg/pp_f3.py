"""PRE-REGISTRATION ROW (supplement to pp_f123) of PLAN_PARTS F3 closed loop (docs/PARTS_RESULTS.md sections 1 and 3): the components
F1 located (pi0.5 KV_14; GR00T Q_6) and the exact code, BEFORE any F3 closed-loop rollout.  The criteria are pp_f123's, unchanged.

    python experiments/prereg/pp_f3.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
assert not os.path.exists(os.path.join(F, "pp_f3_roll")), "F3 rollouts exist"
assert any(r.get("tag") == "pp_f123" for r in ledger.load("m3_prereg")), "pp_f123 missing"
assert not any(r.get("tag") == "pp_f3" for r in ledger.load("m3_prereg")), "pp_f3 already registered"
rd = json.load(open(os.path.join(F, "pp_f1_read.json"), encoding="utf-8"))
loc = {m: rd["models"][m]["located"]["op"] for m in ("pi05", "groot")}
assert loc == {"pi05": "KV_14", "groot": "Q_6"}, loc


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


FILES = ["experiments/pp_purify.py", "experiments/pp_serve_purify.py", "experiments/pp_f3_roll.sh", "experiments/pp_f3_read.py",
         "experiments/pp_f1_mix.py", "results/flawed/pp_f3_branches.json", "results/flawed/pp_f1_read.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_f3", seed=0,
                    command="pre-registration: PLAN_PARTS F3 closed loop (components located by F1; criteria of pp_f123)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS F3 closed loop (docs/PARTS_RESULTS.md section 3)",
                             "components": loc,
                             "arms": "intact re-run and purified, same N2 branch seeds; N1-N6 familiar (primary) and new, states 4-13",
                             "criterion": "pp_f123 F3: ties iff delta pooled p (purified - intact, familiar) > 0 with CI excluding 0",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any F3 closed-loop rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_f3.py"))

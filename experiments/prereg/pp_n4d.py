"""PRE-REGISTRATION ROW of PLAN_PARTS round 1, the N4 diagnostic (docs/PARTS_RESULTS.md section 10).  Written after the N4 carry
generation and the single-task fits (data and training only), BEFORE the oracle runs, any rollout of this round and its read.

    python experiments/prereg/pp_n4d.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_n4d_roll", "pp_n4d_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == "pp_n4d" for r in ledger.load("m3_prereg")), "pp_n4d already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 10\. Round 1 design note.*?(?=\n## 11\.|\Z)", doc, flags=re.S)
assert m, "section 10 missing"
FILES = ["experiments/pp_n4d.py", "experiments/pp_p_roll.py", "experiments/fa1g_plan.py", "experiments/fa1g_read.py",
         "lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "results/flawed/pp_n4d_gen.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_n4d", seed=0,
                    command="pre-registration: PLAN_PARTS round 1, N4 diagnostic (competence vs combination)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS round 1 N4 diagnostic (docs/PARTS_RESULTS.md section 10)",
                             "section_10_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arms": "a: single-task PartsFlow r32 on 50 generated N4 carries (seeds 0-2); P: pp_full_r32_s10-12; controls: "
                                     "single-task fits on generated K4 / K1; o: oracle planner.  Familiar layout, dev states 4-23",
                             "mapping": "D = s_a(N4) - s_P(N4): competence iff D <= 0.10; combination iff D > 0.10 and CI lower bound > 0 "
                                        "(failing phase = largest P deficit among hold, arrival, place-on-arrival); else unresolved",
                             "flag": "a control's single-task fit beating P on its own task by > 0.10 with lower bound > 0 (reported, no "
                                     "verdict change)",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the oracle runs, any rollout of this round and its read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_n4d.py"))

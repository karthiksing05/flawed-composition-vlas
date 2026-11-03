"""PRE-REGISTRATION ROW of PLAN_PARTS step 1, the N4 diagnostic with the fixed instrument (docs/PARTS_RESULTS.md sections 18-19).
Written after the instrument v2 gate passed (single-task fits on familiar servo-generated carries: K1 1.00 vs P 0.97, K4 0.72 vs
P 0.63) and BEFORE any N4 rollout of this step (arm a on N4, P on N4, the oracle) and its read.

    python experiments/prereg/pp_n4d2.py
"""
import glob
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
assert not glob.glob(os.path.join(F, "pp_n4d2_roll", "*_main_22.json")), "an N4 rollout of this step exists"
assert not glob.glob(os.path.join(F, "pp_n4d2_roll", "oracle_*")), "an oracle rollout of this step exists"
assert not os.path.exists(os.path.join(F, "pp_n4d2_read.json")), "the read exists"
assert not any(r.get("tag") == "pp_n4d2" for r in ledger.load("m3_prereg")), "pp_n4d2 already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 18\. Step 1 design note.*?(?=\n## 20\.|\Z)", doc, flags=re.S)
assert m, "sections 18-19 missing"
FILES = ["experiments/pp_n4d.py", "experiments/pp_n4d2.py", "experiments/fa1h_roll.py", "experiments/fa1g_plan.py",
         "lifelong_ood/policy/parts_flow.py", "results/flawed/pp_n4d2v_gen_N4.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_n4d2", seed=0,
                    command="pre-registration: PLAN_PARTS step 1, N4 diagnostic with the fixed instrument (v2)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS N4 diagnostic, instrument v2 (docs/PARTS_RESULTS.md sections 18-19)",
                             "sections_18_19_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "gate": "passed: single-task K1 1.00 vs P 0.97; K4 0.72 vs P 0.63 (within 0.15)",
                             "arms": "a: single-task PartsFlow r32 on 50 familiar servo-generated N4 carries (seeds 0-2); P: pp_full_r32_s10-12; "
                                     "o: oracle planner (stage 1g's, unchanged).  Familiar layout, dev states 4-23; tilt recorded",
                             "mapping": "pp_n4d's, unchanged: D = s_a(N4) - s_P(N4): competence iff D <= 0.10; combination iff D > 0.10 and "
                                        "CI lower bound > 0 (failing phase = largest P deficit among hold, arrival, place-on-arrival); else "
                                        "unresolved",
                             "tilt": "held-but-failed N4 episodes split by final tilt > 45 deg (tipped) vs upright (off-site release)",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any N4 rollout of this step and its read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_n4d2.py"))

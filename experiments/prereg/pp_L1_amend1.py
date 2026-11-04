"""PRE-REGISTRATION AMENDMENT ROW of ladder rung L1 (docs/LADDER_RESULTS.md section 2, Amendment 1): the arm adds --rtclip (L0's own
target-term configuration; the scalar is only the fallback radius of a target part with no held rows), after the registered builds
hit a NaN loss and before any L1 rollout, S1 run or read.

    python experiments/prereg/pp_L1_amend1.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_L1_amend1"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_L1_roll", "pp_ladder_L1_read.json", "pp_L1_smoke_rollagree.json", "pp_s1_mix_full_L1_m0.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert any(r.get("tag") == "pp_L1" for r in ledger.load("m3_prereg")), "pp_L1 not registered"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"
doc = open(os.path.join(ROOT, "docs", "LADDER_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"\*\*Amendment 1 .*?(?=\n## |\Z)", doc, flags=re.S)
assert m, "amendment text missing"
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration amendment: ladder rung L1 adds --rtclip (docs/LADDER_RESULTS.md section 2, Amendment 1)",
                    metrics={"status": "AMENDMENT to pp_L1 before any L1 rollout, S1 run or read",
                             "amendment_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arm": "--rank 32 --rtclip --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize --pgate",
                             "why": "registered builds: NaN loss (target part without event-held rows -> radius falls back to rtclip = 0 "
                                    "-> s_t = 0); L0's target term (P_nsz's) carries --rtclip",
                             "unchanged": "every measure, the counted set, the verdict mapping and the stop rules of pp_L1"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_L1_amend1.py"))

"""PRE-REGISTRATION ROW of the in-hand test (PLAN_PARTS 10.16; docs/PLACE_DIAG.md section 4).  Written after the instruments' code and BEFORE any I1 or I2 run.

    python experiments/prereg/pp_inhand.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_inhand"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_inhand_trace", "pp_inhand_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PLACE_DIAG.md"), encoding="utf-8").read()
m = re.search(r"## 4\. In-hand test.*?(?=\n## 5\.|\Z)", doc, flags=re.S)
assert m and "### 4.1 Pre-registration" in m.group(0), "section 4 missing"
FILES = ["experiments/pp_inhand_trace.py", "experiments/pp_inhand_read.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py", "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: in-hand test (docs/PLACE_DIAG.md section 4; PLAN_PARTS 10.16)",
                    metrics={"status": "PRE-REGISTERED in-hand test (docs/PLACE_DIAG.md section 4)",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "labelled": "post hoc diagnostic; norot is not a candidate fix; builds no model; changes no criterion, GO or ladder verdict",
                             "arms": "P_min pp_full_naf_s{10,11,12}: base, norot (a[3:6] = 0 while the oracle hold is on)",
                             "contrasts": "N1-K1, N4-K1, N6-K6; new layout the headline, familiar beside it",
                             "mapping": "observational, first match wins: G's CI contains 0 -> NO GAP; overlap < 0.5 -> UNDETERMINED; |A| <= 0.5 |G| -> TILT-MEDIATED; else NOT EXPLAINED "
                                        "BY TILT. Intervention, on the contrast's lower-place task: delta's CI lower > 0 and delta > 0.10 -> ROTATION-CAUSAL; else NO EFFECT",
                             "prediction": "new layout: N1-K1 TILT-MEDIATED, NO EFFECT; N4-K1 NO GAP; N6-K6 NOT EXPLAINED BY TILT",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any I1 or I2 run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_inhand.py"))

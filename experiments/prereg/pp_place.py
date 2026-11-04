"""PRE-REGISTRATION ROW of the place diagnostic (PLAN_PARTS 10.15; docs/PLACE_DIAG.md sections 1-2).  Written after the instruments' code and
BEFORE any T0-T4 run or read.

    python experiments/prereg/pp_place.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_place"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_place_trace", "pp_place_drop", "pp_place_read.json", "pp_place_t0.json", "pp_place_t3.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PLACE_DIAG.md"), encoding="utf-8").read()
m = re.search(r"## 1\. Design note.*?(?=\n## 3\.|\Z)", doc, flags=re.S)
assert m and "## 2. Pre-registration" in m.group(0), "sections 1-2 missing"
FILES = ["experiments/pp_place_trace.py", "experiments/pp_place_drop.py", "experiments/pp_place_t3.py", "experiments/pp_place_read.py",
         "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py", "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: place diagnostic (docs/PLACE_DIAG.md sections 1-2; PLAN_PARTS 10.15)",
                    metrics={"status": "PRE-REGISTERED place diagnostic (docs/PLACE_DIAG.md sections 1-2)",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "labelled": "post hoc diagnostic; builds no model; changes no criterion, GO or ladder verdict",
                             "arm": "P_min: pp_full_naf_s{10,11,12}",
                             "contrasts": "N1-K1, N4-K1, N6-K6; new layout the headline, familiar reported beside it",
                             "mapping": "per contrast, first match wins: G's CI contains 0 -> NOISE; G_c's CI contains 0 -> ARRIVAL-DEFINITION; overlap < 0.5 -> "
                                        "UNDETERMINED; |A| <= 0.5 |G_c| -> RELEASE-STATE; D has G_c's sign and |D| >= 0.10 -> SAME-STATE: PHYSICS; else "
                                        "SAME-STATE: UNMEASURED. T3 not exact -> stop",
                             "prediction": "new layout: N1-K1 SAME-STATE: PHYSICS; N4-K1 RELEASE-STATE; N6-K6 SAME-STATE: PHYSICS",
                             "states": "dev only: N2 main states 4-23 (T4: 4-7), N2 seeds; no confirmatory or sealed read",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any T0-T4 run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_place.py"))

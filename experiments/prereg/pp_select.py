"""PRE-REGISTRATION ROW of the selection check (docs/PLACE_DIAG.md section 6), before its read.

    python experiments/prereg/pp_select.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_select"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_select_read.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"
doc = open(os.path.join(ROOT, "docs", "PLACE_DIAG.md"), encoding="utf-8").read()
m = re.search(r"## 6\. Selection check.*?(?=\n## 7\.|\Z)", doc, flags=re.S)
assert m, "section 6 missing"
sha = hashlib.sha256(open(os.path.join(ROOT, "experiments", "pp_select_read.py"), "rb").read()).hexdigest()
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: selection check (docs/PLACE_DIAG.md section 6)",
                    metrics={"status": "PRE-REGISTERED selection check (docs/PLACE_DIAG.md section 6)",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "labelled": "post hoc; read of saved rollouts; changes no criterion, GO or ladder verdict",
                             "mapping": "N1-K1, first match wins: start states match on < 90% -> NOT TESTABLE; shared episodes < 15 -> TOO FEW; G_s's CI contains 0 and "
                                        "|G_s| <= 0.5 |G_all| -> START-STATE SELECTION; else NOT SELECTION",
                             "prediction": "NOT SELECTION", "read_sha256": sha, "written_before": "the read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_select.py"))

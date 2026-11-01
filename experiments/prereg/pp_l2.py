"""PRE-REGISTRATION ROW of PLAN_PARTS Layer 2 (docs/PARTS_RESULTS.md section 4): P-build, S1-S4.  Written after the builds and
creations (training only), BEFORE any S1 read and any Layer-2 rollout.

    python experiments/prereg/pp_l2.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_s2_roll", "pp_s3_roll", "pp_s4_old", "pp_l2_read.json", "pp_s1_mix_full.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == "pp_l2" for r in ledger.load("m3_prereg")), "pp_l2 already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 4\. Layer 2 protocol.*?(?=\n## 5\.|\Z)", doc, flags=re.S)
assert m, "section 4 missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_baselines.py", "experiments/pp_p_roll.py",
         "experiments/pp_s1_mix.py", "experiments/pp_l2_read.py", "experiments/fa1h_roll.py", "experiments/fa1g_read.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_l2", seed=0,
                    command="pre-registration: PLAN_PARTS Layer 2 (P-build, S1 locality, S2 multiplicativity, S3 creation, S4 baselines)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS Layer 2 (docs/PARTS_RESULTS.md section 4)",
                             "section_4_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "S1": "holds iff swap tests exactly 0 (every seed) and every tg component exactly 0 on every frame",
                             "S2": "holds iff in both layouts 3F (every N1-N6 D3 CI contains 0) and object-matched ratio >= 0.8 with CI "
                                   "lower bound >= 0.6",
                             "S3_layer3_go": "S1 and S2 hold and, for bottle and plate, both layouts, every k in {1, 5, 10}: transfer "
                                             "ratio >= 0.8 with CI lower bound >= 0.6",
                             "S4": "descriptive: s_train, s_unseen, ratio, P - baseline on unseen partners, forgetting",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any S1 read and any Layer-2 rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_l2.py"))

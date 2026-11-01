"""PRE-REGISTRATION ROW of PLAN_PARTS Layer 1 (docs/PARTS_RESULTS.md section 1): F1 (role-mixing map), F2 (mixing -> N2 failure),
F3 (purification, criteria fixed now; its component is the F1-located k*_op).  Written after the frames and the smoke runs (whose
numbers were not inspected), BEFORE any F1 main run or read.

    python experiments/prereg/pp_f123.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_f1_pi05.json", "pp_f1_groot.json", "pp_f1_oracle.json", "pp_f1_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists: F1 already run"
assert not any(r.get("tag") == "pp_f123" for r in ledger.load("m3_prereg")), "pp_f123 already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 1\. Layer 1 protocol.*?(?=\n## 2\.|\Z)", doc, flags=re.S)
assert m, "section 1's protocol is missing"
FILES = ["docs/PLAN_PARTS.md", "experiments/pp_f1_frames.py", "experiments/pp_f1_mix.py", "experiments/pp_f1_oracle.py",
         "experiments/pp_f1_read.py", "results/flawed/pp_f1_frames.npz", "results/flawed/n2_tasks.json",
         "lifelong_ood/policy/factored_flow.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_f123", seed=0,
                    command="pre-registration: PLAN_PARTS Layer 1 (F1 role-mixing map, F2 mixing -> N2 failure, F3 purification)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS F1/F2/F3 (docs/PARTS_RESULTS.md section 1)",
                             "section_1_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "F1": "per VLA: holds iff some component (W, V, Rest, KV_l | Q_l, E_j), some direction (tg, op), has the "
                                   "compound-bootstrap CI lower bound of pooled m above floor_dir = max over A's components of A's CI "
                                   "upper bound; location k*_dir = argmax pooled m over {W, V, KV_l | Q_l}",
                             "F2": "per VLA: ties iff rho_g (m_{k*_tg} vs 1-g, 22 grasp cells) > 0 with CI excluding 0, or rho_p "
                                   "(m_{k*_op} vs 1-p, 11 familiar place cells) > 0 with CI excluding 0; N2 main rollouts",
                             "F3": "per VLA: purify k*_op on N1-N6 (familiar primary, states 4-13, 10 eps/task) against an intact "
                                   "re-run on the same branch seeds; ties iff delta pooled p > 0 with CI excluding 0 (bootstrap over "
                                   "tasks and states); open loop descriptive; if k*_op = V stop and report",
                             "go_kill": "GO to Layer 2 iff for at least one VLA F1 holds and (F2 or F3) ties; else KILL K1",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any F1 main run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_f123.py"))

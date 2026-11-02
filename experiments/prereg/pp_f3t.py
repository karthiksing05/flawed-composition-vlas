"""PRE-REGISTRATION ROW of PLAN_PARTS targeted F3 (docs/PARTS_RESULTS.md section 7): purify only the target-identity subspace of
pi0.5's layer-14 KV.  POST-HOC follow-up: designed after pp_f3 showed that wholesale KV_14 purification destroys the grasp.
Written BEFORE the subspace fit, the open-loop read and any closed-loop rollout of this arm.

    python experiments/prereg/pp_f3t.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_f3t_roll", "pp_f3t_open.json", "pp_f3t_read.json", "pp_f3t_subspace.npz", "pp_f3t_subspace.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert any(r.get("tag") == "pp_f3" for r in ledger.load("m3_prereg")), "pp_f3 missing"
assert not any(r.get("tag") == "pp_f3t" for r in ledger.load("m3_prereg")), "pp_f3t already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 7\. Targeted F3 protocol.*?(?=\n## 8\.|\Z)", doc, flags=re.S)
assert m, "section 7 missing"
FILES = ["experiments/pp_f3t.py", "experiments/pp_f3t_read.py", "experiments/pp_f3t_roll.sh", "experiments/pp_serve_purify.py",
         "experiments/pp_purify.py", "experiments/pp_f1_mix.py", "results/flawed/pp_f3_branches.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_f3t", seed=0,
                    command="pre-registration: PLAN_PARTS targeted F3 (pi0.5 KV_14 target-identity subspace; POST-HOC follow-up of pp_f3)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS targeted F3 (docs/PARTS_RESULTS.md section 7); POST-HOC follow-up of pp_f3",
                             "section_7_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "estimator": "top-16 right singular vectors of within-object target contrasts (bowl K1-K2, K1-K3, K2-K3; bottle "
                                          "K4-K5) of layer-14 keys (de-rotated) and values at aligned valid positions, on the F1 frames",
                             "rank": 16,
                             "arms": "T = target subspace, source B (shares the target); R = random 16-dim subspace, source B (control); "
                                     "TA = target subspace, source A (open loop only)",
                             "closed_loop": "N1-N6, familiar (primary) and new, N2 states 4-13, N2 seeds, 10 episodes per task and layout; "
                                            "intact = pp_f3's intact re-run",
                             "verdict": "TIES iff dp_T > 0 with CI excluding 0, dg_T >= -0.10 (point), and dp_T - dp_R > 0 with CI excluding 0 "
                                        "(familiar layout; 2,000-draw paired bootstrap over tasks and states)",
                             "groot": "not run in this pass",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the subspace fit, the open-loop read and any closed-loop rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_f3t.py"))

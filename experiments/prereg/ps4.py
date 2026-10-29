"""PRE-REGISTRATION ROW of PLAN_SOLVE S4 (docs/PLAN_S4.md), written 2026-10-26 BEFORE the robust adapter is trained (asserted)."""
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OD = os.path.join(ROOT, "results", "bind", "s4")
assert not os.path.exists(os.path.join(OD, "adapter.pt")) and not os.path.exists(os.path.join(OD, "ckpt.pt"))
assert not os.path.exists(os.path.join(OD, "roll")) and not ledger.load("s4_robust")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps4", seed=0, command="pre-registration: PLAN_SOLVE S4",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S4 (noise-robust binding adapter; exploratory)",
                             "training": "S0 recipe; own-prompt masks per role: 0.35 oracle / 0.35 synthetic (one_in, one_out, shift, dilate, sibling, empty) / 0.30 S1b|S1c actual masks; cf keeps oracle",
                             "closed_loop": "seeds 4-5; dev 72 units (states 0-2): robust+S1c, robust+oracle; train 50: robust+S1c; zs and S0-oracle reused at seeds 4-5",
                             "bar": "B1 train damage (robust+S1c) >= -0.05; B2 dev binding (robust+S1c) < zs; B3 dev binding (robust+oracle) <= 0.10",
                             "files_sha256": sha(["docs/PLAN_S4.md", "experiments/s4_train.py", "experiments/s4_read.py", "experiments/s2_serve.py",
                                                  "experiments/s2_sandbox.py", "experiments/s4_bmask.py", "experiments/s4_phrases.py"]),
                             "written_before": "robust adapter training and any S4 rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps4.py"))

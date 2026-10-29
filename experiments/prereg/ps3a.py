"""PRE-REGISTRATION ROW of PLAN_SOLVE S3a (docs/PLAN_S3.md), written 2026-10-26 BEFORE the confidence pass runs (asserted)."""
import glob
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

assert not glob.glob(os.path.join(ROOT, "results", "bind", "s3", "*", "*", "*.npz"))
assert not os.path.exists(os.path.join(ROOT, "results", "bind", "s3a.json")) and not ledger.load("s3a_gate")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps3a", seed=0, command="pre-registration: PLAN_SOLVE S3a",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S3a (binder confidence gate; exploratory)",
                             "selection": "train cells only: scores {margin, det, joint}; admissible damage >= -0.02; max train coverage",
                             "closed_loop_if": "dev coverage >= 0.10 and gated-on dev obj masks mostly inside in >= 0.80; seeds 4-5",
                             "bar": "B1 train damage >= -0.05; B2 dev binding(gated) <= zs; closed loop if run, else offline (B1 in-sample)",
                             "files_sha256": sha(["docs/PLAN_S3.md", "experiments/s3_gate.py", "experiments/s3a_read.py",
                                                  "experiments/s2_serve.py", "experiments/s2_sandbox.py"]),
                             "written_before": "the confidence pass and any S3 rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps3a.py"))

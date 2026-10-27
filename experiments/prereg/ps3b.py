"""PRE-REGISTRATION ROW of PLAN_SOLVE S3b (docs/PLAN_S3.md), written 2026-10-26 BEFORE RoboPoint is run on any dev unit (asserted)."""
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

assert not os.path.exists(os.path.join(ROOT, "results", "bind", "s3b", "pred_dev.npz"))
assert not os.path.exists(os.path.join(ROOT, "results", "bind", "s3b.json")) and not ledger.load("s3b_grounder")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps3b", seed=0, command="pre-registration: PLAN_SOLVE S3b",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S3b (RoboPoint-13B point binder, offline mask quality; comparison only)",
                             "model": "wentao-yuan/robopoint-v1-vicuna-v1.5-13b, github wentaoyuan/RoboPoint @ 206cb1e, greedy, own answer format",
                             "query": "object: 'Locate several points on the {phrase}.'; goal: 'Locate several points {phrase}.'",
                             "map": "points -> S1 16x16 tokens; peak = most points; S1 hit metric on S1's 240 dev units",
                             "report": "RoboPoint vs S1b / S1c / B1a by depth x seen / held-out / held-out body-seen; no pass bar",
                             "files_sha256": sha(["docs/PLAN_S3.md", "experiments/s3b_robopoint.py", "experiments/s3b_read.py"]),
                             "written_before": "any RoboPoint dev prediction (smoke on 4 TRAINING units only)"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps3b.py"))

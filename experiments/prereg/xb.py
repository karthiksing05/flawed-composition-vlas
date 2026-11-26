"""PRE-REGISTRATION ROWS tag `xb` (experiments/prereg/xb_pred.txt, committed 3b1bca90): cross-benchmark failure atlas on released checkpoints.
One row per benchmark (mesa, bimesa, robocasa365, robotwin2).  Written BEFORE any xb rollout, cell draw or prediction.  Pattern copied from experiments/prereg/nphj.py.  sbatch (CPU) only."""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "xb" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "bench", "xb")), "xb results exist (cells/rollouts/predictions) - row must come first"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PRED = open(os.path.join(ROOT, "experiments", "prereg", "xb_pred.txt"), encoding="utf-8").read()
CODE = ["experiments/prereg/xb_pred.txt", "experiments/prereg/xb.py", "docs/notes/bench_audit.md", "docs/notes/bench_feasibility.md", "docs/PLAN_GOAL3.md",
        "experiments/bench_audit_mesa.py", "experiments/bench_audit_robocasa.py", "experiments/bench_audit_robotwin.py",
        "results/flawed/fp_dev/v1_frozen.json", "results/flawed/fp_dev/v2_frozen.json", "experiments/fp_fit.py", "experiments/fp_build.py"]
AUD = ["results/bench/mesa/audit/summary.json", "results/bench/mesa/audit/cells__mesa-composite__vs__mesa-70-lerobot.json", "results/bench/mesa/audit/cells__bimesa-composite__vs__bimesa-57-lerobot.json",
       "results/bench/mesa/audit/partner_counts__mesa-70-lerobot.json", "results/bench/mesa/audit/partner_counts__bimesa-57-lerobot.json",
       "results/bench/robocasa/audit/summary.json", "results/bench/robocasa/audit/ember-lab-berkeley__robocasa365-pretrain-atomic.json", "results/bench/robocasa/audit/pepijn223__robocasa_pretrain_human300_v4.json",
       "results/bench/robocasa/audit/object_registry.json",
       "results/bench/robotwin/audit/summary.json", "results/bench/robotwin/audit/place_a2b_left.json", "results/bench/robotwin/audit/place_a2b_right.json"]
SEEN = ("Seen before this row: the audit tables (held-out counts: MESA 17 clean tasks vs MESA-70, BiMESA 8; RoboCasa 566 zero cells in 17 atomic families; RoboTwin place_a2b 1,681 / 1,822 instance zero cells) and the "
        "smoke runs of docs/notes/bench_feasibility.md (MESA pi0.5 mushroom_tray 16/20, mushroom_bowl 14/20; RoboCasa pi0.5 CloseFridge 12/20, CuttingToolSelection 0/20; RoboTwin X-VLA place_container_plate 18/20, beat_block_hammer 19/20).  "
        "No xb cell draw, partner rollout, held-out rollout or prediction has been made.")
DEV = ("hard cap 300 GPU-h for all xb_* jobs, everything on overcap; harness code (experiments/xb_cells.py, xb_roll_*.py, xb_predict.py, xb_read.py) is written after this row and is hashed in docs/notes/xb_status.md; "
       "implementation choices not fixed by the file (operational definitions of lifted / held / on-receptacle for the failure classes, v0 cue-vote cues, receptacle sets used for the nearer-receptacle features) are logged there as deviations/clarifications")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
BENCH = {"mesa": "xbM MESA (4 models, mesa-70 data)", "bimesa": "xbB BiMESA (4 models, bimesa-57 data)", "robocasa365": "xbC RoboCasa365 (pi0, pi0.5, GR00T N1.5, X-VLA)", "robotwin2": "xbT RoboTwin 2.0 (X-VLA, pi0.5, SmolVLA)"}
for b, desc in BENCH.items():
    print(ledger.record("m3_prereg", bench=b, tag="xb", seed=0, command="pre-registration: XB cross-benchmark atlas, " + desc,
                        metrics={"status": "PRE-REGISTERED XB", "arm": desc, "prediction": PRED, "seen": SEEN, "deviation_notes": DEV,
                                 "files_sha256": {p: fsha(p) for p in CODE + AUD}, "prereg_commit": "3b1bca90", "written_before": "any xb cell draw, rollout or prediction"},
                        bytes=Z, config={}, script="experiments/prereg/xb.py"))

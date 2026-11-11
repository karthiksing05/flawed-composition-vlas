"""PRE-REGISTRATION ROW of PU-L (SCAFFOLD_RESULTS 40): U from poses estimated with learned masks (experiments/sc_seg.py pu) vs the
same pipeline on simulator masks (sc_pu percep, SR 38).  BEFORE any non-smoke learned rollout.

    python experiments/prereg/sc_seg_pu.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
od = os.path.join(ROOT, "results", "flawed", "sc_seg", "pu_learned")
assert not os.path.isdir(od) or not os.listdir(od), "non-smoke learned results exist"
assert not any(r.get("tag") == "sc_seg_pu" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SEG = "/coc/flash10/ksingara3/robot/data/fa1/seg/ckpt/unet_main.npz"
with open(SEG, "rb") as fh:
    seg_sha = hashlib.sha256(fh.read()).hexdigest()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_seg_pu", seed=0, command="pre-registration: PU-L (SCAFFOLD_RESULTS 40)",
                    metrics={"status": "PRE-REGISTERED PU-L",
                             "mapping": "experiments/sc_seg_pu_read.py. Four cells (known / new pairings x familiar / new layout; seeds 10-12; "
                                        "N2 main states). Primary per cell: learned - percep, bootstrapped over tasks and states (2,000 draws). COSTS if "
                                        "any cell's CI upper bound < -.10; HOLDS if every lower bound >= -.10; UNRESOLVED otherwise. learned - oracle "
                                        "and per-task rows reported with no verdict.",
                             "prediction": "HOLDS in the familiar layout (learned - percep within -.03; agentview movable mIoU .984 on held-out "
                                           "frames). New layout: UNRESOLVED more likely than HOLDS, with the cost on the plate-target tasks (N1, N4, "
                                           "K1; agentview plate detection .865 on new-layout frames) and the bottle (fallen bottles never in "
                                           "training); point estimates -.03 to -.08 there. Overall label: UNRESOLVED (55%), HOLDS (40%), COSTS (5%).",
                             "seen": "Segmenter val / gt IoU tables (results/flawed/sc_seg/iou_main.json: val movable mIoU .990, gt familiar .984, "
                                     "gt new .899; gt built from smoke states 0-3 with scripted reaches). Smoke: learned PU arm N4 familiar seed 10 "
                                     "states 0-1, 2/2 succeeded; R2/R2b learned-mask smokes on a 500-step format checkpoint (0/2 each, uninformative). "
                                     "SR 38.1's percep and oracle results on these states are known.",
                             "seg_ckpt": SEG, "seg_ckpt_sha256": seg_sha,
                             "files_sha256": {p: fsha(p) for p in ["experiments/sc_seg.py", "experiments/sc_seg_pu_read.py", "experiments/sc_pu.py",
                                                                   "results/flawed/sc_pu_calib.json", "results/flawed/sc_seg/iou_main.json"]},
                             "written_before": "any non-smoke learned PU rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_seg_pu.py"))

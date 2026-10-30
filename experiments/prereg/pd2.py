"""PRE-REGISTRATION ROW of D1x and D2 (docs/FLAWED_COMPOSITION.md section 4.7): the expert-read interchange test on main2 / C1 and
cross-VLA debt on pi0.5, GR00T N1.5 and X-VLA, with the KEY test (per-task debt predicts OOD failure beyond task novelty).
Written after the data-only frame extraction, the novelty controls and code smoke runs, BEFORE any D1x or D2 measurement
(asserted: no d1x_<m>.json, d2_<model>.json or d2_read.json exists).

    python experiments/prereg/pd2.py
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
RR = os.path.join(ROOT, "results", "robot")
assert not glob.glob(os.path.join(F, "d1x_*.json")) and not glob.glob(os.path.join(F, "d2_pi05*.json")) and \
    not glob.glob(os.path.join(F, "d2_xvla*.json")) and not glob.glob(os.path.join(F, "d2_groot*.json")) and \
    not os.path.exists(os.path.join(F, "d2_read.json")), "a D1x / D2 output exists"
assert not any(r.get("tag") == "pd2" for r in ledger.load("m3_prereg")), "pd2 already registered"
fr = json.load(open(os.path.join(F, "d2_frames.json"), encoding="utf-8"))


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def dist(f, keys=None):
    z = [v["zs"] for k, v in json.load(open(os.path.join(RR, f), encoding="utf-8"))["per_task"].items()
         if keys is None or k in keys]
    return dict(n=len(z), at_0=sum(x == 0 for x in z), at_20=sum(x == 20 for x in z), between=sum(0 < x < 20 for x in z),
                mean=sum(z) / (20 * len(z)))


l90keys = {t.split(":")[1] for t in fr["l90"]}
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pd2", seed=0, command="pre-registration: FLAWED_COMPOSITION D1x / D2",
                    metrics={"status": "PRE-REGISTERED D1x / D2 (FLAWED_COMPOSITION 4.7)",
                             "checkpoints": {"main2": "results/bindvla/main2/ckpt/step_060000.pt", "c1": "results/bindvla/c1main/ckpt/step_020000.pt",
                                             "pi05": "pi05_libero (openpi pytorch)", "groot": "ar0s/groot_libero", "xvla": "lerobot/xvla-libero"},
                             "success_distributions": {"pi05-L90": dist("g8_read.json", l90keys), "pi05-LI": dist("g15_read_pi05.json"),
                                                       "groot-LI": dist("g18_read_groot_libero.json"), "xvla-LI": dist("g18_read_xvla.json"),
                                                       "excluded_pi0-LI": dist("g15_read_pi0.json"), "excluded_smolvla-LI": dist("g18_read_smolvla.json")},
                             "l90_sealed_skipped": fr["l90_skipped_sealed"],
                             "predictions": {
                                 "D1x-a": "rho_in (relative excess error of the expert-read interchanged prefix over the true prefix, in-support "
                                          "compounds): holds if upper CI < 0.10; falsified if lower CI > 0.10; else inconclusive (main2 and C1)",
                                 "D1x-b": "delta = rho_dev - rho_in: present if CI excludes 0 above; absent if CI includes 0",
                                 "D2-KEY": "pooled partial Spearman of per-task debt (mean within-block rank of purity and bypass share) with "
                                           "failure 1 - zs/20, controlling for the D0-style score and u with block dummies, task-identity "
                                           "bootstrap 95% CI: holds if rho > 0 and CI excludes 0; falsified (debt adds nothing beyond novelty) "
                                           "otherwise"},
                             "secondary": "per block; purity alone; bypass alone; pi0.5-L90 combination error; uncontrolled; pool of pi05-LI + groot-LI; "
                                          "D1x agreement (IIA)",
                             "deviations_declared": "D2 interface = role-phrase prefix features only (no referent image tokens: no demos / referents "
                                                    "on the OOD tasks); bypass = all non-instruction inputs; no D2 insufficiency",
                             "smoke": "--limit runs of fc_d1x.py, fc_d2_debt.py (3 models) and fc_d2_read.py; only key names and role-span "
                                      "token counts inspected",
                             "files_sha256": {p: fsha(p) for p in ["docs/FLAWED_COMPOSITION.md", "experiments/fc_d1x.py", "experiments/fc_d2_frames.py",
                                                                   "experiments/fc_d2_support.py", "experiments/fc_d2_debt.py", "experiments/fc_d2_read.py",
                                                                   "experiments/fc_d1_debt.py", "results/flawed/d2_frames.json",
                                                                   "results/flawed/d2_support.json"]},
                             "written_before": "any D1x or D2 measurement"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pd2.py"))

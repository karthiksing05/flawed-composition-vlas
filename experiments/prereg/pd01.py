"""PRE-REGISTRATION ROW of D0 / D1 (docs/FLAWED_COMPOSITION.md sections 4.4 and 4.5): compositional debt per interface in main2,
C1 and vanilla, by interventions, with no training, and whether it predicts per-task dev failure.  Written after D0 (split only)
and the data-only dev-frame extraction, and BEFORE any model is scored on the D1 frames (asserted: no d1_<model>.json and no
d1_read.json exist).

    python experiments/prereg/pd01.py
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
CK = {"main2": "results/bindvla/main2/ckpt/step_060000.pt", "c1": "results/bindvla/c1main/ckpt/step_020000.pt",
      "vanilla": "results/bindvla/vanilla/ckpt/step_060000.pt"}
assert not [p for p in glob.glob(os.path.join(F, "d1_*.json")) if not p.endswith("d1_frames.json")], "a D1 model output exists"
assert not any(r.get("tag") == "pd01" for r in ledger.load("m3_prereg")), "pd01 already registered"
assert not ledger.load("fc_d1"), "a D1 result row exists"
d0 = json.load(open(os.path.join(F, "d0_support.json"), encoding="utf-8"))
fr = json.load(open(os.path.join(F, "d1_frames.json"), encoding="utf-8"))


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


print(ledger.record("m3_prereg", bench="libero_90_c", tag="pd01", seed=0, command="pre-registration: FLAWED_COMPOSITION D0 / D1",
                    metrics={"status": "PRE-REGISTERED D0 / D1 (FLAWED_COMPOSITION 4.4, 4.5)",
                             "checkpoints": {k: dict(ckpt=v, sha256=fsha(v)) for k, v in CK.items()},
                             "D0_predicted_order": d0["predicted_order"],
                             "D0_scores": {t: dict(rank=r["predicted_debt_rank"], score_all=r["score_all"], score_binder=r["score_binder"],
                                                   score_motor=r["score_motor"]) for t, r in d0["per_task"].items()},
                             "dev_frames": dict(n_frames=fr["n_frames"], n_demos=fr["n_demos"], pipeline_parity=fr["pipeline_parity"]),
                             "predictions": {
                                 "D1a": "main2: delta_F_rel (combination debt, relative excess phase-local action error of dev over in-support "
                                        "LIBERO-90 cells) > delta_P (purity debt at the binder output, tok). Holds if lower CI of delta_F_rel > "
                                        "upper CI of delta_P; falsified if delta_F_rel <= delta_P (point); else inconclusive",
                                 "D1b": "main2: Spearman rho(delta_F(d), 1 - bv_dev success(d)) over the 29 dev tasks > 0 with task-bootstrap 95% "
                                        "CI excluding 0; falsified if the CI includes 0 or rho < 0",
                                 "D1c": "paired over in-support tasks: dB = bypass share C1 - main2, dI = cut in-support error C1 - main2. Holds if "
                                        "CI(dB) < 0 and CI(dI) > 0; falsified if both fall or neither changes (both CIs include 0); else partial",
                                 "D0": "Spearman rho(D0 score_all, delta_F_main2(d)) over the 29 dev tasks > 0 with CI excluding 0; falsified "
                                       "if the CI includes 0 or rho < 0"},
                             "secondary": "C1 / vanilla correlations; named-contact failure (12.4 logs); D0 sub-scores and CD bin alone; "
                                          "grasp / release / flow / site variants of delta_F; kv-level purity; probes (descriptive)",
                             "weaknesses": "D1a compares two dimensionless ratios of different quantities; BindVLA's binder output is role-pure "
                                           "by construction (tok purity expected 0); dev failure is heavily tied (main2 3/29 tasks succeed); "
                                           "E_dev - E_in includes layout and demo difficulty",
                             "smoke": "a --limit smoke run of fc_d1_debt.py (60 IN and 60 DEV frames per model) and of fc_d1_read.py validated the code; its "
                                      "numbers were written to scratch and not inspected",
                             "files_sha256": {p: fsha(p) for p in ["docs/FLAWED_COMPOSITION.md", "experiments/fc_d0_support.py",
                                                                   "experiments/fc_d1_frames.py", "experiments/fc_d1_debt.py",
                                                                   "experiments/fc_d1_read.py", "lifelong_ood/policy/bindvla.py",
                                                                   "lifelong_ood/policy/backbone_bindvla.py", "experiments/bv_train.py",
                                                                   "results/flawed/d0_support.json", "results/flawed/d1_frames.json"]},
                             "written_before": "any model scored on the D1 frames"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pd01.py"))

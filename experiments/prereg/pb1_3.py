"""PRE-REGISTRATION ROW of PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13), written 2026-10-27 after smokes on generated TRAINING
layouts only (results/bind/b1_3/<arm>/smoke_train.json, b1b/train_log_smoke.json, vlm/smoke) and BEFORE any generated test unit
or S1 dev unit is predicted or scored (asserted).  Freezes the primary arm (the higher held-out-training-layout smoke hit) in
results/bind/b1_3/prereg_frozen.json."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

B13 = os.path.join(ROOT, "results", "bind", "b1_3")
assert not glob.glob(os.path.join(B13, "*", "pred_test*")) and not glob.glob(os.path.join(B13, "*", "pred_dev*"))
assert not glob.glob(os.path.join(B13, "vlm", "*.npz")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "b1_3.json"))
assert not ledger.load("b1_3_bind")
assert not any(r.get("tag") == "pb1_3" for r in ledger.load("m3_prereg"))


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


U = json.load(open(os.path.join(B13, "units.json"), encoding="utf-8"))
plan = json.load(open(os.path.join(B13, "plan.json"), encoding="utf-8"))
sel = {a: json.load(open(os.path.join(B13, a, "select.json"), encoding="utf-8")) for a in ("siglip", "dino")}
smoke = {a: json.load(open(os.path.join(B13, a, "smoke_train.json"), encoding="utf-8")) for a in ("siglip", "dino")}
primary = max(smoke, key=lambda a: (smoke[a]["mean"], a == "siglip"))
json.dump(dict(primary=primary, smoke={a: smoke[a]["mean"] for a in smoke}), open(os.path.join(B13, "prereg_frozen.json"), "w", encoding="utf-8"), indent=1)
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pb1_3", seed=0, command="pre-registration: PLAN_BINDER B1.3",
                    metrics={"status": "PRE-REGISTERED PLAN_BINDER B1.3 (prequential channel MDL + relative-only geometry; generated MCD unit set)",
                             "generator": dict(seed=plan["split"]["seed"], restart=plan["split"]["restart"], n_layouts=len(plan["layouts"]),
                                               layout_seeds="9400000 + layout index", held_pairs=plan["split"]["held_pairs"]),
                             "split": dict(test_compounds=sorted({u["compound"] for u in U["test"]}), stats=U["stats"]),
                             "arms": "factorised: B1.3 siglip and dino (primary frozen by the held-out-training-layout smoke: " + primary + "); "
                                     "holistic T1 comparator: B1b retrained on the generated pool (S1's recipe); S1c as trained in S1; "
                                     "GroundingDINO (B1a); Qwen2.5-VL-7B bbox and Molmo-7B-D point (B0's phrase prompts), scored under "
                                     "LIBERO's mirrored convention (labels) and the literal image-side one",
                             "frozen": {a: dict(chosen=sel[a]["chosen"], n_occ=sel[a]["n_occ"], n_layouts=sel[a]["n_layouts"],
                                                channels=sel[a]["channels"], lexicon_floats=sel[a]["lexicon_floats"],
                                                pca_floats=sel[a]["pca_floats"]) for a in sel},
                             "smoke_heldout_training_layouts": smoke, "primary": primary,
                             "primary_test": "OLS slope of unit binding error (1 - S1 hit) on the realised CD bin (0.1 / 0.3 / 0.5), factorised "
                                             "(primary arm) vs holistic (B1b-gen); 2000-draw compound-cluster bootstrap stratified by depth x bin, "
                                             "same draws for both; CONFIRMED iff the 95% CI of slope(holistic) - slope(factorised) lies above 0. "
                                             "Reported with it: both slopes' CIs, per depth, and x = unit CD",
                             "new_set_bars": "per arm: depth-1 hit >= 0.70 AND depth-2 hit >= 0.70 AND depth-0 hit >= GroundingDINO's on the same "
                                             "units; B1.3 passes iff the primary arm passes all three (compound-cluster two-level bootstrap CIs)",
                             "secondary": "S1's 240 dev units, the B1 bars (held-out d1 >= 0.70, d2 seen-body >= 0.70, d0 >= GroundingDINO 0.786), "
                                          "CD slope on the 11 recombination tasks; per-word channels; mirrored left / right prototypes; atom "
                                          "invariance; T2 budget; drawer / caddy units; parameter count",
                             "files_sha256": sha(["docs/PLAN_BINDER.md", "experiments/b1_3_gen.py", "experiments/b1_3_plan.py", "experiments/b1_3_render.py",
                                                  "experiments/b1_3_units.py", "experiments/b1_3_bind.py", "experiments/b1_3_baselines.py",
                                                  "experiments/b1_3_vlm.py", "experiments/b1_3_read.py", "results/bind/b1_3/units.json",
                                                  "results/bind/b1_3/plan.json"]),
                             "written_before": "any B1.3 test or dev prediction or score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pb1_3.py"))

"""PRE-REGISTRATION ROW of PLAN_BINDER B1.1 (docs/PLAN_BINDER.md section 9), written 2026-10-27 after smokes on TRAINING units only
(results/bind/b1/<arm>_b11/smoke_train.json) and BEFORE any dev unit is predicted or scored (asserted)."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

B1 = os.path.join(ROOT, "results", "bind", "b1")
assert not glob.glob(os.path.join(B1, "*_b11", "pred_dev.pkl")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "b1_1.json"))
assert not ledger.load("b1_1_bind")
assert not any(r.get("tag") == "pb1_1" for r in ledger.load("m3_prereg"))


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


posfree = {a: json.load(open(os.path.join(B1, a + "_b11_posfree.json"), encoding="utf-8"))["final_app"] for a in ("siglip", "dino")}
sel = {a: json.load(open(os.path.join(B1, a + "_b11", "select.json"), encoding="utf-8")) for a in ("siglip", "dino")}
smoke = {a: json.load(open(os.path.join(B1, a + "_b11", "smoke_train.json"), encoding="utf-8")) for a in ("siglip", "dino")}
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pb1_1", seed=0, command="pre-registration: PLAN_BINDER B1.1",
                    metrics={"status": "PRE-REGISTERED PLAN_BINDER B1.1 (position-free appearance by iterative nullspace projection, ordinal geometry, role expert, head composition retried)",
                             "arms": "primary: siglip (pi05_l90c@5000 image tokens); comparison: dino (DINOv2-small)",
                             "frozen": {a: dict(chosen=sel[a]["chosen"], n_occ=sel[a]["n_occ"], n_frames=sel[a]["n_frames"],
                                                n_content=sel[a]["n_content"], lexicon_floats=sel[a]["lexicon_floats"],
                                                pca_floats=sel[a]["pca_floats"]) for a in sel},
                             "variant": "V2: word evidence vs the role referent code, role expert ON, MDL on referent features (frozen by mean training-unit hit among 4 smoked variants; PLAN_BINDER 9)",
                             "smoke_train_units": smoke,
                             "posfree_probe_r2": posfree,
                             "bar": "per arm: held-out depth-1 hit >= 0.70 AND depth-2 seen-body hit >= 0.70 AND depth-0 hit >= GroundingDINO (0.786); B1 PASS iff siglip passes all three",
                             "also": "caddy; part-proposal recall vs sim masks; MDL channels; confounds; CD slope B1 vs B1b (task bootstrap); T2 atom budget (proposal, role, parent, words); atom invariance permutation test; referent purity",
                             "power": "(i) 3 tasks, 15 units; (ii) 4 tasks, 20 units; CD / invariance descriptive",
                             "files_sha256": sha(["docs/PLAN_BINDER.md", "experiments/b1_1_bind.py", "experiments/b1_1_read.py", "experiments/b1_demos.py",
                                                  "experiments/b1_render.py", "experiments/b1_acts.py", "experiments/b1_feats.py",
                                                  "results/bind/s1/labels.json"]),
                             "written_before": "any B1.1 dev prediction or score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pb1_1.py"))

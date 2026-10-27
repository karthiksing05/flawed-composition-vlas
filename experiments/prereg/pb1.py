"""PRE-REGISTRATION ROW of PLAN_BINDER B1 (docs/PLAN_BINDER.md section 7), written 2026-10-27 after smokes on TRAINING units only
(results/bind/b1/<arm>/smoke_train.json) and BEFORE any dev unit is predicted or scored (asserted)."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

B1 = os.path.join(ROOT, "results", "bind", "b1")
assert not glob.glob(os.path.join(B1, "*", "pred_dev.pkl")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "b1.json"))
assert not ledger.load("b1_bind")
assert not any(r.get("tag") == "pb1" for r in ledger.load("m3_prereg"))


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


sel = {a: json.load(open(os.path.join(B1, a, "select.json"), encoding="utf-8")) for a in ("siglip", "dino")}
smoke = {a: json.load(open(os.path.join(B1, a, "smoke_train.json"), encoding="utf-8")) for a in ("siglip", "dino")}
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pb1", seed=0, command="pre-registration: PLAN_BINDER B1",
                    metrics={"status": "PRE-REGISTERED PLAN_BINDER B1 (unsupervised prototype binder: contact-event referents, feature-cluster regions, MDL channels, product of experts)",
                             "arms": "primary: siglip (pi05_l90c@5000 image tokens); comparison: dino (DINOv2-small)",
                             "frozen": {a: dict(chosen=sel[a]["chosen"], n_occ=sel[a]["n_occ"], n_frames=sel[a]["n_frames"],
                                                n_content=sel[a]["n_content"], lexicon_floats=sel[a]["lexicon_floats"],
                                                pca_floats=sel[a]["pca_floats"]) for a in sel},
                             "variant": "A: word evidence vs candidate background, no role expert, MDL on referent features (frozen by mean training-unit hit among 5 smoked variants; PLAN_BINDER 7)",
                             "smoke_train_units": smoke,
                             "bar": "per arm: held-out depth-1 hit >= 0.70 AND depth-2 seen-body hit >= 0.70 AND depth-0 hit >= GroundingDINO (0.786); B1 PASS iff siglip passes all three",
                             "also": "caddy; part-proposal recall vs sim masks; MDL channels; confounds; CD slope B1 vs B1b (task bootstrap); T2 atom budget (proposal, role, parent, words); atom invariance permutation test; referent purity",
                             "power": "(i) 3 tasks, 15 units; (ii) 4 tasks, 20 units; CD / invariance descriptive",
                             "files_sha256": sha(["docs/PLAN_BINDER.md", "experiments/b1_bind.py", "experiments/b1_read.py", "experiments/b1_demos.py",
                                                  "experiments/b1_render.py", "experiments/b1_acts.py", "experiments/b1_feats.py",
                                                  "results/bind/s1/labels.json"]),
                             "written_before": "any B1 dev prediction or score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pb1.py"))

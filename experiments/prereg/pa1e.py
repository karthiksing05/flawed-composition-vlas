"""PRE-REGISTRATION ROW of prototype A, stage 1e (docs/FLAWED_COMPOSITION.md section 5.9): Ad without the target identity (Ae, 3 seeds) and
one seed-0 ablation, ORACLE state, N2's 16 familiar-layout tasks; H / R = stage 1b's Hb / Rb runs.  Criteria C1-C4 and verdict order of
section 5.6 with A = Ae (experiments/fa1e_read.py -> fa1x_read.py); the corrected three-factor split is descriptive.  Written after
training and the swap test (training states), BEFORE any stage-1e rollout (asserted).

    python experiments/prereg/pa1e.py
"""
import hashlib
import json
import os
import re
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
DATA = "/coc/flash10/ksingara3/robot/data/fa1"
RUNS = [("Ae", s) for s in (0, 1, 2)] + [("Ae_nosize", 0)]
assert not os.path.exists(os.path.join(F, "a1e_roll")), "stage-1e rollouts exist"
assert not os.path.exists(os.path.join(F, "a1e_read.json")), "a stage-1e read exists"
assert not any(r.get("tag") == "pa1e" for r in ledger.load("m3_prereg")), "pa1e already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1e_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1c_train_e_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.9 Stage 1e: one place term shared by every target.*?(?=\n\*\*Results|\n---\n|\n### 5\.10)", doc, flags=re.S)
assert m, "section 5.9's protocol is missing"
fam = [t for t in tasks["tasks"] if t["layout"] == "familiar"]
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1c_train.py", "experiments/fa1d_roll.py", "experiments/fa1e_roll.py", "experiments/fa1e_read.py",
         "experiments/fa1x_read.py", "experiments/fa1e_swap.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json", "results/flawed/a1e_swap.json",
         "results/flawed/a1d_read.json", "results/flawed/a1c_support.json"] + \
        [f"results/flawed/a1c_train_e_s{s}.json" for s in (0, 1, 2)] + sorted({t["bddl"] for t in fam}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in fam]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"e_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1e", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.9 prototype A stage 1e",
                    metrics={"status": "PRE-REGISTERED A1e (FLAWED_COMPOSITION 5.9, shared place term, oracle state, familiar layout)",
                             "section_5_9_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train")},
                             "change": "Ad's target term without the target identity one-hot: reads foot (canonical), last command, "
                                       "target size, drawer opening; one place function shared by every target",
                             "baselines": "H = Hb, R = Rb: stage 1b's runs (a1b_roll, 3 seeds each)",
                             "criteria": "C1-C4 and verdict order of section 5.6 (fa1b_read.py), A = Ae",
                             "descriptive": "three-factor split s3 = h_o a_o q_t with the arrival radius max(RAD[t], calibrated dxy_rest "
                                            "+ 0.03) per pair, for Ae and Ad; never changes the verdict",
                             "predictions": "pooled carry-distance support: N4 moves in, N1 ~0.01 out, N3 0.065 out, N2 / N5 / N6 in; N4 "
                                            "improves; cheese pairings stay bound to the cheese's own task ((Suff)), so two-factor C3 is "
                                            "expected to keep failing there; C1 at risk (release heights from size alone)",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1e rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1e.py"))

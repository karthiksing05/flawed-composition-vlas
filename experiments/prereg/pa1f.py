"""PRE-REGISTRATION ROW of prototype A, stage 1f (docs/FLAWED_COMPOSITION.md section 5.11): confirmatory run of Ae_nosize (3 fresh
seeds, tag f), ORACLE state, N2's 16 familiar-layout tasks; H / R = stage 1b's Hb / Rb runs.  Primary: C1-C4 of section 5.6 and the
three-factor law 3F, one verdict (experiments/fa1f_read.py).  Written after training and the swap test (training states), BEFORE any
stage-1f rollout (asserted).

    python experiments/prereg/pa1f.py
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
RUNS = [("Ae_nosize", s) for s in (0, 1, 2)]
assert not os.path.exists(os.path.join(F, "a1f_roll")), "stage-1f rollouts exist"
assert not os.path.exists(os.path.join(F, "a1f_read.json")), "a stage-1f read exists"
assert not any(r.get("tag") == "pa1f" for r in ledger.load("m3_prereg")), "pa1f already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1f_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1c_train_f_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T3", "T5", "T6")) and r["T4"]["max"] <= 1e-4, ("swap test fails", k)


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.11 Stage 1f: confirmatory run.*?(?=\n\*\*Results|\n---\n|\n### 5\.12)", doc, flags=re.S)
assert m, "section 5.11's protocol is missing"
fam = [t for t in tasks["tasks"] if t["layout"] == "familiar"]
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1c_train.py", "experiments/fa1d_roll.py", "experiments/fa1f_roll.py", "experiments/fa1e_read.py",
         "experiments/fa1f_read.py", "experiments/fa1x_read.py", "experiments/fa1e_swap.py", "lifelong_ood/policy/factored_flow.py",
         "results/flawed/n2_tasks.json", "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json",
         "results/flawed/a1f_swap.json", "results/flawed/a1e_read.json"] + \
        [f"results/flawed/a1c_train_f_s{s}.json" for s in (0, 1, 2)] + sorted({t["bddl"] for t in fam}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in fam]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"f_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1f", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.11 prototype A stage 1f",
                    metrics={"status": "PRE-REGISTERED A1f (FLAWED_COMPOSITION 5.11, Ae_nosize confirmatory, oracle state, familiar layout)",
                             "section_5_11_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train")},
                             "change": "none to the architecture: Ae_nosize of 5.9 (place term reads foot, last command, drawer "
                                       "opening only), three fresh seeds (tag f)",
                             "baselines": "H = Hb, R = Rb: stage 1b's runs (a1b_roll, 3 seeds each)",
                             "criteria": "C1-C4 of section 5.6 (A = Ae_nosize) and 3F: every N1-N6 CI of D3 = obs - h_o a_o q_t "
                                         "(corrected arrival radius, states resampled, seeds pooled) contains 0; N6 kept in",
                             "verdict_mapping": "floor -> uninformative; C1-C4 and 3F -> 'compositional; three-factor law holds'; "
                                                "C1-C4 only -> 'compositional by C1-C4; three-factor law fails'; else 'not "
                                                "compositional (fails: C1, C2, C3, C4, 3F as failing)'",
                             "predictions": "swap/grasp identity exact; K1 recovers, C1 holds; s(new) 0.5-0.6, C2 open; C3 fails "
                                            "somewhere; 3F holds N1-N5, at risk on N6; expected 'not compositional (fails: C3, 3F)'; "
                                            "C4 holds",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1f rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1f.py"))

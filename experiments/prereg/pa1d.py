"""PRE-REGISTRATION ROW of prototype A, stage 1d (docs/FLAWED_COMPOSITION.md section 5.8): Ac with the held-object identity bypasses cut
(Ad, 3 seeds) and two seed-0 ablations, ORACLE state, N2's 16 familiar-layout tasks; H / R = stage 1b's Hb / Rb runs.  Criteria C1-C4
and verdict order of section 5.6 with A = Ad (experiments/fa1d_read.py -> fa1x_read.py); the three-factor split is descriptive.
Written after the bypass measurement (training states), training and the swap test, BEFORE any stage-1d rollout (asserted).

    python experiments/prereg/pa1d.py
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
RUNS = [("Ad", s) for s in (0, 1, 2)] + [("Ad_tilt", 0), ("Ad_fing", 0)]
assert not os.path.exists(os.path.join(F, "a1d_roll")), "stage-1d rollouts exist"
assert not os.path.exists(os.path.join(F, "a1d_read.json")), "a stage-1d read exists"
assert not any(r.get("tag") == "pa1d" for r in ledger.load("m3_prereg")), "pa1d already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1d_swap.json"), encoding="utf-8"))
proxy = json.load(open(os.path.join(F, "a1d_proxy.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1c_train_d_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.8 Stage 1d: cut the held-object bypass.*?(?=\n\*\*Results|\n---\n|\n### 5\.9)", doc, flags=re.S)
assert m, "section 5.8's protocol is missing"
fam = [t for t in tasks["tasks"] if t["layout"] == "familiar"]
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1c_train.py", "experiments/fa1d_roll.py", "experiments/fa1d_read.py", "experiments/fa1x_read.py",
         "experiments/fa1d_swap.py", "experiments/fa1d_proxy.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json", "results/flawed/a1d_swap.json",
         "results/flawed/a1d_proxy.json", "results/flawed/a1b_read.json", "results/flawed/a1c_read.json"] + \
        [f"results/flawed/a1c_train_d_s{s}.json" for s in (0, 1, 2)] + sorted({t["bddl"] for t in fam}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in fam]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"d_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1d", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.8 prototype A stage 1d",
                    metrics={"status": "PRE-REGISTERED A1d (FLAWED_COMPOSITION 5.8, held-object bypass cut, oracle state, familiar layout)",
                             "section_5_8_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train")},
                             "change": "Ac's target term without the end-effector tilt and the finger positions (both decode the held "
                                       "object on training states: 0.82 / 0.81 vs chance 0.44); reads foot (canonical), last command, "
                                       "target identity / size / drawer",
                             "baselines": "H = Hb, R = Rb: stage 1b's runs (a1b_roll, 3 seeds each)",
                             "criteria": "C1-C4 and verdict order of section 5.6 (fa1b_read.py), A = Ad",
                             "descriptive": "three-factor split s3 = h_o a_o q_t (arrival = within 0.03 m of the target while held), "
                                            "D3 with a start-state CI; never changes the verdict",
                             "predictions": "bottle pairings N4-N6 improve if the bypass caused their failures (N4 still limited by an "
                                            "out-of-support carry distance); cheese pairings stay limited by the cheese's hold; C1 at "
                                            "risk (Proposition 2: relocation into insufficiency)",
                             "bypass_measurement": {g: v["identity_accuracy"] for g, v in proxy["groups"].items()},
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1d rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1d.py"))

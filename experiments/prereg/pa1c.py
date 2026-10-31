"""PRE-REGISTRATION ROW of prototype A, stage 1c (docs/FLAWED_COMPOSITION.md section 5.7): the canonical place frame (Ac, 3 seeds) and two
seed-0 ablations, on ORACLE state, N2's 16 familiar-layout tasks; H / R = stage 1b's Hb / Rb runs.  Criteria C1-C4 and verdict order of
section 5.6 with A = Ac (experiments/fa1c_read.py -> fa1x_read.py).  Written after training, the swap test (training states) and the
support prediction (from stage 1b's rollouts), BEFORE any stage-1c rollout (asserted).

    python experiments/prereg/pa1c.py
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
RUNS = [("Ac", s) for s in (0, 1, 2)] + [("Ac_nobear", 0), ("Ac_noft", 0)]
assert not os.path.exists(os.path.join(F, "a1c_roll")), "stage-1c rollouts exist"
assert not os.path.exists(os.path.join(F, "a1c_read.json")), "a stage-1c read exists"
assert not any(r.get("tag") == "pa1c" for r in ledger.load("m3_prereg")), "pa1c already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1c_swap.json"), encoding="utf-8"))
sup = json.load(open(os.path.join(F, "a1c_support.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1c_train_c_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.7 Stage 1c: a canonical place frame.*?(?=\n\*\*Results|\n---\n|\n### 5\.8)", doc, flags=re.S)
assert m, "section 5.7's protocol is missing"
fam = [t for t in tasks["tasks"] if t["layout"] == "familiar"]
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1c_train.py", "experiments/fa1c_roll.py", "experiments/fa1c_read.py", "experiments/fa1x_read.py",
         "experiments/fa1c_swap.py", "experiments/fa1c_support.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json", "results/flawed/a1c_swap.json",
         "results/flawed/a1c_support.json", "results/flawed/a1b_read.json"] + \
        [f"results/flawed/a1c_train_c_s{s}.json" for s in (0, 1, 2)] + sorted({t["bddl"] for t in fam}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in fam]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"c_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1c", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.7 prototype A stage 1c",
                    metrics={"status": "PRE-REGISTERED A1c (FLAWED_COMPOSITION 5.7, canonical place frame, oracle state, familiar layout)",
                             "section_5_7_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train")},
                             "change": "target term in the canonical place frame (origin target, x = bearing foot->target frozen at the "
                                       "hold's onset); reads the held object's foot, the end effector's tilt, fingers, last command, target "
                                       "identity / size / drawer; drops held size / yaw / offset, ee position / yaw, target yaw",
                             "baselines": "H = Hb, R = Rb: stage 1b's runs (a1b_roll, 3 seeds each), not retrained",
                             "criteria": "C1-C4 and verdict order of section 5.6 (fa1b_read.py), A = Ac; engine fa1x_read.py reproduces "
                                         "the registered 1b read exactly with A = Ab",
                             "predictions": "(Supp) holds for N2, N5, N6, nearly N1, not N3 / N4 (support table, a1c_support.json); C3 "
                                            "can hold at most on N1, N2, N5, N6 and only if (Suff) holds there; C2 open",
                             "support_prediction": sup["pairs"],
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1c rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1c.py"))

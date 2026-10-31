"""PRE-REGISTRATION ROW of prototype A, stage 1g (docs/FLAWED_COMPOSITION.md section 5.12): every arm trained on the demos plus the
GENERATED varied-layout carries of the known pick-and-place pairings (scripted oracle-state planner, fa1g_plan.py); Ag (Ae_nosize
architecture), Hg, Rg (Hb, Rb architectures), 3 seeds each; ORACLE state; N2's 32 tasks (familiar and new layout).  Primary: per
layout C1-C4 and 3F, one verdict (experiments/fa1g_read.py).  Written after training and the swap test, BEFORE any stage-1g rollout.

    python experiments/prereg/pa1g.py
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
ARMS = {"Ag": "Ae_nosize", "Hg": "Hb", "Rg": "Rb"}
assert not os.path.exists(os.path.join(F, "a1g_roll")), "stage-1g rollouts exist"
assert not os.path.exists(os.path.join(F, "a1g_read.json")), "a stage-1g read exists"
assert not any(r.get("tag") == "pa1g" for r in ledger.load("m3_prereg")), "pa1g already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1g_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1g_train_g_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}
gen = json.load(open(os.path.join(F, "a1g_gen.json"), encoding="utf-8"))
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T3", "T5", "T6")) and r["T4"]["max"] <= 1e-4, ("swap test fails", k)


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.12 Stage 1g: support by generated.*?(?=\n\*\*Results|\n---\n|\n### 5\.13)", doc, flags=re.S)
assert m, "section 5.12's protocol is missing"
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1g_plan.py", "experiments/fa1g_train.py", "experiments/fa1d_roll.py", "experiments/fa1g_roll.py",
         "experiments/fa1g_read.py", "experiments/fa1f_read.py", "experiments/fa1x_read.py", "experiments/fa1e_swap.py",
         "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json", "results/flawed/n2_branches_main.json",
         "results/flawed/a1_calib.json", "results/flawed/a1g_swap.json", "results/flawed/a1g_keypoints.json",
         "results/flawed/a1g_gen.json"] + [f"results/flawed/a1g_gen_K{i}.json" for i in range(1, 7)] + \
        [f"results/flawed/a1g_train_g_s{s}.json" for s in (0, 1, 2)] + sorted({t["bddl"] for t in tasks["tasks"]}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {f"{n}_s{s}": os.path.join(DATA, "ckpt", f"g_s{s}", f"{a}.npz") for n, a in ARMS.items() for s in (0, 1, 2)}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1g", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.12 prototype A stage 1g",
                    metrics={"status": "PRE-REGISTERED A1g (FLAWED_COMPOSITION 5.12, generated varied-layout carries, oracle state, "
                                       "familiar and new layout)",
                             "section_5_12_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arms": ARMS, "runs": {f"{n}_s{s}": dict(params=train[s]["arms"][a]["params"]) for n, a in ARMS.items()
                                                    for s in (0, 1, 2)},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train", "n_generated")},
                             "generated": gen,
                             "change": "data: 50 generated successful carries per known pick-and-place task (K1-K6), scripted "
                                       "oracle-state planner on demo-fitted keypoints, varied layouts (seeds 900000+), no demo "
                                       "action executed; all arms (A, H, R) trained on demos + generated",
                             "criteria": "per layout: C1-C4 (A = Ag, H = Hg, R = Rg) and 3F (every N1-N6 D3 CI contains 0); N6 in",
                             "verdict_mapping": "per layout as 5.11; verdict = 'familiar: <v> | new layout: <v>'; guarantee iff both "
                                                "'compositional; three-factor law holds'",
                             "predictions": "role locality / grasp identity exact; familiar N3, N4 improve, C3 holds; C2 open; 3F "
                                            "cheese bias unless K6 improves; new layout: H/R floor holds, C1 holds, C4 holds; "
                                            "expected not compositional on both layouts (C2, possibly 3F)",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz", "gen_g.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1g rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1g.py"))

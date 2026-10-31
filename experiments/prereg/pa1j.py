"""PRE-REGISTRATION ROW of prototype A, stage 1j (docs/FLAWED_COMPOSITION.md section 5.17): the confirmatory read of Ai, UNCHANGED, on
fresh seeds 3-5 (tag j); ORACLE state; N2's 32 tasks.  Binding: per layout C1 (familiar vs 0.672; new = layout invariance), C2, 3F,
C4; C3 descriptive (user-approved after the 1i read).  Written after training and the swap test, BEFORE any stage-1j rollout.
All hashes are computed here, on the compute node.

    python experiments/prereg/pa1j.py
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
RUNS = [("Ai", s) for s in (3, 4, 5)]
assert not os.path.exists(os.path.join(F, "a1j_roll")), "stage-1j rollouts exist"
assert not os.path.exists(os.path.join(F, "a1j_read.json")), "a stage-1j read exists"
assert not any(r.get("tag") == "pa1j" for r in ledger.load("m3_prereg")), "pa1j already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1j_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1g_train_j_s{s}.json"), encoding="utf-8")) for s in (3, 4, 5)}
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T3", "T7")) and r["T4"]["max"] <= 1e-4 and r["T5p"]["max"] <= 1e-4, \
        ("swap test fails", k)


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


assert fsha("results/flawed/a1b_read.json") == "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760", "reference changed"
doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.17 Stage 1j: confirmatory read.*?(?=\n\*\*Results|\n---\n|\n### 5\.18)", doc, flags=re.S)
assert m, "section 5.17's protocol is missing"
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1g_train.py", "experiments/fa1h_roll.py", "experiments/fa1j_read.py", "experiments/fa1g_read.py",
         "experiments/fa1x_read.py", "experiments/fa1h_swap.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json", "results/flawed/a1j_swap.json",
         "results/flawed/a1b_read.json", "results/flawed/a1i_read.json"] + \
        [f"results/flawed/a1g_train_j_s{s}.json" for s in (3, 4, 5)] + sorted({t["bddl"] for t in tasks["tasks"]}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"j_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1j", seed=3,
                    command="pre-registration: FLAWED_COMPOSITION 5.17 prototype A stage 1j (confirmatory)",
                    metrics={"status": "PRE-REGISTERED A1j (FLAWED_COMPOSITION 5.17, confirmatory read of Ai on fresh seeds 3-5)",
                             "section_5_17_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[3][k] for k in ("steps", "batch", "lr", "horizon", "n_train", "n_generated")},
                             "criteria": "familiar: C1 (s_known - 0.672 >= -0.10), C2, 3F, C4; new: C1 = layout invariance (>= 0.8, "
                                         "CI lower >= 0.6), C2, 3F, C4; C3 descriptive; N6 in",
                             "verdict_mapping": "per layout: floor -> uninformative; C1, C2, 3F, C4 -> 'compositional'; else 'not "
                                                "compositional (fails: ...)'; verdict = 'familiar: <v> | new layout: <v>'; CONFIRMED "
                                                "iff both 'compositional'",
                             "predictions": "familiar: C1, C4 hold, C2 point holds (lower bound ~0.6 either side); 3F N3 D3 -0.05 "
                                            "+/- 0.05 (CI contains 0 ~half the time), others in CI; new: L 0.79 +/- 0.06 (C1 coin "
                                            "flip), C2 fails, 3F at risk N1/N6, C4 holds; confirmation in both layouts not expected",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz", "gen_g.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1j rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1j.py"))

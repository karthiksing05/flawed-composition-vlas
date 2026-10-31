"""PRE-REGISTRATION ROW of prototype A, stage 1h (docs/FLAWED_COMPOSITION.md section 5.14): Ah_rot (Ae_nosize + the end effector's
rotation since the hold's onset in the target term; 3 seeds) and the ablation Ah (+ grasp offset; seed 0), trained on demos + 1g's
generated carries; ORACLE state; N2's 32 tasks.  The user-approved criterion change (floor / C1 reference = stage 1b's demo-only Hb,
0.672) applies.  Written after training, the swap test and the identity probe (training rows), BEFORE any stage-1h rollout.

    python experiments/prereg/pa1h.py
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
RUNS = [("Ah_rot", s) for s in (0, 1, 2)] + [("Ah", 0)]
assert not os.path.exists(os.path.join(F, "a1h_roll")), "stage-1h rollouts exist"
assert not os.path.exists(os.path.join(F, "a1h_read.json")), "a stage-1h read exists"
assert not any(r.get("tag") == "pa1h" for r in ledger.load("m3_prereg")), "pa1h already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1h_swap.json"), encoding="utf-8"))
probe = json.load(open(os.path.join(F, "a1h_probe.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1g_train_h_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T3", "T6", "T7")) and r["T4"]["max"] <= 1e-4 and r["T5p"]["max"] <= 1e-4, \
        ("swap test fails", k)
assert probe["I1"] and probe["I1_rot"] and probe["I2"], "identity check fails"


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.14 Stage 1h: a role-pure route.*?(?=\n\*\*Results|\n---\n|\n### 5\.15)", doc, flags=re.S)
assert m, "section 5.14's protocol is missing"
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1g_train.py", "experiments/fa1h_roll.py", "experiments/fa1h_read.py", "experiments/fa1g_read.py",
         "experiments/fa1x_read.py", "experiments/fa1h_swap.py", "experiments/fa1h_probe.py", "lifelong_ood/policy/factored_flow.py",
         "results/flawed/n2_tasks.json", "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json",
         "results/flawed/a1h_swap.json", "results/flawed/a1h_probe.json", "results/flawed/a1b_read.json", "results/flawed/a1g_read.json"] + \
        [f"results/flawed/a1g_train_h_s{s}.json" for s in (0, 1, 2)] + sorted({t["bddl"] for t in tasks["tasks"]}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"h_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1h", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.14 prototype A stage 1h",
                    metrics={"status": "PRE-REGISTERED A1h (FLAWED_COMPOSITION 5.14, role-pure rotation-since-onset route, oracle state, "
                                       "familiar and new layout)",
                             "section_5_14_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train", "n_generated")},
                             "criterion_change": "user-approved after the 1g read: floor and C1 reference = stage 1b demo-only Hb "
                                                 "familiar s_known 0.672 (results/flawed/a1b_read.json sha256 74fcd712..., commit "
                                                 "3ae1e9fc), fixed, both layouts; C4 vs Hg and Rg (1g); 1g's verdict unchanged",
                             "criteria": "per layout: floor, C1 (vs 0.672), C2, C3, C4 (vs Hg, Rg), 3F; N6 in",
                             "verdict_mapping": "per layout as 5.11 (floor/C1 per the change); verdict = 'familiar: <v> | new layout: "
                                                "<v>'; guarantee iff both 'compositional; three-factor law holds'",
                             "identity_check": probe,
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz", "gen_g.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1h rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1h.py"))

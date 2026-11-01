"""PRE-REGISTRATION ROW of prototype A, stage 1k (docs/FLAWED_COMPOSITION.md section 5.19): Ai, architecture unchanged, trained on the
demos plus EQUAL per-object generated support (gen_k.npz), fresh seeds 6-8 (tag k); ORACLE state; N2's 32 tasks.  FROZEN criteria
(user, 2026-10-31): per layout C1 (familiar vs 0.672; new = layout invariance), C2, 3F, C4 vs Hg / Rg; C3 descriptive.  Written after
training and the swap test, BEFORE any stage-1k rollout.  All hashes are computed here, on the compute node.

    python experiments/prereg/pa1k.py
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
RUNS = [("Ai", s) for s in (6, 7, 8)]
assert not os.path.exists(os.path.join(F, "a1k_roll")), "stage-1k rollouts exist"
assert not os.path.exists(os.path.join(F, "a1k_read.json")), "a stage-1k read exists"
assert not any(r.get("tag") == "pa1k" for r in ledger.load("m3_prereg")), "pa1k already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1k_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1g_train_k_s{s}.json"), encoding="utf-8")) for s in (6, 7, 8)}
assert all(t["gen_file"] == "gen_k.npz" for t in train.values()), "stage-1k models must be trained on gen_k.npz"
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T3", "T7")) and r["T4"]["max"] <= 1e-4 and r["T5p"]["max"] <= 1e-4, \
        ("swap test fails", k)
gen = json.load(open(os.path.join(F, "a1k_gen.json"), encoding="utf-8"))
eps = {o: v["episodes"] for o, v in gen["per_object"].items()}
assert eps.get("akita_black_bowl_1") == eps.get("wine_bottle_1") == eps.get("cream_cheese_1") == 200, ("unequal support", eps)


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


assert fsha("results/flawed/a1b_read.json") == "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760", "reference changed"
doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.19 Stage 1k: equal object-side support.*?(?=\n\*\*Results|\n---\n|\n### 5\.20)", doc, flags=re.S)
assert m, "section 5.19's protocol is missing"
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1g_train.py", "experiments/fa1g_plan.py", "experiments/fa1k_plan.py", "experiments/fa1h_roll.py",
         "experiments/fa1g_roll.py", "experiments/fa1k_read.py", "experiments/fa1j_read.py", "experiments/fa1g_read.py",
         "experiments/fa1x_read.py", "experiments/fa1h_swap.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json", "results/flawed/a1k_swap.json",
         "results/flawed/a1b_read.json", "results/flawed/a1j_read.json", "results/flawed/a1k_gen.json", "results/flawed/a1k_diag.json",
         "results/flawed/a1g_keypoints.json"] + \
        [f"results/flawed/a1g_train_k_s{s}.json" for s in (6, 7, 8)] + sorted({t["bddl"] for t in tasks["tasks"]}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"k_s{s}", f"{a}.npz") for a, s in RUNS}
CKPT.update({"Hk_s6": os.path.join(DATA, "ckpt", "k_s6", "Hb.npz"), "Rk_s6": os.path.join(DATA, "ckpt", "k_s6", "Rb.npz")})
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1k", seed=6,
                    command="pre-registration: FLAWED_COMPOSITION 5.19 prototype A stage 1k (equal object-side support)",
                    metrics={"status": "PRE-REGISTERED A1k (FLAWED_COMPOSITION 5.19, Ai on demos + gen_k, fresh seeds 6-8)",
                             "section_5_19_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[6][k] for k in ("steps", "batch", "lr", "horizon", "n_train", "n_generated", "gen_file")},
                             "generated_per_object": eps,
                             "criteria": "FROZEN: familiar: C1 (s_known - 0.672 >= -0.10), C2, 3F, C4 vs Hg/Rg; new: C1 = layout "
                                         "invariance (>= 0.8, CI lower >= 0.6), C2, 3F, C4; C3 descriptive; N6 in; Hk/Rk and the "
                                         "object-matched C2 descriptive only (post hoc)",
                             "verdict_mapping": "per layout: floor -> uninformative; C1, C2, 3F, C4 -> 'compositional'; else 'not "
                                                "compositional (fails: ...)'; verdict = 'familiar: <v> | new layout: <v>'; CONFIRMED "
                                                "iff both 'compositional'",
                             "predictions": "see section 5.19 (hashed above)",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz", "gen_g.npz", "gen_k.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1k rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1k.py"))

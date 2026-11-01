"""PRE-REGISTRATION ROW of prototype A, stage 1l (docs/FLAWED_COMPOSITION.md section 5.21): Ai, architecture unchanged, trained on the
demos plus gen_l.npz (role-wise acceptance: 1k's carries plus the grasp segments of every failed generated attempt), fresh seeds
9-11 (tag l); ORACLE state; N2's 32 tasks.  FROZEN criteria (user, 2026-10-31).  Written after training and the swap test, BEFORE
any stage-1l rollout.  All hashes are computed here, on the compute node.

    python experiments/prereg/pa1l.py
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
SEEDS = (9, 10, 11)
RUNS = [("Ai", s) for s in SEEDS]
assert not os.path.exists(os.path.join(F, "a1l_roll")), "stage-1l rollouts exist"
assert not os.path.exists(os.path.join(F, "a1l_read.json")), "a stage-1l read exists"
assert not any(r.get("tag") == "pa1l" for r in ledger.load("m3_prereg")), "pa1l already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1l_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1g_train_l_s{s}.json"), encoding="utf-8")) for s in SEEDS}
assert all(t["gen_file"] == "gen_l.npz" for t in train.values()), "stage-1l models must be trained on gen_l.npz"
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T3", "T7")) and r["T4"]["max"] <= 1e-4 and r["T5p"]["max"] <= 1e-4, \
        ("swap test fails", k)
gen = json.load(open(os.path.join(F, "a1l_gen.json"), encoding="utf-8"))
segs = {o: v["segments"] for o, v in gen["per_object"].items()}
assert all(segs.get(o, 0) > 0 for o in ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")), ("segments missing", segs)


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


assert fsha("results/flawed/a1b_read.json") == "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760", "reference changed"
doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.21 Stage 1l: role-wise acceptance.*?(?=\n\*\*Results|\n---\n|\n### 5\.22)", doc, flags=re.S)
assert m, "section 5.21's protocol is missing"
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1g_train.py", "experiments/fa1g_plan.py", "experiments/fa1l_plan.py", "experiments/fa1h_roll.py",
         "experiments/fa1g_roll.py", "experiments/fa1l_read.py", "experiments/fa1k_read.py", "experiments/fa1j_read.py",
         "experiments/fa1g_read.py", "experiments/fa1x_read.py", "experiments/fa1h_swap.py", "experiments/fa1l_diag.py",
         "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json", "results/flawed/n2_branches_main.json",
         "results/flawed/a1_calib.json", "results/flawed/a1l_swap.json", "results/flawed/a1b_read.json", "results/flawed/a1k_read.json",
         "results/flawed/a1l_gen.json", "results/flawed/a1l_diag.json", "results/flawed/a1g_keypoints.json"] + \
        [f"results/flawed/a1l_gen_K{i}.json" for i in range(1, 7)] + \
        [f"results/flawed/a1g_train_l_s{s}.json" for s in SEEDS] + sorted({t["bddl"] for t in tasks["tasks"]}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"l_s{s}", f"{a}.npz") for a, s in RUNS}
CKPT.update({"Hl_s9": os.path.join(DATA, "ckpt", "l_s9", "Hb.npz"), "Rl_s9": os.path.join(DATA, "ckpt", "l_s9", "Rb.npz")})
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1l", seed=9,
                    command="pre-registration: FLAWED_COMPOSITION 5.21 prototype A stage 1l (role-wise acceptance)",
                    metrics={"status": "PRE-REGISTERED A1l (FLAWED_COMPOSITION 5.21, Ai on demos + gen_l, fresh seeds 9-11)",
                             "section_5_21_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[9][k] for k in ("steps", "batch", "lr", "horizon", "n_train", "n_generated", "gen_file")},
                             "grasp_segments_per_object": segs,
                             "criteria": "FROZEN: familiar: C1 (s_known - 0.672 >= -0.10), C2, 3F, C4 vs Hg/Rg; new: C1 = layout "
                                         "invariance (>= 0.8, CI lower >= 0.6), C2, 3F, C4; C3 descriptive; N6 in; Hl/Rl and the "
                                         "object-matched C2 descriptive only (post hoc)",
                             "verdict_mapping": "per layout: floor -> uninformative; C1, C2, 3F, C4 -> 'compositional'; else 'not "
                                                "compositional (fails: ...)'; verdict = 'familiar: <v> | new layout: <v>'; CONFIRMED "
                                                "iff both 'compositional'",
                             "predictions": "see section 5.21 (hashed above)",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz", "gen_k.npz", "gen_l.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1l rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1l.py"))

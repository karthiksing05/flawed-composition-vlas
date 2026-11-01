"""PRE-REGISTRATION ROW of prototype A, stage 1p (docs/FLAWED_COMPOSITION.md section 5.30): arm Aq on the demos plus gen_lp.npz
(gen_lo.npz plus generated carries stratified by the carried object's clearance to the nearest fixture), fresh seeds 24-26 (tag p).
FROZEN criteria (user, 2026-10-31).  Written after training, the swap test, the sensitivity check and the identity probe, BEFORE any
stage-1p rollout.  Hashes computed here.

    python experiments/prereg/pa1p.py
"""
import hashlib
import json
import os
import re
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
import numpy as np
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
DATA = "/coc/flash10/ksingara3/robot/data/fa1"
SEEDS = (24, 25, 26)
assert not os.path.exists(os.path.join(F, "a1p_roll")), "stage-1p rollouts exist"
assert not os.path.exists(os.path.join(F, "a1p_read.json")), "a stage-1p read exists"
assert not any(r.get("tag") == "pa1p" for r in ledger.load("m3_prereg")), "pa1p already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1p_swap.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1g_train_p_s{s}.json"), encoding="utf-8")) for s in SEEDS}
assert all(t["gen_file"] == "gen_lp.npz" and "Aq" in t["arms"] for t in train.values()), "models: Aq on gen_lp.npz"
for k, r in swap["tests"].items():
    assert all(r[t]["max"] == 0.0 for t in ("T1", "T2", "T7", "T3_place")) and r["T4"]["max"] <= 1e-4 and r["T5p"]["max"] <= 1e-4, \
        ("swap test fails", k)
sens = json.load(open(os.path.join(F, "a1p_sens.json"), encoding="utf-8"))
for s in SEEDS:
    assert sens["models"][f"p_s{s}/Aq"]["beyond_radius_max"] == 0.0, ("Aq reacts to entities beyond OCC_R", s)
idp = json.load(open(os.path.join(F, "a1p_idprobe.json"), encoding="utf-8"))
assert idp["arm"] == "Ap" and idp["occ_minus_base_test"] <= 0.05, "identity probe"
sg = json.load(open(os.path.join(F, "a1p_sgen.json"), encoding="utf-8"))
G0, G1 = np.load(os.path.join(DATA, "gen_lo.npz")), np.load(os.path.join(DATA, "gen_lp.npz"))
n0 = len(G0["task"])
assert sg["base_rows"] == n0 and all(np.array_equal(G0[x], G1[x][:n0]) for x in ("task", "demo", "step", "action", "held", "pos")), \
    "gen_lp must start with gen_lo unchanged"


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


assert fsha("results/flawed/a1b_read.json") == "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760", "reference changed"
doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.30 Stage 1p: generated carries stratified by clearance.*?(?=\n\*\*Results|\n---\n|\n### 5\.31)", doc, flags=re.S)
assert m, "section 5.30's protocol is missing"
FILES = ["experiments/fa1_common.py", "experiments/fa1_read.py", "experiments/fa1b_common.py", "experiments/fa1c_common.py",
         "experiments/fa1g_train.py", "experiments/fa1g_plan.py", "experiments/fa1l_cgen.py", "experiments/fa1o_merge.py",
         "experiments/fa1p_sgen.py", "experiments/fa1h_roll.py", "experiments/fa1p_read.py", "experiments/fa1o_read.py",
         "experiments/fa1n_read.py", "experiments/fa1k_read.py", "experiments/fa1j_read.py", "experiments/fa1g_read.py",
         "experiments/fa1x_read.py", "experiments/fa1h_swap.py", "experiments/fa1m_sens.py", "experiments/fa1o_cover.py",
         "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json", "results/flawed/n2_branches_main.json",
         "results/flawed/a1_calib.json", "results/flawed/a1p_swap.json", "results/flawed/a1p_sens.json",
         "results/flawed/a1p_idprobe.json", "results/flawed/a1p_sgen.json", "results/flawed/a1o_cover.json",
         "results/flawed/a1o_gfail.json", "results/flawed/a1o_geom.json", "results/flawed/a1b_read.json",
         "results/flawed/a1o_read.json"] + \
        [f"results/flawed/a1g_train_p_s{s}.json" for s in SEEDS] + sorted({t["bddl"] for t in tasks["tasks"]}) + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {f"Ap1_s{s}": os.path.join(DATA, "ckpt", f"p_s{s}", "Aq.npz") for s in SEEDS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1p", seed=24,
                    command="pre-registration: FLAWED_COMPOSITION 5.30 prototype A stage 1p (clearance-stratified generated carries)",
                    metrics={"status": "PRE-REGISTERED A1p (FLAWED_COMPOSITION 5.30, Aq on demos + gen_lp, fresh seeds 24-26)",
                             "section_5_30_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "runs": {f"Ap1_s{s}": dict(params=train[s]["arms"]["Aq"]["params"]) for s in SEEDS},
                             "training": {k: train[24][k] for k in ("steps", "batch", "lr", "horizon", "n_train", "n_generated", "gen_file")},
                             "sensitivity": sens["models"], "identity_probe_scene_part": idp, "stratified_generation": sg,
                             "criteria": "FROZEN: familiar: C1 (s_known - 0.672 >= -0.10), C2, 3F, C4 vs Hg/Rg; new: C1 = layout "
                                         "invariance (>= 0.8, CI lower >= 0.6), C2, 3F, C4; C3 descriptive; N6 in; Ao1/An/Ak, the "
                                         "near-rack split and the object-matched C2 descriptive only (post hoc)",
                             "verdict_mapping": "per layout: floor -> uninformative; C1, C2, 3F, C4 -> 'compositional'; else 'not "
                                                "compositional (fails: ...)'; verdict = 'familiar: <v> | new layout: <v>'; CONFIRMED "
                                                "iff both 'compositional'",
                             "predictions": "see section 5.30 (hashed above)",
                             "swap_test_before_rollouts": swap["tests"],
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz", "gen_lo.npz", "gen_lp.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1p rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1p.py"))

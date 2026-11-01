"""PLAN_PARTS helper: print the structure (keys, shapes, a few fields) of the inputs Layer 1 reads.  No outcome is summarised.

    python experiments/pp_inspect.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, "results", "flawed")
DATA = "/coc/flash10/ksingara3/robot/data/fa1"

n2 = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
fn = "n2_goal_0.npz"
z = np.load(os.path.join(F, "n2_render", fn), allow_pickle=True)
print(fn, {k: (z[k].shape, z[k].dtype) for k in z.files})
print("names", z["names"] if "names" in z.files else None)
print("n2_states:", sorted(os.listdir(os.path.join(F, "n2_states")))[:5], np.load(os.path.join(F, "n2_states", "n2_goal_0.npy")).shape)
print("n2_roll:", sorted(os.listdir(os.path.join(F, "n2_roll")))[:6], len(os.listdir(os.path.join(F, "n2_roll"))))
r = json.load(open(os.path.join(F, "n2_roll", "pi05_main_0.json"), encoding="utf-8"))
print("roll keys", list(r), {k: list(v) for k, v in r["tasks"].items()})
ep = list(r["tasks"].values())[0]["episodes"][0]
print("episode keys", {k: (v if not isinstance(v, (list, dict)) else type(v).__name__ + str(len(v))) for k, v in ep.items()})
br = json.load(open(os.path.join(F, "n2_branches_main.json"), encoding="utf-8"))
print("branches", type(br).__name__, (list(br)[:5] if isinstance(br, dict) else br[:2]))
print("fa1:", sorted(os.listdir(DATA)))
print("ckpt:", sorted(os.listdir(os.path.join(DATA, "ckpt"))))
for d in ("g_s0", "p_s24"):
    print(d, sorted(os.listdir(os.path.join(DATA, "ckpt", d))))
D = np.load(os.path.join(DATA, "demos.npz"))
print("demos", {k: D[k].shape for k in D.files})
Db = np.load(os.path.join(DATA, "demos_b.npz"))
print("demos_b", {k: Db[k].shape for k in Db.files})
R = np.load(os.path.join(DATA, "raw_K1.npz"))
print("raw_K1", {k: R[k].shape for k in R.files})
print("a1p_roll:", sorted(os.listdir(os.path.join(F, "a1p_roll")))[:10])

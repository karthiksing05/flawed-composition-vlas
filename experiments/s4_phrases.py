"""PLAN_SOLVE S4: role phrases (the parse's split, as s2_plan.py) for every S0 training cell -> results/bind/s4/phrases.json.  Chunks env, CPU."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import s1_labels as L
tasks = json.load(open(os.path.join(ROOT, "results/bind/s1/tasks.json"), encoding="utf-8"))
out = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results/bind/s0/data/*.npz"))):
    z = np.load(f, allow_pickle=True)
    tid = str(z["tid"])
    gs = L.goal_steps(tasks[tid]) or []
    hit = [(o, g) for (o, g), act in gs if act[1] == str(z["named"]) and act[2] == str(z["target"])]
    out[tid] = dict(obj_phrase=L.norm(hit[0][0]) if hit else None, goal_phrase=L.norm(hit[0][1]) if hit else None,
                    file=os.path.relpath(f, ROOT))   # None: the parse does not align -> no binder masks for this cell
os.makedirs(os.path.join(ROOT, "results/bind/s4"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/bind/s4/phrases.json"), "w"), indent=1)
print(f"[s4 phrases] {len(out)} cells, {sum(v['obj_phrase'] is None for v in out.values())} without an aligned parse", flush=True)

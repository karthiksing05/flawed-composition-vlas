"""F (docs/SCAFFOLD_RESULTS.md 7; m3_prereg sc_feas): how much of the layout gap is physically closable?  LIBERO venv, CPU, no training.
fa1l_cgen.episode, unchanged, runs every grasp option on each object's own known task, 3 noise seeds, dev states.  Measurement only: nothing enters any data file.

    python experiments/sc_feas.py smoke
    python experiments/sc_feas.py run --index 0        (0-119: tasks x layouts x states)
    python experiments/sc_feas.py read                 -> results/flawed/sc_feas_read.json, sc_feas_states.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_feas")
OBJ = dict(bowl="K1", cheese="K6", bottle="K4")
NOISE = (0, 1, 2)
BASE = (0, 0)
MIN_SEEDS, GAIN, FLOOR, B = 2, 0.10, 0.90, 2000


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def run_state(comp, lay, trial, outdir):
    import fa1_common as FC
    import fa1l_cgen as CG
    fn = os.path.join(outdir, f"{comp}_{lay}_{trial}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    n2 = tasks()
    t = next(x for x in n2["tasks"] if x["compound"] == comp and x["layout"] == lay)
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][comp]
    oe, te = FC.roles(t["language"])
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    rows = []
    for opt, nz in itertools.product(CG.OPTIONS, NOISE):
        r = CG.episode(os.path.join(ROOT, t["bddl"]), S[int(trial)], oe, te, kp, opt, 1000 * int(trial) + nz, False, True)
        rows.append(dict(option=list(opt), noise=nz, held_lift=bool(r["held_lift"]), success=bool(r["success"]), steps=int(r["steps"])))
        print("[feas]", comp, lay, trial, json.dumps(rows[-1]), flush=True)
    os.makedirs(outdir, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(compound=comp, layout=lay, trial=int(trial), object=oe, rows=rows), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def combos():
    ks = tasks()["states"]["main"]
    return list(itertools.product(OBJ.values(), ("familiar", "new"), ks))


def smoke():
    for comp in OBJ.values():
        for lay in ("familiar", "new"):
            run_state(comp, lay, 0, os.path.join(OUT, "sc_feas_smoke"))


def per_state(R, key, opts):
    """Whether any allowed option achieves `key` in at least MIN_SEEDS of the noise seeds on this state."""
    for o in opts:
        if sum(holds(x) if key == "held_lift" else x[key] for x in R["rows"] if tuple(x["option"]) == o) >= MIN_SEEDS:
            return True
    return False


def holds(x):
    """held_lift samples once, at the lift's end, and the smoke shows it can read false on an episode that succeeds: a success counts as a hold."""
    return bool(x["held_lift"] or x["success"])


def read():
    import fa1l_cgen as CG
    ks = tasks()["states"]["main"]
    opts = [tuple(o) for o in CG.OPTIONS]
    assert BASE in opts
    res, reach = dict(objects={}), {}
    for name, comp in OBJ.items():
        res["objects"][name] = {}
        for lay in ("familiar", "new"):
            A = [json.load(open(os.path.join(OD, f"{comp}_{lay}_{k}.json"), encoding="utf-8")) for k in ks]
            for r in A:
                assert len(r["rows"]) == len(opts) * len(NOISE)
            h_any = np.array([per_state(r, "held_lift", opts) for r in A], float)
            h_base = np.array([per_state(r, "held_lift", [BASE]) for r in A], float)
            s_any = np.array([per_state(r, "success", opts) for r in A], float)
            s_base = np.array([per_state(r, "success", [BASE]) for r in A], float)
            rng = np.random.default_rng(0)
            v = [float((h_any - h_base)[rng.choice(len(ks), len(ks))].mean()) for _ in range(B)]
            res["objects"][name][lay] = dict(hold_any=float(h_any.mean()), hold_base=float(h_base.mean()), gain=float((h_any - h_base).mean()), gain_ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                             success_any=float(s_any.mean()), success_base=float(s_base.mean()),
                                             option_hold_rate={str(o): float(np.mean([holds(x) for r in A for x in r["rows"] if tuple(x["option"]) == o])) for o in opts},
                                             states_blocked=[int(k) for k, x in zip(ks, h_any) if not x])
            if lay == "new":
                reach[name] = [int(k) for k, x in zip(ks, h_any) if x]
            print("[feas]", name, lay, json.dumps(res["objects"][name][lay]), flush=True)
    O = res["objects"]
    fam_ok = all(O[n]["familiar"]["hold_any"] >= FLOOR for n in O)
    gain = any(O[n]["new"]["gain"] >= GAIN and O[n]["new"]["gain_ci"][0] > 0.0 for n in O)
    res["label"] = "INSTRUMENT FAILS" if not fam_ok else ("GAIN" if gain else "NO GAIN")
    print("[feas]", res["label"], flush=True)
    with open(os.path.join(OUT, "sc_feas_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(OUT, "sc_feas_states.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(note="new-layout dev states where some grasp option holds in at least 2 of 3 noise seeds; a lower bound, because the options are not exhaustive", states=reach), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "run", "read"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "run":
        c, l, k = combos()[a.index]
        run_state(c, l, k, OD)
    else:
        {"smoke": smoke, "read": read}[a.what]()

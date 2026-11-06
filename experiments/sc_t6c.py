"""T6c (docs/SCAFFOLD_RESULTS.md 23; m3_prereg sc_t6c): T6b's arm R on three-skill chains.  LIBERO venv, CPU, no training, U's checkpoints.

    python experiments/sc_t6c.py make               writes results/flawed/sc_t6c_bddl/*.bddl
    python experiments/sc_t6c.py roll --index 0
    python experiments/sc_t6c.py read               -> results/flawed/sc_t6c_read.json
SC_SMOKE=1: 2 smoke states, L1 and L2, seed 10, familiar, results under sc_t6c_smoke.
"""
import argparse
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_t6 as T                                                           # noqa: E402
import sc_t6b as B                                                          # noqa: E402

SMOKE = T.SMOKE
OD = os.path.join(T.OUT, "sc_t6c_smoke" if SMOKE else "sc_t6c")
BD = os.path.join(T.OUT, "sc_t6c_bddl")
HORIZON = "ValueError('executing action in terminated episode')"
CHAINS = dict(L1=("K1", "N2", "K4"), L2=("K4", "N1", "K2"), L3=("N4", "K3", "N2"), L4=("K2", "N4", "N3"), L5=("N1", "N5", "K3"), L6=("N3", "K1", "N5"))


def make():
    n2 = T.tasks()
    os.makedirs(BD, exist_ok=True)
    for name, cs in CHAINS.items():
        ps = [os.path.join(ROOT, T.task(n2, c, "familiar")["bddl"]) for c in cs]
        txt = open(ps[0], encoding="utf-8").read()
        gs = [T.goal_of(p) for p in ps]
        new, n = re.subn(r"\(:goal\s*\(And\s*.*?\)\s*\)\s*\)", "(:goal\n    (And " + " ".join(gs) + ")\n  )", txt, flags=re.S)
        assert n == 1, name
        with open(os.path.join(BD, name + ".bddl"), "w", encoding="utf-8") as fh:
            fh.write(new)
        print(name, gs, flush=True)


def combos():
    if SMOKE:
        return [("L1", 10, "familiar"), ("L2", 10, "familiar")]
    return list(itertools.product(CHAINS, T.SEEDS, ("familiar", "new")))


def roll(name, seed, lay):
    import torch
    import fa1_common as FC
    from pp_p_roll import load_any
    n2 = T.tasks()
    t = T.task(n2, CHAINS[name][0], lay)
    fn = os.path.join(OD, f"{name}_s{seed}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
    S = np.load(os.path.join(T.OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    br = json.load(open(os.path.join(T.OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        clauses = [T.task(n2, c, lay)["language"] for c in CHAINS[name]]
        e = B.episode(m, os.path.join(BD, name + ".bddl"), S[int(b["trial"])], clauses, b["seed_base"], "R")
        eps.append(dict(trial=int(b["trial"]), **e))
        print("[t6c]", name, seed, lay, b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(name=name, seed=seed, layout=lay, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = T.tasks()
    ks = n2["states"]["main"]

    def load(n, s, l):
        E = json.load(open(os.path.join(OD, f"{n}_s{s}_{l}.json"), encoding="utf-8"))["eps"]
        for e in E:                                                         # amended after the first read: an episode past the env's 1000-step horizon raises; a failure, goals unmeasured
            assert e.get("error") in (None, HORIZON), (n, e.get("error"))
            assert not (e.get("error") == HORIZON and e["success"]), n
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def u(c, s, l):
        E = json.load(open(os.path.join(T.OUT, "sc_t4", f"U_s{s}_{T.task(n2, c, l)['tid']}.json"), encoding="utf-8"))["eps"]
        d = {int(e["trial"]): e for e in E}
        return np.array([float(d[k]["success"]) for k in ks])

    names = list(CHAINS)

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(T.B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(V={}, per_chain={})
    for l in ("familiar", "new"):
        ch = np.array([[[float(e["success"]) for e in load(n, s, l)] for s in T.SEEDS] for n in names])
        pr = np.array([[np.prod([u(c, s, l) for c in CHAINS[n]], axis=0) for s in T.SEEDS] for n in names])
        res["V"][l] = diff(ch, pr)
        for i, n in enumerate(names):
            E = [e for s in T.SEEDS for e in load(n, s, l)]
            res["per_chain"][f"{n}|{l}"] = dict(
                chain=float(ch[i].mean()), product=float(pr[i].mean()), past_horizon=float(np.mean([e.get("error") == HORIZON for e in E])),
                goal_end=[float(np.mean([len(e["goals"]) > j and e["goals"][j] for e in E])) for j in range(3)],
                alone=[float(np.mean([u(c, s, l).mean() for s in T.SEEDS])) for c in CHAINS[n]],
                goal_at_transition=[float(np.mean([len(e["clauses"]) > j and bool(e["clauses"][j].get("first_goal_at_transition")) for e in E])) for j in range(2)],
                arrived=[float(np.mean([len(e["clauses"]) > j and bool(e["clauses"][j].get("arrived")) for e in E])) for j in range(2)])
    x = res["V"]["familiar"]
    res["label"] = "BREAKS" if x["ci"][1] < -T.MARGIN else ("HOLDS" if x["ci"][0] >= -T.MARGIN else "UNRESOLVED")
    print("[t6c]", json.dumps(res), flush=True)
    with open(os.path.join(T.OUT, "sc_t6c_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("make", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "make":
        make()
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        roll(*combos()[a.index])
    else:
        read()

"""T15b (docs/SCAFFOLD_RESULTS.md 29; m3_prereg sc_t15b): T15's arm R with no end by stillness.  An open clause that is not last ends within rho of its target, or at SKILL_MAX.  LIBERO venv, CPU, no training.

    python experiments/sc_t15b.py roll --index 0
    python experiments/sc_t15b.py read               -> results/flawed/sc_t15b_read.json
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
import sc_t14 as S                                                          # noqa: E402
import sc_t15 as C                                                          # noqa: E402

T = S.T
OD = os.path.join(T.OUT, "sc_t15b")


def combos():
    return list(itertools.product(S.CHAINS, T.SEEDS, ("familiar", "new")))


def roll(name, seed, lay):
    import torch
    import fa1_common as FC
    from pp_p_roll import load_any
    n2 = T.tasks()
    t = T.task(n2, S.CHAINS[name][0], lay)
    fn = os.path.join(OD, f"{name}_s{seed}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    rho = C.rho_demos()
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc11_Um_s{seed}", "P.npz"))
    st = np.load(os.path.join(T.OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    br = json.load(open(os.path.join(T.OUT, "n2_branches_main.json"), encoding="utf-8"))
    eps = []
    for b in br:
        clauses = [T.task(n2, c, lay)["language"] for c in S.CHAINS[name]]
        e = S.episode(m, os.path.join(S.BD, name + ".bddl"), st[int(b["trial"])], clauses, b["seed_base"], end="nearonly", rho=rho)
        eps.append(dict(trial=int(b["trial"]), **e))
        print("[t15b]", name, seed, lay, b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(name=name, seed=seed, layout=lay, rho=rho, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = T.tasks()
    ks = n2["states"]["main"]

    def load(d, pre, n, s, l):
        E = json.load(open(os.path.join(T.OUT, d, f"{pre}{n}_s{s}_{l}.json"), encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") in (None, S.HORIZON), (n, e.get("error"))
            assert not (e.get("error") == S.HORIZON and e["success"]), n
        by = {int(e["trial"]): e for e in E}
        return [by[k] for k in ks]

    def alone(c, s, l):
        tid = T.task(n2, c, l)["tid"]
        p = ("sc_t12", "V") if c in ("K8", "N8") else ("sc_t11", "Um")
        E = json.load(open(os.path.join(T.OUT, p[0], f"{p[1]}_s{s}_{tid}.json"), encoding="utf-8"))["eps"]
        by = {int(e["trial"]): e for e in E}
        return np.array([float(by[k]["success"]) for k in ks])

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(T.B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    def ch(d, pre, names, l):
        return np.array([[[float(e["success"]) for e in load(d, pre, n, s, l)] for s in T.SEEDS] for n in names])

    def pr(names, l):
        return np.array([[alone(S.CHAINS[n][0], s, l) * alone(S.CHAINS[n][1], s, l) for s in T.SEEDS] for n in names])

    vs, dp = list(S.VERDICT), list(S.DEPENDENT)
    res = dict(V={}, dependent={}, vs_R={}, past_horizon={}, per_chain={})
    for l in ("familiar", "new"):
        res["V"][l] = diff(ch("sc_t15b", "", vs, l), pr(vs, l))
        res["dependent"][l] = diff(ch("sc_t15b", "", dp, l), pr(dp, l))
        res["vs_R"][l] = diff(ch("sc_t15b", "", vs, l), ch("sc_t15", "R_", vs, l))
        for n in S.CHAINS:
            E = [e for s in T.SEEDS for e in load("sc_t15b", "", n, s, l)]
            res["per_chain"][f"{n}|{l}"] = dict(chain=float(ch("sc_t15b", "", [n], l).mean()), product=float(pr([n], l).mean()), past_horizon=float(np.mean([e.get("error") == S.HORIZON for e in E])),
                                                first_ended=float(np.mean([len(e["clauses"]) > 0 and e["clauses"][0]["ended"] for e in E])),
                                                first_goal_at_transition=float(np.mean([len(e["clauses"]) > 0 and bool(e["clauses"][0].get("goal_at_transition")) for e in E])),
                                                goal_end=[float(np.mean([len(e["goals"]) > j and e["goals"][j] for e in E])) for j in range(2)])
    x = res["V"]["familiar"]
    res["label"] = "BREAKS" if x["ci"][1] < -T.MARGIN else ("HOLDS" if x["ci"][0] >= -T.MARGIN else "UNRESOLVED")
    print("[t15b]", json.dumps(res), flush=True)
    with open(os.path.join(T.OUT, "sc_t15b_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        roll(*combos()[a.index])
    else:
        read()

"""FLAWED_COMPOSITION 4.4, D0: the projected training support of every dev task, per interface, from the SPLIT ALONE (no model, no
demo, no outcome is read).  CPU.

Atoms and compounds are the split's own (experiments/c_split.py: steps / atoms / compounds; type-level object, target with its
region, verb predicate, BDDL scene).  For each dev task and each of its steps (v, o, t) in scene s, over the 76 TRAINING tasks:

  n_o, n_t      training tasks that contain the object atom o / the target atom t
  P_o, P_t      distinct partners: targets seen with o / objects seen with t (the varied-partner count)
  L_o, L_t      distinct scenes in which o / t appears in training (layout confounding: 1 scene = atom fully confounded with layout)
  seen(v,o), seen(o,t), seen(o,s), seen(t,s)   whether that compound occurs in training

Predicted debt (higher = weaker support = more debt), fixed here before any D1 read:
  components (each oriented so higher = weaker):  1-seen(v,o), 1-seen(o,t), 1-seen(o,s), 1-seen(t,s),
                                                   -log1p(P_o), -log1p(P_t), -log1p(L_o), -log1p(L_t)
  each component is averaged over the task's steps (target terms only over steps that have a target), ranked across the 29 dev
  tasks (average ties), and the task's predicted-debt score is the mean of its available component ranks.  The interface
  sub-scores are reported too: binder = {-log1p(n_o), -log1p(n_t), -log1p(L_o), -log1p(L_t), 1-seen(o,s), 1-seen(t,s)} and
  motor = {1-seen(v,o), 1-seen(o,t), -log1p(P_o), -log1p(P_t)}.
-> results/flawed/d0_support.json

    python experiments/fc_d0_support.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")

ALL = ("u_vo", "u_ot", "u_os", "u_ts", "nP_o", "nP_t", "nL_o", "nL_t")
BINDER = ("nn_o", "nn_t", "nL_o", "nL_t", "u_os", "u_ts")
MOTOR = ("u_vo", "u_ot", "nP_o", "nP_t")


def rankdata(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def main():
    import c_split as CS
    from lifelong_ood import l90c
    from lifelong_ood.policy import factors as F
    sp = l90c.load_split()
    tasks = {t.tid: t for t in F.load(CS.SUITES)}
    train = [tasks[t] for t in sp["train"]]
    n_o, n_t = collections.Counter(), collections.Counter()
    P_o, P_t, L_o, L_t = (collections.defaultdict(set) for _ in range(4))
    seen = set()
    for t in train:
        st = CS.steps(t)
        for o in {o for _, o, _ in st}:
            n_o[o] += 1
        for g in {g for _, _, g in st if g}:
            n_t[g] += 1
        for v, o, g in st:
            P_o[o].add(g)
            L_o[o].add(t.scene)
            if g:
                P_t[g].add(o)
                L_t[g].add(t.scene)
        seen |= set(CS.compounds(t))
    dev = [(b, tid) for b in ("0.1", "0.3", "0.5", "part") for tid in sp["dev"][b]]
    rows = {}
    for b, tid in dev:
        t = tasks[tid]
        s = "s:" + t.scene
        comp = collections.defaultdict(list)
        per_step = []
        for v, o, g in CS.steps(t):
            o_, v_ = "o:" + o, "v:" + v
            r = dict(verb=v, object=o, target=g, n_o=n_o[o], P_o=len(P_o[o]), L_o=len(L_o[o]),
                     seen_vo=(v_, o_) in seen, seen_os=(o_, s) in seen)
            comp["u_vo"].append(1.0 - r["seen_vo"])
            comp["u_os"].append(1.0 - r["seen_os"])
            comp["nP_o"].append(-math.log1p(r["P_o"]))
            comp["nL_o"].append(-math.log1p(r["L_o"]))
            comp["nn_o"].append(-math.log1p(r["n_o"]))
            if g:
                t_ = "t:" + g
                r.update(n_t=n_t[g], P_t=len(P_t[g]), L_t=len(L_t[g]), seen_ot=(o_, t_) in seen, seen_ts=(t_, s) in seen)
                comp["u_ot"].append(1.0 - r["seen_ot"])
                comp["u_ts"].append(1.0 - r["seen_ts"])
                comp["nP_t"].append(-math.log1p(r["P_t"]))
                comp["nL_t"].append(-math.log1p(r["L_t"]))
                comp["nn_t"].append(-math.log1p(r["n_t"]))
            per_step.append(r)
        rows[tid] = dict(bin=b, scene=t.scene, language=t.language, cd=sp["dev_per_task"].get(tid, {}).get("cd"),
                         steps=per_step, components={k: float(np.mean(v)) for k, v in comp.items()})
    tids = [tid for _, tid in dev]
    ranks = {}
    for k in sorted(set(ALL) | set(BINDER) | set(MOTOR)):
        have = [t for t in tids if k in rows[t]["components"]]
        r = rankdata([rows[t]["components"][k] for t in have])
        ranks[k] = dict(zip(have, r.tolist()))

    def score(t, keys):
        v = [ranks[k][t] for k in keys if t in ranks[k]]
        return float(np.mean(v)) if v else None
    for t in tids:
        rows[t]["score_all"] = score(t, ALL)
        rows[t]["score_binder"] = score(t, BINDER)
        rows[t]["score_motor"] = score(t, MOTOR)
    order = sorted(tids, key=lambda t: -rows[t]["score_all"])
    for i, t in enumerate(order):
        rows[t]["predicted_debt_rank"] = i + 1                 # 1 = most debt predicted
    out = dict(n_dev=len(tids), n_train=len(train), components_all=ALL, components_binder=BINDER, components_motor=MOTOR,
               predicted_order=order, per_task=rows, split_sha256=__import__("hashlib").sha256(open(l90c.SPLIT, "rb").read()).hexdigest())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "d0_support.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for t in order:
        r = rows[t]
        print(f"{r['predicted_debt_rank']:2d} {t:14s} bin {r['bin']:4s} cd {r['cd']} score {r['score_all']:.2f} binder {r['score_binder']:.2f} "
              f"motor {r['score_motor']:.2f} | {r['language']}")


if __name__ == "__main__":
    main()

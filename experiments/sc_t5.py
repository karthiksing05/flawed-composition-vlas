"""T5a (docs/SCAFFOLD_RESULTS.md 12; m3_prereg sc_t5): does a trained object part work in a rotated frame, and does rotation beat repeating?  LIBERO venv, CPU, no training.
U's checkpoints run with the object part's yaw frame offset by theta until the hold.  The rep arms repeat theta = 0 under fresh noise: retrying alone also gains by luck.

    python experiments/sc_t5.py roll --index 0
    python experiments/sc_t5.py read       -> results/flawed/sc_t5_read.json
SC_SMOKE=1: 2 smoke states, results under sc_t5_smoke, a small set of combos, U's sc_ checkpoints.
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
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_t5_smoke" if SMOKE else "sc_t5")
SEEDS = (10, 11, 12)
COMPS = tuple(f"K{i}" for i in range(1, 7)) + tuple(f"N{i}" for i in range(1, 7))
OBJECTS = dict(bowl=("K1", "K2", "K3"), cheese=("K6", "N1", "N2", "N3"), bottle=("K4", "K5", "N4", "N5", "N6"))
ARMS = dict(th45=(45, 0), thm45=(-45, 0), th90=(90, 0), rep1=(0, 100000), rep2=(0, 200000), rep3=(0, 300000))     # (theta deg, seed_base offset)
ROTS, REPS = ("th45", "thm45", "th90"), ("rep1", "rep2", "rep3")
B, GAIN = 2000, 0.10


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def task(n2, c, l):
    return next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == l)


def combos():
    n2 = tasks()
    if SMOKE:
        return [(a, 10, task(n2, c, l)["tid"]) for a, c, l in (("th90", "K1", "familiar"), ("th90", "K4", "new"), ("thm45", "K6", "familiar"), ("rep1", "K1", "new"))]
    return [(a, s, task(n2, c, l)["tid"]) for a, s, c, l in itertools.product(ARMS, SEEDS, COMPS, ("familiar", "new"))]


class Framed:
    """The object part's yaw frame turned by theta while nothing is held; everything else is the model's."""

    def __init__(self, m, oi, theta):
        self._m, self._oi, self._th = m, oi, np.radians(theta)

    def __getattr__(self, k):
        return getattr(self._m, k)

    def sample(self, b, **kw):
        if self._th and float(b["held"][0]) < 0.5:
            b = dict(b)
            y = b["yaw"].clone()
            y[0, self._oi] = y[0, self._oi] + float(self._th)
            b["yaw"] = y
        return self._m.sample(b, **kw)


def roll(arm, seed, tid):
    import torch
    import fa1_common as FC
    from fa1h_roll import episode
    from pp_p_roll import load_any
    from lifelong_ood.policy.parts_flow import role_parse
    fn = os.path.join(OD, f"{arm}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
    parsed = role_parse(m.role_lex, t["language"])
    assert parsed is not None, t["language"]
    th, off = ARMS[arm]
    pol = Framed(m, FC.ENT.index(parsed[0]), th)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        e = episode(pol, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"] + off)
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"], steps=e["steps"], bound=e["bound"]))
        print("[t5]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]
    T4 = os.path.join(OUT, "sc_t4")

    def eps(arm, s, c, l):
        tid = task(n2, c, l)["tid"]
        p = os.path.join(T4, f"U_s{s}_{tid}.json") if arm == "U" else os.path.join(OD, f"{arm}_s{s}_{tid}.json")
        E = json.load(open(p, encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") is None, (p, e.get("error"))
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def arr(arm, comps, l, key):
        f = (lambda e: float(e["success"])) if key == "success" else (lambda e: float(e["held_step"] is not None))
        return np.array([[[f(e) for e in eps(arm, s, c, l)] for s in SEEDS] for c in comps])

    def union(arms, comps, l, key="success"):
        return np.max([arr(a, comps, l, key) for a in ("U",) + arms], axis=0)

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(V=dict(), fixed={}, no_hold={})
    for name, cs in OBJECTS.items():
        for l in ("familiar", "new"):
            res["V"][f"{name}|{l}"] = diff(union(ROTS, cs, l), union(REPS, cs, l))
            res.setdefault("retry_ceiling", {})[f"{name}|{l}"] = diff(union(REPS, cs, l), arr("U", cs, l, "success"))     # descriptive: attempts run from fresh scenes, so an upper bound
            res["fixed"][f"{name}|{l}"] = {a: dict(success=float(arr(a, cs, l, "success").mean()), hold=float(arr(a, cs, l, "hold").mean())) for a in ("U",) + ROTS + REPS}
            print("[t5]", name, l, json.dumps(res["V"][f"{name}|{l}"]), json.dumps(res["fixed"][f"{name}|{l}"]), flush=True)
    win = [n for n in ("bowl", "bottle") if res["V"][f"{n}|new"]["delta"] >= GAIN and res["V"][f"{n}|new"]["ci"][0] > 0.0]
    fam = {n: res["fixed"][f"{n}|familiar"] for n in OBJECTS}
    tracks = [n for n in ("bowl", "bottle") if max(fam[n][a]["hold"] for a in ROTS) >= fam[n]["U"]["hold"] - GAIN]
    res["label"] = "PARTS DO NOT TURN" if not tracks else ("ROTATION PAYS" if win else "RETRY ONLY")
    res["win"], res["tracks"] = win, tracks
    print("[t5]", res["label"], win, tracks, flush=True)
    res["per_task"] = {a: {c: {l: float(arr(a, (c,), l, "success").mean()) for l in ("familiar", "new")} for c in COMPS} for a in ("U",) + ROTS + REPS}
    with open(os.path.join(OUT, "sc_t5_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        arm_, s_, tid_ = combos()[a.index]
        roll(arm_, s_, tid_)
    else:
        read()

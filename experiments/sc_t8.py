"""T8a and T9 (docs/SCAFFOLD_RESULTS.md 16; m3_prereg sc_t8).  LIBERO venv for roll and read, CPU; builds in the openpi venv.
T8a, the oracle-state rungs of the removal ladder: U, Ulc (learned combination), RF (role-routed head), HF (one holistic head).
T9, a lifelong stream: a base without bottle or plate; the two parts arrive one at a time, from one demo each, in both orders.

    python experiments/sc_t8.py builds              prints ARGS, one per line ('~' for spaces): first word pp_p_train or pp_baselines
    python experiments/sc_t8.py roll --index 0
    python experiments/sc_t8.py read                -> results/flawed/sc_t8_read.json
SC_SMOKE=1: 50-step builds, sc_sm8_ tags, 2 states, results under sc_t8_smoke.
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
import sc_t4 as T4                                                          # noqa: E402

OUT, SEEDS, COMPS = T4.OUT, T4.SEEDS, T4.COMPS
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_t8_smoke" if SMOKE else "sc_t8")
PFX = "sc_sm8_" if SMOKE else "sc8_"
STEPS = 50 if SMOKE else 30000
ULC = T4.BASE.replace(" --decouple", "") + " --lcomb"
BOTTLE = dict(role="obj", atom="wine_bottle_1", task="K4")
PLATE = dict(role="tgt", atom="plate_1", task="K1")
ORDERS = dict(bp=(BOTTLE, PLATE), pb=(PLATE, BOTTLE))
NEW_CELLS = ("K1", "K4", "N1", "N4", "N5", "N6")
OLD_CELLS = ("K2", "K3", "K6", "N2", "N3")
B, MARGIN = 2000, 0.10


def builds():
    out = []
    for s in SEEDS:
        out.append(f"pp_p_train build --tag {PFX}Ulc --seed {s} --steps {STEPS} {ULC}")
        out.append(f"pp_baselines build --kind RF --tag {PFX}bl --seed {s} --steps {STEPS}")
        out.append(f"pp_baselines build --kind HF --tag {PFX}bl --seed {s} --steps {STEPS}")
        out.append(f"pp_p_train build --tag {PFX}stream --seed {s} --steps {STEPS} --exclude K1+K4+K5 {T4.BASE}")
    return out


def creates():
    """Two lines per seed run first (each order's first part), two after: creates2()."""
    return [_create(s, o, 0) for s in SEEDS for o in ORDERS]


def creates2():
    return [_create(s, o, 1) for s in SEEDS for o in ORDERS]


def _create(s, o, i):
    d = ORDERS[o][i]
    base = "P.npz" if i == 0 else f"{o}_1.npz"
    return f"pp_p_train create --tag {PFX}stream --seed {s} --role {d['role']} --atom {d['atom']} --task {d['task']} --k 1 --pgate --base {base} --out {o}_{i + 1}.npz"


def tasks():
    return T4.tasks()


def combos():
    n2 = tasks()
    if SMOKE:
        out = [(a, 10, T4.task(n2, "N1", "familiar")["tid"]) for a in ("Ulc", "RF", "HF", "base")]
        out += [(o, 10, T4.task(n2, "N4", "familiar")["tid"]) for o in ORDERS]
        return out
    out = [(a, s, T4.task(n2, c, l)["tid"]) for a, s, c, l in itertools.product(("Ulc", "RF", "HF"), SEEDS, COMPS, ("familiar", "new"))]
    out += [(o, s, T4.task(n2, c, l)["tid"]) for o, s, c, l in itertools.product(ORDERS, SEEDS, NEW_CELLS + OLD_CELLS, ("familiar", "new"))]
    out += [("base", s, T4.task(n2, c, l)["tid"]) for s, c, l in itertools.product(SEEDS, OLD_CELLS, ("familiar", "new"))]
    return out


def ckpt(arm, s):
    import fa1_common as FC
    ck = os.path.join(FC.DATA, "ckpt")
    if arm == "Ulc":
        return os.path.join(ck, f"pp_{PFX}Ulc_s{s}", "P.npz")
    if arm in ("RF", "HF"):
        return os.path.join(ck, f"pp_{PFX}bl_s{s}", f"{arm}.npz")
    d = os.path.join(ck, f"pp_{PFX}stream_s{s}")
    if arm == "base":
        return os.path.join(d, "P.npz")
    return os.path.join(d, f"{arm}_2.npz")


def roll(arm, seed, tid):
    import torch
    from fa1h_roll import episode
    from pp_p_roll import load_any
    fn = os.path.join(OD, f"{arm}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(ckpt(arm, seed))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"], roles=e.get("roles"), bound=e["bound"], steps=e["steps"]))
        print("[t8]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]

    def succ(arm, s, c, l):
        d = T4.OD if arm == "U" else OD
        E = json.load(open(os.path.join(d, f"{arm}_s{s}_{T4.task(n2, c, l)['tid']}.json"), encoding="utf-8"))["eps"]
        by = {int(e["trial"]): e for e in E}
        return np.array([float(by[k]["success"]) for k in ks]), E

    def arr(arm, comps, l):
        return np.array([[succ(arm, s, c, l)[0] for s in SEEDS] for c in comps])

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    G = dict(known=tuple(c for c in COMPS if c[0] == "K"), new_pairings=tuple(c for c in COMPS if c[0] == "N"))
    res = dict(ladder={}, stream={}, old={}, per_task={})
    for arm in ("Ulc", "RF", "HF"):
        r = {f"{g}|{l}": diff(arr("U", cs, l), arr(arm, cs, l)) for g, cs in G.items() for l in ("familiar", "new")}
        x, k = r["new_pairings|familiar"], r["known|familiar"]
        lab = "UNDERFIT" if k["delta"] > MARGIN and k["ci"][0] > 0.0 else ("FALLS" if x["ci"][0] > MARGIN else "NO FALL")
        res["ladder"][arm] = dict(label=lab, cells=r)
    res["ladder_label"] = "C1 RUNGS HOLD" if all(res["ladder"][a]["label"] == "FALLS" for a in ("RF", "HF")) else "C1 RUNGS FAIL"
    for o in ORDERS:
        res["stream"][o] = {l: diff(arr(o, NEW_CELLS, l), arr("U", NEW_CELLS, l)) for l in ("familiar", "new")}
        res["stream"][o]["N4_double_new"] = {l: diff(arr(o, ("N4",), l), arr("U", ("N4",), l))["a"] for l in ("familiar", "new")}
        res["old"][o] = {l: diff(arr(o, OLD_CELLS, l), arr("base", OLD_CELLS, l)) for l in ("familiar", "new")}
    res["stream_label"] = "STREAM HOLDS" if all(res["stream"][o]["familiar"]["ci"][0] >= -MARGIN for o in ORDERS) else "NOT YET"
    res["old_label"] = "NOT FORGOTTEN" if all(res["old"][o]["familiar"]["ci"][0] >= -MARGIN for o in ORDERS) else "FORGETS"
    for arm in ("Ulc", "RF", "HF"):
        res["per_task"][arm] = {c: {l: float(arr(arm, (c,), l).mean()) for l in ("familiar", "new")} for c in COMPS}
    for o in ORDERS:
        res["per_task"][o] = {c: {l: float(arr(o, (c,), l).mean()) for l in ("familiar", "new")} for c in NEW_CELLS + OLD_CELLS}
        res["per_task"][o + "_no_roles"] = {c: float(np.mean([not e.get("roles") for s in SEEDS for l in ("familiar", "new") for e in succ(o, s, c, l)[1]])) for c in NEW_CELLS + OLD_CELLS}
    from pp_p_roll import load_any
    from lifelong_ood.policy.parts_flow import role_parse
    res["parses"] = {}
    for arm in ("base", "bp", "pb"):
        lex = load_any(ckpt(arm, SEEDS[0]))[0].role_lex
        res["parses"][arm] = {c: role_parse(lex, T4.task(n2, c, "familiar")["language"]) for c in COMPS}
    print("[t8]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t8_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("builds", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "builds":
        for x in builds() + creates() + creates2():
            print(x.replace(" ", "~"))
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        arm_, s_, tid_ = combos()[a.index]
        roll(arm_, s_, tid_)
    else:
        read()

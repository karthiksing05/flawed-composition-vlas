"""T4 (docs/SCAFFOLD_RESULTS.md 10; m3_prereg sc_t4): the part stores rebuilt under the unsupervised recipe, with removal controls and reuse.  LIBERO venv, CPU.
    python experiments/sc_t4.py builds                 prints the build and creation commands' ARGS, one per line ('~' for spaces)
    python experiments/sc_t4.py roll --index 0         one of combos(): arm, seed, tid
    python experiments/sc_t4.py read                   -> results/flawed/sc_t4_read.json
SC_SMOKE=1: the sc_sm_ checkpoints, 2 smoke states, results under sc_t4_smoke.
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
OD = os.path.join(OUT, "sc_t4_smoke" if SMOKE else "sc_t4")
PFX = "sc_sm_" if SMOKE else "sc_"
SEEDS = (10, 11, 12)
COMPS = tuple(f"K{i}" for i in range(1, 7)) + tuple(f"N{i}" for i in range(1, 7))
BASE = "--rank 32 --rtclip --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize --pgate --iroles --nosize"
ARMS = dict(U=BASE, Unm=BASE.replace(" --imode", ""), Unp=BASE.replace(" --psupport", ""), Unt=BASE.replace(" --tscale", ""),
            Unr=BASE.replace(" --rtransport --bbnom --bbnosize", ""), Ur8=BASE.replace("--rank 32", "--rank 8"))
COND = dict(nobottle=dict(exclude="K4+K5", role="obj", atom="wine_bottle_1", task="K4", cells=("K4", "N4", "N5", "N6")),
            noplate=dict(exclude="K1", role="tgt", atom="plate_1", task="K1", cells=("K1", "N1", "N4")))
KS = (1, 5, 10)
B, MARGIN = 2000, 0.10


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def task(n2, c, l):
    return next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == l)


def builds():
    steps = 50 if SMOKE else 30000
    out = []
    for arm, flags in ARMS.items():
        for s in SEEDS:
            out.append(f"build --tag {PFX}{arm} --seed {s} --steps {steps} {flags}")
    for cond, c in COND.items():
        for s in SEEDS:
            out.append(f"build --tag {PFX}U_{cond} --seed {s} --steps {steps} --exclude {c['exclude']} {BASE}")
    return out


def creates():
    return [f"create --tag {PFX}U_{cond} --seed {s} --role {c['role']} --atom {c['atom']} --task {c['task']} --k {k} --pgate" for cond, c in COND.items() for s in SEEDS for k in KS]


def combos():
    n2 = tasks()
    out = []
    for arm in list(ARMS) + ["U0"]:
        for s, c, l in itertools.product(SEEDS, COMPS, ("familiar", "new")):
            out.append((arm, s, task(n2, c, l)["tid"]))
    for cond, cd in COND.items():
        for k, s, c, l in itertools.product(KS, SEEDS, cd["cells"], ("familiar", "new")):
            out.append((f"{cond}_k{k}", s, task(n2, c, l)["tid"]))
    return out


def ckpt_of(arm, seed):
    import fa1_common as FC
    if "_k" in arm:
        cond, k = arm.split("_k")
        return os.path.join(FC.DATA, "ckpt", f"pp_{PFX}U_{cond}_s{seed}", f"P_{COND[cond]['atom']}_k{k}.npz")
    return os.path.join(FC.DATA, "ckpt", f"pp_{PFX}{'U' if arm == 'U0' else arm}_s{seed}", "P.npz")


def roll(arm, seed, tid, smoke_n=None):
    import torch
    import fa1_common as FC
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
    m, meta = load_any(ckpt_of(arm, seed))
    if arm == "U0":
        with torch.no_grad():
            for net in (m.obj, m.tgt):
                net.emb.zero_()
                for ly in net.hidden:
                    ly.B.zero_()
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"], roles=e.get("roles"), bound=e["bound"], steps=e["steps"]))
        print("[t4]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]

    def succ(path):
        R = json.load(open(path, encoding="utf-8"))
        E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
        d = {int(e["trial"]): e for e in E}
        for e in E:
            assert e.get("error") is None, (path, e.get("error"))
        return np.array([float(bool(d[k]["success"])) for k in ks])

    def arr(arm, comps, lay):
        return np.array([[succ(os.path.join(OD, f"{arm}_s{s}_{task(n2, c, lay)['tid']}.json")) for s in SEEDS] for c in comps])

    def ref_l4(comps, lay):
        return np.array([[succ(os.path.join(OUT, "pp_L4_roll", f"P2L4_s{s}_main_{task(n2, c, lay)['tid']}.json")) for s in SEEDS] for c in comps])

    def diff(a, b):
        """a, b: [tasks, seeds, states]."""
        rng = np.random.default_rng(0)
        d = float((a - b).mean())
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=d, ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    GROUPS = dict(known=tuple(c for c in COMPS if c[0] == "K"), new_pairings=tuple(c for c in COMPS if c[0] == "N"))
    cells = [(g, l) for g in GROUPS for l in ("familiar", "new")]
    res = dict(V1={}, V2={}, V3={}, per_task={})
    for g, l in cells:
        res["V1"][f"{g}|{l}"] = diff(arr("U", GROUPS[g], l), ref_l4(GROUPS[g], l))
    fam = [res["V1"][f"{g}|familiar"] for g in GROUPS]
    res["V1_label"] = "COSTS" if any(x["ci"][1] < -MARGIN for x in fam) else ("HOLDS" if all(x["ci"][0] >= -MARGIN for x in fam) else "UNRESOLVED")
    print("[t4] V1", res["V1_label"], json.dumps(res["V1"]), flush=True)
    for arm in list(ARMS)[1:] + ["U0"]:
        r = {f"{g}|{l}": diff(arr(arm, GROUPS[g], l), arr("U", GROUPS[g], l)) for g, l in cells}
        keep = [k for k, x in r.items() if x["delta"] < -MARGIN and x["ci"][1] < 0.0]
        res["V2"][arm] = dict(label="KEEP" if keep else "DROP", keep_cells=keep, cells=r)
        print("[t4] V2", arm, res["V2"][arm]["label"], json.dumps(r), flush=True)
    labs = []
    for cond, cd in COND.items():
        res["V3"][cond] = {}
        for k in KS:
            for l in ("familiar", "new"):
                res["V3"][cond][f"k{k}|{l}"] = diff(arr(f"{cond}_k{k}", cd["cells"], l), arr("U", cd["cells"], l))
        labs.append(res["V3"][cond]["k1|familiar"]["ci"][0] >= -MARGIN)
        print("[t4] V3", cond, json.dumps(res["V3"][cond]), flush=True)
    res["V3_label"] = "TRANSFERS" if all(labs) else "NOT YET"
    print("[t4] V3", res["V3_label"], flush=True)
    for arm in list(ARMS) + ["U0"]:
        res["per_task"][arm] = {c: {l: float(arr(arm, (c,), l).mean()) for l in ("familiar", "new")} for c in COMPS}
    res["per_task"]["L4"] = {c: {l: float(ref_l4((c,), l).mean()) for l in ("familiar", "new")} for c in COMPS}
    with open(os.path.join(OUT, "sc_t4_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("builds", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "builds":
        for x in builds() + creates():
            print(x.replace(" ", "~"))
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        arm, s, tid = combos()[a.index]
        roll(arm, s, tid)
    else:
        read()

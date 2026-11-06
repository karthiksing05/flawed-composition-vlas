"""T11 (docs/SCAFFOLD_RESULTS.md 24; m3_prereg sc_t11): the event generalized: the hand moves an entity.  Arm Um = U + --mgate.  LIBERO venv for roll and read, CPU; builds in the openpi venv.

    python experiments/sc_t11.py builds             prints ARGS, one per line ('~' for spaces), first word the module
    python experiments/sc_t11.py roll --index 0
    python experiments/sc_t11.py read               -> results/flawed/sc_t11_read.json
SC_SMOKE=1: 50-step builds, sc_sm11_ tags, 2 states, results under sc_t11_smoke.
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
OD = os.path.join(OUT, "sc_t11_smoke" if SMOKE else "sc_t11")
PFX = "sc_sm11_" if SMOKE else "sc11_"
STEPS = 50 if SMOKE else 30000
PUSH = ("K8", "N8")
B, MARGIN = 2000, 0.10


def builds():
    return [f"pp_p_train build --tag {PFX}Um --seed {s} --steps {STEPS} {T4.BASE} --mgate" for s in SEEDS]


def tasks():
    return T4.tasks()


def combos():
    n2 = tasks()
    if SMOKE:
        return [("Um", 10, T4.task(n2, c, "familiar")["tid"]) for c in ("K8", "N8", "K1")]
    return [("Um", s, T4.task(n2, c, l)["tid"]) for s, c, l in itertools.product(SEEDS, COMPS + PUSH, ("familiar", "new"))]


def roll(arm, seed, tid):
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
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_{PFX}Um_s{seed}", "P.npz"))
    assert m.pgate.get("motion"), "no motion flag in the checkpoint"
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], roles=e.get("roles"), steps=e["steps"], pgate=e.get("pgate"), final=e["final"].get(e["roles"][0]) if e.get("roles") else None))
        print("[t11]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
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

    def boot(a):                                                            # [seeds, states], over states with seeds fixed
        rng = np.random.default_rng(0)
        v = [float(a[:, rng.choice(a.shape[1], a.shape[1])].mean()) for _ in range(B)]
        return dict(mean=float(a.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    G = dict(known=tuple(c for c in COMPS if c[0] == "K"), new_pairings=tuple(c for c in COMPS if c[0] == "N"))
    res = dict(cost={f"{g}|{l}": diff(arr("Um", cs, l), arr("U", cs, l)) for g, cs in G.items() for l in ("familiar", "new")}, push={}, per_task={})
    for c in PUSH:
        for l in ("familiar", "new"):
            res["push"][f"{c}|{l}"] = boot(arr("Um", (c,), l)[0])
            E = [e for s in SEEDS for e in succ("Um", s, c, l)[1]]
            res["push"][f"{c}|{l}"].update(event=float(np.mean([bool(e["pgate"] and e["pgate"]["event_step"] is not None) for e in E])), roles=E[0].get("roles"))
    for c in COMPS + PUSH:
        res["per_task"][c] = {l: float(arr("Um", (c,), l).mean()) for l in ("familiar", "new")}
    k8 = res["push"]["K8|familiar"]
    res["push_label"] = "PUSH WORKS" if k8["mean"] >= 0.50 and k8["ci"][0] >= 0.25 else ("PUSH PARTIAL" if k8["ci"][0] > 0.05 else "NO PUSH")
    res["cost_label"] = "NO COST" if all(res["cost"][f"{g}|familiar"]["ci"][0] >= -MARGIN for g in G) else "COST"
    print("[t11]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t11_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("builds", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "builds":
        for x in builds():
            print(x.replace(" ", "~"))
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        roll(*combos()[a.index])
    else:
        read()

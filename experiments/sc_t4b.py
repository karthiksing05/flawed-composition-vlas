"""T4b (docs/SCAFFOLD_RESULTS.md 12; m3_prereg sc_t5): the controls T4 lacked.  LIBERO venv for roll and read, CPU; builds run in the openpi venv.
Z: a created part switched off (its row of emb and B zeroed), k = 1.  J: Unp, Unt and Ur8's removals together, with its own reuse builds.

    python experiments/sc_t4b.py builds             prints pp_p_train ARGS, one per line ('~' for spaces)
    python experiments/sc_t4b.py roll --index 0
    python experiments/sc_t4b.py read               -> results/flawed/sc_t4b_read.json
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

OUT, COND, SEEDS, COMPS = T4.OUT, T4.COND, T4.SEEDS, T4.COMPS
OD = os.path.join(OUT, "sc_t4b")
JFLAGS = T4.BASE.replace("--rank 32", "--rank 8").replace(" --psupport", "").replace(" --tscale", "")
B, MARGIN = 2000, 0.10


def builds():
    out = [f"build --tag sc_Uj --seed {s} --steps 30000 {JFLAGS}" for s in SEEDS]
    out += [f"build --tag sc_Uj_{c} --seed {s} --steps 30000 --exclude {d['exclude']} {JFLAGS}" for c, d in COND.items() for s in SEEDS]
    out += [f"create --tag sc_Uj_{c} --seed {s} --role {d['role']} --atom {d['atom']} --task {d['task']} --k 1 --pgate" for c, d in COND.items() for s in SEEDS]
    return out


def combos():
    n2 = T4.tasks()
    out = []
    for c, d in COND.items():
        for arm in (f"Z_{c}", f"Jc_{c}"):
            out += [(arm, s, T4.task(n2, x, l)["tid"]) for s, x, l in itertools.product(SEEDS, d["cells"], ("familiar", "new"))]
    out += [("Uj", s, T4.task(n2, x, l)["tid"]) for s, x, l in itertools.product(SEEDS, COMPS, ("familiar", "new"))]
    return out


def load(arm, seed):
    import torch
    import fa1_common as FC
    from pp_p_roll import load_any
    from lifelong_ood.policy import parts_flow as PF
    ck = os.path.join(FC.DATA, "ckpt")
    if arm == "Uj":
        return load_any(os.path.join(ck, f"pp_sc_Uj_s{seed}", "P.npz"))[0]
    kind, c = arm.split("_")
    d = COND[c]
    tag = f"pp_sc_U_{c}_s{seed}" if kind == "Z" else f"pp_sc_Uj_{c}_s{seed}"
    m = load_any(os.path.join(ck, tag, f"P_{d['atom']}_k1.npz"))[0]
    if kind == "Z":
        net = m.obj if d["role"] == "obj" else m.tgt
        j = (PF.OBJ_ATOMS if d["role"] == "obj" else PF.TGT_ATOMS).index(d["atom"])
        with torch.no_grad():
            net.emb[j] = 0.0
            for ly in net.hidden:
                ly.B[j] = 0.0
    return m


def roll(arm, seed, tid):
    import torch
    from fa1h_roll import episode
    fn = os.path.join(OD, f"{arm}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = T4.tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m = load(arm, seed)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    eps = []
    for b in br:
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"], roles=e.get("roles"), bound=e["bound"], steps=e["steps"]))
        print("[t4b]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = T4.tasks()
    ks = n2["states"]["main"]

    def succ(p):
        E = json.load(open(p, encoding="utf-8"))["eps"]
        d = {int(e["trial"]): e for e in E}
        for e in E:
            assert e.get("error") is None, (p, e.get("error"))
        return np.array([float(d[k]["success"]) for k in ks])

    def arr(arm, comps, l):
        d = T4.OD if arm == "U" or arm.endswith("_k1") else OD
        return np.array([[succ(os.path.join(d, f"{arm}_s{s}_{T4.task(n2, c, l)['tid']}.json")) for s in SEEDS] for c in comps])

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(Z={}, J={}, Jc={})
    for c, d in COND.items():
        for l in ("familiar", "new"):
            res["Z"][f"{c}|{l}"] = diff(arr(f"{c}_k1", d["cells"], l), arr(f"Z_{c}", d["cells"], l))
            res["Jc"][f"{c}|{l}"] = diff(arr(f"Jc_{c}", d["cells"], l), arr("Uj", d["cells"], l))
        r = res["Z"][f"{c}|familiar"]
        res["Z"][f"{c}_label"] = "CREDITED" if r["delta"] >= MARGIN and r["ci"][0] > 0.0 else "NOT CREDITED"
    res["Jc_label"] = "TRANSFERS" if all(res["Jc"][f"{c}|familiar"]["ci"][0] >= -MARGIN for c in COND) else "NOT YET"
    G = dict(known=tuple(x for x in COMPS if x[0] == "K"), new_pairings=tuple(x for x in COMPS if x[0] == "N"))
    for g, cs in G.items():
        for l in ("familiar", "new"):
            res["J"][f"{g}|{l}"] = diff(arr("Uj", cs, l), arr("U", cs, l))
    fam = [res["J"][f"{g}|familiar"] for g in G]
    res["J_label"] = "COSTS" if any(x["ci"][1] < -MARGIN for x in fam) else ("HOLDS" if all(x["ci"][0] >= -MARGIN for x in fam) else "UNRESOLVED")
    res["per_task_Uj"] = {c: {l: float(arr("Uj", (c,), l).mean()) for l in ("familiar", "new")} for c in COMPS}
    res["per_cell_Z"] = {c: {x: {l: float(arr(f"Z_{c}", (x,), l).mean()) for l in ("familiar", "new")} for x in d["cells"]} for c, d in COND.items()}
    print("[t4b]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t4b_read.json"), "w", encoding="utf-8") as fh:
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
        arm_, s_, tid_ = combos()[a.index]
        roll(arm_, s_, tid_)
    else:
        read()

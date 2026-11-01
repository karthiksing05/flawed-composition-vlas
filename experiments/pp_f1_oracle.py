"""PLAN_PARTS F1 references (docs/PARTS_RESULTS.md section 1): the same role-mixing map for prototype A's oracle-state policies.  CPU.

Arms and checkpoints (robot/data/fa1/ckpt): A = Aq, seeds 24-26 (stage 1p, Ap1: the clean reference); H = Hb trained on gen_g,
seeds 0-2 (Hg) and R = Rb on gen_g, seeds 0-2 (Rg): the known-flawed references of rounds 1g-1p.
Frames, tasks, partners, phases and d: exactly pp_f1_mix.py's (results/flawed/pp_f1_frames.npz; first min(10, H) = 8 steps x 7 dims,
in the arm's own action units, mean over 2 noise seeds per frame).
The instruction of an oracle arm is its role assignment (object entity oi, target entity ti).  A run under another instruction
recomputes every role-derived input by the rollout's own rules (fa1h_roll.py) on the same simulator state:
  grasp frames (N2 start states)  held 0, offset 0, zrest 0, cyaw 0, rot0 = the current rotation, last command -1 (open)
  place frames (demo rows)        if oi is the demo's held object: the row's held flag and offset (demos_b.npz), rot0 at the hold's
                                  onset (fa1g_train.rot0_rows), zrest and cyaw from fa1c_common.Canon stepped over the demo's rows
                                  with (oi, ti); otherwise (the role object is not in the hand) held 0, offset 0, zrest 0, cyaw 0,
                                  rot0 the current rotation.  Last command: the demo row's.
Components (each MLP is Linear-SiLU x 4 - Linear): `<net>.in` its input (only role-dependent columns differ between runs), `<net>.h<j>`
the output of its j-th hidden layer (j = 1..4), patched from the CROSS partner's forward on the same x_t at every Euler step.
  H, R: net = "net" (one fused MLP).  A: nets "obj" (the object term) and "tgt" (the target term); the gate and the frames are the
  fixed combination rule, not components.  full_cross / full_own as in pp_f1_mix.py.
-> results/flawed/pp_f1_oracle.json

    python experiments/pp_f1_oracle.py [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
from fa1g_train import rot0_rows                                            # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402
from pp_f1_mix import PP, SEEDS, hseed, partner, tasks_pp                   # noqa: E402

OUT = os.path.join(FC.ROOT, "results", "flawed")
ARMS = {"A": [("p_s24", "Aq"), ("p_s25", "Aq"), ("p_s26", "Aq")], "H": [("g_s0", "Hb"), ("g_s1", "Hb"), ("g_s2", "Hb")],
        "R": [("g_s0", "Rb"), ("g_s1", "Rb"), ("g_s2", "Rb")]}
HE = 10


def nets_of(m):
    return {"net": m.net} if hasattr(m, "net") else {"obj": m.obj, "tgt": m.tgt}


def comp_modules(m):
    out = {}
    for nm, net in nets_of(m).items():
        out[f"{nm}.in"] = (net[0], "pre")
        for j in range(4):
            out[f"{nm}.h{j + 1}"] = (net[2 * j + 1], "post")
    return out


@torch.no_grad()
def sample(m, b, seed, patch=None, steps=10):
    """FactoredFlow.sample with an optional patch = (module, kind, partner batch)."""
    B = b["pos"].shape[0]
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(B, m.H, 7, generator=g)
    for k in range(steps):
        t = torch.full((B,), k / steps)
        if patch is None:
            v = m.velocity(b, x, t)
        else:
            mod, kind, bp = patch
            st = {}
            if kind == "pre":
                h1 = mod.register_forward_pre_hook(lambda md, a: st.__setitem__("h", a[0].clone()))
            else:
                h1 = mod.register_forward_hook(lambda md, a, o: st.__setitem__("h", o.clone()))
            try:
                m.velocity(bp, x, t)
            finally:
                h1.remove()
            h2 = (mod.register_forward_pre_hook(lambda md, a: (st["h"],)) if kind == "pre"
                  else mod.register_forward_hook(lambda md, a, o: st["h"]))
            try:
                v = m.velocity(b, x, t)
            finally:
                h2.remove()
        x = x + v / steps
    return x.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    T = tasks_pp()
    Z = np.load(os.path.join(OUT, "pp_f1_frames.npz"))
    fr = {k: Z[k] for k in Z.files}
    ENT = list(Z["ent"])
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    R0 = rot0_rows(D, Db["held"])
    key = D["task"].astype(np.int64) * 1000 + D["demo"]

    def role_inputs(i, oi, ti):
        """-> held, off, zrest, cyaw, rot0, prev for frame i under the role assignment (oi, ti)."""
        if fr["phase"][i] == 0:
            return 0.0, np.zeros(3), 0.0, 0.0, fr["ee_rot"][i], -1.0
        r = int(fr["row"][i])
        prev = float(D["prev_grip"][r])
        if oi != int(fr["obj"][i]):
            return 0.0, np.zeros(3), 0.0, 0.0, D["ee_rot"][r], prev
        rows = np.flatnonzero(key == key[r])
        rows = rows[rows <= r]
        c = FK.Canon()
        for q in rows:
            zr, cy = c.step(bool(Db["held"][q] > 0.5), D["pos"][q, oi], lambda q=q: (D["pos"][q, ti], float(D["pos"][q, FK.TABLE, 2])))
        return float(Db["held"][r]), Db["off"][r], zr, cy, R0[r], prev

    def batch(idx, comp):
        oi, ti = ENT.index(T[comp]["obj"]), ENT.index(T[comp]["tgt"])
        rows = [role_inputs(i, oi, ti) for i in idx]
        f = lambda v: torch.tensor(np.asarray(v, np.float32))                # noqa: E731
        return dict(ee_pos=f(fr["ee_pos"][idx]), ee_rot=f(fr["ee_rot"][idx]), grip=f(fr["grip"][idx]), prev=f([r[5] for r in rows]),
                    pos=f(fr["pos"][idx]), yaw=f(fr["yaw"][idx]), drawer=f(fr["drawer"][idx]),
                    oi=torch.full((len(idx),), oi, dtype=torch.long), ti=torch.full((len(idx),), ti, dtype=torch.long),
                    held=f([r[0] for r in rows]), off=f(np.stack([r[1] for r in rows])), zrest=f([r[2] for r in rows]),
                    cyaw=f([r[3] for r in rows]), rot0=f(np.stack([r[4] for r in rows])))

    jobs = [(c, "grasp", lay) for c in PP for lay in (0, 1)] + [(c, "place", 0) for c in PP]
    if a.limit:                                     # smoke: the first N grasp jobs and the first N place jobs
        jobs = [j for j in jobs if j[1] == "grasp"][: a.limit] + [j for j in jobs if j[1] == "place"][: a.limit]
    out = dict(arms={k: [f"{d}/{n}" for d, n in v] for k, v in ARMS.items()}, H=None, seeds=list(SEEDS), runs={})
    for arm, ck in ARMS.items():
        for d_, nm in ck:
            m, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", d_, f"{nm}.npz"))
            H = min(HE, m.H)
            out["H"] = H
            comps = comp_modules(m)
            res = {}
            for c, ph, lay in jobs:
                cross = partner(T, c, "obj" if ph == "grasp" else "tgt")
                own = partner(T, c, "tgt" if ph == "grasp" else "obj")
                k = f"{c}|{ph}|{'familiar' if lay == 0 else 'new'}"
                if cross is None or own is None:
                    res[k] = dict(skipped=True, cross=cross, own=own)
                    continue
                if ph == "grasp":
                    idx = np.flatnonzero((fr["phase"] == 0) & (fr["layout"] == lay))
                else:
                    idx = np.flatnonzero((fr["phase"] == 1) & (fr["obj"] == ENT.index(T[c]["obj"])))
                bb, bx, bo = batch(idx, c), batch(idx, cross), batch(idx, own)
                seeds = [hseed("pp_f1", c, ph, lay, s) for s in SEEDS]
                y0 = [sample(m, bb, s)[:, :H] for s in seeds]
                dd = {}
                for vn, bv in (("full_cross", bx), ("full_own", bo)):
                    ys = [sample(m, bv, s)[:, :H] for s in seeds]
                    dd[vn] = np.mean([((y - b0) ** 2).mean((1, 2)) for y, b0 in zip(ys, y0)], 0).tolist()
                for cn, (mod, kind) in comps.items():
                    ys = [sample(m, bb, s, patch=(mod, kind, bx))[:, :H] for s in seeds]
                    dd[cn] = np.mean([((y - b0) ** 2).mean((1, 2)) for y, b0 in zip(ys, y0)], 0).tolist()
                res[k] = dict(compound=c, phase=ph, layout="familiar" if lay == 0 else "new", cross=cross, own=own, d=dd)
            out["runs"][f"{arm}|{d_}/{nm}"] = res
            print(f"[pp_f1_oracle] {arm} {d_}/{nm} ({FF.n_params(m)} params) {time.time() - t0:.0f}s", flush=True)
    out["secs"] = time.time() - t0
    fn = os.path.join(a.out, f"pp_f1_oracle{'_smoke' if a.limit else ''}.json")
    os.makedirs(a.out, exist_ok=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[pp_f1_oracle] wrote {fn}")


if __name__ == "__main__":
    main()

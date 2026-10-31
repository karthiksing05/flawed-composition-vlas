"""FLAWED_COMPOSITION prototype A, stage 1b (section 5.6): the numerical check of Proposition 3's role locality (the swap test), on
TRAINING states only (DATA/demos.npz + demos_b.npz; no eval cell).  For each trained arm and 4,000 random training steps, the
sampled action chunk with fixed flow noise is compared with the chunk after one input change that the proposition says A cannot see:
  T1 object phase (held = 0): the target role is replaced by 4 random other entities (any slot but the object and the target)
                                                                                               (A's object term never reads it)
  T2 place phase (held = 1):  the object role's IDENTITY is replaced by another movable object whose slot is overwritten with the held
                              object's pose and size (same geometry, other identity)            (A's target term reads geometry only)
  T3 either phase:            every slot that is neither the object nor the target role is moved by 5 cm in a random horizontal
                              direction                                                        (no A term reads them)
Reported per arm and test: mean and max over steps of max |delta action| (action units, before clipping), next to the mean |action|.
CPU.  -> results/flawed/a1b_swap.json

    python experiments/fa1b_swap.py --ckpt b_s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fa1_common as FC                                                     # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

N, TARGETS, OBJS = 4000, list(range(4, 9)), list(range(4))


def sample(m, b, seed):
    g = torch.Generator().manual_seed(seed)
    B = b["pos"].shape[0]
    x0 = torch.randn(B, m.H, 7, generator=g)
    x = x0.clone()
    with torch.no_grad():
        for k in range(10):
            x = x + m.velocity(b, x, torch.full((B,), k / 10)) / 10
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="b_s0")
    a = ap.parse_args()
    torch.set_num_threads(4)
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    rng = np.random.default_rng(0)
    pp = np.nonzero(D["task"] <= 5)[0]                                     # pick-and-place tasks K1-K6
    held = Db["held"] > 0.5
    rows = {"T1": rng.choice(pp[~held[pp]], N), "T2": rng.choice(pp[held[pp]], N), "T3": rng.choice(pp, N)}
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt)   # noqa: E731

    def batch(j, oi_=None, ti_=None, pos=None, yaw=None):
        return dict(ee_pos=T(D["ee_pos"][j]), ee_rot=T(D["ee_rot"][j]), grip=T(D["grip"][j]), prev=T(D["prev_grip"][j]),
                    pos=T(D["pos"][j] if pos is None else pos), yaw=T(D["yaw"][j] if yaw is None else yaw), drawer=T(D["drawer"][j]),
                    oi=T(oi[j] if oi_ is None else oi_, torch.long), ti=T(ti[j] if ti_ is None else ti_, torch.long),
                    held=T(Db["held"][j]), off=T(Db["off"][j]))

    res = {}
    ckdir = os.path.join(FC.DATA, "ckpt", a.ckpt)
    for arm in FF.ARMS_B:
        f = os.path.join(ckdir, f"{arm}.npz")
        if not os.path.exists(f):
            continue
        m, _ = FF.FactoredFlow.load(f)
        r = {}
        # T1: another target role (object phase)
        j = rows["T1"]
        base = sample(m, batch(j), 1)
        d = []
        for k in range(4):                                                 # 4 random other target entities per step
            alt = np.array([rng.choice([e for e in range(FF.N_ENT) if e not in (o, t)]) for o, t in zip(oi[j], ti[j])])
            d.append((sample(m, batch(j, ti_=alt), 1) - base).abs().amax((1, 2)))
        d = torch.stack(d).amax(0)
        r["T1"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        # T2: another object identity with the held object's geometry (place phase)
        j = rows["T2"]
        base = sample(m, batch(j), 2)
        alt = np.array([rng.choice([x for x in OBJS if x != o and x != t]) for o, t in zip(oi[j], ti[j])])
        pos, yaw = D["pos"][j].copy(), D["yaw"][j].copy()
        ii = np.arange(len(j))
        pos[ii, alt], yaw[ii, alt] = pos[ii, oi[j]], yaw[ii, oi[j]]
        size0 = m.size.clone()
        sz = m.size.clone()
        out = []
        for o_new in range(4):                                             # per new identity, give it the held object's size
            sel = alt == o_new
            if not sel.any():
                continue
            for o_old in range(4):
                s2 = sel & (oi[j] == o_old)
                if not s2.any():
                    continue
                jj = j[s2]
                m.size.copy_(size0)
                b0 = sample(m, batch(jj), 2)                               # the same sub-batch, so the same noise draw
                sz = size0.clone()
                sz[o_new] = size0[o_old]
                m.size.copy_(sz)
                bb = batch(jj, oi_=alt[s2], pos=pos[s2], yaw=yaw[s2])
                out.append((sample(m, bb, 2) - b0).abs().amax((1, 2)))
        m.size.copy_(size0)
        d = torch.cat(out)
        r["T2"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        # T3: move every non-role slot by 5 cm
        j = rows["T3"]
        base = sample(m, batch(j), 3)
        pos = D["pos"][j].copy()
        ang = rng.uniform(0, 2 * np.pi, (len(j), FF.N_ENT))
        sh = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1) * 0.05
        keep = np.zeros((len(j), FF.N_ENT), bool)
        keep[np.arange(len(j)), oi[j]] = True
        keep[np.arange(len(j)), ti[j]] = True
        pos = pos + np.where(keep[..., None], 0.0, sh)
        d = (sample(m, batch(j, pos=pos), 3) - base).abs().amax((1, 2))
        r["T3"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        res[arm] = r
        print(f"[fa1b] swap {arm}: {json.dumps(r)}", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", "a1b_swap.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ckpt=a.ckpt, n=N, tests=res), fh, indent=1)


if __name__ == "__main__":
    main()

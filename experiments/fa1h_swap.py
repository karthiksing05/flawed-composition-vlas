"""FLAWED_COMPOSITION prototype A, stage 1h (section 5.14): the swap test of Proposition 3 (i) for the stage-1h arms, on TRAINING states
(demo rows), with the rows' onset rotation rot0 (fa1g_train.rot0_rows) in the batch.  fa1e_swap.py's T1-T4 and T6 unchanged, and
  T5' place phase: the end effector's rotation and the fingers replaced by another object's held row, with rot0 moved WITH them so
      that the rotation since the onset (drot) is kept -- the target term must not read the absolute rotation or the fingers
      (1d's identity proxies); drot itself is the route under test, so T5 cannot replace it.
  T7  place phase: T2 (held-object identity swapped, geometry kept, route inputs drot / offset kept) with the fingers also taken
      from a held row of another object (must be 0: the target term reads no identity and no fingers).
CPU.  -> results/flawed/a1h_swap.json

    FA1H_MODELS="h_s0:Ah+h_s0:Ah_rot" python experiments/fa1h_swap.py
"""
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
import fa1c_common as FK                                                    # noqa: E402
from fa1b_swap import sample                                                # noqa: E402
from fa1c_swap import sample_from                                           # noqa: E402
from fa1g_train import rot0_rows                                            # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

N, OBJS = 4000, list(range(4))
MODELS = [tuple(x.split(":")) for x in os.environ.get("FA1H_MODELS", "h_s0:Ah").split("+")]


def main():
    torch.set_num_threads(4)
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    R0 = rot0_rows(D, Db["held"])
    rng = np.random.default_rng(0)
    pp = np.nonzero(D["task"] <= 5)[0]
    held = Db["held"] > 0.5
    hp = pp[held[pp]]
    rows = {k: rng.choice(hp if k != "T1" else pp[~held[pp]], N) for k in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")}
    rows["T3"] = rng.choice(pp, N)
    donor = np.array([rng.choice(hp[oi[hp] != oi[j]]) for j in rows["T5"]])
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt)   # noqa: E731

    def batch(j, oi_=None, ti_=None, pos=None, yaw=None, ee_pos=None, ee_rot=None, cyaw=None, grip=None, rot0=None, off=None):
        return dict(ee_pos=T(D["ee_pos"][j] if ee_pos is None else ee_pos), ee_rot=T(D["ee_rot"][j] if ee_rot is None else ee_rot),
                    grip=T(D["grip"][j] if grip is None else grip), prev=T(D["prev_grip"][j]), pos=T(D["pos"][j] if pos is None else pos),
                    yaw=T(D["yaw"][j] if yaw is None else yaw), drawer=T(D["drawer"][j]),
                    oi=T(oi[j] if oi_ is None else oi_, torch.long), ti=T(ti[j] if ti_ is None else ti_, torch.long),
                    held=T(Db["held"][j]), off=T(Db["off"][j] if off is None else off), zrest=T(zr[j]), cyaw=T(cy[j] if cyaw is None else cyaw),
                    rot0=T(R0[j] if rot0 is None else rot0))

    def summ(d, base):
        return dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))

    res = {}
    for ck, arm in MODELS:
        m, _ = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", ck, f"{arm}.npz"))
        r = {}
        j = rows["T1"]
        base = sample(m, batch(j), 1)
        d = []
        for _ in range(4):
            alt = np.array([rng.choice([e for e in range(FF.N_ENT) if e not in (o, t)]) for o, t in zip(oi[j], ti[j])])
            d.append((sample(m, batch(j, ti_=alt), 1) - base).abs().amax((1, 2)))
        r["T1"] = summ(torch.stack(d).amax(0), base)
        for key in ("T2", "T7"):
            j = rows[key]
            ii = np.arange(len(j))
            alt = np.array([rng.choice([x for x in OBJS if x != o and x != t]) for o, t in zip(oi[j], ti[j])])
            pos, yaw = D["pos"][j].copy(), D["yaw"][j].copy()
            pos[ii, alt], yaw[ii, alt] = pos[ii, oi[j]], yaw[ii, oi[j]]
            base = sample(m, batch(j), 2)
            if key == "T2":
                r[key] = summ((sample(m, batch(j, oi_=alt, pos=pos, yaw=yaw), 2) - base).abs().amax((1, 2)), base)
            else:                                                   # fingers from a held row of another object, route geometry kept
                dn = np.array([rng.choice(hp[oi[hp] != oi[jj]]) for jj in j])
                b2 = batch(j, oi_=alt, pos=pos, yaw=yaw, grip=D["grip"][dn])
                r[key] = summ((sample(m, b2, 2) - base).abs().amax((1, 2)), base)
        j = rows["T3"]
        ii = np.arange(len(j))
        base = sample(m, batch(j), 3)
        ang = rng.uniform(0, 2 * np.pi, (len(j), FF.N_ENT))
        sh = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1) * 0.05
        keep = np.zeros((len(j), FF.N_ENT), bool)
        keep[ii, oi[j]] = True
        keep[ii, ti[j]] = True
        keep[:, FK.TABLE] = True
        r["T3"] = summ((sample(m, batch(j, pos=D["pos"][j] + np.where(keep[..., None], 0.0, sh)), 3) - base).abs().amax((1, 2)), base)
        j = rows["T4"]
        ii = np.arange(len(j))
        th = rng.uniform(-np.pi, np.pi, len(j))
        Rz = np.stack([np.stack([np.cos(th), -np.sin(th), 0 * th], -1), np.stack([np.sin(th), np.cos(th), 0 * th], -1),
                       np.stack([0 * th, 0 * th, 1 + 0 * th], -1)], -2)
        pt = D["pos"][j, ti[j]][:, None, :]
        pos = np.einsum("bij,bkj->bki", Rz, D["pos"][j] - pt) + pt
        eep = np.einsum("bij,bj->bi", Rz, D["ee_pos"][j] - pt[:, 0]) + pt[:, 0]
        eer = np.einsum("bij,bjk->bik", Rz, D["ee_rot"][j])
        r0r = np.einsum("bij,bjk->bik", Rz, R0[j])
        offr = np.einsum("bij,bj->bi", Rz, Db["off"][j])
        yaw = D["yaw"][j] + th[:, None]
        yaw[ii, ti[j]] = D["yaw"][j][ii, ti[j]]
        Rt = torch.tensor(Rz, dtype=torch.float32)
        x0 = torch.randn(len(j), m.H, 7, generator=torch.Generator().manual_seed(4))
        base = sample_from(m, batch(j), x0)
        rot = sample_from(m, batch(j, pos=pos, yaw=yaw, ee_pos=eep, ee_rot=eer, cyaw=cy[j] + th, rot0=r0r, off=offr), FF.chunk_from(Rt, x0))
        r["T4"] = summ((rot - FF.chunk_from(Rt, base)).abs().amax((1, 2)), base)
        j = rows["T5"]
        base = sample(m, batch(j), 5)
        rel = np.einsum("bij,bkj->bik", D["ee_rot"][j], R0[j])                    # R R0^T, kept
        r0n = np.einsum("bji,bjk->bik", rel, D["ee_rot"][donor])                  # R0' = rel^T R'
        r["T5p"] = summ((sample(m, batch(j, ee_rot=D["ee_rot"][donor], grip=D["grip"][donor], rot0=r0n), 5) - base).abs().amax((1, 2)), base)
        j = rows["T6"]
        alt = np.array([rng.choice([e for e in range(FF.N_ENT) if e not in (o, t) and e != FK.TABLE]) for o, t in zip(oi[j], ti[j])])
        size0 = m.size.clone()
        out = []
        for t_old in np.unique(ti[j]):
            for t_new in np.unique(alt):
                s2 = (ti[j] == t_old) & (alt == t_new)
                if not s2.any():
                    continue
                jj = j[s2]
                sz = size0.clone()
                sz[t_new] = size0[t_old]
                pos, yaw = D["pos"][jj].copy(), D["yaw"][jj].copy()
                pos[:, t_new], yaw[:, t_new] = pos[:, t_old], yaw[:, t_old]
                drw = D["drawer"][jj] if t_new != FF.DRAWER and t_old != FF.DRAWER else np.zeros(len(jj), np.float32)
                bb = batch(jj, ti_=np.full(len(jj), t_new), pos=pos, yaw=yaw)
                bb["drawer"] = T(drw)
                b0b = batch(jj)
                b0b["drawer"] = T(drw)
                m.size.copy_(size0)
                b0 = sample(m, b0b, 6)
                m.size.copy_(sz)
                out.append((sample(m, bb, 6) - b0).abs().amax((1, 2)))
        m.size.copy_(size0)
        r["T6"] = summ(torch.cat(out), base)
        res[f"{ck}/{arm}"] = r
        print(f"[fa1h] swap {ck}/{arm}: {json.dumps({k: (round(v['mean'], 6), round(v['max'], 6)) for k, v in r.items()})}", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", os.environ.get("FA1H_OUT", "a1h_swap.json")), "w", encoding="utf-8") as fh:
        json.dump(dict(models=MODELS, n=N, tests=res), fh, indent=1)


if __name__ == "__main__":
    main()

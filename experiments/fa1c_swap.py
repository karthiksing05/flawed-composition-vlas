"""FLAWED_COMPOSITION prototype A, stage 1c (section 5.7): the swap test of Proposition 3 (i) for the stage-1c arms, on TRAINING states
only.  fa1b_swap.py's three tests (T1 the target role swapped in the object phase; T2 the object identity swapped, its pose and rest
height kept, in the place phase; T3 every non-role slot moved 5 cm), plus
  T4 place phase: the held object's approach bearing (cyaw) and the scene are rotated together about the target's vertical axis by a
     random angle (every slot's position and every slot's yaw except the target's own, the end effector's position and rotation, and
     the bearing; the flow noise rotated with them): Ac's chunk must rotate with the scene (its target term reads only
     frame-relative inputs and the z-row of the end effector's rotation) -> reported as the difference between the rotated scene's
     chunk and the rotated original chunk.
CPU.  -> results/flawed/a1c_swap.json

    python experiments/fa1c_swap.py --ckpt c_s0
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
import fa1c_common as FK                                                    # noqa: E402
from fa1b_swap import sample                                                # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

N, OBJS = 4000, list(range(4))


def sample_from(m, b, x0):
    """The 10-step Euler sampler from a given initial noise x0 (T4 rotates the noise with the scene: the sampler is equivariant per
    sample only when its noise is rotated too; in distribution it is, the noise being isotropic)."""
    x = x0.clone()
    B = x.shape[0]
    with torch.no_grad():
        for k in range(10):
            x = x + m.velocity(b, x, torch.full((B,), k / 10)) / 10
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="c_s0")
    a = ap.parse_args()
    torch.set_num_threads(4)
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    rng = np.random.default_rng(0)
    pp = np.nonzero(D["task"] <= 5)[0]
    held = Db["held"] > 0.5
    rows = {"T1": rng.choice(pp[~held[pp]], N), "T2": rng.choice(pp[held[pp]], N), "T3": rng.choice(pp, N), "T4": rng.choice(pp[held[pp]], N)}
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt)   # noqa: E731

    def batch(j, oi_=None, ti_=None, pos=None, yaw=None, ee_pos=None, ee_rot=None, cyaw=None):
        return dict(ee_pos=T(D["ee_pos"][j] if ee_pos is None else ee_pos), ee_rot=T(D["ee_rot"][j] if ee_rot is None else ee_rot),
                    grip=T(D["grip"][j]), prev=T(D["prev_grip"][j]), pos=T(D["pos"][j] if pos is None else pos),
                    yaw=T(D["yaw"][j] if yaw is None else yaw), drawer=T(D["drawer"][j]),
                    oi=T(oi[j] if oi_ is None else oi_, torch.long), ti=T(ti[j] if ti_ is None else ti_, torch.long),
                    held=T(Db["held"][j]), off=T(Db["off"][j]), zrest=T(zr[j]), cyaw=T(cy[j] if cyaw is None else cyaw))

    res = {}
    for arm in FF.ARMS_C:
        f = os.path.join(FC.DATA, "ckpt", a.ckpt, f"{arm}.npz")
        if not os.path.exists(f):
            continue
        m, _ = FF.FactoredFlow.load(f)
        r = {}
        j = rows["T1"]
        base = sample(m, batch(j), 1)
        d = []
        for _ in range(4):
            alt = np.array([rng.choice([e for e in range(FF.N_ENT) if e not in (o, t)]) for o, t in zip(oi[j], ti[j])])
            d.append((sample(m, batch(j, ti_=alt), 1) - base).abs().amax((1, 2)))
        d = torch.stack(d).amax(0)
        r["T1"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        j = rows["T2"]
        alt = np.array([rng.choice([x for x in OBJS if x != o and x != t]) for o, t in zip(oi[j], ti[j])])
        pos, yaw = D["pos"][j].copy(), D["yaw"][j].copy()
        ii = np.arange(len(j))
        pos[ii, alt], yaw[ii, alt] = pos[ii, oi[j]], yaw[ii, oi[j]]
        base = sample(m, batch(j), 2)
        d = (sample(m, batch(j, oi_=alt, pos=pos, yaw=yaw), 2) - base).abs().amax((1, 2))
        r["T2"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        j = rows["T3"]
        base = sample(m, batch(j), 3)
        pos = D["pos"][j].copy()
        ang = rng.uniform(0, 2 * np.pi, (len(j), FF.N_ENT))
        sh = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1) * 0.05
        keep = np.zeros((len(j), FF.N_ENT), bool)
        keep[ii, oi[j]] = True
        keep[ii, ti[j]] = True
        keep[:, FK.TABLE] = True                                            # the table height is read (a scene constant)
        d = (sample(m, batch(j, pos=pos + np.where(keep[..., None], 0.0, sh)), 3) - base).abs().amax((1, 2))
        r["T3"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        j = rows["T4"]
        th = rng.uniform(-np.pi, np.pi, len(j))
        Rz = np.stack([np.stack([np.cos(th), -np.sin(th), 0 * th], -1), np.stack([np.sin(th), np.cos(th), 0 * th], -1),
                       np.stack([0 * th, 0 * th, 1 + 0 * th], -1)], -2)
        pt = D["pos"][j, ti[j]][:, None, :]
        pos = np.einsum("bij,bkj->bki", Rz, D["pos"][j] - pt) + pt
        eep = np.einsum("bij,bj->bi", Rz, D["ee_pos"][j] - pt[:, 0]) + pt[:, 0]
        eer = np.einsum("bij,bjk->bik", Rz, D["ee_rot"][j])
        yaw = D["yaw"][j] + th[:, None]
        yaw[ii, ti[j]] = D["yaw"][j][ii, ti[j]]                            # the target's own yaw stays: only the approach rotates
        Rt = torch.tensor(Rz, dtype=torch.float32)
        x0 = torch.randn(len(j), m.H, 7, generator=torch.Generator().manual_seed(4))
        base = sample_from(m, batch(j), x0)
        rot = sample_from(m, batch(j, pos=pos, yaw=yaw, ee_pos=eep, ee_rot=eer, cyaw=cy[j] + th), FF.chunk_from(Rt, x0))
        base_rot = FF.chunk_from(Rt, base)
        d = (rot - base_rot).abs().amax((1, 2))
        r["T4"] = dict(mean=float(d.mean()), max=float(d.max()), mean_abs_action=float(base.abs().mean()))
        res[arm] = r
        print(f"[fa1c] swap {arm}: {json.dumps(r)}", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", "a1c_swap.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ckpt=a.ckpt, n=N, tests=res), fh, indent=1)


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION prototype A, stage 1g (section 5.12): train arms on the demos (DATA/demos.npz + demos_b.npz) PLUS the generated
varied-layout carries of the known pick-and-place pairings (DATA/gen_g.npz, fa1g_plan.py; labelled gen = 1), with stage 1c's recipe
and inputs unchanged (fa1c_train.py: canonical rows, rectified flow on 8-step chunks, AdamW 3e-4, batch 512, 30,000 steps, EMA
0.999).  Rows are drawn uniformly from the union.  The baselines (Hb, Rb architectures) are trained on the same union.  GPU.

-> DATA/ckpt/<tag>_s<seed>/<arm>.npz, results/flawed/a1g_train_<tag>_s<seed>.json

    python experiments/fa1g_train.py --tag g --seed 0 --arms Hb+Rb+Ae_nosize
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
from fa1_train import chunks                                                # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

FIELDS = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip")


def rot0_rows(D, held):
    """Stage 1h (5.14): per row, the end effector's rotation at the onset of the current hold (the current rotation while not held),
    stepped over each episode's rows in order -- the rollout's own rule (fa1h_roll.py).  Ignored by every arm before 1h."""
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    out = D["ee_rot"].copy()
    on, r0 = False, None
    for i in range(len(key)):
        if i == 0 or key[i] != key[i - 1]:
            on = False
        h = bool(held[i] > 0.5)
        if h and not on:
            r0 = D["ee_rot"][i].copy()
        on = h
        if h:
            out[i] = r0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--gen", default="gen_g.npz")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D0 = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db0 = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    G = dict(np.load(os.path.join(FC.DATA, a.gen)))
    assert list(D0["ent"]) == FC.ENT and list(D0["tasks"]) == list(FC.KNOWN) and len(Db0["held"]) == len(D0["action"])
    assert G["demo"].min() >= 100 and G["demo"].max() < 1000, "generated demo ids must be 100-999"
    D = {x: np.concatenate([D0[x], G[x].astype(D0[x].dtype)]) for x in FIELDS}
    Db = {x: np.concatenate([Db0[x], G[x].astype(np.float32)]) for x in ("held", "off")}
    gen = np.r_[np.zeros(len(D0["action"]), np.int8), np.ones(len(G["action"]), np.int8)]
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    X = chunks(D, a.horizon)
    center = D0["ee_pos"].mean(0).tolist()                                 # the demos' centre, as every earlier stage
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt, device=dev)   # noqa: E731
    full = dict(ee_pos=T(D["ee_pos"]), ee_rot=T(D["ee_rot"]), grip=T(D["grip"]), prev=T(D["prev_grip"]), pos=T(D["pos"]),
                yaw=T(D["yaw"]), drawer=T(D["drawer"]), oi=T(oi, torch.long), ti=T(ti, torch.long), held=T(Db["held"]), off=T(Db["off"]),
                zrest=T(zr), cyaw=T(cy))
    if any(x in FF.ARMS_H + FF.ARMS_I for x in a.arms.split("+")):           # stage 1h / 1i only; 1g's runs are unchanged
        full["rot0"] = T(rot0_rows(D, Db["held"]))
    act = T(np.clip(X, -1, 1))
    tr_idx = np.arange(len(D["action"]))
    print(f"[fa1g] {len(tr_idx)} train steps ({int(gen.sum())} generated); held share {float((Db['held'] > 0.5).mean()):.3f}; "
          f"seed {a.seed}", flush=True)
    rep = dict(tag=a.tag, steps=a.steps, batch=a.batch, lr=a.lr, horizon=a.horizon, seed=a.seed, n_train=int(len(tr_idx)),
               n_generated=int(gen.sum()), gen_file=a.gen, arms={})
    out_dir = os.path.join(FC.DATA, "ckpt", f"{a.tag}_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    for arm in a.arms.split("+"):
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        m = FF.FactoredFlow(arm, a.horizon, center=center, size=size).to(dev)
        ema = copy.deepcopy(m)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
        curve, t0, run = [], time.time(), []
        for s in range(1, a.steps + 1):
            j = torch.as_tensor(rng.choice(tr_idx, a.batch), device=dev)
            loss = m.loss({k: v[j] for k, v in full.items()}, act[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), m.parameters()):
                    pe.mul_(0.999).add_(pm, alpha=0.001)
            run.append(float(loss))
            if s % 1000 == 0 or s == a.steps:
                curve.append(dict(step=s, train=float(np.mean(run))))
                run = []
                if s % 10000 == 0:
                    print(f"[fa1g] {arm} s{a.seed} step {s}: {curve[-1]} ({time.time() - t0:.0f}s)", flush=True)
        ema.cpu().save(os.path.join(out_dir, f"{arm}.npz"), dict(tag=a.tag, center=center, steps=a.steps, holdout=0, seed=a.seed,
                                                                   generated=a.gen))
        rep["arms"][arm] = dict(params=FF.n_params(m), curve=curve, secs=time.time() - t0)
        print(f"[fa1g] {arm}: {FF.n_params(m)} parameters, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", f"a1g_train_{a.tag}_s{a.seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()

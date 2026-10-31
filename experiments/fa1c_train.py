"""FLAWED_COMPOSITION prototype A, stage 1c (section 5.7): train the stage-1c arms (factored_flow.ARMS_C) on stage 1b's data (DATA/demos.npz
+ demos_b.npz) plus the canonical-frame inputs (fa1c_common.canon_rows: rest height and bearing per row), with stage 1's recipe
unchanged (fa1b_train.py).  GPU (openpi venv).

-> DATA/ckpt/<tag>_s<seed>/<arm>.npz, results/flawed/a1c_train_<tag>_s<seed>.json

    python experiments/fa1c_train.py --tag c --seed 0 --arms Ac
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="+".join(FF.ARMS_C))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    assert list(D["ent"]) == FC.ENT and list(D["tasks"]) == list(FC.KNOWN) and len(Db["held"]) == len(D["action"])
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    X = chunks(D, a.horizon)
    center = D["ee_pos"].mean(0).tolist()
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt, device=dev)   # noqa: E731
    full = dict(ee_pos=T(D["ee_pos"]), ee_rot=T(D["ee_rot"]), grip=T(D["grip"]), prev=T(D["prev_grip"]), pos=T(D["pos"]),
                yaw=T(D["yaw"]), drawer=T(D["drawer"]), oi=T(oi, torch.long), ti=T(ti, torch.long), held=T(Db["held"]), off=T(Db["off"]),
                zrest=T(zr), cyaw=T(cy))
    act = T(np.clip(X, -1, 1))
    tr_idx = np.arange(len(D["action"]))
    h = Db["held"] > 0.5
    foot_dz = (D["pos"][np.arange(len(oi)), oi, 2] - zr + D["pos"][:, FK.TABLE, 2])[h] - D["pos"][h, FK.TABLE, 2]
    print(f"[fa1c] {len(tr_idx)} train steps; held share {float(h.mean()):.3f}; foot height above the table while held: median "
          f"{float(np.median(foot_dz)):.3f} m, 5th pct {float(np.percentile(foot_dz, 5)):.3f}; seed {a.seed}", flush=True)
    rep = dict(tag=a.tag, steps=a.steps, batch=a.batch, lr=a.lr, horizon=a.horizon, seed=a.seed, n_train=int(len(tr_idx)), arms={})
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
                    print(f"[fa1c] {arm} s{a.seed} step {s}: {curve[-1]} ({time.time() - t0:.0f}s)", flush=True)
        ema.cpu().save(os.path.join(out_dir, f"{arm}.npz"), dict(tag=a.tag, center=center, steps=a.steps, holdout=0, seed=a.seed))
        rep["arms"][arm] = dict(params=FF.n_params(m), curve=curve, secs=time.time() - t0)
        print(f"[fa1c] {arm}: {FF.n_params(m)} parameters, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", f"a1c_train_{a.tag}_s{a.seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()

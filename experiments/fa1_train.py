"""FLAWED_COMPOSITION prototype A, stage 1 (section 5.5): train the six arms (lifelong_ood/policy/factored_flow.py ARMS) on the same
oracle-state data (fa1_data.py poses: N2's 8 KNOWN LIBERO-Goal tasks, 50 human demos each, no-op steps dropped) with the same
recipe: rectified-flow matching on 8-step action chunks, AdamW (lr 3e-4, wd 1e-4, cosine), batch 512, the same step budget and
seed for every arm, EMA 0.999 weights saved.  Roles come from the template parse of N2's prompt for each task (oracle grounding).
--holdout k keeps demos 50-k..49 of every task out (dev: the held-out flow loss is logged; fa1_roll.py --phase dev rolls from
their start states); the final models use --holdout 0.  GPU (openpi venv, torch 2.x).

-> DATA/ckpt/<tag>/<arm>.npz, results/flawed/a1_train_<tag>.json (parameter counts, loss curves, wall time)

    python experiments/fa1_train.py --tag final --holdout 0 [--arms H+R+A] [--steps 30000]
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
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402


def chunks(D, H):
    """Per step: the next H actions of the same demo, padded with zero motion and the last gripper command."""
    A, n = D["action"], len(D["action"])
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    out = np.zeros((n, H, 7), np.float32)
    ends = np.r_[np.nonzero(key[1:] != key[:-1])[0], n - 1]                 # the last index of each demo
    endof = ends[np.searchsorted(ends, np.arange(n))]
    pad = np.zeros((n, 7), np.float32)
    pad[:, 6] = A[endof, 6]
    for h in range(H):
        j = np.minimum(np.arange(n) + h, n - 1)
        out[:, h] = np.where((key[j] == key)[:, None], A[j], pad)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--arms", default="+".join(FF.ARMS))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    assert list(D["ent"]) == FC.ENT and list(D["tasks"]) == list(FC.KNOWN)
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    X = chunks(D, a.horizon)
    held = D["demo"] >= 50 - a.holdout
    center = D["ee_pos"].mean(0).tolist()
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt, device=dev)   # noqa: E731
    full = dict(ee_pos=T(D["ee_pos"]), ee_rot=T(D["ee_rot"]), grip=T(D["grip"]), prev=T(D["prev_grip"]), pos=T(D["pos"]),
                yaw=T(D["yaw"]), drawer=T(D["drawer"]), oi=T(oi, torch.long), ti=T(ti, torch.long))
    act = T(np.clip(X, -1, 1))
    tr_idx, ho_idx = np.nonzero(~held)[0], np.nonzero(held)[0]
    print(f"[fa1] {len(tr_idx)} train steps, {len(ho_idx)} held-out; roles {dict(zip(FC.KNOWN, rl))}", flush=True)
    rep = dict(tag=a.tag, holdout=a.holdout, steps=a.steps, batch=a.batch, lr=a.lr, horizon=a.horizon, seed=a.seed,
               n_train=int(len(tr_idx)), n_heldout=int(len(ho_idx)), arms={})
    out_dir = os.path.join(FC.DATA, "ckpt", a.tag)
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
                row = dict(step=s, train=float(np.mean(run)))
                run = []
                if len(ho_idx):
                    with torch.no_grad(), torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                        torch.manual_seed(123)
                        ls = []
                        for c in np.array_split(ho_idx, max(1, len(ho_idx) // 2048)):
                            jj = torch.as_tensor(c, device=dev)
                            ls.append(float(ema.loss({k: v[jj] for k, v in full.items()}, act[jj])) * len(c))
                        row["heldout_ema"] = sum(ls) / len(ho_idx)
                curve.append(row)
                print(f"[fa1] {arm} step {s}: {row} ({time.time() - t0:.0f}s)", flush=True)
        ema.cpu().save(os.path.join(out_dir, f"{arm}.npz"), dict(tag=a.tag, center=center, steps=a.steps, holdout=a.holdout))
        rep["arms"][arm] = dict(params=FF.n_params(m), curve=curve, secs=time.time() - t0)
        print(f"[fa1] {arm}: {FF.n_params(m)} parameters, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", f"a1_train_{a.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()

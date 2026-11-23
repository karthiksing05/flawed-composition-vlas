"""RAA-h head fit (docs/notes/plan_review_atlas.md 6.3; tag rvh).  CPU, openpi venv (torch), seconds.  Fits the frozen learned where-head ONCE and never tunes it.

Data: the held rows (role id 2) of the RAA-c training set (lol/libero_g8v_raac; raas_gate.py's loader imported unchanged), chunk-consistent targets: for every held row t and j = 0..9
the demo action xyz at min(t + j, T - 1).  Input rel_xyz / 0.20.  Head: 3 -> 64 -> 64 -> 30 (SiLU), output M tanh(u), centred h(r) = g(r) - g(0), clipped to +-M (M = .9375, the gate's
max_step).  Split by demo 90 / 10 with the fixed hash (ep * 2654435761 mod 2^32) mod 10 == 0 -> validation.  Loss MSE in raw action units; AdamW lr 1e-3, weight decay 1e-4, full
batch, 3000 steps, seed 0; the step with the lowest validation loss (checked every 100) is kept.  Reported: validation R^2 (chunk-consistent, about the validation mean, pooled xyz) of the
head and of the linear-saturating form refit on the TRAIN split (its K vs the gate's 2.851); slices of h against the servo along x and y; |h| at 2, 5, 10, 20 cm.

    python experiments/raah_fit.py        -> lifelong_ood/policy/raah_head.npz + results/flawed/raah/fit.json   (both opened with "x")
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import raas_gate as RG                                                       # noqa: E402

NPZ = os.path.join(ROOT, "lifelong_ood", "policy", "raah_head.npz")
FIT = os.path.join(ROOT, "results", "flawed", "raah", "fit.json")
SCALE, STEPS, LR_, WD = 0.20, 3000, 1e-3, 1e-4


def r2(a, p):
    a, p = a.reshape(-1, 3), p.reshape(-1, 3)
    return float(1.0 - ((a - p) ** 2).sum() / ((a - a.mean(0)) ** 2).sum())


def main():
    for p in (NPZ, FIT):
        assert not os.path.exists(p), "never overwrite: " + p
    g = json.load(open(os.path.join(ROOT, "results", "flawed", "raas_gate.json"), encoding="utf-8"))
    M, K_gate = float(g["max_step"]), float(g["K"])
    eps = RG.load()
    tr, va = [], []
    for i, (task, st, ac) in enumerate(eps):
        T = len(ac)
        held = np.where(st[:, 3] > 1.5)[0]
        if len(held) == 0:
            continue
        idx = np.minimum(held[:, None] + np.arange(RG.H)[None, :], T - 1)
        row = (st[held, :3], ac[idx][:, :, :3])
        (va if (i * 2654435761) % (2 ** 32) % 10 == 0 else tr).append(row)
    cat = lambda L, k: np.concatenate([r[k] for r in L])                     # noqa: E731
    rtr, atr, rva, ava = cat(tr, 0), cat(tr, 1), cat(va, 0), cat(va, 1)
    print("[raah-fit] held rows train %d val %d (total %d; plan: 13,327); demos train %d val %d" % (len(rtr), len(rva), len(rtr) + len(rva), len(tr), len(va)), flush=True)
    torch.manual_seed(0)
    dims = [(3, 64), (64, 64), (64, 30)]
    Ws = []
    for i, o in dims:
        b = 1.0 / np.sqrt(i)
        Ws += [(torch.rand(i, o) * 2 - 1) * b, (torch.rand(o) * 2 - 1) * b]
    Ws = [w.clone().requires_grad_(True) for w in Ws]

    def g_fn(r, W):
        h = torch.nn.functional.silu((r / SCALE) @ W[0] + W[1])
        h = torch.nn.functional.silu(h @ W[2] + W[3])
        return M * torch.tanh(h @ W[4] + W[5])

    def head(r, W):
        return torch.clamp(g_fn(r, W) - g_fn(torch.zeros(1, 3), W), -M, M).reshape(-1, 10, 3)

    Rtr, Atr, Rva, Ava = (torch.as_tensor(x, dtype=torch.float32) for x in (rtr, atr, rva, ava))
    opt = torch.optim.AdamW(Ws, lr=LR_, weight_decay=WD)
    best = (1e9, None, -1)
    log = []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = ((head(Rtr, Ws) - Atr) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                vl = float(((head(Rva, Ws) - Ava) ** 2).mean())
            log.append((step, float(loss), vl))
            if vl < best[0]:
                best = (vl, [w.detach().clone() for w in Ws], step)
    W = best[1]
    print("[raah-fit] best step %d val loss %.5f (last train loss %.5f)" % (best[2], best[0], log[-1][1]), flush=True)
    with torch.no_grad():
        ph_va = head(Rva, W).numpy()
        ph_tr = head(Rtr, W).numpy()
    Kf = RG.fit_k(rtr, atr, M)
    lin = lambda r, K: np.repeat(np.clip(K * r, -M, M)[:, None, :], 10, axis=1)  # noqa: E731
    res = dict(held_rows=dict(train=int(len(rtr)), val=int(len(rva))), demos=dict(train=len(tr), val=len(va)), M=M, K_gate=K_gate, K_refit_train=Kf,
               K_within_05=bool(abs(Kf - K_gate) <= 0.05), best_step=best[2], val_loss=best[0], curve=log,
               r2_val=dict(head=r2(ava, ph_va), linear_refit=r2(ava, lin(rva, Kf)), linear_gate_K=r2(ava, lin(rva, K_gate))),
               r2_train=dict(head=r2(atr, ph_tr), linear_refit=r2(atr, lin(rtr, Kf))))
    res["r2_val"]["head_minus_linear"] = res["r2_val"]["head"] - res["r2_val"]["linear_refit"]
    sl = {}
    for ax, nm in ((0, "x"), (1, "y")):
        for cm in (-20, -10, -5, -2, 0, 2, 5, 10, 20):
            r = np.zeros((1, 3), np.float32)
            r[0, ax] = cm / 100.0
            with torch.no_grad():
                h = head(torch.as_tensor(r), W).numpy()[0]
            sl["%s=%+dcm" % (nm, cm)] = dict(head_row0=[float(v) for v in h[0]], head_row9=[float(v) for v in h[9]], servo=[float(v) for v in lin(r, K_gate)[0, 0]])
    res["slices"] = sl
    mag = {}
    for cm in (2, 5, 10, 20):
        rr = np.zeros((1, 3), np.float32)
        rr[0, 0] = cm / 100.0
        with torch.no_grad():
            h = head(torch.as_tensor(rr), W).numpy()[0]
        mag["%dcm" % cm] = dict(head_mean_over_rows_x=float(h[:, 0].mean()), servo_x=float(lin(rr, K_gate)[0, 0, 0]))
    res["magnitude_along_x"] = mag
    os.makedirs(os.path.dirname(FIT), exist_ok=True)
    keys = ("W1", "b1", "W2", "b2", "W3", "b3")
    np.savez(NPZ, **{k: w.numpy() for k, w in zip(keys, W)}, scale=np.float32(SCALE), M=np.float32(M))
    res["npz_sha256"] = hashlib.sha256(open(NPZ, "rb").read()).hexdigest()
    with open(FIT, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raah-fit] val R2 head %.4f | linear refit %.4f (K %.3f vs gate %.3f) | linear at gate K %.4f | head - linear %+.4f" % (
        res["r2_val"]["head"], res["r2_val"]["linear_refit"], Kf, K_gate, res["r2_val"]["linear_gate_K"], res["r2_val"]["head_minus_linear"]), flush=True)
    print("[raah-fit] magnitude along x:", json.dumps(mag), flush=True)
    print("[raah-fit] npz sha256", res["npz_sha256"], flush=True)


if __name__ == "__main__":
    main()

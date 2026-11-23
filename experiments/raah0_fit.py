"""NPH head fit H* (docs/notes/plan_necessity_power.md 2.3 / 3.2 step 1; tag nph).  CPU, openpi venv (torch), seconds.  Copy of raah_fit.py with the goal-directed target.

Target: for each held row t of a demo, t_r = the demo's last held row (the frame before the gripper opens), Delta(t) = rel(t) - rel(t_r) (the object's remaining displacement to its
release point), and for every chunk row j = 0..9  y_j = clip(K * Delta(min(t + j, t_r)), +-M), K = 2.851 and M = .9375 (raas_gate.json: the constants behaviour cloning fixed).
Same loader (raas_gate), same 13,327 held rows, same demo split hash, same MLP 3 -> 64 -> 64 -> 30 (SiLU, M tanh), same optimiser (AdamW 1e-3, wd 1e-4, full batch, 3000 steps, seed 0,
best validation step checked every 100).  NOT centred: the head's zero is learned (h = clip(g(rel)), no g(0) subtraction).
Head gate on the VALIDATION demos' held rows (thresholds fixed in the plan):
  G1 mean cos(h_row0(rel(t)), Delta(t)) >= .80 over rows with |Delta| >= 2 cm;  G2 median |h_row0| / min(K |Delta|, M) >= .60 over rows with 2 <= |Delta| <= 6 cm;
  G3 median |h_row0| <= .05 over rows with |Delta| < 1 cm.   The same three numbers for the registered head H (raah_head.npz, centred) are reported (descriptive).
Any gate fails: exit code 3, no refit.   python experiments/raah0_fit.py   -> lifelong_ood/policy/raah0_head.npz + results/flawed/raah0/fit.json (both opened with "x")
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

NPZ = os.path.join(ROOT, "lifelong_ood", "policy", "raah0_head.npz")
FIT = os.path.join(ROOT, "results", "flawed", "raah0", "fit.json")
H_NPZ = os.path.join(ROOT, "lifelong_ood", "policy", "raah_head.npz")
SCALE, STEPS, LR_, WD = 0.20, 3000, 1e-3, 1e-4
G1_MIN, G2_MIN, G3_MAX = 0.80, 0.60, 0.05


def np_head(P, rel, centred):
    P = {k: np.asarray(v, np.float64) for k, v in P.items()}

    def g(r):
        h = (r / float(P["scale"])) @ P["W1"] + P["b1"]
        h = h / (1.0 + np.exp(-h))
        h = h @ P["W2"] + P["b2"]
        h = h / (1.0 + np.exp(-h))
        return float(P["M"]) * np.tanh(h @ P["W3"] + P["b3"])
    out = g(np.asarray(rel, np.float64)) - (g(np.zeros((1, 3))) if centred else 0.0)
    return np.clip(out, -float(P["M"]), float(P["M"])).reshape(-1, 10, 3)


def gates(h0, delta, K, M):
    """h0 [n, 3] head row 0 on val rows; delta [n, 3] remaining displacement."""
    dn = np.linalg.norm(delta, axis=1)
    hn = np.linalg.norm(h0, axis=1)
    m1 = dn >= 0.02
    cos = (h0[m1] * delta[m1]).sum(1) / (hn[m1] * dn[m1] + 1e-12)
    m2 = (dn >= 0.02) & (dn <= 0.06)
    ratio = hn[m2] / np.minimum(K * dn[m2], M)
    m3 = dn < 0.01
    return dict(G1_mean_cos=float(cos.mean()), n_G1=int(m1.sum()), G2_median_ratio=float(np.median(ratio)), n_G2=int(m2.sum()),
                G3_median_abs=float(np.median(hn[m3])), n_G3=int(m3.sum()))


def main():
    for p in (NPZ, FIT):
        assert not os.path.exists(p), "never overwrite: " + p
    g = json.load(open(os.path.join(ROOT, "results", "flawed", "raas_gate.json"), encoding="utf-8"))
    M, K = float(g["max_step"]), float(g["K"])
    eps = RG.load()
    tr, va = [], []
    for i, (task, st, ac) in enumerate(eps):
        T = len(ac)
        held = np.where(st[:, 3] > 1.5)[0]
        if len(held) == 0:
            continue
        t_r = int(held.max())
        rel = st[:, :3]
        idx = np.minimum(held[:, None] + np.arange(RG.H)[None, :], t_r)                  # min(t + j, t_r)
        delta_rows = rel[idx] - rel[t_r][None, None, :]                                  # [n, 10, 3]
        y = np.clip(K * delta_rows, -M, M)
        row = (st[held, :3], y, rel[held] - rel[t_r][None, :])
        (va if (i * 2654435761) % (2 ** 32) % 10 == 0 else tr).append(row)
    cat = lambda L, k: np.concatenate([r[k] for r in L])                     # noqa: E731
    rtr, ytr, rva, yva, dva = cat(tr, 0), cat(tr, 1), cat(va, 0), cat(va, 1), cat(va, 2)
    print("[raah0-fit] held rows train %d val %d (total %d; plan: 13,327); demos train %d val %d" % (len(rtr), len(rva), len(rtr) + len(rva), len(tr), len(va)), flush=True)
    torch.manual_seed(0)
    dims = [(3, 64), (64, 64), (64, 30)]
    Ws = []
    for i, o in dims:
        b = 1.0 / np.sqrt(i)
        Ws += [(torch.rand(i, o) * 2 - 1) * b, (torch.rand(o) * 2 - 1) * b]
    Ws = [w.clone().requires_grad_(True) for w in Ws]

    def head(r, W):
        h = torch.nn.functional.silu((r / SCALE) @ W[0] + W[1])
        h = torch.nn.functional.silu(h @ W[2] + W[3])
        return torch.clamp(M * torch.tanh(h @ W[4] + W[5]), -M, M).reshape(-1, 10, 3)

    Rtr, Ytr, Rva, Yva = (torch.as_tensor(x, dtype=torch.float32) for x in (rtr, ytr, rva, yva))
    opt = torch.optim.AdamW(Ws, lr=LR_, weight_decay=WD)
    best = (1e9, None, -1)
    log = []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = ((head(Rtr, Ws) - Ytr) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                vl = float(((head(Rva, Ws) - Yva) ** 2).mean())
            log.append((step, float(loss), vl))
            if vl < best[0]:
                best = (vl, [w.detach().clone() for w in Ws], step)
    W = best[1]
    print("[raah0-fit] best step %d val loss %.5f (last train loss %.5f)" % (best[2], best[0], log[-1][1]), flush=True)
    keys = ("W1", "b1", "W2", "b2", "W3", "b3")
    P = {k: w.numpy() for k, w in zip(keys, W)}
    P.update(scale=np.float32(SCALE), M=np.float32(M), centred=np.float32(0.0))
    h0 = np_head(P, rva, False)[:, 0, :]
    gs = gates(h0, dva, K, M)
    PH = {k: np.asarray(v) for k, v in np.load(H_NPZ).items()}
    gh = gates(np_head(PH, rva, True)[:, 0, :], dva, K, M)
    r2 = 1.0 - ((np_head(P, rva, False) - yva) ** 2).sum() / ((yva - yva.mean((0, 1))) ** 2).sum()
    ok = dict(G1=gs["G1_mean_cos"] >= G1_MIN, G2=gs["G2_median_ratio"] >= G2_MIN, G3=gs["G3_median_abs"] <= G3_MAX)
    res = dict(held_rows=dict(train=int(len(rtr)), val=int(len(rva))), demos=dict(train=len(tr), val=len(va)), M=M, K=K, best_step=best[2], val_loss=best[0], val_r2_target=float(r2),
               curve=log, gates_Hstar=gs, gates_H_registered=gh, thresholds=dict(G1_min=G1_MIN, G2_min=G2_MIN, G3_max=G3_MAX), pass_=ok, all_pass=bool(all(ok.values())))
    os.makedirs(os.path.dirname(FIT), exist_ok=True)
    np.savez(NPZ, **P)
    res["npz_sha256"] = hashlib.sha256(open(NPZ, "rb").read()).hexdigest()
    with open(FIT, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raah0-fit] H* gates", json.dumps(gs), "pass", json.dumps(ok), flush=True)
    print("[raah0-fit] H (registered) gates", json.dumps(gh), flush=True)
    print("[raah0-fit] val R2 of y %.4f | npz sha256 %s" % (r2, res["npz_sha256"]), flush=True)
    print("[raah0-fit] ALL GATES PASS" if res["all_pass"] else "[raah0-fit] GATE FAIL: STOP", flush=True)
    sys.exit(0 if res["all_pass"] else 3)


if __name__ == "__main__":
    main()

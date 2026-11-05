"""D (docs/SCAFFOLD_RESULTS.md 8; m3_prereg sc_disc): does the part inventory come out of the demos?  openpi venv, GPU, no rollouts.
From a build on demos 0-39 with no given roles, sizes or grasp message, every decision reads held-out demos 40-49.
    split:  refit one part on one of its tasks' training rows.  gain = (loss(shared part) - loss(refit)) / loss(shared part), held out, on that task.
    noise:  the same refit for a part with one task (it has nothing to split): the gain that refitting alone gives.
    merge:  use part p on part q's held-out rows.  cost = (loss(p) - loss(q)) / loss(q).
    k = 0:  the part zeroed: what the part contributes at all.
Given roles and atom names score the result in the read, and nothing else.

    python experiments/sc_disc.py score --seed 10   -> results/flawed/sc_disc/s<seed>.json
    python experiments/sc_disc.py read              -> results/flawed/sc_disc_read.json
SC_SMOKE=1: 200 refit steps, the _smoke checkpoint tag, results under sc_disc_smoke.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_disc_smoke" if SMOKE else "sc_disc")
TAG = "sc_u40_smoke" if SMOKE else "sc_u40"
TRAIN_DEMOS, DRAWS = 40, 8
SEEDS = (10, 11, 12)


def role_mask(b, role):
    return (b["held"] < 0.5) if role == "obj" else (b["held"] > 0.5)


def row_losses(m, b, act, role, seed=0, bs=4096):
    """Per-row flow loss on one role's rows, with common noise across calls: the same seed gives the same x0 and t for a row."""
    import torch
    keep = role_mask(b, role).nonzero()[:, 0]
    out = torch.zeros(len(keep), device=act.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(DRAWS):
        x0 = torch.randn((len(keep),) + act.shape[1:], generator=g).to(act.device)
        t = torch.rand(len(keep), generator=g).to(act.device)
        for i in range(0, len(keep), bs):
            j = keep[i:i + bs]
            bb = {k: v[j] for k, v in b.items()}
            a1, x, tt = act[j], x0[i:i + bs], t[i:i + bs]
            xt = (1 - tt[:, None, None]) * x + tt[:, None, None] * a1
            with torch.no_grad():
                v = m.velocity(bb, xt, tt)
            out[i:i + bs] += ((v - (a1 - x)) ** 2).mean((1, 2))
    return keep, out / DRAWS


def demo_mean(vals, dkey):
    """-> per-demo mean loss, for the CI's unit."""
    return {int(k): float(vals[dkey == k].mean()) for k in np.unique(dkey)}


def boot_rel(a, b, n=2000):
    """a, b: {demo: loss} on the same demos.  (a - b) / a over resampled demos -> the estimate and its 95% CI."""
    ks = sorted(a)
    A, B = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        i = rng.choice(len(ks), len(ks))
        v.append((A[i].mean() - B[i].mean()) / A[i].mean())
    return float((A.mean() - B.mean()) / A.mean()), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score(seed):
    import torch
    import fa1_common as FC
    import pp_p_train as T
    from lifelong_ood.policy import parts_flow as PF
    fn = os.path.join(OD, f"s{seed}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    T.deterministic()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, meta = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", f"pp_{TAG}_s{seed}", "P.npz"))
    assert meta.get("demos") == TRAIN_DEMOS and meta.get("nosize") and m.nmode == 0 and meta.get("pgate") and meta.get("roles"), meta
    m = m.to(dev)
    lex, pg, tasks = meta["roles"], meta["pgate"], list(meta["tasks"])
    names = list(FC.KNOWN)
    b, act, _, _, _, _ = T.rows(dev, tasks, None, pgate=pg, roles=lex)
    tr = (b["dkey"] % 1000) < TRAIN_DEMOS
    bh = {k: v[~tr] for k, v in b.items()}
    ah = act[~tr]
    bt = {k: v[tr] for k, v in b.items()}
    at = act[tr]
    task_of = (bh["dkey"] // 1000).cpu().numpy()
    tti = (bt["dkey"] // 1000)
    res = dict(seed=seed, tasks=tasks, parts={}, split={}, merge={}, k0={})
    ATOMS = dict(obj=PF.OBJ_ATOMS, tgt=PF.TGT_ATOMS)
    for role in ("obj", "tgt"):
        net_name = role
        keep, base = row_losses(m, bh, ah, role)
        base = base.cpu().numpy()
        kidx = keep.cpu().numpy()
        dk = bh["dkey"].cpu().numpy()[kidx]
        ent = (bh["oi"] if role == "obj" else bh["ti"]).cpu().numpy()[kidx]
        tk = task_of[kidx]
        uses = {}
        for e, t in sorted(set(zip(ent.tolist(), tk.tolist()))):
            uses.setdefault(e, []).append(t)
        res["parts"][role] = {FC.ENT[e]: [names[t] for t in v] for e, v in uses.items()}
        print("[disc]", role, json.dumps(res["parts"][role]), flush=True)
        table = "opart" if role == "obj" else "tpart"
        for e, ts in uses.items():
            j = int(getattr(m, table)[e])
            assert j >= 0
            own = ent == e
            # k = 0
            m0 = copy.deepcopy(m)
            net0 = getattr(m0, net_name)
            with torch.no_grad():
                net0.emb[j] = 0.0
                for ly in net0.hidden:
                    ly.B[j] = 0.0
            _, l0 = row_losses(m0, bh, ah, role)
            l0 = l0.cpu().numpy()
            res["k0"][f"{role}|{FC.ENT[e]}"] = boot_rel(demo_mean(l0[own], dk[own]), demo_mean(base[own], dk[own]))
            # split (or noise, for a part with one task)
            for t in ts:
                mf = copy.deepcopy(m)
                net = getattr(mf, net_name)
                for p in mf.parameters():
                    p.requires_grad_(False)
                g = torch.Generator().manual_seed(1000 + seed + t)
                with torch.no_grad():
                    net.emb[j] = torch.randn(PF.EMB, generator=g).to(dev) * 0.1
                    for ly in net.hidden:
                        ly.A[j] = (torch.randn(ly.A.shape[1:], generator=g) / np.sqrt(ly.A.shape[1])).to(dev)
                        ly.B[j] = 0.0
                parts = net.part_parameters()
                for p in parts:
                    p.requires_grad_(True)

                def mask(parts=parts, j=j):
                    for p in parts:
                        if p.grad is not None:
                            kj = p.grad[j].clone()
                            p.grad.zero_()
                            p.grad[j] = kj
                ent_t = (bt["oi"] if role == "obj" else bt["ti"])
                sel = (tti == t) & (ent_t == e) & role_mask(bt, role)
                bs_ = {k: v[sel] for k, v in bt.items()}
                mf.training_bb = False
                steps = 200 if SMOKE else T.CREATE_STEPS
                mf, _, secs = T.fit(mf, parts, bs_, at[sel], steps, 256, T.CREATE_LR, seed + t, 0.0, mask_fn=mask, log=f"refit {role} {FC.ENT[e]} {names[t]}")
                _, lr = row_losses(mf, bh, ah, role)
                lr = lr.cpu().numpy()
                here = own & (tk == t)
                est, ci = boot_rel(demo_mean(base[here], dk[here]), demo_mean(lr[here], dk[here]))
                res["split"][f"{role}|{FC.ENT[e]}|{names[t]}"] = dict(gain=est, ci=ci, n_tasks=len(ts), secs=secs)
                print("[disc] refit", role, FC.ENT[e], names[t], "gain", round(est, 4), [round(x, 4) for x in ci], "tasks", len(ts), flush=True)
            # merge: every other part on this part's rows
            for e2 in uses:
                if e2 == e:
                    continue
                ms = copy.deepcopy(m)
                getattr(ms, table)[e] = getattr(ms, table)[e2]
                _, l2 = row_losses(ms, bh, ah, role)
                l2 = l2.cpu().numpy()
                est, ci = boot_rel(demo_mean(l2[own], dk[own]), demo_mean(base[own], dk[own]))
                res["merge"][f"{role}|{FC.ENT[e2]}->{FC.ENT[e]}"] = dict(cost=est, ci=ci)
                print("[disc] use", role, FC.ENT[e2], "for", FC.ENT[e], "cost", round(est, 4), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    R = {s: json.load(open(os.path.join(OD, f"s{s}.json"), encoding="utf-8")) for s in SEEDS}
    out = dict(seeds={}, splits={}, merges={})
    votes_s, votes_m = {}, {}
    for s, r in R.items():
        noise = [v["gain"] for v in r["split"].values() if v["n_tasks"] == 1]
        floor = max(noise) if noise else 0.0
        mag = max(abs(x) for x in noise) if noise else 0.0
        sp = {k: bool(v["n_tasks"] > 1 and v["ci"][0] > 0.0 and v["gain"] > floor) for k, v in r["split"].items()}
        mg = {k: bool(v["ci"][1] <= mag) for k, v in r["merge"].items()}             # boot_rel's sign: p's loss above q's gives a positive (a - b) / a
        out["seeds"][str(s)] = dict(noise_floor=floor, noise_abs=mag, n_noise=len(noise), splits=[k for k, x in sp.items() if x], merges=[k for k, x in mg.items() if x], k0=r["k0"])
        for k, x in sp.items():
            votes_s[k] = votes_s.get(k, 0) + int(x)
        for k, x in mg.items():
            votes_m[k] = votes_m.get(k, 0) + int(x)
    out["splits"] = sorted(k for k, n in votes_s.items() if n >= 2)
    out["merges"] = sorted(k for k, n in votes_m.items() if n >= 2)
    k5 = any(x.endswith("wine_bottle_1|K5") or x.endswith("wine_bottle_1|K4") for x in out["splits"])
    others = [x for x in out["splits"] if "wine_bottle_1" not in x]
    out["label"] = "UNDER-SPLITS" if not k5 else ("OVER-SPLITS" if len(others) >= 2 else "RECOVERS")
    print("[disc]", out["label"], json.dumps(out), flush=True)
    with open(os.path.join(OUT, "sc_disc_read.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("score", "read"))
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    score(a.seed) if a.what == "score" else read()

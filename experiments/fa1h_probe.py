"""FLAWED_COMPOSITION prototype A, stage 1h (section 5.14): the IDENTITY CHECK of the held-geometry route, on training rows (demos +
generated, held rows of the pick-and-place tasks K1-K6).  Label: the held object (bowl, cheese, bottle).  Probe: multinomial
logistic regression on standardised features, 5 folds split by episode, balanced accuracy (chance 1/3).
  route_model   the route exactly as the stage-1h network receives it (FactoredFlow("Ah").geo_route on the training batch)
  route_geom    the same geometric quantities recomputed independently in numpy from the raw rows: Q^T (R R0^T) Q (2 columns) and
                Q^T off * 5, Q = Rz(cyaw)
  drot, off     each part of the route alone;  drot_onset  drot on each hold's first row only (the identity matrix by construction)
  proxy_1d      1d's cut inputs, the end effector's tilt (world-z row) and the finger positions
Pre-registered checks (5.14):
  I1  |acc(route_model) - acc(route_geom)| <= 0.02   -- the route carries nothing beyond its geometry
  I2  acc(drot_onset) <= 1/3 + 0.02                   -- the rotation part starts identity-free at every grasp
  (descriptive) acc(route_model), acc(off), acc(drot) against acc(proxy_1d): how much identity the geometry itself predicts
CPU (openpi venv).  -> results/flawed/a1h_probe.json

    python experiments/fa1h_probe.py
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
from fa1g_train import FIELDS, rot0_rows                                    # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402


def probe(X, y, grp, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(grp)
    fold = dict(zip(ug, rng.integers(0, folds, len(ug))))
    f = np.array([fold[g] for g in grp])
    pred = np.zeros(len(y), int)
    for k in range(folds):
        tr, te = f != k, f == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        W = torch.zeros(X.shape[1], 3, requires_grad=True)
        bb = torch.zeros(3, requires_grad=True)
        opt = torch.optim.LBFGS([W, bb], max_iter=200)
        yt = torch.tensor(y[tr])
        cw = torch.tensor(len(y[tr]) / (3 * np.bincount(y[tr], minlength=3) + 1e-9), dtype=torch.float32)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ W + bb, yt, weight=cw) + 1e-4 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pred[te] = (torch.tensor((X[te] - mu) / sd, dtype=torch.float32) @ W + bb).argmax(1).numpy()
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(3) if np.any(y == c)]))


def main():
    D0 = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db0 = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    G = dict(np.load(os.path.join(FC.DATA, "gen_g.npz")))
    D = {x: np.concatenate([D0[x], G[x].astype(D0[x].dtype)]) for x in FIELDS}
    Db = {x: np.concatenate([Db0[x], G[x].astype(np.float32)]) for x in ("held", "off")}
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    R0 = rot0_rows(D, Db["held"])
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    held = Db["held"] > 0.5
    sel = held & (D["task"] <= 5)
    onset = held & np.r_[True, (~held[:-1]) | (key[1:] != key[:-1])] & (D["task"] <= 5)
    objs = sorted(set(oi[sel]))
    y_all = np.array([objs.index(o) if o in objs else -1 for o in oi])
    m = FF.FactoredFlow("Ah")
    T = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32)          # noqa: E731
    b = dict(ee_rot=T(D["ee_rot"]), rot0=T(R0), off=T(Db["off"]))
    Q = FF.rz(T(cy))
    with torch.no_grad():
        route_model = torch.cat(m.geo_route(b, Q), -1).numpy()
        route_model_rot = torch.cat(FF.FactoredFlow("Ah_rot").geo_route(b, Q), -1).numpy()
    c, s = np.cos(cy), np.sin(cy)
    Qn = np.zeros((len(cy), 3, 3))
    Qn[:, 0, 0], Qn[:, 0, 1], Qn[:, 1, 0], Qn[:, 1, 1], Qn[:, 2, 2] = c, -s, s, c, 1.0
    rel = np.einsum("bij,bkj->bik", D["ee_rot"].astype(np.float64), R0.astype(np.float64))
    dr = np.einsum("bji,bjk,bkl->bil", Qn, rel, Qn)[:, :, :2].reshape(-1, 6)
    of = np.einsum("bji,bj->bi", Qn, Db["off"].astype(np.float64)) * FF.POS_SCALE
    route_geom = np.concatenate([dr, of], 1)
    proxy = np.concatenate([D["ee_rot"][:, 2, :], D["grip"]], 1)
    feats = {"route_model": route_model, "route_geom": route_geom, "route_model_rot": route_model_rot, "drot": dr, "off": of,
             "proxy_1d": proxy}
    out = {k: probe(v[sel], y_all[sel], key[sel]) for k, v in feats.items()}
    out["drot_onset"] = probe(dr[onset], y_all[onset], key[onset])
    out["max_abs_route_model_minus_geom"] = float(np.abs(route_model[sel] - route_geom[sel]).max())
    out["max_abs_route_model_rot_minus_drot"] = float(np.abs(route_model_rot[sel] - dr[sel]).max())
    out["n_rows"], out["n_onset"] = int(sel.sum()), int(onset.sum())
    out["I1"] = abs(out["route_model"] - out["route_geom"]) <= 0.02
    out["I1_rot"] = abs(out["route_model_rot"] - out["drot"]) <= 0.02
    out["I2"] = out["drot_onset"] <= 1 / 3 + 0.02
    print("[fa1h] probe", json.dumps(out), flush=True)
    with open(os.path.join(ROOT, "results", "flawed", "a1h_probe.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(objects=[FC.ENT[o] for o in objs], probe="multinomial logistic, 5 folds by episode, balanced accuracy", **out), fh, indent=1)


if __name__ == "__main__":
    main()

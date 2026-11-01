"""FLAWED_COMPOSITION prototype A, stage 1l IDENTITY PROBE on the new input (section 5.21; before any rollout): does the role-symmetric
occupancy (factored_flow.FactoredFlow._occ, the object term's new input) carry the TARGET's identity beyond what the object's own
identity already carries?  On the approach rows (hold off) of the known pick-and-place tasks K1-K6, demos + the generated data the
models train on, a multinomial logistic probe predicts the target entity from
  base    the object's identity (one-hot)                          -- what the object term already reads
  occ     the object's identity + the occupancy features           -- the new object-term input
Split by episode (80 / 20, seed 0).  A bypass would show as occ accuracy above base accuracy on held-out episodes.
-> results/flawed/a1l_idprobe.json

    python experiments/fa1l_idprobe.py --gen gen_lc.npz
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


def fit(Xtr, ytr, Xte, yte, ncls, epochs=400):
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    A, B = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    W = torch.nn.Linear(A.shape[1], ncls)
    opt = torch.optim.Adam(W.parameters(), lr=0.05, weight_decay=1e-4)
    yt = torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(W(A), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((W(A).argmax(1).numpy() == ytr).mean()), float((W(B).argmax(1).numpy() == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen_lc.npz")
    ap.add_argument("--arm", default="Ao", help="Ao (1l occupancy) or Ap (1m local free space)")
    ap.add_argument("--out", default="a1l_idprobe.json")
    a = ap.parse_args()
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    G = dict(np.load(os.path.join(FC.DATA, a.gen)))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    task = np.r_[D["task"], G["task"]].astype(int)
    demo = np.r_[D["demo"], G["demo"]].astype(int)
    pos = np.r_[D["pos"], G["pos"]].astype(np.float32)
    yaw = np.r_[D["yaw"], G["yaw"]].astype(np.float32)
    held = np.r_[Db["held"], G["held"]] > 0.5
    keep = (task <= 5) & ~held
    oi = np.array([FC.ENT.index(rl[t][0]) for t in task])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in task])
    m = FF.FactoredFlow(a.arm, size=np.array([FC.SIZE[e] for e in FC.ENT], np.float32))
    j = np.nonzero(keep)[0]
    with torch.no_grad():
        P, Y, E = torch.tensor(pos[j]), torch.tensor(yaw[j]), torch.tensor(oi[j])
        idx = torch.arange(len(j))
        Q, o = FF.rz(Y[idx, E]), P[idx, E]
        occ = (m._free if a.arm == "Ap" else m._occ)(dict(pos=P), Q, o, E).numpy()
    one = np.eye(FF.N_ENT, dtype=np.float32)[oi[j]]
    ep = task[j] * 1000 + demo[j]
    rng = np.random.default_rng(0)
    ue = np.unique(ep)
    te_ep = set(rng.choice(ue, max(1, len(ue) // 5), replace=False).tolist())
    te = np.array([e in te_ep for e in ep])
    y = ti[j]
    out = dict(rows=int(len(j)), episodes=int(len(ue)), gen_file=a.gen, arm=a.arm, majority=float(np.mean(y[te] == np.bincount(y[~te]).argmax())))
    for name, X in (("base", one), ("occ", np.c_[one, occ])):
        tr_acc, te_acc = fit(X[~te], y[~te], X[te], y[te], FF.N_ENT)
        out[name] = dict(train=tr_acc, test=te_acc)
    out["occ_minus_base_test"] = out["occ"]["test"] - out["base"]["test"]
    print("[fa1l] idprobe", json.dumps(out), flush=True)
    with open(os.path.join(ROOT, "results", "flawed", a.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

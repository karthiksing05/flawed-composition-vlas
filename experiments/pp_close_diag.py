"""PLAN_PARTS round diagnostics (descriptive, dev data only): the object term's input at the moment it decides to close.
fa1h_roll.episode, unchanged, with model.sample wrapped to record each call's batch and sampled chunk.  The probe is the batch of
the first call before the hold whose chunk's first gripper command closes (> 0); an episode that never closes before a hold
contributes the call at its closest horizontal approach instead ("hover").  The training cloud is the object term's input
(pp_ood_diag.feats, the model's own feature map) on the task's NOT-held demo rows whose action chunk's first gripper command
closes.  Per input group: the probe's standardized nearest-cloud distance, plus the horizontal ee-object offset at the probe and
in the cloud.  LIBERO venv, CPU.  -> results/flawed/pp_close_<name>.json

    python experiments/pp_close_diag.py --ckpt pp_full_rt_s10/P.npz --name rt10 --task K4 --states 4+5+6
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1h_roll as RH                                                      # noqa: E402
import pp_ood_diag as OD                                                    # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


class Wrap:
    def __init__(self, m):
        self.m, self.calls = m, []

    def __getattr__(self, k):
        return getattr(self.m, k)

    def sample(self, b, **kw):
        x = self.m.sample(b, **kw)
        self.calls.append(({k: v.clone() for k, v in b.items()}, x.clone()))
        return x


def dxy(b):
    return np.linalg.norm((b["ee_pos"] - b["pos"][torch.arange(len(b["oi"])), b["oi"]])[:, :2].numpy(), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--states", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    m, meta = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    import pp_p_train as PT
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"), [a.task])
    keep = (b["held"] < 0.5) & (X[:, 0, 6] > 0)
    cb = {k: v[keep] for k, v in b.items()}
    with torch.no_grad():
        Ft = OD.feats(m, cb).numpy()
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-3
    Z = (Ft - mu) / sd
    cd = dxy(cb)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    res = dict(cloud=dict(n=int(keep.sum()), dxy_median=float(np.median(cd)), dxy_p90=float(np.percentile(cd, 90)),
                          dxy_max=float(cd.max())), episodes=[])
    print("[close] cloud", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["cloud"].items()}), flush=True)
    for lay in ("familiar", "new"):
        t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == lay)
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
        for k in [int(x) for x in a.states.split("+")]:
            w = Wrap(m)
            e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
            pre = [(bb, x) for bb, x in w.calls if float(bb["held"][0]) < 0.5]
            if not pre:
                continue
            j = next((i for i, (bb, x) in enumerate(pre) if float(x[0, 0, 6]) > 0), None)
            kind = "close" if j is not None else "hover"
            if j is None:
                j = int(np.argmin([dxy(bb)[0] for bb, _ in pre]))
            bb = pre[j][0]
            with torch.no_grad():
                Fp = OD.feats(m, bb).numpy()
            Zp = (Fp - mu) / sd
            g = {nm: float(np.sqrt(((Zp[:, None, sl] - Z[None, :, sl]) ** 2).sum(-1)).min()) for nm, sl in OD.GROUPS.items()}
            r = dict(layout=lay, state=k, success=e["success"], kind=kind, call=j, dxy=float(dxy(bb)[0]),
                     dz=float((bb["ee_pos"] - bb["pos"][torch.arange(1), bb["oi"]])[0, 2]), nn_z=g)
            res["episodes"].append(r)
            print("[close]", a.name, a.task, json.dumps({x: (round(v, 3) if isinstance(v, float) else
                  ({q: round(u, 2) for q, u in v.items()} if isinstance(v, dict) else v)) for x, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(OUT, f"pp_close_{a.name}_{a.task}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

"""PLAN_PARTS round diagnostics (descriptive, dev data only): does the object term's commanded horizontal translation saturate the
executor's [-1, 1] clip during the approach, and where does the first close fire?  Separates the two candidate causes of round 7's
loss (PARTS_RESULTS 28): saturation of the scaled commands (x 1/s) against the loss of round 5's per-step clip.

fa1h_roll.episode, unchanged, with model.sample wrapped to record each call's batch and its raw sampled chunk (before the
executor's clip).  Per episode, over the executed steps (the first 5 of each chunk) of calls made before the hold:
  sat        fraction of steps whose raw horizontal command has max(|x|, |y|) > 1 (clipped by the executor)
  ratio_med  median of max(|x|, |y|) / 1 over those steps
  s          the episode's horizontal scale (P_hs), else None
  close      first call whose chunk's first gripper command closes: hand-object dxy and dz there (None: never closes)
Also, on the demos (K1-K6 not-held rows with dxy > 0.06 m), the fraction of rows whose first command has max(|x|, |y|) >= 0.99.
LIBERO venv, CPU.  -> results/flawed/pp_sat_<name>_<task>.json

    python experiments/pp_sat_diag.py --ckpt pp_full_hs_s10/P.npz --name hs10 --task K4 --states 4+5+6
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


def rel(b):
    d = (b["ee_pos"] - b["pos"][torch.arange(len(b["oi"])), b["oi"]])[0].numpy()
    return float(np.linalg.norm(d[:2])), float(d[2])


def demo_sat(task):
    import pp_p_train as PT
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"), [task])
    d = (b["ee_pos"] - b["pos"][torch.arange(len(b["oi"])), b["oi"]]).numpy()
    far = (b["held"].numpy() < 0.5) & (np.linalg.norm(d[:, :2], axis=1) > 0.06)
    m = np.abs(X[:, 0, :2].numpy()).max(1)
    return dict(n=int(far.sum()), sat=float((m[far] >= 0.99).mean()), med=float(np.median(m[far])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--states", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    m, meta = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    res = dict(demo=demo_sat(a.task), episodes=[])
    print("[sat] demo", a.task, json.dumps(res["demo"]), flush=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    for lay in ("familiar", "new"):
        t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == lay)
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
        for k in [int(x) for x in a.states.split("+")]:
            w = Wrap(m)
            e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
            pre = [(bb, x) for bb, x in w.calls if float(bb["held"][0]) < 0.5]
            if not pre:
                continue
            mx = np.concatenate([np.abs(x[0, :RH.REPLAN, :2].numpy()).max(1) for _, x in pre])
            s = None
            if float(getattr(m, "hscale", 0.0)) > 0:
                bb = pre[0][0]
                o = bb["pos"][torch.arange(1), bb["oi"]]
                s = float(m._scale(bb, o)[0])
            j = next((i for i, (bb, x) in enumerate(pre) if float(x[0, 0, 6]) > 0), None)
            d0 = rel(pre[0][0])
            r = dict(layout=lay, state=k, success=e["success"], held=e["held_step"] is not None, d0_xy=d0[0], d0_z=d0[1], s=s,
                     sat=float((mx > 1).mean()), ratio_med=float(np.median(mx)), ratio_p90=float(np.percentile(mx, 90)),
                     close_dxy=rel(pre[j][0])[0] if j is not None else None, close_dz=rel(pre[j][0])[1] if j is not None else None,
                     close_call=j, n_pre=len(pre))
            res["episodes"].append(r)
            print("[sat]", a.name, a.task, json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items()}), flush=True)
    for lay in ("familiar", "new"):
        E = [e for e in res["episodes"] if e["layout"] == lay]
        if E:
            res[f"summary_{lay}"] = dict(n=len(E), success=float(np.mean([e["success"] for e in E])), held=float(np.mean([e["held"] for e in E])),
                                         sat_med=float(np.median([e["sat"] for e in E])), ratio_med=float(np.median([e["ratio_med"] for e in E])),
                                         close_dxy_med=float(np.median([e["close_dxy"] for e in E if e["close_dxy"] is not None]))
                                         if any(e["close_dxy"] is not None for e in E) else None,
                                         never_close=int(sum(e["close_dxy"] is None for e in E)))
            print("[sat] summary", a.name, a.task, lay, json.dumps(res[f"summary_{lay}"]), flush=True)
    json.dump(res, open(os.path.join(OUT, f"pp_sat_{a.name}_{a.task}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

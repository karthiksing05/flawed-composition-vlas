"""PLAN_PARTS round diagnostics (descriptive, dev data only): the carry's height-to-progress profile, as the target term sees it.
fa1h_roll.episode, unchanged, with model.sample wrapped to record each call's batch.  On held calls, the held object's foot (its
position with z - zrest + the table's height; the target term's own input before any clip) relative to the target: horizontal
distance d and height h.  Binned by d, the median h over episodes, against the same on the held demo rows of a reference task
(--demo, e.g. K1 for the plate part).  Per episode also the d at which h first drops below 0.03 m (the 'touch-down distance').
LIBERO venv, CPU.  -> results/flawed/pp_cprof_<name>.json

    python experiments/pp_carry_profile.py --ckpt pp_full_pp_s10/P.npz --name pp10_N4 --tid 22 --demo K1 --states 4+5+6
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
import fa1c_common as FK                                                    # noqa: E402
import fa1h_roll as RH                                                      # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
BINS = [0.0, 0.03, 0.06, 0.10, 0.16, 0.22, 0.30, 0.45]


class Wrap:
    def __init__(self, m):
        self.m, self.calls = m, []

    def __getattr__(self, k):
        return getattr(self.m, k)

    def sample(self, b, **kw):
        self.calls.append({k: v.clone() for k, v in b.items()})
        return self.m.sample(b, **kw)


def dh(b):
    i = torch.arange(len(b["oi"]))
    p = b["pos"][i, b["oi"]].clone()
    p[:, 2] = p[:, 2] - b["zrest"] + b["pos"][:, FK.TABLE, 2]
    r = (p - b["pos"][i, b["ti"]]).numpy()
    return np.linalg.norm(r[:, :2], axis=1), r[:, 2]


def binned(d, h):
    out = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d >= lo) & (d < hi)
        if m.any():
            out[f"{lo:.2f}-{hi:.2f}"] = dict(n=int(m.sum()), h_med=float(np.median(h[m])), h_p25=float(np.percentile(h[m], 25)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--states", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    import pp_p_train as PT
    b, _, _, _, _, _ = PT.rows(torch.device("cpu"), [a.demo])
    hd = b["held"] > 0.5
    dd, hh = dh({k: v[hd] for k, v in b.items()})
    res = dict(demo=dict(task=a.demo, bins=binned(dd, hh)), episodes=[])
    print("[cprof] demo", a.demo, json.dumps(res["demo"]["bins"]), flush=True)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    t = n2["tasks"][a.tid]
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{a.tid}.npy"))
    D, Hh = [], []
    for k in [int(x) for x in a.states.split("+")]:
        w = Wrap(m)
        e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
        held = [c for c in w.calls if float(c["held"][0]) > 0.5]
        if not held:
            continue
        d, h = (np.concatenate(x) for x in zip(*[dh(c) for c in held]))
        D.append(d)
        Hh.append(h)
        up = np.nonzero(h > 0.05)[0]                                         # lifted, then the first return below 0.03 m
        low = np.nonzero((h < 0.03) & (np.arange(len(h)) > (up[0] if len(up) else len(h))))[0]
        r = dict(state=k, success=e["success"], d_onset=float(d[0]), h_onset=float(h[0]), h_max=float(h.max()),
                 touch_d=float(d[low[0]]) if len(low) else None, bins=binned(d, h))
        res["episodes"].append(r)
        print("[cprof]", a.name, t["key"], json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items() if x != "bins"}), flush=True)
    if D:
        res["pooled"] = binned(np.concatenate(D), np.concatenate(Hh))
        res["touch_d_med"] = float(np.median([e["touch_d"] for e in res["episodes"] if e["touch_d"] is not None] or [np.nan]))
        print("[cprof] pooled", a.name, json.dumps(res["pooled"]), "touch_d_med", res["touch_d_med"], flush=True)
    json.dump(res, open(os.path.join(OUT, f"pp_cprof_{a.name}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

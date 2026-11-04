"""PLAN_PARTS round 16 diagnostic (descriptive, dev data only; no model change): why does K5's carry (bottle -> rack) not arrive
in the new layout?

demo   Every target part's carries (held demo rows of the 8 known tasks; fitted rows, never executed).  Per carry (one hold
       segment): the horizontal object-to-target distance at the onset d0 and the foot's height above the target h0 there; the carry
       bearing (cyaw, the canonical frame's x-axis) relative to the target's own yaw; the foot at the last held row (d, h, and its
       canonical-frame offset).  Per target: percentiles of these, and the share of rows whose first-step horizontal command is at
       the executor's clip (|a_x| or |a_y| >= 0.99).  This is the support of each target part in (d0, bearing, h0).
roll   fa1h_roll.episode unchanged, model.sample wrapped (pp_close_diag.Wrap).  Per held call: d, h (the foot, as the target term
       reads it), d0t, the carry scale s_t, the scaled distance s_t * d (what the term sees), the bearing relative to the target's
       yaw, and over the chunk's 5 executed steps: the raw (pre-clip) horizontal command, the share of its x/y entries beyond 1,
       the clipped step's progress toward the target and its mean vertical command.  Per episode: onset (d0, h0, s_t, bearing),
       closest approach and the height there, held calls, drops (held -> not held), the steps left at the onset, success.
LIBERO venv, CPU.  -> results/flawed/pp_rack_demo.json (demo), results/flawed/pp_rack_<name>_<task>_<layout>.json (roll)

    python experiments/pp_rack_diag.py demo
    python experiments/pp_rack_diag.py roll --ckpt pp_full_nsz_s10/P.npz --name nsz10 --task K5 --layout new
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
DBINS = [0.0, 0.03, 0.06, 0.10, 0.16, 0.22, 0.30, 0.45, 9.0]


def wrap(a):
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def foot_rel(b):
    """The foot (object with z - zrest + table) minus the target: (d, h, bearing of the target-to-foot vector), per row."""
    i = torch.arange(len(b["oi"]))
    p = b["pos"][i, b["oi"]].clone()
    p[:, 2] = p[:, 2] - b["zrest"] + b["pos"][:, FK.TABLE, 2]
    r = (p - b["pos"][i, b["ti"]]).numpy()
    return np.linalg.norm(r[:, :2], axis=1), r[:, 2], r


def pct(v, q=(10, 50, 90)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return np.percentile(v, q).round(4).tolist() if len(v) else None


def demo():
    import pp_p_train as PT
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"))
    d, h, r = foot_rel(b)
    held = b["held"].numpy() > 0.5
    key = b["dkey"].numpy()
    ti, oi = b["ti"].numpy(), b["oi"].numpy()
    tyaw = b["yaw"][torch.arange(len(ti)), b["ti"]].numpy()
    cy = b["cyaw"].numpy()
    X = X.numpy()
    sat = (np.abs(X[:, 0, :2]) >= 0.99).any(1)
    seg_on = held & np.r_[True, (~held[:-1]) | (key[1:] != key[:-1])]
    seg_end = held & np.r_[(~held[1:]) | (key[1:] != key[:-1]), True]
    starts, ends = np.nonzero(seg_on)[0], np.nonzero(seg_end)[0]
    res = {}
    for t in np.unique(ti[held]):
        name = FC.ENT[t]
        car = []
        for s, e in zip(starts, ends):
            if ti[s] != t:
                continue
            Q = np.array([[math.cos(cy[s]), math.sin(cy[s])], [-math.sin(cy[s]), math.cos(cy[s])]])
            car.append(dict(task=int(key[s] // 1000), obj=FC.ENT[oi[s]], d0=float(d[s]), h0=float(h[s]),
                            bear_rel=float(math.degrees(wrap(cy[s] - tyaw[s]))), cyaw=float(math.degrees(cy[s])),
                            tyaw=float(math.degrees(tyaw[s])), d_end=float(d[e]), h_end=float(h[e]),
                            end_canon=(Q @ r[e, :2]).round(4).tolist(), rows=int(e - s + 1)))
        sel = held & (ti == t)
        prof = {}
        for lo, hi in zip(DBINS[:-1], DBINS[1:]):
            m = sel & (d >= lo) & (d < hi)
            if m.any():
                prof[f"{lo:.2f}-{hi:.2f}"] = dict(n=int(m.sum()), h_med=float(np.median(h[m])), sat_share=float(sat[m].mean()),
                                                 vz_med=float(np.median(X[m, :5, 2].mean(1))))
        res[name] = dict(n_carries=len(car), tasks=sorted({c["task"] for c in car}), d0=pct([c["d0"] for c in car], (10, 50, 90, 100)),
                         h0=pct([c["h0"] for c in car]), bear_rel=pct([c["bear_rel"] for c in car], (0, 10, 50, 90, 100)),
                         tyaw=pct([c["tyaw"] for c in car], (0, 50, 100)), d_end=pct([c["d_end"] for c in car]),
                         h_end=pct([c["h_end"] for c in car]), sat_share=float(sat[sel].mean()), profile=prof, carries=car)
        print("[rack] demo", name, json.dumps({k: v for k, v in res[name].items() if k not in ("carries",)}), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_rack_demo.json"), "w", encoding="utf-8"), indent=1)


def roll(a):
    import fa1h_roll as RH
    from fa1g_read import radius
    from pp_close_diag import Wrap
    from pp_p_roll import load_any
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == a.layout)
    rad = radius(t["object"], t["target"], cal)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    eps = []
    for k in [int(x) for x in a.states.split("+")]:
        w = Wrap(m)
        e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
        calls, was, drops, onset_call = [], False, 0, None
        for ci, (bb, x) in enumerate(w.calls):
            hd = float(bb["held"][0]) > 0.5
            if was and not hd:
                drops += 1
            was = hd
            if not hd:
                continue
            if onset_call is None:
                onset_call = ci
            d, h, r = foot_rel(bb)
            st = m._tscale(bb)
            s = float(st[0]) if st is not None else 1.0
            ty = float(bb["yaw"][0, bb["ti"][0]])
            raw = x[0, :RH.REPLAN].numpy()
            cl = np.clip(raw, -1.0, 1.0)
            u = -r[0, :2] / max(float(d[0]), 1e-9)                           # toward the target, horizontal
            calls.append(dict(call=ci, d=float(d[0]), h=float(h[0]), d0t=float(bb["d0t"][0]) if "d0t" in bb else None, s_t=s,
                              d_seen=float(d[0]) * s, bear_rel=float(math.degrees(wrap(float(bb["cyaw"][0]) - ty))),
                              raw_xy=float(np.abs(raw[:, :2]).max()), over1=float((np.abs(raw[:, :2]) > 1.0).mean()),
                              prog=float((cl[:, :2] @ u).mean()), vz=float(cl[:, 2].mean()), grip=float(cl[:, 6].mean())))
        ep = dict(state=k, success=bool(e["success"]), held_step=e["held_step"], bound=e["bound"], steps=e["steps"], drops=drops,
                  n_held_calls=len(calls), radius=rad)
        if calls:
            c0 = calls[0]
            j = int(np.argmin([c["d"] for c in calls]))
            ep.update(d0=c0["d"], h0=c0["h"], s_t=c0["s_t"], bear_rel=c0["bear_rel"], min_d=calls[j]["d"], h_at_min=calls[j]["h"],
                      arrived=bool(e.get("min_dxy_held") is not None and e["min_dxy_held"] <= rad), min_dxy_held=e.get("min_dxy_held"),
                      steps_left=300 - (e["held_step"] or 0), over1_med=float(np.median([c["over1"] for c in calls])),
                      prog_med=float(np.median([c["prog"] for c in calls])), d_end=calls[-1]["d"], h_end=calls[-1]["h"])
        ep["calls"] = calls
        eps.append(ep)
        print("[rack]", a.name, a.task, a.layout, json.dumps({q: (round(v, 3) if isinstance(v, float) else v) for q, v in ep.items() if q != "calls"}), flush=True)
    al = [c for e_ in eps for c in e_["calls"]]
    summ = {}
    for lo, hi in zip(DBINS[:-1], DBINS[1:]):
        sel = [c for c in al if lo <= c["d"] < hi]
        if sel:
            summ[f"{lo:.2f}-{hi:.2f}"] = dict(n=len(sel), h_med=float(np.median([c["h"] for c in sel])),
                                             s_t_med=float(np.median([c["s_t"] for c in sel])),
                                             over1_mean=float(np.mean([c["over1"] for c in sel])),
                                             prog_med=float(np.median([c["prog"] for c in sel])), vz_med=float(np.median([c["vz"] for c in sel])))
    held = [e_ for e_ in eps if e_["n_held_calls"]]
    agg = dict(n=len(eps), held=len(held), arrived=sum(bool(e_.get("arrived")) for e_ in held), success=sum(e_["success"] for e_ in eps),
               d0=pct([e_["d0"] for e_ in held], (10, 50, 90)), s_t=pct([e_["s_t"] for e_ in held]),
               bear_rel=pct([e_["bear_rel"] for e_ in held], (0, 10, 50, 90, 100)), min_d=pct([e_["min_d"] for e_ in held]),
               h_at_min=pct([e_["h_at_min"] for e_ in held]), drops=pct([e_["drops"] for e_ in held]),
               steps_left=pct([e_["steps_left"] for e_ in held]), over1=pct([e_["over1_med"] for e_ in held]))
    print("[rack] agg", a.name, a.task, a.layout, json.dumps(agg), flush=True)
    for q, v in summ.items():
        print("[rack] bin", a.name, a.task, a.layout, q, json.dumps(v), flush=True)
    fn = os.path.join(OUT, f"pp_rack_{a.name}_{a.task}_{a.layout}.json")
    assert not os.path.exists(fn), fn
    json.dump(dict(task=a.task, layout=a.layout, ckpt=a.ckpt, radius=rad, agg=agg, bins=summ, episodes=eps),
              open(fn, "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("demo", "roll"))
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--task")
    ap.add_argument("--layout")
    ap.add_argument("--states", default="+".join(str(k) for k in range(4, 24)))
    a = ap.parse_args()
    demo() if a.mode == "demo" else roll(a)


if __name__ == "__main__":
    main()

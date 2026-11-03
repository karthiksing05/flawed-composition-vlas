"""PLAN_PARTS round diagnostics (descriptive, dev data only): how does the object term approach from a far start?

demo   The demos' coupling of the hand's height to its horizontal distance from the object, per known task (fitted rows, never
       executed): per demo, the start offset (dxy, dz) and the first close (dxy, dz); over not-held rows, the median dz and the
       median commanded vertical step (first chunk step, action z) per dxy bin.
roll   fa1h_roll.episode unchanged, model.sample wrapped (pp_close_diag.Wrap).  Per pre-hold call: the true dxy and dz of the hand
       from the object, the term's apparent dxy (after the object part's radial clip), the message m, the chunk's first gripper
       command and the mean commanded z over the executed steps.  Per episode: the start offset, the dz when the clip releases
       (first call with dxy <= r), the dxy at which the hand first comes within 4 cm of its close height ("low"), the first
       close's (dxy, dz), the number of close calls, held step, the object moved first, success and the yaw at the onset.
LIBERO venv, CPU.  -> results/flawed/pp_far_<name>_<task>_<layout>.json (roll), results/flawed/pp_far_demo.json (demo)

    python experiments/pp_far_diag.py demo
    python experiments/pp_far_diag.py roll --ckpt pp_full_imd_s10/P.npz --name imd10 --task K4 --layout new
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

OUT = os.path.join(ROOT, "results", "flawed")
BINS = [0.0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.18, 0.25, 0.40]


def demo():
    import pp_p_train as PT
    res = {}
    for task in ("K1", "K4", "K5", "K6"):
        b, X, _, _, _, _ = PT.rows(torch.device("cpu"), [task])
        i = torch.arange(len(b["oi"]))
        off = (b["ee_pos"] - b["pos"][i, b["oi"]]).numpy()
        dxy, dz = np.linalg.norm(off[:, :2], axis=1), off[:, 2]
        nh = (b["held"] < 0.5).numpy()
        cl = nh & (X[:, 0, 6] > 0).numpy()
        key = b["dkey"].numpy()
        starts, closes = [], []
        for k in np.unique(key):
            j = np.nonzero(key == k)[0]
            starts.append((float(dxy[j[0]]), float(dz[j[0]])))
            c = j[cl[j]]
            if len(c):
                closes.append((float(dxy[c[0]]), float(dz[c[0]])))
        prof = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = nh & (dxy >= lo) & (dxy < hi)
            if s.sum():
                prof.append(dict(bin=[lo, hi], n=int(s.sum()), dz_med=float(np.median(dz[s])), dz_p10=float(np.percentile(dz[s], 10)),
                                 dz_p90=float(np.percentile(dz[s], 90)), vz_med=float(np.median(X[s.nonzero()[0], 0, 2].numpy()))))
        st, cs = np.array(starts), np.array(closes)
        res[task] = dict(n_demos=len(starts), start_dxy=np.percentile(st[:, 0], [10, 50, 90]).tolist(),
                         start_dz=np.percentile(st[:, 1], [10, 50, 90]).tolist(),
                         close_dxy=np.percentile(cs[:, 0], [10, 50, 90]).tolist(), close_dz=np.percentile(cs[:, 1], [10, 50, 90]).tolist(),
                         profile=prof)
        print("[far] demo", task, json.dumps(res[task]), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_far_demo.json"), "w", encoding="utf-8"), indent=1)


def roll(a):
    import fa1h_roll as RH
    from pp_close_diag import Wrap
    from pp_grasp_modes import yaw2
    from pp_p_roll import load_any
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    dm = json.load(open(os.path.join(OUT, "pp_far_demo.json"), encoding="utf-8"))
    zc = float(dm[a.task if a.task in dm else "K4"]["close_dz"][1])         # the demos' median close height for this object
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == a.layout)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    eps = []
    for k in [int(x) for x in a.states.split("+")]:
        w = Wrap(m)
        e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
        calls = []
        for bb, x in w.calls:
            if float(bb["held"][0]) > 0.5:
                break
            o = (bb["ee_pos"] - bb["pos"][torch.arange(1), bb["oi"]])[0].numpy()
            pr = m.opart[bb["oi"]]
            r = float(m._radius(m.rclip, m.rclip_p, pr).reshape(-1)[0]) if float(m.rclip) > 0 else float("inf")
            d = float(np.linalg.norm(o[:2]))
            mm = int(m.message(bb)[0]) if getattr(m, "nmode", 0) > 0 else None
            calls.append(dict(dxy=d, dz=float(o[2]), app=min(d, r), r=r, m=mm, grip=float(x[0, 0, 6]), vz=float(x[0, :5, 2].mean()),
                              yaw=float(np.degrees(yaw2(bb["ee_rot"].numpy())[0]) / 2)))
        rel = next((c for c in calls if c["dxy"] <= c["r"]), None)
        low = next((c for c in calls if c["dz"] <= zc + 0.04), None)
        cl = [c for c in calls if c["grip"] > 0]
        on = None
        for bb, _ in w.calls:
            if float(bb["held"][0]) > 0.5:
                on = float(np.degrees(yaw2(bb["ee_rot"].numpy())[0]) / 2)
                break
        r = dict(state=k, success=bool(e["success"]), held_step=e["held_step"], bound=e["bound"], n_pre=len(calls),
                 m=calls[0]["m"] if calls else None, r=calls[0]["r"] if calls else None,
                 start=dict(dxy=calls[0]["dxy"], dz=calls[0]["dz"]) if calls else None,
                 at_release=None if rel is None else dict(dxy=rel["dxy"], dz=rel["dz"], call=calls.index(rel)),
                 low=None if low is None else dict(dxy=low["dxy"], dz=low["dz"], call=calls.index(low)),
                 first_close=None if not cl else dict(dxy=cl[0]["dxy"], dz=cl[0]["dz"], call=calls.index(cl[0])), n_close=len(cl),
                 yaw_onset=on, calls=calls)
        eps.append(r)
        print("[far]", a.name, a.task, a.layout, json.dumps({q: v for q, v in r.items() if q != "calls"}), flush=True)
    json.dump(dict(task=a.task, layout=a.layout, ckpt=a.ckpt, close_dz_demo=zc, episodes=eps),
              open(os.path.join(OUT, f"pp_far_{a.name}_{a.task}_{a.layout}.json"), "w", encoding="utf-8"), indent=1)


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

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
    if a.bb_m >= 0:                                                        # diagnostic intervention (P_rtr): backbone-only rows read m = bb_m
        vel = m.velocity

        def velocity(b, x, t):
            on = m.part_on(b, b["pos"][torch.arange(len(b["oi"])), b["oi"]])
            b = dict(b, m=torch.where(on, m.message(b), torch.full_like(m.message(b), a.bb_m)))
            return vel(b, x, t)
        m.velocity = velocity
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
            on = m.part_on(bb, bb["pos"][torch.arange(1), bb["oi"]]) if getattr(m, "rtransport", False) else None
            calls.append(dict(dxy=d, dz=float(o[2]), app=min(d, r), r=r, m=mm, grip=float(x[0, 0, 6]), vz=float(x[0, :5, 2].mean()),
                              yaw=float(np.degrees(yaw2(bb["ee_rot"].numpy())[0]) / 2),
                              backbone_only=None if on is None else bool(not on[0])))       # PARTS_RESULTS 41 (P_rtr)
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


SETTINGS = ("built", "m0", "m2", "none")                                    # PARTS_RESULTS 46: the m read by the backbone-alone path
FBINS = ("in_rs", "rs_0.14", "0.14_rrole", "beyond_rrole")                # horizontal distance bins (rs: the part's start radius)


def bb_chunk(m, bb, setting, size_from=None, draws=4):
    """PARTS_RESULTS 46: the object backbone ALONE (part off on every row, as P_rtr's backbone-only rows; relocated onto rrole), sampled
    with draws fixed noise seeds, under one m setting: built (the model's own rule), m0 / m2 (one-hot m fixed), none (one-hot m = 0).
    size_from: the object's hand-set size replaced by that entity's (the size probe).  -> (draws, H, 7)."""
    from lifelong_ood.policy.factored_flow import temb
    b = dict(bb)
    nom, size = m.bbnom, m.size
    try:
        if setting == "none":
            m.bbnom = True
        elif setting in ("m0", "m2"):
            m.bbnom = False
            b["m"] = torch.full_like(m.message(bb), int(setting[1]))
        if size_from is not None:
            s = size.clone()
            s[bb["oi"][0]] = size[size_from]
            m.size = s
        out = []
        for d in range(draws):
            x = torch.randn(1, m.H, 7, generator=torch.Generator().manual_seed(1000 + d))
            for k in range(10):
                x = x + m._vobj(b, x, temb(torch.full((1,), k / 10)), part_off=True) / 10
            out.append(x[0])
        return torch.stack(out)
    finally:
        m.bbnom, m.size = nom, size


def bb_demo():
    """PARTS_RESULTS 46: the backbone-only term's rows (not-held, every object) per object and task, beyond the bottle's start radius:
    n, median dz, median commanded vertical step, share rising (> 0.1), median commanded horizontal step toward the object."""
    import pp_p_train as PT
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"))
    i = torch.arange(len(b["oi"]))
    off = (b["ee_pos"] - b["pos"][i, b["oi"]]).numpy()
    d, dz = np.linalg.norm(off[:, :2], axis=1), off[:, 2]
    u = -off[:, :2] / np.maximum(d[:, None], 1e-9)
    X = X.numpy()
    vz, prog = X[:, :5, 2].mean(1), (X[:, :5, :2] * u[:, None, :]).sum(-1).mean(1)
    nh = (b["held"] < 0.5).numpy()
    task = (b["dkey"].numpy() // 1000)
    oi = b["oi"].numpy()
    res, KN = {}, list(FC.KNOWN)
    for e in np.unique(oi[nh]):
        for t in np.unique(task[nh & (oi == e)]):
            for lo, hi in ((0.0, 0.05), (0.05, 0.079), (0.079, 0.11), (0.11, 0.14), (0.14, 0.225), (0.225, 9.0)):
                s = nh & (oi == e) & (task == t) & (d >= lo) & (d < hi)
                if not s.any():
                    continue
                hi_ = s & (dz > 0.12)
                res[f"{FC.ENT[e]}|{KN[t]}|{lo}-{hi}"] = dict(
                    n=int(s.sum()), dz_med=float(np.median(dz[s])), vz_med=float(np.median(vz[s])), rise_share=float(np.mean(vz[s] > 0.1)),
                    prog_med=float(np.median(prog[s])), n_high=int(hi_.sum()),
                    vz_med_high=float(np.median(vz[hi_])) if hi_.any() else None, prog_med_high=float(np.median(prog[hi_])) if hi_.any() else None)
                print("[bb] demo", FC.ENT[e], KN[t], lo, hi, json.dumps(res[f"{FC.ENT[e]}|{KN[t]}|{lo}-{hi}"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_bb_demo.json"), "w", encoding="utf-8"), indent=1)


def bb_roll(a):
    """PARTS_RESULTS 46: matched far states.  The driver (--ckpt, --bb-m as in roll) runs fa1h_roll.episode unchanged; at every
    pre-hold call each model of --evals (name=ckpt+...) gives its backbone-alone chunk under each m setting (and size probes), on the
    same state and the same noise.  Per call: dxy, dz, and per (model, setting) the mean commanded vertical step and horizontal step
    toward the object over the first 5 steps.  -> results/flawed/pp_bb_<name>_<task>_<layout>.json"""
    import fa1h_roll as RH
    from pp_close_diag import Wrap
    from pp_p_roll import load_any
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    if a.bb_m >= 0:
        vel = m.velocity

        def velocity(b, x, t):
            on = m.part_on(b, b["pos"][torch.arange(len(b["oi"])), b["oi"]])
            b = dict(b, m=torch.where(on, m.message(b), torch.full_like(m.message(b), a.bb_m)))
            return vel(b, x, t)
        m.velocity = velocity
    ev = {nm: load_any(os.path.join(FC.DATA, "ckpt", ck))[0] for nm, ck in (s.split("=") for s in a.evals.split("+"))}
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == a.layout)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    calls, eps = [], []
    for k in [int(x) for x in a.states.split("+")]:
        w = Wrap(m)
        e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
        eps.append(dict(state=k, success=bool(e["success"]), held_step=e["held_step"], bound=e["bound"]))
        for ci, (bb, _) in enumerate(w.calls):
            if float(bb["held"][0]) > 0.5:
                break
            o = (bb["ee_pos"] - bb["pos"][torch.arange(1), bb["oi"]])[0].numpy()
            d = float(np.linalg.norm(o[:2]))
            u = torch.tensor(-o[:2] / max(d, 1e-9), dtype=torch.float32)
            rec = dict(state=k, call=ci, dxy=d, dz=float(o[2]), m_built=int(m.message(bb)[0]), out={})
            oe = int(bb["oi"][0])
            probes = [p for p in (0, 1, 2) if p != oe]
            for nm, me in ev.items():
                rec["rs"] = float(me.rs_p[me.opart[bb["oi"]].clamp_min(0)][0])
                rec["rrole"] = float(me.rrole)
                for st in SETTINGS + tuple(f"{s_}+size{p}" for s_ in ("built", "m0", "none") for p in probes):
                    s_, _, p = st.partition("+size")
                    x = bb_chunk(me, bb, s_, int(p) if p else None)
                    rec["out"][f"{nm}|{st}"] = dict(vz=float(x[:, :5, 2].mean()), prog=float((x[:, :5, :2] * u).sum(-1).mean()))
            calls.append(rec)
        print("[bb]", a.name, a.task, a.layout, k, json.dumps(eps[-1]), flush=True)
    summ = {}
    for key in (calls[0]["out"] if calls else {}):
        for nm_b in FBINS:
            sel = [c for c in calls if (c["dxy"] <= c["rs"] if nm_b == "in_rs" else
                                        c["rs"] < c["dxy"] <= 0.14 if nm_b == "rs_0.14" else
                                        0.14 < c["dxy"] <= c["rrole"] if nm_b == "0.14_rrole" else c["dxy"] > c["rrole"])]
            if sel:
                vz, pg = np.array([c["out"][key]["vz"] for c in sel]), np.array([c["out"][key]["prog"] for c in sel])
                summ[f"{key}|{nm_b}"] = dict(n=len(sel), vz_med=float(np.median(vz)), rise_share=float(np.mean(vz > 0.1)),
                                             prog_med=float(np.median(pg)), dz_med=float(np.median([c["dz"] for c in sel])))
    for k_, v in summ.items():
        print("[bb] summ", a.name, a.task, a.layout, k_, json.dumps(v), flush=True)
    json.dump(dict(task=a.task, layout=a.layout, driver=a.ckpt, driver_bb_m=a.bb_m, evals=a.evals, episodes=eps, summary=summ, calls=calls),
              open(os.path.join(OUT, f"pp_bb_{a.name}_{a.task}_{a.layout}.json"), "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("demo", "roll", "bbdemo", "bb"))
    ap.add_argument("--evals", default="")                                 # PARTS_RESULTS 46 (bb): name=ckpt+name=ckpt
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--task")
    ap.add_argument("--layout")
    ap.add_argument("--states", default="+".join(str(k) for k in range(4, 24)))
    ap.add_argument("--bb-m", type=int, default=-1)                       # P_rtr diagnostic: m read by backbone-only rows (-1: off)
    a = ap.parse_args()
    dict(demo=lambda: demo(), roll=lambda: roll(a), bbdemo=lambda: bb_demo(), bb=lambda: bb_roll(a))[a.mode]()


if __name__ == "__main__":
    main()

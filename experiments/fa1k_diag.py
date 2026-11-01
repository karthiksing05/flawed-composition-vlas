"""FLAWED_COMPOSITION prototype A, stage 1k DIAGNOSTIC summary (section 5.18; descriptive only): grasp and carry of the cheese, the
bottle and the bowl on familiar against relocated layouts, from results/flawed/a1k_trace (fa1k_trace.py).
Per (model group, object, layout), pooled over seeds and the object's traced tasks:
  hold classes          success / never_closed / closed_no_hold / slipped / released_far / carried_to_target (fa1k_trace)
  grasp pose error      |ee - object| in the OBJECT frame at the first close command, minus the median of the same quantity over
                        the training episodes of that object (demos.npz + gen_g.npz, first close command per episode); xy and z
  support               the episode's start object position (xy) and yaw against the training starts of that object (step-0 rows of
                        demos + generated): nearest-neighbour xy distance, and whether it exceeds the 95th percentile of the training
                        starts' own leave-one-out nearest-neighbour distances (out_xy); yaw distance (mod pi) to the nearest training
                        yaw, out_yaw if > 0.2 rad
  success | in / out of support   (descriptive)
Also: the training starts per object (demo vs generated): count, xy range, yaw range.
openpi venv, CPU -> results/flawed/a1k_diag.json

    python experiments/fa1k_diag.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")
CLASSES = ["success", "never_closed", "closed_no_hold", "slipped", "released_far", "carried_to_target"]
YAW_OUT = 0.2


def objframe(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def ydist(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b) + math.pi / 2) % math.pi - math.pi / 2)
    return d


def training(n2):
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    G = dict(np.load(os.path.join(FC.DATA, "gen_g.npz")))
    per = {}
    for src, X in (("demo", D), ("gen", G)):
        for ti, k in enumerate(FC.KNOWN):
            if k not in lang:
                continue
            oe = FC.roles(lang[k])[0]
            oi = FC.ENT.index(oe)
            m = X["task"] == ti
            for dm in np.unique(X["demo"][m]):
                idx = np.nonzero(m & (X["demo"] == dm))[0]
                idx = idx[np.argsort(X["step"][idx])]
                p = per.setdefault(oe, dict(xy=[], yaw=[], src=[], grasp=[], task=[]))
                p["xy"].append(X["pos"][idx[0], oi, :2]); p["yaw"].append(float(X["yaw"][idx[0], oi])); p["src"].append(src)
                p["task"].append(k)
                a = X["action"][idx, 6]
                if (a > 0).any():
                    i = idx[int(np.argmax(a > 0))]
                    p["grasp"].append(objframe(X["ee_pos"][i] - X["pos"][i, oi], float(X["yaw"][i, oi])))
    out = {}
    for oe, p in per.items():
        xy, yaw, src = np.array(p["xy"]), np.array(p["yaw"]), np.array(p["src"])
        dd = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 9
        out[oe] = dict(xy=xy, yaw=yaw, src=src, grasp_med=np.median(np.array(p["grasp"]), 0), thr=float(np.percentile(dd.min(1), 95)),
                       summary={s: dict(n=int((src == s).sum()), x=[float(xy[src == s, 0].min()), float(xy[src == s, 0].max())] if (src == s).any() else None,
                                        y=[float(xy[src == s, 1].min()), float(xy[src == s, 1].max())] if (src == s).any() else None,
                                        yaw=[float(yaw[src == s].min()), float(yaw[src == s].max())] if (src == s).any() else None,
                                        tasks=sorted(set(np.array(p["task"])[src == s].tolist())))
                                 for s in ("demo", "gen")})
        out[oe]["summary"]["grasp_med_objframe"] = out[oe]["grasp_med"].tolist()
        out[oe]["summary"]["grasp_mad_objframe"] = np.median(np.abs(np.array(p["grasp"]) - out[oe]["grasp_med"]), 0).tolist()
        out[oe]["summary"]["nn_thr_95"] = out[oe]["thr"]
    return out


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    tr = training(n2)
    groups = {}
    for f in sorted(glob.glob(os.path.join(OUT, "a1k_trace", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        T = tr[d["object"]]
        g = groups.setdefault((d["name"], d["object"], d["layout"]), dict(rows=[], reproduced=[], tasks=set()))
        g["tasks"].add(d["key"])
        if d["reproduced"] is not None:
            g["reproduced"].append(d["reproduced"])
        for e in d["episodes"]:
            if e.get("start") is None:
                continue
            xy, yaw = np.array(e["start"]["obj"][:2]), e["start"]["yaw"]
            nn = float(np.linalg.norm(T["xy"] - xy, axis=1).min())
            yd = float(ydist(T["yaw"], yaw).min())
            ge = (np.array(e["grasp_ee_obj"]) - T["grasp_med"]) if "grasp_ee_obj" in e else None
            g["rows"].append(dict(cls=e["hold_class"], success=e["success"], nn=nn, out_xy=nn > T["thr"], yd=yd, out_yaw=yd > YAW_OUT,
                                  yaw=yaw, gxy=None if ge is None else float(np.linalg.norm(ge[:2])), gz=None if ge is None else float(ge[2]),
                                  lift=e.get("max_lift"), task=d["key"]))
    rep = {}
    for (nm, ob, lay), g in sorted(groups.items()):
        R = g["rows"]
        n = len(R)
        sh = {c: float(np.mean([r["cls"] == c for r in R])) for c in CLASSES}
        gx = [r["gxy"] for r in R if r["gxy"] is not None]
        gz = [r["gz"] for r in R if r["gz"] is not None]
        out = [r["out_xy"] or r["out_yaw"] for r in R]
        so = [r["success"] for r, o in zip(R, out) if o]
        si = [r["success"] for r, o in zip(R, out) if not o]
        ok = [r for r in R if r["success"]]
        bad = [r for r in R if not r["success"]]
        rep[f"{nm}|{ob}|{lay}"] = dict(
            name=nm, object=ob, layout=lay, tasks=sorted(g["tasks"]), n=n,
            reproduced=float(np.mean(g["reproduced"])) if g["reproduced"] else None, classes=sh,
            grasp_err_xy_med=float(np.median(gx)) if gx else None, grasp_err_z_med=float(np.median(gz)) if gz else None,
            grasp_err_xy_med_success=float(np.median([r["gxy"] for r in ok if r["gxy"] is not None])) if ok else None,
            grasp_err_xy_med_fail=float(np.median([r["gxy"] for r in bad if r["gxy"] is not None])) if [r for r in bad if r["gxy"] is not None] else None,
            nn_med=float(np.median([r["nn"] for r in R])), out_xy=float(np.mean([r["out_xy"] for r in R])),
            yaw_dist_med=float(np.median([r["yd"] for r in R])), out_yaw=float(np.mean([r["out_yaw"] for r in R])),
            start_yaw_range=[float(min(r["yaw"] for r in R)), float(max(r["yaw"] for r in R))],
            out_support=float(np.mean(out)), success_out=float(np.mean(so)) if so else None, n_out=len(so),
            success_in=float(np.mean(si)) if si else None, n_in=len(si))
        x = rep[f"{nm}|{ob}|{lay}"]
        print(f"[fa1k] {nm:6s} {ob:20s} {lay:8s} n={n:3d} repro={x['reproduced']} " + " ".join(f"{c}={v:.2f}" for c, v in sh.items())
              + f" | gerr_xy={x['grasp_err_xy_med']} (ok {x['grasp_err_xy_med_success']}, fail {x['grasp_err_xy_med_fail']}) gerr_z={x['grasp_err_z_med']}"
              + f" | nn={x['nn_med']:.3f} out_xy={x['out_xy']:.2f} yawd={x['yaw_dist_med']:.2f} out_yaw={x['out_yaw']:.2f} yaw={x['start_yaw_range']}"
              + f" | succ in {x['success_in']} (n {x['n_in']}) out {x['success_out']} (n {x['n_out']})", flush=True)
    for ob, T in tr.items():
        print(f"[fa1k] support {ob}: {json.dumps(T['summary'])}", flush=True)
    with open(os.path.join(OUT, "a1k_diag.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(groups=rep, support={ob: T["summary"] for ob, T in tr.items()}, yaw_out=YAW_OUT, descriptive=True), fh, indent=1)


if __name__ == "__main__":
    main()

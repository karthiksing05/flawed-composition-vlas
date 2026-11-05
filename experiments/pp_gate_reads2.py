"""Reads E and F on existing tracks (docs/HAND_OVER.md 4; m3_prereg pp_gate_reads).  CPU, no rollouts.  -> results/flawed/pp_gate_reads2.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402
from pp_l1_floor import kp_for                                              # noqa: E402

n2 = PA.tasks()
ks = n2["states"]["main"]
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
GD = os.path.join(PA.OUT, "pp_gate")
R_REACH, R_PUSH, R_STALL, LAST = 0.05, 0.03, 0.02, 50
CLASSES = ("PUSHED", "NEVER REACHED", "NO CLOSE", "CLOSED, NO HOLD")


def episodes(stem):
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    Tr = PA.load_track(stem + "_track.npz")
    assert len(E) == len(ks) == len(Tr), stem
    return E, Tr


def grasp_class(r, off):
    a = np.linalg.norm(r["ee"] - (r["obj"] + off), axis=1)
    disp = np.linalg.norm(r["obj"] - r["obj"][0], axis=1)
    near = np.flatnonzero(a <= R_REACH)
    first = int(near[0]) if len(near) else len(a)
    if (disp[:first] > R_PUSH).any():
        return "PUSHED", float(a.min())
    if not len(near):
        return "NEVER REACHED", float(a.min())
    if not (r["a6"][near] > 0).any():
        return "NO CLOSE", float(a.min())
    return "CLOSED, NO HOLD", float(a.min())


def read_e():
    out = {}
    for name, fmt in (("pi05", "pi05_{t}"), ("groot", "groot_{t}"), ("pmin_s10", "pmin_s10_{t}"), ("servo", "servo_s10_{t}")):
        count, mins, per, n = {c: 0 for c in CLASSES}, [], {}, 0
        for c in PA.NEW:
            t = tid[(c, "new")]
            off = np.array(kp_for(c)["grasp_off"], np.float64)
            E, Tr = episodes(os.path.join(PA.OD, fmt.format(t=t["tid"])))
            per[c] = {x: 0 for x in CLASSES}
            for e, r in zip(E, Tr):
                d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
                if PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0] != "GRASP":
                    continue
                k, m = grasp_class(r, off)
                count[k] += 1
                per[c][k] += 1
                mins.append(m)
                n += 1
        share = {k: (count[k] / n if n else 0.0) for k in CLASSES}
        top = [k for k in CLASSES if share[k] > 0.5]
        out[name] = dict(n=n, share=share, verdict=top[0] if top else "MIXED", min_a_quartiles=np.percentile(mins, [25, 50, 75]).tolist() if mins else None, per_task=per)
        print("E", name, json.dumps(out[name]), flush=True)
    return out


def read_f():
    out = {}
    for model in ("pi05", "groot"):
        out[model] = {}
        for arm in ("G", "GA"):
            net, dend, ever3 = [], [], []
            for c in PA.NEW:
                t = tid[(c, "familiar")]
                E, Tr = episodes(os.path.join(GD, f"{model}_{arm}_{t['tid']}"))
                for e, r in zip(E, Tr):
                    d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
                    if e["success"] or not r["held"][-1] or d[-1] <= R_REACH:
                        continue
                    k = max(0, len(d) - LAST)
                    net.append(float(np.linalg.norm((r["obj"][-1] - r["obj"][k])[:2])))
                    dend.append(float(d[-1]))
                    ever3.append(float(d[r["held"]].min() <= PA.SEAT))
            stall = float(np.mean(np.array(net) < R_STALL)) if net else None
            out[model][arm] = dict(n=len(net), stall_share=stall, median_net=float(np.median(net)) if net else None, median_d_end=float(np.median(dend)) if dend else None, ever_within_3cm=float(np.mean(ever3)) if ever3 else None)
        out[model]["verdict"] = "STALLS" if (out[model]["G"]["stall_share"] or 0.0) > 0.5 else "KEEPS MOVING"
        print("F", model, json.dumps(out[model]), flush=True)
    return out


res = dict(E=read_e(), F=read_f())
json.dump(res, open(os.path.join(PA.OUT, "pp_gate_reads2.json"), "w", encoding="utf-8"), indent=1)

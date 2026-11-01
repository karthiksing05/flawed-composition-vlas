"""FLAWED_COMPOSITION prototype A, stage 1o DIAGNOSTIC (section 5.29; descriptive only): where the cheese sits when a new-layout
grasp fails.  From the traced episodes (results/flawed/a1k_trace, fa1k_trace.py) of the chosen models on the cheese tasks, per
episode start: the xy clearance from the cheese to every other entity (centre distance minus that entity's hand-set footprint
radius; the table reference slot excluded) and the nearest entity.  Per (model, layout, outcome): the median nearest clearance,
the nearest entity's identity counts, and the success rate per clearance bin.  The same start-state statistic over the training
data's cheese carries (demos, generated) says which configurations the data covers.
-> results/flawed/a1o_geom.json

    python experiments/fa1o_geom.py --names Ao1+An+Ak
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")
OI = FC.ENT.index("cream_cheese_1")
OTHERS = [i for i in range(len(FC.ENT)) if i not in (OI, 8)]
RAD = np.array([FC.SIZE[e][0] for e in FC.ENT])
BINS = (0.0, 0.03, 0.06, 0.10, 1.0)


def nearest(pos):
    pos = np.asarray(pos)
    c = np.array([np.linalg.norm(pos[i, :2] - pos[OI, :2]) - RAD[i] for i in OTHERS])
    k = int(np.argmin(c))
    return float(c[k]), FC.ENT[OTHERS[k]]


def summary(rows):
    c = np.array([r[0] for r in rows])
    ids = {}
    for r in rows:
        ids[r[1]] = ids.get(r[1], 0) + 1
    return dict(n=len(rows), clr_med_cm=float(100 * np.median(c)) if len(c) else None,
                clr_iqr_cm=[float(100 * np.percentile(c, q)) for q in (25, 75)] if len(c) else None, nearest=ids,
                bins={f"{BINS[i]:.2f}-{BINS[i + 1]:.2f}": int(((c >= BINS[i]) & (c < BINS[i + 1])).sum()) for i in range(len(BINS) - 1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="Ao1+An+Ak")
    ap.add_argument("--gen", default="gen_lo.npz")
    a = ap.parse_args()
    out = {"episodes": {}, "data": {}}
    for nm in a.names.split("+"):
        g = {}
        for f in sorted(glob.glob(os.path.join(OUT, "a1k_trace", f"{nm}_s*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            if d["name"] != nm or d["object"] != "cream_cheese_1":
                continue
            for e in d["episodes"]:
                if not e.get("start"):
                    continue
                cls = "success" if e["success"] else e["hold_class"]
                c, who = nearest(e["start"]["all"])
                g.setdefault((d["layout"], cls), []).append((c, who))
                g.setdefault((d["layout"], "all_" + ("success" if e["success"] else "fail")), []).append((c, who))
        for k, rows in sorted(g.items()):
            s = summary(rows)
            out["episodes"][f"{nm}|{k[0]}|{k[1]}"] = s
            print(f"[fa1o] geom {nm:4s} {k[0]:8s} {k[1]:18s} {json.dumps(s)}", flush=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    for src, fn in (("demos", "demos.npz"), ("generated", a.gen)):
        D = dict(np.load(os.path.join(FC.DATA, fn)))
        task, step = np.asarray(D["task"]), np.asarray(D["step"])
        m = np.nonzero((step == 0) & np.array([rl[int(t)][0] == "cream_cheese_1" for t in task]))[0]
        rows = [nearest(D["pos"][i]) for i in m]
        s = summary(rows)
        out["data"][src] = s
        print(f"[fa1o] geom data {src:9s} {json.dumps(s)}", flush=True)
    with open(os.path.join(OUT, "a1o_geom.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

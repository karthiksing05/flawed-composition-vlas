"""SR 86 Q2 part c (exploratory): reads sr86_q2b.json + sr86_q2.json + the wa.json query records.  (1) which part of the held channel error is the object's (err_obj_cm) vs the plate's (err_tgt_cm);
(2) correlation of the first-release offset with the channel error, W3 vs RAA-sL; (3) classification of every success->failure flip.   python experiments/sr86_q2c.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr86_q2 as Q
D = Q.D
rows = json.load(open(os.path.join(D, "sr86_q2b.json"), encoding="utf-8"))
fl = json.load(open(os.path.join(D, "sr86_q2.json"), encoding="utf-8"))["flips"]


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


for arm in ("wasw3", "wasl"):
    eo, et = [], []
    for tid in (0, 1, 4, 5, 9):
        js = json.load(open(os.path.join(D, arm, "full_%d_wa.json" % tid), encoding="utf-8"))
        for e in js["eps"]:
            q = [x for x in e["queries"] if x.get("held")]
            if q:
                eo.append(np.median([x["err_obj_cm"] for x in q])); et.append(np.median([x["err_tgt_cm"] for x in q]))
    print(arm, "held queries per-episode median: err_obj_cm med %.2f p90 %.2f | err_tgt_cm med %.2f p90 %.2f (n=%d)" % (np.median(eo), np.percentile(eo, 90), np.median(et), np.percentile(et, 90), len(eo)))
    R = [r for r in rows if r["arm"] == arm and r.get("off_xy") is not None and r.get("err_at_release") is not None]
    x = np.array([r["err_at_release"] for r in R]); y = np.array([r["off_xy"] for r in R])
    print(arm, "first-release offset vs channel error at release: n=%d pearson %.2f spearman %.2f; median(off - err) %.2f; off/err median %.2f" % (
        len(R), np.corrcoef(x, y)[0, 1], np.corrcoef(rank(x), rank(y))[0, 1], np.median(y - x), np.median(y / np.maximum(x, .2))))
# flips classification (W3 loses)
L = [f for f in fl if f["kind"] == "lose"]
byw = {(r["tid"], r["trial"]): r for r in rows if r["arm"] == "wasw3"}
cls = {}
for f in L:
    r = byw.get((f["tid"], f["trial"]))
    if r is None: c = "other cell (K1t / K2t)"
    elif r.get("off_xy") is None: c = "no release of a carry"
    elif r["off_xy"] >= 3.0: c = "first release >= 3 cm off the plate"
    else: c = "first release < 3 cm off"
    cls.setdefault(c, []).append((f["cell"], f["trial"], round(f["f3"].get("held_err_cm_med") or -1, 2)))
for c, v in cls.items(): print(c, len(v), "med held err %.2f" % np.median([x[2] for x in v]), {k: sum(1 for z in v if z[0] == k) for k in sorted(set(z[0] for z in v))})
ss = [f for f in fl if f["kind"] == "ss"]
print("ss med held err %.2f; lose med held err %.2f" % (np.median([f["f3"]["held_err_cm_med"] for f in ss if f["f3"]["held_err_cm_med"] is not None and f["cell"] not in ("K1t", "K2t")]), np.median([f["f3"]["held_err_cm_med"] for f in L if f["f3"]["held_err_cm_med"] is not None and f["cell"] not in ("K1t", "K2t")])))

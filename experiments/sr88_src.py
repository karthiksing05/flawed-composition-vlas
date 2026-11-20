"""SR 88: held-object estimate error vs success on the SR 87 Phase B cells (all 60 fresh states per arm; read only).  err_obj_cm per held query = the channel's
object estimate vs the simulator body (wa.py's per-query log)."""
import json
import numpy as np
RW = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/raasw5"
CELLS = {"uniL": 4, "F6_b": 8, "F6_a": 9, "F7relL": 0, "F7relR": 1, "uniR": 5}
for cell, tid in CELLS.items():
    print("== %s (tid %d)" % (cell, tid))
    for arm in ("w5", "w4", "w5c1", "w5c2", "raasl"):
        d = json.load(open("%s/b_%s/full_%d_wa.json" % (RW, arm, tid)))
        res = json.load(open("%s/b_%s/full_%d.json" % (RW, arm, tid)))["tasks"][str(tid)]["episodes"]
        succ = {e["trial"]: bool(e["success"]) for e in res}
        rows = []
        for e in d["eps"]:
            q = [x for x in e["queries"] if x["held"]]
            if not q:
                continue
            er = np.array([x["err_obj_cm"] for x in q])
            mem = np.mean([x["src_obj"] != "agent" for x in q])
            rows.append((e["trial"], float(np.median(er)), float(np.percentile(er, 90)), float(mem), succ.get(e["trial"], False)))
        if not rows:
            continue
        a = np.array([[r[1], r[2], r[3], float(r[4])] for r in rows])
        s = a[:, 3] > 0.5
        out = ["n=%d succ=%.2f" % (len(rows), s.mean()), "held err med: ok=%.2f fail=%.2f" % (np.median(a[s, 0]) if s.any() else float("nan"), np.median(a[~s, 0]) if (~s).any() else float("nan")),
               "mem frac: ok=%.2f fail=%.2f" % (a[s, 2].mean() if s.any() else float("nan"), a[~s, 2].mean() if (~s).any() else float("nan"))]
        for lo, hi in ((0, 1), (1, 2), (2, 99)):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            out.append("err %g-%g: n=%d succ=%s" % (lo, hi, m.sum(), None if not m.any() else round(float(a[m, 3].mean()), 2)))
        print("  %-5s " % arm + " | ".join(out))

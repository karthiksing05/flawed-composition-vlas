"""SR 88 (exploratory diagnosis): paired outcomes of the SR 87 rollouts on the five weak cells; picks the states for the instrumented re-runs.
Reads results/flawed/raasw5 (read only).  Writes results/flawed/sr88/pick.json and br_<cell>.json (branch subsets)."""
import json, os, sys
import numpy as np
R = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
RW = R + "/raasw5"
CELLS = {"N4": ("a", 22), "N6": ("a", 26), "K6": ("a", 10), "K1": ("a", 0), "F6_b": ("b", 8), "uniL": ("b", 4)}
ARMS = ["w5", "w4", "w5c1", "w5c2", "raasl"]
def load(ph, arm, tid):
    p = "%s/%s_%s/%s_main_%d.json" % (RW, ph, arm, arm, tid) if ph == "a" else "%s/%s_%s/full_%d.json" % (RW, ph, arm, tid)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t = d["tasks"][str(tid)]
    return {e["trial"]: e for e in t["episodes"]}
out = {}
for cell, (ph, tid) in CELLS.items():
    ep = {a: load(ph, a, tid) for a in ARMS}
    print("== %s (phase %s tid %d)" % (cell, ph, tid), {a: (len(v) if v else None) for a, v in ep.items()})
    tr = sorted(ep["w5"].keys())
    rows = []
    for t in tr:
        s = {a: (int(bool(ep[a][t]["success"])) if ep[a] and t in ep[a] else -1) for a in ARMS}
        rows.append((t, s, ep["w5"][t].get("branch")))
    n = len(rows)
    print(" success:", {a: round(sum(1 for r in rows if r[1][a] == 1) / n, 3) for a in ARMS})
    lose4 = [r[0] for r in rows if r[1]["w5"] == 0 and r[1]["w4"] == 1]
    lose1 = [r[0] for r in rows if r[1]["w5"] == 0 and r[1]["w5c1"] == 1]
    gain4 = [r[0] for r in rows if r[1]["w5"] == 1 and r[1]["w4"] == 0]
    ff = [r[0] for r in rows if r[1]["w5"] == 0 and r[1]["w4"] == 0 and r[1]["w5c1"] == 0]
    w5f = [r[0] for r in rows if r[1]["w5"] == 0]
    print(" W5 fails:", w5f)
    print(" lose vs W4:", lose4, " lose vs W5c1:", lose1, " gain vs W4:", gain4, " fail all three:", ff)
    out[cell] = dict(phase=ph, tid=tid, rows=[(r[0], r[1], r[2]) for r in rows], lose_w4=lose4, lose_w5c1=lose1, gain_w4=gain4, ff=ff, w5_fail=w5f)
json.dump(out, open(R + "/sr88/pick.json", "w"), indent=1)
# inspect one track npz and wa json (keys only)
p = RW + "/b_w5/full_8_track.npz"
z = np.load(p, allow_pickle=True)
print("track keys", {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
w = json.load(open(RW + "/b_w5/full_8_wa.json"))
print("wa json type", type(w).__name__, (list(w.keys())[:20] if isinstance(w, dict) else len(w)))

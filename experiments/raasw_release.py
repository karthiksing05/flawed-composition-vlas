"""SR 84 EXPLORATORY (not registered): release xy error on F6_b (rgc tid 8) for RAA-sL (SR 82, results/flawed/wasl), W1 (wasw1), W2 (wasw2); arms full and noM.
Read-only on <dir>/<arm>_8.json, _wa.json, _track.npz and chan_*.jsonl (same table as experiments/sr82_diag.py release_table).  Numpy only; run via sbatch cpu.
  python experiments/raasw_release.py        (refuses to overwrite results/flawed/raasw_release.json)
"""
import collections
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL = os.path.join(ROOT, "results", "flawed")
OUTP = os.path.join(FL, "raasw_release.json")
TID = 8
BINS = ((0, 3), (3, 5), (5, 7), (7, 99))


def chan(d, arm):
    out = collections.defaultdict(list)
    for f in sorted(os.listdir(d)):
        if f.startswith("chan_") and f.endswith(".jsonl"):
            for l in open(os.path.join(d, f), encoding="utf-8"):
                r = json.loads(l)
                if r["tid"] == TID and r["arm"] == arm:
                    out[r["ep"]].append(r)
    return out


def table(d, arm):
    J = json.load(open(f"{d}/{arm}_{TID}.json", encoding="utf-8"))["tasks"][str(TID)]["episodes"]
    W = json.load(open(f"{d}/{arm}_{TID}_wa.json", encoding="utf-8"))["eps"]
    C = chan(d, arm)
    Z = np.load(f"{d}/{arm}_{TID}_track.npz")
    names = list(Z["names"])
    b, p = names.index("akita_black_bowl_1"), names.index("plate_1")
    rows, nolift, norel = [], 0, 0
    for i, w in enumerate(W):
        s = bool([e for e in J if e["trial"] == w["trial"]][0]["success"])
        a6, pos = Z[f"e{i}_a6"], Z[f"e{i}_pos"]
        close = a6 > 0
        lift = np.nonzero(pos[:, b, 2] > pos[0, b, 2] + 0.03)[0]
        if not len(lift):
            nolift += 1
            continue
        ch = [c for c in np.nonzero(np.diff(close.astype(int)))[0] + 1 if not close[c] and c > lift[0]]
        if not ch:
            norel += 1
            continue
        o = int(ch[0])
        xy = 100 * np.linalg.norm(pos[o][b][:2] - pos[o][p][:2])
        tq = [r for r in C.get(w["trial"], []) if r["role"] == "target" and r["t"] <= o]
        rx = 100 * float(np.linalg.norm(np.array(tq[-1]["rel_cm"][:2]))) if tq else float("nan")
        rows.append((s, o, xy, rx, len(tq)))
    R = np.array(rows, float) if rows else np.zeros((0, 5))
    res = dict(n_episodes=len(W), n_release_after_lift=len(R), no_lift=nolift, lift_no_release=norel, success=sum(bool(e["success"]) for e in J))
    if len(R):
        res.update(release_step_median=float(np.median(R[:, 1])), bowl_plate_xy_cm_median=float(np.median(R[:, 2])), bowl_plate_xy_cm_p25_p75=[float(np.percentile(R[:, 2], 25)), float(np.percentile(R[:, 2], 75))],
                   last_target_query_rel_xy_cm_median=float(np.nanmedian(R[:, 3])), frac_xy_under_3cm=float(np.mean(R[:, 2] < 3)),
                   by_bin={"[%d,%d)" % bn: dict(n=int(((R[:, 2] >= bn[0]) & (R[:, 2] < bn[1])).sum()),
                                                success=(float(R[(R[:, 2] >= bn[0]) & (R[:, 2] < bn[1]), 0].mean()) if ((R[:, 2] >= bn[0]) & (R[:, 2] < bn[1])).any() else None)) for bn in BINS})
    return res


if __name__ == "__main__":
    assert not os.path.exists(OUTP), OUTP
    out = {}
    for name, sub in (("RAA-sL", "wasl"), ("W1", "wasw1"), ("W2", "wasw2")):
        for arm in ("full", "noM"):
            d = os.path.join(FL, sub)
            if os.path.exists(f"{d}/{arm}_{TID}_track.npz"):
                out["%s|%s" % (name, arm)] = table(d, arm)
                print("[raasw-release]", name, arm, json.dumps(out["%s|%s" % (name, arm)]), flush=True)
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("[raasw-release] -> " + OUTP, flush=True)

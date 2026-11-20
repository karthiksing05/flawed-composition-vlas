"""SR 86 Q2 part b (exploratory): release offset (simulator truth, plate-object xy at the first release of a carry) vs the channel's held-object error, W3 vs RAA-sL, per cell.
    python experiments/sr86_q2b.py --out results/flawed/sr86_q2b.json   (refuses to overwrite)"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr86_q2 as Q
D = Q.D


def first_release(a6, pos_o, pos_t):
    """first step after >= 8 consecutive closed-command steps with the object lifted >= 2 cm above its start where the command turns open (< 0); None if never"""
    closed = a6 > 0
    lifted = pos_o[:, 2] > pos_o[0, 2] + 0.02
    run = 0
    for i in range(len(a6)):
        run = run + 1 if (closed[i] and lifted[i]) else 0
        if run >= 8 and i + 1 < len(a6) and a6[i + 1] <= 0 and False: pass
    idx = np.where(closed & lifted)[0]
    if not len(idx): return None
    # carry = first contiguous block of closed&lifted; release = first open command after its start
    s = idx[0]
    op = np.where(a6[s:] <= 0)[0]
    return int(s + op[0]) if len(op) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    LG = {k: Q.load_logs(k)[0] for k in ("wasw3", "wasl")}
    rows = []
    for tid in (0, 1, 4, 5, 9):
        for arm in ("wasw3", "wasl"):
            E = Q.eps(arm, tid)
            js = json.load(open(os.path.join(D, arm, "full_%d_wa.json" % tid), encoding="utf-8"))
            trk = np.load(os.path.join(D, arm, "full_%d_track.npz" % tid), allow_pickle=True)
            names = [str(x) for x in trk["names"]]; io, it = names.index(js["roles"][0]), names.index(js["roles"][1])
            for k, e in enumerate(js["eps"]):
                t = int(e["trial"]); pos, a6 = trk["e%d_pos" % k], trk["e%d_a6" % k]
                r = first_release(a6, pos[:, io], pos[:, it])
                held = [c for c, w in LG[arm].get((tid, t), []) if c["role"] == "target"]
                row = dict(tid=tid, cell=Q.KEYS[tid], arm=arm, trial=t, succ=bool(E[t]["success"]), steps=int(E[t]["steps"]), release=r,
                           held_err_med=float(np.median([c["err_cm"] for c in held])) if held else None)
                if r is not None:
                    d = pos[r, io] - pos[r, it]
                    row.update(off_xy=100 * float(np.linalg.norm(d[:2])), off_dz=100 * float(d[2]))
                    ch = [c["err_cm"] for c in held if c["t"] <= r]
                    row["err_at_release"] = float(np.mean(ch[-4:])) if ch else None
                    ok = np.where(trk["e%d_success" % k] if False else np.zeros(1, bool))[0]
                rows.append(row)
    json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=0)
    import collections
    for cell in sorted(set(r["cell"] for r in rows)):
        for arm in ("wasw3", "wasl"):
            R = [r for r in rows if r["cell"] == cell and r["arm"] == arm]
            off = [r["off_xy"] for r in R if "off_xy" in r]
            em = [r["held_err_med"] for r in R if r["held_err_med"] is not None]
            print("%-9s %-6s n=%d succ=%.3f released=%d off_xy med %.2f p75 %.2f frac>3cm %.2f | held err med %.2f | multi-release n/a" % (
                cell, arm, len(R), np.mean([r["succ"] for r in R]), len(off), np.median(off) if off else -1, np.percentile(off, 75) if off else -1,
                np.mean(np.array(off) > 3) if off else -1, np.median(em) if em else -1))
    # success vs channel error bins (held_err_med), pooled over these cells
    for arm in ("wasw3", "wasl"):
        R = [r for r in rows if r["arm"] == arm and r["held_err_med"] is not None and r["cell"] != "K2t"]
        for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)):
            S = [r["succ"] for r in R if lo <= r["held_err_med"] < hi]
            print("bin err %.1f-%.1f %-6s n=%3d succ=%.3f" % (lo, hi, arm, len(S), np.mean(S) if S else -1))
        # first-release offset <3 vs >=3
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 99)):
            S = [r["succ"] for r in R if "off_xy" in r and lo <= r["off_xy"] < hi]
            print("bin off_xy %.1f-%.1f %-6s n=%3d succ=%.3f" % (lo, hi, arm, len(S), np.mean(S) if S else -1))


if __name__ == "__main__":
    main()

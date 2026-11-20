"""SR 86 Q2 timelines (exploratory): per-query rows for chosen (tid, trial) of wasw3 and wasl full arm.  python experiments/sr86_q2t.py --out results/flawed/sr86_q2t.txt 4:6 4:7 ..."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr86_q2 as Q
D = Q.D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("cells", nargs="+"); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    LG = {k: Q.load_logs(k)[0] for k in ("wasw3", "wasl")}
    o = open(a.out, "w", encoding="utf-8")
    for cell in a.cells:
        tid, trial = (int(x) for x in cell.split(":"))
        for arm in ("wasw3", "wasl"):
            js = json.load(open(os.path.join(D, arm, "full_%d_wa.json" % tid), encoding="utf-8"))
            trk = np.load(os.path.join(D, arm, "full_%d_track.npz" % tid), allow_pickle=True)
            k = [int(e["trial"]) for e in js["eps"]].index(trial)
            names = [str(x) for x in trk["names"]]; io, it = names.index(js["roles"][0]), names.index(js["roles"][1])
            pos, ee, a6 = trk["e%d_pos" % k], trk["e%d_ee" % k], trk["e%d_a6" % k]
            suc = bool(trk["e%d_success" % k])
            o.write("== %s tid %d (%s) trial %d success %s names %s\n" % (arm, tid, Q.KEYS[tid], trial, suc, names))
            o.write("t role own c_cm relxy_cm err_cm | expxy wxy | g_old | obj_z ee_z a6 | obj-tgt xy cm, dz cm | ee-obj xy cm\n")
            for c, w in LG[arm][(tid, trial)]:
                t = int(c["t"]); tt = min(t, len(pos) - 1)
                d = pos[tt, io] - pos[tt, it]
                o.write("%3d %s %d %5.1f %5.1f %4.1f | %+.2f %+.2f  %+.2f %+.2f | %+.2f | %.3f %.3f %+.2f | %5.1f %5.1f %5.1f | %4.1f\n" % (
                    t, c["role"][:3], int(bool(w["own"])), 100 * w["c"], 100 * w["rel_xy"], c["err_cm"], w["exp_xy"][0], w["exp_xy"][1], w["w_xy"][0], w["w_xy"][1], w["g_old"],
                    pos[tt, io, 2], ee[tt, 2], a6[tt], 100 * d[0], 100 * d[1], 100 * d[2], 100 * np.linalg.norm(ee[tt, :2] - pos[tt, io, :2])))
    o.close()


if __name__ == "__main__":
    main()

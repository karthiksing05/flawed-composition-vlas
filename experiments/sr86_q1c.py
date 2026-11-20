"""SR 86 Q1 part c (exploratory): per-step timelines of chosen K5 episodes from experiments/sr86_instr.py output.  python experiments/sr86_q1c.py --out results/flawed/sr86_q1c.txt W3:24 SL:24 ..."""
import argparse, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed", "sr86")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=4); ap.add_argument("eps", nargs="+"); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    o = open(a.out, "w", encoding="utf-8")
    for e in a.eps:
        arm, t = e.split(":"); dd = {"W3": "s86w3", "SL": "s86w0"}[arm]
        d = np.load(os.path.join(D, dd, "instr_raac_8_t%s.npz" % t), allow_pickle=True)
        obj, tgt, eef, A = (np.asarray(d[k], float) for k in ("obj", "tgt", "eef", "a")); grip = np.asarray(d["grip"], float); cons = [str(s) for s in d["cons"]]
        under = np.asarray(d["under"]).reshape(-1); con = np.asarray(d["contact"]).reshape(-1); succ = np.asarray(d["succ"]).reshape(-1)
        # query alignment: chan/where
        cf = sorted(glob.glob(os.path.join(D, dd, "chan_*.jsonl")))[0]; wf = cf.replace("chan_", "where_")
        ch = [json.loads(l) for l in open(cf, encoding="utf-8")]; wh = [json.loads(l) for l in open(wf, encoding="utf-8")] if os.path.exists(wf) else [dict(own=False, c=float("nan"), rel_xy=float("nan"), exp_xy=[0, 0], w_xy=[0, 0])] * len(ch)
        wh = wh[len(wh) - len(ch):]
        Q = {int(c["t"]): (c, w) for c, w in zip(ch, wh) if c["ep"] == int(t)}
        o.write("== %s ep %s success %s final site frame rel_cm %s\nstep(env n) | obj z cm above start | rel(tgt-obj) cm x y z | eef-obj xy cm | exec a xy z g | query: role own c rel_xy exp_xy w_xy | gripper width mm | contacts O || R\n" % (arm, t, bool(succ.any()), [round(100 * v) for v in (tgt[-1] - obj[-1])]))
        for n in range(0, len(obj), a.every):
            tq = n - 10
            q = Q.get(tq - tq % 5) if tq >= 0 else None
            qs = "" if q is None else "%s %d c%5.1f rxy%5.1f e(%+.2f %+.2f) w(%+.2f %+.2f)" % (q[0]["role"][:3], int(bool(q[1]["own"])), 100 * q[1]["c"], 100 * q[1]["rel_xy"], q[1]["exp_xy"][0], q[1]["exp_xy"][1], q[1]["w_xy"][0], q[1]["w_xy"][1])
            rel = 100 * (tgt[n] - obj[n])
            o.write("%3d | %5.1f | %+6.1f %+6.1f %+6.1f | %5.1f | a %+.2f %+.2f %+.2f %+.2f | %s | gw %.1f | %s\n" % (n, 100 * (obj[n, 2] - obj[0, 2]), rel[0], rel[1], rel[2], 100 * np.linalg.norm(eef[n, :2] - obj[n, :2]),
                    A[n, 0], A[n, 1], A[n, 2], A[n, 6], qs, 1000 * abs(grip[n, 0] - grip[n, 1]), cons[n][:150]))
    o.close()


if __name__ == "__main__":
    main()

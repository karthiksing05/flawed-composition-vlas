"""NPHJ E1 (prereg nphj): failure classes of the 7th-set Phase A arms with rva1_read.py's classifier unchanged; geometry results/flawed/rva1/geom_w7.json (nphj_geom.py).
  arm --arm pzJ|pzA24 --seed 43 --out <json>      per (arm, seed)  (copy of nph_e1.py arm)
  pool --arm pzJ --out <json>               sums the class counts of every results/flawed/nphj7/e1_<arm>_s*.json present; share = count / n_fail.   Outputs opened with "x"."""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import rva1_read as R                                                        # noqa: E402

BASE = os.path.join(R.OUT, "nphj7")


def arm_grid7(arm, seed):
    meta = json.load(open(os.path.join(R.OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["tid"]: t["language"] for t in meta["tasks"]}
    geom = json.load(open(os.path.join(R.OUT, "rva1", "geom_w7.json"), encoding="utf-8"))
    nm = "%s_s%d" % (arm, seed)
    d = os.path.join(BASE, "a_" + nm)
    chan = R.load_chan(d, nm)
    res = {}
    for c in R.NT:
        tid = R.NTID[c]
        eps = R.episodes(os.path.join(d, "%s_main_%d.json" % (nm, tid)), tid)
        res[c] = {t: R.classify(eps[t], chan.get((lang[tid], t), []), R.OBJ[c], geom[str(t)]) for t in range(60)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("arm", "pool"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trials = list(range(60))
    if a.cmd == "arm":
        grid = arm_grid7(a.arm, a.seed)
        res, _ = R.arm_summary(grid, trials)
        res.update(arm=a.arm, seed=a.seed, set="w7", B=R.B)
        res["episodes"] = {c: {str(t): grid[c][t] for t in trials} for c in R.NT}
    else:
        files = sorted(f for f in glob.glob(os.path.join(BASE, "e1_%s_s[0-9]*.json" % a.arm)))
        cnt, nfail, n, seeds = {}, 0, 0, []
        for f in files:
            r = json.load(open(f, encoding="utf-8"))
            seeds.append(r["seed"])
            n += r["n_episodes"]
            nfail += r["n_fail"]
            for k, v in r["counts"].items():
                cnt[k] = cnt.get(k, 0) + v
        res = dict(arm=a.arm, seeds=seeds, n_episodes=n, n_fail=nfail, counts=cnt, share={k: (cnt.get(k, 0) / nfail if nfail else None) for k in R.FAIL},
                   note="pooled over the seeds present (arm summary counts of rva1_read.arm_summary); share = class count / pooled failures")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nphj-e1] -> %s" % out, flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"})[:3000], flush=True)


if __name__ == "__main__":
    main()

"""NPH E1 (docs/notes/plan_necessity_power.md 3.3): the failure-class split of the 6th-set Phase A arms (pzH, pzA0, pzsL) with rva1_read.py's classifier unchanged; geometry results/flawed/rva1/geom_w6.json
(experiments/nph_geom.py).  Same outputs as rva1_read.py arm / decomp.   python experiments/nph_e1.py arm --arm pzH --out <json>   |   decomp --a pzA0 --b pzH --out <json>"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import rva1_read as R                                                        # noqa: E402

BASE = os.path.join(R.OUT, "nph6")
ARMDIR = {a: (os.path.join(BASE, "a_" + a), a, a) for a in ("pzH", "pzA0", "pzsL")}


def arm_grid6(arm):
    meta = json.load(open(os.path.join(R.OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["tid"]: t["language"] for t in meta["tasks"]}
    geom = json.load(open(os.path.join(R.OUT, "rva1", "geom_w6.json"), encoding="utf-8"))
    d, stem, pre = ARMDIR[arm]
    chan = R.load_chan(d, pre)
    res = {}
    for c in R.NT:
        tid = R.NTID[c]
        eps = R.episodes(os.path.join(d, "%s_main_%d.json" % (stem, tid)), tid)
        res[c] = {t: R.classify(eps[t], chan.get((lang[tid], t), []), R.OBJ[c], geom[str(t)]) for t in range(60)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("arm", "decomp"))
    ap.add_argument("--arm")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trials = list(range(60))
    if a.cmd == "arm":
        grid = arm_grid6(a.arm)
        res, _ = R.arm_summary(grid, trials)
        res.update(arm=a.arm, set="w6", B=R.B)
        res["episodes"] = {c: {str(t): grid[c][t] for t in trials} for c in R.NT}
    else:
        res = R.decomp(R.to_arrays(arm_grid6(a.a), trials), R.to_arrays(arm_grid6(a.b), trials), trials)
        res.update(a=a.a, b=a.b, set="w6", B=R.B)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nph-e1] -> %s" % out, flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"})[:3000], flush=True)


if __name__ == "__main__":
    main()

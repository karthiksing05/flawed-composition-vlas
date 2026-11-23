"""RVA1 Phase B split (docs/notes/plan_review_atlas.md 2.5; prereg rva1_pred.txt "Phase B (tracks)"): the failures of F6_b (rgc tid 8; bowl -> plate) classified from the per-step tracks
(<dir>/full_8_track.npz: pos [steps, 4, 3] of akita_black_bowl_1, cream_cheese_1, wine_bottle_1, plate_1; episodes in the order of the json).  Classes (failed episodes only):
  G never lifted (the bowl's z never exceeds its start by 2 cm)   L lost the target: final bowl-plate xy distance > 10 cm   M3-10   M<3 (inside LIBERO's 3 cm xy tolerance).
Limits (stated with the result): the F7 pool (two bowls; the second bowl and the stove site are not in the track) and K1t are not split here; only F6_b is.  CPU, numpy.  Separate file from
rva1_read.py (whose hash the rva1 row holds).
    python experiments/rva1_phaseb.py --set dev|w5 --arms pzsL,pzA0,raasl,A,A0 --out results/flawed/rva1/rva1_phaseB.json
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "flawed")
B = 2000
DEVDIR = {"raasl": os.path.join(R, "wasl")}


def wdir(arm, sset):
    if sset == "w5":
        return os.path.join(R, "rvw5", "b_" + arm)
    return DEVDIR.get(arm, os.path.join(R, "raaf", "dev_%s" % arm, "wa"))


def classify_arm(arm, sset, tid=8):
    base = os.path.join(wdir(arm, sset), "full_%d" % tid)
    eps = list(json.load(open(base + ".json", encoding="utf-8"))["tasks"].values())[0]["episodes"]
    z = np.load(base + "_track.npz")
    names = [str(x) for x in z["names"]]
    ib, ip = names.index("akita_black_bowl_1"), names.index("plate_1")
    out = {}
    for i, e in enumerate(eps):
        p = z["e%d_pos" % i]
        s = z["e%d_start" % i]
        ok = bool(e["success"])
        if ok:
            c, d = "S", None
        else:
            lifted = float(p[:, ib, 2].max() - s[ib, 2]) > 0.02
            d = float(100 * np.linalg.norm(p[-1, ib, :2] - p[-1, ip, :2]))
            c = "G" if not lifted else ("L" if d > 10 else ("M3-10" if d >= 3 else "M<3"))
        out[int(e["trial"])] = dict(cls=c, d=d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    res = {}
    for arm in a.arms.replace("+", ",").split(","):
        try:
            ep = classify_arm(arm, a.set)
        except FileNotFoundError as ex:
            res[arm] = "MISSING %s" % ex
            continue
        trials = sorted(ep)
        C = np.array([[1.0 if ep[t]["cls"] == c else 0.0 for t in trials] for c in ("S", "G", "L", "M3-10", "M<3")])
        rng = np.random.default_rng(0)
        shares = {}
        for k, c in enumerate(("G", "L", "M3-10", "M<3")):
            v = []
            for _ in range(B):
                s = rng.integers(0, len(trials), len(trials))
                nf = len(s) - C[0][s].sum()
                v.append(C[k + 1][s].sum() / nf if nf > 0 else np.nan)
            v = np.array(v)
            v = v[np.isfinite(v)]
            nf0 = len(trials) - C[0].sum()
            shares[c] = [float(C[k + 1].sum() / nf0) if nf0 else None, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res[arm] = dict(n=len(trials), success=float(C[0].mean()), n_fail=int(len(trials) - C[0].sum()),
                        counts={c: int(C[k].sum()) for k, c in enumerate(("S", "G", "L", "M3-10", "M<3"))}, share_of_failures=shares)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

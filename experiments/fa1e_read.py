"""FLAWED_COMPOSITION prototype A, stage 1e read (section 5.9; m3_prereg tag pa1e).  CPU.  Primary: the engine experiments/fa1x_read.py
(stage 1b's estimators, C1-C4 and verdict order) with A = Ae (seeds 0-2, results/flawed/a1e_roll), ablation Ae_nosize (seed 0), H = Hb
and R = Rb (stage 1b's runs), Ad (1d) and Ac (1c) beside it.
Descriptive, pre-registered as descriptive (never changes the verdict): the three-factor split s3 = h_o a_o q_t (fa1d_read.py) with
the ARRIVAL RADIUS of each (object, target) pair set to the relaxed test's horizontal radius, max(RAD[t], mean calibrated dxy_rest(o, t)
+ 0.03) (fa1_read.at; results/flawed/a1_calib.json), in place of 1d's fixed 0.03 m, which never fired at the cabinet and rack sites.
Computed for Ae, Ad (stage 1d's rollouts carry min_dxy_held) and pooled per arm over its seeds.
-> results/flawed/a1e_read.json (the engine's read plus "three_factor")

    python experiments/fa1e_read.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1_read as R1                                                        # noqa: E402
import fa1x_read as RX                                                       # noqa: E402


def radius(o, t, cal):
    return max(R1.RAD[t], float(np.mean(cal[f"{o}|{t}"]["dxy"])) + R1.DXY_PAD)


def three_factor(A, seeds=(0, 1, 2)):
    n2 = json.load(open(os.path.join(RX.OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(RX.OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    fam = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6}
    ks = n2["states"]["main"]
    H, AR, S = {}, {}, {}
    for c, t in fam.items():
        rad = radius(t["object"], t["target"], cal)
        h, ar, s = (np.zeros((len(seeds), len(ks))) for _ in range(3))
        for i, sd in enumerate(seeds):
            f = os.path.join(RX.OUT, RX.ROLLDIR[A], f"{A}_s{sd}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                j = ks.index(int(e["trial"]))
                h[i, j] = e["held_step"] is not None
                ar[i, j] = e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad
                s[i, j] = bool(e["success"])
        H[c], AR[c], S[c] = h, ar, s

    def est(kk):
        out = {}
        known = [c for c in fam if c[0] == "K"]
        for c in sorted(x for x in fam if x[0] == "N"):
            o, t = fam[c]["object"], fam[c]["target"]
            ko = [k for k in known if fam[k]["object"] == o]
            kt = [k for k in known if fam[k]["target"] == t]
            hs = np.concatenate([H[k][:, kk].ravel() for k in ko])
            ars = np.concatenate([AR[k][:, kk].ravel() for k in ko])
            arq = np.concatenate([AR[k][:, kk].ravel() for k in kt])
            sq = np.concatenate([S[k][:, kk].ravel() for k in kt])
            h_o, a_o = hs.mean(), (ars * hs).sum() / max(hs.sum(), 1)
            q_t = (sq * arq).sum() / max(arq.sum(), 1)
            hc, ac, sc = H[c][:, kk], AR[c][:, kk], S[c][:, kk]
            out[c] = dict(h_o=float(h_o), a_o=float(a_o), q_t=float(q_t), pred=float(h_o * a_o * q_t), obs=float(sc.mean()),
                          h_obs=float(hc.mean()), a_obs=float((ac * hc).sum() / max(hc.sum(), 1)),
                          q_obs=float((sc * ac).sum() / max(ac.sum(), 1)))
            out[c]["D3"] = out[c]["obs"] - out[c]["pred"]
        return out

    n = len(ks)
    point = est(np.arange(n))
    rng = np.random.default_rng(0)
    draws = [est(rng.choice(n, n)) for _ in range(2000)]
    for c in point:
        v = np.array([d[c]["D3"] for d in draws])
        point[c]["D3_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return point


if __name__ == "__main__":
    RX.ROLLDIR.update({"Ad": "a1d_roll", "Ad_tilt": "a1d_roll", "Ad_fing": "a1d_roll", "Ae": "a1e_roll", "Ae_nosize": "a1e_roll"})
    res = RX.run(RX.Read("Ae", ("Ae_nosize",), [("Ad", (0, 1, 2)), ("Ac", (0, 1, 2))]), "a1e_read.json")
    res["three_factor"] = {A: three_factor(A) for A in ("Ae", "Ad")}
    json.dump(res, open(os.path.join(RX.OUT, "a1e_read.json"), "w", encoding="utf-8"), indent=1)
    for A, tf in res["three_factor"].items():
        for c, d in tf.items():
            print("3F", A, c, json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in d.items()}))

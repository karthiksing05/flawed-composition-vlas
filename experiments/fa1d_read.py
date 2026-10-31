"""FLAWED_COMPOSITION prototype A, stage 1d read (section 5.8; m3_prereg tag pa1d).  CPU.  Primary: the engine experiments/fa1x_read.py
(stage 1b's estimators, C1-C4 and verdict order) with A = Ad (seeds 0-2, results/flawed/a1d_roll), ablations Ad_tilt, Ad_fing (seed 0),
H = Hb and R = Rb (stage 1b's runs), Ac (1c) and Ab (1b) beside it.
Descriptive, pre-registered as descriptive (never changes the verdict): the three-factor split of Proposition 3 (iii) for Ad, per new
pick-and-place pairing (o, t), from the known pick-and-place tasks:
  h_o  P(a hold is reached | o)                          (known tasks with object o)
  a_o  P(arrived | hold, o): the held object came within 0.03 m (horizontal) of the target while held   (known tasks with o)
  q_t  P(success | arrived, t)                           (known tasks with target t)
  predicted s3 = h_o * a_o * q_t; D3 = observed s - s3, 95% CI over 2,000 start-state draws (seed 0); and per pairing the observed
  a and q next to the predicted ones.
-> results/flawed/a1d_read.json (the engine's read plus "three_factor")

    python experiments/fa1d_read.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1x_read as RX                                                       # noqa: E402

ARRIVE = 0.03


def three_factor(A="Ad", seeds=(0, 1, 2)):
    n2 = json.load(open(os.path.join(RX.OUT, "n2_tasks.json"), encoding="utf-8"))
    fam = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6}
    ks = n2["states"]["main"]
    H, AR, S = {}, {}, {}
    for c, t in fam.items():
        h, ar, s = (np.zeros((len(seeds), len(ks))) for _ in range(3))
        for i, sd in enumerate(seeds):
            f = os.path.join(RX.OUT, RX.ROLLDIR[A], f"{A}_s{sd}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                j = ks.index(int(e["trial"]))
                h[i, j] = e["held_step"] is not None
                ar[i, j] = e["min_dxy_held"] is not None and e["min_dxy_held"] <= ARRIVE
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
            h_o = hs.mean()
            a_o = (ars * hs).sum() / max(hs.sum(), 1)
            arq = np.concatenate([AR[k][:, kk].ravel() for k in kt])
            sq = np.concatenate([S[k][:, kk].ravel() for k in kt])
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
    RX.ROLLDIR.update({"Ad": "a1d_roll", "Ad_tilt": "a1d_roll", "Ad_fing": "a1d_roll"})
    res = RX.run(RX.Read("Ad", ("Ad_tilt", "Ad_fing"), [("Ac", (0, 1, 2)), ("Ab", (0, 1, 2))]), "a1d_read.json")
    tf = three_factor()
    res["three_factor"] = tf
    json.dump(res, open(os.path.join(RX.OUT, "a1d_read.json"), "w", encoding="utf-8"), indent=1)
    for c, d in tf.items():
        print("3F", c, json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in d.items()}))

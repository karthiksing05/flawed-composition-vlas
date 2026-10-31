"""FLAWED_COMPOSITION prototype A, stage 1g read (section 5.12; m3_prereg tag pa1g).  CPU.
All arms trained on the demos plus the generated varied-layout carries of the known pairings (tag g, fa1g_train.py): A = Ag (the
factored arm, seeds 0-2), H = Hg and R = Rg (the Hb / Rb architectures on the same data, seeds 0-2); rollouts results/flawed/a1g_roll.
Read once per N2 layout (familiar: even tids; new: odd tids), each with the engine experiments/fa1x_read.py (C1-C4, stage 1b's
estimators) and 3F (the three-factor law, fa1e_read's estimator with the layout's own cells; every N1-N6 CI of D3 contains 0).
  per layout: floor s_H(known) < 0.5 -> "uninformative"; C1-C4 and 3F -> "compositional; three-factor law holds";
              C1-C4 only -> "compositional by C1-C4; three-factor law fails"; else "not compositional (fails: ...)"
  verdict: "<familiar verdict> | <new-layout verdict>"; COMPOSITIONAL WITH THE GUARANTEE = both layouts "compositional; three-factor
           law holds".
Stage 1b-1f arms are shown beside the familiar read (Ae_nosize from a1f_roll).
-> results/flawed/a1g_read.json

    python experiments/fa1g_read.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1_read as R1                                                        # noqa: E402
import fa1x_read as RX                                                       # noqa: E402
from fa1f_read import verdict                                                # noqa: E402

A, H, R = "Ag", "Hg", "Rg"


def radius(o, t, cal):
    return max(R1.RAD[t], float(np.mean(cal[f"{o}|{t}"]["dxy"])) + R1.DXY_PAD)


def three_factor(m, layout, seeds=(0, 1, 2)):
    """fa1e_read.three_factor on one layout's cells."""
    n2 = json.load(open(os.path.join(RX.OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(RX.OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    fam = {t["compound"]: t for t in n2["tasks"] if t["layout"] == layout and int(t["compound"][1:]) <= 6}
    ks = n2["states"]["main"]
    HH, AR, S = {}, {}, {}
    for c, t in fam.items():
        rad = radius(t["object"], t["target"], cal)
        h, ar, s = (np.zeros((len(seeds), len(ks))) for _ in range(3))
        for i, sd in enumerate(seeds):
            f = os.path.join(RX.OUT, RX.ROLLDIR[m], f"{m}_s{sd}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                j = ks.index(int(e["trial"]))
                h[i, j] = e["held_step"] is not None
                ar[i, j] = e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad
                s[i, j] = bool(e["success"])
        HH[c], AR[c], S[c] = h, ar, s

    def est(kk):
        out = {}
        known = [c for c in fam if c[0] == "K"]
        for c in sorted(x for x in fam if x[0] == "N"):
            o, t = fam[c]["object"], fam[c]["target"]
            ko = [k for k in known if fam[k]["object"] == o]
            kt = [k for k in known if fam[k]["target"] == t]
            hs = np.concatenate([HH[k][:, kk].ravel() for k in ko])
            ars = np.concatenate([AR[k][:, kk].ravel() for k in ko])
            arq = np.concatenate([AR[k][:, kk].ravel() for k in kt])
            sq = np.concatenate([S[k][:, kk].ravel() for k in kt])
            h_o, a_o = hs.mean(), (ars * hs).sum() / max(hs.sum(), 1)
            q_t = (sq * arq).sum() / max(arq.sum(), 1)
            hc, ac, sc = HH[c][:, kk], AR[c][:, kk], S[c][:, kk]
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
    RX.ROLLDIR.update({A: "a1g_roll", H: "a1g_roll", R: "a1g_roll", "Ae_nosize": "a1f_roll", "Ad": "a1d_roll"})
    out = {}
    for lay in ("familiar", "new"):
        refs = [("Hb", (0, 1, 2)), ("Rb", (0, 1, 2)), ("Ae_nosize", (0, 1, 2)), ("Ad", (0, 1, 2))] if lay == "familiar" else []
        res = RX.run(RX.Read(A, (), refs, H=H, R=R, layout=lay), f"a1g_read_{lay}.json")
        res["three_factor"] = {m: three_factor(m, lay) for m in (A,)}
        res["verdict"] = verdict(res["verdict"], res["three_factor"][A])
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"])
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["guarantee"] = all(x == "compositional; three-factor law holds" for x in (v["familiar"], v["new"]))
    final = dict(A=A, H=H, R=R, verdict=v, layouts=out)
    json.dump(final, open(os.path.join(RX.OUT, "a1g_read.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(verdict=v), indent=1))
    for lay in out:
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))

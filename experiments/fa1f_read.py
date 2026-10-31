"""FLAWED_COMPOSITION prototype A, stage 1f read (section 5.11; m3_prereg tag pa1f).  CPU.
Primary, two parts, one verdict:
  (a) the engine experiments/fa1x_read.py (stage 1b's estimators, C1-C4) with A = Ae_nosize, seeds 0-2 retrained under tag f,
      rollouts in results/flawed/a1f_roll; H = Hb, R = Rb (stage 1b's runs); Ae and Ad (1e, 1d) beside it.
  (b) 3F, the three-factor law s(o, t) = h_o a_o q_t (fa1e_read.three_factor: corrected arrival radius, seeds pooled, the 20 states
      resampled, 2,000 draws, seed 0): 3F holds iff the 95% CI of D3 = obs - h_o a_o q_t contains 0 on every one of N1-N6
      (N6 is kept in; section 5.11 gives the reason).
  verdict: floor -> "uninformative"; C1-C4 and 3F -> "compositional; three-factor law holds";
           C1-C4 but not 3F -> "compositional by C1-C4; three-factor law fails";
           else "not compositional (fails: <C1, C2, C3, C4, 3F in that order, those failing>)"
-> results/flawed/a1f_read.json

    python experiments/fa1f_read.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1x_read as RX                                                       # noqa: E402
from fa1e_read import three_factor                                           # noqa: E402

A = "Ae_nosize"


def verdict(v, tf):
    v = dict(v)
    v["3F"] = all(d["D3_ci"][0] <= 0 <= d["D3_ci"][1] for d in tf.values())
    v["3F_pairings_in_ci"] = sorted(c for c, d in tf.items() if d["D3_ci"][0] <= 0 <= d["D3_ci"][1])
    v["engine_verdict"] = v.pop("verdict")
    order = ("C1", "C2", "C3", "C4", "3F")
    if v["floor"]:
        v["verdict"] = "uninformative"
    elif all(v[k] for k in order):
        v["verdict"] = "compositional; three-factor law holds"
    elif all(v[k] for k in order[:4]):
        v["verdict"] = "compositional by C1-C4; three-factor law fails"
    else:
        v["verdict"] = "not compositional (fails: " + ", ".join(k for k in order if not v[k]) + ")"
    return v


if __name__ == "__main__":
    RX.ROLLDIR.update({"Ad": "a1d_roll", "Ae": "a1e_roll", A: "a1f_roll"})
    res = RX.run(RX.Read(A, (), [("Ae", (0, 1, 2)), ("Ad", (0, 1, 2))]), "a1f_read.json")
    res["three_factor"] = {m: three_factor(m) for m in (A, "Ae")}
    res["verdict"] = verdict(res["verdict"], res["three_factor"][A])
    json.dump(res, open(os.path.join(RX.OUT, "a1f_read.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(verdict=res["verdict"]), indent=1))
    for m, tf in res["three_factor"].items():
        for c, d in tf.items():
            print("3F", m, c, json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in d.items()}))

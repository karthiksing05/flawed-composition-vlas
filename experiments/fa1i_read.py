"""FLAWED_COMPOSITION prototype A, stage 1i read (section 5.16; m3_prereg tag pa1i).  CPU.
A = Ai (Ah_rot + the target's identity one-hot and size in the target term; seeds 0-2, tag i, demos + 1g's generated carries);
rollouts results/flawed/a1i_roll.  C4 against Hg and Rg (1g); Ah_rot (1h) and Ag (1g) beside.
Criteria, per layout (familiar: even tids, new: odd tids), with BOTH user-approved changes:
  floor  the fixed reference REF = 0.672 (stage 1b's demo-only Hb, results/flawed/a1b_read.json; hash checked in this script) < 0.5
  C1     familiar: s_A(known) - REF >= -0.10 (point)
         new layout: LAYOUT INVARIANCE  L = s_A(known, new) / s_A(known, familiar) >= 0.8 (point) and CI lower bound >= 0.6; the CI
         is a 2,000-draw percentile bootstrap (seed 0) resampling the 6 known compounds and the 20 states JOINTLY across layouts
  C2, C3, C4 (engine, 5.6), 3F (5.11)
  per layout: the 5.11 mapping; verdict = "familiar: <v> | new layout: <v>"; guarantee iff both "compositional; three-factor law holds"
-> results/flawed/a1i_read.json (+ a1i_read_familiar.json, a1i_read_new.json)

    python experiments/fa1i_read.py
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import three_factor                                           # noqa: E402

A, H, R = "Ai", "Hg", "Rg"
REF_FILE, REF_SHA = "a1b_read.json", "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760"


def layout_invariance():
    """L = s_A(known, new) / s_A(known, familiar), seeds pooled; joint bootstrap over known compounds and states."""
    S = {}
    for lay in ("familiar", "new"):
        rd = RX.Read(A, (), [], H=H, R=R, layout=lay)
        rd.all_runs = [(A, s) for s in RX.SEEDS]
        comps, ks, info, X, *_ = rd.load()
        kn = [comps.index(c) for c in RX.PRIMARY if c[0] == "K"]
        S[lay] = np.stack([X[(A, s)]["S"][kn] for s in RX.SEEDS])              # [seeds, 6, 20]
    n_c, n_k = S["familiar"].shape[1], S["familiar"].shape[2]
    est = float(S["new"].mean() / S["familiar"].mean())
    rng = np.random.default_rng(RX.SEED)
    draws = []
    for _ in range(RX.B):
        a, kk = rng.choice(n_c, n_c), rng.choice(n_k, n_k)
        f, nw = S["familiar"][:, a][:, :, kk].mean(), S["new"][:, a][:, :, kk].mean()
        if f > 0:
            draws.append(nw / f)
    return dict(est=est, ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def verdict(v, lay, s_known, tf, ref, L):
    v = dict(v)
    v["engine_verdict"] = v.pop("verdict")
    v["engine_floor_Hg"], v["engine_C1_vs_Hg"] = v.pop("floor"), v.pop("C1")
    v["floor"] = ref < 0.5
    if lay == "familiar":
        v["C1_value"] = s_known - ref
        v["C1"] = v["C1_value"] >= -0.10
    else:
        v["C1_value"], v["C1_ci"] = L["est"], L["ci"]
        v["C1"] = L["est"] >= 0.8 and L["ci"][0] >= 0.6
    v["3F"] = all(d["D3_ci"][0] <= 0 <= d["D3_ci"][1] for d in tf.values())
    v["3F_pairings_in_ci"] = sorted(c for c, d in tf.items() if d["D3_ci"][0] <= 0 <= d["D3_ci"][1])
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
    p = os.path.join(RX.OUT, REF_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == REF_SHA, "the 1b reference file changed"
    ref = float(json.load(open(p, encoding="utf-8"))["cells"]["Hb"]["s_known"]["est"])
    RX.ROLLDIR.update({A: "a1i_roll", H: "a1g_roll", R: "a1g_roll", "Ag": "a1g_roll", "Ah_rot": "a1h_roll"})
    L = layout_invariance()
    out = {}
    for lay in ("familiar", "new"):
        res = RX.run(RX.Read(A, (), [("Ah_rot", (0, 1, 2)), ("Ag", (0, 1, 2))], H=H, R=R, layout=lay), f"a1i_read_{lay}.json")
        res["three_factor"] = {m: three_factor(m, lay) for m in (A, "Ah_rot")}
        res["verdict"] = verdict(res["verdict"], lay, res["cells"][A]["s_known"]["est"], res["three_factor"][A], ref, L)
        res["K5_K6"] = {c: {f"{m}_s{s}": res["per_compound"][c][f"{m}_s{s}"]["s"] for m in (A, "Ah_rot", "Ag") for s in (0, 1, 2)}
                        for c in ("K5", "K6")}
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"], reference=ref, reference_file=REF_FILE,
             reference_sha256=REF_SHA, layout_invariance=L)
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["guarantee"] = all(x == "compositional; three-factor law holds" for x in (v["familiar"], v["new"]))
    json.dump(dict(A=A, H=H, R=R, verdict=v, layouts=out), open(os.path.join(RX.OUT, "a1i_read.json"), "w", encoding="utf-8"), indent=1)
    print("VERDICT_FINAL", json.dumps(v))
    for lay in out:
        print("PERLAYOUT", lay, json.dumps({k: x for k, x in out[lay]["verdict"].items()}))
        print("K5K6", lay, json.dumps(out[lay]["K5_K6"]))
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))

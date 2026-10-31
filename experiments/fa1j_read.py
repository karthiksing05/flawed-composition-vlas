"""FLAWED_COMPOSITION prototype A, stage 1j read (section 5.17; m3_prereg tag pa1j): the CONFIRMATORY read of Ai, unchanged, on three
FRESH seeds (3, 4, 5; tag j; rollouts named Aj in results/flawed/a1j_roll).  CPU.
Binding criteria (user-approved, 5.16): per layout
  familiar  C1  s_A(known) - 0.672 >= -0.10 (point; 0.672 = stage 1b's demo-only Hb, a1b_read.json, hash checked)
  new       C1  layout invariance L = s_A(known, new) / s_A(known, familiar) >= 0.8 (point) and CI lower bound >= 0.6 (joint
                bootstrap over the 6 known compounds and 20 states, 2,000 draws, seed 0)
  both      C2  (engine: ratio >= 0.8 point, CI lower >= 0.6), 3F (every N1-N6 CI of D3 = obs - h_o a_o q_t contains 0; N6 in),
            C4  (vs Hg and Rg, 1g's runs: both differences > 0 with CI excluding 0)
  C3 (two-factor) is DESCRIPTIVE only.
  per layout: floor (0.672 < 0.5) -> "uninformative"; all four -> "compositional"; else "not compositional (fails: <C1, C2, 3F, C4
  in that order, those failing>)"
  verdict = "familiar: <v> | new layout: <v>"; CONFIRMED iff both layouts read "compositional".
Ai (1i, seeds 0-2) shown beside.
-> results/flawed/a1j_read.json (+ a1j_read_familiar.json, a1j_read_new.json)

    python experiments/fa1j_read.py
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import three_factor                                           # noqa: E402

A, H, R, SEEDS_A = "Aj", "Hg", "Rg", (3, 4, 5)
REF_FILE, REF_SHA = "a1b_read.json", "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760"


def layout_invariance():
    S = {}
    for lay in ("familiar", "new"):
        rd = RX.Read(A, (), [], H=H, R=R, layout=lay, a_seeds=SEEDS_A)
        rd.all_runs = [(A, s) for s in SEEDS_A]
        comps, ks, info, X, *_ = rd.load()
        kn = [comps.index(c) for c in RX.PRIMARY if c[0] == "K"]
        S[lay] = np.stack([X[(A, s)]["S"][kn] for s in SEEDS_A])
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
    v["C3_descriptive"] = v.pop("C3")
    v["floor"] = ref < 0.5
    if lay == "familiar":
        v["C1_value"] = s_known - ref
        v["C1"] = v["C1_value"] >= -0.10
    else:
        v["C1_value"], v["C1_ci"] = L["est"], L["ci"]
        v["C1"] = L["est"] >= 0.8 and L["ci"][0] >= 0.6
    v["3F"] = all(d["D3_ci"][0] <= 0 <= d["D3_ci"][1] for d in tf.values())
    v["3F_pairings_in_ci"] = sorted(c for c, d in tf.items() if d["D3_ci"][0] <= 0 <= d["D3_ci"][1])
    order = ("C1", "C2", "3F", "C4")
    if v["floor"]:
        v["verdict"] = "uninformative"
    elif all(v[k] for k in order):
        v["verdict"] = "compositional"
    else:
        v["verdict"] = "not compositional (fails: " + ", ".join(k for k in order if not v[k]) + ")"
    return v


if __name__ == "__main__":
    p = os.path.join(RX.OUT, REF_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == REF_SHA, "the 1b reference file changed"
    ref = float(json.load(open(p, encoding="utf-8"))["cells"]["Hb"]["s_known"]["est"])
    RX.ROLLDIR.update({A: "a1j_roll", H: "a1g_roll", R: "a1g_roll", "Ai": "a1i_roll"})
    L = layout_invariance()
    out = {}
    for lay in ("familiar", "new"):
        res = RX.run(RX.Read(A, (), [("Ai", (0, 1, 2))], H=H, R=R, layout=lay, a_seeds=SEEDS_A), f"a1j_read_{lay}.json")
        res["three_factor"] = {A: three_factor(A, lay, seeds=SEEDS_A), "Ai": three_factor("Ai", lay)}
        res["verdict"] = verdict(res["verdict"], lay, res["cells"][A]["s_known"]["est"], res["three_factor"][A], ref, L)
        res["K5_K6"] = {c: {f"{m}_s{s}": res["per_compound"][c][f"{m}_s{s}"]["s"] for m, ss in ((A, SEEDS_A), ("Ai", (0, 1, 2)))
                            for s in ss} for c in ("K5", "K6")}
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"], reference=ref, reference_file=REF_FILE,
             reference_sha256=REF_SHA, layout_invariance=L)
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["confirmed"] = v["familiar"] == "compositional" and v["new"] == "compositional"
    json.dump(dict(A=A, arm="Ai", seeds=SEEDS_A, H=H, R=R, verdict=v, layouts=out),
              open(os.path.join(RX.OUT, "a1j_read.json"), "w", encoding="utf-8"), indent=1)
    print("VERDICT_FINAL", json.dumps(v))
    for lay in out:
        print("PERLAYOUT", lay, json.dumps(out[lay]["verdict"]))
        print("K5K6", lay, json.dumps(out[lay]["K5_K6"]))
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))

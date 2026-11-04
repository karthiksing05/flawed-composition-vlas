"""Layer-3 ladder read (docs/LADDER_RESULTS.md; PLAN_PARTS 5 and 10.10): a rung against L0 = P_min on the composition measures,
with paired state-bootstrap CIs of the rung-minus-L0 difference.  CPU.  The estimators are pp_l2r32_read's (3F, OM, S3), recomputed
here on shared draws so that each difference is paired; their point values are asserted equal to pp_l2r32_read's.

Measures, per layout (seeds 10-12, dev states 4-23, seeds pooled):
  S1        rung: every S1 file holds (pp_l2r32_read.s1: both swap tests and every target-into-grasp component exactly 0)
  OM        object-matched ratio (pp_l2r32_read.s2)
  3F|Nx     D3 = obs - pred of the three-factor law, per new pairing N1-N6 (pp_l2r32_read.three_factor)
  S3|cond   the mean over k in {1, 5, 10} of the transfer ratio s_unseen / s_train (pp_l2r32_read.s3_one), per held-out atom
Delta = rung - L0; CI: 2,000 joint draws of the 20 states (rng seed 0), the same draw for both arms, 2.5/97.5 percentiles.
COUNTED (familiar layout only): S1, OM, 3F|Nx and S3|cond for which L0 passes its own Layer-2 criterion (OM: r >= 0.8, LB >= 0.6;
3F: D3 CI contains 0; S3: all of k = 1, 5, 10 ratio >= 0.8, LB >= 0.6).
A counted measure BREAKS iff |Delta| > 0.10 and its CI excludes 0.  The rung HOLDS iff S1 holds and no counted measure breaks.
New layout: the same measures and Deltas, reported with no verdict.

  python experiments/pp_ladder_read.py --rung L1 --s1 <f1+f2+...> --s2 <dir>:<name> --s3 <dir>:<prefix>
-> results/flawed/pp_ladder_<rung>_read.json (must not exist)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_l2_read as L                                                       # noqa: E402
import pp_l2r32_read as C                                                    # noqa: E402
import pp_round_read as R                                                    # noqa: E402

OUT = C.OUT
SEEDS = (10, 11, 12)
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]
KNOWN = ["K1", "K2", "K3", "K4", "K5", "K6"]
MATCH = {"N1": ["K6"], "N2": ["K6"], "N3": ["K6"], "N4": ["K4", "K5"], "N5": ["K4", "K5"], "N6": ["K4", "K5"]}
L0 = dict(s1="pp_s1_mix_full_naf_m0.json+pp_s1_mix_full_naf_m1.json+pp_s1_mix_full_naf_m2.json", s2="pp_cons_roll:P2naf",
          s3="pp_cons_roll:Pnaf")
TOL, NB = 0.10, 2000


def cfg(s1, s2, s3):
    c = R.cfg_of(argparse.Namespace(phase="main", s1=s1.split("+")[0], s2=s2, s3=s3), SEEDS)
    c.s1_all = s1.split("+")
    return c


def arrays(c, lay):
    """-> {compound: (held, arrived, success) [seeds, states]} for K1-K6 and N1-N6 (no missing episode allowed)."""
    n2 = L.n2()
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    out = {}
    for comp in KNOWN + NEW:
        t = next(x for x in n2["tasks"] if x["compound"] == comp and x["layout"] == lay)
        rad = C.radius(t["object"], t["target"], cal)
        h, ar, s = (np.full((len(SEEDS), len(c.states)), np.nan) for _ in range(3))
        for i, sd in enumerate(SEEDS):
            f = os.path.join(OUT, c.s2[0], f"{c.s2[1]}_s{sd}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                j = c.states.index(int(e["trial"]))
                h[i, j] = e["held_step"] is not None
                ar[i, j] = e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad
                s[i, j] = bool(e["success"])
        assert not np.isnan(s).any(), ("missing episodes", c.s2, comp, lay)
        out[comp] = (h, ar, s)
    return out


def d3(A, kk):
    """pp_l2r32_read.three_factor's est on the state draw kk -> {Nx: D3}."""
    n2 = L.n2()
    fam = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6}
    out = {}
    for comp in NEW:
        o, t = fam[comp]["object"], fam[comp]["target"]
        ko = [k for k in KNOWN if fam[k]["object"] == o]
        kt = [k for k in KNOWN if fam[k]["target"] == t]
        hs = np.concatenate([A[k][0][:, kk].ravel() for k in ko])
        ars = np.concatenate([A[k][1][:, kk].ravel() for k in ko])
        arq = np.concatenate([A[k][1][:, kk].ravel() for k in kt])
        sq = np.concatenate([A[k][2][:, kk].ravel() for k in kt])
        h_o, a_o = hs.mean(), (ars * hs).sum() / max(hs.sum(), 1)
        q_t = (sq * arq).sum() / max(arq.sum(), 1)
        out[comp] = float(A[comp][2][:, kk].mean() - h_o * a_o * q_t)
    return out


def om(A, kk):
    num = np.mean([np.nanmean(A[c][2][:, kk]) for c in NEW])
    den = np.mean([np.mean([np.nanmean(A[k][2][:, kk]) for k in MATCH[c]]) for c in NEW])
    return float(num / den) if den > 0 else np.nan


def s3arr(c, lay):
    out = {}
    for cond, CC in L.COND.items():
        out[cond] = [C.stack(c, c.s3dir, c.pname(cond, k), (CC["train"],) + CC["unseen"], lay) for k in (1, 5, 10)]
        assert not any(np.isnan(S).any() for S in out[cond]), ("missing S3 episodes", c.s3dir, cond, lay)
    return out


def s3m(S3, cond, kk):
    r = []
    for S in S3[cond]:
        st = np.nanmean(S[:, 0][:, kk])
        r.append(np.nanmean(S[:, 1:][:, :, kk]) / st if st > 0 else np.nan)
    return float(np.mean(r))


def measures(A, S3, kk):
    m = {"OM": om(A, kk)}
    m.update({f"3F|{p}": v for p, v in d3(A, kk).items()})
    m.update({f"S3|{cond}": s3m(S3, cond, kk) for cond in L.COND})
    return m


def s1_all(c):
    res = {}
    for f in c.s1_all:
        c.s1 = f
        res[f] = bool(C.s1(c)["holds"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--parse-locality", default="")                         # LADDER_RESULTS 5.5 (L4): pp_L4_smoke roles' file
    a = ap.parse_args()
    fn = os.path.join(OUT, f"pp_ladder_{a.rung}_read.json")
    assert not os.path.exists(fn), f"{fn} exists"
    ref, cand = cfg(L0["s1"], L0["s2"], L0["s3"]), cfg(a.s1, a.s2, a.s3)
    s1r, s1c = s1_all(ref), s1_all(cand)
    S2r, S2c = C.s2(ref), C.s2(cand)                                         # pp_l2r32_read's own estimates and criteria
    S3r, _ = C.s3s4(ref)
    S3c, _ = C.s3s4(cand)
    out = dict(rung=a.rung, inputs=dict(L0=L0, rung=dict(s1=a.s1, s2=a.s2, s3=a.s3)), S1=dict(L0=s1r, rung=s1c, holds=bool(all(s1c.values()))),
               layouts={}, rule=f"counted familiar measure breaks iff |delta| > {TOL} and the paired state-bootstrap CI excludes 0; "
                                 "the rung holds iff S1 holds and no counted measure breaks; new layout reported only")
    if a.parse_locality:                                                     # L4: S1 holds iff the swaps and parse locality both hold
        pl = json.load(open(os.path.join(OUT, a.parse_locality), encoding="utf-8"))["parse_locality"]
        out["S1"]["parse_locality"] = pl
        out["S1"]["holds"] = bool(out["S1"]["holds"] and pl["holds"])
    n = len(ref.states)
    rng = np.random.default_rng(0)
    draws = [rng.choice(n, n) for _ in range(NB)]
    for lay in ("familiar", "new"):
        Ar, Ac = arrays(ref, lay), arrays(cand, lay)
        Tr, Tc = s3arr(ref, lay), s3arr(cand, lay)
        pr, pc = measures(Ar, Tr, np.arange(n)), measures(Ac, Tc, np.arange(n))
        for S2, p in ((S2r, pr), (S2c, pc)):                                 # the paired estimators reproduce pp_l2r32_read's
            assert np.isclose(p["OM"], S2[lay]["object_matched"][0]), ("OM", lay)
            for x in NEW:
                assert np.isclose(p[f"3F|{x}"], S2[lay]["three_factor"][x]["D3"]), ("3F", x, lay)
        dd = {k: [] for k in pr}
        for kk in draws:
            mr, mc = measures(Ar, Tr, kk), measures(Ac, Tc, kk)
            for k in dd:
                dd[k].append(mc[k] - mr[k])
        L0pass = {"OM": bool(S2r[lay]["object_matched_holds"])}
        L0pass.update({f"3F|{x}": bool(S2r[lay]["three_factor"][x]["D3_ci"][0] <= 0.0 <= S2r[lay]["three_factor"][x]["D3_ci"][1]) for x in NEW})
        L0pass.update({f"S3|{cond}": bool(all(S3r["transfer_criterion"][f"{cond}|{lay}|k{k}"] for k in (1, 5, 10))) for cond in L.COND})
        rows = {}
        for k in pr:
            v = np.array(dd[k], float)
            v = v[np.isfinite(v)]
            ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]
            delta = pc[k] - pr[k]
            brk = bool(np.isfinite(delta) and abs(delta) > TOL and ci[0] is not None and (ci[0] > 0 or ci[1] < 0))
            rows[k] = dict(L0=pr[k], rung=pc[k], delta=float(delta), delta_ci=ci, n_draws=int(len(v)), L0_passes=L0pass[k],
                           counted=bool(lay == "familiar" and L0pass[k]), breaks=brk if lay == "familiar" else None)
        out["layouts"][lay] = dict(measures=rows, competence=dict(
            L0=dict(s_known=S2r[lay]["s_known"], s_new=S2r[lay]["s_new"], per_compound=S2r[lay]["per_compound"]),
            rung=dict(s_known=S2c[lay]["s_known"], s_new=S2c[lay]["s_new"], per_compound=S2c[lay]["per_compound"])),
            rung_own_criteria=dict(OM=S2c[lay]["object_matched"] + [S2c[lay]["object_matched_holds"]],
                                   three_factor={x: [S2c[lay]["three_factor"][x]["D3"]] + S2c[lay]["three_factor"][x]["D3_ci"] for x in NEW},
                                   S3={kk: v for kk, v in S3c["transfer_criterion"].items() if f"|{lay}|" in kk}))
    fam = out["layouts"]["familiar"]["measures"]
    broken = [k for k, v in fam.items() if v["counted"] and v["breaks"]]
    out["counted"] = ["S1"] + [k for k, v in fam.items() if v["counted"]]
    out["broken"] = (["S1"] if not out["S1"]["holds"] else []) + broken
    out["verdict"] = "HOLDS" if not out["broken"] else "BREAKS"
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    r = lambda x: None if x is None else round(float(x), 3)                  # noqa: E731
    print("LADDER", a.rung, "VERDICT", out["verdict"], "broken", out["broken"], "S1", out["S1"], flush=True)
    for lay, d in out["layouts"].items():
        for k, v in d["measures"].items():
            print(lay, k, "L0", r(v["L0"]), "rung", r(v["rung"]), "delta", r(v["delta"]), [r(x) for x in v["delta_ci"]], "counted", v["counted"],
                  "breaks", v["breaks"], flush=True)
        print(lay, "competence", json.dumps(d["competence"]), flush=True)
        print(lay, "rung own criteria", json.dumps(d["rung_own_criteria"]), flush=True)


if __name__ == "__main__":
    main()

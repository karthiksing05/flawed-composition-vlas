"""PLAN_PARTS improvement rounds (docs/PARTS_RESULTS.md sections 12 onward): the Layer-2 dev read of a candidate architecture with
pp_l2's estimators and frozen mapping (pp_l2r32_read.py's functions, unchanged), plus the round's pre-registered weakness metric
against a reference arm.  CPU.

  --tag      round tag (output results/flawed/pp_<tag>_read.json)
  --seeds    training seeds of the candidate (and of the reference unless --ref-seeds)
  --s1       S1 file (pp_s1_mix.py output) of the candidate
  --s2       <dir>:<name> of the candidate's S2 rollouts       --ref-s2  <dir>:<name> of the reference's S2 rollouts
  --s3       <dir>:<name prefix>; S3 names <prefix>_<cond>_k<k>
  --metric   n4_fam   N4's S2 familiar success, candidate - reference (state bootstrap, seeds pooled; improves iff > 0.10, LB > 0)
             s_known_new   pooled known-pairing success in the new layout, candidate - reference (same rule)
  --phase    main (dev states 4-23) | conf

    python experiments/pp_round_read.py --tag r2 --seeds 10+11+12 --s1 pp_s1_mix_full_sz.json --s2 pp_r2_roll:P2sz \
        --ref-s2 pp_r2_roll:P2r32 --s3 pp_r2_roll:Psz --metric n4_fam
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

OUT = C.OUT


def cfg_of(a, seeds):
    c = C.Cfg(a.phase == "conf")
    c.seeds = seeds
    c.s1 = a.s1
    d, n = a.s2.split(":")
    c.s2 = (d, n)
    d3, p3 = a.s3.split(":")
    c.s3dir, c.olddir = d3, "pp_none"
    c.pname = lambda cond, k: f"{p3}_{cond}_k{k}"                             # noqa: E731
    return c


def held_arrived(cfg, comps, lay):
    """-> (held, arrived) arrays [comps x seeds, states] from a cfg's S2 rollouts (fa1g_read's arrival: held within the pair's
    calibrated radius)."""
    n2 = L.n2()
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    H, R = [], []
    for c in comps:
        t = next(x for x in n2["tasks"] if x["compound"] == c and x["layout"] == lay)
        rad = C.radius(t["object"], t["target"], cal)
        for sd in cfg.seeds:
            h, r = np.full(len(cfg.states), np.nan), np.full(len(cfg.states), np.nan)
            f = os.path.join(OUT, cfg.s2[0], f"{cfg.s2[1]}_s{sd}_{cfg.phase}_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                j = cfg.states.index(int(e["trial"]))
                h[j] = e["held_step"] is not None
                r[j] = h[j] and e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad
            H.append(h)
            R.append(r)
    return np.array(H), np.array(R)


def metric(a, cand, ref):
    known = ["K1", "K2", "K3", "K4", "K5", "K6"]
    if a.metric == "cab_arr_fam":                                            # PARTS_RESULTS 30: cabinet arrival given hold
        cab = ["K3", "K4", "N3"]
        Ha, Ra = held_arrived(cand, cab, "familiar")
        Hb, Rb = held_arrived(ref, cab, "familiar")
        rate = lambda H, R, kk: np.nansum(R[:, kk]) / max(np.nansum(H[:, kk]), 1)   # noqa: E731
        n = Ha.shape[-1]
        f = lambda kk: float(rate(Ha, Ra, kk) - rate(Hb, Rb, kk))           # noqa: E731
        d = [f(np.arange(n)), *L.boot(f, n)]
        return dict(metric=a.metric, candidate=float(rate(Ha, Ra, np.arange(n))), reference=float(rate(Hb, Rb, np.arange(n))), delta=d,
                    rule="improves iff the CI lower bound > 0 (PARTS_RESULTS 30)", improves=bool(d[1] is not None and d[1] > 0))
    if a.metric in ("bottle_hold_fam", "bottle_hold_new"):                   # PARTS_RESULTS 38 / 41: hold over the bottle's tasks
        bt = ["K4", "K5", "N4", "N5", "N6"]
        lay = "familiar" if a.metric == "bottle_hold_fam" else "new"
        Ha, _ = held_arrived(cand, bt, lay)
        Hb, _ = held_arrived(ref, bt, lay)
        n = Ha.shape[-1]
        f = lambda kk: float(np.nanmean(Ha[:, kk]) - np.nanmean(Hb[:, kk]))  # noqa: E731
        d = [f(np.arange(n)), *L.boot(f, n)]
        out = dict(metric=a.metric, candidate=float(np.nanmean(Ha)), reference=float(np.nanmean(Hb)), delta=d,
                   rule=f"improves iff delta > 0.10 and the CI lower bound > 0 (PARTS_RESULTS {38 if lay == 'familiar' else 41})",
                   improves=bool(d[0] > 0.10 and d[1] is not None and d[1] > 0))
        if lay == "new":                                                     # PARTS_RESULTS 41: familiar non-regression (reported)
            A = C.stack(cand, cand.s2[0], cand.s2[1], known, "familiar").transpose(1, 0, 2).reshape(-1, len(cand.states))
            B_ = C.stack(ref, ref.s2[0], ref.s2[1], known, "familiar").transpose(1, 0, 2).reshape(-1, len(ref.states))
            g = lambda kk: float(np.nanmean(A[:, kk]) - np.nanmean(B_[:, kk]))  # noqa: E731
            dg = [g(np.arange(n)), *L.boot(g, n)]
            out["familiar_nonregression"] = dict(candidate=float(np.nanmean(A)), reference=float(np.nanmean(B_)), delta=dg,
                                                 rule="regresses iff the CI upper bound < 0 (reported, no verdict)",
                                                 regresses=bool(dg[2] is not None and dg[2] < 0))
        return out
    if a.metric == "n4_fam":
        A = C.stack(cand, cand.s2[0], cand.s2[1], ["N4"], "familiar")[:, 0]   # [seeds, states]
        B_ = C.stack(ref, ref.s2[0], ref.s2[1], ["N4"], "familiar")[:, 0]
    else:
        A = C.stack(cand, cand.s2[0], cand.s2[1], known, "new").transpose(1, 0, 2).reshape(-1, len(cand.states))
        B_ = C.stack(ref, ref.s2[0], ref.s2[1], known, "new").transpose(1, 0, 2).reshape(-1, len(ref.states))
    n = A.shape[-1]
    f = lambda kk: float(np.nanmean(A[:, kk]) - np.nanmean(B_[:, kk]))       # noqa: E731
    d = [f(np.arange(n)), *L.boot(f, n)]
    return dict(metric=a.metric, candidate=float(np.nanmean(A)), reference=float(np.nanmean(B_)), delta=d,
                improves=bool(d[0] > 0.10 and d[1] is not None and d[1] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--ref-seeds", default="")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--ref-s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--metric", required=True, choices=("n4_fam", "s_known_new", "cab_arr_fam", "bottle_hold_fam", "bottle_hold_new"))
    ap.add_argument("--s1-extra", default="")                               # PARTS_RESULTS 38: further S1 files that must all hold
    ap.add_argument("--phase", default="main", choices=("main", "conf"))
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split("+"))
    cand = cfg_of(a, seeds)
    ref = cfg_of(a, tuple(int(x) for x in a.ref_seeds.split("+")) if a.ref_seeds else seeds)
    ref.s2 = tuple(a.ref_s2.split(":"))
    out = dict(S1=C.s1(cand), S2=C.s2(cand))
    if a.s1_extra:                                                           # S1 restated (m held fixed at each value): all must hold
        out["S1_parts"] = {cand.s1: out["S1"]["holds"]}
        for f in a.s1_extra.split("+"):
            cand.s1 = f
            out["S1_parts"][f] = C.s1(cand)["holds"]
        out["S1"]["holds"] = bool(all(out["S1_parts"].values()))
    out["S3"], _ = C.s3s4(cand)
    out["layer3_go"] = bool(out["S1"]["holds"] and out["S2"]["holds"] and out["S3"]["all_transfer"])
    out["round_metric"] = metric(a, cand, ref)
    out["reference_S2"] = C.s2(ref)
    json.dump(out, open(os.path.join(OUT, f"pp_{a.tag}_read.json"), "w", encoding="utf-8"), indent=1, default=float)
    r = lambda x: None if x is None else round(float(x), 3)                  # noqa: E731
    print("ROUND", a.tag, json.dumps(out["round_metric"]), "GO", out["layer3_go"], flush=True)
    print("S1", out["S1"]["holds"], "S2", out["S2"]["holds"], "S3", json.dumps(out["S3"]["transfer_criterion"]), flush=True)
    for nm, S2 in (("cand", out["S2"]), ("ref", out["reference_S2"])):
        for lay in ("familiar", "new"):
            d = S2[lay]
            print("S2", nm, lay, "OM", [r(x) for x in d["object_matched"]], "3F", d["three_factor_holds"], "s_known", r(d["s_known"]), "s_new",
                  r(d["s_new"]), "miss", d["n_missing"], {c: (r(v["obs"]), r(v["pred"]), r(v["D3"]), [r(x) for x in v["D3_ci"]]) for c, v in d["three_factor"].items()},
                  {c: r(v) for c, v in d["per_compound"].items()}, flush=True)
    for key, c in out["S3"]["cells"].items():
        for lay, v in (c or {}).items():
            if v:
                print("S3", key, lay, r(v["s_train"]), r(v["s_unseen"]), r(v["ratio"]), [r(x) for x in v["ratio_ci"]],
                      {u: r(x) for u, x in v["per_unseen"].items()}, "miss", v["n_missing"], flush=True)


if __name__ == "__main__":
    main()

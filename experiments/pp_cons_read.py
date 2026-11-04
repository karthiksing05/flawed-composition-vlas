"""PARTS_RESULTS 52 (consolidation, m3_prereg pp_cons): the leave-one-out and combined-arm reads against P_nsz, with the frozen
keep/drop rule.  CPU.  Estimators are pp_l2r32_read's (S1, S2, S3) and pp_round_read's (the options' own metrics), unchanged.

  --arm      name (output results/flawed/pp_cons_<arm>_read.json; must not exist)
  --removed  '+'-separated options removed from P_nsz in this arm (their own metrics are read)
  --s1       '+'-separated S1 files (m fixed at each inventory value); all must hold
  --s2       <dir>:<name> of the arm's S2 rollouts        --s3  <dir>:<prefix> of its S3 rollouts (<prefix>_<cond>_k<k>)
  --exact    an exactness file (pp_cons_check.py): the arm is P_nsz's function exactly, so its read IS P_nsz's read (delta 0)

Rule (frozen in PARTS_RESULTS 52 and m3_prereg pp_cons; never changed after a read).  Removing the options WORSENS iff any of
  (a) an own metric of a removed option: delta = arm - P_nsz (2,000-draw paired state bootstrap, seeds pooled) < -0.10 and its
      CI upper bound < 0;
  (b) familiar non-regression (pooled familiar known-pairing success K1-K6): the same rule;
  (c) a GO item that passes for P_nsz fails for the arm.  GO items: S1 (restated, m fixed); per layout, OM; per layout and new
      pairing N1-N6, 3F (D3 CI contains 0); per atom and layout, S3 (all of k = 1, 5, 10 meet ratio >= 0.8 and LB >= 0.6).
The verdict is DROP iff the arm does not worsen, else KEEP.  `go_both_layouts` flags the plan's stop condition (the arm passes
Layer-2 GO on dev under pp_l2's mapping).
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
OWN = dict(rtclip="n4_fam", tscale="n4_fam", aframe="s_known_new", bbnom="k5_hold_new", bbnosize="bottle_hold_new")
REF = dict(s1="pp_s1_mix_full_nsz_m0.json+pp_s1_mix_full_nsz_m1.json+pp_s1_mix_full_nsz_m2.json", s2="pp_r15_roll:P2nsz",
           s3="pp_r15_roll:Pnsz")
KNOWN = ["K1", "K2", "K3", "K4", "K5", "K6"]
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]


def cfg(s1, s2, s3):
    c = R.cfg_of(argparse.Namespace(phase="main", s1=s1.split("+")[0], s2=s2, s3=s3), SEEDS)
    c.s1_all = s1.split("+")
    return c


def components(c):
    s1 = {}
    for f in c.s1_all:
        c.s1 = f
        s1[f] = bool(C.s1(c)["holds"])
    S2 = C.s2(c)
    S3, _ = C.s3s4(c)
    items = {"S1": bool(all(s1.values()))}
    for lay in ("familiar", "new"):
        items[f"OM|{lay}"] = bool(S2[lay]["object_matched_holds"])
        for p in NEW:
            ci = S2[lay]["three_factor"][p]["D3_ci"]
            items[f"3F|{lay}|{p}"] = bool(ci[0] <= 0.0 <= ci[1])
    for cond in L.COND:
        for lay in ("familiar", "new"):
            items[f"S3|{cond}|{lay}"] = bool(all(S3["transfer_criterion"][f"{cond}|{lay}|k{k}"] for k in (1, 5, 10)))
    go = bool(items["S1"] and S2["holds"] and S3["all_transfer"])
    return dict(items=items, S1_files=s1, S2=S2, S3=dict(transfer_criterion=S3["transfer_criterion"], cells=S3["cells"]), layer2_go=go)


def fam_nonreg(cand, ref):
    A = C.stack(cand, cand.s2[0], cand.s2[1], KNOWN, "familiar").transpose(1, 0, 2).reshape(-1, len(cand.states))
    B = C.stack(ref, ref.s2[0], ref.s2[1], KNOWN, "familiar").transpose(1, 0, 2).reshape(-1, len(ref.states))
    n = A.shape[-1]
    g = lambda kk: float(np.nanmean(A[:, kk]) - np.nanmean(B[:, kk]))       # noqa: E731
    return dict(candidate=float(np.nanmean(A)), reference=float(np.nanmean(B)), delta=[g(np.arange(n)), *L.boot(g, n)])


def worse(d):
    return bool(d[0] < -0.10 and d[2] is not None and d[2] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--removed", required=True)
    ap.add_argument("--s1", default="")
    ap.add_argument("--s2", default="")
    ap.add_argument("--s3", default="")
    ap.add_argument("--exact", default="")
    a = ap.parse_args()
    fn = os.path.join(OUT, f"pp_cons_{a.arm}_read.json")
    assert not os.path.exists(fn), f"{fn} exists"
    removed = a.removed.split("+")
    assert all(x in OWN for x in removed), removed
    ref = cfg(REF["s1"], REF["s2"], REF["s3"])
    rc = components(ref)
    if a.exact:
        ex = json.load(open(os.path.join(OUT, a.exact), encoding="utf-8"))
        assert ex["all_exact"], "exactness check failed: this arm needs rollouts"
        cc = rc
        own = {x: dict(metric=OWN[x], delta=[0.0, 0.0, 0.0], exact=True) for x in removed}
        fnr = dict(delta=[0.0, 0.0, 0.0], exact=True)
    else:
        cand = cfg(a.s1, a.s2, a.s3)
        cc = components(cand)
        own = {}
        for x in removed:
            r = R.metric(argparse.Namespace(metric=OWN[x]), cand, ref)
            own[x] = dict(metric=OWN[x], candidate=r["candidate"], reference=r["reference"], delta=r["delta"])
        fnr = fam_nonreg(cand, ref)
    flips = [k for k, v in rc["items"].items() if v and not cc["items"][k]]
    gains = [k for k, v in rc["items"].items() if not v and cc["items"][k]]
    why = [f"own metric {x} ({own[x]['metric']})" for x in removed if worse(own[x]["delta"])]
    why += ["familiar non-regression"] if worse(fnr["delta"]) else []
    why += [f"GO item flips: {k}" for k in flips]
    out = dict(arm=a.arm, removed=removed, own_metrics=own, familiar_nonregression=fnr, flips_pass_to_fail=flips,
               flips_fail_to_pass=gains, worsens=bool(why), why=why, verdict="KEEP" if why else "DROP",
               go_both_layouts=bool(cc["layer2_go"]), candidate=cc, reference=rc,
               rule="worsens iff (a) an own metric or (b) familiar non-regression has delta < -0.10 with CI upper bound < 0, or (c) a "
                    "GO item passing for P_nsz fails (PARTS_RESULTS 52)")
    json.dump(out, open(fn, "w", encoding="utf-8"), indent=1, default=float)
    r = lambda v: None if v is None else round(float(v), 3)                  # noqa: E731
    print("CONS", a.arm, "removed", removed, "VERDICT", out["verdict"], "why", why, "GO", cc["layer2_go"], flush=True)
    for x, v in own.items():
        print("OWN", x, v["metric"], [r(z) for z in v["delta"]], flush=True)
    print("FAMNR", [r(z) for z in fnr["delta"]], "flips", flips, "gains", gains, flush=True)
    for nm, comp in (("arm", cc), ("ref", rc)):
        for lay in ("familiar", "new"):
            d = comp["S2"][lay]
            print("S2", nm, lay, "OM", [r(x) for x in d["object_matched"]], "s_known", r(d["s_known"]), "s_new", r(d["s_new"]), "miss",
                  d["n_missing"], {p: (r(v["obs"]), r(v["pred"]), r(v["D3"]), [r(x) for x in v["D3_ci"]]) for p, v in d["three_factor"].items()},
                  flush=True)
        for key, cell in comp["S3"]["cells"].items():
            for lay, v in (cell or {}).items():
                if v:
                    print("S3", nm, key, lay, r(v["s_train"]), r(v["s_unseen"]), r(v["ratio"]), [r(x) for x in v["ratio_ci"]], "miss",
                          v["n_missing"], flush=True)


if __name__ == "__main__":
    main()

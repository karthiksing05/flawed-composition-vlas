"""Coverage arm read (docs/COVERAGE_RESULTS.md 1.8 and 2; m3_prereg pp_cov).  CPU.  Dev states 4-23, N2 seeds, seeds 10-12 only.

The GO components are pp_cons_read.components (pp_l2r32_read's and pp_l2_read's frozen estimators, unchanged) on P_cov and on P_min.
Mapping (frozen in COVERAGE_RESULTS 1.8; never changed after the read): the condition-S account is SUPPORTED iff
  (a) S1 restated is exact for P_cov;
  (b) new-layout 3F passes (D3 CI contains 0) on every pairing N1-N6;
  (c) new-layout OM passes (ratio >= 0.8, LB >= 0.6);
  (d) new-layout S3 passes for each atom (bottle, plate): all k in {1, 5, 10} meet ratio >= 0.8 with LB >= 0.6;
  (e) familiar non-regression: every familiar GO item passing for P_min passes for P_cov, and pooled familiar K1-K6 success
      delta = P_cov - P_min (2,000-draw paired state bootstrap, seeds pooled) does not have delta < -0.10 with CI upper < 0.
Otherwise NOT SUPPORTED.

    python experiments/pp_cov_read.py --s1 pp_s1_mix_cov_full_m0.json+... --s2 pp_cov_roll:P2cov --s3 pp_cov_roll:Pcov
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_cons_read as CR                                                    # noqa: E402
import pp_l2_read as L                                                       # noqa: E402
import pp_l2r32_read as C                                                    # noqa: E402

OUT = C.OUT
REF = dict(s1="pp_s1_mix_full_naf_m0.json+pp_s1_mix_full_naf_m1.json+pp_s1_mix_full_naf_m2.json", s2="pp_cons_roll:P2naf",
           s3="pp_cons_roll:Pnaf")
NEW = CR.NEW
KNOWN = CR.KNOWN
ALL = [f"K{i}" for i in range(1, 9)] + [f"N{i}" for i in range(1, 9)]


def competence(c):
    out = {}
    for lay in ("familiar", "new"):
        S = C.stack(c, c.s2[0], c.s2[1], ALL, lay)                         # [seeds, compounds, states]
        n = S.shape[-1]
        out[lay] = {}
        for i, comp in enumerate(ALL):
            g = lambda kk, i=i: float(np.nanmean(S[:, i][:, kk]))           # noqa: E731
            out[lay][comp] = [g(np.arange(n)), *L.boot(g, n)]
    return out


def paired(cand, ref, comps, lay):
    A = C.stack(cand, cand.s2[0], cand.s2[1], comps, lay).transpose(1, 0, 2).reshape(-1, len(cand.states))
    B = C.stack(ref, ref.s2[0], ref.s2[1], comps, lay).transpose(1, 0, 2).reshape(-1, len(ref.states))
    n = A.shape[-1]
    g = lambda kk: float(np.nanmean(A[:, kk]) - np.nanmean(B[:, kk]))       # noqa: E731
    return dict(candidate=float(np.nanmean(A)), reference=float(np.nanmean(B)), delta=[g(np.arange(n)), *L.boot(g, n)])


def gap(c, comps):
    """familiar - new pooled success over `comps`, paired by state (the same state index in both layouts' state lists)."""
    F = C.stack(c, c.s2[0], c.s2[1], comps, "familiar").transpose(1, 0, 2).reshape(-1, len(c.states))
    N = C.stack(c, c.s2[0], c.s2[1], comps, "new").transpose(1, 0, 2).reshape(-1, len(c.states))
    n = F.shape[-1]
    g = lambda kk: float(np.nanmean(F[:, kk]) - np.nanmean(N[:, kk]))       # noqa: E731
    return [g(np.arange(n)), *L.boot(g, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", default="pp_cov_read.json")
    a = ap.parse_args()
    fn = os.path.join(OUT, a.out)
    assert not os.path.exists(fn), f"{fn} exists"
    ref, cand = CR.cfg(REF["s1"], REF["s2"], REF["s3"]), CR.cfg(a.s1, a.s2, a.s3)
    assert tuple(cand.seeds) == (10, 11, 12) and cand.phase == "main" and ref.phase == "main"
    rc, cc = CR.components(ref), CR.components(cand)
    fnr = CR.fam_nonreg(cand, ref)
    it = cc["items"]
    fam_lost = [k for k, v in rc["items"].items() if v and "familiar" in k and not it[k]]
    clauses = dict(a_S1=bool(it["S1"]),
                   b_3F_new_all=bool(all(it[f"3F|new|{p}"] for p in NEW)),
                   c_OM_new=bool(it["OM|new"]),
                   d_S3_new_each_atom=bool(all(it[f"S3|{cond}|new"] for cond in L.COND)),
                   e_familiar_nonregression=bool(not fam_lost and not CR.worse(fnr["delta"])))
    verdict = "SUPPORTED" if all(clauses.values()) else "NOT SUPPORTED"
    out = dict(verdict=verdict, clauses=clauses, failing=[k for k, v in clauses.items() if not v], familiar_items_lost=fam_lost,
               familiar_nonregression=fnr, new_known_delta=paired(cand, ref, KNOWN, "new"), new_new_delta=paired(cand, ref, NEW, "new"),
               gap_known_fam_minus_new=dict(P_cov=gap(cand, KNOWN), P_min=gap(ref, KNOWN)),
               gap_new_fam_minus_new=dict(P_cov=gap(cand, NEW), P_min=gap(ref, NEW)),
               competence=dict(P_cov=competence(cand), P_min=competence(ref)),
               flips_pass_to_fail=[k for k, v in rc["items"].items() if v and not it[k]],
               flips_fail_to_pass=[k for k, v in rc["items"].items() if not v and it[k]],
               candidate=cc, reference=rc, states="dev: N2 main states 4-23, N2 seeds; seeds 10-12",
               mapping="COVERAGE_RESULTS 1.8 (m3_prereg pp_cov)")
    json.dump(out, open(fn, "w", encoding="utf-8"), indent=1, default=float)
    r = lambda v: None if v is None else round(float(v), 3)                  # noqa: E731
    print("COV VERDICT", verdict, json.dumps(clauses), "familiar lost", fam_lost, flush=True)
    print("FAMNR", [r(z) for z in fnr["delta"]], "NEWK", [r(z) for z in out["new_known_delta"]["delta"]],
          "NEWN", [r(z) for z in out["new_new_delta"]["delta"]], flush=True)
    for arm in ("P_cov", "P_min"):
        print("GAP", arm, "known", [r(z) for z in out["gap_known_fam_minus_new"][arm]], "new", [r(z) for z in out["gap_new_fam_minus_new"][arm]],
              flush=True)
    for nm, comp in (("P_cov", cc), ("P_min", rc)):
        print("ITEMS", nm, json.dumps(comp["items"]), flush=True)
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
    for arm, comp in out["competence"].items():
        for lay, v in comp.items():
            print("COMP", arm, lay, {c: r(x[0]) for c, x in v.items()}, flush=True)


if __name__ == "__main__":
    main()

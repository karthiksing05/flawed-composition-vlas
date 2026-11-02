"""PLAN_PARTS Layer 2 confirmatory read, rank-32 parts (docs/PARTS_RESULTS.md section 6; m3_prereg pp_l2r32).  CPU.

pp_l2_read.py's estimators and criteria, unchanged, on the confirmatory data:
  seeds 10, 11, 12 (training and creation); FRESH start states 0-49 of results/flawed/pp_conf_states (pp_conf_states.py) with their
  noise seeds; rollout files <name>_s<seed>_conf_<tid>.json.
  S1  results/flawed/pp_s1_mix_full_r32c.json (pp_s1_mix.py on pp_full_r32_s10-12)
  S2  pp_s2c_roll, name P2r32       S3  pp_s3c_roll, names P_r32_<cond>_k<k>       S4  pp_s3c_roll <K>_<cond>_k<k>; pp_s4c_old
The three-factor test is fa1g_read.three_factor's estimator with the state list and file phase as parameters (identical code path
otherwise).  `--check` re-runs this module's estimators on pp_l2's development data (seeds 0-2, states 4-23, phase main) and asserts
that they reproduce results/flawed/pp_l2_read.json; it reads nothing new.
-> results/flawed/pp_l2r32_read.json

    python experiments/pp_l2r32_read.py [--check]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import radius                                                 # noqa: E402
import pp_l2_read as L                                                       # noqa: E402

OUT = RX.OUT
B, SEED = 2000, 0
KS = (1, 5, 10, 50)


class Cfg:
    def __init__(self, conf):
        if conf:
            self.seeds, self.phase = (10, 11, 12), "conf"
            self.states = json.load(open(os.path.join(OUT, "pp_conf_states.json"), encoding="utf-8"))["states"]
            self.s1 = "pp_s1_mix_full_r32c.json"
            self.s2 = ("pp_s2c_roll", "P2r32")
            self.s3dir, self.olddir = "pp_s3c_roll", "pp_s4c_old"
            self.pname = lambda cond, k: f"P_r32_{cond}_k{k}"                               # noqa: E731
        else:                                                               # pp_l2's development data (the --check)
            self.seeds, self.phase = (0, 1, 2), "main"
            self.states = L.n2()["states"]["main"]
            self.s1 = "pp_s1_mix_full.json"
            self.s2 = ("pp_s2_roll", "P2")
            self.s3dir, self.olddir = "pp_s3_roll", "pp_s4_old"
            self.pname = lambda cond, k: f"P_{cond}_k{k}"                                   # noqa: E731


def succ(cfg, path):
    s = np.full(len(cfg.states), np.nan)
    if not os.path.exists(path):
        return s
    for e in [e for v in json.load(open(path, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
        s[cfg.states.index(int(e["trial"]))] = float(bool(e["success"]))
    return s


def stack(cfg, outdir, name, comps, lay):
    return np.array([[succ(cfg, os.path.join(OUT, outdir, f"{name}_s{sd}_{cfg.phase}_{L.tid_of(c, lay)}.json")) for c in comps]
                     for sd in cfg.seeds])


def three_factor(cfg, outdir, m, layout):
    """fa1g_read.three_factor with the states, seeds and file phase as parameters."""
    n2 = L.n2()
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    fam = {t["compound"]: t for t in n2["tasks"] if t["layout"] == layout and int(t["compound"][1:]) <= 6}
    ks = cfg.states
    HH, AR, S = {}, {}, {}
    for c, t in fam.items():
        rad = radius(t["object"], t["target"], cal)
        h, ar, s = (np.zeros((len(cfg.seeds), len(ks))) for _ in range(3))
        for i, sd in enumerate(cfg.seeds):
            f = os.path.join(OUT, outdir, f"{m}_s{sd}_{cfg.phase}_{t['tid']}.json")
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


def s2(cfg):
    res = {}
    known = ["K1", "K2", "K3", "K4", "K5", "K6"]
    new = ["N1", "N2", "N3", "N4", "N5", "N6"]
    match = {"N1": ["K6"], "N2": ["K6"], "N3": ["K6"], "N4": ["K4", "K5"], "N5": ["K4", "K5"], "N6": ["K4", "K5"]}
    d, nm = cfg.s2
    for lay in ("familiar", "new"):
        tf = three_factor(cfg, d, nm, lay)
        f3 = all(v["D3_ci"][0] <= 0.0 <= v["D3_ci"][1] for v in tf.values())
        S = stack(cfg, d, nm, known + new, lay)
        idx = {c: i for i, c in enumerate(known + new)}

        def r_of(kk):
            num = np.mean([np.nanmean(S[:, idx[c]][:, kk]) for c in new])
            den = np.mean([np.mean([np.nanmean(S[:, idx[k]][:, kk]) for k in match[c]]) for c in new])
            return num / den if den > 0 else None
        n = S.shape[-1]
        r = r_of(np.arange(n))
        ci = L.boot(r_of, n)
        om = bool(r is not None and r >= 0.8 and ci[0] is not None and ci[0] >= 0.6)
        res[lay] = dict(three_factor=tf, three_factor_holds=f3, object_matched=[r, *ci], object_matched_holds=om,
                        s_known=float(np.nanmean(S[:, :6])), s_new=float(np.nanmean(S[:, 6:])),
                        per_compound={c: float(np.nanmean(S[:, i])) for c, i in idx.items()}, n_missing=int(np.isnan(S).sum()))
    res["holds"] = all(res[l]["three_factor_holds"] and res[l]["object_matched_holds"] for l in ("familiar", "new"))
    return res


def s3_one(cfg, name, cond):
    C = L.COND[cond]
    out = {}
    for lay in ("familiar", "new"):
        S = stack(cfg, cfg.s3dir, name, (C["train"],) + C["unseen"], lay)
        if np.isnan(S).all():
            out[lay] = None
            continue

        def rat(kk):
            st = np.nanmean(S[:, 0][:, kk])
            su = np.nanmean(S[:, 1:][:, :, kk])
            return su / st if st > 0 else None
        n = S.shape[-1]
        out[lay] = dict(s_train=float(np.nanmean(S[:, 0])), s_unseen=float(np.nanmean(S[:, 1:])), ratio=rat(np.arange(n)), ratio_ci=L.boot(rat, n),
                        per_unseen={c: float(np.nanmean(S[:, 1 + i])) for i, c in enumerate(C["unseen"])}, n_missing=int(np.isnan(S).sum()), S=S)
    return out


def forgetting(cfg, method, cond, k):
    C = L.COND[cond]
    base = stack(cfg, cfg.olddir, f"{method}_{cond}_base", C["old"], "familiar")
    ad = stack(cfg, cfg.olddir, f"{method}_{cond}_k{k}", C["old"], "familiar")
    if np.isnan(base).all() or np.isnan(ad).all():
        return None
    n = base.shape[-1]
    d = lambda kk: float(np.nanmean(ad[:, :, kk]) - np.nanmean(base[:, :, kk]))     # noqa: E731
    return dict(base=float(np.nanmean(base)), adapted=float(np.nanmean(ad)), delta=d(np.arange(n)), ci=L.boot(d, n))


def s3s4(cfg):
    S3, S4 = {}, {}
    for cond in L.COND:
        for k in KS:
            S3[f"P|{cond}|k{k}"] = s3_one(cfg, cfg.pname(cond, k), cond)
            for m in ("HF", "HL", "RF", "LIB"):
                S4[f"{m}|{cond}|k{k}"] = s3_one(cfg, f"{m}_{cond}_k{k}", cond)
    diffs = {}
    for key, r in S4.items():
        m, cond, kk = key.split("|")
        p = S3.get(f"P|{cond}|{kk}")
        for lay in ("familiar", "new"):
            if not r or not p or r.get(lay) is None or p.get(lay) is None:
                continue
            A, Bm = p[lay]["S"][:, 1:], r[lay]["S"][:, 1:]
            n = A.shape[-1]
            f = lambda q: float(np.nanmean(A[:, :, q]) - np.nanmean(Bm[:, :, q]))   # noqa: E731
            diffs[f"P-{m}|{cond}|{kk}|{lay}"] = [f(np.arange(n)), *L.boot(f, n)]
    forg = {f"{m}|{c}|k{k}": forgetting(cfg, m, c, k) for m in ("HF", "RF", "LIB") for c in L.COND for k in KS}
    go = {}
    for cond in L.COND:
        for lay in ("familiar", "new"):
            for k in (1, 5, 10):
                r = (S3.get(f"P|{cond}|k{k}") or {}).get(lay)
                go[f"{cond}|{lay}|k{k}"] = bool(r and r["ratio"] is not None and r["ratio"] >= 0.8 and r["ratio_ci"][0] is not None
                                                and r["ratio_ci"][0] >= 0.6)
    strip = lambda d: {k: ({l: ({kk: vv for kk, vv in v.items() if kk != "S"} if v else None) for l, v in r.items()} if r else None)  # noqa: E731
                       for k, r in d.items()}
    return dict(cells=strip(S3), transfer_criterion=go, all_transfer=all(go.values())), dict(cells=strip(S4), P_minus_baseline_unseen=diffs,
                                                                                              forgetting=forg)


def s1(cfg):
    r = json.load(open(os.path.join(OUT, cfg.s1), encoding="utf-8"))
    sw = r["swap_tests"]
    swaps0 = all(v["T_obj_target_swap"] == 0.0 and v["T_tgt_object_part_swap"] == 0.0 for v in sw.values())
    tg0, opmax = True, {}
    for run, cells in r["runs"].items():
        for u, c in cells.items():
            if c.get("skipped"):
                continue
            for comp, d in c["d"].items():
                if comp == "full_own":
                    continue
                if c["phase"] == "grasp" and any(x != 0.0 for x in d):
                    tg0 = False
                if c["phase"] == "place":
                    opmax[comp] = max(opmax.get(comp, 0.0), float(np.sum(d) / max(np.sum(c["d"]["full_own"]), 1e-12)))
    return dict(swap_tests=sw, swaps_exact_0=swaps0, tg_profile_exact_0=tg0, op_max_m=opmax, holds=bool(swaps0 and tg0))


def check():
    """This module on pp_l2's development data must reproduce pp_l2_read.json (P rank 8, the baselines, S1, S2)."""
    cfg = Cfg(False)
    ref = json.load(open(os.path.join(OUT, "pp_l2_read.json"), encoding="utf-8"))
    a = s2(cfg)
    for lay in ("familiar", "new"):
        assert np.allclose(a[lay]["object_matched"], ref["S2"][lay]["object_matched"]), lay
        for c, v in a[lay]["three_factor"].items():
            assert np.allclose([v["D3"], *v["D3_ci"]], [ref["S2"][lay]["three_factor"][c]["D3"], *ref["S2"][lay]["three_factor"][c]["D3_ci"]]), c
    assert a["holds"] == ref["S2"]["holds"]
    b3, b4 = s3s4(cfg)
    for key, r in b3["cells"].items():
        for lay in ("familiar", "new"):
            x, y = r[lay], ref["S3"]["cells"][key][lay]
            assert np.isclose(x["ratio"], y["ratio"]) and np.allclose(x["ratio_ci"], y["ratio_ci"]), key
    for key, v in b4["P_minus_baseline_unseen"].items():
        assert np.allclose(v, ref["S4"]["P_minus_baseline_unseen"][key]), key
    assert s1(cfg)["holds"] == ref["S1"]["holds"]
    print("[pp_l2r32_read] check ok: reproduces pp_l2_read.json on the development data", flush=True)


def main():
    if "--check" in sys.argv:
        check()
        return
    cfg = Cfg(True)
    out = dict(S1=s1(cfg), S2=s2(cfg))
    out["S3"], out["S4"] = s3s4(cfg)
    out["layer3_go"] = bool(out["S1"]["holds"] and out["S2"]["holds"] and out["S3"]["all_transfer"])
    json.dump(out, open(os.path.join(OUT, "pp_l2r32_read.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("S1", out["S1"]["holds"], "S2", out["S2"]["holds"], "S3", json.dumps(out["S3"]["transfer_criterion"]), "LAYER3_GO", out["layer3_go"],
          flush=True)


if __name__ == "__main__":
    main()

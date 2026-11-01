"""PLAN_PARTS Layer 2 read: S1, S2, S3, S4 (docs/PARTS_RESULTS.md section 4; m3_prereg tag pp_l2).  CPU.

S1  results/flawed/pp_s1_mix_full.json: holds iff both swap tests are exactly 0 for every seed and every target-into-grasp component
    of P has m exactly 0 on every frame.  Object-into-place is reported (A's foot route is unchanged in P).
S2  rollouts results/flawed/pp_s2_roll/P2_s<seed>_main_<tid>.json (P built on all known tasks, seeds 0-2, states 4-23).  Per layout:
      3F   every N1-N6 CI of D3 contains 0 (fa1g_read.three_factor, unchanged)
      OM   object-matched ratio r = mean over N1-N6 of s(new pairing) / mean over N1-N6 of s(the known tasks of the same object)
           (cheese: K6; bottle: K4, K5); 95% CI: 2,000-draw bootstrap (seed 0) over states, seeds pooled.  Holds iff r >= 0.8 and
           its lower bound >= 0.6.
    S2 HOLDS iff 3F and OM hold in both layouts.  Pooled success s_known, s_new is reported as competence.
S3  per held-out atom (bottle: base without K4, K5, part created from K4; plate: base without K1, part created from K1), k, layout:
    s_train = success on the training-partner task, s_unseen = mean success over the unseen-partner tasks (bottle: N4, N5, N6; plate:
    N1, N4), over 3 seeds x 20 states; transfer ratio = s_unseen / s_train; CI: bootstrap over states (seeds and tasks pooled).
    Layer-3 GO (plan section 4.3) iff S1 holds, S2 holds, and for both atoms, both layouts and every k in {1, 5, 10}: ratio >= 0.8
    with CI lower bound >= 0.6.  Old tasks: P's created part changes no other parameter (asserted bitwise at creation), so the change
    on old tasks is exactly 0.  Secondary arm: rank-32 parts (tag *_r32), the same read.
S4  the same s_train, s_unseen and ratio for HF, HL, RF, LIB; forgetting = s(adapted) - s(base) on the old pick-and-place tasks
    (familiar layout; nobottle K1 K2 K3 K6, noplate K2-K6) for HF, RF, LIB (HL is off on old tasks: exactly 0); the difference
    s_unseen(P) - s_unseen(baseline) per k with a state-bootstrap CI.  Descriptive: no verdict is attached to S4.
-> results/flawed/pp_l2_read.json

    python experiments/pp_l2_read.py [--only s1|s2|s3]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import three_factor                                           # noqa: E402

OUT = RX.OUT
B, SEED, SEEDS = 2000, 0, (0, 1, 2)
KS = (1, 5, 10, 50)
COND = {"nobottle": dict(atom="wine_bottle_1", train="K4", unseen=("N4", "N5", "N6"), old=("K1", "K2", "K3", "K6")),
        "noplate": dict(atom="plate_1", train="K1", unseen=("N1", "N4"), old=("K2", "K3", "K4", "K5", "K6"))}
METHODS = ("P", "HF", "HL", "RF", "LIB")


def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def tid_of(c, lay):
    return next(t["tid"] for t in n2()["tasks"] if t["compound"] == c and t["layout"] == lay)


def succ(path):
    """-> success array over states 4-23 (NaN if missing)."""
    ks = n2()["states"]["main"]
    s = np.full(len(ks), np.nan)
    if not os.path.exists(path):
        return s
    for e in [e for v in json.load(open(path, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
        s[ks.index(int(e["trial"]))] = float(bool(e["success"]))
    return s


def stack(outdir, name, comps, lay):
    """-> [seeds, compounds, states]."""
    return np.array([[succ(os.path.join(OUT, outdir, f"{name}_s{sd}_main_{tid_of(c, lay)}.json")) for c in comps] for sd in SEEDS])


def boot(fn, n):
    rng = np.random.default_rng(SEED)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]


def s1():
    r = json.load(open(os.path.join(OUT, "pp_s1_mix_full.json"), encoding="utf-8"))
    sw = r["swap_tests"]
    swaps0 = all(v["T_obj_target_swap"] == 0.0 and v["T_tgt_object_part_swap"] == 0.0 for v in sw.values())
    tg0, opmax = True, {}
    for run, cells in r["runs"].items():
        for u, c in cells.items():
            if c.get("skipped"):
                continue
            for comp, d in c["d"].items():
                if comp in ("full_own",):
                    continue
                if c["phase"] == "grasp" and comp != "full_own" and any(x != 0.0 for x in d):
                    tg0 = False
                if c["phase"] == "place":
                    opmax[comp] = max(opmax.get(comp, 0.0), float(np.sum(d) / max(np.sum(c["d"]["full_own"]), 1e-12)))
    return dict(swap_tests=sw, swaps_exact_0=swaps0, tg_profile_exact_0=tg0, op_max_m=opmax, holds=bool(swaps0 and tg0))


def s2():
    RX.ROLLDIR["P2"] = "pp_s2_roll"
    res = {}
    known = ["K1", "K2", "K3", "K4", "K5", "K6"]
    new = ["N1", "N2", "N3", "N4", "N5", "N6"]
    match = {"N1": ["K6"], "N2": ["K6"], "N3": ["K6"], "N4": ["K4", "K5"], "N5": ["K4", "K5"], "N6": ["K4", "K5"]}
    for lay in ("familiar", "new"):
        tf = three_factor("P2", lay, seeds=SEEDS)
        f3 = all(v["D3_ci"][0] <= 0.0 <= v["D3_ci"][1] for v in tf.values())
        S = stack("pp_s2_roll", "P2", known + new, lay)                 # [3, 12, 20]
        idx = {c: i for i, c in enumerate(known + new)}

        def r_of(kk):
            num = np.mean([np.nanmean(S[:, idx[c]][:, kk]) for c in new])
            den = np.mean([np.mean([np.nanmean(S[:, idx[k]][:, kk]) for k in match[c]]) for c in new])
            return num / den if den > 0 else None
        n = S.shape[-1]
        r = r_of(np.arange(n))
        ci = boot(r_of, n)
        om = bool(r is not None and r >= 0.8 and ci[0] is not None and ci[0] >= 0.6)
        res[lay] = dict(three_factor=tf, three_factor_holds=f3, object_matched=[r, *ci], object_matched_holds=om,
                        s_known=float(np.nanmean(S[:, :6])), s_new=float(np.nanmean(S[:, 6:])),
                        per_compound={c: float(np.nanmean(S[:, i])) for c, i in idx.items()})
    res["holds"] = all(res[l]["three_factor_holds"] and res[l]["object_matched_holds"] for l in ("familiar", "new"))
    return res


def s3_one(method, cond, k, tag_suffix=""):
    C = COND[cond]
    name = f"{method}{tag_suffix}_{cond}_k{k}"
    out = {}
    for lay in ("familiar", "new"):
        S = stack("pp_s3_roll", name, (C["train"],) + C["unseen"], lay)  # [3, 1 + u, 20]
        if np.isnan(S).all():
            out[lay] = None
            continue

        def rat(kk):
            st = np.nanmean(S[:, 0][:, kk])
            su = np.nanmean(S[:, 1:][:, :, kk])
            return su / st if st > 0 else None
        n = S.shape[-1]
        out[lay] = dict(s_train=float(np.nanmean(S[:, 0])), s_unseen=float(np.nanmean(S[:, 1:])), ratio=rat(np.arange(n)), ratio_ci=boot(rat, n),
                        per_unseen={c: float(np.nanmean(S[:, 1 + i])) for i, c in enumerate(C["unseen"])}, S=S)
    return out


def forgetting(method, cond, k):
    C = COND[cond]
    base = stack("pp_s4_old", f"{method}_{cond}_base", C["old"], "familiar")
    ad = stack("pp_s4_old", f"{method}_{cond}_k{k}", C["old"], "familiar")
    if np.isnan(base).all() or np.isnan(ad).all():
        return None
    n = base.shape[-1]
    d = lambda kk: float(np.nanmean(ad[:, :, kk]) - np.nanmean(base[:, :, kk]))     # noqa: E731
    return dict(base=float(np.nanmean(base)), adapted=float(np.nanmean(ad)), delta=d(np.arange(n)), ci=boot(d, n))


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    out = {}
    if only in (None, "s1"):
        out["S1"] = s1()
        print("S1", out["S1"]["holds"], flush=True)
    if only in (None, "s2"):
        out["S2"] = s2()
        print("S2", out["S2"]["holds"], {l: (out["S2"][l]["three_factor_holds"], out["S2"][l]["object_matched"]) for l in ("familiar", "new")}, flush=True)
    if only in (None, "s3"):
        S3, S4 = {}, {}
        for cond in COND:
            for method in METHODS + ("P_r32",):
                for k in KS:
                    sfx = "_r32" if method == "P_r32" else ""
                    r = s3_one("P" if method == "P_r32" else method, cond, k, sfx)
                    (S3 if method.startswith("P") else S4)[f"{method}|{cond}|k{k}"] = r
        # P minus each baseline on unseen partners, per k, cond, layout
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
                diffs[f"P-{m}|{cond}|{kk}|{lay}"] = [f(np.arange(n)), *boot(f, n)]
        forg = {f"{m}|{c}|k{k}": forgetting(m, c, k) for m in ("HF", "RF", "LIB") for c in COND for k in KS}
        go = {}
        for cond in COND:
            for lay in ("familiar", "new"):
                for k in (1, 5, 10):
                    r = S3.get(f"P|{cond}|k{k}", {}).get(lay)
                    go[f"{cond}|{lay}|k{k}"] = bool(r and r["ratio"] is not None and r["ratio"] >= 0.8 and r["ratio_ci"][0] is not None
                                                    and r["ratio_ci"][0] >= 0.6)
        strip = lambda d: {k: ({l: ({kk: vv for kk, vv in v.items() if kk != "S"} if v else None) for l, v in r.items()} if r else None)  # noqa: E731
                           for k, r in d.items()}
        out["S3"] = dict(cells=strip(S3), transfer_criterion=go, all_transfer=all(go.values()))
        out["S4"] = dict(cells=strip(S4), P_minus_baseline_unseen=diffs, forgetting=forg)
        print("S3 transfer", json.dumps(go), flush=True)
    prev = os.path.join(OUT, "pp_l2_read.json")
    old = json.load(open(prev, encoding="utf-8")) if os.path.exists(prev) else {}
    old.update(out)
    if all(x in old for x in ("S1", "S2", "S3")):
        old["layer3_go"] = bool(old["S1"]["holds"] and old["S2"]["holds"] and old["S3"]["all_transfer"])
        print("LAYER3_GO", old["layer3_go"])
    json.dump(old, open(prev, "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()

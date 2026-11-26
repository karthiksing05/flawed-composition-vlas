"""XB read: excess failure E, gates, x1-x5, p1, p2, Holm, failure-class shares exactly as registered in experiments/prereg/xb_pred.txt.  CPU, numpy + scipy.
Writes results/bench/xb/read_main.json (refuses to overwrite).  k2: if any expected (model, cell) is incomplete the script writes read_INCOMPLETE.json instead (no read).

Operational choices (logged in docs/notes/xb_status.md):  cell failure = 1 - mean success;  E = mean failure over held-out cells - mean failure over partner cells;
bootstrap = cells resampled with replacement within cell type, then episodes within each cell, B = 2000, np.random.default_rng(0) per interval, 95% percentile;
one-sided p = share of bootstrap draws on the wrong side of the null; contrasts that need k of n models use the k-th smallest model p (intersection-union);
x2 per-model TOST p = max(P(E <= -.10), P(E >= +.10)), equivalent iff TOST p < .05;  Holm over {x1, x2, x3, x5, p1}, survives iff adjusted p < .025.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
XB = ROOT + "/results/bench/xb"
B = 2000
BENCH_MODELS = {"mesa": ["pi0", "pi05", "pi0fast", "groot"], "bimesa": ["pi0", "pi05", "pi0fast", "groot"], "robocasa": ["pi0", "pi05", "groot15", "xvla"], "robotwin": ["xvla", "pi05", "smolvla"]}
PUBLISHED_ID = {("mesa", "pi05"): .628, ("mesa", "groot"): .250}      # MESA paper Table 2 (MESA-ID), model cards; others not fetched -> k1 not evaluated


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def expected_cells(bench):
    if bench in ("mesa", "bimesa"):
        c = json.load(open(f"{XB}/cells_{bench}.json"))
        return {"heldout": {t: dict(n=50, op=c["heldout_info"][t]["object_partner_count"], rp=c["heldout_info"][t]["receptacle_partner_count"]) for t in c["heldout_tasks"]},
                "partner": {t: dict(n=50, op=c["partner_info"][t]["object_partner_count"], rp=c["partner_info"][t]["receptacle_partner_count"]) for t in c["partner_tasks"]}}
    if bench == "robocasa":
        c = json.load(open(f"{XB}/cells_robocasa.json"))["families"]; f = json.load(open(f"{XB}/cells_robocasa_final.json"))["families"]
        out = {"heldout": {}, "partner": {}}
        for fam, d in f.items():
            for ct in ("heldout", "partner"):
                for o in d[ct]["cells"]:
                    cid = f"{fam.replace('->', '_to_')}__{o.replace(' ', '_')}"
                    op = len(c[fam]["heldout_object_partners"].get(o, [])) if ct == "heldout" else None
                    out[ct][cid] = dict(n=20, op=op if op is not None else 1, rp=c[fam]["family_n_objects"])
        return out
    c = json.load(open(f"{XB}/cells_robotwin.json"))["tasks"]; f = json.load(open(f"{XB}/cells_robotwin_final.json"))["tasks"]
    out = {"heldout": {}, "partner": {}}
    for task, d in f.items():
        for ct in ("heldout", "partner"):
            for A, Bb in d[ct]:
                cid = f"{task}/{A.replace('/', '-')}__{Bb.replace('/', '-')}"
                out[ct][cid] = dict(n=10, op=c[task]["A_partners"].get(A, 1), rp=c[task]["B_partners"].get(Bb, 1))
    return out


def load(bench):
    exp = expected_cells(bench); data, missing = {}, []
    for m in BENCH_MODELS[bench]:
        data[m] = {"heldout": {}, "partner": {}}
        any_file = bool(glob.glob(f"{XB}/roll/{bench}/{m}/*/ep*.json")) or bool(glob.glob(f"{XB}/roll/{bench}/{m}/*/*/ep*.json"))
        if not any_file:
            data.pop(m); continue                                            # model never rolled out (excluded, reported)
        for ct, cells in exp.items():
            for cid, info in cells.items():
                fs = sorted(glob.glob(f"{XB}/roll/{bench}/{m}/{cid}/ep*.json")) if bench != "robotwin" else sorted(glob.glob(f"{XB}/roll/robotwin/{m}/{cid}/ep*.json"))
                succ = [bool(json.load(open(f))["success"]) for f in fs]
                if len(succ) < info["n"]:
                    missing.append((bench, m, ct, cid, len(succ), info["n"]))
                data[m][ct][cid] = dict(s=np.array(succ, float), op=info["op"], rp=info["rp"])
    return data, missing


def cell_means(cells):
    return np.array([c["s"].mean() if len(c["s"]) else np.nan for c in cells.values()])


def boot_E(ho, pa, rng):
    Hs, Ps = list(ho.values()), list(pa.values())
    out = np.empty(B)
    for b in range(B):
        def draw(cl):
            idx = rng.integers(0, len(cl), len(cl))
            return np.mean([1 - rng.choice(cl[i]["s"], size=len(cl[i]["s"]), replace=True).mean() for i in idx])
        out[b] = draw(Hs) - draw(Ps)
    return out


def E_point(ho, pa):
    return float(np.mean([1 - c["s"].mean() for c in ho.values()]) - np.mean([1 - c["s"].mean() for c in pa.values()]))


def kth(ps, k):
    ps = sorted(ps)
    return ps[k - 1] if len(ps) >= k else 1.0


def holm(named):
    items = sorted(named.items(), key=lambda t: t[1]); m = len(items); res, prev = {}, 0.0
    for i, (n, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p)); prev = adj; res[n] = dict(p=p, adj=adj, survives=adj < .025)
    return res


def main():
    out = f"{XB}/read_main.json"
    assert not os.path.exists(out), "never overwrite"
    R = {"bench": {}, "missing": []}; D = {}
    for bench in BENCH_MODELS:
        data, missing = load(bench)
        D[bench] = data; R["missing"] += missing
    if R["missing"]:
        json.dump(R, open(f"{XB}/read_INCOMPLETE.json", "w"), indent=1)
        print("k2: incomplete cells:", len(R["missing"]), "-> read_INCOMPLETE.json; no read"); return
    rng = np.random.default_rng(0)
    Ep, pv = {}, {}
    for bench, data in D.items():
        R["bench"][bench] = {}
        for m, d in data.items():
            ho, pa = d["heldout"], d["partner"]
            base = float(np.mean([c["s"].mean() for c in pa.values()])); fail_ho = float(np.mean([1 - c["s"].mean() for c in ho.values()]))
            bs = boot_E(ho, pa, np.random.default_rng(0)); e = E_point(ho, pa)
            testable = base >= .30
            r = dict(partner_success=base, heldout_success=1 - fail_ho, E=e, E_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], testable=testable,
                     p_E_gt0=float(np.mean(bs <= 0)), tost_p=float(max(np.mean(bs <= -.10), np.mean(bs >= .10))), n_heldout_cells=len(ho), n_partner_cells=len(pa),
                     published_id=PUBLISHED_ID.get((bench, m)), k1_flag=None if (bench, m) not in PUBLISHED_ID else bool(abs(base - PUBLISHED_ID[(bench, m)]) > .15))
            R["bench"][bench][m] = r
    def sel(bench, key, test=True):
        return [(m, r) for m, r in R["bench"].get(bench, {}).items() if r["testable"] or not test]
    # contrasts
    C = {}
    def count_contrast(bench, cond, k, nmax, pname):
        t = sel(bench, None); ok = [m for m, r in t if cond(r)]
        return dict(testable_models=[m for m, _ in t], n_testable=len(t), satisfied=ok, k_needed=k, supported=len(ok) >= k and len(t) >= k,
                    p=kth([pname(r) for _, r in t], k) if len(t) >= k else 1.0)
    C["x1"] = count_contrast("mesa", lambda r: r["E_ci"][0] > 0, 3, 4, lambda r: r["p_E_gt0"])
    C["x2"] = count_contrast("robotwin", lambda r: r["tost_p"] < .05, 2, 3, lambda r: r["tost_p"])
    C["x3"] = count_contrast("robocasa", lambda r: r["E_ci"][0] > 0, 2, 4, lambda r: r["p_E_gt0"])
    C["x4"] = count_contrast("bimesa", lambda r: r["E_ci"][0] > 0, 3, 4, lambda r: r["p_E_gt0"])
    # x5: partner-count law over all held-out cells of all benchmarks / testable models
    rows = []
    for bench, data in D.items():
        for m, d in data.items():
            if not R["bench"][bench][m]["testable"]:
                continue
            pf = 1 - float(np.mean([c["s"].mean() for c in d["partner"].values()]))
            for cid, c in d["heldout"].items():
                rows.append((bench, m, cid, (1 - c["s"].mean()) - pf, np.log(max(1, min(c["op"], c["rp"]))), 1 - c["s"].mean(), 0))
    if len(rows) > 5:
        y = np.array([r[3] for r in rows]); x = np.array([r[4] for r in rows])
        rho = float(spearmanr(x, y).statistic)
        keys = sorted({(r[0], r[2]) for r in rows}); rb = []
        r2 = np.random.default_rng(0)
        for b in range(B):
            pick = [keys[i] for i in r2.integers(0, len(keys), len(keys))]
            cnt = {}
            for k in pick:
                cnt[k] = cnt.get(k, 0) + 1
            ii = [i for i, r in enumerate(rows) for _ in range(cnt.get((r[0], r[2]), 0))]
            if len(set(x[ii])) > 1:
                rb.append(spearmanr(x[ii], y[ii]).statistic)
        C["x5"] = dict(rho=rho, ci=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))], n_rows=len(rows), supported=bool(np.percentile(rb, 97.5) < 0), p=float(np.mean(np.array(rb) >= 0)))
    else:
        C["x5"] = dict(supported=False, p=1.0, note="too few rows")
    # p1 / p2: AUC of cell failure on held-out cells
    def auc_rows(bench):
        rs = []
        for m, d in D[bench].items():
            if not R["bench"][bench][m]["testable"]:
                continue
            pr = json.load(open(f"{XB}/pred_{bench}.json"))["models"].get(m, {}).get("cells", {})
            for cid, c in d["heldout"].items():
                if cid in pr and pr[cid]["v1"] is not None:
                    rs.append(dict(cell=cid, y=int(c["s"].mean() < .5), v1=pr[cid]["v1"], v0=pr[cid]["v0"], b2=pr[cid]["b2"]))
        return rs
    p1_b, p2_rows = {}, []
    for bench in ("mesa", "robocasa", "robotwin"):
        if not os.path.exists(f"{XB}/pred_{bench}.json"):
            continue
        rs = auc_rows(bench); p2_rows += [dict(r, bench=bench) for r in rs]
        if len(rs) < 6:
            continue
        y = np.array([r["y"] for r in rs]); cells = sorted({r["cell"] for r in rs}); r3 = np.random.default_rng(0); diff = []
        for b in range(B):
            pick = r3.integers(0, len(cells), len(cells)); cnt = {}
            for i in pick:
                cnt[cells[i]] = cnt.get(cells[i], 0) + 1
            ii = [i for i, r in enumerate(rs) for _ in range(cnt.get(r["cell"], 0))]
            yy = y[ii]
            a1, a2 = auc(yy, [rs[i]["v1"] for i in ii]), auc(yy, [-rs[i]["b2"] for i in ii])
            if not (np.isnan(a1) or np.isnan(a2)):
                diff.append(a1 - a2)
        p1_b[bench] = dict(auc_v1=auc(y, [r["v1"] for r in rs]), auc_b2=auc(y, [-r["b2"] for r in rs]), diff=float(np.mean(diff)) if diff else None,
                           ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))] if diff else None, p=float(np.mean(np.array(diff) <= 0)) if diff else 1.0, n=len(rs))
    C["p1"] = dict(per_bench=p1_b, supported=sum(1 for v in p1_b.values() if v["ci"] and v["ci"][0] > 0) >= 2, p=kth([v["p"] for v in p1_b.values()], 2) if len(p1_b) >= 2 else 1.0)
    if p2_rows:
        y = np.array([r["y"] for r in p2_rows]); s = np.array([r["v0"] for r in p2_rows]); cells = sorted({(r["bench"], r["cell"]) for r in p2_rows}); r4 = np.random.default_rng(0); ab = []
        for b in range(B):
            pick = r4.integers(0, len(cells), len(cells)); cnt = {}
            for i in pick:
                cnt[cells[i]] = cnt.get(cells[i], 0) + 1
            ii = [i for i, r in enumerate(p2_rows) for _ in range(cnt.get((r["bench"], r["cell"]), 0))]
            a = auc(y[ii], s[ii])
            if not np.isnan(a):
                ab.append(a)
        C["p2"] = dict(auc=auc(y, s), ci=[float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))], supported=bool(np.percentile(ab, 2.5) > .5), n=len(p2_rows),
                       note="pooled over held-out cells of all benchmarks with predictions, testable models")
    names = {k: C[k]["p"] for k in ("x1", "x2", "x3", "x5", "p1") if k in C}
    H = holm(names)
    for k in H:
        C[k].update(holm=H[k])
    R["contrasts"] = C; R["prereg_sha256"] = __import__("hashlib").sha256(open(ROOT + "/experiments/prereg/xb_pred.txt", "rb").read()).hexdigest()
    with open(out, "x") as fh:
        json.dump(R, fh, indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("supported", "p", "holm", "rho", "ci", "satisfied", "auc")} for k, v in C.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()

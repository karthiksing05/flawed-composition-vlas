"""Failure predictor development fit (docs/PLAN_GOAL3.md pillar B).  CPU, numpy + scipy; run through sbatch.
Reads results/flawed/fp_dev/table.csv (experiments/fp_build.py).  Predictors: v0 cue-vote rule (no fitting), v1 L2 logistic regression on
standardized features (+ multinomial class head), b1 novelty only, b2 known-partner base rate only (+ b12 = b1+b2 as a stronger reference).
CV: leave-one-model-out and leave-one-scene-out; bootstrap over state clusters (B=2000).  Writes fit.json, v1_frozen.json, v1_frozen.sha256 (never overwrites)."""
import csv, hashlib, json, os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed", "fp_dev")
CLASSES = ["S", "G", "W", "O", "H", "M"]
LAM, B = 10.0, 2000
FEATS = ["f_novel", "f_obj_targets", "f_tgt_holders", "f_tc_conflict", "f_nearer_conf", "f_near_margin", "f_instr_rank", "f_sal_conf", "f_sal_missing",
         "f_lang", "f_occl", "f_base_rate"]
SETS = {"v1": FEATS, "b1": ["f_novel"], "b2": ["f_base_rate"], "b12": ["f_novel", "f_base_rate"]}
rng = np.random.default_rng(0)


def load():
    rows = list(csv.DictReader(open(os.path.join(D, "table.csv"), encoding="utf-8")))
    n = len(rows)
    X = np.zeros((n, len(FEATS)))
    miss = {}
    for j, f in enumerate(FEATS):
        for i, r in enumerate(rows):
            if f == "f_sal_missing":
                X[i, j] = float(r["f_sal_conf"] == "")
            elif r[f] == "":
                X[i, j] = np.nan
            else:
                X[i, j] = float(r[f])
        miss[f] = float(np.mean(np.isnan(X[:, j]))) if f != "f_sal_missing" else float(X[:, j].mean())
    X[:, FEATS.index("f_obj_targets")] = np.log(X[:, FEATS.index("f_obj_targets")])
    X[:, FEATS.index("f_tgt_holders")] = np.log(X[:, FEATS.index("f_tgt_holders")])
    meta = dict(model=np.array([r["model"] for r in rows]), scene=np.array([int(r["scene"]) for r in rows]), cluster=np.array([r["cluster"] for r in rows]),
                set=np.array([r["set"] for r in rows]), task=np.array([r["task"] for r in rows]))
    y = 1 - np.array([int(r["success"]) for r in rows])
    cls = np.array([r["cls"] for r in rows])
    v0s = np.array([float(r["v0_score"]) for r in rows])
    v0c = np.array([r["v0_cls"] for r in rows])
    return rows, X, y, cls, v0s, v0c, meta, miss


def prep(Xtr, Xte, cols):
    A, T = Xtr[:, cols].copy(), Xte[:, cols].copy()
    mu = np.nanmean(A, 0)
    mu = np.where(np.isnan(mu), 0, mu)
    for M in (A, T):
        idx = np.where(np.isnan(M))
        M[idx] = mu[idx[1]]
    m, s = A.mean(0), A.std(0)
    s = np.where(s < 1e-9, 1.0, s)
    return (A - m) / s, (T - m) / s, dict(impute=mu.tolist(), mean=m.tolist(), std=s.tolist())


def fit_bin(A, y, lam):
    n, p = A.shape
    def f(w):
        z = A @ w[:p] + w[p]
        l = np.logaddexp(0, z) - y * z
        pr = 1 / (1 + np.exp(-z))
        g = A.T @ (pr - y) + lam * w[:p]
        return l.sum() + 0.5 * lam * (w[:p] ** 2).sum(), np.append(g, (pr - y).sum())
    r = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B")
    return r.x


def fit_multi(A, c, lam):
    n, p = A.shape
    K = len(CLASSES)
    Y = np.zeros((n, K))
    for k, name in enumerate(CLASSES):
        Y[:, k] = (c == name)
    def f(w):
        W = w[: p * K].reshape(p, K); b = w[p * K:]
        Z = A @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        l = -(Y * np.log(P + 1e-12)).sum() + 0.5 * lam * (W ** 2).sum()
        G = P - Y
        return l, np.concatenate([(A.T @ G + lam * W).ravel(), G.sum(0)])
    r = minimize(f, np.zeros(p * K + K), jac=True, method="L-BFGS-B")
    return r.x[: p * K].reshape(p, K), r.x[p * K:]


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def predict_set(name, Xtr, ytr, ctr, Xte, lam=LAM, cols=None):
    cols = cols if cols is not None else [FEATS.index(f) for f in SETS[name]]
    A, T, st = prep(Xtr, Xte, cols)
    w = fit_bin(A, ytr, lam)
    p = 1 / (1 + np.exp(-(T @ w[:-1] + w[-1])))
    W, b = fit_multi(A, ctr, lam)
    Z = T @ W + b
    return p, np.array(CLASSES)[Z.argmax(1)], dict(w=w, W=W, b=b, st=st, cols=cols)


def cv_oof(X, y, cls, v0s, v0c, groups, lam=LAM, which=("v1", "b1", "b2", "b12")):
    """out-of-fold predictions; groups = list of (train_idx, test_idx)."""
    n = len(y)
    out = {k: dict(p=np.full(n, np.nan), c=np.array([""] * n, dtype=object)) for k in which}
    out["v0"] = dict(p=np.full(n, np.nan), c=np.array([""] * n, dtype=object))
    fold = np.full(n, -1)
    for gi, (tr, te) in enumerate(groups):
        fold[te] = gi
        for k in which:
            p, c, _ = predict_set(k, X[tr], y[tr], cls[tr], X[te], lam)
            out[k]["p"][te], out[k]["c"][te] = p, c
        out["v0"]["p"][te], out["v0"]["c"][te] = v0s[te], v0c[te]
    return out, fold


def macro(y, p, fold, ii):
    v = [auc(y[ii][fold[ii] == f], p[ii][fold[ii] == f]) for f in np.unique(fold[ii])]
    v = [a for a in v if a == a]
    return float(np.mean(v)) if v else np.nan


def metrics(out, y, cls, idx, fold=None):
    res = {}
    fail = y[idx] == 1
    for k, o in out.items():
        pc = o["c"][idx]
        # failure-conditional class prediction: the best non-S class (v0 can only say W)
        res[k] = dict(auc=auc(y[idx], o["p"][idx]), macro_auc=macro(y, o["p"], fold, idx) if fold is not None else None, top1=float(np.mean(pc == cls[idx])),
                      top1_fail=float(np.mean((np.where(pc == "S", "W", pc))[fail] == cls[idx][fail])) if fail.any() else np.nan,
                      pfail_acc=float(np.mean((o["p"][idx] > 0.5) == (y[idx] == 1))))
    return res


def boot(out, y, cls, idx, clusters, pairs, fold):
    uc, inv = np.unique(clusters[idx], return_inverse=True)
    members = [np.nonzero(inv == i)[0] for i in range(len(uc))]
    ids = idx
    acc = {k: [] for k in out}
    mac = {k: [] for k in out}
    mdif = {"%s-%s" % p: [] for p in pairs}
    t1 = {k: [] for k in out}
    dif = {"%s-%s" % p: [] for p in pairs}
    for _ in range(B):
        draw = rng.integers(0, len(uc), len(uc))
        sel = np.concatenate([members[i] for i in draw])
        ii = ids[sel]
        a = {k: auc(y[ii], o["p"][ii]) for k, o in out.items()}
        ma = {k: macro(y, o["p"], fold, ii) for k, o in out.items()}
        for k in out:
            mac[k].append(ma[k])
        for p in pairs:
            mdif["%s-%s" % p].append(ma[p[0]] - ma[p[1]])
        for k, o in out.items():
            acc[k].append(a[k]); t1[k].append(float(np.mean(o["c"][ii] == cls[ii])))
        for p in pairs:
            dif["%s-%s" % p].append(a[p[0]] - a[p[1]])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return dict(auc_ci={k: ci(v) for k, v in acc.items()}, top1_ci={k: ci(v) for k, v in t1.items()},
                macro_auc_ci={k: ci(v) for k, v in mac.items()}, macro_auc_diff_ci={k: ci(v) for k, v in mdif.items()}, auc_diff_ci={k: ci(v) for k, v in dif.items()}, auc_diff_p_le0={k: float(np.mean(np.array(v) <= 0)) for k, v in dif.items()})


def main():
    rows, X, y, cls, v0s, v0c, meta, miss = load()
    n = len(y)
    res = dict(n_rows=n, features=FEATS, lam=LAM, bootstrap_B=B, missing_share=miss, base_rates=dict(
        fail_rate=float(y.mean()), by_scene={int(s): float(y[meta["scene"] == s].mean()) for s in (1, 2)},
        n_by_model_scene={"%s/s%d" % (m, s): int(((meta["model"] == m) & (meta["scene"] == s)).sum()) for m in sorted(set(meta["model"])) for s in (1, 2)
                          if ((meta["model"] == m) & (meta["scene"] == s)).any()},
        class_counts={c: int((cls == c).sum()) for c in CLASSES}))
    res["fail_rate_known_vs_new"] = {int(s): {t: float(y[(meta["scene"] == s) & (np.array([x.startswith(t) for x in meta["task"]]))].mean()) for t in ("K", "N")} for s in (1, 2)}
    schemes = {}
    models = sorted(set(meta["model"]))
    schemes["LOMO"] = [(np.nonzero(meta["model"] != m)[0], np.nonzero(meta["model"] == m)[0]) for m in models]
    schemes["LOSO"] = [(np.nonzero(meta["scene"] != s)[0], np.nonzero(meta["scene"] == s)[0]) for s in (1, 2)]
    pairs = [("v1", "b2"), ("v1", "b1"), ("v1", "v0"), ("v1", "b12"), ("v0", "b2")]
    allidx = np.arange(n)
    for sname, groups in schemes.items():
        out, fold = cv_oof(X, y, cls, v0s, v0c, groups)
        subs = {"all": allidx, "scene1": allidx[meta["scene"] == 1], "scene2": allidx[meta["scene"] == 2],
                "new_only": allidx[np.array([t.startswith("N") for t in meta["task"]])]}
        r = dict(subsets={})
        for nm, idx in subs.items():
            m = metrics(out, y, cls, idx, fold)
            m["n"] = int(len(idx)); m["fail_rate"] = float(y[idx].mean())
            m["boot"] = boot(out, y, cls, idx, meta["cluster"], pairs, fold) if nm in ("all", "new_only") else None
            r["subsets"][nm] = m
        r["per_fold"] = {}
        for gi, (tr, te) in enumerate(groups):
            label = models[gi] if sname == "LOMO" else "test_scene%d" % (gi + 1)
            r["per_fold"][label] = {k: v for k, v in metrics(out, y, cls, te).items()}
            r["per_fold"][label]["n"] = int(len(te)); r["per_fold"][label]["fail_rate"] = float(y[te].mean())
        # drop-one-feature and lambda sensitivity (pooled AUC on all)
        abl = {}
        for f in FEATS:
            cols = [FEATS.index(g) for g in FEATS if g != f]
            o = {"v1": dict(p=np.full(n, np.nan), c=np.array([""] * n, dtype=object))}
            for tr, te in groups:
                p, c, _ = predict_set("v1", X[tr], y[tr], cls[tr], X[te], LAM, cols)
                o["v1"]["p"][te], o["v1"]["c"][te] = p, c
            abl[f] = dict(auc_all=auc(y, o["v1"]["p"]), auc_new_only=auc(y[subs["new_only"]], o["v1"]["p"][subs["new_only"]]),
                          top1=float(np.mean(o["v1"]["c"] == cls)))
        r["drop_one_v1"] = abl
        only = {}
        for f in FEATS:
            cols = [FEATS.index(f)]
            p_ = np.full(n, np.nan)
            for tr, te in groups:
                p, c, _ = predict_set("v1", X[tr], y[tr], cls[tr], X[te], LAM, cols)
                p_[te] = p
            only[f] = dict(auc_all=auc(y, p_), auc_new_only=auc(y[subs["new_only"]], p_[subs["new_only"]]))
        r["single_feature_v1"] = only
        r["lambda_sensitivity_auc_all"] = {}
        for lam in (1.0, 10.0, 100.0):
            o2, _ = cv_oof(X, y, cls, v0s, v0c, groups, lam, which=("v1",))
            r["lambda_sensitivity_auc_all"][str(lam)] = auc(y, o2["v1"]["p"])
        schemes[sname] = r
    res["cv"] = {k: v for k, v in schemes.items() if isinstance(v, dict)}
    # raw univariate feature AUC (no fitting): sign-free, all and new-only (leakage screen)
    uni = {}
    nw = np.array([t.startswith("N") for t in meta["task"]])
    for j, f in enumerate(FEATS):
        col = np.nan_to_num(X[:, j], nan=np.nanmean(X[:, j]) if not np.all(np.isnan(X[:, j])) else 0.0)
        a1, a2 = auc(y, col), auc(y[nw], col[nw])
        uni[f] = dict(auc_all=a1, auc_new=a2, corr_with_fail=float(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else None)
    res["univariate_feature_auc"] = uni
    # frozen v1: fit on every row, then freeze
    p, c, st = predict_set("v1", X, y, cls, X, LAM)
    res["v1_in_sample_auc"] = auc(y, p)
    frozen = dict(spec="docs/PLAN_GOAL3.md pillar B predictor v1; logistic regression L2 (sum-scale lambda=%g, intercept unpenalised) on standardized features; fit on all LIBERO atlas rows"
                       " (both scenes, five VLAs); prediction = sigmoid(intercept + sum_j coef_j * (x_j_imputed - mean_j) / std_j); class head = softmax(b + W^T z)" % LAM,
                  features=FEATS, feature_transform={"f_obj_targets": "log", "f_tgt_holders": "log", "f_sal_missing": "1 iff salience unavailable (f_sal_conf blank; imputed to the training mean)"},
                  impute=st["st"]["impute"] if "st" in st else st["impute"], mean=st["mean"] if "mean" in st else None, std=None)
    stt = st["st"]
    frozen.update(impute=stt["impute"], mean=stt["mean"], std=stt["std"], coef=dict(zip(FEATS, map(float, st["w"][:-1]))), intercept=float(st["w"][-1]),
                  class_order=CLASSES, class_W={f: dict(zip(CLASSES, map(float, st["W"][j]))) for j, f in enumerate(FEATS)}, class_b=dict(zip(CLASSES, map(float, st["b"]))),
                  n_rows=n, table_sha256=hashlib.sha256(open(os.path.join(D, "table.csv"), "rb").read()).hexdigest())
    fp = os.path.join(D, "v1_frozen.json")
    with open(fp, "x", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=1)
    sha = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    with open(os.path.join(D, "v1_frozen.sha256"), "x", encoding="utf-8") as fh:
        fh.write("%s  v1_frozen.json\n" % sha)
    res["v1_frozen_sha256"] = sha
    res["v1_coef"] = frozen["coef"]
    with open(os.path.join(D, "fit.json"), "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: None if o != o else str(o))
    # console summary
    for sname, r in res["cv"].items():
        for nm in ("all", "scene1", "scene2", "new_only"):
            m = r["subsets"][nm]
            print(sname, nm, "n=%d fail=%.3f" % (m["n"], m["fail_rate"]), {k: (round(m[k]["auc"], 3), round(m[k]["macro_auc"], 3), round(m[k]["top1"], 3)) for k in ("v0", "v1", "b1", "b2", "b12")}, flush=True)
            if m["boot"]:
                print("    auc CI", {k: [round(x, 3) for x in v] for k, v in m["boot"]["auc_ci"].items()}, flush=True)
                print("    macro CI", {k: [round(x, 3) for x in v] for k, v in m["boot"]["macro_auc_ci"].items()}, flush=True)
                print("    macro diff CI", {k: [round(x, 3) for x in v] for k, v in m["boot"]["macro_auc_diff_ci"].items()}, flush=True)
                print("    diff CI", {k: [round(x, 3) for x in v] for k, v in m["boot"]["auc_diff_ci"].items()}, flush=True)
    print("sha256", sha); print("coef", {k: round(v, 3) for k, v in res["v1_coef"].items()})


if __name__ == "__main__":
    main()

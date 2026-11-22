"""ORC read (docs/notes/plan_oracle_pi0.md 2.4).  CPU (numpy only).  Writes with open(..., "x"): refuses to overwrite.

  read      per model: phi (ratio of sums, clustered bootstrap over layouts, B = 2000, rng 0), c_clean, phi_zero / c_zero, sigma_demo, phi_net / phi_total for servo models,
            label LEARNED (CI lower >= .70) / NOT LEARNED (CI upper < .30) / PARTIAL; UNTESTABLE if fewer than 120 observations or sum den <= 0.
  contrast  cluster-paired differences of two read files on the same observations (phi and c_clean).
  selftest  synthetic chunks: a perfect follower gives 1, an ignorer 0, a half rotation .5.

For observation i (clean d5 = the sum of the first 5 unnormalised chunk rows, xy):  num_i = d5(cf).u_j - d5(clean).u_j;  den_i = |d5(clean)| - d5(clean).u_j.

    python experiments/orc_read.py read --label A+0 --obs DIR --probe DIR --out DIR
    python experiments/orc_read.py contrast --a A+0 --b A0 --dir DIR
    python experiments/orc_read.py selftest
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

B_BOOT, MIN_OBS = 2000, 120


def d5(ch):
    return np.asarray(ch, np.float64)[..., :5, :2].sum(-2)


def core(d5c, d5x, uj):
    """-> num_i, den_i (arrays over observations)."""
    return (d5x * uj).sum(-1) - (d5c * uj).sum(-1), np.linalg.norm(d5c, axis=-1) - (d5c * uj).sum(-1)


def boot_ratio(num, den, cl, B=B_BOOT, seed=0):
    """ratio of sums, clustered bootstrap.  cl: cluster ids per observation."""
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    sn, sd = np.array([num[idx[c]].sum() for c in u]), np.array([den[idx[c]].sum() for c in u])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(u), size=(B, len(u)))
    r = sn[draws].sum(1) / np.where(sd[draws].sum(1) == 0, np.nan, sd[draws].sum(1))
    return float(num.sum() / den.sum()) if den.sum() != 0 else float("nan"), [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]


def boot_mean(x, cl, B=B_BOOT, seed=0):
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    sx, n = np.array([x[idx[c]].sum() for c in u]), np.array([len(idx[c]) for c in u])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(u), size=(B, len(u)))
    r = sx[draws].sum(1) / n[draws].sum(1)
    return float(x.mean()), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def label(ci, n, sden):
    if n < MIN_OBS or not sden > 0:
        return "UNTESTABLE"
    return "LEARNED" if ci[0] >= 0.70 else "NOT LEARNED" if ci[1] < 0.30 else "PARTIAL"


def phi_block(d5c, d5x, uj, cl):
    num, den = core(d5c, d5x, uj)
    p, ci = boot_ratio(num, den, cl)
    return dict(phi=p, ci=ci, sum_num=float(num.sum()), sum_den=float(den.sum()), label=label(ci, len(num), den.sum())), num, den


def read(a):
    sel = json.load(open(os.path.join(a.obs, "selection.json"), encoding="utf-8"))
    ids = sel["ids"]
    D = dict(c=[], x=[], z=[], t=[], ui=[], uj=[], cl=[], dd=[])
    servo = None
    for oid in ids:
        z = np.load(os.path.join(a.obs, oid + ".npz"), allow_pickle=False)
        p = np.load(os.path.join(a.probe, oid + ".npz"), allow_pickle=False)
        names = list(p["names"])
        ch = p["chunks"]
        servo = "chan_cf" in names
        D["c"].append(d5(ch[names.index("ref")]))
        D["x"].append(d5(ch[names.index("tok_cf")]))
        D["z"].append(d5(ch[names.index("abl.tok")]))
        D["t"].append(d5(ch[names.index("chan_cf")]) if servo else np.zeros(2))
        D["ui"].append(z["u_i"][:2].astype(np.float64))
        D["uj"].append(z["u_j"][:2].astype(np.float64))
        D["cl"].append(json.loads(str(z["meta"]))["layout"])
        D["dd"].append(float(np.dot(z["d5_first_j"], z["u_j"][:2]) - np.dot(z["d5_first_i"], z["u_j"][:2])))
    A = {k: (np.array(v) if k != "cl" else np.array(v)) for k, v in D.items()}
    cl = A["cl"]
    out = dict(label=a.label, n=len(ids), n_clusters=int(len(np.unique(cl))), servo=bool(servo), B=B_BOOT, rng=0)
    blk, num, den = phi_block(A["c"], A["x"], A["uj"], cl)
    out["phi_net" if servo else "phi"] = blk
    out["phi_registered"] = blk
    cc = (A["c"] * A["ui"]).sum(-1) / np.maximum(np.linalg.norm(A["c"], axis=-1), 1e-9)
    cz = (A["z"] * A["ui"]).sum(-1) / np.maximum(np.linalg.norm(A["z"], axis=-1), 1e-9)
    m, ci = boot_mean(cc, cl)
    out["c_clean"] = dict(mean=m, ci=ci)
    m, ci = boot_mean(cz, cl)
    out["c_zero"] = dict(mean=m, ci=ci)
    out["phi_zero"] = phi_block(A["c"], A["z"], A["uj"], cl)[0]
    sd, ci = boot_ratio(num, np.where(True, A["dd"], 0.0), cl)
    out["sigma_demo"] = dict(sigma=sd, ci=ci, sum_demo_diff=float(A["dd"].sum()))
    if servo:
        out["phi_total"] = phi_block(A["c"], A["t"], A["uj"], cl)[0]
    out["per_obs"] = dict(id=ids, cluster=cl.tolist(), num=num.tolist(), den=den.tolist(), c_clean=cc.tolist())
    p = os.path.join(a.out, "orc_read_%s.json" % a.label)
    os.makedirs(a.out, exist_ok=True)
    with open(p, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("[orc_read] %s: n %d, clusters %d, phi%s %.3f [%.3f, %.3f] %s | c_clean %.3f | phi_zero %.3f | sigma_demo %.3f" % (
        a.label, out["n"], out["n_clusters"], "_net" if servo else "", blk["phi"], blk["ci"][0], blk["ci"][1], blk["label"], out["c_clean"]["mean"], out["phi_zero"]["phi"], sd), flush=True)


def contrast(a):
    ra = json.load(open(os.path.join(a.dir, "orc_read_%s.json" % a.a), encoding="utf-8"))
    rb = json.load(open(os.path.join(a.dir, "orc_read_%s.json" % a.b), encoding="utf-8"))
    pa, pb = ra["per_obs"], rb["per_obs"]
    assert pa["id"] == pb["id"], "different observation sets"
    cl = np.array(pa["cluster"])
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    rng = np.random.default_rng(0)
    draws = rng.integers(len(u), size=(B_BOOT, len(u)))
    res = {}
    na, da, nb, db = (np.array(x) for x in (pa["num"], pa["den"], pb["num"], pb["den"]))
    sn_a, sd_a = np.array([na[idx[c]].sum() for c in u]), np.array([da[idx[c]].sum() for c in u])
    sn_b, sd_b = np.array([nb[idx[c]].sum() for c in u]), np.array([db[idx[c]].sum() for c in u])
    r = sn_a[draws].sum(1) / sd_a[draws].sum(1) - sn_b[draws].sum(1) / sd_b[draws].sum(1)
    res["phi"] = dict(diff=float(na.sum() / da.sum() - nb.sum() / db.sum()), ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])
    ca, cb = np.array(pa["c_clean"]), np.array(pb["c_clean"])
    n = np.array([len(idx[c]) for c in u])
    sa, sb = np.array([ca[idx[c]].sum() for c in u]), np.array([cb[idx[c]].sum() for c in u])
    r = sa[draws].sum(1) / n[draws].sum(1) - sb[draws].sum(1) / n[draws].sum(1)
    res["c_clean"] = dict(diff=float(ca.mean() - cb.mean()), ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])
    p = os.path.join(a.dir, "orc_contrast_%s_%s.json" % (a.a, a.b))
    with open(p, "x", encoding="utf-8") as fh:
        json.dump(dict(a=a.a, b=a.b, n=len(ca), B=B_BOOT, **res), fh, indent=1)
    print("[orc_read] contrast %s - %s: phi %+.3f [%+.3f, %+.3f]; c_clean %+.3f [%+.3f, %+.3f]" % (
        a.a, a.b, res["phi"]["diff"], *res["phi"]["ci"], res["c_clean"]["diff"], *res["c_clean"]["ci"]), flush=True)


def selftest():
    rng = np.random.default_rng(1)
    n = 240
    th_i, th_j = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    ui, uj = np.stack([np.cos(th_i), np.sin(th_i)], 1), np.stack([np.cos(th_j), np.sin(th_j)], 1)
    sp = rng.uniform(1.0, 3.0, n)[:, None]
    c = sp * ui
    cl = np.array(["L%d" % (k // 6) for k in range(n)])
    out = {}
    for nm, x in (("perfect", sp * uj), ("ignorer", c), ("half", (c + sp * uj) / 2)):
        b, _, _ = phi_block(c, x, uj, cl)
        out[nm] = b["phi"]
    print("[orc_read] selftest phi:", {k: round(v, 4) for k, v in out.items()}, flush=True)
    assert abs(out["perfect"] - 1) < 1e-9 and abs(out["ignorer"]) < 1e-9 and abs(out["half"] - 0.5) < 1e-9, out
    print("[orc_read] selftest passed (1 / 0 / .5)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read", "contrast", "selftest"))
    ap.add_argument("--label")
    ap.add_argument("--obs")
    ap.add_argument("--probe")
    ap.add_argument("--out")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--dir")
    a = ap.parse_args()
    {"read": read, "contrast": contrast, "selftest": lambda a: selftest()}[a.mode](a)

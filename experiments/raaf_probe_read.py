"""RAA-F selection read, PRIMARY criterion (docs/notes/raaf_plan.md 6.1): the learned where-share on the pooled N + K sets.  CPU, openpi venv.

Uses a2_read.py UNCHANGED (load, table, boot_idx, ci, analyze).  Statistic (a2_read's effect measure and bootstrap: B = 2000, rng 0, clusters resampled within set):
    share(name) = mean_i E[i, name] / mean_i T[i]        (E = the effect of patching `name` from the counterfactual run; T = the effect of the counterfactual run itself)
    learned where-input:  in.tok for A / A0 / A+;   in.tokcrop (state token and crop image patched together, one row) for B.   The servo is not part of these models.
Pass rule, applied exactly (the plan, 6.1):
    PASS   iff  the where-share's 95% CI lower bound >= .70
           AND  the pooled N + K in.prefix share POINT is below RAA-sL's (recomputed here, same code, from the probe files of RAA-sL: --ref-dirs)
           AND  the gate T CI lower bound > 0.
    FAILED for the probe iff the where-share CI upper bound < .30;  otherwise NOT PASSED (CI straddles).
For B, in.prefix as defined by a2_probe.py contains the crop image (it is in the prefix); the literal rule uses it.  `pass_alt_nocrop` repeats the rule with in.prefix_nocrop
(the prefix with the crop image kept clean) and `sum_tok_crop` reports share(in.tok) + share(in.crop): both descriptive, not the registered rule.
Reported always: the share per set (N, K), the token-ablation drop (abl.tok, needs the full rows), the prefix share, top1 / PRn of the 72 writes (full rows).
Writes the JSON once (refuses to overwrite).

    python experiments/raaf_probe_read.py selftest
    python experiments/raaf_probe_read.py read --route A|B|Aplus|A0|raasl --dirs DIR[+DIR..] --ref-dirs DIR[+DIR..] --out PATH.json [--crop 0|1]
      (--route raasl --dirs X --ref-dirs X reads RAA-sL itself, e.g. the SR 83 reproduction; --sets N+K restricts the pooled sets)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import a2_read as AR                                                       # noqa: E402

PASS_LO, FAIL_HI = 0.70, 0.30
NAMES = ["in.tok", "in.crop", "in.tokcrop", "in.prefix", "in.prefix_nocrop", "in.servo", "in.chan"]


def select(obs, sets):
    return [o for o in obs if o["meta"]["set"] in sets]


def share_stats(obs, names, B=AR.B_BOOT):
    """-> {share name: {pooled, per set, CI}} plus T, for the pooled `obs` (point + paired bootstrap, rng 0)."""
    E, T = AR.table(obs, names)
    sets = np.array([o["meta"]["set"] for o in obs])
    col = {n: k for k, n in enumerate(names)}

    def calc(idx):
        r = {}
        for tag, m in [("ALL", np.ones(len(idx), bool))] + [(s, sets[idx] == s) for s in sorted(set(sets))]:
            ii = idx[m]
            if len(ii) == 0:
                continue
            Tm = float(np.nanmean(T[ii]))
            r[tag + ".T"] = Tm
            for n in names:
                with np.errstate(all="ignore"):
                    e = np.nanmean(E[ii][:, col[n]]) if np.isfinite(E[ii][:, col[n]]).any() else np.nan
                r["%s.%s" % (tag, n)] = float(e / Tm) if Tm != 0 and np.isfinite(e) else float("nan")
        return r
    pt = calc(np.arange(len(obs)))
    rng = np.random.default_rng(0)
    reps = [calc(AR.boot_idx(obs, rng)) for _ in range(B)]
    out = {}
    for k in pt:
        out[k] = dict(point=pt[k], ci=AR.ci([r.get(k, np.nan) for r in reps]))
    return out


def decide(route, st, ref, crop):
    """The registered rule on the share stats of the route (st) and of RAA-sL (ref)."""
    where = "in.tokcrop" if crop else "in.tok"
    w = st["ALL." + where]
    gate_lo = st["ALL.T"]["ci"][0]
    gate = bool(np.isfinite(gate_lo) and gate_lo > 0)
    p_route, p_ref = st["ALL.in.prefix"]["point"], ref["ALL.in.prefix"]["point"]
    pass_where = bool(np.isfinite(w["ci"][0]) and w["ci"][0] >= PASS_LO)
    below = bool(np.isfinite(p_route) and np.isfinite(p_ref) and p_route < p_ref)
    failed = bool(np.isfinite(w["ci"][1]) and w["ci"][1] < FAIL_HI)
    v = dict(route=route, where_input=where, where_share=w, gate_T_lower_gt_0=gate, T=st["ALL.T"],
             prefix_share_route=p_route, prefix_share_raasl=p_ref, prefix_below_raasl=below,
             )
    v["where_ci_lower_ge_0.70"] = pass_where
    v["where_ci_upper_lt_0.30"] = failed
    v["PASS"] = bool(pass_where and below and gate)
    v["verdict"] = "PASS" if v["PASS"] else ("FAILED (share CI upper < .30)" if failed else "NOT PASSED (CI straddles / prefix not below RAA-sL / gate)")
    if not gate:
        v["verdict"] += " [gate: T CI lower bound not > 0: UNTESTABLE]"
    if crop:
        pn = st["ALL.in.prefix_nocrop"]["point"]
        v["pass_alt_nocrop"] = bool(pass_where and np.isfinite(pn) and pn < p_ref and gate)
        v["prefix_nocrop_share_route"] = pn
        v["sum_tok_crop"] = st["ALL.in.tok"]["point"] + st["ALL.in.crop"]["point"]
    return v


def full_rows_report(obs):
    """top1 / PRn of the 72 writes and the token-ablation drop, when the probe ran with --rows full."""
    if "vlm.0.attn.all" not in obs[0]["names"]:
        return dict(available=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = AR.analyze("raasl", obs)
    s = doc["stats"]
    rep = dict(available=True, n_obs=doc["n_obs"], n_by_set=doc["n_by_set"])
    for k in ("ALL.top1", "ALL.PRn", "ALL.top3", "ALL.coverage"):
        rep[k] = s[k]
    for tag in ("N", "K"):
        for k in ("top1", "PRn"):
            if "%s.%s" % (tag, k) in s:
                rep["%s.%s" % (tag, k)] = s["%s.%s" % (tag, k)]
    rep["abl.tok_drop"] = {k: v["drop"] for k, v in doc["selectivity"].items() if k.startswith("abl.tok|")}
    rep["intact"] = {k: s[k] for k in s if k.startswith("intact.")} if any(k.startswith("intact.") for k in s) else None
    return rep


def run(route, dirs, ref_dirs, sets, crop):
    obs = select(AR.load(dirs), sets)
    ref = select(AR.load(ref_dirs), sets)
    assert obs and ref, "no observations (route %d, ref %d)" % (len(obs), len(ref))
    names = [n for n in NAMES if any(n in o["names"] for o in obs)]
    st = share_stats(obs, names)
    rst = share_stats(ref, [n for n in NAMES if any(n in o["names"] for o in ref)])
    v = decide(route, st, {k: dict(point=x["point"]) for k, x in rst.items()}, crop)
    return dict(route=route, crop=crop, sets=sorted(sets), dirs=dirs, ref_dirs=ref_dirs, n_obs=len(obs), n_ref=len(ref),
                n_by_set={s: sum(1 for o in obs if o["meta"]["set"] == s) for s in sorted(sets)}, B=AR.B_BOOT, rule=v,
                shares=st, ref_shares=rst, full_rows=full_rows_report(obs))


def show(doc):
    r, st = doc["rule"], doc["shares"]
    f = lambda d: "%.3f [%.3f, %.3f]" % (d["point"], d["ci"][0], d["ci"][1])
    print("[raaf-read] route %s n %d %s  ref n %d" % (doc["route"], doc["n_obs"], doc["n_by_set"], doc["n_ref"]))
    print("[raaf-read] T %s  where-input %s share %s" % (f(r["T"]), r["where_input"], f(r["where_share"])))
    for tag in ("N", "K"):
        for n in ("in.tok", "in.crop", "in.tokcrop", "in.prefix"):
            if "%s.%s" % (tag, n) in st and np.isfinite(st["%s.%s" % (tag, n)]["point"]):
                print("[raaf-read]   %s %-18s %s" % (tag, n, f(st["%s.%s" % (tag, n)])))
    print("[raaf-read] prefix share route %.3f  RAA-sL %.3f  below %s" % (r["prefix_share_route"], r["prefix_share_raasl"], r["prefix_below_raasl"]))
    print("[raaf-read] VERDICT %s" % r["verdict"])
    if "pass_alt_nocrop" in r:
        print("[raaf-read] B extras: pass_alt_nocrop %s  sum(in.tok + in.crop) %.3f  prefix_nocrop %.3f" % (r["pass_alt_nocrop"], r["sum_tok_crop"], r["prefix_nocrop_share_route"]))
    fr = doc["full_rows"]
    if fr.get("available"):
        print("[raaf-read] ALL top1 %s PRn %s ; abl.tok drop %s" % (f(fr["ALL.top1"]), f(fr["ALL.PRn"]), json.dumps(fr["abl.tok_drop"])))
    else:
        print("[raaf-read] full rows not present (top1 / PRn / token-ablation drop not computed)")


def read(route, dirs, ref_dirs, out, sets, crop):
    if os.path.exists(out):
        raise SystemExit("refusing to overwrite %s" % out)
    doc = run(route, dirs, ref_dirs, sets, crop)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:                           # 'x': never overwrites
        json.dump(doc, fh, indent=1, default=float)
    show(doc)


def _synth(share_where, share_prefix, crop, n=24, seed=0):
    rng = np.random.default_rng(seed)
    names = ["ref", "total", "in.tok", "in.crop", "in.tokcrop", "in.prefix", "in.prefix_nocrop"]
    obs = []
    for i in range(n):
        st = ("N", "K")[i % 2]
        scf = np.zeros(len(names))
        ix = {k: j for j, k in enumerate(names)}
        scf[ix["total"]] = 1.0 + 0.05 * rng.standard_normal()
        scf[ix["in.tok"]] = share_where * (0.6 if crop else 1.0)
        scf[ix["in.crop"]] = share_where * 0.4 if crop else 0.0
        scf[ix["in.tokcrop"]] = share_where
        scf[ix["in.prefix"]] = share_prefix
        scf[ix["in.prefix_nocrop"]] = share_prefix * 0.5
        obs.append(dict(meta=dict(set=st, cluster="%s_%d" % (st, i // 2)), names=names, scf=scf, scor=np.ones(len(names)), ref=np.zeros(len(names), int), d5=None))
    return obs


def selftest():
    ref = share_stats(_synth(-0.007, 0.13, False), ["in.tok", "in.prefix"], B=200)
    refp = {k: dict(point=v["point"]) for k, v in ref.items()}
    assert abs(ref["ALL.in.tok"]["point"] + 0.007) < 0.03
    nm = ["in.tok", "in.crop", "in.tokcrop", "in.prefix", "in.prefix_nocrop"]
    a = decide("A", share_stats(_synth(0.95, 0.05, False, seed=1), nm, B=200), refp, False)
    assert a["PASS"] and a["verdict"] == "PASS", a
    b = decide("A", share_stats(_synth(0.95, 0.40, False, seed=1), nm, B=200), refp, False)
    assert not b["PASS"] and not b["prefix_below_raasl"], b                 # where-share passes, prefix share not below RAA-sL
    c = decide("A", share_stats(_synth(0.05, 0.05, False, seed=2), nm, B=200), refp, False)
    assert c["verdict"].startswith("FAILED"), c
    d = decide("A", share_stats(_synth(0.50, 0.05, False, seed=3), nm, B=200), refp, False)
    assert d["verdict"].startswith("NOT PASSED") and not d["where_ci_lower_ge_0.70"] and not d["where_ci_upper_lt_0.30"], d
    e = decide("B", share_stats(_synth(0.90, 0.05, True, seed=4), nm, B=200), refp, True)
    assert e["PASS"] and abs(e["sum_tok_crop"] - 0.90) < 0.05 and e["where_input"] == "in.tokcrop", e
    f_ = decide("B", share_stats(_synth(0.90, 0.20, True, seed=4), nm, B=200), refp, True)
    assert not f_["PASS"] and f_["pass_alt_nocrop"], f_                     # the crop sits in the prefix: literal rule fails, the no-crop variant passes
    g = d
    assert g["gate_T_lower_gt_0"]
    h = os.path.join(os.environ.get("TMPDIR", "/tmp"), "raaf_probe_read_selftest_%d.json" % os.getpid())
    with open(h, "x") as fh:
        fh.write("{}")
    try:
        read("A", [], [], h, {"N", "K"}, False)
        raise AssertionError("overwrote")
    except SystemExit:
        pass
    os.remove(h)
    print("[raaf-read] selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "read"))
    ap.add_argument("--route", default="")
    ap.add_argument("--dirs", default="")
    ap.add_argument("--ref-dirs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--sets", default="N+K")
    ap.add_argument("--crop", type=int, default=-1, help="default: on for route B")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        crop = bool(a.crop) if a.crop >= 0 else a.route == "B"
        read(a.route, [d for d in a.dirs.split("+") if d], [d for d in a.ref_dirs.split("+") if d], a.out, set(a.sets.split("+")), crop)

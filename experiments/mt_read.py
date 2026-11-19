"""MT read (SR 81; docs/notes/mt_plan.md; m3_prereg tag mt).  Numpy only (any venv; run on a compute node).

Cells (results/flawed/mt: experiments/mt_tasks.py; 40 main states each, trial = state; the 4 smoke states are never read):
  K1_in K2_in (nothing moved), T_near T_far (K1, PLATE moved), O1_near O1_far (K1, BOWL moved), O2_near O2_far (K2, BOWL moved).
Arms: I (frozen pi0.5), r2 (R2), raa (RAA), rgp (ReGuide-perc); rollouts results/flawed/mt_roll/<arm>_<cell>_main.json + _track.npz.

ENDPOINT ATTRIBUTION (target moved; T cells and K1_in), from the track (pp_atlas.Rec: held flag, the object's, the target's and the eef's positions):
  carry          the object was held for >= 5 steps.  The release is the LAST held step L (the end of the carry).
  d_act          xy distance of the held object at L to the target's actual position at L (the plate's own track);
  d_tr           the same to the target's trained position (n2_tasks.json training.centre, the mean of the stored sampler states = the mode);
  TRAINED RELEASE  d_act - d_tr > 0.03 m  (closer to the trained position than to the actual one by more than 3 cm).
  share_all      mean over ALL states of the indicator (an episode without a carry counts 0);  share_carry = over carry episodes only.
  lam            ((rel - act) . (tr - act)) / |tr - act|^2 at L: 0 = at the actual target, 1 = at the trained one (descriptive).
  Property of the rule: a displacement of d <= 3 cm cannot give a trained release, so in-range cells have share 0 by construction; M1 therefore
  reads as "the far-out share is > 0".  (n2's demo range: spread .0273 m; the 50 official init states of K1 lie within .018 m of the centre.)
REACH ATTRIBUTION (object moved; O cells and the in-range cells, descriptive): the first step with the gripper commanded closed (a6 > 0) while the eef
  is within 10 cm above the object's start height; attempt at the eef xy: trained attempt iff |ee - object_start| - |ee - trained object position| > 0.03 m.

REGISTERED CONTRASTS (paired by state within a bin; bootstrap B = 2000, numpy default_rng(0), percentile 2.5 / 97.5, states resampled; M1's two bins are different states
and are resampled independently in the same draw)
  M1  I: share_all(T_far) - share_all(K1_in)         SUPPORTED iff lower bound > 0  (REVERSED iff upper bound < 0, else NOT SUPPORTED).
  M2  RAA - I success on T_far                        SUPPORTED iff lower bound > 0  (REVERSED iff upper < 0, else NOT SUPPORTED).
  M3  RAA - I share_all on T_far                      SUPPORTED iff upper bound < 0  (REVERSED iff lower > 0, else NOT SUPPORTED).
  MK  every arm's in-range success (K1_in, K2_in): FIT iff the point >= .70, else BELOW FLOOR.
  Testability annotations (the label is kept): M1 needs I's K1_in success >= .70; M2 needs I's T_far success <= .85 (room to improve); M3 needs
  I's T_far share_all > .10 (room to reduce).  Descriptive: rgp, r2, every arm x cell, the O cells, lam, release error to the actual target.

    python experiments/mt_read.py read --phase smoke|main [--roll DIR] --out X.json
    python experiments/mt_read.py selftest
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "flawed")
MT = os.path.join(OUT, "mt")
ROLL = os.path.join(OUT, "mt_roll")
B = 2000
TH, MIN_HELD, FLOOR, ROOM_S, ROOM_C = 0.03, 5, 0.70, 0.85, 0.10
ARMS = ("I", "r2", "raa", "rgp")
CELLS = ("K1_in", "T_near", "T_far", "O1_near", "O1_far", "K2_in", "O2_near", "O2_far")
TARGET_MOVED = ("T_near", "T_far")
BIN = dict(K1_in=(0.0, 0.0273), K2_in=(0.0, 0.0273), T_near=(0.04, 0.07), O1_near=(0.04, 0.07), O2_near=(0.04, 0.07),
           T_far=(0.14, 0.30), O1_far=(0.14, 0.30), O2_far=(0.14, 0.30))


def bootm(fn, ns, seed=0):
    rng = np.random.default_rng(seed)
    pt = fn([np.arange(n) for n in ns])
    v = np.array([fn([rng.integers(0, n, n) for n in ns]) for _ in range(B)], float)
    v = v[np.isfinite(v)]
    if not len(v) or not np.isfinite(pt):
        return [float(pt), None, None]
    return [float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def label(ci, hi_good=True):
    """SUPPORTED iff lo > 0 (hi_good) / hi < 0 (not hi_good); REVERSED the other way; else NOT SUPPORTED."""
    pt, lo, hi = ci
    if lo is None:
        return "NOT TESTABLE"
    if hi_good:
        return "SUPPORTED" if lo > 0 else ("REVERSED" if hi < 0 else "NOT SUPPORTED")
    return "SUPPORTED" if hi < 0 else ("REVERSED" if lo > 0 else "NOT SUPPORTED")


def fmt(ci):
    return "%+.3f [%+.3f, %+.3f]" % tuple(ci) if ci[1] is not None else str(ci)


# ---------------------------------------------------------------- attribution
def attribute(tr, trained_t, trained_o):
    """One episode's track -> dict (carry, release attribution, reach attribution).  tr: held, a6, obj, tgt, ee (per step)."""
    held = np.asarray(tr["held"], bool)
    obj, tgt, ee, a6 = (np.asarray(tr[k], np.float64) for k in ("obj", "tgt", "ee", "a6"))
    r = dict(carry=bool(held.sum() >= MIN_HELD), n_held=int(held.sum()))
    if r["carry"]:
        L = int(np.flatnonzero(held)[-1])
        rel, act = obj[L, :2], tgt[L, :2]
        d_act, d_tr = float(np.linalg.norm(rel - act)), float(np.linalg.norm(rel - trained_t))
        r.update(L=L, d_act=d_act, d_tr=d_tr, trained_release=bool(d_act - d_tr > TH))
        v = trained_t - act
        r["lam"] = float(np.dot(rel - act, v) / np.dot(v, v)) if np.dot(v, v) > 1e-6 else None
        r["tgt_moved_cm"] = float(100 * np.linalg.norm(tgt[L, :2] - tgt[0, :2]))
    close = np.flatnonzero((a6 > 0) & (ee[:, 2] - obj[0, 2] < 0.10))
    r["attempt"] = bool(len(close))
    if len(close):
        s = int(close[0])
        e = ee[s, :2]
        d_act, d_tr = float(np.linalg.norm(e - obj[0, :2])), float(np.linalg.norm(e - trained_o))
        r.update(att_d_act=d_act, att_d_tr=d_tr, trained_attempt=bool(d_act - d_tr > TH))
    r["obj0"], r["tgt0"] = obj[0, :2].tolist(), tgt[0, :2].tolist()
    return r


def load_cell(roll, arm, cell, phase, br):
    """-> (success array, [track dict per episode]) in branch order, or None."""
    p = os.path.join(roll, "%s_%s_%s.json" % (arm, cell, phase))
    tp = p.replace(".json", "_track.npz")
    if not (os.path.exists(p) and os.path.exists(tp)):
        return None
    res = json.load(open(p, encoding="utf-8"))
    eps = list(res["tasks"].values())[0]["episodes"]
    z = np.load(tp)
    n = int(z["n"])
    tr = [{k: z["e%d_%s" % (i, k)] for k in ("held", "a6", "obj", "tgt", "ee")} for i in range(n)]
    trials = [b["trial"] for b in br]
    if [e["trial"] for e in eps] != trials or n != len(eps):
        return dict(error="trial / track mismatch", n_eps=len(eps), n_track=n)
    return dict(success=np.array([bool(e["success"]) for e in eps]), tracks=tr, errors=[e.get("error") for e in eps])


def cell_stats(D, arm, cell, trained, ns):
    """Point statistics and CIs of one arm x cell."""
    succ = D["success"]
    A = [attribute(t, trained["plate"], trained["bowl"]) for t in D["tracks"]]
    n = len(A)
    carry = np.array([a["carry"] for a in A])
    tr_rel = np.array([a["carry"] and a["trained_release"] for a in A])
    att = np.array([a["attempt"] for a in A])
    tr_att = np.array([a["attempt"] and a["trained_attempt"] for a in A])
    st = dict(n=n, success=bootm(lambda ix: succ[ix[0]].mean(), [n]), carry=float(carry.mean()), attempt=float(att.mean()),
              n_errors=int(sum(1 for e in D["errors"] if e)))
    if cell in TARGET_MOVED or cell == "K1_in":
        st["share_all"] = bootm(lambda ix: tr_rel[ix[0]].mean(), [n])
        st["share_carry"] = float(tr_rel[carry].mean()) if carry.any() else None
        st["n_trained_release"] = int(tr_rel.sum())
        lam = [a["lam"] for a in A if a["carry"] and a.get("lam") is not None]
        st["lam_median"] = float(np.median(lam)) if lam else None
        dd = [a["d_act"] for a in A if a["carry"]]
        st["release_err_to_actual_cm_median"] = float(100 * np.median(dd)) if dd else None
        st["failed_with_trained_release"] = int((tr_rel & ~succ).sum())
    st["share_attempt_trained_all"] = bootm(lambda ix: tr_att[ix[0]].mean(), [n])
    st["share_attempt_trained_of_attempts"] = float(tr_att[att].mean()) if att.any() else None
    return st, A, succ, tr_rel


def read(a):
    roll = a.roll or ROLL
    meta = json.load(open(os.path.join(MT, "mt_tasks.json"), encoding="utf-8"))
    trained = dict(plate=np.array(meta["trained"]["plate"]), bowl=np.array(meta["trained"]["bowl"]))
    rows = {}
    ident = dict(n_bad=0, detail=[])
    for arm in ARMS:
        for cell in CELLS:
            br = json.load(open(os.path.join(MT, "mt_br_%s_%s.json" % (cell, a.phase)), encoding="utf-8"))
            D = load_cell(roll, arm, cell, a.phase, br)
            if D is None:
                continue
            if "error" in D:
                rows[(arm, cell)] = D
                continue
            st, A, succ, tr_rel = cell_stats(D, arm, cell, trained, len(br))
            rows[(arm, cell)] = dict(stats=st, A=A, succ=succ, tr_rel=tr_rel)
            # identity: the moved entity starts at the cell's displacement
            mv = {"T": "tgt0", "O": "obj0"}.get(cell[0])
            if mv:
                c = trained["plate"] if mv == "tgt0" else trained["bowl"]
                d = np.array([np.linalg.norm(np.array(x[mv]) - c) for x in A])
                lo_, hi_ = BIN[cell]
                bad = int(((d < lo_ - 0.01) | (d > hi_ + 0.01)).sum())
                ident["n_bad"] += bad
                st["moved_dist_cm_range"] = [float(100 * d.min()), float(100 * d.max())]
                st["identity_bad"] = bad
    # start-state pairing across arms (the object's start position)
    for cell in CELLS:
        ref = None
        for arm in ARMS:
            r = rows.get((arm, cell))
            if not r or "A" not in r:
                continue
            o = np.array([x["obj0"] for x in r["A"]])
            if ref is None:
                ref = (arm, o)
            else:
                dmax = float(np.abs(o - ref[1]).max())
                if dmax > 0.005:
                    ident["n_bad"] += 1
                    ident["detail"].append([cell, arm, ref[0], dmax])
    out = dict(phase=a.phase, B=B, mapping=__doc__, cells={}, contrasts={}, identity=ident)
    for (arm, cell), r in rows.items():
        out["cells"].setdefault(arm, {})[cell] = r if "error" in r else r["stats"]

    def S(arm, cell):
        r = rows.get((arm, cell))
        return r if r and "A" in r else None
    # M1
    r1, r0 = S("I", "T_far"), S("I", "K1_in")
    if r1 and r0:
        s1, s0 = r1["tr_rel"].astype(float), r0["tr_rel"].astype(float)
        ci = bootm(lambda ix: s1[ix[0]].mean() - s0[ix[1]].mean(), [len(s1), len(s0)])
        out["contrasts"]["M1"] = dict(ci=ci, label=label(ci), testable=bool(r0["succ"].mean() >= FLOOR), I_K1_in_success=float(r0["succ"].mean()),
                                      I_T_far_share=float(s1.mean()), I_K1_in_share=float(s0.mean()))
    # M2, M3
    ra, ri = S("raa", "T_far"), S("I", "T_far")
    if ra and ri:
        n = len(ra["succ"])
        sa, si = ra["succ"].astype(float), ri["succ"].astype(float)
        ci2 = bootm(lambda ix: sa[ix[0]].mean() - si[ix[0]].mean(), [n])
        out["contrasts"]["M2"] = dict(ci=ci2, label=label(ci2), testable=bool(si.mean() <= ROOM_S), I_T_far_success=float(si.mean()), RAA_T_far_success=float(sa.mean()))
        ta, ti = ra["tr_rel"].astype(float), ri["tr_rel"].astype(float)
        ci3 = bootm(lambda ix: ta[ix[0]].mean() - ti[ix[0]].mean(), [n])
        out["contrasts"]["M3"] = dict(ci=ci3, label=label(ci3, hi_good=False), testable=bool(ti.mean() > ROOM_C), I_share=float(ti.mean()), RAA_share=float(ta.mean()))
    # MK
    mk = {}
    for arm in ARMS:
        for cell in ("K1_in", "K2_in"):
            r = S(arm, cell)
            if r:
                mk["%s/%s" % (arm, cell)] = dict(success=r["stats"]["success"], label="FIT" if r["stats"]["success"][0] >= FLOOR else "BELOW FLOOR")
    out["contrasts"]["MK"] = mk
    # descriptive: rgp, r2 minus I per cell
    desc = {}
    for arm in ("raa", "r2", "rgp"):
        for cell in CELLS:
            x, y = S(arm, cell), S("I", cell)
            if x and y:
                sx, sy = x["succ"].astype(float), y["succ"].astype(float)
                d = dict(success_diff=bootm(lambda ix: sx[ix[0]].mean() - sy[ix[0]].mean(), [len(sx)]))
                if cell in TARGET_MOVED:
                    tx, ty = x["tr_rel"].astype(float), y["tr_rel"].astype(float)
                    d["share_diff"] = bootm(lambda ix: tx[ix[0]].mean() - ty[ix[0]].mean(), [len(tx)])
                desc["%s-I/%s" % (arm, cell)] = d
    out["contrasts"]["descriptive"] = desc
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("== MT read, phase %s: identity bad %d" % (a.phase, ident["n_bad"]))
    for arm in ARMS:
        for cell in CELLS:
            r = out["cells"].get(arm, {}).get(cell)
            if not r or "error" in r:
                if r:
                    print(arm, cell, "ERROR", r)
                continue
            print("%-4s %-8s n %2d success %s carry %.2f share_all %s share_carry %s lam_med %s relerr_cm %s att_tr %s" % (
                arm, cell, r["n"], fmt(r["success"]), r["carry"],
                fmt(r["share_all"]).split(" [")[0] if "share_all" in r else "-", r.get("share_carry"), r.get("lam_median"),
                r.get("release_err_to_actual_cm_median"), r["share_attempt_trained_of_attempts"]))
    for k, v in out["contrasts"].items():
        if k not in ("MK", "descriptive"):
            print(k, v)
    for k, v in mk.items():
        print("MK", k, v["label"], fmt(v["success"]))
    return out


def selftest(a):
    """CPU: the attribution rule on constructed tracks, bootm on a known case, label rules."""
    T = np.array([0.05, -0.02])
    O = np.array([-0.09, 0.0])
    n = 30

    def trk(rel_xy, act_xy, extra=0):
        held = np.zeros(n, bool)
        held[5:20] = True
        obj = np.zeros((n, 3))
        obj[5:20, :2] = np.array(rel_xy)
        obj[0, :2] = O
        tgt = np.zeros((n, 3))
        tgt[:, :2] = np.array(act_xy)
        ee = np.zeros((n, 3))
        ee[:, 2] = 0.5
        a6 = -np.ones(n)
        return dict(held=held, a6=a6, obj=obj, tgt=tgt, ee=ee)
    act = T + np.array([0.0, 0.20])
    r = attribute(trk(T + np.array([0.01, 0.0]), act), T, O)
    assert r["carry"] and r["trained_release"] and abs(r["lam"] - 1.0) < 0.1, r              # released near the trained spot
    r = attribute(trk(act + np.array([0.01, 0.0]), act), T, O)
    assert r["carry"] and not r["trained_release"] and abs(r["lam"]) < 0.1, r                 # released at the actual target
    mid = (T + act) / 2
    r = attribute(trk(mid, act), T, O)
    assert not r["trained_release"], r                                                      # halfway: neither wins by > 3 cm
    near = T + np.array([0.0, 0.02])
    r = attribute(trk(T, near), T, O)
    assert not r["trained_release"], r                                                      # displacement 2 cm: cannot exceed 3 cm
    t0 = trk(T, act)
    t0["held"][:] = False
    assert not attribute(t0, T, O)["carry"]
    t1 = trk(T, act)
    t1["a6"][12:] = 1.0
    t1["ee"][:, :2] = T + np.array([0.0, 0.005])
    t1["ee"][:, 2] = 0.05
    r = attribute(t1, T, O)
    assert r["attempt"] is True and r["trained_attempt"] is False, r                        # trained object spot = O: the eef is far from it
    assert attribute(t1, T, T)["trained_attempt"] is True                                   # trained object spot = T: the eef is there
    ci = bootm(lambda ix: np.ones(40)[ix[0]].mean(), [40])
    assert ci == [1.0, 1.0, 1.0], ci
    x = np.r_[np.ones(20), np.zeros(20)]
    ci = bootm(lambda ix: x[ix[0]].mean(), [40])
    assert 0.3 < ci[1] < 0.4 and 0.6 < ci[2] < 0.7, ci
    assert label([0.1, 0.01, 0.2]) == "SUPPORTED" and label([-0.1, -0.2, -0.01]) == "REVERSED" and label([0, -0.1, 0.1]) == "NOT SUPPORTED"
    assert label([-0.1, -0.2, -0.01], hi_good=False) == "SUPPORTED" and label([0.1, 0.01, 0.2], hi_good=False) == "REVERSED"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--phase", required=True, choices=("smoke", "main"))
    r.add_argument("--roll", default="")
    r.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a)
    if os.path.exists(a.out):
        raise SystemExit("exists: " + a.out)
    read(a)


if __name__ == "__main__":
    main()

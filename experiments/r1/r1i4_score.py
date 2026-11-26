"""I4 scorer (numpy only).  Registration: interp_I45_registered; PLAN_INTERP_I45.md v3 section I4.
  python r1i4_score.py [--out DIR] [--smoke] [--boot 2000]  ->  {OUT}/i4.json
Order: checks C1-C4 first (a failure voids I4: "void": true; C5 is descriptive only, addendum interp_I4_amended), then I4-P1..P4.
Whatever data are missing (groups, seeds, activity.npz, pinned T1) is reported "not_evaluable" and the rest still runs.

Literal-reading choices:
 - Episode logs: {OUT}/s{S}/{group}.npz/.json written by r1i4.py; activity from {OUT}/s{S}/activity.npz (r1i4_activity.py).
 - Bit for bit = np.array_equal on every logged array (actions pre/post clip, eef, object, contact, segment, success flag per step; chunk
   state, x0, path, executed plan) plus success, steps, on_t, off_t.
 - C2: for every episode of arm grasp or scramble that is in a cell with gating: t_g = first chunk start with a gated active set.  Before t_g
   (steps t < t_g: pre/post actions; t <= t_g: eef, object, contact, segment; chunks t <= t_g: state, x0) and on_t must equal none's.  An
   episode with no gated chunk must equal none in full.
 - C4: episode 0 of every cell (success, steps) against {I4_T1 or results/r1/t1p}/none/s{S}/{group}.json.
 - C5 (descriptive): distribution of Q_scr/Q_grasp (post-clip, first gated chunk, common steps of the three arms) and per-DOF clip rates.
   P3/P4 are scored on all P3 episodes and on the size-matched subset (Q_scr within +-10% of Q_grasp; < 50% of episodes -> unresolved); each arm uses its own Q.
 - Q_e = L2 norm of the post-clip action difference grasp - none over the executed steps of the first gated chunk (up to 8; the episode may
   end earlier).  P3 set = episodes of the grasp arm with a gated chunk, in cells that run scramble (single_OW, pair_HBOW, pair_OWTC seeds 0-1).
   Discordance D uses none's success.
 - Strata: seed x cell x none success.  Eligible: both arms have >= 1 discordant and >= 1 concordant; weights n_s (strata sizes), common.
   Bootstrap: resample (cell name, episode index) blocks jointly across seeds (a block holds every seed's episode at that cell and index),
   strata recomputed in each replicate.  Pooled AUCs (secondary) weight strata by n1*n0.
 - P4 equivalence uses the Wald 95% CI of (b - c)/N, SE = sqrt(b + c - (b-c)^2/N)/N (the SE the plan quotes); the block-bootstrap CI is reported too.
 - R_k per episode: none arm; A_k(t) = mean over the 10 flow steps of ||d_k|| at chunk start t; S0 = chunk starts t < on_t, S12^5 = the first 5
   chunk starts t >= on_t (episodes without onset or with < 3 such starts dropped and counted).  P1/P2 pooled CI = block bootstrap of the mean over
   all eligible episodes of all seeds (episodes equal weight).  Secondary: R_k minus the mean of the 8 random deltas' R (draws resampled inside).
"""
import argparse, json, math, os, sys
import numpy as np

RES = "/coc/flash10/ksingara3/flawed-composition-vlas/results/r1"
TASKS = ["HB", "OW", "TC"]
ARMS = ["none", "grasp", "scramble", "none2"]
GROUPS = ["single_HB", "single_OW", "single_TC", "pair_HBOW", "pair_HBTC", "pair_OWTC"]
SEEDS = (0, 1, 2)
LOG125 = math.log(1.25)


def jdump(o):
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, np.floating): return clean(float(x))
    return x


class Grp:
    def __init__(self, OUT, S, g):
        self.S, self.g = S, g
        self.j = json.load(open(f"{OUT}/s{S}/{g}.json", encoding="utf-8"))
        self.z = dict(np.load(f"{OUT}/s{S}/{g}.npz"))
        z = self.z; n = len(z["ep_arm"])
        self.row = {(int(c), ARMS[int(a)], int(e)): i for i, (c, a, e) in enumerate(zip(z["ep_cell"], z["ep_arm"], z["ep_e"]))}
        ar = np.arange(n)
        self.c0, self.c1 = np.searchsorted(z["ch_ep"], ar, "left"), np.searchsorted(z["ch_ep"], ar, "right")
        self.s0, self.s1 = np.searchsorted(z["sp_ep"], ar, "left"), np.searchsorted(z["sp_ep"], ar, "right")
        self.cell_names = self.j["cell_names"]

    def ch(self, k, r): return self.z["ch_" + k][self.c0[r]:self.c1[r]]
    def sp(self, k, r): return self.z["sp_" + k][self.s0[r]:self.s1[r]]
    def ep(self, k, r): return self.z["ep_" + k][r]
    def arms_of(self, ci): return self.j["cells"][self.cell_names[ci]]["arms"]

    def tg(self, r):
        gt = self.ch("gated", r)
        return int(self.ch("t", r)[np.argmax(gt)]) if gt.any() else None


def eq(a, b): return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def same_full(G, r1, r2):
    bad = []
    for k in ("success", "steps", "on_t", "off_t"):
        if G.ep(k, r1) != G.ep(k, r2): bad.append("ep_" + k)
    for k in ("pre", "post", "eef", "obj", "contact", "seg", "succ"):
        if not eq(G.sp(k, r1), G.sp(k, r2)): bad.append("sp_" + k)
    for k in ("t", "state", "x0", "path", "plan_pre", "seg", "active"):
        if not eq(G.ch(k, r1), G.ch(k, r2)): bad.append("ch_" + k)
    return bad


def same_prefix(G, r1, r2, tg):
    bad = []
    vis = lambda x: int(x) if 0 <= int(x) <= tg else None   # an onset only counts if it falls inside the shared prefix (gating from t = 0 leaves none)
    if vis(G.ep("on_t", r1)) != vis(G.ep("on_t", r2)): bad.append("on_t")
    t1, t2 = G.sp("t", r1), G.sp("t", r2)
    if tg not in t1 or tg not in t2: return bad + ["missing step tg"]
    for k in ("pre", "post"):
        m1, m2 = t1 < tg, t2 < tg
        if not eq(G.sp(k, r1)[m1], G.sp(k, r2)[m2]): bad.append("sp_" + k)
    for k in ("eef", "obj", "contact", "seg"):
        m1, m2 = t1 <= tg, t2 <= tg
        if not eq(G.sp(k, r1)[m1], G.sp(k, r2)[m2]): bad.append("sp_" + k)
    c1, c2 = G.ch("t", r1) <= tg, G.ch("t", r2) <= tg
    for k in ("state", "x0"):
        if not eq(G.ch(k, r1)[c1], G.ch(k, r2)[c2]): bad.append("ch_" + k)
    return bad


def status(name, n, fails, note=""):
    if n == 0: return dict(status="not_evaluable", n=0, note=note)
    return dict(status="pass" if not fails else "fail", n=n, n_fail=len(fails), first_failures=fails[:5], note=note)


def q_info(G, ci, e):
    """first gated chunk of grasp episode (ci, e): Q, scramble size, clipping; None if no grasp gated chunk"""
    rg, rn = G.row.get((ci, "grasp", e)), G.row.get((ci, "none", e))
    if rg is None or rn is None: return None
    tg = G.tg(rg)
    if tg is None: return None
    rs = G.row.get((ci, "scramble", e))
    def seg(r):
        t = G.sp("t", r); m = (t >= tg) & (t < tg + 8)
        return G.sp("post", r)[m], G.sp("pre", r)[m]
    pg, prg = seg(rg); pn, prn = seg(rn)
    ps = prs = None
    if rs is not None: ps, prs = seg(rs)
    n = min(len(pg), len(pn), len(ps) if ps is not None else 8)
    if n == 0: return None
    q = float(np.linalg.norm((pg[:n] - pn[:n]).ravel()))
    clip = bool((prg[:n].astype(np.float64) != pg[:n]).any() or (prn[:n].astype(np.float64) != pn[:n]).any())
    qs = None
    if ps is not None:
        qs = float(np.linalg.norm((ps[:n] - pn[:n]).ravel()))
        clip = clip or bool((prs[:n].astype(np.float64) != ps[:n]).any())
    return dict(tg=tg, q=q, q_scr=qs, clipped=clip, n=n, dof_clip=(prn[:n].astype(np.float64) != pn[:n]).mean(0))


def run_checks(Gs, OUT, T1):
    out = {}
    # C1
    n, fails = 0, []
    for g, seeds in (("single_HB", (0, 1, 2)), ("single_TC", (0, 1)), ("pair_HBTC", (0, 1))):
        for S in seeds:
            G = Gs.get((S, g))
            if G is None or "grasp" not in G.arms_of(0): continue
            for e in range(G.j["n_ep"]):
                ra, rb = G.row.get((0, "none", e)), G.row.get((0, "grasp", e))
                if ra is None or rb is None: continue
                n += 1; bad = same_full(G, ra, rb)
                if bad: fails.append(dict(seed=S, group=g, e=e, differs=bad))
    out["C1"] = status("C1", n, fails, "grasp == none on ungated cells (single_HB cell 0 every seed; single_TC, pair_HBTC cell 0 seeds 0-1)")
    # C2
    n, fails = 0, []
    for (S, g), G in Gs.items():
        for ci, cn in enumerate(G.cell_names):
            info = G.j["cells"][cn]
            if not info["gated_parts"]: continue
            for arm in ("grasp", "scramble"):
                if arm not in info["arms"]: continue
                for e in range(G.j["n_ep"]):
                    ra, rn = G.row.get((ci, arm, e)), G.row.get((ci, "none", e))
                    if ra is None or rn is None: continue
                    n += 1; tg = G.tg(ra)
                    bad = same_full(G, ra, rn) if tg is None else same_prefix(G, ra, rn, tg)
                    if bad: fails.append(dict(seed=S, group=g, cell=cn, arm=arm, e=e, tg=tg, differs=bad))
    out["C2"] = status("C2", n, fails, "grasp/scramble match none until the first gated chunk start; onset equal")
    # C3
    n, fails = 0, []
    for S in SEEDS:
        G = Gs.get((S, "pair_OWTC"))
        if G is None or "none2" not in G.arms_of(0): continue
        for e in range(G.j["n_ep"]):
            ra, rb = G.row[(0, "none", e)], G.row[(0, "none2", e)]
            n += 1; bad = same_full(G, ra, rb)
            if bad: fails.append(dict(seed=S, e=e, differs=bad))
    out["C3"] = status("C3", n, fails, "second none run of pair_OWTC cell 0")
    # C4
    n, fails = 0, []
    for (S, g), G in Gs.items():
        f = f"{T1}/none/s{S}/{g}.json"
        if not os.path.exists(f): continue
        t1 = json.load(open(f, encoding="utf-8"))["cells"]
        for ci, cn in enumerate(G.cell_names):
            if cn not in t1 or (ci, "none", 0) not in G.row: continue
            r = G.row[(ci, "none", 0)]; n += 1
            a, b = (bool(G.ep("success", r)), int(G.ep("steps", r))), (bool(t1[cn]["success"][0]), int(t1[cn]["steps"][0]))
            if a != b: fails.append(dict(seed=S, group=g, cell=cn, i4=a, t1=b))
    out["C4"] = status("C4", n, fails, f"episode 0 under none vs pinned T1 none ({T1})")
    # C5
    n, fails, nclip, ntot = 0, [], 0, 0
    dof, gap = [], []
    for (S, g), G in Gs.items():
        for ci, cn in enumerate(G.cell_names):
            if "scramble" not in G.arms_of(ci): continue
            for e in range(G.j["n_ep"]):
                qi = q_info(G, ci, e)
                if qi is None or qi["q_scr"] is None: continue
                ntot += 1; dof.append(qi["dof_clip"]); gap.append(qi["q_scr"] / max(qi["q"], 1e-12))
                nclip += qi["clipped"]
    # amended (interp_I4_amended): C5 is descriptive only and never voids I4
    out["C5"] = dict(status="descriptive", n_gated_episodes=ntot, n_episodes_with_any_clipping=nclip,
                     note="distribution of Q_scr/Q_grasp (post-clip) and per-DOF clip rates; does not void I4")
    if ntot:
        out["C5"]["q_scr_over_q_grasp_quantiles_0_10_50_90_100"] = np.percentile(gap, [0, 10, 50, 90, 100])
        out["C5"]["frac_within_10pct"] = float(np.mean(np.abs(np.array(gap) - 1) <= 0.10))
        out["C5"]["none_clip_frac_by_dof"] = np.mean(dof, 0)
    out["void"] = bool(any(out[c]["status"] == "fail" for c in ("C1", "C2", "C3", "C4")))
    return out


# ---------------------------------------------------------------- statistics
def auc(q, d):
    npos = int(d.sum()); nneg = len(d) - npos
    if npos == 0 or nneg == 0: return float("nan")
    srt = np.sort(q); l = np.searchsorted(srt, q, "left"); r = np.searchsorted(srt, q, "right")
    rk = (l + 1 + r) / 2
    return float((rk[d].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def strat_stats(q, qs, dg, ds, st):
    """-> dAUC (common weights over strata eligible for both arms), eligible n, pooled AUC grasp / scramble (n1*n0 weights)"""
    num = den = 0.0; pg = [0.0, 0.0]; ps = [0.0, 0.0]
    order = np.argsort(st, kind="stable"); sts = st[order]
    cuts = np.flatnonzero(np.diff(sts)) + 1
    for ix in np.split(order, cuts):
        n = len(ix); a, b = dg[ix], ds[ix]; qa, qb = q[ix], qs[ix]
        for d_, qq, acc in ((a, qa, pg), (b, qb, ps)):
            if d_.any() and (~d_).any():
                w = d_.sum() * (~d_).sum(); acc[0] += w * auc(qq, d_); acc[1] += w
        if a.any() and (~a).any() and b.any() and (~b).any():
            num += n * (auc(qa, a) - auc(qb, b)); den += n
    return (num / den if den else float("nan")), den, (pg[0] / pg[1] if pg[1] else float("nan")), (ps[0] / ps[1] if ps[1] else float("nan"))


def mcnemar_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def block_mean_boot(vals, blocks, B, rng, weights=None):
    """pooled mean over resampled blocks; vals (n,) or (B', n) with B' = B"""
    ub, inv = np.unique(blocks, return_inverse=True); nb = len(ub)
    cnt = np.bincount(inv, minlength=nb).astype(float)
    out = np.empty(B)
    for b in range(B):
        v = vals if vals.ndim == 1 else vals[b]
        bs = np.bincount(inv, weights=v, minlength=nb)
        idx = rng.randint(nb, size=nb)
        out[b] = bs[idx].sum() / cnt[idx].sum()
    return out


def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def p3p4(rows, B, rng):
    """rows: list of dict(seed, cell, e, q, dg, ds, nsucc); returns the P3/P4 block"""
    N = len(rows)
    if N == 0: return dict(status="not_evaluable", N=0)
    q = np.array([r["q"] for r in rows]); qs = np.array([r["qs"] for r in rows]); dg = np.array([r["dg"] for r in rows], bool); ds = np.array([r["ds"] for r in rows], bool)
    keys = [(r["seed"], r["cell"], r["ns"]) for r in rows]
    _, st = np.unique(np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in keys]), return_inverse=True)
    blk = np.array([f"{r['cell']}|{r['e']}" for r in rows])
    d, elig, pag, pas = strat_stats(q, qs, dg, ds, st)
    b_ = int((dg & ~ds).sum()); c_ = int((~dg & ds).sum())
    diff = (b_ - c_) / N
    se = math.sqrt(max(b_ + c_ - (b_ - c_) ** 2 / N, 0.0)) / N
    wald = [diff - 1.96 * se, diff + 1.96 * se]
    ub, binv = np.unique(blk, return_inverse=True)
    members = [np.flatnonzero(binv == i) for i in range(len(ub))]
    bd, bg, bsc, bdiff = [], [], [], []
    for _ in range(B):
        ix = np.concatenate([members[i] for i in rng.randint(len(ub), size=len(ub))])
        x = strat_stats(q[ix], qs[ix], dg[ix], ds[ix], st[ix])
        if np.isfinite(x[0]): bd.append(x[0])
        if np.isfinite(x[2]): bg.append(x[2])
        if np.isfinite(x[3]): bsc.append(x[3])
        bdiff.append(((dg[ix] & ~ds[ix]).sum() - (~dg[ix] & ds[ix]).sum()) / len(ix))
    res = dict(N=N, n_discordant_grasp=int(dg.sum()), n_discordant_scramble=int(ds.sum()), b=b_, c=c_, delta_auc=d,
               eligible_fraction=elig / N, auc_grasp_pooled=pag, auc_scramble_pooled=pas)
    if bd:
        lo, hi = ci(np.array(bd)); res["delta_auc_ci95"] = [lo, hi]
        if elig / N < 0.5: v = "unresolved (eligible strata < 50% of episodes)"
        elif d >= 0.10 and lo > 0: v = "holds"
        elif lo >= -0.10 and hi <= 0.10: v = "fails (CI within +-.10)"
        elif d <= -0.10 and hi < 0: v = "reverse effect"
        else: v = "unresolved"
        res["P3"] = v
    if bg: res["auc_grasp_pooled_ci95"] = ci(np.array(bg))
    if bsc: res["auc_scramble_pooled_ci95"] = ci(np.array(bsc))
    p = mcnemar_p(b_, c_)
    res.update(mcnemar_p=p, bc_over_N=diff, bc_wald_ci95=wald, bc_boot_ci95=ci(np.array(bdiff)))
    if p < 0.05 and abs(diff) >= 0.03: v4 = "differs (b > c)" if b_ > c_ else "differs (b < c)"
    elif wald[0] >= -0.03 and wald[1] <= 0.03: v4 = "equivalent"
    else: v4 = "unresolved"
    res["P4"] = v4
    # reading table
    p3, a_lo = res.get("P3", ""), (res.get("auc_scramble_pooled_ci95") or [float("nan")])[0]
    if p3 == "holds": rd = "gating acts through the part's content" if v4 == "differs (b > c)" else ("the part's content decides which outcomes change, not how many" if v4 in ("equivalent", "differs (b < c)") else "unresolved")
    elif p3 == "reverse effect": rd = "a redirection's size predicts outcome change better than a removal's"
    elif p3.startswith("fails"):
        if v4.startswith("differs"): rd = "direction sets how many outcomes change, not which"
        elif v4 == "equivalent": rd = "size alone" if a_lo > 0.5 else "Q predicts neither arm's outcome changes"
        else: rd = "unresolved"
    else: rd = "unresolved"
    res["reading"] = rd
    return res


def episode_R(G, act, S):
    """per-episode R for the none arm of group G: dict (ci, e) -> dict(R (3,), Rr (3, ndraw)) ; and drop counts"""
    z = G.z; rows = act[G.g + "__rows"]; Ap = act[G.g + "__A_part"].mean(-1); Ar = act[G.g + "__A_rand"].mean(-1)   # (N,3), (N,3,ndraw)
    pos = -np.ones(len(z["ch_ep"]), int); pos[rows] = np.arange(len(rows))
    out, ndrop_no_onset, ndrop_short = {}, 0, 0
    for ci, cn in enumerate(G.cell_names):
        for e in range(G.j["n_ep"]):
            r = G.row.get((ci, "none", e))
            if r is None: continue
            on = int(G.ep("on_t", r))
            if on < 0: ndrop_no_onset += 1; continue
            ix = pos[G.c0[r]:G.c1[r]]; t = G.ch("t", r)
            s0 = ix[t < on]; s1 = ix[t >= on][:5]
            if len(s1) < 3 or len(s0) < 1: ndrop_short += 1; continue
            lg = lambda a: np.log(a.mean(0) + 1e-12)
            R = lg(Ap[s0]) - lg(Ap[s1]); Rr = lg(Ar[s0]) - lg(Ar[s1])
            out[(ci, e)] = dict(R=R, Rr=Rr)
    return out, ndrop_no_onset, ndrop_short


def p12(Gs, acts, grp, a_idx, b_idx, name, B, rng):
    """mean over eligible episodes of R_a - R_b on group grp per seed + pooled block-bootstrap CI"""
    vals, seeds, blocks, Rrs, dropped = [], [], [], [], {}
    for S in SEEDS:
        G = Gs.get((S, grp))
        if G is None or S not in acts or grp + "__rows" not in acts[S]: continue
        er, d0, d1 = episode_R(G, acts[S], S)
        dropped[S] = dict(no_onset=d0, fewer_than_3_post_onset_chunks=d1, kept=len(er))
        for (cj, e), v in er.items():
            vals.append(v["R"][a_idx] - v["R"][b_idx]); seeds.append(S); blocks.append(f"{G.cell_names[cj]}|{e}")
            Rrs.append(v["Rr"][[a_idx, b_idx]])
    if not vals: return dict(status="not_evaluable")
    vals, seeds, blocks, Rrs = np.array(vals), np.array(seeds), np.array(blocks), np.array(Rrs)   # Rrs (n,2,ndraw)
    per_seed = {int(S): float(vals[seeds == S].mean()) for S in np.unique(seeds)}
    bs = block_mean_boot(vals, blocks, B, rng); lo, hi = ci(bs)
    # secondary: R_a - R_b minus the random deltas' (draws resampled inside the bootstrap)
    nd = Rrs.shape[2]
    sec = np.empty((B, len(vals)))
    for b in range(B):
        ix = rng.randint(nd, size=nd)
        sec[b] = vals - (Rrs[:, 0, ix].mean(1) - Rrs[:, 1, ix].mean(1))
    sb = block_mean_boot(sec, blocks, B, rng)
    res = dict(per_seed_mean=per_seed, pooled_mean=float(vals.mean()), pooled_ci95=[lo, hi], n=len(vals), episodes_by_seed=dropped,
               secondary_minus_random=dict(pooled_mean=float(sec.mean()), ci95=ci(sb)), threshold_log1p25=LOG125)
    ms = list(per_seed.values())
    res["holds"] = bool(len(ms) == 3 and all(m >= LOG125 for m in ms) and lo > 0)
    res["falsified"] = bool(sum(m <= 0 for m in ms) >= 2)
    res["seeds_with_data"] = len(ms)
    return res


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default=""); p.add_argument("--smoke", action="store_true")
    p.add_argument("--boot", type=int, default=2000); a = p.parse_args()
    OUT = a.out or os.environ.get("I4_OUT") or RES + ("/i4_smoke" if a.smoke else "/i4")
    T1 = os.environ.get("I4_T1", RES + "/t1p")
    rng = np.random.RandomState(20261130)
    Gs = {}
    for S in SEEDS:
        for g in GROUPS:
            if os.path.exists(f"{OUT}/s{S}/{g}.npz") and os.path.exists(f"{OUT}/s{S}/{g}.json"):
                Gs[(S, g)] = Grp(OUT, S, g)
    acts = {}
    for S in SEEDS:
        f = f"{OUT}/s{S}/activity.npz"
        if os.path.exists(f): acts[S] = dict(np.load(f))
    res = dict(out=OUT, loaded=[f"s{S}/{g}" for (S, g) in Gs], activity_seeds=sorted(acts))
    res["checks"] = run_checks(Gs, OUT, T1)
    res["void"] = res["checks"]["void"]
    P = {}
    for name, grp, ai, bi in (("I4-P1", "single_OW", 1, 0), ("I4-P2", "single_TC", 0, 2)):
        try: P[name] = p12(Gs, acts, grp, ai, bi, name, a.boot, rng)
        except Exception as ex:
            import traceback; P[name] = dict(status="error", error=traceback.format_exc())
    # P3 / P4
    try:
        rows = []; nset = 0; nonset = 0
        for (S, g), G in Gs.items():
            for ci, cn in enumerate(G.cell_names):
                if "scramble" not in G.arms_of(ci): continue
                for e in range(G.j["n_ep"]):
                    rn, rg, rs = G.row.get((ci, "none", e)), G.row.get((ci, "grasp", e)), G.row.get((ci, "scramble", e))
                    if rn is None or rg is None or rs is None: continue
                    nset += 1
                    qi = q_info(G, ci, e)
                    if qi is None: continue
                    sn = bool(G.ep("success", rn))
                    rows.append(dict(seed=S, cell=cn, e=e, q=qi["q"], qs=qi["q_scr"], clipped=qi["clipped"], ns=sn,
                                     dg=bool(G.ep("success", rg)) != sn, ds=bool(G.ep("success", rs)) != sn))
        P34 = dict(n_p3_episodes_per_arm=nset, n_with_gated_chunk=len(rows))
        P34["all"] = p3p4(rows, a.boot, rng)
        sm = [r for r in rows if abs(r["qs"] - r["q"]) <= 0.10 * r["q"]]   # size-matched: Q_scr within +-10% of Q_grasp
        P34["size_matched_fraction"] = len(sm) / len(rows) if rows else None
        P34["size_matched"] = p3p4(sm, a.boot, rng)
        if not rows or len(sm) < 0.5 * len(rows):
            P34["verdict"] = "P3 and P4 unresolved (size-matched subset < 50% of P3 episodes)"
        else:
            P34["verdict"] = dict(P3_all=P34["all"].get("P3"), P3_size_matched=P34["size_matched"].get("P3"),
                                  P4_all=P34["all"].get("P4"), P4_size_matched=P34["size_matched"].get("P4"),
                                  note="each of P3 and P4 must hold on both sets")
        P34["note"] = "amended: each arm's AUC uses its own Q (post-clip); C5 descriptive"
        P["I4-P3_P4"] = P34
    except Exception:
        import traceback; P["I4-P3_P4"] = dict(status="error", error=traceback.format_exc())
    res["predictions"] = P
    os.makedirs(OUT, exist_ok=True)
    json.dump(clean(res), open(f"{OUT}/i4.json", "w", encoding="utf-8"), indent=1, default=jdump)
    print(json.dumps(clean(dict(void=res["void"], checks={k: ({kk: vv for kk, vv in v.items() if kk != "first_failures"} if isinstance(v, dict) else v)
                                                    for k, v in res["checks"].items()})), default=jdump, indent=1))
    print(json.dumps(clean(P), default=jdump)[:3000])


if __name__ == "__main__":
    main()

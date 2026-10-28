"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): the unsupervised prototype binder of B1.2 (b1_2_bind.py) with TWO changes,
fitted on the GENERATED training pool (results/bind/b1_3/units.json, b1_3_units.py).  CPU, chunks env (numpy + scikit-learn).

(1) PREQUENTIAL channel assignment.  Folds = the training LAYOUTS (5 folds by a fixed hash of the layout id).  For word w and code
    c in {app, geo, both}:  DL(w, c) = sum over w's referents i of -log P(referent_i | candidates), where P = softmax over the frame's
    candidates of [role expert + w's expert in c], BOTH fitted only on the folds that do not hold i's layout, and the referent is
    the token's chain (the probability of all nodes containing the referent token).  DL(w, null) = the same with the role expert
    alone (for a B word of "A of B": uniform over the candidates).  The cheapest code wins; null = a function word.  No separate
    parameter penalty (a prequential code already pays for its parameters).  A channel wins only if it transfers to new layouts.
(2) RELATIVE-ONLY geometry.  No absolute image coordinate enters any expert.  geometry = [centroid row / col and log height / width
    relative to the PARENT region (4): the region "A of B" resolves as B, else the node's body; a node with no body above it and an
    unresolved B use the node's own box, i.e. carry no position] + [position within the SIBLING set (2), ordinal rank in it (2),
    log height / width relative to the siblings' median (2)].  Siblings = the non-overlapping candidates that every appearance
    word of the phrase accepts (diagonal Mahalanobis^2 within chi-square 99%), inside the parent when one is resolved.  The role
    expert's geometry is the parent-relative part only.
Appearance = B1.1's position-free code (results/bind/b1/<arm>_b11_posfree.pkl, fitted on LIBERO TRAINING frames only).
Everything else is B1.2's: regions (Ward merge tree on the code), hard EM over the referent token's chain, product of experts,
"A of B" head composition, selection of K x composition x variance floor by held-out-LAYOUT referent likelihood (2:1 by hash).

    python experiments/b1_3_bind.py --arm siglip --smoke      fit + select, score held-out TRAINING layouts only
    python experiments/b1_3_bind.py --arm siglip --test --dev  (needs the m3_prereg pb1_3 row) predict generated test + S1 dev units
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from b1_2_bind import (PCA, PosFreePCA, affine_floats, box, chi2_99, fit_gauss, gauss_ll, merge_tree, parse,  # noqa: E402,F401
                       rel, words, RR, CC, MIN_SZ, MAX_SZ, N_EM)

S1 = os.path.join(ROOT, "results", "bind", "s1")
B1 = os.path.join(ROOT, "results", "bind", "b1")
B13 = os.path.join(ROOT, "results", "bind", "b1_3")
GRID_K = (8, 16, 32)
COMPOSE = ("head", "flat")
FLOORS = (0.0, 0.05, 0.2)
N_FOLD = 5
D_GEO = 10
GEO_FLOOR = (1.0 / 16) ** 2      # absolute variance floor of every geometry Gaussian: one token at the 16 x 16 grid (fixed before
                                  # registration: without it near-constant training dims gave held-out geometry codes ~1e9 nats)


def gfl(g):
    return g[0], np.maximum(g[1], GEO_FLOOR), g[2]


def lsm(x):
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def chain_lp(s, chain):
    """log P(referent token's chain) under softmax(s) over candidates."""
    l = lsm(s)[chain]
    m = l.max()
    return float(m + np.log(np.exp(l - m).sum()))


def fold_of(layout):
    return int(hashlib.sha256(("b13-fold" + layout).encode()).hexdigest(), 16) % N_FOLD


def heldout_layout(layout):
    return int(hashlib.sha256(("b13-select" + layout).encode()).hexdigest(), 16) % 3 == 0


# ── features ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_feats(arm):
    fe = os.path.join(B13, "feats")
    src = {s: (os.path.join(fe, f"{arm}_{s}.npy"), os.path.join(fe, f"{arm}_{s}_frames.json")) for s in ("train", "test")}
    src["dev"] = ((os.path.join(S1, "feats", "dev_img.npy"), os.path.join(S1, "feats", "dev_frames.json")) if arm == "siglip" else
                  (os.path.join(B1, "feats", "dino_dev.npy"), os.path.join(B1, "feats", "dino_dev_frames.json")))
    arr = {k: np.load(a, mmap_mode="r") for k, (a, _) in src.items()}
    idx = {f: (k, i) for k, (_, j) in src.items() for i, f in enumerate(json.load(open(j)))}
    return idx, arr


# ── regions (B1.2's, with no absolute position) ────────────────────────────────────────────────────────────────────────
class Frame:
    def __init__(self, Z, K):
        ch = merge_tree(Z)
        n = 256
        mem = [np.array([i]) for i in range(n)]
        par = np.full(2 * n - 1, -1)
        for i, (a, b) in enumerate(ch):
            mem.append(np.concatenate([mem[a], mem[b]]))
            par[a] = par[b] = n + i
        cut = n - K
        made = np.concatenate([np.full(n, -1), np.arange(n - 1)])
        pidx = np.array([made[p] if p >= 0 else 10 ** 9 for p in par])
        is_body = (made < cut) & (pidx >= cut)
        above = made >= cut
        body_of = np.full(2 * n - 1, -1)
        for j in range(2 * n - 1):
            k = j
            while k >= 0 and not is_body[k] and not above[k]:
                k = par[k]
            body_of[j] = k if (k >= 0 and is_body[k]) else -1
        self.nodes = [j for j in range(2 * n - 1) if MIN_SZ <= len(mem[j]) <= MAX_SZ]
        self.mem = [mem[j] for j in self.nodes]
        self.is_body = np.array([is_body[j] for j in self.nodes])
        self.body = np.array([body_of[j] for j in self.nodes])
        self.app = np.stack([Z[m].mean(0) for m in self.mem])
        self.box = np.array([box(m) for m in self.mem])
        self.mask = np.zeros((len(self.nodes), 256), bool)
        for i, m in enumerate(self.mem):
            self.mask[i, m] = True
        self.peak = np.array([m[np.argmin((RR[m] - b[4]) ** 2 + (CC[m] - b[5]) ** 2)] for m, b in zip(self.mem, self.box)])
        bb = {b: box(mem[b]) for b in set(self.body.tolist()) if b >= 0}
        # B1.3: a node above body level has no parent region -> its own box (no absolute image coordinate)
        self.bodybox = np.stack([bb[b] if b >= 0 else self.box[i] for i, b in enumerate(self.body)])
        self.body_node = {b: self.nodes.index(b) for b in bb if b in self.nodes}
        self.g_body = rel(self.box, self.bodybox)

    def chain(self, tok):
        return np.where(self.mask[:, tok])[0]


# ── lexicon with prequential channel assignment ──────────────────────────────────────────────────────────────────────
class Lexicon:
    def __init__(self, floor=0.0):
        self.p, self.ch, self.role, self.bg, self.null, self.floor, self.dl, self.gain = {}, {}, {}, {}, {}, floor, {}, {}

    def _role_params(self, R, G_rows):
        out = {"app": fit_gauss([o[2] for o in R], self.bg["app"][1], self.floor),
               "gbody": gfl(fit_gauss([o[4] for o in R], self.bg["gbody"][1], self.floor))}
        G = np.stack(G_rows)
        null = {"app": out["app"][:2], "geo": (G.mean(0), np.maximum(np.maximum(G.var(0) + 1e-6, self.floor * self.bg["geo"][1]), GEO_FLOOR))}
        return out, null

    def _role_score(self, rp, app, gbody):
        s = 0.0
        for k, X in (("app", app), ("gbody", gbody)):
            mu, var, _ = rp[k]
            bm, bv = self.bg[k]
            s = s + gauss_ll(X, mu, var) - gauss_ll(X, bm, bv)
        return s

    def role_score(self, role, app, gbody):
        return self._role_score(self.role[role], app, gbody)

    def llr(self, w, ctx, X, c=None):
        c = c or self.ch[w]
        mu, var, _ = self.p[w][c]
        nm, nv = self.null[ctx][c]
        return gauss_ll(X, mu, var) - gauss_ll(X, nm, nv)

    def fit(self, rows, cand_bg):
        """rows: (word or None, ctx, app x, geo x, gbody x, group, frame, G [n, D_GEO] or None, referent node, layout, chain)."""
        self.bg = {k: (X.mean(0), X.var(0) + 1e-6 if k == "app" else np.maximum(X.var(0), GEO_FLOOR)) for k, X in cand_bg.items()}
        self.null = {"cand": {"app": self.bg["app"], "geo": self.bg["geo"]}}
        self.role = {}
        roleR = {r: [o for o in rows if o[0] is None and o[1] == r] for r in ("object", "goal")}
        for r, R in roleR.items():
            self.role[r], self.null[r] = self._role_params(R, [o[3] for o in R])
        by = {}
        for o in rows:
            if o[0] is not None:
                by.setdefault(o[0], []).append(o)
        self.p = {}
        for w, os_ in by.items():
            self.p[w] = {}
            for k, j in (("app", 2), ("geo", 3)):
                rr = [o for o in os_ if o[j] is not None]
                if rr:
                    self.p[w][k] = fit_gauss(np.stack([o[j] for o in rr]), self.null["cand"][k][1], self.floor) if k == "app" else gfl(fit_gauss(np.stack([o[j] for o in rr]), self.null["cand"][k][1], self.floor))
        # ── prequential (leave-one-fold-of-layouts-out) description length of the referent, per word and code ──
        fold_role, fold_null = {}, {}
        for f in range(N_FOLD):
            for r, R in roleR.items():
                Rf = [o for o in R if fold_of(o[9]) != f]
                fold_role[(f, r)], fold_null[(f, r)] = self._role_params(Rf, [o[3] for o in Rf])
        rcache = {}

        def role_s(f, ctx, fr):
            if ctx == "cand":
                return np.zeros(len(fr.nodes))
            k = (f, ctx, id(fr))
            if k not in rcache:
                rcache[k] = self._role_score(fold_role[(f, ctx)], fr.app, fr.g_body)
            return rcache[k]
        self.dl, self.gain = {}, {}
        for w in sorted(by):
            dl = {"null": 0.0, "app": 0.0, "geo": 0.0, "both": 0.0}
            n_noref = 0
            for f in range(N_FOLD):
                held = [o for o in by[w] if fold_of(o[9]) == f]
                if not held:
                    continue
                rest = [o for o in by[w] if fold_of(o[9]) != f]
                pa = fit_gauss(np.stack([o[2] for o in rest]), self.null["cand"]["app"][1], self.floor) if rest else None
                rg = [o for o in rest if o[3] is not None]
                pg = gfl(fit_gauss(np.stack([o[3] for o in rg]), self.null["cand"]["geo"][1], self.floor)) if rg else None
                for o in held:
                    ctx, fr, G, chain = o[1], o[6], o[7], o[10]
                    nl = (fold_null[(f, ctx)] if ctx != "cand" else self.null["cand"])
                    s0 = role_s(f, ctx, fr)
                    if pa is None:
                        n_noref += 1
                    sa = (gauss_ll(fr.app, *pa[:2]) - gauss_ll(fr.app, *nl["app"])) if pa is not None else 0.0
                    sg = (gauss_ll(G, *pg[:2]) - gauss_ll(G, *nl["geo"])) if (pg is not None and G is not None) else 0.0
                    dl["null"] -= chain_lp(s0, chain)
                    dl["app"] -= chain_lp(s0 + sa, chain)
                    dl["geo"] -= chain_lp(s0 + sg, chain)
                    dl["both"] -= chain_lp(s0 + sa + sg, chain)
            if all(o[3] is None for o in by[w]):                 # a B word: no geometry row
                dl["geo"] = dl["both"] = np.inf
            self.dl[w] = {k: float(v) for k, v in dl.items()}
            self.gain[w] = {k: float(dl["null"] - v) for k, v in dl.items() if k != "null"}
            best = min(dl, key=dl.get)
            self.ch[w] = None if best == "null" else best

    def nfloats(self):
        n = sum(2 * len(self.p[w][k][0]) for w, c in self.ch.items() if c for k in (("app", "geo") if c == "both" else (c,)))
        n += sum(2 * len(self.role[r][k][0]) for r in self.role for k in self.role[r])
        return n


# ── binding ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class Binder:
    def __init__(self, lex, K, compose):
        self.lex, self.K, self.compose = lex, K, compose

    def app_evidence(self, fr, ws, ctx):
        s = np.zeros(len(fr.nodes))
        for w in ws:
            if self.lex.ch.get(w) in ("app", "both"):
                s += self.lex.llr(w, ctx, fr.app, "app")
        return s

    def geo_feats(self, fr, ws, ctx, parent_box):
        """[n, 10] RELATIVE geometry: parent-relative (4) + sibling set-position (2), rank (2), log size vs the siblings' median (2)."""
        g = fr.g_body if parent_box is None else rel(fr.box, parent_box)
        n = len(fr.nodes)
        a = self.app_evidence(fr, ws, ctx)
        flat = np.concatenate([g, np.full((n, 4), 0.5), np.zeros((n, 2))], 1)
        if not np.any(a != 0):
            return flat
        acc = np.ones(n, bool)
        for w in ws:
            if self.lex.ch.get(w) in ("app", "both"):
                mu, var, _ = self.lex.p[w]["app"]
                acc &= (((fr.app - mu) ** 2) / var).sum(1) <= chi2_99(len(mu))
        if parent_box is not None:                                  # siblings inside the resolved parent
            pb = parent_box
            acc &= (fr.box[:, 4] >= pb[0] - 0.5) & (fr.box[:, 4] <= pb[1] + 0.5) & (fr.box[:, 5] >= pb[2] - 0.5) & (fr.box[:, 5] <= pb[3] + 0.5)
        order = [i for i in np.argsort(-a) if acc[i]]
        keep, used = [], np.zeros(256, bool)
        for i in order:
            if len(keep) >= 8:
                break
            if not (fr.mask[i] & used).any():
                keep.append(i)
                used |= fr.mask[i]
        if not keep:
            keep = [int(np.argmax(a))]
        cs = fr.box[keep][:, 4:6]
        lo, hi = cs.min(0), cs.max(0)
        ok = hi - lo > 0.5
        sib = np.where(ok, (fr.box[:, 4:6] - lo) / np.where(ok, hi - lo, 1.0), 0.5)
        if len(keep) > 1:
            below = (cs[None, :, :] < fr.box[:, None, 4:6] - 0.5).sum(1) + 0.5 * (np.abs(cs[None, :, :] - fr.box[:, None, 4:6]) <= 0.5).sum(1) - 0.5
            rank = np.clip(below / (len(keep) - 1), 0.0, 1.0)
            hw = np.stack([fr.box[:, 1] - fr.box[:, 0] + 1, fr.box[:, 3] - fr.box[:, 2] + 1], 1)
            med = np.median(hw[keep], 0)
            size = np.log(hw / med)
        else:
            rank = np.full((n, 2), 0.5)
            size = np.zeros((n, 2))
        return np.concatenate([g, np.clip(sib, -0.5, 1.5), rank, np.clip(size, -3, 3)], 1)

    def scores(self, fr, phrase, role):
        A, B = parse(phrase)
        ctx = role or "cand"
        pidx, parent_box = None, None
        if B is not None and self.compose == "head":
            sb, _, _, _, _ = self.scores(fr, B, None)
            if np.any(sb != 0):
                pidx = int(np.argmax(sb))
                parent_box = fr.box[pidx]
            ws = A
        else:
            ws = [w for w in (A + (words(B) if B is not None else [])) if w != "of"]
        G = self.geo_feats(fr, ws, ctx, parent_box)
        per = {}
        if role is not None:
            per["@role"] = self.lex.role_score(role, fr.app, fr.g_body)
        for w in ws:
            c = self.lex.ch.get(w)
            if c is None or w not in self.lex.p:
                continue
            per[w] = sum(self.lex.llr(w, ctx, fr.app if k == "app" else G, k) for k in (("app", "geo") if c == "both" else (c,)))
        tot = sum(per.values()) if per else np.zeros(len(fr.nodes))
        return np.asarray(tot, np.float64), per, parent_box, pidx, ws

    def resolve(self, fr, phrase, role):
        tot, per, _, pidx, _ = self.scores(fr, phrase, role)
        if not [k for k in per if k != "@role"]:
            return None, tot, per, pidx
        return int(np.argmax(tot)), tot, per, pidx


# ── training referents (generated pool, scripted grasp / release tokens) ──────────────────────────────────────────────
def training_occurrences():
    U = json.load(open(os.path.join(B13, "units.json"), encoding="utf-8"))
    occ = [dict(tid=u["layout"], frame=u["frame"], role=u["role"], phrase=u["phrase"], token=int(u["token"])) for u in U["train"]]
    return occ, U


def fit(occ, frames, K, compose, floor):
    lex = Lexicon(floor)
    binder = Binder(lex, K, compose)
    cand = {"app": np.concatenate([fr.app for fr in frames.values()]), "gbody": np.concatenate([fr.g_body for fr in frames.values()])}
    ref = {}
    for n, o in enumerate(occ):
        fr = frames[o["frame"]]
        ch = fr.chain(o["token"])
        bod = [i for i in ch if fr.is_body[i]]
        ref[n] = int(bod[0]) if bod else (int(ch[0]) if len(ch) else None)
    for it in range(N_EM + 1):
        rows, gcand = [], []
        for n, o in enumerate(occ):
            fr, r = frames[o["frame"]], ref[n]
            if r is None:
                continue
            chain = fr.chain(o["token"])
            A, B = parse(o["phrase"])
            parent_box = None
            if B is not None and compose == "head":
                if it == 0:
                    pi = fr.body_node.get(int(fr.body[r]))
                    parent_box = fr.bodybox[r]
                else:
                    _, _, parent_box, pi, _ = binder.scores(fr, o["phrase"], o["role"])
                if pi is not None:
                    for w in words(B):
                        rows.append((w, "cand", fr.app[pi], None, None, ("B", n), fr, None, pi, o["tid"], np.array([pi])))
                ws = A
            else:
                ws = [w for w in words(o["phrase"]) if w != "of"]
            if it == 0:
                g = fr.g_body if parent_box is None else rel(fr.box, parent_box)
                G = np.concatenate([g, np.full((len(fr.nodes), 4), 0.5), np.zeros((len(fr.nodes), 2))], 1)
            else:
                G = binder.geo_feats(fr, ws, o["role"], parent_box)
            gcand.append(G)
            rows.append((None, o["role"], fr.app[r], G[r], fr.g_body[r], n, fr, G, r, o["tid"], chain))
            for w in ws:
                rows.append((w, o["role"], fr.app[r], G[r], None, n, fr, G, r, o["tid"], chain))
        cand["geo"] = np.concatenate(gcand)
        lex.fit(rows, cand)
        if it == N_EM:
            break
        for n, o in enumerate(occ):
            fr = frames[o["frame"]]
            ch = fr.chain(o["token"])
            if len(ch):
                tot, _, _, _, _ = binder.scores(fr, o["phrase"], o["role"])
                ref[n] = int(ch[np.argmax(tot[ch])])
    return lex, binder, ref


def referent_ll(binder, occ, frames):
    ll = []
    for o in occ:
        fr = frames[o["frame"]]
        ch = fr.chain(o["token"])
        if not len(ch):
            continue
        tot, _, _, _, _ = binder.scores(fr, o["phrase"], o["role"])
        ll.append(chain_lp(tot, ch))
    return float(np.mean(ll))


def build_frames(fs, idx, feats, pca, K):
    out = {}
    for f in fs:
        if f not in out:
            s, i = idx[f]
            out[f] = Frame(pca(np.asarray(feats[s][i], np.float32)), K)
    return out


def _sel_job(args):
    K, compose, floor, occ, (idx, feats, pca) = args
    frames = build_frames([o["frame"] for o in occ], idx, feats, pca, K)
    tr = [o for o in occ if not heldout_layout(o["tid"])]
    ho = [o for o in occ if heldout_layout(o["tid"])]
    _, binder, _ = fit(tr, frames, K, compose, floor)
    return dict(K=K, compose=compose, floor=floor, ll_heldout=referent_ll(binder, ho, frames), ll_train=referent_ll(binder, tr, frames))


def predict(binder, units, idx, feats, pca, K):
    out, cache = [], {}
    for u in units:
        f = u["frame"]
        if f not in cache:
            s, i = idx[f]
            cache = {f: Frame(pca(np.asarray(feats[s][i], np.float32)), K)}
        fr = cache[f]
        j, tot, per, pidx = binder.resolve(fr, u["phrase"], u["role"])
        out.append(dict(peak=int(fr.peak[j]) if j is not None else -1, chosen=-1 if j is None else int(j),
                        parent=-1 if pidx is None else int(pidx), cand_peak=fr.peak.astype(np.int16),
                        cand_mask=np.packbits(fr.mask, axis=1), score=tot.astype(np.float32),
                        per={w: np.asarray(v, np.float32) for w, v in per.items()}, is_body=fr.is_body, n_cand=len(fr.nodes)))
    return out


def score_units(units, pr):
    import s1_read as SR
    hits = {}
    for u, p in zip(units, pr):
        z = np.load(os.path.join(ROOT, u["frame"]), allow_pickle=True)
        names, cov = SR.masks(z)
        bodies = [str(b) for b in z["body_names"]]
        tg = u["targets"][0]
        comp, _ = SR.competitors(tg, names, bodies)
        h = SR.hit(p["peak"], cov[names.index(tg)], [cov[names.index(c)] for c in comp if c != tg])
        hits.setdefault(f"d{u['depth']}|{u['role']}", []).append(h)
    return {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(hits.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("siglip", "dino"), default="siglip")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    idx, feats = load_feats(a.arm)
    od = os.path.join(B13, a.arm)
    os.makedirs(od, exist_ok=True)
    occ, U = training_occurrences()
    pca = pickle.load(open(os.path.join(B1, a.arm + "_b11_posfree.pkl"), "rb"))      # B1.1's position-free code (LIBERO training frames)
    mp = os.path.join(od, "model.pkl")
    if os.path.exists(mp):
        M = pickle.load(open(mp, "rb"))
        lex, K, compose, floor = M["lex"], M["K"], M["compose"], M["floor"]
        binder = Binder(lex, K, compose)
    else:
        import multiprocessing as mpc
        jobs = [(K, c, fl, occ, (idx, feats, pca)) for K in GRID_K for c in COMPOSE for fl in FLOORS]
        with mpc.get_context("fork").Pool(a.procs) as pool:
            table = pool.map(_sel_job, jobs)
        for t in table:
            print(f"[b1.3 bind] select {a.arm} K {t['K']} {t['compose']} floor {t['floor']}: held-out-layout referent ll "
                  f"{t['ll_heldout']:.3f} (train {t['ll_train']:.3f})", flush=True)
        best = max(table, key=lambda t: t["ll_heldout"])
        K, compose, floor = best["K"], best["compose"], best["floor"]
        frames = build_frames([o["frame"] for o in occ], idx, feats, pca, K)
        # held-out-layout smoke model (2/3 of the training layouts) BEFORE the final fit
        tr = [o for o in occ if not heldout_layout(o["tid"])]
        _, b_sm, _ = fit(tr, frames, K, compose, floor)
        lex, binder, ref = fit(occ, frames, K, compose, floor)
        # fresh linear probe of absolute (row, col) from the 32-d appearance code on generated TRAINING frames (diagnostic)
        tf = sorted({o["frame"] for o in occ})
        X = np.concatenate([pca(np.asarray(feats[idx[f][0]][idx[f][1]], np.float32)) for f in tf])
        tok = np.tile(np.arange(256), len(tf))
        Y = np.stack([tok // 16, tok % 16], 1).astype(np.float64)
        fid = np.repeat(np.arange(len(tf)), 256) % 2 == 0
        A_ = np.column_stack([np.ones(len(X)), X])
        bcoef, *_ = np.linalg.lstsq(A_[fid], Y[fid], rcond=None)
        res = Y[~fid] - A_[~fid] @ bcoef
        r2 = (1 - (res ** 2).sum(0) / ((Y[~fid] - Y[~fid].mean(0)) ** 2).sum(0)).tolist()
        word_occ = {}
        for n, o in enumerate(occ):
            for w in set(words(o["phrase"])):
                word_occ.setdefault(w, []).append(n)
        json.dump(dict(arm=a.arm, chosen=best, table=table, n_occ=len(occ), n_frames=len(frames), n_layouts=len({o["tid"] for o in occ}),
                       dl=lex.dl, gain=lex.gain, channels=lex.ch, n_words=len(lex.p), n_content=sum(1 for c in lex.ch.values() if c),
                       proto_geo={w: lex.p[w]["geo"][0].tolist() for w in lex.p if "geo" in lex.p[w]},
                       proto_geo_var={w: lex.p[w]["geo"][1].tolist() for w in lex.p if "geo" in lex.p[w]},
                       lexicon_floats=lex.nfloats(), pca_floats=affine_floats(pca), posfree_probe_r2_generated=r2,
                       phrases=sorted({(o["role"], o["phrase"]) for o in occ}),
                       word_occ={w: [occ[n]["tid"] + "|" + occ[n]["role"] + "|" + occ[n]["phrase"] for n in v] for w, v in word_occ.items()}),
                  open(os.path.join(od, "select.json"), "w", encoding="utf-8"), indent=1, default=float)
        pickle.dump(dict(lex=lex, K=K, compose=compose, floor=floor, smoke_binder=b_sm), open(mp, "wb"))
        print(f"[b1.3 bind] {a.arm}: chosen K {K} {compose} floor {floor}; {len(occ)} referents on {len(frames)} frames; {len(lex.p)} words, "
              f"content {sum(1 for c in lex.ch.values() if c)}; lexicon {lex.nfloats()} floats; posfree code {affine_floats(pca)} floats; "
              f"probe R^2 on generated frames {r2}", flush=True)
        for w in sorted(lex.p):
            print(f"[b1.3 bind]   {w:12s} -> {lex.ch[w]}  DL {json.dumps({k: round(v, 1) for k, v in lex.dl[w].items()})}", flush=True)
    if a.smoke:
        M = pickle.load(open(mp, "rb"))
        rng = np.random.default_rng(0)
        ho = [u for u in U["train"] if heldout_layout(u["layout"])]
        units = [ho[i] for i in sorted(rng.choice(len(ho), min(800, len(ho)), replace=False))]
        pr = predict(M["smoke_binder"], units, idx, feats, pca, K)
        sm = score_units(units, pr)
        print("[b1.3 bind] SMOKE (held-out TRAINING layouts, model fitted on the other 2/3): " + json.dumps(sm), flush=True)
        allh = [h for v in sm.values() for h in [v["mean"]] * v["n"]]
        json.dump(dict(cells=sm, mean=float(np.mean(allh)), n=len(units)), open(os.path.join(od, "smoke_train.json"), "w"))
    if a.test or a.dev:
        from lifelong_ood import ledger
        assert any(r.get("tag") == "pb1_3" for r in ledger.load("m3_prereg")), "pre-register first (experiments/prereg/pb1_3.py)"
    if a.test:
        pr = predict(binder, U["test"], idx, feats, pca, K)
        pickle.dump(dict(units=[dict(frame=u["frame"], role=u["role"], phrase=u["phrase"]) for u in U["test"]], pred=pr),
                    open(os.path.join(od, "pred_test.pkl"), "wb"))
        print(f"[b1.3 bind] test: {len(pr)} units predicted", flush=True)
    if a.dev:
        lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
        pr = predict(binder, lab["dev"], idx, feats, pca, K)
        pickle.dump(dict(units=[dict(frame=u["frame"], role=u["role"], phrase=u["phrase"]) for u in lab["dev"]], pred=pr),
                    open(os.path.join(od, "pred_dev.pkl"), "wb"))
        print(f"[b1.3 bind] dev: {len(pr)} units predicted", flush=True)


if __name__ == "__main__":
    main()

"""PLAN_BINDER B1.2 (docs/PLAN_BINDER.md section 11): b1_1_bind.py (B1.1, section 9) with three changes: (1) channel assignment by
the description length of the referent's IDENTITY among its frame's candidates, per channel and for BOTH channels jointly (a
two-part code; null = log #candidates = a function word); (2) the ordinal / set-position siblings are the regions every TYPE word
of the phrase accepts (diagonal Mahalanobis within the chi-square 99% radius), not a DELTA-nats window; (3) a third arm "both"
(appearance = the position-free siglip and dino codes concatenated).  Outputs go to results/bind/b1/<arm>_b12.

B1: the unsupervised prototype binder.  CPU, chunks env (numpy + scikit-learn).

NOTHING here reads a simulator label.  Inputs: frozen patch features (results/bind/s1/feats, results/bind/b1/feats), the TRAINING
cells' demos with their contact-event referents (results/bind/b1/demos, b1_demos.py: frame 0 of every recorded demo and the
projected gripper token at each grasp / release) and the role phrases (the training instruction split as S1 splits it).

  regions   per frame: PCA (D_APP dims, whitened; fitted on TRAINING frames only) of the patch features, then Ward agglomerative
            clustering of the 256 tokens under 4-neighbour grid connectivity -> a merge tree.  Cutting it at K clusters gives the
            BODIES; every node inside a body is a PART (the recursive split).  Candidates: every tree node of 2..128 tokens.
  channels  appearance = the node's mean PCA feature; geometry = [centroid row / col and log height / width relative to its
            PARENT's box (4)] + [its centroid's position among the phrase's instance set (2)].  Parent = the region the parse
            resolves ("A of B": B), else the node's body (the image above body level).  Instance set = the non-overlapping
            candidates whose appearance evidence for the phrase is within DELTA nats of the best.
  atoms     ROLE: a diagonal Gaussian over the role's referents (object / goal) in both channels, scored against all candidates.
            WORDS: every word of a role phrase gets a diagonal-Gaussian prototype in BOTH channels from its co-occurring referents
            (cross-situational; hard EM over which tree node at the referent token is the referent, N_EM rounds).
            MDL: per word and channel, the nats its expert saves in coding WHICH candidate of the frame is the referent (over the
            role expert alone), minus the parameter cost (d log n).  The word goes to the channel with the larger saving, or is a FUNCTION word
            when neither saving is positive.  Words in B of "A of B" are fitted on B's resolved region against the candidates.
  binding   score(node | role, phrase) = role expert + sum over content words of log N(word) - log N(role referent code), in the
            word's channel: a product of experts.  "A of B" (head composition): B is resolved first by its words alone, then A
            scores every node with geometry relative to B.  The chosen node's peak = its member token nearest its centroid.
            Unknown words carry no evidence.
  selection K in {8, 16, 32} x composition {head, flat} x DELTA {2, 8}: the setting with the highest held-out log-likelihood of
            the training referents (cells split 2:1 by a fixed hash; training demos only).  Fixed before any dev unit is scored.

    python experiments/b1_bind.py --arm siglip --smoke        select + fit, score S1's TRAINING units only
    python experiments/b1_bind.py --arm siglip --dev          (needs the m3_prereg pb1 row) predict the 240 dev units
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
B1 = os.path.join(ROOT, "results", "bind", "b1")
D_APP, MIN_SZ, MAX_SZ, N_EM, MOVE_M = 32, 2, 128, 3, 0.05
MDL_SWEEPS = 0                    # no conditional sweeps (variants tried on training smokes only, dropped)
# design variants (frozen by the training-unit smoke, docs/PLAN_BINDER.md 7): the code words are scored against ("cand": all candidates;
# "role": the role's referents), whether a role expert is an atom of the score, and what MDL codes ("feature": the referent's features;
# "identity": which candidate is the referent)
NULL_CODE, ROLE_EXPERT, MDL_CODE = os.environ.get("B1_NULL", "role"), os.environ.get("B1_ROLE", "1") == "1", os.environ.get("B1_MDL", "feature")   # B1.1 FROZEN: variant V2
GRID_K = (8, 16, 32)
COMPOSE = ("head", "flat")
DELTAS = (0.0,)  # B1.2: unused (siblings = regions the type words accept); was          # B1.1: instance-set width in nats; at 4 the set was almost always one region (training smoke)
FLOORS = (0.0, 0.05, 0.2)         # variance floor, x the candidate background variance (overconfidence across cells)
VERB = re.compile(r"^(open|close|turn on|turn off|pick up|push|put|place|stack)\s+")
SKIP_CELLS = ("libero_10:1", "libero_10:7")            # "both X and Y": S1 skips them too (one phrase, two referents)
RR, CC = np.divmod(np.arange(256), 16)
IMAGE_BOX = np.array([0, 15, 0, 15, 7.5, 7.5], np.float64)


# ── features ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_feats(arm):
    fe = os.path.join(B1, "feats")
    if arm == "siglip":
        src = {"train": (os.path.join(S1, "feats", "train_img.npy"), os.path.join(S1, "feats", "train_frames.json")),
               "dev": (os.path.join(S1, "feats", "dev_img.npy"), os.path.join(S1, "feats", "dev_frames.json")),
               "demos": (os.path.join(fe, "demos_img.npy"), os.path.join(fe, "demos_frames.json"))}
    else:
        src = {k: (os.path.join(fe, f"dino_{k}.npy"), os.path.join(fe, f"dino_{k}_frames.json")) for k in ("train", "dev", "demos")}
    arr = {k: np.load(a, mmap_mode="r") for k, (a, _) in src.items()}
    idx = {f: (k, i) for k, (_, j) in src.items() for i, f in enumerate(json.load(open(j)))}
    return idx, arr


class _Cat:
    """B1.2 concat arm: per-token [siglip 2048 | dino 384] features (same frame order, asserted)."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0], 256, a.shape[-1] + b.shape[-1])

    def __getitem__(self, i):
        return np.concatenate([np.asarray(self.a[i], np.float32), np.asarray(self.b[i], np.float32)], -1)


class CatCode:
    """appearance = [position-free siglip code | position-free dino code] (64-d)."""

    def __init__(self, ca, cb, da=2048):
        self.ca, self.cb, self.da = ca, cb, da
        self.W = np.zeros((da + cb.V.shape[0], ca.W.shape[1] + cb.W.shape[1]))   # size bookkeeping: the composed affine map
        self.mu = np.zeros(ca.W.shape[1] + cb.W.shape[1])

    def __call__(self, X):
        X = np.asarray(X)
        return np.concatenate([self.ca(X[..., :self.da]), self.cb(X[..., self.da:])], -1)


def affine_floats(c):
    """floats of the appearance code as ONE composed affine map (input dims x 32 + 32 per tower): PCA, nullspace projection and
    whitening are all linear.  (B1.1 reported W + mu only, which undercounted.)"""
    if isinstance(c, CatCode):
        return affine_floats(c.ca) + affine_floats(c.cb)
    if hasattr(c, "V"):
        return int(c.V.shape[0] * c.W.shape[1] + c.W.shape[1])
    return int(c.W.size + c.mu.size)


def load_arm(arm):
    if arm != "both":
        return load_feats(arm)
    i1, f1 = load_feats("siglip")
    i2, f2 = load_feats("dino")
    assert i1 == i2
    return i1, {k: _Cat(f1[k], f2[k]) for k in f1}


class PCA:
    def __init__(self, X, d):
        X = np.asarray(X, np.float64)
        self.mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = Vt[:d].T / (S[:d] / np.sqrt(len(X) - 1))
        self.d = d

    def __call__(self, X):
        return (np.asarray(X, np.float64) - self.mu) @ self.W


class PosFreePCA:
    """B1.1: a POSITION-FREE appearance code, fitted on TRAINING frames only.
    x -> PCA to D_PRE dims -> iterative nullspace projection (INLP): fit the linear map to the token's (row, col) on half of the
    training frames, project its 2-d column space out, refit a FRESH probe, repeat until the fresh probe's held-out R^2 (the
    other half of the training frames) is < R2_STOP for both row and col -> PCA (whitened) to D_APP dims."""

    def __init__(self, X, frame_id, d=D_APP, d_pre=256, r2_stop=0.05, max_iter=120):
        X = np.asarray(X, np.float64)
        tok = np.tile(np.arange(256), len(X) // 256)
        Y = np.stack([tok // 16, tok % 16], 1).astype(np.float64)
        self.mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[:d_pre].T
        Z = (X - self.mu) @ self.V
        fit = (frame_id % 2) == 0
        self.Q = np.zeros((d_pre, 0))
        self.log = []

        def probe(Zp):
            A = np.column_stack([np.ones(len(Zp)), Zp])
            b, *_ = np.linalg.lstsq(A[fit], Y[fit], rcond=None)
            res = Y[~fit] - A[~fit] @ b
            r2 = 1 - (res ** 2).sum(0) / ((Y[~fit] - Y[~fit].mean(0)) ** 2).sum(0)
            return b[1:], r2
        for it in range(max_iter):
            Zp = Z - (Z @ self.Q) @ self.Q.T
            b, r2 = probe(Zp)
            self.log.append(dict(iter=it, removed_dims=int(self.Q.shape[1]), r2_row=float(r2[0]), r2_col=float(r2[1])))
            print(f"[b1.1 posfree] iter {it}: removed {self.Q.shape[1]} dims, fresh-probe held-out R^2 row {r2[0]:.3f} col {r2[1]:.3f}", flush=True)
            if max(r2) < r2_stop:
                break
            Bq = b - self.Q @ (self.Q.T @ b)
            q, _ = np.linalg.qr(Bq)
            self.Q = np.concatenate([self.Q, q], 1)
        self.final_r2 = self.log[-1]
        Zp = Z - (Z @ self.Q) @ self.Q.T
        self.mu2 = Zp.mean(0)
        _, S2, Vt2 = np.linalg.svd(Zp - self.mu2, full_matrices=False)
        self.W = Vt2[:d].T / (S2[:d] / np.sqrt(len(Zp) - 1))
        self.d = d
        A = np.column_stack([np.ones(len(Zp)), (Zp - self.mu2) @ self.W])     # the 32-d code itself, fresh probe
        b, *_ = np.linalg.lstsq(A[fit], Y[fit], rcond=None)
        res = Y[~fit] - A[~fit] @ b
        r2 = 1 - (res ** 2).sum(0) / ((Y[~fit] - Y[~fit].mean(0)) ** 2).sum(0)
        self.final_r2_app = dict(r2_row=float(r2[0]), r2_col=float(r2[1]))
        print(f"[b1.1 posfree] final 32-d appearance code: fresh-probe held-out R^2 row {r2[0]:.3f} col {r2[1]:.3f}", flush=True)

    def __call__(self, X):
        Z = (np.asarray(X, np.float64) - self.mu) @ self.V
        Z = Z - (Z @ self.Q) @ self.Q.T
        return (Z - self.mu2) @ self.W


# ── regions ──────────────────────────────────────────────────────────────────────────────────────────────────────────
_CONN = None


def merge_tree(Z):
    global _CONN
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.image import grid_to_graph
    if _CONN is None:
        _CONN = grid_to_graph(16, 16)
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=0.0, linkage="ward", connectivity=_CONN, compute_full_tree=True)
    ac.fit(Z)
    return ac.children_


def box(m):
    r, c = RR[m], CC[m]
    return np.array([r.min(), r.max(), c.min(), c.max(), r.mean(), c.mean()], np.float64)


def rel(bx, pb):
    """bx [n, 6] relative to parent boxes pb [6] or [n, 6] -> [n, 4]."""
    pb = np.broadcast_to(pb, bx.shape)
    h, w = pb[:, 1] - pb[:, 0] + 1, pb[:, 3] - pb[:, 2] + 1
    return np.stack([(bx[:, 4] - pb[:, 0]) / h, (bx[:, 5] - pb[:, 2]) / w, np.log((bx[:, 1] - bx[:, 0] + 1) / h),
                     np.log((bx[:, 3] - bx[:, 2] + 1) / w)], 1)


class Frame:
    """The candidate regions of one frame (features only)."""

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
        self.bodybox = np.stack([bb.get(b, IMAGE_BOX) for b in self.body])
        self.body_node = {b: self.nodes.index(b) for b in bb if b in self.nodes}
        self.g_body = rel(self.box, self.bodybox)

    def chain(self, tok):
        return np.where(self.mask[:, tok])[0]


# ── lexicon ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def gauss_ll(X, mu, var):
    return -0.5 * (((X - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(-1)


def fit_gauss(X, prior_var, floor=0.0):
    X = np.asarray(X, np.float64)
    n = len(X)
    var = (X.var(0) * n + prior_var) / (n + 1) + 1e-4 * prior_var
    return X.mean(0), np.maximum(var, floor * prior_var), n


def pog_ll(x, protos):
    """log density of x under the normalised product of diagonal Gaussians [(mu, var)]."""
    lam = sum(1.0 / v for _, v in protos)
    mu = sum(m / v for m, v in protos) / lam
    return gauss_ll(x, mu, 1.0 / lam)


class Lexicon:
    def __init__(self, floor=0.0):
        self.p, self.ch, self.gain, self.role, self.bg, self.floor = {}, {}, {}, {}, {}, floor

    def role_score(self, role, app, gbody):
        s = 0.0
        for k, X in (("app", app), ("gbody", gbody)):
            mu, var, _ = self.role[role][k]
            bm, bv = self.bg[k]
            s = s + gauss_ll(X, mu, var) - gauss_ll(X, bm, bv)
        return s

    def llr(self, w, ctx, X, c=None):
        """word w's evidence in channel c (default: its channel), against the context's null code (a role's referents, or "cand")."""
        c = c or self.ch[w]
        mu, var, _ = self.p[w][c]
        nm, nv = self.null[ctx][c]
        return gauss_ll(X, mu, var) - gauss_ll(X, nm, nv)

    def fit(self, occ, cand_bg):
        """occ: rows (word, ctx, app x, geo x, gbody x, group); ctx in {object, goal, cand}; group = the coded phrase (an occurrence,
        or the B phrase of one).  Word None = the referent row of a role code.  cand_bg: candidate features {"app", "geo", "gbody"}.

        Channel assignment = MDL on the referent's IDENTITY: per word and channel, the nats its expert saves in coding which of
        the frame's candidates is the referent (log-softmax of role expert + word expert vs role expert alone; uniform for B
        words), summed over its occurrences, minus its parameter cost d log n.  The larger saving wins; none positive = a
        function word.  (Two coding-the-features variants were tried on training smokes and dropped; see PLAN_BINDER 7.)"""
        self.bg = {k: (X.mean(0), X.var(0) + 1e-6) for k, X in cand_bg.items()}
        self.null = {"cand": {"app": self.bg["app"], "geo": self.bg["geo"]}}
        self.role = {}
        for r in ("object", "goal"):
            R = [o for o in occ if o[0] is None and o[1] == r]
            self.role[r] = {"app": fit_gauss([o[2] for o in R], self.bg["app"][1], self.floor),
                            "gbody": fit_gauss([o[4] for o in R], self.bg["gbody"][1], self.floor)}
            G = np.stack([o[3] for o in R])
            self.null[r] = ({"app": self.role[r]["app"][:2], "geo": (G.mean(0), np.maximum(G.var(0) + 1e-6, self.floor * self.bg["geo"][1]))}
                            if NULL_CODE == "role" else self.null["cand"])
        self.p, self.ch, self.gain = {}, {}, {}
        by, grp = {}, {}
        for o in occ:
            if o[0] is not None:
                by.setdefault(o[0], []).append(o)
                grp.setdefault(o[5], set()).add(o[0])
        for w, os_ in by.items():
            self.p[w] = {}
            for k, j in (("app", 2), ("geo", 3)):
                rows = [o for o in os_ if o[j] is not None]
                if rows:
                    self.p[w][k] = fit_gauss(np.stack([o[j] for o in rows]), self.null["cand"][k][1], self.floor)

        base_cache = {}

        def lsm(x):
            m = x.max()
            return x - (m + np.log(np.exp(x - m).sum()))
        # B1.2 channel assignment: description length of the referent's IDENTITY among its frame's candidates (a two-part code)
        #   DL(w, c)    = sum_i -log softmax_candidates(log N(x^c; prototype_w^c))[referent_i] + 0.5 k_c log n,  k_c = 2 dims(c)
        #   DL(w, both) = the same with the summed log-likelihood of both channels, k = k_app + k_geo
        #   DL(w, null) = sum_i log(#candidates_i)
        # the word takes the cheapest code; null = a FUNCTION word.  (B words have no geometry row: their geometry code is null.)
        self.dl = {}
        for w in sorted(by):
            n = len(by[w])
            dl = {"null": 0.0, "app": 0.0, "geo": 0.0, "both": 0.0}
            for o in by[w]:
                fr, G, r = o[6], o[7], o[8]
                la = lsm(gauss_ll(fr.app, *self.p[w]["app"][:2]))[r] if "app" in self.p[w] else -np.log(len(fr.nodes))
                nul = np.log(len(fr.nodes))
                dl["null"] += nul
                dl["app"] -= la
                if G is not None and "geo" in self.p[w]:
                    lgg = gauss_ll(G, *self.p[w]["geo"][:2])
                    dl["geo"] -= lsm(lgg)[r]
                    dl["both"] -= lsm(gauss_ll(fr.app, *self.p[w]["app"][:2]) + lgg)[r] if "app" in self.p[w] else -lsm(lgg)[r]
                else:
                    dl["geo"] += nul
                    dl["both"] -= la
            ka = 2 * len(self.p[w]["app"][0]) if "app" in self.p[w] else 0
            kg = 2 * len(self.p[w]["geo"][0]) if "geo" in self.p[w] else 0
            dl["app"] += 0.5 * ka * np.log(max(n, 2))
            dl["geo"] += 0.5 * kg * np.log(max(n, 2)) if kg else np.inf
            dl["both"] += 0.5 * (ka + kg) * np.log(max(n, 2)) if kg else np.inf
            self.dl[w] = {k: float(v) for k, v in dl.items()}
            self.gain[w] = {k: float(dl["null"] - v) for k, v in dl.items() if k != "null"}
            best = min(dl, key=dl.get)
            self.ch[w] = None if best == "null" else best
        self.gain0 = dict(self.gain)

    def nfloats(self):
        n = sum(2 * len(self.p[w][k][0]) for w, c in self.ch.items() if c for k in (("app", "geo") if c == "both" else (c,)))
        n += sum(2 * len(self.role[r][k][0]) for r in self.role for k in self.role[r])
        return n


# ── parse ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def words(p):
    return re.findall(r"[a-z]+", p.lower())


def parse(p):
    """-> (A words, B phrase or None): the FIRST "of" splits "A of B" (right-branching: B may itself contain "of")."""
    ws = words(p)
    if "of" in ws:
        i = ws.index("of")
        return ws[:i], " ".join(ws[i + 1:])
    return ws, None


# ── binding ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def chi2_99(d):
    """99% quantile of chi-square(d) (Wilson-Hilferty)."""
    return d * (1 - 2 / (9 * d) + 2.326 * np.sqrt(2 / (9 * d))) ** 3


class Binder:
    def __init__(self, lex, K, compose, delta):
        self.lex, self.K, self.compose, self.delta = lex, K, compose, delta

    def app_evidence(self, fr, ws, ctx):
        s = np.zeros(len(fr.nodes))
        for w in ws:
            if self.lex.ch.get(w) in ("app", "both"):
                s += self.lex.llr(w, ctx, fr.app, "app")
        return s

    def geo_feats(self, fr, ws, ctx, parent_box):
        g = fr.g_body if parent_box is None else rel(fr.box, parent_box)
        a = self.app_evidence(fr, ws, ctx)
        if not np.any(a != 0):
            return np.concatenate([g, np.full((len(fr.nodes), 4), 0.5)], 1)
        acc = np.ones(len(fr.nodes), bool)
        for w in ws:
            if self.lex.ch.get(w) in ("app", "both"):
                mu, var, _ = self.lex.p[w]["app"]
                acc &= (((fr.app - mu) ** 2) / var).sum(1) <= chi2_99(len(mu))
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
        # B1.1 ordinal feature: the node's rank among the instance set along image y (row) and x (col), in [0, 1]
        # (fraction of the set's instances strictly before it, plus half the ties); 0.5 for a singleton set
        if len(keep) > 1:
            below = (cs[None, :, :] < fr.box[:, None, 4:6] - 0.5).sum(1) + 0.5 * (np.abs(cs[None, :, :] - fr.box[:, None, 4:6]) <= 0.5).sum(1) - 0.5
            rank = np.clip(below / (len(keep) - 1), 0.0, 1.0)
        else:
            rank = np.full((len(fr.nodes), 2), 0.5)
        return np.concatenate([g, np.clip(sib, -0.5, 1.5), rank], 1)

    def scores(self, fr, phrase, role):
        """-> (total [n], per-atom {name: [n]}, parent box or None, parent node or None, A words).  role None: a B phrase."""
        A, B = parse(phrase)
        ctx = role or "cand"
        pidx, parent_box = None, None
        if B is not None and self.compose == "head":
            sb, _, _, _, _ = self.scores(fr, B, None)
            if np.any(sb != 0):
                pidx = int(np.argmax(sb))
                parent_box = fr.box[pidx]
            else:
                parent_box = IMAGE_BOX
            ws = A
        else:
            ws = [w for w in (A + (words(B) if B is not None else [])) if w != "of"]
        G = self.geo_feats(fr, ws, ctx, parent_box)
        per = {}
        if role is not None:
            if ROLE_EXPERT:
                per["@role"] = self.lex.role_score(role, fr.app, fr.g_body)
        for w in ws:
            c = self.lex.ch.get(w)
            if c is None:
                continue
            per[w] = sum(self.lex.llr(w, ctx, fr.app if k == "app" else G, k) for k in (("app", "geo") if c == "both" else (c,)))
        tot = sum(per.values()) if per else np.zeros(len(fr.nodes))
        return np.asarray(tot, np.float64), per, parent_box, pidx, ws

    def resolve(self, fr, phrase, role):
        tot, per, _, pidx, _ = self.scores(fr, phrase, role)
        if not [k for k in per if k != "@role"]:
            return None, tot, per, pidx
        return int(np.argmax(tot)), tot, per, pidx


# ── training referents ───────────────────────────────────────────────────────────────────────────────────────────────
def transports(ev):
    """(grasp, release or None) pairs; a release within MOVE_M of its grasp is a regrasp, not a placement, and is dropped."""
    out, k = [], 0
    while k < len(ev):
        if ev[k]["kind"] == "grasp":
            r = ev[k + 1] if k + 1 < len(ev) and ev[k + 1]["kind"] == "release" else None
            if r is None or np.linalg.norm(np.subtract(r["eef"], ev[k]["eef"])) >= MOVE_M:
                out.append((ev[k], r))
            k += 2 if r is not None else 1
        else:
            k += 1
    return out


def training_occurrences():
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    C = json.load(open(os.path.join(B1, "cells.json"), encoding="utf-8"))["cells"]
    occ, log = [], {}
    for c in C:
        if c["tid"] in SKIP_CELLS:
            log[c["tid"]] = "skipped: both X and Y"
            continue
        f = os.path.join(B1, "demos", c["tid"].replace(":", "_") + ".npz")
        if not os.path.exists(f):
            log[c["tid"]] = "no demos"
            continue
        evs = json.loads(str(np.load(f, allow_pickle=True)["events"]))
        steps = [SL.split_step(s) for s in P.steps_of(c["lang"])]
        st = dict(demos=len(evs), used=0, short=0, extra=0)
        for k, ev in enumerate(evs):
            pairs = transports(ev)
            if len(pairs) < len(steps):
                st["short"] += 1
                continue
            st["extra"] += len(pairs) > len(steps)
            st["used"] += 1
            key = f"{c['tid']}#{k}"
            for j, (o, g) in enumerate(steps):
                gr, re_ = pairs[j]
                occ.append(dict(tid=c["tid"], frame=key, role="object", phrase=SL.norm(VERB.sub("", o.strip())), token=gr["token"]))
                if g and re_ is not None:
                    occ.append(dict(tid=c["tid"], frame=key, role="goal", phrase=g, token=re_["token"]))
        log[c["tid"]] = st
    return occ, log


def heldout_cell(tid):
    return int(hashlib.sha256(("b1-select" + tid).encode()).hexdigest(), 16) % 3 == 0


def fit(occ, frames, K, compose, delta, floor):
    lex = Lexicon(floor)
    binder = Binder(lex, K, compose, delta)
    cand = {"app": np.concatenate([fr.app for fr in frames.values()]), "gbody": np.concatenate([fr.g_body for fr in frames.values()])}
    ref = {}
    for n, o in enumerate(occ):                                       # init: the body at the token (else its smallest node)
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
                        rows.append((w, "cand", fr.app[pi], None, None, ("B", n), fr, None, pi))
                ws = A
            else:
                ws = [w for w in words(o["phrase"]) if w != "of"]
            if it == 0:
                g = fr.g_body if parent_box is None else rel(fr.box, parent_box)
                G = np.concatenate([g, np.full((len(fr.nodes), 4), 0.5)], 1)
            else:
                G = binder.geo_feats(fr, ws, o["role"], parent_box)
            gcand.append(G)
            rows.append((None, o["role"], fr.app[r], G[r], fr.g_body[r], n, fr, G, r))
            for w in ws:
                rows.append((w, o["role"], fr.app[r], G[r], None, n, fr, G, r))
        cand["geo"] = np.concatenate(gcand)
        lex.fit(rows, cand)
        if it == N_EM:
            break
        for n, o in enumerate(occ):                                   # hard EM: the chain node the phrase explains best
            fr = frames[o["frame"]]
            ch = fr.chain(o["token"])
            if len(ch):
                tot, _, _, _, _ = binder.scores(fr, o["phrase"], o["role"])
                ref[n] = int(ch[np.argmax(tot[ch])])
    return lex, binder, ref


def referent_ll(binder, occ, frames, ref=None):
    """Mean log-softmax over candidates of the best-scoring node at the referent token (the token's chain)."""
    ll = []
    for n, o in enumerate(occ):
        fr = frames[o["frame"]]
        ch = fr.chain(o["token"])
        if not len(ch):
            continue
        tot, _, _, _, _ = binder.scores(fr, o["phrase"], o["role"])
        m = tot.max()
        lse = m + np.log(np.exp(tot - m).sum())
        ll.append(tot[ch].max() - lse)
    return float(np.mean(ll))


def build_frames(occ, idx, feats, pca, K):
    out = {}
    for o in occ:
        if o["frame"] not in out:
            s, i = idx[o["frame"]]
            out[o["frame"]] = Frame(pca(np.asarray(feats[s][i], np.float32)), K)
    return out


def _sel_job(args):
    K, compose, delta, floor, occ, idx_feats_pca = args
    idx, feats, pca = idx_feats_pca
    frames = build_frames(occ, idx, feats, pca, K)
    tr = [o for o in occ if not heldout_cell(o["tid"])]
    ho = [o for o in occ if heldout_cell(o["tid"])]
    _, binder, _ = fit(tr, frames, K, compose, delta, floor)
    return dict(K=K, compose=compose, delta=delta, floor=floor, ll_heldout=referent_ll(binder, ho, frames), ll_train=referent_ll(binder, tr, frames))


def predict(binder, units, idx, feats, pca, K, roles=True):
    out, cache = [], {}
    for u in units:
        f = u["frame"]
        if f not in cache:
            s, i = idx[f]
            cache = {f: Frame(pca(np.asarray(feats[s][i], np.float32)), K)}
        fr = cache[f]
        j, tot, per, pidx = binder.resolve(fr, u["phrase"], u["role"] if roles else None)
        out.append(dict(peak=int(fr.peak[j]) if j is not None else -1, chosen=-1 if j is None else int(j),
                        parent=-1 if pidx is None else int(pidx), cand_peak=fr.peak.astype(np.int16),
                        cand_mask=np.packbits(fr.mask, axis=1), score=tot.astype(np.float32),
                        per={w: np.asarray(v, np.float32) for w, v in per.items()}, is_body=fr.is_body, n_cand=len(fr.nodes)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("siglip", "dino", "both"), default="siglip")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--posfree-only", action="store_true")
    a = ap.parse_args()
    idx, feats = load_arm(a.arm)
    od = os.path.join(B1, a.arm + "_b12" + os.environ.get("B1_TAG", ""))
    os.makedirs(od, exist_ok=True)
    occ, log = training_occurrences()
    mp = os.path.join(od, "model.pkl")
    pf = os.path.join(B1, a.arm + "_b11_posfree.pkl")              # the position-free code is shared by every variant of an arm
    if a.arm == "both":                                            # B1.2 reuses B1.1's training-only position-free codes
        pca = CatCode(pickle.load(open(os.path.join(B1, "siglip_b11_posfree.pkl"), "rb")), pickle.load(open(os.path.join(B1, "dino_b11_posfree.pkl"), "rb")))
    elif os.path.exists(pf):
        pca = pickle.load(open(pf, "rb"))
    elif not os.path.exists(mp):
        tA, tB = np.asarray(feats["train"][::4]), np.asarray(feats["demos"][::8])
        tr = np.concatenate([tA.reshape(-1, tA.shape[-1]), tB.reshape(-1, tB.shape[-1])]).astype(np.float32)
        fid = np.repeat(np.arange(len(tA) + len(tB)), 256)
        pca = PosFreePCA(tr, fid)                                  # TRAINING frames only (S1 training frames + training demos)
        del tr
        pickle.dump(pca, open(pf, "wb"))
        json.dump(dict(log=pca.log, final=pca.final_r2, final_app=pca.final_r2_app), open(os.path.join(B1, a.arm + "_b11_posfree.json"), "w"), indent=1)
    if a.posfree_only:
        return
    if os.path.exists(mp):
        M = pickle.load(open(mp, "rb"))
        lex, K, compose, delta, floor, pca = M["lex"], M["K"], M["compose"], M["delta"], M["floor"], M["pca"]
        binder = Binder(lex, K, compose, delta)
    else:
        import multiprocessing as mpc
        jobs = [(K, c, d, fl, occ, (idx, feats, pca)) for K in GRID_K for c in COMPOSE for d in DELTAS for fl in FLOORS]
        with mpc.get_context("fork").Pool(a.procs) as pool:
            table = pool.map(_sel_job, jobs)
        for t in table:
            print(f"[b1 bind] select {a.arm} K {t['K']} {t['compose']} delta {t['delta']} floor {t['floor']}: held-out-cell referent ll "
                  f"{t['ll_heldout']:.3f} (train {t['ll_train']:.3f})", flush=True)
        best = max(table, key=lambda t: t["ll_heldout"])
        K, compose, delta, floor = best["K"], best["compose"], best["delta"], best["floor"]
        frames = build_frames(occ, idx, feats, pca, K)
        lex, binder, ref = fit(occ, frames, K, compose, delta, floor)
        word_occ = {}
        for n, o in enumerate(occ):
            for w in set(words(o["phrase"])):
                word_occ.setdefault(w, []).append(n)
        json.dump(dict(arm=a.arm, chosen=best, table=table, occ_log=log, n_occ=len(occ), n_frames=len(frames),
                       dl=lex.dl,
                       proto_geo={w: lex.p[w]["geo"][0].tolist() for w in lex.p if "geo" in lex.p[w]},
                       proto_geo_var={w: lex.p[w]["geo"][1].tolist() for w in lex.p if "geo" in lex.p[w]},
                       channels=lex.ch, gain=lex.gain, gain_marginal=lex.gain0, n_words=len(lex.p), n_content=sum(1 for c in lex.ch.values() if c),
                       lexicon_floats=lex.nfloats(), pca_floats=affine_floats(pca),
                       phrases=sorted({(o["role"], o["phrase"]) for o in occ}),
                       word_occ={w: [occ[n]["tid"] + "|" + occ[n]["role"] + "|" + occ[n]["phrase"] for n in v] for w, v in word_occ.items()},
                       ref_sizes=[int(frames[o["frame"]].mask[ref[n]].sum()) if ref[n] is not None else 0 for n, o in enumerate(occ)]),
                  open(os.path.join(od, "select.json"), "w", encoding="utf-8"), indent=1, default=float)
        pickle.dump(dict(lex=lex, K=K, compose=compose, delta=delta, floor=floor, pca=pca), open(mp, "wb"))
        print(f"[b1 bind] {a.arm}: chosen K {K} {compose} delta {delta} floor {floor}; {len(occ)} referents on {len(frames)} demo frames; "
              f"{len(lex.p)} words, content {sum(1 for c in lex.ch.values() if c)}; lexicon {lex.nfloats()} floats, PCA {pca.W.size + pca.mu.size}",
              flush=True)
        for w in sorted(lex.p):
            print(f"[b1 bind]   {w:12s} -> {lex.ch[w]}  gain app {lex.gain[w].get('app', 0):.1f} geo {lex.gain[w].get('geo', 0):.1f}", flush=True)
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    if a.smoke:
        import s1_read as SR
        rng = np.random.default_rng(0)
        units = [lab["train"][i] for i in sorted(rng.choice(len(lab["train"]), 800, replace=False))]
        pr = predict(binder, units, idx, feats, pca, K)
        phr = {p for _, p in json.load(open(os.path.join(od, "select.json")))["phrases"]}
        hits = {}
        for u, p in zip(units, pr):
            z = np.load(os.path.join(ROOT, u["frame"]), allow_pickle=True)
            names, cov = SR.masks(z)
            bodies = [str(b) for b in z["body_names"]]
            h = 0.0
            for tg in u["targets"]:
                comp, _ = SR.competitors(tg, names, bodies)
                comp = [c for c in comp if c not in u["targets"]]
                h = max(h, SR.hit(p["peak"], cov[names.index(tg)], [cov[names.index(c)] for c in comp]))
            k = f"{u['role']}|{'demo_phrase' if u['phrase'] in phr else 'other_phrase'}"
            hits.setdefault(k, []).append(h)
        print("[b1 bind] SMOKE (S1 TRAINING units, 800 sampled): " + " ".join(f"{k} {np.mean(v):.3f} (n {len(v)})" for k, v in sorted(hits.items())), flush=True)
        json.dump({k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in hits.items()}, open(os.path.join(od, "smoke_train.json"), "w"))
    if a.dev:
        from lifelong_ood import ledger
        assert any(r.get("tag") == "pb1_2" for r in ledger.load("m3_prereg")), "pre-register first (experiments/prereg/pb1_2.py)"
        pr = predict(binder, lab["dev"], idx, feats, pca, K)
        pickle.dump(dict(units=[dict(frame=u["frame"], role=u["role"], phrase=u["phrase"]) for u in lab["dev"]], pred=pr),
                    open(os.path.join(od, "pred_dev.pkl"), "wb"))
        print(f"[b1 bind] dev: {len(pr)} units predicted", flush=True)


if __name__ == "__main__":
    main()

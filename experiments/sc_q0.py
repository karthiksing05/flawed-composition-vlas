"""Q0 (docs/THEORY_COMPOSITION.md 4b): from ONE demonstration and NO instruction, does the prototype-discovery operator ("Test-Time
Compositional Generalization in Diffusion Models via Prototype Discovery"), run over U's part fields, recover which entity fills the
object role and which fills the target role?  U = the unsupervised parts policy (lifelong_ood/policy/parts_flow.py, checkpoints
DATA/ckpt/pp_sc_U_s{10,11,12}/P.npz).  Offline, CPU, LIBERO venv, no rollouts, no training.

The operator, per demo row (state b, the demo's 8-step chunk a_q, flat 56-d), for each candidate assignment c of the role being parsed:
  1. discover: noise a_q (or draw noise) at each level sigma, ascend U's score under c with Adam, map each mode to clean-action space
     by Tweedie, fit a diagonal Gaussian by Hutchinson/Laplace (finite differences), de-duplicate by rmse.
  2. score (setting "score"; `row_scores`), summed over the parsed phase's rows: "uncapped" F_c = sum_r max_{j in c's prototypes}
     log N(a_q[r]; m_j[r], var_j[r]); "capped" F of c's own greedy K-subset; "pooled" the weight mass of c's prototypes in the paper's
     K-selection and per-dimension softmax over the whole pool.  Every record also carries the other two rules' parses (same pools).
     The variance comes from the score Jacobian's diagonal at the mode (setting "extent": deconv / posterior / marginal, `extent_var`;
     floor = setting "var_floor").
  3. parse: the role's filler = argmax_c of its score.  Object role on pre-hold rows, target role on held rows (with the true object and with
     the inferred one).  The phase is U's own proprioceptive hold event (meta "pgate"); the instruction is never read by the operator
     (it gives the ground-truth roles for scoring only: `truth`).
  4. the paper's composition, recorded per row: greedy submodular selection of K prototypes from the whole pool, per-dimension softmax
     weights at temperature tau, the product of experts' precision-weighted mean, its rmse to a_q (next to U's own sample's rmse).
Baselines on the same rows: (a) flow-matching-loss ranking, (b) demo_sites' motion heuristic (pp_p_train.py), (c) chance.

Candidate sets (from U's buffers, so they follow the checkpoint):
  object role: every entity with an object part (opart >= 0): bowl, cream cheese, wine bottle, plate (FC.MOVABLE).  U's object term also
               reads the target part's grasp-mode message m (P_imd); before the hold it depends on the target only through m, so each
               object candidate runs once per value of m (b["m"] set) and its prototypes are the union over m (the FM baseline takes the
               min over m).  The dummy ti set on those rows never reaches the output (smoke checks it).
  target role: every entity with a target part (tpart >= 0) other than the object: plate, stove, cabinet top, rack, drawer (cabinet top
               region), stove front, bowl.

Splits (demo index within each task K1-K6): demos 0-9 = SEARCH (seed-10 checkpoint only); demos 25-49 = READ (seeds 10, 11, 12);
demos 10-24 unused.

    python experiments/sc_q0.py settings
    python experiments/sc_q0.py smoke [--index i]              the only command run before registration (default: smoke_settings())
    python experiments/sc_q0.py search --index i [--task K1]   -> results/flawed/sc_q0/search_{i}.json (with --task: one part file only)
    python experiments/sc_q0.py pick                           -> results/flawed/sc_q0/picked.json
    python experiments/sc_q0.py read [--part j]                -> results/flawed/sc_q0_read.json (with --part: one of 36 part files only)
Part files (results/flawed/sc_q0/parts/) cache one (split, setting, seed, task) unit each; they are never overwritten, so search/read can
be split over jobs and the final command only aggregates.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
import pp_p_train as PT                                                     # noqa: E402
from fa1_train import chunks                                                # noqa: E402
from fa1g_train import rot0_rows                                            # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402
from lifelong_ood.policy.factored_flow import chunk_from, chunk_to, temb    # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_q0")
PARTS = os.path.join(OD, "parts")
READ_FN = os.path.join(OUT, "sc_q0_read.json")
TASKS = ("K1", "K2", "K3", "K4", "K5", "K6")
SPLITS = dict(search=dict(demos=tuple(range(0, 10)), seeds=(10,)), read=dict(demos=tuple(range(25, 50)), seeds=(10, 11, 12)))
FM_T, FM_DRAWS = (0.5, 0.7, 0.9), 8                                         # baseline (a): U's t (t = 1 is data), i.e. sigma = .5, .3, .1
FD_DELTA, VAR_CAP = 5e-3, 1.0                                             # the variance floor is a setting (var_floor)
SAMPLE_STEPS = 10                                                           # U's own sampler (PartsFlow.sample's default)
B_BOOT = 2000
H, DA = 8, 56


# ---- hyperparameter settings ---------------------------------------------------------------------------------------------------
def _levels(n, lo, hi):
    return [round(float(x), 6) for x in np.linspace(lo, hi, n)]


GRID = dict(levels=(3, 6, 12), sigma_range=((0.05, 0.4), (0.1, 0.7), (0.3, 0.9)), n_starts=(4, 16, 64), init=("query", "hybrid"),
            iters=(50, 150), lr=(0.02, 0.1), K=(1, 3, 5), tau=(0.05, 0.5), score=("uncapped", "capped", "pooled"),
            extent=("posterior", "marginal"), var_floor=(1e-4, 1e-3, 1e-2))
COST_MAX = 12 * 16 * 150                                                    # draws costing more (levels * n_starts * iters) are redrawn
RULES = ("uncapped", "capped", "pooled")
EXTENTS = ("deconv", "posterior", "marginal")
FLOORS = (1e-4, 1e-3, 1e-2)


def settings():
    """0: plan v4's setting (uncapped coverage, the original "deconv" extent, floor 1e-4); 1: the paper-like setting (pooled, posterior,
    floor 1e-4); 2-49: 48 distinct draws: numpy default_rng(0), rng.integers over the 23,328 points of GRID in itertools.product order,
    a draw redrawn if it repeats an earlier one or costs more than COST_MAX."""
    common = dict(rows_per_phase=6, probes=4, dedup=0.05)
    out = [dict(levels=_levels(6, 0.1, 0.7), sigma_range=[0.1, 0.7], n_starts=4, init="query", iters=80, lr=0.1, K=3, tau=0.5,
                score="uncapped", extent="deconv", var_floor=1e-4, **common),
           dict(levels=_levels(15, 0.05, 0.4), sigma_range=[0.05, 0.4], n_starts=128, init="hybrid", iters=150, lr=0.1, K=3, tau=0.5,
                score="pooled", extent="posterior", var_floor=1e-4, **common)]
    grid = list(itertools.product(*GRID.values()))
    rng = np.random.default_rng(0)
    seen = set()
    while len(seen) < 48:
        j = int(rng.integers(len(grid)))
        n, (lo, hi), ns, init, it, lr, K, tau, sc, ext, fl = grid[j]
        if j in seen or n * ns * it > COST_MAX:
            continue
        seen.add(j)
        out.append(dict(levels=_levels(n, lo, hi), sigma_range=[lo, hi], n_starts=ns, init=init, iters=it, lr=lr, K=K, tau=tau,
                        score=sc, extent=ext, var_floor=fl, **common))
    for i, h in enumerate(out):
        h["index"] = i
        h["cost"] = h["n_starts"] * len(h["levels"]) * h["iters"]
    return out


# ---- U's field in the paper's convention ---------------------------------------------------------------------------------------
_ORIG_LORA = PF.LoRALinear.forward
EVALS = [0]                                                                 # velocity row-evaluations (for timing)
VERBOSE = False
DIAG = None                                                                 # smoke: a list collecting each role call's pools


def _fast_lora_forward(self, x, atom, on=None):
    """LoRALinear.forward grouped by atom: one matmul per distinct atom instead of a per-row gather of A and B (same arithmetic)."""
    us = torch.unique(atom)
    if len(us) == 1:
        d = x @ self.A[us[0]]
        if on is not None:
            d = d * on[:, None].to(d.dtype)
        return self.base(x) + d @ self.B[us[0]]
    out = self.base(x)
    delta = torch.zeros_like(out)
    onf = None if on is None else on[:, None].to(x.dtype)
    for u in us:
        idx = torch.nonzero(atom == u, as_tuple=True)[0]
        d = x[idx] @ self.A[u]
        if onf is not None:
            d = d * onf[idx]
        delta[idx] = d @ self.B[u]
    return out + delta


def fast_lora(model, enable=True):
    """Patch (instance attributes only; parts_flow.py is untouched) the LoRA layers' forward."""
    for net in (model.obj, model.tgt):
        for ly in net.hidden:
            if enable:
                ly.forward = types.MethodType(_fast_lora_forward, ly)
            elif "forward" in ly.__dict__:
                del ly.forward


def _vtgt(m, b, x, te):
    """The target term of PartsFlow.velocity (parts_flow.py lines 446-456), alone."""
    B = x.shape[0]
    Qc, ct = m._canon(b)
    pt = m.tpart[b["ti"]]
    assert bool((pt >= 0).all()), "an atom without a part"
    st = m._tscale(b)
    xc = chunk_to(Qc, x)
    if st is not None:
        xc = torch.cat([xc[..., :2] * st[:, None, None], xc[..., 2:]], -1)
    oc = m.tgt(torch.cat([ct, xc.reshape(B, -1), te], -1), pt).reshape(B, m.H, 7)
    if st is not None:
        oc = torch.cat([oc[..., :2] / st[:, None, None], oc[..., 2:]], -1)
    return chunk_from(Qc, oc)


def term_velocity(m, b, x, tU):
    """U's velocity, computing only the term the gate selects: velocity() = (1 - held) v_o + held v_t with held in {0, 1}."""
    if m.lcomb:
        return m.velocity(b, x, tU)
    h = b["held"]
    if bool((h == 0).all()):
        return m._vobj(b, x, temb(tU))
    if bool((h == 1).all()):
        return _vtgt(m, b, x, temb(tU))
    return m.velocity(b, x, tU)


class UField:
    """U's flow field under a fixed state batch and candidate assignment (oi, ti, held, optional m) per chain, as the paper's eps/score
    over flat 56-d chunks.  b: a batch with one row per chain (or one row, expanded to the chains).

    Time convention.  U (PartsFlow.loss): x_t = (1 - t_U) noise + t_U a, target velocity v = a - noise; t_U = 0 is NOISE, 1 is data.
    The paper (and score_ops.py): x = alpha a + sigma eps, the noise level sigma rising toward noise.  Matching terms: alpha = t_U,
    sigma = 1 - t_U, eps = noise.  From x and v = a - noise:
        a_hat     = x + (1 - t_U) v      [(1-t)n + t a + (1-t)(a - n) = a]
        noise_hat = x - t_U v            [(1-t)n + t a - t(a - n)     = n]
    so eps_hat(x, sigma) = x - (1 - sigma) v(x, t_U = 1 - sigma), score = -eps_hat / sigma, and Tweedie's clean estimate
    (x + sigma^2 score) / alpha = (x - sigma eps_hat) / (1 - sigma) = x + sigma v, i.e. a_hat above.
    """

    def __init__(self, model, b1):
        self.m, self.b1, self._b = model, b1, None

    def _batch(self, n):
        """The batch for n chains: b itself (n rows), or b tiled (n a multiple of its rows: one row expanded, or [b; b] for x +- d v)."""
        if self._b is None or len(self._b["oi"]) != n:
            r = len(self.b1["oi"])
            assert n % r == 0, (n, r)
            self._b = {k: v.repeat(n // r, *([1] * (v.dim() - 1))) for k, v in self.b1.items()}
        return self._b

    def velocity(self, x, tU):
        n = x.shape[0]
        EVALS[0] += n
        return term_velocity(self.m, self._batch(n), x.reshape(n, H, 7), tU).reshape(n, -1)

    def eps(self, x, sigma):
        return x - (1 - sigma)[:, None] * self.velocity(x, 1 - sigma)

    def score(self, x, sigma):
        return -self.eps(x, sigma) / sigma[:, None]

    def tweedie(self, x, sigma):
        return x + sigma[:, None] * self.velocity(x, 1 - sigma)


@torch.no_grad()
def adam_ascent(f, x, sig, iters, lr):
    """score_ops.mode_ascent's update: Adam on +score (b1 .9, b2 .999, eps 1e-8, bias-corrected), no clamp."""
    m1, m2 = torch.zeros_like(x), torch.zeros_like(x)
    for i in range(1, int(iters) + 1):
        g = f.score(x, sig)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        x = x + lr * (m1 / (1 - 0.9 ** i)) / (torch.sqrt(m2 / (1 - 0.999 ** i)) + 1e-8)
    return x


@torch.no_grad()
def score_jac_diag(f, x, sig, probes):
    """Hutchinson estimate of diag(d score / dx) at the modes: h = mean over Rademacher probes v of v * (s(x + d v) - s(x - d v)) / 2d.
    probes: [P, n, 56].  The Tweedie (denoiser) Jacobian's diagonal comes from the same probes: a_hat = (x + sigma^2 s) / alpha, so
    v * (a_hat(x + d v) - a_hat(x - d v)) / 2d = (1 + sigma^2 h) / alpha exactly (v * v = 1); one h serves every extent below."""
    h = torch.zeros_like(x)
    n = x.shape[0]
    for v in probes:
        s = f.score(torch.cat([x + FD_DELTA * v, x - FD_DELTA * v]), torch.cat([sig, sig]))
        h += v * (s[:n] - s[n:]) / (2 * FD_DELTA)
    return h / len(probes)


def extent_var(h, sig, extent, floor):
    """Diagonal clean-space variance of a prototype from h (diag of the score Jacobian at its mode) at level sigma (alpha = 1 - sigma):
      deconv     (the original implementation, score_ops.laplace_diag): Sigma_t = -1/h (h < 0; else the cap), Sigma_0 = (Sigma_t - sigma^2) / alpha^2
      marginal   the noisy marginal's Laplace variance rescaled, no deconvolution: Sigma_0 = -1 / (h alpha^2)  (h < 0; else the cap)
      posterior  the paper's appendix: Cov[a | x_t] = (sigma^2 / alpha) d a_hat / dx = sigma^2 (1 + sigma^2 h) / alpha^2
    An isolated sharp mode (a point mass) has h = -1/sigma^2: deconv -> 0 and posterior -> 0 (both at the floor), marginal -> sigma^2/alpha^2.
    All clipped to [floor, VAR_CAP].  h, sig: numpy [n, 56], [n]."""
    s2, a2 = (sig ** 2)[:, None], ((1 - sig) ** 2)[:, None]
    neg = h < -1e-9
    if extent == "posterior":
        v = s2 * (1 + s2 * h) / a2
    elif extent == "marginal":
        v = np.where(neg, -1.0 / np.where(neg, h, -1.0) / a2, VAR_CAP)
    elif extent == "deconv":
        v = (np.where(neg, -1.0 / np.where(neg, h, -1.0), VAR_CAP) - s2) / a2
    else:
        raise ValueError(extent)
    return np.clip(v, floor, VAR_CAP).astype(np.float32)


def dedup(M, tol):
    """Greedy, in pool order: keep a mode iff its rmse to every kept mode exceeds tol."""
    keep = []
    for i in range(len(M)):
        if not keep or bool((((M[keep] - M[i]) ** 2).mean(1) > tol ** 2).all()):
            keep.append(i)
    return np.array(keep, int)


def with_cand(b_row, cand):
    """A one-row batch with the candidate's assignment (oi, ti, held) and, before the hold, the grasp-mode message m."""
    b = dict(b_row[(cand["oi"], cand["ti"])])
    if cand.get("m") is not None:
        b["m"] = torch.tensor([int(cand["m"])], dtype=torch.long)
    return b


MAXB = 8192                                                                 # chains per block (memory bound; blocks are independent)


@torch.no_grad()
def discover_many(model, jobs, hp):
    """jobs: list of (b_row, a_row, cands, seed), one per demo row.  All chains of all rows and candidates run together, in blocks of at
    most MAXB chains (chains are independent, so blocking changes nothing).  -> one pool per job (see `discover`)."""
    lv = np.asarray(hp["levels"], np.float32)
    L, S = len(lv), int(hp["n_starts"])
    nq = S if hp["init"] == "query" else S - S // 2
    sig1 = torch.tensor(np.repeat(lv, S))
    isq1 = np.tile(np.r_[np.ones(nq, bool), np.zeros(S - nq, bool)], L)
    al, sg = (1 - sig1)[:, None], sig1[:, None]
    bs, xs, vs, meta = [], [], [], []
    for j, (b_row, a_row, cands, seed) in enumerate(jobs):
        gen = torch.Generator().manual_seed(int(seed))                      # common random numbers: every candidate of a row
        z = torch.randn(L * S, DA, generator=gen)                           # starts from the same noise and uses the same probes
        pv = torch.randint(0, 2, (int(hp["probes"]), L * S, DA), generator=gen).float() * 2 - 1
        a = torch.as_tensor(np.asarray(a_row, np.float32)).reshape(1, -1)
        x0 = torch.where(torch.as_tensor(isq1)[:, None], al * a + sg * z, torch.sqrt(al ** 2 + sg ** 2) * z)
        for ci, cand in enumerate(cands):
            bs.append({k: v.expand(L * S, *v.shape[1:]) for k, v in with_cand(b_row, cand).items()})
            xs.append(x0)
            vs.append(pv)
            meta.append((j, ci))
    keys = set(bs[0])
    assert all(set(b) == keys for b in bs), "candidates of one call must agree on the batch fields (m on all or none)"
    B = {k: torch.cat([b[k] for b in bs]) for k in keys}
    X = torch.cat(xs)
    PV = torch.cat(vs, 1)
    SIG = sig1.repeat(len(bs))
    M0, H0 = torch.empty_like(X), torch.empty_like(X)
    for s in range(0, len(X), MAXB):
        e = min(s + MAXB, len(X))
        f = UField(model, {k: v[s:e] for k, v in B.items()})
        x = adam_ascent(f, X[s:e], SIG[s:e], hp["iters"], hp["lr"])
        M0[s:e] = f.tweedie(x, SIG[s:e])
        H0[s:e] = score_jac_diag(f, x, SIG[s:e], PV[:, s:e])
    M0, H0 = M0.numpy(), H0.numpy()
    pools = [dict(M=[], H=[], cand=[], level=[], query_start=[], n_raw=0) for _ in jobs]
    for k, (j, ci) in enumerate(meta):
        m0, h0 = M0[k * L * S:(k + 1) * L * S], H0[k * L * S:(k + 1) * L * S]
        keep = dedup(m0, hp["dedup"])
        p = pools[j]
        p["n_raw"] += len(m0)
        p["M"].append(m0[keep])
        p["H"].append(h0[keep])
        p["cand"].append(np.full(len(keep), ci))
        p["level"].append(np.repeat(lv, S)[keep])
        p["query_start"].append(isq1[keep])
    for p in pools:
        for k in ("M", "H", "cand", "level", "query_start"):
            p[k] = np.concatenate(p[k])
        p["V"] = extent_var(p["H"], p["level"], hp["extent"], hp["var_floor"])
    return pools


def discover(model, b_row, a_row, cands, hp, seed=0):
    """b_row: {(oi, ti): one-row batch} for this demo row; a_row: the demo's chunk [56]; cands: list of dict(oi, ti, m).
    For each candidate and each level, hp n_starts chains (init "query": all from the noised chunk; "hybrid": ceil(half) from it, the
    rest from pure noise, N(0, (alpha^2 + sigma^2) I), the flow path's analogue of score_ops' randn), hp iters Adam steps of hp lr on the
    score, Tweedie to clean space, diag of the score Jacobian H (hp probes) and from it the variance V (hp extent, hp var_floor;
    `extent_var`), de-duplication within the candidate's pool at rmse hp dedup.  Every candidate gets the same noise (common random
    numbers).  -> dict(M [P, 56], H [P, 56], V [P, 56], cand [P], level [P], query_start [P], n_raw)."""
    return discover_many(model, [(b_row, a_row, cands, seed)], hp)[0]


# ---- the paper's selection and composition (compose.py's numpy lines, re-implemented: lifelong_ood/compose/ shadows that module) -----
def loglik_perdim(xq, M, V):
    return -0.5 * (((np.ravel(xq) - M) ** 2) / V + np.log(2 * np.pi * V))


def submodular_select(Lk, K):
    """Greedy max of F(S) = sum_r max_{j in S} L[j, r] -> indices in pick order."""
    best, sel = np.full(Lk.shape[1], -np.inf), []
    for _ in range(min(K, len(Lk))):
        gains = np.maximum(Lk, best).sum(1)
        gains[sel] = -np.inf
        j = int(np.argmax(gains))
        sel.append(j)
        best = np.maximum(best, Lk[j])
    return np.array(sel, int)


def poe_mean(xq, M, V, K, tau):
    """Selection, per-dimension softmax weights at fixed temperature tau (the paper's Eq. 10), precision-weighted product mean."""
    Lk = loglik_perdim(xq, M, V)
    sel = submodular_select(Lk, K)
    Z = Lk[sel] / tau
    Z -= Z.max(0, keepdims=True)
    w = np.exp(Z)
    w /= w.sum(0, keepdims=True)
    prec = (w / V[sel]).sum(0)
    return (w * M[sel] / V[sel]).sum(0) / prec, sel


def coverage(xq, pool, groups):
    """groups: {name: [candidate indices]} -> {name: sum_r max over the group's prototypes of the per-dim log-likelihood}."""
    return row_scores(xq, pool["M"], pool["V"], pool["cand"], groups, "uncapped", 1, 1.0)


def row_scores(xq, M, V, cand, groups, rule, K, tau):
    """One row's score per candidate name (higher wins) under a scoring rule:
      uncapped  sum_r max over ALL the candidate's prototypes of l_r (the original rule)
      capped    the candidate's own greedy submodular K-subset S_c, score F(S_c) = sum_r max_{j in S_c} l_r (equal budget per candidate)
      pooled    the paper's operator: greedy K over the whole pool, per-dimension softmax weights at temperature tau over the selected
                set; a candidate's score is the weight mass of its selected prototypes summed over dimensions (0 if none selected)."""
    Lk = loglik_perdim(xq, M, V)
    out = {}
    if rule == "pooled":
        sel = submodular_select(Lk, K)
        Z = Lk[sel] / tau
        Z -= Z.max(0, keepdims=True)
        w = np.exp(Z)
        w /= w.sum(0, keepdims=True)
        mass = w.sum(1)
        for name, cis in groups.items():
            out[name] = float(mass[np.isin(cand[sel], cis)].sum())
        return out
    for name, cis in groups.items():
        sel = np.isin(cand, cis)
        if not sel.any():
            out[name] = -np.inf
        elif rule == "uncapped":
            out[name] = float(Lk[sel].max(0).sum())
        elif rule == "capped":
            Lc = Lk[sel]
            out[name] = float(Lc[submodular_select(Lc, K)].max(0).sum())
        else:
            raise ValueError(rule)
    return out


# ---- demos ---------------------------------------------------------------------------------------------------------------------
_D = {}


def demos():
    if not _D:
        _D["D"] = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
        _D["Db"] = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
        _D["lang"] = PT.instructions()
    return _D


def truth(task):
    """Ground-truth roles, from the instruction's template parse.  Used to SCORE the parse only; the operator never sees it."""
    o, t = FC.roles(demos()["lang"][task])
    return FC.ENT.index(o), FC.ENT.index(t)


def demo_ex(task, demo, pg):
    """One demo's rows: the state arrays, U's hold event (its meta pgate; else the contact label), the clipped chunks, ee0 and rot0."""
    Z = demos()
    D = Z["D"]
    r = np.nonzero((D["task"] == list(FC.KNOWN).index(task)) & (D["demo"] == demo))[0]
    assert len(r) and np.all(np.diff(r) == 1), (task, demo)
    Dd = {k: v[r] for k, v in D.items() if hasattr(v, "shape") and v.shape[:1] == D["task"].shape}
    if pg:
        assert not pg.get("motion"), "pgate motion makes the hold depend on the object: not handled"
        held = PF.pgate_rows(np.zeros(len(r), np.int64), Dd["grip"], Dd["prev_grip"], Dd["ee_pos"][:, 2], pg["tw"], pg["lift"])
    else:
        held = Z["Db"]["held"][r].astype(np.float32)
    return dict(task=task, demo=demo, D=Dd, held=held, n=len(r), X=np.clip(chunks(Dd, H), -1, 1).reshape(len(r), DA),
                ee0=PT.start_rows(Dd), rot0=rot0_rows(Dd, held), _cache={})


def batch_rows(ex, rows, oi, ti):
    """U's batch fields for `rows` of the demo under assignment (oi, ti) (pp_p_train.rows' fields; zrest, cyaw, d0t follow (oi, ti))."""
    key = (oi, ti)
    if key not in ex["_cache"]:
        n, Dd, held = ex["n"], ex["D"], ex["held"]
        oa, ta = np.full(n, oi), np.full(n, ti)
        zr, cy = FK.canon_rows(Dd, held, oa, ta)
        ex["_cache"][key] = dict(oa=oa, ta=ta, zr=zr, cy=cy, d0t=PT.d0t_rows(Dd, held, oa, ta))
    c, Dd = ex["_cache"][key], ex["D"]
    rows = np.asarray(rows)
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v)[rows], dtype=dt)   # noqa: E731
    return dict(ee_pos=T(Dd["ee_pos"]), ee_rot=T(Dd["ee_rot"]), grip=T(Dd["grip"]), prev=T(Dd["prev_grip"]), pos=T(Dd["pos"]),
                yaw=T(Dd["yaw"]), drawer=T(Dd["drawer"]), oi=T(c["oa"], torch.long), ti=T(c["ta"], torch.long), held=T(ex["held"]),
                zrest=T(c["zr"]), cyaw=T(c["cy"]), rot0=T(ex["rot0"]), ee0=T(ex["ee0"]), d0t=T(c["d0t"]))


def spaced(idx, k):
    if len(idx) == 0:
        return idx
    return idx[np.unique(np.round(np.linspace(0, len(idx) - 1, k)).astype(int))]


def phase_rows(ex, k):
    """k evenly spaced rows before the first hold onset, and k evenly spaced held rows (U's own event)."""
    h = ex["held"] > 0.5
    if not h.any():
        return spaced(np.arange(ex["n"]), k), np.zeros(0, int)
    first = int(np.argmax(h))
    return spaced(np.arange(first), k), spaced(np.nonzero(h)[0], k)


def cand_sets(model):
    ents = list(range(len(FC.ENT)))
    objs = [e for e in ents if int(model.opart[e]) >= 0]
    tgts = [e for e in ents if int(model.tpart[e]) >= 0]
    msgs = list(range(model.nmode)) if model.nmode > 0 else [None]
    return objs, tgts, msgs


def obj_cands(model):
    """Object role, before the hold: (object e, message m) for every object entity and every m.  ti = a dummy valid target (unused)."""
    objs, tgts, msgs = cand_sets(model)
    out = []
    for e in objs:
        for mm in msgs:
            out.append(dict(oi=e, ti=next(t for t in tgts if t != e), m=mm, name=FC.ENT[e]))
    return out


def tgt_cands(model, obj):
    _, tgts, _ = cand_sets(model)
    return [dict(oi=obj, ti=t, m=None, name=FC.ENT[t]) for t in tgts if t != obj]


def row_seed(ckpt_seed, task, demo, row):
    return ((int(ckpt_seed) * 10 + TASKS.index(task)) * 100 + int(demo)) * 1000 + int(row)


# ---- parsing ---------------------------------------------------------------------------------------------------------------------
def run_role(model, ex, rows, cands, hp, ckpt_seed):
    """Discover over `cands` at each row -> ({rule: {candidate name: score summed over rows}} for every rule in RULES (the setting's own
    rule is hp["score"]; the others come free from the same pools), per-row composition records)."""
    names = list(dict.fromkeys(c["name"] for c in cands))
    groups = {nm: [i for i, c in enumerate(cands) if c["name"] == nm] for nm in names}
    F = {rule: {nm: 0.0 for nm in names} for rule in RULES}
    recs = []
    jobs = [({(c["oi"], c["ti"]): batch_rows(ex, [r], c["oi"], c["ti"]) for c in cands}, ex["X"][r], cands,
             row_seed(ckpt_seed, ex["task"], ex["demo"], r)) for r in rows]
    t0, e0 = time.time(), EVALS[0]
    pools = discover_many(model, jobs, hp) if jobs else []
    if VERBOSE:
        print(f"[q0] discover {len(rows)} rows x {len(cands)} cands: {time.time() - t0:.1f}s, {EVALS[0] - e0} velocity row-evals", flush=True)
    if DIAG is not None:
        DIAG.append(dict(rows=[int(r) for r in rows], aq=[ex["X"][r] for r in rows], pools=pools, groups=groups,
                         oi=cands[0]["oi"] if len({c["oi"] for c in cands}) == 1 else None))
    for r, pool in zip(rows, pools):
        aq = ex["X"][r]
        cov = {}
        for rule in RULES:
            cov[rule] = row_scores(aq, pool["M"], pool["V"], pool["cand"], groups, rule, hp["K"], hp["tau"])
            for nm in names:
                F[rule][nm] += cov[rule][nm]
        mu, sel = poe_mean(aq, pool["M"], pool["V"], hp["K"], hp["tau"])
        recs.append(dict(row=int(r), coverage=cov[hp["score"]], scores_by_rule=cov, n_pool=int(len(pool["M"])), n_raw=int(pool["n_raw"]),
                         n_per_cand={nm: int(np.isin(pool["cand"], groups[nm]).sum()) for nm in names},
                         selected=[cands[int(pool["cand"][j])]["name"] + ("" if cands[int(pool["cand"][j])]["m"] is None else f"|m{cands[int(pool['cand'][j])]['m']}")
                                   for j in sel],
                         selected_level=[float(pool["level"][j]) for j in sel],
                         selected_query_start=[bool(pool["query_start"][j]) for j in sel],
                         rmse_poe=float(np.sqrt(np.mean((mu - aq) ** 2)))))
    return F, recs


@torch.no_grad()
def fm_role(model, ex, rows, cands, ckpt_seed):
    """Baseline (a): mean flow-matching loss of the demo chunk over t_U in FM_T and FM_DRAWS noise draws, same noise for every candidate
    -> {name: loss} (a name with several m takes the min over m)."""
    rows = np.asarray(rows)
    nT = len(FM_T) * FM_DRAWS
    out = {}
    for c in cands:
        gen = torch.Generator().manual_seed(row_seed(ckpt_seed, ex["task"], ex["demo"], 999))
        b = batch_rows(ex, rows, c["oi"], c["ti"])
        if c["m"] is not None:
            b["m"] = torch.full((len(rows),), int(c["m"]), dtype=torch.long)
        b = {k: v.repeat_interleave(nT, 0) for k, v in b.items()}
        a = torch.as_tensor(ex["X"][rows]).repeat_interleave(nT, 0).reshape(-1, H, 7)
        t = torch.tensor(np.tile(np.repeat(FM_T, FM_DRAWS), len(rows)), dtype=torch.float32)
        n0 = torch.randn(a.shape, generator=gen)
        xt = (1 - t[:, None, None]) * n0 + t[:, None, None] * a
        loss = float(((term_velocity(model, b, xt, t) - (a - n0)) ** 2).mean())
        out[c["name"]] = min(loss, out.get(c["name"], np.inf))
    return out


@torch.no_grad()
def u_sample_rmse(model, ex, rows, oi, ti, ckpt_seed):
    """U's own sample under the TRUE assignment (its default message), per row: rmse to the demo chunk."""
    if len(rows) == 0:
        return []
    gen = torch.Generator().manual_seed(row_seed(ckpt_seed, ex["task"], ex["demo"], 998))
    x = model.sample(batch_rows(ex, rows, oi, ti), steps=SAMPLE_STEPS, generator=gen).reshape(len(rows), -1).numpy()
    return [float(v) for v in np.sqrt(((x - ex["X"][np.asarray(rows)]) ** 2).mean(1))]


def sites_baseline(ex, true_obj):
    """Baseline (b): pp_p_train.demo_sites (no instruction) on this demo with U's event: g, r; and r by the same rule with g = the true
    object (target given the true object)."""
    s = PT.demo_sites(ex["D"], ex["held"])
    if not s:
        return dict(g=None, r=None, r_true=None)
    v = next(iter(s.values()))
    h = np.nonzero(ex["held"] > 0.5)[0]
    pb = ex["D"]["pos"][h[-1]]
    dist = np.linalg.norm(pb - pb[true_obj], axis=-1)
    dist[true_obj] = np.inf
    return dict(g=v["g"], r=v["r"], r_true=FC.ENT[int(np.argsort(dist, kind="stable")[0])])


def argmax_name(d, low=False):
    return (min if low else max)(d, key=lambda k: d[k]) if d else None


def parse_demo(model, ex, hp, ckpt_seed):
    """The whole Q0 parse of one demo -> a JSON-able record (truth used only for the *_ok fields, the chance level and U's sample)."""
    t0 = time.time()
    to, tt = truth(ex["task"])
    pre, hrows = phase_rows(ex, hp["rows_per_phase"])
    rec = dict(task=ex["task"], demo=int(ex["demo"]), seed=int(ckpt_seed), setting=int(hp["index"]), n_rows=int(ex["n"]),
               pre_rows=[int(r) for r in pre], held_rows=[int(r) for r in hrows], no_event=bool(len(hrows) == 0),
               true_obj=FC.ENT[to], true_tgt=FC.ENT[tt])
    oc = obj_cands(model)
    F, recs = run_role(model, ex, pre, oc, hp, ckpt_seed)
    rec["obj_rows"] = recs
    fo = fm_role(model, ex, pre, oc, ckpt_seed)
    rec["fm_obj"], rec["fm_obj_pred"] = fo, argmax_name(fo, low=True)
    rec["u_rmse_pre"] = u_sample_rmse(model, ex, pre, to, tt, ckpt_seed)
    if VERBOSE:
        print(f"[q0] object role incl. FM baseline and U sample: {time.time() - t0:.1f}s", flush=True)
    T, O = FC.ENT[tt], FC.ENT[to]
    rules = {rule: dict(obj_scores=F[rule], obj_pred=argmax_name(F[rule])) for rule in RULES}
    if len(hrows):
        tpass = {}                                                          # object -> target pass (one per distinct object)
        for obj in [to] + [FC.ENT.index(rules[rule]["obj_pred"]) for rule in [hp["score"]] + [x for x in RULES if x != hp["score"]]]:
            if obj not in tpass:
                tpass[obj] = run_role(model, ex, hrows, tgt_cands(model, obj), hp, ckpt_seed)
        for rule in RULES:
            Ft, Fi = tpass[to][0][rule], tpass[FC.ENT.index(rules[rule]["obj_pred"])][0][rule]
            rules[rule].update(tgt_scores_true=Ft, tgt_pred_true=argmax_name(Ft), tgt_scores_inf=Fi, tgt_pred_inf=argmax_name(Fi))
        rec["tgt_rows_true"] = tpass[to][1]
        rec["tgt_rows_inf"] = tpass[FC.ENT.index(rules[hp["score"]]["obj_pred"])][1]
    else:
        for rule in RULES:
            rules[rule].update(tgt_scores_true={}, tgt_pred_true=None, tgt_scores_inf={}, tgt_pred_inf=None)
    for rule in RULES:
        r_ = rules[rule]
        r_["ok"] = dict(obj=r_["obj_pred"] == O, tgt_true=r_["tgt_pred_true"] == T, tgt_inf=r_["tgt_pred_inf"] == T)
        r_["ok"]["joint"] = r_["ok"]["obj"] and r_["ok"]["tgt_inf"]
    rec["rules"] = rules
    rec.update({k: rules[hp["score"]][k] for k in ("obj_scores", "obj_pred", "tgt_scores_true", "tgt_pred_true", "tgt_scores_inf", "tgt_pred_inf")})
    if len(hrows):
        ft = fm_role(model, ex, hrows, tgt_cands(model, to), ckpt_seed)
        fio = FC.ENT.index(rec["fm_obj_pred"])
        fi = ft if fio == to else fm_role(model, ex, hrows, tgt_cands(model, fio), ckpt_seed)
        rec.update(fm_tgt_true=ft, fm_tgt_pred_true=argmax_name(ft, low=True), fm_tgt_inf=fi, fm_tgt_pred_inf=argmax_name(fi, low=True))
        rec["u_rmse_held"] = u_sample_rmse(model, ex, hrows, to, tt, ckpt_seed)
    else:
        rec.update(fm_tgt_pred_true=None, fm_tgt_pred_inf=None, u_rmse_held=[])
    rec["sites"] = sites_baseline(ex, to)
    objs, tgts, _ = cand_sets(model)
    rec["chance"] = dict(obj=1.0 / len(objs), tgt=1.0 / len([t for t in tgts if t != to]))
    rec["chance"]["joint"] = rec["chance"]["obj"] * rec["chance"]["tgt"]
    rec["ok"] = rules[hp["score"]]["ok"]
    rec["fm_ok"] = dict(obj=rec["fm_obj_pred"] == O, tgt_true=rec["fm_tgt_pred_true"] == T, tgt_inf=rec["fm_tgt_pred_inf"] == T)
    rec["fm_ok"]["joint"] = rec["fm_ok"]["obj"] and rec["fm_ok"]["tgt_inf"]
    s = rec["sites"]
    rec["sites_ok"] = dict(obj=s["g"] == O, tgt_true=s["r_true"] == T, tgt_inf=s["r"] == T)
    rec["sites_ok"]["joint"] = rec["sites_ok"]["obj"] and rec["sites_ok"]["tgt_inf"]
    rec["secs"] = time.time() - t0
    return rec


# ---- units, summaries ----------------------------------------------------------------------------------------------------------
_MODELS = {}


def load_u(seed):
    if seed not in _MODELS:
        from pp_p_roll import load_any
        m, meta = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
        fast_lora(m)
        _MODELS[seed] = (m, meta)
    return _MODELS[seed]


def part_fn(split, h, seed, task):
    return os.path.join(PARTS, f"{split}_h{h}_s{seed}_{task}.json")


def run_unit(split, h, seed, task):
    """One (split, setting, seed, task) unit -> its records; cached in a part file that is never overwritten."""
    fn = part_fn(split, h, seed, task)
    if os.path.exists(fn):
        return json.load(open(fn, encoding="utf-8"))["records"]
    hp = settings()[h]
    m, meta = load_u(seed)
    recs = []
    for d in SPLITS[split]["demos"]:
        ex = demo_ex(task, d, meta.get("pgate"))
        recs.append(parse_demo(m, ex, hp, seed))
        r = recs[-1]
        print(f"[q0] {split} h{h} s{seed} {task} d{d} obj {r['obj_pred']} ({r['true_obj']}) tgt_inf {r['tgt_pred_inf']} ({r['true_tgt']}) "
              f"fm {r['fm_obj_pred']}/{r['fm_tgt_pred_inf']} sites {r['sites']['g']}/{r['sites']['r']} {r['secs']:.1f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    if os.path.exists(fn):
        raise SystemExit(f"{fn} appeared while computing; not overwriting")
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(split=split, setting=hp, seed=seed, task=task, host=os.uname().nodename, records=recs), fh)
    os.replace(fn + ".tmp", fn)
    return recs


KEYS = ("obj", "tgt_true", "tgt_inf", "joint")


def summarize(recs):
    """Point summaries (no CI) over a list of demo records, per task and pooled."""
    def one(rs):
        out = {}
        for fam in ("ok", "fm_ok", "sites_ok"):
            out[fam.replace("ok", "acc").strip("_") if fam != "ok" else "acc"] = {k: float(np.mean([r[fam][k] for r in rs])) for k in KEYS}
        out["acc_by_rule"] = {rule: {k: float(np.mean([r["rules"][rule]["ok"][k] for r in rs])) for k in KEYS} for rule in RULES}
        out["chance"] = {k: float(np.mean([r["chance"][k] for r in rs])) for k in ("obj", "tgt", "joint")}
        out["no_event"] = int(sum(r["no_event"] for r in rs))
        rows = [x for r in rs for x in r["obj_rows"] + r.get("tgt_rows_inf", [])]
        out["rmse_poe"] = dict(pre=_mean([x["rmse_poe"] for r in rs for x in r["obj_rows"]]),
                               held=_mean([x["rmse_poe"] for r in rs for x in r.get("tgt_rows_inf", [])]))
        out["rmse_u"] = dict(pre=_mean([v for r in rs for v in r["u_rmse_pre"]]), held=_mean([v for r in rs for v in r["u_rmse_held"]]))
        out["protos_per_row"] = dict(mean=_mean([x["n_pool"] for x in rows]), median=float(np.median([x["n_pool"] for x in rows])) if rows else None,
                                     raw=_mean([x["n_raw"] for x in rows]))
        out["secs_per_demo"] = _mean([r["secs"] for r in rs])
        out["n"] = len(rs)
        return out
    res = {t: one([r for r in recs if r["task"] == t]) for t in TASKS if any(r["task"] == t for r in recs)}
    res["pooled"] = one(recs)
    return res


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def boot(arr_by_task):
    """arr_by_task: {task: [n_demos, n_seeds] values}.  Per task: mean and 95% CI resampling demos (each demo keeps all its seeds);
    pooled: resampling demos within each task (stratified), the mean over all.  2,000 draws, default_rng(0)."""
    rng = np.random.default_rng(0)
    tasks = list(arr_by_task)
    draws = {t: [] for t in tasks}
    pooled = []
    for _ in range(B_BOOT):
        tot, cnt = 0.0, 0
        for t in tasks:
            a = arr_by_task[t]
            s = a[rng.integers(0, len(a), len(a))]
            draws[t].append(float(s.mean()))
            tot, cnt = tot + float(s.sum()), cnt + s.size
        pooled.append(tot / cnt)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]   # noqa: E731
    out = {t: dict(mean=float(arr_by_task[t].mean()), ci=ci(draws[t])) for t in tasks}
    allv = np.concatenate([arr_by_task[t].ravel() for t in tasks])
    out["pooled"] = dict(mean=float(allv.mean()), ci=ci(pooled))
    return out


# ---- commands ------------------------------------------------------------------------------------------------------------------
def cmd_settings():
    for h in settings():
        print(json.dumps(h))


def cmd_search(i, task=None):
    fn = os.path.join(OD, f"search_{i}.json")
    if task:
        run_unit("search", i, 10, task)
        return
    if os.path.exists(fn):
        raise SystemExit(f"{fn} exists")
    t0 = time.time()
    recs = [r for t in TASKS for r in run_unit("search", i, 10, t)]
    res = dict(setting=settings()[i], split="search", demos=list(SPLITS["search"]["demos"]), seeds=[10], summary=summarize(recs),
               records=recs, secs_aggregate=time.time() - t0, host=os.uname().nodename)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    os.replace(fn + ".tmp", fn)
    print("[q0] search", i, json.dumps(res["summary"]["pooled"]), flush=True)


def cmd_pick():
    fn = os.path.join(OD, "picked.json")
    if os.path.exists(fn):
        raise SystemExit(f"{fn} exists")
    hs = settings()
    missing = [h["index"] for h in hs if not os.path.exists(os.path.join(OD, f"search_{h['index']}.json"))]
    if missing:
        raise SystemExit(f"search files missing: {missing}")
    table = []
    for h in hs:
        s = json.load(open(os.path.join(OD, f"search_{h['index']}.json"), encoding="utf-8"))["summary"]["pooled"]
        table.append(dict(index=h["index"], joint=s["acc"]["joint"], obj=s["acc"]["obj"], tgt_inf=s["acc"]["tgt_inf"], cost=h["cost"]))
    best = sorted(table, key=lambda r: (-r["joint"], r["cost"], r["index"]))[0]
    res = dict(index=best["index"], rule="highest search-split joint accuracy; ties: lower n_starts*levels*iters, then lower index",
               setting=hs[best["index"]], table=table)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[q0] picked", json.dumps(best), flush=True)


def read_units():
    p = json.load(open(os.path.join(OD, "picked.json"), encoding="utf-8"))["index"]
    hs = [p] if p == 0 else [p, 0]
    return p, [(h, s, t) for h in hs for s in SPLITS["read"]["seeds"] for t in TASKS]


def cmd_read(part=None):
    if os.path.exists(READ_FN):
        raise SystemExit(f"{READ_FN} exists")
    p, units = read_units()
    if part is not None:
        if part < len(units):
            run_unit("read", *units[part])
        return
    t0 = time.time()
    demos_ = SPLITS["read"]["demos"]
    seeds = SPLITS["read"]["seeds"]
    R = {}
    for h, s, t in units:
        for r in run_unit("read", h, s, t):
            R[(h, s, t, r["demo"])] = r

    def arr(h, fam, key, rule=None):
        get = (lambda r: r[fam][key]) if rule is None else (lambda r: r["rules"][rule]["ok"][key])   # noqa: E731
        return {t: np.array([[float(get(R[(h, s, t, d)])) for s in seeds] for d in demos_]) for t in TASKS}

    res = dict(split="read", demos=list(demos_), seeds=list(seeds), picked=p, settings={str(h): settings()[h] for h in sorted({p, 0})})
    for h in sorted({p, 0}):
        recs = [R[(h, s, t, d)] for t in TASKS for s in seeds for d in demos_]
        res[f"setting_{h}"] = dict(
            operator={k: boot(arr(h, "ok", k)) for k in KEYS},
            fm_loss={k: boot(arr(h, "fm_ok", k)) for k in KEYS},
            summary=summarize(recs))
    # secondary, no verdict: the picked setting's hyperparameters and pools scored by the other two rules
    res["picked_other_rules"] = {rule: {k: boot(arr(p, None, k, rule)) for k in KEYS} for rule in RULES if rule != settings()[p]["score"]}
    res["sites"] = {k: boot(arr(0, "sites_ok", k)) for k in KEYS}
    res["chance"] = res["setting_0"]["summary"]["pooled"]["chance"]
    a, b = arr(p, "ok", "joint"), arr(0, "ok", "joint")
    res["picked_minus_0_joint"] = boot({t: a[t] - b[t] for t in TASKS})
    j = res[f"setting_{p}"]["operator"]["joint"]["pooled"]["ci"]
    res["label_operator"] = "OPERATOR PARSES" if j[0] > 0.80 else ("OPERATOR FAILS" if j[1] < 0.50 else "PARTIAL")
    res["label_search"] = "SEARCH MATTERS" if res["picked_minus_0_joint"]["pooled"]["ci"][0] > 0.10 else "SEARCH DOES NOT MATTER"
    res["secs_aggregate"] = time.time() - t0
    with open(READ_FN + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(READ_FN + ".tmp", READ_FN)
    print("[q0] read", res["label_operator"], res["label_search"], flush=True)


def cmd_smoke(only=None):
    """search-split demo 0 of K1 and K5, seed 10, at smoke_settings() (or only setting `only`): equivalence checks, the decisions under
    every scoring rule, baselines, setting 0's diagnosis, timing, projections."""
    global VERBOSE, DIAG
    VERBOSE = True
    hp = settings()[0]
    m, meta = load_u(10)
    objs, tgts, msgs = cand_sets(m)
    print("[smoke] meta tasks", meta.get("tasks"), "pgate", meta.get("pgate"), "nmode", m.nmode, "tmode",
          {a: int(m.tmode[i]) for i, a in enumerate(PF.TGT_ATOMS)}, "lcomb", m.lcomb, "rtransport", m.rtransport, "tscale", m.tscale, flush=True)
    print("[smoke] trained parse", json.dumps((meta.get("roles") or {}).get("situations")), flush=True)
    print("[smoke] object candidates", [FC.ENT[e] for e in objs], "messages", msgs, "target candidates", [FC.ENT[e] for e in tgts], flush=True)
    # -- equivalence checks
    ex = demo_ex("K1", 0, meta.get("pgate"))
    pre, hrows = phase_rows(ex, hp["rows_per_phase"])
    print("[smoke] K1 d0 rows", ex["n"], "held rows", int((ex["held"] > 0.5).sum()), "pre", pre.tolist(), "held", hrows.tolist(), flush=True)
    to, tt = truth("K1")
    g = torch.Generator().manual_seed(0)
    for nm, rr in (("pre", pre), ("held", hrows)):
        if len(rr) == 0:
            print(f"[smoke] check {nm}: no rows", flush=True)
            continue
        r = int(rr[len(rr) // 2])
        b = {k: v.expand(64, *v.shape[1:]).contiguous() for k, v in batch_rows(ex, [r], to, tt).items()}
        x = torch.randn(64, H, 7, generator=g)
        t = torch.rand(64, generator=g)
        with torch.no_grad():
            v_fast = term_velocity(m, b, x, t)
            v_full = m.velocity(b, x, t)
            fast_lora(m, False)
            v_orig = m.velocity(b, x, t)
            fast_lora(m, True)
            msg = ""
            if nm == "pre":
                b2 = dict(b, ti=torch.full_like(b["ti"], next(e for e in tgts if e not in (to, tt))), m=m.message(b))
                msg = f" | pre-hold output with another ti, same m: max diff {float((m.velocity(b2, x, t) - v_full).abs().max()):.2e}"
        print(f"[smoke] check {nm}: term-only vs velocity {float((v_fast - v_full).abs().max()):.2e}; fast LoRA vs parts_flow "
              f"{float((v_full - v_orig).abs().max()):.2e}{msg}", flush=True)
    # time-convention sanity: Tweedie from a lightly noised true chunk should land near the chunk
    r = int(pre[len(pre) // 2])
    f = UField(m, batch_rows(ex, [r], to, tt))
    a = torch.as_tensor(ex["X"][r]).reshape(1, -1).repeat(32, 1)
    for s in (0.05, 0.2, 0.5, 0.8):
        sig = torch.full((32,), s)
        x = (1 - s) * a + s * torch.randn(a.shape, generator=g)
        with torch.no_grad():
            ah = f.tweedie(x, sig)
        print(f"[smoke] tweedie sigma {s}: rmse(x/alpha, a) {float(((x / (1 - s) - a) ** 2).mean().sqrt()):.4f} -> rmse(a_hat, a) "
              f"{float(((ah - a) ** 2).mean().sqrt()):.4f}", flush=True)
    # -- the two demos, at each smoke setting
    for hi in smoke_settings() if only is None else [only]:
        hp = settings()[hi]
        print(f"[smoke] ===== setting {hi}: {json.dumps(hp)}", flush=True)
        tot_ev, tot_s, nfields = 0, 0.0, []
        for task in ("K1", "K5"):
            ex = demo_ex(task, 0, meta.get("pgate"))
            EVALS[0] = 0
            DIAG = [] if hi == 0 else None
            rec = parse_demo(m, ex, hp, 10)
            tot_ev += EVALS[0]
            tot_s += rec["secs"]
            nf = len(rec["pre_rows"]) * len(obj_cands(m)) + len(rec["held_rows"]) * len(tgt_cands(m, truth(task)[0])) * (1 if rec["ok"]["obj"] else 2)
            nfields.append(nf)
            print(f"[smoke] h{hi} {task} d0 truth obj {rec['true_obj']} tgt {rec['true_tgt']}", flush=True)
            for rule in RULES:
                R_ = rec["rules"][rule]
                top = lambda d: ", ".join(f"{k} {v:.4g}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:3])   # noqa: E731
                print(f"[smoke] h{hi} {task} rule {rule}{' (setting)' if rule == hp['score'] else ''}: ok {R_['ok']} | obj: {top(R_['obj_scores'])} | "
                      f"tgt|true: {top(R_['tgt_scores_true'])} | tgt|inferred {R_['obj_pred']}: {top(R_['tgt_scores_inf'])}", flush=True)
            print(f"[smoke] h{hi} {task} fm_ok {rec['fm_ok']} sites_ok {rec['sites_ok']}; protos per row "
                  f"{[x['n_pool'] for x in rec['obj_rows']]} / {[x['n_pool'] for x in rec.get('tgt_rows_inf', [])]}; rmse_poe "
                  f"{_mean([x['rmse_poe'] for x in rec['obj_rows']]):.3f} / {_mean([x['rmse_poe'] for x in rec.get('tgt_rows_inf', [])]):.3f}; "
                  f"U sample {_mean(rec['u_rmse_pre']):.3f} / {_mean(rec['u_rmse_held']):.3f}", flush=True)
            print(f"[smoke] h{hi} {task} secs {rec['secs']:.1f}, velocity row-evals {EVALS[0]}, fields {nf}", flush=True)
            if DIAG is not None:
                diagnose(task, hp, DIAG)
        rate = tot_ev / tot_s
        nf = float(np.mean(nfields))
        print(f"[smoke] h{hi} rate {rate:.0f} velocity row-evals per second ({torch.get_num_threads()} threads); fields per demo ~{nf:.0f}", flush=True)
        draws = sorted(settings()[2:], key=lambda h: (h["cost"], h["index"]))
        for name, h in (("heaviest draw", draws[-1]), ("median draw", draws[24]),
                        ("setting 0", settings()[0]), ("setting 1", settings()[1])):
            per_field = len(h["levels"]) * h["n_starts"] * (h["iters"] + 1 + 2 * h["probes"])
            demo_s = nf * per_field / rate
            print(f"[smoke] h{hi} projection {name} (index {h['index']}, cost {h['cost']}): {demo_s:.1f} s/demo; search (60 demos) "
                  f"{60 * demo_s / 3600:.2f} h; read part (25 demos) {25 * demo_s / 3600:.2f} h", flush=True)
        tot = sum(len(h["levels"]) * h["n_starts"] * (h["iters"] + 1 + 2 * h["probes"]) for h in settings())
        print(f"[smoke] h{hi} projection all 50 settings' search: {60 * nf * tot / rate / 3600:.1f} CPU-job hours at this rate", flush=True)


def smoke_settings():
    """0, 1, and the cheapest capped and the cheapest pooled draw at var_floor 1e-2 (ties: lower index)."""
    hs = settings()[2:]
    pick = lambda sc: min([h for h in hs if h["score"] == sc and h["var_floor"] == 1e-2], key=lambda h: (h["cost"], h["index"]))["index"]   # noqa: E731
    return [0, 1, pick("capped"), pick("pooled")]


def diagnose(task, hp, diag):
    """Setting-0 diagnosis per role call: per candidate the prototypes per row, the variance per level (setting's extent and floor),
    the floor share under each extent, the nearest prototype mean's rmse to the chunk; then the decision under every extent x floor x
    rule on the same modes (Y = the truth tops)."""
    to, tt = truth(task)
    for e in diag:
        pools, groups = e["pools"], e["groups"]
        is_obj = e["oi"] is None                                            # object calls vary oi; target calls fix it
        truth_name = FC.ENT[to] if is_obj else FC.ENT[tt]
        label = "object" if is_obj else f"target|obj={FC.ENT[e['oi']]}"
        print(f"[diag] {task} {label} decision over rows {e['rows']} (truth {truth_name}); extent {hp['extent']}, floor {hp['var_floor']}", flush=True)
        for nm, cis in groups.items():
            n, near, H_, V_, L_ = [], [], [], [], []
            for aq, p in zip(e["aq"], pools):
                s = np.isin(p["cand"], cis)
                n.append(int(s.sum()))
                if s.any():
                    near.append(float(np.sqrt(((p["M"][s] - aq) ** 2).mean(1)).min()))
                    H_.append(p["H"][s])
                    V_.append(p["V"][s])
                    L_.append(p["level"][s])
            if not H_:
                continue
            H_, V_, L_ = np.concatenate(H_), np.concatenate(V_), np.concatenate(L_)
            per = []
            for lv in sorted(set(L_.tolist())):
                v = V_[L_ == lv]
                per.append(f"{lv:.2f}:{np.median(v):.1e}[{v.min():.0e},{v.max():.0e}]")
            fl = {ext: float((extent_var(H_, L_, ext, 1e-4) <= 1e-4 * (1 + 1e-6)).mean()) for ext in EXTENTS}
            print(f"[diag] {task} {label} {nm}: protos/row {np.mean(n):.1f}; nearest-mean rmse {np.mean(near):.4f}; floor share (1e-4) "
                  + " ".join(f"{k} {v:.2f}" for k, v in fl.items()) + "; var by level " + " ".join(per), flush=True)
        for rule in RULES:
            cells = []
            for ext in EXTENTS:
                marks = []
                for fl_ in FLOORS:
                    tot = {nm: 0.0 for nm in groups}
                    for aq, p in zip(e["aq"], pools):
                        sc = row_scores(aq, p["M"], extent_var(p["H"], p["level"], ext, fl_), p["cand"], groups, rule, hp["K"], hp["tau"])
                        for nm in groups:
                            tot[nm] += sc[nm]
                    marks.append(f"{fl_:.0e}:{'Y' if argmax_name(tot) == truth_name else 'N(' + argmax_name(tot)[:8] + ')'}")
                cells.append(f"{ext}[{' '.join(marks)}]")
            print(f"[diag] {task} {label} rule {rule}: " + " ".join(cells), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("settings", "smoke", "search", "pick", "read"))
    ap.add_argument("--index", type=int)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--part", type=int)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if a.what == "settings":
        cmd_settings()
    elif a.what == "smoke":
        cmd_smoke(a.index)
    elif a.what == "search":
        cmd_search(a.index, a.task)
    elif a.what == "pick":
        cmd_pick()
    else:
        cmd_read(a.part)

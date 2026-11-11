"""Q2 (docs/THEORY_COMPOSITION.md 4b; SCAFFOLD_RESULTS 36-37): from ONE demonstration and NO instruction, does sc_q0's prototype-
discovery operator, run over a pi0.5 rung's action field, pick the role phrases the demonstration is made of?  Offline, one A40, the
openpi venv; no rollouts, no training.

Rungs (all fine-tuned on K1-K8 of the re-rendered frames, experiments/sc_rv.py; config = pi05_g8v_<arm>_ae):
    r0  plain    whole instructions                   candidates: the six instructions of K1-K6 (retrieve a trained class)
    r1  rsup     role phrases (sc_t16.given)          candidates: object phrases before the hold, target phrases during it
    r2  rview2   r1 + every movable object except the candidate's filler painted magenta by its mask (sc_rv.role_paint/paint_many)
    r2b rbox     r2 with bounding rectangles
  Each painted-rung candidate is evaluated on ITS OWN view: "object: the X" on the row's images painted as if X were the active object
  (role_paint(X, -, held=False)), "target: the Y" as if Y were the active target (role_paint(-, Y, held=True): every movable object but Y,
  all four when Y is a fixture), from the demonstration's own masks, exactly as the dataset painted (sc_rv.frames_of).
  `--cands roles` runs r0 on the role phrases (unpainted) instead of instructions (same code path).

The operator per demo row r (the row's next 10 actions, normalised by the checkpoint's own norm stats through the policy's input
transform, flattened to the 70 real dims), for each candidate c of the role being parsed: sc_q0's discover (Adam ascent on the score
from noised-query / pure-noise starts at each level, Tweedie to clean space, Hutchinson diag of the score Jacobian, dedup, extent_var),
then sc_q0.row_scores summed over the phase's rows (uncapped / capped / pooled; every record carries all three), sc_q0.poe_mean per row.
Object role on 6 evenly spaced rows before the first hold onset, target role on 6 evenly spaced held rows; the hold event is
sc_t16.held_rows (ProprioHeld) at the g8v threshold (results/flawed/sc_rv_lex.json), never the instruction.  r0's instruction
candidates are scored over all 12 rows.  Baseline: flow-matching loss ranking on the same rows and candidates (lowest mean loss of the
demo chunk at openpi t in FM_T, FM_DRAWS noise draws, the same noise for every candidate).  Chance.

From sc_q0.py (imported, unchanged): settings() (for settings 0 and 31), _levels, adam_ascent, extent_var, dedup, loglik_perdim via
row_scores / poe_mean, phase_rows, spaced, argmax_name, boot, RULES, VAR_CAP.  Copied and adapted here (tied to U there):
discover_many's chain layout (-> `discover`), score_jac_diag (with the finite-difference step as an argument, `jac_diag`), fm_role's
loss (-> `run_role`, sharing the operator's prefix), and the part-file / summary / bootstrap plumbing of run_unit / cmd_read.

Splits (demo index per task K1-K6, rendered under /coc/flash10/ksingara3/robot/data/fa1/rv): search 0-4, read 25-34.

    python experiments/sc_q2.py settings  --rung R --ckpt DIR
    python experiments/sc_q2.py smoke     --rung R --ckpt DIR [--config NAME] [--index i] [--task K1]
    python experiments/sc_q2.py search    --rung R --ckpt DIR --index i [--task K1]   -> results/flawed/sc_q2/<R>/search_{i}.json
    python experiments/sc_q2.py pick      --rung R --ckpt DIR                         -> results/flawed/sc_q2/<R>/picked.json
    python experiments/sc_q2.py read      --rung R --ckpt DIR [--task K1]             -> results/flawed/sc_q2/<R>/read.json
    python experiments/sc_q2.py demo      --rung R --ckpt DIR --path FILE.npz [--tid T] [--index i]
Part files (results/flawed/sc_q2/<R>/parts/) cache one (split, setting, task) unit each and are never overwritten; with --task only
that task's part file(s) are computed.  <R> is the rung, plus "_roles" for r0 under --cands roles.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OPENPI = "/coc/flash10/ksingara3/robot/openpi"
sys.path[:0] = [HERE, ROOT, OPENPI + "/src"]
import sc_q0 as Q0                                                          # noqa: E402
import sc_rv as RV                                                          # noqa: E402
from lifelong_ood.policy import data as D                                   # noqa: E402

T16 = RV.t16()                                                              # sc_t16 in its supervised stage: given(), held_rows(), PHRASE
OUT = os.path.join(ROOT, "results", "flawed")
TASKS = ("K1", "K2", "K3", "K4", "K5", "K6")
SPLITS = dict(search=tuple(range(0, 5)), read=tuple(range(25, 35)))
RUNGS = dict(r0=dict(config="pi05_g8v_plain_ae", paint=None, cands="instr"),
             r1=dict(config="pi05_g8v_rsup_ae", paint=None, cands="roles"),
             r2=dict(config="pi05_g8v_rview2_ae", paint="mask", cands="roles"),
             r2b=dict(config="pi05_g8v_rbox_ae", paint="box", cands="roles"))
H_ACT, A_REAL = 10, 7
DA = H_ACT * A_REAL                                                         # 70 real dims
FM_T, FM_DRAWS = (0.1, 0.3, 0.5), 8                                         # openpi t (t = 1 noise): sigma .1/.3/.5 as sc_q0's FM_T
FD_DELTA = 5e-3                                                             # finite-difference step of the Hutchinson probes (sc_q0's)
MAXCH = 384                                                                 # chains per action-expert forward (KV copies ~18 MB each)
PRE_BLOCK = 16                                                              # (row, candidate) prefixes per VLM pass
EVALS = [0]                                                                 # chain-evaluations of the expert (timing)
VERBOSE = False


# ---- settings ------------------------------------------------------------------------------------------------------------------
def settings():
    """0 = plan v4's (sc_q0 setting 0); 1 = sc_q0's picked setting 31; 2-9 = setting 1 with one field changed each."""
    q = Q0.settings()
    s0, s1 = dict(q[0]), dict(q[31])
    assert s1["levels"] == Q0._levels(3, 0.05, 0.4) and s1["n_starts"] == 4 and s1["init"] == "hybrid" and s1["iters"] == 50 \
        and s1["lr"] == 0.02 and s1["K"] == 5 and s1["tau"] == 0.05 and s1["score"] == "capped" and s1["extent"] == "marginal" \
        and s1["var_floor"] == 1e-2, s1
    s0["src"], s1["src"] = "sc_q0 setting 0 (plan v4)", "sc_q0 setting 31 (picked)"
    var = [("levels (0.1, 0.7)", dict(levels=Q0._levels(3, 0.1, 0.7), sigma_range=[0.1, 0.7])),
           ("levels (0.3, 0.9)", dict(levels=Q0._levels(3, 0.3, 0.9), sigma_range=[0.3, 0.9])),
           ("16 starts", dict(n_starts=16)),
           ("floor 1e-3", dict(var_floor=1e-3)),
           ("floor 1e-1", dict(var_floor=1e-1)),
           ("extent posterior", dict(extent="posterior")),
           ("score pooled", dict(score="pooled")),
           ("K 1", dict(K=1))]
    out = [s0, s1] + [dict(s1, src="setting 1, " + nm, **ch) for nm, ch in var]
    for i, h in enumerate(out):
        h["index"] = i
        h["cost"] = h["n_starts"] * len(h["levels"]) * h["iters"]
    return out


# ---- candidates and truth ------------------------------------------------------------------------------------------------------
SLOT_OF = {v: k for k, v in T16.PHRASE.items()}                             # "the bowl" -> akita_black_bowl_1


def phrase_slot(p):
    return SLOT_OF[p.split(": ", 1)[1]]


def role_phrases():
    """The object and target phrases the role rungs were trained with for K1-K6 (sc_t16.given), in task order."""
    L = RV.lang()
    P = [T16.given(L[c]) for c in TASKS]
    return list(dict.fromkeys(p[0] for p in P)), list(dict.fromkeys(p[1] for p in P))


def cands_roles(paint):
    """-> (object candidates, target candidates); each dict(name, prompt, view, obj, tgt); view = (slots to paint, active slot)."""
    objs, tgts = role_phrases()
    oc = [dict(name=p, prompt=p, obj=phrase_slot(p), tgt=None, view=RV.role_paint(phrase_slot(p), None, False) if paint else None)
          for p in objs]
    tc = [dict(name=p, prompt=p, obj=None, tgt=phrase_slot(p), view=RV.role_paint(None, phrase_slot(p), True) if paint else None)
          for p in tgts]
    return oc, tc


def cands_instr():
    L, (by_c, _) = RV.lang(), RV.roles_of()
    return [dict(name=c, prompt=L[c], obj=by_c[c][0], tgt=by_c[c][1], view=None) for c in TASKS]


def truth_of(ex, tid=None):
    """Ground truth for SCORING only (never seen by the operator): (object slot, target slot, compound or None)."""
    meta = RV.n2()
    if tid is not None:
        t = next(t for t in meta["tasks"] if int(t["tid"]) == int(tid))
        return t["object"], t["target"], t["compound"]
    by_c, by_l = RV.roles_of()
    if ex["task"] in by_c and RV.lang().get(ex["task"]) == ex["language"]:
        o, t = by_c[ex["task"]]
        return o, t, ex["task"]
    if ex["language"] in by_l:
        o, t = by_l[ex["language"]]
        return o, t, next((c for c, l in RV.lang().items() if l == ex["language"]), None)
    return None, None, None


# ---- demos ---------------------------------------------------------------------------------------------------------------------
_LEX = {}


def lex():
    if not _LEX:
        _LEX.update(json.load(open(RV.lex_path(""), encoding="utf-8")))
    return _LEX


def load_demo(path):
    """A rendered demo (sc_rv render layout) -> rows, the proprioceptive hold event (g8v threshold), raw 10-step chunks as LeRobot
    builds them (indices clamped to the episode end, D.chunk_indices)."""
    z = dict(np.load(path))
    X = lex()
    held = T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]).astype(np.float32)
    idx, _ = D.chunk_indices(len(z["actions"]), H_ACT)
    return dict(z=z, held=held, n=len(held), chunks=np.asarray(z["actions"], np.float32)[idx], task=str(z["task"]),
                demo=int(z["demo"]), language=str(z["language"]), path=path, _q={})


def demo_ex(task, demo):
    return load_demo(RV.demo_path(RV.RV_ROOT, task, demo))


def view(ex, r, cand, paint):
    """Row r's two images under the candidate's view (unchanged when the rung does not paint)."""
    z = ex["z"]
    img, wimg = z["image"][r], z["wrist"][r]
    if paint and cand["view"] is not None:
        slots, act = cand["view"]
        sid = lambda s: RV.slot_index(z["names"], z["slot_names"], z["slot_ents"], s)   # noqa: E731
        pid = [sid(s) for s in slots]
        assert all(x is not None for x in pid), (ex["path"], slots, list(z["names"]))
        aid = None if act is None else sid(act)
        img, _ = RV.paint_many(img, z["seg"][r], pid, aid, box=paint == "box")
        wimg, _ = RV.paint_many(wimg, z["wseg"][r], pid, aid, box=paint == "box")
    return img, wimg


def obs_dict(ex, r, cand, paint):
    img, wimg = view(ex, r, cand, paint)
    return {"observation/image": np.asarray(img), "observation/wrist_image": np.asarray(wimg),
            "observation/state": np.asarray(ex["z"]["state"][r], np.float32), "prompt": cand["prompt"]}


def query(bk, ex, r):
    """The demo's chunk at row r in the model's normalised action space, 70 real dims: the policy's own input transform (Normalize
    with the checkpoint's norm stats, quantile for pi0.5, then PadStatesAndActions), exactly the training target."""
    if r not in ex["_q"]:
        o = obs_dict(ex, r, dict(prompt="", view=None), None)
        o["actions"] = ex["chunks"][r]
        a = np.asarray(bk.policy._input_transform(o)["actions"], np.float32)
        assert a.shape == (H_ACT, bk.A), a.shape
        ex["_q"][r] = a[:, :A_REAL].reshape(-1).copy()
    return ex["_q"][r]


def row_seed(task, demo, row):
    k = TASKS.index(task) if task in TASKS else 9
    return ((7 * 10 + k) * 100 + int(demo)) * 1000 + int(row)


# ---- the field adapter ---------------------------------------------------------------------------------------------------------
def rep_rows(pre, counts):
    """A Prefix whose row i is repeated counts[i] times (0 drops it), as PolicyBackbone.repeat / store.Live._rows build it."""
    c = torch.as_tensor(counts, dtype=torch.long, device=pre.state.device)
    rep = lambda t: t.repeat_interleave(c, dim=0)                           # noqa: E731
    kv2 = type(pre.kv)()
    for layer, (kk, vv) in enumerate(zip(pre.kv.key_cache, pre.kv.value_cache)):
        kv2.update(rep(kk), rep(vv), layer)
    return dataclasses.replace(pre, state=rep(pre.state), pad_masks=rep(pre.pad_masks), kv=kv2, n=int(c.sum()))


class VLAField:
    """A pi0.5 checkpoint's field over the 70 real action dims at fixed (observation, candidate prompt, candidate view) per chain, in
    the convention sc_q0's operator code expects (UField's interface: eps / score / tweedie / velocity of (x [n, 70], sigma [n])).

    Time convention.  openpi (pi0_pytorch.forward): x_t = t eps + (1 - t) a, target u = eps - a, t = 1 is NOISE.  The paper / sc_q0:
    x = alpha a + sigma eps.  Matching terms: sigma = t, alpha = 1 - t (the same alpha = 1 - sigma that sc_q0.extent_var and the
    hybrid starts assume).  From x and v = eps - a:
        a_hat   = x - t v          [t e + (1-t) a - t (e - a) = a]
        eps_hat = x + (1 - t) v    [t e + (1-t) a + (1-t)(e - a) = e]
    so eps(x, sigma) = x + (1 - sigma) v(x, t = sigma), score = -eps / sigma, Tweedie (x + sigma^2 score) / alpha = x - sigma v = a_hat.
    (U's t_U = 1 - sigma; this model's t = sigma.  The same field, two parameterisations.)

    Padded dims.  The model sees [10, 32]; dims 7-31 are trained on a = 0, so at time t their marginal is t * eps_pad, independent of
    the real dims (backbone.py's `pad`).  backbone.pad redraws that noise on every call, so x + d v and x - d v of one finite
    difference (sent as one batch of 2n rows) would see different padding.  Here each chain draws its padding z_pad ONCE and feeds
    sigma * z_pad on every call (same law, held fixed along the chain: ascent steps, Tweedie and both sides of every probe).

    pre: a Prefix already repeated to the chain count n (one row per chain); calls with a multiple of n rows are split into n-row
    expert forwards against the same cached prefix (x +- d v)."""

    def __init__(self, bk, pre, zpad):
        self.bk, self.pre, self.zpad = bk, pre, zpad
        self.real = torch.as_tensor(bk.real, device=bk.device)
        padm = np.ones(bk.dim_full, bool)
        padm[bk.real] = False
        self.padi = torch.as_tensor(np.flatnonzero(padm), device=bk.device)

    def velocity(self, x, sig):
        n0 = self.pre.n
        assert len(x) % n0 == 0, (len(x), n0)
        outs = []
        for s in range(0, len(x), n0):
            xs, ss = x[s:s + n0], sig[s:s + n0]
            full = torch.empty((n0, self.bk.dim_full), dtype=torch.float32, device=self.bk.device)
            full[:, self.real] = xs
            full[:, self.padi] = ss[:, None] * self.zpad
            outs.append(self.bk.v_full(full, ss, self.pre)[:, self.real])
            EVALS[0] += n0
        return torch.cat(outs)

    def eps(self, x, sigma):
        return x + (1 - sigma)[:, None] * self.velocity(x, sigma)

    def score(self, x, sigma):
        return -self.eps(x, sigma) / sigma[:, None]

    def tweedie(self, x, sigma):
        return x - sigma[:, None] * self.velocity(x, sigma)


@torch.no_grad()
def jac_diag(f, x, sig, probes, delta):
    """sc_q0.score_jac_diag with the finite-difference step as an argument (copied: sc_q0 reads its module constant)."""
    h = torch.zeros_like(x)
    n = x.shape[0]
    for v in probes:
        s = f.score(torch.cat([x + delta * v, x - delta * v]), torch.cat([sig, sig]))
        h += v * (s[:n] - s[n:]) / (2 * delta)
    return h / len(probes)


def blocks_of(n_jobs, per_job, n_cands):
    """Consecutive job ranges of at most PRE_BLOCK jobs and MAXCH chains, cut at row boundaries (jobs are row-major) so that every
    candidate of a row runs in the same expert batch: bf16 kernels give batch-shape-dependent outputs (smoke: up to ~0.1 in v), and
    candidates compared on one row must not differ by that."""
    k = max(1, min(PRE_BLOCK, MAXCH // max(1, per_job)))
    k = max(n_cands, (k // n_cands) * n_cands)
    return [(s, min(s + k, n_jobs)) for s in range(0, n_jobs, k)]


# ---- one role ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_role(bk, ex, rows, cands, hp, paint, fm=True):
    """Discover over `cands` at each row (sc_q0.discover_many's layout: per row common random numbers for every candidate) and the
    flow-matching loss on the same prefixes.  -> dict(scores {rule: {name: sum over rows}}, fm {name: mean loss}, rows [...])."""
    dev = bk.device
    names = [c["name"] for c in cands]
    groups = {nm: [i] for i, nm in enumerate(names)}
    lv = np.asarray(hp["levels"], np.float32)
    L, S = len(lv), int(hp["n_starts"])
    N = L * S
    nq = S if hp["init"] == "query" else S - S // 2
    sig1 = torch.tensor(np.repeat(lv, S), device=dev)
    isq1 = np.tile(np.r_[np.ones(nq, bool), np.zeros(S - nq, bool)], L)
    al, sg = (1 - sig1)[:, None], sig1[:, None]
    npad = bk.dim_full - DA
    real_t = torch.as_tensor(bk.real, device=dev)
    per_row = {}
    for r in rows:                                                          # common random numbers per row
        gen = torch.Generator().manual_seed(row_seed(ex["task"], ex["demo"], r))
        z = torch.randn(N, DA, generator=gen)
        zp = torch.randn(N, npad, generator=gen)
        pv = torch.randint(0, 2, (int(hp["probes"]), N, DA), generator=gen).float() * 2 - 1
        fe = torch.randn(len(FM_T) * FM_DRAWS, bk.dim_full, generator=gen)
        a = torch.as_tensor(query(bk, ex, r)).reshape(1, -1).to(dev)
        z, zp, pv, fe = z.to(dev), zp.to(dev), pv.to(dev), fe.to(dev)
        x0 = torch.where(torch.as_tensor(isq1, device=dev)[:, None], al * a + sg * z, torch.sqrt(al ** 2 + sg ** 2) * z)
        per_row[r] = dict(a=a, x0=x0, zp=zp, pv=pv, fe=fe)
    jobs = [(r, ci) for r in rows for ci in range(len(cands))]
    M0 = np.zeros((len(jobs), N, DA), np.float32)
    H0 = np.zeros_like(M0)
    FMl = np.zeros(len(jobs), np.float64)
    t_pre = t_asc = t_fm = 0.0
    nT = len(FM_T) * FM_DRAWS
    tF = torch.tensor(np.repeat(FM_T, FM_DRAWS), dtype=torch.float32, device=dev)
    for s, e in blocks_of(len(jobs), max(N, nT), len(cands)):
        t0 = time.time()
        pre = bk.prefix(bk.observations([obs_dict(ex, r, cands[ci], paint) for r, ci in jobs[s:e]]))
        t_pre += time.time() - t0
        t0 = time.time()
        B = e - s
        rep = rep_rows(pre, [N] * B)
        X = torch.cat([per_row[r]["x0"] for r, _ in jobs[s:e]])
        ZP = torch.cat([per_row[r]["zp"] for r, _ in jobs[s:e]])
        PV = torch.cat([per_row[r]["pv"] for r, _ in jobs[s:e]], 1)
        SIG = sig1.repeat(B)
        f = VLAField(bk, rep, ZP)
        x = Q0.adam_ascent(f, X, SIG, hp["iters"], hp["lr"])
        M0[s:e] = f.tweedie(x, SIG).reshape(B, N, DA).float().cpu().numpy()
        H0[s:e] = jac_diag(f, x, SIG, PV, FD_DELTA).reshape(B, N, DA).float().cpu().numpy()
        del rep, f
        t_asc += time.time() - t0
        if fm:                                                              # the baseline on the same cached prefixes
            t0 = time.time()
            repf = rep_rows(pre, [nT] * B)
            a_full = torch.zeros((B * nT, bk.dim_full), device=dev)          # padded dims: data value 0, as in training
            a_full[:, real_t] = torch.cat([per_row[r]["a"].expand(nT, -1) for r, _ in jobs[s:e]])
            eps = torch.cat([per_row[r]["fe"] for r, _ in jobs[s:e]])
            tt = tF.repeat(B)
            xt = tt[:, None] * eps + (1 - tt[:, None]) * a_full               # openpi's x_t; the loss is openpi's, on the 70 real dims
            v = bk.v_full(xt, tt, repf)
            EVALS[0] += B * nT
            err = ((v - (eps - a_full))[:, real_t] ** 2).mean(1).reshape(B, nT).mean(1)
            FMl[s:e] = err.double().cpu().numpy()
            del repf
            t_fm += time.time() - t0
        del pre
    out = dict(scores={rule: {nm: 0.0 for nm in names} for rule in Q0.RULES}, rows=[], fm={nm: 0.0 for nm in names},
               secs=dict(prefix=t_pre, ascent=t_asc, fm=t_fm))
    for j, r in enumerate(rows):
        ks = [jobs.index((r, ci)) for ci in range(len(cands))]
        Ms, Hs, cs, lvs, qs = [], [], [], [], []
        n_raw = 0
        for ci, k in enumerate(ks):
            keep = Q0.dedup(M0[k], hp["dedup"])
            n_raw += N
            Ms.append(M0[k][keep])
            Hs.append(H0[k][keep])
            cs.append(np.full(len(keep), ci))
            lvs.append(np.repeat(lv, S)[keep])
            qs.append(isq1[keep])
        M, Hh, cand, level, qst = (np.concatenate(v) for v in (Ms, Hs, cs, lvs, qs))
        V = Q0.extent_var(Hh, level, hp["extent"], hp["var_floor"])
        aq = query(bk, ex, r)
        cov = {rule: Q0.row_scores(aq, M, V, cand, groups, rule, hp["K"], hp["tau"]) for rule in Q0.RULES}
        for rule in Q0.RULES:
            for nm in names:
                out["scores"][rule][nm] += cov[rule][nm]
        fm_row = {names[ci]: float(FMl[k]) for ci, k in enumerate(ks)}
        for nm in names:
            out["fm"][nm] += fm_row[nm] / len(rows)
        mu, sel = Q0.poe_mean(aq, M, V, hp["K"], hp["tau"])
        floor_share = float((V <= hp["var_floor"] * (1 + 1e-6)).mean())
        out["rows"].append(dict(row=int(r), scores_by_rule=cov, fm=fm_row, n_pool=int(len(M)), n_raw=n_raw,
                                n_per_cand={nm: int((cand == ci).sum()) for ci, nm in enumerate(names)},
                                nearest_rmse={nm: float(np.sqrt(((M[cand == ci] - aq) ** 2).mean(1)).min()) for ci, nm in enumerate(names)},
                                selected=[names[int(cand[k])] for k in sel], selected_level=[float(level[k]) for k in sel],
                                floor_share=floor_share, rmse_poe=float(np.sqrt(np.mean((mu - aq) ** 2)))))
    if VERBOSE:
        print(f"[q2] role {len(rows)} rows x {len(cands)} cands, {N} chains each: prefix {t_pre:.1f}s ascent {t_asc:.1f}s fm {t_fm:.1f}s",
              flush=True)
    return out


def argbest(d, low=False):
    return Q0.argmax_name(d, low=low)


KEYS = ("obj", "tgt", "joint")


def parse_demo(bk, ex, hp, rung, mode, tid=None):
    """The Q2 parse of one demo -> JSON-able record.  Truth only fills the *_ok fields."""
    t0 = time.time()
    paint = RUNGS[rung]["paint"]
    to, tt, tc = truth_of(ex, tid)
    pre, hrows = Q0.phase_rows(ex, hp["rows_per_phase"])
    rec = dict(task=ex["task"], demo=ex["demo"], path=ex["path"], setting=int(hp["index"]), rung=rung, mode=mode, n_rows=ex["n"],
               pre_rows=[int(r) for r in pre], held_rows=[int(r) for r in hrows], no_event=bool(len(hrows) == 0),
               true_obj=to, true_tgt=tt, true_task=tc)
    rules = {}
    if mode == "roles":
        oc, tc_ = cands_roles(paint)
        ro = run_role(bk, ex, pre, oc, hp, paint)
        rt = run_role(bk, ex, hrows, tc_, hp, paint) if len(hrows) else None
        slot_o = {c["name"]: c["obj"] for c in oc}
        slot_t = {c["name"]: c["tgt"] for c in tc_}
        rec["obj_role"], rec["tgt_role"] = ro, rt
        for rule in Q0.RULES:
            po = argbest(ro["scores"][rule])
            pt = argbest(rt["scores"][rule]) if rt else None
            ok = dict(obj=slot_o.get(po) == to, tgt=pt is not None and slot_t.get(pt) == tt)
            ok["joint"] = ok["obj"] and ok["tgt"]
            rules[rule] = dict(obj_pred=po, tgt_pred=pt, ok=ok)
        fo, ft = argbest(ro["fm"], low=True), (argbest(rt["fm"], low=True) if rt else None)
        rec["fm_pred"] = dict(obj=fo, tgt=ft)
        rec["fm_ok"] = dict(obj=slot_o.get(fo) == to, tgt=ft is not None and slot_t.get(ft) == tt)
        rec["chance"] = dict(obj=1.0 / len(oc), tgt=1.0 / len(tc_))
        rec["chance"]["joint"] = rec["chance"]["obj"] * rec["chance"]["tgt"]
    else:
        ic = cands_instr()
        rows = np.concatenate([pre, hrows])
        ri = run_role(bk, ex, rows, ic, hp, paint)
        rec["instr_role"] = ri
        by = {c["name"]: c for c in ic}
        for rule in Q0.RULES:
            p = argbest(ri["scores"][rule])
            ok = dict(obj=by[p]["obj"] == to, tgt=by[p]["tgt"] == tt)
            ok["joint"] = p == tc
            rules[rule] = dict(obj_pred=by[p]["obj"], tgt_pred=by[p]["tgt"], instr_pred=p, ok=ok)
        f = argbest(ri["fm"], low=True)
        rec["fm_pred"] = dict(obj=by[f]["obj"], tgt=by[f]["tgt"], instr=f)
        rec["fm_ok"] = dict(obj=by[f]["obj"] == to, tgt=by[f]["tgt"] == tt)
        rec["fm_ok"]["joint"] = f == tc
        rec["chance"] = dict(obj=float(np.mean([c["obj"] == to for c in ic])), tgt=float(np.mean([c["tgt"] == tt for c in ic])),
                             joint=1.0 / len(ic))
    if mode == "roles":
        rec["fm_ok"]["joint"] = rec["fm_ok"]["obj"] and rec["fm_ok"]["tgt"]
    rec["rules"] = rules
    rec["ok"] = rules[hp["score"]]["ok"]
    rec["pred"] = {k: v for k, v in rules[hp["score"]].items() if k != "ok"}
    rec["secs"] = time.time() - t0
    return rec


# ---- units, summaries ----------------------------------------------------------------------------------------------------------
def tag_of(a):
    return a.rung + ("_roles" if a.rung == "r0" and a.cands == "roles" else "")


def od(a):
    return os.path.join(OUT, "sc_q2", tag_of(a))


def mode_of(a):
    return a.cands or RUNGS[a.rung]["cands"]


_BK = {}


def backbone(a):
    if "bk" not in _BK:
        from lifelong_ood.policy.backbone import PolicyBackbone
        name = a.config or RUNGS[a.rung]["config"]
        _BK["bk"] = PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
        print("[q2]", _BK["bk"].describe(), "ckpt", a.ckpt, flush=True)
    return _BK["bk"]


def run_unit(a, split, h, task):
    fn = os.path.join(od(a), "parts", f"{split}_h{h}_{task}.json")
    if os.path.exists(fn):
        d = json.load(open(fn, encoding="utf-8"))
        assert os.path.realpath(d["ckpt"]) == os.path.realpath(a.ckpt), (fn, d["ckpt"], a.ckpt)
        return d["records"]
    hp = settings()[h]
    bk = backbone(a)
    recs = []
    for dm in SPLITS[split]:
        r = parse_demo(bk, demo_ex(task, dm), hp, a.rung, mode_of(a))
        recs.append(r)
        print(f"[q2] {tag_of(a)} {split} h{h} {task} d{dm} pred {r['pred']} ok {r['ok']} fm {r['fm_pred']} {r['secs']:.1f}s", flush=True)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if os.path.exists(fn):
        raise SystemExit(f"{fn} appeared while computing; not overwriting")
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(split=split, setting=hp, task=task, rung=a.rung, mode=mode_of(a), ckpt=a.ckpt, config=a.config or RUNGS[a.rung]["config"],
                       host=os.uname().nodename, records=recs), fh)
    os.replace(fn + ".tmp", fn)
    return recs


def summarize(recs):
    def one(rs):
        o = dict(acc={k: float(np.mean([r["ok"][k] for r in rs])) for k in KEYS},
                 fm_acc={k: float(np.mean([r["fm_ok"][k] for r in rs])) for k in KEYS},
                 acc_by_rule={rule: {k: float(np.mean([r["rules"][rule]["ok"][k] for r in rs])) for k in KEYS} for rule in Q0.RULES},
                 chance={k: float(np.mean([r["chance"][k] for r in rs])) for k in KEYS},
                 no_event=int(sum(r["no_event"] for r in rs)), secs_per_demo=float(np.mean([r["secs"] for r in rs])), n=len(rs))
        rows = [x for r in rs for k in ("obj_role", "tgt_role", "instr_role") if r.get(k) for x in r[k]["rows"]]
        o["rmse_poe"] = float(np.mean([x["rmse_poe"] for x in rows])) if rows else None
        o["floor_share"] = float(np.mean([x["floor_share"] for x in rows])) if rows else None
        o["protos_per_row"] = float(np.mean([x["n_pool"] for x in rows])) if rows else None
        return o
    res = {t: one([r for r in recs if r["task"] == t]) for t in TASKS if any(r["task"] == t for r in recs)}
    res["pooled"] = one(recs)
    return res


# ---- commands ------------------------------------------------------------------------------------------------------------------
def cmd_settings(a):
    for h in settings():
        print(json.dumps(h))


def cmd_search(a):
    i = a.index
    assert i is not None and 0 <= i < len(settings()), i
    fn = os.path.join(od(a), f"search_{i}.json")
    if a.task:
        run_unit(a, "search", i, a.task)
        return
    if os.path.exists(fn):
        raise SystemExit(f"{fn} exists")
    t0 = time.time()
    recs = [r for t in TASKS for r in run_unit(a, "search", i, t)]
    res = dict(setting=settings()[i], split="search", demos=list(SPLITS["search"]), rung=a.rung, mode=mode_of(a), ckpt=a.ckpt,
               summary=summarize(recs), records=recs, secs_aggregate=time.time() - t0, host=os.uname().nodename)
    os.makedirs(od(a), exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    os.replace(fn + ".tmp", fn)
    print("[q2] search", i, json.dumps(res["summary"]["pooled"]), flush=True)


def cmd_pick(a):
    fn = os.path.join(od(a), "picked.json")
    if os.path.exists(fn):
        raise SystemExit(f"{fn} exists")
    hs = settings()
    missing = [h["index"] for h in hs if not os.path.exists(os.path.join(od(a), f"search_{h['index']}.json"))]
    if missing:
        raise SystemExit(f"search files missing: {missing}")
    table = []
    for h in hs:
        s = json.load(open(os.path.join(od(a), f"search_{h['index']}.json"), encoding="utf-8"))["summary"]["pooled"]
        table.append(dict(index=h["index"], joint=s["acc"]["joint"], obj=s["acc"]["obj"], tgt=s["acc"]["tgt"],
                          fm_joint=s["fm_acc"]["joint"], secs_per_demo=s["secs_per_demo"]))
    best = sorted(table, key=lambda r: (-r["joint"], r["index"]))[0]
    res = dict(index=best["index"], rule="highest search-split joint accuracy; ties to the lower index", setting=hs[best["index"]],
               table=table, ckpt=a.ckpt)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[q2] picked", json.dumps(best), flush=True)


def picked(a):
    return json.load(open(os.path.join(od(a), "picked.json"), encoding="utf-8"))["index"]


def cmd_read(a):
    fn = os.path.join(od(a), "read.json")
    if os.path.exists(fn):
        raise SystemExit(f"{fn} exists")
    p = picked(a)
    hs = [p] if p == 0 else [p, 0]
    if a.task:
        for h in hs:
            run_unit(a, "read", h, a.task)
        return
    t0 = time.time()
    R = {}
    for h in hs:
        for t in TASKS:
            for r in run_unit(a, "read", h, t):
                R[(h, t, r["demo"])] = r
    dm = SPLITS["read"]

    def arr(h, fam, key, rule=None):
        get = (lambda r: r[fam][key]) if rule is None else (lambda r: r["rules"][rule]["ok"][key])   # noqa: E731
        return {t: np.array([[float(get(R[(h, t, d)]))] for d in dm]) for t in TASKS}

    res = dict(split="read", demos=list(dm), rung=a.rung, mode=mode_of(a), ckpt=a.ckpt, picked=p,
               settings={str(h): settings()[h] for h in sorted(set(hs))})
    for h in sorted(set(hs)):
        recs = [R[(h, t, d)] for t in TASKS for d in dm]
        res[f"setting_{h}"] = dict(operator={k: Q0.boot(arr(h, "ok", k)) for k in KEYS},
                                   fm_loss={k: Q0.boot(arr(h, "fm_ok", k)) for k in KEYS},
                                   other_rules={rule: {k: Q0.boot(arr(h, None, k, rule)) for k in KEYS}
                                                for rule in Q0.RULES if rule != settings()[h]["score"]},
                                   summary=summarize(recs),
                                   wall_secs_per_demo=float(np.mean([r["secs"] for r in recs])))
    res["chance"] = res[f"setting_{p}"]["summary"]["pooled"]["chance"]
    res["picked_minus_0"] = {k: Q0.boot({t: arr(p, "ok", k)[t] - arr(0, "ok", k)[t] for t in TASKS}) for k in KEYS}
    res["picked_minus_fm"] = {k: Q0.boot({t: arr(p, "ok", k)[t] - arr(p, "fm_ok", k)[t] for t in TASKS}) for k in KEYS}
    res["secs_aggregate"] = time.time() - t0
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(fn + ".tmp", fn)
    print("[q2] read", json.dumps({k: res[f"setting_{p}"]["operator"][k]["pooled"] for k in KEYS}), flush=True)


def print_role(tag, role, hp, lowfm=True):
    if role is None:
        print(f"[q2] {tag}: no rows", flush=True)
        return
    for rule in Q0.RULES:
        d = role["scores"][rule]
        print(f"[q2] {tag} rule {rule}{' (setting)' if rule == hp['score'] else ''}: "
              + ", ".join(f"{k!r} {v:.5g}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])), flush=True)
    print(f"[q2] {tag} flow-loss (lower wins): " + ", ".join(f"{k!r} {v:.5f}" for k, v in sorted(role["fm"].items(), key=lambda kv: kv[1])),
          flush=True)


def cmd_demo(a):
    global VERBOSE
    VERBOSE = True
    h = a.index if a.index is not None else picked(a)
    hp = settings()[h]
    bk = backbone(a)
    ex = load_demo(a.path)
    rec = parse_demo(bk, ex, hp, a.rung, mode_of(a), tid=a.tid)
    print(f"[q2] demo {a.path} (task field {ex['task']!r}; the instruction is not read) setting {h}; rows {ex['n']}, pre {rec['pre_rows']}, "
          f"held {rec['held_rows']}", flush=True)
    for k in ("obj_role", "tgt_role", "instr_role"):
        if k in rec:
            print_role(k, rec[k], hp)
    print(f"[q2] CHOSEN {json.dumps(rec['pred'])}; flow-loss baseline {json.dumps(rec['fm_pred'])}; truth (scoring only) obj {rec['true_obj']} "
          f"tgt {rec['true_tgt']} task {rec['true_task']}; ok {rec['ok']} fm_ok {rec['fm_ok']}; {rec['secs']:.1f}s", flush=True)


# ---- smoke ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def smoke_checks(bk, a):
    """Plumbing checks on demo 0 of K1: prefix cache not mutated, batch invariance, padded-dim sensitivity, the finite-difference step
    in bf16, and Tweedie from a noised demo chunk under the true and the wrong candidates."""
    paint = RUNGS[a.rung]["paint"]
    mode = mode_of(a)
    ex = demo_ex("K1", 0)
    pre, hrows = Q0.phase_rows(ex, 6)
    to, tt, tc = truth_of(ex)
    print(f"[smoke] K1 d0 rows {ex['n']} held {int(ex['held'].sum())} pre {pre.tolist()} held {hrows.tolist()} truth {to} {tt}", flush=True)
    dev = bk.device
    g = torch.Generator().manual_seed(0)
    r = int(pre[len(pre) // 2])
    aq = torch.as_tensor(query(bk, ex, r), device=dev)
    raw = ex["chunks"][r]
    print(f"[smoke] row {r} raw chunk[0] {np.round(raw[0], 3).tolist()} -> normalised {np.round(aq[:7].cpu().numpy(), 3).tolist()}; "
          f"normalised range [{float(aq.min()):.2f}, {float(aq.max()):.2f}]", flush=True)
    if mode == "roles":
        oc, tc_ = cands_roles(paint)
    else:
        oc = tc_ = cands_instr()
    true_o = next(c for c in oc if c["obj"] == to)
    p1 = bk.prefix(bk.observations([obs_dict(ex, r, true_o, paint)]))
    n = 32
    x = torch.randn(n, DA, generator=g).to(dev)
    sig = torch.full((n,), 0.3, device=dev)
    zp = torch.randn(n, bk.dim_full - DA, generator=g).to(dev)
    f = VLAField(bk, rep_rows(p1, [n]), zp)
    v1, v2 = f.velocity(x, sig), f.velocity(x, sig)
    print(f"[smoke] same prefix twice: max |dv| {float((v1 - v2).abs().max()):.2e} (0 = the cache is not mutated)", flush=True)
    f2 = VLAField(bk, rep_rows(p1, [2 * n]), torch.cat([zp, zp]))
    v3 = f2.velocity(torch.cat([x, x]), torch.cat([sig, sig]))[:n]
    print(f"[smoke] batch 32 vs 64: max |dv| {float((v1 - v3).abs().max()):.2e} mean |dv| {float((v1 - v3).abs().mean()):.2e}; "
          f"|v| mean {float(v1.abs().mean()):.3f}", flush=True)
    f0 = VLAField(bk, rep_rows(p1, [n]), torch.zeros_like(zp))
    fz = VLAField(bk, rep_rows(p1, [n]), torch.randn(zp.shape, generator=g).to(dev))
    print(f"[smoke] padded dims: |v(pad 0) - v(pad sigma z)| {float((f0.velocity(x, sig) - v1).abs().mean()):.3e}, "
          f"|v(pad z') - v(pad z)| {float((fz.velocity(x, sig) - v1).abs().mean()):.3e} (mean abs, real dims)", flush=True)
    # finite-difference step in bf16: directional second derivative estimates at noised demo points
    for s in (0.05, 0.2, 0.4):
        sg = torch.full((n,), s, device=dev)
        xs = (1 - s) * aq[None] + s * torch.randn(n, DA, generator=g).to(dev)
        pv = (torch.randint(0, 2, (4, n, DA), generator=g).float() * 2 - 1).to(dev)
        hs = {d: jac_diag(f, xs, sg, pv, d) for d in (1e-3, 5e-3, 2e-2, 5e-2)}
        ref = hs[5e-2].flatten()
        msg = " ".join(f"d{d:g}: mean {float(hh.mean()):.2f} sd {float(hh.std()):.2f} corr-to-d.05 "
                       f"{float(torch.corrcoef(torch.stack([hh.flatten(), ref]))[0, 1]):.3f}" for d, hh in hs.items())
        print(f"[smoke] jac diag at sigma {s} (a point mass gives -1/sigma^2 = {-1 / s ** 2:.1f}): {msg}", flush=True)
    # Tweedie from the noised demo chunk, true vs wrong candidates, a pre row and a held row
    for nm, rr, cs, tslot, key in (("pre", pre, oc, to, "obj"), ("held", hrows, tc_, tt, "tgt")):
        if len(rr) == 0:
            continue
        r = int(rr[len(rr) // 2])
        aq = torch.as_tensor(query(bk, ex, r), device=dev)
        ps = bk.prefix(bk.observations([obs_dict(ex, r, c, paint) for c in cs]))
        for s in (0.05, 0.2, 0.5, 0.8):
            gg = torch.Generator().manual_seed(1)
            e = torch.randn(n, DA, generator=gg).to(dev)
            zz = torch.randn(n, bk.dim_full - DA, generator=gg).to(dev)
            xs = (1 - s) * aq[None] + s * e
            sg = torch.full((n,), s, device=dev)
            base = float(((xs / (1 - s) - aq[None]) ** 2).mean().sqrt())
            cells = []
            for ci, c in enumerate(cs):
                cnt = [0] * len(cs)
                cnt[ci] = n
                fc = VLAField(bk, rep_rows(ps, cnt), zz)
                ah = fc.tweedie(xs, sg)
                tag = "*" if c[key] == tslot else ""
                cells.append(f"{tag}{c['name']!r} {float(((ah - aq[None]) ** 2).mean().sqrt()):.4f}")
            print(f"[smoke] tweedie {nm} row {r} sigma {s}: rmse(x/alpha, a) {base:.4f} -> rmse(a_hat, a) " + "; ".join(cells), flush=True)


def cmd_smoke(a):
    global VERBOSE
    VERBOSE = True
    torch.manual_seed(0)
    bk = backbone(a)
    mode = mode_of(a)
    paint = RUNGS[a.rung]["paint"]
    if mode == "roles":
        oc, tc_ = cands_roles(paint)
        print("[smoke] object candidates", [c["prompt"] for c in oc], "views", [c["view"] for c in oc], flush=True)
        print("[smoke] target candidates", [c["prompt"] for c in tc_], "views", [c["view"] for c in tc_], flush=True)
    else:
        print("[smoke] instruction candidates", [(c["name"], c["prompt"]) for c in cands_instr()], flush=True)
    print("[smoke] hold threshold", {k: lex()[k] for k in ("tw", "lift")}, "FD_DELTA", FD_DELTA, "MAXCH", MAXCH, flush=True)
    t0 = time.time()
    smoke_checks(bk, a)
    print(f"[smoke] checks {time.time() - t0:.1f}s", flush=True)
    tasks = [a.task] if a.task else ["K1", "K5"]
    secs = {}
    fm_rows = [0, 0]
    for hi in ([a.index] if a.index is not None else [0, 1]):
        hp = settings()[hi]
        print(f"[smoke] ===== setting {hi}: {json.dumps(hp)}", flush=True)
        for task in tasks:
            ex = demo_ex(task, 0)
            EVALS[0] = 0
            torch.cuda.reset_peak_memory_stats()
            rec = parse_demo(bk, ex, hp, a.rung, mode)
            secs.setdefault(hi, []).append(rec["secs"])
            print(f"[smoke] h{hi} {task} d0 truth obj {rec['true_obj']} tgt {rec['true_tgt']}; rows pre {rec['pre_rows']} held {rec['held_rows']}",
                  flush=True)
            for k in ("obj_role", "tgt_role", "instr_role"):
                if k in rec:
                    print_role(f"h{hi} {task} {k}", rec[k], hp)
                    role = rec[k]
                    if role is None:
                        continue
                    for x in role["rows"]:
                        print(f"[smoke] h{hi} {task} {k} row {x['row']}: pool {x['n_pool']}/{x['n_raw']} floor share {x['floor_share']:.2f} "
                              f"rmse_poe {x['rmse_poe']:.3f} nearest " + " ".join(f"{n_!r}:{v:.3f}" for n_, v in x['nearest_rmse'].items())
                              + " fm " + " ".join(f"{n_!r}:{v:.4f}" for n_, v in x["fm"].items()), flush=True)
                        if hi == (a.index if a.index is not None else 0):
                            truth = rec["true_obj"] if k == "obj_role" else rec["true_tgt"]
                            win = argbest(x["fm"], low=True)
                            cs = cands_roles(paint)[0 if k == "obj_role" else 1] if mode == "roles" else cands_instr()
                            ok = (next(c for c in cs if c["name"] == win)["obj" if k == "obj_role" else "tgt"] == truth) if k != "instr_role" \
                                else win == rec["true_task"]
                            fm_rows[0] += int(ok)
                            fm_rows[1] += 1
            print(f"[smoke] h{hi} {task} CHOSEN {json.dumps(rec['pred'])} ok {rec['ok']} | other rules "
                  + " ".join(f"{rule}:{rec['rules'][rule]['ok']}" for rule in Q0.RULES) + f" | flow-loss {json.dumps(rec['fm_pred'])} ok {rec['fm_ok']}",
                  flush=True)
            print(f"[smoke] h{hi} {task} secs {rec['secs']:.1f}, expert chain-evals {EVALS[0]}, peak GPU mem "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    print(f"[smoke] flow-loss row-level: true candidate lowest on {fm_rows[0]} of {fm_rows[1]} rows", flush=True)
    for hi, v in secs.items():
        print(f"[smoke] setting {hi}: {np.mean(v):.1f} s/demo", flush=True)
    if 0 in secs and 1 in secs:
        s0, s1 = float(np.mean(secs[0])), float(np.mean(secs[1]))
        hs = settings()
        # settings 2-9 have setting 1's cost except 16 starts (4x the chains; time scaled linearly as an upper bound)
        per = [s0, s1] + [s1 * (4 if h["n_starts"] == 16 else 1) for h in hs[2:]]
        n_s, n_r = len(TASKS) * len(SPLITS["search"]), len(TASKS) * len(SPLITS["read"])
        print(f"[smoke] projection: search all 10 settings x {n_s} demos ~{sum(per) * n_s / 3600:.2f} GPU-h (+ model load per job); "
              f"read picked + setting 0 x {n_r} demos ~{(s1 + s0) * n_r / 3600:.2f} GPU-h (taking setting 1's cost for the picked)",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("settings", "smoke", "search", "pick", "read", "demo"))
    ap.add_argument("--rung", choices=tuple(RUNGS), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="", help="config name override (default: the rung's pi05_g8v_*_ae)")
    ap.add_argument("--cands", choices=("roles", "instr"), default=None, help="r0 only: role phrases instead of instructions")
    ap.add_argument("--index", type=int)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--path", help="demo: one .npz in sc_rv's render layout")
    ap.add_argument("--tid", type=int, help="demo: n2 task id, for the truth (scoring only)")
    a = ap.parse_args()
    if a.cands is not None and a.rung != "r0" and a.cands != "roles":
        raise SystemExit("--cands instr is only defined for r0")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    dict(settings=cmd_settings, smoke=cmd_smoke, search=cmd_search, pick=cmd_pick, read=cmd_read, demo=cmd_demo)[a.what](a)

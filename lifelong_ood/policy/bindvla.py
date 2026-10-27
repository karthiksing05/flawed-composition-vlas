"""BindVLA (docs/PLAN_BINDVLA.md): SmolVLA with an explicit, HIERARCHICAL binding bottleneck.  Every theory condition is a flag.

  frozen     SmolVLM2-500M-Video-Instruct (vision tower, connector, token embeddings, the first 16 text layers), as SmolVLA builds it
  roles      per task step: a verb unit (verb + goal preposition: the operator) and two binding roles (object, goal).  Each role
             phrase is parsed into a PATH from coarse to fine, L levels (default 3: body / fixture -> object type -> attribute or
             part): "top drawer of the cabinet" -> cabinet / drawer / top; "black bowl on the left" -> bowl / black left.
             One query per (role, level), built ONLY from that unit's phrase tokens (frozen token embeddings -> a small
             transformer; cross_role=off: no attention across units) + learned role / level / step embeddings.
  binding    over the 16 x 16 agentview SigLIP grid with POSITION-FREE keys (pos_keys: a fixed nullspace projection removes the
             linear position subspace).  hier=on: NESTED masks, m_1 = softmax(q_1 . k / tau) over the scene, m_l = m_{l-1} x
             softmax(q_l . k / tau + geometry in the PARENT's frame) renormalised, so a child never leaves its parent; level 1 has no
             geometry term (no absolute layout).  hier=off (flat): one query from all the role's words, one mask, geometry in the
             IMAGE frame.  foveate=on: levels >= 2 re-encode a crop around the parent's mask with the frozen tower.
  bottleneck the action expert reads the scene ONLY through the role tokens (slot features + pose + relative geometry), one global
             context token and the state token; bottleneck=off gives it SmolVLA's full prefix (+ the role tokens if the binder is on)
  loss       SmolVLA's flow-matching velocity MSE (per element); the invariance penalty lives in the trainer

VLAFlowMatching underneath: lerobot's embed_suffix / vlm_with_expert / action_out_proj / denoise_step are used unchanged, only the
prefix differs.  No binding supervision anywhere.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRID = 16
SLOT_ROLES = ("verb", "object", "goal")
VERB = re.compile(r"^(open|close|turn on|turn off|pick up|push|put|place|stack)\s+")
PREPS = ("on top of", "to the right of", "to the left of", "to the front of", "to the back of", "under", "into", "onto", "in", "on", "at")
ATTR = {"left", "right", "front", "back", "middle", "top", "bottom", "upper", "lower", "center", "first", "second", "third",
        "black", "white", "red", "yellow", "green", "blue", "brown", "pink", "purple", "grey", "gray", "silver"}
FUNC = {"the", "a", "an", "and", "at", "on", "in", "of", "to", "into", "onto", "with", "it", "both", "side"}


@dataclasses.dataclass
class BVConfig:
    binder: bool = True
    parser: str = "rule"            # rule | noisy | learned | generic | token
    cross_role: bool = False
    hier: bool = True
    levels: int = 3
    foveate: bool = False
    pos_keys: bool = True
    bottleneck: bool = True
    wrist_slots: bool = True
    inv: float = 0.0                # V-REx weight (trainer)
    d: int = 384
    n_layers: int = 2
    n_steps: int = 2
    phrase_len: int = 12
    k_generic: int = 6
    noisy_p: float = 0.3
    sigma_min: float = 0.05
    chunk: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    tok_len: int = 48

    @property
    def units(self):
        return ("verb",) + tuple(f"obj{l}" for l in range(self.levels)) + tuple(f"goal{l}" for l in range(self.levels))

    @property
    def n_units(self):
        return self.n_steps * len(self.units)

    @property
    def n_slots(self):
        return 1 + self.k_generic if self.parser in ("generic", "token") else self.n_steps * len(SLOT_ROLES)


SETTINGS = {
    "bindvla": {},
    "vanilla_frozen": dict(binder=False, bottleneck=False),
    "flat": dict(hier=False),
    "levels2": dict(levels=2),
    "foveate": dict(foveate=True),
    "no_bottleneck": dict(bottleneck=False),
    "no_pos_keys": dict(pos_keys=False),
    "cross_role": dict(cross_role=True),
    "noisy_parser": dict(parser="noisy"),
    "learned_parser": dict(parser="learned"),
    "roboground": dict(cross_role=True, bottleneck=False, hier=False),
    "pelican": dict(parser="generic"),
    "otter": dict(parser="token", pos_keys=False),
}


def make_config(setting="bindvla", **kw):
    d = dict(SETTINGS[setting])
    d.update(kw)
    return BVConfig(**d)


# ── the role parse (S1 / B1's step and role rules + a coarse-to-fine path) ────────────────────────────────────────
def _labels():
    p = os.path.join(ROOT, "experiments")
    if p not in sys.path:
        sys.path.insert(0, p)
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    return SL, P


def words(p):
    return re.findall(r"[a-z]+", p.lower())


def strip_prep(g):
    """goal phrase -> (preposition, rest)."""
    ws = words(g)
    for p in PREPS:
        pw = p.split()
        if ws[: len(pw)] == pw:
            return p, " ".join(ws[len(pw):])
    return "", " ".join(ws)


def np_levels(p):
    """a noun phrase without "of" -> (type words, attribute words), function words dropped."""
    ws = [w for w in words(p) if w not in FUNC]
    return [w for w in ws if w not in ATTR], [w for w in ws if w in ATTR]


def role_path(p, L):
    """coarse -> fine path of L strings.  "A of B": B (whole) / A's type / A's attributes; else type / attributes.  L = 2 folds the
    finest two levels; L = 1 is the flat phrase."""
    ws = words(p)
    if "of" in ws:
        i = ws.index("of")
        A, B = " ".join(ws[:i]), " ".join(ws[i + 1:])
        bt, ba = np_levels(B)
        at, aa = np_levels(A)
        path = [" ".join(ba + bt), " ".join(at), " ".join(aa)]
    else:
        t, a = np_levels(p)
        path = [" ".join(t), " ".join(a), ""]
    if L == 1:
        return [" ".join(x for x in path if x)]
    if L == 2:
        return [path[0], " ".join(x for x in path[1:] if x)] if "of" in ws else [path[0], path[1]]
    return path[:L] + [""] * (L - 3)


def role_parse(text, cfg):
    """-> [{unit: phrase}] per step (at most cfg.n_steps)."""
    SL, P = _labels()
    L = cfg.levels if cfg.hier else 1
    out = []
    for c in P.steps_of(text.lower().strip().rstrip(".")):
        c = " ".join(c.strip().lower().split())
        o, g = SL.split_step(c)
        m = VERB.match(c)
        verb = m.group(1) if m else ""
        prep, grest = strip_prep(g) if g else ("", "")
        st = {"verb": " ".join(x for x in (verb, prep) if x)}
        for r, ph in (("obj", SL.norm(VERB.sub("", o.strip()))), ("goal", grest)):
            path = role_path(ph, L) if ph else [""] * L
            for l in range(cfg.levels):
                st[f"{r}{l}"] = path[l] if l < len(path) else ""
        out.append(st)
    return out[: cfg.n_steps]


def corrupt(units, rng, p, order):
    """parser=noisy: per unit with prob p, move 1-2 boundary words into the neighbouring unit of the step, or drop one word."""
    out = []
    for st in units:
        st = {k: st[k].split() for k in order}
        for i, k in enumerate(order):
            if not st[k] or rng.random() >= p:
                continue
            r = rng.random()
            if r < 0.33 and len(st[k]) > 1:
                st[k].pop(int(rng.integers(len(st[k]))))
            else:
                j = i + 1 if (r < 0.66 and i + 1 < len(order)) else i - 1
                if 0 <= j < len(order):
                    n = int(rng.integers(1, 3))
                    if j > i:
                        st[order[j]] = st[k][-n:] + st[order[j]]
                        st[k] = st[k][:-n]
                    else:
                        st[order[j]] = st[order[j]] + st[k][:n]
                        st[k] = st[k][n:]
        out.append({k: " ".join(v) for k, v in st.items()})
    return out


def encode_units(tokenizer, units, cfg):
    """-> tokens [n_units, L] (pad 0), mask [n_units, L] bool."""
    U = cfg.units
    T = np.zeros((cfg.n_units, cfg.phrase_len), np.int64)
    M = np.zeros((cfg.n_units, cfg.phrase_len), bool)
    for s, st in enumerate(units[: cfg.n_steps]):
        for u, k in enumerate(U):
            if not st.get(k):
                continue
            ids = tokenizer(" " + st[k], add_special_tokens=False)["input_ids"][: cfg.phrase_len]
            T[s * len(U) + u, : len(ids)] = ids
            M[s * len(U) + u, : len(ids)] = True
    return T, M


# ── the position-free projection ───────────────────────────────────────────────────────────────────────────────────
def fit_nullspace(X, pos, max_iter=300, r2_stop=0.05, ridge=1e-3, seed=0):
    """Iterative nullspace projection (B1.1): fit a ridge probe X -> (row, col) on half the FRAMES, remove its 2-d column space,
    repeat until a FRESH probe's held-out R^2 (other half) is < r2_stop for both.  X [n_frames, T, D], pos [T, 2].
    -> (P [D, D], mu [D], log)."""
    n, T, D = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    pos = np.asarray(pos, np.float64)
    mu = X[a].reshape(-1, D).astype(np.float64).mean(0)
    ya = pos.mean(0)                                  # every frame has the same token grid

    def gram(idx):
        C, Cy = np.zeros((D, D)), np.zeros((D, 2))
        for i in idx:                                 # frame by frame: no [n T, D] float64 copy
            Xi = X[i].astype(np.float64) - mu
            C += Xi.T @ Xi
            Cy += Xi.T @ (pos - ya)
        return C, Cy, len(idx) * T
    Ca, Cay, na = gram(a)
    Cb, Cby, nb = gram(b)
    Syy = len(b) * ((pos - ya) ** 2).sum(0)           # held-out targets centred on the TRAINING mean (= the grid mean)
    Sst = len(b) * ((pos - pos.mean(0)) ** 2).sum(0)
    lam = ridge * np.trace(Ca) / D
    P = np.eye(D)
    log = []
    for it in range(max_iter):
        W = np.linalg.solve(P @ Ca @ P + lam * np.eye(D), P @ Cay)        # probe on the projected features X P
        sse = Syy - 2 * (W * (P @ Cby)).sum(0) + np.einsum("dk,de,ek->k", W, P @ Cb @ P, W)
        rr = 1 - sse / Sst
        log.append(dict(it=it, r2_row=float(rr[0]), r2_col=float(rr[1]), removed=2 * it))
        if (rr < r2_stop).all():
            break
        Q, _ = np.linalg.qr(P @ W)                    # the probe's directions, in the original space (P is symmetric)
        P = P - Q @ Q.T
    return P.astype(np.float32), mu.astype(np.float32), log


# ── the model ──────────────────────────────────────────────────────────────────────────────────────────────────────
class RMSScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.log_s = nn.Parameter(torch.tensor(float(math.log(scale))))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.log_s.exp()


def grid_pos(device):
    r, c = torch.meshgrid(torch.arange(GRID, device=device), torch.arange(GRID, device=device), indexing="ij")
    return (torch.stack([r.reshape(-1), c.reshape(-1)], -1).float() + 0.5) / GRID      # token centres in (0, 1)


def moments(m, pos):
    """m [..., T], pos [..., T, 2] or [T, 2] -> mu [..., 2], sigma [..., 2]."""
    mu = (m[..., None] * pos).sum(-2)
    sig = torch.sqrt((m[..., None] * (pos - mu[..., None, :]) ** 2).sum(-2) + 1e-8)
    return mu, sig


class Binder(nn.Module):
    def __init__(self, cfg: BVConfig, d_tok=960, d_vis=768, emb_scale=1.0):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        nu = len(cfg.units)
        self.inp = nn.Linear(d_tok, d)
        self.pos_emb = nn.Parameter(torch.randn(64, d) * 0.02)
        self.unit_emb = nn.Parameter(torch.randn(nu, d) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(cfg.n_steps, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.q_norm = nn.LayerNorm(d)
        if cfg.parser == "learned":
            tl = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.tagger = nn.TransformerEncoder(tl, cfg.n_layers, enable_nested_tensor=False)
            self.tag_out = nn.Linear(d, cfg.n_units + 1)
        if cfg.parser == "generic":
            self.generic_q = nn.Parameter(torch.randn(cfg.k_generic, d) * 0.5)
        if cfg.parser in ("generic", "token"):
            self.instr_q = nn.Parameter(torch.randn(1, d) * 0.02)
        if cfg.parser == "token":
            self.pool_q = nn.Parameter(torch.randn(cfg.k_generic, d) * 0.5)
        self.v_norm = nn.LayerNorm(d_vis)
        self.wk, self.wv = nn.Linear(d_vis, d), nn.Linear(d_vis, d)
        self.wkw, self.wvw = nn.Linear(d_vis, d), nn.Linear(d_vis, d)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.log_tau_w = nn.Parameter(torch.zeros(()))
        self.geo = nn.Linear(d, 5)
        nn.init.zeros_(self.geo.weight), nn.init.zeros_(self.geo.bias)
        n_pose = 2 + 2 + 2 + 2 + 2 + (2 if cfg.parser == "token" else 0)   # mu, sigma, rank in parent, centroid rel. parent, wrist mu (+abs)
        self.slot_mlp = nn.Sequential(nn.Linear(4 * d + n_pose + len(SLOT_ROLES) + cfg.n_steps, 2 * d), nn.GELU(), nn.Linear(2 * d, d_tok))
        self.verb_mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_tok))
        self.out_norm = RMSScale(emb_scale)
        self.glob = nn.Linear(2 * d_tok, d_tok)
        self.glob_norm = RMSScale(emb_scale)
        self.register_buffer("P", torch.eye(d_vis), persistent=True)
        self.register_buffer("P_mu", torch.zeros(d_vis), persistent=True)

    # queries ------------------------------------------------------------------------------------------------------
    def queries(self, tok_emb, mask, full_emb=None, full_mask=None):
        """tok_emb [B, U, L, Dt], mask [B, U, L] -> q [B, U, d], valid [B, U]."""
        cfg = self.cfg
        B, U, L, _ = tok_emb.shape
        nu = len(cfg.units)
        uid = torch.arange(U, device=tok_emb.device)
        add = self.unit_emb[uid % nu] + self.step_emb[uid // nu]                                 # [U, d]
        if cfg.parser == "learned":
            h = self.inp(full_emb) + self.pos_emb[: full_emb.shape[1]]
            h = self.tagger(h, src_key_padding_mask=~full_mask)
            p = self.tag_out(h).softmax(-1)[..., :U] * full_mask[..., None]                   # [B, Lf, U]
            q = torch.einsum("blu,bld->bud", p, h) / (p.sum(1)[..., None] + 1e-3) + add
            valid = (p.sum(1) > 0.5) | (uid % nu == 0)[None]
            return self.q_norm(q), valid
        h = self.inp(tok_emb) + self.pos_emb[:L] + add[None, :, None]
        if cfg.cross_role:
            mm = mask.reshape(B, U * L).clone()
            mm[:, 0] |= ~mm.any(1)
            hs = self.enc(h.reshape(B, U * L, -1), src_key_padding_mask=~mm).reshape(B, U, L, -1)
        else:
            mm = mask.reshape(B * U, L).clone()
            mm[:, 0] |= ~mm.any(1)                                                            # empty units: avoid all-masked rows
            hs = self.enc(h.reshape(B * U, L, -1), src_key_padding_mask=~mm).reshape(B, U, L, -1)
        w = mask.float()
        q = (hs * w[..., None]).sum(2) / w.sum(2).clamp(min=1)[..., None]
        return self.q_norm(q), mask.any(-1)

    def keys(self, f, pos_free=True):
        f = f.float()
        if pos_free and self.cfg.pos_keys:
            f = (f - self.P_mu) @ self.P
        f = self.v_norm(f)
        return self.wk(f), self.wv(f)

    def phi(self, pos, mu, sig):
        """geometry features of every token in the frame (mu, sig): [.., T, 5]."""
        r = (pos - mu[..., None, :]) / sig[..., None, :]
        return torch.cat([r, r ** 2, r[..., :1] * r[..., 1:]], -1)

    # the nested hierarchy -----------------------------------------------------------------------------------------
    def hierarchy(self, qr, vr, k, v, pos, fov=None):
        """qr [B, n, L, d] per (role slot, level) queries, vr [B, n, L] valid.  -> per-level masks [B, n, L, T] on the scene grid,
        final mask, frames, slot features."""
        cfg = self.cfg
        B, n, L, d = qr.shape
        T = k.shape[1]
        tau = self.log_tau.exp()
        m_par = torch.full((B, n, T), 1.0 / T, device=k.device)
        mu_par = torch.full((B, n, 2), 0.5, device=k.device)
        sig_par = torch.full((B, n, 2), 0.29, device=k.device)            # the image frame (sd of uniform on [0, 1])
        masks, s1 = [], None
        par_of_final = (m_par, mu_par, sig_par)
        for l in range(L):
            q = qr[:, :, l]
            if l > 0 and fov is not None:
                a, kpos, vfeat = fov(q, mu_par, sig_par)                    # crop tokens: logits [B, n, Tc], pos [B, n, Tc, 2]
                pm = self._sample_grid(m_par, kpos)
            else:
                a = torch.einsum("bnd,btd->bnt", q, k) / math.sqrt(d) * tau
                kpos, vfeat, pm = pos[None, None].expand(B, n, T, 2), None, m_par
            geo_on = (l > 0) if cfg.hier else True                           # hier: no geometry at the scene level
            if geo_on:
                a = a + (self.geo(q)[:, :, None] * self.phi(kpos, mu_par, sig_par)).sum(-1)
            child = a.softmax(-1)
            m = pm * child if (cfg.hier and l > 0) else child
            m = m / m.sum(-1, keepdim=True).clamp(min=1e-12)
            if vfeat is not None:
                sl = torch.einsum("bnt,bntd->bnd", m, vfeat)
                m = self._scatter_grid(m, kpos, T)
            else:
                sl = torch.einsum("bnt,btd->bnd", m, v)
            ok = vr[:, :, l][..., None].float()
            m_new = ok * m + (1 - ok) * m_par                                # an empty level keeps its parent
            if l == 0:
                s1 = ok * sl + (1 - ok) * torch.einsum("bnt,btd->bnd", m_par, v)
                s_fin = s1
            else:
                s_fin = ok * sl + (1 - ok) * s_fin
            okb = ok.bool()[..., 0]
            par_of_final = tuple(torch.where(okb[..., None] if x.dim() == 3 else okb, x, y)
                                 for x, y in zip((m_par, mu_par, sig_par), par_of_final)) if l > 0 else par_of_final
            mu_new, sig_new = moments(m_new, pos)
            masks.append(m_new)
            m_par, mu_par, sig_par = m_new, mu_new, sig_new.clamp(min=cfg.sigma_min)
        final = masks[-1]
        mu_f, sig_f = moments(final, pos)
        pm, pmu, psig = par_of_final
        rank = (pm[..., None] * torch.sigmoid((mu_f[:, :, None] - pos[None, None]) / 0.05)).sum(-2)   # parent mass above / left of
        relp = (mu_f - pmu) / psig.clamp(min=cfg.sigma_min)
        return torch.stack(masks, 2), final, mu_f, sig_f, rank, relp, s1, s_fin

    @staticmethod
    def _sample_grid(m, kpos):
        """the scene-grid mask m [B, n, 256] at positions kpos [B, n, Tc, 2] (bilinear)."""
        B, n, T = m.shape
        img = m.reshape(B * n, 1, GRID, GRID)
        g = kpos.reshape(B * n, 1, -1, 2)[..., [1, 0]] * 2 - 1              # (x = col, y = row) in [-1, 1]
        return F.grid_sample(img, g, align_corners=False).reshape(B, n, -1)

    @staticmethod
    def _scatter_grid(m, kpos, T):
        """crop-token masses -> the scene grid (nearest token)."""
        B, n, Tc = m.shape
        rc = (kpos * GRID).long().clamp(0, GRID - 1)
        idx = rc[..., 0] * GRID + rc[..., 1]
        out = torch.zeros(B, n, T, device=m.device, dtype=m.dtype)
        return out.scatter_add(-1, idx, m)

    def forward(self, feats, wrist, conn, tok_emb, tmask, full_emb=None, full_mask=None, fov=None):
        """feats / wrist [B, 256, 768] SigLIP grids; conn [B, 2, 960] mean connector tokens (scaled as SmolVLA scales them);
        tok_emb [B, U, L, 960] frozen unit embeddings, tmask [B, U, L]; full_emb / full_mask the whole instruction's embeddings.
        -> slot tokens [B, S, 960], slot valid [B, S], global token [B, 1, 960], diagnostics."""
        cfg = self.cfg
        B = feats.shape[0]
        dev = feats.device
        pos = grid_pos(dev)
        k, v = self.keys(feats)
        kw, vw = self.keys(wrist, pos_free=False)
        g = self.glob_norm(self.glob(conn.reshape(B, -1).float()))[:, None]
        if cfg.parser in ("generic", "token"):
            return self._generic(full_emb, full_mask, k, v, kw, vw, pos, g, B)
        q, valid = self.queries(tok_emb, tmask, full_emb, full_mask)                            # [B, U, d]
        nu, S, L = len(cfg.units), cfg.n_steps, cfg.levels
        q = q.reshape(B, S, nu, -1)
        valid = valid.reshape(B, S, nu)
        # role slots: (step, role) -> n = 2S slots of L levels
        qr = torch.stack([q[:, :, 1:1 + L], q[:, :, 1 + L:1 + 2 * L]], 2).reshape(B, 2 * S, L, -1)
        vr = torch.stack([valid[:, :, 1:1 + L], valid[:, :, 1 + L:1 + 2 * L]], 2).reshape(B, 2 * S, L)
        if not cfg.hier:
            vr = vr.clone()
            vr[:, :, 1:] = False
        masks, final, mu_f, sig_f, rank, relp, s1, s_fin = self.hierarchy(qr, vr, k, v, pos, fov=fov)
        qmean = (qr * vr[..., None]).sum(2) / vr.sum(2, keepdim=True).clamp(min=1)
        if cfg.wrist_slots:
            mw = (torch.einsum("bnd,btd->bnt", qmean, kw) / math.sqrt(qmean.shape[-1]) * self.log_tau_w.exp()).softmax(-1)
            sw = torch.einsum("bnt,btd->bnd", mw, vw)
            muw = torch.einsum("bnt,tc->bnc", mw, pos)
        else:
            sw, muw = torch.zeros_like(s1), torch.zeros_like(mu_f)
        role_oh = torch.zeros(B, 2 * S, len(SLOT_ROLES), device=dev)
        role_oh[:, 0::2, 1] = 1
        role_oh[:, 1::2, 2] = 1
        step_oh = torch.zeros(B, 2 * S, S, device=dev)
        for s in range(S):
            step_oh[:, 2 * s:2 * s + 2, s] = 1
        z = torch.cat([s_fin, s1, sw, qmean, mu_f, sig_f, rank, relp, muw, role_oh, step_oh], -1)
        bind_tok = self.out_norm(self.slot_mlp(z))                                              # [B, 2S, 960]
        verb_tok = self.out_norm(self.verb_mlp(q[:, :, 0]))                                      # [B, S, 960]
        toks = torch.stack([verb_tok, bind_tok[:, 0::2], bind_tok[:, 1::2]], 2).reshape(B, 3 * S, -1)
        rv = vr.any(-1)
        slot_valid = torch.stack([valid[:, :, 0], rv[:, 0::2], rv[:, 1::2]], 2).reshape(B, 3 * S)
        diag = dict(levels=masks.reshape(B, S, 2, L, -1), mask=final.reshape(B, S, 2, -1), level_valid=vr.reshape(B, S, 2, L),
                    valid=rv.reshape(B, S, 2))
        return toks, slot_valid, g, diag

    def _generic(self, full_emb, full_mask, k, v, kw, vw, pos, g, B):
        """parser generic (Pelican-style K learned slots) / token (OTTER-style per-token selection pooled to K)."""
        cfg = self.cfg
        h = self.inp(full_emb) + self.pos_emb[: full_emb.shape[1]]
        h = self.enc(h, src_key_padding_mask=~full_mask)
        instr = self.q_norm((h * full_mask[..., None]).sum(1) / full_mask.sum(1, keepdim=True).clamp(min=1) + self.instr_q[0])
        if cfg.parser == "generic":
            q = self.q_norm(self.generic_q)[None].expand(B, -1, -1)
        else:
            q = self.q_norm(self.inp(full_emb))                                                  # context-free per-token queries
        m = (torch.einsum("bnd,btd->bnt", F.normalize(q, dim=-1), F.normalize(k, dim=-1)) * 10 * self.log_tau.exp()).softmax(-1)
        s = torch.einsum("bnt,btd->bnd", m, v)
        mw = (torch.einsum("bnd,btd->bnt", q, kw) / math.sqrt(q.shape[-1]) * self.log_tau_w.exp()).softmax(-1)
        sw = torch.einsum("bnt,btd->bnd", mw, vw)
        mu, sig = moments(m, pos)
        muw = torch.einsum("bnt,tc->bnc", mw, pos)
        n = q.shape[1]
        zeros = torch.zeros(B, n, 4, device=q.device)
        extra = [mu] if cfg.parser == "token" else []                                         # OTTER: absolute position in the values
        oh = torch.zeros(B, n, len(SLOT_ROLES) + cfg.n_steps, device=q.device)
        z = torch.cat([s, s, sw, q, mu, sig, zeros, muw] + extra + [oh], -1)
        tok = self.out_norm(self.slot_mlp(z))
        if cfg.parser == "token":
            att = torch.einsum("kd,bnd->bkn", self.pool_q, self.q_norm(q)).masked_fill(~full_mask[:, None], -1e9).softmax(-1)
            tok = torch.einsum("bkn,bnd->bkd", att, tok)
        valid = torch.ones(B, cfg.k_generic, dtype=torch.bool, device=q.device)
        itok = self.out_norm(self.verb_mlp(instr))[:, None]
        return torch.cat([itok, tok], 1), torch.cat([valid[:, :1], valid], 1), g, dict(mask=m)


class BindVLA(nn.Module):
    """VLAFlowMatching (frozen SmolVLM2, fresh expert) + the Binder."""

    def __init__(self, cfg: BVConfig, device="cuda"):
        super().__init__()
        os.environ.setdefault("HF_HOME", "/coc/flash10/ksingara3/robot/data/hf/home")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
        from lerobot.configs.types import FeatureType, PolicyFeature
        self.cfg = cfg
        scfg = SmolVLAConfig(vlm_model_name=VLM_ID, load_vlm_weights=True, train_expert_only=True, freeze_vision_encoder=True,
                             num_vlm_layers=16, expert_width_multiplier=0.75, attention_mode="cross_attn", self_attn_every_n_layers=2,
                             chunk_size=cfg.chunk, n_action_steps=cfg.chunk, max_state_dim=cfg.max_state_dim,
                             max_action_dim=cfg.max_action_dim, tokenizer_max_length=cfg.tok_len, add_image_special_tokens=False,
                             prefix_length=0, pad_language_to="max_length", num_steps=10, device=device,
                             input_features={"observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
                                             "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
                                             "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
                             output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        self.scfg = scfg
        self.flow = VLAFlowMatching(scfg)
        for p in self.flow.vlm_with_expert.vlm.parameters():
            p.requires_grad_(False)
        self.tokenizer = self.flow.vlm_with_expert.processor.tokenizer
        E = self.flow.vlm_with_expert.get_vlm_model().text_model.get_input_embeddings().weight
        self.d_tok = int(E.shape[1])
        emb_scale = float(E.float().pow(2).mean().sqrt()) * math.sqrt(self.d_tok)          # the rms of SmolVLA's scaled language tokens
        self.binder = Binder(cfg, d_tok=self.d_tok, emb_scale=emb_scale) if cfg.binder else None

    def train(self, mode=True):
        super().train(mode)
        self.flow.vlm_with_expert.vlm.eval()                                                     # frozen: always eval
        return self

    # ── parameters ───────────────────────────────────────────────────────────────────────────────────────────────────
    def trainable_state(self):
        return {n: p.detach() for n, p in self.named_parameters() if p.requires_grad} | \
               ({"binder.P": self.binder.P, "binder.P_mu": self.binder.P_mu} if self.binder is not None else {})

    def n_trainable(self):
        out = dict(expert=sum(p.numel() for p in self.flow.vlm_with_expert.lm_expert.parameters() if p.requires_grad),
                   heads=sum(p.numel() for n, p in self.flow.named_parameters() if p.requires_grad and not n.startswith("vlm_with_expert")),
                   binder=sum(p.numel() for p in self.binder.parameters() if p.requires_grad) if self.binder is not None else 0,
                   frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad))
        out["total_trainable"] = out["expert"] + out["heads"] + out["binder"]
        return out

    # ── frozen encoders ──────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def vision(self, img):
        """img [B, 3, h, w] in [0, 1] -> (SigLIP 16 x 16 grid [B, 256, 768], connector tokens x sqrt(960) [B, 64, 960])."""
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
        vm = self.flow.vlm_with_expert.get_vlm_model()
        x = resize_with_pad(img.float(), 512, 512, pad_value=0) * 2.0 - 1.0
        h = vm.vision_model(pixel_values=x.to(vm.vision_model.dtype)).last_hidden_state                 # [B, 1024, 768]
        conn = vm.connector(h).float() * math.sqrt(self.d_tok)
        n = int(math.isqrt(h.shape[1]))
        gr = h.reshape(h.shape[0], n, n, -1).permute(0, 3, 1, 2)
        gr = F.avg_pool2d(gr.float(), n // GRID).permute(0, 2, 3, 1).reshape(h.shape[0], GRID * GRID, -1)
        return gr, conn

    @torch.no_grad()
    def embed_tokens(self, ids):
        return self.flow.vlm_with_expert.embed_language_tokens(ids).float()

    def foveator(self, img):
        """foveate=on: for each (slot, level >= 2) query, crop a square around the parent's mask (centre mu, half-size 2.5 sigma,
        clamped to [0.08, 0.5]), re-encode it with the frozen tower and return the crop's logits, token positions in image
        coordinates and values."""
        b = self.binder

        def fov(q, mu, sig):
            B, n, d = q.shape
            half = (2.5 * sig.max(-1).values).clamp(0.08, 0.5).detach()                         # [B, n]
            c = mu.detach().clamp(0, 1)
            lin = (torch.arange(224, device=img.device).float() + 0.5) / 224 * 2 - 1           # crop pixel grid in [-1, 1]
            yy, xx = torch.meshgrid(lin, lin, indexing="ij")
            gy = (c[..., 0, None, None] * 2 - 1) + yy[None, None] * half[..., None, None]
            gx = (c[..., 1, None, None] * 2 - 1) + xx[None, None] * half[..., None, None]
            grid = torch.stack([gx, gy], -1).reshape(B * n, 224, 224, 2)
            src = img.float()[:, None].expand(B, n, *img.shape[1:]).reshape(B * n, *img.shape[1:])
            crop = F.grid_sample(src, grid, align_corners=False, padding_mode="zeros")
            gr, _ = self.vision(crop)                                                           # [B n, 256, 768]
            kc, vc = b.keys(gr)
            kc, vc = kc.reshape(B, n, 256, -1), vc.reshape(B, n, 256, -1)
            a = torch.einsum("bnd,bntd->bnt", q, kc) / math.sqrt(d) * b.log_tau.exp()
            gp = grid_pos(img.device)
            kpos = (c[:, :, None] + (gp[None, None] * 2 - 1) * half[..., None, None]).clamp(0, 1 - 1e-4)
            return a, kpos, vc
        return fov

    # ── the prefix ───────────────────────────────────────────────────────────────────────────────────────────────────
    def build_prefix(self, batch):
        """batch: img, wrist [B, 3, 224, 224] float [0, 1]; state [B, 32] (normalised, padded); lang_tokens / lang_masks [B, 48];
        unit_tokens / unit_masks [B, U, L].  -> (embs, pad_masks, att_masks, diag)."""
        cfg = self.cfg
        img, wrist = batch["img"], batch["wrist"]
        B = img.shape[0]
        gA, cA = self.vision(img)
        gW, cW = self.vision(wrist)
        lang = self.embed_tokens(batch["lang_tokens"]) * math.sqrt(self.d_tok)
        st = self.flow.state_proj(batch["state"].float())[:, None]
        diag = {}
        parts, pads = [], []
        one = lambda n: torch.ones(B, n, dtype=torch.bool, device=img.device)                    # noqa: E731
        if not cfg.bottleneck:
            parts += [cA, cW, lang]
            pads += [one(cA.shape[1] + cW.shape[1]), batch["lang_masks"].bool()]
        if self.binder is not None:
            temb = self.embed_tokens(batch["unit_tokens"])
            fov = self.foveator(img) if (cfg.foveate and cfg.hier) else None
            toks, valid, g, diag = self.binder(gA, gW, torch.stack([cA.mean(1), cW.mean(1)], 1), temb, batch["unit_masks"].bool(),
                                               full_emb=self.embed_tokens(batch["lang_tokens"]), full_mask=batch["lang_masks"].bool(),
                                               fov=fov)
            parts += [toks]
            pads += [valid]
            if cfg.bottleneck:
                parts += [g]
                pads += [one(1)]
        parts += [st]
        pads += [one(1)]
        dt = self.flow.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
        embs = torch.cat([p.to(dt) for p in parts], 1)
        pad = torch.cat(pads, 1)
        att = torch.zeros(B, embs.shape[1], dtype=torch.bool, device=img.device)
        att[:, -1] = True                                                                          # the state token (SmolVLA)
        return embs, pad, att, diag

    def forward(self, batch, noise, time):
        """-> per-element losses [B, H, A] (SmolVLA's VLAFlowMatching.forward with our prefix), diag."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        f = self.flow
        actions = batch["actions"]
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions
        pe, pp, pa, diag = self.build_prefix(batch)
        se, sp, sa = f.embed_suffix(x_t, time)
        pad = torch.cat([pp, sp], 1)
        att = torch.cat([pa.to(sa.dtype), sa], 1)
        att2 = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        (_, out), _ = f.vlm_with_expert.forward(attention_mask=att2, position_ids=pos, past_key_values=None,
                                                inputs_embeds=[pe, se], use_cache=False, fill_kv_cache=False)
        v_t = f.action_out_proj(out[:, -self.cfg.chunk:].float())
        return F.mse_loss(u_t, v_t, reduction="none"), diag

    @torch.no_grad()
    def prefix_cache(self, batch):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        pe, pp, pa, diag = self.build_prefix(batch)
        att2 = make_att_2d_masks(pp, pa)
        pos = torch.cumsum(pp, dim=1) - 1
        _, kv = self.flow.vlm_with_expert.forward(attention_mask=att2, position_ids=pos, past_key_values=None,
                                                  inputs_embeds=[pe, None], use_cache=True, fill_kv_cache=True)
        return pp, kv, diag

    @torch.no_grad()
    def sample(self, batch, noise, num_steps=10):
        pp, kv, _ = self.prefix_cache(batch)
        x = noise
        dt = -1.0 / num_steps
        for s in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 + s * dt, device=x.device)
            x = x + dt * self.flow.denoise_step(prefix_pad_masks=pp, past_key_values=kv, x_t=x, timestep=t)
        return x


# ── conditioning shared by the trainer and the server ─────────────────────────────────────────────────────────────
class Conditioner:
    """text -> (lang tokens, lang mask, unit tokens, unit masks), cached per text (noisy draws are not cached)."""

    def __init__(self, tokenizer, cfg: BVConfig):
        self.tok, self.cfg = tokenizer, cfg
        self.cache = {}

    def lang(self, text):
        t = text.strip()
        if not t.endswith("\n"):
            t = t + "\n"                                  # SmolVLA's pre-processor appends a newline
        e = self.tok(t, max_length=self.cfg.tok_len, padding="max_length", truncation=True, return_tensors="np")
        return e["input_ids"][0].astype(np.int64), e["attention_mask"][0].astype(bool)

    def parse(self, text, rng=None):
        u = role_parse(text, self.cfg)
        if self.cfg.parser == "noisy" and rng is not None:
            u = corrupt(u, rng, self.cfg.noisy_p, self.cfg.units)
        return u

    def __call__(self, text, rng=None):
        noisy = self.cfg.parser == "noisy" and rng is not None
        if not noisy and text in self.cache:
            return self.cache[text]
        out = self.lang(text) + encode_units(self.tok, self.parse(text, rng), self.cfg)
        if not noisy:
            self.cache[text] = out
        return out

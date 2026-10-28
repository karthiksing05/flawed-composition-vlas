"""BindVLA-C (docs/PLAN_BINDVLA.md section 12): the opt-in training pieces, used by bv_train.py only when their flags are set.

  held-out layouts  --hold-layouts N: N training ENVIRONMENTS (layouts) with contact-event referents, chosen by a fixed hash, are
                    removed from training; frame 0 of their episodes scores the leave-layout-out slot hit (the selection criterion)
  swap contrast     --swap W: on a pre-grasp training frame of task A (scene S), step 0's object phrase is replaced by an entity
                    named by ANOTHER training task of the same environment; the slot must move to that entity's contact-event
                    referent (its modal grasp / release token over the training demos of the environment).  Training-cell demos
                    only; the entity map is logged with its source task ids.
  prototypes        --proto W: each level-1 phrase keeps one EMA prototype of its level-1 slot feature, shared across layouts; the
                    slot is pulled to it (cosine)
  adversary         --adv W: an MLP probe predicts each agentview token's (row, col) from the binder's keys through gradient reversal
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]


# ── held-out layouts ─────────────────────────────────────────────────────────────────────────────────────────────────
def heldout_layouts(store, n, max_tasks=3):
    """the n environments (of those whose training episodes carry >= 20 referent episodes and that hold at most max_tasks
    training tasks, so no whole suite such as libero_goal leaves training) with the smallest sha256 key."""
    has = (store.ref >= 0).any((1, 2))
    cnt, tids = Counter(), {}
    for e in np.unique(store.ep[has]):
        ep = store.eps[int(e)]
        if not ep["held_out"]:
            cnt[ep["env"]] += 1
    for ep in store.eps.values():
        tids.setdefault(ep["env"], set()).add(ep["tid"])
    cand = sorted((v for v, c in cnt.items() if c >= 20 and len(tids[v]) <= max_tasks),
                  key=lambda v: hashlib.sha256(("bindvla-layout" + v).encode()).hexdigest())
    return cand[:n]


def layout_frames(store, envs, max_eps=200):
    """frame 0 of every episode of the held-out environments that carries referents -> [(frame, [(step, role, token)])]."""
    out = []
    first = np.flatnonzero(np.r_[True, np.diff(store.ep) != 0])
    for i in first:
        e = store.eps[int(store.ep[i])]
        if e["env"] not in envs:
            continue
        rows = [(s, ("object", "goal")[r], int(store.ref[i, s, r])) for s in range(store.ref.shape[1]) for r in range(2) if store.ref[i, s, r] >= 0]
        if rows:
            out.append((int(i), rows))
    rng = np.random.default_rng(0)
    if len(out) > max_eps:
        out = [out[j] for j in sorted(rng.choice(len(out), max_eps, replace=False))]
    return out


# ── the swap contrast's entity map ───────────────────────────────────────────────────────────────────────────────────
def role_phrases(text):
    """-> [(object phrase, goal phrase without its preposition)] per step, the strings role_parse parses."""
    from lifelong_ood.policy import bindvla as BV
    SL, P = BV._labels()
    out = []
    for c in P.steps_of(text.lower().strip().rstrip(".")):
        c = " ".join(c.strip().lower().split())
        o, g = SL.split_step(c)
        out.append((SL.norm(BV.VERB.sub("", o.strip())), SL.norm(BV.strip_prep(g)[1]) if g else ""))
    return out


def entity_map(store):
    """env -> {phrase: dict(token=modal referent token, n, tids)} from the TRAINING episodes' frame-0 referents."""
    first = np.flatnonzero(np.r_[True, np.diff(store.ep) != 0])
    acc = {}
    train = set(np.unique(store.ep[store.train_idx]).tolist())
    for i in first:
        epi = int(store.ep[i])
        if epi not in train:
            continue
        e = store.eps[epi]
        for s, (o, g) in enumerate(role_phrases(e["text"])[: store.ref.shape[1]]):
            for r, ph in ((0, o), (1, g)):
                t = int(store.ref[i, s, r])
                if ph and t >= 0:
                    d = acc.setdefault(e["env"], {}).setdefault(ph, dict(tokens=Counter(), tids=set()))
                    d["tokens"][t] += 1
                    d["tids"].add(e["tid"])
    out = {}
    for env, m in acc.items():
        out[env] = {ph: dict(token=int(d["tokens"].most_common(1)[0][0]), n=int(sum(d["tokens"].values())), tids=sorted(d["tids"]),
                             purity=float(d["tokens"].most_common(1)[0][1] / sum(d["tokens"].values()))) for ph, d in m.items()}
    return out


def swap_items(store, emap, idx, cond, rng):
    """per sampled frame: (unit tokens, unit masks) with step 0's object swapped to another entity of the environment, and the
    entity's token (-1 when the frame is not pre-grasp or no other entity exists)."""
    U, M, T = [], [], []
    for i in idx:
        e = store.eps[int(store.ep[i])]
        own = role_phrases(e["text"])[0][0]
        tok_own = int(store.ref[i, 0, 0])
        cands = []
        if tok_own >= 0:
            tr, tc = divmod(tok_own, 16)
            for ph, d in emap.get(e["env"], {}).items():
                r, c = divmod(d["token"], 16)
                if ph != own and max(abs(r - tr), abs(c - tc)) >= 2:
                    cands.append((ph, d["token"]))
        if cands:
            ph, t = cands[int(rng.integers(len(cands)))]
            u, m = cond.with_object(e["text"], ph)
        else:
            _, _, u, m = cond(e["text"])
            t = -1
        U.append(u), M.append(m), T.append(t)
    return np.stack(U), np.stack(M), np.asarray(T, np.int64)


# ── losses ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


class Extras:
    def __init__(self, model, a, dev):
        self.a, self.dev = a, dev
        d = model.cfg.d
        self.probe = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 2)).to(dev) if a.adv > 0 else None
        self.proto = torch.zeros(4096, d, device=dev)
        self.pcount = torch.zeros(4096, device=dev)

    def params(self):
        return list(self.probe.parameters()) if self.probe is not None else []

    def __call__(self, model, b, dgm, win):
        a, out, logs = self.a, {}, {}
        if a.swap > 0 and "gA" in dgm:
            ok = b["swap_tok"] >= 0
            if ok.any():
                temb = model.embed_tokens(b["swap_units"])
                fm = model.binder.final_masks(dgm["gA"], temb, b["swap_masks"].bool())[:, 0, 0]       # [B, T]
                mass = (fm.float() * win[b["swap_tok"].clamp(min=0)]).sum(-1)
                out["swap"] = -(torch.log(mass + 1e-4) * ok).sum() / ok.sum()
                logs["swap_hit"] = float(((fm.argmax(-1)[:, None] == torch.arange(256, device=fm.device)[None]) * win[b["swap_tok"].clamp(min=0)]).sum(-1)[ok].mean())
        if a.proto > 0 and "s1" in dgm:
            ids = b["l1"]                                                                                  # [B, S, 2]
            s1 = F.normalize(dgm["s1"].float(), dim=-1)
            ok = ids >= 0
            if ok.any():
                sel, idv = s1[ok], ids[ok]
                pr = F.normalize(self.proto[idv], dim=-1)
                have = self.pcount[idv] > 0
                if have.any():
                    out["proto"] = (1 - (sel[have] * pr[have]).sum(-1)).mean()
                with torch.no_grad():
                    for j in torch.unique(idv):
                        m = sel[idv == j].mean(0)
                        self.proto[j] = m if self.pcount[j] == 0 else 0.99 * self.proto[j] + 0.01 * m
                        self.pcount[j] += 1
        if a.adv > 0 and "keys" in dgm:
            k = _GRL.apply(dgm["keys"].float())                                                           # [B, T, d]
            from lifelong_ood.policy import bindvla as BV
            pos = BV.grid_pos(k.device)[None].expand(k.shape[0], -1, -1)
            pred = self.probe(k)
            out["adv"] = F.mse_loss(pred, pos)
            with torch.no_grad():
                logs["adv_r2"] = float(1 - ((pred - pos) ** 2).sum() / ((pos - pos.mean((0, 1))) ** 2).sum())
        return out, logs


@torch.no_grad()
def lol_diag(model, cond, cfg, store, frames, dev, collate, to_dev, bs=16):
    """leave-layout-out final-level slot hit (the BindVLA-C selection criterion)."""
    if model.binder is None or not frames:
        return {}
    rec = []
    model.eval()
    for k in range(0, len(frames), bs):
        chunk = frames[k:k + bs]
        b = to_dev(collate([store.item(fi) for fi, _ in chunk], cond, cfg), dev)
        _, _, _, dg = model.build_prefix(b)
        M = dg["mask"].float().cpu().numpy()
        VV = dg["level_valid"].cpu().numpy()
        for j, (_, rows) in enumerate(chunk):
            for s, role, tok in rows:
                if s >= cfg.n_steps:
                    continue
                r = 0 if role == "object" else 1
                if not VV[j, s, r].any():
                    continue
                tr, tc = divmod(tok, 16)
                pr, pc = divmod(int(M[j, s, r].argmax()), 16)
                rec.append(dict(role=role, depth=int(VV[j, s, r].sum()), hit=float(max(abs(pr - tr), abs(pc - tc)) <= 1)))
    model.train()
    out = {}
    for key, f in [("object", lambda x: x["role"] == "object"), ("goal", lambda x: x["role"] == "goal")] + \
                  [(f"depth{d}", (lambda d: lambda x: x["depth"] == d)(d)) for d in (1, 2, 3)]:
        rs = [x["hit"] for x in rec if f(x)]
        if rs:
            out[key] = dict(n=len(rs), hit=float(np.mean(rs)))
    if "object" in out and "goal" in out:
        out["lol_hit"] = 0.5 * (out["object"]["hit"] + out["goal"]["hit"])
    return out


def key_probe(model, store, dev, n_frames=200, iters=3000, seed=0):
    """a FRESH nonlinear (2-hidden-layer MLP) probe of agentview token (row, col) from the binder's keys, fitted on half of
    n_frames training frames and scored (R^2 per axis) on the other half: the adversary's before / after read."""
    from lifelong_ood.policy import bindvla as BV
    if model.binder is None:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(store.train_idx, n_frames, replace=False)
    Z = []
    model.eval()
    with torch.no_grad():
        for k in range(0, len(idx), 32):
            img = torch.from_numpy(np.stack([store._img("a", int(i)) for i in idx[k:k + 32]])).permute(0, 3, 1, 2).float().div(255).to(dev)
            g, _ = model.vision(img)
            kk, _ = model.binder.keys(g)
            Z.append(kk.float())
    model.train()
    Z = torch.cat(Z)
    pos = BV.grid_pos(dev).float()
    h = n_frames // 2
    Xt, Xe = Z[:h].reshape(-1, Z.shape[-1]), Z[h:].reshape(-1, Z.shape[-1])
    Yt, Ye = pos.repeat(h, 1), pos.repeat(n_frames - h, 1)
    m_, s_ = Xt.mean(0), Xt.std(0) + 1e-6
    with torch.enable_grad():
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(Z.shape[-1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(),
                                  torch.nn.Linear(256, 2)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for _ in range(iters):
            j = torch.randint(0, len(Xt), (4096,), device=dev)
            loss = ((net((Xt[j] - m_) / s_) - Yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pr = net((Xe - m_) / s_)
        r2 = 1 - ((Ye - pr) ** 2).sum(0) / ((Ye - Ye.mean(0)) ** 2).sum(0)
    return dict(mlp_r2_row=float(r2[0]), mlp_r2_col=float(r2[1]), frames=n_frames)

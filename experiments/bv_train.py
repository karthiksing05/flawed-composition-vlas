"""PLAN_BINDVLA section 5: train BindVLA (or any flag setting of it) on the LIBERO-90-C TRAINING cells.  SmolVLA venv, one GPU.

    --posfit                  fit the position-free projection on TRAINING frames (agentview SigLIP grid), report linear and MLP
                              probes before / after -> CACHE/posfree.npz, results/bindvla/posfree.json
    --setting S --exp NAME --steps N [--inv L] [--batch 32] [--envs 4]
                              train; resumable (newest complete checkpoint in results/bindvla/<exp>/ckpt is restored)
    --timing N                N steps then seconds / step and peak memory (no checkpoint)

Data: experiments/bv_data.py's cache (training episodes minus the held-out ones).  Env-balanced batches (envs x frames per env);
the frames of every environment share the same stratified flow times and noise draws; V-REx on per-t-bin standardised losses.
Every 100 steps: loss, per-env spread, penalty, gradient ratio of the penalty to BC on the binder; every --eval-every steps:
validation flow loss on held-out episodes (fixed t / noise) and the slot-mask referent diagnostic on their frame 0.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
OUT = os.path.join(ROOT, "results", "bindvla")
NT = 8                                       # t strata per environment
BASE = "/coc/flash10/ksingara3/robot/data/smolvla_base"   # the public lerobot/smolvla_base (anonymous download, PLAN_BINDVLA 11)
_rc = np.divmod(np.arange(256), 16)
WIN = torch.from_numpy((np.maximum(np.abs(_rc[0][:, None] - _rc[0][None]), np.abs(_rc[1][:, None] - _rc[1][None])) <= 1).astype(np.float32))


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Store:
    def __init__(self):
        self.meta = json.load(open(os.path.join(CACHE, "meta.json"), encoding="utf-8"))
        z = np.load(os.path.join(CACHE, "frames.npz"))
        self.state, self.actions, self.ep, self.fidx = z["state"], z["actions"], z["ep"], z["fidx"]
        self.aoff, self.woff = z["agent_off"], z["wrist_off"]
        self._fa = self._fw = None
        st = self.meta["stats"]
        self.s_mu, self.s_sd = np.asarray(st["state"]["mean"], np.float32), np.asarray(st["state"]["std"], np.float32)
        self.a_mu, self.a_sd = np.asarray(st["actions"]["mean"], np.float32), np.asarray(st["actions"]["std"], np.float32)
        eps = self.meta["episodes"]
        self.eps = {e["ep"]: e for e in eps}
        n = len(self.ep)
        self.start = np.zeros(n, np.int64)
        self.end = np.zeros(n, np.int64)
        bounds = np.flatnonzero(np.diff(self.ep)) + 1
        s = np.concatenate([[0], bounds])
        e = np.concatenate([bounds, [n]])
        for a, b in zip(s, e):
            self.start[a:b], self.end[a:b] = a, b
        held = np.array([self.eps[int(x)]["held_out"] for x in self.ep])
        self.train_idx = np.flatnonzero(~held)
        self.held_idx = np.flatnonzero(held)
        env_of = np.array([self.eps[int(x)]["env"] for x in self.ep])
        self.envs = sorted(set(env_of[self.train_idx]))
        self.by_env = {v: self.train_idx[env_of[self.train_idx] == v] for v in self.envs}
        rf = os.path.join(CACHE, "refs.npz")                  # contact-event referents (bv_refs.py); -1 = none
        self.ref = np.load(rf)["ref"] if os.path.exists(rf) else np.full((n, 2, 2), -1, np.int16)
        self.ho_envs = []

    def hold(self, envs):
        """BindVLA-C leave-layout-out: drop these environments from training (their frames score the selection criterion)."""
        self.ho_envs = list(envs)
        keep = ~np.isin(np.array([self.eps[int(x)]["env"] for x in self.ep[self.train_idx]]), self.ho_envs)
        self.train_idx = self.train_idx[keep]
        self.envs = [v for v in self.envs if v not in self.ho_envs]
        self.by_env = {v: self.by_env[v] for v in self.envs}

    def _img(self, which, i):
        if which == "a":
            if self._fa is None:
                self._fa = np.memmap(os.path.join(CACHE, "agent.bin"), dtype=np.uint8, mode="r")
            f, off = self._fa, self.aoff
        else:
            if self._fw is None:
                self._fw = np.memmap(os.path.join(CACHE, "wrist.bin"), dtype=np.uint8, mode="r")
            f, off = self._fw, self.woff
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(f[off[i]:off[i + 1]].tobytes())).convert("RGB"))

    def item(self, i, H=50):
        j = np.arange(i, i + H)
        pad = j >= self.end[i]
        j = np.minimum(j, self.end[i] - 1)
        a = (self.actions[j] - self.a_mu) / (self.a_sd + 1e-8)
        s = (self.state[i] - self.s_mu) / (self.s_sd + 1e-8)
        return dict(img=self._img("a", i), wrist=self._img("w", i), state=s.astype(np.float32), actions=a.astype(np.float32),
                    pad=pad, text=self.eps[int(self.ep[i])]["text"], env=self.eps[int(self.ep[i])]["env"],
                    ref=self.ref[i].astype(np.int64))


def collate(items, cond, cfg, rng=None):
    T = lambda k: torch.from_numpy(np.stack([it[k] for it in items]))                  # noqa: E731
    img = T("img").permute(0, 3, 1, 2).float() / 255.0
    wr = T("wrist").permute(0, 3, 1, 2).float() / 255.0
    st = torch.zeros(len(items), cfg.max_state_dim)
    st[:, :8] = T("state")
    act = torch.zeros(len(items), cfg.chunk, cfg.max_action_dim)
    act[:, :, :7] = T("actions")
    cs = [cond(it["text"], rng) for it in items]
    return dict(img=img, wrist=wr, state=st, actions=act, pad=T("pad"), ref=T("ref"),
                lang_tokens=torch.from_numpy(np.stack([c[0] for c in cs])), lang_masks=torch.from_numpy(np.stack([c[1] for c in cs])),
                unit_tokens=torch.from_numpy(np.stack([c[2] for c in cs])), unit_masks=torch.from_numpy(np.stack([c[3] for c in cs])))


class Batches(torch.utils.data.Dataset):
    """item(step) -> the whole env-balanced batch of that step (deterministic in (seed, step): resume-safe)."""

    def __init__(self, store, cond, cfg, n_env, per_env, seed):
        self.s, self.cond, self.cfg, self.n_env, self.per_env, self.seed = store, cond, cfg, n_env, per_env, seed
        self.emap, self.l1 = None, False                      # BindVLA-C (bv_c.py): swap-contrast entity map, prototype ids

    def __len__(self):
        return 10 ** 9

    def __getitem__(self, step):
        rng = np.random.default_rng([self.seed, int(step)])
        envs = rng.choice(len(self.s.envs), self.n_env, replace=False)
        idx = [int(rng.choice(self.s.by_env[self.s.envs[e]])) for e in envs for _ in range(self.per_env)]
        b = collate([self.s.item(i) for i in idx], self.cond, self.cfg, rng)
        b["env"] = torch.from_numpy(np.repeat(np.arange(self.n_env), self.per_env))
        b["u"] = torch.from_numpy(np.tile((np.arange(self.per_env) + rng.random(self.per_env)) / self.per_env, self.n_env)).float()
        b["noise"] = torch.from_numpy(np.tile(rng.standard_normal((self.per_env, self.cfg.chunk, self.cfg.max_action_dim)),
                                              (self.n_env, 1, 1))).float()
        if self.emap is not None:
            import bv_c
            u, m, t = bv_c.swap_items(self.s, self.emap, idx, self.cond, np.random.default_rng([self.seed, int(step), 7]))
            b["swap_units"], b["swap_masks"], b["swap_tok"] = torch.from_numpy(u), torch.from_numpy(m), torch.from_numpy(t)
        if self.l1:
            b["l1"] = torch.from_numpy(np.stack([self.cond.level1_ids(self.s.eps[int(self.s.ep[i])]["text"]) for i in idx]))
        b["step"] = int(step)
        return b


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def t_of(u):
    """stratified u -> SmolVLA's t ~ Beta(1.5, 1) x 0.999 + 0.001 (CDF x^1.5)."""
    return u.clamp(1e-6, 1) ** (1 / 1.5) * 0.999 + 0.001


def per_sample(loss, pad, cfg):
    m = (~pad).float()[..., None]
    return (loss[:, :, :7] * m).sum((1, 2)) / (m.sum((1, 2)) * 7).clamp(min=1)


# ── the model ────────────────────────────────────────────────────────────────────────────────────────────────────────
def build(setting, dev, init_base=False, **kw):
    from lifelong_ood.policy import bindvla as BV
    cfg = BV.make_config(setting, **kw)
    model = BV.BindVLA(cfg, device=dev).to(dev)
    if init_base:                                            # the public smolvla_base action expert + flow heads (PLAN_BINDVLA 11)
        from safetensors.torch import load_file
        sb = load_file(os.path.join(BASE, "model.safetensors"))
        sd = {k[len("model."):]: v for k, v in sb.items() if k.startswith("model.") and ".vlm_with_expert.vlm." not in "." + k[len("model."):]
              and not k[len("model."):].startswith("vlm_with_expert.vlm.")}
        own = model.flow.state_dict()
        bad = [k for k in sd if k not in own or own[k].shape != sd[k].shape]
        assert not bad, bad[:5]
        miss = [k for k in own if not k.startswith("vlm_with_expert.vlm.") and k not in sd]
        assert not miss, miss[:5]
        model.flow.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()}, strict=False)
        print(f"[bv] initialised {len(sd)} expert / head tensors from {BASE}", flush=True)
    if model.binder is not None and cfg.pos_keys:
        f = os.path.join(CACHE, "posfree.npz")
        assert os.path.exists(f), "run --posfit first"
        z = np.load(f)
        model.binder.P.copy_(torch.from_numpy(z["P"]))
        model.binder.P_mu.copy_(torch.from_numpy(z["mu"]))
    return cfg, model, BV.Conditioner(model.tokenizer, cfg)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * p))


# ── checkpoints ──────────────────────────────────────────────────────────────────────────────────────────────────────
def save(d, step, model, opt, extra):
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"step_{step:06d}.pt")
    torch.save(dict(step=step, model=model.trainable_state(), opt=opt.state_dict(), cfg=dataclasses.asdict(model.cfg), **extra), f + ".tmp")
    os.replace(f + ".tmp", f)
    have = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    for g in have[:-2]:
        s = int(os.path.basename(g)[5:11])
        if s % 10000:
            os.remove(g)
    return f


def latest(d):
    have = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    return have[-1] if have else None


def load_into(model, f, opt=None):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    missing = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing.unexpected_keys]
    assert not bad, bad
    if opt is not None:
        opt.load_state_dict(ck["opt"])
    return ck


# ── diagnostics ──────────────────────────────────────────────────────────────────────────────────────────────────────
def diag_frames(store):
    """held-out episodes with contact-event referents -> [(frame index, [(step, role, token, phrase)])]."""
    out = []
    for ep, recs in sorted(store.meta["diag"].items(), key=lambda x: int(x[0])):
        fi = int(np.flatnonzero(store.ep == int(ep))[0])
        step, rows = -1, []
        for r in recs:
            if r["role"] == "object":
                step += 1
            rows.append((step, r["role"], int(r["token"]), r["phrase"]))
        out.append((fi, rows))
    return out


@torch.no_grad()
def run_diag(model, cond, cfg, store, frames, dev, bs=16):
    """slot-mask referent hits on held-out episodes' frame 0 (contact-event referents): per role and per path depth (number of
    non-empty levels), final-level argmax within Chebyshev 1 of the referent token; chance 9 / 256."""
    if model.binder is None or cfg.parser in ("generic", "token"):
        return {}
    rec = []
    model.eval()
    for k in range(0, len(frames), bs):
        chunk = frames[k:k + bs]
        b = to_dev(collate([store.item(fi) for fi, _ in chunk], cond, cfg), dev)
        _, _, _, dg = model.build_prefix(b)
        M = dg["mask"].float().cpu().numpy()                    # [B, S, 2, 256]
        LV = dg["levels"].float().cpu().numpy()                 # [B, S, 2, L, 256]
        VV = dg["level_valid"].cpu().numpy()                    # [B, S, 2, L]
        for j, (_, rows) in enumerate(chunk):
            for step, role, tok, _ in rows:
                if step >= cfg.n_steps:
                    continue
                r = 0 if role == "object" else 1
                tr, tc = divmod(tok, 16)
                win = [(rr * 16 + cc) for rr in range(max(0, tr - 1), min(16, tr + 2)) for cc in range(max(0, tc - 1), min(16, tc + 2))]
                m = M[j, step, r]
                pr, pc = divmod(int(m.argmax()), 16)
                m1 = LV[j, step, r, 0]
                p1r, p1c = divmod(int(m1.argmax()), 16)
                rec.append(dict(role=role, depth=int(VV[j, step, r].sum()), hit=float(max(abs(pr - tr), abs(pc - tc)) <= 1),
                                hit1=float(max(abs(p1r - tr), abs(p1c - tc)) <= 1), mass=float(m[win].sum())))
    model.train()
    out = {}

    def summ(rs):
        h = np.array([x["hit"] for x in rs])
        rng = np.random.default_rng(0)
        bt = [rng.choice(h, len(h)).mean() for _ in range(1000)]
        return dict(n=len(rs), hit=float(h.mean()), hit_ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                    hit_level1=float(np.mean([x["hit1"] for x in rs])), mass3x3=float(np.mean([x["mass"] for x in rs])), chance=9 / 256)
    for role in ("object", "goal"):
        rs = [x for x in rec if x["role"] == role]
        if rs:
            out[role] = summ(rs)
    for d in (1, 2, 3):
        rs = [x for x in rec if x["depth"] == d]
        if rs:
            out[f"depth{d}"] = summ(rs)
    return out

@torch.no_grad()
def validation(model, cond, cfg, store, dev, n=256, bs=32, seed=12345):
    rng = np.random.default_rng(seed)
    idx = rng.choice(store.held_idx, n, replace=False)
    g = torch.Generator().manual_seed(seed)
    model.eval()
    tot = []
    for k in range(0, n, bs):
        b = to_dev(collate([store.item(int(i)) for i in idx[k:k + bs]], cond, cfg), dev)
        B = b["state"].shape[0]
        noise = torch.randn(B, cfg.chunk, cfg.max_action_dim, generator=g).to(dev)
        t = t_of(torch.rand(B, generator=g)).to(dev)
        loss, _ = model(b, noise, t)
        tot.append(per_sample(loss, b["pad"], cfg).cpu())
    model.train()
    return float(torch.cat(tot).mean())


# ── position fit ─────────────────────────────────────────────────────────────────────────────────────────────────────
def posfit(a):
    from lifelong_ood.policy import bindvla as BV
    dev = "cuda"
    store = Store()
    cfg, model, cond = build("vanilla_frozen", dev)
    rng = np.random.default_rng(0)
    idx = rng.choice(store.train_idx, a.posfit_frames, replace=False)
    X = []
    for k in range(0, len(idx), 32):
        img = torch.from_numpy(np.stack([store._img("a", int(i)) for i in idx[k:k + 32]])).permute(0, 3, 1, 2).float().div(255).to(dev)
        g, _ = model.vision(img)
        X.append(g.cpu().numpy().astype(np.float32))
    X = np.concatenate(X)
    pos = BV.grid_pos("cpu").numpy()
    t0 = time.time()
    P, mu, log = BV.fit_nullspace(X, pos)
    fit_s = time.time() - t0

    def mlp_r2(Z):
        n = len(Z)
        tr, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
        Xt = torch.from_numpy(Z[tr].reshape(-1, Z.shape[-1])).to(dev)
        Yt = torch.from_numpy(np.tile(pos, (int(tr.sum()), 1))).to(dev)
        Xe = torch.from_numpy(Z[te].reshape(-1, Z.shape[-1])).to(dev)
        Ye = torch.from_numpy(np.tile(pos, (int(te.sum()), 1))).to(dev)
        m_, s_ = Xt.mean(0), Xt.std(0) + 1e-6
        net = torch.nn.Sequential(torch.nn.Linear(Z.shape[-1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(),
                                  torch.nn.Linear(256, 2)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        g = torch.Generator(device=dev).manual_seed(0)
        for it in range(3000):
            j = torch.randint(0, len(Xt), (4096,), device=dev, generator=g)
            loss = ((net((Xt[j] - m_) / s_) - Yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pr = net((Xe - m_) / s_)
            r2 = 1 - ((Ye - pr) ** 2).sum(0) / ((Ye - Ye.mean(0)) ** 2).sum(0)
        return [float(x) for x in r2.cpu()]
    Xp = (X - mu) @ P
    res = dict(frames=int(len(X)), tokens=int(X.shape[0] * X.shape[1]), dim=int(X.shape[-1]), iterations=len(log) - 1,
               removed_dims=2 * (len(log) - 1), linear_r2_before=[log[0]["r2_row"], log[0]["r2_col"]],
               linear_r2_after=[log[-1]["r2_row"], log[-1]["r2_col"]], mlp_r2_before=mlp_r2(X), mlp_r2_after=mlp_r2(Xp),
               fit_secs=fit_s, log=log[:: max(1, len(log) // 20)] + [log[-1]])
    np.savez(os.path.join(CACHE, "posfree.npz"), P=P, mu=mu)
    import hashlib
    res["posfree_sha256"] = hashlib.sha256(open(os.path.join(CACHE, "posfree.npz"), "rb").read()).hexdigest()
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "posfree.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


# ── training ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def train(a):
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    kw = {}
    if a.inv is not None:
        kw["inv"] = a.inv
    if a.levels:
        kw["levels"] = a.levels
    if a.motor != "full":
        kw["motor"] = a.motor
    if a.no_glob:
        kw["glob"] = False
    store = Store()
    cflag = a.hold_layouts or a.swap > 0 or a.proto > 0 or a.adv > 0
    if cflag:
        import bv_c
    if a.hold_layouts:
        store.hold(bv_c.heldout_layouts(store, a.hold_layouts))
        print(f"[bv] held-out layouts {store.ho_envs}", flush=True)
    cfg, model, cond = build(a.setting, dev, init_base=a.init_base and not latest(os.path.join(OUT, a.exp, "ckpt")), **kw)
    d = os.path.join(OUT, a.exp)
    ckd = os.path.join(d, "ckpt")
    os.makedirs(d, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    extras = bv_c.Extras(model, a, dev) if cflag else None
    xparams = extras.params() if extras is not None else []
    opt = torch.optim.AdamW(params + xparams, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    start, ema = 0, None
    f = latest(ckd)
    if f and not a.timing:
        ck = load_into(model, f, opt)
        start, ema = int(ck["step"]), ck.get("ema")
        print(f"[bv] resumed {f} at step {start}", flush=True)
    ntr = model.n_trainable()
    print(f"[bv] {a.setting} {dataclasses.asdict(cfg)}\n[bv] trainable {ntr}", flush=True)
    per_env = a.batch // a.envs
    ds = Batches(store, cond, cfg, a.envs, per_env, seed=a.seed)
    xinfo = {}
    if cflag:
        if a.swap > 0:
            ds.emap = bv_c.entity_map(store)
            xinfo["entity_map"] = ds.emap
        ds.l1 = a.proto > 0
        xinfo["ho_envs"] = store.ho_envs
    json.dump(dict(setting=a.setting, cfg=dataclasses.asdict(cfg), params=ntr, args=vars(a), envs=store.envs,
                   n_train_frames=int(len(store.train_idx)), n_held_frames=int(len(store.held_idx)), **xinfo),
              open(os.path.join(d, "config.json"), "w", encoding="utf-8"), indent=1)
    lol_frames = bv_c.layout_frames(store, store.ho_envs) if cflag and store.ho_envs else []
    steps = a.timing or a.steps
    sampler = list(range(start, steps))
    dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=a.workers, prefetch_factor=4,
                                     persistent_workers=False, pin_memory=True)
    binder_params = [p for p in model.binder.parameters() if p.requires_grad] if model.binder is not None else []
    ema = torch.tensor(ema) if ema is not None else None
    frames = diag_frames(store)
    logf = open(os.path.join(d, "train_log.jsonl"), "a", encoding="utf-8")
    model.train()
    t0, tstep, n_t = time.time(), [], 0
    torch.cuda.reset_peak_memory_stats()
    acc = dict(loss=0.0, pen=0.0, spread=0.0, n=0)
    for b in dl:
        step = b.pop("step")
        b = to_dev(b, dev)
        ts = time.time()
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, a)
        t = t_of(b["u"])
        loss, dgm = model(b, b["noise"], t)
        ls = per_sample(loss, b["pad"], cfg)                                   # [B]
        bins = (b["u"] * NT).long().clamp(0, NT - 1)
        with torch.no_grad():
            bm = torch.zeros(NT, device=dev).scatter_add(0, bins, ls.detach()) / torch.bincount(bins, minlength=NT).clamp(min=1)
            ema = bm.cpu() if ema is None else 0.99 * ema + 0.01 * bm.cpu()
        z = ls / ema.to(dev)[bins]
        R = torch.zeros(a.envs, device=dev).scatter_add(0, b["env"], z) / per_env
        Rraw = torch.zeros(a.envs, device=dev).scatter_add(0, b["env"], ls.detach()) / per_env
        pen = R.var(unbiased=False)
        bc = ls.mean()
        aux = torch.zeros((), device=dev)
        if a.aux > 0 and dgm and "mask" in dgm:                               # contact-event referent signal (PLAN_BINDVLA 11)
            ref = b["ref"][:, : cfg.n_steps]                                   # [B, S, 2]
            ok = ref >= 0
            if ok.any():
                mass = (dgm["mask"].float() * WIN.to(dev)[ref.clamp(min=0)]).sum(-1)   # [B, S, 2]
                aux = -(torch.log(mass + 1e-4) * ok).sum() / ok.sum()
        total = bc + cfg.inv * pen + a.aux * aux
        if extras is not None:
            xl, xlog = extras(model, b, dgm or {}, WIN.to(dev))
            for k_, w_ in (("swap", a.swap), ("proto", a.proto), ("adv", a.adv)):
                if k_ in xl:
                    total = total + w_ * xl[k_]
                    acc["x_" + k_] = acc.get("x_" + k_, 0.0) + float(xl[k_].detach())
            for k_, v_ in xlog.items():
                acc["x_" + k_] = acc.get("x_" + k_, 0.0) + v_
        log_now = (step % 100 == 0)
        gr = None
        if log_now and binder_params and cfg.inv > 0:
            g1 = torch.autograd.grad(bc, binder_params, retain_graph=True, allow_unused=True)
            g2 = torch.autograd.grad(cfg.inv * pen, binder_params, retain_graph=True, allow_unused=True)
            n1 = math.sqrt(sum(float((x ** 2).sum()) for x in g1 if x is not None))
            n2 = math.sqrt(sum(float((x ** 2).sum()) for x in g2 if x is not None))
            gr = n2 / max(n1, 1e-12)
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        torch.cuda.synchronize()
        tstep.append(time.time() - ts)
        acc["loss"] += float(bc)
        acc["aux"] = acc.get("aux", 0.0) + float(aux)
        acc["pen"] += float(pen)
        acc["spread"] += float(Rraw.max() - Rraw.min())
        acc["n"] += 1
        if log_now:
            row = dict(step=step, loss=acc["loss"] / acc["n"], aux=acc.get("aux", 0.0) / acc["n"], pen=acc["pen"] / acc["n"], env_spread=acc["spread"] / acc["n"],
                       grad_ratio=gr, gnorm=float(gn), lr=lr_at(step, a), secs_step=float(np.mean(tstep[-100:])),
                       wall=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            row.update({k_: v_ / acc["n"] for k_, v_ in acc.items() if k_.startswith("x_")})
            if model.binder is not None:
                bd = model.binder
                row.update(tau=float(bd.log_tau.exp()), geo_w=float(bd.geo.weight.norm()), out_scale=float(bd.out_norm.log_s.exp()))
            acc = dict(loss=0.0, pen=0.0, spread=0.0, n=0)
            print(json.dumps(row), flush=True)
            logf.write(json.dumps(row) + "\n")
            logf.flush()
        done = step + 1
        if a.timing:
            continue
        if done % a.eval_every == 0 or done == a.steps:
            row = dict(step=done, val_loss=validation(model, cond, cfg, store, dev), diag=run_diag(model, cond, cfg, store, frames, dev))
            if lol_frames:
                row["lol"] = bv_c.lol_diag(model, cond, cfg, store, lol_frames, dev, collate, to_dev)
            if cflag and done == a.steps:
                row["key_probe"] = bv_c.key_probe(model, store, dev)
            print(json.dumps(row), flush=True)
            logf.write(json.dumps(row) + "\n")
            logf.flush()
        if done % a.save_every == 0 or done == a.steps:
            save(ckd, done, model, opt, dict(ema=ema.tolist(), setting=a.setting, stats=store.meta["stats"]))
    if a.timing:
        ts = np.asarray(tstep[5:])
        res = dict(setting=a.setting, steps=a.timing, batch=a.batch, secs_step=float(ts.mean()), secs_step_p90=float(np.percentile(ts, 90)),
                   wall_secs=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 1e9, params=ntr,
                   a40h_per_1k=float(ts.mean()) * 1000 / 3600, diag=run_diag(model, cond, cfg, store, frames, dev),
                   val_loss=validation(model, cond, cfg, store, dev, n=64))
        json.dump(res, open(os.path.join(d, "timing.json"), "w", encoding="utf-8"), indent=1)
        print("[timing]", json.dumps(res), flush=True)
    print(f"[bv] done at step {steps} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posfit", action="store_true")
    ap.add_argument("--posfit-frames", type=int, default=1200)
    ap.add_argument("--setting", default="bindvla")
    ap.add_argument("--exp", default="main")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--inv", type=float, default=None)
    ap.add_argument("--levels", type=int, default=0)
    ap.add_argument("--init-base", action="store_true", help="initialise the expert + heads from lerobot/smolvla_base")
    ap.add_argument("--aux", type=float, default=0.0, help="weight of the contact-event referent loss on the final slot masks")
    ap.add_argument("--hold-layouts", type=int, default=0, help="BindVLA-C: hold out N training layouts (envs) for the selection hit")
    ap.add_argument("--swap", type=float, default=0.0, help="BindVLA-C: weight of the swap contrast")
    ap.add_argument("--proto", type=float, default=0.0, help="BindVLA-C: weight of cross-layout prototype consistency")
    ap.add_argument("--adv", type=float, default=0.0, help="BindVLA-C: weight of the gradient-reversed position probe")
    ap.add_argument("--motor", default="full", choices=["full", "pose"], help="BindVLA-C: motor interface")
    ap.add_argument("--no-glob", action="store_true", help="BindVLA-C: drop the global token g from the prefix")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=1000)
    a = ap.parse_args()
    if a.posfit:
        posfit(a)
    else:
        train(a)


if __name__ == "__main__":
    main()

"""PLAN_PARTS S4 baselines (docs/PARTS_RESULTS.md section 3).  Each gets S3's data: the same base tasks (the held-out atom's tasks
excluded), then the same k demos of the same one task.  GPU (or CPU).

  HF   holistic policy (stage 1b's Hb architecture) trained on the base tasks, then FINE-TUNED (every parameter) on the k demos
  HL   the same holistic base, frozen; a rank-8 LoRA on its 4 hidden layers trained on the k demos; at test the LoRA is on for every
       task that contains the new atom (its most favourable use) and off otherwise
  RF   role-routed policy (Rb) trained on the base tasks, then fine-tuned on the k demos
  LIB  DMPEL / CLARE-style expert library: the holistic backbone plus rank-8 LoRA experts on its 4 hidden layers, mixed by a LEARNED
       ROUTER (an MLP on the instruction's role code and proprio -> softmax over the active experts).  Base: backbone, one expert per
       base task and the router trained jointly.  New atom: one expert is added and the new expert and the router are trained on the
       k demos (backbone and old experts frozen); the router's change is what can move old tasks.
Training: base exactly pp_p_train.py build's recipe; adaptation exactly pp_p_train.py create's (CREATE_STEPS, AdamW, batch 256, no
EMA), on the trainable set named above.
-> DATA/ckpt/pp_<tag>_s<seed>/<KIND>.npz (base) and <KIND>_<atom>_k<k>.npz; results/flawed/pp_bl_<tag>_s<seed>_<KIND>[_k<k>].json

    python experiments/pp_baselines.py build --kind HF --tag nobottle --exclude K4+K5 --seed 0
    python experiments/pp_baselines.py create --kind HF --tag nobottle --seed 0 --atom wine_bottle_1 --task K4 --k 5
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

RANK, EMAX = 8, 16
ARCH = {"HF": "Hb", "HL": "Hb", "RF": "Rb", "LIB": "Hb"}


class ExpertLinear(nn.Module):
    def __init__(self, base, n, rank=RANK):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(n, base.in_features, rank) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(n, rank, base.out_features))
        self.w = None                                                # [B, n] mixing weights, set per call

    def forward(self, x):
        y = self.base(x)
        if self.w is None:
            return y
        d = torch.einsum("bi,nir->bnr", x, self.A)
        return y + torch.einsum("bn,bnr,nro->bo", self.w, d, self.B)


class Baseline(nn.Module):
    def __init__(self, kind, horizon=8, center=(0, 0, 0), size=None, atom=-1):
        super().__init__()
        self.kind = kind
        self.core = FF.FactoredFlow(ARCH[kind], horizon, center=center, size=size)
        self.H = horizon
        self.register_buffer("atom", torch.tensor(int(atom)))       # HL: the entity whose tasks switch the LoRA on (-1: none)
        self.register_buffer("active", torch.tensor(0))              # LIB: number of active experts
        if kind in ("HL", "LIB"):
            n = 1 if kind == "HL" else EMAX
            for j in (0, 2, 4, 6):
                self.core.net[j] = ExpertLinear(self.core.net[j], n)
        if kind == "LIB":
            self.router = nn.Sequential(nn.Linear(2 * FF.N_ENT + 12, 64), nn.SiLU(), nn.Linear(64, EMAX))

    def experts(self):
        return [self.core.net[j] for j in (0, 2, 4, 6)] if self.kind in ("HL", "LIB") else []

    def weights(self, b):
        B = b["pos"].shape[0]
        if self.kind == "HL":
            on = ((b["oi"] == self.atom) | (b["ti"] == self.atom)).float()[:, None]
            return on
        oh = torch.eye(FF.N_ENT, device=b["pos"].device)
        prop = torch.cat([b["ee_pos"], b["ee_rot"][:, :, :2].reshape(B, 6), b["grip"] * 25.0, b["prev"][:, None]], -1)
        logits = self.router(torch.cat([oh[b["oi"]], oh[b["ti"]], prop], -1))
        mask = torch.arange(EMAX, device=logits.device) >= self.active
        return torch.softmax(logits.masked_fill(mask, -1e9), -1)

    def velocity(self, b, x, t):
        if self.experts():
            w = self.weights(b)
            for e in self.experts():
                e.w = w
        try:
            return self.core.velocity(b, x, t)
        finally:
            for e in self.experts():
                e.w = None

    def loss(self, b, a):
        B = a.shape[0]
        x0 = torch.randn_like(a)
        t = torch.rand(B, device=a.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
        return ((self.velocity(b, xt, t) - (a - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, b, steps=10, generator=None):
        B = b["pos"].shape[0]
        x = torch.randn(B, self.H, 7, generator=generator)
        for k in range(steps):
            t = torch.full((B,), k / steps)
            x = x + self.velocity(b, x, t) / steps
        return x

    def save(self, path, meta):
        np.savez(path, __meta__=json.dumps(dict(meta, baseline=self.kind, horizon=self.H)),
                 **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    m = Baseline(meta["baseline"], meta["horizon"])
    m.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "__meta__"})
    m.eval()
    return m, meta


def main():
    import pp_p_train as PT
    import fa1_common as FC
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "create"))
    ap.add_argument("--kind", required=True, choices=("HF", "HL", "RF", "LIB"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--atom")
    ap.add_argument("--task")
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = os.path.join(FC.DATA, "ckpt", f"pp_{a.tag}_s{a.seed}")
    os.makedirs(d, exist_ok=True)
    base_kind = "HF" if a.kind == "HL" else a.kind                   # HL adapts the HF base
    if a.mode == "build":
        assert a.kind != "HL", "HL uses HF's base"
        tasks = [c for c in FC.KNOWN if c not in a.exclude.split("+")]
        b, act, center, size, n, _ = PT.rows(dev, tasks)
        m = Baseline(a.kind, 8, center=center, size=size).to(dev)
        if a.kind == "LIB":                                         # one expert per base task, router learned jointly
            m.active.fill_(len(tasks))
        ema, curve, secs = PT.fit(m, list(m.parameters()), b, act, a.steps, 512, 3e-4, a.seed, 1e-4, log=f"build {a.kind} {a.tag}")
        ema.cpu().save(os.path.join(d, f"{a.kind}.npz"), dict(tag=a.tag, seed=a.seed, tasks=tasks, center=center, steps=a.steps))
        rep = dict(mode="build", kind=a.kind, tag=a.tag, seed=a.seed, tasks=tasks, n_rows=n, secs=secs, curve=curve,
                   params=int(sum(p.numel() for p in m.parameters())))
        fn = os.path.join(ROOT, "results", "flawed", f"pp_bl_{a.tag}_s{a.seed}_{a.kind}.json")
    else:
        src, meta = load(os.path.join(d, f"{base_kind}.npz"))
        if a.kind == "HL":
            m = Baseline("HL", 8, atom=FC.ENT.index(a.atom))
            sd = {k.replace("core.net.0.", "core.net.0.base.").replace("core.net.2.", "core.net.2.base.")
                  .replace("core.net.4.", "core.net.4.base.").replace("core.net.6.", "core.net.6.base."): v for k, v in src.state_dict().items()}
            missing = m.load_state_dict(sd, strict=False)
            assert all(k.endswith((".A", ".B")) for k in missing.missing_keys) and not missing.unexpected_keys, missing
            m.atom.fill_(FC.ENT.index(a.atom))
        else:
            m = src
        m = m.to(dev).train()
        for p in m.parameters():
            p.requires_grad_(False)
        mask_fn = None
        if a.kind in ("HF", "RF"):
            train = list(m.parameters())
        elif a.kind == "HL":
            train = [p for e in m.experts() for p in (e.A, e.B)]
        else:
            j = int(m.active)
            m.active.fill_(j + 1)
            train = [p for e in m.experts() for p in (e.A, e.B)] + list(m.router.parameters())
            ex = [p for e in m.experts() for p in (e.A, e.B)]

            def mask_fn():                                           # the old experts stay; only expert j and the router move
                for p in ex:
                    if p.grad is not None:
                        g = p.grad[j].clone()
                        p.grad.zero_()
                        p.grad[j] = g
        for p in train:
            p.requires_grad_(True)
        b, act, _, _, n, demos = PT.rows(dev, [a.task], a.k)
        t0 = time.time()
        mk = mask_fn if mask_fn is not None else (lambda: None)
        out, curve, secs = PT.fit(m, train, b, act, PT.CREATE_STEPS, 256, PT.CREATE_LR, a.seed, 0.0, mask_fn=mk, log=f"create {a.kind} k{a.k}")
        out.cpu().save(os.path.join(d, f"{a.kind}_{a.atom}_k{a.k}.npz"), dict(meta, created=a.atom, task=a.task, k=a.k, demos=demos))
        rep = dict(mode="create", kind=a.kind, tag=a.tag, seed=a.seed, atom=a.atom, task=a.task, k=a.k, demos=demos, n_rows=n, secs=secs,
                   gpu_seconds_to_ready=time.time() - t0, curve=curve, params_trained=int(sum(p.numel() for p in train)))
        fn = os.path.join(ROOT, "results", "flawed", f"pp_bl_{a.tag}_s{a.seed}_{a.kind}_{a.atom}_k{a.k}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[pp_bl] wrote {fn}", flush=True)


if __name__ == "__main__":
    main()

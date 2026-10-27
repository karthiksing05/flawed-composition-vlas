"""PLAN_SOLVE S4 (docs/PLAN_S4.md; pre-registration experiments/prereg/ps4.py): the NOISE-ROBUST binding adapter.  S0's recipe
(s0_train.py: same 56 training cells, frames, flow loss, frozen policy, AdamW 3e-4 / wd 0.01, batch 16, 3000 steps, seed 0, P_CF 0.5,
same held-out frames) with ONE change: the role masks of OWN-prompt examples are drawn from a mixture, per role independently:
  0.35 oracle (S0's masks)
  0.35 synthetic corruption, uniform over
         one_in   a single token of the oracle mask (an S1c-like peak that is right)
         one_out  a single token outside the oracle mask (an S1c-like peak that is wrong)
         shift    the oracle mask rolled by (dy, dx) in [-3, 3]^2 \\ {0} on the 16 x 16 grid
         dilate   the oracle mask dilated by 1-2 steps of a 3 x 3 max
         sibling  the mask of a competitor (S1's rule: same-type instance / same-body region / same-reference region); none ->
                  another body or region visible in the frame
         empty    zero
  0.30 the actual deployable binder masks of this frame (S1b or S1c, 50 / 50; results/bind/s4/bmask), or a synthetic corruption
       for the 2 cells without an aligned parse
cf examples (a same-scene partner's prompt) keep the oracle masks, as in S0 (the masks are then the only correct binding signal).
The target stays the demo chunk: a mask that disagrees with the demo can only raise the loss if the adapter follows it.

    .venv/bin/python experiments/s4_train.py [--smoke]   -> results/bind/s4/adapter.pt, train_log.json, heldout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s0_train as S0T  # noqa: E402
import s1_read as SR    # noqa: E402

OD = os.path.join(ROOT, "results", "bind", "s4")
P_ORACLE, P_SYN, P_BIND = 0.35, 0.35, 0.30
SYN = ("one_in", "one_out", "shift", "dilate", "sibling", "empty")


def dilate(m, k):
    g = m.reshape(16, 16) > 0
    for _ in range(k):
        p = np.pad(g, 1)
        g = np.max([p[dy:dy + 16, dx:dx + 16] for dy in range(3) for dx in range(3)], 0)
    return g.reshape(-1).astype(np.float32)


def corrupt(kind, m, others, rng):
    """m: [256] oracle cov; others: [k, 256] covs of the alternatives (siblings first, then any visible entry)."""
    on = np.flatnonzero(m > 0)
    off = np.flatnonzero(m <= 0)
    out = np.zeros(256, np.float32)
    if kind == "one_in" and len(on):
        out[rng.choice(on)] = 1.0
    elif kind == "one_out" and len(off):
        out[rng.choice(off)] = 1.0
    elif kind == "shift":
        while True:
            dy, dx = rng.integers(-3, 4, 2)
            if dy or dx:
                break
        g = np.zeros((16, 16), np.float32)
        src = (m > 0).reshape(16, 16)
        ys, xs = np.nonzero(src)
        ys, xs = ys + dy, xs + dx
        ok = (ys >= 0) & (ys < 16) & (xs >= 0) & (xs < 16)
        g[ys[ok], xs[ok]] = 1.0
        out = g.reshape(-1)
    elif kind == "dilate":
        out = dilate(m, int(rng.integers(1, 3)))
    elif kind == "sibling" and len(others):
        out = (others[0] > 0).astype(np.float32) if others[0].any() else out
    return out


def alternatives(c, i, role_name):
    """covs of competitors (S1's rule) then of every other visible entry, for the frame i of cell c."""
    names, cov = c["names"], c["cov"][i]
    bodies = names[:c["n_bodies"]]
    comp, _ = SR.competitors(role_name, names, bodies)
    comp = [n for n in comp if cov[names.index(n)].any()]
    rest = [n for n in names if n != role_name and n not in comp and cov[names.index(n)].any()]
    return np.stack([cov[names.index(n)] for n in comp + rest]) if comp + rest else np.zeros((0, 256), np.float32), len(comp)


def main():
    import glob
    import torch
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.adapt import _flow_batch
    from lifelong_ood.policy.backbone import PolicyBackbone
    import s0_adapter as SA
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cells = S0T.load_data()
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", "s0", "data", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        c = cells[str(z["tid"])]
        c["n_bodies"] = len(z["body_names"])
        c["named"], c["target"] = str(z["named"]), str(z["target"])
        bf = os.path.join(OD, "bmask", str(z["tid"]).replace(":", "_") + ".npz")
        c["bm"] = dict(np.load(bf)) if os.path.exists(bf) else None
    os.makedirs(OD, exist_ok=True)
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None)
    norm = D.Norm.for_backbone(bk)
    dev = bk.device
    items = [(t, i) for t, c in cells.items() for i, f in enumerate(c["frames"]) if f % S0T.VAL_MOD != 0]
    val = [(t, i) for t, c in cells.items() for i, f in enumerate(c["frames"]) if f % S0T.VAL_MOD == 0]
    print(f"[s4 train] {len(cells)} cells ({sum(c['bm'] is not None for c in cells.values())} with binder masks), {len(items)} train "
          f"frames, {len(val)} held-out frames", flush=True)

    def obs(rows):
        return bk.observations([{"observation/image": cells[t]["img"][i], "observation/wrist_image": cells[t]["wimg"][i],
                                 "observation/state": cells[t]["state8"][i], "prompt": p} for t, i, p in rows])

    def chunks(rows):
        c = np.stack([norm.normalize(cells[t]["chunk"][i]) for t, i, _ in rows])
        return torch.tensor(D.pad_rows(c.reshape(len(rows), -1), bk.H), device=dev)

    def role_mask(t, i, role, rng, src=None):
        c = cells[t]
        m = c["mo"][i] if role == "obj" else c["mg"][i]
        name = c["named"] if role == "obj" else c["target"]
        if src is None:
            r = rng.random()
            src = "oracle" if r < P_ORACLE else ("syn" if r < P_ORACLE + P_SYN else "bind")
        if src == "bind" and c["bm"] is None:
            src = "syn"
        if src == "oracle":
            return m.astype(np.float32), "oracle"
        if src in ("s1b", "s1c", "bind"):
            k = src if src != "bind" else ("s1b" if rng.random() < 0.5 else "s1c")
            return c["bm"][f"{k}_m{'o' if role == 'obj' else 'g'}"][i].astype(np.float32), k
        kind = src if src in SYN else SYN[int(rng.integers(len(SYN)))]
        alt, _ = alternatives(c, i, name)
        return corrupt(kind, m, alt, rng), kind

    ad = SA.BindAdapter().to(dev)
    opt = torch.optim.AdamW(ad.parameters(), lr=S0T.LR, weight_decay=S0T.WD)
    ck = os.path.join(OD, "ckpt.pt")
    step, hist, srcs = 0, [], {}
    if os.path.exists(ck) and not a.smoke:
        s = torch.load(ck, map_location=dev)
        ad.load_state_dict(s["adapter"])
        opt.load_state_dict(s["opt"])
        step, hist, srcs = s["step"], s["hist"], s.get("srcs", {})
        print(f"[s4 train] resumed at step {step}", flush=True)
    rng = np.random.default_rng(S0T.SEED + step)
    gen = torch.Generator(device="cpu").manual_seed(S0T.SEED + step)
    steps = 3 if a.smoke else S0T.STEPS
    t0 = time.time()
    while step < steps:
        pick = [items[j] for j in rng.integers(0, len(items), S0T.BATCH)]
        rows, MO, MG = [], [], []
        for t, i in pick:
            cf = cells[t]["cf"]
            if cf and rng.random() < S0T.P_CF:
                rows.append((t, i, cf[rng.integers(len(cf))]))
                mo, mg, so, sg = cells[t]["mo"][i], cells[t]["mg"][i], "cf_oracle", "cf_oracle"
            else:
                rows.append((t, i, cells[t]["prompt"]))
                (mo, so), (mg, sg) = role_mask(t, i, "obj", rng), role_mask(t, i, "goal", rng)
            MO.append(mo)
            MG.append(mg)
            for s_ in (so, sg):
                srcs[s_] = srcs.get(s_, 0) + 1
        with torch.no_grad():
            pre = bk.prefix(obs(rows))
        A = chunks(rows)
        pe = ad.apply(pre, torch.tensor(np.stack(MO), device=dev), torch.tensor(np.stack(MG), device=dev))
        tt, eps = _flow_batch(len(rows), bk.H, bk.A, gen, dev)
        xt = tt[:, None, None] * eps + (1 - tt[:, None, None]) * A
        u = (eps - A).reshape(len(rows), -1)
        loss = ((bk.v_full(xt, tt, pe) - u) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        hist.append(float(loss))
        if step % 25 == 0 or a.smoke:
            print(f"[s4 train] step {step}: loss {np.mean(hist[-25:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % S0T.CKPT_EVERY == 0 and not a.smoke:
            torch.save(dict(adapter=ad.state_dict(), opt=opt.state_dict(), step=step, hist=hist, srcs=srcs), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if not a.smoke:
        torch.save(ad.state_dict(), os.path.join(OD, "adapter.pt"))
    json.dump(dict(steps=step, loss=hist, sources=srcs, bytes=ad.nbytes(), n_train=len(items)),
              open(os.path.join(OD, "train_log.json" if not a.smoke else "train_log_smoke.json"), "w"))
    print(f"[s4 train] done: {step} steps; mask sources {srcs}", flush=True)
    # held-out frames, OWN prompt, fixed noise / times: loss of {S0 adapter, S4 adapter} x mask source (base = no edit)
    s0ad = SA.BindAdapter().to(dev)
    s0ad.load_state_dict(torch.load(os.path.join(ROOT, "results", "bind", "s0", "adapter.pt"), map_location=dev))
    ad.eval()
    s0ad.eval()
    vb = val[:64] if a.smoke else val
    out = {}
    g2 = torch.Generator(device="cpu").manual_seed(123)
    for b0 in range(0, len(vb), 16):
        rows = [(t, i, cells[t]["prompt"]) for t, i in vb[b0:b0 + 16]]
        with torch.no_grad():
            pre = bk.prefix(obs(rows))
            A = chunks(rows)
            tt, eps = _flow_batch(len(rows), bk.H, bk.A, g2, dev)
            xt = tt[:, None, None] * eps + (1 - tt[:, None, None]) * A
            u = (eps - A).reshape(len(rows), -1)
            L = lambda p: ((bk.v_full(xt, tt, p) - u) ** 2).mean(-1).cpu().numpy()  # noqa: E731
            out.setdefault("base", []).extend(L(pre))
            r2 = np.random.default_rng(b0)
            for src in ("oracle", "s1b", "s1c") + SYN:
                ok = np.array([src not in ("s1b", "s1c") or cells[t]["bm"] is not None for t, i, _ in rows])
                ms = [[role_mask(t, i, role, r2, src)[0] if k else np.zeros(256, np.float32) for role in ("obj", "goal")]
                      for (t, i, _), k in zip(rows, ok)]
                mo = torch.tensor(np.stack([m[0] for m in ms]), device=dev)
                mg = torch.tensor(np.stack([m[1] for m in ms]), device=dev)
                for nm, A_ in (("s0", s0ad), ("s4", ad)):
                    out.setdefault(f"{nm}_{src}", []).extend(L(A_.apply(pre, mo, mg))[ok])
                out.setdefault(f"base_{src}", []).extend(L(pre)[ok])
    res = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in out.items()}
    json.dump(res, open(os.path.join(OD, "heldout.json" if not a.smoke else "heldout_smoke.json"), "w"), indent=1)
    for k in sorted(res):
        print(f"[s4 heldout] {k}: {res[k]['mean']:.4f} (n {res[k]['n']})", flush=True)


if __name__ == "__main__":
    main()

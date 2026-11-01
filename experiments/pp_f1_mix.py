"""PLAN_PARTS F1 (docs/PLAN_PARTS.md section 3; protocol in docs/PARTS_RESULTS.md section 1): the role-mixing map of a VLA by
activation patching on the N2 scene.  Open loop, no rollouts; demos are scored only.  One GPU, the model's own venv
(pi0.5: openpi; GR00T: groot).

Frames: results/flawed/pp_f1_frames.npz (pp_f1_frames.py).  Tasks: N2's 12 pick-and-place compounds K1-K6, N1-N6.
  grasp phase  every compound x layout, on that layout's 10 start-state frames.  base = the task's instruction;
               CROSS partner = the target-changed task (same object, another target); OWN partner = the object-changed task.
  place phase  familiar layout, on the 10 just-lifted demo frames of the task's object (pp_f1_frames.SRC).  CROSS partner = the
               object-changed task (same target); OWN partner = the target-changed task.
  partner rule among the 12: same atom in the kept role, another atom in the changed role; known before new, then compound order.
  A task without an OWN or a CROSS partner in a phase is skipped there (K5 in both: the rack has only the bottle).
Components (the cross-role patch: the CROSS partner's activations patched into the base run):
  W      the changed role's instruction words (the token span where the two instructions differ), at the expert-read interface:
         pi0.5 every layer's K and V at those positions; GR00T the backbone features the DiT cross-attends (kv["vl"])
  V      the changed role's agentview image tokens (Chebyshev radius 1 on the model's own patch grid around the entity's projected
         point), at the expert-read interface
  Rest   every other expert-read position (the remaining image tokens of both views, the other words, pi0.5's state tokens)
  KV_l   pi0.5 only, l = 0..17: layer l's K and V at every position (the role span included when the two spans have equal length;
         otherwise the span keeps the base values at that layer, `kv_span_aligned` records which)
  Q_l    GR00T only, l = 0..11: the changed role's words in the residual stream entering Eagle layer l, propagated through the
         later layers (equal-length spans only; NaN otherwise)
  E_j    the action expert's layer j output on the action tokens, patched at every denoising step from the partner run on the same
         x_t (pi0.5 Gemma expert layers 0..17; GR00T DiT blocks 0..15)
  full_cross / full_own   the whole instruction swapped to the CROSS / OWN partner.
Per frame: d = MSE between the patched and the base sample over the first min(10, H) steps x 7 real dims, in the model's normalised
rows, mean over 2 noise seeds shared by every variant of a frame.  -> results/flawed/pp_f1_<model>.json (per task and phase: per-
frame d for every component; spans; partners)

    python experiments/pp_f1_mix.py --model pi05|groot [--limit N --out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
SEEDS = (0, 1)
HE = 10
PP = ["K1", "K2", "K3", "K4", "K5", "K6", "N1", "N2", "N3", "N4", "N5", "N6"]


def hseed(*k):
    return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:8], 16)


def tasks_pp():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    T = {}
    for t in n2["tasks"]:
        if t["compound"] in PP and t["layout"] == "familiar":
            T[t["compound"]] = dict(text=t["language"], obj=t["object"], tgt=t["target"], status=t["status"])
    return T


def partner(T, c, keep):
    """keep = 'obj' (target-changed partner) or 'tgt' (object-changed partner)."""
    chg = "tgt" if keep == "obj" else "obj"
    cand = [u for u in PP if u != c and T[u][keep] == T[c][keep] and T[u][chg] != T[c][chg]]
    cand.sort(key=lambda u: (T[u]["status"] != "known", PP.index(u)))
    return cand[0] if cand else None


def diff_span(a, b):
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return p, len(a) - p - s, len(b) - p - s          # start, base span length, partner span length


# ── model adapters ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Pi05:
    name = "pi05"

    def __init__(self):
        from lifelong_ood.policy.backbone import PolicyBackbone
        self.bk = PolicyBackbone("pi05_libero", device="cuda")
        self.layers = list(self.bk.model.paligemma_with_expert.gemma_expert.model.layers)
        self.nread = len(self.bk.model.paligemma_with_expert.paligemma.language_model.layers)
        self.grid, self.cell, self.img0 = 16, 14, 0               # base image tokens 0..255 (16 x 16 SigLIP patches of 224)

    def obs(self, fr, idx, text):
        return self.bk.observations([{"observation/image": fr["img"][i], "observation/wrist_image": fr["wimg"][i],
                                      "observation/state": fr["state8"][i], "prompt": text} for i in idx])

    def prefix(self, o):
        p = self.bk.prefix(o)
        m = o.tokenized_prompt_mask[0].bool()
        p.ids = o.tokenized_prompt[0][m].tolist()
        p.base = 256 * len(o.images)
        return p

    def kv(self, p):
        kv = p.kv
        if hasattr(kv, "key_cache"):
            return list(kv.key_cache), list(kv.value_cache)
        return [l.keys for l in kv.layers], [l.values for l in kv.layers]

    def make(self, p, K, V, mask):
        c = type(p.kv)()
        for l, (k, v) in enumerate(zip(K, V)):
            c.update(k.contiguous(), v.contiguous(), l)
        from lifelong_ood.policy.backbone import Prefix
        q = Prefix(state=p.state, pad_masks=mask, kv=c, n=p.n)
        q.ids, q.base = p.ids, p.base
        return q

    def pos_mask(self, p, rows_pos):
        """rows_pos: list per row of sequence positions -> bool [B, T]."""
        T = p.pad_masks.shape[1]
        m = torch.zeros((p.n, T), dtype=torch.bool, device=p.pad_masks.device)
        for b, ps in enumerate(rows_pos):
            m[b, ps] = True
        return m

    def splice_mask(self, pb, pp, m, layers=None):
        """base everywhere, partner where m (same alignment: positions before the text span) at the given layers (all)."""
        Kb, Vb = self.kv(pb)
        Kp, Vp = self.kv(pp)
        mm = m[:, None, :, None]
        L = range(len(Kb)) if layers is None else layers
        K = [torch.where(mm, Kp[l], Kb[l]) if l in L else Kb[l] for l in range(len(Kb))]
        V = [torch.where(mm, Vp[l], Vb[l]) if l in L else Vb[l] for l in range(len(Vb))]
        return self.make(pb, K, V, pb.pad_masks)

    def _cat_seq(self, parts, T):
        """parts: list of (tensor [B, H, t, d] or mask [B, t]) along the sequence; trimmed / padded at the end to length T."""
        x = torch.cat(parts, dim=-2 if parts[0].dim() == 4 else 1)
        n = x.shape[-2] if x.dim() == 4 else x.shape[1]
        if n > T:
            return x[..., :T, :] if x.dim() == 4 else x[:, :T]
        if n < T:
            pad = list(x.shape)
            if x.dim() == 4:
                pad[2] = T - n
            else:
                pad[1] = T - n
            return torch.cat([x, torch.zeros(pad, dtype=x.dtype, device=x.device)], dim=2 if x.dim() == 4 else 1)
        return x

    def splice_span(self, pb, pp, sp, which):
        """which = 'W': base with the partner's role span; 'Rest': partner with the base's role span.  sp = (start, Lb, Lp) in token
        coordinates relative to the text region."""
        s, Lb, Lp = sp
        o = pb.base
        T = pb.pad_masks.shape[1]
        Kb, Vb = self.kv(pb)
        Kp, Vp = self.kv(pp)
        nb, np_ = len(pb.ids), len(pp.ids)
        if which == "W":
            A, a0, a1, B_, b0, b1 = (Kb, Vb, pb.pad_masks), o + s, o + s + Lb, (Kp, Vp, pp.pad_masks), o + s, o + s + Lp
        else:
            A, a0, a1, B_, b0, b1 = (Kp, Vp, pp.pad_masks), o + s, o + s + Lp, (Kb, Vb, pb.pad_masks), o + s, o + s + Lb
        out_K, out_V = [], []
        for l in range(len(Kb)):
            out_K.append(self._cat_seq([A[0][l][:, :, :a0], B_[0][l][:, :, b0:b1], A[0][l][:, :, a1:]], T))
            out_V.append(self._cat_seq([A[1][l][:, :, :a0], B_[1][l][:, :, b0:b1], A[1][l][:, :, a1:]], T))
        full = torch.cat([A[2][:, :a0], B_[2][:, b0:b1], A[2][:, a1:]], 1)
        assert not bool(full[:, T:].any()), "splice: a valid position fell beyond the prefix length"
        mask = self._cat_seq([A[2][:, :a0], B_[2][:, b0:b1], A[2][:, a1:]], T)
        assert int(mask[0].sum()) == int(A[2][0].sum()) - (a1 - a0) + (b1 - b0), "splice: valid-token count"
        q = self.make(pb, out_K, out_V, mask)
        q.ids = (pb if which == "W" else pp).ids
        return q

    def splice_layer(self, pb, pp, sp, l):
        s, Lb, Lp = sp
        o = pb.base
        Kb, Vb = self.kv(pb)
        Kp, Vp = self.kv(pp)
        nb, np_ = len(pb.ids), len(pp.ids)
        if Lb == Lp:
            K = [Kp[j] if j == l else Kb[j] for j in range(len(Kb))]
            V = [Vp[j] if j == l else Vb[j] for j in range(len(Vb))]
            assert torch.equal(pb.pad_masks, pp.pad_masks)
            return self.make(pb, K, V, pb.pad_masks), True

        def al(X, Y):              # partner layer-l values on the base alignment, the span kept from the base
            return torch.cat([Y[:, :, :o + s], X[:, :, o + s:o + s + Lb], Y[:, :, o + s + Lp:o + np_], X[:, :, o + nb:]], 2)
        K = [al(Kb[j], Kp[j]) if j == l else Kb[j] for j in range(len(Kb))]
        V = [al(Vb[j], Vp[j]) if j == l else Vb[j] for j in range(len(Vb))]
        assert K[l].shape == Kb[l].shape
        return self.make(pb, K, V, pb.pad_masks), False

    def span(self, pb, pp):
        return diff_span(pb.ids, pp.ids)

    def img_positions(self, pix):
        """pix [B, 2] (row, col) in the 224 model image -> per row, the base-image token positions within Chebyshev 1."""
        out = []
        for r, c in np.asarray(pix):
            i, j = int(np.clip(r // self.cell, 0, self.grid - 1)), int(np.clip(c // self.cell, 0, self.grid - 1))
            out.append([self.img0 + a * self.grid + b for a in range(max(0, i - 1), min(self.grid, i + 2))
                        for b in range(max(0, j - 1), min(self.grid, j + 2))])
        return out

    def text_positions(self, p, sp):
        s, Lb, _ = sp
        return list(range(p.base + s, p.base + s + Lb))


class Groot:
    name = "groot"

    def __init__(self):
        from lifelong_ood.policy.backbone_groot import GrootBackbone
        self.bk = GrootBackbone(None, device="cuda")
        self.layers = list(self.bk.head.model.transformer_blocks)
        self.nread = len(self.bk.cond_layers())
        self.img_id = int(self.bk.gm.backbone.eagle_model.config.image_token_index)
        self._rec = None

    def obs(self, fr, idx, text):
        return self.bk.observations([{"observation/image": fr["img"][i], "observation/wrist_image": fr["wimg"][i],
                                      "observation/state": fr["state8"][i], "prompt": text} for i in idx])

    def prefix(self, o, hook=None):
        hs = self.bk.cond_hook(hook) if hook is not None else []
        try:
            p = self.bk.prefix(o)
        finally:
            for h in hs:
                h.remove()
        p.ids = o.eagle["eagle_input_ids"][0].tolist()
        p.o = o
        ii = np.flatnonzero(np.asarray(p.ids) == self.img_id)
        side = int(round(np.sqrt(len(ii) / 2)))
        assert 2 * side * side == len(ii), ("image token count", len(ii))
        p.img = ii[: side * side]
        p.side = side
        return p

    def with_vl(self, p, vl):
        from lifelong_ood.policy.backbone import Prefix
        kv = dict(p.kv)
        kv["vl"] = vl
        q = Prefix(state=p.state, pad_masks=p.pad_masks, kv=kv, n=p.n)
        q.ids, q.img, q.side = p.ids, p.img, p.side
        return q

    def splice_mask(self, pb, pp, m, layers=None):
        """base everywhere, partner where m; m marks positions before the role span (image tokens), aligned in both runs."""
        vb, vp = pb.kv["vl"], pp.kv["vl"]
        T = min(vb.shape[1], vp.shape[1])
        assert not bool(m[:, T:].any()), "splice_mask: a masked position beyond the shorter run"
        v = vb.clone()
        v[:, :T] = torch.where(m[:, :T, None], vp[:, :T], vb[:, :T])
        return self.with_vl(pb, v)

    def splice_span(self, pb, pp, sp, which):
        s, Lb, Lp = sp
        vb, vp = pb.kv["vl"], pp.kv["vl"]
        if which == "W":
            return self.with_vl(pb, torch.cat([vb[:, :s], vp[:, s:s + Lp], vb[:, s + Lb:]], 1))
        return self.with_vl(pb, torch.cat([vp[:, :s], vb[:, s:s + Lb], vp[:, s + Lp:]], 1))

    def pos_mask(self, p, rows_pos):
        T = p.kv["vl"].shape[1]
        m = torch.zeros((p.n, T), dtype=torch.bool, device=p.kv["vl"].device)
        for b, ps in enumerate(rows_pos):
            m[b, ps] = True
        return m

    def span(self, pb, pp):
        return diff_span(pb.ids, pp.ids)

    def img_positions_for(self, p, pix):
        side, cell = p.side, 224.0 / p.side
        out = []
        for r, c in np.asarray(pix):
            i, j = int(np.clip(r // cell, 0, side - 1)), int(np.clip(c // cell, 0, side - 1))
            out.append([int(p.img[a * side + b]) for a in range(max(0, i - 1), min(side, i + 2)) for b in range(max(0, j - 1), min(side, j + 2))])
        return out

    def text_positions(self, p, sp):
        s, Lb, _ = sp
        return list(range(s, s + Lb))

    def trace(self, o_b, pb, pp, sp, l):
        """Q_l: the partner's role-word residual entering Eagle layer l, propagated (equal-length spans only)."""
        s, Lb, Lp = sp
        if Lb != Lp:
            return None
        store = {}

        def rec(i, h, span):
            if i == l:
                store["h"] = h[:, s:s + Lp].clone()
            return None
        self.prefix(pp.o, hook=rec)

        def rep(i, h, span):
            if i == l:
                h = h.clone()
                h[:, s:s + Lb] = store["h"].to(h.dtype)
                return h
            return None
        return self.prefix(o_b, hook=rep)


# ── sampling (the backbone's own Euler loop, with an optional expert-layer patch) ────────────────────────────────────────────────
def _out0(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def sample(M, pre, seed, patch=None):
    """-> real rows [n, H, 7].  patch = (layer module, partner prefix): at every step the layer's output on the base run is replaced
    by the partner run's output at the same x_t."""
    bk = M.bk
    n = pre.n
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn((n, bk.dim_full), generator=g).to(bk.device)
    K = bk.num_steps
    dt = torch.tensor(-1.0 / K, dtype=torch.float32, device=bk.device)
    t = torch.tensor(1.0, dtype=torch.float32, device=bk.device)
    while t >= -dt / 2:
        if patch is None:
            v = bk.v_full(x, t.expand(n), pre)
        else:
            mod, pp = patch
            st = {}
            h1 = mod.register_forward_hook(lambda m, a, o: st.__setitem__("h", _out0(o).detach().clone()))
            try:
                bk.v_full(x, t.expand(n), pp)
            finally:
                h1.remove()

            def rep(m, a, o):
                if isinstance(o, tuple):
                    return (st["h"].to(o[0].dtype),) + tuple(o[1:])
                return st["h"].to(o.dtype)
            h2 = mod.register_forward_hook(rep)
            try:
                v = bk.v_full(x, t.expand(n), pre)
            finally:
                h2.remove()
        x = x + dt * v
        t = t + dt
    return x[:, bk.real].float().reshape(n, bk.H, bk.a).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("pi05", "groot"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    M = Pi05() if a.model == "pi05" else Groot()
    H = min(HE, M.bk.H)
    T = tasks_pp()
    Z = np.load(os.path.join(OUT, "pp_f1_frames.npz"))
    fr = {k: Z[k] for k in Z.files}
    ENT = list(Z["ent"])
    jobs = []
    for c in PP:
        for lay in (0, 1):
            jobs.append((c, "grasp", lay))
        jobs.append((c, "place", 0))
    if a.limit:                                     # smoke: the first N grasp jobs and the first N place jobs
        jobs = [j for j in jobs if j[1] == "grasp"][: a.limit] + [j for j in jobs if j[1] == "place"][: a.limit]
    res = {}
    for c, ph, lay in jobs:
        cross = partner(T, c, "obj" if ph == "grasp" else "tgt")
        own = partner(T, c, "tgt" if ph == "grasp" else "obj")
        key = f"{c}|{ph}|{'familiar' if lay == 0 else 'new'}"
        if cross is None or own is None:
            res[key] = dict(skipped=True, cross=cross, own=own)
            continue
        if ph == "grasp":
            idx = np.flatnonzero((fr["phase"] == 0) & (fr["layout"] == lay))
        else:
            idx = np.flatnonzero((fr["phase"] == 1) & (fr["obj"] == ENT.index(T[c]["obj"])))
        role = "tgt" if ph == "grasp" else "obj"                     # the changed (cross) role
        ent = ENT.index(T[c][role])
        o_b, o_x, o_o = M.obs(fr, idx, T[c]["text"]), M.obs(fr, idx, T[cross]["text"]), M.obs(fr, idx, T[own]["text"])
        with torch.no_grad():
            pb, px, po = M.prefix(o_b), M.prefix(o_x), M.prefix(o_o)
            sp = M.span(pb, px)
            if a.model == "pi05":
                ipos = M.img_positions(fr["pix"][idx, ent])
            else:
                ipos = M.img_positions_for(pb, fr["pix"][idx, ent])
            wpos = M.text_positions(pb, sp)
            mV = M.pos_mask(pb, ipos)
            variants = {"full_cross": px, "full_own": po, "W": M.splice_span(pb, px, sp, "W"),
                        "V": M.splice_mask(pb, px, mV), "Rest": M.splice_span(pb, M.splice_mask(px, pb, M.pos_mask(px, ipos)), sp, "Rest")}
            aligned = None
            if a.model == "pi05":
                for l in range(M.nread):
                    variants[f"KV_{l}"], aligned = M.splice_layer(pb, px, sp, l)
            else:
                for l in range(M.nread):
                    q = M.trace(o_b, pb, px, sp, l)
                    if q is not None:
                        variants[f"Q_{l}"] = q
            seeds = [hseed("pp_f1", c, ph, lay, s) for s in SEEDS]
            y0 = [sample(M, pb, s)[:, :H] for s in seeds]
            d = {}
            for vn, pre in variants.items():
                ys = [sample(M, pre, s)[:, :H] for s in seeds]
                d[vn] = np.mean([((y - b) ** 2).mean((1, 2)) for y, b in zip(ys, y0)], 0).tolist()
            for j, mod in enumerate(M.layers):
                ys = [sample(M, pb, s, patch=(mod, px))[:, :H] for s in seeds]
                d[f"E_{j}"] = np.mean([((y - b) ** 2).mean((1, 2)) for y, b in zip(ys, y0)], 0).tolist()
            if a.model == "groot":
                for l in range(M.nread):
                    d.setdefault(f"Q_{l}", [float("nan")] * len(idx))
        res[key] = dict(compound=c, phase=ph, layout="familiar" if lay == 0 else "new", cross=cross, own=own, frames=idx.tolist(),
                        span=list(sp), kv_span_aligned=aligned, n_img_tokens=[len(x) for x in ipos], word_positions=wpos, d=d)
        print(f"[pp_f1] {a.model} {key} cross={cross} own={own} span={sp} {time.time() - t0:.0f}s", flush=True)
    out = dict(model=a.model, H=H, seeds=list(SEEDS), components=sorted({k for r in res.values() for k in r.get("d", {})}),
               tasks=res, secs=time.time() - t0, describe=M.bk.describe())
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"pp_f1_{a.model}{'_smoke' if a.limit else ''}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[pp_f1] wrote {fn} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

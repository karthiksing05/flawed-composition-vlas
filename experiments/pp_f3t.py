"""PLAN_PARTS targeted F3 (docs/PARTS_RESULTS.md section 7; m3_prereg pp_f3t): purify only the TARGET-IDENTITY SUBSPACE of pi0.5's
layer-14 KV (the component F1 located), keeping its orthogonal complement.  GPU, openpi venv.

Subspace estimator (known tasks only, no rollout, no outcome):
- Contrasts: for every pair of KNOWN tasks that share the object and differ in the target (bowl: K1-K2, K1-K3, K2-K3; bottle:
  K4-K5), on every F1 frame that pair can be read on (the 20 grasp frames of both layouts, and the object's own 10 place frames),
  the difference of the two runs' layer-14 keys and values at every aligned valid position (every image token; every instruction
  token outside the span where the two instructions differ).
- Keys are compared before RoPE: k is de-rotated at its own position ids (u = k cos - rot_half(k) sin); values as they are.
- U_K, U_V = the top R right singular vectors (R = 16, fixed) of the stacked key / value contrasts (uncentred: differences).
- Control: a random R-dim subspace of each (QR of a Gaussian, torch seed 0 for K, 1 for V).
- Reported (diagnostic): the share of contrast energy each subspace captures, in-sample and on odd frames when fit on even frames.
Targeted purification of a new-pairing run (o, t), source S:
- u' = u + P_U (u_S - u), v' = v + P_V (v_S - v) at every aligned valid position (positions mapped across the instructions'
  differing span as above), then k' = rot(u') at the base run's own positions.  Every other layer and position is unchanged.
- Arms: T  (U = the target subspace, S = B, the known task sharing the TARGET: plate K1, stove K2, cabinet K3, bowl K6) - primary
        R  (U = the random subspace, S = B) - control
        TA (U = the target subspace, S = A, the known task sharing the OBJECT: cheese K6, bottle K4) - open loop only, descriptive
Open loop (`open`): on the place frames of N1-N6 (familiar) and the grasp frames of N1-N6 (both layouts), the cosine between the
chunk's summed translation (first 10 steps) and the direction to the target (place) / object (grasp), per arm; mean over 2 seeds.
-> results/flawed/pp_f3t_subspace.npz + pp_f3t_subspace.json (`fit`), results/flawed/pp_f3t_open.json (`open`)

    python experiments/pp_f3t.py fit
    python experiments/pp_f3t.py open
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from pp_f1_mix import OUT, SEEDS, Pi05, diff_span, hseed, sample, tasks_pp   # noqa: E402
from pp_purify import NEW, partners_of                                         # noqa: E402

LAYER, R = 14, 16
PAIRS = [("K1", "K2"), ("K1", "K3"), ("K2", "K3"), ("K4", "K5")]
SUBFILE = os.path.join(OUT, "pp_f3t_subspace.npz")
_SUB = {}


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(M, p):
    """-> cos, sin [B, 1, T, d] at the prefix's own position ids (cumsum of the pad mask - 1, as openpi)."""
    pos = torch.cumsum(p.pad_masks.long(), dim=1) - 1
    re = M.bk.model.paligemma_with_expert.paligemma.model.language_model.rotary_emb
    d = M.kv(p)[0][LAYER].shape[-1]
    cos, sin = re(torch.zeros(1, 1, d, device=pos.device, dtype=torch.float32), pos)
    return cos.float()[:, None], sin.float()[:, None]


def derot(k, cs):
    return k * cs[0] - rot_half(k) * cs[1]


def rerot(u, cs):
    return u * cs[0] + rot_half(u) * cs[1]


def row_ids(o):
    m = o.tokenized_prompt_mask.bool()
    return [o.tokenized_prompt[b][m[b]].tolist() for b in range(m.shape[0])]


def align(ib, is_, base, vb, vs):
    """-> (base positions, source positions): every valid image position, and every instruction token outside the differing span."""
    s, Lb, Ls = diff_span(ib, is_)
    keep = [j for j in range(len(ib)) if j < s or j >= s + Lb]
    pb = [i for i in range(base) if vb[i] and vs[i]] + [base + j for j in keep]
    ps = [i for i in range(base) if vb[i] and vs[i]] + [base + (j if j < s else j + Ls - Lb) for j in keep]
    return pb, ps


def prefix_rows(M, fr, idx, text):
    o = M.obs(fr, idx, text)
    p = M.prefix(o)
    return p, row_ids(o)


def layer_uv(M, p):
    K, V = M.kv(p)
    return derot(K[LAYER].float(), rope(M, p)), V[LAYER].float()


def subspaces():
    if not _SUB:
        z = np.load(SUBFILE)
        for k in z.files:
            _SUB[k] = torch.tensor(z[k])
    return _SUB


def targeted(M, pb, ids_b, ps, ids_s, UK, UV):
    """The base prefix with layer 14's keys / values moved to the source's inside span(U), at aligned positions."""
    K, V = M.kv(pb)
    csb = rope(M, pb)
    ub, vb = derot(K[LAYER].float(), csb), V[LAYER].float()
    us, vs = layer_uv(M, ps)
    UK, UV = UK.to(ub.device), UV.to(ub.device)
    Kn, Vn = K[LAYER].clone(), V[LAYER].clone()
    for b in range(ub.shape[0]):
        ab, as_ = align(ids_b[b], ids_s[b], pb.base, pb.pad_masks[b].tolist(), ps.pad_masks[b].tolist())
        ab_t, as_t = torch.tensor(ab, device=ub.device), torch.tensor(as_, device=ub.device)
        du = us[b][:, as_t] - ub[b][:, ab_t]
        dv = vs[b][:, as_t] - vb[b][:, ab_t]
        u_new = ub[b].clone()
        u_new[:, ab_t] = ub[b][:, ab_t] + du @ UK @ UK.T
        k_new = rerot(u_new[None], (csb[0][b:b + 1], csb[1][b:b + 1]))[0]
        Kn[b][:, ab_t] = k_new[:, ab_t].to(Kn.dtype)
        Vn[b][:, ab_t] = (vb[b][:, ab_t] + dv @ UV @ UV.T).to(Vn.dtype)
    K2 = [Kn if j == LAYER else K[j] for j in range(len(K))]
    V2 = [Vn if j == LAYER else V[j] for j in range(len(V))]
    return M.make(pb, K2, V2, pb.pad_masks)


def targeted_prefix(M, fr, idx, T, c, arm):
    """-> (base prefix, purified prefix) for arm T / R / TA."""
    S = subspaces()
    a, b = partners_of(T, c)
    src = a if arm == "TA" else b
    UK, UV = (S["RK"], S["RV"]) if arm == "R" else (S["UK"], S["UV"])
    pb, ib = prefix_rows(M, fr, idx, T[c]["text"])
    ps, is_ = prefix_rows(M, fr, idx, T[src]["text"])
    return pb, targeted(M, pb, ib, ps, is_, UK, UV)


def fit():
    t0 = time.time()
    M = Pi05()
    T = tasks_pp()
    Z = np.load(os.path.join(OUT, "pp_f1_frames.npz"))
    fr = {k: Z[k] for k in Z.files}
    ENT = list(Z["ent"])
    n = len(fr["phase"])
    d = None
    C = {}                                                              # (which, split) -> d x d
    nvec = 0
    with torch.no_grad():
        for ci, cj in PAIRS:
            obj = T[ci]["obj"]
            assert T[cj]["obj"] == obj and T[ci]["tgt"] != T[cj]["tgt"]
            idx = np.flatnonzero((fr["phase"] == 0) | ((fr["phase"] == 1) & (fr["obj"] == ENT.index(obj))))
            for chunk in np.array_split(idx, max(1, len(idx) // 10)):
                pi, ii = prefix_rows(M, fr, chunk, T[ci]["text"])
                pj, ij = prefix_rows(M, fr, chunk, T[cj]["text"])
                ui, vi = layer_uv(M, pi)
                uj, vj = layer_uv(M, pj)
                for b, f in enumerate(chunk):
                    ab, as_ = align(ii[b], ij[b], pi.base, pi.pad_masks[b].tolist(), pj.pad_masks[b].tolist())
                    ab_t, as_t = torch.tensor(ab, device=ui.device), torch.tensor(as_, device=ui.device)
                    du = (uj[b][:, as_t] - ui[b][:, ab_t]).reshape(-1, ui.shape[-1])
                    dv = (vj[b][:, as_t] - vi[b][:, ab_t]).reshape(-1, vi.shape[-1])
                    d = ui.shape[-1]
                    for which, x in (("K", du), ("V", dv)):
                        for split in ("all", "even" if f % 2 == 0 else "odd"):
                            C[(which, split)] = C.get((which, split), torch.zeros(d, d, device=x.device, dtype=torch.float64)) + (x.T.double() @ x.double())
                    nvec += du.shape[0]
            print(f"[pp_f3t] fit {ci}-{cj}: {len(idx)} frames ({time.time() - t0:.0f}s)", flush=True)
    out, stats = {}, {}
    for which in ("K", "V"):
        ev, vec = torch.linalg.eigh(C[(which, "all")])
        order = torch.argsort(ev, descending=True)
        ev, vec = ev[order], vec[:, order]
        U = vec[:, :R].float()
        g = torch.Generator().manual_seed(0 if which == "K" else 1)
        Rm, _ = torch.linalg.qr(torch.randn(d, R, generator=g, dtype=torch.float64))
        Rm = Rm.float().to(U.device)
        ev_e, vec_e = torch.linalg.eigh(C[(which, "even")])
        Ue = vec_e[:, torch.argsort(ev_e, descending=True)[:R]].float()
        Co = C[(which, "odd")].float()
        share = lambda W, Cm: float(torch.trace(W.T @ Cm @ W) / torch.trace(Cm))      # noqa: E731
        stats[which] = dict(dim=d, rank=R, energy_top=share(U, C[(which, "all")].float()), energy_random=share(Rm, C[(which, "all")].float()),
                            heldout_odd_from_even=share(Ue, Co), heldout_random=share(Rm, Co),
                            spectrum_top32=(ev[:32] / ev.sum()).tolist())
        out["U" + which], out["R" + which] = U.cpu().numpy(), Rm.cpu().numpy()
    np.savez(SUBFILE, **out)
    with open(os.path.join(OUT, "pp_f3t_subspace.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(layer=LAYER, rank=R, pairs=PAIRS, n_contrast_rows=nvec, n_frames=n, stats=stats, secs=time.time() - t0), fh, indent=1)
    print("[pp_f3t] subspace", json.dumps({w: {k: v for k, v in s.items() if k != "spectrum_top32"} for w, s in stats.items()}), flush=True)


def open_loop():
    t0 = time.time()
    M = Pi05()
    T = tasks_pp()
    Z = np.load(os.path.join(OUT, "pp_f1_frames.npz"))
    fr = {k: Z[k] for k in Z.files}
    ENT = list(Z["ent"])
    H = min(10, M.bk.H)

    def cosines(pre, idx, seeds, ref_ent):
        cs = []
        for s in seeds:
            rows = sample(M, pre, s).reshape(len(idx), -1)
            tr = np.asarray(M.bk.unnormalize(rows))[:, :H, :3].sum(1)
            to = fr["pos"][idx, ref_ent] - fr["ee_pos"][idx]
            cs.append((tr * to).sum(1) / (np.linalg.norm(tr, axis=1) * np.linalg.norm(to, axis=1) + 1e-9))
        return np.mean(cs, 0).tolist()

    res = {}
    with torch.no_grad():
        for c in NEW:
            for ph, lay in (("place", 0), ("grasp", 0), ("grasp", 1)):
                if ph == "grasp":
                    idx = np.flatnonzero((fr["phase"] == 0) & (fr["layout"] == lay))
                    ref = ENT.index(T[c]["obj"])
                else:
                    idx = np.flatnonzero((fr["phase"] == 1) & (fr["obj"] == ENT.index(T[c]["obj"])))
                    ref = ENT.index(T[c]["tgt"])
                seeds = [hseed("pp_f3_open", c, ph, lay, s) for s in SEEDS]           # pp_f3's open-loop seeds
                key = f"{c}|{ph}|{'familiar' if lay == 0 else 'new'}"
                r = {}
                for arm in ("T", "R", "TA"):
                    pb, pp = targeted_prefix(M, fr, idx, T, c, arm)
                    if "intact" not in r:
                        r["intact"] = cosines(pb, idx, seeds, ref)
                    r[arm] = cosines(pp, idx, seeds, ref)
                res[key] = dict(cos=r, partners=partners_of(T, c))
                print(f"[pp_f3t_open] {key} {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for ph in ("place|familiar", "grasp|familiar", "grasp|new"):
        ks = [k for k in res if k.endswith(ph)]
        summ[ph] = {arm: float(np.mean([np.mean(res[k]["cos"][arm]) for k in ks])) for arm in ("intact", "T", "R", "TA")}
    with open(os.path.join(OUT, "pp_f3t_open.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=res, summary=summ, secs=time.time() - t0), fh, indent=1)
    print("[pp_f3t_open] summary", json.dumps(summ), flush=True)


def smoke():
    """Code checks on 2 frames; writes nothing and prints only pass / fail."""
    M = Pi05()
    T = tasks_pp()
    Z = np.load(os.path.join(OUT, "pp_f1_frames.npz"))
    fr = {k: Z[k] for k in Z.files}
    idx = np.array([0, 25])
    with torch.no_grad():
        pb, ib = prefix_rows(M, fr, idx, T["N4"]["text"])
        ps, is_ = prefix_rows(M, fr, idx, T["K1"]["text"])
        K = M.kv(pb)[0][LAYER].float()
        cs = rope(M, pb)
        e_rt = float((rerot(derot(K, cs), cs) - K).abs().max() / K.abs().max())
        d = K.shape[-1]
        Z0 = torch.zeros(d, 0)
        q0 = targeted(M, pb, ib, ps, is_, Z0, Z0)
        e0 = max(float((a.float() - b.float()).abs().max()) for a, b in zip(M.kv(q0)[0] + M.kv(q0)[1], M.kv(pb)[0] + M.kv(pb)[1]))
        I = torch.eye(d)
        qs = targeted(M, pb, ib, pb, ib, I, I)
        es = max(float((a.float() - b.float()).abs().max()) for a, b in zip(M.kv(qs)[0] + M.kv(qs)[1], M.kv(pb)[0] + M.kv(pb)[1]))
        qI = targeted(M, pb, ib, ps, is_, I, I)
        ab, as_ = align(ib[0], is_[0], pb.base, pb.pad_masks[0].tolist(), ps.pad_masks[0].tolist())
        eV = float((M.kv(qI)[1][LAYER][0][:, ab].float() - M.kv(ps)[1][LAYER][0][:, as_].float()).abs().max())
        img = [x for x in ab if x < pb.base]                               # aligned (valid) image positions: same position ids
        eKimg = float((M.kv(qI)[0][LAYER][0][:, img].float() - M.kv(ps)[0][LAYER][0][:, img].float()).abs().max())
        n_text = len(ab) - sum(1 for x in ab if x < pb.base)
    print(f"[pp_f3t smoke] roundtrip rel err {e_rt:.2e}; U=0 max diff {e0:.2e}; source=base max diff {es:.2e}; "
          f"U=I values = source's {eV:.2e}; U=I image keys = source's {eKimg:.2e}; aligned text tokens {n_text} of {len(ib[0])}", flush=True)
    assert e_rt < 1e-4 and e0 < 0.05 and es < 0.05 and eV < 0.05 and eKimg < 0.05, "smoke failed"
    print("[pp_f3t smoke] ok", flush=True)


if __name__ == "__main__":
    {"fit": fit, "open": open_loop, "smoke": smoke}[sys.argv[1]]()

"""PLAN_PARTS F3 (docs/PARTS_RESULTS.md section 1, F3): purify the located component of a new-pairing run, and the open-loop read.

Purification of component k in a new-pairing run (o, t): both role messages within k's scope are replaced by messages of known
tasks.  A = the known task sharing the object (cheese K6, bottle K4, bowl K1); B = the known task sharing the target (plate K1,
stove K2, cabinet K3, bowl K6).  Object positions take A's values, target positions B's (aligned by the tokens the instructions
share: the object words are A's common prefix with the base, the target words B's common suffix).
  W     the object-word and target-word spans, at every expert-read layer (pi0.5: K, V; GR00T: vl)
  KV_l  pi0.5: layer l takes B's K, V at every position (realigned to the base's tokens), except the object words, which take A's
  Q_l   GR00T: the object and target words entering Eagle layer l take A's / B's residuals, propagated
  V     not available online (stop and report)
Open loop (descriptive): on the place frames of N1-N6 (k*_op purified) and on the grasp frames of N1-N6 in both layouts (k*_tg
purified), the cosine between the chunk's summed translation (unnormalised actions, first min(10, H) steps) and the direction from
the end effector to the target (place) / object (grasp), intact and purified; mean over 2 seeds.
-> results/flawed/pp_f3_open_<model>.json

    python experiments/pp_purify.py --model pi05|groot --comp-op KV_3 --comp-tg W
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from pp_f1_mix import OUT, SEEDS, Groot, Pi05, diff_span, hseed, sample, tasks_pp   # noqa: E402

A_OF = {"cream_cheese_1": "K6", "wine_bottle_1": "K4", "akita_black_bowl_1": "K1"}
B_OF = {"plate_1": "K1", "flat_stove_1_cook_region": "K2", "wooden_cabinet_1_top_side": "K3", "akita_black_bowl_1": "K6"}
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]


def partners_of(T, c):
    return A_OF[T[c]["obj"]], B_OF[T[c]["tgt"]]


def spans(pb, pa, pB):
    """-> (s_o, L_o), (s_t, L_t) in the base's token coordinates, and the shift of B's tokens after its object span."""
    s_t, L_t, L_ta = diff_span(pb.ids, pa.ids)            # base vs A: the target phrase differs
    s_o, L_o, L_oB = diff_span(pb.ids, pB.ids)            # base vs B: the object phrase differs
    assert s_o + L_o <= s_t, "object words must precede the target words"
    assert pa.ids[s_o:s_o + L_o] == pb.ids[s_o:s_o + L_o], "A's object words are not aligned with the base's"
    sh = len(pB.ids) - len(pb.ids)
    assert pB.ids[s_t + sh:s_t + L_t + sh] == pb.ids[s_t:s_t + L_t], "B's target words are not aligned with the base's"
    return (s_o, L_o), (s_t, L_t), sh, L_oB


def purify_pi05(M, pb, pa, pB, comp):
    (s_o, L_o), (s_t, L_t), sh, L_oB = spans(pb, pa, pB)
    o = pb.base
    Kb, Vb = M.kv(pb)
    Ka, Va = M.kv(pa)
    KB, VB = M.kv(pB)
    nb = len(pb.ids)
    if comp == "W":
        K, V = [], []
        for l in range(len(Kb)):
            k, v = Kb[l].clone(), Vb[l].clone()
            k[:, :, o + s_o:o + s_o + L_o], v[:, :, o + s_o:o + s_o + L_o] = Ka[l][:, :, o + s_o:o + s_o + L_o], Va[l][:, :, o + s_o:o + s_o + L_o]
            k[:, :, o + s_t:o + s_t + L_t] = KB[l][:, :, o + s_t + sh:o + s_t + L_t + sh]
            v[:, :, o + s_t:o + s_t + L_t] = VB[l][:, :, o + s_t + sh:o + s_t + L_t + sh]
            K.append(k)
            V.append(v)
        return M.make(pb, K, V, pb.pad_masks)
    if comp.startswith("KV_"):
        l = int(comp[3:])
        nB = len(pB.ids)

        def al(Xb, Xa, XB):
            return torch.cat([XB[:, :, :o + s_o], Xa[:, :, o + s_o:o + s_o + L_o], XB[:, :, o + s_o + L_oB:o + nB], Xb[:, :, o + nb:]], 2)
        K = [al(Kb[j], Ka[j], KB[j]) if j == l else Kb[j] for j in range(len(Kb))]
        V = [al(Vb[j], Va[j], VB[j]) if j == l else Vb[j] for j in range(len(Vb))]
        assert K[l].shape == Kb[l].shape
        return M.make(pb, K, V, pb.pad_masks)
    raise SystemExit(f"purification of {comp} is not defined online (stop and report)")


def purify_groot(M, o_b, pb, pa, pB, comp):
    (s_o, L_o), (s_t, L_t), sh, _ = spans(pb, pa, pB)
    if comp == "W":
        v = pb.kv["vl"].clone()
        v[:, s_o:s_o + L_o] = pa.kv["vl"][:, s_o:s_o + L_o]
        v[:, s_t:s_t + L_t] = pB.kv["vl"][:, s_t + sh:s_t + L_t + sh]
        return M.with_vl(pb, v)
    if comp.startswith("Q_"):
        l = int(comp[2:])
        st = {}

        def rec(name, a0, a1):
            def f(i, h, span):
                if i == l:
                    st[name] = h[:, a0:a1].clone()
                return None
            return f
        M.prefix(pa.o, hook=rec("a", s_o, s_o + L_o))
        M.prefix(pB.o, hook=rec("b", s_t + sh, s_t + L_t + sh))

        def rep(i, h, span):
            if i == l:
                h = h.clone()
                h[:, s_o:s_o + L_o] = st["a"].to(h.dtype)
                h[:, s_t:s_t + L_t] = st["b"].to(h.dtype)
                return h
            return None
        return M.prefix(o_b, hook=rep)
    raise SystemExit(f"purification of {comp} is not defined online (stop and report)")


def purified_prefix(M, frames_or_obs, idx, T, c, comp):
    """frames: the pp_f1 frames dict (idx rows) -> (base prefix, purified prefix)."""
    a, b = partners_of(T, c)
    o_b, o_a, o_B = M.obs(frames_or_obs, idx, T[c]["text"]), M.obs(frames_or_obs, idx, T[a]["text"]), M.obs(frames_or_obs, idx, T[b]["text"])
    pb, pa, pB = M.prefix(o_b), M.prefix(o_a), M.prefix(o_B)
    if M.name == "pi05":
        return pb, purify_pi05(M, pb, pa, pB, comp)
    return pb, purify_groot(M, o_b, pb, pa, pB, comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("pi05", "groot"))
    ap.add_argument("--comp-op", required=True)
    ap.add_argument("--comp-tg", required=True)
    a = ap.parse_args()
    t0 = time.time()
    M = Pi05() if a.model == "pi05" else Groot()
    T = tasks_pp()
    Z = np.load(os.path.join(OUT, "pp_f1_frames.npz"))
    fr = {k: Z[k] for k in Z.files}
    ENT = list(Z["ent"])
    H = min(10, M.bk.H)

    def cosines(pre, idx, seeds, ref_ent):
        cs = []
        for s in seeds:
            rows = sample(M, pre, s).reshape(len(idx), -1)
            act = M.bk.unnormalize(rows) if M.name == "groot" else M.bk.unnormalize(rows)
            tr = np.asarray(act)[:, :H, :3].sum(1)
            to = fr["pos"][idx, ref_ent] - fr["ee_pos"][idx]
            cs.append((tr * to).sum(1) / (np.linalg.norm(tr, axis=1) * np.linalg.norm(to, axis=1) + 1e-9))
        return np.mean(cs, 0).tolist()

    res = {}
    with torch.no_grad():
        for c in NEW:
            for ph, lay, comp in (("place", 0, a.comp_op), ("grasp", 0, a.comp_tg), ("grasp", 1, a.comp_tg)):
                if ph == "grasp":
                    idx = np.flatnonzero((fr["phase"] == 0) & (fr["layout"] == lay))
                    ref = ENT.index(T[c]["obj"])
                else:
                    idx = np.flatnonzero((fr["phase"] == 1) & (fr["obj"] == ENT.index(T[c]["obj"])))
                    ref = ENT.index(T[c]["tgt"])
                pb, pp = purified_prefix(M, fr, idx, T, c, comp)
                seeds = [hseed("pp_f3_open", c, ph, lay, s) for s in SEEDS]
                key = f"{c}|{ph}|{'familiar' if lay == 0 else 'new'}"
                res[key] = dict(component=comp, cos_intact=cosines(pb, idx, seeds, ref), cos_purified=cosines(pp, idx, seeds, ref),
                                partners=partners_of(T, c))
                print(f"[pp_f3_open] {a.model} {key} {comp} {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, f"pp_f3_open_{a.model}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(model=a.model, comp_op=a.comp_op, comp_tg=a.comp_tg, tasks=res, secs=time.time() - t0), fh)


if __name__ == "__main__":
    main()

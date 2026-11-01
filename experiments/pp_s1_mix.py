"""PLAN_PARTS S1 (docs/PARTS_RESULTS.md section 3): F1's role-mixing map (pp_f1_oracle.py's frames, pairs, d and components) on
architecture P, plus the exact swap tests.  CPU.

Components of P: per role net (obj, tgt): `<net>.in` its input (the slot embedding included), `<net>.h<j>` the output of its j-th
LoRA-carrying hidden Linear layer (j = 1..4).
Swap tests (on every F1 frame, 2 noise seeds): T_obj = max |v_o(target part a) - v_o(target part b)| over the object term's output
when only the TARGET atom changes; T_tgt = the same for the target term when only the OBJECT part's parameters are replaced by
another atom's (the held object's position, a physical input, unchanged).  Both are 0 by construction.
-> results/flawed/pp_s1_mix_<tag>.json

    python experiments/pp_s1_mix.py --tag full --ckpts pp_full_s0/P.npz+pp_full_s1/P.npz+pp_full_s2/P.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fa1_common as FC                                                     # noqa: E402
import pp_f1_oracle as O                                                    # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402


def comp_modules_p(m):
    out = {}
    for nm, net in (("obj", m.obj), ("tgt", m.tgt)):
        out[f"{nm}.in"] = (net.hidden[0], "pre")
        for j in range(4):
            out[f"{nm}.h{j + 1}"] = (net.hidden[j], "post")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpts", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    # P's LoRA layers take (x, atom): the pre-hook must keep the atom argument
    orig = O.sample

    def sample(m, b, seed, patch=None, steps=10):
        if patch is not None and patch[1] == "pre":
            mod, kind, bp = patch
            st = {}
            B = b["pos"].shape[0]
            g = torch.Generator().manual_seed(int(seed))
            x = torch.randn(B, m.H, 7, generator=g)
            for k in range(steps):
                t = torch.full((B,), k / steps)
                h1 = mod.register_forward_pre_hook(lambda md, ar: st.__setitem__("h", ar[0].clone()))
                try:
                    m.velocity(bp, x, t)
                finally:
                    h1.remove()
                h2 = mod.register_forward_pre_hook(lambda md, ar: (st["h"],) + tuple(ar[1:]))
                try:
                    v = m.velocity(b, x, t)
                finally:
                    h2.remove()
                x = x + v / steps
            return x.numpy()
        return orig(m, b, seed, patch, steps)
    O.sample = sample
    O.comp_modules = lambda m: comp_modules_p(m) if hasattr(m, "opart") else O.comp_modules(m)
    O.ARMS = {"P": [(c.split("/")[0], c.split("/")[1][:-4]) for c in a.ckpts.split("+")]}
    O.FF.FactoredFlow.load = staticmethod(lambda p: load_any(p))
    sys.argv = [sys.argv[0], "--out", os.path.join(FC.ROOT, "results", "flawed", f"pp_s1_tmp_{a.tag}")]
    O.main()
    res = json.load(open(os.path.join(FC.ROOT, "results", "flawed", f"pp_s1_tmp_{a.tag}", "pp_f1_oracle.json"), encoding="utf-8"))
    # exact swap tests
    Z = np.load(os.path.join(FC.ROOT, "results", "flawed", "pp_f1_frames.npz"))
    swaps = {}
    for c in a.ckpts.split("+"):
        m, _ = load_any(os.path.join(FC.DATA, "ckpt", c))
        n = len(Z["pos"])
        f = lambda v: torch.tensor(np.asarray(v, np.float32))                # noqa: E731
        b = dict(ee_pos=f(Z["ee_pos"]), ee_rot=f(Z["ee_rot"]), grip=f(Z["grip"]), prev=f(np.full(n, -1.0)), pos=f(Z["pos"]), yaw=f(Z["yaw"]),
                 drawer=f(Z["drawer"]), held=f(np.zeros(n)), off=f(np.zeros((n, 3))), zrest=f(np.zeros(n)), cyaw=f(np.zeros(n)), rot0=f(Z["ee_rot"]))
        ent = FC.ENT
        tobj, ttgt = 0.0, 0.0
        x = torch.randn(n, m.H, 7, generator=torch.Generator().manual_seed(0))
        t = torch.full((n,), 0.5)
        with torch.no_grad():
            for oe in ("akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1"):
                if int(m.opart[ent.index(oe)]) < 0:
                    continue
                outs = []
                for te in ("plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side"):
                    bb = dict(b, oi=torch.full((n,), ent.index(oe)), ti=torch.full((n,), ent.index(te)))
                    outs.append(m.velocity(bb, x, t))                     # held 0: the object term alone
                tobj = max(tobj, max(float((o - outs[0]).abs().max()) for o in outs))
            bb = dict(b, held=torch.ones(n), oi=torch.full((n,), ent.index("akita_black_bowl_1")), ti=torch.full((n,), ent.index("plate_1")))
            v0 = m.velocity(bb, x, t)
            net = m.obj                                                   # replace the bowl's object part by another atom's
            i0 = int(m.opart[ent.index("akita_black_bowl_1")])
            i1 = int(m.opart[ent.index("wine_bottle_1")])
            saved = [p.detach().clone() for p in net.part_parameters()]
            for p in net.part_parameters():
                p.data[i0] = p.data[i1]
            v1 = m.velocity(bb, x, t)
            for p, s in zip(net.part_parameters(), saved):
                p.data.copy_(s)
            ttgt = float((v1 - v0).abs().max())
        swaps[c] = dict(T_obj_target_swap=tobj, T_tgt_object_part_swap=ttgt)
    res["swap_tests"] = swaps
    json.dump(res, open(os.path.join(FC.ROOT, "results", "flawed", f"pp_s1_mix_{a.tag}.json"), "w", encoding="utf-8"))
    print("[pp_s1] swaps", json.dumps(swaps))


if __name__ == "__main__":
    main()

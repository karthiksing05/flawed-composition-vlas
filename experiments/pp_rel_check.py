"""PARTS_RESULTS 50 code check (smoke): (a) with --tnear off the edited PartsFlow gives the previous code's velocity bitwise on a
P_nsz checkpoint (the previous module is read from git at --old-rev); (b) on a P_rel checkpoint, the carry scale is 1 on every held
row with d0 <= r_t or d <= rho, and equals round 9's s_t = min(1, r_t / d0) at d = d0; rows with d0 > r_t and d > rho get a scale in
(s_t, 1]; (c) the object term does not depend on the flag; (d) the fitted release radii lie inside the carry radii.

    python experiments/pp_rel_check.py --old-rev 28015f14 --ckpt pp_full_nsz_s10/P.npz --rel pp_smk_relf_s10/P.npz
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as PT                                                     # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402
from pp_nsz_check import old_module                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-rev", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rel", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"))
    j = torch.randperm(len(X))[:4096]
    bb = {k: v[j] for k, v in b.items()}
    x, t = torch.randn(len(j), 8, 7), torch.rand(len(j))
    old = old_module(a.old_rev)
    p = os.path.join(FC.DATA, "ckpt", a.ckpt)
    mn, _ = PF.PartsFlow.load(p)
    mo, _ = old.PartsFlow.load(p)
    with torch.no_grad():
        dv = (mn.velocity(bb, x, t) - mo.velocity(bb, x, t)).abs().max().item()
    print("[rel_check] (a) tnear off vs old code, max |dv| =", dv, "bitwise" if dv == 0.0 else "DIFFERS", flush=True)
    m, _ = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", a.rel))
    print("[rel_check] tnear", m.tnear, "tscale", m.tscale, "rrel_p", [round(float(v), 4) for v in m.rrel_p],
          "rtclip_p", [round(float(v), 4) for v in m.rtclip_p], flush=True)
    cover = [PF.TGT_ATOMS[i] for i in range(len(PF.TGT_ATOMS)) if 0 < float(m.rrel_p[i]) and float(m.rrel_p[i]) >= float(m.rtclip_p[i])]
    print("[rel_check] (d) parts whose release region covers the carry support (scale 1 on all their rows):", cover, flush=True)
    # far carries: synthesise d0 beyond r_t on held rows by scaling the onset distance
    hd = bb["held"] > 0.5
    q = {k: v[hd] for k, v in bb.items()}
    n = len(q["oi"])
    idx = torch.arange(n)
    d = torch.linalg.norm((q["pos"][idx, q["oi"]] - q["pos"][idx, q["ti"]])[:, :2], dim=-1)
    r = m.rtclip_p[m.tpart[q["ti"]]]
    rho = m.rrel_p[m.tpart[q["ti"]]]
    for nm, d0 in (("demo d0", q["d0t"]), ("d0 = 2.5 x max(d, r_t)", 2.5 * torch.maximum(d, r))):
        qq = dict(q, d0t=d0)
        s = m._tscale(qq)
        st = torch.clamp(r / d0.clamp_min(1e-9), max=1.0)
        near = (d0 <= r) | (d <= rho) | (rho >= r)
        far = ~near & (d <= d0)                                        # beyond the onset (d > d0) the scale falls below s_t, as phi/d decreases
        print(f"[rel_check] (b) {nm}: rows {n}; scale 1 where d0 <= r_t or d <= rho:",
              bool((s[near] == 1.0).all()), f"(n {int(near.sum())}); far rows {int(far.sum())}, scale in [s_t, 1]:",
              bool(((s[far] >= st[far] - 1e-6) & (s[far] <= 1.0)).all()), flush=True)
    at = dict(q, d0t=d)                                                    # d = d0: the onset itself
    s_at = m._tscale(at)
    st_at = torch.clamp(r / d.clamp_min(1e-9), max=1.0)
    v = rho < r
    print("[rel_check] (b) at d = d0 (parts with rho < r_t): max |scale - round 9's s_t| =", (s_at[v] - st_at[v]).abs().max().item(),
          flush=True)
    te = PF.temb(t)
    with torch.no_grad():
        v1 = m._vobj(bb, x, te)
        m.tnear = False
        v0 = m._vobj(bb, x, te)
        m.tnear = True
    print("[rel_check] (c) object term, tnear on vs off: max |dv| =", (v1 - v0).abs().max().item(), flush=True)


if __name__ == "__main__":
    main()

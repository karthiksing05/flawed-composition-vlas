"""PARTS_RESULTS 41 code check (smoke, CPU or GPU): (a) with --rtransport off the edited PartsFlow gives the pre-round-13 velocity
bitwise on a saved checkpoint (the old module file is passed as an argument); (b) on a P_rtr checkpoint the part is off exactly on
rows beyond its start radius, the backbone-only path ignores the part (changing the part's parameters leaves it unchanged), and
sample() runs.

    python experiments/pp_rtr_check.py --old /tmp/parts_flow_old.py --ckpt pp_full_imd_s10/P.npz --rtr pp_smk_rtr_s10/P.npz
"""
import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as PT                                                     # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rtr", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"))
    j = torch.randperm(len(X))[:2048]
    bb = {k: v[j] for k, v in b.items()}
    x, t = torch.randn(len(j), 8, 7), torch.rand(len(j))
    spec = importlib.util.spec_from_file_location("lifelong_ood.policy.parts_flow_old", a.old)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    p = os.path.join(FC.DATA, "ckpt", a.ckpt)
    mn, _ = PF.PartsFlow.load(p)
    mo, _ = old.PartsFlow.load(p)
    with torch.no_grad():
        dv = (mn.velocity(bb, x, t) - mo.velocity(bb, x, t)).abs().max().item()
    print("[rtr_check] (a) default off vs old code, max |dv| =", dv, "bitwise" if dv == 0.0 else "DIFFERS", flush=True)
    m, meta = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", a.rtr))
    print("[rtr_check] rtransport", m.rtransport, "rs_p", m.rs_p.tolist(), "rrole", float(m.rrole), flush=True)
    idx = torch.arange(len(j))
    oo = bb["pos"][idx, bb["oi"]]
    on = m.part_on(bb, oo)
    dxy = torch.linalg.norm((bb["ee_pos"] - oo)[:, :2], dim=-1)
    ok = bool((on == (dxy <= m.rs_p[m.opart[bb["oi"]]])).all())
    print("[rtr_check] (b) part on iff dxy <= r_s:", ok, "share on (not-held rows):", float(on[bb["held"] < 0.5].float().mean()), flush=True)
    with torch.no_grad():
        te = PF.temb(t)
        v1 = m._vobj(bb, x, te, part_off=True)
        for ly in m.obj.hidden:
            ly.B.add_(1.0)
        m.obj.emb.add_(1.0)
        v2 = m._vobj(bb, x, te, part_off=True)
        v3 = m._vobj(bb, x, te)
    print("[rtr_check] (b) backbone-only path ignores the part: max |dv| =", (v1 - v2).abs().max().item(), flush=True)
    off = ~on
    print("[rtr_check] (b) switched path equals backbone-only on part-off rows: max |dv| =",
          (v3[off] - v2[off]).abs().max().item() if bool(off.any()) else None, flush=True)
    m2, _ = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", a.rtr))
    s = m2.sample({k: v[:4] for k, v in bb.items()})
    print("[rtr_check] sample ok", tuple(s.shape), bool(torch.isfinite(s).all()), flush=True)


if __name__ == "__main__":
    main()

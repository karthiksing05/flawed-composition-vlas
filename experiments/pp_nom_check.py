"""PARTS_RESULTS 44 code check (smoke): (a) with --bbnom off the edited PartsFlow gives the previous code's velocity bitwise on a
P_rtr checkpoint (the old module file is passed as an argument); (b) on a P_nom checkpoint, changing m leaves the backbone-only
path and every part-off row of the switched field unchanged, and changes part-on rows.

    python experiments/pp_nom_check.py --old /tmp/parts_flow_old.py --ckpt pp_full_rtr_s10/P.npz --nom pp_smk_nomf_s10/P.npz
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
    ap.add_argument("--nom", required=True)
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
    print("[nom_check] (a) bbnom off vs old code, max |dv| =", dv, "bitwise" if dv == 0.0 else "DIFFERS", flush=True)
    m, _ = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", a.nom))
    print("[nom_check] bbnom", m.bbnom, "rtransport", m.rtransport, "nmode", m.nmode, flush=True)
    on = m.part_on(bb, bb["pos"][torch.arange(len(j)), bb["oi"]])
    te = PF.temb(t)
    with torch.no_grad():
        outs = []
        for v in range(m.nmode):
            q = dict(bb, m=torch.full((len(j),), v))
            outs.append((m._vobj(q, x, te, part_off=True), m._vobj(q, x, te)))
    d_bb = max((o[0] - outs[0][0]).abs().max().item() for o in outs)
    d_off = max((o[1][~on] - outs[0][1][~on]).abs().max().item() for o in outs) if bool((~on).any()) else None
    d_on = max((o[1][on] - outs[0][1][on]).abs().max().item() for o in outs)
    print("[nom_check] (b) m change: backbone-only path max |dv| =", d_bb, "; part-off rows =", d_off, "; part-on rows =", d_on, flush=True)


if __name__ == "__main__":
    main()

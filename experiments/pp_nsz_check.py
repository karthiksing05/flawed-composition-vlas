"""PARTS_RESULTS 47 code check (smoke): (a) with --bbnosize off the edited PartsFlow gives the previous code's velocity bitwise on a
P_nom checkpoint (the previous module is read from git at --old-rev); (b) on a P_nsz checkpoint, changing every object's hand-set
size leaves the backbone-only path and every part-off row of the switched field unchanged, and changes part-on rows; (c) with m
fixed, changing the target atom leaves the object term unchanged.

    python experiments/pp_nsz_check.py --old-rev 26f11b06 --ckpt pp_full_nom_s10/P.npz --nsz pp_smk_nszf_s10/P.npz
"""
import argparse
import importlib.util
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as PT                                                     # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402


def old_module(rev):
    src = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:lifelong_ood/policy/parts_flow.py"], check=True, capture_output=True,
                         text=True).stdout
    spec = importlib.util.spec_from_loader("lifelong_ood.policy.parts_flow_old", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__package__"] = "lifelong_ood.policy"
    exec(compile(src, f"{rev}:parts_flow.py", "exec"), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-rev", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nsz", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"))
    j = torch.randperm(len(X))[:2048]
    bb = {k: v[j] for k, v in b.items()}
    x, t = torch.randn(len(j), 8, 7), torch.rand(len(j))
    old = old_module(a.old_rev)
    p = os.path.join(FC.DATA, "ckpt", a.ckpt)
    mn, _ = PF.PartsFlow.load(p)
    mo, _ = old.PartsFlow.load(p)
    with torch.no_grad():
        dv = (mn.velocity(bb, x, t) - mo.velocity(bb, x, t)).abs().max().item()
    print("[nsz_check] (a) bbnosize off vs old code, max |dv| =", dv, "bitwise" if dv == 0.0 else "DIFFERS", flush=True)
    m, _ = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", a.nsz))
    print("[nsz_check] bbnosize", m.bbnosize, "bbnom", m.bbnom, "rtransport", m.rtransport, "nmode", m.nmode, flush=True)
    on = m.part_on(bb, bb["pos"][torch.arange(len(j)), bb["oi"]])
    te = PF.temb(t)
    with torch.no_grad():
        ref = (m._vobj(bb, x, te, part_off=True), m._vobj(bb, x, te))
        s0 = m.size.clone()
        m.size = s0 * 1.7 + 0.01
        new = (m._vobj(bb, x, te, part_off=True), m._vobj(bb, x, te))
        m.size = s0
    d_bb = (new[0] - ref[0]).abs().max().item()
    d_off = (new[1][~on] - ref[1][~on]).abs().max().item() if bool((~on).any()) else None
    d_on = (new[1][on] - ref[1][on]).abs().max().item()
    print("[nsz_check] (b) size change: backbone-only path max |dv| =", d_bb, "; part-off rows =", d_off, "(n", int((~on).sum()),
          "); part-on rows =", d_on, "(n", int(on.sum()), ")", flush=True)
    with torch.no_grad():
        for v in range(max(m.nmode, 1)):
            q = dict(bb, m=torch.full((len(j),), v))
            q2 = dict(q, ti=bb["ti"].roll(1))
            d = (m._vobj(q, x, te) - m._vobj(q2, x, te)).abs().max().item()
            print(f"[nsz_check] (c) m fixed at {v}: target atom changed, object term max |dv| =", d, flush=True)


if __name__ == "__main__":
    main()

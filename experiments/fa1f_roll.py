"""FLAWED_COMPOSITION prototype A, stages 1f onward (sections 5.11+): closed-loop rollouts of one factored arm, in process on CPU (LIBERO
venv).  The episode is stage 1d's, unchanged (fa1d_roll.episode).  Differs from fa1e_roll.py only in taking the output directory and
any arm of factored_flow.CANON_IN, so that one job can run all of a model's tasks on one node (5.9: cross-CPU rounding broke the
grasp-identity equality at 1e-6 when a model's halves ran on different CPU types).

-> results/flawed/<outdir>/<arm>_s<seed>_main_<tid>.json

    python experiments/fa1f_roll.py --arm Ae_nosize --ckpt f_s0 --outdir a1f_roll --tids 0+2+4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fa1_common as FC                                                     # noqa: E402
from fa1d_roll import episode                                               # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True, help="dir under DATA/ckpt, e.g. f_s0")
    ap.add_argument("--outdir", required=True, help="dir under results/flawed, e.g. a1f_roll")
    ap.add_argument("--tids", required=True)
    a = ap.parse_args()
    assert a.arm in FF.CANON_IN, a.arm
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, a.outdir)
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.arm}_s{meta['seed']}_main_{tid}.json")
        if os.path.exists(out):
            print(f"[fa1f] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
            print(f"[fa1f] {a.arm} s{meta['seed']} task {tid} {t['key']} trial {b['trial']}: success={e['success']} "
                  f"bound={e['bound']} held_step={e['held_step']} ({e['secs']:.0f}s)", flush=True)
        res = dict(arm=a.arm, ckpt=a.ckpt, ckpt_meta=meta, phase="main", host=os.uname().nodename,
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION prototype A, stage 1e (section 5.9): closed-loop rollouts of one stage-1e arm (factored_flow.ARMS_E), in process on
CPU (LIBERO venv).  The episode is stage 1d's, unchanged (fa1d_roll.episode: stage 1b's protocol, the held machine and the canonical
tracker stepped after every simulator step, the onset and min_dxy_held logged).

-> results/flawed/a1e_roll/<arm>_s<seed>_main_<tid>.json

    python experiments/fa1e_roll.py --arm Ae --ckpt e_s0 --tids 0+2+4
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
    ap.add_argument("--arm", required=True, choices=FF.ARMS_E)
    ap.add_argument("--ckpt", required=True, help="dir under DATA/ckpt, e.g. e_s0")
    ap.add_argument("--tids", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, "a1e_roll")
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.arm}_s{meta['seed']}_main_{tid}.json")
        if os.path.exists(out):
            print(f"[fa1e] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
            print(f"[fa1e] {a.arm} s{meta['seed']} task {tid} {t['key']} trial {b['trial']}: success={e['success']} "
                  f"bound={e['bound']} held_step={e['held_step']} ({e['secs']:.0f}s)", flush=True)
        res = dict(arm=a.arm, ckpt=a.ckpt, ckpt_meta=meta, phase="main",
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

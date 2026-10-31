"""FLAWED_COMPOSITION prototype A, stage 1g (section 5.12): closed-loop rollouts, as fa1f_roll.py (stage 1d's episode unchanged, one job
per model and layout so that every task of a layout runs on one node), with --name: the name the rollout files carry, so that the
baselines retrained on the generated data (Hb / Rb architectures, tag g) are read as Hg / Rg beside stage 1b's Hb / Rb.

-> results/flawed/<outdir>/<name>_s<seed>_main_<tid>.json

    python experiments/fa1g_roll.py --arm Hb --name Hg --ckpt g_s0 --outdir a1g_roll --tids 1+3+5
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True, help="dir under DATA/ckpt, e.g. g_s0")
    ap.add_argument("--outdir", required=True, help="dir under results/flawed, e.g. a1g_roll")
    ap.add_argument("--tids", required=True)
    a = ap.parse_args()
    assert a.arm in FF.CANON_IN or a.arm in FF.BASE, a.arm
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, a.outdir)
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.name}_s{meta['seed']}_main_{tid}.json")
        if os.path.exists(out):
            print(f"[fa1g] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
            print(f"[fa1g] {a.name} s{meta['seed']} task {tid} {t['key']} trial {b['trial']}: success={e['success']} "
                  f"bound={e['bound']} held_step={e['held_step']} ({e['secs']:.0f}s)", flush=True)
        res = dict(arm=a.arm, name=a.name, ckpt=a.ckpt, ckpt_meta=meta, phase="main", host=os.uname().nodename,
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

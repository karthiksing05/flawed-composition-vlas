"""PLAN_PARTS Layer 2 closed-loop rollouts (docs/PARTS_RESULTS.md section 3): prototype A's episode (fa1h_roll.episode: oracle state,
contact gate, rot0, N2's start states 4-23 and seeds, 300 steps, replan every 5), unchanged, for a model loaded from any npz under
DATA/ckpt: architecture P (lifelong_ood/policy/parts_flow.py), a FactoredFlow arm, or an S4 baseline (pp_baselines.py).  LIBERO venv.

-> results/flawed/<outdir>/<name>_s<seed>_main_<tid>.json   (fa1x_read's naming)

    python experiments/pp_p_roll.py --ckpt pp_full_s0/P.npz --name P2 --outdir pp_s2_roll --tids 0+2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
from fa1h_roll import episode                                               # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


def load_any(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    arm = meta.get("arm")
    if arm == "P":
        from lifelong_ood.policy.parts_flow import PartsFlow
        return PartsFlow.load(path)
    if meta.get("baseline"):
        import pp_baselines as BL
        return BL.load(path)
    from lifelong_ood.policy import factored_flow as FF
    return FF.FactoredFlow.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tids", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    od = os.path.join(OUT, a.outdir)
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.name}_s{meta['seed']}_main_{tid}.json")
        if os.path.exists(out):
            print(f"[pp_roll] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
        print(f"[pp_roll] {a.name} s{meta['seed']} task {tid} {t['key']}: success {np.mean([e['success'] for e in eps]):.2f} "
              f"errors {sum(e['error'] is not None for e in eps)}", flush=True)
        res = dict(ckpt=a.ckpt, name=a.name, ckpt_meta=meta, phase="main", host=os.uname().nodename,
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

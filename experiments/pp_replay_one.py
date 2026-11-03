"""PLAN_PARTS round 10 check (descriptive): re-run ONE dev episode (fa1h_roll.episode unchanged, N2's state and seed) for several
checkpoints in one process on one host, --repeats times each, and print the hold step, success and a hash of the executed
first-call chunks.  Separates a model difference from host-to-host simulator / CPU float differences.  LIBERO venv, CPU.

    python experiments/pp_replay_one.py --ckpts pp_full_pdc_s10/P.npz+pp_full_tdc_s10/P.npz --tid 7 --trial 6 --repeats 2
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1h_roll as RH                                                      # noqa: E402
from pp_close_diag import Wrap                                              # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(1)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    t = n2["tasks"][a.tid]
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{a.tid}.npy"))
    for c in a.ckpts.split("+"):
        m, _ = load_any(os.path.join(FC.DATA, "ckpt", c))
        for r in range(a.repeats):
            w = Wrap(m)
            e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[a.trial], t["language"], br[a.trial]["seed_base"])
            pre = [x for bb, x in w.calls if float(bb["held"][0]) < 0.5]
            h = hashlib.sha256(b"".join(x.numpy().tobytes() for x in pre)).hexdigest()[:16]
            print("[replay]", os.uname().nodename, c, "rep", r, json.dumps(dict(held_step=e["held_step"], success=e["success"], steps=e["steps"],
                  pre_hold_calls=len(pre), pre_hold_chunks_sha=h)), flush=True)


if __name__ == "__main__":
    main()

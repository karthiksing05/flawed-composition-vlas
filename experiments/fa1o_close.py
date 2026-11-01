"""FLAWED_COMPOSITION prototype A, stage 1n DIAGNOSTIC (section 5.27; descriptive only): what happens to a cheese grasp AFTER the
first close command.  5.25 read the grasp at the first close step only; in An (5.26) missed and successful cheese grasps close at
nearly the same height (0.9 vs 0.4 cm) with no finger touching in either, so the miss is decided later, while the fingers close.
Each registered episode of the chosen tasks is re-run with fa1k_trace's protocol (same model, state, noise seeds; fresh re-seeded
env); for the W steps from the first close command it records, per episode:
  closed_frac     fraction of the window with the gripper commanded closed
  reopen          first step (relative) the gripper is commanded open again, or None
  ee_dz, ee_dxy   end-effector motion over the window, object frame (cm)
  obj_disp        object displacement over the window (cm; pushed or knocked)
  touch_any, touch_both   any / both fingers touching the object at any step of the window
  z_close, z_end  end-effector height above the object (object frame, cm) at the first close and at the window's end
CPU (LIBERO venv).  -> results/flawed/a1o_close/<name>_s<seed>_<tid>.json

    python experiments/fa1o_close.py --arm Aq --name An --ckpt n_s18 --tids 10+11
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1k_trace as KT                                                     # noqa: E402

W = 20


def window(e):
    tr = e["trace"]
    cmd, H = np.array(tr["cmd"]), np.array(tr["held"], bool)
    O, EE, Y = np.array(tr["obj"]), np.array(tr["ee"]), np.array(tr["yaw"])
    lt, rt = np.array(tr["lt"], bool), np.array(tr["rt"], bool)
    closed = np.nonzero(cmd > 0)[0]
    s = dict(success=e["success"], held_any=bool(H.any()), error=e["error"])
    if not len(closed):
        s["cls"] = "never_closed"
        return s
    i = int(closed[0])
    j = min(i + W, len(cmd))
    s["cls"] = "success" if e["success"] else ("miss" if not H.any() else "other")
    s["first_close"] = i
    c = cmd[i:j]
    s["closed_frac"] = float(np.mean(c > 0))
    ro = np.nonzero(c < 0)[0]
    s["reopen"] = int(ro[0]) if len(ro) else None
    a, b = KT.objframe(EE[i] - O[i], float(Y[i])), KT.objframe(EE[j - 1] - O[j - 1], float(Y[j - 1]))
    s["z_close"], s["z_end"] = 100 * a[2], 100 * b[2]
    d = KT.objframe(EE[j - 1] - EE[i], float(Y[i]))
    s["ee_dz"], s["ee_dxy"] = 100 * d[2], 100 * float(np.hypot(d[0], d[1]))
    s["obj_disp"] = 100 * float(np.linalg.norm(O[j - 1] - O[i]))
    s["touch_any"] = bool((lt[i:j] | rt[i:j]).any())
    s["touch_both"] = bool((lt[i:j] & rt[i:j]).any())
    hh = np.nonzero(H[i:j])[0]
    s["hold_step"] = int(hh[0]) if len(hh) else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tids", default="10+11")
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = KT.FF.FactoredFlow.load(os.path.join(KT.FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(KT.OUT, "a1o_close")
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(KT.OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(KT.OUT, "n2_branches_main.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        t = n2["tasks"][tid]
        S = np.load(os.path.join(KT.OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps, t0 = [], time.time()
        for b in br:
            e = KT.episode(model, os.path.join(KT.ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            s = window(e)
            s["trial"] = int(b["trial"])
            eps.append(s)
            print(f"[fa1o] {a.name} s{meta['seed']} {t['key']} trial {b['trial']}: "
                  f"{json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})} ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(od, f"{a.name}_s{meta['seed']}_{tid}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(arm=a.arm, name=a.name, ckpt=a.ckpt, tid=tid, key=t["key"], layout=t["layout"], object=t["object"],
                           episodes=eps), fh, indent=0)


if __name__ == "__main__":
    main()

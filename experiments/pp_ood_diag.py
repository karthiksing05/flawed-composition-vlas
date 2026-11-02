"""PLAN_PARTS step 2 diagnostic (descriptive, dev data only; docs/PARTS_RESULTS.md section 20): WHICH input of the object term
leaves the training distribution when the object starts away from its familiar spot.

For each checkpoint (P, P_af, P_af2: their own feature maps) and each known pick-and-place task K1-K6:
- training cloud: the object term's input features (PartsFlow._obj_in, exactly as the model computes them) on every NOT-held
  demo row of that task's object (the rows the object term is trained on), plus the Φ gate input (held) and the start offset;
- probe: the same features at the episode start (after the 10 settle steps) of N2's dev states 4-23, familiar and new layout,
  computed from the simulator scene (fa1_common.scene -> factored_flow.batch_from_scene, the rollout's own path);
- per feature group (ee position in the frame; ee rotation in the frame; gripper; last command; slot yaw; slot size):
  the probe's distance outside the training range (0 inside), and its standardized distance to the nearest training row,
  both per group and for the whole input.
LIBERO venv, CPU.  -> results/flawed/pp_ood_diag.json

    python experiments/pp_ood_diag.py --ckpts pp_full_r32_s10/P.npz+pp_full_af_s10/P.npz+pp_full_af2_s10/P.npz
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
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402
from lifelong_ood.policy.factored_flow import rz                            # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
GROUPS = {"ee_pos": slice(0, 3), "ee_rot": slice(3, 9), "grip": slice(9, 11), "prev": slice(11, 12), "slot_pos0": slice(12, 15),
          "slot_yaw": slice(15, 17), "slot_size": slice(17, 19), "slot_aux": slice(19, 20)}
DEV = list(range(4, 24))


def feats(m, b):
    idx = torch.arange(len(b["oi"]))
    Qo, oo, yo = rz(b["yaw"][idx, b["oi"]]), b["pos"][idx, b["oi"]], b["yaw"][idx, b["oi"]]
    if m.aframe == 2:
        ay = torch.atan2(oo[:, 1] - m.home[1], oo[:, 0] - m.home[0])
        return m._obj_in(b, rz(ay), oo, ay)
    if m.aframe:
        Qa = rz(torch.atan2(oo[:, 1] - m.home[1], oo[:, 0] - m.home[0]))
        return m._obj_in(b, Qo, oo, yo, Qp=Qa)
    return m._obj_in(b, Qo, oo, yo)


def demo_batch(task):
    import pp_p_train as PT
    b, _, _, _, _, _ = PT.rows(torch.device("cpu"), [task])
    keep = b["held"] < 0.5
    return {k: v[keep] for k, v in b.items()}


def probes(task):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    out = {}
    for lay in ("familiar", "new"):
        t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == lay)
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
        oi, ti = FC.ENT.index(t["object"]), FC.ENT.index(t["target"])
        bs = []
        for k in DEV:
            env = FC.fresh_env(os.path.join(ROOT, t["bddl"]))
            env.reset()
            obs = env.set_init_state(S[k])
            dj = FC.drawer_joint(env)
            for _ in range(10):
                obs, _, _, _ = env.step(FC.DUMMY)
            bs.append(FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, -1.0))
            env.close()
        out[lay] = {k: torch.cat([b[k] for b in bs]) for k in bs[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    res = {}
    P = {t: probes(t) for t in ("K1", "K2", "K3", "K4", "K5", "K6")}
    for c in a.ckpts.split("+"):
        m, meta = load_any(os.path.join(FC.DATA, "ckpt", c))
        res[c] = {}
        with torch.no_grad():
            for task in P:
                Ft = feats(m, demo_batch(task)).numpy()
                mu, sd = Ft.mean(0), Ft.std(0) + 1e-3
                lo, hi = Ft.min(0), Ft.max(0)
                Z = (Ft - mu) / sd
                r = {}
                for lay, b in P[task].items():
                    Fp = feats(m, b).numpy()
                    Zp = (Fp - mu) / sd
                    g = {}
                    for nm, sl in GROUPS.items():
                        outside = np.maximum(0, np.maximum(lo[sl] - Fp[:, sl], Fp[:, sl] - hi[sl])).max(1)
                        nn = np.sqrt(((Zp[:, None, sl] - Z[None, :, sl]) ** 2).sum(-1)).min(1)
                        g[nm] = dict(outside_median=float(np.median(outside)), nn_z_median=float(np.median(nn)))
                    nn_all = np.sqrt(((Zp[:, None] - Z[None]) ** 2).sum(-1)).min(1)
                    d0 = np.linalg.norm((b["ee_pos"] - b["pos"][torch.arange(len(b["oi"])), b["oi"]])[:, :2].numpy(), axis=1)
                    r[lay] = dict(groups=g, nn_z_all_median=float(np.median(nn_all)), start_dxy_median=float(np.median(d0)),
                                  start_dxy_range=[float(d0.min()), float(d0.max())])
                dd = np.linalg.norm((demo_batch(task)["ee_pos"] - demo_batch(task)["pos"][:, FC.ENT.index(FC.roles(
                    {"K1": "put the bowl on the plate", "K2": "put the bowl on the stove", "K3": "put the bowl on top of the cabinet",
                     "K4": "put the wine bottle on top of the cabinet", "K5": "put the wine bottle on the rack",
                     "K6": "put the cream cheese in the bowl"}[task])[0])])[:, :2].numpy(), axis=1)
                r["demo_dxy_range"] = [float(dd.min()), float(np.percentile(dd, 99)), float(dd.max())]
                res[c][task] = r
                print(f"[ood] {c} {task}", json.dumps({lay: (round(v['nn_z_all_median'], 2), round(v['start_dxy_median'], 3),
                      {k: round(x['nn_z_median'], 2) for k, x in v['groups'].items()}) for lay, v in r.items() if lay != 'demo_dxy_range'}),
                      "demo dxy", [round(x, 3) for x in r["demo_dxy_range"]], flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_ood_diag.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

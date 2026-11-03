"""PLAN_PARTS round diagnostics (descriptive, dev data only): in which frame is an object part's grasp rotation invariant?
The object term reads the hand's rotation in the object's BODY yaw frame (rz(object yaw)) and emits its rotation velocity there.
For each task's object, three candidate frames for that rotation: body (rz(object yaw)), approach (rz(bearing from the demos' home
end effector to the object), the frame P_af already uses for translation) and world (identity).  Per frame:
- demo cloud: the hand-rotation feature (frame^T ee_rot, first two columns, 6 numbers; the object term's own encoding) on the task's
  NOT-held demo rows whose action chunk's first gripper command closes (the close rows, pp_close_diag's cloud), and on all not-held
  rows; its spread (median distance to the cloud's mean) and the object's yaw spread over the demos (circular SD, degrees);
- probe: the same feature at the first close of a closed-loop episode of --ckpt (fa1h_roll.episode unchanged, model.sample wrapped
  as in pp_close_diag; an episode that never closes contributes its closest approach, "hover"); its raw distance to the nearest
  cloud row, per frame; plus the object's yaw at the probe.
LIBERO venv, CPU.  -> results/flawed/pp_yaw_<name>_<task>.json

    python experiments/pp_yaw_diag.py --ckpt pp_full_tdc_s10/P.npz --name tdc10 --task K4 --states 4+5+6
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
import fa1h_roll as RH                                                      # noqa: E402
from lifelong_ood.policy.factored_flow import rz                            # noqa: E402
from pp_close_diag import Wrap, dxy                                         # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
FRAMES = ("body", "approach", "world")


def rotfeat(m, b):
    """-> {frame: [N, 6]} hand-rotation features, and the object's yaw [N]."""
    idx = torch.arange(len(b["oi"]))
    oo, yo = b["pos"][idx, b["oi"]], b["yaw"][idx, b["oi"]]
    Q = dict(body=rz(yo), approach=rz(torch.atan2(oo[:, 1] - m.home[1], oo[:, 0] - m.home[0])),
             world=torch.eye(3).expand(len(yo), 3, 3))
    return {f: torch.einsum("bji,bjk->bik", Q[f], b["ee_rot"])[:, :, :2].reshape(-1, 6).numpy() for f in FRAMES}, yo.numpy()


def circ_sd_deg(a):
    R = np.abs(np.mean(np.exp(1j * np.asarray(a, np.float64))))
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def spread(F):
    return float(np.median(np.linalg.norm(F - F.mean(0), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--states", default="+".join(str(k) for k in range(4, 24)))
    ap.add_argument("--layouts", default="familiar+new")
    a = ap.parse_args()
    torch.set_num_threads(1)
    m, meta = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    import pp_p_train as PT
    b, X, _, _, _, _ = PT.rows(torch.device("cpu"), [a.task])
    nh = b["held"] < 0.5
    close = nh & (X[:, 0, 6] > 0)
    Fc, yc = rotfeat(m, {k: v[close] for k, v in b.items()})
    Fa, ya = rotfeat(m, {k: v[nh] for k, v in b.items()})
    res = dict(task=a.task, ckpt=a.ckpt, cloud=dict(n_close=int(close.sum()), n_not_held=int(nh.sum()),
                                                    object_yaw_circ_sd_deg=circ_sd_deg(yc),
                                                    object_yaw_range_deg=[float(np.degrees(yc.min())), float(np.degrees(yc.max()))],
                                                    spread_close={f: spread(Fc[f]) for f in FRAMES},
                                                    spread_not_held={f: spread(Fa[f]) for f in FRAMES}), episodes=[])
    print("[yaw] cloud", a.task, json.dumps(res["cloud"]), flush=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    for lay in a.layouts.split("+"):
        t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == lay)
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
        for k in [int(x) for x in a.states.split("+")]:
            w = Wrap(m)
            e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
            pre = [(bb, x) for bb, x in w.calls if float(bb["held"][0]) < 0.5]
            if not pre:
                continue
            j = next((i for i, (bb, x) in enumerate(pre) if float(x[0, 0, 6]) > 0), None)
            kind = "close" if j is not None else "hover"
            if j is None:
                j = int(np.argmin([dxy(bb)[0] for bb, _ in pre]))
            bb = pre[j][0]
            Fp, yp = rotfeat(m, bb)
            start, _ = rotfeat(m, pre[0][0])
            r = dict(layout=lay, state=k, success=bool(e["success"]), held=e["held_step"] is not None, kind=kind, call=j,
                     dxy=float(dxy(bb)[0]), object_yaw_deg=float(np.degrees(yp[0])),
                     nn_close={f: float(np.linalg.norm(Fc[f] - Fp[f][0], axis=1).min()) for f in FRAMES},
                     nn_start={f: float(np.linalg.norm(Fa[f] - start[f][0], axis=1).min()) for f in FRAMES})
            res["episodes"].append(r)
            print("[yaw]", a.name, a.task, json.dumps({x: (round(v, 3) if isinstance(v, float) else
                  ({q: round(u, 3) for q, u in v.items()} if isinstance(v, dict) else v)) for x, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(OUT, f"pp_yaw_{a.name}_{a.task}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

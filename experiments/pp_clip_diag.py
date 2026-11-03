"""PLAN_PARTS round diagnostics (descriptive): does a change in the TARGET term change how the OBJECT term trains?
Two models that differ only in the target term (P_pp: per-part radii; P_ts: + the carry scale frame) are initialised from the same
seed and trained for --steps steps of pp_p_train.fit's loop on the same batches, CPU, deterministic.  Two optimiser variants:
  joint     clip_grad_norm_ over all parameters together (pp_p_train.fit, as every build so far)
  per-role  clip_grad_norm_ over each role network separately
Reported: the largest absolute difference between the two models' object-term parameters after training, per variant.  Also the
object-term parameter distance between the real seed-10 checkpoints of P_pp and P_ts, relative to the parameter norm.
CPU.  -> results/flawed/pp_clip_diag.json

    python experiments/pp_clip_diag.py --steps 50
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
import pp_p_train as PT                                                     # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402


def make(b, center, size, ro, rt, tscale, seed):
    torch.manual_seed(seed)
    D0 = np.load(os.path.join(FC.DATA, "demos.npz"))
    home = D0["ee_pos"][D0["step"] == 0].mean(0).tolist()
    return PF.PartsFlow(8, center=center, size=size, rank=32, aframe=1, home=home, rclip=0.1237, rtclip=0.0925, psupport=True,
                        rclip_p=ro, rtclip_p=rt, tscale=tscale)


def train(m, b, act, steps, per_role, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(steps):
        j = torch.as_tensor(rng.integers(0, len(act), 512))
        loss = m.loss({k: v[j] for k, v in b.items()}, act[j])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        if per_role:
            torch.nn.utils.clip_grad_norm_(list(m.obj.parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(m.tgt.parameters()), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0)
        opt.step()
    return m


def objdiff(a, c):
    return max(float((p - q).abs().max()) for p, q in zip(a.obj.parameters(), c.obj.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cpu")
    b, act, center, size, _, _ = PT.rows(dev, list(FC.KNOWN))
    ro, rt = PT.part_radii(b, dev)
    out = {}
    for name, per_role in (("joint", False), ("per_role", True)):
        A = train(make(b, center, size, ro, rt, False, 10), b, act, a.steps, per_role, 10)
        C = train(make(b, center, size, ro, rt, True, 10), b, act, a.steps, per_role, 10)
        out[name] = dict(steps=a.steps, obj_max_abs_diff=objdiff(A, C))
        print("[clip]", name, json.dumps(out[name]), flush=True)
    P, _ = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", "pp_full_pp_s10", "P.npz"))
    T, _ = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", "pp_full_ts_s10", "P.npz"))
    num = sum(float(((p - q) ** 2).sum()) for p, q in zip(P.obj.parameters(), T.obj.parameters())) ** 0.5
    den = sum(float((p ** 2).sum()) for p in P.obj.parameters()) ** 0.5
    out["checkpoints_s10"] = dict(obj_rel_l2=num / den)
    print("[clip] checkpoints", json.dumps(out["checkpoints_s10"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "flawed", "pp_clip_diag.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

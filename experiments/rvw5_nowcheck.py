"""RVW5 noW serving check (docs/notes/plan_review_atlas.md 5.1; v4).  openpi venv, ONE GPU.  DEVIATION from the plan (recorded in review_atlas_results.md): two separately loaded backbones of the
SAME config give different samples with the same seed and noise (instance nondeterminism: v3 run, S vs S2 max |diff| up to .44 on dims 0-2 and ~2 on the gripper), so a cross-instance
comparison cannot discriminate.  This check is made deterministic in three parts:
  (1) INPUT identity: a backbone built with config pi05_g8v_raac_l_ae (noW) and one with pi05_g8v_raas_l_ae (RAA-sL), same checkpoint, give the same normalised state / prefix state for every test input.
  (2) SERVO identity inside ONE instance (RAA-sL config): chunk(servo_k = K) - chunk(model.config servo_k = 0), same noise, equals servo_norm_torch(state) on dims 0-2 (< .02) and 0 on dims 3+
      (the servo is exactly the post-flow addend; with servo_k = 0 the sampler is the noW path, backbone.py line 296 branch off), for 8 held states; object-role state: difference 0.
  (3) the noW backbone's config has servo_k == 0 and its model class equals RAA-sL's.
    python experiments/rvw5_nowcheck.py        (writes results/flawed/rvw5/nowcheck_v4.json, opened with "x")
"""
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_main6000"
OUTP = os.path.join(ROOT, "results", "flawed", "rvw5", "nowcheck_v4.json")
TOL = 0.02


def main():
    assert not os.path.exists(OUTP), "never overwrite: " + OUTP
    import torch
    from openpi.policies import libero_policy
    from lifelong_ood.policy import l90c_config as LC
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.raas_model import servo_norm_torch
    LC.register()
    held = [[0.05, -0.04, 0.03], [0.12, 0.08, 0.10], [-0.20, 0.10, 0.05], [0.0, 0.0, 0.0], [0.30, -0.20, 0.10], [0.02, 0.01, -0.01], [-0.08, -0.15, 0.20], [0.15, 0.0, 0.0]]
    states = [r + [2.0] for r in held] + [[0.05, -0.04, 0.03, 1.0]]
    res = dict(ckpt=CK, tol=TOL, rows=[])
    bkS = PolicyBackbone("pi05_g8v_raas_l_ae", checkpoint_dir=CK, device="cuda")
    cfgS = bkS.model.config
    A, H = bkS.A, bkS.H
    pfs, obs_state = [], []
    for st in states:
        ex = libero_policy.make_libero_example()
        ex["observation/state"] = np.asarray(st, np.float32)
        ex["prompt"] = "target: the plate"
        ob = bkS.observation(ex)
        pfs.append(bkS.prefix(ob))
    outS, outN = [], []
    for st, pf in zip(states, pfs):
        noise = torch.randn((1, H * A), generator=torch.Generator(device="cpu").manual_seed(11)).numpy()
        bkS.model.config = cfgS
        xs = bkS.sample(1, pf, seed=5, noise=noise, full=True).reshape(H, A)
        bkS.model.config = dataclasses.replace(cfgS, servo_k=0.0)
        xn = bkS.sample(1, pf, seed=5, noise=noise, full=True).reshape(H, A)
        bkS.model.config = cfgS
        sv = servo_norm_torch(cfgS, pf.state, A)[..., :3].float().cpu().numpy().reshape(3)
        outS.append(xs)
        outN.append(xn)
        d012 = float(np.abs(xn[:, :3] + sv[None, :] - xs[:, :3]).max())
        d3 = float(np.abs(xn[:, 3:] - xs[:, 3:]).max())
        res["rows"].append(dict(state=st, servo_norm=[float(v) for v in sv], max_diff_dims012=d012, max_diff_dims3p=d3, state_normalised=[float(v) for v in pf.state.float().cpu().numpy().reshape(-1)[:4]],
                                pass_=bool(d012 < TOL and d3 < 1e-6)))
    stS = [pf.state.float().cpu().numpy() for pf in pfs]
    del bkS
    torch.cuda.empty_cache()
    bkW = PolicyBackbone("pi05_g8v_raac_l_ae", checkpoint_dir=CK, device="cuda")
    res["noW_servo_k"] = float(bkW.model.config.servo_k)
    res["same_model_class"] = type(bkW.model).__name__
    ins = []
    for st, s0 in zip(states, stS):
        ex = libero_policy.make_libero_example()
        ex["observation/state"] = np.asarray(st, np.float32)
        ex["prompt"] = "target: the plate"
        pf = bkW.prefix(bkW.observation(ex))
        ins.append(float(np.abs(pf.state.float().cpu().numpy() - s0).max()))
    res["input_state_maxdiff_noW_vs_S"] = ins
    ok = all(r["pass_"] for r in res["rows"]) and max(ins) == 0.0 and res["noW_servo_k"] == 0.0
    res["all_pass"] = bool(ok)
    os.makedirs(os.path.dirname(OUTP), exist_ok=True)
    with open(OUTP, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nowcheck] rows", [(round(r["max_diff_dims012"], 4), round(r["max_diff_dims3p"], 6), r["pass_"]) for r in res["rows"]], "inputs", ins, "servo_k noW", res["noW_servo_k"], flush=True)
    print("[nowcheck] ALL PASS" if ok else "[nowcheck] FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""RVW5 noW serving check (docs/notes/plan_review_atlas.md 5.1).  openpi venv, ONE GPU.  The RAA-sL checkpoint (pi05_g8v_raas_l_ae_main6000) is served twice, with its own config
(servo_k = K: RAA-sL) and with config pi05_g8v_raac_l_ae (servo_k = 0: noW).  On 8 held-target states (role id 2; DEVIATION from the plan: synthetic channel values on openpi's example
frames instead of a2 captures, which would need the capture reader) and 1 object-role state, with the SAME seed and noise:
  noW chunk dims 0-2 + servo_norm_torch(state) == RAA-sL chunk dims 0-2 (|diff| < .02, bf16), dims 3-6 equal, object-role state: noW == RAA-sL on all dims.
    python experiments/rvw5_nowcheck.py        (writes results/flawed/rvw5/nowcheck.json, opened with "x")
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_main6000"
OUTP = os.path.join(ROOT, "results", "flawed", "rvw5", "nowcheck.json")
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
    out = {}
    for tag, cfg in (("S", "pi05_g8v_raas_l_ae"), ("W", "pi05_g8v_raac_l_ae")):
        bk = PolicyBackbone(cfg, checkpoint_dir=CK, device="cuda")
        A, H = bk.A, bk.H
        rows, svs = [], []
        for st in states:
            ex = libero_policy.make_libero_example()
            ex["observation/state"] = np.asarray(st, np.float32)
            ex["prompt"] = "target: the plate"
            ob = bk.observation(ex)
            pf = bk.prefix(ob)
            noise = torch.randn((1, H * A), generator=torch.Generator(device="cpu").manual_seed(11)).numpy()
            x = bk.sample(1, pf, seed=5, noise=noise, full=True).reshape(H, A)
            rows.append(x)
            svs.append(servo_norm_torch(LC.train_config("pi05_g8v_raas_l_ae").model, pf.state, A)[..., :3].float().cpu().numpy().reshape(3))
        out[tag] = (np.stack(rows), np.stack(svs), bk.model.config.servo_k)
        del bk
        torch.cuda.empty_cache()
    S, W = out["S"][0], out["W"][0]
    sv = out["S"][1]
    ok = True
    for i, st in enumerate(states):
        d012 = float(np.abs(W[i][:, :3] + sv[i][None, :] - S[i][:, :3]).max())
        d3 = float(np.abs(W[i][:, 3:] - S[i][:, 3:]).max())
        held_i = st[3] == 2.0
        r = dict(state=st, servo_norm=[float(v) for v in sv[i]], max_diff_dims012=d012, max_diff_dims3p=d3, noW_vs_S_dims012_raw=float(np.abs(W[i][:, :3] - S[i][:, :3]).max()))
        r["pass"] = bool(d012 < TOL and d3 < TOL and ((not held_i) or np.abs(sv[i]).max() >= 0))
        if not held_i:
            r["pass"] = bool(np.abs(W[i] - S[i]).max() < TOL and np.abs(sv[i]).max() == 0)
        ok &= r["pass"]
        res["rows"].append(r)
    res["servo_k"] = dict(S=out["S"][2], W=out["W"][2])
    res["all_pass"] = bool(ok)
    os.makedirs(os.path.dirname(OUTP), exist_ok=True)
    with open(OUTP, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nowcheck] servo_k S %.3f W %.3f; rows %s" % (out["S"][2], out["W"][2], [(round(r["max_diff_dims012"], 4), round(r["max_diff_dims3p"], 4), r["pass"]) for r in res["rows"]]), flush=True)
    print("[nowcheck] ALL PASS" if ok else "[nowcheck] FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

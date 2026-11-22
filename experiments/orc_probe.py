"""ORC readout (docs/notes/plan_oracle_pi0.md 2.7): one batch of rows per oracle observation.  GPU, openpi venv.  Imports a2_probe (Patcher) unchanged.

Rows per observation (images, prompt, noise seed identical in all rows):
  ref       the observation as rendered (clean token, clean servo input)
  tok_cf    pre = clean, tok = state_cf (the token the front end would give for the paired spot), srv = clean        (phi, phi_net for servo models)
  abl.tok   tok = zeros (the unrouted convention)                                                                      (phi_zero, c_zero)
  chan_cf   (servo models only) tok = srv = state_cf                                                                   (phi_total, descriptive)
Saves the UNNORMALISED chunks of every row, one npz per observation (skip-if-exists, so a requeued job resumes).  The ref row of the first --ident-n observations is
checked against bk.sample with the same seed (max abs diff < 1e-3, plan section 6 row orc_probe).

    python experiments/orc_probe.py --label A+0 --config pi05_g8v_raac_lp_ae --ckpt <pytorch dir> --obs DIR --out DIR
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import a2_probe as AP                                                      # noqa: E402


def odict(z, state):
    return {"observation/image": z["image"], "observation/wrist_image": z["wrist"], "observation/state": np.asarray(state, np.float32), "prompt": str(z["prompt"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ident-n", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(a.config, checkpoint_dir=a.ckpt, device="cuda")
    print("[orc_probe] %s label %s" % (bk.describe(), a.label), flush=True)
    P = AP.Patcher(bk)
    fs = sorted(f for f in glob.glob(os.path.join(a.obs, "orc_*.npz")))
    if a.limit:
        fs = fs[:a.limit]
    maxd = 0.0
    for n, f in enumerate(fs):
        oid = os.path.basename(f)
        path = os.path.join(a.out, oid)
        if os.path.exists(path):
            continue
        t0 = time.time()
        z = np.load(f, allow_pickle=False)
        clean, cf = odict(z, z["state"]), odict(z, z["state_cf"])
        zero = odict(z, np.zeros_like(z["state"]))
        seed = int(z["seed"])
        rows = [("ref", dict(pre=clean, tok=clean, srv=clean, comps=[])), ("tok_cf", dict(pre=clean, tok=cf, srv=clean, comps=[])),
                ("abl.tok", dict(pre=clean, tok=zero, srv=clean, comps=[]))]
        if P.servo:
            rows.append(("chan_cf", dict(pre=clean, tok=cf, srv=cf, comps=[])))
        ch = P.run([r for _, r in rows], seed)
        if n < a.ident_n:
            ob = bk.observation(clean)
            pre = bk.prefix(ob)
            ref = bk.unnormalize(bk.sample(1, pre, seed=seed), state=pre.state[0].float().cpu().numpy())[0]
            d = float(np.abs(ref - ch[0]).max())
            maxd = max(maxd, d)
            print("[orc_probe] ident %s: P.run ref vs bk.sample max abs diff %.3g" % (oid, d), flush=True)
            assert d < 1e-3, d
        np.savez(path, names=np.array([k for k, _ in rows]), chunks=ch.astype(np.float32))
        print("[orc_probe] %d/%d %s %.1fs" % (n + 1, len(fs), oid, time.time() - t0), flush=True)
    print("[orc_probe] done; ident max diff %.3g" % maxd, flush=True)


if __name__ == "__main__":
    main()

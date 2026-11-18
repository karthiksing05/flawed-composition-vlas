"""RAA-c (SR 77) plumbing check, GPU, openpi venv: the exported PyTorch checkpoint loads into the state-token model, the normalised 4-number state reaches
obs.state (first 4 dims, rest 0), and the sampled action chunk CHANGES with the channel (same images, prompt and noise) -- the token is wired in.
Label: plumbing on a quick checkpoint, not a result.
    python experiments/raac_sens_check.py --ckpt DIR [--config pi05_g8v_raac_ae]"""
import argparse
import os
import sys

import numpy as np

sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", default="pi05_g8v_raac_ae")
a = ap.parse_args()
from lifelong_ood.policy.backbone import PolicyBackbone
from lifelong_ood.policy import raac_model as RM
from openpi.policies import libero_policy
bk = PolicyBackbone(a.config, checkpoint_dir=a.ckpt, device="cuda")
m = bk.model
print("[raac-sens] model class %s, state_proj weight %s, |w| mean %.4g, |b| mean %.4g" % (
    type(m).__name__, tuple(m.state_proj.weight.shape), float(m.state_proj.weight.float().abs().mean()), float(m.state_proj.bias.float().abs().mean())), flush=True)
assert isinstance(m, RM.PI0PytorchC)
ex = libero_policy.make_libero_example()
ex["prompt"] = "target: the plate"
outs = []
for st in ([0.0, 0.0, 0.0, 0.0], [0.10, 0.0, 0.0, 2.0], [-0.12, 0.05, -0.20, 2.0], [-0.12, 0.05, -0.20, 1.0]):
    e = dict(ex)
    e["observation/state"] = np.asarray(st, np.float32)
    ob = bk.observation(e)
    s = ob.state[0].float().cpu().numpy()
    pre = bk.prefix(ob)
    rows = bk.sample(1, pre, seed=3, num_steps=10)
    outs.append(np.asarray(rows)[0])
    print("[raac-sens] state %s -> obs.state[:6] %s (rest zero: %s); chunk row0 %s" % (st, np.round(s[:6], 3).tolist(), bool(np.all(s[4:] == 0)), np.round(outs[-1][:7], 3).tolist()), flush=True)
    assert s.shape[0] == 32 and np.all(s[4:] == 0)
d = [float(np.abs(outs[0] - o).max()) for o in outs[1:]]
print("[raac-sens] max |chunk(state) - chunk(zeros)| for the 3 channels: %s; role id 1 vs 2 at equal offsets: %.4g" % (np.round(d, 4).tolist(), float(np.abs(outs[2] - outs[3]).max())))
assert min(d) > 1e-4, "the chunk does not depend on the state: the token is not wired in"
print("[raac-sens] done")

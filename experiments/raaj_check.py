"""NPHJ checks (openpi venv, one GPU).
  identity   : (random-initialised full models, no checkpoint)
     1. parameter tree of pi0_g8v_raaj_l_ae = pi0-A0's (pi0_g8v_raac_l_ae) + exactly where_head/l{1,2,3}/{kernel,bias}; frozen set equal; where_head trainable; l3 zero-initialised.
     2. step-0 identity: J's head output is exactly 0, and with the same init and noise J's sample and loss equal pi0-A0's (< 1e-6).
     3. random head weights: jax head_norm == torch head_norm (< 1e-4) on 100 states; zero on non-role-2 rows and in dims >= 3.
  post --jax_dir D --torch_dir T : the smoke checkpoint (30 training steps) and its export: where_head tensors in the safetensors equal the JAX ones exactly, last layer moved off zero (the flow loss
     reaches the branch), and PI0PytorchZJ.sample_actions - PI0Pytorch.sample_actions = head_norm_torch(random branch weights) through PolicyBackboneJ's model.
    python experiments/raaj_check.py identity | post --jax_dir .. --torch_dir ..
"""
import argparse
import dataclasses
import gc
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]


def rand_w(seed, sc=0.3):
    r = np.random.default_rng(seed)
    w = {}
    for k, (i, o) in (("l1", (3, 64)), ("l2", (64, 64)), ("l3", (64, 30))):
        w["where_head.%s.weight" % k] = (r.normal(0, sc, (o, i))).astype(np.float32)
        w["where_head.%s.bias" % k] = (r.normal(0, sc, (o,))).astype(np.float32)
    return w


def states(cfg, n=100, role=2.0, seed=0):
    r = np.random.default_rng(seed)
    raw = np.concatenate([r.uniform(-0.15, 0.15, (n, 3)), np.full((n, 1), role)], axis=1)
    q01, q99 = np.asarray(cfg.st_q01, np.float64), np.asarray(cfg.st_q99, np.float64)
    return raw, np.pad(((raw - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32), [(0, 0), (0, 28)])


def identity():
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import torch
    from lifelong_ood.policy import l90c_config as LC, raaj_model as RJ
    LC.register()
    import openpi.training.config as C
    cj, cc = C.get_config("pi0_g8v_raaj_l_ae"), C.get_config("pi0_g8v_raac_l_ae")
    assert type(cj.model) is RJ.Pi0ConfigZJ and cj.model.servo_k == 0.0 and not cj.model.pi05
    sh = lambda c, f=None: {k: tuple(v.shape) for k, v in tu.flatten_dict((nnx.state(nnx.eval_shape(c.model.create, jax.random.key(0)), f) if f else nnx.state(nnx.eval_shape(c.model.create, jax.random.key(0)))).to_pure_dict(), sep="/").items()}   # noqa: E731
    ta, tb = sh(cc), sh(cj)
    extra = set(tb) - set(ta)
    assert not (set(ta) - set(tb)), set(ta) - set(tb)
    assert extra and all(k.startswith("where_head/") for k in extra), extra
    shapes = {(k.split("/")[1], k.split("/")[2]): tb[k] for k in extra}
    assert shapes == {("l1", "kernel"): (3, 64), ("l1", "bias"): (64,), ("l2", "kernel"): (64, 64), ("l2", "bias"): (64,), ("l3", "kernel"): (64, 30), ("l3", "bias"): (30,)}, shapes
    assert all(tb[k] == ta[k] for k in ta)
    fa, fb = set(sh(cc, cc.freeze_filter)), set(sh(cj, cj.freeze_filter))
    assert fa == fb and not (extra & fb) and any("state_proj" in k for k in set(tb) - fb), "frozen set differs or where_head frozen"
    print("[raaj-check] 1 ok: pi0-A0's %d tensors + %d where_head tensors; same %d frozen; where_head and state_proj trainable" % (len(ta), len(extra), len(fa)), flush=True)
    raw, S = states(cj.model)
    obs_spec, _ = cc.model.inputs_spec(batch_size=4)
    obs = jax.tree.map(lambda s: jnp.ones(s.shape, s.dtype) if s.dtype in (jnp.float32, jnp.bool_) else jnp.zeros(s.shape, s.dtype), obs_spec)
    obs = dataclasses.replace(obs, state=jnp.asarray(S[:4]), tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask))
    acts = jax.random.normal(jax.random.key(3), (4, cc.model.action_horizon, 32), jnp.float32)
    noise = jax.random.normal(jax.random.key(4), (4, cc.model.action_horizon, 32), jnp.float32)
    out = {}
    for tag, cfg in (("A0", cc.model), ("J", cj.model)):
        mdl = cfg.create(jax.random.key(0))
        mdl.eval()
        if tag == "J":
            h0 = np.asarray(RJ.head_norm_jax(cj.model, mdl.where_head, jnp.asarray(S), 32))
            assert np.abs(h0).max() == 0.0, np.abs(h0).max()
        out[tag] = (np.asarray(mdl.sample_actions(jax.random.key(5), obs, noise=noise), np.float32), np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts, train=False), np.float32))
        del mdl
        gc.collect()
        jax.clear_caches()
    d1, d2 = np.abs(out["A0"][0] - out["J"][0]).max(), np.abs(out["A0"][1] - out["J"][1]).max()
    assert d1 < 1e-6 and d2 < 1e-6, (d1, d2)
    print("[raaj-check] 2 ok: step-0 J head output is exactly 0; J == pi0-A0 on sample (%.2g) and loss (%.2g), same init and noise" % (d1, d2), flush=True)
    w = rand_w(1)
    hj, ht = RJ.jax_head_from_numpy(w), RJ.torch_head_from_numpy(w)
    yj = np.asarray(RJ.head_norm_jax(cj.model, hj, jnp.asarray(S), 32))
    yt = RJ.head_norm_torch(cj.model, ht, torch.from_numpy(S), 32).detach().numpy()
    d = float(np.abs(yj - yt).max())
    assert d < 1e-4 and np.abs(yj).max() > 1e-3, (d, np.abs(yj).max())
    assert np.allclose(yj[:, :, 3:], 0) and np.allclose(yt[:, :, 3:], 0)
    raw0, S0 = states(cj.model, role=1.0)
    y0 = np.asarray(RJ.head_norm_jax(cj.model, hj, jnp.asarray(S0), 32))
    y0t = RJ.head_norm_torch(cj.model, ht, torch.from_numpy(S0), 32).detach().numpy()
    assert np.abs(y0).max() == 0 and np.abs(y0t).max() == 0, "branch not gated off the object role"
    print("[raaj-check] 3 ok: random branch weights: jax == torch (max |diff| %.2g, max |head| %.3f normalised) on 100 states; zero on role 1 and in dims >= 3" % (d, np.abs(yj).max()), flush=True)
    print("[raaj-check] identity done", flush=True)


def post(a):
    import jax
    import numpy as np
    import safetensors.torch as st
    import torch
    from lifelong_ood.policy import l90c_config as LC, raaj_model as RJ
    from openpi.models import model as _m0
    LC.register()
    prm = _m0.restore_params(os.path.join(a.jax_dir, "params") + "/", restore_type=np.ndarray, dtype="float32")
    hj = RJ.torch_state_from_jax(prm)
    back = st.load_file(os.path.join(a.torch_dir, "model.safetensors"))
    for k, v in hj.items():
        assert torch.equal(back[k], v.float()), k
    mv = float(hj["where_head.l3.weight"].abs().max())
    assert mv > 0, "last layer still exactly zero after the smoke training: the flow loss does not reach the branch"
    print("[raaj-check] post 1 ok: %d where_head tensors in the export equal the JAX checkpoint; |l3.kernel| max %.3g after the smoke steps (> 0: the flow loss reaches the branch)" % (len(hj), mv), flush=True)
    import openpi.models_pytorch.pi0_pytorch as _pt
    import openpi.models.model as _m
    from openpi.policies import libero_policy
    from lifelong_ood.policy.backbone_j import PolicyBackboneJ
    bk = PolicyBackboneJ("pi0_g8v_raaj_l_ae", checkpoint_dir=a.torch_dir, device="cuda")
    pol, mdl = bk.policy, bk.policy._model
    assert type(mdl) is RJ.PI0PytorchZJ, type(mdl)
    w = rand_w(2, 0.2)
    mdl.where_head.load_state_dict({k.split("where_head.", 1)[1]: torch.as_tensor(v) for k, v in w.items()})
    mdl.where_head.float().to(pol._pytorch_device)
    for r in ([0.05, -0.04, 0.03, 2.0], [0.05, -0.04, 0.03, 1.0], [-0.02, 0.01, 0.0, 2.0]):
        ex = libero_policy.make_libero_example()
        ex["observation/state"] = np.asarray(r, np.float32)
        inputs = pol._input_transform(jax.tree.map(lambda x: x, ex))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(pol._pytorch_device)[None, ...], inputs)
        ob = _m.Observation.from_dict(inputs)
        nz = torch.randn(1, mdl.config.action_horizon, mdl.config.action_dim, device=pol._pytorch_device, generator=torch.Generator(device=pol._pytorch_device).manual_seed(7))
        with torch.no_grad():
            a_h = mdl.sample_actions(pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
            a_c = _pt.PI0Pytorch.sample_actions(mdl, pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
            want = RJ.head_norm_torch(mdl.config, mdl.where_head, inputs["state"].float(), mdl.config.action_dim).cpu().numpy()
        got = a_h - a_c
        assert np.abs(got - want).max() < 5e-3, (r, np.abs(got - want).max())
        assert np.allclose(got[..., 3:], 0, atol=1e-6)
        print("[raaj-check]   state %s: branch applied %s (row 0), want %s" % (r, np.round(got[0, 0, :3], 4).tolist(), np.round(want[0, 0, :3], 4).tolist()), flush=True)
    print("[raaj-check] post 2 ok: PI0PytorchZJ = PI0Pytorch + branch on the same weights and noise; PolicyBackboneJ constructs", flush=True)
    print("[raaj-check] post done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("identity", "post"))
    ap.add_argument("--jax_dir")
    ap.add_argument("--torch_dir")
    a = ap.parse_args()
    identity() if a.what == "identity" else post(a)

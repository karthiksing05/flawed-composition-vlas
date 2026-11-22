"""PZ model checks (docs/notes/plan_oracle_pi0.md 3.4 step 5, section 6 row pz_check).  A copy of raas_check.py's items 1-4 retargeted to openpi pi0.  openpi venv, ONE GPU.

  1. k = 0 (Pi0ZS with servo_k = 0) equals openpi's plain Pi0 exactly (same rng, same random init): compute_loss and sample_actions.
  2. real K: compute_loss(actions) = Pi0.compute_loss(actions - servo); sample_actions = Pi0.sample_actions + servo (same noise); the servo is nonzero only for role 2.
  3. parameter tree of the pi0_g8v_* configs = pi0_libero's saved tree (no missing, no extra key, same shapes); the config's weight loader returns every leaf (nothing
     left at init) and state_proj is bit-equal to pi0_libero's; state_proj is trainable (not frozen); the data config has use_quantile_norm True and no DeltaActions.
  4. [--torch-ckpt DIR --name CFG] an exported PyTorch checkpoint: PI0PytorchZS.sample_actions - PI0Pytorch.sample_actions on the same weights and noise = the servo of the
     policy's own normalised state (held target on, object role off, zeros off).

    python experiments/pz_check.py [--torch-ckpt <pytorch dir> --name pi0_g8v_raas_l_ae]
"""
import argparse
import dataclasses
import gc
import json
import os
import pathlib
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT, os.path.join(ROOT, "experiments")]
import raas_check as RC                                                      # noqa: E402  (servo_ref, real_states, pad32; main() is not run)

PZ = ("pi0_g8v_raas_l_ae", "pi0_g8v_raac_l_ae", "pi0_g8v_raac_lf_ae")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-ckpt", default=None)
    ap.add_argument("--name", default="pi0_g8v_raas_l_ae")
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import torch
    from lifelong_ood.policy import l90c_config as LC, raas0_model as RZ, raas_model as RS
    LC.register()
    import openpi.models.pi0_config as PC
    import openpi.models.model as _m
    import openpi.training.config as C
    g = json.load(open(RS.GATE, encoding="utf-8"))
    ns = json.load(open(RS.NORM, encoding="utf-8"))["norm_stats"]
    cs = C.get_config("pi0_g8v_raas_l_ae")

    # ---- 1, 2. identity with openpi Pi0 (same rng, same random init, one model alive at a time)
    raw, nrm = RC.real_states(ns)
    base = PC.Pi0Config(pi05=False, action_horizon=cs.model.action_horizon)
    assert type(cs.model) is RZ.Pi0ConfigZS and not cs.model.pi05 and cs.model.action_horizon == 10
    obs_spec, _ = cs.model.inputs_spec(batch_size=4)
    obs = jax.tree.map(lambda s: jnp.ones(s.shape, s.dtype) if s.dtype in (jnp.float32, jnp.bool_) else jnp.zeros(s.shape, s.dtype), obs_spec)
    obs = dataclasses.replace(obs, state=jnp.asarray(RC.pad32(nrm)), tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask))
    acts = jax.random.normal(jax.random.key(3), (4, 10, 32), jnp.float32)
    noise = jax.random.normal(jax.random.key(4), (4, 10, 32), jnp.float32)
    srv = np.asarray(RS.servo_norm_jax(cs.model, obs.state, 32))
    assert np.abs(srv[0]).max() > 0 and np.allclose(srv[1:3], 0), "servo must be on for the held row only"
    out = {}
    for tag, cfg in (("P", base), ("Z0", dataclasses.replace(cs.model, servo_k=0.0)), ("Z", cs.model)):
        mdl = cfg.create(jax.random.key(0))
        mdl.eval()
        out[tag] = dict(sample=np.asarray(mdl.sample_actions(jax.random.key(5), obs, noise=noise), np.float32),
                        loss=np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts, train=False), np.float32))
        if tag == "P":
            out[tag]["loss_sh"] = np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts - jnp.asarray(srv), train=False), np.float32)
        del mdl
        gc.collect()
    d1 = float(np.abs(out["Z0"]["sample"] - out["P"]["sample"]).max())
    d2 = float(np.abs(out["Z0"]["loss"] - out["P"]["loss"]).max())
    assert d1 == 0 and d2 == 0, ("servo_k = 0 differs from openpi Pi0", d1, d2)
    d3 = float(np.abs(out["Z"]["sample"] - (out["P"]["sample"] + srv)).max())
    d4 = float(np.abs(out["Z"]["loss"] - out["P"]["loss_sh"]).max())
    assert d3 < 1e-5 and d4 == 0, (d3, d4)
    assert np.abs(out["Z"]["loss"] - out["P"]["loss"]).max() > 1e-6, "servo has no effect on the loss"
    print("[pz-check] 1-2 ok: Pi0ZS(k = 0) == openpi Pi0 exactly (sample max|diff| %.1g, loss %.1g); real K: sample = Pi0 sample + servo (%.2g), loss = Pi0 loss on (action - servo) (%.1g)" % (
        d1, d2, d3, d4), flush=True)

    # ---- 3. parameter tree = pi0_libero's, loader returns every leaf, state_proj bit-equal, trainable, data config
    saved = _m.restore_params(LC.PI0_LIBERO_PARAMS, restore_type=np.ndarray)
    sf = {k.replace("/value", ""): tuple(np.shape(v)) for k, v in tu.flatten_dict(saved, sep="/").items()}
    for n in PZ:
        c = C.get_config(n)
        assert type(c.model) is RZ.Pi0ConfigZS and not c.model.pi05 and c.model.max_token_len == 48 and c.model.action_horizon == 10
        assert str(c.weight_loader.params_path) == LC.PI0_LIBERO_PARAMS and "pi05" not in LC.PI0_LIBERO_PARAMS
        shp = nnx.eval_shape(c.model.create, jax.random.key(0))
        tree = nnx.state(shp).to_pure_dict()
        mf = {k.replace("/value", ""): tuple(v.shape) for k, v in tu.flatten_dict(tree, sep="/").items()}
        assert mf == sf, (n, sorted(set(mf) ^ set(sf))[:10], [k for k in mf if k in sf and mf[k] != sf[k]][:5])
        loaded = c.weight_loader.load(tree)
        lf = {k.replace("/value", ""): v for k, v in tu.flatten_dict(loaded, sep="/").items()}
        left = [k for k, v in lf.items() if isinstance(v, jax.ShapeDtypeStruct)]
        assert not left, (n, left[:5])
        lk = np.asarray(lf["state_proj/kernel"])
        sk = np.asarray(saved["state_proj"]["kernel"])
        assert lk.shape == (32, 1024) and np.array_equal(lk, sk), (n, np.abs(lk - sk).max())
        fz = set(tu.flatten_dict(nnx.state(shp, c.freeze_filter).to_pure_dict(), sep="/"))
        assert not any("state_proj" in k or "action_in_proj" in k or "action_out_proj" in k or "action_time_mlp" in k for k in fz), "a projection is frozen"
        assert any("llm" in k and "_1" not in k for k in fz) and any("img" in k for k in fz) and not any("llm" in k and "_1" in k for k in fz)
        dc = c.data.create(pathlib.Path(c.assets_dirs), c.model)
        ins = [type(t).__name__ for t in dc.data_transforms.inputs]
        assert dc.use_quantile_norm and "DeltaActions" not in ins and dc.norm_stats is not None, (n, ins)
        print("[pz-check] 3 ok %s: %d tensors = pi0_libero's tree, loader leaves all concrete, state_proj bit-equal (|W|max %.3g), %d frozen, quantile norm, no DeltaActions, repo %s" % (
            n, len(mf), float(np.abs(sk).max()), len(fz), dc.repo_id), flush=True)

    # ---- 4. exported PyTorch checkpoint
    if a.torch_ckpt:
        import openpi.models_pytorch.pi0_pytorch as _pt
        from openpi.policies import libero_policy
        from lifelong_ood.policy.backbone import PolicyBackbone
        bk = PolicyBackbone(a.name, checkpoint_dir=a.torch_ckpt, device="cuda")
        pol = bk.policy
        mdl = pol._model
        assert type(mdl) is RZ.PI0PytorchZS, type(mdl)
        for r in ([0.05, -0.04, 0.03, 2.0], [0.05, -0.04, 0.03, 1.0], [0, 0, 0, 0]):
            ex = libero_policy.make_libero_example()
            ex["observation/state"] = np.asarray(r, np.float32)
            inputs = pol._input_transform(jax.tree.map(lambda x: x, ex))
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(pol._pytorch_device)[None, ...], inputs)
            ob = _m.Observation.from_dict(inputs)
            nz = torch.randn(1, mdl.config.action_horizon, mdl.config.action_dim, device=pol._pytorch_device, generator=torch.Generator(device=pol._pytorch_device).manual_seed(7))
            with torch.no_grad():
                a_s = mdl.sample_actions(pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
                a_c = _pt.PI0Pytorch.sample_actions(mdl, pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
            want = RC.servo_ref(inputs["state"].float().cpu().numpy()[:, :4], g, ns) if mdl.config.servo_k > 0 else np.zeros((1, 3))
            got = (a_s - a_c)[0, :, :3]
            assert np.abs(got - want).max() < 5e-3, (r, np.abs(got - want).max(), got[0], want)
            assert np.allclose((a_s - a_c)[..., 3:], 0, atol=1e-6)
            print("[pz-check]   state %s: servo_norm applied %s, reference %s" % (r, np.round(got[0], 4).tolist(), np.round(want[0], 4).tolist()), flush=True)
        print("[pz-check] 4 ok: PI0PytorchZS = PI0Pytorch + servo on the same weights and noise", flush=True)
    print("[pz-check] done", flush=True)


if __name__ == "__main__":
    main()

"""RAA-s (docs/notes/raas_plan.md, SR 78) model checks.  openpi venv, ONE GPU (random-initialised full model, no checkpoint needed except for --torch-ckpt).
No rollout, no training data beyond reading 4 state columns of the RAA-c LeRobot set.

  1. config / parameter tree: the raas config's JAX parameter tree equals raac's exactly (no new parameter), its freeze filter freezes the same tensors, the raac config is
     untouched (type Pi0ConfigC, no servo fields), the loader is RaacLoader; the servo constants equal results/flawed/raas_gate.json (K, max_step) and the RAA-c norm stats.
  2. servo function: servo_norm_jax and servo_norm_torch equal a numpy reference (written here from the plan's formula, not imported) on real RAA-c state rows
     (held target role 2, object role 1, no-channel zeros), m = 1 only for role 2, zeros off dims 0-2, clip binds on an extreme row.
  3. raac unchanged (JAX, same rng, same random init): (a) Pi0S with servo_k = 0: compute_loss and sample_actions equal Pi0C's exactly; (b) with the real K: compute_loss(actions)
     equals Pi0C.compute_loss(actions - servo) and sample_actions equals Pi0C.sample_actions + servo (same noise).
  4. [--torch-ckpt DIR] an exported PyTorch RAA-s checkpoint: PI0PytorchS.sample_actions - PI0PytorchC.sample_actions on the SAME weights and noise = servo_np of the policy's own
     normalised state, for a held-target state (role 2), an object state (role 1) and zeros.

    python experiments/raas_check.py [--torch-ckpt <pytorch dir>]
"""
import argparse
import dataclasses
import gc
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"


def servo_ref(state_norm, g, ns):
    """numpy reference of the plan's formula on a normalised state [b, >=4] -> normalised-action servo [b, 3]."""
    q01, q99 = np.asarray(ns["state"]["q01"][:4], np.float64), np.asarray(ns["state"]["q99"][:4], np.float64)
    raw = (np.asarray(state_norm[:, :4], np.float64) + 1) / 2 * (q99 - q01 + 1e-6) + q01
    m = raw[:, 3] > 1.5
    s = np.clip(g["K"] * raw[:, :3], -g["max_step"], g["max_step"]) * m[:, None]
    aq01, aq99 = np.asarray(ns["actions"]["q01"][:3], np.float64), np.asarray(ns["actions"]["q99"][:3], np.float64)
    return s * 2 / (aq99 - aq01 + 1e-6)


def norm_state(raw, ns):
    q01, q99 = np.asarray(ns["state"]["q01"][:4], np.float64), np.asarray(ns["state"]["q99"][:4], np.float64)
    return ((np.asarray(raw, np.float64) - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32)


def real_states(ns):
    import pyarrow.parquet as pq
    rows = []
    for ep, want in ((0, 2.0), (0, 1.0), (350, 0.0)):                         # K1 held, K1 object role, K8 (no channel: zeros)
        st = np.asarray(pq.read_table(os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % ep), columns=["state"]).to_pydict()["state"], np.float64)
        idx = np.where(st[:, 3] == want)[0]
        assert len(idx), (ep, want)
        rows.append(st[idx[len(idx) // 2]])
    rows.append(np.array([0.30, -0.25, 0.28, 2.0]))                           # extreme held row: clip must bind (K * .3 = .86 < .9375, so use .40)
    rows[-1][0] = 0.40
    return np.stack(rows), norm_state(np.stack(rows), ns)


def pad32(x):
    return np.pad(x, [(0, 0), (0, 32 - x.shape[1])]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-ckpt", default=None)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import flax.traverse_util as tu
    from lifelong_ood.policy import l90c_config as LC, raac_model as RM, raas_model as RS
    LC.register()
    import openpi.training.config as C
    g = json.load(open(RS.GATE, encoding="utf-8"))
    ns = json.load(open(RS.NORM, encoding="utf-8"))["norm_stats"]

    # ---- 1. config / tree
    cc, cs = C.get_config("pi05_g8v_raac_ae"), C.get_config("pi05_g8v_raas_ae")
    assert type(cc.model) is RM.Pi0ConfigC and not hasattr(cc.model, "servo_k"), "raac config changed"
    assert type(cs.model) is RS.Pi0ConfigS and isinstance(cs.weight_loader, RM.RaacLoader)
    assert abs(cs.model.servo_k - g["K"]) < 1e-12 and abs(cs.model.servo_max - g["max_step"]) < 1e-12 and cs.model.servo_k > 0
    assert cs.model.st_q01 == tuple(ns["state"]["q01"][:4]) and cs.model.act_q99 == tuple(ns["actions"]["q99"][:3])
    assert cs.data.repo_id == cc.data.repo_id and str(cs.data.assets.assets_dir) == str(cc.data.assets.assets_dir), "raas must train on the RAA-c data / norm stats"
    for f in ("batch_size", "num_train_steps", "ema_decay", "lr_schedule", "optimizer"):
        assert getattr(cs, f) == getattr(cc, f), f
    ta = {k: tuple(v.shape) for k, v in tu.flatten_dict(nnx.state(nnx.eval_shape(cc.model.create, jax.random.key(0))).to_pure_dict(), sep="/").items()}
    tb = {k: tuple(v.shape) for k, v in tu.flatten_dict(nnx.state(nnx.eval_shape(cs.model.create, jax.random.key(0))).to_pure_dict(), sep="/").items()}
    assert ta == tb, (set(ta) ^ set(tb))
    fa = set(tu.flatten_dict(nnx.state(nnx.eval_shape(cc.model.create, jax.random.key(0)), cc.freeze_filter).to_pure_dict(), sep="/"))
    fb = set(tu.flatten_dict(nnx.state(nnx.eval_shape(cs.model.create, jax.random.key(0)), cs.freeze_filter).to_pure_dict(), sep="/"))
    assert fa == fb
    print("[raas-check] 1 ok: raac config untouched (Pi0ConfigC), raas = same %d parameter tensors, same %d frozen, same data / norm stats / recipe; K %.4f max_step %.4f" % (
        len(ta), len(fa), cs.model.servo_k, cs.model.servo_max), flush=True)

    # ---- 2. servo function
    raw, nrm = real_states(ns)
    ref = servo_ref(nrm, g, ns)
    sj = np.asarray(RS.servo_norm_jax(cs.model, jnp.asarray(pad32(nrm)), 32))
    import torch
    st_ = RS.servo_norm_torch(cs.model, torch.from_numpy(pad32(nrm)), 32).numpy()
    assert sj.shape == (4, 1, 32) and np.allclose(sj[:, 0, 3:], 0) and np.allclose(st_[:, 0, 3:], 0)
    assert np.allclose(sj[:, 0, :3], ref, atol=1e-5) and np.allclose(st_[:, 0, :3], ref, atol=1e-5), (sj[:, 0, :3], st_[:, 0, :3], ref)
    assert np.abs(ref[0]).max() > 0 and np.allclose(ref[1], 0) and np.allclose(ref[2], 0), ref        # held target: on; object role: off; zeros: off
    assert abs(ref[3, 0] - 2 * g["max_step"] / (ns["actions"]["q99"][0] - ns["actions"]["q01"][0])) < 1e-4, "clip did not bind on the extreme row"
    print("[raas-check] 2 ok: jax / torch servo = numpy reference on real RAA-c states (held target %s cm -> servo_norm %s; object role 0; zeros 0; clip binds)" % (
        np.round(100 * raw[0, :3], 1).tolist(), np.round(ref[0], 4).tolist()), flush=True)

    # ---- 3. raac unchanged, JAX identity (same rng, same random init, one model alive at a time)
    from openpi.models import model as _m
    obs_spec, act_spec = cs.model.inputs_spec(batch_size=4)
    obs = jax.tree.map(lambda s: jnp.ones(s.shape, s.dtype) if s.dtype in (jnp.float32, jnp.bool_) else jnp.zeros(s.shape, s.dtype), obs_spec)
    obs = dataclasses.replace(obs, state=jnp.asarray(pad32(nrm)))
    # token ids need a non-empty prompt mask for the LLM to attend to something
    obs = dataclasses.replace(obs, tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask))
    acts = jax.random.normal(jax.random.key(3), (4, cs.model.action_horizon, 32), jnp.float32)
    noise = jax.random.normal(jax.random.key(4), (4, cs.model.action_horizon, 32), jnp.float32)
    srv = np.asarray(RS.servo_norm_jax(cs.model, obs.state, 32))
    out = {}
    cfg0 = dataclasses.replace(cs.model, servo_k=0.0)
    for tag, cfg in (("C", cc.model), ("S0", cfg0), ("S", cs.model)):
        mdl = cfg.create(jax.random.key(0))
        mdl.eval()
        out[tag] = dict(sample=np.asarray(mdl.sample_actions(jax.random.key(5), obs, noise=noise), np.float32))
        if tag == "C":
            out[tag]["loss_sh"] = np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts - jnp.asarray(srv), train=False), np.float32)
            out[tag]["loss"] = np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts, train=False), np.float32)
        else:
            out[tag]["loss"] = np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts, train=False), np.float32)
        del mdl
        gc.collect()
    d1 = np.abs(out["S0"]["sample"] - out["C"]["sample"]).max()
    d2 = np.abs(out["S0"]["loss"] - out["C"]["loss"]).max()
    assert d1 == 0 and d2 == 0, ("servo_k = 0 differs from raac", d1, d2)
    d3 = np.abs(out["S"]["sample"] - (out["C"]["sample"] + srv)).max()
    d4 = np.abs(out["S"]["loss"] - out["C"]["loss_sh"]).max()
    assert d3 < 1e-5 and d4 == 0, (d3, d4)
    assert np.abs(out["S"]["loss"] - out["C"]["loss"]).max() > 1e-6, "servo has no effect on the loss"
    print("[raas-check] 3 ok: servo_k = 0 reproduces raac exactly (sample max|diff| %.1g, loss %.1g); real K: sample = raac sample + servo (max|diff| %.2g), loss = raac loss on (action - servo) (%.1g)" % (
        d1, d2, d3, d4), flush=True)

    # ---- 4. exported PyTorch checkpoint
    if a.torch_ckpt:
        from openpi.policies import libero_policy
        from lifelong_ood.policy.backbone import PolicyBackbone
        bk = PolicyBackbone("pi05_g8v_raas_ae", checkpoint_dir=a.torch_ckpt, device="cuda")
        pol = bk.policy
        mdl = pol._model
        assert type(mdl) is RS.PI0PytorchS, type(mdl)
        rows = [[0.05, -0.04, 0.03, 2.0], [0.05, -0.04, 0.03, 1.0], [0, 0, 0, 0]]
        for r in rows:
            ex = libero_policy.make_libero_example()
            ex["observation/state"] = np.asarray(r, np.float32)
            inputs = pol._input_transform(jax.tree.map(lambda x: x, ex))
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(pol._pytorch_device)[None, ...], inputs)
            ob = _m.Observation.from_dict(inputs)
            nz = torch.randn(1, mdl.config.action_horizon, mdl.config.action_dim, device=pol._pytorch_device, generator=torch.Generator(device=pol._pytorch_device).manual_seed(7))
            with torch.no_grad():
                a_s = mdl.sample_actions(pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
                a_c = RM.PI0PytorchC.sample_actions(mdl, pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
            want = servo_ref(inputs["state"].float().cpu().numpy()[:, :4], g, ns)
            got = (a_s - a_c)[0, :, :3]
            assert np.abs(got - want).max() < 5e-3, (r, np.abs(got - want).max(), got[0], want)           # bf16 model dtype: the add is in bf16 when the chunk is bf16
            assert np.allclose((a_s - a_c)[..., 3:], 0, atol=1e-6)
            print("[raas-check]   state %s: servo_norm applied %s, reference %s" % (r, np.round(got[0], 4).tolist(), np.round(want[0], 4).tolist()), flush=True)
        print("[raas-check] 4 ok: PI0PytorchS = PI0PytorchC + servo on the same weights and noise (held target on, object / zeros off, dims 3+ untouched)", flush=True)
    print("[raas-check] done", flush=True)


if __name__ == "__main__":
    main()

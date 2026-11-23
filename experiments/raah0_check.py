"""NPH model checks (docs/notes/plan_necessity_power.md 3.2 step 4; copy of raah_check.py retargeted to pi0).  openpi venv, one GPU (random-initialised full models; no checkpoint) unless --torch-ckpt.
  1. identity: an npz in LINEAR form (lin_K = the gate's K, clip M) makes head_norm_{np,jax,torch} equal the servo's servo_norm under pi0's quantile normalisation on 1,000 RAA-c dataset states
     (max |diff| < 1e-6), and Pi0ZH with it gives Pi0ZS(pi0_g8v_raas_l_ae)'s loss and samples on one batch (same init, same noise).
  2. the learned head H*: head_norm_np == jax == torch on the same 1,000 states (< 1e-5); zero where the role id != 2; zero in dims >= 3; and Pi0ZH(H*) == Pi0ZS(k = 0) + head (sample) / Pi0 loss on (action - head).
  3. parameter tree (and frozen set) equals pi0-A0's (pi0_g8v_raac_l_ae); state_proj trainable; the real npz's sha256 equals results/flawed/raah0/fit.json's.
  4. [--torch-ckpt DIR] an exported PyTorch checkpoint: PI0PytorchZH.sample_actions - PI0Pytorch.sample_actions on the same weights and noise = head_norm_torch of the policy's own normalised state.
    python experiments/raah0_check.py [--torch-ckpt <pytorch dir>]
"""
import argparse
import dataclasses
import gc
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"
LIN = "/coc/flash10/ksingara3/.claude_tmp/pp/raah0_lin_head.npz"


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
    import pyarrow.parquet as pq
    import torch
    from lifelong_ood.policy import l90c_config as LC, raas_model as RS, raah0_model as RH
    LC.register()
    import openpi.training.config as C
    g = json.load(open(RS.GATE, encoding="utf-8"))
    ns = json.load(open(RS.NORM, encoding="utf-8"))["norm_stats"]
    ch, cs, cc = C.get_config("pi0_g8v_raah_l_ae"), C.get_config("pi0_g8v_raas_l_ae"), C.get_config("pi0_g8v_raac_l_ae")
    assert type(ch.model) is RH.Pi0ConfigZH and ch.model.servo_k == 0.0 and not ch.model.pi05
    q01, q99 = np.asarray(ns["state"]["q01"][:4], np.float64), np.asarray(ns["state"]["q99"][:4], np.float64)
    assert np.allclose(np.asarray(ch.model.st_q01, np.float64), q01) and np.allclose(np.asarray(ch.model.st_q99, np.float64), q99), "config state stats differ from the norm file"
    rows = []
    files = sorted(glob.glob(os.path.join(LR, "data", "chunk-000", "episode_*.parquet")))
    for f in files[::8]:
        st = np.asarray(pq.read_table(f, columns=["state"]).to_pydict()["state"], np.float64)
        rows.append(st[:: max(len(st) // 20, 1)])
    raw = np.concatenate(rows)[:1000]
    print("[raah0-check] states", raw.shape, "role counts", {int(k): int((raw[:, 3] == k).sum()) for k in np.unique(raw[:, 3])}, flush=True)
    nrm = ((raw - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32)
    S = pad32(nrm)
    P = {k: np.asarray(v) for k, v in np.load(RH.NPZ).items()}
    assert float(P["centred"]) == 0.0
    np.savez(LIN, lin_K=np.float32(g["K"]), M=P["M"], scale=P["scale"])
    cfg_lin = dataclasses.replace(ch.model, head_npz=LIN)
    # ---- 1. identity of the linear-form head with the servo under pi0's normalisation
    sv = np.asarray(RS.servo_norm_jax(cs.model, jnp.asarray(S), 32))[:, 0, :]
    for nm, hv in (("np", RH.head_norm_np(cfg_lin, S, 32)), ("jax", np.asarray(RH.head_norm_jax(cfg_lin, jnp.asarray(S), 32))),
                   ("torch", RH.head_norm_torch(cfg_lin, torch.from_numpy(S), 32).numpy())):
        assert hv.shape == (len(S), 10, 32), hv.shape
        d = max(np.abs(hv[:, j, :] - sv).max() for j in range(10))
        assert d < 1e-6, (nm, d)
        print("[raah0-check] 1a ok: linear-form head_norm_%s == servo_norm on %d states (max |diff| %.2g; max |servo| %.3f)" % (nm, len(S), d, np.abs(sv).max()), flush=True)
    # ---- 2. learned head: backends agree, zero off role 2, zero in dims >= 3
    hn = RH.head_norm_np(ch.model, S, 32)
    hj = np.asarray(RH.head_norm_jax(ch.model, jnp.asarray(S), 32))
    ht = RH.head_norm_torch(ch.model, torch.from_numpy(S), 32).numpy()
    d1, d2 = np.abs(hn - hj).max(), np.abs(hn - ht).max()
    assert d1 < 1e-5 and d2 < 1e-5, (d1, d2)
    off = raw[:, 3] != 2.0
    assert np.abs(hn[off]).max() == 0 and np.abs(hj[off]).max() == 0 and np.abs(ht[off]).max() == 0, "head not zero off role 2"
    assert np.allclose(hn[:, :, 3:], 0) and np.allclose(hj[:, :, 3:], 0) and np.allclose(ht[:, :, 3:], 0), "head leaks into dims >= 3"
    assert np.abs(hn[~off]).max() > 0, "head is zero on role-2 rows"
    print("[raah0-check] 2a ok: head np == jax (%.2g) == torch (%.2g) on %d states; zero on %d non-target rows; |head| max %.4f normalised" % (d1, d2, len(S), int(off.sum()), np.abs(hn).max()), flush=True)
    # ---- 3. parameter tree = pi0-A0's; sha256
    ta = {k: tuple(v.shape) for k, v in tu.flatten_dict(nnx.state(nnx.eval_shape(cc.model.create, jax.random.key(0))).to_pure_dict(), sep="/").items()}
    tb = {k: tuple(v.shape) for k, v in tu.flatten_dict(nnx.state(nnx.eval_shape(ch.model.create, jax.random.key(0))).to_pure_dict(), sep="/").items()}
    assert ta == tb, (set(ta) ^ set(tb))
    fa = set(tu.flatten_dict(nnx.state(nnx.eval_shape(cc.model.create, jax.random.key(0)), cc.freeze_filter).to_pure_dict(), sep="/"))
    fb = set(tu.flatten_dict(nnx.state(nnx.eval_shape(ch.model.create, jax.random.key(0)), ch.freeze_filter).to_pure_dict(), sep="/"))
    assert fa == fb and any("state_proj" in k for k in set(tb) - fb), "state_proj must be trainable"
    sha = hashlib.sha256(open(RH.NPZ, "rb").read()).hexdigest()
    fit = json.load(open(os.path.join(ROOT, "results", "flawed", "raah0", "fit.json"), encoding="utf-8"))
    assert sha == fit["npz_sha256"], (sha, fit["npz_sha256"])
    print("[raah0-check] 3 ok: same %d parameter tensors, same %d frozen as pi0-A0, state_proj trainable; npz sha256 %s == fit.json" % (len(ta), len(fa), sha), flush=True)
    # ---- 1b / 2b. Pi0ZH reproduces Pi0ZS (linear head) and Pi0ZS(k = 0) + head (learned head), same init, same noise
    obs_spec, _ = cs.model.inputs_spec(batch_size=4)
    obs = jax.tree.map(lambda s: jnp.ones(s.shape, s.dtype) if s.dtype in (jnp.float32, jnp.bool_) else jnp.zeros(s.shape, s.dtype), obs_spec)
    sel = np.flatnonzero(raw[:, 3] == 2.0)[:4]
    obs = dataclasses.replace(obs, state=jnp.asarray(S[sel]), tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask))
    acts = jax.random.normal(jax.random.key(3), (4, cs.model.action_horizon, 32), jnp.float32)
    noise = jax.random.normal(jax.random.key(4), (4, cs.model.action_horizon, 32), jnp.float32)
    hd = np.asarray(RH.head_norm_jax(ch.model, jnp.asarray(S[sel]), 32), np.float32)           # the jax head itself: a float64-numpy shift differs by ~1e-7, which bf16 rounding inside the model amplifies
    out = {}
    for tag, cfg in (("S", cs.model), ("Hlin", cfg_lin), ("A0", cc.model), ("H", ch.model)):
        mdl = cfg.create(jax.random.key(0))
        mdl.eval()
        out[tag] = (np.asarray(mdl.sample_actions(jax.random.key(5), obs, noise=noise), np.float32), np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts, train=False), np.float32))
        if tag == "A0":
            out["A0_shift"] = np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts - jnp.asarray(hd), train=False), np.float32)
        del mdl
        gc.collect()
        jax.clear_caches()
    dsm = np.abs(out["S"][0] - out["Hlin"][0]).max()
    dls = np.abs(out["S"][1] - out["Hlin"][1]).max()
    assert dsm < 1e-5 and dls < 1e-5, (dsm, dls)
    print("[raah0-check] 1b ok: Pi0ZH with the linear-form head == Pi0ZS: sample max |diff| %.2g, loss %.2g" % (dsm, dls), flush=True)
    e1 = np.abs(out["H"][0] - (out["A0"][0] + hd)).max()
    e2 = np.abs(out["H"][1] - out["A0_shift"]).max()
    assert e1 < 1e-5 and e2 < 1e-6, (e1, e2)
    assert np.abs(out["H"][1] - out["A0"][1]).max() > 1e-6, "the head has no effect on the loss"
    print("[raah0-check] 2b ok: Pi0ZH(H*) sample = Pi0ZS(k = 0) sample + head (%.2g), loss = Pi0 loss on (action - head) (%.2g)" % (e1, e2), flush=True)
    # ---- 4. exported PyTorch checkpoint
    if a.torch_ckpt:
        import openpi.models_pytorch.pi0_pytorch as _pt
        import openpi.models.model as _m
        from openpi.policies import libero_policy
        from lifelong_ood.policy.backbone_h0 import PolicyBackboneH0
        bk = PolicyBackboneH0("pi0_g8v_raah_l_ae", checkpoint_dir=a.torch_ckpt, device="cuda")
        pol = bk.policy
        mdl = pol._model
        assert type(mdl) is RH.PI0PytorchZH, type(mdl)
        for r in ([0.05, -0.04, 0.03, 2.0], [0.05, -0.04, 0.03, 1.0], [0, 0, 0, 0], [-0.02, 0.01, 0.0, 2.0]):
            ex = libero_policy.make_libero_example()
            ex["observation/state"] = np.asarray(r, np.float32)
            inputs = pol._input_transform(jax.tree.map(lambda x: x, ex))
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(pol._pytorch_device)[None, ...], inputs)
            ob = _m.Observation.from_dict(inputs)
            nz = torch.randn(1, mdl.config.action_horizon, mdl.config.action_dim, device=pol._pytorch_device, generator=torch.Generator(device=pol._pytorch_device).manual_seed(7))
            with torch.no_grad():
                a_h = mdl.sample_actions(pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
                a_c = _pt.PI0Pytorch.sample_actions(mdl, pol._pytorch_device, ob, noise=nz.clone()).float().cpu().numpy()
            want = RH.head_norm_torch(mdl.config, inputs["state"].float(), mdl.config.action_dim).cpu().numpy()
            got = a_h - a_c
            assert np.abs(got - want).max() < 5e-3, (r, np.abs(got - want).max())
            assert np.allclose(got[..., 3:], 0, atol=1e-6)
            print("[raah0-check]   state %s: head applied %s (row 0), want %s" % (r, np.round(got[0, 0, :3], 4).tolist(), np.round(want[0, 0, :3], 4).tolist()), flush=True)
        print("[raah0-check] 4 ok: PI0PytorchZH = PI0Pytorch + head on the same weights and noise; PolicyBackboneH0 constructs", flush=True)
    print("[raah0-check] done", flush=True)


if __name__ == "__main__":
    main()

"""RAA-h model checks (docs/notes/plan_review_atlas.md 6.6; copy of raas_check.py retargeted).  openpi venv, one GPU (random-initialised full model; no checkpoint).
  1. identity: an npz in LINEAR form (key lin_K = the gate's K, clip M, constant over the 10 rows) makes head_norm_{np,jax,torch} equal the servo's servo_norm on 1,000 RAA-c dataset
     states (max |diff| < 1e-6), and Pi0H with it gives Pi0S's loss and samples on one batch (same init, same noise).
  2. the learned head: head_norm_np == jax == torch on the same 1,000 states (< 1e-5); zero where the role id != 2; zero at rel = 0 (role 2).
  3. parameter tree (and frozen set) equals RAA-c's (no missing / extra keys); the real npz's sha256 equals results/flawed/raah/fit.json's.
    python experiments/raah_check.py
"""
import dataclasses
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"
LIN = "/coc/flash10/ksingara3/.claude_tmp/pp/raah_lin_head.npz"


def pad32(x):
    return np.pad(x, [(0, 0), (0, 32 - x.shape[1])]).astype(np.float32)


def main():
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import pyarrow.parquet as pq
    import torch
    from lifelong_ood.policy import l90c_config as LC, raac_model as RM, raas_model as RS, raah_model as RH
    LC.register()
    import openpi.training.config as C
    g = json.load(open(RS.GATE, encoding="utf-8"))
    ns = json.load(open(RS.NORM, encoding="utf-8"))["norm_stats"]
    ch, cs, cc = C.get_config("pi05_g8v_raah_l_ae"), C.get_config("pi05_g8v_raas_l_ae"), C.get_config("pi05_g8v_raac_ae")
    assert type(ch.model) is RH.Pi0ConfigH and ch.model.servo_k == 0.0
    # 1000 real states: spread over the 400 episodes (held, object, zeros rows)
    rows = []
    files = sorted(glob.glob(os.path.join(LR, "data", "chunk-000", "episode_*.parquet")))
    for f in files[::8]:
        st = np.asarray(pq.read_table(f, columns=["state"]).to_pydict()["state"], np.float64)
        rows.append(st[:: max(len(st) // 20, 1)])
    raw = np.concatenate(rows)[:1000]
    print("[raah-check] states", raw.shape, "role counts", {int(k): int((raw[:, 3] == k).sum()) for k in np.unique(raw[:, 3])}, flush=True)
    q01, q99 = np.asarray(ns["state"]["q01"][:4], np.float64), np.asarray(ns["state"]["q99"][:4], np.float64)
    nrm = ((raw - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32)
    S = pad32(nrm)
    P = {k: np.asarray(v) for k, v in np.load(RH.NPZ).items()}
    np.savez(LIN, lin_K=np.float32(g["K"]), M=P["M"], scale=P["scale"])
    cfg_lin = dataclasses.replace(ch.model, head_npz=LIN)
    # ---- 1. identity of the linear-form head with the servo
    sv = np.asarray(RS.servo_norm_jax(cs.model, jnp.asarray(S), 32))[:, 0, :]
    for nm, hv in (("np", RH.head_norm_np(cfg_lin, S, 32)), ("jax", np.asarray(RH.head_norm_jax(cfg_lin, jnp.asarray(S), 32))),
                   ("torch", RH.head_norm_torch(cfg_lin, torch.from_numpy(S), 32).numpy())):
        assert hv.shape == (len(S), 10, 32), hv.shape
        d = max(np.abs(hv[:, j, :] - sv).max() for j in range(10))
        assert d < 1e-6, (nm, d)
        print("[raah-check] 1a ok: linear-form head_norm_%s == servo_norm on %d states (max |diff| %.2g)" % (nm, len(S), d), flush=True)
    # ---- 2. learned head: backends agree, zero off role 2, zero at rel = 0
    hn = RH.head_norm_np(ch.model, S, 32)
    hj = np.asarray(RH.head_norm_jax(ch.model, jnp.asarray(S), 32))
    ht = RH.head_norm_torch(ch.model, torch.from_numpy(S), 32).numpy()
    d1, d2 = np.abs(hn - hj).max(), np.abs(hn - ht).max()
    assert d1 < 1e-5 and d2 < 1e-5, (d1, d2)
    off = raw[:, 3] != 2.0
    assert np.abs(hn[off]).max() == 0 and np.abs(hj[off]).max() == 0 and np.abs(ht[off]).max() == 0, "head not zero off role 2"
    assert np.allclose(hn[:, :, 3:], 0) and np.allclose(hj[:, :, 3:], 0) and np.allclose(ht[:, :, 3:], 0), "head leaks into dims >= 3"
    z = np.array([[0.0, 0.0, 0.0, 2.0]])
    zn = pad32(((z - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32))
    zmax = max(np.abs(RH.head_norm_np(ch.model, zn, 32)).max(), np.abs(np.asarray(RH.head_norm_jax(ch.model, jnp.asarray(zn), 32))).max(),
               np.abs(RH.head_norm_torch(ch.model, torch.from_numpy(zn), 32).numpy()).max())
    assert zmax < 1e-6, zmax
    print("[raah-check] 2 ok: head np == jax (%.2g) == torch (%.2g) on %d states; zero on %d non-target rows; zero at rel = 0 (%.2g); |head| max %.4f normalised" % (
        d1, d2, len(S), int(off.sum()), zmax, np.abs(hn).max()), flush=True)
    # ---- 3. parameter tree = RAA-c's; sha256
    ta = {k: tuple(v.shape) for k, v in tu.flatten_dict(nnx.state(nnx.eval_shape(cc.model.create, jax.random.key(0))).to_pure_dict(), sep="/").items()}
    tb = {k: tuple(v.shape) for k, v in tu.flatten_dict(nnx.state(nnx.eval_shape(ch.model.create, jax.random.key(0))).to_pure_dict(), sep="/").items()}
    assert ta == tb, (set(ta) ^ set(tb))
    fa = set(tu.flatten_dict(nnx.state(nnx.eval_shape(cc.model.create, jax.random.key(0)), cc.freeze_filter).to_pure_dict(), sep="/"))
    fb = set(tu.flatten_dict(nnx.state(nnx.eval_shape(ch.model.create, jax.random.key(0)), ch.freeze_filter).to_pure_dict(), sep="/"))
    assert fa == fb and any("state_proj" in k for k in set(tb) - fb), "state_proj must be trainable"
    sha = hashlib.sha256(open(RH.NPZ, "rb").read()).hexdigest()
    fit = json.load(open(os.path.join(ROOT, "results", "flawed", "raah", "fit.json"), encoding="utf-8"))
    assert sha == fit["npz_sha256"], (sha, fit["npz_sha256"])
    print("[raah-check] 3 ok: same %d parameter tensors, same %d frozen as RAA-c, state_proj trainable; npz sha256 %s == fit.json" % (len(ta), len(fa), sha), flush=True)
    # ---- 1b. Pi0H(linear head) reproduces Pi0S on loss and samples (same init, same noise)
    from openpi.models import model as _m                                       # noqa: F401
    obs_spec, act_spec = cs.model.inputs_spec(batch_size=4)
    obs = jax.tree.map(lambda s: jnp.ones(s.shape, s.dtype) if s.dtype in (jnp.float32, jnp.bool_) else jnp.zeros(s.shape, s.dtype), obs_spec)
    obs = dataclasses.replace(obs, state=jnp.asarray(S[:4][[0, 0, 0, 0]] if False else S[np.flatnonzero(raw[:, 3] == 2.0)[:4]]))
    obs = dataclasses.replace(obs, tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask))
    acts = jax.random.normal(jax.random.key(3), (4, cs.model.action_horizon, 32), jnp.float32)
    noise = jax.random.normal(jax.random.key(4), (4, cs.model.action_horizon, 32), jnp.float32)
    out = {}
    for tag, cfg in (("S", cs.model), ("H", cfg_lin)):
        mdl = cfg.create(jax.random.key(0))
        mdl.eval()
        out[tag] = (np.asarray(mdl.sample_actions(jax.random.key(5), obs, noise=noise), np.float32), np.asarray(mdl.compute_loss(jax.random.key(6), obs, acts, train=False), np.float32))
        del mdl
    dsm = np.abs(out["S"][0] - out["H"][0]).max()
    dls = np.abs(out["S"][1] - out["H"][1]).max()
    assert dsm < 1e-5 and dls < 1e-5, (dsm, dls)
    print("[raah-check] 1b ok: Pi0H with the linear-form head == Pi0S: sample max |diff| %.2g, loss %.2g" % (dsm, dls), flush=True)
    print("[raah-check] done", flush=True)


if __name__ == "__main__":
    main()

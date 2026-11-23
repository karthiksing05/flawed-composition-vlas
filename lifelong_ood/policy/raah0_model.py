"""NPH (docs/notes/plan_necessity_power.md 3.2 step 3, tag nph): pi0 (Pi0ZS family, raas0_model.py) with the servo slot filled by the frozen goal-directed learned where-head H*.

    action_xyz(row j) = H*_j(rel_xyz) * m + expert_xyz;   rotation and gripper: expert only

Copy of raah_model.py with (a) the head NOT centred (its zero is learned; raah0_head.npz carries centred = 0), and (b) the Pi0ZS base (openpi pi0, pi05 = False, native state_proj) instead of Pi0S.
Same state un-normalisation as raas_model.servo_norm (pi0's quantile stats are in the config's st_q01 / st_q99); m = 1 iff the un-normalised role id exceeds 1.5.  The head's weights are module-level
constants, not nnx parameters, so the parameter tree equals pi0-A0's.  servo_k = 0 keeps backbone.py's built-in servo loop inert; the head term is added by PolicyBackboneH0 (serving) and by
PI0PytorchZH / Pi0ZH (export / training).  An npz with key lin_K is the identity-test (linear-form) head: the servo itself.
"""
from __future__ import annotations

import dataclasses
import os

import jax.numpy as jnp
import numpy as np
import safetensors.torch
import torch

from . import raas_model as RS
from . import raas0_model as RZ

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ = os.path.join(ROOT, "lifelong_ood", "policy", "raah0_head.npz")
EPS, ROLE_TARGET = RS.EPS, RS.ROLE_TARGET
_CACHE = {}


def _load(path):
    if path not in _CACHE:
        z = np.load(path)
        _CACHE[path] = {k: np.asarray(z[k], np.float32) for k in z.files}
    return _CACHE[path]


def _centred(P):
    return float(P.get("centred", 0.0)) > 0.5


def head_raw_np(P, rel):
    """rel [b, 3] metres -> [b, 10, 3] raw action units (numpy float64 reference of the head)."""
    P = {k: np.asarray(v, np.float64) for k, v in P.items()}
    if "lin_K" in P:
        s = np.clip(float(P["lin_K"]) * np.asarray(rel, np.float64), -float(P["M"]), float(P["M"]))
        return np.repeat(s[:, None, :], 10, axis=1)

    def g(r):
        h = (r / float(P["scale"])) @ P["W1"] + P["b1"]
        h = h / (1.0 + np.exp(-h))
        h = h @ P["W2"] + P["b2"]
        h = h / (1.0 + np.exp(-h))
        return float(P["M"]) * np.tanh(h @ P["W3"] + P["b3"])
    out = g(np.asarray(rel, np.float64))
    if _centred(P):
        out = out - g(np.zeros((1, 3)))
    return np.clip(out, -float(P["M"]), float(P["M"])).reshape(-1, 10, 3)


def head_norm_np(cfg, state, action_dim):
    """state [b, >= 4] normalised -> [b, 10, action_dim] in normalised action units (numpy; float64 reference)."""
    P = _load(cfg.head_npz)
    q01, q99 = np.asarray(cfg.st_q01, np.float64), np.asarray(cfg.st_q99, np.float64)
    raw = (np.asarray(state[..., :4], np.float64) + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).astype(np.float64)
    s = head_raw_np(P, raw[..., :3]) * m[..., None, None]
    aq01, aq99 = np.asarray(cfg.act_q01, np.float64), np.asarray(cfg.act_q99, np.float64)
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return np.pad(sn, [(0, 0), (0, 0), (0, action_dim - 3)])


def _fwd(P, r, act, tanh, mm):
    h = act(mm(r / float(P["scale"]), P["W1"]) + P["b1"])
    h = act(mm(h, P["W2"]) + P["b2"])
    return float(P["M"]) * tanh(mm(h, P["W3"]) + P["b3"])


def head_norm_jax(cfg, state, action_dim):
    import jax
    P = {k: (jnp.asarray(v, jnp.float32) if k[0] in "Wb" else v) for k, v in _load(cfg.head_npz).items()}     # scalars stay numpy (static under jit)
    lin = "lin_K" in P
    q01, q99 = jnp.asarray(cfg.st_q01, jnp.float32), jnp.asarray(cfg.st_q99, jnp.float32)
    raw = (state[..., :4].astype(jnp.float32) + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).astype(jnp.float32)
    g = lambda r: _fwd(P, r, jax.nn.silu, jnp.tanh, lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST))                    # noqa: E731
    M = float(np.asarray(_load(cfg.head_npz)["M"]))
    if lin:
        s = jnp.repeat(jnp.clip(float(P["lin_K"]) * raw[..., :3], -M, M)[:, None, :], 10, axis=1) * m[:, None, None]
    else:
        out = g(raw[..., :3])
        if _centred(P):
            out = out - g(jnp.zeros((1, 3), jnp.float32))
        s = jnp.clip(out, -M, M).reshape(-1, 10, 3) * m[:, None, None]
    aq01, aq99 = jnp.asarray(cfg.act_q01, jnp.float32), jnp.asarray(cfg.act_q99, jnp.float32)
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return jnp.pad(sn, [(0, 0), (0, 0), (0, action_dim - 3)])


def head_norm_torch(cfg, state, action_dim):
    P = {k: torch.as_tensor(v, dtype=torch.float32, device=state.device) for k, v in _load(cfg.head_npz).items()}
    q01, q99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.st_q01, cfg.st_q99))
    raw = (state[..., :4].float() + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).float()
    g = lambda r: _fwd(P, r, torch.nn.functional.silu, torch.tanh, torch.matmul)   # noqa: E731
    M = float(_load(cfg.head_npz)["M"])
    if "lin_K" in P:
        s = torch.clamp(float(P["lin_K"]) * raw[..., :3], -M, M)[:, None, :].expand(-1, 10, 3) * m[:, None, None]
    else:
        out = g(raw[..., :3])
        if _centred(_load(cfg.head_npz)):
            out = out - g(torch.zeros((1, 3), dtype=torch.float32, device=state.device))
        s = torch.clamp(out, -M, M).reshape(-1, 10, 3) * m[:, None, None]
    aq01, aq99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.act_q01, cfg.act_q99))
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return torch.nn.functional.pad(sn, (0, action_dim - 3))


class Pi0ZH(RZ.Pi0ZS):
    def servo(self, observation):
        return head_norm_jax(self.servo_cfg, observation.state, self.action_dim)


@dataclasses.dataclass(frozen=True)
class Pi0ConfigZH(RZ.Pi0ConfigZS):
    head_npz: str = NPZ

    def create(self, rng):
        import flax.nnx as nnx
        return Pi0ZH(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchZH(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchZH(RZ.PI0PytorchZS):
    """PI0PytorchZS adds servo_norm_torch (zero here: servo_k = 0); this adds the head term."""

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        out = super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        return out + head_norm_torch(self.config, observation.state, out.shape[-1]).to(out.dtype)

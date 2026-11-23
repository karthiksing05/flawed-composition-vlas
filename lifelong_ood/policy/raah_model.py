"""RAA-h (docs/notes/plan_review_atlas.md 6, tag rvh): RAA-sL's model with the fixed-form servo replaced by a LEARNED separate where-head.

    action_xyz(row j) = H_j(rel_xyz) * m + expert_xyz;   rotation and gripper: expert only

H is a tiny MLP 3 -> 64 -> 64 -> 30 (10 chunk rows x 3, SiLU), fit ONCE on CPU by least squares of the demos' action xyz(t + j) on rel(t) over the held rows of K1-K6
(experiments/raah_fit.py -> raah_head.npz), then FROZEN: its weights are module-level constants loaded from the npz (not nnx parameters), so the parameter tree equals RAA-c's.
Output M tanh(u), centred so H(0) = 0 (g(r) - g(0), clipped to +-M); m = 1 iff the un-normalised role id exceeds 1.5 (target, while held); the same state un-normalisation as
raas_model.servo_norm.  The config keeps servo_k = 0 so backbone.py's built-in servo loop adds nothing; the head term is added by PolicyBackboneH (serving) and by
PI0PytorchH / Pi0H (export / training).  raas_model.py is not edited.
"""
from __future__ import annotations

import dataclasses
import os

import jax.numpy as jnp
import numpy as np
import safetensors.torch
import torch

from . import raas_model as RS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ = os.path.join(ROOT, "lifelong_ood", "policy", "raah_head.npz")
EPS, ROLE_TARGET = RS.EPS, RS.ROLE_TARGET
_CACHE = {}


def _load(path):
    if path not in _CACHE:
        z = np.load(path)
        _CACHE[path] = {k: np.asarray(z[k], np.float32) for k in z.files}
    return _CACHE[path]


def head_raw_np(P, rel):
    """rel [b, 3] metres -> [b, 10, 3] raw action units (numpy float64 reference of the head)."""
    P = {k: np.asarray(v, np.float64) for k, v in P.items()}
    if "lin_K" in P:                                                         # identity-test npz (raah_check): the servo itself, constant over the 10 rows
        s = np.clip(float(P["lin_K"]) * np.asarray(rel, np.float64), -float(P["M"]), float(P["M"]))
        return np.repeat(s[:, None, :], 10, axis=1)
    def g(r):
        h = (r / float(P["scale"])) @ P["W1"] + P["b1"]
        h = h / (1.0 + np.exp(-h))
        h = h @ P["W2"] + P["b2"]
        h = h / (1.0 + np.exp(-h))
        return float(P["M"]) * np.tanh(h @ P["W3"] + P["b3"])
    out = g(np.asarray(rel, np.float64)) - g(np.zeros((1, 3)))
    return np.clip(out, -float(P["M"]), float(P["M"])).reshape(-1, 10, 3)


def _norm_consts(cfg):
    return (np.asarray(cfg.act_q01, np.float32), np.asarray(cfg.act_q99, np.float32))


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


def _fwd(P, r, act, tanh, xp, mm=None):
    mm = mm or xp.matmul
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
    g = lambda r: _fwd(P, r, jax.nn.silu, jnp.tanh, jnp, lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST))                    # noqa: E731
    M = float(np.asarray(_load(cfg.head_npz)["M"]))
    if lin:
        s = jnp.repeat(jnp.clip(float(P["lin_K"]) * raw[..., :3], -M, M)[:, None, :], 10, axis=1) * m[:, None, None]
    else:
        out = g(raw[..., :3]) - g(jnp.zeros((1, 3), jnp.float32))
        s = jnp.clip(out, -M, M).reshape(-1, 10, 3) * m[:, None, None]
    aq01, aq99 = jnp.asarray(cfg.act_q01, jnp.float32), jnp.asarray(cfg.act_q99, jnp.float32)
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return jnp.pad(sn, [(0, 0), (0, 0), (0, action_dim - 3)])


def head_norm_torch(cfg, state, action_dim):
    P = {k: torch.as_tensor(v, dtype=torch.float32, device=state.device) for k, v in _load(cfg.head_npz).items()}
    q01, q99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.st_q01, cfg.st_q99))
    raw = (state[..., :4].float() + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).float()
    g = lambda r: _fwd(P, r, torch.nn.functional.silu, torch.tanh, torch)   # noqa: E731
    M = float(_load(cfg.head_npz)["M"])
    if "lin_K" in P:
        s = torch.clamp(float(P["lin_K"]) * raw[..., :3], -M, M)[:, None, :].expand(-1, 10, 3) * m[:, None, None]
    else:
        out = g(raw[..., :3]) - g(torch.zeros((1, 3), dtype=torch.float32, device=state.device))
        s = torch.clamp(out, -M, M).reshape(-1, 10, 3) * m[:, None, None]
    aq01, aq99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.act_q01, cfg.act_q99))
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return torch.nn.functional.pad(sn, (0, action_dim - 3))


class Pi0H(RS.Pi0S):
    def servo(self, observation):
        return head_norm_jax(self.servo_cfg, observation.state, self.action_dim)


@dataclasses.dataclass(frozen=True)
class Pi0ConfigH(RS.Pi0ConfigS):
    head_npz: str = NPZ

    def create(self, rng):
        import flax.nnx as nnx
        return Pi0H(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchH(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchH(RS.PI0PytorchS):
    """PI0PytorchS adds servo_norm_torch (zero here: servo_k = 0); this adds the head term."""

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        out = super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        return out + head_norm_torch(self.config, observation.state, out.shape[-1]).to(out.dtype)

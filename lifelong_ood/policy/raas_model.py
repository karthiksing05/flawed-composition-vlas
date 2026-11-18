"""RAA-s (docs/notes/raas_plan.md, SR 78): RAA-c's model plus a SERVO RESIDUAL action head (translation dims only).

    action_xyz = clip(K * rel_xyz, -M, +M) * m + expert_xyz;   rotation and gripper: expert only

rel_xyz = the first three numbers of the continuous channel state [dx, dy, dz (metres), role id]; m = 1 iff the role id is 2 (target, while held).  Both are read
from obs.state INSIDE the model: the state arrives quantile-normalised and zero-padded to 32 (openpi), so it is un-normalised with the dataset's own stats (the inverse
of openpi's Normalize, (x + 1) / 2 * (q99 - q01 + 1e-6) + q01), then the servo is mapped into the normalised action space, where openpi's quantile Normalize is affine
and the offset cancels: servo_norm = s * 2 / (aq99 - aq01 + 1e-6) on dims 0-2.  The flow-matching loss is trained on (action - servo_norm); sampling returns the
denoised residual + servo_norm.  The servo is a fixed function (K, M from results/flawed/raas_gate.json, the offline phase-0 gate; no trained parameter), the same
parameter tree as RAA-c (state_proj only is new relative to pi0.5).  raac_model.py is not edited: Pi0S / PI0PytorchS subclass Pi0C / PI0PytorchC and with servo_k = 0
reproduce them exactly (experiments/raas_check.py).

The PyTorch model (serving) overrides sample_actions only; training is JAX (c_train_pi05.py).  Export = raac_export.py (the servo constants live in the config).
"""
from __future__ import annotations

import dataclasses
import json
import os

import jax.numpy as jnp
import safetensors.torch
import torch

from . import raac_model as RM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = os.path.join(ROOT, "results", "flawed", "raas_gate.json")
NORM = "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json"
EPS = 1e-6                                                       # openpi's quantile Normalize epsilon
ROLE_TARGET = 1.5                                                # m = 1 iff the un-normalised role id exceeds this (target = 2, object = 1, none = 0)


def servo_fields(k=None, max_step=None):
    """The Pi0ConfigS servo fields from the phase-0 gate file (K, max_step) and the RAA-c dataset's norm stats (state q01 / q99 [4], action q01 / q99 [3])."""
    g = json.load(open(GATE, encoding="utf-8"))
    assert g["gate_pass"], "raas_gate.json: gate not passed"
    ns = json.load(open(NORM, encoding="utf-8"))["norm_stats"]
    return dict(servo_k=float(g["K"] if k is None else k), servo_max=float(g["max_step"] if max_step is None else max_step),
                st_q01=tuple(float(v) for v in ns["state"]["q01"][:4]), st_q99=tuple(float(v) for v in ns["state"]["q99"][:4]),
                act_q01=tuple(float(v) for v in ns["actions"]["q01"][:3]), act_q99=tuple(float(v) for v in ns["actions"]["q99"][:3]))


def servo_norm_jax(cfg, state, action_dim):
    """state [b, 32] normalised -> [b, 1, action_dim] servo in normalised action units (zeros off dims 0-2 and when m = 0)."""
    q01, q99 = jnp.asarray(cfg.st_q01, jnp.float32), jnp.asarray(cfg.st_q99, jnp.float32)
    raw = (state[..., :4].astype(jnp.float32) + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).astype(jnp.float32)
    s = jnp.clip(cfg.servo_k * raw[..., :3], -cfg.servo_max, cfg.servo_max) * m[..., None]
    aq01, aq99 = jnp.asarray(cfg.act_q01, jnp.float32), jnp.asarray(cfg.act_q99, jnp.float32)
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return jnp.pad(sn, [(0, 0), (0, action_dim - 3)])[:, None, :]


def servo_norm_torch(cfg, state, action_dim):
    q01, q99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.st_q01, cfg.st_q99))
    raw = (state[..., :4].float() + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).float()
    s = torch.clamp(cfg.servo_k * raw[..., :3], -cfg.servo_max, cfg.servo_max) * m[..., None]
    aq01, aq99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.act_q01, cfg.act_q99))
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return torch.nn.functional.pad(sn, (0, action_dim - 3))[:, None, :]


class Pi0S(RM.Pi0C):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.servo_cfg = config                                  # a frozen dataclass of floats / tuples: static for nnx

    def servo(self, observation):
        return servo_norm_jax(self.servo_cfg, observation.state, self.action_dim)

    def compute_loss(self, rng, observation, actions, *, train=False):
        return super().compute_loss(rng, observation, actions - self.servo(observation).astype(actions.dtype), train=train)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        out = super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        return out + self.servo(observation).astype(out.dtype)


@dataclasses.dataclass(frozen=True)
class Pi0ConfigS(RM.Pi0ConfigC):
    servo_k: float = 0.0
    servo_max: float = 1.0
    st_q01: tuple = (0.0, 0.0, 0.0, 0.0)
    st_q99: tuple = (1.0, 1.0, 1.0, 1.0)
    act_q01: tuple = (-1.0, -1.0, -1.0)
    act_q99: tuple = (1.0, 1.0, 1.0)

    def create(self, rng):
        import flax.nnx as nnx
        return Pi0S(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchS(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchS(RM.PI0PytorchC):
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        out = super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        sv = servo_norm_torch(self.config, observation.state, out.shape[-1])
        return out + sv.to(out.dtype)

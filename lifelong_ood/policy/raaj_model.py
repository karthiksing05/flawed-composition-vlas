"""NPHJ (tag nphj, docs/notes/nphj_results.md): pi0 (Pi0ZS family, raas0_model.py) with a JOINTLY TRAINED where-branch in the servo slot.

    action_xyz(row j) = J_j(rel_xyz) * m + expert_xyz;   rotation and gripper: expert only

J = a small MLP 3 -> 64 -> 64 -> 30 (SiLU) on the same token input as H* (raah0_model: un-normalised object-to-target offset rel = raw state[:3], role gate m = 1 iff raw role id > 1.5,
input rel / H*'s scale), output reshaped to 10 rows x 3 dims, scaled by H*'s output bound M (a fixed output scale, as in H*; no tanh, no clip), converted to normalised action units exactly as H*.
Its parameters are trainable nnx parameters (path where_head/l{1,2,3}/{kernel,bias}), trained end to end by the flow loss with the expert (the loss is on actions - J(state); J is added back at
sampling).  No servo target, no distillation.  The last layer is zero-initialised, so at step 0 J = 0 and the model is pi0-A0 exactly (servo_k = 0).
Torch side: PI0PytorchZJ carries torch where_head (keys where_head.l{1,2,3}.{weight,bias}); experiments/raaj_export.py adds them to the converted safetensors.
"""
from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import safetensors.torch
import torch

import flax.nnx as nnx

from . import raas_model as RS
from . import raas0_model as RZ
from . import raah0_model as RH0

EPS, ROLE_TARGET = RS.EPS, RS.ROLE_TARGET
HID = 64


def head_consts():
    """(scale, M) of H*'s npz: the input scale and the output bound, used here only as FIXED constants of the branch's input / output scaling."""
    z = np.load(RH0.NPZ)
    return float(z["scale"]), float(z["M"])


class WhereHead(nnx.Module):
    def __init__(self, rngs):
        P = jax.lax.Precision.HIGHEST
        self.l1 = nnx.Linear(3, HID, precision=P, rngs=rngs)
        self.l2 = nnx.Linear(HID, HID, precision=P, rngs=rngs)
        self.l3 = nnx.Linear(HID, 30, precision=P, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros, rngs=rngs)

    def __call__(self, x):
        h = jax.nn.silu(self.l1(x))
        h = jax.nn.silu(self.l2(h))
        return self.l3(h)


def _gate_rel_jax(cfg, state):
    q01, q99 = jnp.asarray(cfg.st_q01, jnp.float32), jnp.asarray(cfg.st_q99, jnp.float32)
    raw = (state[..., :4].astype(jnp.float32) + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    return raw[..., :3], (raw[..., 3] > ROLE_TARGET).astype(jnp.float32)


def _to_norm_jax(cfg, s, action_dim):
    aq01, aq99 = jnp.asarray(cfg.act_q01, jnp.float32), jnp.asarray(cfg.act_q99, jnp.float32)
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return jnp.pad(sn, [(0, 0), (0, 0), (0, action_dim - 3)])


def head_norm_jax(cfg, head, state, action_dim):
    """state [b, >= 4] normalised -> [b, 10, action_dim] normalised action units (head: WhereHead)."""
    scale, M = head_consts()
    rel, m = _gate_rel_jax(cfg, state)
    s = (M * head(rel / scale)).reshape(-1, 10, 3) * m[:, None, None]
    return _to_norm_jax(cfg, s, action_dim)


class Pi0ZJ(RZ.Pi0ZS):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.where_head = WhereHead(rngs)                  # created LAST: every other parameter has the init pi0-A0's model has under the same key

    def servo(self, observation):
        return head_norm_jax(self.servo_cfg, self.where_head, observation.state, self.action_dim)


# ---------------------------------------------------------------- torch
class WhereHeadT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = torch.nn.Linear(3, HID), torch.nn.Linear(HID, HID), torch.nn.Linear(HID, 30)

    def forward(self, x):
        f = torch.nn.functional
        x = x.float()
        h = f.silu(f.linear(x, self.l1.weight.float(), self.l1.bias.float()))
        h = f.silu(f.linear(h, self.l2.weight.float(), self.l2.bias.float()))
        return f.linear(h, self.l3.weight.float(), self.l3.bias.float())


def head_norm_torch(cfg, head, state, action_dim):
    scale, M = head_consts()
    q01, q99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.st_q01, cfg.st_q99))
    raw = (state[..., :4].float() + 1.0) / 2.0 * (q99 - q01 + EPS) + q01
    m = (raw[..., 3] > ROLE_TARGET).float()
    s = (M * head(raw[..., :3] / scale)).reshape(-1, 10, 3) * m[:, None, None]
    aq01, aq99 = (torch.tensor(v, dtype=torch.float32, device=state.device) for v in (cfg.act_q01, cfg.act_q99))
    sn = s * 2.0 / (aq99 - aq01 + EPS)
    return torch.nn.functional.pad(sn, (0, action_dim - 3))


@dataclasses.dataclass(frozen=True)
class Pi0ConfigZJ(RZ.Pi0ConfigZS):
    def create(self, rng):
        return Pi0ZJ(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchZJ(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchZJ(RZ.PI0PytorchZS):
    def __init__(self, config):
        super().__init__(config)
        self.where_head = WhereHeadT().float()

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        out = super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        return out + head_norm_torch(self.config, self.where_head, observation.state, out.shape[-1]).to(out.dtype)


# ---------------------------------------------------------------- weights <-> arrays
KEYS = ("l1", "l2", "l3")


def torch_state_from_jax(params):
    """params: the restored JAX params tree (pure dict, value-suffix removed or {'value': ..}) -> dict of torch tensors where_head.l*.{weight,bias}."""
    out = {}
    wh = params["where_head"]
    for k in KEYS:
        ker, b = wh[k]["kernel"], wh[k]["bias"]
        ker = np.array(ker["value"] if isinstance(ker, dict) else ker, np.float32)
        b = np.array(b["value"] if isinstance(b, dict) else b, np.float32)
        out["where_head.%s.weight" % k] = torch.from_numpy(ker.T.copy())
        out["where_head.%s.bias" % k] = torch.from_numpy(b.copy())
    return out


def jax_head_from_numpy(w):
    """w: dict where_head.l*.{weight,bias} (numpy, torch layout) -> a WhereHead (nnx) with those values."""
    h = WhereHead(nnx.Rngs(0))
    for k in KEYS:
        lin = getattr(h, k)
        lin.kernel.value = jnp.asarray(np.asarray(w["where_head.%s.weight" % k], np.float32).T)
        lin.bias.value = jnp.asarray(np.asarray(w["where_head.%s.bias" % k], np.float32))
    return h


def torch_head_from_numpy(w):
    t = WhereHeadT()
    t.load_state_dict({k.split("where_head.", 1)[1]: torch.as_tensor(np.asarray(v, np.float32)) for k, v in w.items()})
    return t.float()


# ---------------------------------------------------------------- weight loader
from openpi.models import model as _model                                    # noqa: E402
from openpi.shared import download as _download                              # noqa: E402
from openpi.training import weight_loaders as _W                             # noqa: E402


class RaajLoader(_W.CheckpointWeightLoader):
    """pi0_libero's params; the missing where_head parameters stay at the model's init (kernel 3 / 64 random, last layer zero)."""

    def load(self, params):
        loaded = _model.restore_params(_download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _W._merge_params(loaded, params, missing_regex=".*lora.*|.*where_head.*")

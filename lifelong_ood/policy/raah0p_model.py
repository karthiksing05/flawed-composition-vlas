"""NPHP (tag nphp): pi0.5 (Pi0S family, raas_model.py) with the servo slot filled by the frozen goal-directed learned where-head H* (raah0_head.npz, NOT centred).

    action_xyz(row j) = H*_j(rel_xyz) * m + expert_xyz;   rotation and gripper: expert only

The head functions are raah0_model's (raw-unit head, un-normalised with the config's st_q01 / st_q99 / act_q01 / act_q99, which servo_fields() fills from the RAA-c dataset norm stats for pi0.5 and pi0 alike).
Only the model classes are new: Pi0H0P / Pi0ConfigH0P / PI0PytorchH0P replace raah_model's Pi0H / Pi0ConfigH / PI0PytorchH (same structure; the head is uncentred).  servo_k = 0 keeps the built-in servo loop inert;
the head term is added by PolicyBackboneH0 (serving; it only needs head_npz and servo_k = 0) and by Pi0H0P / PI0PytorchH0P (training / export).  The head weights are module constants: the parameter tree equals A0's.
"""
from __future__ import annotations

import dataclasses

import safetensors.torch

from . import raas_model as RS
from .raah0_model import NPZ, head_norm_jax, head_norm_np, head_norm_torch, head_raw_np      # noqa: F401


class Pi0H0P(RS.Pi0S):
    def servo(self, observation):
        return head_norm_jax(self.servo_cfg, observation.state, self.action_dim)


@dataclasses.dataclass(frozen=True)
class Pi0ConfigH0P(RS.Pi0ConfigS):
    head_npz: str = NPZ

    def create(self, rng):
        import flax.nnx as nnx
        return Pi0H0P(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchH0P(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchH0P(RS.PI0PytorchS):
    """PI0PytorchS adds servo_norm_torch (zero here: servo_k = 0); this adds the head term."""

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        out = super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        return out + head_norm_torch(self.config, observation.state, out.shape[-1]).to(out.dtype)

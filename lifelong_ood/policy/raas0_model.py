"""PZ (docs/notes/plan_oracle_pi0.md 3.4): the servo residual head of raas_model.py on openpi's plain pi0 (Pi0Config pi05=False).

pi0 embeds obs.state with its OWN (trained) state_proj into suffix position 0; nothing is added to the parameter tree.  Pi0ZS is a copy of raas_model.Pi0S's three
methods on openpi.models.pi0.Pi0 (not on raac_model.Pi0C, which would add a second, random state_proj).  With servo_k = 0 it is openpi's Pi0 exactly
(experiments/pz_check.py).  The servo reuses raas_model.servo_norm_{jax,torch}, so it reads state[:4] after the dataset's QUANTILE norm: the config's data must force
use_quantile_norm = True (l90c_config._quantile_data).  Export = experiments/pi0z_export.py (openpi's converter handles pi0's state_proj).
"""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import safetensors.torch

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _cfg
import openpi.models_pytorch.pi0_pytorch as _pt

from . import raas_model as RS


class Pi0ZS(_pi0.Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.servo_cfg = config

    def servo(self, observation):
        return RS.servo_norm_jax(self.servo_cfg, observation.state, self.action_dim)

    def compute_loss(self, rng, observation, actions, *, train=False):
        return super().compute_loss(rng, observation, actions - self.servo(observation).astype(actions.dtype), train=train)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        out = super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        return out + self.servo(observation).astype(out.dtype)


@dataclasses.dataclass(frozen=True)
class Pi0ConfigZS(_cfg.Pi0Config):
    servo_k: float = 0.0
    servo_max: float = 1.0
    st_q01: tuple = (0.0, 0.0, 0.0, 0.0)
    st_q99: tuple = (1.0, 1.0, 1.0, 1.0)
    act_q01: tuple = (-1.0, -1.0, -1.0)
    act_q99: tuple = (1.0, 1.0, 1.0)

    def create(self, rng):
        return Pi0ZS(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchZS(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchZS(_pt.PI0Pytorch):
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        out = super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        sv = RS.servo_norm_torch(self.config, observation.state, out.shape[-1])
        return out + sv.to(out.dtype)

"""RAA-c (docs/SCAFFOLD_RESULTS.md 77): pi0.5 plus ONE new input projection, a continuous 'state' token for the role channel.

openpi's pi0 embeds `obs.state` with a linear `state_proj` into one suffix token; openpi's pi0.5 drops it (the state goes into the prompt as
text, or, with discrete_state_input=False as in l90c_config.py, is never read).  RAA-c turns the pi0 mechanism back on in the pi0.5 model,
unchanged otherwise: the suffix is [state token, action tokens]; the state token is causal as in pi0 (the action tokens attend to it,
it attends to the prefix and itself).  No openpi file is edited: the subclasses below override `create` / `load_pytorch` / `embed_suffix`.

  obs.state = the 4-number channel [dx, dy, dz (metres), role id] (0 none, 1 object, 2 target), quantile-normalised by the dataset's own
  norm stats, zero-padded to 32 by openpi's PadStatesAndActions.  state_proj: Linear(32, 1024) (the action expert's width), the only new
  parameters; initialised at random, trained with the action expert (the freeze filter leaves it trainable).

Pieces: Pi0ConfigC (config; JAX create and PyTorch load_pytorch), Pi0C (JAX), PI0PytorchC (PyTorch), RaacLoader (pi05_base weights, the
missing state_proj keeps its random init), plus the PyTorch export in experiments/raac_export.py.
"""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp
import safetensors.torch
import torch
from torch import nn

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _cfg
import openpi.models_pytorch.pi0_pytorch as _pt
import openpi.training.weight_loaders as _W
import openpi.shared.download as _download
import numpy as np


class Pi0C(_pi0.Pi0):
    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.state_proj = nnx.Linear(config.action_dim, _pi0._gemma.get_config(config.action_expert_variant).width, rngs=rngs)

    def embed_suffix(self, obs, noisy_actions, timestep):
        tokens, mask, ar, cond = super().embed_suffix(obs, noisy_actions, timestep)
        st = self.state_proj(obs.state)[:, None, :]
        tokens = jnp.concatenate([st.astype(tokens.dtype), tokens], axis=1)
        mask = jnp.concatenate([jnp.ones((tokens.shape[0], 1), dtype=jnp.bool_), mask], axis=1)
        ar = jnp.concatenate([jnp.array([True]), ar], axis=0)               # image/language do not attend to the state token
        return tokens, mask, ar, cond


@dataclasses.dataclass(frozen=True)
class Pi0ConfigC(_cfg.Pi0Config):
    def create(self, rng):
        return Pi0C(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path: str):
        model = PI0PytorchC(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model


class PI0PytorchC(_pt.PI0Pytorch):
    def __init__(self, config):
        super().__init__(config)
        self.state_proj = nn.Linear(config.action_dim, self.action_in_proj.out_features)

    def embed_suffix(self, state, noisy_actions, timestep):
        embs, pad, att, cond = super().embed_suffix(state, noisy_actions, timestep)
        st = self.state_proj(state.to(self.state_proj.weight.dtype))[:, None, :].to(embs.dtype)
        embs = torch.cat([st, embs], dim=1)
        pad = torch.cat([torch.ones(pad.shape[0], 1, dtype=pad.dtype, device=pad.device), pad], dim=1)
        att = torch.cat([torch.ones(att.shape[0], 1, dtype=att.dtype, device=att.device), att], dim=1)
        return embs, pad, att, cond


@dataclasses.dataclass(frozen=True)
class RaacLoader(_W.CheckpointWeightLoader):
    """CheckpointWeightLoader plus: the missing state_proj parameters stay at the model's random initialisation."""

    def load(self, params):
        loaded = _model.restore_params(_download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _W._merge_params(loaded, params, missing_regex=".*lora.*|.*state_proj.*")

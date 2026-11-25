"""NPHJ serving: copy of backbone_h0.py with the jointly trained where-branch (raaj_model).  Overrides `sample` ONLY."""
from __future__ import annotations

import json

import numpy as np
import torch

from .backbone import PolicyBackbone
from .raaj_model import head_norm_torch


class PolicyBackboneJ(PolicyBackbone):
    def __init__(self, *a, log_path=None, **kw):
        super().__init__(*a, **kw)
        assert float(getattr(self.model.config, "servo_k", 0.0) or 0.0) == 0.0, "NPHJ config must have servo_k = 0"
        assert hasattr(self.model, "where_head"), "not a PI0PytorchZJ"
        self.log_path = log_path

    @torch.no_grad()
    def sample(self, n, prefix, seed=0, num_steps=None, guide=None, step_gate=None, noise=None, full=False, start=None):
        x = super().sample(n, prefix, seed=seed, num_steps=num_steps, guide=guide, step_gate=step_gate, noise=noise, full=True, start=start)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(n, self.H, self.A).clone()
        hd = head_norm_torch(self.model.config, self.model.where_head, prefix.state, self.A)[..., :3]               # [b, 10, 3]
        xt[:, :, :3] = xt[:, :, :3] + hd.to(xt.dtype)
        y = xt.reshape(n, self.H * self.A).cpu().numpy().astype(x.dtype)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(head_abs_max=float(hd.abs().max()), head_row0=[float(v) for v in hd[0, 0]])) + "\n")
        return y if n >= 0 and full else y[:, self.real]

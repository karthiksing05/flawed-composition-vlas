"""SR 84 RAA-sL-W: test-time-only "where is owned by the module" serving of an RAA-s family checkpoint (docs/notes/raasw_plan.md).

PolicyBackboneW subclasses PolicyBackbone and overrides `sample` ONLY; backbone.py, raas_model.py and every existing config / arm are untouched, and the
serving scripts use this class only when `--where_own 1|2` is given (default 0 = the plain PolicyBackbone class, structurally the old path).

    where_own = 1 (W1):  while a target is held (role id 2, m = 1) the action's xy (dims 0, 1, all H chunk rows) are the module's alone:
                         xy = clip(K_w * rel_xy, -M, +M) (metres -> raw action units), NO expert contribution (the flow residual's xy is dropped).
                         z, rotation and gripper are exactly what PolicyBackbone.sample returns (expert residual + the existing proportional z servo).
    where_own = 2 (W2):  W1 plus a release gate: while m = 1 and |rel_xy| > REL_RELEASE (2 cm), a gripper command below 0 (open) is replaced by the
                         the closed command (raw +1; q-normalised).
    m = 0 (no role / object role): the returned chunk is bit-identical to PolicyBackbone.sample.

K_w = servo_max / D_SAT with D_SAT = 0.02 m: the command reaches the demos' own per-step clip (servo_max = M = .9375 raw action units, results/flawed/raas_gate.json,
pooled p99 of |demo action xyz| on held rows) at 2 cm from the target and stays there beyond; 46.875 per metre, 16.4 x the existing proportional K (2.851).
Fixed before any read; not tuned on any registered cell.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .backbone import PolicyBackbone
from .raas_model import EPS, NORM, ROLE_TARGET

D_SAT = 0.02            # m: distance at which the saturating controller reaches max_step
REL_RELEASE = 0.02      # m: W2 blocks a release while |rel_xy| exceeds this


def k_w(servo_max):
    return float(servo_max) / D_SAT


def _grip_norm():
    ns = json.load(open(NORM, encoding="utf-8"))["norm_stats"]["actions"]
    return float(ns["q01"][6]), float(ns["q99"][6])


def where_own_apply(x, state, cfg, mode, gq, A, H, kw=None, return_info=False):
    """x [n, H * A] float tensor = PolicyBackbone.sample(full=True) output; state [b, 32] normalised prefix state (b = 1 or n);
    cfg the servo config (st_q01 .., act_q01 .., servo_max); mode 1 | 2; gq = (q01, q99) of action dim 6.  -> new x (same shape), info dict."""
    assert mode in (1, 2), mode
    kw = k_w(cfg.servo_max) if kw is None else float(kw)
    dev = x.device
    q01, q99 = (torch.tensor(v, dtype=torch.float32, device=dev) for v in (cfg.st_q01, cfg.st_q99))
    raw = (state[..., :4].float().to(dev) + 1.0) / 2.0 * (q99 - q01 + EPS) + q01          # [b, 4]: dx, dy, dz (m), role id
    m = (raw[..., 3] > ROLE_TARGET)                                                     # [b]
    aq01, aq99 = (torch.tensor(v, dtype=torch.float32, device=dev) for v in (cfg.act_q01, cfg.act_q99))
    sat = torch.clamp(kw * raw[..., :2], -cfg.servo_max, cfg.servo_max)                  # [b, 2] raw action units
    sn = sat * 2.0 / (aq99[:2] - aq01[:2] + EPS)                                         # normalised action units (the affine offset cancels, as raas_model)
    n = x.shape[0]
    y = x.reshape(n, H, A).clone()
    mm = m.reshape(-1, 1, 1).expand(n, 1, 1) if m.numel() in (1, n) else None
    assert mm is not None, (m.shape, n)
    y[:, :, :2] = torch.where(mm, sn.reshape(-1, 1, 2).expand(n, H, 2).to(y.dtype), y[:, :, :2])
    relxy = torch.linalg.norm(raw[..., :2], dim=-1)                                      # [b] metres
    fired = torch.zeros((), dtype=torch.bool, device=dev)
    if mode == 2:
        g01, g99 = gq
        graw = (y[:, :, 6] + 1.0) / 2.0 * (g99 - g01 + EPS) + g01
        closed = (1.0 - g01) / (g99 - g01 + EPS) * 2.0 - 1.0                              # normalised value of the raw closed command +1 (demos: held-frame gripper mean .94-.96)
        gate = mm[:, :, 0] & (relxy.reshape(-1, 1).expand(n, 1) > REL_RELEASE) & (graw < 0.0)
        fired = gate.any()
        y[:, :, 6] = torch.where(gate, torch.full_like(y[:, :, 6], closed), y[:, :, 6])
    info = dict(m=bool(m.any()), rel_xy=float(relxy[0]), gate=bool(fired))
    out = y.reshape(n, H * A)
    return (out, info) if return_info else out


class PolicyBackboneW(PolicyBackbone):
    """PolicyBackbone with the SR 84 guarded branch after the servo.  `where_own` 1 | 2; log_path (optional) appends one jsonl line per sample call."""

    def __init__(self, *a, where_own=1, log_path=None, **kw):
        super().__init__(*a, **kw)
        assert where_own in (1, 2), where_own
        assert float(getattr(self.model.config, "servo_k", 0.0) or 0.0) != 0.0, "where_own needs a servo config (RAA-s family)"
        self.where_own = int(where_own)
        self.log_path = log_path
        self._gq = _grip_norm()
        assert self._gq[0] < 0.0 < self._gq[1], self._gq          # open (-1) and closed (+1) both inside the quantile range

    @torch.no_grad()
    def sample(self, n, prefix, seed=0, num_steps=None, guide=None, step_gate=None, noise=None, full=False, start=None):
        x = super().sample(n, prefix, seed=seed, num_steps=num_steps, guide=guide, step_gate=step_gate, noise=noise, full=True, start=start)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        yt, info = where_own_apply(xt, prefix.state, self.model.config, self.where_own, self._gq, self.A, self.H, return_info=True)
        y = yt.cpu().numpy().astype(x.dtype)
        if self.log_path:
            yy = y.reshape(-1, self.H, self.A)[0]
            xx = x.reshape(-1, self.H, self.A)[0]
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(info, mode=self.where_own, exp_xy=[float(v) for v in xx[:5, :2].mean(0)], w_xy=[float(v) for v in yy[:5, :2].mean(0)],
                                         g_old=float(xx[:5, 6].mean()), g_new=float(yy[:5, 6].mean()))) + "\n")
        return y if full else y[:, self.real]

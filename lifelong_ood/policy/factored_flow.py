"""PROTOTYPE A (docs/FLAWED_COMPOSITION.md 5.5): a flow-matching motor whose velocity is a FIXED, phase-gated SUM of per-role terms,
each reading only its own role's slot and acting in that slot's frame; and the comparison arms.  Stage 1 reads ORACLE slots
(experiments/fa1_common.scene: ground-truth poses, simple identity / shape features; roles from a template parse).

    v(a | o, instr) = sum_r g_r(phase) * T_r^-1 v_r(T_r a | slot_r, held object, proprio in frame_r)

  r = object: frame = the object's translation + yaw; reads the object slot; on while the commanded gripper is open (approach,
      grasp).
  r = target: frame = the target's translation + yaw; reads the target slot and the HELD object's geometry (its position in the
      target frame and its size; masked to 0 while the gripper is open; no identity); on while the gripper is closed (carry, place).
  T_r rotates every step's translation and rotation delta by -yaw_r (gripper unchanged): actions in the slot frame, mapped back.

Arms (one class, `arm`):
  H         holistic: one MLP on every slot (world frame, role flags) + the instruction's role code; world-frame actions
  R         OA-WAM-like: one fused MLP on the two role-routed slots and proprio, all in the END-EFFECTOR frame; EE-frame actions
  A         the factored motor above
  A_noframe A with world-frame terms (inputs and actions)
  A_nogate  A with both terms always on (the sum still fixed)
  A_nosum   A's role-local inputs (both frames, both chunks) into one learned joint MLP; world-frame actions

Stage 1b arms (section 5.6; ARMS_B).  Every one also receives the PHYSICAL-CONTACT signal b["held"] (the role object touches both
fingers, from simulator state; experiments/fa1b_common.py) and the grasp offset b["off"] (held object minus end effector, world
frame, captured at the grasp, 0 while not held):
  Hb        H + (held flag, offset) in its proprio (world frame)
  Rb        R + (held flag, offset) in its proprio (end-effector frame)
  Ab        A with the gate on CONTACT (c = held) and the target term's held-object geometry masked by held and extended by the
            offset in the target frame (still no held-object identity)
  Ab_cmd    Ab with stage 1's gate (c = last gripper command closed); held features as Ab   (ablates the contact gate)
  Ab_nooff  Ab without the offset                                                          (ablates the offset)
  Ab_nosum  Ab's role-local inputs (+ held flag) into one learned joint MLP                (the joint-head control)
Stage 1's arms are unchanged (their checkpoints load and run as before).

Rectified flow: x_t = (1 - t) x0 + t x1, target x1 - x0, t ~ U(0, 1); sampling = Euler from N(0, I).  Works in torch 1.11 (the
LIBERO venv, rollouts on CPU) and 2.x (training).  Weights travel as npz (save / load).
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn

N_ENT, DRAWER = 9, 7                    # fa1_common.ENT order; index of wooden_cabinet_1_top_region (carries the drawer opening)
POS_SCALE, GRIP_SCALE = 5.0, 25.0
ARMS = ("H", "R", "A", "A_noframe", "A_nogate", "A_nosum")
ARMS_B = ("Hb", "Rb", "Ab", "Ab_cmd", "Ab_nooff", "Ab_nosum")
BASE = {"Hb": "H", "Rb": "R", "Ab": "A", "Ab_cmd": "A", "Ab_nooff": "A", "Ab_nosum": "A_nosum"}
WIDTH = {"H": 512, "R": 512, "A": 352, "A_noframe": 352, "A_nogate": 352, "A_nosum": 512,
         "Hb": 512, "Rb": 512, "Ab": 352, "Ab_cmd": 352, "Ab_nooff": 352, "Ab_nosum": 512}


def mlp(din, dout, width, depth=4):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


def temb(t, dim=32):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    a = t[:, None] * f[None] * 1000.0
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def rz(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)   # [B, 3, 3]


def chunk_to(Q, x):
    """Express a world-frame action chunk x [B, H, 7] in the frame with rotation Q [B, 3, 3] (vectors -> Q^T v)."""
    QT = Q.transpose(1, 2)
    p = torch.einsum("bij,bhj->bhi", QT, x[..., 0:3])
    r = torch.einsum("bij,bhj->bhi", QT, x[..., 3:6])
    return torch.cat([p, r, x[..., 6:7]], -1)


def chunk_from(Q, x):
    p = torch.einsum("bij,bhj->bhi", Q, x[..., 0:3])
    r = torch.einsum("bij,bhj->bhi", Q, x[..., 3:6])
    return torch.cat([p, r, x[..., 6:7]], -1)


class FactoredFlow(nn.Module):
    def __init__(self, arm, horizon=8, center=(0.0, 0.0, 0.0), size=None):
        super().__init__()
        assert arm in ARMS + ARMS_B, arm
        self.arm, self.H = arm, horizon
        self.b = arm in ARMS_B                                  # stage 1b: contact flag and grasp offset available
        self.base = BASE.get(arm, arm)
        self.gate_contact = arm in ("Ab", "Ab_nooff", "Ab_nosum")
        self.offset = self.b and arm != "Ab_nooff"
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("size", torch.tensor(size if size is not None else np.zeros((N_ENT, 2)), dtype=torch.float32))
        self.register_buffer("onehot", torch.eye(N_ENT))
        da, dt, w = 7 * horizon, 32, WIDTH[arm]
        dprop = 3 + 6 + 2 + 1                                   # ee position, ee rotation (2 columns), finger qpos, last gripper cmd
        dslot = 3 + 2 + 2 + N_ENT + 1                           # position, (sin, cos) yaw, size, identity, drawer opening
        dhold = 3 + 2 + 2 + (3 if self.offset else 0)           # held object's position + yaw in the target frame, its size (+ offset)
        if self.base in ("H", "R") and self.b:
            dprop += 1 + 3                                      # held flag, grasp offset (in the arm's own frame)
        arm = self.base
        if arm == "H":
            self.net = mlp(dprop + N_ENT * (dslot + 2) + 2 * N_ENT + da + dt, da, w)
        elif arm == "R":
            self.net = mlp(dprop + 1 + 2 * dslot + da + dt, da, w)  # + world height of the ee
        elif arm == "A_nosum":
            self.net = mlp(2 * (dprop + dslot) + dhold + 1 + 2 * da + dt, da, w)
        else:
            self.obj = mlp(dprop + dslot + da + dt, da, w)
            self.tgt = mlp(dprop + dslot + dhold + da + dt, da, w)

    # ---- features -------------------------------------------------------------------------------------------------------------
    def _prop(self, b, Q, o):
        QT = Q.transpose(1, 2)
        p = torch.einsum("bij,bj->bi", QT, b["ee_pos"] - o) * POS_SCALE
        R = torch.einsum("bij,bjk->bik", QT, b["ee_rot"])[:, :, :2].reshape(-1, 6)
        out = [p, R, b["grip"] * GRIP_SCALE, b["prev"][:, None]]
        if self.b and self.base in ("H", "R"):
            out += [b["held"][:, None], torch.einsum("bij,bj->bi", QT, b["off"]) * POS_SCALE]
        return torch.cat(out, -1)

    def _slot(self, b, e, Q, o, yaw0):
        """Slot e [B] in the frame (Q, o) whose yaw is yaw0 [B]."""
        idx = torch.arange(len(e), device=e.device)
        p = b["pos"][idx, e]
        QT = Q.transpose(1, 2)
        pr = torch.einsum("bij,bj->bi", QT, p - o) * POS_SCALE
        dy = b["yaw"][idx, e] - yaw0
        aux = b["drawer"] * (e == DRAWER).float()
        return torch.cat([pr, torch.sin(dy)[:, None], torch.cos(dy)[:, None], self.size[e], self.onehot[e], aux[:, None] * 10.0], -1)

    def _frame(self, b, e, world):
        B = len(e)
        idx = torch.arange(B, device=e.device)
        if world:
            return torch.eye(3, device=e.device).expand(B, 3, 3), self.center.expand(B, 3), torch.zeros(B, device=e.device)
        yaw = b["yaw"][idx, e]
        return rz(yaw), b["pos"][idx, e], yaw

    def _held(self, b, Qt, ot, yt):
        closed = (b["held"] if self.b else (b["prev"] > 0).float())[:, None]
        s = self._slot(b, b["oi"], Qt, ot, yt)
        parts = [s[:, 0:3], s[:, 3:5], s[:, 5:7]]
        if self.offset:
            parts.append(torch.einsum("bij,bj->bi", Qt.transpose(1, 2), b["off"]) * POS_SCALE)
        return closed * torch.cat(parts, -1)

    def _gate(self, b):
        return b["held"] if self.gate_contact else (b["prev"] > 0).float()

    def velocity(self, b, x, t):
        """b: batch dict of tensors (ee_pos [B,3], ee_rot [B,3,3], grip [B,2], prev [B], pos [B,9,3], yaw [B,9], drawer [B],
        oi [B], ti [B] long); x [B, H, 7] the noisy world-frame chunk; t [B]."""
        B = x.shape[0]
        te = temb(t)
        flat = lambda z: z.reshape(B, -1)                                       # noqa: E731
        if self.base == "H":
            Q, o, y0 = self._frame(b, b["oi"], True)
            ents = torch.arange(N_ENT, device=x.device)
            slots = []
            for e in range(N_ENT):
                ee = ents[e].expand(B)
                flags = torch.stack([(b["oi"] == e).float(), (b["ti"] == e).float()], -1)
                slots.append(torch.cat([self._slot(b, ee, Q, o, y0), flags], -1))
            instr = torch.cat([self.onehot[b["oi"]], self.onehot[b["ti"]]], -1)
            v = self.net(torch.cat([self._prop(b, Q, o)] + slots + [instr, flat(x), te], -1))
            return v.reshape(B, self.H, 7)
        if self.base == "R":
            Q, o = b["ee_rot"], b["ee_pos"]
            y0 = torch.atan2(Q[:, 1, 0], Q[:, 0, 0])
            h = (b["ee_pos"][:, 2:3] - self.center[2]) * POS_SCALE
            c = torch.cat([self._prop(b, Q, o), h, self._slot(b, b["oi"], Q, o, y0), self._slot(b, b["ti"], Q, o, y0)], -1)
            v = self.net(torch.cat([c, flat(chunk_to(Q, x)), te], -1)).reshape(B, self.H, 7)
            return chunk_from(Q, v)
        world = self.arm == "A_noframe"
        Qo, oo, yo = self._frame(b, b["oi"], world)
        Qt, ot, yt = self._frame(b, b["ti"], world)
        co = torch.cat([self._prop(b, Qo, oo), self._slot(b, b["oi"], Qo, oo, yo)], -1)
        ct = torch.cat([self._prop(b, Qt, ot), self._slot(b, b["ti"], Qt, ot, yt), self._held(b, Qt, ot, yt)], -1)
        closed = self._gate(b)
        if self.base == "A_nosum":
            v = self.net(torch.cat([co, ct, closed[:, None], flat(chunk_to(Qo, x)), flat(chunk_to(Qt, x)), te], -1))
            return v.reshape(B, self.H, 7)
        vo = chunk_from(Qo, self.obj(torch.cat([co, flat(chunk_to(Qo, x)), te], -1)).reshape(B, self.H, 7))
        vt = chunk_from(Qt, self.tgt(torch.cat([ct, flat(chunk_to(Qt, x)), te], -1)).reshape(B, self.H, 7))
        if self.arm == "A_nogate":
            return vo + vt
        g = closed[:, None, None]
        return (1 - g) * vo + g * vt

    def loss(self, b, a):
        B = a.shape[0]
        x0 = torch.randn_like(a)
        t = torch.rand(B, device=a.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
        return ((self.velocity(b, xt, t) - (a - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, b, steps=10, generator=None):
        B = b["pos"].shape[0]
        x = torch.randn(B, self.H, 7, generator=generator)
        for k in range(steps):
            t = torch.full((B,), k / steps)
            x = x + self.velocity(b, x, t) / steps
        return x

    # ---- io -------------------------------------------------------------------------------------------------------------------
    def save(self, path, meta):
        np.savez(path, __meta__=json.dumps(dict(meta, arm=self.arm, horizon=self.H)),
                 **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        m = cls(meta["arm"], meta["horizon"])
        m.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "__meta__"})
        m.eval()
        return m, meta


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def batch_from_scene(sc, oi, ti, prev, held=0.0, off=(0.0, 0.0, 0.0)):
    """One oracle scene (fa1_common.scene) -> a batch of 1.  held / off: stage 1b's contact flag and grasp offset (ignored by
    stage 1's arms)."""
    f = lambda v: torch.tensor(np.asarray(v, np.float32))[None]               # noqa: E731
    return dict(ee_pos=f(sc["ee_pos"]), ee_rot=f(sc["ee_rot"]), grip=f(sc["grip"]), prev=f(prev).reshape(1), pos=f(sc["pos"]),
                yaw=f(sc["yaw"]), drawer=f(sc["drawer"]).reshape(1), oi=torch.tensor([oi]), ti=torch.tensor([ti]),
                held=f(float(held)).reshape(1), off=f(off))

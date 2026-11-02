"""PLAN_PARTS architecture P (docs/PLAN_PARTS.md section 4.1; protocol docs/PARTS_RESULTS.md section 3): a sparse PART LIBRARY on
prototype A's factored, frame-equivariant flow.

    v = (1 - c) T_o^-1 v_o(. | object part) + c T_c^-1 v_t(. | target part)        (Phi of PLAN_PARTS section 2, fixed; c = held)

Role backbones.  v_o is stage 1i's object term (fa1 arm Ai) and v_t its canonical target term, with every ATOM IDENTITY input removed
from the backbone: the object term reads proprio and the object slot's pose, yaw and hand-set size in the object's frame; the target
term reads the foot, the last command, the target's hand-set size and drawer opening, and the rotation since the hold's onset, in the
canonical place frame.  Neither reads the other role's atom.
Parts.  One per atom of a role: a slot embedding (EMB dims, an input of that role's backbone) and a rank-RANK LoRA (A x B, B = 0 at
init) on the backbone's four hidden Linear layers.  A task activates exactly its object's part in the object term and its target's
part in the target term; nothing else is atom-specific.  Parameters per part: EMB + RANK * sum over the 4 layers of (in + out).
None of stages 1l-1p's executor devices (occupancy, free space, the object-unit end-effector input) is present.
Oracle state, frames and the contact gate stay (Layer 3 removes them).
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn

from .factored_flow import DRAWER, GRIP_SCALE, N_ENT, POS_SCALE, TABLE, chunk_from, chunk_to, rz, temb

EMB, RANK, WIDTH = 16, 8, 352
OBJ_ATOMS = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"]
TGT_ATOMS = ["plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side", "wine_rack_1_top_region", "wooden_cabinet_1_top_region",
             "main_table_stove_front_region", "akita_black_bowl_1"]


class LoRALinear(nn.Module):
    """A frozen-able base Linear plus per-atom low-rank deltas; the atom of each row selects its delta."""

    def __init__(self, din, dout, n_atoms, rank=RANK):
        super().__init__()
        self.base = nn.Linear(din, dout)
        self.A = nn.Parameter(torch.randn(n_atoms, din, rank) / math.sqrt(din))
        self.B = nn.Parameter(torch.zeros(n_atoms, rank, dout))
        self.rank = rank

    def forward(self, x, atom):
        d = torch.einsum("bi,bir->br", x, self.A[atom])
        return self.base(x) + torch.einsum("br,bro->bo", d, self.B[atom])


class RoleNet(nn.Module):
    """Linear-SiLU x 4 - Linear; the 4 hidden Linear layers carry the parts' LoRA; the slot embedding is appended to the input."""

    def __init__(self, din, dout, n_atoms, width=WIDTH, rank=RANK):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n_atoms, EMB) * 0.1)
        dims = [din + EMB] + [width] * 4
        self.hidden = nn.ModuleList([LoRALinear(dims[i], dims[i + 1], n_atoms, rank) for i in range(4)])
        self.act = nn.SiLU()
        self.out = nn.Linear(width, dout)

    def forward(self, x, atom):
        h = torch.cat([x, self.emb[atom]], -1)
        for ly in self.hidden:
            h = self.act(ly(h, atom))
        return self.out(h)

    def part_params(self):
        return EMB + sum(ly.rank * (ly.base.in_features + ly.base.out_features) for ly in self.hidden)

    def backbone_parameters(self):
        return [p for ly in self.hidden for p in ly.base.parameters()] + list(self.out.parameters())

    def part_parameters(self):
        return [self.emb] + [p for ly in self.hidden for p in (ly.A, ly.B)]


class PartsFlow(nn.Module):
    arm = "P"

    def __init__(self, horizon=8, center=(0.0, 0.0, 0.0), size=None, rank=RANK, tsize=False, aframe=False, home=(0.0, 0.0, 0.0)):
        super().__init__()
        self.H, self.rank = horizon, rank
        self.tsize = bool(tsize)                                    # PARTS_RESULTS 12 (P_sz): the target term reads the held object's size
        # PARTS_RESULTS 14 (P_af, aframe=1): the object term's translation in the approach frame;
        # PARTS_RESULTS 16 (P_af2, aframe=2): translation AND rotation in the approach frame, the object's yaw relative to it as input
        self.aframe = int(aframe)
        self.register_buffer("home", torch.tensor(home, dtype=torch.float32))
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("size", torch.tensor(size if size is not None else np.zeros((N_ENT, 2)), dtype=torch.float32))
        da, dt = 7 * horizon, 32
        dprop = 3 + 6 + 2 + 1
        dslot = 3 + 2 + 2 + 1                                       # position, (sin, cos) yaw, size, drawer opening: no identity
        self.obj = RoleNet(dprop + dslot + da + dt, da, len(OBJ_ATOMS), rank=rank)
        self.tgt = RoleNet(3 + 1 + 2 + 1 + 6 + 2 * self.tsize + da + dt, da, len(TGT_ATOMS), rank=rank)
        # entity index -> part index (-1: no part)
        self.register_buffer("opart", torch.tensor([OBJ_ATOMS.index(e) if e in OBJ_ATOMS else -1 for e in _ENT()], dtype=torch.long))
        self.register_buffer("tpart", torch.tensor([TGT_ATOMS.index(e) if e in TGT_ATOMS else -1 for e in _ENT()], dtype=torch.long))

    # ---- features (stage 1i's, identity removed) ---------------------------------------------------------------------------------
    def _obj_in(self, b, Q, o, yaw0, Qp=None):
        idx = torch.arange(len(b["oi"]), device=b["oi"].device)
        QT = Q.transpose(1, 2)
        QpT = QT if Qp is None else Qp.transpose(1, 2)                      # translation frame (P_af: the approach frame)
        p = torch.einsum("bij,bj->bi", QpT, b["ee_pos"] - o) * POS_SCALE
        R = torch.einsum("bij,bjk->bik", QT, b["ee_rot"])[:, :, :2].reshape(-1, 6)
        e = b["oi"]
        dy = b["yaw"][idx, e] - yaw0
        aux = b["drawer"] * (e == DRAWER).float()
        slot = torch.cat([torch.zeros_like(p), torch.sin(dy)[:, None], torch.cos(dy)[:, None], self.size[e], aux[:, None] * 10.0], -1)
        return torch.cat([p, R, b["grip"] * GRIP_SCALE, b["prev"][:, None], slot], -1)

    def _canon(self, b):
        B = b["pos"].shape[0]
        idx = torch.arange(B, device=b["pos"].device)
        pt = b["pos"][idx, b["ti"]]
        Q = rz(b["cyaw"])
        QT = Q.transpose(1, 2)
        p = b["pos"][idx, b["oi"]].clone()
        p[:, 2] = p[:, 2] - b["zrest"] + b["pos"][:, TABLE, 2]
        pr = torch.einsum("bij,bj->bi", QT, p - pt) * POS_SCALE
        e = b["ti"]
        aux = b["drawer"] * (e == DRAWER).float()
        r0 = b["rot0"] if "rot0" in b else b["ee_rot"]
        rel = torch.einsum("bij,bkj->bik", b["ee_rot"], r0)
        drot = torch.einsum("bij,bjk,bkl->bil", QT, rel, Q)[:, :, :2].reshape(-1, 6)
        held_geom = [self.size[b["oi"]]] if self.tsize else []                 # the held object's footprint radius and height
        return Q, torch.cat([pr, b["prev"][:, None], self.size[e], aux[:, None] * 10.0, drot] + held_geom, -1)

    def velocity(self, b, x, t):
        B = x.shape[0]
        te = temb(t)
        flat = lambda z: z.reshape(B, -1)                                       # noqa: E731
        idx = torch.arange(B, device=x.device)
        Qo, oo, yo = rz(b["yaw"][idx, b["oi"]]), b["pos"][idx, b["oi"]], b["yaw"][idx, b["oi"]]
        if self.aframe == 2:                                        # one frame: positions, rotations and yaw relative to the bearing
            ay = torch.atan2(oo[:, 1] - self.home[1], oo[:, 0] - self.home[0])
            Qa = rz(ay)
            co = self._obj_in(b, Qa, oo, ay)
            Qo = Qa
        elif self.aframe:                                           # x-axis: the bearing from the home end effector to the object
            Qa = rz(torch.atan2(oo[:, 1] - self.home[1], oo[:, 0] - self.home[0]))
            co = self._obj_in(b, Qo, oo, yo, Qp=Qa)
        else:
            Qa = Qo
            co = self._obj_in(b, Qo, oo, yo)
        Qc, ct = self._canon(b)
        po, pt = self.opart[b["oi"]], self.tpart[b["ti"]]
        assert bool((po >= 0).all()) and bool((pt >= 0).all()), "an atom without a part"
        vo = _from2(Qa, Qo, self.obj(torch.cat([co, flat(_to2(Qa, Qo, x)), te], -1), po).reshape(B, self.H, 7))
        vt = chunk_from(Qc, self.tgt(torch.cat([ct, flat(chunk_to(Qc, x)), te], -1), pt).reshape(B, self.H, 7))
        g = b["held"][:, None, None]
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

    def save(self, path, meta):
        np.savez(path, __meta__=json.dumps(dict(meta, arm="P", horizon=self.H, rank=self.rank, tsize=self.tsize, aframe=self.aframe)),
                 **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        m = cls(meta["horizon"], rank=meta.get("rank", RANK), tsize=meta.get("tsize", False), aframe=meta.get("aframe", False))
        sd = {k: torch.from_numpy(z[k]) for k in z.files if k != "__meta__"}
        sd.setdefault("home", torch.zeros(3))                                   # checkpoints before P_af carry no home buffer
        m.load_state_dict(sd)
        m.eval()
        return m, meta


def _to2(Qp, Qr, x):
    """A world-frame chunk in a frame whose translation axes are Qp and rotation axes Qr (Qp = Qr: chunk_to)."""
    p = torch.einsum("bij,bhj->bhi", Qp.transpose(1, 2), x[..., 0:3])
    r = torch.einsum("bij,bhj->bhi", Qr.transpose(1, 2), x[..., 3:6])
    return torch.cat([p, r, x[..., 6:7]], -1)


def _from2(Qp, Qr, x):
    p = torch.einsum("bij,bhj->bhi", Qp, x[..., 0:3])
    r = torch.einsum("bij,bhj->bhi", Qr, x[..., 3:6])
    return torch.cat([p, r, x[..., 6:7]], -1)


def _ENT():
    return ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side",
            "wine_rack_1_top_region", "wooden_cabinet_1_top_region", "main_table_stove_front_region"]

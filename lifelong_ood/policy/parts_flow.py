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
Ladder L1 (docs/LADDER_RESULTS.md section 1; default off): with a checkpoint's meta "pgate", c is the proprioceptive event
ProprioHeld (finger width, last gripper command, hand height) instead of the sim-contact gate, in training rows and rollouts.
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


PG_LIFT = 0.01                                                      # fa1b_common.LIFT, the oracle rule's own constant
MV_ON, MV_STILL, MV_GAP = 0.02, 0.005, 15                           # SCAFFOLD_RESULTS 24: HandMoved's constants


class ProprioHeld:
    """Ladder L1 (LADDER_RESULTS 1.2): the held event from proprioception only, stepped once per step (per kept demo row).
    Inputs: grip = the two finger joint positions (width w = grip[0] - grip[1]), cmd = the last gripper command (> 0 closes),
    z = the hand's height.  Command open: off, close reference cleared.  Command closing: z_min = the lowest hand height since the
    close began; turns ON when w >= tw and z - z_min >= lift; turns OFF when w < tw.  Reads no object pose, contact or identity."""

    def __init__(self, tw, lift=PG_LIFT):
        self.tw, self.lift = float(tw), float(lift)
        self.on, self.zmin = False, None

    def step(self, grip, cmd, z):
        w, z = float(grip[0]) - float(grip[1]), float(z)
        if not float(cmd) > 0:
            self.on, self.zmin = False, None
            return False
        self.zmin = z if self.zmin is None else min(self.zmin, z)
        if not self.on and w >= self.tw and z - self.zmin >= self.lift:
            self.on = True
        elif self.on and w < self.tw:
            self.on = False
        return self.on


class HandMoved:
    """SCAFFOLD_RESULTS 24: the hand moves an entity with nothing held.  Stepped once per step on the entity's position and the last gripper
    command (> 0 closes).  With the command open: ON once the entity sits MV_ON from its rest position; OFF once it has moved under MV_STILL over
    the last MV_GAP steps, and that position becomes its rest.  The first closing command disarms it for the episode: from there ProprioHeld owns
    the object, and an object falling after a release is no push."""

    def __init__(self, p0):
        self.rest = np.asarray(p0, np.float64).copy()
        self.on, self.hist, self.armed = False, [], True

    def step(self, p, cmd):
        p = np.asarray(p, np.float64)
        if float(cmd) > 0:
            self.on, self.armed = False, False
        if not self.armed:
            return False
        if not self.on:
            if np.linalg.norm(p - self.rest) > MV_ON:
                self.on, self.hist = True, [p.copy()]
            return self.on
        self.hist.append(p.copy())
        if len(self.hist) > MV_GAP:
            self.hist.pop(0)
            if np.linalg.norm(p - self.hist[0]) < MV_STILL:
                self.on, self.hist, self.rest = False, [], p.copy()
        return self.on


def move_rows(key, pos, cmd):
    """HandMoved over each episode's rows in order (key = episode id per row; pos = the entity's position per row) -> bool per row."""
    out = np.zeros(len(key), bool)
    det = None
    for i in range(len(key)):
        if i == 0 or key[i] != key[i - 1]:
            det = HandMoved(pos[i])
        out[i] = det.step(pos[i], cmd[i])
    return out


def pgate_fit(grip, cmd):
    """LADDER_RESULTS 1.2's fixed rule on a build's demo rows: tw = (min width + 1st pct of the width on close-commanded rows) / 2."""
    w = np.asarray(grip, np.float64)[:, 0] - np.asarray(grip, np.float64)[:, 1]
    c = np.asarray(cmd) > 0
    return dict(tw=float(0.5 * (w.min() + np.quantile(w[c], 0.01))), lift=PG_LIFT, w_min=float(w.min()),
                w_close_q01=float(np.quantile(w[c], 0.01)))


def pgate_rows(key, grip, cmd, z, tw, lift=PG_LIFT):
    """The event per row, ProprioHeld stepped over each episode's rows in order (key = episode id per row) -> float32 0/1."""
    out = np.zeros(len(key), np.float32)
    det = None
    for i in range(len(key)):
        if i == 0 or key[i] != key[i - 1]:
            det = ProprioHeld(tw, lift)
        out[i] = float(det.step(grip[i], cmd[i], z[i]))
    return out


def role_lexicon(situations, rarest=False):
    """Ladder L4 (LADDER_RESULTS 5.2, steps 3-4).  situations: {instruction: [grasp slot, release slot]}.  -> dict(situations, ref
    {token: slot}, pg {token: share of its situations with g = ref}, pr likewise for r).  Reads no given role.
    rarest (SCAFFOLD_RESULTS 24): after the fixpoint, a slot its situation leaves unexplained goes to that situation's rarest contradicted token
    (no candidate left, so it would read as a function word), repeated to a fixpoint.  "stove" in "the front of the stove" no longer erases "stove"."""
    toks = {s: s.lower().split() for s in situations}
    words = sorted({w for v in toks.values() for w in v})
    has = {w: [s for s in situations if w in toks[s]] for w in words}
    cand = {w: set.intersection(*[set(situations[s]) for s in has[w]]) for w in words}
    ref = {w: next(iter(c)) for w, c in cand.items() if len(c) == 1}
    while True:                                                             # explaining away: simultaneous passes to a fixpoint
        new = {}
        for w in words:
            if w in ref or len(cand[w]) < 2:
                continue
            gone = set.intersection(*[{ref[u] for u in toks[s] if u != w and u in ref} for s in has[w]])
            left = cand[w] - gone
            if len(left) == 1:
                new[w] = next(iter(left))
        if not new:
            break
        ref.update(new)
    while rarest:
        pick = None
        for s in sorted(situations):
            left = set(situations[s]) - {ref[u] for u in toks[s] if u in ref}
            free = sorted({w for w in toks[s] if w not in ref and not cand[w]}, key=lambda w: (len(has[w]), w))
            if len(left) == 1 and free and (pick is None or len(has[free[0]]) < len(has[pick[0]])):
                pick = (free[0], next(iter(left)))
        if pick is None:
            break
        ref[pick[0]] = pick[1]
    pg = {w: sum(situations[s][0] == ref[w] for s in has[w]) / len(has[w]) for w in ref}
    pr = {w: sum(situations[s][1] == ref[w] for s in has[w]) / len(has[w]) for w in ref}
    return dict(situations={s: list(v) for s, v in situations.items()}, ref=ref, pg=pg, pr=pr)


def role_parse(lex, instruction):
    """Ladder L4 (LADDER_RESULTS 5.2, step 5): -> (object slot, target slot), or None when the instruction has no induced roles."""
    toks = [w for w in instruction.lower().split() if w in lex["ref"]]
    og = [w for w in toks if lex["pg"][w] > 0]
    if not og:
        return None
    best = max(lex["pg"][w] for w in og)
    obj = lex["ref"][next(w for w in og if lex["pg"][w] == best)]           # ties: the earliest token
    tg = [w for w in toks if lex["pr"][w] > 0 and lex["ref"][w] != obj]
    if not tg:
        return None
    best = max(lex["pr"][w] for w in tg)
    return obj, lex["ref"][[w for w in tg if lex["pr"][w] == best][-1]]     # ties: the latest token


class LoRALinear(nn.Module):
    """A frozen-able base Linear plus per-atom low-rank deltas; the atom of each row selects its delta."""

    def __init__(self, din, dout, n_atoms, rank=RANK):
        super().__init__()
        self.base = nn.Linear(din, dout)
        self.A = nn.Parameter(torch.randn(n_atoms, din, rank) / math.sqrt(din))
        self.B = nn.Parameter(torch.zeros(n_atoms, rank, dout))
        self.rank = rank

    def forward(self, x, atom, on=None):
        d = torch.einsum("bi,bir->br", x, self.A[atom])
        if on is not None:                                          # PARTS_RESULTS 41: rows with the part off get no delta
            d = d * on[:, None].to(d.dtype)
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

    def forward(self, x, atom, on=None):
        """on (PARTS_RESULTS 41): per-row bool, the part acts (None: every row).  Part off = slot embedding 0 and LoRA delta 0."""
        e = self.emb[atom]
        if on is not None:
            e = e * on[:, None].to(e.dtype)
        h = torch.cat([x, e], -1)
        for ly in self.hidden:
            h = self.act(ly(h, atom, on))
        return self.out(h)

    def part_params(self):
        return EMB + sum(ly.rank * (ly.base.in_features + ly.base.out_features) for ly in self.hidden)

    def backbone_parameters(self):
        return [p for ly in self.hidden for p in ly.base.parameters()] + list(self.out.parameters())

    def part_parameters(self):
        return [self.emb] + [p for ly in self.hidden for p in (ly.A, ly.B)]


class PartsFlow(nn.Module):
    arm = "P"

    def __init__(self, horizon=8, center=(0.0, 0.0, 0.0), size=None, rank=RANK, tsize=False, aframe=False, home=(0.0, 0.0, 0.0),
                 rclip=0.0, rtclip=0.0, hscale=0.0, psupport=False, rclip_p=None, rtclip_p=None, tscale=False, nmode=0, tmode=None,
                 gm=None, rtransport=False, rs_p=None, rrole=0.0, bbnom=False, bbnosize=False, tnear=False, rrel_p=None, lcomb=False):
        super().__init__()
        # SCAFFOLD_RESULTS 16 (T8a): the fixed rule replaced by a learned combination of the two role velocities, the hold and the flow time
        self.lcomb = bool(lcomb)
        # PARTS_RESULTS 50 (P_rel): with tscale, the carry scale frame leaves the target part's release region unstretched.  rrel_p
        # [target part] = its release radius rho (99th pct of its demos' object-to-target distance at each carry's last held row).
        # For d0 > r_t: phi(d) = d (d <= rho), rho + (d - rho)(r_t - rho)/(d0 - rho) (d > rho); the row's scale is phi(d)/d.
        self.tnear = bool(tnear)
        assert not self.tnear or tscale, "tnear needs tscale"
        self.register_buffer("rrel_p", torch.tensor(rrel_p if rrel_p is not None else np.zeros(len(TGT_ATOMS)), dtype=torch.float32))
        # PARTS_RESULTS 47 (P_nsz): with rtransport, the backbone-alone path reads no atom-specific slot attribute (the object's
        # hand-set size is 0 on backbone-only rows); part-on rows read it as before
        self.bbnosize = bool(bbnosize)
        assert not self.bbnosize or rtransport, "bbnosize needs rtransport"
        # PARTS_RESULTS 44 (P_nom; PLAN_PARTS 10.7): with rtransport, the backbone-alone path does not read m (its one-hot is 0)
        self.bbnom = bool(bbnom)
        assert not self.bbnom or rtransport, "bbnom needs rtransport"
        # PARTS_RESULTS 41 (P_rtr; PLAN_PARTS 10.6): role-level transport.  An object part acts only where the horizontal
        # hand-object distance is <= its start radius rs_p[part] (99th pct of its demos' start offsets); beyond it the object
        # backbone acts alone (part off), relocated onto the role's support radius rrole.  The per-part rclip is not used.
        self.rtransport = bool(rtransport)
        self.register_buffer("rs_p", torch.tensor(rs_p if rs_p is not None else np.zeros(len(OBJ_ATOMS)), dtype=torch.float32))
        self.register_buffer("rrole", torch.tensor(float(rrole), dtype=torch.float32))
        self.H, self.rank = horizon, rank
        self.tsize = bool(tsize)                                    # PARTS_RESULTS 12 (P_sz): the target term reads the held object's size
        # PARTS_RESULTS 14 (P_af, aframe=1): the object term's translation in the approach frame;
        # PARTS_RESULTS 16 (P_af2, aframe=2): translation AND rotation in the approach frame, the object's yaw relative to it as input
        self.aframe = int(aframe)
        self.register_buffer("home", torch.tensor(home, dtype=torch.float32))
        # PARTS_RESULTS 21 (P_rc): the object term's horizontal ee offset clipped to the demos' support radius (0: off)
        self.register_buffer("rclip", torch.tensor(float(rclip), dtype=torch.float32))
        # PARTS_RESULTS 24 (P_rt): the target term's horizontal object-to-target offset clipped to the demos' carry radius (0: off)
        self.register_buffer("rtclip", torch.tensor(float(rtclip), dtype=torch.float32))
        # PARTS_RESULTS 27 (P_hs): the object term in a horizontal scale frame frozen at the episode start,
        # s = min(1, hscale / |start offset_xy|); input offsets and chunk translations x s, output translation velocity / s (0: off)
        self.register_buffer("hscale", torch.tensor(float(hscale), dtype=torch.float32))
        # PARTS_RESULTS 30 (P_pp): each part clips with its OWN support radius (per object part / per target part; 0 = the scalar
        # library-wide radius).  The clips stay switched on by rclip / rtclip > 0.  psupport False: the scalar radii, as before.
        self.psupport = bool(psupport)
        # PARTS_RESULTS 33 (P_ts): the target term in a horizontal scale frame frozen at the hold's onset,
        # s_t = min(1, r_t(part) / d_onset) (b["d0t"]); replaces the carry's radial clip.  Needs rtclip > 0 (the radii).  False: off
        self.tscale = bool(tscale)
        # PARTS_RESULTS 38 (P_imd; PLAN_PARTS 10.5): the explicit interface.  nmode = K, the size of the grasp-mode inventory
        # (0: off).  tmode[target part] = the value of m that part emits (fitted from its own demos); the object term reads
        # one-hot(m).  gm = (mu, kappa, weight) of the inventory's von Mises components on the doubled grasp yaw, kept so a created
        # target part can label its own demos.  b["m"], when present, overrides the emitted value (S1 with m held fixed).
        self.nmode = int(nmode)
        self.register_buffer("tmode", torch.tensor(tmode if tmode is not None else np.zeros(len(TGT_ATOMS)), dtype=torch.long))
        self.register_buffer("gm", torch.tensor(gm if gm is not None else np.zeros((3, max(self.nmode, 1))), dtype=torch.float32))
        self.register_buffer("rclip_p", torch.tensor(rclip_p if rclip_p is not None else np.zeros(len(OBJ_ATOMS)), dtype=torch.float32))
        self.register_buffer("rtclip_p", torch.tensor(rtclip_p if rtclip_p is not None else np.zeros(len(TGT_ATOMS)), dtype=torch.float32))
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("size", torch.tensor(size if size is not None else np.zeros((N_ENT, 2)), dtype=torch.float32))
        da, dt = 7 * horizon, 32
        dprop = 3 + 6 + 2 + 1
        dslot = 3 + 2 + 2 + 1                                       # position, (sin, cos) yaw, size, drawer opening: no identity
        self.obj = RoleNet(dprop + dslot + self.nmode + da + dt, da, len(OBJ_ATOMS), rank=rank)
        self.tgt = RoleNet(3 + 1 + 2 + 1 + 6 + 2 * self.tsize + da + dt, da, len(TGT_ATOMS), rank=rank)
        if self.lcomb:
            self.comb = nn.Sequential(nn.Linear(2 * da + 1 + dt, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, da))
        # entity index -> part index (-1: no part)
        self.register_buffer("opart", torch.tensor([OBJ_ATOMS.index(e) if e in OBJ_ATOMS else -1 for e in _ENT()], dtype=torch.long))
        self.register_buffer("tpart", torch.tensor([TGT_ATOMS.index(e) if e in TGT_ATOMS else -1 for e in _ENT()], dtype=torch.long))

    # ---- features (stage 1i's, identity removed) ---------------------------------------------------------------------------------
    def _scale(self, b, o):
        """P_hs: per row, the episode's horizontal scale s (None when off).  Start = b["ee0"], else the current hand position."""
        if float(self.hscale) <= 0:
            return None
        e0 = b["ee0"] if "ee0" in b else b["ee_pos"]
        r0 = torch.linalg.norm((e0 - o)[:, :2], dim=-1).clamp_min(1e-9)
        return torch.clamp(self.hscale / r0, max=1.0)

    def _radius(self, scalar, per_part, part):
        """The clip radius per row: the scalar (library-wide) one, or, with psupport, the activated part's own (fallback: scalar)."""
        if not self.psupport:
            return scalar
        rp = per_part[part.clamp_min(0)]
        return torch.where(rp > 0, rp, scalar)[:, None]

    def _tscale(self, b):
        """P_ts: per row, the carry's horizontal scale s_t (None when off).  d_onset = b["d0t"], else the current distance."""
        if not self.tscale:
            return None
        idx = torch.arange(len(b["oi"]), device=b["oi"].device)
        d0 = b["d0t"] if "d0t" in b else torch.linalg.norm((b["pos"][idx, b["oi"]] - b["pos"][idx, b["ti"]])[:, :2], dim=-1)
        r = self._radius(self.rtclip, self.rtclip_p, self.tpart[b["ti"]])
        r = r[:, 0] if r.dim() == 2 else r.expand(len(d0))
        if not self.tnear:
            return torch.clamp(r / d0.clamp_min(1e-9), max=1.0)
        d = torch.linalg.norm((b["pos"][idx, b["oi"]] - b["pos"][idx, b["ti"]])[:, :2], dim=-1)      # PARTS_RESULTS 50
        rho = self.rrel_p[self.tpart[b["ti"]].clamp_min(0)]
        sfar = (r - rho) / (d0 - rho).clamp_min(1e-9)
        phi = torch.where(d > rho, rho + (d - rho) * sfar, d)
        # rho >= r_t: the release region covers the part's whole carry support, so nothing is stretched (scale 1)
        return torch.where((d0 > r) & (rho < r), torch.clamp(phi / d.clamp_min(1e-9), max=1.0), torch.ones_like(d))

    def part_on(self, b, o):
        """P_rtr: per row, does the object part act (horizontal hand-object distance <= the part's start radius)?  None when off."""
        if not self.rtransport:
            return None
        dxy = torch.linalg.norm((b["ee_pos"] - o)[:, :2], dim=-1)
        return dxy <= self.rs_p[self.opart[b["oi"]].clamp_min(0)]

    def _obj_in(self, b, Q, o, yaw0, Qp=None, on=None):
        idx = torch.arange(len(b["oi"]), device=b["oi"].device)
        QT = Q.transpose(1, 2)
        QpT = QT if Qp is None else Qp.transpose(1, 2)                      # translation frame (P_af: the approach frame)
        d = torch.einsum("bij,bj->bi", QpT, b["ee_pos"] - o)
        s = self._scale(b, o)
        if s is not None:                                                       # horizontal scaling commutes with the yaw frame
            d = torch.cat([d[:, :2] * s[:, None], d[:, 2:]], -1)
        if on is not None and float(self.rrole) > 0:                            # P_rtr: backbone rows relocated onto the role's radius
            r = torch.linalg.norm(d[:, :2], dim=-1, keepdim=True).clamp_min(1e-9)
            f = torch.where(on[:, None], torch.ones_like(r), torch.clamp(self.rrole / r, max=1.0))
            d = torch.cat([d[:, :2] * f, d[:, 2:]], -1)
        if float(self.rclip) > 0:                                               # relocate far states onto the support radius
            r = torch.linalg.norm(d[:, :2], dim=-1, keepdim=True).clamp_min(1e-9)
            d = torch.cat([d[:, :2] * torch.clamp(self._radius(self.rclip, self.rclip_p, self.opart[b["oi"]]) / r, max=1.0), d[:, 2:]], -1)
        p = d * POS_SCALE
        R = torch.einsum("bij,bjk->bik", QT, b["ee_rot"])[:, :, :2].reshape(-1, 6)
        e = b["oi"]
        dy = b["yaw"][idx, e] - yaw0
        aux = b["drawer"] * (e == DRAWER).float()
        sz = self.size[e]
        if self.bbnosize and on is not None:                                    # PARTS_RESULTS 47: backbone-only rows read no size
            sz = sz * on[:, None].to(sz.dtype)
        slot = torch.cat([torch.zeros_like(p), torch.sin(dy)[:, None], torch.cos(dy)[:, None], sz, aux[:, None] * 10.0], -1)
        msg = [torch.nn.functional.one_hot(self.message(b), self.nmode).float()] if self.nmode > 0 else []
        if msg and self.bbnom and on is not None:                               # PARTS_RESULTS 44: backbone-only rows read no m
            msg = [msg[0] * on[:, None].to(msg[0].dtype)]
        return torch.cat([p, R, b["grip"] * GRIP_SCALE, b["prev"][:, None], slot] + msg, -1)

    def message(self, b):
        """P_imd: the grasp-mode message m per row: the activated target part's fitted value (b["m"] overrides it)."""
        if "m" in b:
            return b["m"].long()
        return self.tmode[self.tpart[b["ti"]].clamp_min(0)].clamp(0, max(self.nmode - 1, 0))

    def _canon(self, b):
        B = b["pos"].shape[0]
        idx = torch.arange(B, device=b["pos"].device)
        pt = b["pos"][idx, b["ti"]]
        Q = rz(b["cyaw"])
        QT = Q.transpose(1, 2)
        p = b["pos"][idx, b["oi"]].clone()
        p[:, 2] = p[:, 2] - b["zrest"] + b["pos"][:, TABLE, 2]
        pr = torch.einsum("bij,bj->bi", QT, p - pt)
        st = self._tscale(b)
        if st is not None:                                                      # P_ts: scaled, not clipped (horizontal scale commutes with Q)
            pr = torch.cat([pr[:, :2] * st[:, None], pr[:, 2:]], -1)
        elif float(self.rtclip) > 0:                                            # relocate far carry states onto the support radius
            r = torch.linalg.norm(pr[:, :2], dim=-1, keepdim=True).clamp_min(1e-9)
            pr = torch.cat([pr[:, :2] * torch.clamp(self._radius(self.rtclip, self.rtclip_p, self.tpart[b["ti"]]) / r, max=1.0), pr[:, 2:]], -1)
        pr = pr * POS_SCALE
        e = b["ti"]
        aux = b["drawer"] * (e == DRAWER).float()
        r0 = b["rot0"] if "rot0" in b else b["ee_rot"]
        rel = torch.einsum("bij,bkj->bik", b["ee_rot"], r0)
        drot = torch.einsum("bij,bjk,bkl->bil", QT, rel, Q)[:, :, :2].reshape(-1, 6)
        held_geom = [self.size[b["oi"]]] if self.tsize else []                 # the held object's footprint radius and height
        return Q, torch.cat([pr, b["prev"][:, None], self.size[e], aux[:, None] * 10.0, drot] + held_geom, -1)

    def _vobj(self, b, x, te, part_off=False):
        """The object term's velocity (world frame).  part_off (P_rtr's backbone-only loss): every row with the part off."""
        B = x.shape[0]
        flat = lambda z: z.reshape(B, -1)                                       # noqa: E731
        idx = torch.arange(B, device=x.device)
        Qo, oo, yo = rz(b["yaw"][idx, b["oi"]]), b["pos"][idx, b["oi"]], b["yaw"][idx, b["oi"]]
        on = self.part_on(b, oo)
        if on is not None and part_off:
            on = torch.zeros_like(on)
        if self.aframe == 2:                                        # one frame: positions, rotations and yaw relative to the bearing
            ay = torch.atan2(oo[:, 1] - self.home[1], oo[:, 0] - self.home[0])
            Qa = rz(ay)
            co = self._obj_in(b, Qa, oo, ay, on=on)
            Qo = Qa
        elif self.aframe:                                           # x-axis: the bearing from the home end effector to the object
            Qa = rz(torch.atan2(oo[:, 1] - self.home[1], oo[:, 0] - self.home[0]))
            co = self._obj_in(b, Qo, oo, yo, Qp=Qa, on=on)
        else:
            Qa = Qo
            co = self._obj_in(b, Qo, oo, yo, on=on)
        po = self.opart[b["oi"]]
        assert bool((po >= 0).all()), "an atom without a part"
        s = self._scale(b, oo)
        xo = x if s is None else torch.cat([x[..., :2] * s[:, None, None], x[..., 2:]], -1)
        vo = _from2(Qa, Qo, self.obj(torch.cat([co, flat(_to2(Qa, Qo, xo)), te], -1), po, on).reshape(B, self.H, 7))
        if s is not None:
            vo = torch.cat([vo[..., :2] / s[:, None, None], vo[..., 2:]], -1)
        return vo

    def velocity(self, b, x, t):
        B = x.shape[0]
        te = temb(t)
        flat = lambda z: z.reshape(B, -1)                                       # noqa: E731
        vo = self._vobj(b, x, te)
        Qc, ct = self._canon(b)
        pt = self.tpart[b["ti"]]
        assert bool((pt >= 0).all()), "an atom without a part"
        st = self._tscale(b)
        xc = chunk_to(Qc, x)
        if st is not None:                                          # P_ts: the chunk's horizontal translation in scaled units
            xc = torch.cat([xc[..., :2] * st[:, None, None], xc[..., 2:]], -1)
        oc = self.tgt(torch.cat([ct, flat(xc), te], -1), pt).reshape(B, self.H, 7)
        if st is not None:
            oc = torch.cat([oc[..., :2] / st[:, None, None], oc[..., 2:]], -1)
        vt = chunk_from(Qc, oc)
        g = b["held"][:, None, None]
        if self.lcomb:
            return self.comb(torch.cat([flat(vo), flat(vt), b["held"][:, None], te], -1)).reshape(B, self.H, 7)
        return (1 - g) * vo + g * vt

    def loss(self, b, a):
        B = a.shape[0]
        x0 = torch.randn_like(a)
        t = torch.rand(B, device=a.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
        loss = ((self.velocity(b, xt, t) - (a - x0)) ** 2).mean()
        if self.rtransport and self.training_bb:                            # PARTS_RESULTS 41: the backbone-only term, not-held rows
            vbb = self._vobj(b, xt, temb(t), part_off=True)
            loss = loss + ((1 - b["held"])[:, None, None] * (vbb - (a - x0)) ** 2).mean()
        return loss

    training_bb = True                                                      # creation turns it off (the backbone is frozen there)

    @torch.no_grad()
    def sample(self, b, steps=10, generator=None):
        B = b["pos"].shape[0]
        x = torch.randn(B, self.H, 7, generator=generator)
        for k in range(steps):
            t = torch.full((B,), k / steps)
            x = x + self.velocity(b, x, t) / steps
        return x

    def save(self, path, meta):
        np.savez(path, __meta__=json.dumps(dict(meta, arm="P", horizon=self.H, rank=self.rank, tsize=self.tsize, aframe=self.aframe,
                                                psupport=self.psupport, tscale=self.tscale, nmode=self.nmode,
                                                rtransport=self.rtransport, bbnom=self.bbnom, bbnosize=self.bbnosize,
                                                tnear=self.tnear, **(dict(lcomb=True) if self.lcomb else {}))),
                 **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        m = cls(meta["horizon"], rank=meta.get("rank", RANK), tsize=meta.get("tsize", False), aframe=meta.get("aframe", False),
                psupport=meta.get("psupport", False), tscale=meta.get("tscale", False), nmode=meta.get("nmode", 0),
                rtransport=meta.get("rtransport", False), bbnom=meta.get("bbnom", False), bbnosize=meta.get("bbnosize", False),
                tnear=meta.get("tnear", False), lcomb=meta.get("lcomb", False))
        sd ={k: torch.from_numpy(z[k]) for k in z.files if k != "__meta__"}
        sd.setdefault("rrel_p", m.rrel_p.clone())                               # checkpoints before P_rel carry no release radii
        sd.setdefault("rs_p", m.rs_p.clone())                                   # checkpoints before P_rtr carry no start radii
        sd.setdefault("rrole", torch.tensor(0.0))
        sd.setdefault("tmode", m.tmode.clone())                                 # checkpoints before P_imd carry no interface
        sd.setdefault("gm", m.gm.clone())
        sd.setdefault("home", torch.zeros(3))                                   # checkpoints before P_af carry no home buffer
        sd.setdefault("rclip", torch.tensor(0.0))                               # nor before P_rc
        sd.setdefault("rtclip", torch.tensor(0.0))                              # nor before P_rt
        sd.setdefault("hscale", torch.tensor(0.0))                              # nor before P_hs
        sd.setdefault("rclip_p", torch.zeros(len(OBJ_ATOMS)))                   # nor per-part radii before P_pp
        sd.setdefault("rtclip_p", torch.zeros(len(TGT_ATOMS)))
        m.load_state_dict(sd)
        m.eval()
        if meta.get("pgate"):                                                   # ladder L1: the build's proprioceptive event
            m.pgate = dict(tw=float(meta["pgate"]["tw"]), lift=float(meta["pgate"]["lift"]), **({"motion": True} if meta["pgate"].get("motion") else {}))
        if meta.get("roles"):                                                   # ladder L4: the library's induced lexicon
            m.role_lex = meta["roles"]
        return m, meta

    pgate = None                                                            # ladder L1 off: the rollout steps the oracle gate only
    role_lex = None                                                         # ladder L4 off: the rollout uses the given roles


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

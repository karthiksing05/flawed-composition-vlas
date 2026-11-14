"""ReGuide (Zhang et al., "Correcting WHERE, Preserving HOW", arXiv 2609.38616): a training-free wrapper on a FROZEN VLA that rebinds
the instructed referent and steers the end effector with object poses plus training-demonstration statistics, then hands control
back to the frozen policy.  Reimplemented from the paper's text (main method, Algorithm 1, App. A.4-A.7, B, C.6, C.7) for
docs/SCAFFOLD_RESULTS.md 52 (m3_prereg tag reguide).  No code was released; every unspecified point takes the simplest reading and
is listed in SR 52 "implementation choices" as a deviation.

Harness = the break atlas's (pp_atlas_client.py: fc_n2_client -> libero_client, pp_atlas.Rec's oracle track as the read's
instrument, pi0.5 served intact by pp_serve_purify.py --mode intact --comp= --seed 13, client seed 7, --bind track), as SR 47-48.
This file wraps the env (every action passes through ReGuide) and the websocket client (the chunk ReGuide predicts from; no server
query while ReGuide steers), and gives libero_client's action queue a handle so a hand-back can discard the pending chunk (paper:
"the pending chunk is discarded and the frozen policy is queried on the true observation").  Prompt and images are never changed.

Arms:
    rgo   RG-oracle (PRIVILEGED): entity poses from the simulator every step (paper App. B oracle state interface); the interaction
          state from the simulator (the object rose > 3 cm above rest and sits within 12 cm of the hand: A.4's effect onset).
    rgp   RG-perc: entity poses registered ONCE, at the first policy query, from sc_seg's learned U-Net masks + the rendered depth
          back-projected through the camera calibration (sc_pu.Estimator with sc_hp's monkeypatched seg_std, as fx_r / sc_hp / PU-L;
          the depth render and calibration are the remaining privilege, as SR 40).  No tracking afterwards (paper C.6).  The held
          object then follows the hand rigidly from the hold onset; the hold event is ProprioHeld (sc_rv lex threshold), so no object
          pose is read during the episode.  Fixture sites: sc_pu calib's site mode.
Statistics: results/flawed/rg_stats.json (experiments/rg_stats.py; trained-task demos K1-K6 only).

The controller per policy step (Algorithm 1, paraphrased in the paper):
  1. read ee pose, entities, interaction state; moving point q = ee (free) or the held object (attached).
  2. transport active: dissolve on a premise change (grasp / drop) or no error decrease > eps over n replans (n * 5 steps); end it
     (hand-back) on entering the pre-contact set K_e (approach) or one step above release height inside Omega_d (carry; masked
     destination classes: on entering Omega_d); else issue the bounded command (Eq. 6) and continue.
  3./4. the policy's chunk remainder -> predicted endpoint and posture (Eq. 2) -> the policy's referent (Eq. 3).
  5. a trigger fires and the stage's transport budget (2) is not spent -> start a transport (the second one in a stage only on a
     completion certificate).
  6. otherwise the policy's action, clamped to the demonstrated envelope (per-step translation s, rotation s_R).
  7. completion certificates: re-arm (the held object leaves Omega_d for n steps after a carry transport; stall N; dwell outside
     Omega_d within the carry radius > Q.95 dwell); dwell inside Omega_d without descent > Q.95 dwell -> descend to the Q.5 release
     height and open for the Q.95 release duration (not on masked classes: cabinet top side, wine rack); after a release outside
     Omega_d the object stage re-arms.

Modes (LIBERO venv, py3.8, EGL, one GPU for the client):
    python experiments/rg.py client --arm rgo|rgp <pp_atlas_client.py's arguments>   one n2 task per process
        -> <out>.json (libero_client), <out>_track.npz (pp_atlas.Rec), <out>_rg.json (per episode: events, transports, perception)
    python experiments/rg.py summary DIR                                          per-episode table (smoke)
The registered read is experiments/rg_read.py.
"""
import argparse
import collections
import json
import math
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

ARMS = ("rgo", "rgp")
WAIT, REPLAN = 10, 5
EFFECT, HOLD_NEAR, CO_MOTION = 0.03, 0.12, 0.02
OUT = os.path.join(ROOT, "results", "flawed")
STATS = os.environ.get("RG_STATS") or os.path.join(OUT, "rg_stats.json")
RG_DIR = os.path.join(OUT, "rg")
MOVABLE = ("akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1")
# destinations in the retrieved training scene (every LIBERO-Goal task's goal region / body; the scene is the same for all of them)
DESTS = ("plate_1", "akita_black_bowl_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side", "wine_rack_1_top_region",
         "wooden_cabinet_1_top_region", "main_table_stove_front_region")
MASKED = ("wooden_cabinet_1_top_side", "wine_rack_1_top_region")          # paper A.6: microwave, wine rack, cabinet-top side
HALF = np.diag([-1.0, -1.0, 1.0])


# ---------------------------------------------------------------- geometry helpers
def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_angle(R):
    return float(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def rot_err(A, B):
    return min(rot_angle(A.T @ B), rot_angle((A @ HALF).T @ B))


def up_err(A, B):
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(A[:, 2], B[:, 2]))))))


def rotvec_to_mat(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(v, np.float64)).as_matrix()


def mat_to_rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, np.float64)).as_rotvec()


def box_out(p, o, yaw, g):
    C = np.eye(3) if g["symmetric"] else rz(yaw)
    L = C.T @ (np.asarray(p, np.float64) - np.asarray(o, np.float64))
    return np.array([max(0.0, abs(L[0]) - g["hx"]), max(0.0, abs(L[1]) - g["hy"]), max(0.0, g["bottom"] - L[2], L[2] - g["top"])])


# ---------------------------------------------------------------- the controller (no simulator access; it gets a state dict)
class ReGuide:
    """One episode.  decide(s, a, rem) -> (action to execute, handback: bool).  s = dict(ee [3], R [3, 3], width, pos {name: [3]},
    yaw {name: float}, held bool, q (moving point when held) [3])."""

    def __init__(self, S, oe, de):
        self.S, self.oe, self.de = S, oe, de
        self.G = S["geometry"]
        self.g = self.G[oe]
        self.os = S["obj"].get(oe) or S["obj"][S["inherit"][oe]]
        self.kp, self.kr = S["kappa"]
        self.sp, self.sr = S["sigma"]
        self.s_app, self.s_car, self.s_R = S["s_app"], S["s_carry"], S["s_R"]
        self.eps, self.n, self.N, self.N2, self.budget_max = S["eps"], int(S["n"]), int(S["N"]), int(S["N_after"]), int(S["budget"])
        self.ds = S["dst"][de]
        crs = [v["carry_radius"] for v in S["dst"].values()]
        self.cr = {d: (S["dst"][d]["carry_radius"] if d in S["dst"] else float(np.median(crs))) for d in DESTS}
        self.masked = de in MASKED
        self.stage, self.mode, self.done = 1, None, False
        self.budget = {1: 0, 2: 0}
        self.t = 0
        self.events, self.clamped, self.steps_rg = [], 0, 0
        self.best, self.t_best = None, 0                   # failure-to-bind / stall progress
        self.mis_count, self.prev_q, self.qhist = 0, None, collections.deque(maxlen=self.n + 1)
        self.after_carry, self.out_count, self.dwell_in, self.dwell_out, self.zmin_in = False, 0, 0, 0, None
        self.tr = None                                     # active transport: dict(kind, t0, best_err, t_best, ...)
        self.wq = collections.deque(maxlen=self.n)
        self.rearm = False
        self.dp_counts = collections.Counter()

    # -- targets
    def precontact(self, s):
        o, yaw = np.asarray(s["pos"][self.oe], np.float64), float(s["yaw"][self.oe])
        C = np.eye(3) if self.g["symmetric"] else rz(yaw)
        off = np.asarray(self.os["off"], np.float64)
        p = o + C @ np.array([off[0], off[1], 0.0])
        p[2] = o[2] + self.g["top"] + float(self.os["zbar"])
        Rbar = np.asarray(self.os["Rbar"], np.float64)
        if self.g["symmetric"]:                            # only the up axis: the yaw of the reference nearest the current posture
            R = s["R"]
            best = min((rot_err(rz(a) @ Rbar, R), a) for a in np.linspace(-math.pi, math.pi, 73))
            Rs = rz(best[1]) @ Rbar
        else:
            Rs = C @ Rbar
        return p, Rs

    def posture(self, A, B):
        return up_err(A, B) if self.g["symmetric"] else rot_err(A, B)

    def dest_xy(self, s):
        d = np.asarray(s["pos"][self.de], np.float64)
        off = np.asarray(self.ds["off"], np.float64)
        return d[:2] + (rz(float(s["yaw"][self.de])) @ np.array([off[0], off[1], 0.0]))[:2], d[2]

    def profile(self, dist):
        P = self.ds["profile"]
        if not P:
            return None, None
        c = np.array([p[0] for p in P])
        i = int(np.argmin(np.abs(c - dist)))
        crest = max((p[2] for p in P if p[0] <= max(dist, c.min())), default=P[0][2])
        return P[i][1], crest

    # -- prediction (Eq. 2) and commitment (Eq. 3)
    def predict(self, s, rem, q):
        if rem is None or not len(rem):
            return None, None
        rem = np.asarray(rem, np.float64)
        qh = np.asarray(q, np.float64) + self.kp * self.sp * rem[:, :3].sum(0)
        R = np.array(s["R"], copy=True)
        for w in rem[:, 3:6]:
            R = rotvec_to_mat(self.kr * self.sr * w) @ R
        return qh, R

    def obj_referent(self, s, qh):
        best = None
        cur = s["ee"]
        for e in MOVABLE:
            if e not in s["pos"]:
                continue
            g = self.G[e]
            st = self.S["obj"].get(e) or self.S["obj"][self.S["inherit"][e]]
            oh = box_out(qh, s["pos"][e], s["yaw"][e], g)
            oc = box_out(cur, s["pos"][e], s["yaw"][e], g)
            if int((oh > 0).sum()) > 1:
                continue
            ax = int(np.argmax(oh))
            if oh[ax] < st["rho"] and (oh[ax] < oc[ax] or oc[ax] == 0.0):
                dd = float(np.linalg.norm(qh - np.asarray(s["pos"][e])))
                if best is None or dd < best[0]:
                    best = (dd, e)
        return None if best is None else best[1]

    def dst_half(self, d):
        """Horizontal half-widths of a destination's box: a movable body's declared (mesh) footprint, a fixture region's declared
        footprint radius (fa1_common.SIZE)."""
        if d in self.G:
            return np.array([self.G[d]["hx"], self.G[d]["hy"]])
        import fa1_common as FC
        return np.array([FC.SIZE[d][0], FC.SIZE[d][0]])

    def dst_out(self, p, d, s):
        c = np.asarray(s["pos"][d], np.float64)[:2]
        return np.maximum(0.0, np.abs(np.asarray(p, np.float64)[:2] - c) - self.dst_half(d))

    def dst_committed(self, s, qh, q, dwell=False):
        """Eq. 3 on the held object's (predicted) horizontal position against each destination's box (A.5): inside along one of
        the two horizontal axes, the other offset below the carry radius and smaller than the current one."""
        out = {}
        for d in DESTS:
            if d == self.oe or d not in s["pos"]:
                continue
            oh, oc = self.dst_out(qh, d, s), self.dst_out(q, d, s)
            if int((oh > 0).sum()) > 1:
                continue
            ax = int(np.argmax(oh))
            if oh[ax] < self.cr[d] and (dwell or oh[ax] < oc[ax] or oc[ax] == 0.0):
                out[d] = float(np.linalg.norm(np.asarray(qh)[:2] - np.asarray(s["pos"][d])[:2]))
        return out

    def dst_referent(self, s, qh, q):
        c = self.dst_committed(s, qh, q)
        if not c or self.de in c:                          # no mismatch while committed to the instructed destination
            return None if not c else self.de
        return min(c, key=c.get)

    # -- commands (Eq. 6)
    def cmd(self, s, p_target, R_target, s_bound, grip, move=None):
        cur = s["ee"] if move is None else move
        D = np.asarray(p_target, np.float64) - np.asarray(cur, np.float64)
        nD = float(np.linalg.norm(D))
        u = np.zeros(3) if nD < 1e-9 else D / nD * min(nD, s_bound) / (self.kp * self.sp)
        w = np.zeros(3)
        if R_target is not None:
            E = mat_to_rotvec(R_target @ s["R"].T)
            th = float(np.linalg.norm(E))
            if th > 1e-9:
                w = E / th * min(th, self.s_R) / (self.kr * self.sr)
        return np.concatenate([np.clip(u, -1, 1), np.clip(w, -1, 1), [grip]]).tolist()

    def clamp(self, a):
        a = np.array(a, np.float64)
        sb = self.s_app if self.stage != 2 else self.s_car
        nu = float(np.linalg.norm(a[:3])) * self.kp * self.sp
        if nu > sb:
            a[:3] *= sb / nu
            self.clamped += 1
        nw = float(np.linalg.norm(a[3:6])) * self.kr * self.sr
        if nw > self.s_R:
            a[3:6] *= self.s_R / nw
        return a.tolist()

    def ev(self, kind, **kw):
        e = dict(t=self.t, kind=kind, stage=self.stage)
        e.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()})
        self.events.append(e)

    # -- transports
    def start(self, kind, why, s, **kw):
        st = 1 if kind == "approach" else 2
        if kind != "release":                               # descent-and-release completes a transport; it is not a new one
            if self.budget[st] >= self.budget_max:
                self.ev("budget_spent", why=why)
                return False
            self.budget[st] += 1
        self.tr = dict(kind=kind, t0=self.t, best=None, t_best=self.t, why=why, R0=np.array(s["R"], copy=True),
                       grip_off=None if s.get("q") is None else (np.asarray(s["ee"]) - np.asarray(s["q"])), z0=None, phase="move")
        self.tr.update(kw)
        self.mode = kind
        self.ev("start", what=kind, why=why)
        return True

    def end(self, why, handback=True):
        self.ev("end", what=self.mode, why=why)
        if self.mode == "carry":
            self.after_carry, self.out_count = True, 0
        self.tr, self.mode = None, None
        return handback

    def transport_step(self, s):
        """-> (action or None if ended, handback)."""
        tr = self.tr
        if tr["kind"] == "approach":
            p, Rs = self.precontact(s)
            err = float(np.linalg.norm(p - s["ee"]))
            lat = float(np.linalg.norm((p - s["ee"])[:2]))
            if err <= self.s_app and lat <= self.os["r_e"] and self.posture(Rs, s["R"]) <= self.os["tau_R"]:
                return None, self.end("entered K_e")
            a = self.cmd(s, p, Rs, self.s_app, -1.0)
        elif tr["kind"] == "carry":
            q = np.asarray(s["q"], np.float64)
            pxy, dz = self.dest_xy(s)
            dist = float(np.linalg.norm(q[:2] - pxy))
            zb = q[2] + self.g["bottom"]                    # the held object's bottom height
            if tr["z0"] is None:
                tr["z0"] = zb
            inside = dist <= self.ds["omega"]
            rel = dz + self.ds["h_rel"]
            if self.masked:
                if inside:
                    return None, self.end("entered Omega_d (masked class)")
                _, crest = self.profile(dist)               # keeps the current height (the policy places), floored by the crest
                zt = tr["z0"] if crest is None else max(tr["z0"], crest)
                if zb < zt - self.eps:                      # lift-then-translate
                    pxy = q[:2]
                err = dist + abs(zb - zt)
            else:
                m, crest = self.profile(dist)
                if inside:
                    tr["phase"] = "descend"
                if tr["phase"] == "descend":
                    zt = rel + self.s_car
                    if zb <= zt + self.eps:
                        return None, self.end("one step above release height in Omega_d")
                else:
                    zt = max(m, crest) if m is not None else zb
                    if zb < crest - self.eps:               # lift-then-translate: the floor first
                        pxy = q[:2]
                err = dist + abs(zb - zt)
            tgt = np.array([pxy[0], pxy[1], zt - self.g["bottom"]])
            a = self.cmd(s, tgt, tr["R0"], self.s_car, 1.0, move=q)
        else:                                               # release procedure: descend to the Q.5 release height, then open
            q = np.asarray(s["q"], np.float64)
            pxy, dz = self.dest_xy(s)
            zb = q[2] + self.g["bottom"]
            rel = dz + self.ds["h_rel"]
            if tr["phase"] == "move":
                if zb <= rel + self.eps:
                    tr["phase"], tr["t_open"] = "open", self.t
                else:
                    tgt = np.array([pxy[0], pxy[1], rel - self.g["bottom"]])
                    return self.cmd(s, tgt, tr["R0"], self.s_car, 1.0, move=q), False
            if self.t - tr["t_open"] >= int(math.ceil(self.ds["rel_dur"])):
                return None, self.end("released")
            return [0.0] * 6 + [-1.0], False
        if tr["best"] is None or err < tr["best"] - self.eps:
            tr["best"], tr["t_best"] = err, self.t
        elif self.t - tr["t_best"] > self.n * REPLAN:
            return None, self.end("no error decrease over n replans")
        return a, False

    # -- main
    def decide(self, s, a, rem):
        self.t += 1
        held = bool(s["held"])
        hb = False
        # interaction state changes (premise changes dissolve transports)
        if self.stage == 1 and held:
            if self.mode == "approach":
                hb = self.end("premise changed: grasp")
            self.stage, self.mis_count, self.best, self.t_best = 2, 0, None, self.t
            self.after_carry, self.dwell_in, self.dwell_out = False, 0, 0
            self.ev("attached")
        elif self.stage == 2 and not held:
            if self.mode in ("carry",):
                hb = self.end("premise changed: drop")
            elif self.mode == "release":
                self.end("released", handback=True)
                hb = True
            q = np.asarray(s["q"] if s.get("q") is not None else s["pos"][self.oe], np.float64)
            pxy, _ = self.dest_xy(s)
            inside = float(np.linalg.norm(q[:2] - pxy)) <= self.ds["omega"]
            self.ev("detached", inside=bool(inside))
            if inside:
                self.stage, self.done = 3, True
            else:
                self.stage, self.best, self.t_best, self.rearm = 1, None, self.t, True
        if self.mode is not None:
            act, h2 = self.transport_step(s)
            if act is not None:
                self.steps_rg += 1
                return act, False
            return self.finish(s, a, hb or h2)
        return self.finish(s, a, hb)

    def finish(self, s, a, hb):
        """Triggers (no transport active) -> a new transport, else the clamped policy action."""
        trig = None if self.done else (self.trig1(s, a) if self.stage == 1 else self.trig2(s))
        if trig is not None:
            kind, why = trig
            if self.start(kind, why, s):
                act, h2 = self.transport_step(s)
                if act is not None:
                    self.steps_rg += 1
                    return act, True                        # the policy's pending chunk is discarded
                hb = hb or h2
        if hb:                                              # hand-back: the pending chunk is stale; hold this one step, and the
            return [0.0] * 6 + [float(s["cmd"])], True      # policy is queried on the next observation
        return self.clamp(a), False

    def trig1(self, s, a):
        ee = np.asarray(s["ee"], np.float64)
        p, Rs = self.precontact(s)
        rem = s.get("rem")
        qh, Rh = self.predict(s, rem, ee)
        oo = box_out(ee, s["pos"][self.oe], s["yaw"][self.oe], self.g)
        free = int((oo > 0).sum()) > 1 or oo.max() > self.os["rho"]   # not yet in the instructed object's entry region
        # failure to bind: no demonstrated step of progress in position, posture or gripper closing for N steps (approach segments)
        err, perr = float(np.linalg.norm(p - ee)), self.posture(Rs, s["R"])
        prog = (self.best is None or err < self.best[0] - self.s_app or perr < self.best[1] - self.s_R or float(a[6]) > 0)
        if prog or not free:
            self.best = (err if self.best is None else min(err, self.best[0]), perr if self.best is None else min(perr, self.best[1]))
            self.t_best = self.t
        cert = getattr(self, "rearm", False)
        if free and self.t - self.t_best > self.N:
            self.t_best = self.t
            return ("approach", "failure to bind (N steps)")
        if cert and free:
            self.rearm = False
            return ("approach", "re-arm: released outside Omega_d")
        if self.budget[1] >= 1:                              # the second transport in a stage needs a certificate
            return None
        if qh is None:
            return None
        ep = self.obj_referent(s, qh)
        if ep is not None and ep != self.oe and free:
            return ("approach", "wrong binding -> %s" % ep)
        if ep == self.oe and ee[2] >= p[2] - self.s_app:       # posture, before descent beside the object
            ph, pc = self.posture(Rs, Rh), self.posture(Rs, s["R"])
            if ph > self.os["tau_R"] and ph >= pc - self.s_R:
                return ("approach", "posture mismatch %.2f" % ph)
        return None

    def trig2(self, s):
        q = np.asarray(s["q"], np.float64)
        pxy, dz = self.dest_xy(s)
        dist = float(np.linalg.norm(q[:2] - pxy))
        inside = dist <= self.ds["omega"]
        self.qhist.append(q.copy())
        still = len(self.qhist) > self.n and float(np.linalg.norm(self.qhist[-1] - self.qhist[0])) < self.eps
        w = s.get("width")
        self.wq.append(w)
        band = self.os["width_band"]                        # A.5: width at rest for n steps within the demonstrated band
        stable = (len(self.wq) == self.n and w is not None and (max(self.wq) - min(self.wq)) < 0.001
                  and band[0] - self.eps <= w <= band[1] + self.eps)
        # stall progress
        if self.best is None or dist < self.best - self.s_car:
            self.best, self.t_best = dist, self.t
        # certificates
        cert = None
        if self.after_carry:
            self.out_count = 0 if inside else self.out_count + 1
            if self.out_count >= self.n:
                cert = "re-arm: left Omega_d for n steps"
        if inside and not self.masked:
            zb = q[2] + self.g["bottom"]
            if self.zmin_in is None or zb < self.zmin_in - self.eps:
                self.zmin_in, self.dwell_in = zb, 0
            else:
                self.dwell_in += 1
            if self.dwell_in > self.ds["dwell"]:
                self.dwell_in = 0
                return ("release", "dwell inside Omega_d without descent")
        else:
            self.zmin_in, self.dwell_in = None, 0
        if not inside and dist <= self.cr[self.de] and still:
            self.dwell_out += 1
            if self.dwell_out > self.ds["dwell"]:
                cert = "dwell outside Omega_d within the carry radius"
        else:
            self.dwell_out = 0
        if self.t - self.t_best > self.N2 and not inside:
            cert = "stall (N steps)"
            self.t_best = self.t
        if cert is not None and stable:
            return ("carry", cert)
        if self.budget[2] >= 1 or not stable:
            return None
        qh, _ = self.predict(s, s.get("rem"), q)
        if qh is not None:
            dp = self.dst_referent(s, qh, q)
            self.dp_counts[str(dp)] += 1
            self.mis_count = self.mis_count + 1 if (dp is not None and dp != self.de) else 0
            if self.mis_count >= self.n:
                return ("carry", "wrong destination -> %s (n steps)" % dp)
        if still and not inside:                            # dwelling: the held object itself sits committed (Eq. 3 at q) and still
            c = self.dst_committed(s, q, q, dwell=True)
            if c and self.de not in c:
                return ("carry", "dwell at habitual %s" % min(c, key=c.get))
        return None

    def record(self):
        return dict(events=self.events, budget=self.budget, dst_referent_counts=dict(self.dp_counts), clamped=self.clamped, steps_rg=self.steps_rg, stage=self.stage,
                    done=self.done, transports=sum(1 for e in self.events if e["kind"] == "start" and e["what"] != "release"),
                    releases=sum(1 for e in self.events if e["kind"] == "start" and e["what"] == "release"))


# ---------------------------------------------------------------- client (LIBERO venv)
def client_main():
    def take(name, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return v
        return default
    sys.argv.pop(1)                                                         # "client"
    arm = take("--arm")
    assert arm in ARMS, arm
    import fc_n2_client as FN
    import pp_atlas as PA
    import fa1_common as FC
    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID and TASK["layout"] == "familiar", TASK
    S = json.load(open(STATS, encoding="utf-8"))
    OE, DE = FC.roles(TASK["language"])                                    # the cell's referents (paper B: from the cell spec)
    assert (OE, DE) == (TASK["object"], TASK["target"]) or DE.startswith(TASK["target"]), (OE, DE, TASK)
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], dict(env=None, plan=None, chunk=None, k=0)

    class PlanDeque(collections.deque):                                     # libero_client's action queue, with a handle
        def __init__(self, *a):
            super().__init__(*a)
            CUR["plan"] = self
    LC.collections = types.SimpleNamespace(deque=PlanDeque)

    P = None
    if arm == "rgp":
        import sc_pu as PU
        import sc_rv as RV
        import sc_seg as SEG
        from lifelong_ood.policy.parts_flow import ProprioHeld
        X = json.load(open(RV.lex_path(""), encoding="utf-8"))
        Cal = PU.load_calib()
        SEGM = SEG.load(os.environ.get("SC_SEG_CKPT") or None, "cuda")
        CACHE = {}

        def seg_std(env, E, cam):
            """sc_hp's seg_std: the learned U-Net on an RGB render, in sc_pu.seg_std's convention (no simulator labels)."""
            sim = env.env.sim
            key = (id(sim), float(sim.data.time), cam)
            if key in CACHE:
                return CACHE[key]
            img = RV.flip(sim.render(camera_name=cam, width=PU.SIZE, height=PU.SIZE, depth=False))
            p = SEGM.ent(img, cam, E["names"])
            out = np.ascontiguousarray(np.asarray(p)[:, ::-1])
            if len(CACHE) > 8:
                CACHE.clear()
            CACHE[key] = out
            return out
        PU.seg_std = seg_std
        P = dict(PU=PU, RV=RV, X=X, C=Cal, PH=ProprioHeld, seg=SEGM)

    class RGEnv(PA.Rec):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.rg, self.obs, self.cmd, self.reg, self.det, self.rest, self.att = None, None, -1.0, None, None, None, None
            self.diag = dict(chunk_len=[], held_onset=None, reg=None, n_q=0)
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.rg, self.obs, self.cmd, self.reg, self.det, self.rest, self.att = ReGuide(S, OE, DE), obs, -1.0, None, None, None, None
            self.diag = dict(chunk_len=[], held_onset=None, reg=None, n_q=0)
            return obs

        def state(self):
            obs = self.obs
            ee = np.asarray(obs["robot0_eef_pos"], np.float64)
            R = FC._quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
            grip = np.asarray(obs["robot0_gripper_qpos"], np.float64)
            if arm == "rgo" or self.reg is None:                    # rgp: the simulator only for the logged registration error
                sim = FC.scene(self._e, obs, self._dj)
                simpos = {n: sim["pos"][i].copy() for i, n in enumerate(FC.ENT)}
                simyaw = {n: float(sim["yaw"][i]) for i, n in enumerate(FC.ENT)}
            if arm == "rgo":
                pos, yaw = simpos, simyaw
                o = pos[OE]
                rel = R.T @ (o - ee)                               # the object in the hand frame
                if self.att is None:                               # free: attach at the effect onset (A.4), object near the hand
                    self.rest = float(o[2]) if self.rest is None else min(self.rest, float(o[2]))
                    if o[2] > self.rest + EFFECT and np.linalg.norm(o - ee) < HOLD_NEAR:
                        self.att = rel
                elif np.linalg.norm(rel - self.att) > CO_MOTION:   # co-motion ended: detached
                    self.att, self.rest = None, float(o[2])
                held = self.att is not None
                q = o.copy()
            else:
                if self.reg is None:
                    self.reg = self.register(ee, R, simpos)
                pos, yaw = {k: v.copy() for k, v in self.reg["pos"].items()}, dict(self.reg["yaw"])
                if self.det is None:
                    self.det = P["PH"](P["X"]["tw"], P["X"]["lift"])
                held = bool(self.det.step(grip, self.cmd, ee[2]))
                if held and self.att is None:
                    o0 = pos[OE] if self.reg.get("last") is None else self.reg["last"]
                    self.att = R.T @ (o0 - ee)                     # rigid offset in the hand frame, fixed at the hold onset
                if not held and self.att is not None:
                    self.reg["last"] = ee + R @ self.att           # the released object stays where the hand left it
                    self.att = None
                q = ee + R @ self.att if self.att is not None else (pos[OE] if self.reg.get("last") is None else self.reg["last"])
                pos[OE] = q.copy()
            if held and self.diag["held_onset"] is None:
                self.diag["held_onset"] = len(self.rows["held"]) - 1
            return dict(ee=ee, R=R, width=float(grip[0] - grip[1]), pos=pos, yaw=yaw, held=held, q=q, cmd=self.cmd)

        def register(self, ee, R, simpos):
            """C.6: registration once, at the first policy query, from the learned masks + depth (sc_pu.Estimator, first step)."""
            PU, RV = P["PU"], P["RV"]
            E = RV.entity_tables(self._e)
            est = PU.Estimator(P["C"], OE)
            A, _ = PU.measure(self._e, E, PU.AGENT, PU.MOV)
            F = PU.measure(self._e, E, PU.AGENT, PU.fixture_names(E))[0]
            est.step(A, lambda: PU.measure(self._e, E, PU.WRIST, PU.MOV)[0], F, False, ee, R, -1.0)
            pos = {n: est.pos[i].copy() for i, n in enumerate(FC.ENT)}
            yaw = {n: float(est.yaw[i]) for i, n in enumerate(FC.ENT)}
            err = {n: round(100 * float(np.linalg.norm(pos[n] - simpos[n])), 3) for n in FC.ENT}   # logged only
            self.diag["reg"] = dict(src=dict(est.src), err_cm=err)
            return dict(pos=pos, yaw=yaw, last=None, src=dict(est.src))

        def step(self, a):
            if self.n < WAIT or self.rg is None:
                r = super().step(a)
                self.obs = r[0]
                return r
            ch = CUR.get("chunk")
            rem = None
            if ch is not None:
                rem = ch[CUR["k"]:]
                CUR["k"] += 1
            s = self.state()
            s["rem"] = rem
            act, hb = self.rg.decide(s, list(a), rem)
            r = super().step(act)
            self.obs, self.cmd = r[0], float(act[6])
            if hb:
                pl = CUR.get("plan")
                if pl is not None:
                    pl.clear()
                CUR["chunk"] = None
            return r

        def close(self):
            if self.n > 0 and self.rg is not None:
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                held = np.asarray(self.rows["held"], bool)
                rec = self.rg.record()
                rec.update(trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"], arm=arm,
                           oracle_first_held=int(np.argmax(held)) if held.any() else None, **self.diag)
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], roles=[OE, DE], stats=STATS,
                                   seg_ckpt=None if P is None else P["seg"].path, eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[rg] %s %s trial %s: transports %d releases %d events %s" % (
                    arm, TASK["key"], rec["trial"], rec["transports"], rec["releases"],
                    [(e["t"], e["kind"], e.get("what"), e.get("why")) for e in rec["events"]][:8]), flush=True)
                self.rg = None
            self._e.close()

    SIDEF = OUTF.replace(".json", "_rg.json")
    Base = LC._wcp.WebsocketClientPolicy

    class RGClient(Base):
        def infer(self, el):
            env = CUR.get("env")
            if env is not None and env.rg is not None and env.rg.mode is not None:
                CUR["chunk"] = None                             # ReGuide steers: no query; the env replaces these actions
                return {"actions": np.zeros((REPLAN, 7))}
            out = super().infer(el)
            CUR["chunk"], CUR["k"] = np.asarray(out["actions"], np.float64), 0
            if env is not None:
                env.diag["chunk_len"].append(int(len(out["actions"])))
                env.diag["n_q"] += 1
            return out
    LC._wcp.WebsocketClientPolicy = RGClient
    LC.OffScreenRenderEnv = lambda *a, **k: RGEnv(ORIG(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    print("[rg] arm %s task %s %r roles %s -> %s stats %s" % (arm, TASK["key"], TASK["language"], OE, DE, STATS), flush=True)
    LC.main()


# ---------------------------------------------------------------- summary (smoke)
def summary(d):
    import pp_atlas as PA
    T = {t["tid"]: t for t in PA.tasks()["tasks"]}
    tot = []
    for f in sorted(x for x in os.listdir(d) if x.endswith("_rg.json")):
        Sd = json.load(open(os.path.join(d, f), encoding="utf-8"))
        stem = os.path.join(d, f[:-len("_rg.json")])
        if not os.path.exists(stem + ".json"):
            print("== %s: no client json yet" % f, flush=True)
            continue
        R = json.load(open(stem + ".json", encoding="utf-8"))
        E = [e for v in R["tasks"].values() for e in v["episodes"]]
        Tr = PA.load_track(stem + "_track.npz")
        t = T[Sd["tid"]]
        print("== %s %s %r" % (f, t["key"], t["language"]), flush=True)
        for e, r, s in zip(E, Tr, Sd["eps"]):
            dd = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
            st = PA.classify(r["held"], dd, bool(e["success"]), e.get("bound"), t["object"])
            tot.append(bool(e["success"]))
            print(json.dumps(dict(trial=e["trial"], success=e["success"], stage=st[0], sub=st[1], steps=e["steps"], err=e["error"],
                                  transports=s["transports"], releases=s["releases"], clamped=s["clamped"], steps_rg=s["steps_rg"],
                                  chunk=sorted(set(s["chunk_len"])), held_onset=s["held_onset"], oracle_first_held=s["oracle_first_held"],
                                  reg=s.get("reg"), dst=s.get("dst_referent_counts"), events=s["events"][:12])), flush=True)
    if tot:
        print("TOTAL %d episodes, success %.3f" % (len(tot), float(np.mean(tot))), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        return summary(sys.argv[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

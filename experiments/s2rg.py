"""S2RG (docs/SCAFFOLD_RESULTS.md 68; m3_prereg tag s2_rg): SR 52's ReGuide reimplementation (experiments/rg.py's controller class,
UNCHANGED) ported to scene 2 (LIBERO-10 LIVING_ROOM_SCENE5: white / yellow-and-white / red mug, left / right plate), against I and
SR 63's named-mug hand-over Hp, on NEW states, layout O, tasks K1, K2, N1, N2, frozen pi0.5 (PRIMARY) and pi0 (SECONDARY).

Facts (SR 50's task table): K1 white -> left (plate_1), K2 yellow-and-white -> right (plate_2), N1 white -> right, N2 yellow-and-white
-> left; the scene's only training task is the compound (white -> left, then yellow-and-white -> right).

THE PORT (what changes against SR 52 / 65's rgp; everything else is rg.py's ReGuide class, imported and unchanged):
  - Entities.  rg.MOVABLE (the referent set of Eq. 3 for objects) = the three mugs; rg.DESTS (destinations) = the two plates (the
    scene's only goal entities: the habitual set of the single training scene, as SR 52 D3); rg.MASKED = () (no masked class here).
    The instructed mug and plate are parsed from the instruction (s2fix4.parse_named, s2fix6.parse_plates; asserted equal to the
    task's object and target).
  - Declared geometry (mode `stats`): the asset meshes of the 3 mugs and 2 plates at rest in a scene-2 env (rg_stats.geom_points).
    Every entity is treated as ROUND (symmetric: world-axis box, posture = the approach axis only), because OWLv2 boxes give no yaw;
    a mug's footprint half-width = its mesh radius (handle included).  rgo uses the same declaration.
  - Statistics (mode `stats`, rg_stats.py's rules and Table 5's quantile levels, unchanged in kind): from scene 2's only
    demonstrations, the LIBERO-10 compound's 50 raw demos (s2_tasks.DUMP: recorded states + actions, replayed, never executed), one
    close-open cycle per mug: the white mug's cycle is a K1 demo (-> plate_1), the yellow-and-white mug's a K2 demo (-> plate_2).  No
    N1 / N2 demo exists.  The approach segment of the second cycle starts after the first cycle's release.  Plate frames and
    release offsets in world axes (round plates).  The red mug inherits the nearest mug's statistics (A.6).
  - rgp perception: SR 63's (s2fix3.Percept: OWLv2 sidecar on the policy's agentview image + rendered depth + camera calibration,
    CAL / PLATE_LOC / SIDE_RULE unchanged), the same privilege as Hp.  Registration once at the first gated step (n = WAIT, as Hp
    and paper C.6): every detected mug, and the plates by SIDE_RULE (left = plate_1, right = plate_2).  If the instructed mug or
    plate is not perceived, ReGuide stays inactive in that episode (the policy acts alone; logged).  Hold events: ProprioHeld (sc_rv
    lex), as rgp.  At each hold onset OWLv2 runs again (Hp's rule): detected mugs update their positions, and the held mug's identity
    is Hp's (the nearest perceived mug within HELD_R of the hand).  ReGuide's interaction state is "the INSTRUCTED mug is held"
    (as rgo's oracle state, which tracks the instructed object only): a hold of another mug is not an attachment.  While attached the
    instructed mug follows the hand rigidly from the onset (rgp's rule); at a release it stays where the hand left it.
  - rgo (PRIVILEGED): simulator body positions of every mug and plate every step; rg.py's oracle interaction rule on the instructed
    mug (rose > 3 cm above rest within 12 cm of the hand; detached when co-motion ends).  A ProprioHeld detector and the simulator's
    held-mug rule are stepped for the LOG only (first-hold identity), never fed.
  - Harness: s2.py's (s2_living suite, --fresh / --branches, s2.Rec2's oracle track), client seed 7, --bind track; ReGuide wraps the
    env and the websocket client exactly as rg.py does (the pending chunk -> predicted endpoint; no query while ReGuide steers; a
    hand-back discards the pending chunk).  Prompt and images are never changed.
END PORT.

New states: LIBERO's sampler on K1's BDDL, seeds 980000 + k (none equal to an SR 50-67 state), k 0-3 smoke, 4-63 main; branches
seed_base 46000000 + 1000 k.  Servers: SR 63's (pi0.5 pp_serve_purify --mode intact --comp= --seed 13; pi0 serve_guided --mode ours
--seed 13 with BACKBONE_CONFIG=pi0_libero).

Modes (LIBERO venv, py3.8):
    stats [--out X]                       (EGL)   -> results/flawed/s2rg/rg_stats_s2.json (geometry + demo statistics)
    states                                (EGL)   -> results/flawed/s2rg/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    client --arm rgp|rgo <s2.py client args>      (rgp needs the OWLv2 sidecar on $S2FIX3_DET_PORT)
    ref --tid T --branches B --out-dir D  (CPU, C1)
    summary DIR                           (smoke)
    read [DIR --selftest --out X]         (CPU)
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import s2                                                                   # noqa: E402
import s2cue                                                                # noqa: E402
import s2fix3 as F3                                                         # noqa: E402
import s2fix4 as F4                                                         # noqa: E402
import s2fix6 as F6                                                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2rg")
ROLL = os.path.join(OUT, "s2rg_roll")
READ = os.path.join(OUT, "s2rg_read.json")
STATS = os.path.join(CD, "rg_stats_s2.json")
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
MUGS, PLATES = s2.MUGS, s2.PLATES
BASE, BR_BASE, N_STATES = 980000, 46000000, 64
SMOKE = list(range(0, 4))
MAIN = list(range(4, 64))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
HELD_R = F3.HELD_R
NI_MARGIN = F3.NI_MARGIN
FLOOR = F3.FLOOR
BAR = 0.50
EQ_BAND = 0.10
RUNS = dict(pi05=("I", "Hp", "rgp", "rgo"), pi0=("I", "Hp", "rgp", "rgo"))
PRIMARY_MODEL = "pi05"
HALF = {W: ("K1", PL), Y: ("K2", PR)}                                       # each mug's demonstrated cycle in the compound demos
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 980000 + k, none equal to an SR 50-67 state), main k = 4-63 (60), branches "
    "seed_base 46000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Models: frozen pi0.5 (PRIMARY; "
    "pp_serve_purify --model pi05 --mode intact --comp= --seed 13) and frozen pi0 (SECONDARY; serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero).  Arms per model: I (s2.py client), Hp (SR 63, s2fix3.py client --arm Hp, unchanged), rgp (SR 52's "
    "ReGuide controller, unchanged, ported to scene 2 with SR 63's OWLv2 + depth + calibration poses; s2rg.py client --arm rgp), rgo "
    "(the same controller on simulator poses, PRIVILEGED; s2rg.py client --arm rgo).  Success = the BDDL goal.  Matrices [task, "
    "state].  Paired bootstrap over states, tasks fixed (s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5).  Per "
    "model, registered labels for rgp (PRIMARY arm; the same labels reported for rgo as SECONDARY): (V1, SR 62's rule) dN = rgp - I on "
    "N1+N2, dK = rgp - I on K1+K2: FIX WORKS iff dN lo > 0 AND dK lo > -.05 (NON-INFERIOR at margin -.05); PARTIAL iff exactly one "
    "holds; FAILS iff neither.  (V2) rgp - Hp on N1+N2: RG WINS iff lo > 0; HP WINS iff hi < 0; EQUIVALENT iff the CI lies inside "
    "[-.10, +.10]; otherwise NO CLEAR DIFFERENCE; a win label takes precedence.  (V3) BAR: MEETS BAR iff rgp's N1+N2 success >= .50 "
    "AND dN lo > 0 (SR 54's FIXED); else BELOW BAR; descriptive beside it: the state-bootstrap CI of rgp's N1+N2 success.  PRIMARY: "
    "V1-V3 on pi0.5.  SR 50's floor on every model: TESTABLE iff its I K1+K2 success >= .70 on these states; a NOT TESTABLE model's "
    "labels read NOT TESTABLE with the rule's outcome descriptive.  MECHANISM (registered, descriptive): (a) the share of episodes "
    "whose FIRST hold-detector onset holds the named mug by the logged simulator rule (an episode with no onset counts as not named), "
    "per task, for Hp (identical to I up to the first named hold, by construction), rgp and rgo; on N2 rgp - Hp paired over states, "
    "and N2 success given a named / other first hold; (b) the plate each mug ends on (final LIBERO On predicates) per arm and task, "
    "and on N1 / N2 the share of episodes in which the named mug ends on the plate the instruction does not name (the F1 place cue: "
    "its trained plate); (c) ReGuide's transports per task (approach / carry / release, and why).  SECONDARY: Hp's FIX label per "
    "model (SR 63 replication); rgp - Hp on K1+K2 and K2 (GAIN iff lo > 0, COST iff hi < 0, else NO CLEAR DIFFERENCE); rgo - rgp on "
    "N1+N2.  Descriptive: rgp perception (registration of the instructed mug and both plates, target plate xy error, side, onset "
    "held-mug identity against the simulator rule), episode length and time-outs.  C1: the reference planner (s2.plan_episode via "
    "s2cue3.ref_eps, unchanged) on the main states; MET iff its N1-N2 success >= .80.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def short(m):
    return None if m is None else m.split("_")[0]


# ---------------------------------------------------------------- statistics (LIBERO venv, EGL)
def geometry(env):
    """rg_stats.geometry's rule on scene 2's movables: asset geom points at rest, in each body's frame.  All declared ROUND (yaw is not
    perceived): a mug's footprint half-widths are both its radius."""
    import rg_stats as RS
    sim = env.env.sim
    sim.forward()
    m, d = sim.model, sim.data
    G = {}
    for n in MUGS + PLATES:
        root = int(env.env.obj_body_id[n])
        bodies = set()
        for b in range(m.nbody):
            x = b
            while True:
                if x == root:
                    bodies.add(b)
                    break
                p = int(m.body_parentid[x])
                if p == x or x == 0:
                    break
                x = p
        P = [RS.geom_points(m, d, g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
        P = np.concatenate([x for x in P if len(x)], 0)
        Rb = np.asarray(d.body_xmat[root]).reshape(3, 3)
        L = (P - np.asarray(d.body_xpos[root])) @ Rb
        hx = float(max(abs(L[:, 0].min()), abs(L[:, 0].max())))
        hy = float(max(abs(L[:, 1].min()), abs(L[:, 1].max())))
        rad = max(hx, hy)
        G[n] = dict(top=float(L[:, 2].max()), bottom=float(L[:, 2].min()), hx=rad if n in MUGS else hx, hy=rad if n in MUGS else hy,
                    radius=rad, symmetric=True, npts=int(len(P)), mesh_hx=hx, mesh_hy=hy)
    return G


def stats(a):
    import rg_stats as RS
    import s2_tasks as T2
    out = os.path.abspath(a.out or STATS)
    assert not os.path.exists(out), "never overwrite: %s" % out
    env = T2.make_env(T2.TEMPLATE)
    env.seed(0)
    env.reset()
    G = geometry(env)
    print("[s2rg stats] geometry %s" % json.dumps(G), flush=True)
    z = np.load(T2.DUMP)
    assert str(z["source"]).endswith(os.path.basename(T2.TEMPLATE).replace(".bddl", "_demo.hdf5")), str(z["source"])
    NAMES = list(MUGS + PLATES)
    obj_s = {n: dict(off=[], zbar=[], rho=[], Rent=[], ax=[], width=[], s_app=[]) for n in MUGS}
    dst_s = {}
    kap_p, kap_r, s_rot, s_carry, ev_log = [], [], [], [], []
    for di in range(int(z["n"])):
        A, Sx = np.asarray(z["a%d" % di], np.float64), z["s%d" % di]
        T = min(len(A), len(Sx))
        ee, Rr, gr, P = np.zeros((T, 3)), np.zeros((T, 3, 3)), np.zeros((T, 2)), np.zeros((T, len(NAMES), 3))
        for t in range(T):
            env.env.sim.set_state_from_flattened(Sx[t])
            env.env.sim.forward()
            ob = env.env._get_observations(force_update=True)
            ee[t] = np.asarray(ob["robot0_eef_pos"], np.float64)
            Rr[t] = T2.quat_mat(np.asarray(ob["robot0_eef_quat"], np.float64))
            gr[t] = np.asarray(ob["robot0_gripper_qpos"], np.float64)[:2]
            P[t] = T2.positions(env, NAMES)
        act = A[:T]
        for t in range(T - 1):                                              # kappa and the rotation step bound (every step)
            u = act[t, :3]
            if np.linalg.norm(u) > 0.2:
                kap_p.append(np.linalg.norm(ee[t + 1] - ee[t]) / (RS.SIG_P * np.linalg.norm(u)))
            w = act[t, 3:6]
            if np.linalg.norm(w) > 0.2:
                kap_r.append(RS.rot_angle(Rr[t + 1] @ Rr[t].T) / (RS.SIG_R * np.linalg.norm(w)))
            s_rot.append(RS.rot_angle(Rr[t + 1] @ Rr[t].T))
        cyc = []
        for oe in (W, Y):
            oi = NAMES.index(oe)
            o = P[:, oi]
            eff = np.flatnonzero(o[:, 2] > o[0, 2] + RS.EFFECT)
            if not len(eff):
                ev_log.append((di, short(oe), "no effect"))
                continue
            cyc.append((int(eff[0]), oe))
        cyc.sort()
        a0 = 0
        for t_eff, oe in cyc:
            oi = NAMES.index(oe)
            de = HALF[oe][1]
            di_ = NAMES.index(de)
            g = G[oe]
            o = P[:, oi]
            outd_all = np.stack([RS.box_out(ee[t], o[t], 0.0, g) for t in range(T)])
            inside = outd_all <= 0.0
            cl = np.flatnonzero(act[a0:t_eff, 6] > 0)
            t_cl0 = a0 + int(cl[-1]) if len(cl) else t_eff
            while t_cl0 > a0 and act[t_cl0 - 1, 6] > 0:
                t_cl0 -= 1
            two = inside[:t_cl0 + 1].sum(1) >= 2
            rel = np.flatnonzero(act[t_eff:, 6] <= 0)
            t_rel = t_eff + int(rel[0]) if len(rel) else T - 1
            if not two[t_cl0]:
                ev_log.append((di, short(oe), "close outside the entry region"))
                a0 = t_rel + 1
                continue
            t_ent = t_cl0
            while t_ent > a0 and two[t_ent - 1]:
                t_ent -= 1
            S_ = obj_s[oe]
            S_["rho"].append(float(outd_all[t_ent].max()))
            S_["zbar"].append(float(ee[t_ent, 2] - (o[t_ent, 2] + g["top"])))
            S_["Rent"].append(Rr[t_ent])
            S_["off"].append((ee[t_cl0] - o[t_cl0]).tolist())
            S_["ax"].append(Rr[t_cl0][:, 1].tolist())
            S_["width"].append(float(gr[t_eff, 0] - gr[t_eff, 1]))
            S_["s_app"] += [float(np.linalg.norm(ee[t + 1] - ee[t])) for t in range(a0, t_ent)]
            s_carry.extend(float(np.linalg.norm(ee[t + 1] - ee[t])) for t in range(t_eff, t_rel))
            d = P[:, di_]
            Dd = dst_s.setdefault(de, dict(off=[], h=[], dist=[], zb=[], dwell_xy=[], rel_dur=[], demos=0, tasks=set()))
            Dd["demos"] += 1
            Dd["tasks"].add(HALF[oe][0])
            Dd["off"].append((o[t_rel] - d[t_rel])[:2].tolist())
            Dd["h"].append(float(o[t_rel, 2] + g["bottom"] - d[t_rel, 2]))
            dist = np.linalg.norm((o - d)[:, :2], axis=1)
            Dd["dist"] += dist[t_eff:t_rel + 1].tolist()
            Dd["zb"] += (o[t_eff:t_rel + 1, 2] + g["bottom"]).tolist()
            Dd["dwell_xy"].append((o[t_eff:t_rel + 1] - d[t_eff:t_rel + 1])[:, :2].tolist())
            w_ = gr[t_rel:, 0] - gr[t_rel:, 1]
            wm = float(w_.max()) if len(w_) else 0.0
            op = np.flatnonzero(w_ >= 0.95 * wm) if len(w_) else []
            Dd["rel_dur"].append(int(op[0]) + 1 if len(op) else 1)
            a0 = t_rel + 1
        if di % 10 == 0:
            print("[s2rg stats] demo %d: %d steps, cycles %s" % (di, T, [(t, short(o_)) for t, o_ in cyc]), flush=True)
    env.close()
    q = RS.q
    OBJ = {}
    for n, S_ in obj_s.items():
        if not S_["off"]:
            continue
        off = np.array(S_["off"])
        med = np.median(off, 0)
        ax = np.array(S_["ax"])
        scat = np.abs(np.sum((off - med) * ax, 1))
        i = RS.medoid(S_["Rent"], True)
        Rbar = S_["Rent"][i]
        OBJ[n] = dict(n=len(off), off=med.tolist(), zbar=q(S_["zbar"], .5), rho=q(S_["rho"], .25), r_e=min(q(scat, .95), G[n]["radius"]),
                      Rbar=Rbar.tolist(), tau_R=q([RS.posture_err(Rbar, Rx, True) for Rx in S_["Rent"]], .95),
                      width_band=[q(S_["width"], .05), q(S_["width"], .95)], s_app_obj=q(S_["s_app"], .95))
    s_app_all = [x for S_ in obj_s.values() for x in S_["s_app"]]

    def phi(n):
        return np.log(np.array([G[n]["top"] - G[n]["bottom"], 2 * G[n]["radius"], 2 * G[n]["radius"]]))
    inherit = {n: min(OBJ, key=lambda m_: float(np.linalg.norm(phi(n) - phi(m_)))) for n in MUGS if n not in OBJ}
    DST = {}
    for de, Dd in dst_s.items():
        off = np.array(Dd["off"])
        med = np.median(off, 0)
        omega = q(np.linalg.norm(off - med, axis=1), .95)
        dist, zb = np.array(Dd["dist"]), np.array(Dd["zb"])
        prof = []
        for b in range(int(dist.max() / RS.BIN) + 1):
            mk = (dist >= b * RS.BIN) & (dist < (b + 1) * RS.BIN)
            if mk.sum() >= RS.MIN_BIN:
                prof.append([b * RS.BIN + RS.BIN / 2, q(zb[mk], .5), q(zb[mk], .05), int(mk.sum())])
        dw = [int((np.linalg.norm(np.array(xy) - med, axis=1) <= omega).sum()) for xy in Dd["dwell_xy"]]
        DST[de] = dict(demos=Dd["demos"], tasks=sorted(Dd["tasks"]), off=med.tolist(), omega=omega, h_rel=q(Dd["h"], .5),
                       carry_radius=q(dist, .25), profile=prof, dwell=q(dw, .95), rel_dur=q(Dd["rel_dur"], .95),
                       enough=bool(Dd["demos"] >= RS.MIN_DEMOS))
    res = dict(source="s2_tasks.DUMP (the LIBERO-10 LIVING_ROOM_SCENE5 compound's raw demos; one close-open cycle per mug: white = K1 "
                      "-> plate_1, yellow-and-white = K2 -> plate_2)", geometry=G, sigma=[RS.SIG_P, RS.SIG_R],
               kappa=[float(np.median(kap_p)), float(np.median(kap_r))], s_app=q(s_app_all, .95), s_carry=q(s_carry, .95),
               s_R=q(s_rot, .95), obj=OBJ, inherit=inherit, dst=DST, skipped=ev_log, eps=0.004, n=3, N=40, N_after=75, budget=2,
               symmetric_all=True, frames="world axes for every entity (round declaration)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(out + ".tmp", out)
    print("[s2rg stats]", json.dumps({k: v for k, v in res.items() if k not in ("obj", "dst", "geometry")}), flush=True)
    for n, v in OBJ.items():
        print("[s2rg stats] obj %s %s" % (short(n), json.dumps({k: (np.round(v[k], 4).tolist() if k != "n" else v[k]) for k in v})), flush=True)
    for n, v in DST.items():
        vv = dict(v)
        vv["profile"] = [[round(x, 3) for x in p[:3]] + [p[3]] for p in v["profile"]]
        print("[s2rg stats] dst %s %s" % (n, json.dumps(vv)), flush=True)
    print("[s2rg stats] wrote %s" % out, flush=True)


# ---------------------------------------------------------------- states (LIBERO venv, EGL)
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    for t in M_["tasks"][:4]:
        assert F4.parse_named(t["language"]) == {t["object"]}, (t["key"], t["language"])
        assert F6.parse_plates(t["language"]) == {t["target"]}, (t["key"], t["language"])
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE + k)
        env.seed(BASE + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))] + [
        np.load(os.path.join(OUT, d, L, "%s_0.npy" % s2.SUITE)) for d, L in
        (("s2cue3", "O"), ("s2cue4", "O"), ("s2cue5", "O"), ("s2fix2", "O"), ("s2fix2", "PD2"), ("s2fix3", "O"), ("s2fix4", "O"),
         ("s2fix5", "O"), ("s2fix6", "O"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-67 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2rg states] %d new states (seed %d + k), none equal to an SR 50-67 state; wrote %s" % (len(S), BASE, CD), flush=True)


# ---------------------------------------------------------------- client: arms rgp / rgo (LIBERO venv)
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
    assert arm in ("rgp", "rgo"), arm
    import rg as RG
    RG.MOVABLE, RG.DESTS, RG.MASKED = tuple(MUGS), tuple(PLATES), ()       # the port: this scene's referents and destinations
    import pp_atlas as PA
    import libero_client as LC
    import sc_rv as RV
    import s2_tasks as T2
    from lifelong_ood.policy.parts_flow import ProprioHeld
    M = s2.register_suite(LC)
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = M["tasks"][TID]
    assert TASK["tid"] == TID and len(TASK["atoms"]) == 1
    NAMED, NPL = F4.parse_named(TASK["language"]), F6.parse_plates(TASK["language"])
    assert len(NAMED) == 1 and len(NPL) == 1, (NAMED, NPL)
    OE, DE = list(NAMED)[0], list(NPL)[0]                                   # the cell's referents, from the instruction
    assert (OE, DE) == (TASK["object"], TASK["target"]), (OE, DE, TASK)
    ms = int(TASK["max_steps"])
    assert "--max-steps" not in sys.argv
    sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    S = json.load(open(STATS, encoding="utf-8"))
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    PER = F3.Percept() if arm == "rgp" else None
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], dict(env=None, plan=None, chunk=None, k=0)
    SIDEF = OUTF.replace(".json", "_rg.json")

    class PlanDeque(collections.deque):                                     # libero_client's action queue, with a handle (rg.py)
        def __init__(self, *a):
            super().__init__(*a)
            CUR["plan"] = self
    LC.collections = types.SimpleNamespace(deque=PlanDeque)

    class RGEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.rg = None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.rg, self.obs, self.cmd = RG.ReGuide(S, OE, DE), obs, -1.0
            self.reg, self.mem, self.att, self.rest, self.hid, self.inactive = None, {}, None, None, None, False
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.diag = dict(chunk_len=[], n_q=0, reg=None, onsets=[], attached_rows=[], inactive=False)
            return obs

        def _yaw0(self):
            return {n: 0.0 for n in MUGS + PLATES}

        def _sim_held(self, ee):
            d = {m: float(np.linalg.norm(self._pos(m)[:2] - np.asarray(ee, np.float64)[:2])) for m in MUGS}
            m = min(d, key=d.get)
            return m if d[m] < HELD_R else None

        def _per_held(self, ee):
            d = {m: float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(ee, np.float64)[:2])) for m, p in self.mem.items()}
            if not d:
                return None
            m = min(d, key=d.get)
            return m if d[m] < HELD_R else None

        def register(self, obs):
            sc = PER.scene(self._e, obs)
            self.mem = {m: np.asarray(v["pos"], np.float64) for m, v in sc["mugs"].items()}
            self.plates = {}
            if sc["left"] is not None:
                self.plates[PL] = np.asarray(sc["left"], np.float64)
            if sc["right"] is not None:
                self.plates[PR] = np.asarray(sc["right"], np.float64)
            err = {short(n): round(100 * float(np.linalg.norm((p - self._pos(n))[:2])), 3)
                   for n, p in list(self.mem.items()) + list(self.plates.items())}
            tp, op = self._pos(DE), self._pos([p for p in PLATES if p != DE][0])
            side_ok = None
            if DE in self.plates:
                side_ok = bool(np.linalg.norm(self.plates[DE][:2] - tp[:2]) < np.linalg.norm(self.plates[DE][:2] - op[:2]))
            self.inactive = OE not in self.mem or DE not in self.plates
            self.diag["inactive"] = self.inactive
            self.diag["reg"] = dict(row=len(self.rows["held"]) - 1, err_xy_cm=err, found=sorted(short(n) for n in list(self.mem) + list(self.plates)),
                                    tgt_err_xy_cm=err.get(short(DE)), tgt_side_ok=side_ok, obj_err_xy_cm=err.get(short(OE)),
                                    scores={short(m): v["score"] for m, v in sc["mugs"].items()})
            self.reg = True

        def state(self):
            obs = self.obs
            ee = np.asarray(obs["robot0_eef_pos"], np.float64)
            Rm = T2.quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
            grip = np.asarray(obs["robot0_gripper_qpos"], np.float64)
            row = len(self.rows["held"]) - 1
            was = self.det.on
            on = bool(self.det.step(grip, self.cmd, ee[2]))                  # rgp: the hold event; rgo: logged only
            onset = on and not was
            if arm == "rgo":
                pos = {n: self._pos(n) for n in MUGS + PLATES}
                o = pos[OE]
                rel = Rm.T @ (o - ee)
                if self.att is None:
                    self.rest = float(o[2]) if self.rest is None else min(self.rest, float(o[2]))
                    if o[2] > self.rest + RG.EFFECT and np.linalg.norm(o - ee) < RG.HOLD_NEAR:
                        self.att = rel
                        self.diag["attached_rows"].append(row)
                elif np.linalg.norm(rel - self.att) > RG.CO_MOTION:
                    self.att, self.rest = None, float(o[2])
                held = self.att is not None
                q = o.copy()
                if onset:
                    self.diag["onsets"].append(dict(row=row, sim_held=short(self._sim_held(ee)), held=None, attached=held))
            else:
                if onset:
                    sc = PER.scene(self._e, obs)
                    for m, v in sc["mugs"].items():
                        self.mem[m] = np.asarray(v["pos"], np.float64)
                    self.hid = self._per_held(ee)
                    self.diag["onsets"].append(dict(row=row, sim_held=short(self._sim_held(ee)), held=short(self.hid),
                                                    attached=self.hid == OE))
                if not on:
                    self.hid = None
                held = bool(on and self.hid == OE)
                if held and self.att is None:
                    self.att = Rm.T @ (self.mem[OE] - ee)                    # rigid offset in the hand frame, fixed at the hold onset
                    self.diag["attached_rows"].append(row)
                if not held and self.att is not None:
                    self.mem[OE] = ee + Rm @ self.att                        # the released mug stays where the hand left it
                    self.att = None
                q = ee + Rm @ self.att if self.att is not None else self.mem[OE]
                pos = {m: np.array(p, copy=True) for m, p in self.mem.items()}
                pos.update({p: np.array(v, copy=True) for p, v in self.plates.items()})
                pos[OE] = np.array(q, copy=True)
            return dict(ee=ee, R=Rm, width=float(grip[0] - grip[1]), pos=pos, yaw=self._yaw0(), held=held, q=q, cmd=self.cmd)

        def step(self, a):
            if self.n < s2.WAIT or self.rg is None:
                r = super().step(a)
                self.obs = r[0]
                return r
            if arm == "rgp" and self.reg is None:
                self.register(self.obs)
            if self.inactive:                                               # instructed mug or plate not perceived: the policy alone
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
                self.finish()
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
                                   movable=list(RG.MOVABLE), dests=list(RG.DESTS), masked=list(RG.MASKED), lex=X, held_r=HELD_R,
                                   detector=None if PER is None else PER.model, cal=F3.CAL, side_rule=F3.SIDE_RULE, eps=SIDE),
                              fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                os.replace(tmp, SIDEF)
                print("[s2rg] %s %s trial %s: inactive %s reg %s | onsets %s | transports %d releases %d steps_rg %d clamped %d | events %s" % (
                    arm, TASK["key"], rec["trial"], rec["inactive"], None if rec["reg"] is None else rec["reg"].get("tgt_err_xy_cm"),
                    [(o_["row"], o_["sim_held"], o_["held"], o_["attached"]) for o_ in rec["onsets"][:4]],
                    rec["transports"], rec["releases"], rec["steps_rg"], rec["clamped"],
                    [(e["t"], e["kind"], e.get("what"), e.get("why")) for e in rec["events"]][:8]), flush=True)
                self.rg = None
            self._e.close()

    Base = LC._wcp.WebsocketClientPolicy

    class RGClient(Base):
        def infer(self, el):
            env = CUR.get("env")
            if env is not None and env.rg is not None and env.rg.mode is not None:
                CUR["chunk"] = None                                         # ReGuide steers: no query; the env replaces these actions
                return {"actions": np.zeros((RG.REPLAN, 7))}
            out = super().infer(el)
            CUR["chunk"], CUR["k"] = np.asarray(out["actions"], np.float64), 0
            if env is not None and env.rg is not None:
                env.diag["chunk_len"].append(int(len(out["actions"])))
                env.diag["n_q"] += 1
            return out
    LC._wcp.WebsocketClientPolicy = RGClient
    LC.OffScreenRenderEnv = lambda *a, **k: RGEnv(ORIG(*a, **k))
    print("[s2rg] arm %s task %s %r roles %s -> %s max_steps %d stats %s kappa %s movable %s dests %s detector %s" % (
        arm, TASK["key"], TASK["language"], OE, DE, ms, STATS, S["kappa"], RG.MOVABLE, RG.DESTS,
        None if PER is None else PER.model), flush=True)
    LC.main()


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    import s2cue3
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_O_%d.json" % a.tid)
    assert not os.path.exists(fn), "exists: " + fn
    S = np.load(os.path.join(CD, "O", "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = s2cue3.ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(layout="O", tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- read helpers (CPU)
def load_rg(RD, model, arm, ks):
    out = {}
    for tid in TIDS:
        t = s2.meta()["tasks"][tid]
        f = F3.stem_of(RD, model, arm, tid) + "_rg.json"
        if not os.path.exists(f):
            return "MISSING %s" % f
        E = json.load(open(f, encoding="utf-8"))["eps"]
        assert [e["trial"] for e in E] == ks, f
        out[t["key"]] = E
    return out


def first_hold(run, side, arm, c):
    """-> per episode the simulator rule's held mug at the first hold-detector onset (None: no onset or no mug)."""
    if arm == "Hp":
        H = run["side"][c]
        return [h["events"][0]["sim_held"] if h["events"] else None for h in H]
    if arm in ("rgp", "rgo"):
        return [((e["onsets"][0]["sim_held"] and [m for m in MUGS if short(m) == e["onsets"][0]["sim_held"]][0])
                 if e["onsets"] else None) for e in side[c]]
    return None


def rg_diag(side, run):
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c, E in side.items():
        succ = run["succ"][c]
        n = len(E)
        st = [[ev for ev in e["events"] if ev["kind"] == "start"] for e in E]
        why = collections.Counter()
        for s_ in st:
            for ev in s_:
                w_ = str(ev.get("why")).split(" ->")[0].split(" (")[0]
                why["%s: %s" % (ev["what"], w_)] += 1
        has = {k: np.array([float(any(ev["what"] == k for ev in s_)) for s_ in st]) for k in ("approach", "carry", "release")}
        out[c] = dict(n=n, inactive=int(sum(bool(e.get("inactive")) for e in E)),
                      any_approach=float(has["approach"].mean()), any_carry=float(has["carry"].mean()),
                      any_release_proc=float(has["release"].mean()),
                      success_given_carry=float(succ[has["carry"] > 0].mean()) if (has["carry"] > 0).any() else None,
                      success_given_no_carry=float(succ[has["carry"] == 0].mean()) if (has["carry"] == 0).any() else None,
                      why=dict(why), steps_rg_mean=float(np.mean([e["steps_rg"] for e in E])),
                      clamped_mean=float(np.mean([e["clamped"] for e in E])), done=int(sum(bool(e["done"]) for e in E)),
                      attached_any=int(sum(bool(e.get("attached_rows")) for e in E)),
                      budget_spent=int(sum(any(ev["kind"] == "budget_spent" for ev in e["events"]) for e in E)),
                      object=short(T[c]["object"]))
    return out


def perception(side):
    xy, oxy, side_ok, nreg, inact, ag, nn = [], [], [0, 0], 0, 0, 0, 0
    conf = collections.Counter()
    for c, E in side.items():
        for e in E:
            rg_ = e.get("reg")
            if rg_ is None:
                continue
            nreg += 1
            inact += int(bool(e.get("inactive")))
            if rg_.get("tgt_err_xy_cm") is not None:
                xy.append(rg_["tgt_err_xy_cm"])
            if rg_.get("obj_err_xy_cm") is not None:
                oxy.append(rg_["obj_err_xy_cm"])
            if rg_.get("tgt_side_ok") is not None:
                side_ok[0] += int(rg_["tgt_side_ok"])
                side_ok[1] += 1
            for o_ in e["onsets"]:
                nn += 1
                ag += int(o_["held"] == o_["sim_held"])
                conf["%s->%s" % (o_["sim_held"], o_["held"])] += 1
    return dict(registrations=nreg, inactive=inact,
                target_xy_cm=dict(median=float(np.median(xy)) if xy else None, max=float(np.max(xy)) if xy else None, n=len(xy)),
                object_xy_cm=dict(median=float(np.median(oxy)) if oxy else None, max=float(np.max(oxy)) if oxy else None, n=len(oxy)),
                target_side_ok=side_ok, onset_identity_agree=[ag, nn, (ag / nn) if nn else None], confusion=dict(conf))


def plates_of(run):
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c, rr in run["rows"].items():
        o, tg = T[c]["object"], T[c]["target"]
        other = [p for p in PLATES if p != tg][0]
        out[c] = dict(white_on={str(p): int(sum(s2cue.plate_of(r["on"], W) == p for r in rr)) for p in (PL, PR, None, "both")},
                      yw_on={str(p): int(sum(s2cue.plate_of(r["on"], Y) == p for r in rr)) for p in (PL, PR, None, "both")},
                      named_on_unnamed_plate=float(np.mean([s2cue.plate_of(r["on"], o) == other for r in rr])),
                      named_on_target=float(np.mean([s2cue.plate_of(r["on"], o) == tg for r in rr])))
    return out


def v2_label(d):
    if d[1] > 0:
        return "RG WINS"
    if d[2] < 0:
        return "HP WINS"
    if d[1] >= -EQ_BAND and d[2] <= EQ_BAND:
        return "EQUIVALENT"
    return "NO CLEAR DIFFERENCE"


def gate(lab, testable):
    return lab if testable else "NOT TESTABLE (rule: %s)" % lab


def summary(d):
    M_ = s2.meta()
    for model, arms in RUNS.items():
        for arm in arms:
            for tid in TIDS:
                st = F3.stem_of(d, model, arm, tid)
                if not (os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")):
                    continue
                t = M_["tasks"][tid]
                Rj = json.load(open(st + ".json", encoding="utf-8"))
                E = [e for v in Rj["tasks"].values() for e in v["episodes"]]
                ks = [int(e["trial"]) for e in E]
                side = None
                for suf in ("_rg.json", "_h.json"):
                    if os.path.exists(st + suf):
                        side = json.load(open(st + suf, encoding="utf-8"))["eps"]
                side = side or [None] * len(ks)
                for k, e, r, h in zip(ks, E, s2.stage_rows(st, t, ks), side):
                    on = {"%s>%s" % (m.split("_")[0][:5], p): 1 for m in MUGS for p in PLATES if s2.on_of(r["on"], m, p)}
                    x = None
                    if h is not None and "onsets" in h and isinstance(h["onsets"], list) and h["onsets"] and isinstance(h["onsets"][0], dict):
                        x = dict(onsets=[(o_["row"], o_["sim_held"], o_["held"], o_["attached"]) for o_ in h["onsets"][:4]],
                                 starts=[(ev["t"], ev["what"], ev.get("why")) for ev in h["events"] if ev["kind"] == "start"][:6],
                                 reg=None if not h.get("reg") else h["reg"].get("tgt_err_xy_cm"), inactive=h.get("inactive"))
                    elif h is not None and "events" in h:
                        x = dict(events=[(v["row"], short(v.get("held")), short(v.get("sim_held")), v.get("decision")) for v in h["events"][:4]])
                    print("[s2rg summary] %s %s %s k%d success %s steps %s bound %s on %s | %s" % (
                        model, arm, t["key"], k, r["success"], e["steps"], r["bound"], on, json.dumps(x)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2rg summary] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_rg"]
        assert len(rows) == 1, "expected exactly one s2_rg prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, models={})
    ALLT = s2.KNOWN + s2.NEW
    one = np.ones(len(ks))
    for model, arms in RUNS.items():
        runs = {arm: F3.load_run(RD, model, arm, ks) for arm in arms}
        sides = {arm: load_rg(RD, model, arm, ks) for arm in arms if arm in ("rgp", "rgo")}
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}
        for arm in ("rgp", "rgo"):
            if isinstance(sides.get(arm), str):
                ok[arm] = False
        R_ = dict(arms={}, deltas={}, verdicts={}, mechanism={})
        for arm, run in runs.items():
            if not ok[arm]:
                R_["arms"][arm] = run if isinstance(run, str) else sides[arm]
                print("== %s %s: %s" % (model, arm, R_["arms"][arm]), flush=True)
                continue
            R_["arms"][arm] = dict(success={c: float(v.mean()) for c, v in run["succ"].items()},
                                   new=float(F3.mat(run, s2.NEW).mean()), known=float(F3.mat(run, s2.KNOWN).mean()),
                                   plates=plates_of(run))
            print("== %s %s: per task %s | N1+N2 %.3f | K1+K2 %.3f" % (
                model, arm, {c: round(v, 3) for c, v in R_["arms"][arm]["success"].items()}, R_["arms"][arm]["new"],
                R_["arms"][arm]["known"]), flush=True)
            print("      plates %s" % json.dumps(R_["arms"][arm]["plates"]), flush=True)

        def dd(x, y, tasks):
            return s2.boot2_states(F3.mat(runs[x], list(tasks)), F3.mat(runs[y], list(tasks)))
        tst = None
        if ok.get("I"):
            kI = float(F3.mat(runs["I"], s2.KNOWN).mean())
            tst = bool(kI >= FLOOR)
            R_["testable"] = dict(intact_K=kI, floor=FLOOR, TESTABLE=tst)
            for arm in ("Hp", "rgp", "rgo"):
                if not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                lab = F3.fix_label(dN, dK)
                R_["verdicts"]["%s V1 FIX label (rule)" % arm] = lab
                R_["verdicts"]["%s V1 FIX label (registered)" % arm] = gate(lab, tst)
                R_["verdicts"]["%s-I K non-inferiority" % arm] = F3.ni_label(dK)
                R_["deltas"]["%s-I per task" % arm] = {c: s2.boot2_states(runs[arm]["succ"][c][None], runs["I"]["succ"][c][None]) for c in ALLT}
                xN = float(F3.mat(runs[arm], s2.NEW).mean())
                bar = "MEETS BAR" if (dN[1] > 0 and xN >= BAR) else "BELOW BAR"
                R_["verdicts"]["%s V3 BAR (rule)" % arm] = bar
                R_["verdicts"]["%s V3 BAR (registered)" % arm] = gate(bar, tst)
                ci = s2.boot2_states(F3.mat(runs[arm], s2.NEW), np.zeros((2, len(ks))))
                R_.setdefault("bar", {})[arm] = dict(new=xN, bar=BAR, new_ci=ci, lower_bound_ge_bar=bool(ci[1] >= BAR))
        for arm in ("rgp", "rgo"):
            if ok.get(arm) and ok.get("Hp"):
                d2 = dd(arm, "Hp", s2.NEW)
                R_["deltas"]["%s-Hp N" % arm] = d2
                R_["verdicts"]["%s V2 vs Hp (rule)" % arm] = v2_label(d2)
                R_["verdicts"]["%s V2 vs Hp (registered)" % arm] = gate(v2_label(d2), bool(tst))
                for nm, tasks in (("K", s2.KNOWN), ("K2", ("K2",)), ("all", ALLT)):
                    d_ = dd(arm, "Hp", tasks)
                    R_["deltas"]["%s-Hp %s" % (arm, nm)] = d_
                    R_["verdicts"]["%s-Hp %s (secondary)" % (arm, nm)] = F3.diff_label(d_, "GAIN", "COST")
                R_["deltas"]["%s-Hp per task" % arm] = {c: s2.boot2_states(runs[arm]["succ"][c][None], runs["Hp"]["succ"][c][None]) for c in ALLT}
        if ok.get("rgo") and ok.get("rgp"):
            R_["deltas"]["rgo-rgp N"] = dd("rgo", "rgp", s2.NEW)
            R_["deltas"]["rgo-rgp K"] = dd("rgo", "rgp", s2.KNOWN)
        # mechanism (a): first hold
        FH = {}
        OBJ_OF = {t["key"]: t["object"] for t in s2.meta()["tasks"]}
        for arm in ("Hp", "rgp", "rgo"):
            if not ok.get(arm):
                continue
            FH[arm] = {}
            for c in ALLT:
                o = OBJ_OF[c]
                fh = first_hold(runs[arm], sides.get(arm), arm, c)
                ind = np.array([float(x == o) for x in fh])
                succ = runs[arm]["succ"][c]
                FH[arm][c] = dict(named_first=s2cue.boot_share(ind, one),
                                  first={str(short(m)): int(sum(x == m for x in fh)) for m in MUGS + (None,)},
                                  success_named_first=float(succ[ind > 0].mean()) if (ind > 0).any() else None,
                                  success_other_first=float(succ[ind == 0].mean()) if (ind == 0).any() else None,
                                  ind=ind.tolist())
        R_["mechanism"]["first_hold"] = FH
        for arm in ("rgp", "rgo"):
            if arm in FH and "Hp" in FH:
                R_["mechanism"]["first_hold_named_N2_%s-Hp" % arm] = s2.boot2_states(np.array([FH[arm]["N2"]["ind"]]), np.array([FH["Hp"]["N2"]["ind"]]))
        print("== %s MECHANISM first hold: %s" % (model, json.dumps({a_: {c: {k: v for k, v in x.items() if k != "ind"} for c, x in y.items()}
                                                                    for a_, y in FH.items()})), flush=True)
        for arm in ("rgp", "rgo"):
            if ok.get(arm):
                R_["mechanism"]["reguide_%s" % arm] = rg_diag(sides[arm], runs[arm])
                print("== %s %s ReGuide: %s" % (model, arm, json.dumps(R_["mechanism"]["reguide_%s" % arm])), flush=True)
        if ok.get("rgp"):
            R_["perception_rgp"] = perception(sides["rgp"])
            print("== %s rgp perception: %s" % (model, json.dumps(R_["perception_rgp"])), flush=True)
        if ok.get("Hp"):
            R_["perception_Hp"] = F3.accuracy(runs["Hp"], ks)
        for arm in arms:
            if ok.get(arm):
                st_ = {t["key"]: F6.ep_steps(RD, model, arm, t["tid"]) for t in s2.meta()["tasks"][:4]}
                R_.setdefault("steps", {})[arm] = {c: float(np.median(v)) for c, v in st_.items()}
                R_.setdefault("timed_out", {})[arm] = {c: int(sum(s >= 400 and not r["success"] for s, r in zip(v, runs[arm]["rows"][c])))
                                                      for c, v in st_.items()}
        for k_, v in R_["deltas"].items():
            print("   %s %s %s" % (model, k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s | testable %s | bar %s | median steps %s | timed out %s" % (
            model, json.dumps(R_["verdicts"]), json.dumps(R_.get("testable")), json.dumps(R_.get("bar")), json.dumps(R_.get("steps")),
            json.dumps(R_.get("timed_out"))), flush=True)
        res["models"][model] = R_
    ref_ = {}
    for tid in TIDS:
        f = os.path.join(RD, "ref_O_%d.json" % tid)
        if os.path.exists(f):
            ee = json.load(open(f, encoding="utf-8"))["eps"]
            assert [e["trial"] for e in ee] == ks, f
            ref_[s2.meta()["tasks"][tid]["key"]] = dict(success=float(np.mean([e["success"] for e in ee])),
                                                        errors=int(sum(e.get("error") is not None for e in ee)))
    nn = [ref_[c]["success"] for c in s2.NEW if c in ref_]
    res["reference"] = dict(per_task=ref_, success_new=float(np.mean(nn)) if len(nn) == 2 else None,
                            C1=None if len(nn) < 2 else ("C1 MET" if np.mean(nn) >= 0.80 else "C1 UNMET"))
    print("== reference: %s" % json.dumps(res["reference"]), flush=True)
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2rg read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stats", "states", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(stats=stats, states=states, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

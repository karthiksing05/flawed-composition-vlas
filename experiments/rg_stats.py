"""ReGuide (Zhang et al., arXiv 2609.38616) reimplementation, part 1: the training-demonstration statistics (paper Sec. 3, App. A.4-A.7,
Table 5), mined from the TRAINED tasks' demonstrations only: the raw LIBERO-Goal demos of K1-K6 (50 each; experiments/fa1_data.py
extract + poses -> FC.DATA/demos.npz: per kept step ee pose, finger qpos, action, every entity's pose).  No N1-N6 demo exists or is
read; K7 (drawer) and K8 (push) are not pick-and-place and give no grasp / carry segments, so they are not used.

Declared geometry (paper C.6 "declared category geometry from object assets"): each movable object's asset meshes in a LIBERO-Goal
env at rest -> top / bottom offsets and footprint half-widths in the body frame (robosuite's top_offset / horizontal_radius are
placeholders for these assets).  Boxes: footprint rectangles in the canonical frame (world axes for round objects).

Events per demo (paper A.4, simplest readings, see experiments/prereg/reguide_choices.txt):
    entry     start of the run of frames inside the object's box along two of three axes that holds the last close before the effect
    close     first frame at or after entry with a closing command (action[6] > 0)
    effect    first frame where the object rises > 3 cm above its rest height (A.4)
    release   first frame after effect with an opening command (action[6] <= 0) (end of co-motion)
Statistics (Table 5 levels): grasp offset / entry altitude Q.5; rotation medoid; tau_R, r_e, s, s_R, dwell, release duration Q.95;
rho_e (commitment radius) Q.25 of the box-surface distance at entry; carry radius Q.25 of the object-destination distance during
carries; release offset / release height Q.5 pooled by destination class; height profile m_f Q.5 and envelope Q.05 in 2 cm bins of
horizontal distance (>= 20 samples per bin, >= 10 demos per class); kappa = median realized / commanded motion (sigma_p = .05 m,
sigma_R = .5 rad: LIBERO's OSC_POSE output limits).

    LIBERO venv, CPU:  python experiments/rg_stats.py [--out results/flawed/rg_stats.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402

SIG_P, SIG_R = 0.05, 0.5
EFFECT = 0.03
BIN = 0.02
MIN_BIN, MIN_DEMOS = 20, 10
PICK = ("K1", "K2", "K3", "K4", "K5", "K6")
SYMMETRIC = ("akita_black_bowl_1", "wine_bottle_1", "plate_1")             # round footprints (sc_pu.SYMMETRIC): yaw unobservable
OUT_DEF = os.path.join(ROOT, "results", "flawed", "rg_stats.json")


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_angle(R):
    return float(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


HALF = np.diag([-1.0, -1.0, 1.0])                                           # the gripper's half-turn about its approach (z) axis


def rot_err(A, B):
    """Geodesic angle between two ee orientations modulo the gripper's half-turn symmetry (paper Eq. 5)."""
    return min(rot_angle(A.T @ B), rot_angle((A @ HALF).T @ B))


def up_err(A, B):
    """Angle between the approach (z) axes: the posture error for rotationally symmetric objects (A.6: up axis only)."""
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(A[:, 2], B[:, 2]))))))


def geom_points(m, d, g):
    """World points bounding one geom: mesh vertices, or the corners of a primitive's local box."""
    R = np.asarray(d.geom_xmat[g]).reshape(3, 3)
    p = np.asarray(d.geom_xpos[g])
    t, s = int(m.geom_type[g]), np.asarray(m.geom_size[g])
    if t == 7:                                                              # mesh
        k = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
        V = np.asarray(m.mesh_vert[a:a + n])
    else:
        h = {6: s[:3], 5: np.array([s[0], s[0], s[1]]), 2: np.array([s[0]] * 3), 3: np.array([s[0], s[0], s[1] + s[0]]),
             4: s[:3]}.get(t)
        if h is None:
            return np.zeros((0, 3))
        V = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
    return V @ R.T + p


def geometry():
    """Declared geometry from the object assets: every geom of each movable object's body tree (meshes' vertices), at rest in a
    LIBERO-Goal env, expressed in the body's frame -> top / bottom offsets (z above / below the body origin) and the footprint
    half-widths hx, hy (body frame).  (robosuite's top_offset / horizontal_radius are placeholders for these assets.)"""
    env = FC.make_env(os.path.join(FC.LIBBDDL, FC.KNOWN["K6"] + ".bddl"))
    env.seed(0)
    env.reset()
    sim = env.env.sim
    sim.forward()
    m, d = sim.model, sim.data
    G = {}
    for n in FC.MOVABLE:
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
        P = [geom_points(m, d, g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
        P = np.concatenate([x for x in P if len(x)], 0)
        Rb = np.asarray(d.body_xmat[root]).reshape(3, 3)
        L = (P - np.asarray(d.body_xpos[root])) @ Rb                         # body frame
        hx = float(max(abs(L[:, 0].min()), abs(L[:, 0].max())))
        hy = float(max(abs(L[:, 1].min()), abs(L[:, 1].max())))
        G[n] = dict(top=float(L[:, 2].max()), bottom=float(L[:, 2].min()), hx=hx, hy=hy, radius=max(hx, hy),
                    symmetric=n in SYMMETRIC, npts=int(len(P)))
    env.close()
    return G


def box_out(p, o, yaw, g):
    """Per-axis distance of point p outside the object's box (canonical frame: rz(yaw) for a box, world axes for a round object;
    footprint half-widths hx, hy; z between bottom and top offsets).  0 where inside along that axis."""
    C = np.eye(3) if g["symmetric"] else rz(yaw)
    L = C.T @ (np.asarray(p, np.float64) - np.asarray(o, np.float64))
    return np.array([max(0.0, abs(L[0]) - g["hx"]), max(0.0, abs(L[1]) - g["hy"]), max(0.0, g["bottom"] - L[2], L[2] - g["top"])])


def posture_err(A, B, symmetric):
    return up_err(A, B) if symmetric else rot_err(A, B)


def q(x, p):
    return float(np.percentile(np.asarray(x, np.float64), 100 * p))


def medoid(Rs, symmetric):
    n = len(Rs)
    D = np.array([[posture_err(Rs[i], Rs[j], symmetric) for j in range(n)] for i in range(n)])
    return int(np.argmin(D.sum(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEF)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: %s" % out
    G = geometry()
    print("[rg_stats] geometry", json.dumps(G), flush=True)
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    tasks = [str(t) for t in D["tasks"]]
    print("[rg_stats] demos.npz keys", sorted(D.keys()), "tasks", tasks, "rows", len(D["task"]), flush=True)
    obj_s = {n: dict(off=[], zbar=[], rho=[], Rent=[], ax=[], width=[], s_app=[]) for n in FC.MOVABLE}
    dst_s = {}
    kap_p, kap_r, s_rot, s_carry = [], [], [], []
    ev_log = []
    for k in PICK:
        ti = tasks.index(k)
        lang = FC.KNOWN[k].replace("_", " ")
        oe, de = FC.roles(lang)
        oi, di = FC.ENT.index(oe), FC.ENT.index(de)
        g = G[oe]
        rows = np.flatnonzero(D["task"] == ti)
        for dm in np.unique(D["demo"][rows]):
            r = rows[D["demo"][rows] == dm]
            r = r[np.argsort(D["step"][r])]
            ee, R, gr, act = D["ee_pos"][r].astype(np.float64), D["ee_rot"][r].astype(np.float64), D["grip"][r].astype(np.float64), D["action"][r].astype(np.float64)
            P, Y = D["pos"][r].astype(np.float64), D["yaw"][r].astype(np.float64)
            T = len(r)
            o = P[:, oi]
            rest = o[0, 2]
            # kappa (every step): realized vs commanded
            for t in range(T - 1):
                u = act[t, :3]
                if np.linalg.norm(u) > 0.2:
                    kap_p.append(np.linalg.norm(ee[t + 1] - ee[t]) / (SIG_P * np.linalg.norm(u)))
                w = act[t, 3:6]
                if np.linalg.norm(w) > 0.2:
                    kap_r.append(rot_angle(R[t + 1] @ R[t].T) / (SIG_R * np.linalg.norm(w)))
                s_rot.append(rot_angle(R[t + 1] @ R[t].T))
            outd_all = np.stack([box_out(ee[t], o[t], Y[t, oi], g) for t in range(T)])
            inside = outd_all <= 0.0
            eff = np.flatnonzero(o[:, 2] > rest + EFFECT)
            if not len(eff):
                ev_log.append((k, int(dm), "no effect"))
                continue
            t_eff = int(eff[0])
            cl = np.flatnonzero(act[:t_eff, 6] > 0)
            t_cl0 = int(cl[-1]) if len(cl) else t_eff                       # inside the last closing run before the effect
            while t_cl0 > 0 and act[t_cl0 - 1, 6] > 0:
                t_cl0 -= 1
            two = inside[:t_cl0 + 1].sum(1) >= 2
            if not two[t_cl0]:
                ev_log.append((k, int(dm), "close outside the entry region"))
                continue
            t_ent = t_cl0                                                    # entry = start of the inside run that holds the close
            while t_ent > 0 and two[t_ent - 1]:
                t_ent -= 1
            t_cl = t_cl0
            rel = np.flatnonzero(act[t_eff:, 6] <= 0)
            t_rel = t_eff + int(rel[0]) if len(rel) else T - 1
            C = np.eye(3) if g["symmetric"] else rz(Y[t_ent, oi])
            # entry: box-surface distance along the remaining axis, altitude above the top, posture in the canonical frame
            outd = outd_all[t_ent]
            S = obj_s[oe]
            S["rho"].append(float(outd.max()))
            S["zbar"].append(float(ee[t_ent, 2] - (o[t_ent, 2] + g["top"])))
            S["Rent"].append(C.T @ R[t_ent])
            Cg = np.eye(3) if g["symmetric"] else rz(Y[t_cl, oi])
            S["off"].append((Cg.T @ (ee[t_cl] - o[t_cl])).tolist())
            S["ax"].append((Cg.T @ R[t_cl][:, 1]).tolist())
            S["width"].append(float(gr[t_eff, 0] - gr[t_eff, 1]))
            S["s_app"] += [float(np.linalg.norm(ee[t + 1] - ee[t])) for t in range(t_ent)]
            s_carry.extend(float(np.linalg.norm(ee[t + 1] - ee[t])) for t in range(t_eff, t_rel))
            # destination
            d = P[:, di]
            Dd = dst_s.setdefault(de, dict(off=[], h=[], dist=[], zb=[], dwell_xy=[], rel_dur=[], demos=0, tasks=set()))
            Dd["demos"] += 1
            Dd["tasks"].add(k)
            Cd = rz(Y[t_rel, di])
            Dd["off"].append((Cd.T @ (o[t_rel] - d[t_rel]))[:2].tolist())
            Dd["h"].append(float(o[t_rel, 2] + g["bottom"] - d[t_rel, 2]))
            dist = np.linalg.norm((o - d)[:, :2], axis=1)
            Dd["dist"] += dist[t_eff:t_rel + 1].tolist()
            Dd["zb"] += (o[t_eff:t_rel + 1, 2] + g["bottom"]).tolist()
            Dd["dwell_xy"].append(((Cd.T @ (o[t_eff:t_rel + 1] - d[t_eff:t_rel + 1]).T).T[:, :2]).tolist())
            w = gr[t_rel:, 0] - gr[t_rel:, 1]
            wm = float(w.max()) if len(w) else 0.0
            op = np.flatnonzero(w >= 0.95 * wm) if len(w) else []
            Dd["rel_dur"].append(int(op[0]) + 1 if len(op) else 1)
    # per object type
    OBJ = {}
    for n, S in obj_s.items():
        if not S["off"]:
            continue
        off = np.array(S["off"])
        med = np.median(off, 0)
        ax = np.array(S["ax"])
        scat = np.abs(np.sum((off - med) * ax, 1))
        sym = G[n]["symmetric"]
        i = medoid(S["Rent"], sym)
        Rbar = S["Rent"][i]
        axm = np.median(np.abs(ax), 0)
        half = float(axm[0] * G[n]["hx"] + axm[1] * G[n]["hy"]) / max(1e-9, float(np.hypot(axm[0], axm[1])))
        OBJ[n] = dict(n=len(off), off=med.tolist(), zbar=q(S["zbar"], .5), rho=q(S["rho"], .25),
                      r_e=min(q(scat, .95), half), Rbar=Rbar.tolist(),
                      tau_R=q([posture_err(Rbar, Rx, sym) for Rx in S["Rent"]], .95),
                      width_band=[q(S["width"], .05), q(S["width"], .95)], s_app_obj=q(S["s_app"], .95))
    s_app_all = [x for S in obj_s.values() for x in S["s_app"]]
    # inheritance (A.6): an object type without grasp demos takes the nearest type's statistics under ||log phi - log phi'||
    # (phi = height, two widths; top fill not computed: simplest reading)
    def phi(n):
        return np.log(np.array([G[n]["top"] - G[n]["bottom"], 2 * G[n]["radius"], 2 * G[n]["radius"]]))
    inherit = {}
    for n in FC.MOVABLE:
        if n not in OBJ:
            inherit[n] = min(OBJ, key=lambda m: float(np.linalg.norm(phi(n) - phi(m))))
    DST = {}
    for de, Dd in dst_s.items():
        off = np.array(Dd["off"])
        med = np.median(off, 0)
        scat = np.linalg.norm(off - med, axis=1)
        omega = q(scat, .95)
        dist, zb = np.array(Dd["dist"]), np.array(Dd["zb"])
        prof = []
        for b in range(int(dist.max() / BIN) + 1):
            m = (dist >= b * BIN) & (dist < (b + 1) * BIN)
            if m.sum() >= MIN_BIN:
                prof.append([b * BIN + BIN / 2, q(zb[m], .5), q(zb[m], .05), int(m.sum())])
        dw = []
        for xy in Dd["dwell_xy"]:
            xy = np.array(xy)
            dw.append(int((np.linalg.norm(xy - med, axis=1) <= omega).sum()))
        DST[de] = dict(demos=Dd["demos"], tasks=sorted(Dd["tasks"]), off=med.tolist(), omega=omega, h_rel=q(Dd["h"], .5),
                       carry_radius=q(dist, .25), profile=prof, dwell=q(dw, .95), rel_dur=q(Dd["rel_dur"], .95),
                       enough=bool(Dd["demos"] >= MIN_DEMOS))
    res = dict(source="FC.DATA/demos.npz (LIBERO-Goal raw demos of trained tasks %s only)" % ",".join(PICK), geometry=G,
               sigma=[SIG_P, SIG_R], kappa=[float(np.median(kap_p)), float(np.median(kap_r))],
               s_app=q(s_app_all, .95), s_carry=q(s_carry, .95), s_R=q(s_rot, .95), obj=OBJ, inherit=inherit, dst=DST,
               skipped=ev_log, eps=0.004, n=3, N=40, N_after=75, budget=2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    pr = {k: v for k, v in res.items() if k not in ("obj", "dst")}
    print("[rg_stats]", json.dumps(pr), flush=True)
    for n, v in OBJ.items():
        print("[rg_stats] obj %s" % n, json.dumps({k: (np.round(v[k], 4).tolist() if k != "n" else v[k]) for k in v}), flush=True)
    for n, v in DST.items():
        vv = dict(v)
        vv["profile"] = [[round(x, 3) for x in p[:3]] + [p[3]] for p in v["profile"]]
        print("[rg_stats] dst %s" % n, json.dumps(vv), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

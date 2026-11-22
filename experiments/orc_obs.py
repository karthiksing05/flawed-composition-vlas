"""ORC test observations (docs/notes/plan_oracle_pi0.md 2.4, 2.6 step 4).  openpi venv, CPU.

Mirrors raafp_rv.convert's planner loop (planner_rel, RV.frames_of with roles (obj, "main_table"), RV.chan_state) on the rendered test pairs (rvp_orc_test: demo 2p = spot 0,
demo 2p+1 = spot 1 of pair p).  The observation set is fixed by the demo labels alone, before any model is run: a held frame of a test demo qualifies iff
  (i) the object is >= 3 cm above its resting height, (ii) it is <= 6 cm (xy) from its start, (iii) the demo's own label has |d5_xy| >= 1.0 action units (d5 = the sum of the
  first 5 label rows, xy), (iv) angle(u_i, u_j) >= 30 deg (u = unit xy vector from the object to the commanded / paired spot).
Up to 3 frames per demo, spaced >= 3 frames apart (the earliest qualifying ones).  Both directions of a pair are used.
One npz per observation (a2's capture format): image, wrist, state [raw 4-dim token], prompt, seed (crc32 of the id), actions (the label chunk, 10 rows), plus
state_cf, spot_i, spot_j, p_obj, u_i, u_j, layout (cluster), obj, d5_label_i, and the partner's / own first-qualifying-frame label d5 (for sigma_demo).

    python experiments/orc_obs.py --rvp /coc/.../fa1/rvp_orc_test --out results/flawed/orc/obs_test
"""
import argparse
import json
import os
import sys
import types
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sc_rv as RV                                                           # noqa: E402
import raafp_rv as RP                                                        # noqa: E402

LIFT, NEAR, D5MIN, ANG, SPACE, PER_DEMO, H = 0.03, 0.06, 1.0, 30.0, 3, 3, 10


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(unit(a), unit(b))), -1, 1))))


def d5_of(ac, i):
    return np.asarray(ac[i:i + 5, :2], np.float64).sum(0)


def qualifying(z, ch, gi, Hh, s_i, s_j):
    pos = np.asarray(ch["pos"], np.float64)
    ac = np.asarray(z["actions"], np.float64)
    p0 = pos[0, gi]
    out = []
    for i in range(len(ac) - 4):
        if Hh[i] <= 0.5:
            continue
        p = pos[i, gi]
        if p[2] - p0[2] < LIFT or np.linalg.norm(p[:2] - p0[:2]) > NEAR:
            continue
        if np.linalg.norm(d5_of(ac, i)) < D5MIN:
            continue
        if angle_deg(s_i - p[:2], s_j - p[:2]) < ANG:
            continue
        out.append(i)
    return out


def pick(frames):
    keep, last = [], -10 ** 9
    for i in frames:
        if i - last >= SPACE and len(keep) < PER_DEMO:
            keep.append(i)
            last = i
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    RV._openpi()
    T16 = RV.t16()
    ns = types.SimpleNamespace(suffix="", root=RV.RV_ROOT, arm="raac", ndemo=0)
    X = RV.lex(ns)
    import fa1_common as FC
    os.makedirs(a.out, exist_ok=True)
    sel_path = os.path.join(a.out, "selection.json")
    assert not os.path.exists(sel_path), "refusing to overwrite " + sel_path
    obs, per = [], {}
    for name, c, obj in RP.OBJS:
        gi = FC.ENT.index(obj)
        fs = RV.demo_files(a.rvp, c)
        assert len(fs) % 2 == 0 and fs, (name, len(fs))
        Z = [dict(np.load(f)) for f in fs]
        CH = [np.load(os.path.join(a.rvp, c, "chan_%s" % os.path.basename(f)[5:])) for f in fs]
        info = {}
        PR = [RP.planner_rel(Z[k], CH[k], obj, T16, X, info) for k in range(len(fs))]
        for k in range(len(fs)):
            m = k ^ 1                                                          # the pair partner (2p <-> 2p+1)
            s_i, s_j = np.asarray(Z[k]["spot"], np.float64)[:2], np.asarray(Z[m]["spot"], np.float64)[:2]
            assert int(Z[k]["seed"]) == int(Z[m]["seed"]), "pair partners must share the layout seed"
            assert np.linalg.norm(s_i - s_j) >= 0.15
            rel, Hh = PR[k]
            q = qualifying(Z[k], CH[k], gi, Hh, s_i, s_j)
            q_partner = qualifying(Z[m], CH[m], gi, PR[m][1], s_j, s_i)
            per[(name, k)] = len(pick(q))
            if not q or not q_partner:
                continue
            fi, fj = q[0], q_partner[0]                                        # first qualifying frame of each demo (sigma_demo reference)
            d5_first_i, d5_first_j = d5_of(np.asarray(Z[k]["actions"], np.float64), fi), d5_of(np.asarray(Z[m]["actions"], np.float64), fj)
            roles = {c: (obj, "main_table")}
            frames = list(RV.frames_of(Z[k], c, "raac", X, T16, roles, rel))
            for i in pick(q):
                img, wimg, task, h, na, nw = frames[i]
                assert h, "held frame expected"
                tok = RV.chan_state(rel[i, 1], True, True)
                p = np.asarray(CH[k]["pos"], np.float64)[i, gi]
                d = np.array([s_j[0] - s_i[0], s_j[1] - s_i[1], 0.0, 0.0], np.float32)
                tok_cf = (tok + d).astype(np.float32)
                assert tok[3] == 2.0 and tok_cf[3] == 2.0 and np.array_equal(tok_cf[2:], tok[2:])
                oid = "orc_%s_p%02d_s%d_f%03d" % (name, k // 2, k % 2, i)
                lab = np.asarray(Z[k]["actions"], np.float32)
                ch10 = lab[i:i + H]
                if len(ch10) < H:
                    ch10 = np.concatenate([ch10, np.repeat(ch10[-1:], H - len(ch10), 0)])
                meta = dict(id=oid, obj=name, layout="%s_%d" % (name, k // 2), pair=k // 2, spot_idx=k % 2, frame=i, demo=os.path.basename(fs[k]))
                np.savez(os.path.join(a.out, oid + ".npz"), image=np.asarray(img), wrist=np.asarray(wimg), state=tok, state_cf=tok_cf, prompt=RP.TABLE_PHRASE,
                         seed=zlib.crc32(oid.encode()) % (2 ** 31), actions=ch10, spot_i=s_i, spot_j=s_j, p_obj=p.astype(np.float64), u_i=unit(s_i - p[:2]), u_j=unit(s_j - p[:2]),
                         d5_label_i=d5_of(np.asarray(Z[k]["actions"], np.float64), i), d5_first_i=d5_first_i, d5_first_j=d5_first_j, meta=np.array(json.dumps(meta)))
                obs.append(meta)
    layouts = sorted({o["layout"] for o in obs})
    sel = dict(n=len(obs), n_layouts=len(layouts), per_demo_counts={"%s_%d" % k: v for k, v in per.items()}, by_obj={n: sum(o["obj"] == n for o in obs) for n, _, _ in RP.OBJS},
               rules=dict(lift=LIFT, near=NEAR, d5min=D5MIN, angle_deg=ANG, spacing=SPACE, per_demo=PER_DEMO), untestable=len(obs) < 120, ids=[o["id"] for o in obs])
    with open(sel_path, "x", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    print("[orc_obs] %d observations, %d layouts (k1: need >= 120); by object %s" % (len(obs), len(layouts), sel["by_obj"]), flush=True)


if __name__ == "__main__":
    main()

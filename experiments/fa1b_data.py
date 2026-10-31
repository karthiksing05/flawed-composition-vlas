"""FLAWED_COMPOSITION prototype A, stage 1b (section 5.6), DATA: the physical-contact phase label and the grasp offset for stage 1's
training set (DATA/demos.npz, fa1_data.py poses).  Demos are READ: each kept step's recorded simulator state is set in the task's own
LIBERO-Goal env (exactly fa1_data.py poses' loop and order) and the contacts of that state are read (fa1b_common); no demo action is
executed.  LIBERO venv.

    held  -> DATA/demos_b.npz (held [n] = fa1b_common.Held for the task's role object, stepped over each demo's kept steps;
             off [n, 3] the grasp offset, world frame; touch [n, 2] the raw finger contacts; lift [n]; aligned row for row with
             demos.npz, checked on ee_pos) and results/flawed/a1b_held.json (per task and rule: held fraction, held segments per demo,
             steps from the first close command to the first held step, the share of close-commanded steps that are held)

    python experiments/fa1b_data.py held
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402
import fa1b_common as FB                                                    # noqa: E402

OUT = os.path.join(FC.ROOT, "results", "flawed")


def seg_stats(h, demo, close):
    out = dict(held_frac=float(h.mean()), segments_hist=[], lag=[], close_share=float(h[close].mean()) if close.any() else None)
    for dm in np.unique(demo):
        m = demo == dm
        x, c = h[m], close[m]
        out["segments_hist"].append(int(np.sum(x[1:] & ~x[:-1]) + int(x[0])))
        if x.any() and c.any():
            out["lag"].append(int(np.argmax(x)) - int(np.argmax(c)))
    out["segments_hist"] = np.bincount(out["segments_hist"]).tolist()
    lag = out.pop("lag")
    out["lag_median"] = float(np.median(lag)) if lag else None
    out["lag_min_max"] = [int(min(lag)), int(max(lag))] if lag else None
    return out


def held():
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    H, OFF, EE, TOUCH, LIFT = [], [], [], [], []
    rep = {}
    for ti, (k, name) in enumerate(FC.KNOWN.items()):
        r = np.load(os.path.join(FC.DATA, f"raw_{k}.npz"))
        env = FC.make_env(os.path.join(FC.LIBBDDL, name + ".bddl"))
        env.seed(0)
        env.reset()
        o = FC.roles(lang[k])[0]
        con = FB.Contact(env, [o])
        if ti == 0:
            m = env.env.sim.model
            print("[fa1b] left finger geoms", sorted(m.geom_id2name(g) for g in con.left), flush=True)
            print("[fa1b] right finger geoms", sorted(m.geom_id2name(g) for g in con.right), flush=True)
        last, hm, n0 = -1, None, len(H)
        for st, dm in zip(r["states"], r["demo"]):
            if dm != last:
                hm, last = FB.Held(), dm
            obs = env.set_init_state(st)
            lt, rt = con.touching(o)
            p = con.pos(o)
            ee = np.asarray(obs["robot0_eef_pos"], np.float64)
            h, off = hm.step(lt, rt, p, ee)
            H.append(h); OFF.append(off); EE.append(ee); TOUCH.append((lt, rt)); LIFT.append(p[2] - hm.z0)
        env.close()
        h = np.asarray(H[n0:], bool)
        tc = np.asarray(TOUCH[n0:], bool)
        lf = np.asarray(LIFT[n0:])
        close = r["actions"][:, 6] > 0
        rep[k] = dict(object=o, held_machine=seg_stats(h, r["demo"], close), both_contact=seg_stats(tc.all(1), r["demo"], close),
                      both_contact_and_lift=seg_stats(tc.all(1) & (lf >= FB.LIFT), r["demo"], close),
                      max_lift_median=float(np.median([lf[r["demo"] == d].max() for d in np.unique(r["demo"])])))
        print(f"[fa1b] {k} {name} ({o}): {json.dumps(rep[k])}", flush=True)
    EE = np.asarray(EE, np.float32)
    assert EE.shape == D["ee_pos"].shape and np.abs(EE - D["ee_pos"]).max() < 1e-5, "row order differs from demos.npz"
    np.savez_compressed(os.path.join(FC.DATA, "demos_b.npz"), held=np.asarray(H, np.float32), off=np.asarray(OFF, np.float32),
                        touch=np.asarray(TOUCH, np.float32), lift=np.asarray(LIFT, np.float32))
    with open(os.path.join(OUT, "a1b_held.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rep, n=len(H), lift=FB.LIFT,
                       held_rule="on: role object touches both fingers and is lifted >= LIFT above its first-step height; stays on "
                                 "while it touches either finger; off when it touches neither (MuJoCo contacts of the recorded state)"),
                  fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("held",))
    {"held": held}[ap.parse_args().mode]()

"""FLAWED_COMPOSITION prototype A, stage 1 (section 5.5), DATA: the oracle-state training set and the relaxed-placement calibration.
Demos are READ (their recorded simulator states are set and the objects' poses read); no demo action is ever executed.

    extract (openpi venv, h5py): the raw LIBERO-Goal hdf5 demos of N2's 8 KNOWN tasks (50 each) -> DATA/raw_<K>.npz
                                 (recorded states, actions; no-op steps dropped as in p0_demo_init.noop_mask)
    poses   (LIBERO venv): per kept step, set the recorded state in the task's own LIBERO-Goal env and read the oracle scene
                           (fa1_common.scene) -> DATA/demos.npz; prints the action-frame check, yaw spreads, contact events
    calib   (LIBERO venv): the relaxed predicate's rest heights.  On N2's smoke states 0-3 (familiar layout), each movable object is
                           teleported 0.12 m above each target reference (plate / bowl body, stove cook site, cabinet top-side site,
                           rack top site, stove-front site), 60 dummy steps settle it, and dz = object z - reference z and the
                           horizontal offset are recorded; for the 14 N2 compounds whose BDDL has that goal (all but the drawer
                           pair), LIBERO's strict check too (feasibility) -> results/flawed/a1_calib.json
    equiv   (LIBERO venv, GPU node): N2's OffScreenRenderEnv against fa1_common's camera-free ControlEnv on one N2 state and a fixed
                           random action sequence: the simulator states must agree -> results/flawed/a1_equiv.json

    python experiments/fa1_data.py extract|poses|calib|equiv
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(FC.ROOT, "results", "flawed")


def noop_mask(actions):
    keep, prev = [], None
    for a in actions:
        keep.append(not (np.linalg.norm(a[:-1]) < 1e-4 and prev is not None and a[-1] == prev[-1]))
        prev = a
    return np.asarray(keep)


def extract():
    import h5py
    os.makedirs(FC.DATA, exist_ok=True)
    for k, name in FC.KNOWN.items():
        with h5py.File(os.path.join(FC.RAW, f"{name}_demo.hdf5"), "r") as f:
            d = f["data"]
            demos = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
            S, A, I, Tn = [], [], [], []
            for j, dm in enumerate(demos):
                st, ac = np.asarray(d[dm]["states"]), np.asarray(d[dm]["actions"])
                keep = noop_mask(ac)
                S.append(st[keep]); A.append(ac[keep]); I.append(np.full(keep.sum(), j)); Tn.append(len(ac))
        np.savez_compressed(os.path.join(FC.DATA, f"raw_{k}.npz"), states=np.concatenate(S), actions=np.concatenate(A),
                            demo=np.concatenate(I), raw_len=np.array(Tn))
        print(f"[fa1] {k} {name}: {len(demos)} demos, {sum(len(a) for a in A)} kept of {sum(Tn)} steps", flush=True)


def poses():
    rows = {x: [] for x in ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip")}
    for ti, (k, name) in enumerate(FC.KNOWN.items()):
        r = np.load(os.path.join(FC.DATA, f"raw_{k}.npz"))
        env = FC.make_env(os.path.join(FC.LIBBDDL, name + ".bddl"))
        env.seed(0)
        env.reset()
        dj = FC.drawer_joint(env)
        assert r["states"].shape[1] == env.get_sim_state().shape[0], (r["states"].shape, env.get_sim_state().shape)
        prev, last_demo, step = -1.0, -1, 0
        for st, ac, dm in zip(r["states"], r["actions"], r["demo"]):
            if dm != last_demo:
                prev, last_demo, step = -1.0, dm, 0
            obs = env.set_init_state(st)
            sc = FC.scene(env, obs, dj)
            for x in ("ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                rows[x].append(sc[x])
            rows["task"].append(ti); rows["demo"].append(dm); rows["step"].append(step); rows["action"].append(ac)
            rows["prev_grip"].append(prev)
            prev, step = float(ac[6]), step + 1
        env.close()
        print(f"[fa1] poses {k}: {len(r['actions'])} steps (drawer joint {dj})", flush=True)
    D = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in rows.items()}
    np.savez_compressed(os.path.join(FC.DATA, "demos.npz"), ent=np.array(FC.ENT), tasks=np.array(list(FC.KNOWN)), **D)
    # diagnostics: does the action's translation act along the world axes?  next ee position - this one, within a demo
    same = (D["demo"][1:] == D["demo"][:-1]) & (D["task"][1:] == D["task"][:-1])
    dx = (D["ee_pos"][1:] - D["ee_pos"][:-1])[same]
    a = D["action"][:-1][same][:, :3]
    M, *_ = np.linalg.lstsq(a, dx, rcond=None)
    print("[fa1] action->world displacement map (rows: action dims):\n", np.round(M, 4), flush=True)
    for ti, k in enumerate(FC.KNOWN):
        m = D["task"] == ti
        first = D["step"][m] == 0
        y = D["yaw"][m][first]
        p = D["pos"][m][first]
        closes = [int(np.sum((D["action"][m][D["demo"][m] == dm][1:, 6] > 0) & (D["action"][m][D["demo"][m] == dm][:-1, 6] <= 0)))
                  for dm in np.unique(D["demo"][m])]
        print(f"[fa1] {k}: start pos spread {np.round(np.linalg.norm(p - p.mean(0), axis=-1).max(0), 3).tolist()} m; yaw spread "
              f"{np.round(y.max(0) - y.min(0), 2).tolist()}; close events per demo {np.bincount(closes).tolist()}", flush=True)


TREF = ["plate_1", "akita_black_bowl_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side", "wine_rack_1_top_region",
        "main_table_stove_front_region"]


def calib():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    fam = [t for t in n2["tasks"] if t["layout"] == "familiar"]
    by_pair = {(t["object"], t["target"]): t for t in fam}
    res = {}
    for o in FC.MOVABLE:
        for tg in TREF:
            if tg == o:
                continue
            t = by_pair.get((o, tg))
            bddl = os.path.join(FC.ROOT, (t or by_pair[("akita_black_bowl_1", "plate_1")])["bddl"])
            S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{(t or by_pair[('akita_black_bowl_1', 'plate_1')])['tid']}.npy"))
            out = []
            for k in range(4):
                env = FC.fresh_env(bddl)
                env.reset()
                obs = env.set_init_state(S[k])
                for _ in range(10):
                    obs, _, _, _ = env.step(FC.DUMMY)
                sim = env.env.sim
                dj = FC.drawer_joint(env)
                sc = FC.scene(env, obs, dj)
                ref = sc["pos"][FC.ENT.index(tg)]
                m = sim.model
                a = int(m.jnt_qposadr[m.body_jntadr[env.env.obj_body_id[o]]])
                va = int(m.jnt_dofadr[m.body_jntadr[env.env.obj_body_id[o]]])
                sim.data.qpos[a:a + 3] = [ref[0], ref[1], ref[2] + 0.12 + FC.SIZE[o][1] / 2]
                sim.data.qvel[va:va + 6] = 0.0
                sim.forward()
                done = False
                for _ in range(60):
                    obs, _, d_, _ = env.step(FC.DUMMY)
                    done = done or bool(d_)
                sc = FC.scene(env, obs, dj)
                p, ref = sc["pos"][FC.ENT.index(o)], sc["pos"][FC.ENT.index(tg)]
                out.append(dict(dz=float(p[2] - ref[2]), dxy=float(np.linalg.norm(p[:2] - ref[:2])),
                                strict=(bool(done or env.check_success()) if t else None)))
                env.close()
            res[f"{o}|{tg}"] = dict(object=o, target=tg, compound=t["compound"] if t else None,
                                    dz_mean=float(np.mean([x["dz"] for x in out])), dz=[x["dz"] for x in out],
                                    dxy=[x["dxy"] for x in out], strict=[x["strict"] for x in out])
            print(f"[fa1] calib {o} -> {tg} ({t['compound'] if t else '-'}): dz {np.round([x['dz'] for x in out], 3).tolist()} "
                  f"dxy {np.round([x['dxy'] for x in out], 3).tolist()} strict {[x['strict'] for x in out]}", flush=True)
    with open(os.path.join(OUT, "a1_calib.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(pairs=res, note="teleport 0.12 m above the reference, 60 dummy steps, N2 familiar smoke states 0-3"), fh, indent=1)


def equiv():
    from libero.libero.envs import OffScreenRenderEnv
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    t = n2["tasks"][0]
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    rng = np.random.default_rng(0)
    acts = np.concatenate([rng.uniform(-1, 1, (120, 6)) * 0.5, np.where(rng.uniform(size=(120, 1)) > 0.7, 1.0, -1.0)], 1)
    finals = []
    for kind in ("offscreen", "control"):
        np.random.seed(7)
        env = (OffScreenRenderEnv(bddl_file_name=os.path.join(FC.ROOT, t["bddl"]), camera_heights=256, camera_widths=256)
               if kind == "offscreen" else FC.make_env(os.path.join(FC.ROOT, t["bddl"])))
        env.seed(7)
        env.reset()
        env.set_init_state(S[4])
        for _ in range(10):
            env.step(FC.DUMMY)
        for a in acts:
            env.step(a.tolist())
        finals.append(env.get_sim_state())
        env.close()
    d = float(np.abs(finals[0] - finals[1]).max())
    print(f"[fa1] equiv: max |state difference| after 130 steps = {d:.3e}", flush=True)
    with open(os.path.join(OUT, "a1_equiv.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(max_abs_state_diff=d, steps=130, task=t["key"], state=4), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("extract", "poses", "calib", "equiv"))
    {"extract": extract, "poses": poses, "calib": calib, "equiv": equiv}[ap.parse_args().mode]()

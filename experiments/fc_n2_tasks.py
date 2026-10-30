"""FLAWED_COMPOSITION N2 (section 4.11), DATA ONLY: the 2 x 2 factorial's tasks and start states, inside ONE scene (LIBERO-Goal's
kitchen table: akita black bowl, cream cheese, wine bottle, plate; wooden cabinet, flat stove, wine rack).  No policy is run, no
demo is read or executed.  The LIBERO venv (py3.8), EGL on one GPU.

Factor A, compound (relative to the 40 LIBERO tasks pi0.5_libero and ar0s/groot_libero were fine-tuned on; checked by rule in
fc_n2_novelty.py): 8 KNOWN = LIBERO-Goal's own 8 object -> target tasks (its instruction, its goal atom); 8 NEW = the same scene's
objects and targets re-paired so that no training task has the (object type, target, region) slot, while the object and the target
each occur in some training task's goal.  The 8 pairs share a target (or verb) with their known partner where the scene allows.

Factor B, layout: FAMILIAR = LIBERO's own placement sampler on the scene's training regions (the BDDL of every task is the template
put_the_cream_cheese_in_the_bowl.bddl with only its language, obj_of_interest and goal replaced), seeds 600000 + k.  NEW = the same
state k with the four movable objects relocated (seed 700000 + k): each >= 0.10 m from its own training position (the mean over the
template task's 50 stored init states; their spread is ~1 cm) and >= 0.06 m from every other movable object's, inside the box the four
training positions span widened by 0.06 m, >= r_i + r_j + 0.02 from each other and from the fixtures, and clear of the stove-front
region; accepted only if 10 settle steps of the dummy action move no object by more than 2 cm.  Fixtures do not move.
States 0-3 are the smoke set, 4-23 the main set; every task of a layout uses the same 24 states (so compounds differ only in the
instruction and goal).  Task id = 2 * compound index + layout (0 familiar, 1 new).

-> results/flawed/n2_bddl/<key>.bddl, n2_states/n2_goal_<tid>.npy [24, state_dim], n2_tasks.json, n2_branches_{smoke,main}.json
   (one fresh, re-seeded env per episode through the client's branch path; noise seed_base 13000000 + 1000 k, shared by every
   cell and model), n2_render/n2_goal_<tid>.npz (the N1 render convention on main states 4-13) and check_*.png.

    python experiments/fc_n2_tasks.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
BD, ST, RD = (os.path.join(OUT, d) for d in ("n2_bddl", "n2_states", "n2_render"))
LIB = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/bddl_files/libero_goal"
TEMPLATE = os.path.join(LIB, "put_the_cream_cheese_in_the_bowl.bddl")
SUITE, MAX_STEPS = "n2_goal", 300
N_SMOKE, N_MAIN, BASE_FAM, BASE_NEW = 4, 20, 600000, 700000
DUMMY = [0.0] * 6 + [-1.0]
# key, status, language (LIBERO's file-name prompt), goal atom, object instance, partner / training file
PAIRS = [
    ("K1", "known", "put the bowl on the plate", "On akita_black_bowl_1 plate_1", "put_the_bowl_on_the_plate"),
    ("K2", "known", "put the bowl on the stove", "On akita_black_bowl_1 flat_stove_1_cook_region", "put_the_bowl_on_the_stove"),
    ("K3", "known", "put the bowl on top of the cabinet", "On akita_black_bowl_1 wooden_cabinet_1_top_side", "put_the_bowl_on_top_of_the_cabinet"),
    ("K4", "known", "put the wine bottle on top of the cabinet", "On wine_bottle_1 wooden_cabinet_1_top_side", "put_the_wine_bottle_on_top_of_the_cabinet"),
    ("K5", "known", "put the wine bottle on the rack", "On wine_bottle_1 wine_rack_1_top_region", "put_the_wine_bottle_on_the_rack"),
    ("K6", "known", "put the cream cheese in the bowl", "On cream_cheese_1 akita_black_bowl_1", "put_the_cream_cheese_in_the_bowl"),
    ("K7", "known", "open the top drawer and put the bowl inside", "In akita_black_bowl_1 wooden_cabinet_1_top_region",
     "open_the_top_drawer_and_put_the_bowl_inside"),
    ("K8", "known", "push the plate to the front of the stove", "On plate_1 main_table_stove_front_region", "push_the_plate_to_the_front_of_the_stove"),
    ("N1", "new", "put the cream cheese on the plate", "On cream_cheese_1 plate_1", "K1"),
    ("N2", "new", "put the cream cheese on the stove", "On cream_cheese_1 flat_stove_1_cook_region", "K2"),
    ("N3", "new", "put the cream cheese on top of the cabinet", "On cream_cheese_1 wooden_cabinet_1_top_side", "K3"),
    ("N4", "new", "put the wine bottle on the plate", "On wine_bottle_1 plate_1", "K1"),
    ("N5", "new", "put the wine bottle on the stove", "On wine_bottle_1 flat_stove_1_cook_region", "K2"),
    ("N6", "new", "put the wine bottle in the bowl", "On wine_bottle_1 akita_black_bowl_1", "K6"),
    ("N7", "new", "open the top drawer and put the cream cheese inside", "In cream_cheese_1 wooden_cabinet_1_top_region", "K7"),
    ("N8", "new", "push the bowl to the front of the stove", "On akita_black_bowl_1 main_table_stove_front_region", "K8"),
]
MOVR = {"akita_black_bowl_1": 0.06, "cream_cheese_1": 0.045, "wine_bottle_1": 0.045, "plate_1": 0.10}   # b1_3_gen footprints
FIXR = {"wooden_cabinet_1": 0.13, "flat_stove_1": 0.11, "wine_rack_1": 0.10}
REGION = {"akita_black_bowl_1": (-0.09, 0.0), "cream_cheese_1": (-0.05, 0.13), "wine_bottle_1": (-0.20, -0.05), "plate_1": (0.05, -0.02)}
STOVE_FRONT = (-0.09, 0.17, -0.01, 0.25)            # table frame (x0, y0, x1, y1), the template's stove_front_region
OWN_MIN, OTHER_MIN, BOX_PAD, GAP, SETTLE_MAX = 0.10, 0.0, 0.08, 0.02, 0.02


def bddl_of(text, lang, atom, obj):
    tgt = atom.split()[2]
    interest = [obj] + ([tgt] if tgt in MOVR else [])
    t = re.sub(r"\(:language [^)]*\)", f"(:language {lang})", text, count=1)
    t = re.sub(r"\(:obj_of_interest[^)]*\)", "(:obj_of_interest\n    " + "\n    ".join(interest) + "\n  )", t, count=1)
    t = re.sub(r"\(:goal\s*\(And \([^)]*\)\)\s*\)", f"(:goal\n    (And ({atom}))\n  )", t, count=1)
    assert f"({atom})" in t and f"(:language {lang})" in t, atom
    return t


def make_env(path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=path, camera_heights=256, camera_widths=256)


def positions(env, names):
    sim, inner = env.env.sim, env.env
    return np.array([np.array(sim.data.body_xpos[inner.obj_body_id[n]]) for n in names])


def qadr(env, n):
    m = env.env.sim.model
    return int(m.jnt_qposadr[m.body_jntadr[env.env.obj_body_id[n]]])


def main():
    from libero.libero import benchmark
    os.makedirs(BD, exist_ok=True)
    os.makedirs(ST, exist_ok=True)
    os.makedirs(RD, exist_ok=True)
    text = open(TEMPLATE, encoding="utf-8").read()
    paths = {}
    for key, _, lang, atom, _ in PAIRS:
        obj = atom.split()[1]
        paths[key] = os.path.join(BD, key + ".bddl")
        with open(paths[key], "w", encoding="utf-8") as fh:
            fh.write(bddl_of(text, lang, atom, obj))
    mov = list(MOVR)
    # the training distribution: the template task's 50 stored init states
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tidx = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == "put_the_cream_cheese_in_the_bowl")
    stored = suite.get_task_init_states(tidx)
    env = make_env(paths["K1"])
    env.seed(0)
    tr = []
    for s in stored:
        env.reset()
        env.set_init_state(s)
        tr.append(positions(env, mov)[:, :2])
    tr = np.stack(tr)                                                          # [50, 4, 2]
    centre = tr.mean(0)
    spread = float(np.linalg.norm(tr - centre[None], axis=-1).max())
    offset = np.mean([centre[i] - np.array(REGION[n]) for i, n in enumerate(mov)], axis=0)   # table frame -> world
    fixn = list(FIXR)
    fixp = positions(env, fixn)[:, :2]
    lo, hi = centre.min(0) - BOX_PAD, centre.max(0) + BOX_PAD
    sf = (STOVE_FRONT[0] + offset[0], STOVE_FRONT[1] + offset[1], STOVE_FRONT[2] + offset[0], STOVE_FRONT[3] + offset[1])
    print(f"[n2] training centres {np.round(centre, 3).tolist()}, spread {spread:.3f} m, box {np.round(lo, 3)}-{np.round(hi, 3)}", flush=True)

    # familiar states: LIBERO's sampler, the same seeds for every BDDL (checked identical on the first two)
    n = N_SMOKE + N_MAIN
    fam = []
    for k in range(n):
        np.random.seed(BASE_FAM + k)
        env.seed(BASE_FAM + k)
        env.reset()
        fam.append(env.sim.get_state().flatten())
    fam = np.stack(fam)
    assert fam.shape[1] == stored.shape[1], (fam.shape, stored.shape)
    for key in paths:
        if key == "K1":
            continue
        e2 = make_env(paths[key])
        for k in range(2):
            np.random.seed(BASE_FAM + k)
            e2.seed(BASE_FAM + k)
            e2.reset()
            assert np.allclose(e2.sim.get_state().flatten(), fam[k]), f"{key}: state {k} differs from K1's"
        e2.close()

    def ok(p):
        for i, nm in enumerate(mov):
            r = MOVR[nm]
            if np.any(p[i] < lo) or np.any(p[i] > hi):
                return False
            if np.linalg.norm(p[i] - centre[i]) < OWN_MIN:
                return False
            if any(np.linalg.norm(p[i] - centre[j]) < OTHER_MIN for j in range(len(mov)) if j != i):
                return False
            if any(np.linalg.norm(p[i] - p[j]) < r + MOVR[mov[j]] + GAP for j in range(i)):
                return False
            if any(np.linalg.norm(p[i] - fixp[f]) < r + FIXR[fn] + GAP for f, fn in enumerate(fixn)):
                return False
            cx, cy = np.clip(p[i][0], sf[0], sf[2]), np.clip(p[i][1], sf[1], sf[3])
            if np.hypot(p[i][0] - cx, p[i][1] - cy) < r + GAP:
                return False
        return True

    new, newpos, fampos, settle = [], [], [], []
    for k in range(n):
        rng = np.random.default_rng(BASE_NEW + k)
        env.reset()
        env.set_init_state(fam[k])
        p0 = positions(env, mov)
        fampos.append(p0[:, :2])
        for _ in range(10):
            env.step(DUMMY)
        dz_fam = positions(env, mov)[:, 2] - p0[:, 2]          # the familiar state's own settle (objects start slightly above rest)
        done, n_ok, last = False, 0, None
        for _ in range(200000):
            p = rng.uniform(lo, hi, size=(len(mov), 2))
            if not ok(p):
                continue
            n_ok += 1
            if n_ok > 400:
                break
            env.reset()
            env.set_init_state(fam[k])
            sim = env.env.sim
            for i, nm in enumerate(mov):
                a = qadr(env, nm)
                sim.data.qpos[a:a + 2] = p[i]
            sim.forward()
            s_new = sim.get_state().flatten()
            env.reset()
            env.set_init_state(s_new)
            q0 = positions(env, mov)
            for _ in range(10):
                env.step(DUMMY)
            q1 = positions(env, mov)
            d = float(max(np.linalg.norm(q1[:, :2] - q0[:, :2], axis=-1).max(), np.abs((q1[:, 2] - q0[:, 2]) - dz_fam).max()))
            last = (d, float(np.abs(q0[:, :2] - p).max()), np.round(q1 - q0, 3).tolist(), np.round(dz_fam, 3).tolist())
            if d <= SETTLE_MAX and np.allclose(q0[:, :2], p, atol=1e-4):
                new.append(s_new)
                newpos.append(p)
                settle.append(d)
                done = True
                break
        assert done, f"state {k}: no valid new layout ({n_ok} geometric candidates; last settle / placement error {last})"
    new = np.stack(new)
    env.close()
    print(f"[n2] {n} new layouts, max settle displacement {max(settle):.4f} m", flush=True)

    tasks = []
    for ci, (key, status, lang, atom, partner) in enumerate(PAIRS):
        for li, lay in enumerate(("familiar", "new")):
            tid = 2 * ci + li
            np.save(os.path.join(ST, f"{SUITE}_{tid}.npy"), fam if li == 0 else new)
            g = atom.split()
            tasks.append(dict(tid=tid, key=f"{key}_{lay[:3]}", compound=key, status=status, layout=lay, language=lang,
                              bddl=os.path.relpath(paths[key], ROOT), goal=g, object=g[1], target=g[2],
                              training_file=partner if status == "known" else None, partner=partner if status == "new" else None))
    br = {ph: [dict(id=f"k{k}", trial=k, seed_base=13000000 + 1000 * k, at=0, variant="n2") for k in ks]
          for ph, ks in (("smoke", range(N_SMOKE)), ("main", range(N_SMOKE, n)))}
    for ph, v in br.items():
        with open(os.path.join(OUT, f"n2_branches_{ph}.json"), "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1)
    rel = lambda P: np.linalg.norm(P - centre[None], axis=-1)                   # noqa: E731
    meta = dict(suite=SUITE, max_steps=MAX_STEPS, template=TEMPLATE, tasks=tasks, movable=mov, fixtures=fixn,
                states=dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, n)), base_familiar=BASE_FAM, base_new=BASE_NEW),
                training=dict(centre=centre.tolist(), spread=spread, n_stored=len(stored), table_offset=offset.tolist()),
                constraints=dict(own_min=OWN_MIN, other_min=OTHER_MIN, box=[lo.tolist(), hi.tolist()], gap=GAP, settle_max=SETTLE_MAX,
                                 fixture_xy=fixp.tolist(), stove_front_world=list(sf)),
                positions=dict(familiar=np.stack(fampos).tolist(), new=np.stack(newpos).tolist()),
                shift=dict(familiar_mean=rel(np.stack(fampos)).mean(0).tolist(), new_mean=rel(np.stack(newpos)).mean(0).tolist(),
                           new_min=rel(np.stack(newpos)).min(0).tolist()),
                settle_max_observed=max(settle))
    with open(os.path.join(OUT, "n2_tasks.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    # renders (fc_n1_render's convention) on main states 4-13, for the N1 input-novelty measures
    from openpi_client import image_tools
    import imageio
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from fc_n1_render import quat2axisangle
    grid = []
    for t in tasks:
        env = make_env(os.path.join(ROOT, t["bddl"]))
        env.seed(0)
        inner = env.env
        names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
        kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
        S = np.load(os.path.join(ST, f"{SUITE}_{t['tid']}.npy"))
        rec = {k_: [] for k_ in ("img", "wimg", "state8", "pos", "base")}
        for s in S[N_SMOKE:N_SMOKE + 10]:
            env.reset()
            obs = env.set_init_state(s)
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY)
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            rec["img"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
            rec["wimg"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)))
            rec["state8"].append(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                 obs["robot0_gripper_qpos"])).astype(np.float32))
            sim = inner.sim
            rec["pos"].append(np.asarray([np.asarray(obs[nm + "_pos"], np.float64) if (nm + "_pos") in obs
                                          else np.array(sim.data.body_xpos[inner.obj_body_id[nm]]) for nm in names], np.float32))
            rec["base"].append(np.array(sim.data.body_xpos[sim.model.body_name2id("robot0_base")], np.float32))
        np.savez_compressed(os.path.join(RD, f"{SUITE}_{t['tid']}.npz"), names=np.array(names), kinds=np.array(kinds),
                            language=t["language"], **{k_: np.stack(v) for k_, v in rec.items()})
        if t["compound"] == "K1":
            grid.append(np.concatenate(rec["img"][:4], axis=1))
        env.close()
        print(f"[n2] rendered {t['key']} '{t['language']}'", flush=True)
    imageio.imwrite(os.path.join(RD, "check_K1_fam_new.png"), np.concatenate(grid, axis=0))
    print("[n2] done", flush=True)


if __name__ == "__main__":
    main()

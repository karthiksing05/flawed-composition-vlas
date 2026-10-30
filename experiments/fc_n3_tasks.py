"""FLAWED_COMPOSITION N3 (section 4.12), DATA ONLY: the 2 x 2 compound x layout factorial in LIBERO-Spatial's scene (two akita black
bowls, cookie box, glazed ramekin, plate; wooden cabinet and flat stove as fixtures).  No policy is run, no demo is read or executed.
The LIBERO venv (py3.8), EGL on one GPU.

Why this scene.  Of the scenes the 40 training tasks use, only LIBERO-Goal's (N2) has two moved object types with two trained
targets each; LIBERO-Spatial's has 10 trained referents ("the black bowl next to the ramekin", ...) with ONE target (the plate).  N3
therefore re-pairs REFERENT and TARGET: a known compound K_i is LIBERO-Spatial task i itself (its BDDL, its file-name prompt, its
goal On bowl_1 plate_1); its new partner N_i is the same BDDL with only the language and the goal replaced, the target now the stove
(On bowl_1 flat_stove_1_cook_region, "... and place it on the stove") or the cabinet top (On bowl_1 wooden_cabinet_1_top_side,
"... and place it on top of the cabinet").  Both targets are trained with this very object (LIBERO-Goal: put the bowl on the stove /
on top of the cabinet), so object, grasp and target are each matched; only the (referent, target) joint and the (scene, target)
joint are new.  Every pair shares its grasp phase exactly (same states, same bowl).
Tasks used: spatial 0, 1, 2, 3, 4, 5, 6, 8.  Excluded: 7 and 9 (the named bowl starts on one fixture and the other bowl on the
other, so no free target).  The new target is the free fixture; where both are free they alternate (cabinet first) in index order.

Factor B, layout: FAMILIAR = LIBERO's sampler on the task's own BDDL (seeds 600000 + k).  NEW = the same state k with every
table-resting movable object (and whatever is stacked on it) translated by ONE vector v, |v| in [0.10, 0.16] m (seed 700000 + k),
so every spatial relation a referent names ("next to the ramekin", "on the cookie box", "between the plate and the ramekin") is kept;
objects resting on a fixture (a bowl on the cabinet top, on the stove, in the drawer) stay.  Accepted only if every moved object stays
inside the box the movers' training positions span (all 8 tasks) widened by 0.08 m, clears the fixtures (cabinet body, stove body,
stove cook site) by r_obj + r_fix + 0.02, and 10 settle steps of the dummy action move no object more than 2 cm in the plane nor drop
it more than 2 cm beyond the familiar state's own settle.
Feasibility and the relaxed predicate: for every pair, on smoke states 0-3, bowl_1 is teleported 0.12 m above each FREE target (plate
centre, stove cook site, cabinet top-side site), the dummy action runs 60 steps, and LIBERO's strict goal check of the BDDL whose
target that is is read; the rest height dz (bowl z - target reference z) is recorded per target (the relaxed predicate's band).
States 0-3 are the smoke set, 4-23 the main set; both tasks of a pair use the same 24 states.  Task id = 2 * compound index + layout.

-> results/flawed/n3_bddl/<key>.bddl, n3_states/n3_spatial_<tid>.npy, n3_tasks.json, n3_branches_{smoke,main}.json

    python experiments/fc_n3_tasks.py
"""
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
BD, ST = (os.path.join(OUT, d) for d in ("n3_bddl", "n3_states"))
LIB = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/bddl_files/libero_spatial"
SUITE, MAX_STEPS = "n3_spatial", 300
N_SMOKE, N_MAIN, BASE_FAM, BASE_NEW = 4, 20, 600000, 700000
DUMMY = [0.0] * 6 + [-1.0]
USE = [0, 1, 2, 3, 4, 5, 6, 8]
TARGETS = {"stove": ("flat_stove_1_cook_region", "on the stove"),
           "cabinet": ("wooden_cabinet_1_top_side", "on top of the cabinet"),
           "plate": ("plate_1", "on the plate")}
MOVR = {"akita_black_bowl_1": 0.06, "akita_black_bowl_2": 0.06, "cookies_1": 0.06, "glazed_rim_porcelain_ramekin_1": 0.05,
        "plate_1": 0.10}
FIXD = (("wooden_cabinet_1", "body", 0.13), ("flat_stove_1", "body", 0.11), ("flat_stove_1_cook_region", "site", 0.09))
VMIN, VMAX, BOX_PAD, GAP, SETTLE_MAX = 0.10, 0.16, 0.08, 0.02, 0.02


def spatial_tasks():
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()["libero_spatial"]()
    return s, [s.get_task(i) for i in range(s.n_tasks)]


def init_atoms(text):
    body = re.search(r"\(:init(.*?)\)\s*\(:goal", text, re.S).group(1)
    return [tuple(x.split()) for x in re.findall(r"\(([^()]*)\)", body)]


def replace(text, lang, atom):
    t = re.sub(r"\(:language [^)]*\)", f"(:language {lang})", text, count=1)
    t = re.sub(r"\(:goal\s*\(And \([^)]*\)\)\s*\)", f"(:goal\n    (And ({atom}))\n  )", t, count=1)
    assert f"({atom})" in t and f"(:language {lang})" in t, atom
    return t


def make_env(path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=path, camera_heights=128, camera_widths=128)


def pos(env, n):
    return np.array(env.env.sim.data.body_xpos[env.env.obj_body_id[n]])


def ref_xyz(env, key):
    site = TARGETS[key][0]
    if key == "plate":
        return pos(env, "plate_1")
    sim = env.env.sim
    return np.array(sim.data.site_xpos[sim.model.site_name2id(site)])


def fix_xy(env):
    sim, out = env.env.sim, []
    for n, kind, r in FIXD:
        p = sim.data.site_xpos[sim.model.site_name2id(n)] if kind == "site" else sim.data.body_xpos[env.env.obj_body_id[n]]
        out.append((np.array(p[:2]), r))
    return out


def qadr(env, n):
    m = env.env.sim.model
    return int(m.jnt_qposadr[m.body_jntadr[env.env.obj_body_id[n]]])


def main():
    os.makedirs(BD, exist_ok=True)
    os.makedirs(ST, exist_ok=True)
    suite, tasks = spatial_tasks()
    pairs, free_toggle = [], 0
    for ci, i in enumerate(USE):
        t = tasks[i]
        text = open(os.path.join(LIB, t.bddl_file), encoding="utf-8").read()
        atoms = init_atoms(text)
        on_fix = {a[1]: a[2] for a in atoms if a[0] in ("On", "In") and a[1] in MOVR and
                  (a[2].startswith("wooden_cabinet_1") or a[2].startswith("flat_stove_1"))}
        occupied = {"stove": any(v.startswith("flat_stove_1") for v in on_fix.values()),
                    "cabinet": any(v == "wooden_cabinet_1_top_side" for v in on_fix.values())}
        free = [k for k in ("cabinet", "stove") if not occupied[k]]
        assert free, (i, on_fix)
        if len(free) == 2:
            tgt = free[free_toggle % 2]
            free_toggle += 1
        else:
            tgt = free[0]
        lang_k = " ".join(t.name.split("_"))
        assert lang_k.endswith(" and place it on the plate"), lang_k
        lang_n = lang_k[: -len("on the plate")] + TARGETS[tgt][1]
        ref = lang_k[len("pick up the black bowl "): -len(" and place it on the plate")]
        kp, np_ = os.path.join(BD, f"K{ci}.bddl"), os.path.join(BD, f"N{ci}.bddl")
        with open(kp, "w", encoding="utf-8") as fh:
            fh.write(text)                                                  # the training BDDL itself
        with open(np_, "w", encoding="utf-8") as fh:
            fh.write(replace(text, lang_n, f"On akita_black_bowl_1 {TARGETS[tgt][0]}"))
        pairs.append(dict(ci=ci, spatial_index=i, spatial_name=t.name, referent=ref, new_target=tgt, stay=sorted(on_fix),
                          occupied=occupied, known=dict(key=f"K{ci}", language=lang_k, bddl=os.path.relpath(kp, ROOT), target="plate",
                                                        goal=["On", "akita_black_bowl_1", "plate_1"]),
                          new=dict(key=f"N{ci}", language=lang_n, bddl=os.path.relpath(np_, ROOT), target=tgt,
                                   goal=["On", "akita_black_bowl_1", TARGETS[tgt][0]])))
        print(f"[n3] K{ci} '{lang_k}'  ->  N{ci} '{lang_n}' (stay {sorted(on_fix)})", flush=True)

    # training positions of the movers (stored init states of the 8 tasks) -> the layout box
    tr = []
    for p in pairs:
        env = make_env(os.path.join(ROOT, p["known"]["bddl"]))
        env.seed(0)
        for s in suite.get_task_init_states(p["spatial_index"])[:50]:
            env.reset()
            env.set_init_state(s)
            tr += [pos(env, n)[:2] for n in MOVR if n not in p["stay"]]
        env.close()
    tr = np.array(tr)
    lo, hi = tr.min(0) - BOX_PAD, tr.max(0) + BOX_PAD
    print(f"[n3] box {np.round(lo, 3)}-{np.round(hi, 3)}", flush=True)

    n = N_SMOKE + N_MAIN
    meta_pairs, calib = [], {k: [] for k in TARGETS}
    for p in pairs:
        kp = os.path.join(ROOT, p["known"]["bddl"])
        env = make_env(kp)
        fam = []
        for k in range(n):
            np.random.seed(BASE_FAM + k)
            env.seed(BASE_FAM + k)
            env.reset()
            fam.append(env.sim.get_state().flatten())
        fam = np.stack(fam)
        e2 = make_env(os.path.join(ROOT, p["new"]["bddl"]))
        for k in range(2):
            np.random.seed(BASE_FAM + k)
            e2.seed(BASE_FAM + k)
            e2.reset()
            assert np.allclose(e2.sim.get_state().flatten(), fam[k]), f"{p['new']['key']}: state {k} differs from its known partner's"
        e2.close()
        movers = [m for m in MOVR if m not in p["stay"]]
        new, vs, settle = [], [], []
        for k in range(n):
            rng = np.random.default_rng(BASE_NEW + k + 1000 * p["ci"])
            env.reset()
            env.set_init_state(fam[k])
            p0 = np.array([pos(env, m) for m in movers])
            fx = fix_xy(env)
            for _ in range(10):
                env.step(DUMMY)
            dz_fam = np.array([pos(env, m) for m in movers])[:, 2] - p0[:, 2]
            done, last = False, None
            for _ in range(4000):
                ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(VMIN, VMAX)
                v = mag * np.array([np.cos(ang), np.sin(ang)])
                q = p0[:, :2] + v
                if np.any(q < lo) or np.any(q > hi):
                    continue
                if any(np.linalg.norm(q[j] - c) < MOVR[m] + r + GAP for j, m in enumerate(movers) for c, r in fx):
                    continue
                env.reset()
                env.set_init_state(fam[k])
                sim = env.env.sim
                for m in movers:
                    a = qadr(env, m)
                    sim.data.qpos[a:a + 2] += v
                sim.forward()
                s_new = sim.get_state().flatten()
                env.reset()
                env.set_init_state(s_new)
                q0 = np.array([pos(env, m) for m in movers])
                for _ in range(10):
                    env.step(DUMMY)
                q1 = np.array([pos(env, m) for m in movers])
                d = float(max(np.linalg.norm(q1[:, :2] - q0[:, :2], axis=-1).max(), np.abs((q1[:, 2] - q0[:, 2]) - dz_fam).max()))
                last = d
                if d <= SETTLE_MAX and np.allclose(q0[:, :2], q, atol=1e-4):
                    new.append(s_new)
                    vs.append(v.tolist())
                    settle.append(d)
                    done = True
                    break
            assert done, f"{p['known']['key']} state {k}: no valid translation (last settle {last})"
        new = np.stack(new)

        # feasibility + rest heights: teleport bowl_1 above each free target, settle, read the strict check of that target's BDDL
        feas = {}
        for tgt in ("plate", p["new_target"]):
            path = os.path.join(ROOT, p["known" if tgt == "plate" else "new"]["bddl"])
            ev = make_env(path)
            ev.seed(0)
            oks, dzs = [], []
            for k in range(N_SMOKE):
                ev.reset()
                ev.set_init_state(fam[k])
                sim = ev.env.sim
                r = ref_xyz(ev, tgt)
                a = qadr(ev, "akita_black_bowl_1")
                sim.data.qpos[a:a + 3] = [r[0], r[1], r[2] + 0.12]          # orientation kept (upright as sampled)
                dadr = int(sim.model.jnt_dofadr[sim.model.body_jntadr[ev.env.obj_body_id["akita_black_bowl_1"]]])
                sim.data.qvel[dadr:dadr + 6] = 0
                sim.forward()
                for _ in range(60):
                    ev.step(DUMMY)
                oks.append(bool(ev.env._check_success()))
                dzs.append(float(pos(ev, "akita_black_bowl_1")[2] - ref_xyz(ev, tgt)[2]))
            ev.close()
            feas[tgt] = dict(strict_ok=oks, dz_rest=dzs)
            calib[tgt] += dzs
            print(f"[n3] {p['known']['key']} feasibility {tgt}: strict {oks} dz {np.round(dzs, 3).tolist()}", flush=True)
        env.close()
        for li, S in enumerate((fam, new)):
            np.save(os.path.join(ST, f"{SUITE}_{2 * p['ci'] + li}.npy"), S)        # known: tids 2ci, 2ci+1
            np.save(os.path.join(ST, f"{SUITE}_{2 * (p['ci'] + len(pairs)) + li}.npy"), S)   # new partner: same states
        meta_pairs.append(dict(p, translation=vs, settle_max_observed=max(settle), feasibility=feas))
        print(f"[n3] {p['known']['key']}: {n} new layouts, |v| mean {np.mean(np.linalg.norm(vs, axis=1)):.3f}, "
              f"max settle {max(settle):.4f}", flush=True)

    tasks_out = []
    for status in ("known", "new"):
        for p in meta_pairs:
            ci = p["ci"] + (0 if status == "known" else len(pairs))
            d = p[status]
            for li, lay in enumerate(("familiar", "new")):
                tasks_out.append(dict(tid=2 * ci + li, key=f"{d['key']}_{lay[:3]}", compound=d["key"], pair=p["ci"], status=status,
                                      layout=lay, language=d["language"], bddl=d["bddl"], goal=d["goal"], object="akita_black_bowl_1",
                                      target=d["target"], habit_target="plate", referent=p["referent"],
                                      spatial_index=p["spatial_index"]))
    br = {ph: [dict(id=f"k{k}", trial=k, seed_base=13000000 + 1000 * k, at=0, variant="n3") for k in ks]
          for ph, ks in (("smoke", range(N_SMOKE)), ("main", range(N_SMOKE, n)))}
    for ph, v in br.items():
        with open(os.path.join(OUT, f"n3_branches_{ph}.json"), "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1)
    meta = dict(suite=SUITE, max_steps=MAX_STEPS, tasks=sorted(tasks_out, key=lambda x: x["tid"]), pairs=meta_pairs,
                states=dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, n)), base_familiar=BASE_FAM, base_new=BASE_NEW),
                layout=dict(vmin=VMIN, vmax=VMAX, box=[lo.tolist(), hi.tolist()], gap=GAP, settle_max=SETTLE_MAX,
                            mover_radius=MOVR, fixture_discs=[list(x) for x in FIXD]),
                dz_rest={k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v)) for k, v in calib.items() if v},
                targets={k: v[0] for k, v in TARGETS.items()})
    with open(os.path.join(OUT, "n3_tasks.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("[n3] dz_rest", json.dumps(meta["dz_rest"]), flush=True)
    print("[n3] done", flush=True)


if __name__ == "__main__":
    main()

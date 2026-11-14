"""SCENE 2 (docs/SCAFFOLD_RESULTS.md 50): the F1 atlas and the test-time role views in a SECOND trained scene.  DATA ONLY: no policy.

The scene.  LIBERO-10's LIVING_ROOM_SCENE5 (living-room table: white porcelain mug, yellow-and-white mug, red coffee mug, a left and a
right plate).  It is the only scene among the 40 tasks pi0.5_libero, ar0s/groot_libero and pi0_libero were fine-tuned on (LIBERO-
Spatial, -Object, -Goal, -10) with two movable objects that each have a trained target and two targets: its one training task
"put the white mug on the left plate and put the yellow and white mug on the right plate" binds white -> left and yellow-white ->
right.  Spatial and Object have one target (plate / basket); the other LIBERO-10 scenes have one pick-place pairing or one target.
In other scenes the white mug goes onto a single plate (LIBERO-10 LIVING_ROOM_SCENE6) and the yellow-and-white mug into a microwave
(KITCHEN_SCENE6), so the type-level pairing mug -> plate is trained; what is new here is which plate.  LIBERO-90's single-step tasks
of this scene are not in the 40.  The compound's own raw LIBERO-10 demos (recorded simulator states) fit the reference planner's
keypoints (mode keypoints); they are never executed and no VLA sees anything from this step.

Tasks (the template is the LIBERO-10 BDDL with only language, obj_of_interest and goal replaced; every task shares the states):
    tid 0 K1  put the white mug on the left plate                     On porcelain_mug_1 plate_1        trained half
    tid 1 K2  put the yellow and white mug on the right plate          On white_yellow_mug_1 plate_2     trained half
    tid 2 N1  put the white mug on the right plate                    On porcelain_mug_1 plate_2        new; partners K1 (object), K2 (target)
    tid 3 N2  put the yellow and white mug on the left plate           On white_yellow_mug_1 plate_1     new; partners K2 (object), K1 (target)
    tid 4 KC  the training instruction, verbatim (both atoms)                                          trained compound
    tid 5 NC  put the white mug on the right plate and put the yellow and white mug on the left plate   swapped compound (new)
Single-step episodes run 400 policy steps (LIBERO's limit for LIBERO-90's single-step tasks of this scene), compounds 520 (LIBERO-10's).

States.  LIBERO's sampler on K1's BDDL, np.random.seed = env.seed = 830000 + k, k = 0..63: 0-3 smoke, 4-63 main.  Branches: trial = k,
seed_base 23000000 + 1000 k (never an N2 seed base).  The state rows are identical for every task's BDDL (checked on k = 0, 1).

Modes (LIBERO venv, py3.8, EGL):
    build      -> results/flawed/s2/{bddl/*.bddl, states/s2_living_<tid>.npy, s2_tasks.json, branches_smoke.json, branches_main.json}
    keypoints  -> results/flawed/s2/keypoints.json   (from the LIBERO-10 compound's raw demos' recorded states; fa1g_plan.stats' rule)
    check      -> .claude_tmp/pp/s2_check/{sheet.png, segcheck.json}: start renders of smoke states 0-3 and the learned segmenter
                  (sc_seg) against the simulator's entity map on them (does it transfer?)
Refuses to overwrite.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
SD = os.path.join(OUT, "s2")
BD, ST = os.path.join(SD, "bddl"), os.path.join(SD, "states")
CHK = "/coc/flash10/ksingara3/.claude_tmp/pp/s2_check"
LIB = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/bddl_files"
TEMPLATE = os.path.join(LIB, "libero_10", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_"
                                          "the_right_plate.bddl")
RAW10 = "/coc/flash10/ksingara3/robot/data/libero_raw/libero_10"         # the compound's own raw demos (recorded states only)
DUMP = "/coc/flash10/ksingara3/robot/data/fa1/s2_raw10.npz"
SUITE = "s2_living"
N_SMOKE, N_MAIN, BASE, BR_BASE = 4, 60, 830000, 23000000
DUMMY = [0.0] * 6 + [-1.0]
W, Y, R, PL, PR = "porcelain_mug_1", "white_yellow_mug_1", "red_coffee_mug_1", "plate_1", "plate_2"
MOVABLE = (W, Y, R, PL, PR)
PHRASE = {W: "the white mug", Y: "the yellow and white mug", R: "the red mug", PL: "the left plate", PR: "the right plate"}
# key, status, language, goal atoms [(object, target)], max steps, partners (object partner, target partner) for new pairings
TASKS = [
    ("K1", "known", "put the white mug on the left plate", [(W, PL)], 400, None),
    ("K2", "known", "put the yellow and white mug on the right plate", [(Y, PR)], 400, None),
    ("N1", "new", "put the white mug on the right plate", [(W, PR)], 400, ("K1", "K2")),
    ("N2", "new", "put the yellow and white mug on the left plate", [(Y, PL)], 400, ("K2", "K1")),
    ("KC", "known_compound", "put the white mug on the left plate and put the yellow and white mug on the right plate", [(W, PL), (Y, PR)], 520, None),
    ("NC", "new_compound", "put the white mug on the right plate and put the yellow and white mug on the left plate", [(W, PR), (Y, PL)], 520, None),
]


def bddl_of(text, lang, atoms):
    interest = []
    for o, t in atoms:
        interest += [x for x in (o, t) if x not in interest]
    t = re.sub(r"\(:language [^)]*\)", "(:language %s)" % lang, text, count=1)
    t = re.sub(r"\(:obj_of_interest[^)]*\)", "(:obj_of_interest\n    " + "\n    ".join(interest) + "\n  )", t, count=1)
    head = t[:t.index("(:goal")]
    goal = " ".join("(On %s %s)" % a for a in atoms)
    t = head + "(:goal\n    (And %s)\n  )\n\n)\n" % goal
    assert "(:language %s)" % lang in t
    return t


def make_env(path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=path, camera_heights=256, camera_widths=256)


def positions(env, names):
    sim, inner = env.env.sim, env.env
    return np.array([np.array(sim.data.body_xpos[inner.obj_body_id[n]]) for n in names])


def training_check():
    """The 40 training tasks: no task other than LIBERO-10's compound names a mug; the new atoms occur in none."""
    from libero.libero import benchmark
    seen, langs = [], []
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = benchmark.get_benchmark_dict()[s]()
        for i in range(suite.n_tasks):
            tk = suite.get_task(i)
            langs.append(str(tk.language))
            txt = open(os.path.join(LIB, tk.problem_folder, tk.bddl_file), encoding="utf-8").read()
            if "mug" in txt:
                seen.append(dict(suite=s, index=i, name=tk.name, goal=txt[txt.index("(:goal"):].split("\n")[1].strip(),
                                 scene5=tk.name.startswith("LIVING_ROOM_SCENE5")))
    assert len(langs) == 40, len(langs)
    s5 = [x for x in seen if x["scene5"]]
    assert len(s5) == 1, s5                                                 # the compound is the scene's only training task
    for a in ("(On porcelain_mug_1 plate_2)", "(On white_yellow_mug_1 plate_1)"):
        assert a not in s5[0]["goal"], (a, s5)
    for _, st, lang, _, _, _ in TASKS:
        if st.startswith("new"):
            assert lang not in langs, lang
    return seen


def build():
    for p in (SD,):
        assert not os.path.exists(os.path.join(p, "s2_tasks.json")), "exists"
    os.makedirs(BD, exist_ok=True)
    os.makedirs(ST, exist_ok=True)
    seen = training_check()
    print("[s2] training tasks naming a mug: %s" % json.dumps(seen), flush=True)
    text = open(TEMPLATE, encoding="utf-8").read()
    paths = {}
    for key, _, lang, atoms, _, _ in TASKS:
        paths[key] = os.path.join(BD, key + ".bddl")
        assert not os.path.exists(paths[key])
        with open(paths[key], "w", encoding="utf-8") as fh:
            fh.write(text if key == "KC" else bddl_of(text, lang, atoms))
    n = N_SMOKE + N_MAIN
    env = make_env(paths["K1"])

    def draw(e, k):
        np.random.seed(BASE + k)
        e.seed(BASE + k)
        e.reset()
        return e.sim.get_state().flatten()

    S = np.stack([draw(env, k) for k in range(n)])
    for key in paths:
        if key == "K1":
            continue
        e2 = make_env(paths[key])
        for k in range(2):
            assert np.allclose(draw(e2, k), S[k]), "%s: state %d differs from K1's" % (key, k)
        e2.close()
    for i in range(n):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
    pos, settle = [], []
    for s in S:
        env.reset()
        env.set_init_state(s)
        p0 = positions(env, MOVABLE)
        for _ in range(10):
            env.step(DUMMY)
        p1 = positions(env, MOVABLE)
        pos.append(p0.tolist())
        settle.append(float(np.linalg.norm((p1 - p0)[:, :2], axis=-1).max()))
    env.close()
    pos = np.array(pos)
    print("[s2] %d states; xy spread per object (max dist from mean, m) %s; max settle xy %.4f m" % (
        n, np.round(np.linalg.norm(pos[:, :, :2] - pos[:, :, :2].mean(0), axis=-1).max(0), 3).tolist(), max(settle)), flush=True)
    tasks = []
    for tid, (key, status, lang, atoms, ms, partner) in enumerate(TASKS):
        np.save(os.path.join(ST, "%s_%d.npy" % (SUITE, tid)), S)
        tasks.append(dict(tid=tid, key=key, compound=key, status=status, layout="familiar", language=lang,
                          bddl=os.path.relpath(paths[key], ROOT), atoms=[list(a) for a in atoms], object=atoms[0][0], target=atoms[0][1],
                          max_steps=ms, partner=list(partner) if partner else None,
                          phrases=["object: " + PHRASE[atoms[0][0]], "target: " + PHRASE[atoms[0][1]]] if len(atoms) == 1 else None))
    br = {ph: [dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks]
          for ph, ks in (("smoke", range(N_SMOKE)), ("main", range(N_SMOKE, n)))}
    for ph, v in br.items():
        with open(os.path.join(SD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1)
    meta = dict(suite=SUITE, max_steps=400, template=TEMPLATE, tasks=tasks, movable=list(MOVABLE), phrase=PHRASE,
                states=dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, n)), base=BASE, rule="np.random.seed = env.seed = %d + k on K1.bddl" % BASE),
                branch_seed_base="%d + 1000 k" % BR_BASE, training_mug_tasks=seen, positions=pos.tolist(), settle_xy_max=max(settle))
    with open(os.path.join(SD, "s2_tasks.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("[s2] wrote %s" % SD, flush=True)


def quat_mat(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_mean(Rs):
    M = np.mean(Rs, 0)
    return Rs[int(np.argmin([np.linalg.norm(R - M) for R in Rs]))]


def keypoints():
    """fa1g_plan.stats' rule on the LIBERO-10 compound's raw demos, one close -> open cycle per mug (the mug nearest the hand at the
    close names the half; recorded simulator states only, never executed): grasp offset = end effector - object at the close command; grasp rotation; carry height = highest end effector between close
    and the first open after it, minus the target's height; release offset = end effector - target at that open; release rotation.
    Medians over demos; rotations = the demo rotation nearest the mean."""
    fn = os.path.join(SD, "keypoints.json")
    assert not os.path.exists(fn), "exists"
    ROLE = {W: ("K1", PL), Y: ("K2", PR)}
    f = os.path.join(RAW10, os.path.basename(TEMPLATE).replace(".bddl", "_demo.hdf5"))
    z = np.load(DUMP)                                                       # rawdump's copy of f (the LIBERO venv has no h5py)
    assert str(z["source"]) == f
    env = make_env(TEMPLATE)
    env.seed(0)
    env.reset()
    acc = {k: dict(G=[], RG=[], CZ=[], RO=[], RR=[]) for k in ("K1", "K2")}
    if True:
        for di in range(int(z["n"])):
            A, S = z["a%d" % di], z["s%d" % di]

            def at(i):
                env.env.sim.set_state_from_flattened(S[i])
                env.env.sim.forward()
                ob = env.env._get_observations(force_update=True)
                return np.asarray(ob["robot0_eef_pos"], np.float64), quat_mat(np.asarray(ob["robot0_eef_quat"], np.float64)), \
                    positions(env, [W, Y, PL, PR])
            # every close -> open cycle; the mug nearest the hand at the close names the half; the first cycle per mug counts
            i, done = 0, set()
            while i < len(A):
                if A[i, 6] <= 0:
                    i += 1
                    continue
                ic = i
                io = ic + int(np.argmax(A[ic:, 6] < 0)) if (A[ic:, 6] < 0).any() else len(A) - 1
                e1, r1, p1 = at(ic)
                mug = (W, Y)[int(np.argmin([np.linalg.norm(e1 - p1[0]), np.linalg.norm(e1 - p1[1])]))]
                k, te = ROLE[mug]
                if mug not in done:
                    done.add(mug)
                    e2, r2, p2 = at(io)
                    ti = 2 if te == PL else 3
                    zs = [at(j)[0][2] for j in range(ic, io + 1, 4)]
                    a_ = acc[k]
                    a_["G"].append(e1 - p1[0 if mug == W else 1])
                    a_["RG"].append(r1)
                    a_["CZ"].append(float(max(zs) - p1[ti][2]))
                    a_["RO"].append(e2 - p2[ti])
                    a_["RR"].append(r2)
                i = io + 1
    env.close()
    out = {}
    for k, a_ in acc.items():
        oe, te = (W, PL) if k == "K1" else (Y, PR)
        G, RO = np.array(a_["G"]), np.array(a_["RO"])
        out[k] = dict(object=oe, target=te, n=len(G), grasp_off=np.median(G, 0).tolist(), grasp_off_mad=np.median(np.abs(G - np.median(G, 0)), 0).tolist(),
                      grasp_rot=rot_mean(np.array(a_["RG"])).tolist(), carry_z_above_tgt=float(np.median(a_["CZ"])),
                      release_off=np.median(RO, 0).tolist(), release_off_mad=np.median(np.abs(RO - np.median(RO, 0)), 0).tolist(),
                      release_rot=rot_mean(np.array(a_["RR"])).tolist(), source=f)
        print("[s2kp] %s: %s" % (k, json.dumps({x: (np.round(v, 3).tolist() if isinstance(v, list) else v) for x, v in out[k].items() if "rot" not in x})), flush=True)
    assert all(out[k]["n"] >= 20 for k in out), {k: out[k]["n"] for k in out}
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=out, note="LIBERO-10 compound raw demos, one close-open cycle per mug; recorded states only"), fh, indent=1)


def rawdump():
    """openpi venv (h5py): the compound's raw demos -> DUMP (actions and recorded simulator states per demo; nothing else)."""
    import h5py
    f = os.path.join(RAW10, os.path.basename(TEMPLATE).replace(".bddl", "_demo.hdf5"))
    assert not os.path.exists(DUMP), "exists"
    arr = {}
    with h5py.File(f, "r") as h:
        demos = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for i, dm in enumerate(demos):
            arr["a%d" % i] = np.asarray(h["data"][dm]["actions"])
            arr["s%d" % i] = np.asarray(h["data"][dm]["states"])
    os.makedirs(os.path.dirname(DUMP), exist_ok=True)
    np.savez_compressed(DUMP, n=np.array(len(demos)), source=np.array(f), **arr)
    print("[s2raw] %d demos from %s -> %s" % (len(demos), f, DUMP), flush=True)


def check():
    """Start renders of smoke states 0-3 (both cameras, after the 10 settle steps) -> sheet.png, and the learned segmenter's classes on
    each simulator entity's pixels -> segcheck.json.  The segmenter's class table has no mug class and one plate class."""
    import imageio
    import sc_rv as RV
    import sc_seg as SG
    os.makedirs(CHK, exist_ok=True)
    meta = json.load(open(os.path.join(SD, "s2_tasks.json"), encoding="utf-8"))
    S = np.load(os.path.join(ST, "%s_0.npy" % SUITE))
    seg = SG.load(None, "cuda")
    env = make_env(os.path.join(ROOT, meta["tasks"][0]["bddl"]))
    env.seed(7)
    rows, hist = [], {}
    for k in range(N_SMOKE):
        env.reset()
        obs = env.set_init_state(S[k])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
        E = RV.entity_tables(env)
        row = []
        for cam, key in zip(RV.CAMS, ("agentview_image", "robot0_eye_in_hand_image")):
            img = np.ascontiguousarray(np.asarray(obs[key])[::-1, ::-1])
            g = np.asarray(RV.seg_px(env, E, cam))
            c = seg.classes_of(img, [SG.cam_id(cam)])[0]
            for i, nm in enumerate(E["names"]):
                m = g == i
                if not m.any():
                    continue
                hh = hist.setdefault(cam, {}).setdefault(nm, {})
                for cl, cnt in zip(*np.unique(c[m], return_counts=True)):
                    hh[seg.classes[int(cl)]] = hh.get(seg.classes[int(cl)], 0) + int(cnt)
            vis = np.array(img, copy=True)
            for i, nm in enumerate(E["names"]):
                if nm in (W, Y, R):
                    vis[g == i] = (vis[g == i] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            row += [img, vis]
        rows.append(np.concatenate(row, 1))
    env.close()
    imageio.imwrite(os.path.join(CHK, "sheet.png"), np.concatenate(rows, 0))
    share = {cam: {nm: {cl: round(v / float(sum(h.values())), 3) for cl, v in sorted(h.items(), key=lambda x: -x[1])[:4]}
                   for nm, h in hc.items()} for cam, hc in hist.items()}
    res = dict(classes=seg.classes, ckpt=seg.path, names=E["names"], share=share,
               transfers=False, why="no class for any mug; one class for both plates (no instance), so no role view can be painted from it")
    with open(os.path.join(CHK, "segcheck.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[s2check] classes %s" % seg.classes, flush=True)
    for cam, hc in share.items():
        for nm, h in hc.items():
            print("[s2check] %s %s -> %s" % (cam, nm, h), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "rawdump", "keypoints", "check"))
    a = ap.parse_args()
    {"build": build, "rawdump": rawdump, "keypoints": keypoints, "check": check}[a.mode]()

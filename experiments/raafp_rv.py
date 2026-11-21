"""RAA-F Route A+ data build (docs/notes/raaf_plan.md 3, docs/notes/raafp_diag.md): the planner's table-spot demos (experiments/raafp_gen.py, rule h_j)
rendered like sc_rv.render and converted twice, each time on top of the K1-K8 base episodes re-converted from their rendered demos:

    raacp   lol/libero_g8v_raacp   RAA-c frames (rview2 painted views, 4-d state token) + planner demos; held prompt "target: the table".  A+ trains on it.
    rview2p lol/libero_g8v_rview2p the same frames and prompts in the rview2 format (8-d proprio state, no token).                           A+ctrl trains on it.

    render   (LIBERO venv, GPU/EGL)  python experiments/raafp_rv.py render  --gen DIR [--keep 50]
             per generated demo: replay check (k3), then sc_rv.render's arrays -> RVP/<K>/demo_XX.npz and chan_XX.npz
    convert  (openpi venv, CPU)      python experiments/raafp_rv.py convert --arm raac|rview2 [--ndemo N] [--suffix S]
    check    (openpi venv, CPU)      python experiments/raafp_rv.py check   --arm raac|rview2 [--suffix S]

Planner demos: bowl -> K1's language / roles, cream cheese -> K6's; the token's TARGET role = the spot at the object's resting height minus the hand minus the in-hand
offset (sc_rv.held_target_rel), the object role = object - hand; the views follow role_paint with the target = main_table (a fixture: nothing is un-painted
while held, all four movables are painted).  sc_rv.py is imported, not edited.
"""
import argparse
import glob
import json
import os
import pathlib
import shutil
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sc_rv as RV                                                           # noqa: E402

RVP = "/coc/flash10/ksingara3/robot/data/fa1/rvp"
OBJS = [("bowl", "K1", "akita_black_bowl_1"), ("cream_cheese", "K6", "cream_cheese_1")]
REPO = {"raac": "lol/libero_g8v_raacp", "rview2": "lol/libero_g8v_rview2p"}
BASE_REPO = {"raac": "lol/libero_g8v_raac", "rview2": "lol/libero_g8v_rview2"}
TABLE_PHRASE = "target: the table"
OUTD = os.path.join(RV.OUT, "raaf")


def gen_files(gen, name, keep=None):
    fs = sorted(glob.glob(os.path.join(gen, name, "demo_*.npz")), key=lambda f: int(os.path.basename(f)[5:-4]))
    return fs[:keep] if keep else fs


def render(a):
    from libero.libero.envs import OffScreenRenderEnv
    import fa1_common as FC
    L = RV.lang()
    rep = {}
    for name, c, obj in OBJS:
        files = gen_files(a.gen, name, a.keep)
        assert len(files) == (a.keep or len(files)) and files, (name, len(files))
        bddl = str(np.load(files[0])["bddl"])
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RV.SIZE, camera_widths=RV.SIZE)
        env.seed(0)
        env.reset()
        E = RV.entity_tables(env)
        dj = FC.drawer_joint(env)
        nstate = env.get_sim_state().shape[0]
        rows = []
        for j, f in enumerate(files):
            out = RV.demo_path(RVP, c, j)
            if os.path.exists(out):
                continue
            z = np.load(f)
            st, ac = z["states_0"], z["actions_0"]
            T = len(ac)
            assert st.shape[0] == T + 1 and st.shape[1] == nstate, (st.shape, T, nstate)
            env.env.timestep, env.env.done = 0, False                 # robosuite counts steps across demos (horizon 1000) and refuses to step after done
            # k3 (1): open-loop replay of the stored actions from the stored first state vs the stored states' end effector
            obs = env.set_init_state(st[0])
            ee_rep = [np.asarray(obs["robot0_eef_pos"], np.float64)]
            for t in range(T):
                obs, _, _, _ = env.step(ac[t].tolist())
                ee_rep.append(np.asarray(obs["robot0_eef_pos"], np.float64))
            keep = RV.noop_keep(ac)
            R = {k: [] for k in ("image", "wrist", "seg", "wseg", "state")}
            POS, EE, ee_sto, ee_one = [], [], [], []
            kept_idx = set(np.flatnonzero(keep).tolist())
            for t in range(T + 1):
                obs = env.set_init_state(st[t])
                sc = FC.scene(env, obs, dj)
                ee_sto.append(sc["ee_pos"].copy())
                if t < T and t in kept_idx:
                    R["image"].append(RV.flip(obs["agentview_image"]))
                    R["wrist"].append(RV.flip(obs["robot0_eye_in_hand_image"]))
                    R["seg"].append(RV.seg_px(env, E, RV.CAMS[0]))
                    R["wseg"].append(RV.seg_px(env, E, RV.CAMS[1]))
                    R["state"].append(np.concatenate((obs["robot0_eef_pos"], RV.quat2axisangle(obs["robot0_eef_quat"]),
                                                      obs["robot0_gripper_qpos"])).astype(np.float32))
                    POS.append(sc["pos"].astype(np.float32))
                    EE.append(sc["ee_pos"].astype(np.float32))
                if t < T:                                                 # k3 (2): one step from the stored state vs the next stored state
                    o2, _, _, _ = env.step(ac[t].tolist())
                    ee_one.append(np.asarray(o2["robot0_eef_pos"], np.float64))
            ee_sto = np.asarray(ee_sto)
            d_open = float(np.abs(np.asarray(ee_rep) - ee_sto).max())
            d_one = float(np.abs(np.asarray(ee_one) - ee_sto[1:]).max())
            d_ee = float(np.abs(np.asarray(EE, np.float64) - np.asarray(R["state"])[:, :3]).max())
            RV.atomic_savez(out, image=np.stack(R["image"]).astype(np.uint8), wrist=np.stack(R["wrist"]).astype(np.uint8), seg=np.stack(R["seg"]), wseg=np.stack(R["wseg"]),
                            state=np.stack(R["state"]), actions=np.asarray(ac[keep], np.float32), kept=np.flatnonzero(keep), raw_len=T, demo=j, task=c, language=L[c],
                            names=np.array(E["names"]), slot_names=np.array(list(E["slots"])), slot_ents=np.array(list(E["slots"].values())),
                            spot=np.asarray(z["spot"], np.float64), seed=int(z["seed"]), src=os.path.basename(f))
            RV.atomic_savez(os.path.join(RVP, c, "chan_%02d.npz" % j), pos=np.stack(POS), ee=np.stack(EE), kept=np.flatnonzero(keep), ee_maxdiff=np.float32(d_ee),
                            ent=np.array(FC.ENT))
            rows.append(dict(demo=j, src=os.path.basename(f), T=T, kept=int(keep.sum()), open_loop_ee_maxdiff_mm=1e3 * d_open, one_step_ee_maxdiff_mm=1e3 * d_one,
                             state_vs_chan_ee_maxdiff_mm=1e3 * d_ee))
            print("[raafp_rv] %s demo %d (%s): T %d kept %d; open-loop ee diff %.2f mm, one-step %.2f mm, state-vs-scene %.3f mm" % (
                name, j, os.path.basename(f), T, int(keep.sum()), 1e3 * d_open, 1e3 * d_one, 1e3 * d_ee), flush=True)
        env.close()
        rep[name] = rows
    p = os.path.join(OUTD, "planner", "replay_check_%s.json" % os.path.basename(os.path.normpath(a.gen)))
    assert not os.path.exists(p), p
    allr = [r for v in rep.values() for r in v]
    summ = {k: dict(max=max(r[k] for r in allr), median=float(np.median([r[k] for r in allr])), n_over_1mm=int(sum(r[k] > 1.0 for r in allr)))
            for k in ("open_loop_ee_maxdiff_mm", "one_step_ee_maxdiff_mm", "state_vs_chan_ee_maxdiff_mm")} if allr else {}
    json.dump(dict(gen=a.gen, summary=summ, demos=rep), open(p, "w", encoding="utf-8"), indent=1)
    print("[raafp_rv] replay check %s" % json.dumps(summ), flush=True)


def planner_rel(z, ch, obj, T16, X, info):
    import fa1_common as FC
    pos, ee = np.asarray(ch["pos"], np.float64), np.asarray(ch["ee"], np.float64)
    assert len(pos) == len(z["actions"]) and list(ch["ent"]) == list(FC.ENT)
    H = np.asarray(T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]), np.float64)
    gi = FC.ENT.index(obj)
    sp = np.asarray(z["spot"], np.float64)
    tgt = np.tile(np.array([sp[0], sp[1], pos[0, gi, 2]]), (len(pos), 1))          # the spot at the object's resting height
    out = np.stack([pos[:, gi] - ee, RV.held_target_rel(pos[:, gi], tgt, ee, H > 0.5, info)], axis=1)
    return out, H


def convert(a):
    os.environ["HF_LEROBOT_HOME"] = RV.LEROBOT_HOME
    RV._openpi()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    chan = a.arm == "raac"
    ns = types.SimpleNamespace(suffix="", root=RV.RV_ROOT, arm=a.arm, ndemo=a.ndemo)
    T16 = RV.t16()
    X = RV.lex(ns)
    by_c, _ = RV.roles_of()
    repo = REPO[a.arm] + a.suffix
    root = pathlib.Path(RV.LEROBOT_HOME) / repo
    man_path = os.path.join(OUTD, "raafp_rv_manifest_%s%s.json" % (a.arm, a.suffix))
    assert not (root.exists() and os.path.exists(man_path)), "%s exists and is complete" % root
    if root.exists():
        print("[raafp_rv] removing incomplete %s" % root, flush=True)
        shutil.rmtree(root)
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (RV.CHAN_DIM if chan else 8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=repo, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8, image_writer_processes=4)
    info = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
    pinfo = dict(offset_cm=[])
    n_base, n_plan, fr_base, fr_plan, per_task, prompts, tok_last = 0, 0, 0, 0, {}, {}, []
    for c in RV.KNOWN:                                       # the base episodes: exactly sc_rv.convert's rows
        for f in RV.demo_files(RV.RV_ROOT, c)[:(a.ndemo or None)]:
            z = dict(np.load(f))
            routed = T16.given(str(z["language"])) is not None
            rel = (RV.demo_rel(z, c, f, ns, X, T16, by_c, info) if routed else np.zeros((len(z["actions"]), 2, 3))) if chan else None
            for i, (img, wimg, task, h, na, nw) in enumerate(RV.frames_of(z, c, a.arm, X, T16, by_c, rel)):
                ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "state": (RV.chan_state(rel[i, int(h)], h, routed) if chan else np.asarray(z["state"][i], np.float32)),
                              "actions": np.asarray(z["actions"][i], np.float32), "task": task})
                fr_base += 1
            ds.save_episode()
            n_base += 1
    for name, c, obj in OBJS:                                # the planner's table-spot episodes
        for f in RV.demo_files(RVP, c)[:(a.ndemo or None)]:
            z = dict(np.load(f))
            ch = np.load(os.path.join(RVP, c, "chan_%s" % os.path.basename(f)[5:]))
            rel, H = planner_rel(z, ch, obj, T16, X, pinfo)
            roles = {c: (obj, "main_table")}
            assert T16.given(str(z["language"])) is not None
            last_h = None
            for i, (img, wimg, task, h, na, nw) in enumerate(RV.frames_of(z, c, a.arm, X, T16, roles, rel if chan else None)):
                if h:
                    task = TABLE_PHRASE
                    last_h = i
                prompts[task] = prompts.get(task, 0) + 1
                ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "state": (RV.chan_state(rel[i, int(h)], h, True) if chan else np.asarray(z["state"][i], np.float32)),
                              "actions": np.asarray(z["actions"][i], np.float32), "task": task})
                fr_plan += 1
            if chan and last_h is not None:
                tok_last.append(np.abs(rel[last_h, 1]).max())
            ds.save_episode()
            n_plan += 1
            per_task[name] = per_task.get(name, 0) + 1
    ds.stop_image_writer()
    man = dict(repo_id=repo, arm=a.arm, episodes_base=n_base, episodes_planner=n_plan, episodes=n_base + n_plan, frames_base=fr_base, frames_planner=fr_plan,
               frames=int(ds.meta.total_frames), planner_per_object=per_task, planner_prompts=prompts, base_channel=info if chan else None,
               planner_offset_cm_median=(np.median(np.asarray(pinfo["offset_cm"]), 0).tolist() if chan and pinfo["offset_cm"] else None),
               token_last_held_row_max_abs_m_median=(float(np.median(tok_last)) if tok_last else None), ndemo=a.ndemo, tw=X["tw"], lift=X["lift"], gen_root=RVP)
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("[raafp_rv] done %s: %s" % (repo, json.dumps(man)), flush=True)


def check(a):
    RV._openpi()
    import pyarrow.parquet as pq
    from PIL import Image
    import io
    repo, base = REPO[a.arm] + a.suffix, BASE_REPO[a.arm]
    R = pathlib.Path(RV.LEROBOT_HOME)
    fs = sorted((R / repo / "data").rglob("episode_*.parquet"))
    bs = sorted((R / base / "data").rglob("episode_*.parquet"))
    nb = len(bs) if not a.suffix else None
    man = json.load(open(os.path.join(OUTD, "raafp_rv_manifest_%s%s.json" % (a.arm, a.suffix)), encoding="utf-8"))
    nbase = man["episodes_base"]
    print("[raafp_rv] check %s: %d episodes (manifest %d = %d base + %d planner); base repo %s has %d" % (repo, len(fs), man["episodes"], nbase, man["episodes_planner"], base, len(bs)), flush=True)
    assert len(fs) == man["episodes"]
    tasks = {}
    for l in open(R / repo / "meta" / "tasks.jsonl", encoding="utf-8"):
        d = json.loads(l)
        tasks[d["task_index"]] = d["task"]
    ok_base, bad = 0, []
    nd = man.get("ndemo", 0) or 50
    bidx = [ci * 50 + j for ci in range(8) for j in range(nd)]
    assert len(bidx) == nbase, (len(bidx), nbase)
    for i, bi in enumerate(bidx):
        t, u = pq.read_table(fs[i], columns=["state", "actions", "task_index"]).to_pydict(), pq.read_table(bs[bi], columns=["state", "actions"]).to_pydict()
        s1, s2 = np.asarray(t["state"], np.float32), np.asarray(u["state"], np.float32)
        a1, a2 = np.asarray(t["actions"], np.float32), np.asarray(u["actions"], np.float32)
        if s1.shape == s2.shape and np.array_equal(s1, s2) and np.array_equal(a1, a2):
            ok_base += 1
        else:
            bad.append(i)
    print("[raafp_rv] base episodes identical to %s (state, actions): %d of %d; mismatches %s" % (base, ok_base, nbase, bad[:10]), flush=True)
    held_prompt, tot, magenta, toks, shifts = 0, 0, [], [], []
    for i in range(nbase, len(fs)):
        t = pq.read_table(fs[i]).to_pydict()
        pr = [tasks[int(k)] for k in t["task_index"]]
        tot += len(pr)
        held_prompt += sum(p == TABLE_PHRASE for p in pr)
        st, ac = np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32)
        assert st.shape[1] == (4 if a.arm == "raac" else 8) and ac.shape[1] == 7 and np.isfinite(st).all() and np.isfinite(ac).all() and np.abs(ac[:, :6]).max() <= 1.0 + 1e-6
        if a.arm == "raac":
            role = st[:, 3]
            assert set(np.unique(role)) <= {1.0, 2.0}, np.unique(role)
            assert (role == 2.0).sum() == sum(p == TABLE_PHRASE for p in pr), "token role 2 must be exactly the held frames"
            toks.append(float(np.abs(st[role == 2.0][-1, :3]).max()) if (role == 2.0).any() else np.nan)
        if (i - nbase) % 10 == 0:
            im = np.asarray(Image.open(io.BytesIO(t["image"][0]["bytes"])))
            magenta.append(float(((im[..., 0] > 250) & (im[..., 1] < 5) & (im[..., 2] > 250)).mean()))
    print("[raafp_rv] planner episodes %d, frames %d, held-prompt frames '%s' %d (%.1f%%); first-frame magenta fraction (sampled) %s" % (
        len(fs) - nbase, tot, TABLE_PHRASE, held_prompt, 100.0 * held_prompt / max(tot, 1), np.round(magenta, 3).tolist()), flush=True)
    if toks:
        print("[raafp_rv] token |dx,dy,dz|max on the last held row: median %.3f m, max %.3f m" % (np.nanmedian(toks), np.nanmax(toks)), flush=True)
    assert ok_base == nbase, "base episodes differ from the base repo"
    assert held_prompt > 0 and all(m > 0 for m in magenta)
    print("[raafp_rv] CHECK PASSED %s" % repo, flush=True)




def ctrl_replay(a):
    """k3 control: the same one-step / open-loop replay numbers on the genuine LIBERO demos (raw states and actions of K1 and K6, sc_rv.extract's files)."""
    from libero.libero.envs import OffScreenRenderEnv
    out = {}
    for c in ("K1", "K6"):
        raw = np.load(RV.raw_path(RV.RV_ROOT, c))
        env = OffScreenRenderEnv(bddl_file_name=RV._bddl(c), camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        rows = []
        for j in [int(x) for x in raw["demos"]][:a.keep]:
            st, ac = raw["states_%d" % j], raw["actions_%d" % j]
            T = len(ac)
            obs = env.set_init_state(st[0])
            ee_rep = [np.asarray(obs["robot0_eef_pos"], np.float64)]
            for t in range(T):
                obs, _, _, _ = env.step(ac[t].tolist())
                ee_rep.append(np.asarray(obs["robot0_eef_pos"], np.float64))
            ee_sto, ee_one = [], []
            for t in range(T):
                obs = env.set_init_state(st[t])
                ee_sto.append(np.asarray(obs["robot0_eef_pos"], np.float64))
                o2, _, _, _ = env.step(ac[t].tolist())
                ee_one.append(np.asarray(o2["robot0_eef_pos"], np.float64))
            ee_sto = np.asarray(ee_sto)
            n = min(len(ee_sto) - 1, T - 1)
            rows.append(dict(demo=j, T=T, open_loop_mm=1e3 * float(np.abs(np.asarray(ee_rep)[:T] - ee_sto).max()), one_step_mm=1e3 * float(np.abs(np.asarray(ee_one)[:n] - ee_sto[1:n + 1]).max())))
        out[c] = rows
        print("[raafp_rv] ctrl_replay %s: %s" % (c, json.dumps(rows)), flush=True)
        env.close()
    p = os.path.join(OUTD, "planner", "replay_check_genuine_demos.json")
    assert not os.path.exists(p)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("render", "convert", "check", "ctrl_replay"))
    ap.add_argument("--gen", default="/coc/flash10/ksingara3/.claude_tmp/raaf/gen_aplus")
    ap.add_argument("--keep", type=int, default=50)
    ap.add_argument("--arm", choices=("raac", "rview2"), default="raac")
    ap.add_argument("--ndemo", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rvp", default=RVP)
    a = ap.parse_args()
    RVP = a.rvp
    dict(render=render, convert=convert, check=check, ctrl_replay=ctrl_replay)[a.what](a)

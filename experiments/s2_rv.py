"""SR 89 step 1 (docs/notes/scene2_plan.md; docs/notes/sr89_build.md): the scene-2 role-view / role-channel training data for RAA-sL-S2
(pi05_g8v_raas_ls2_ae = RAA-sL's recipe, only the data differs).  Adapted from sc_rv.py (extract / render / chan / convert), unchanged.

Source: the 50 raw demos of LIBERO-10's LIVING_ROOM_SCENE5 compound ("put the white mug on the left plate and put the yellow and white mug on
the right plate"), s2_tasks.DUMP.  Each demo holds two close-open cycles (one per mug); a cycle is one single-step training episode:
    white mug -> plate_1 (the left plate)           = a K1 cycle   prompt "object: the white mug"  /  "target: the left plate"
    yellow-and-white mug -> plate_2 (right plate)   = a K2 cycle   prompt "object: the yellow and white mug" / "target: the right plate"
N1 and N2 (the swapped pairings) contribute no frame.  Views: the rview2 rule (sc_rv.role_paint) over this scene's five movables (3 mugs, 2 plates):
every movable is painted magenta (exact simulator mask, both cameras) except the active filler (before the hold the cycle's mug; while held
the cycle's plate).  Channel: the continuous state token [dx, dy, dz (m), role id] of RAA-c / RAA-s (sc_rv.chan_state): role 1 (object) =
mug minus end effector until the hold; role 2 (target) while held = plate minus end effector minus the in-hand offset (the mug's offset to the
end effector at the hold's onset, fixed to the hold's end; sc_rv.held_target_rel).  Positions are simulator body positions (privileged
labels, as the kitchen data's).  Hold = ProprioHeld with the kitchen threshold (results/flawed/sc_rv_lex.json, unchanged).

Cycle split (s2rg.py stats' rule on the kept steps): a mug's cycle starts at the previous cycle's release (0 for the first) and its release is
the first open command (gripper action <= 0) after the mug first rose EFFECT = 3 cm above its start; cycle 1 = [0, release_1], cycle 2 =
[release_1 + 1, end].

Venvs: render in LIBERO's (examples/libero/.venv, py3.8, EGL, GPU); convert / norm in openpi's (.venv).  This file must parse under python 3.8.

    python experiments/s2_rv.py render  [--part i/n] [--demos 0+1] [--root R]   -> R/demo_XX.npz (skips existing)
    python experiments/s2_rv.py convert [--suffix S]    merge kitchen lol/libero_g8v_raac + the scene-2 cycles -> lol/libero_g8v_raacs2<S>
    python experiments/s2_rv.py norm    [--suffix S]    side norm stats of the union (the training config keeps RAA-c's, see l90c_config.py)
"""
import argparse
import glob
import io
import json
import math
import os
import pathlib
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
LEROBOT_HOME = "/coc/flash10/ksingara3/robot/data/hf/lerobot"
S2_ROOT = "/coc/flash10/ksingara3/robot/data/fa1/rv_s2"
OUT = os.path.join(ROOT, "results", "flawed")
SR89 = os.path.join(OUT, "sr89")
KITCHEN_REPO = "lol/libero_g8v_raac"
REPO = "lol/libero_g8v_raacs2"
W, Y, R, PL, PR = "porcelain_mug_1", "white_yellow_mug_1", "red_coffee_mug_1", "plate_1", "plate_2"
MOVABLE = (W, Y, R, PL, PR)
PHRASE = {W: "the white mug", Y: "the yellow and white mug", R: "the red mug", PL: "the left plate", PR: "the right plate"}
HALF = {W: ("K1", PL), Y: ("K2", PR)}                                       # each mug's demonstrated cycle in the compound demos
EFFECT = 0.03                                                               # m: a mug has risen (rg_stats.EFFECT)
SIZE = 256


def demo_path(root, i):
    return os.path.join(root, "demo_%02d.npz" % int(i))


def demo_files(root):
    return sorted(glob.glob(os.path.join(root, "demo_*.npz")))


# ---------------------------------------------------------------- render (LIBERO venv, GPU)
def render(a):
    """Each recorded state of each compound demo set in the scene's env: both cameras, both simulator segmentations, the eef state, the five
    movables' body positions and the eef at EVERY raw step; no-op steps dropped from the frames (sc_rv.noop_keep).  The last state's goal check
    (both mugs on their plates) is recorded."""
    import sc_rv as RV
    import s2_tasks as T2
    from libero.libero.envs import OffScreenRenderEnv
    z = np.load(T2.DUMP)
    n = int(z["n"])
    demos = list(range(n)) if not a.demos else [int(x) for x in a.demos.split("+") if x]
    if a.part:
        i, k = [int(x) for x in a.part.split("/")]
        demos = [d for d in demos if d % k == i]
    todo = [j for j in demos if not os.path.exists(demo_path(a.root, j))]
    print("[s2rv] render: %d demos requested, %d to do (root %s)" % (len(demos), len(todo), a.root), flush=True)
    if not todo:
        return
    env = OffScreenRenderEnv(bddl_file_name=T2.TEMPLATE, camera_heights=SIZE, camera_widths=SIZE)
    env.seed(0)
    env.reset()
    E = RV.entity_tables(env)
    names = list(E["names"])
    for m in MOVABLE:
        assert m in names, (m, names)
    print("[s2rv] entities %s; without a body id %s; invisible geoms hidden %d" % (names, E["missing"], len(E["hide"])), flush=True)
    nstate = env.get_sim_state().shape[0]
    for j in todo:
        st, ac = np.asarray(z["s%d" % j]), np.asarray(z["a%d" % j], np.float64)
        T = min(len(ac), len(st))
        assert st.shape[1] == nstate, (st.shape, nstate)
        ac = ac[:T]
        keep = RV.noop_keep(ac)
        pos_all, ee_all = np.zeros((T, len(MOVABLE), 3), np.float32), np.zeros((T, 3), np.float32)
        Rr = {k: [] for k in ("image", "wrist", "seg", "wseg", "state")}
        for t in range(T):
            obs = env.set_init_state(st[t])
            pos_all[t] = T2.positions(env, MOVABLE)
            ee_all[t] = np.asarray(obs["robot0_eef_pos"], np.float32)
            if keep[t]:
                Rr["image"].append(RV.flip(obs["agentview_image"]))
                Rr["wrist"].append(RV.flip(obs["robot0_eye_in_hand_image"]))
                Rr["seg"].append(RV.seg_px(env, E, RV.CAMS[0]))
                Rr["wseg"].append(RV.seg_px(env, E, RV.CAMS[1]))
                Rr["state"].append(np.concatenate((obs["robot0_eef_pos"], RV.quat2axisangle(obs["robot0_eef_quat"]),
                                                   obs["robot0_gripper_qpos"])).astype(np.float32))
        goal = bool(env.env._check_success())                               # the last state of the replay: the compound's goal
        RV.atomic_savez(demo_path(a.root, j), image=np.stack(Rr["image"]).astype(np.uint8), wrist=np.stack(Rr["wrist"]).astype(np.uint8),
                        seg=np.stack(Rr["seg"]), wseg=np.stack(Rr["wseg"]), state=np.stack(Rr["state"]),
                        actions=np.asarray(ac[keep], np.float32), kept=np.flatnonzero(keep), raw_len=T, demo=j,
                        pos=pos_all[keep], ee=ee_all[keep], pos_all=pos_all, ee_all=ee_all, goal_at_end=np.asarray(goal),
                        names=np.array(names), movable=np.array(MOVABLE))
        print("[s2rv] demo %d: %d of %d steps kept; goal at the last replayed state %s -> %s" % (j, int(keep.sum()), T, goal, demo_path(a.root, j)), flush=True)
    env.close()
    print("[s2rv] render done", flush=True)


# ---------------------------------------------------------------- cycles and channel (numpy only)
def split_cycles(pos, actions):
    """pos [n, 5, 3] (MOVABLE order), actions [n, 7] of the KEPT steps -> list of (mug, t0, t1) in time order (t1 = the release step, inclusive),
    or None when a mug never rose."""
    cyc = []
    for m in (W, Y):
        o = pos[:, MOVABLE.index(m), :]
        eff = np.flatnonzero(o[:, 2] > o[0, 2] + EFFECT)
        if not len(eff):
            return None
        t_eff = int(eff[0])
        rel = np.flatnonzero(np.asarray(actions)[t_eff:, 6] <= 0)
        cyc.append((t_eff, m, t_eff + int(rel[0]) if len(rel) else len(actions) - 1))
    cyc.sort()
    out, a0 = [], 0
    for _, m, t_rel in cyc:
        out.append((m, a0, t_rel))
        a0 = t_rel + 1
    out[-1] = (out[-1][0], out[-1][1], len(actions) - 1)                      # the last cycle runs to the end of the demo
    return out


def cycle_channel(pos, ee, held, mug, plate):
    """One cycle (arrays sliced to it).  -> rel [n, 3] (the frame's role channel for ITS phase), role-on [n] (= held), info.
    Object role = mug - eef; target role while held = plate - eef - offset, offset = mug - eef at the hold's onset (sc_rv.held_target_rel)."""
    import sc_rv as RV
    po, pt = pos[:, MOVABLE.index(mug)], pos[:, MOVABLE.index(plate)]
    rel_o = po - ee
    info = {}
    rel_t = RV.held_target_rel(po, pt, ee, held, info)
    rel = np.where(held[:, None], rel_t, rel_o)
    return rel, held.copy(), info


# ---------------------------------------------------------------- convert (openpi venv, CPU)
def _openpi():
    if "/coc/flash10/ksingara3/robot/openpi/src" not in sys.path:
        sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")


def convert(a):
    os.environ["HF_LEROBOT_HOME"] = LEROBOT_HOME
    _openpi()
    from PIL import Image
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import sc_rv as RV
    T16 = RV.t16()
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    RV.MOVABLE_OBJ = MOVABLE                                                  # role_paint reads this scene's movables
    repo = REPO + a.suffix
    root = pathlib.Path(LEROBOT_HOME) / repo
    man_path = os.path.join(SR89, "s2_rv_manifest%s.json" % a.suffix)
    assert not (root.exists() and os.path.exists(man_path)), "%s exists and is complete (%s)" % (root, man_path)
    if root.exists():
        print("[s2rv] removing incomplete %s (no %s)" % (root, man_path), flush=True)
        shutil.rmtree(root)
    os.makedirs(SR89, exist_ok=True)
    files = demo_files(a.root)[:(a.ndemo or None)]
    print("[s2rv] convert -> %s; %d rendered compound demos in %s" % (repo, len(files), a.root), flush=True)
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (RV.CHAN_DIM,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=repo, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8,
                               image_writer_processes=4)
    # ---- 1. the kitchen episodes, copied frame by frame from lol/libero_g8v_raac (lossless PNGs decoded and re-encoded)
    kroot = pathlib.Path(LEROBOT_HOME) / KITCHEN_REPO
    tasks = {}
    for ln in open(kroot / "meta" / "tasks.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        tasks[int(d["task_index"])] = d["task"]
    kfiles = sorted((kroot / "data").rglob("episode_*.parquet"))
    nk, kframes = 0, 0
    for f in kfiles[:(a.ndemo or None)]:
        t = pq.read_table(f).to_pydict()
        for i in range(len(t["actions"])):
            im, wi = t["image"][i], t["wrist_image"][i]
            ds.add_frame({"image": np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")),
                          "wrist_image": np.asarray(Image.open(io.BytesIO(wi["bytes"])).convert("RGB")),
                          "state": np.asarray(t["state"][i], np.float32), "actions": np.asarray(t["actions"][i], np.float32),
                          "task": tasks[int(t["task_index"][i])]})
            kframes += 1
        ds.save_episode()
        nk += 1
        if nk % 50 == 0:
            print("[s2rv] kitchen %d episodes, %d frames" % (nk, kframes), flush=True)
    print("[s2rv] kitchen copied: %d episodes, %d frames" % (nk, kframes), flush=True)
    # ---- 2. the scene-2 cycles
    cyc_rows, counts, paint, held_p99, rel_rel = [], {}, {}, [], []
    skipped, agree, onset_gap = [], [], []
    mug_first = {"K1": 0, "K2": 0}
    for f in files:
        z = dict(np.load(f))
        names = [str(x) for x in z["names"]]
        H_all = np.asarray(T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]), bool)
        cyc = split_cycles(z["pos"], z["actions"])
        j = int(z["demo"])
        if cyc is None or len(cyc) != 2:
            skipped.append(j)
            continue
        mug_first[HALF[cyc[0][0]][0]] += 1
        for ci, (mug, t0, t1) in enumerate(cyc):
            key, plate = HALF[mug]
            sl = slice(t0, t1 + 1)
            held = H_all[sl]
            pos, ee = z["pos"][sl].astype(np.float64), z["ee"][sl].astype(np.float64)
            rel, _, info = cycle_channel(pos, ee, held, mug, plate)
            # sanity: the detector's hold against the simulator's (the mug rose > 2 cm above its cycle-start height within 12 cm of the hand)
            o = pos[:, MOVABLE.index(mug)]
            simh = (o[:, 2] > o[0, 2] + 0.02) & (np.linalg.norm(o[:, :2] - ee[:, :2], axis=1) < 0.12)
            agree.append(float((simh == held).mean()))
            if held.any() and simh.any():
                onset_gap.append(int(np.argmax(held)) - int(np.argmax(simh)))
            h_i = np.flatnonzero(held)
            if len(h_i):
                rel_rel.append([round(100.0 * float(v), 2) for v in rel[h_i[-1]]])              # target channel at the last held row (cm)
                held_p99.append(np.abs(z["actions"][sl][held][:, :3]))
            plan = []
            for h in (False, True):
                slots, act = RV.role_paint(mug, plate, h)
                plan.append(([names.index(s) for s in slots], None if act is None else names.index(act)))
            ph_o, ph_t = "object: " + PHRASE[mug], "target: " + PHRASE[plate]
            for i in range(t0, t1 + 1):
                h = bool(H_all[i])
                pid, aid = plan[int(h)]
                img, na = RV.paint_many(z["image"][i], z["seg"][i], pid, aid)
                wimg, nw = RV.paint_many(z["wrist"][i], z["wseg"][i], pid, aid)
                task = ph_t if h else ph_o
                counts[task] = counts.get(task, 0) + 1
                s = paint.setdefault("%s|%s" % (key, "held" if h else "pre"), dict(frames=0, agent_px=0, wrist_px=0, agent_empty=0, wrist_empty=0))
                s["frames"] += 1
                s["agent_px"] += na
                s["wrist_px"] += nw
                s["agent_empty"] += int(na == 0)
                s["wrist_empty"] += int(nw == 0)
                ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "state": RV.chan_state(rel[i - t0], h, True),
                              "actions": np.asarray(z["actions"][i], np.float32), "task": task})
            ds.save_episode()
            cyc_rows.append(dict(demo=j, cycle=ci, task=key, mug=mug, frames=int(t1 - t0 + 1), held_frames=int(held.sum()),
                                 offset_cm=info["offset_cm"][0] if info.get("offset_cm") else None,
                                 sim_agree=agree[-1], goal_at_end=bool(z["goal_at_end"])))
        print("[s2rv] demo %d: cycles %s" % (j, [(HALF[m][0], t0, t1) for m, t0, t1 in cyc]), flush=True)
    ds.stop_image_writer()
    for s in paint.values():
        s["agent_frac"] = s["agent_px"] / float(max(1, s["frames"]) * SIZE * SIZE)
        s["wrist_frac"] = s["wrist_px"] / float(max(1, s["frames"]) * SIZE * SIZE)
    hp = np.concatenate(held_p99, 0) if held_p99 else np.zeros((0, 3))
    rr = np.asarray(rel_rel) if rel_rel else np.zeros((0, 3))
    gate = json.load(open(os.path.join(OUT, "raas_gate.json"), encoding="utf-8"))
    man = dict(repo_id=repo, kitchen_repo=KITCHEN_REPO, kitchen_episodes=nk, kitchen_frames=kframes, scene2_demos=len(files),
               scene2_demos_skipped=skipped, scene2_cycles=len(cyc_rows), scene2_cycles_per_task={k: sum(1 for r in cyc_rows if r["task"] == k) for k in ("K1", "K2")},
               scene2_frames=int(sum(r["frames"] for r in cyc_rows)), scene2_held_frames=int(sum(r["held_frames"] for r in cyc_rows)),
               episodes=int(ds.meta.total_episodes), frames=int(ds.meta.total_frames), first_cycle_mug_per_demo=mug_first,
               frames_per_scene2_prompt=counts, paint=paint, tw=X["tw"], lift=X["lift"],
               sanity=dict(hold_vs_sim_agree_mean=float(np.mean(agree)) if agree else None, hold_vs_sim_agree_min=float(np.min(agree)) if agree else None,
                           onset_gap_frames_median=float(np.median(onset_gap)) if onset_gap else None,
                           onset_gap_frames_range=[int(min(onset_gap)), int(max(onset_gap))] if onset_gap else None,
                           cycles_without_hold=[r for r in cyc_rows if r["held_frames"] == 0],
                           goal_at_end_all=bool(all(r["goal_at_end"] for r in cyc_rows)) if cyc_rows else None,
                           target_channel_at_last_held_row_cm_median=np.median(np.abs(rr), 0).round(2).tolist() if len(rr) else None,
                           target_channel_at_last_held_row_xy_cm_median=float(np.median(np.linalg.norm(rr[:, :2], axis=1))) if len(rr) else None,
                           in_hand_offset_cm_median=np.median(np.asarray([r["offset_cm"] for r in cyc_rows if r["offset_cm"]]), 0).round(2).tolist()
                           if any(r["offset_cm"] for r in cyc_rows) else None,
                           channel_error_vs_simulator_truth="0 by construction (the labels ARE simulator body positions); the mask + depth estimator's "
                                                            "error at test time is logged by wa_s2.py's chan diag"),
               servo_check=dict(kitchen_servo_max=gate["max_step"], kitchen_K=gate["K"],
                                scene2_p99_abs_action_xyz_on_held_rows=np.percentile(hp, 99).round(4).item() if len(hp) else None,
                                scene2_p99_per_axis=np.percentile(hp, 99, axis=0).round(4).tolist() if len(hp) else None,
                                note="servo constants unchanged (RAA-sL's recipe); this is the check the plan asks for, not a refit"),
               cycles=cyc_rows, root=a.root)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("[s2rv] done %s: %d episodes, %d frames (kitchen %d + scene 2 %d)" % (repo, man["episodes"], man["frames"], kframes, man["scene2_frames"]), flush=True)
    print("[s2rv] frames per scene-2 prompt %s" % json.dumps(counts), flush=True)
    print("[s2rv] sanity %s" % json.dumps(man["sanity"]), flush=True)
    print("[s2rv] servo check %s" % json.dumps(man["servo_check"]), flush=True)
    for key, s in sorted(paint.items()):
        print("[s2rv] paint %-9s frames %6d  agent frac %.4f (empty %d)  wrist frac %.4f (empty %d)" % (
            key, s["frames"], s["agent_frac"], s["agent_empty"], s["wrist_frac"], s["wrist_empty"]), flush=True)


def norm(a):
    """Side statistics of the union (state, actions): written next to the manifest, NOT used by the training config (which keeps RAA-c's norm
    stats, as RAA-sL does: the servo's constants and the init expert's normalisation stay as they were).  Prints the quantile shift."""
    _openpi()
    import pyarrow.parquet as pq
    import openpi.shared.normalize as normalize
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import l90c_config as LC
    cfg = LC.train_config("pi05_g8v_raas_ls2_ae")
    repo = REPO + a.suffix
    files = sorted((pathlib.Path(LEROBOT_HOME) / repo / "data").rglob("episode_*.parquet"))
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    n = 0
    for f in files:
        t = pq.read_table(f, columns=["state", "actions"]).to_pydict()
        st, ac = np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32)
        idx, _ = D.chunk_indices(len(ac), cfg.model.action_horizon)
        stats["state"].update(st)
        stats["actions"].update(ac[idx])
        n += len(ac)
    res = {k: s.get_statistics() for k, s in stats.items()}
    old = json.load(open(os.path.join(LC.L90C, "assets", "pi05_g8v_raac", KITCHEN_REPO, "norm_stats.json"), encoding="utf-8"))["norm_stats"]
    out = {}
    for k in ("state", "actions"):
        out[k] = dict(q01_union=[round(float(v), 4) for v in res[k].q01], q99_union=[round(float(v), 4) for v in res[k].q99],
                      q01_raac=[round(float(v), 4) for v in old[k]["q01"]], q99_raac=[round(float(v), 4) for v in old[k]["q99"]])
    p = os.path.join(SR89, "s2_rv_norm_side%s.json" % a.suffix)
    os.makedirs(SR89, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(frames=n, episodes=len(files), **out), fh, indent=1)
    print("[s2rv] norm side stats over %d frames of %d episodes -> %s" % (n, len(files), p), flush=True)
    for k in out:
        print("[s2rv] %s %s" % (k, json.dumps(out[k])), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("render", "convert", "norm"))
    ap.add_argument("--root", default=S2_ROOT)
    ap.add_argument("--part", default="", help="render: i/n, the demos with index %% n == i")
    ap.add_argument("--demos", default="", help="render: demo indices, '+'-joined")
    ap.add_argument("--suffix", default="", help="appended to the dataset repo and manifest (smoke: _smoke)")
    ap.add_argument("--ndemo", type=int, default=0, help="convert: only the first N rendered demos / kitchen episodes (smoke dataset)")
    a = ap.parse_args()
    {"render": render, "convert": convert, "norm": norm}[a.what](a)

"""PLAN_PARTS F1 (docs/PLAN_PARTS.md section 3): the frames every Layer-1 intervention is scored on.  LIBERO venv, EGL (GPU node).
No policy is run; no demo action is executed (demo frames are recorded simulator states, set and rendered).

  grasp frames  N2's main start states 4-13 of each layout (results/flawed/n2_states; every N2 task of a layout shares them):
                reset, set the init state, 10 settle steps of the dummy action -- the observation N2's client sends first.
  place frames  the first 10 demos of the lowest-index known task of each movable object (bowl K1, wine bottle K4, cream cheese K6):
                the recorded state HOLD_AFTER kept steps after the stage-1b held flag (demos_b.npz) first turns on (the object just
                lifted in the gripper, before the carry heads to the task's own target).  Set and rendered, no settle.
Per frame: the model images (agentview and wrist, flipped 180 degrees and resize_with_pad to 224, as N2's client), the 8-D state, the
oracle scene (fa1_common.scene), and the projection of every oracle entity point into the 224 model image (row, col), from robosuite's
agentview camera matrix.  Place frames also carry their demos.npz row index (the oracle arms' held / offset / canonical inputs).
-> results/flawed/pp_f1_frames.npz (+ pp_f1_frames_check.png: entity projections drawn on two frames, for a visual check)

    python experiments/pp_f1_frames.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(FC.ROOT, "results", "flawed")
STATES = list(range(4, 14))
SRC = {"akita_black_bowl_1": "K1", "wine_bottle_1": "K4", "cream_cheese_1": "K6"}
N_DEMO, HOLD_AFTER = 10, 5


def quat2axisangle(quat):
    q = np.asarray(quat, np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def project(sim, pts, H, W, cam="agentview"):
    """robosuite.utils.camera_utils' world -> pixel projection (that module imports h5py, absent from the LIBERO venv): -> (row, col)
    in the upright image."""
    cid = sim.model.camera_name2id(cam)
    f = 0.5 * H / np.tan(sim.model.cam_fovy[cid] * np.pi / 360)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    R = np.eye(4)
    R[:3, :3] = np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)
    R[:3, 3] = np.asarray(sim.data.cam_xpos[cid])
    R = R @ np.diag([1.0, -1.0, -1.0, 1.0])
    Ke = np.eye(4)
    Ke[:3, :3] = K
    P = Ke @ np.linalg.inv(R)
    x = P @ np.concatenate([np.asarray(pts, np.float64), np.ones((len(pts), 1))], 1).T
    x = (x[:2] / x[2:3]).T
    return np.stack([np.clip(x[:, 1], 0, H - 1), np.clip(x[:, 0], 0, W - 1)], -1)


def main():
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    Db = np.load(os.path.join(FC.DATA, "demos_b.npz"))
    rec = {k: [] for k in ("img", "wimg", "state8", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "pix", "phase", "layout",
                           "state_k", "src_task", "src_demo", "row", "obj")}

    def grab(env, obs, dj):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        H, W = img.shape[:2]
        rec["img"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
        rec["wimg"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)))
        rec["state8"].append(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                             obs["robot0_gripper_qpos"])).astype(np.float32))
        sc = FC.scene(env, obs, dj)
        for k in ("ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
            rec[k].append(np.asarray(sc[k], np.float32))
        rc = project(env.env.sim, sc["pos"], H, W)                                                  # (row, col), upright image
        # the model image is the raw (OpenGL, upside-down) frame flipped in both axes = the upright frame mirrored left-right
        pix = np.stack([rc[:, 0], (W - 1) - rc[:, 1]], -1).astype(np.float32) * (224.0 / H)
        rec["pix"].append(pix)

    # grasp frames: N2 start states (shared by every task of a layout)
    for lay in ("familiar", "new"):
        tids = [t["tid"] for t in n2["tasks"] if t["layout"] == lay]
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tids[0]}.npy"))
        for t in tids[1:]:
            assert np.array_equal(S, np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t}.npy"))), ("layout states differ", lay, t)
        bddl = os.path.join(FC.ROOT, next(t["bddl"] for t in n2["tasks"] if t["tid"] == tids[0]))
        for k in STATES:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(7)
            env.reset()
            dj = FC.drawer_joint(env)
            obs = env.set_init_state(S[k])
            for _ in range(10):
                obs, _, _, _ = env.step(FC.DUMMY)
            grab(env, obs, dj)
            rec["phase"].append(0); rec["layout"].append(0 if lay == "familiar" else 1); rec["state_k"].append(k)
            rec["src_task"].append(-1); rec["src_demo"].append(-1); rec["row"].append(-1); rec["obj"].append(-1)
            env.close()
        print(f"[pp_f1_frames] grasp {lay}: {len(STATES)} states", flush=True)

    # place frames: demo states just after the hold's onset
    tasks = list(FC.KNOWN)
    for obj, K in SRC.items():
        ti = tasks.index(K)
        raw = np.load(os.path.join(FC.DATA, f"raw_{K}.npz"))
        rows = np.flatnonzero(D["task"] == ti)
        assert len(rows) == len(raw["states"]) and np.array_equal(D["demo"][rows], raw["demo"]), ("demos.npz / raw rows differ", K)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(FC.LIBBDDL, FC.KNOWN[K] + ".bddl"), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        dj = FC.drawer_joint(env)
        for dm in range(N_DEMO):
            r = rows[D["demo"][rows] == dm]
            h = np.flatnonzero(Db["held"][r] > 0.5)
            assert len(h), ("no hold", K, dm)
            j = int(min(h[0] + HOLD_AFTER, len(r) - 1))
            assert Db["held"][r[j]] > 0.5, ("not held at the chosen step", K, dm)
            obs = env.set_init_state(raw["states"][r[j] - rows[0]])
            grab(env, obs, dj)
            assert np.allclose(rec["pos"][-1], D["pos"][r[j]], atol=1e-4), ("oracle scene differs from demos.npz", K, dm)
            rec["phase"].append(1); rec["layout"].append(0); rec["state_k"].append(-1)
            rec["src_task"].append(ti); rec["src_demo"].append(dm); rec["row"].append(int(r[j])); rec["obj"].append(FC.ENT.index(obj))
        env.close()
        print(f"[pp_f1_frames] place {obj} from {K}: {N_DEMO} frames", flush=True)
    Z = {k: np.stack(v) if k in ("img", "wimg", "state8", "ee_pos", "ee_rot", "grip", "pos", "yaw", "pix") else np.asarray(v)
         for k, v in rec.items()}
    np.savez_compressed(os.path.join(OUT, "pp_f1_frames.npz"), ent=np.array(FC.ENT), **Z)
    from PIL import Image, ImageDraw
    ims = []
    for i in (0, len(Z["img"]) - 1):
        im = Image.fromarray(Z["img"][i]).resize((448, 448))
        d = ImageDraw.Draw(im)
        for e, (r, c) in enumerate(Z["pix"][i]):
            d.ellipse([2 * c - 4, 2 * r - 4, 2 * c + 4, 2 * r + 4], outline=(255, 0, 0))
            d.text((2 * c + 5, 2 * r - 5), FC.ENT[e][:10], fill=(255, 255, 0))
        ims.append(im)
    canvas = Image.new("RGB", (896, 448))
    canvas.paste(ims[0], (0, 0)); canvas.paste(ims[1], (448, 0))
    canvas.save(os.path.join(OUT, "pp_f1_frames_check.png"))
    print(f"[pp_f1_frames] wrote {len(Z['img'])} frames", flush=True)


if __name__ == "__main__":
    main()

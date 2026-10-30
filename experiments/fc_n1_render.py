"""FLAWED_COMPOSITION N1 (section 4.9), DATA ONLY: the initial scene of every N1 task, as the client starts an episode (reset,
set_init_state, 10 settle steps of the dummy action; experiments/pl0_render.py's convention).  No policy is run, no demo is
executed.  The LIBERO venv (py3.8), EGL on one GPU.

  default  the tasks of results/flawed/n1_tasks.json, standard LIBERO, the benchmark's init states 0-9 (the BindVLA dev / parity
           episodes use init states 0-9)
  --li     Li's 20 OOD tasks in Li's modified LIBERO, the 10 practice start states of set a (results/robot/fresh_ood_a; the same states
           results/lang/frames was rendered from), re-rendered here to add the robot base position
Per state: agentview / wrist image (flipped, padded to 224), the 8-d state, every object's and fixture's position and the robot base
position (world frame).  -> results/flawed/n1_render/<suite>_<index>.npz.  Resumable.  --shard i/n splits the task list.

    python experiments/fc_n1_render.py [--li] [--shard 0/3]
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed", "n1_render")
FRESH = os.path.join(ROOT, "results", "robot", "fresh_ood_a")
DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    import numpy as np
    q = np.asarray(quat, np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--li", action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.li:
        sys.path.insert(0, "/coc/flash10/ksingara3/ext_dl/pi0tl/pi0-text-latent/third_party/modified_libero")
        os.environ["LIBERO_CONFIG_PATH"] = "/coc/flash10/ksingara3/robot/libero_ood_config"
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    os.makedirs(OUT, exist_ok=True)
    if a.li:
        todo = [(s, i) for s in ("libero_goal_ood", "libero_spatial_ood") for i in range(10)]
    else:
        tl = json.load(open(os.path.join(ROOT, "results", "flawed", "n1_tasks.json")))["tasks"]
        todo = [(t.split(":")[0], int(t.split(":")[1])) for t in tl]
    k, n = map(int, a.shard.split("/"))
    todo = todo[k::n]
    bd = benchmark.get_benchmark_dict()
    for sname, tid in todo:
        out = os.path.join(OUT, "%s_%d.npz" % (sname, tid))
        if os.path.exists(out):
            continue
        suite = bd[sname]()
        task = suite.get_task(tid)
        states = np.load(os.path.join(FRESH, "%s_%d.npy" % (sname, tid))) if a.li else suite.get_task_init_states(tid)[:10]
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        inner = env.env
        names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
        kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
        rec = {k_: [] for k_ in ("img", "wimg", "state8", "pos", "base")}
        for s in states:
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
        tmp = out + ".%d.tmp.npz" % os.getpid()
        np.savez_compressed(tmp, names=np.array(names), kinds=np.array(kinds), language=str(task.language),
                            **{k_: np.stack(v) for k_, v in rec.items()})
        os.replace(tmp, out)
        env.close()
        print("[n1_render] %s %d '%s': %d states" % (sname, tid, task.language, len(states)), flush=True)


if __name__ == "__main__":
    main()

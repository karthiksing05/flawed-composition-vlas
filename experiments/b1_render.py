"""PLAN_BINDER B1 (docs/PLAN_BINDER.md section 7): the unsupervised referents of every TRAINING cell's matched demo.  LIBERO env.

Per cell of results/bind/b1/cells.json (b1_acts.py): the demo's simulator states (its match file) are SET, never stepped:
  frame 0      the scene before any contact, rendered as the policy sees it (s0_data.frame_no_settle: agentview[::-1, ::-1],
               224) + the camera's K / E (robosuite) -> the image the binder's regions come from
  events       contact events from the recorded gripper command (pi convention, +1 = close): a GRASP is the first frame of a
               closed run of >= RUN frames, its RELEASE the frame that run ends (primitives.releases' rule, RUN = 10)
  referents    the gripper's position (robot0_eef_pos, proprioception) at each event, projected into frame 0 with K / E
               (where.project, client orientation) -> (row, col) at 256 and the 16 x 16 token.  The object role's referent is the
               token under the gripper at the grasp, the goal role's the token under the gripper at the release.
Simulator segmentation of frame 0 (body_cov / site_cov, S1's masks) is stored for SCORING / DIAGNOSTICS ONLY (part-proposal
recall, referent purity); the binder never reads it.

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/b1_render.py --shard i/n
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
OD = os.path.join(ROOT, "results", "bind", "b1", "demo")
RUN = 10


def events(grip, run=RUN):
    """[(grasp frame, release frame or None)] of every closed run of >= run frames."""
    c = np.asarray(grip) > 0
    out, i, T = [], 0, len(c)
    while i < T:
        if c[i]:
            j = i
            while j < T and c[j]:
                j += 1
            if j - i >= run:
                out.append((i, j if j < T else None))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import f0b_firstframes_seg as F0
    import libero_client as LC
    import s0_data as S0
    import s1_render as SR
    W = SR._where()
    C = json.load(open(os.path.join(ROOT, "results", "bind", "b1", "cells.json"), encoding="utf-8"))["cells"]
    i, n = map(int, a.shard.split("/"))
    os.makedirs(OD, exist_ok=True)
    for c in C[i::n]:
        p = os.path.join(OD, c["tid"].replace(":", "_") + ".npz")
        if os.path.exists(p):
            continue
        t0 = time.time()
        suite, idx = c["tid"].split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(idx))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        states = np.load(os.path.join(ROOT, c["match"]), allow_pickle=True)["states"]
        acts = np.load(os.path.join(ROOT, "results", "bind", "b1", "acts", c["tid"].replace(":", "_") + ".npy"))
        T = min(len(states), len(acts))
        env.set_init_state(states[0])
        sim = env.env.sim
        fr = S0.frame_no_settle(env, sim, W, LC)
        K = F0.CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
        E = F0.CU.get_camera_extrinsic_matrix(sim, "agentview")
        ev = []
        for g, r in events(acts[:T, 6]):
            for kind, f in (("grasp", g), ("release", r)):
                if f is None:
                    continue
                env.set_init_state(states[min(f, T - 1)])
                obs = env.env._get_observations(force_update=True)
                eef = np.asarray(obs["robot0_eef_pos"], np.float64)
                rc = W.project(eef, K, E)
                tok = int(np.clip(rc[0] // 16, 0, 15)) * 16 + int(np.clip(rc[1] // 16, 0, 15))
                ev.append(dict(kind=kind, frame=int(f), eef=eef.tolist(), rc=[float(rc[0]), float(rc[1])], token=tok))
        env.close()
        np.savez_compressed(p + ".tmp.npz", tid=c["tid"], prompt=c["lang"], K=K, E=E, events=json.dumps(ev), T=T, **fr)
        os.replace(p + ".tmp.npz", p)
        print(f"[b1 render] {c['tid']}: T {T}, events {[(e['kind'], e['frame'], e['token']) for e in ev]}, {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()

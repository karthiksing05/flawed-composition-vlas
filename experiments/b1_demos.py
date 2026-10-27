"""PLAN_BINDER B1 (docs/PLAN_BINDER.md section 7): cross-situational referents from ALL of a training cell's demos.  Chunks env, CPU.

b1_render.py renders the ONE matched demo per cell in the simulator (its frame-0 masks serve diagnostics).  The binder's statistics
come from every recorded demo of the cell instead (LeRobot ports already on disk: physical-intelligence/libero for libero_10 /
goal / object / spatial, IPEC libero_90; HF_HUB_OFFLINE=1, a file not on disk is skipped):
  frame 0     the demo's first agentview frame as stored (the policy's orientation), resized 256 -> 224 with openpi_client's
              resize_with_pad (the policy's own preprocessing)
  events      the recorded gripper command (pi convention) -> grasp / release frames (b1_render.events, RUN = 10)
  referents   the recorded proprioceptive eef position (state[:3]) at each event, projected into frame 0 with the cell's agentview
              K / E (b1_render; the camera is fixed per task) -> token
No simulator label is read.  -> results/bind/b1/demos/<tid>.npz: img [n, 224, 224, 3] uint8, episodes [n], events (json).

    HF_HUB_OFFLINE=1 python experiments/b1_demos.py [--max-eps 50]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments"), "/coc/flash10/ksingara3/robot/openpi/packages/openpi-client/src"]
OD = os.path.join(ROOT, "results", "bind", "b1", "demos")


def first_frame_mp4(path):
    import av
    with av.open(path) as c:
        for f in c.decode(video=0):
            return f.to_ndarray(format="rgb24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eps", type=int, default=50)
    a = ap.parse_args()
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import where as W
    from openpi_client import image_tools
    from PIL import Image
    import b1_render as BR
    os.makedirs(OD, exist_ok=True)
    C = json.load(open(os.path.join(ROOT, "results", "bind", "b1", "cells.json"), encoding="utf-8"))["cells"]
    demos, d90 = D.Demos(), D.Demos90()
    sp = os.path.join(OD, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    for c in C:
        p = os.path.join(OD, c["tid"].replace(":", "_") + ".npz")
        rf = os.path.join(ROOT, "results", "bind", "b1", "demo", c["tid"].replace(":", "_") + ".npz")
        if os.path.exists(p) or not os.path.exists(rf):
            continue
        z = np.load(rf, allow_pickle=True)
        K, E = z["K"], z["E"]
        if c["ipec"]:
            eps = d90.episodes_of_task(int(c["tid"].split(":")[1]))
        else:
            ti = demos.load(c["episode"])["task_index"]
            eps = demos.episodes(ti)
        imgs, kept, evs, skipped = [], [], [], 0
        for e in eps[:a.max_eps]:
            try:
                if c["ipec"]:
                    ep = d90.load(e)
                    rel = d90.info["video_path"].format(episode_chunk=e // d90.chunks_size, video_key="observation.images.image", episode_index=e)
                    vp = os.path.join(d90.root, rel)
                    if not os.path.exists(vp):
                        raise FileNotFoundError(vp)
                    im = first_frame_mp4(vp)
                else:
                    ep = demos.load(e, images=True)
                    im = np.asarray(Image.open(io.BytesIO(ep["image"][0])).convert("RGB"), np.uint8)
            except Exception:                                       # noqa: BLE001
                skipped += 1
                continue
            ev = []
            for g, r in BR.events(ep["actions"][:, 6]):
                for kind, f in (("grasp", g), ("release", r)):
                    if f is None:
                        continue
                    eef = np.asarray(ep["state"][f][:3], np.float64)
                    rc = W.project(eef, K, E)
                    tok = int(np.clip(rc[0] // 16, 0, 15)) * 16 + int(np.clip(rc[1] // 16, 0, 15))
                    ev.append(dict(kind=kind, frame=int(f), eef=eef.tolist(), rc=[float(rc[0]), float(rc[1])], token=tok))
            imgs.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(im, 224, 224)))
            kept.append(int(e))
            evs.append(ev)
        if not imgs:
            summary[c["tid"]] = dict(n=0, skipped=skipped)
            continue
        np.savez_compressed(p + ".tmp.npz", tid=c["tid"], img=np.stack(imgs), episodes=np.array(kept), events=json.dumps(evs))
        os.replace(p + ".tmp.npz", p)
        m = c["episode"] in kept and kept.index(c["episode"])
        chk = ""
        if m is not False:
            mine = [x["token"] for x in json.loads(str(z["events"]))]
            chk = f" | matched ep {c['episode']}: dataset tokens {[x['token'] for x in evs[m]]} vs rendered {mine}"
        summary[c["tid"]] = dict(n=len(imgs), skipped=skipped)
        print(f"[b1 demos] {c['tid']}: {len(imgs)} demos ({skipped} skipped){chk}", flush=True)
    json.dump(summary, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main()

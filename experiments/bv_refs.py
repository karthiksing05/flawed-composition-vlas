"""PLAN_BINDVLA iteration 1: per-frame contact-event referents for the BindVLA training cache (label-free: the demo's own gripper
command + proprioception, projected with the cell's fixed agentview camera; the referent source of B1 / B1.3, PLAN_BINDER 3.1).
No simulator mask or annotation is read.

For every cache episode whose cell has b1 demo events (results/bind/b1/demos/<tid>.npz, every recorded demo of the 61 cells), step j
of the instruction is paired with the j-th transport (b1_bind.transports: grasp, release).  Per frame f of the episode:
  object referent of step j = the grasp token, for f < the grasp frame (the object has not moved yet), else -1
  goal referent of step j   = the release token, for f <= the release frame (the receptacle is static), else -1
Sanity: the grasp frame must be where the cache's gripper action closes (pi convention +1), within 2 frames.
-> CACHE/refs.npz: ref [n_frames, n_steps, 2] int16 (-1 = none); results/bindvla/refs.json (summary).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
S = 2


def main():
    import b1_bind as B
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    meta = json.load(open(os.path.join(CACHE, "meta.json"), encoding="utf-8"))
    z = np.load(os.path.join(CACHE, "frames.npz"))
    E, act = z["ep"], z["actions"]
    n = len(E)
    first = {}
    for i in np.flatnonzero(np.r_[True, np.diff(E) != 0]):
        first[int(E[i])] = int(i)
    by_key = {(e["tid"], int(e["src_ep"])): e for e in meta["episodes"]}
    ref = np.full((n, S, 2), -1, np.int16)
    st = dict(episodes=0, frames_obj=0, frames_goal=0, grasp_check=[], skipped_short=0, cells=0)
    for f in sorted(os.listdir(os.path.join(B.B1, "demos"))):
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(B.B1, "demos", f), allow_pickle=True)
        tid = str(d["tid"])
        if tid in B.SKIP_CELLS:
            continue
        evs = json.loads(str(d["events"]))
        lang = next((e["text"] for e in meta["episodes"] if e["tid"] == tid), None)
        if lang is None:
            continue
        steps = [SL.split_step(s) for s in P.steps_of(lang)]
        st["cells"] += 1
        for k, src in enumerate(d["episodes"]):
            e = by_key.get((tid, int(src)))
            if e is None:
                continue
            pairs = B.transports(evs[k])
            if len(pairs) < len(steps):
                st["skipped_short"] += 1
                continue
            i0, L = first[e["ep"]], e["length"]
            st["episodes"] += 1
            for j, (o, g) in enumerate(steps[:S]):
                gr, re_ = pairs[j]
                fg = int(gr["frame"])
                closed = np.flatnonzero(act[i0:i0 + L, 6] > 0)
                if len(closed):
                    near = closed[np.argmin(np.abs(closed - fg))]
                    st["grasp_check"].append(int(abs(near - fg) <= 2))
                ref[i0:i0 + min(fg, L), j, 0] = gr["token"]
                if g and re_ is not None:
                    ref[i0:i0 + min(int(re_["frame"]) + 1, L), j, 1] = re_["token"]
    st["frames_obj"] = int((ref[:, :, 0] >= 0).any(1).sum())
    st["frames_goal"] = int((ref[:, :, 1] >= 0).any(1).sum())
    st["grasp_check"] = float(np.mean(st["grasp_check"])) if st["grasp_check"] else None
    held = {e["ep"] for e in meta["episodes"] if e["held_out"]}
    st["frames_obj_heldout"] = int(((ref[:, :, 0] >= 0).any(1) & np.isin(E, sorted(held))).sum())
    st["n_frames"] = n
    np.savez(os.path.join(CACHE, "refs.npz"), ref=ref)
    os.makedirs(os.path.join(ROOT, "results", "bindvla"), exist_ok=True)
    json.dump(st, open(os.path.join(ROOT, "results", "bindvla", "refs.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()

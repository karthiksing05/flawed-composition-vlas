"""FLAWED_COMPOSITION D2, DATA ONLY (no model is run here): the frames on which the archive VLAs' debt is measured.  SmolVLA venv, CPU.

Blocks (the OOD tasks that have per-task zero-shot successes in results/robot, 20 episodes each):
  L90  pi0.5 (pi05_libero) on the LIBERO-90 tasks of the G8 read (results/robot/g8_read.json, arm zs).  A task that the LIBERO-90-C
       split seals is skipped (l90c.is_sealed reads only the salted hashes in split.json; split_sealed.json is never opened).  Per
       task, the first 10 IPEC demos (l90c.task_episodes), passed through the BindVLA cache pipeline as in fc_d1_frames.py:
       frame 0 of each demo (purity and bypass) and the training-referent event frames of fc_d1_frames.frames_of (combination;
       scored only, never trained on, never executed).
  LI   pi0.5, GR00T N1.5 and X-VLA on Li's 20 OOD tasks (g15_read_pi05.json, g18_read_groot_libero.json, g18_read_xvla.json):
       the first observations of the 10 practice start states of set a already rendered by experiments/pl0_render.py
       (results/lang/frames; set a is a practice set, never a scoring set).
-> CACHE/d2_l90.npz, results/flawed/d2_frames.json

    python experiments/fc_d2_frames.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
OUT = os.path.join(ROOT, "results", "flawed")
R = os.path.join(ROOT, "results", "robot")


def main():
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    from fc_d1_frames import chunk, frames_of, pipe
    g8 = json.load(open(os.path.join(R, "g8_read.json"), encoding="utf-8"))["per_task"]
    d90 = D.Demos90()
    A, W, S, ACT, PAD, TID, EP, FR, PH, KIND = ([] for _ in range(10))
    skipped, per = [], {}
    for k in sorted(g8, key=int):
        tid = f"libero_90:{k}"
        if l90c.is_sealed(tid):
            skipped.append(tid)
            continue
        nf = 0
        for e in l90c.task_episodes(tid)[:10]:
            ep = d90.load(e)
            img, wr = d90.frames(e)
            assert len(img) == len(ep["actions"]) == len(wr), (tid, e)
            fl = [(0, "f0", -1)] + [x for x in frames_of(ep["actions"], ep["state"]) if x[0] != 0]
            for f, ph, _ in fl:
                a, p = chunk(ep["actions"], f)
                A.append(pipe(img[f])), W.append(pipe(wr[f])), S.append(ep["state"][f]), ACT.append(a), PAD.append(p)
                TID.append(tid), EP.append(int(e)), FR.append(f), PH.append(ph), KIND.append("f0" if ph == "f0" else "ev")
            nf += len(fl)
        per[tid] = dict(frames=nf, zs=g8[k]["zs"])
        print(f"[d2_frames] {tid} {nf} frames", flush=True)
    np.savez(os.path.join(CACHE, "d2_l90.npz"), agent=np.stack(A), wrist=np.stack(W), state=np.stack(S).astype(np.float32),
             actions=np.stack(ACT).astype(np.float32), pad=np.stack(PAD), tid=np.array(TID), ep=np.array(EP), frame=np.array(FR),
             phase=np.array(PH), kind=np.array(KIND))
    li = {}
    for s, pre in (("libero_goal_ood", "goal"), ("libero_spatial_ood", "spatial")):
        for i in range(10):
            z = np.load(os.path.join(ROOT, "results", "lang", "frames", f"{s}_{i}.npz"))
            li[f"{pre}:{i}"] = dict(n=int(len(z["img"])), shape=list(z["img"].shape), language=str(z["language"]))
    out = dict(l90_tasks=len(per), l90_skipped_sealed=skipped, l90_frames=len(TID), l90=per, li=li)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "d2_frames.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(dict(l90_tasks=len(per), skipped=len(skipped), frames=len(TID), li=len(li)), indent=1))


if __name__ == "__main__":
    main()

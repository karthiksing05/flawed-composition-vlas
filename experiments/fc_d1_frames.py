"""FLAWED_COMPOSITION D1, DATA ONLY (no model is run here): the phase-local scoring frames of the 29 dev tasks' own demos.

The demos are SCORED ONLY: never trained on, never executed.  They are the same 10 demos per dev task that bv_devrefs.py used
(l90c.task_episodes(tid)[:10]; every tid passes l90c.check_task; split_sealed.json is never opened).

Frames go through the TRAINING cache's pipeline (experiments/bv_data.py): the IPEC mp4 frame (Demos90.frames, 256 px) ->
PIL bilinear resize to 224 -> JPEG q95 -> decode.  Events are the training-referent rule: b1_render.events on the recorded
gripper command (pi convention), eef = state[:3], b1_bind.transports drops regrasps.  For each of the first two transports
(grasp g, release r) the frames are g-10, g-5, g (phase "grasp") and r-10, r-5, r (phase "release"), clipped to the episode.
An episode with no transport (no closed gripper run, e.g. a drawer pushed shut) is scored at 0.25, 0.5 and 0.75 of its length
(phase "none").
Stored per frame: agentview / wrist images, raw state [8], the raw action chunk a[f : f + 50] (pi convention) and its pad mask.

Pipeline parity check (training data, recorded): the same extraction is applied to 2 held-out TRAINING episodes of LIBERO-90
cells, and their images are compared with the cache's images of the same frames (mean abs pixel difference).
-> CACHE/d1_dev.npz, results/flawed/d1_frames.json

    python experiments/fc_d1_frames.py          (SmolVLA venv, CPU)
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
OUT = os.path.join(ROOT, "results", "flawed")
H = 50
OFFS = (10, 5, 0)
FALLBACK = (0.25, 0.5, 0.75)


def pipe(img):
    from PIL import Image
    import bv_data as BD
    return np.asarray(Image.open(io.BytesIO(BD.jpeg(BD.resize(np.asarray(img))))).convert("RGB"))


def frames_of(actions, state):
    """-> [(frame, phase, transport k)] by the training-referent rule."""
    import b1_bind as B
    import b1_render as BR
    ev = []
    for g, r in BR.events(actions[:, 6]):
        for kind, fr in (("grasp", g), ("release", r)):
            if fr is not None:
                ev.append(dict(kind=kind, frame=int(fr), eef=[float(x) for x in state[fr][:3]]))
    out = []
    T = len(actions)
    for k, (gr, re_) in enumerate(B.transports(ev)[:2]):
        for ph, e in (("grasp", gr), ("release", re_)):
            if e is None:
                continue
            for o in OFFS:
                f = int(np.clip(e["frame"] - o, 0, T - 1))
                if all(f != x[0] for x in out):
                    out.append((f, ph, k))
    if not out:                                   # no gripper event (e.g. a drawer pushed shut): fixed phase fractions
        for q in FALLBACK:
            f = int(round(q * (T - 1)))
            if all(f != x[0] for x in out):
                out.append((f, "none", -1))
    return out


def chunk(actions, f):
    j = np.arange(f, f + H)
    pad = j >= len(actions)
    return actions[np.minimum(j, len(actions) - 1)], pad


def main():
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    sp = l90c.load_split()
    tasks = {t.tid: t for t in F.load(("libero_90",))}
    d90 = D.Demos90()
    A, W, S, ACT, PAD, TID, EP, FR, PH, TK, BIN = ([] for _ in range(11))
    summary = {}
    for b in ("0.1", "0.3", "0.5", "part"):
        for tid in sp["dev"][b]:
            l90c.check_task(tid)
            n_f, n_d = 0, 0
            for e in l90c.task_episodes(tid)[:10]:
                ep = d90.load(e)
                img, wr = d90.frames(e)
                assert len(img) == len(ep["actions"]) == len(wr), (tid, e)
                fl = frames_of(ep["actions"], ep["state"])
                for f, ph, k in fl:
                    a, p = chunk(ep["actions"], f)
                    A.append(pipe(img[f])), W.append(pipe(wr[f])), S.append(ep["state"][f]), ACT.append(a), PAD.append(p)
                    TID.append(tid), EP.append(int(e)), FR.append(f), PH.append(ph), TK.append(k), BIN.append(b)
                n_f += len(fl)
                n_d += 1
            summary[tid] = dict(bin=b, language=tasks[tid].language, demos=n_d, frames=n_f)
            print(f"[d1_frames] {tid} {n_d} demos {n_f} frames", flush=True)
    np.savez(os.path.join(CACHE, "d1_dev.npz"), agent=np.stack(A), wrist=np.stack(W), state=np.stack(S).astype(np.float32),
             actions=np.stack(ACT).astype(np.float32), pad=np.stack(PAD), tid=np.array(TID), ep=np.array(EP), frame=np.array(FR),
             phase=np.array(PH), transport=np.array(TK), bin=np.array(BIN))
    # pipeline parity on 2 held-out TRAINING episodes of LIBERO-90 cells
    meta = json.load(open(os.path.join(CACHE, "meta.json"), encoding="utf-8"))
    z = np.load(os.path.join(CACHE, "frames.npz"))
    held = [e for e in meta["episodes"] if e["held_out"] and e["src"] == "ipec"][:2]
    par = []
    fa = np.memmap(os.path.join(CACHE, "agent.bin"), dtype=np.uint8, mode="r")
    from PIL import Image
    for e in held:
        idx = np.flatnonzero(z["ep"] == e["ep"])
        img, _ = d90.frames(e["src_ep"])
        for f in (0, len(idx) // 2):
            i = idx[f]
            c = np.asarray(Image.open(io.BytesIO(fa[z["agent_off"][i]:z["agent_off"][i + 1]].tobytes())).convert("RGB")).astype(float)
            par.append(dict(tid=e["tid"], src_ep=e["src_ep"], frame=int(f), mean_abs_px=float(np.abs(pipe(img[f]).astype(float) - c).mean())))
    out = dict(n_frames=len(TID), n_demos=int(sum(v["demos"] for v in summary.values())), per_task=summary, offsets=OFFS,
               pipeline_parity=par)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "d1_frames.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(dict(n_frames=out["n_frames"], n_demos=out["n_demos"], parity=par), indent=1))


if __name__ == "__main__":
    main()

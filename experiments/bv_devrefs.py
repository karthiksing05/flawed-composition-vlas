"""PLAN_BINDVLA 11, offline dev slot-hit diagnostic, DATA ONLY (no model is scored here): contact-event referents of the DEV tasks'
own demos, built exactly as b1_demos.py builds them for training cells (frame 0 of the recorded demo as stored, resize_with_pad 224;
grasp / release frames from the recorded gripper command; the eef position (state[:3]) at each event projected into frame 0 with
the agentview camera).  The demos are SCORED ONLY: never trained on, never executed.  Chunks env, CPU.

Camera: dev cells were never rendered, so K / E are borrowed from a rendered TRAINING cell (results/bind/b1/demo) of the SAME
scene; if the scene has none, from the same arena family (KITCHEN / LIVING_ROOM / STUDY) when every rendered cell of that family
shares one K / E (checked and recorded); otherwise the task is skipped and listed.  Every dev tid passes l90c.check_task
(sealed tasks are refused; split_sealed.json is never opened).
-> CACHE/devrefs.npz (img [n, 224, 224, 3], tid, src_ep, events json), results/bindvla/devrefs.json (summary, camera sources).

    python experiments/bv_devrefs.py [--max-eps 10]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments"), "/coc/flash10/ksingara3/robot/openpi/packages/openpi-client/src"]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"


def family(scene):
    return scene.split("_SCENE")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eps", type=int, default=10)
    a = ap.parse_args()
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy import where as W
    from openpi_client import image_tools
    import b1_demos as BD
    import b1_render as BR
    sp = l90c.load_split()
    tasks = {t.tid: t for t in F.load(("libero_90",))}
    cams = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", "b1", "demo", "libero_90_*.npz"))):
        z = np.load(f, allow_pickle=True)
        tid = str(z["tid"])
        cams.setdefault(tasks[tid].scene, []).append((tid, np.asarray(z["K"]), np.asarray(z["E"])))
    fam = {}
    for sc, lst in cams.items():
        for tid, K, E in lst:
            fam.setdefault(family(sc), []).append((tid, K, E))
    fam_ok = {k: all(np.allclose(K, v[0][1], atol=1e-6) and np.allclose(E, v[0][2], atol=1e-6) for _, K, E in v) for k, v in fam.items()}
    scene_ok = {sc: all(np.allclose(K, v[0][1], atol=1e-6) and np.allclose(E, v[0][2], atol=1e-6) for _, K, E in v) for sc, v in cams.items()}
    d90 = D.Demos90()
    imgs, tids, srcs, evs, bins, summary = [], [], [], [], [], {}
    for b in ("0.1", "0.3", "0.5", "part"):
        for tid in sp["dev"][b]:
            l90c.check_task(tid)
            t = tasks[tid]
            if t.scene in cams and scene_ok[t.scene]:
                src_cam, K, E = "scene:" + cams[t.scene][0][0], cams[t.scene][0][1], cams[t.scene][0][2]
            elif family(t.scene) in fam and fam_ok[family(t.scene)]:
                src_cam, K, E = "family:" + fam[family(t.scene)][0][0], fam[family(t.scene)][0][1], fam[family(t.scene)][0][2]
            else:
                summary[tid] = dict(bin=b, skipped="no rendered training cell with this camera")
                continue
            n = 0
            for e in l90c.task_episodes(tid)[: a.max_eps]:
                try:
                    ep = d90.load(e)
                    rel = d90.info["video_path"].format(episode_chunk=e // d90.chunks_size, video_key="observation.images.image", episode_index=e)
                    im = BD.first_frame_mp4(os.path.join(d90.root, rel))
                except Exception as ex:                                 # noqa: BLE001
                    summary.setdefault("_errors", []).append(f"{tid} ep {e}: {type(ex).__name__}: {str(ex)[:160]}")
                    continue
                ev = []
                for g, r in BR.events(ep["actions"][:, 6]):
                    for kind, fr in (("grasp", g), ("release", r)):
                        if fr is None:
                            continue
                        rc = W.project(np.asarray(ep["state"][fr][:3], np.float64), K, E)
                        tok = int(np.clip(rc[0] // 16, 0, 15)) * 16 + int(np.clip(rc[1] // 16, 0, 15))
                        ev.append(dict(kind=kind, frame=int(fr), token=tok))
                imgs.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(im, 224, 224)))
                tids.append(tid), srcs.append(int(e)), evs.append(ev), bins.append(b)
                n += 1
            summary[tid] = dict(bin=b, scene=t.scene, camera=src_cam, demos=n, language=t.language)
    if not imgs:
        raise SystemExit("no dev demo loaded: " + json.dumps(summary.get("_errors", [])[:5]))
    np.savez_compressed(os.path.join(CACHE, "devrefs.npz"), img=np.stack(imgs), tid=np.array(tids), src_ep=np.array(srcs),
                        bin=np.array(bins), events=json.dumps(evs))
    out = dict(n_demos=len(imgs), per_task=summary, scene_camera_consistent=scene_ok, family_camera_consistent=fam_ok)
    os.makedirs(os.path.join(ROOT, "results", "bindvla"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "bindvla", "devrefs.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(n_demos=len(imgs), family_camera_consistent=fam_ok,
                          skipped=[k for k, v in summary.items() if "skipped" in v],
                          cams={k: v.get("camera") for k, v in summary.items()}), indent=1))


if __name__ == "__main__":
    main()

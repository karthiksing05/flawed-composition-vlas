"""RAA-F Route B training data (docs/notes/raaf_plan.md section 4): sc_rv.py's raac arm (RAA-c / RAA-sL data: rview2 painted views, role-phrase
prompts, the 4-dim state token [dx, dy, dz, role]) PLUS the target-centred crop as a fourth feature `crop_image` (224 x 224 x 3), which
becomes pi0.5's third image slot right_wrist_0_rgb (docs/notes/raafb_build.md).  sc_rv.py is imported, not edited.  openpi venv, CPU.

Crop (raafb_crop): CROP = 96 px square of the PAINTED 256 agentview, centred on the ground-truth world position of the active role's slot
(the object before the hold, the target while held: the same slot and phase as the token), projected with the task's robosuite K / E
(camKE.npz from raafb_proj.py), plus Gaussian jitter N(0, --jitter_px) on the centre (the test estimator's error, about 2 px); resized to 224.
Routed means K1-K6; K7 / K8 carry no channel and get an all-zero crop (the server's no-channel crop).  The state token, prompts, images and
actions are byte-identical to sc_rv's raac dataset (the same functions).  Norm stats / assets dir pi05_g8v_raac are reused (plan section 1),
so there is no norm step.

    python experiments/raafb_rv.py --task K1+K4+K6 --ndemo 1 --suffix _btest --out DIR/ds --cam DIR/camKE.npz --png_dir DIR/png
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sc_rv as RV                                                          # noqa: E402
import raafb_crop as CR                                                     # noqa: E402

REPO = "lol/libero_g8v_raacb"


def save_pngs(png_dir, tag, i, img, crop, uv_true, uv_used):
    from PIL import Image
    os.makedirs(png_dir, exist_ok=True)
    Image.fromarray(np.asarray(crop)).save(os.path.join(png_dir, "%s_f%03d_crop.png" % (tag, i)))
    full = np.array(img, copy=True)
    r, c = int(round(uv_true[0])), int(round(uv_true[1]))
    for d in range(-6, 7):                                                  # green cross at the projected ground truth, white box = the crop window
        for rr, cc in ((r + d, c), (r, c + d)):
            if 0 <= rr < 256 and 0 <= cc < 256:
                full[rr, cc] = (0, 255, 0)
    r0, c0 = int(round(uv_used[0])) - CR.CROP // 2, int(round(uv_used[1])) - CR.CROP // 2
    for k in range(CR.CROP):
        for rr, cc in ((r0, c0 + k), (r0 + CR.CROP - 1, c0 + k), (r0 + k, c0), (r0 + k, c0 + CR.CROP - 1)):
            if 0 <= rr < 256 and 0 <= cc < 256:
                full[rr, cc] = (255, 255, 255)
    both = np.concatenate([full, np.asarray(Image.fromarray(np.asarray(crop)).resize((256, 256)))], axis=1)
    Image.fromarray(both).save(os.path.join(png_dir, "%s_f%03d_full_crop.png" % (tag, i)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="", help="K1..K8, '+'-joined (default: all; a partial set needs --suffix)")
    ap.add_argument("--ndemo", type=int, default=0)
    ap.add_argument("--root", default=RV.RV_ROOT)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="", help="LeRobot dataset root (default: <HF_LEROBOT_HOME>/lol/libero_g8v_raacb<suffix>)")
    ap.add_argument("--cam", required=True, help="camKE.npz from raafb_proj.py (K_<task>, E_<task>)")
    ap.add_argument("--jitter_px", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=93)
    ap.add_argument("--png_dir", default="")
    ap.add_argument("--png_n", type=int, default=6, help="PNG frames per demo, evenly spaced")
    a = ap.parse_args()
    os.environ["HF_LEROBOT_HOME"] = RV.LEROBOT_HOME
    RV._openpi()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    T16 = RV.t16()
    ka = argparse.Namespace(root=a.root, suffix="", arm="raac", task=a.task)        # lex(): the full lexicon, as raac
    ka_raw = argparse.Namespace(root=a.root, suffix="", arm="raa", task=a.task)      # demo_rel without raao's held-object offset: slot - eef
    X = RV.lex(ka)
    by_c, _ = RV.roles_of()
    KE = np.load(a.cam)
    tasks = [t for t in RV.KNOWN if not a.task or t in a.task.split("+")]
    if not a.suffix:
        assert tasks == list(RV.KNOWN), "a partial task set needs --suffix"
    repo = REPO + a.suffix
    root = pathlib.Path(a.out) if a.out else pathlib.Path(RV.LEROBOT_HOME) / repo
    man_path = os.path.join(os.path.dirname(a.out) if a.out else RV.OUT, "raafb_rv%s_manifest.json" % a.suffix)
    assert not (root.exists() and os.path.exists(man_path)), "%s exists and is complete (%s)" % (root, man_path)
    if root.exists():
        print("[rvb] removing incomplete %s" % root, flush=True)
        shutil.rmtree(root)
    files = {c: RV.demo_files(a.root, c)[:(a.ndemo or None)] for c in tasks}
    print("[rvb] convert raac + crop -> %s (%s); demos per task %s; jitter %.1f px" % (repo, root, {c: len(v) for c, v in files.items()}, a.jitter_px),
          flush=True)
    img_ft = lambda: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}          # noqa: E731
    features = {"image": img_ft(), "wrist_image": img_ft(),
                "crop_image": {"dtype": "image", "shape": (CR.OUT, CR.OUT, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (RV.CHAN_DIM,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=repo, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8,
                               image_writer_processes=4)
    chan_info = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
    stats, per_task, k = {}, {}, 0
    for c in tasks:
        K, E = KE["K_" + c], KE["E_" + c]
        for f in files[c]:
            z = dict(np.load(f))
            assert str(z["task"]) == c
            j = int(z["demo"])
            routed = T16.given(str(z["language"])) is not None
            n = len(z["actions"])
            if routed:
                rel = RV.demo_rel(z, c, f, ka, X, T16, by_c, chan_info)
                raw = RV.demo_rel(z, c, f, ka_raw, X, T16, by_c, dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0))
                ee = np.asarray(np.load(RV.chan_path(a.root, c, j))["ee"], np.float64)
                p_obj, p_tgt = raw[:, 0] + ee, raw[:, 1] + ee               # the slots' world positions (token = these minus eef (minus the in-hand offset))
            else:
                rel = np.zeros((n, 2, 3))
            rng = np.random.default_rng(a.seed * 100000 + 1000 * int(c[1:]) + j)
            pick = set(np.unique(np.linspace(0, n - 1, a.png_n).astype(int)).tolist()) if a.png_dir else set()
            for i, (img, wimg, task, h, na, nw) in enumerate(RV.frames_of(z, c, "raac", X, T16, by_c, rel)):
                if routed:
                    uv0 = CR.project_client(p_tgt[i] if h else p_obj[i], K, E)
                    uv = uv0 + rng.normal(0.0, a.jitter_px, 2) if a.jitter_px > 0 else uv0
                    crop = CR.make_crop(img, uv)
                    obj, tgt = by_c[c]
                    act = tgt if h else obj
                    ei = RV.slot_index(z["names"], z["slot_names"], z["slot_ents"], act)
                    m = RV.clean_mask(np.asarray(z["seg"][i]) == ei) if ei is not None else None
                    s = stats.setdefault("%s|%s" % (c, "held" if h else "pre"), dict(n=0, jit=[], mask_err=[], movable=act in MOV))
                    s["n"] += 1
                    s["jit"].append(float(np.linalg.norm(uv - uv0)))
                    if m is not None and m.sum() >= 20:
                        ys, xs = np.nonzero(m)
                        s["mask_err"].append(float(np.linalg.norm(uv0 - np.array([ys.mean(), xs.mean()]))))
                    if i in pick:
                        save_pngs(a.png_dir, "%s_d%02d_%s" % (c, j, "held" if h else "pre"), i, img, crop, uv0, uv)
                else:
                    crop = np.zeros((CR.OUT, CR.OUT, 3), np.uint8)
                ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "crop_image": crop,
                              "state": RV.chan_state(rel[i, int(h)], h, routed), "actions": np.asarray(z["actions"][i], np.float32), "task": task})
            ds.save_episode()
            per_task[c] = per_task.get(c, 0) + 1
            k += 1
            print("[rvb] %s demo %d: %d frames" % (c, j, n), flush=True)
    ds.stop_image_writer()
    summ = {key: dict(n=s["n"], jitter_med_px=float(np.median(s["jit"])), mask_err_med_px=float(np.median(s["mask_err"])) if s["mask_err"] else None,
                      mask_err_p90_px=float(np.percentile(s["mask_err"], 90)) if s["mask_err"] else None, movable_active=bool(s["movable"]))
            for key, s in stats.items()}
    man = dict(repo_id=repo, arm="raac+crop", episodes=k, frames=int(ds.meta.total_frames), per_task=per_task, jitter_px=a.jitter_px, crop=CR.CROP,
               out=CR.OUT, per_phase=summ, channel=chan_info)
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("[rvb] done %s: %d episodes %d frames; %s" % (repo, k, int(ds.meta.total_frames), json.dumps(summ)), flush=True)


MOV = RV.MOVABLE_OBJ

if __name__ == "__main__":
    main()

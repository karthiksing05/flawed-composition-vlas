"""RAA-F Route A dataset (docs/notes/raaf_plan.md 2.1-2.2): lol/libero_g8v_raacf = the RAA-c dataset's episodes (sc_rv.convert's own frames, unchanged) plus,
per eligible K1-K6 demo, M = 2 relabelled copies of the held carry.  openpi venv, CPU (the rendered demos are read; nothing is rendered).  sbatch only.
sc_rv.py is imported, not edited; lol/libero_g8v_raac is never touched.

A copy d' of demo d (rows h0 .. e - 1: the held rows from the hold onset to the open command or the first non-held row): the same images, the same held
role-phrase prompt, the state token chan_state(rel_t + delta, held), actions xyz + K delta (rotation and gripper unchanged: closed on every row of d').
delta, per demo and copy, is read from the registered file (experiments/raaf_geom.py -> results/flawed/raaf_delta.json).

    python experiments/raaf_rv.py convert --delta results/flawed/raaf_delta.json [--ndemo N] [--suffix S]
    python experiments/raaf_rv.py norm    [--suffix S]        (informational: the dataset's own statistics; training keeps RAA-c's, plan section 1)
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sc_rv as RV                                                           # noqa: E402

K = 2.851209548740752
REPO = "lol/libero_g8v_raacf"


def convert(a):
    os.environ["HF_LEROBOT_HOME"] = RV.LEROBOT_HOME
    RV._openpi()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ns = types.SimpleNamespace(suffix="", root=RV.RV_ROOT, arm="raac", ndemo=a.ndemo)
    T16 = RV.t16()
    X = RV.lex(ns)
    by_c, _ = RV.roles_of()
    D = json.load(open(a.delta, encoding="utf-8"))
    assert abs(D["k"] - K) < 1e-12
    cps = {(d["task"], d["demo"]): d for d in D["demos"]}
    repo = REPO + a.suffix
    root = pathlib.Path(RV.LEROBOT_HOME) / repo
    man_path = os.path.join(RV.OUT, "raaf", "raaf_rv_manifest%s.json" % a.suffix)
    assert not (root.exists() and os.path.exists(man_path)), "%s exists and is complete" % root
    if root.exists():
        print("[raaf_rv] removing incomplete %s" % root, flush=True)
        shutil.rmtree(root)
    files = {c: RV.demo_files(ns.root, c)[:(a.ndemo or None)] for c in RV.KNOWN}
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (RV.CHAN_DIM,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=repo, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8, image_writer_processes=4)
    chan_info = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
    n_orig, n_copy, fr_orig, fr_copy, per_task = 0, 0, 0, 0, {}
    for c in RV.KNOWN:
        for f in files[c]:
            z = dict(np.load(f))
            routed = T16.given(str(z["language"])) is not None
            rel = RV.demo_rel(z, c, f, ns, X, T16, by_c, chan_info) if routed else np.zeros((len(z["actions"]), 2, 3))
            frames = list(RV.frames_of(z, c, "raac", X, T16, by_c, rel))
            st = [RV.chan_state(rel[i, int(fr[3])], fr[3], routed) for i, fr in enumerate(frames)]
            for i, (img, wimg, task, h, na, nw) in enumerate(frames):                   # the original episode, exactly sc_rv.convert's rows
                ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "state": st[i], "actions": np.asarray(z["actions"][i], np.float32), "task": task})
            ds.save_episode()
            n_orig += 1
            fr_orig += len(frames)
            per_task[c] = per_task.get(c, 0) + 1
            if not routed:
                continue
            dd = cps.get((c, int(z["demo"])))
            if dd is None:
                continue
            h0, e = dd["h0"], dd["e"]
            for cp in dd["copies"]:
                dl = np.asarray(cp["delta"], np.float64)
                for i in range(h0, e):
                    img, wimg, task, h, na, nw = frames[i]
                    assert h, (c, f, i)
                    act = np.asarray(z["actions"][i], np.float64).copy()
                    act[:3] += K * dl
                    assert np.all(np.abs(act[:3]) <= 0.9375 + 1e-9) and act[6] == z["actions"][i][6], (c, f, i)
                    s = RV.chan_state(rel[i, 1] + dl, True, True)
                    ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "state": s, "actions": act.astype(np.float32), "task": task})
                ds.save_episode()
                n_copy += 1
                fr_copy += e - h0
            if (n_orig % 20) == 0:
                print("[raaf_rv] %d originals, %d copies, at %s %s" % (n_orig, n_copy, c, os.path.basename(f)), flush=True)
    ds.stop_image_writer()
    man = dict(repo_id=repo, episodes_original=n_orig, episodes_copies=n_copy, episodes=n_orig + n_copy, frames_original=fr_orig, frames_copies=fr_copy,
               frames=int(ds.meta.total_frames), per_task_originals=per_task, delta=a.delta, k=K, channel=chan_info)
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("[raaf_rv] done %s: %s" % (repo, json.dumps(man)), flush=True)


def norm(a):
    RV._openpi()
    import pyarrow.parquet as pq
    import openpi.shared.normalize as normalize
    from lifelong_ood.policy import data as DD
    from lifelong_ood.policy import l90c_config as LC
    cfg = LC.train_config("pi05_g8v_raac_lf_ae")
    repo = REPO + a.suffix
    out = pathlib.Path(LC.L90C) / "assets" / ("pi05_g8v_raacf" + a.suffix) / repo           # a NEW assets dir, informational: training keeps pi05_g8v_raac's
    assert not out.exists(), out
    files = sorted((pathlib.Path(RV.LEROBOT_HOME) / repo / "data").rglob("episode_*.parquet"))
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    n = 0
    for f in files:
        t = pq.read_table(f, columns=["state", "actions"]).to_pydict()
        st, ac = np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32)
        idx, _ = DD.chunk_indices(len(ac), cfg.model.action_horizon)
        stats["state"].update(st)
        stats["actions"].update(ac[idx])
        n += len(ac)
    normalize.save(out, {k: s.get_statistics() for k, s in stats.items()})
    ref = json.load(open("/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json", encoding="utf-8"))["norm_stats"]
    new = json.load(open(out / "norm_stats.json", encoding="utf-8"))["norm_stats"]
    cmp = {k: dict(raac_q01=ref[k]["q01"][:4], raacf_q01=new[k]["q01"][:4], raac_q99=ref[k]["q99"][:4], raacf_q99=new[k]["q99"][:4]) for k in ("state", "actions")}
    print("[raaf_rv] norm -> %s over %d frames of %d episodes (informational; training keeps RAA-c's)" % (out, n, len(files)), flush=True)
    print("[raaf_rv] quantiles raac vs raacf: %s" % json.dumps(cmp), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("convert", "norm"))
    ap.add_argument("--delta", default=os.path.join(RV.OUT, "raaf_delta.json"))
    ap.add_argument("--ndemo", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    dict(convert=convert, norm=norm)[a.what](a)

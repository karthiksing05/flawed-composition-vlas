"""T8b (docs/SCAFFOLD_RESULTS.md 30; m3_prereg sc_t8v): the retrained VLA.  pi0.5 from pi05_base on the eight known tasks' demos only.  openpi venv for convert and norm; read in any venv.

    python experiments/sc_t8v.py convert            LeRobot dataset lol/libero_goal8 from pi05_libero's own data: every episode of K1-K8, nothing else
    python experiments/sc_t8v.py norm               norm stats over it
    python experiments/sc_t8v.py read               -> results/flawed/sc_t8v_read.json
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OPENPI = "/coc/flash10/ksingara3/robot/openpi"
LEROBOT_HOME = "/coc/flash10/ksingara3/robot/data/hf/lerobot"
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_t8v")
MANIFEST = os.path.join(OUT, "sc_t8v_manifest.json")
KNOWN = tuple(f"K{i}" for i in range(1, 9))
READ_C = tuple(f"K{i}" for i in range(1, 7)) + tuple(f"N{i}" for i in range(1, 7))
ALL_TIDS = list(range(32))
B, MARGIN = 2000, 0.10


def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def lang():
    return {t["compound"]: t["language"] for t in n2()["tasks"]}


def convert():
    sys.path.insert(0, OPENPI + "/src")
    os.environ["HF_LEROBOT_HOME"] = LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import l90c_config as LC
    L = lang()
    d = D.Demos()
    jobs = []
    for c in KNOWN:
        ti = d.task_of_text[L[c]]
        jobs += [(c, L[c], int(e)) for e in d.episodes(ti)]
    root = pathlib.Path(LEROBOT_HOME) / LC.GOAL8_REPO
    assert not (root / "meta" / "info.json").exists(), f"{root} exists"
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=LC.GOAL8_REPO, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8, image_writer_processes=4)
    t0 = time.time()
    per = {}
    for k, (c, text, e) in enumerate(jobs):
        ep = d.load(e, images=True)
        assert ep["task"] == text, (c, e, ep["task"])
        img, wrist = [D.Demos.decode(x) for x in ep["image"]], [D.Demos.decode(x) for x in ep["wrist_image"]]
        for i in range(len(ep["actions"])):
            ds.add_frame({"image": np.asarray(img[i]), "wrist_image": np.asarray(wrist[i]), "state": np.asarray(ep["state"][i], np.float32), "actions": np.asarray(ep["actions"][i], np.float32), "task": text})
        ds.save_episode()
        per[c] = per.get(c, 0) + 1
        if k % 20 == 0:
            print(f"[t8v] {k + 1}/{len(jobs)} {c} ep {e} {time.time() - t0:.0f}s", flush=True)
    ds.stop_image_writer()
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(dict(repo_id=LC.GOAL8_REPO, episodes=len(jobs), frames=int(ds.meta.total_frames), per_task=per, source="physical-intelligence/libero"), fh, indent=1)
    print(f"[t8v] done: {len(jobs)} episodes, {ds.meta.total_frames} frames, per task {per}", flush=True)


def norm():
    sys.path.insert(0, OPENPI + "/src")
    import pyarrow.parquet as pq
    import openpi.shared.normalize as normalize
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import l90c_config as LC
    cfg = LC.train_config("pi05_goal8")
    files = sorted((pathlib.Path(LEROBOT_HOME) / LC.GOAL8_REPO / "data").rglob("episode_*.parquet"))
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    n = 0
    for f in files:
        t = pq.read_table(f, columns=["state", "actions"]).to_pydict()
        st, ac = np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32)
        idx, _ = D.chunk_indices(len(ac), cfg.model.action_horizon)
        stats["state"].update(st)
        stats["actions"].update(ac[idx])
        n += len(ac)
    ns = {k: s.get_statistics() for k, s in stats.items()}
    out = cfg.assets_dirs / LC.GOAL8_REPO
    normalize.save(out, ns)
    print(f"[t8v] norm stats -> {out} over {n} frames of {len(files)} episodes", flush=True)


def read():
    meta = n2()
    ks = meta["states"]["main"]

    def tid(c, l):
        return next(t["tid"] for t in meta["tasks"] if t["compound"] == c and t["layout"] == l)

    def client(path, t):
        E = json.load(open(path, encoding="utf-8"))["tasks"][str(t)]["episodes"]
        by = {int(e.get("trial", e.get("init_state", -1))): e for e in E}
        return np.array([float(by[k]["success"]) for k in ks])

    def vla(c, l):
        return client(os.path.join(OD, f"pi05g8_main_{tid(c, l)}.json"), tid(c, l))

    def frozen(c, l):
        return client(os.path.join(OUT, "n2_roll", f"pi05_main_{tid(c, l)}.json"), tid(c, l))

    def u(c, l):
        v = []
        for s in (10, 11, 12):
            E = json.load(open(os.path.join(OUT, "sc_t4", f"U_s{s}_{tid(c, l)}.json"), encoding="utf-8"))["eps"]
            by = {int(e["trial"]): e for e in E}
            v.append([float(by[k]["success"]) for k in ks])
        return np.mean(v, 0)

    def diff(a, b):                                                         # [tasks, states]
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[1], a.shape[1])
            v.append(float((a[t][:, s] - b[t][:, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    G = dict(known=READ_C[:6], new_pairings=READ_C[6:])
    res = dict(vla_minus_u={}, vla_minus_frozen={}, per_task={})
    for g, cs in G.items():
        for l in ("familiar", "new"):
            A = np.array([vla(c, l) for c in cs])
            res["vla_minus_u"][f"{g}|{l}"] = diff(A, np.array([u(c, l) for c in cs]))
            res["vla_minus_frozen"][f"{g}|{l}"] = diff(A, np.array([frozen(c, l) for c in cs]))
    for c in list(READ_C) + ["K7", "K8", "N7", "N8"]:
        res["per_task"][c] = {l: float(vla(c, l).mean()) for l in ("familiar", "new")}
    kn, np_ = res["vla_minus_u"]["known|familiar"], res["vla_minus_u"]["new_pairings|familiar"]
    res["underfit"] = bool(kn["ci"][1] < -MARGIN)
    res["label"] = "VLA MATCHES" if np_["ci"][0] >= -MARGIN else ("RUNG HOLDS" if np_["ci"][1] < -MARGIN else "UNRESOLVED")
    if res["underfit"] and res["label"] == "RUNG HOLDS":
        res["label"] = "UNDERFIT"
    print("[t8v]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t8v_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("convert", "norm", "read"))
    a = ap.parse_args()
    dict(convert=convert, norm=norm, read=read)[a.what]()

"""PLAN_BINDVLA section 1 / 5: the BindVLA training cache from lol/libero_l90c (the frozen split's TRAINING tasks only).  CPU.

  frames   every frame of every episode: agentview + wrist image resized 256 -> 224 with openpi_client's resize_with_pad (PIL
           bilinear, what libero_client sends) and stored as JPEG q95; state [8], actions [7], episode, frame index
  episodes per l90c episode: its tid (meta/l90c_done.jsonl line order = episode index; asserted against episodes.jsonl), the text,
           the environment for the invariance penalty (the task's BDDL scene; the task itself for LIBERO-Spatial, whose tasks share a
           scene name but not a layout) and whether it is HELD OUT
  held out 2 episodes per training task, by a fixed hash among the episodes that carry contact-event referents in
           results/bind/b1/demos (b1_bind.training_occurrences: grasp / release tokens on frame 0's 16 x 16 grid); tasks without b1
           referents hold out 2 episodes by the same hash among all of theirs
  stats    MEAN_STD of state and actions over the NON-held-out frames
  diag     the referent records of the held-out episodes (role, phrase, token, step) for the slot-mask diagnostic

Refuses any episode whose tid is not in split.json's training list.  Output: CACHE (outside the repo; about 15 GB) +
results/bindvla/data.json (small summary).

    python experiments/bv_data.py [--workers 16] [--limit-eps N]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
DS = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_l90c"
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
OUT = os.path.join(ROOT, "results", "bindvla")
N_HOLD = 2
RES = 224


def resize(img):
    from PIL import Image
    if img.shape[:2] == (RES, RES):
        return img
    return np.asarray(Image.fromarray(img).resize((RES, RES), resample=Image.BILINEAR))    # square input: resize_with_pad = resize


def jpeg(img):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=95)
    return b.getvalue()


def decode_png(x):
    from PIL import Image
    if isinstance(x, dict):
        x = x["bytes"]
    return np.asarray(Image.open(io.BytesIO(x)).convert("RGB"))


def episode_file(ep):
    return os.path.join(DS, "data", f"chunk-{ep // 1000:03d}", f"episode_{ep:06d}.parquet")


def load_episode(ep):
    import pyarrow.parquet as pq
    t = pq.read_table(episode_file(ep), columns=["image", "wrist_image", "state", "actions", "frame_index", "episode_index"])
    d = t.to_pydict()
    assert all(int(e) == ep for e in d["episode_index"]), ep
    A = [jpeg(resize(decode_png(x))) for x in d["image"]]
    W = [jpeg(resize(decode_png(x))) for x in d["wrist_image"]]
    return dict(ep=ep, agent=A, wrist=W, state=np.asarray(d["state"], np.float32), actions=np.asarray(d["actions"], np.float32),
                fidx=np.asarray(d["frame_index"], np.int64))


def hsh(s):
    return int(hashlib.sha256(("bindvla-holdout" + s).encode()).hexdigest(), 16)


def environment(t):
    return f"{t.suite}:{t.index}" if t.suite == "libero_spatial" else t.scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-eps", type=int, default=0)
    a = ap.parse_args()
    from lifelong_ood import l90c
    from lifelong_ood.policy import factors as F
    split = l90c.load_split()
    train = set(split["train"])
    done = [json.loads(x) for x in open(os.path.join(DS, "meta", "l90c_done.jsonl"), encoding="utf-8")]
    epmeta = [json.loads(x) for x in open(os.path.join(DS, "meta", "episodes.jsonl"), encoding="utf-8")]
    assert len(done) == len(epmeta), (len(done), len(epmeta))
    tasks = {t.tid: t for t in F.load(("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"))}
    manifest = json.load(open(os.path.join(ROOT, "results", "l90c", "data_manifest.json"), encoding="utf-8"))["per_task"]
    eps = []
    for k, (d, m) in enumerate(zip(done, epmeta)):
        assert int(m["episode_index"]) == k
        tid = d["tid"]
        if tid not in train:
            raise SystemExit(f"episode {k}: {tid} is not a training task")
        assert m["tasks"][0] == manifest[tid]["text"], (k, tid)
        eps.append(dict(ep=k, tid=tid, src=d["src"], src_ep=int(d["episode"]), text=m["tasks"][0], length=int(m["length"]),
                        env=environment(tasks[tid])))
    # referents from the b1 demos (no simulator label): occ rows keyed "tid#k", k = index into the npz's episodes
    import b1_bind as B
    occ, _ = B.training_occurrences()
    src_of = {}
    for tid in {o["tid"] for o in occ}:
        z = np.load(os.path.join(B.B1, "demos", tid.replace(":", "_") + ".npz"), allow_pickle=True)
        src_of[tid] = [int(e) for e in z["episodes"]]
    by_key = {(e["tid"], e["src_ep"]): e["ep"] for e in eps}
    diag_all = {}
    for o in occ:
        tid, k = o["frame"].split("#")
        ep = by_key.get((tid, src_of[tid][int(k)]))
        if ep is None:
            continue
        diag_all.setdefault(ep, []).append(dict(role=o["role"], phrase=o["phrase"], token=int(o["token"])))
    hold = set()
    for tid in sorted(train):
        mine = [e["ep"] for e in eps if e["tid"] == tid]
        cand = [x for x in mine if x in diag_all] or mine
        hold |= set(sorted(cand, key=lambda x: hsh(f"{tid}#{x}"))[:N_HOLD])
    for e in eps:
        e["held_out"] = e["ep"] in hold
    todo = [e["ep"] for e in eps][: a.limit_eps or None]
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    from multiprocessing import Pool
    agent_off, wrist_off = [0], [0]
    S, Aa, E, FI = [], [], [], []
    with open(os.path.join(CACHE, "agent.bin.tmp"), "wb") as fa, open(os.path.join(CACHE, "wrist.bin.tmp"), "wb") as fw, \
            Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap(load_episode, todo, chunksize=2)):
            for x in r["agent"]:
                fa.write(x)
                agent_off.append(agent_off[-1] + len(x))
            for x in r["wrist"]:
                fw.write(x)
                wrist_off.append(wrist_off[-1] + len(x))
            S.append(r["state"]), Aa.append(r["actions"]), FI.append(r["fidx"]), E.append(np.full(len(r["fidx"]), r["ep"], np.int64))
            if n % 200 == 0:
                print(f"[bv_data] {n}/{len(todo)} episodes, {agent_off[-1] / 1e9:.2f} GB agent, {time.time() - t0:.0f}s", flush=True)
    os.replace(os.path.join(CACHE, "agent.bin.tmp"), os.path.join(CACHE, "agent.bin"))
    os.replace(os.path.join(CACHE, "wrist.bin.tmp"), os.path.join(CACHE, "wrist.bin"))
    S, Aa, E, FI = map(np.concatenate, (S, Aa, E, FI))
    tr = ~np.isin(E, sorted(hold))
    stats = {k: dict(mean=v[tr].mean(0).tolist(), std=v[tr].std(0).tolist()) for k, v in (("state", S), ("actions", Aa))}
    np.savez(os.path.join(CACHE, "frames.npz"), state=S, actions=Aa, ep=E, fidx=FI, agent_off=np.asarray(agent_off, np.int64),
             wrist_off=np.asarray(wrist_off, np.int64))
    meta = dict(episodes=eps, stats=stats, diag={str(k): v for k, v in diag_all.items() if k in hold}, n_frames=int(len(E)),
                n_train_frames=int(tr.sum()), n_held_out_eps=len(hold), res=RES, source=DS, split_train=sorted(train))
    json.dump(meta, open(os.path.join(CACHE, "meta.json"), "w", encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    envs = sorted({e["env"] for e in eps})
    summ = dict(n_episodes=len(eps), n_frames=int(len(E)), n_train_frames=int(tr.sum()), n_held_out_eps=len(hold),
                n_held_out_with_referents=sum(1 for x in hold if x in diag_all), n_diag_records=sum(len(diag_all[x]) for x in hold if x in diag_all),
                n_envs=len(envs), envs=envs, agent_gb=agent_off[-1] / 1e9, wrist_gb=wrist_off[-1] / 1e9, secs=time.time() - t0, stats=stats)
    json.dump(summ, open(os.path.join(OUT, "data.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("stats", "envs")}), flush=True)


if __name__ == "__main__":
    main()

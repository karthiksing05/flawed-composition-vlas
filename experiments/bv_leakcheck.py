"""PLAN_BINDVLA 11 leakage check (coordinator request, before the phase-2 dev read): which tasks' demos reach main2 / vanilla.

Checks, each recorded with task ids in results/bindvla/leakcheck.json:
  1  every cache episode's tid is a split.json TRAINING task, and none is a dev, part or two-step-test task
  2  every frame carrying a contact-event referent (refs.npz) belongs to a training-task episode; the referent source files
     (results/bind/b1/demos/<tid>.npz) are all training cells
  3  the trainer samples only non-held-out episodes (Store.train_idx): referent frames inside the held-out training episodes are
     counted (they are present in refs.npz but never sampled)
  4  the training runs' own config.json records n_train_frames equal to the cache's non-held-out frame count
  5  scene + role-phrase overlap: for each dev task, the (scene, object phrase) and (scene, goal phrase) pairs that also occur in
     a training task (by design of the MCD split atoms are seen; this lists where a dev phrase is seen in the SAME scene)
Nothing here opens split_sealed.json (l90c.load_split only).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"


def main():
    from lifelong_ood import l90c
    from lifelong_ood.policy import bindvla as BV
    from lifelong_ood.policy import factors as F
    sp = l90c.load_split()
    train = set(sp["train"])
    dev = {t for b in ("0.1", "0.3", "0.5", "part") for t in sp["dev"][b]}
    two = set(sp.get("two_step_test", []))
    meta = json.load(open(os.path.join(CACHE, "meta.json"), encoding="utf-8"))
    eps = meta["episodes"]
    out = {}
    cache_tids = sorted({e["tid"] for e in eps})
    out["1_cache_tasks"] = dict(n=len(cache_tids), all_training=all(t in train for t in cache_tids),
                                in_dev=[t for t in cache_tids if t in dev], in_two_step_test=[t for t in cache_tids if t in two],
                                tids=cache_tids)
    z = np.load(os.path.join(CACHE, "frames.npz"))
    E = z["ep"]
    ref = np.load(os.path.join(CACHE, "refs.npz"))["ref"]
    has = (ref >= 0).any((1, 2))
    tid_of = {e["ep"]: e["tid"] for e in eps}
    held = {e["ep"] for e in eps if e["held_out"]}
    ref_eps = sorted({int(x) for x in np.unique(E[has])})
    ref_tids = sorted({tid_of[x] for x in ref_eps})
    src = sorted(":".join(f[:-4].rsplit("_", 1)) for f in os.listdir(os.path.join(ROOT, "results", "bind", "b1", "demos")) if f.endswith(".npz"))
    out["2_referent_frames"] = dict(n_frames=int(has.sum()), n_episodes=len(ref_eps), tids=ref_tids,
                                    all_training=all(t in train for t in ref_tids), in_dev=[t for t in ref_tids if t in dev],
                                    source_files=src, source_all_training=all(t in train for t in src),
                                    source_in_dev=[t for t in src if t in dev])
    held_frames = int((has & np.isin(E, sorted(held))).sum())
    n_train_frames = int((~np.isin(E, sorted(held))).sum())
    out["3_heldout_episodes"] = dict(n_heldout_episodes=len(held), referent_frames_in_heldout=held_frames,
                                     sampled_by_trainer=False, n_train_frames_cache=n_train_frames)
    runs = {}
    for r in ("main", "main2", "vanilla"):
        f = os.path.join(ROOT, "results", "bindvla", r, "config.json")
        if os.path.exists(f):
            c = json.load(open(f, encoding="utf-8"))
            runs[r] = dict(n_train_frames=c["n_train_frames"], matches_cache=c["n_train_frames"] == n_train_frames,
                           aux=c["args"].get("aux", 0.0), init_base=c["args"].get("init_base", False))
    out["4_runs"] = runs
    tasks = {t.tid: t for t in F.load(("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"))}
    cfg = BV.BVConfig(hier=False)

    def pairs(tid):
        t = tasks[tid]
        ps = set()
        for st in BV.role_parse(t.language, cfg):
            for r in ("obj0", "goal0"):
                if st.get(r):
                    ps.add((t.scene, r[:-1], st[r]))
        return ps
    tr_pairs = set().union(*[pairs(t) for t in train])
    ov = {}
    for t in sorted(dev):
        hit = sorted(f"{s} | {r}: {p}" for s, r, p in pairs(t) if (s, r, p) in tr_pairs)
        if hit:
            ov[t] = hit
    norm = lambda s: " ".join(s.lower().replace("place it", "put it").split())             # noqa: E731
    same_text = {}
    for t in sorted(dev):
        m = [(u, tasks[u].scene) for u in sorted(train) if norm(tasks[u].language) == norm(tasks[t].language)]
        if m:
            same_text[t] = dict(scene=tasks[t].scene, language=tasks[t].language, training=m,
                                same_scene=any(s == tasks[t].scene for _, s in m))
    out["5_scene_phrase_overlap"] = dict(n_dev=len(dev), dev_with_overlap=len(ov), overlaps=ov,
                                         dev_instruction_identical_to_training=same_text,
                                         identical_instruction_in_same_scene=[t for t, v in same_text.items() if v["same_scene"]])
    ok = (out["1_cache_tasks"]["all_training"] and not out["1_cache_tasks"]["in_dev"] and not out["1_cache_tasks"]["in_two_step_test"]
          and out["2_referent_frames"]["all_training"] and out["2_referent_frames"]["source_all_training"]
          and all(v["matches_cache"] for v in runs.values()) and not out["5_scene_phrase_overlap"]["identical_instruction_in_same_scene"])
    out["PASS"] = bool(ok)
    json.dump(out, open(os.path.join(ROOT, "results", "bindvla", "leakcheck.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: (v if k in ("PASS", "3_heldout_episodes", "4_runs") else {kk: vv for kk, vv in v.items() if kk not in ("tids", "source_files", "overlaps")})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()

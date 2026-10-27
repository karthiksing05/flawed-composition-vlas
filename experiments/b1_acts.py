"""PLAN_BINDER B1 (docs/PLAN_BINDER.md section 7): each TRAINING cell's matched demo actions, aligned to its match file's states
(the files S0 / D1 / D4 / D5 use; s0_data.cells' path rule, extended to every training cell with a match file, any goal type)
-> results/bind/b1/acts/<tid>.npy, and results/bind/b1/cells.json.  Chunks env, CPU.  The actions are only read for the gripper
command (contact events); nothing is executed.  Offline only: HF_HUB_OFFLINE=1, a demo file not on disk is skipped and listed.

    HF_HUB_OFFLINE=1 python experiments/b1_acts.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OD = os.path.join(ROOT, "results", "bind", "b1")


def cells():
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(ROOT, "results", "l90c", "split.json"), encoding="utf-8"))
    pp = json.load(open(os.path.join(ROOT, "results", "protos", "index_l90c.json"), encoding="utf-8"))
    eps = {}
    for o in pp["obs"]:
        if o["set"] == "train":
            eps.setdefault(f"{o['suite']}:{o['index']}", (o.get("src"), o.get("hf_task"), o.get("src_hf"), o.get("l90_task"), o["episode"]))
    out, missing = [], []
    for tid in split["train"]:
        if tid not in eps:
            missing.append((tid, "no proto index entry"))
            continue
        src, hf, shf, l90t, ep = eps[tid]
        mp = (os.path.join(ROOT, "results", "robot", "p0_demo", f"{l90t}_match.npz") if src == "ipec"
              else os.path.join(ROOT, "results", "protos", "match", f"tr{(shf if shf is not None else hf):02d}.npz"))
        if not os.path.exists(mp):
            missing.append((tid, "no match file"))
            continue
        out.append(dict(tid=tid, match=os.path.relpath(mp, ROOT), episode=int(ep), ipec=src == "ipec", lang=tasks[tid]["lang"],
                        goal=tasks[tid]["goal"], scene=tasks[tid].get("scene")))
    return out, missing


def main():
    from lifelong_ood.policy import data as D
    os.makedirs(os.path.join(OD, "acts"), exist_ok=True)
    C, missing = cells()
    demos, d90 = D.Demos(), D.Demos90()
    ok = []
    for c in C:
        p = os.path.join(OD, "acts", c["tid"].replace(":", "_") + ".npy")
        s0 = os.path.join(ROOT, "results", "bind", "s0", "acts", c["tid"].replace(":", "_") + ".npy")
        mz = np.load(os.path.join(ROOT, c["match"]), allow_pickle=True)
        n = len(mz["states"])
        try:
            if os.path.exists(s0):
                a, how = np.load(s0), "s0 acts"
            elif c["ipec"] and "actions" in mz.files:
                a, how = np.asarray(mz["actions"], np.float32), "match"
            else:
                a = np.asarray((d90 if c["ipec"] else demos).load(c["episode"])["actions"], np.float32)
                how = f"loader T {len(a)}"
                if len(a) != n:
                    a, how = a[np.asarray(mz["kept"]).astype(int)], how + " [kept]"
        except Exception as ex:                                     # noqa: BLE001
            missing.append((c["tid"], f"actions unavailable offline: {type(ex).__name__}"))
            continue
        if len(a) != n:
            missing.append((c["tid"], f"length mismatch {len(a)} vs {n}"))
            continue
        np.save(p, a)
        ok.append(c)
        print(f"[b1 acts] {c['tid']}: {n} frames ({how})", flush=True)
    json.dump(dict(cells=ok, missing=missing), open(os.path.join(OD, "cells.json"), "w", encoding="utf-8"), indent=1)
    print(f"[b1 acts] {len(ok)} cells; missing {missing}", flush=True)


if __name__ == "__main__":
    main()

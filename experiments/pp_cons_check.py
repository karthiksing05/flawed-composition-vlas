"""PARTS_RESULTS 52 (consolidation): the rtclip leave-one-out is exact.  CPU.

On P_nsz (--tscale --psupport --rtclip) the radial carry clip is never applied (`_canon` takes the scale-frame branch), and the
scale frame takes each target part's own radius rtclip_p; the library-wide scalar rtclip is read only for a target part whose own
radius is 0, i.e. a part with no held demo row in the build.  This script checks, for every P_nsz checkpoint a consolidation read
uses (full / nobottle / noplate builds, seeds 10-12, and the S3 creations k = 1, 5, 10):

  (a) reads:    no task the checkpoint is read on (S2: N2 tids 0-31 on the full build; S3: the condition's tids) has a target part
                whose own radius is 0;
  (b) training: no held demo row of the build's own tasks has such a target part (so --rtclip changes no training row);
  (c) function: on every demo row of the 8 known tasks, with the target re-assigned to each target part read in (a), at flow
                time 0.4 and fixed noise, and again with every object pushed 3x farther from its target (far carries, scale frame
                active), the velocity with rtclip set to 0 equals the velocity as built, bitwise.

    python experiments/pp_cons_check.py --tag nsz --out results/flawed/pp_cons_rtclip_check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_l2_read as L                                                      # noqa: E402
from pp_p_train import rows                                                 # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402

CK = os.path.join(FC.DATA, "ckpt")
S3TIDS = dict(nobottle=(6, 7, 22, 23, 24, 25, 26, 27), noplate=(0, 1, 16, 17, 22, 23))
ATOM = dict(nobottle="wine_bottle_1", noplate="plate_1")


def read_targets(tids):
    n2 = L.n2()
    out = set()
    for t in n2["tasks"]:
        if int(t["tid"]) in tids:
            out.add(FC.roles(t["language"])[1])
    return sorted(out)


def check_one(rel, tids, build_tasks, b, act):
    m, meta = PF.PartsFlow.load(os.path.join(CK, rel))
    assert m.tscale and m.psupport and float(m.rtclip) > 0, ("not a P_nsz-type checkpoint", rel)
    ent = list(FC.ENT)
    zero = [a for i, a in enumerate(PF.TGT_ATOMS) if float(m.rtclip_p[i]) <= 0]
    tg = read_targets(tids)
    reads_ok = not any(t in zero for t in tg)
    # (b) training rows of this build's tasks
    names = list(FC.KNOWN)
    D_task = b["task_idx"]
    own = torch.isin(D_task, torch.tensor([names.index(c) for c in build_tasks]))
    held_t = {ent[int(i)] for i in torch.unique(b["ti"][own & (b["held"] > 0.5)])}
    train_ok = not any(t in zero for t in held_t)
    # (c) function, bitwise
    m0, _ = PF.PartsFlow.load(os.path.join(CK, rel))
    with torch.no_grad():
        m0.rtclip.fill_(0.0)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(act.shape, generator=g)
    worst, n = 0.0, 0
    for tgt in tg:
        bb = {k: v for k, v in b.items() if k != "task_idx"}
        bb["ti"] = torch.full_like(b["ti"], ent.index(tgt))
        for far in (1.0, 3.0):
            bf = dict(bb)
            if far != 1.0:
                idx = torch.arange(len(bf["oi"]))
                pos = bf["pos"].clone()
                off = pos[idx, bf["oi"], :2] - pos[idx, bf["ti"], :2]
                pos[idx, bf["oi"], :2] = pos[idx, bf["ti"], :2] + far * off
                bf["pos"] = pos
                bf["d0t"] = bf["d0t"] * far
            for tv in (0.4,):
                t = torch.full((len(x),), tv)
                with torch.no_grad():
                    v1 = m.velocity(bf, x, t)
                    v0 = m0.velocity(bf, x, t)
                assert torch.isfinite(v1).all()
                worst = max(worst, float((v1 - v0).abs().max()))
                n += 1
    return dict(ckpt=rel, zero_radius_parts=zero, read_targets=tg, reads_ok=reads_ok, held_targets_in_training=sorted(held_t),
                training_ok=train_ok, n_velocity_evals=n, max_abs_velocity_diff=worst, function_exact=bool(worst == 0.0),
                exact=bool(reads_ok and train_ok and worst == 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nsz")
    ap.add_argument("--seeds", default="10+11+12")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    assert not os.path.exists(out), f"{out} exists"
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    b, act, _, _, _, _ = rows(dev)
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    b["task_idx"] = torch.tensor(D["task"].astype(np.int64))
    exc = dict(full=(), nobottle=("K4", "K5"), noplate=("K1",))
    res = []
    for sd in a.seeds.split("+"):
        for cond in ("full", "nobottle", "noplate"):
            bt = [c for c in FC.KNOWN if c not in exc[cond]]
            if cond == "full":
                res.append(check_one(f"pp_full_{a.tag}_s{sd}/P.npz", tuple(range(32)), bt, b, act))
            else:
                for k in (1, 5, 10):
                    res.append(check_one(f"pp_{cond}_{a.tag}_s{sd}/P_{ATOM[cond]}_k{k}.npz", S3TIDS[cond], bt, b, act))
            print("[cons_check]", json.dumps(res[-1]), flush=True)
    rep = dict(what="rtclip leave-one-out on P_nsz is exact (PARTS_RESULTS 52)", checks=res, all_exact=bool(all(r["exact"] for r in res)))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("[cons_check] all_exact", rep["all_exact"], flush=True)


if __name__ == "__main__":
    main()

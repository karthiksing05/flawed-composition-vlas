"""PLAN_PARTS P-build and S3 creation (docs/PARTS_RESULTS.md section 3).  GPU (or CPU).

build   train architecture P (lifelong_ood/policy/parts_flow.py: role backbones + one part per training atom) on the DEMOS ONLY
        (DATA/demos.npz + demos_b.npz; fitted, never executed; no generated carries), every known task except --exclude, with
        prototype A's recipe (rectified flow on 8-step chunks, AdamW 3e-4, batch 512, 30,000 steps, EMA 0.999, seed).
        -> DATA/ckpt/pp_<tag>_s<seed>/P.npz, results/flawed/pp_train_<tag>_s<seed>.json
create  from a built library: re-initialise ONE atom's part (--atom, role --role) and train only that part (its slot embedding and
        its LoRA rows; every other parameter receives a zero gradient and no weight decay, so it is bitwise unchanged, which is
        asserted) on the first --k demos of ONE task (--task), CREATE_STEPS steps, AdamW CREATE_LR (no decay), batch 256, no EMA.
        -> DATA/ckpt/pp_<tag>_s<seed>/P_<atom>_k<k>.npz, results/flawed/pp_create_<tag>_s<seed>_k<k>.json

    python experiments/pp_p_train.py build --tag full --seed 0
    python experiments/pp_p_train.py build --tag nobottle --exclude K4+K5 --seed 0
    python experiments/pp_p_train.py create --tag nobottle --seed 0 --role obj --atom wine_bottle_1 --task K4 --k 5
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
from fa1_train import chunks                                                # noqa: E402
from fa1g_train import rot0_rows                                            # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402

CREATE_STEPS, CREATE_LR = 4000, 3e-4
FIELDS = ("ee_pos", "ee_rot", "grip", "prev", "pos", "yaw", "drawer", "oi", "ti", "held", "off", "zrest", "cyaw", "rot0")


def start_rows(D):
    """PARTS_RESULTS 27 (P_hs): per row, the end effector at its demo's first row (the episode start)."""
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    first = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]
    return D["ee_pos"][first[np.searchsorted(first, np.arange(len(key)), side="right") - 1]]


def rows(dev, tasks=None, k=None):
    """-> (batch dict of tensors, actions, centre, size) over the demo rows of `tasks` (compound names; None = all), first k demos."""
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[c]) for c in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    r0 = rot0_rows(D, Db["held"])
    X = chunks(D, 8)
    center = D["ee_pos"].mean(0).tolist()                           # the demos' centre over all 8 known tasks, as every stage
    names = list(FC.KNOWN)
    keep = np.ones(len(oi), bool)
    if tasks is not None:
        keep &= np.isin(D["task"], [names.index(c) for c in tasks])
    if k is not None:
        keep &= D["demo"] < k
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v)[keep], dtype=dt, device=dev)   # noqa: E731
    b = dict(ee_pos=T(D["ee_pos"]), ee_rot=T(D["ee_rot"]), grip=T(D["grip"]), prev=T(D["prev_grip"]), pos=T(D["pos"]), yaw=T(D["yaw"]),
             drawer=T(D["drawer"]), oi=T(oi, torch.long), ti=T(ti, torch.long), held=T(Db["held"]), off=T(Db["off"]), zrest=T(zr),
             cyaw=T(cy), rot0=T(r0), ee0=T(start_rows(D)))
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    return b, T(np.clip(X, -1, 1)), center, size, int(keep.sum()), sorted({int(x) for x in np.asarray(D["demo"])[keep]})


def fit(m, params, b, act, steps, batch, lr, seed, wd, mask_fn=None, log=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    n = len(act)
    curve, run, t0 = [], [], time.time()
    for s in range(1, steps + 1):
        j = torch.as_tensor(rng.integers(0, n, batch), device=act.device)
        loss = m.loss({k: v[j] for k, v in b.items()}, act[j])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        if mask_fn is not None:
            mask_fn()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if mask_fn is None:                                          # creation keeps no EMA (an EMA of unchanged rows is not bitwise exact)
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), m.parameters()):
                    pe.mul_(0.999).add_(pm, alpha=0.001)
        run.append(float(loss))
        if s % 1000 == 0 or s == steps:
            curve.append(dict(step=s, train=float(np.mean(run))))
            run = []
            print(f"[pp_p] {log} step {s}: {curve[-1]['train']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return (ema if mask_fn is None else m), curve, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "create"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--rank", type=int, default=PF.RANK)
    ap.add_argument("--tsize", action="store_true")                       # PARTS_RESULTS 12: P_sz
    ap.add_argument("--aframe", action="store_true")                      # PARTS_RESULTS 14: P_af
    ap.add_argument("--aframe2", action="store_true")                     # PARTS_RESULTS 16: P_af2
    ap.add_argument("--rclip", action="store_true")                       # PARTS_RESULTS 21: P_rc
    ap.add_argument("--rtclip", action="store_true")                      # PARTS_RESULTS 24: P_rt
    ap.add_argument("--hscale", action="store_true")                      # PARTS_RESULTS 27: P_hs
    ap.add_argument("--role", choices=("obj", "tgt"))
    ap.add_argument("--atom")
    ap.add_argument("--task")
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = os.path.join(FC.DATA, "ckpt", f"pp_{a.tag}_s{a.seed}")
    os.makedirs(d, exist_ok=True)
    if a.mode == "build":
        tasks = [c for c in FC.KNOWN if c not in a.exclude.split("+")]
        b, act, center, size, n, _ = rows(dev, tasks)
        D0 = np.load(os.path.join(FC.DATA, "demos.npz"))
        home = D0["ee_pos"][D0["step"] == 0].mean(0).tolist()               # the demos' mean end effector at step 0 (P_af)
        rclip = hscale = 0.0
        if a.rclip or a.hscale:                                                        # P_rc: min over training objects of the 99th pct of not-held dxy
            nh = b["held"] < 0.5
            idx = torch.arange(len(b["oi"]), device=dev)
            dxy = torch.linalg.norm((b["ee_pos"] - b["pos"][idx, b["oi"]])[:, :2], dim=-1)
            rclip = float(min(torch.quantile(dxy[nh & (b["oi"] == o)], 0.99) for o in torch.unique(b["oi"][nh])))
            if a.hscale:                                                     # P_hs: the same radius, as the scale frame's support
                rclip, hscale = 0.0, rclip
        rtclip = 0.0
        if a.rtclip:                                                         # P_rt: min over training targets of the 99th pct of held object-target dxy
            hd = b["held"] > 0.5
            idx = torch.arange(len(b["oi"]), device=dev)
            txy = torch.linalg.norm((b["pos"][idx, b["oi"]] - b["pos"][idx, b["ti"]])[:, :2], dim=-1)
            rtclip = float(min(torch.quantile(txy[hd & (b["ti"] == t)], 0.99) for t in torch.unique(b["ti"][hd])))
        print(f"[pp_p] rclip {rclip:.4f} rtclip {rtclip:.4f} hscale {hscale:.4f}", flush=True)
        m = PF.PartsFlow(8, center=center, size=size, rank=a.rank, tsize=a.tsize, aframe=2 if a.aframe2 else int(a.aframe), home=home,
                         rclip=rclip, rtclip=rtclip, hscale=hscale).to(dev)
        ema, curve, secs = fit(m, list(m.parameters()), b, act, a.steps, 512, 3e-4, a.seed, 1e-4, log=f"build {a.tag} s{a.seed}")
        ema.cpu().save(os.path.join(d, "P.npz"), dict(tag=a.tag, seed=a.seed, tasks=tasks, center=center, steps=a.steps))
        rep = dict(mode="build", tag=a.tag, seed=a.seed, tasks=tasks, n_rows=n, steps=a.steps, secs=secs, curve=curve,
                   params_total=int(sum(p.numel() for p in m.parameters())), part_params=dict(obj=m.obj.part_params(), tgt=m.tgt.part_params()),
                   backbone_params=dict(obj=int(sum(p.numel() for p in m.obj.backbone_parameters())),
                                        tgt=int(sum(p.numel() for p in m.tgt.backbone_parameters()))))
        fn = os.path.join(ROOT, "results", "flawed", f"pp_train_{a.tag}_s{a.seed}.json")
    else:
        m, meta = PF.PartsFlow.load(os.path.join(d, "P.npz"))
        m = m.to(dev)
        net = m.obj if a.role == "obj" else m.tgt
        j = (PF.OBJ_ATOMS if a.role == "obj" else PF.TGT_ATOMS).index(a.atom)
        g = torch.Generator().manual_seed(1000 + a.seed)
        fresh = PF.RoleNet(net.hidden[0].base.in_features - PF.EMB, net.out.out_features, net.emb.shape[0], rank=net.hidden[0].rank)
        with torch.no_grad():                                        # re-initialise the atom's part (the build never trained it)
            net.emb[j] = torch.randn(PF.EMB, generator=g).to(dev) * 0.1
            for ly, fl in zip(net.hidden, fresh.hidden):
                ly.A[j] = (torch.randn(ly.A.shape[1:], generator=g) / np.sqrt(ly.A.shape[1])).to(dev)
                ly.B[j] = 0.0
        before = {k: v.detach().clone() for k, v in m.state_dict().items()}
        for p in m.parameters():
            p.requires_grad_(False)
        parts = net.part_parameters()
        for p in parts:
            p.requires_grad_(True)

        def mask():                                                  # only row j of the shared part tensors moves
            for p in parts:
                if p.grad is not None:
                    keepj = p.grad[j].clone()
                    p.grad.zero_()
                    p.grad[j] = keepj
        b, act, _, _, n, demos = rows(dev, [a.task], a.k)
        assert len(demos) == a.k, ("demos", demos)
        t0 = time.time()
        ema, curve, secs = fit(m, parts, b, act, CREATE_STEPS, 256, CREATE_LR, a.seed, 0.0, mask_fn=mask, log=f"create {a.atom} k{a.k}")
        after = ema.state_dict()
        changed = []
        for k, v in before.items():
            w = after[k].to(v.device)
            if k.endswith(("emb", ".A", ".B")) and k.startswith("obj." if a.role == "obj" else "tgt."):
                other = torch.ones(v.shape[0], dtype=torch.bool)
                other[j] = False
                if not torch.equal(v[other], w[other]):
                    changed.append(k)
            elif not torch.equal(v, w):
                changed.append(k)
        assert not changed, ("parameters outside the created part changed", changed)
        name = f"P_{a.atom}_k{a.k}.npz"
        ema.cpu().save(os.path.join(d, name), dict(meta, created=a.atom, role=a.role, task=a.task, k=a.k, demos=demos))
        rep = dict(mode="create", tag=a.tag, seed=a.seed, atom=a.atom, role=a.role, task=a.task, k=a.k, demos=demos, n_rows=n,
                   steps=CREATE_STEPS, secs=secs, gpu_seconds_to_ready=time.time() - t0, curve=curve, params_added=net.part_params(),
                   old_parameters_changed=0)
        fn = os.path.join(ROOT, "results", "flawed", f"pp_create_{a.tag}_s{a.seed}_{a.atom}_k{a.k}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[pp_p] wrote {fn}", flush=True)


if __name__ == "__main__":
    main()

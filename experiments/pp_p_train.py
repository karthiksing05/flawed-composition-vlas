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


def d0t_rows(D, held, oi, ti):
    """PARTS_RESULTS 33 (P_ts): per row, the horizontal object-to-target distance at the onset of the current hold (the current
    distance while not held), stepped over each episode's rows in order -- the rollout's own rule (fa1h_roll.py)."""
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    i = np.arange(len(key))
    cur = np.linalg.norm(D["pos"][i, oi, :2] - D["pos"][i, ti, :2], axis=1).astype(np.float32)
    out = cur.copy()
    on, d0 = False, 0.0
    for j in range(len(key)):
        if j == 0 or key[j] != key[j - 1]:
            on = False
        h = bool(held[j] > 0.5)
        if h and not on:
            d0 = cur[j]
        on = h
        if h:
            out[j] = d0
    return out


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
             cyaw=T(cy), rot0=T(r0), ee0=T(start_rows(D)), d0t=T(d0t_rows(D, Db["held"], oi, ti)),
             dkey=T(D["task"].astype(np.int64) * 1000 + D["demo"], torch.long))
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    return b, T(np.clip(X, -1, 1)), center, size, int(keep.sum()), sorted({int(x) for x in np.asarray(D["demo"])[keep]})


def part_radii(b, dev):
    """PARTS_RESULTS 30 (P_pp): each part's own support radius by rounds 5-6's rule, on the rows in b.  Object part: the 99th pct of
    the horizontal hand-to-object distance on its not-held rows; target part: the 99th pct of the horizontal object-to-target
    distance on its held rows.  Parts without rows: 0 (the model then falls back to the library-wide radius)."""
    idx = torch.arange(len(b["oi"]), device=dev)
    nh, hd = b["held"] < 0.5, b["held"] > 0.5
    dxy = torch.linalg.norm((b["ee_pos"] - b["pos"][idx, b["oi"]])[:, :2], dim=-1)
    txy = torch.linalg.norm((b["pos"][idx, b["oi"]] - b["pos"][idx, b["ti"]])[:, :2], dim=-1)
    ent = list(FC.ENT)
    ro, rt = np.zeros(len(PF.OBJ_ATOMS)), np.zeros(len(PF.TGT_ATOMS))
    for i, a in enumerate(PF.OBJ_ATOMS):
        m = nh & (b["oi"] == ent.index(a))
        if bool(m.any()):
            ro[i] = float(torch.quantile(dxy[m], 0.99))
    for i, a in enumerate(PF.TGT_ATOMS):
        m = hd & (b["ti"] == ent.index(a))
        if bool(m.any()):
            rt[i] = float(torch.quantile(txy[m], 0.99))
    return ro, rt


def deterministic():
    """PARTS_RESULTS 35 (decoupled training, part 3): deterministic kernels.  Called before the first cuBLAS call."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def role_init(m, seed):
    """PARTS_RESULTS 35 (decoupled training, part 2): each role network's initial parameters from its own generator, seeded by
    (build seed, role): independent of the other role's architecture and of the construction order."""
    for i, net in enumerate((m.obj, m.tgt)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1_000_003 * (i + 1) + int(seed))
            fresh = PF.RoleNet(net.hidden[0].base.in_features - PF.EMB, net.out.out_features, net.emb.shape[0], rank=net.hidden[0].rank)
        net.load_state_dict(fresh.state_dict())


ROLE_BUFFERS = dict(obj=("rclip", "rclip_p", "home", "hscale"), tgt=("rtclip", "rtclip_p"))
ROLE_FLAGS = dict(obj=("aframe",), tgt=("tsize", "tscale"))


def reuse_role(m, spec, seed, steps):
    """PARTS_RESULTS 35: load ONE role's term (`spec` = "<role>:<tag>") from a decoupled build of the same seed and freeze it.  The
    role's buffers, flags and the shared buffers must equal this build's (asserted).  -> the role reused."""
    role, tag = spec.split(":")
    src, meta = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", f"pp_{tag}_s{seed}", "P.npz"))
    assert meta.get("decouple") and int(meta["seed"]) == int(seed) and int(meta["steps"]) == int(steps), ("donor", meta)
    sd, mine = src.state_dict(), m.state_dict()
    for k in ROLE_BUFFERS[role] + ("center", "size"):
        assert torch.equal(sd[k].cpu(), mine[k].cpu()), ("reuse: buffer differs", k)
    for k in ROLE_FLAGS[role]:
        assert getattr(src, k) == getattr(m, k), ("reuse: flag differs", k)
    net, snet = getattr(m, role), getattr(src, role)
    with torch.no_grad():
        for p, q in zip(net.parameters(), snet.parameters()):
            p.copy_(q.to(p.device))
    for p in net.parameters():
        p.requires_grad_(False)
    return role


KMAX, KAPPA_MAX, EM_ITERS = 4, 1000.0, 300


def grasp_angles(b, X):
    """PARTS_RESULTS 38 (as amended there): per demo with a close row, (its target atom's entity index, the doubled turn of the hand's
    yaw from the demo's first row to its first close row).  A close row: not held, the chunk's first gripper command closes."""
    R = b["ee_rot"].cpu().numpy()
    psi = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    close = ((b["held"] < 0.5) & (X[:, 0, 6] > 0)).cpu().numpy()
    key, ti = b["dkey"].cpu().numpy(), b["ti"].cpu().numpy()
    first = {}
    for i in range(len(key)):                                        # rows are in episode order: the demo's first row
        first.setdefault(int(key[i]), i)
    out = {}
    for i in np.nonzero(close)[0]:                                   # ... and its first close row
        if int(key[i]) not in out:
            out[int(key[i])] = (int(ti[i]), float(np.angle(np.exp(2j * (psi[i] - psi[first[int(key[i])]])))))
    return [v for _, v in sorted(out.items())]


def _vm_logpdf(a, mu, kap):
    from scipy.special import i0e
    return kap * (np.cos(a - mu) - 1.0) - np.log(2 * np.pi * i0e(kap))


def vm_mixture(a, K):
    """EM for a K-component von Mises mixture on angles a (deterministic farthest-point init from the circular mean; kappa <= cap).
    -> (mu, kappa, weight, log-likelihood)."""
    mus = [float(np.angle(np.mean(np.exp(1j * a))))]
    while len(mus) < K:
        d = np.min([np.abs(np.angle(np.exp(1j * (a - u)))) for u in mus], 0)
        mus.append(float(a[int(np.argmax(d))]))
    mu, kap, w = np.array(mus), np.full(K, 10.0), np.full(K, 1.0 / K)
    for _ in range(EM_ITERS):
        lp = np.log(w)[None] + np.stack([_vm_logpdf(a, mu[k], kap[k]) for k in range(K)], 1)
        mx = lp.max(1, keepdims=True)
        r = np.exp(lp - mx)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        w = nk / nk.sum()
        z = (r * np.exp(1j * a)[:, None]).sum(0) / nk
        mu = np.angle(z)
        Rb = np.clip(np.abs(z), 1e-6, 1 - 1e-9)
        kap = np.minimum(Rb * (2 - Rb ** 2) / (1 - Rb ** 2), KAPPA_MAX)        # Banerjee et al.'s approximation
    lp = np.log(w)[None] + np.stack([_vm_logpdf(a, mu[k], kap[k]) for k in range(K)], 1)
    mx = lp.max(1, keepdims=True)
    ll = float((mx[:, 0] + np.log(np.exp(lp - mx).sum(1))).sum())
    return mu, kap, w, ll


def labels(a, gm):
    mu, kap, w = gm
    lp = np.log(np.maximum(w, 1e-300))[None] + np.stack([_vm_logpdf(a, mu[k], kap[k]) for k in range(len(mu))], 1)
    return lp.argmax(1)


def grasp_inventory(b, X):
    """PARTS_RESULTS 38: the grasp-mode inventory by the fixed rule (K in 1..4 by BIC, ties to the smaller K; components ordered by
    weight) and each target part's value of m (most frequent label of its own demos; ties to the lower value; no demos: 0)."""
    g = grasp_angles(b, X)
    a = np.array([x for _, x in g])
    tis = np.array([t for t, _ in g])
    fits = {}
    for K in range(1, KMAX + 1):
        mu, kap, w, ll = vm_mixture(a, K)
        fits[K] = dict(mu=mu, kappa=kap, weight=w, ll=ll, bic=-2 * ll + (3 * K - 1) * np.log(len(a)))
    K = min(fits, key=lambda k: (round(fits[k]["bic"], 9), k))
    o = np.argsort(-fits[K]["weight"], kind="stable")
    gm = np.stack([fits[K]["mu"][o], fits[K]["kappa"][o], fits[K]["weight"][o]])
    lab = labels(a, gm)
    ent = list(FC.ENT)
    tmode = np.zeros(len(PF.TGT_ATOMS), np.int64)
    for i, atom in enumerate(PF.TGT_ATOMS):
        mine = lab[tis == ent.index(atom)]
        if len(mine):
            tmode[i] = int(np.argmax(np.bincount(mine, minlength=K)))
    rep = dict(K=K, n_demos=len(a), bic={k: float(v["bic"]) for k, v in fits.items()},
               mu_deg=(np.degrees(gm[0]) / 2).tolist(), kappa=gm[1].tolist(), weight=gm[2].tolist(),
               tmode={a_: int(tmode[i]) for i, a_ in enumerate(PF.TGT_ATOMS)}, bits_capacity=float(np.log2(K)))
    return K, tmode, gm, rep


def task_entropy_bits(tasks, tmode):
    """Entropy of m over the build's known tasks, tasks weighted equally."""
    import fa1_common as F
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    ms = [int(tmode[PF.TGT_ATOMS.index(F.roles(lang[c])[1])]) for c in tasks]
    p = np.bincount(ms) / len(ms)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def fit(m, params, b, act, steps, batch, lr, seed, wd, mask_fn=None, log="", groups=None):
    """groups (PARTS_RESULTS 35, decoupled training, part 1): parameter lists clipped separately (one per role); None = one joint
    clip over `params`, as every build before round 10.  The EMA updates only parameters that train."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    if groups is not None:
        ids = sorted(id(p) for g in groups for p in g)
        assert ids == sorted(id(p) for p in params) and len(set(ids)) == len(ids), "clip groups must partition the trained parameters"
    n = len(act)
    curve, run, t0 = [], [], time.time()
    for s in range(1, steps + 1):
        j = torch.as_tensor(rng.integers(0, n, batch), device=act.device)
        loss = m.loss({k: v[j] for k, v in b.items()}, act[j])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        if mask_fn is not None:
            mask_fn()
        for g in (groups if groups is not None else [params]):
            torch.nn.utils.clip_grad_norm_(g, 1.0)
        opt.step()
        sched.step()
        if mask_fn is None:                                          # creation keeps no EMA (an EMA of unchanged rows is not bitwise exact)
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), m.parameters()):
                    if pm.requires_grad:                             # a reused (frozen) role keeps its loaded values exactly
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
    ap.add_argument("--psupport", action="store_true")                    # PARTS_RESULTS 30: P_pp (per-part support radii)
    ap.add_argument("--tscale", action="store_true")                      # PARTS_RESULTS 33: P_ts (carry scale frame)
    ap.add_argument("--decouple", action="store_true")                    # PARTS_RESULTS 35: one training per role
    ap.add_argument("--reuse", default="")                                # PARTS_RESULTS 35: "<role>:<tag>", needs --decouple
    ap.add_argument("--imode", action="store_true")                       # PARTS_RESULTS 38: P_imd (grasp-mode message m)
    ap.add_argument("--role", choices=("obj", "tgt"))
    ap.add_argument("--atom")
    ap.add_argument("--task")
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    assert not a.reuse or a.decouple, "--reuse needs --decouple"
    if a.decouple:
        deterministic()
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
        ro, rt = part_radii(b, dev) if a.psupport else (None, None)
        print(f"[pp_p] rclip {rclip:.4f} rtclip {rtclip:.4f} hscale {hscale:.4f} psupport {a.psupport} "
              f"obj {None if ro is None else [round(x, 4) for x in ro]} tgt {None if rt is None else [round(x, 4) for x in rt]}", flush=True)
        nmode, tmode, gm, inv = 0, None, None, None
        if a.imode:                                                        # PARTS_RESULTS 38: the grasp-mode inventory and each target's m
            nmode, tmode, gm, inv = grasp_inventory(b, act)
            inv["bits_task_entropy"] = task_entropy_bits(tasks, tmode)
            print("[pp_p] interface", json.dumps(inv), flush=True)
        m = PF.PartsFlow(8, center=center, size=size, rank=a.rank, tsize=a.tsize, aframe=2 if a.aframe2 else int(a.aframe), home=home,
                         rclip=rclip, rtclip=rtclip, hscale=hscale, psupport=a.psupport, rclip_p=ro, rtclip_p=rt, tscale=a.tscale,
                         nmode=nmode, tmode=tmode, gm=gm)
        groups, reused = None, None
        if a.decouple:                                                     # PARTS_RESULTS 35
            hv = torch.unique(b["held"]).cpu().tolist()
            assert set(hv) <= {0.0, 1.0}, ("held must be 0/1 for the loss to split by role", hv)
            role_init(m, a.seed)
        m = m.to(dev)
        if a.reuse:
            reused = reuse_role(m, a.reuse, a.seed, a.steps)
        if a.decouple:
            groups = [[p for p in net.parameters() if p.requires_grad] for net in (m.obj, m.tgt)]
            groups = [g for g in groups if g]
        params = [p for p in m.parameters() if p.requires_grad]
        ema, curve, secs = fit(m, params, b, act, a.steps, 512, 3e-4, a.seed, 1e-4, log=f"build {a.tag} s{a.seed}", groups=groups)
        ema.cpu().save(os.path.join(d, "P.npz"), dict(tag=a.tag, seed=a.seed, tasks=tasks, center=center, steps=a.steps,
                                                      decouple=a.decouple, reuse=a.reuse or None))
        rep = dict(mode="build", tag=a.tag, seed=a.seed, tasks=tasks, n_rows=n, steps=a.steps, secs=secs, curve=curve,
                   decouple=a.decouple, reuse=a.reuse or None, reused_role=reused, interface=inv,
                   rclip=rclip, rtclip=rtclip, hscale=hscale, rclip_p=None if ro is None else list(ro), rtclip_p=None if rt is None else list(rt),
                   params_total=int(sum(p.numel() for p in m.parameters())), part_params=dict(obj=m.obj.part_params(), tgt=m.tgt.part_params()),
                   backbone_params=dict(obj=int(sum(p.numel() for p in m.obj.backbone_parameters())),
                                        tgt=int(sum(p.numel() for p in m.tgt.backbone_parameters()))))
        fn = os.path.join(ROOT, "results", "flawed", f"pp_train_{a.tag}_s{a.seed}.json")
    else:
        m, meta = PF.PartsFlow.load(os.path.join(d, "P.npz"))
        if meta.get("decouple") and not a.decouple:                  # PARTS_RESULTS 35: a decoupled library's parts train deterministically
            deterministic()
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
        radius = None
        if m.psupport:                                               # P_pp: the created part's own radius, from its k demos only
            ro, rt = part_radii(b, dev)
            radius = float((ro if a.role == "obj" else rt)[j])
            assert radius > 0, "no creation rows for the atom's radius"
            with torch.no_grad():
                (m.rclip_p if a.role == "obj" else m.rtclip_p)[j] = radius
            print(f"[pp_p] created part radius {radius:.4f}", flush=True)
        cmode = None
        if m.nmode > 0 and a.role == "tgt":                          # P_imd: the created target part's m, from its own k demos
            g = grasp_angles(b, act)
            assert g, "no close row in the creation demos"
            lab = labels(np.array([x for _, x in g]), m.gm.cpu().numpy())
            cmode = int(np.argmax(np.bincount(lab, minlength=m.nmode)))
            with torch.no_grad():
                m.tmode[j] = cmode
            print(f"[pp_p] created part m {cmode} (labels {lab.tolist()})", flush=True)
        t0 = time.time()
        ema, curve, secs = fit(m, parts, b, act, CREATE_STEPS, 256, CREATE_LR, a.seed, 0.0, mask_fn=mask, log=f"create {a.atom} k{a.k}")
        after = ema.state_dict()
        changed = []
        for k, v in before.items():
            w = after[k].to(v.device)
            own = k == ("rclip_p" if a.role == "obj" else "rtclip_p") and radius is not None     # P_pp: the part's radius row
            own = own or (k == "tmode" and cmode is not None)                                  # P_imd: the part's m
            if own or (k.endswith(("emb", ".A", ".B")) and k.startswith("obj." if a.role == "obj" else "tgt.")):
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
                   old_parameters_changed=0, part_radius=radius, part_mode=cmode)
        fn = os.path.join(ROOT, "results", "flawed", f"pp_create_{a.tag}_s{a.seed}_{a.atom}_k{a.k}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[pp_p] wrote {fn}", flush=True)


if __name__ == "__main__":
    main()

"""The demo-labelled locator (docs/LOCATOR.md; m3_prereg pp_loc).

    frames        (LIBERO venv, EGL) set and render demo states and development states -> DATA/loc_frames.npz.  No demo action is executed.
    train ARM S   (openpi venv, GPU) ARM C or A, seed S -> DATA/ckpt/pp_loc_<ARM>_s<S>.pt
    read          (openpi venv, GPU) -> results/flawed/pp_loc.json

LOC_SMOKE=1 uses 2 demos per task, 2 states per layout and 40 steps, writes _smoke files, and prints no error on a development state.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fa1_common as FC                                                     # noqa: E402

SMOKE = os.environ.get("LOC_SMOKE") == "1"
SFX = "_smoke" if SMOKE else ""
OUT = os.path.join(FC.ROOT, "results", "flawed")
FRAMES = os.path.join(FC.DATA, f"loc_frames{SFX}.npz")
H = 256
STRIDE, N_TRAIN_DEMO = 5, 40
TASKS = ("K1", "K2", "K3", "K4", "K5", "K6")
HEADS = (("cream_cheese_1", "obj", ("K6",)), ("wine_bottle_1", "obj", ("K4", "K5")), ("akita_black_bowl_1", "obj", ("K1", "K2", "K3")),
         ("plate_1", "tgt", ("K1",)), ("akita_black_bowl_1", "tgt", ("K6",)))
NAMES = ("cheese_obj", "bottle_obj", "bowl_obj", "plate_tgt", "bowl_tgt")
STATES = range(4, 6) if SMOKE else range(4, 24)
BUDGET = 0.02


def camera(sim, cam="agentview"):
    """pp_f1_frames.project's matrices: -> K [3, 3], R [4, 4] (camera to world, with its axis flip)."""
    cid = sim.model.camera_name2id(cam)
    f = 0.5 * H / np.tan(sim.model.cam_fovy[cid] * np.pi / 360)
    K = np.array([[f, 0, H / 2], [0, f, H / 2], [0, 0, 1]])
    R = np.eye(4)
    R[:3, :3] = np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)
    R[:3, 3] = np.asarray(sim.data.cam_xpos[cid])
    return K, R @ np.diag([1.0, -1.0, -1.0, 1.0])


def project(K, R, pts):
    """world [N, 3] -> (row, col) [N, 2] in the upright image."""
    Ke = np.eye(4)
    Ke[:3, :3] = K
    x = Ke @ np.linalg.inv(R) @ np.concatenate([pts, np.ones((len(pts), 1))], 1).T
    x = (x[:2] / x[2:3]).T
    return np.stack([x[:, 1], x[:, 0]], -1)


def lift(K, R, rc, z):
    """(row, col) [N, 2] and a plane height [N] -> world [N, 3]."""
    d = np.linalg.inv(K) @ np.stack([rc[:, 1], rc[:, 0], np.ones(len(rc))])
    w = (R[:3, :3] @ d).T
    t = (np.asarray(z) - R[2, 3]) / w[:, 2]
    return R[:3, 3][None] + t[:, None] * w


def events(D, ti, dm):
    """-> (rows of this demo, close index, release index), from proprioception and commands only; None without a full hold."""
    r = np.flatnonzero((D["task"] == ti) & (D["demo"] == dm))
    h = np.flatnonzero(D["_held"][r])
    if not len(h):
        return None
    a6 = D["action"][r, 6]
    c = int(h[0])
    while c > 0 and a6[c - 1] > 0:
        c -= 1
    op = np.flatnonzero(a6[h[0]:] <= 0)
    if not len(op):
        return None
    return r, c, int(h[0] + op[0])


def frames():
    from libero.libero.envs import OffScreenRenderEnv
    from lifelong_ood.policy import parts_flow as PF
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    Z = np.load(os.path.join(FC.DATA, "demos.npz"))
    D = {k: Z[k] for k in ("task", "demo", "step", "action", "ee_pos", "pos", "grip", "prev_grip")}
    tidx = [list(FC.KNOWN).index(k) for k in TASKS]
    keep = np.isin(D["task"], tidx)
    pg = PF.pgate_fit(D["grip"][keep], D["prev_grip"][keep])
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    D["_held"] = np.asarray(PF.pgate_rows(key, D["grip"], D["prev_grip"], D["ee_pos"][:, 2], pg["tw"], pg["lift"]), bool)
    print("[loc] event", pg, flush=True)
    rec = {k: [] for k in ("img", "task", "demo", "row", "valid", "lab_C", "lab_A")}
    cam, dropped = None, {}
    for k, ti in zip(TASKS, tidx):
        raw = np.load(os.path.join(FC.DATA, f"raw_{k}.npz"))
        rows0 = np.flatnonzero(D["task"] == ti)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(FC.LIBBDDL, FC.KNOWN[k] + ".bddl"), camera_heights=H, camera_widths=H)
        env.seed(0)
        env.reset()
        cn = camera(env.env.sim)
        cam = cam or cn
        assert np.allclose(cam[0], cn[0]) and np.allclose(cam[1], cn[1]), ("camera differs", k)
        for dm in np.unique(D["demo"][rows0])[:2 if SMOKE else None]:
            ev = events(D, ti, dm)
            if ev is None:
                dropped[k] = dropped.get(k, 0) + 1
                continue
            r, c, u = ev
            for j in range(0, u, STRIDE):
                val, lc, la = np.zeros(len(HEADS), bool), np.full((len(HEADS), 3), np.nan), np.full((len(HEADS), 3), np.nan)
                for hi, (ent, role, tks) in enumerate(HEADS):
                    if k in tks and (j < c if role == "obj" else True):
                        val[hi] = True
                        lc[hi] = D["ee_pos"][r[c if role == "obj" else u]]
                        la[hi] = D["pos"][r[j], FC.ENT.index(ent)]
                if not val.any():
                    continue
                obs = env.set_init_state(raw["states"][r[j] - rows0[0]])
                rec["img"].append(np.ascontiguousarray(obs["agentview_image"][::-1]))
                rec["task"].append(ti); rec["demo"].append(dm); rec["row"].append(r[j]); rec["valid"].append(val)
                rec["lab_C"].append(lc); rec["lab_A"].append(la)
        env.close()
        print(f"[loc] rendered {k}: {len(rec['img'])} frames so far; dropped demos {dropped}", flush=True)
    dev = {k: [] for k in ("d_img", "d_pos", "d_layout", "d_state")}
    for li, lay in enumerate(("familiar", "new")):
        t = next(x for x in n2["tasks"] if x["layout"] == lay)
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
        for s in STATES:
            env = OffScreenRenderEnv(bddl_file_name=os.path.join(FC.ROOT, t["bddl"]), camera_heights=H, camera_widths=H)
            env.seed(7)
            env.reset()
            dj = FC.drawer_joint(env)
            obs = env.set_init_state(S[s])
            for _ in range(10):
                obs, _, _, _ = env.step(FC.DUMMY)
            cn = camera(env.env.sim)
            assert np.allclose(cam[0], cn[0]) and np.allclose(cam[1], cn[1], atol=1e-6), ("camera differs", lay, s)
            dev["d_img"].append(np.ascontiguousarray(obs["agentview_image"][::-1]))
            dev["d_pos"].append(FC.scene(env, obs, dj)["pos"])
            dev["d_layout"].append(li); dev["d_state"].append(s)
            env.close()
        print(f"[loc] rendered development states of {lay}", flush=True)
    np.savez(FRAMES, K=cam[0], R=cam[1], dropped=json.dumps(dropped), **{k: np.asarray(v) for k, v in rec.items()},
             **{k: np.asarray(v) for k, v in dev.items()})
    print(f"[loc] wrote {len(rec['img'])} demo frames, {len(dev['d_img'])} development frames", flush=True)


def net():
    import torch.nn as nn
    L = [nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU()]
    for d in (1, 2, 2, 4, 4):
        L += [nn.Conv2d(64, 64, 3, 1, d, dilation=d), nn.GroupNorm(8, 64), nn.ReLU()]
    return nn.Sequential(*L, nn.Conv2d(64, len(HEADS), 1))


def to_pix(F, arm):
    """Labels projected into the upright image: [frames, heads, 2], nan where invalid."""
    lab = F[f"lab_{arm}"]
    n, h = lab.shape[:2]
    bad = np.isnan(lab[..., 0])
    pts = np.nan_to_num(lab).reshape(-1, 3)
    p = project(F["K"], F["R"], pts).reshape(n, h, 2)
    p[bad] = np.nan
    return p


def train(arm, seed):
    import torch
    import torch.nn.functional as TF
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[loc] device", dev, flush=True)
    F = np.load(FRAMES)
    F = {k: F[k] for k in ("img", "demo", "valid", "lab_C", "lab_A", "K", "R")}
    pix = to_pix(F, arm)
    tr = (F["demo"] < N_TRAIN_DEMO) if not SMOKE else np.ones(len(F["demo"]), bool)
    pool = [np.flatnonzero(tr & F["valid"][:, h]) for h in range(len(HEADS))]
    print("[loc] frames per head", [len(p) for p in pool], flush=True)
    m = net().to(dev)
    steps, batch = (40, 8) if SMOKE else (4000, 32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    C = H // 2
    yy, xx = torch.meshgrid(torch.arange(C, device=dev), torch.arange(C, device=dev), indexing="ij")
    for it in range(steps):
        hs = rng.integers(0, len(HEADS), batch)
        ix = np.array([rng.choice(pool[h]) for h in hs])
        x = torch.as_tensor(F["img"][ix], device=dev).permute(0, 3, 1, 2).float() / 255.0
        s = torch.as_tensor(rng.uniform(0.85, 1.15, batch), device=dev, dtype=torch.float32)
        t = torch.as_tensor(rng.uniform(-64, 64, (batch, 2)) / (H / 2), device=dev, dtype=torch.float32)   # normalised (x, y)
        th = torch.zeros(batch, 2, 3, device=dev)
        th[:, 0, 0], th[:, 1, 1], th[:, 0, 2], th[:, 1, 2] = s, s, t[:, 0], t[:, 1]
        g = TF.affine_grid(th, x.shape, align_corners=False)
        x = TF.grid_sample(x, g, padding_mode="border", align_corners=False)
        br = torch.as_tensor(rng.uniform(-0.15, 0.15, (batch, 1, 1, 1)), device=dev, dtype=torch.float32)
        ct = torch.as_tensor(rng.uniform(0.85, 1.15, (batch, 1, 1, 1)), device=dev, dtype=torch.float32)
        x = ((x - 0.5) * ct + 0.5 + br).clamp(0, 1) - 0.5
        rc = torch.as_tensor(pix[ix, hs], device=dev, dtype=torch.float32)
        nx, ny = (rc[:, 1] + 0.5) / (H / 2) - 1.0, (rc[:, 0] + 0.5) / (H / 2) - 1.0     # the image's normalised coordinates
        ox, oy = (nx - t[:, 0]) / s, (ny - t[:, 1]) / s                                 # where the label lands after the warp
        cx, cy = (ox + 1.0) * C / 2.0 - 0.5, (oy + 1.0) * C / 2.0 - 0.5
        lg = m(x)[torch.arange(batch, device=dev), torch.as_tensor(hs, device=dev)]
        tgt = torch.exp(-(((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2) / (2 * 1.5 ** 2)))
        tgt = tgt / tgt.sum((1, 2), keepdim=True).clamp_min(1e-9)
        loss = -(tgt * TF.log_softmax(lg.flatten(1), 1).view(batch, C, C)).sum((1, 2)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"[loc] {arm} s{seed} step {it} loss {float(loss):.3f}", flush=True)
    os.makedirs(os.path.join(FC.DATA, "ckpt"), exist_ok=True)
    torch.save(m.state_dict(), os.path.join(FC.DATA, "ckpt", f"pp_loc_{arm}_s{seed}{SFX}.pt"))


def locate(m, img, dev):
    """Images [N, H, H, 3] -> (row, col) [N, heads, 2]: the heatmap's peak, refined over a 7x7 window."""
    import torch
    out = []
    for i in range(0, len(img), 16):
        x = torch.as_tensor(img[i:i + 16], device=dev).permute(0, 3, 1, 2).float() / 255.0 - 0.5
        with torch.no_grad():
            lg = m(x)
        B, h, C, _ = lg.shape
        p = torch.softmax(lg.flatten(2), 2).view(B, h, C, C).cpu().numpy()
        for pb in p:
            rows = []
            for ph in pb:
                r0, c0 = np.unravel_index(int(ph.argmax()), ph.shape)
                a, b, c, d = max(r0 - 3, 0), min(r0 + 4, C), max(c0 - 3, 0), min(c0 + 4, C)
                w = ph[a:b, c:d]
                rr, cc = np.mgrid[a:b, c:d]
                rows.append([2.0 * (rr * w).sum() / w.sum() + 0.5, 2.0 * (cc * w).sum() / w.sum() + 0.5])
            out.append(rows)
    return np.asarray(out)


def stat(pred, oracle, q=95):
    return float(np.percentile(np.linalg.norm(pred[:, :2] - oracle[:, :2], axis=-1), q))


def displacement():
    import pp_atlas as PA
    n2 = PA.tasks()
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    res = {}
    for model in ("pi05", "groot"):
        for lay in ("familiar", "new"):
            d = {}
            for c in PA.NEW + PA.KNOWN:
                t = tid[(c, lay)]
                z = np.load(os.path.join(PA.OUT, "pp_hand", f"{model}_H_{t['tid']}_track.npz"))
                for i in range(int(z["n"])):
                    k = int(z[f"e{i}_handed"])
                    if k >= 0:
                        o = z[f"e{i}_obj"]
                        d.setdefault(t["object"], []).append(float(np.linalg.norm(o[k][:2] - o[0][:2])))
            res[f"{model}_{lay}"] = {k: dict(n=len(v), p50=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in d.items()}
    return res


def read():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = np.load(FRAMES)
    F = {k: F[k] for k in F.files}
    K, R = F["K"], F["R"]
    pts = np.array([[0.0, 0.0, 0.9], [0.1, -0.2, 0.92], [-0.1, 0.2, 0.95]])
    assert np.abs(lift(K, R, project(K, R, pts), pts[:, 2]) - pts).max() < 1e-6, "lift is not project's inverse"
    a0, a1 = lift(K, R, np.array([[128.0, 128.0]]), [0.9]), lift(K, R, np.array([[128.0, 129.0]]), [0.9])
    res = dict(metres_per_pixel=float(np.linalg.norm(a0 - a1)), dropped=json.loads(str(F["dropped"])), arms={})
    D = np.load(os.path.join(FC.DATA, "demos.npz"))["pos"]
    tr, he = F["demo"] < N_TRAIN_DEMO, F["demo"] >= N_TRAIN_DEMO
    for arm in ("C", "A"):
        per = {}
        for hi, (ent, role, tks) in enumerate(HEADS):
            v = F["valid"][:, hi]
            lab = F[f"lab_{arm}"][:, hi]
            ei = FC.ENT.index(ent)
            orc = D[F["row"], ei]
            b = np.mean(lab[tr & v, :2] - orc[tr & v, :2], 0)
            noise = np.linalg.norm(lab[v, :2] - orc[v, :2] - b, axis=-1)
            per[NAMES[hi]] = dict(z=float(lab[tr & v, 2].mean()), b=b.tolist(), table=lab[tr & v, :2].mean(0).tolist(),
                                  label_noise_p95=float(np.percentile(noise, 95)), n_train_frames=int((tr & v).sum()), ent=ei)
        seeds = {}
        for s in (0, 1, 2):
            m = net().to(dev)
            m.load_state_dict(torch.load(os.path.join(FC.DATA, "ckpt", f"pp_loc_{arm}_s{s}.pt"), map_location=dev))
            m.eval()
            pd = locate(m, F["d_img"], dev)
            ph = locate(m, F["img"][he], dev)
            seeds[s] = dict(dev=pd, held=ph)
        rows = {}
        for hi, nm in enumerate(NAMES):
            p = per[nm]
            ei = p["ent"]
            r = dict(p)
            r.pop("ent")
            for li, lay in enumerate(("familiar", "new")):
                mk = F["d_layout"] == li
                orc = F["d_pos"][mk, ei]
                e, raw, med = [], [], []
                for s in seeds:
                    w = lift(K, R, seeds[s]["dev"][mk, hi], np.full(mk.sum(), p["z"]))
                    raw.append(stat(w, orc))
                    w2 = w.copy()
                    w2[:, :2] -= np.asarray(p["b"])
                    e.append(stat(w2, orc))
                    med.append(stat(w2, orc, 50))
                tb = np.tile(np.asarray(p["table"]) - np.asarray(p["b"]), (mk.sum(), 1))
                r[lay] = dict(e=float(np.median(e)), e_seeds=e, e_raw=float(np.median(raw)), e_median=float(np.median(med)),
                              T=stat(tb, orc), oracle_z=[float(orc[:, 2].min()), float(orc[:, 2].max())])
            vh = F["valid"][he, hi]
            hd = []
            for s in seeds:
                w = lift(K, R, seeds[s]["held"][vh, hi], np.full(vh.sum(), p["z"]))
                w[:, :2] -= np.asarray(p["b"])
                hd.append(stat(w, D[F["row"][he][vh], ei]))
            r["demos40_49_e"] = float(np.median(hd))
            rows[nm] = r
        within = all(rows[n][lay]["e"] <= BUDGET for n in NAMES for lay in ("familiar", "new"))
        beats = all(rows[n]["new"]["e"] <= rows[n]["new"]["T"] / 2 for n in NAMES)
        v = "WITHIN BUDGET" if within else ("BEATS A TABLE, OVER BUDGET" if beats else "NO BETTER THAN A TABLE")
        res["arms"][arm] = dict(verdict=v, heads=rows)
        print(f"[loc] arm {arm}: {v}", flush=True)
        print(json.dumps(rows), flush=True)
    res["grasp_displacement"] = displacement()
    print(json.dumps(res["grasp_displacement"]), flush=True)
    print("[loc] metres per pixel", res["metres_per_pixel"], flush=True)
    with open(os.path.join(OUT, "pp_loc.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def smoke_check():
    """Runs train's outputs through locate and lift, and prints shapes only."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = np.load(FRAMES)
    pts = np.array([[0.0, 0.0, 0.9], [0.1, -0.2, 0.92]])
    print("[loc] round trip", float(np.abs(lift(F["K"], F["R"], project(F["K"], F["R"], pts), pts[:, 2]) - pts).max()), flush=True)
    m = net().to(dev)
    m.load_state_dict(torch.load(os.path.join(FC.DATA, "ckpt", f"pp_loc_C_s0{SFX}.pt"), map_location=dev))
    print("[loc] locate", locate(m.eval(), F["d_img"], dev).shape, "frames", F["img"].shape, "valid per head", F["valid"].sum(0).tolist(),
          "pix in range", bool(np.nanmin(to_pix(F, "C")) >= 0 and np.nanmax(to_pix(F, "C")) < H), flush=True)
    print("[loc] displacement read runs on", len(displacement()), "groups", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "frames":
        frames()
    elif mode == "train":
        train(sys.argv[2], int(sys.argv[3]))
    elif mode == "read":
        read()
    elif mode == "smoke_check":
        smoke_check()

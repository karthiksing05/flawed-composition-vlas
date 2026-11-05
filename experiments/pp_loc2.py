"""The locator's second round (docs/LOCATOR.md 9; m3_prereg pp_loc2).  openpi venv, GPU.

    train ARM S    ARM in CP, AP, CS, CPS -> DATA/ckpt/pp_loc2_<ARM>_s<S>.pt (S arms also _labels.npy)
    read           -> results/flawed/pp_loc2.json
    smoke_check    shapes only; prints no error on a development state

LOC_SMOKE=1 reuses pp_loc's smoke frames: 40 steps, _smoke files.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_loc as L                                                          # noqa: E402

FC, H, SFX = L.FC, L.H, L.SFX
ENT_OF_HEAD = (0, 1, 2, 3, 2)                                               # cheese, bottle, bowl, plate
SRC_HEAD = (0, 1, 2, 3)                                                     # the head whose mean label stands for each entity
P_PASTE, GUARD, CUT_ROI, CUT_DIFF, PLATE_R = 0.8, 20.0, 70.0, 25.0, 40.0
BAD = 0.05


def ck(arm, s):
    return os.path.join(FC.DATA, "ckpt", f"pp_loc2_{arm}_s{s}{SFX}")


def first_net(arm, s):
    """S arms' first network: section 7's C for CS, CP for CPS."""
    return os.path.join(FC.DATA, "ckpt", f"pp_loc_C_s{s}{SFX}.pt") if arm == "CS" else ck("CP", s) + ".pt"


def load():
    F = np.load(L.FRAMES)
    return {k: F[k] for k in ("img", "demo", "task", "row", "valid", "lab_C", "lab_A", "K", "R")}


def clean_labels(F, base_pix, arm, s, dev):
    """Mechanism S: per demo and head, the median peak over its training frames, unless over GUARD px from the frame's own label."""
    import torch
    m = L.net().to(dev)
    m.load_state_dict(torch.load(first_net(arm, s), map_location=dev))
    m.eval()
    pk = L.locate(m, F["img"], dev)
    out = base_pix.copy()
    moved, kept = 0, 0
    key = F["task"].astype(np.int64) * 1000 + F["demo"]
    for hi in range(len(L.HEADS)):
        v = F["valid"][:, hi]
        for k in np.unique(key[v]):
            ix = np.flatnonzero(v & (key == k))
            med = np.median(pk[ix, hi], 0)
            far = np.linalg.norm(med - base_pix[ix, hi], axis=-1) > GUARD
            out[ix[~far], hi] = med
            moved += int((~far).sum()); kept += int(far.sum())
    print(f"[loc2] {arm} s{s}: labels moved on {moved} frame-heads, kept on {kept}", flush=True)
    return out


def cut_frames():
    """LIBERO venv, EGL: each training demo's first and last recorded states, set and rendered -> DATA/loc_cut.npz."""
    from libero.libero.envs import OffScreenRenderEnv
    Z = np.load(os.path.join(FC.DATA, "demos.npz"))
    rec = {k: [] for k in ("first", "last", "task", "demo")}
    for k in L.TASKS:
        ti = list(FC.KNOWN).index(k)
        raw = np.load(os.path.join(FC.DATA, f"raw_{k}.npz"))
        rows0 = np.flatnonzero(Z["task"] == ti)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(FC.LIBBDDL, FC.KNOWN[k] + ".bddl"), camera_heights=H, camera_widths=H)
        env.seed(0)
        env.reset()
        for dm in range(2 if L.SMOKE else L.N_TRAIN_DEMO):
            r = np.flatnonzero((Z["task"] == ti) & (Z["demo"] == dm))
            for key, j in (("first", r[0]), ("last", r[-1])):
                obs = env.set_init_state(raw["states"][j - rows0[0]])
                rec[key].append(np.ascontiguousarray(obs["agentview_image"][::-1]))
            rec["task"].append(ti); rec["demo"].append(dm)
        env.close()
        print(f"[loc2] cut frames {k}", flush=True)
    np.savez(os.path.join(FC.DATA, f"loc_cut{SFX}.npz"), **{k: np.asarray(v) for k, v in rec.items()})


def make_bank(F, pix, pool, arm, dev):
    """Cutouts per entity: what changed when the robot moved it (first frame against last), or a disc for the plate, which no K1-K6 demo moves.
    -> cuts [entity][N, 4, 128, 128], offset [head] (label minus anchor), and the empty-scene image with its alpha."""
    import torch
    from scipy import ndimage as ni
    Z = np.load(os.path.join(FC.DATA, f"loc_cut{SFX}.npz"))
    first, last, task, demo = Z["first"], Z["last"], Z["task"], Z["demo"]
    tidx = {k: list(FC.KNOWN).index(k) for k in L.TASKS}
    rr, cc = np.mgrid[0:H, 0:H]
    cuts, anchors, areas = {}, {}, {}
    A = np.zeros((H, H))
    B = first[0].astype(np.float32) / 255.0
    for e in range(3):
        cs, an, ar, ms, lasts = [], [], [], [], []
        for k in L.HEADS[e][2]:
            for i in np.flatnonzero(task == tidx[k]):
                f = (F["task"] == task[i]) & (F["demo"] == demo[i]) & F["valid"][:, e]
                if not f.any():
                    continue
                lb = np.nanmean(pix[f, e], 0)
                dsk = (rr - lb[0]) ** 2 + (cc - lb[1]) ** 2 <= CUT_ROI ** 2
                d = ni.binary_opening((np.abs(first[i].astype(np.float32) - last[i]).max(-1) > CUT_DIFF) & dsk)
                lab, n = ni.label(d)
                best, sz = None, 0
                for c in range(1, n + 1):
                    p = lab == c
                    if np.sqrt((rr[p] - lb[0]) ** 2 + (cc[p] - lb[1]) ** 2).min() <= GUARD and p.sum() > sz:
                        best, sz = p, int(p.sum())
                if best is None:
                    continue
                best = ni.binary_fill_holes(best)
                if not 100 <= best.sum() <= 9000:
                    continue
                cen = np.round([rr[best].mean(), cc[best].mean()]).astype(int)
                al = ni.gaussian_filter(best.astype(np.float32), 1.0)
                rgba = np.concatenate([first[i].astype(np.float32) / 255.0, al[..., None]], -1)
                pad = np.pad(rgba, ((64, 64), (64, 64), (0, 0)))
                cs.append(pad[cen[0]:cen[0] + 128, cen[1]:cen[1] + 128].transpose(2, 0, 1))
                an.append(cen); ar.append(int(best.sum())); ms.append(best); lasts.append(last[i].astype(np.float32) / 255.0)
        cuts[e], anchors[e], areas[e] = np.stack(cs), np.mean(an, 0), np.median(ar)
        m = ni.binary_dilation(np.mean(ms, 0) > 0.2, iterations=12)
        a = ni.gaussian_filter(m.astype(np.float32), 1.5)
        B = B * (1 - a[..., None]) + np.median(lasts, 0) * a[..., None]
        A = np.maximum(A, a)
    pm = np.nanmean(pix[pool[3], 3], 0)
    cs = []
    for i in np.flatnonzero(task == tidx["K1"]):
        f = (F["task"] == task[i]) & (F["demo"] == demo[i]) & F["valid"][:, 3]
        cen = np.nanmean(pix[f, 3], 0) if (arm[0] == "A" and f.any()) else pm
        cen = np.round(cen).astype(int)
        al = ni.gaussian_filter((((rr - cen[0]) ** 2 + (cc - cen[1]) ** 2) <= PLATE_R ** 2).astype(np.float32), 1.0)
        rgba = np.concatenate([first[i].astype(np.float32) / 255.0, al[..., None]], -1)
        cs.append(np.pad(rgba, ((64, 64), (64, 64), (0, 0)))[cen[0]:cen[0] + 128, cen[1]:cen[1] + 128].transpose(2, 0, 1))
    cuts[3], anchors[3], areas[3] = np.stack(cs), pm, float(np.pi * PLATE_R ** 2)
    dp = ((rr - pm[0]) ** 2 + (cc - pm[1]) ** 2) <= (PLATE_R + 4) ** 2                # the plate's patch of table: filled row by row across its disc
    for r in np.flatnonzero(dp.any(1)):
        col = np.flatnonzero(dp[r])
        a0, a1 = max(col[0] - 1, 0), min(col[-1] + 1, H - 1)
        t = (col - a0) / max(a1 - a0, 1)
        B[r, col] = B[r, a0] * (1 - t[:, None]) + B[r, a1] * t[:, None]
    A = np.maximum(A, ni.gaussian_filter(dp.astype(np.float32), 1.5))
    off = np.array([np.nanmean(pix[pool[h], h], 0) - anchors[ENT_OF_HEAD[h]] for h in range(len(L.HEADS))])
    print("[loc2] bank", {e: (len(cuts[e]), float(areas[e])) for e in cuts}, "offsets", np.round(off, 1).tolist(), flush=True)
    return ({e: torch.as_tensor(cuts[e], device=dev, dtype=torch.float32) for e in cuts}, off,
            torch.as_tensor(B, device=dev, dtype=torch.float32).permute(2, 0, 1)[None], torch.as_tensor(A, device=dev, dtype=torch.float32)[None, None])


def paste_cut(x, cut, ctr, scale, mask):
    """Composite cutouts [B, 4, 128, 128] with their centres at ctr [B, 2] (row, col), rescaled by scale."""
    import torch
    import torch.nn.functional as TF
    rr, cc = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32), torch.arange(H, device=x.device, dtype=torch.float32), indexing="ij")
    rel_r = (rr[None] - ctr[:, 0, None, None]) / scale[:, None, None]
    rel_c = (cc[None] - ctr[:, 1, None, None]) / scale[:, None, None]
    g = torch.stack([(rel_c + 64.0 + 0.5) / 64.0 - 1.0, (rel_r + 64.0 + 0.5) / 64.0 - 1.0], -1)
    s = TF.grid_sample(cut, g, padding_mode="zeros", align_corners=False)
    a = s[:, 3:] * mask.view(len(x), 1, 1, 1)
    return x * (1 - a) + s[:, :3] * a


def depth(R, pts):
    return -(np.linalg.inv(R) @ np.concatenate([pts, np.ones((len(pts), 1))], 1).T).T[:, 2]


def train(arm, seed):
    import torch
    import torch.nn.functional as TF
    assert arm in ("CP", "AP", "CS", "CPS")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = load()
    K, R = F["K"], F["R"]
    pix = L.to_pix(F, arm[0])
    if "S" in arm:
        pix = clean_labels(F, pix, arm, seed, dev)
        np.save(ck(arm, seed) + "_labels.npy", pix)
    tr = (F["demo"] < L.N_TRAIN_DEMO) if not L.SMOKE else np.ones(len(F["demo"]), bool)
    pool = [np.flatnonzero(tr & F["valid"][:, h]) for h in range(len(L.HEADS))]
    lab = F[f"lab_{arm[0]}"]
    hz = np.array([np.nanmean(lab[pool[h], h, 2]) for h in range(len(L.HEADS))])
    msrc = np.array([np.nanmean(pix[pool[h], h], 0) for h in range(len(L.HEADS))])   # head -> mean label pixel
    use = "P" in arm
    if use:
        bank, off, BG, ALPHA = make_bank(F, pix, pool, arm, dev)
    Z = np.load(os.path.join(FC.DATA, "demos.npz"))
    ee = Z["ee_pos"][np.isin(Z["task"], [list(FC.KNOWN).index(k) for k in L.TASKS])]
    lo, hi_ = ee[:, :2].min(0), ee[:, :2].max(0)
    print(f"[loc2] {arm} s{seed} device {dev}; workspace {lo.tolist()} {hi_.tolist()}", flush=True)

    def world_of_pixel(rc, z):
        return L.lift(K, R, rc, z)

    m = L.net().to(dev)
    steps, batch = (40, 8) if L.SMOKE else (4000, 32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    C = H // 2
    yy, xx = torch.meshgrid(torch.arange(C, device=dev), torch.arange(C, device=dev), indexing="ij")
    T_ = lambda v: torch.as_tensor(np.asarray(v), device=dev, dtype=torch.float32)  # noqa: E731
    for it in range(steps):
        hs = rng.integers(0, len(L.HEADS), batch)
        ix = np.array([rng.choice(pool[h]) for h in hs])
        x = torch.as_tensor(F["img"][ix], device=dev).permute(0, 3, 1, 2).float() / 255.0
        rc = pix[ix, hs].copy()
        if use:
            mask = T_(rng.uniform(size=batch) < P_PASTE)
            a = ALPHA * mask.view(batch, 1, 1, 1)
            x = x * (1 - a) + BG * a                                            # every object's usual place, emptied
            own = np.array([ENT_OF_HEAD[h] for h in hs])
            new_own = rc.copy()
            order = [[e for e in range(4) if e != own[b]] + [own[b]] for b in range(batch)]   # the trained head's object last
            for pos in range(4):
                es = np.array([order[b][pos] for b in range(batch)])
                he = np.array([hs[b] if es[b] == own[b] else SRC_HEAD[es[b]] for b in range(batch)])
                zs = hz[he]
                wd = np.stack([rng.uniform(lo[0], hi_[0], batch), rng.uniform(lo[1], hi_[1], batch), zs], -1)
                ds = L.project(K, R, wd)
                ws = world_of_pixel(msrc[he], zs)
                sc = np.clip(depth(R, ws) / depth(R, wd), 0.5, 2.0)
                ct = ds - off[he] * sc[:, None]
                cut = torch.stack([bank[int(es[b])][rng.integers(len(bank[int(es[b])]))] for b in range(batch)])
                x = paste_cut(x, cut, T_(ct), T_(sc), mask)
                if pos == 3:
                    new_own = ds
            rc = np.where(mask.cpu().numpy()[:, None] > 0.5, new_own, rc)
            if L.SMOKE and it == 0:                                             # a visual check of the patches; demo frames only
                from PIL import Image, ImageDraw
                canvas = Image.new("RGB", (H * 4, H * 2))
                for b in range(batch):
                    im = Image.fromarray((x[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                    d = ImageDraw.Draw(im)
                    d.ellipse([rc[b, 1] - 5, rc[b, 0] - 5, rc[b, 1] + 5, rc[b, 0] + 5], outline=(0, 255, 0))
                    d.text((3, 3), L.NAMES[hs[b]], fill=(255, 255, 0))
                    canvas.paste(im, (H * (b % 4), H * (b // 4)))
                canvas.save(os.path.join(L.OUT, f"pp_loc2_patches_{arm}.png"))
        s = T_(rng.uniform(0.85, 1.15, batch))
        t = T_(rng.uniform(-64, 64, (batch, 2)) / (H / 2))
        th = torch.zeros(batch, 2, 3, device=dev)
        th[:, 0, 0], th[:, 1, 1], th[:, 0, 2], th[:, 1, 2] = s, s, t[:, 0], t[:, 1]
        g = TF.affine_grid(th, x.shape, align_corners=False)
        x = TF.grid_sample(x, g, padding_mode="border", align_corners=False)
        br, ct = T_(rng.uniform(-0.15, 0.15, (batch, 1, 1, 1))), T_(rng.uniform(0.85, 1.15, (batch, 1, 1, 1)))
        x = ((x - 0.5) * ct + 0.5 + br).clamp(0, 1) - 0.5
        rct = T_(rc)
        nx, ny = (rct[:, 1] + 0.5) / (H / 2) - 1.0, (rct[:, 0] + 0.5) / (H / 2) - 1.0
        ox, oy = (nx - t[:, 0]) / s, (ny - t[:, 1]) / s
        cx, cy = (ox + 1.0) * C / 2.0 - 0.5, (oy + 1.0) * C / 2.0 - 0.5
        lg = m(x)[torch.arange(batch, device=dev), torch.as_tensor(hs, device=dev)]
        tgt = torch.exp(-(((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2) / (2 * 1.5 ** 2)))
        tgt = tgt / tgt.sum((1, 2), keepdim=True).clamp_min(1e-9)
        loss = -(tgt * TF.log_softmax(lg.flatten(1), 1).view(batch, C, C)).sum((1, 2)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"[loc2] {arm} s{seed} step {it} loss {float(loss):.3f}", flush=True)
    torch.save(m.state_dict(), ck(arm, seed) + ".pt")


def read():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = np.load(L.FRAMES)
    F = {k: F[k] for k in F.files}
    K, R = F["K"], F["R"]
    Dp = np.load(os.path.join(FC.DATA, "demos.npz"))["pos"]
    tr = F["demo"] < L.N_TRAIN_DEMO
    res = {}
    for arm in ("CPS", "CP", "CS", "AP"):
        lab = F[f"lab_{arm[0]}"]
        rows = {n: dict(familiar=dict(e=[], raw=[], med=[], T=[]), new=dict(e=[], raw=[], med=[], T=[], n_bad=[])) for n in L.NAMES}
        base = L.to_pix(F, arm[0])
        for s in (0, 1, 2):
            pix = np.load(ck(arm, s) + "_labels.npy") if "S" in arm else base
            m = L.net().to(dev)
            m.load_state_dict(torch.load(ck(arm, s) + ".pt", map_location=dev))
            pr = L.locate(m.eval(), F["d_img"], dev)
            for hi, nm in enumerate(L.NAMES):
                ei = FC.ENT.index(L.HEADS[hi][0])
                v = tr & F["valid"][:, hi]
                z = float(lab[v, hi, 2].mean())
                wl = L.lift(K, R, pix[v, hi], np.full(v.sum(), z))
                orc = Dp[F["row"][v], ei]
                b = (wl[:, :2] - orc[:, :2]).mean(0)
                tb = wl[:, :2].mean(0)
                for li, lay in enumerate(("familiar", "new")):
                    mk = F["d_layout"] == li
                    o = F["d_pos"][mk, ei]
                    w = L.lift(K, R, pr[mk, hi], np.full(mk.sum(), z))
                    w2 = w.copy()
                    w2[:, :2] -= b
                    tab = np.tile(tb - b, (mk.sum(), 1))
                    d = rows[nm][lay]
                    d["e"].append(L.stat(w2, o)); d["raw"].append(L.stat(w, o)); d["med"].append(L.stat(w2, o, 50)); d["T"].append(L.stat(tab, o))
                    if lay == "new":
                        d["n_bad"].append(int((np.linalg.norm(w2[:, :2] - o[:, :2], axis=-1) > BAD).sum()))
        out = {}
        for nm, r in rows.items():
            out[nm] = {}
            for lay, d in r.items():
                out[nm][lay] = dict(e=float(np.median(d["e"])), e_seeds=d["e"], e_raw=float(np.median(d["raw"])), e_median=float(np.median(d["med"])),
                                    T=float(np.median(d["T"])))
                if lay == "new":
                    out[nm][lay]["n_bad"] = d["n_bad"]
                    out[nm][lay]["few_states"] = all(n <= 3 and q <= L.BUDGET for n, q in zip(d["n_bad"], d["med"]))
        within = all(out[n][y]["e"] <= L.BUDGET for n in L.NAMES for y in ("familiar", "new"))
        beats = all(out[n]["new"]["e"] <= out[n]["new"]["T"] / 2 for n in L.NAMES)
        v = "WITHIN BUDGET" if within else ("BEATS A TABLE, OVER BUDGET" if beats else "NO BETTER THAN A TABLE")
        res[arm] = dict(label=v, heads=out)
        print(f"[loc2] {arm}: {v}", json.dumps(out), flush=True)
    with open(os.path.join(L.OUT, "pp_loc2.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def smoke_check():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = np.load(L.FRAMES)
    for arm in ("CP", "AP", "CS", "CPS"):
        m = L.net().to(dev)
        m.load_state_dict(torch.load(ck(arm, 0) + ".pt", map_location=dev))
        print("[loc2]", arm, L.locate(m.eval(), F["d_img"], dev).shape, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cut":
        cut_frames()
    elif mode == "train":
        train(sys.argv[2], int(sys.argv[3]))
    elif mode == "read":
        read()
    elif mode == "smoke_check":
        smoke_check()

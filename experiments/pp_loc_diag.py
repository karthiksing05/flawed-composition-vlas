"""The locator's diagnostic (docs/LOCATOR.md 8; m3_prereg pp_loc_diag): a few states, or a broad failure?  openpi venv.
-> results/flawed/pp_loc_diag.json, results/flawed/pp_loc_diag.png"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pp_loc as L                                                          # noqa: E402
import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402

FC = L.FC
BAD = 0.05
torch.set_num_threads(2)
dev = torch.device("cpu")
F = np.load(L.FRAMES)
F = {k: F[k] for k in F.files if k != "img"}
K, R = F["K"], F["R"]
LOC = json.load(open(os.path.join(L.OUT, "pp_loc.json"), encoding="utf-8"))
mk = F["d_layout"] == 1
img, pos, st = F["d_img"][mk], F["d_pos"][mk], F["d_state"][mk]
res, tiles = {}, []
for arm in ("C", "A"):
    res[arm] = {}
    pr = {}
    for s in (0, 1, 2):
        m = L.net()
        m.load_state_dict(torch.load(os.path.join(FC.DATA, "ckpt", f"pp_loc_{arm}_s{s}.pt"), map_location=dev))
        pr[s] = L.locate(m.eval(), img, dev)
    for hi, nm in enumerate(L.NAMES):
        h = LOC["arms"][arm]["heads"][nm]
        ei = FC.ENT.index(L.HEADS[hi][0])
        rows, ok = {}, True
        for s in pr:
            w = L.lift(K, R, pr[s][:, hi], np.full(len(img), h["z"]))[:, :2] - np.asarray(h["b"])
            err = np.linalg.norm(w - pos[:, ei, :2], axis=-1)
            bad = np.flatnonzero(err > BAD)
            ok = ok and len(bad) <= 3 and float(np.median(err)) <= 0.02
            near = {}
            for b in bad:
                dist = [float(np.linalg.norm(w[b] - pos[b, j, :2])) if j != ei else 9.0 for j in range(len(FC.ENT))]
                j = int(np.argmin(dist))
                near[int(st[b])] = dict(err=float(err[b]), nearest=FC.ENT[j] if dist[j] <= BAD else None, oracle_z=float(pos[b, ei, 2]))
                if s == 0 and len(tiles) < 24:
                    im = Image.fromarray(img[b]).resize((384, 384))
                    d = ImageDraw.Draw(im)
                    o = L.project(K, R, pos[b, ei][None])[0] * 1.5
                    p = pr[s][b, hi] * 1.5
                    d.ellipse([o[1] - 6, o[0] - 6, o[1] + 6, o[0] + 6], outline=(0, 255, 0), width=2)
                    d.line([p[1] - 7, p[0] - 7, p[1] + 7, p[0] + 7], fill=(255, 0, 0), width=2)
                    d.line([p[1] - 7, p[0] + 7, p[1] + 7, p[0] - 7], fill=(255, 0, 0), width=2)
                    d.text((4, 4), f"{arm} {nm} state {int(st[b])} err {err[b]:.3f}", fill=(255, 255, 0))
                    tiles.append(im)
            rows[s] = dict(n_bad=len(bad), median=float(np.median(err)), bad=near)
        res[arm][nm] = dict(verdict="FEW STATES" if ok else "BROAD", seeds=rows)
        print(f"[diag] {arm} {nm}: {res[arm][nm]['verdict']}", json.dumps(rows), flush=True)
with open(os.path.join(L.OUT, "pp_loc_diag.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
n = max(len(tiles), 1)
c = 4
canvas = Image.new("RGB", (384 * c, 384 * ((n + c - 1) // c)))
for i, t in enumerate(tiles):
    canvas.paste(t, (384 * (i % c), 384 * (i // c)))
canvas.save(os.path.join(L.OUT, "pp_loc_diag.png"))
print(f"[diag] {len(tiles)} tiles", flush=True)

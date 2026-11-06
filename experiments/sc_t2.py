"""T2 (docs/SCAFFOLD_RESULTS.md 19; m3_prereg sc_t2): do frozen pre-fusion VLM features locate entities from one demo?  openpi venv, GPU, no training.
Key: pi05_libero's image features (embed_image, unit rows) at the token under the demo's event label in the demo's first frame, averaged with its 4 neighbours.
Locate: the most similar token in a development frame, refined by a softmax over its 3 x 3 window, lifted to the head's label height.  Same frames, heads, labels and camera matrices as docs/LOCATOR.md.

    python experiments/sc_t2.py        -> results/flawed/sc_t2_read.json
SC_SMOKE=1: writes sc_t2_smoke.json and prints no error on a development state.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fa1_common as FC                                                     # noqa: E402
import pp_loc as PL                                                         # noqa: E402

SMOKE = bool(os.environ.get("SC_SMOKE"))
OUT = PL.OUT
TEMP, REPS, K10, BUDGET = 0.02, (0, 1, 2), 10, 0.02
P = 16                                                                      # one token = 16 px of the 256 render (lifelong_ood/policy/where.py)


def main():
    import torch
    from PIL import Image
    from lifelong_ood.policy.backbone import PolicyBackbone
    F = np.load(PL.FRAMES)
    F = {k: F[k] for k in F.files}
    K, R = F["K"], F["R"]
    D = np.load(os.path.join(FC.DATA, "demos.npz"))["pos"]
    bk = PolicyBackbone("pi05_libero", device="cuda")

    def feats(imgs):
        out = []
        for g in range(0, len(imgs), 16):
            obs = []
            for up in imgs[g:g + 16]:
                im = np.asarray(Image.fromarray(up[:, ::-1]).resize((224, 224), Image.BICUBIC))     # the client's image: raw[::-1, ::-1] = upright[:, ::-1]
                obs.append({"observation/image": im, "observation/wrist_image": im, "observation/state": np.zeros(8, np.float32), "prompt": ""})
            ob = bk.observations(obs)
            images, _, _, _, _ = bk.model._preprocess_observation(ob, train=False)
            with torch.no_grad():
                f = bk.model.paligemma_with_expert.embed_image(images[0]).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)                                          # [N, 256, D]

    idx = lambda up_rc: (int(np.clip(up_rc[0] // P, 0, 15)), int(np.clip((255 - up_rc[1]) // P, 0, 15)))     # noqa: E731

    def key(f, rc):
        i, j = idx(rc)
        cells = [(a, b) for a, b in ((i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if 0 <= a < 16 and 0 <= b < 16]
        v = np.mean([f[a * 16 + b] for a, b in cells], 0)
        return v / np.linalg.norm(v)

    def locate(fs, v):
        out = []
        for f in fs:
            s = (f @ v).reshape(16, 16)
            i, j = np.unravel_index(int(np.argmax(s)), s.shape)
            a0, a1, b0, b1 = max(i - 1, 0), min(i + 2, 16), max(j - 1, 0), min(j + 2, 16)
            w = np.exp((s[a0:a1, b0:b1] - s[i, j]) / TEMP)
            w = w / w.sum()
            rr, cc = np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij")
            r = float((w * (rr + 0.5)).sum() * P)
            c = float((w * (cc + 0.5)).sum() * P)
            out.append((r, 255.0 - c))
        return np.array(out), None

    fdev = feats(F["d_img"] if not SMOKE else F["d_img"][:4])
    lay = F["d_layout"] if not SMOKE else F["d_layout"][:4]
    dpos = F["d_pos"] if not SMOKE else F["d_pos"][:4]
    res = dict(heads={}, smoke=SMOKE)
    for hi, (ent, role, tks) in enumerate(PL.HEADS):
        nm = PL.NAMES[hi]
        ei = FC.ENT.index(ent)
        ti = list(FC.KNOWN).index(tks[0])
        v = F["valid"][:, hi]
        lab = F["lab_C"][:, hi]
        orc = D[F["row"], ei]
        tr = v & (F["demo"] < PL.N_TRAIN_DEMO)
        b = np.mean(lab[tr, :2] - orc[tr, :2], 0)
        z = float(lab[tr, 2].mean())
        table = lab[tr, :2].mean(0)
        demos = sorted({int(d) for d in F["demo"][v & (F["task"] == ti)]})
        rows = []
        for dm in demos[:K10]:
            i0 = np.flatnonzero(v & (F["task"] == ti) & (F["demo"] == dm))[0]
            rc = PL.project(K, R, lab[i0][None])[0]
            rows.append((i0, rc))
        kf = feats(F["img"][[i for i, _ in rows]])
        keys = [key(f, rc) for f, (_, rc) in zip(kf, rows)]
        arms = {f"k1_{r}": keys[r] for r in REPS}
        v10 = np.mean(keys, 0)
        arms["k10"] = v10 / np.linalg.norm(v10)
        out = {}
        for a, vk in arms.items():
            rcs, _ = locate(fdev, vk)
            w = PL.lift(K, R, rcs, np.full(len(rcs), z))
            w2 = w.copy()
            w2[:, :2] -= b
            out[a] = {}
            for li, l in enumerate(("familiar", "new")):
                mk = lay == li
                if not mk.any():
                    continue
                o = dpos[mk, ei]
                out[a][l] = dict(e=PL.stat(w2[mk], o), e_raw=PL.stat(w[mk], o), median=PL.stat(w2[mk], o, 50))
        tb = {}
        for li, l in enumerate(("familiar", "new")):
            mk = lay == li
            if mk.any():
                tb[l] = PL.stat(np.tile(table - b, (mk.sum(), 1)), dpos[mk, ei])
        res["heads"][nm] = dict(b=b.tolist(), z=z, T=tb, demos=demos[:K10], arms=out)
        if not SMOKE:
            print("[t2]", nm, json.dumps(res["heads"][nm]), flush=True)
    if SMOKE:
        print("[t2] smoke", {n: {a: sorted(x) for a, x in h["arms"].items()} for n, h in res["heads"].items()}, flush=True)
        return
    med = {n: {l: float(np.median([h["arms"][f"k1_{r}"][l]["e"] for r in REPS])) for l in ("familiar", "new")} for n, h in res["heads"].items()}
    res["k1_median_p95"] = med
    within = all(med[n][l] <= BUDGET for n in med for l in ("familiar", "new"))
    beats = all(med[n]["new"] <= res["heads"][n]["T"]["new"] / 2 for n in med)
    res["label"] = "WITHIN BUDGET" if within else ("BEATS A TABLE, OVER BUDGET" if beats else "NO BETTER THAN A TABLE")
    print("[t2]", res["label"], json.dumps(med), flush=True)
    with open(os.path.join(OUT, "sc_t2_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

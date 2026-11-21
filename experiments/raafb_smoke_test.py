"""RAA-F Route B smoke unit test, no model, no simulator (openpi venv, CPU).  Feeds RECORDED observations (rendered K-task demo frames: painted
agentview inputs, the demo's own ground-truth slot positions, the task's K / E from raafb_proj.py) through the SERVER-side crop
(raafb_crop.attach_crop) exactly as wa_sb.py / raa_b.py call it, and checks the crop centre against ground truth.
  T1 synthetic: a marker at (r, c) lands at the crop centre.
  T2 recorded frames: the Estimator's estimate is modelled as truth + 0.6 cm (the SR 82 target median); the true entity's projected position
     lands within 4 crop px of the crop centre, and the entity's segmentation mask centroid within 8 px * 224 / 96 (the plan's kill, in crop px).
  T3 skip (noM) and no-centre queries give an all-zero 224 x 224 x 3 uint8 crop and return None.
  T4 border: a centre near the image edge zero-pads, keeps shape / dtype.
  T5 wa_sb.py and raa_b.py parse.
    python experiments/raafb_smoke_test.py --cam .../camKE.npz [--root RV_ROOT]
"""
import argparse
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import raafb_crop as CR                                                     # noqa: E402

SC = CR.OUT / float(CR.CROP)
CEN = (CR.OUT - 1) / 2.0


def t1():
    img = np.zeros((256, 256, 3), np.uint8)
    for (r, c) in ((128, 128), (60, 200), (200, 40)):
        im = img.copy()
        im[r - 1:r + 2, c - 1:c + 2] = 255
        o = {}
        raa = {"crop_uv_obj": [r, c], "crop_uv_tgt": [0.0, 0.0]}
        CR.attach_crop(o, im, raa, on=False)
        cr = o["observation/crop_image"]
        assert cr.shape == (CR.OUT, CR.OUT, 3) and cr.dtype == np.uint8
        ys, xs = np.nonzero(cr[..., 0] > 60)
        assert abs(ys.mean() - CEN) < 2.0 and abs(xs.mean() - CEN) < 2.0, (r, c, ys.mean(), xs.mean())
        o2 = {}
        CR.attach_crop(o2, im, {"crop_uv_obj": [0, 0], "crop_uv_tgt": [r, c]}, on=True)       # held: the TARGET key is used
        assert np.array_equal(o2["observation/crop_image"], cr)
    print("[smoke] T1 synthetic marker centred ok")


def t2(a):
    import sc_rv as RV
    import fa1_common as FC
    KE = np.load(a.cam)
    by_c, _ = RV.roles_of()
    rows = []
    rng = np.random.default_rng(0)
    for c in ("K1", "K4", "K6"):
        K, E = KE["K_" + c], KE["E_" + c]
        f = RV.demo_files(a.root, c)[0]
        z = dict(np.load(f))
        j = int(z["demo"])
        pos = np.asarray(np.load(RV.chan_path(a.root, c, j))["pos"], np.float64)
        n = len(z["actions"])
        obj, tgt = by_c[c]
        for role, slot in (("object", obj), ("target", tgt)):
            si = FC.ENT.index(slot)
            ei = RV.slot_index(z["names"], z["slot_names"], z["slot_ents"], slot)
            for t in np.linspace(0, n - 1, 6).astype(int):
                m = RV.clean_mask(np.asarray(z["seg"][t]) == ei)
                if m.sum() < 20 or slot not in FC.MOVABLE:
                    continue
                est = pos[t, si] + rng.normal(0, 0.006 / np.sqrt(3), 3)                # ~0.6 cm Estimator error
                raa = {"crop_uv_obj": CR.project_client(est, K, E).tolist(), "crop_uv_tgt": CR.project_client(est, K, E).tolist()}
                o = {}
                used = CR.attach_crop(o, z["image"][t], raa, on=(role == "target"))
                assert used is not None
                uv_true = CR.project_client(pos[t, si], K, E)
                p = CR.crop_centre_px(used, uv_true)
                d_true = float(np.hypot(p[0] - CEN, p[1] - CEN))
                mk = np.repeat(m[..., None].astype(np.uint8) * 255, 3, 2)
                mc = CR.make_crop(mk, used)[..., 0] > 127
                ys, xs = np.nonzero(mc)
                d_mask = float(np.hypot(ys.mean() - CEN, xs.mean() - CEN)) if len(ys) else np.nan
                rows.append((c, role, int(t), d_true, d_mask))
                expect = float(np.linalg.norm(CR.project_client(est, K, E) - uv_true)) * SC               # the Estimator error in crop px; the rest is the pixel rounding of the window (<= 0.71 client px)
                assert abs(d_true - expect) < 1.5 * SC, ("true position off the estimate-centred crop", c, role, t, d_true, expect)
                u0 = CR.attach_crop({}, z["image"][t], {"crop_uv_obj": uv_true.tolist(), "crop_uv_tgt": uv_true.tolist()}, on=False)   # exact centre -> truth in the middle
                assert float(np.hypot(*(CR.crop_centre_px(u0, uv_true) - CEN))) < 1.5 * SC
    dm = np.array([r[4] for r in rows])
    dm = dm[~np.isnan(dm)]
    print("[smoke] T2 recorded frames: n=%d  true-pos offset from crop centre: max %.2f crop px;  mask-centroid offset: median %.1f, max %.1f crop px (kill %.1f)" % (
        len(rows), max(r[3] for r in rows), float(np.median(dm)), float(dm.max()), 8 * SC))
    assert np.median(dm) < 8 * SC, "median mask-centroid offset above the 8 px kill"


def t3_t4():
    img = (np.random.default_rng(1).random((256, 256, 3)) * 255).astype(np.uint8)
    for kw in (dict(skip=True), dict(skip=False)):
        o = {}
        raa = {"crop_uv_obj": [100, 100], "crop_uv_tgt": [100, 100]} if kw["skip"] else {}
        assert CR.attach_crop(o, img, raa, False, **kw) is None
        cr = o["observation/crop_image"]
        assert cr.shape == (CR.OUT, CR.OUT, 3) and cr.dtype == np.uint8 and not cr.any()
    o = {}
    CR.attach_crop(o, img, {"crop_uv_obj": [2.0, 250.0]}, False)
    cr = o["observation/crop_image"]
    assert cr.shape == (CR.OUT, CR.OUT, 3) and cr.dtype == np.uint8
    sq = CR.crop_square(img, (2.0, 250.0))
    assert sq.shape == (CR.CROP, CR.CROP, 3) and not sq[:CR.CROP // 2 - 2].any() and not sq[:, -(CR.CROP // 2 - 6):].any()
    print("[smoke] T3 skip / no-centre zero crop ok; T4 border zero-pad ok")


def t5():
    for f in ("wa_sb.py", "raa_b.py", "raafb_rv.py", "raafb_proj.py"):
        ast.parse(open(os.path.join(HERE, f), encoding="utf-8").read(), f)
    print("[smoke] T5 files parse ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--root", default="/coc/flash10/ksingara3/robot/data/fa1/rv")
    a = ap.parse_args()
    t1()
    t2(a)
    t3_t4()
    t5()
    print("[smoke] ALL OK")

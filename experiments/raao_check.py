"""RAA-o (docs/SCAFFOLD_RESULTS.md 76) channel unit check, CPU, openpi venv, reads only the rendered demos' npz and the chan files
(no simulator).  For two demos (default K5 demo 0 = wine bottle, K1 demo 0 = bowl) prints, at the first held row (the grasp event) and
the last held row (the release), RAA's target channel (target - eef) and RAA-o's (target - eef - offset), the in-hand offset, and the
reference (target - grasped object's own position at that row).  At the onset RAA-o's channel equals the reference by construction;
the gap at the release is the object's slip in the hand.  Also checks that frames before the hold are identical between arms and that
the object-role channel is identical.

    python experiments/raao_check.py [--demos K5:0+K1:0]
"""
import argparse
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sc_rv as RV                                                          # noqa: E402


def cm(v):
    return [round(100.0 * float(x), 1) for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="K5:0+K1:0")
    a = ap.parse_args()
    import fa1_common as FC
    T16 = RV.t16()
    ns = types.SimpleNamespace(root=RV.RV_ROOT, suffix="", arm="raao")
    X = RV.lex(ns)
    by_c, _ = RV.roles_of()
    for item in a.demos.split("+"):
        c, j = item.split(":")
        f = RV.demo_path(RV.RV_ROOT, c, int(j))
        z = dict(np.load(f))
        ch = np.load(RV.chan_path(RV.RV_ROOT, c, int(j)))
        pos, ee = np.asarray(ch["pos"], np.float64), np.asarray(ch["ee"], np.float64)
        info_o, info_a = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0), dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
        rel_o = RV.demo_rel(z, c, f, types.SimpleNamespace(root=RV.RV_ROOT, arm="raao"), X, T16, by_c, info_o)
        rel_a = RV.demo_rel(z, c, f, types.SimpleNamespace(root=RV.RV_ROOT, arm="raa"), X, T16, by_c, info_a)
        H = np.asarray(T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]), bool)
        idx = np.nonzero(H)[0]
        assert len(idx), "no hold event"
        i0, i1 = int(idx[0]), int(idx[-1])
        # the grasped object's slot: the one whose relative position equals rel_a[:, 0] (demo_rel's g)
        gi = int(np.argmin([np.abs((pos[:, k] - ee) - rel_a[:, 0]).max() for k in range(pos.shape[1])]))
        print("[raao-check] %s demo %s: '%s'; %d frames, held rows %d..%d (%d segments); g = %s; fallback %s" % (
            c, j, str(z["language"]), len(H), i0, i1, int(np.sum(np.diff(H.astype(int)) == 1) + int(H[0])), FC.ENT[gi], info_o["fallback"]))
        print("[raao-check]   in-hand offset at onset (object - eef, cm): %s" % info_o.get("offset_cm"))
        for name, i in (("grasp event (first held row)", i0), ("release (last held row)", i1)):
            ref = rel_a[i, 1] - rel_a[i, 0]                                  # target - grasped object, from the same row
            print("[raao-check]   %-30s row %3d  eef %s  object role (cm) %s | target channel RAA (target-eef) %s  RAA-o (target-held) %s  "
                  "reference target-object-now %s  RAA-o minus reference (slip) %s" % (
                      name, i, cm(ee[i]), cm(rel_o[i, 0]), cm(rel_a[i, 1]), cm(rel_o[i, 1]), cm(ref), cm(rel_o[i, 1] - ref)))
        pre = ~H
        assert np.array_equal(rel_o[:, 0], rel_a[:, 0]), "object-role channel differs"
        assert np.array_equal(rel_o[pre, 1], rel_a[pre, 1]), "a not-held frame's target channel differs"
        d = np.linalg.norm((rel_o[H, 1] - rel_a[H, 1])[:, :], axis=1)
        print("[raao-check]   held rows: |RAA-o - RAA| target channel min %.1f median %.1f max %.1f cm; object-role channel and pre-hold frames identical (asserted)" % (
            100 * d.min(), 100 * np.median(d), 100 * d.max()))
        # the text the policy sees at those rows
        for name, i in (("grasp", i0), ("release", i1)):
            print("[raao-check]   prompt at %-8s RAA: %r | RAA-o: %r" % (
                name, RV.chan_text(T16.given(str(z["language"]))[1], rel_a[i, 1], True), RV.chan_text(T16.given(str(z["language"]))[1], rel_o[i, 1], True)))
    # unit test of held_target_rel on a synthetic episode with two holds
    n = 12
    ee = np.zeros((n, 3))
    ee[:, 0] = np.arange(n) * 0.01
    pg = ee + np.array([0.03, 0.0, -0.05])
    pr = np.tile(np.array([0.5, 0.2, 0.0]), (n, 1))
    h = np.zeros(n, bool)
    h[3:7], h[9:11] = True, True
    out = RV.held_target_rel(pg, pr, ee, h)
    assert np.allclose(out[~h], (pr - ee)[~h])
    assert np.allclose(out[h], (pr - pg)[h]), out[h]                         # rigid offset: target minus the object, exactly
    assert np.allclose(RV.held_target_rel(pg, pr, ee, np.zeros(n, bool)), pr - ee)
    print("[raao-check] synthetic unit test ok (offset fixed per hold, no hold = unchanged)")
    print("[raao-check] done")


if __name__ == "__main__":
    main()

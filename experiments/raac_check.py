"""RAA-c (docs/SCAFFOLD_RESULTS.md 77) channel-tensor unit check, CPU, openpi venv, reads only the rendered demos' npz and chan files.
For two demos (default K5:0 and K1:0) it builds every frame's training row three ways and asserts:
  * the state vector of frame i = [rel x, y, z (metres, float32), role id] with rel = RAA-o's channel of the frame's phase (object role
    before the hold, target - held object while held) and role id 1 / 2 by phase; the vector is 4 numbers and carries no eef state;
  * rounding that vector to whole cm reproduces EXACTLY the numbers of RAA-o's text channel at the same frame (the same information, only
    unrounded), and the prompt of an RAA-c frame is the role phrase with NO suffix = RAA-o's prompt with its ". rel: ..." removed;
  * a frame with no channel (routed=False / rel None) is [0, 0, 0, 0]; the object role id differs from the target's.
It prints the vector at the grasp event, the release and a mid-transport row.

    python experiments/raac_check.py [--demos K5:0+K1:0]
"""
import argparse
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sc_rv as RV                                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="K5:0+K1:0")
    a = ap.parse_args()
    T16 = RV.t16()
    X = RV.lex(types.SimpleNamespace(root=RV.RV_ROOT, suffix="", arm="raac"))
    by_c, _ = RV.roles_of()
    for item in a.demos.split("+"):
        c, j = item.split(":")
        f = RV.demo_path(RV.RV_ROOT, c, int(j))
        z = dict(np.load(f))
        info = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
        rel_c = RV.demo_rel(z, c, f, types.SimpleNamespace(root=RV.RV_ROOT, arm="raac"), X, T16, by_c, info)
        rel_o = RV.demo_rel(z, c, f, types.SimpleNamespace(root=RV.RV_ROOT, arm="raao"), X, T16, by_c, dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0))
        assert np.array_equal(rel_c, rel_o), "raac channel values differ from raao's"
        H = np.asarray(T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]), bool)
        P = T16.given(str(z["language"]))
        assert P is not None, "demo not routed"
        txt_o = list(RV.frames_of(z, c, "raao", X, T16, by_c, rel_o))
        txt_c = list(RV.frames_of(z, c, "raac", X, T16, by_c, rel_c))
        assert len(txt_o) == len(txt_c) == len(H)
        vecs = np.stack([RV.chan_state(rel_c[i, int(H[i])], bool(H[i]), True) for i in range(len(H))])
        assert vecs.shape == (len(H), RV.CHAN_DIM) and vecs.dtype == np.float32
        for i in range(len(H)):
            assert txt_c[i][2] == P[int(H[i])], (i, txt_c[i][2])                       # no suffix: the role phrase itself
            assert txt_o[i][2] == RV.chan_text(P[int(H[i])], rel_o[i, int(H[i])], bool(H[i])), i
            assert vecs[i, 3] == (2.0 if H[i] else 1.0)
            cm = RV.chan_cm(vecs[i, :3])
            want = RV.chan_cm(rel_o[i, int(H[i])])
            assert cm == want, (i, cm, want)
            assert "rel:" in txt_o[i][2] and "rel:" not in txt_c[i][2]
            assert np.array_equal(txt_c[i][0], txt_o[i][0]) and np.array_equal(txt_c[i][1], txt_o[i][1]), "images differ"
        assert np.all(RV.chan_state(None, False, False) == 0) and RV.chan_state(rel_c[0, 0], False, False).shape == (4,)
        idx = np.nonzero(H)[0]
        i0, i1 = int(idx[0]), int(idx[-1])
        im = (i0 + i1) // 2
        print("[raac-check] %s demo %s '%s': %d frames, held rows %d..%d; vectors (m, m, m, role):" % (c, j, str(z["language"]), len(H), i0, i1))
        for name, i in (("pre-hold row 0", 0), ("grasp event", i0), ("mid transport", im), ("release", i1)):
            print("[raac-check]   %-15s row %3d  state %s  | RAA-o prompt: %r" % (name, i, np.round(vecs[i], 4).tolist(), txt_o[i][2]))
        print("[raac-check]   all %d frames: whole-cm rounding of the vector == RAA-o's text numbers; prompts = role phrase, no suffix; images identical" % len(H))
    print("[raac-check] done")


if __name__ == "__main__":
    main()

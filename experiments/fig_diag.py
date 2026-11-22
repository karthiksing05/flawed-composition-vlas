"""Diagnostic printout of the selection signatures' raw quantities per trial (no selection).  usage: fig_diag.py DIR/stem [DIR/stem ...]"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_frames as FR                                                                  # noqa: E402

for sk in sys.argv[1:]:
    d, f = sk.split("/")
    p = os.path.join(FR.FF, d, f)
    L = FR.load_episodes(p, FR.task_of(sk))
    print("==", sk, flush=True)
    for e in L:
        tk = e["tk"]
        h = np.flatnonzero(tk["held"])
        r = int(h[-1]) if len(h) else None
        o = tk["obj"]
        s = "trial %2d succ %d bound %s steps %d held_rows %d" % (e["trial"], e["success"], e["bound"], e["steps"], len(h))
        if r is not None:
            fin = tk["final"]
            s += " | rel obj xy to ENT0 %.3f ENT3 %.3f | to tgt %.3f | lift %.3f | z_obj_end %.3f" % (
                np.linalg.norm(o[r][:2] - fin[0][:2]), np.linalg.norm(o[r][:2] - fin[3][:2]), np.linalg.norm(o[r][:2] - tk["tgt"][r][:2]),
                float(o[h, 2].max() - o[0, 2]), float(o[r, 2]))
        print(s, flush=True)

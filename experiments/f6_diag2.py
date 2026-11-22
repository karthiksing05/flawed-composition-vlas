"""F6 diagnosis part 2 (docs/notes/f6_diag.md, "RAA family"): the same classification (f6_diag.feat / summ) on the registered F6_b (tid 8) and
F6_a (tid 9) tracks of the arms SECTION 6 tests.  Registered dirs are read only; output results/flawed/fig_frames/f6_diag2.json + stdout.  numpy, via sbatch."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f6_diag as D                                                                      # noqa: E402

RES = D.RES
# (label, dir under results/flawed, file stem)
ARMS = [
    ("RAA-L full (wal)", "wal", "full_{t}"), ("RAA-L noM", "wal", "noM_{t}"), ("RAA-L noR", "wal", "noR_{t}"),
    ("RAA-sL full (wasl, dev)", "wasl", "full_{t}"), ("RAA-sL noM (wasl)", "wasl", "noM_{t}"), ("RAA-sL noR (wasl)", "wasl", "noR_{t}"),
    ("RAA-sL (raasw5 set 2)", "raasw5/b_raasl", "full_{t}"), ("RAA-sL (raasw6 set 3)", "raasw6/b_raasl", "full_{t}"),
    ("RAA-sL (raasw5r set 4)", "raasw5r/b_raasl", "full_{t}"),
    ("W4 (raasw5 set 2)", "raasw5/b_w4", "full_{t}"),
    ("W5 full (raasw5 set 2)", "raasw5/b_w5", "full_{t}"), ("W5 noM (raasw5 set 2)", "raasw5/b_w5", "noM_{t}"), ("W5 noR (raasw5 set 2)", "raasw5/b_w5", "noR_{t}"),
    ("W5 full (raasw6 set 3)", "raasw6/b_w5", "full_{t}"),
    ("W5 full (raasw5r set 4)", "raasw5r/b_w5", "full_{t}"), ("W5 noM (raasw5r set 4)", "raasw5r/b_w5", "noM_{t}"), ("W5 noR (raasw5r set 4)", "raasw5r/b_w5", "noR_{t}"),
    ("I (raasw5 ref)", "raasw5/b_ref", "pi05_I_{t}"), ("rgp (raasw5 ref)", "raasw5/b_ref", "pi05_rgp_{t}"),
    ("I (raasw5r ref)", "raasw5r/b_ref", "pi05_I_{t}"), ("rgp (raasw5r ref)", "raasw5r/b_ref", "pi05_rgp_{t}"),
    ("A0 (raaf dev)", "raaf/dev_A0/wa", "full_{t}"), ("A (raaf dev)", "raaf/dev_A/wa", "full_{t}"), ("B (raaf dev)", "raaf/dev_B/wa", "full_{t}"),
    ("A+ (raaf dev)", "raaf/dev_Aplus/wa", "full_{t}"), ("A+ctrl (raaf dev)", "raaf/dev_Actrl/wa", "full_{t}"),
]


def main():
    out = {}
    for lab, d, f in ARMS:
        for t, cell in ((8, "F6_b"), (9, "F6_a")):
            stem = os.path.join(RES, d, f.format(t=t))
            if not os.path.exists(stem + "_track.npz") or not os.path.exists(stem + ".json"):
                if t == 8:
                    print("-- no track:", lab, stem, flush=True)
                continue
            try:
                E = D.episodes(stem)
                F = [D.feat(e) for e in E]
            except Exception as ex:                                                      # noqa: BLE001
                print("-- fail:", lab, cell, type(ex).__name__, ex, flush=True)
                continue
            S = D.summ(F)
            out["%s | %s" % (lab, cell)] = S
            c = S["classes"]
            fl, su = S["fail"], S["succ"]
            print("%-30s %s n=%3d succ %.3f | cls %s | rel_dxy succ %s fail %s | end_dxy fail %s | rel_ee_z fail %s | shift10 fail %s | end_over_3cm %s" % (
                lab, cell, S["n"], S["success"], json.dumps(c), su["rel_dxy"], fl["rel_dxy"], fl["end_dxy"], fl["rel_ee_z"], fl.get("shift10"),
                fl.get("end_over_3cm")), flush=True)
    with open(os.path.join(RES, "fig_frames", "f6_diag2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

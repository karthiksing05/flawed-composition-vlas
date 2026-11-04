"""Place diagnostic T3 (docs/PLACE_DIAG.md 1.3; m3_prereg pp_place): on P_min, is the target term exactly blind to which object is held?
Held demo rows of the tasks below: all object slots set equal to the held object's, then oi changed, comparing the velocity (exact: all zero).
A control moves the foot by 1 cm and must change the velocity.  CPU.  -> results/flawed/pp_place_t3.json

    python experiments/pp_place_t3.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as PT                                                     # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OBJ = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1"]
torch.set_num_threads(1)
out = dict(seeds={}, exact=True, control_alive=True)
for sd in (10, 11, 12):
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{sd}", "P.npz"))
    m.eval()
    assert not m.tsize, "P_min has tsize off"
    per = {}
    for task in ("K1", "K6"):
        b, *_ = PT.rows(torch.device("cpu"), [task])
        keep = torch.nonzero(b["held"] > 0.5)[:, 0][:256]
        b = {k: v[keep] for k, v in b.items()}
        B = len(keep)
        idx = torch.arange(B)
        g = torch.Generator().manual_seed(sd)
        x, t = torch.randn(B, m.H, 7, generator=g), torch.rand(B, generator=g)
        hold = b["pos"][idx, b["oi"]].clone()
        y = b["yaw"][idx, b["oi"]].clone()
        # PLACE_DIAG 2, amendment 1: never the target's own slot (the first version overwrote the bowl at K6, leaving its control dead)
        others = [e for e in OBJ if e != FC.ENT[int(b["ti"][0])]]
        for e in others:
            j = FC.ENT.index(e)
            b["pos"][:, j], b["yaw"][:, j] = hold, y
        with torch.no_grad():
            v0 = m.velocity(b, x, t)
            diffs = {}
            for e in others:
                b2 = dict(b, oi=torch.full_like(b["oi"], FC.ENT.index(e)))
                diffs[e] = float((m.velocity(b2, x, t) - v0).abs().max())
            b3 = dict(b, pos=b["pos"].clone())
            for e in others:
                b3["pos"][:, FC.ENT.index(e), 0] += 0.01
            ctl = float((m.velocity(b3, x, t) - v0).abs().max())
        per[task] = dict(rows=B, max_abs_diff=diffs, control_max_abs_diff=ctl)
        out["exact"] = out["exact"] and all(v == 0.0 for v in diffs.values())
        out["control_alive"] = out["control_alive"] and ctl > 0.0
    out["seeds"][str(sd)] = per
    print("[t3]", sd, json.dumps(per), flush=True)
print("T3 exact", out["exact"], "control alive", out["control_alive"], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "flawed", "pp_place_t3_a1.json"), "w", encoding="utf-8"), indent=1)

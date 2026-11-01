"""Summarise fa1o_close.py (section 5.27; descriptive): per (model, layout, outcome) the medians / rates of the close-window fields.
-> results/flawed/a1o_close.json

    python experiments/fa1o_close_sum.py
"""
import glob
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed")
MED = ("first_close", "closed_frac", "z_close", "z_end", "ee_dz", "ee_dxy", "obj_disp")
RATE = ("touch_any", "touch_both")


def main():
    g = {}
    for f in sorted(glob.glob(os.path.join(OUT, "a1o_close", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for e in d["episodes"]:
            g.setdefault((d["name"], d["layout"], e["cls"]), []).append(e)
    out = {}
    for k, E in sorted(g.items()):
        r = dict(n=len(E))
        if k[2] != "never_closed":
            for m in MED:
                r[m] = float(np.median([e[m] for e in E]))
            for m in RATE:
                r[m] = float(np.mean([e[m] for e in E]))
            r["reopen_rate"] = float(np.mean([e["reopen"] is not None for e in E]))
        out["|".join(k)] = r
        print(f"[fa1o] {k[0]:4s} {k[1]:8s} {k[2]:12s} {json.dumps({a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items()})}", flush=True)
    with open(os.path.join(OUT, "a1o_close.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

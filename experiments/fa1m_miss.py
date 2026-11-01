"""FLAWED_COMPOSITION prototype A, stage 1m DIAGNOSTIC (section 5.25; descriptive): what a missed cheese grasp looks like.  From the
traced episodes (results/flawed/a1k_trace, fa1k_trace.py summaries), per (model, object, layout) and outcome (missed = closed without
a hold, success): the finger contacts at the first close command (both / one / none), the end-effector height above the object at
that step (object frame z, cm) and the step of the first close.
-> results/flawed/a1m_miss.json

    python experiments/fa1m_miss.py
"""
import glob
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed")


def main():
    g = {}
    for f in sorted(glob.glob(os.path.join(OUT, "a1k_trace", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for e in d["episodes"]:
            cls = "miss" if e["hold_class"] == "closed_no_hold" else ("success" if e["success"] else None)
            if cls is None or "grasp_touch" not in e:
                continue
            g.setdefault((d["name"], d["object"], d["layout"], cls), []).append(e)
    out = {}
    for k, E in sorted(g.items()):
        t = np.array([e["grasp_touch"] for e in E], bool)
        z = np.array([e["grasp_ee_obj"][2] for e in E]) * 100
        st = np.array([e["first_close"] for e in E])
        r = dict(n=len(E), touch_both=float(t.all(1).mean()), touch_one=float((t.any(1) & ~t.all(1)).mean()), touch_none=float((~t.any(1)).mean()),
                 z_cm_med=float(np.median(z)), z_cm_iqr=[float(np.percentile(z, 25)), float(np.percentile(z, 75))], first_close_med=float(np.median(st)))
        out["|".join(k)] = r
        print(f"[fa1m] miss {' '.join(f'{x:20s}' if i == 1 else f'{x:8s}' for i, x in enumerate(k))} {json.dumps({a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items()})}", flush=True)
    with open(os.path.join(OUT, "a1m_miss.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

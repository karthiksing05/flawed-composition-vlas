"""FLAWED_COMPOSITION 5.6, descriptive diagnosis of stage 1b's place failures (after the pa1b read; not a test).  For each Ab episode
(3 seeds) on the pick-and-place tasks with g = 1: the held object's end position relative to the target reference (horizontal
distance, height above it), whether it is still held at the end, and the share of episodes in each class: placed (s), still held at
the end, released within 3 cm (horizontal) of the target, released elsewhere.  Also the known task's rest height (a1_calib) for the
pairing.  CPU.  -> results/flawed/a1b_diag.json

    python experiments/fa1b_diag.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    fam = [t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6]
    res = {}
    for t in sorted(fam, key=lambda t: (t["compound"][0] != "K", int(t["compound"][1:]))):
        o, tg = t["object"], t["target"]
        rows = []
        for s in (0, 1, 2):
            f = os.path.join(OUT, "a1b_roll", f"Ab_s{s}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                if not (e["success"] or e.get("bound") == o):
                    continue
                p, r = np.array(e["final"][o]), np.array(e["final"][tg])
                rows.append(dict(s=bool(e["success"]), held_end=bool(e["held_end"]), dxy=float(np.linalg.norm(p[:2] - r[:2])),
                                 dz=float(p[2] - r[2]), dx=float(p[0] - r[0]), dy=float(p[1] - r[1])))
        fail = [x for x in rows if not x["s"]]
        c = cal.get(f"{o}|{tg}", {})
        res[t["compound"]] = dict(
            object=o, target=tg, n_g=len(rows), placed=float(np.mean([x["s"] for x in rows])) if rows else None,
            fail_held_end=float(np.mean([x["held_end"] for x in fail])) if fail else None,
            fail_released_near=float(np.mean([(not x["held_end"]) and x["dxy"] <= 0.03 for x in fail])) if fail else None,
            fail_released_far=float(np.mean([(not x["held_end"]) and x["dxy"] > 0.03 for x in fail])) if fail else None,
            fail_dxy_median=float(np.median([x["dxy"] for x in fail])) if fail else None,
            fail_dz_median=float(np.median([x["dz"] for x in fail])) if fail else None,
            fail_dxdy_median=[float(np.median([x["dx"] for x in fail])), float(np.median([x["dy"] for x in fail]))] if fail else None,
            succ_dz_median=float(np.median([x["dz"] for x in rows if x["s"]])) if any(x["s"] for x in rows) else None,
            calib_dz=c.get("dz_mean"))
        print(t["compound"], json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[t["compound"]].items()}), flush=True)
    json.dump(res, open(os.path.join(OUT, "a1b_diag.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

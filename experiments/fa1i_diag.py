"""FLAWED_COMPOSITION prototype A, stage 1i DIAGNOSTIC summary (section 5.15; descriptive only): classifies every traced K5 / K6
episode (results/flawed/a1i_trace, fa1i_trace.py) and pools seeds per model group.
Radius r = max(RAD_t, calibrated dxy_rest + 0.03) (the relaxed test's; fa1_read), rest dz = calibrated dz_rest (a1_calib.json).
Classes (first that applies):
  success
  never_held            the hold never turned on
  held_at_end_arrived   still held at the end, min dxy while held <= r      (arrived, never released)
  held_at_end_short     still held at the end, never within r
  dropped_in_transit    last release with dxy > r
  released_fell         released within r, final dz < rest dz - 0.04         (fell off / through)
  released_tipped       released within r, final tilt > 30 deg
  released_resting      released within r, came to rest, predicate false
Per group: class shares, median release height above rest (release dz - rest dz), median release dxy, median tilt at release and
at the end, open commands while held, held segments, re-run reproduction rate.
-> results/flawed/a1i_diag.json

    python experiments/fa1i_diag.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1_read as R1                                                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed")
CLASSES = ["success", "never_held", "held_at_end_arrived", "held_at_end_short", "dropped_in_transit", "released_fell",
           "released_tipped", "released_resting"]


def classify(s, r):
    if s["success"]:
        return "success"
    if not s.get("hold_steps"):
        return "never_held"
    if s.get("held_at_end"):
        return "held_at_end_arrived" if s["min_dxy_held"] is not None and s["min_dxy_held"] <= r else "held_at_end_short"
    if s.get("release_dxy") is None or s["release_dxy"] > r:
        return "dropped_in_transit"
    if s["final_dz"] < s["rest_dz"] - 0.04:
        return "released_fell"
    if s["final_tilt"] > 30.0:
        return "released_tipped"
    return "released_resting"


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    files = sorted(glob.glob(os.path.join(OUT, "a1i_trace", "*.json")))
    groups = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        t = n2["tasks"][d["tid"]]
        r = max(R1.RAD[t["target"]], float(np.mean(cal[f"{t['object']}|{t['target']}"]["dxy"])) + R1.DXY_PAD)
        g = groups.setdefault((d["name"], t["compound"]), dict(rows=[], reproduced=[], radius=r))
        g["reproduced"].append(d["reproduced"])
        for e in d["episodes"]:
            s = e["summary"]
            s["class"] = classify(s, r)
            g["rows"].append(s)
    out = {}
    med = lambda v: (float(np.median(v)) if len(v) else None)                  # noqa: E731
    for (name, comp), g in sorted(groups.items()):
        rows = g["rows"]
        rel = [x for x in rows if x.get("release_dxy") is not None]
        out[f"{name}|{comp}"] = dict(
            n=len(rows), radius=g["radius"], reproduced=float(np.mean(g["reproduced"])),
            classes={c: float(np.mean([x["class"] == c for x in rows])) for c in CLASSES},
            release_height_above_rest=med([x["release_dz"] - x["rest_dz"] for x in rel]),
            release_dxy=med([x["release_dxy"] for x in rel]),
            release_tilt=med([x["release_tilt"] for x in rel]), final_tilt=med([x["final_tilt"] for x in rows if "final_tilt" in x]),
            max_drop_after_release=med([x["max_drop_after"] for x in rel]),
            open_cmds_while_held=float(np.mean([x.get("open_cmds_while_held", 0) for x in rows])),
            held_segments=float(np.mean([x.get("held_segments", 0) for x in rows])),
            min_dxy_held=med([x["min_dxy_held"] for x in rows if x.get("min_dxy_held") is not None]),
            failed_release_height_above_rest=med([x["release_dz"] - x["rest_dz"] for x in rel if not x["success"]]),
            success_release_height_above_rest=med([x["release_dz"] - x["rest_dz"] for x in rel if x["success"]]))
        o = out[f"{name}|{comp}"]
        print(f"[fa1i] {name:9s} {comp} n={o['n']} repro={o['reproduced']:.2f} " +
              " ".join(f"{c}={v:.2f}" for c, v in o["classes"].items() if v > 0) +
              f" | rel_h={o['release_height_above_rest']} rel_dxy={o['release_dxy']} rel_tilt={o['release_tilt']} "
              f"fin_tilt={o['final_tilt']} opens={o['open_cmds_while_held']:.2f} segs={o['held_segments']:.2f} "
              f"fail_h={o['failed_release_height_above_rest']} ok_h={o['success_release_height_above_rest']}", flush=True)
    with open(os.path.join(OUT, "a1i_diag.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(classes=CLASSES, groups=out, note="descriptive; re-runs of registered K5 / K6 familiar episodes"), fh, indent=1)


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION prototype A, stage 1o DIAGNOSTIC (section 5.29; descriptive only): does the generated data cover the new-layout
cheese starts that fail?  In the layout (table) frame of n2_tasks.json: the cheese's clearance to the wine rack (centre distance minus
the sampler's radii, fa1g_plan.MOVR / FIXR) and to the nearest fixture, for
  (a) the new-layout evaluation states of the cheese tasks (K6 tid 11, N1 tid 17), with Ao1's success per state (a1o_roll), and
  (b) every generator attempt for K6 (a1g_gen_K6, a1k_gen_K6, a1l_cgen_K6): layout found, grasp option chosen, kept.
-> results/flawed/a1o_cover.json

    python experiments/fa1o_cover.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1g_plan as FG                                                      # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")
BINS = (-1.0, 0.03, 0.06, 0.10, 0.20, 9.0)


def clear(p_cheese, fix, fixn):
    c = {fn: float(np.linalg.norm(np.asarray(p_cheese) - np.asarray(fix[i])) - FG.FIXR[fn] - FG.MOVR["cream_cheese_1"])
         for i, fn in enumerate(fixn)}
    return c


def binof(x):
    for i in range(len(BINS) - 1):
        if BINS[i] <= x < BINS[i + 1]:
            return f"{BINS[i]:.2f}-{BINS[i + 1]:.2f}"
    return "out"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Ao1")
    ap.add_argument("--roll", default="a1o_roll")
    ap.add_argument("--seeds", default="21+22+23")
    ap.add_argument("--out", default="a1o_cover.json")
    a = ap.parse_args()
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    fix, fixn = n2["constraints"]["fixture_xy"], n2["fixtures"]
    ci = n2["movable"].index("cream_cheese_1")
    out = {"eval": {}, "gen": {}}
    for tid in (11, 17):
        succ = {}
        for s in [int(x) for x in a.seeds.split("+")]:
            p = os.path.join(OUT, a.roll, f"{a.name}_s{s}_main_{tid}.json")
            reg = json.load(open(p, encoding="utf-8"))
            for v in reg["tasks"].values():
                for x in v["episodes"]:
                    succ.setdefault(int(x["trial"]), []).append(bool(x["success"]))
        rows = {}
        for trial, ok in sorted(succ.items()):
            c = clear(n2["positions"]["new"][trial][ci], fix, fixn)
            b = binof(c["wine_rack_1"])
            r = rows.setdefault(b, dict(states=0, episodes=0, success=0))
            r["states"] += 1
            r["episodes"] += len(ok)
            r["success"] += int(sum(ok))
        out["eval"][str(tid)] = rows
        print(f"[fa1o] cover eval {a.name} tid {tid} by cheese-rack clearance: {json.dumps(rows)}", flush=True)
    for name in ("a1g_gen_K6", "a1k_gen_K6", "a1l_cgen_K6"):
        d = json.load(open(os.path.join(OUT, f"{name}.json"), encoding="utf-8"))
        rows = {}
        for e in d["episodes"]:
            if not e.get("layout"):
                continue
            c = clear(e["positions"][ci], fix, fixn)
            b = binof(c["wine_rack_1"])
            r = rows.setdefault(b, dict(attempts=0, kept=0, kept_episode=0, no_option=0))
            r["attempts"] += 1
            k = e.get("kept")
            if k is None and "kept" not in e and e.get("success") is not None:
                k = "episode" if e["success"] else None
            r["kept"] += int(bool(k))
            r["kept_episode"] += int(k == "episode")
            r["no_option"] += int("option" in e and e["option"] is None)
        out["gen"][name] = rows
        print(f"[fa1o] cover gen {name} by cheese-rack clearance: {json.dumps(dict(sorted(rows.items())))}", flush=True)
    out["name"] = a.name
    with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

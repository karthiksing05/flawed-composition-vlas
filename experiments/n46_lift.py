"""N46 follow-up (descriptive, no verdict): carry height against outcome, from results/flawed/n46_phase.json (experiments/n46_phase.py).
High carry = lift while held > .22 m (K4's cabinet carry is ~.28-.33 m; low-target successes ~.13-.20 m).  Per (set, model, arm, task):
success rate among high- and low-carry grasped episodes, and the share of high carries.  docs/SCAFFOLD_RESULTS.md 51.

    python experiments/n46_lift.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUT = 0.22


def main():
    R = json.load(open(os.path.join(ROOT, "results", "flawed", "n46_phase.json"), encoding="utf-8"))["per_episode"]
    tot = {}
    for k, rows in R.items():
        task = k.split("|")[-1]
        if task not in ("N4", "N6", "N5", "N1", "K4"):
            continue
        g = [r for r in rows if r.get("lift_held") is not None]
        hi = [r for r in g if r["lift_held"] > CUT]
        lo = [r for r in g if r["lift_held"] <= CUT]
        f = lambda v: "%.2f" % (sum(r["success"] for r in v) / float(len(v))) if v else "  - "
        print("%-28s grasped %3d high share %.2f | success high %s (n %2d) low %s (n %2d)" % (
            k, len(g), len(hi) / float(max(1, len(g))), f(hi), len(hi), f(lo), len(lo)), flush=True)
        if task in ("N4", "N6") and k.split("|")[1] in ("pi05", "groot"):
            m = k.split("|")[1]
            t = tot.setdefault(m, [0, 0, 0, 0])
            t[0] += len(hi); t[1] += sum(r["success"] for r in hi); t[2] += len(lo); t[3] += sum(r["success"] for r in lo)
    for m, t in tot.items():
        print("POOLED N4+N6 all sets/arms %s: high %d (success %.2f) low %d (success %.2f)" % (
            m, t[0], t[1] / float(max(1, t[0])), t[2], t[3] / float(max(1, t[2]))), flush=True)


if __name__ == "__main__":
    main()

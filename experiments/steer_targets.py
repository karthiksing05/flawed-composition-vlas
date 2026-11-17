"""SR 74: the T0 targets, read from the named rollout JSONs (CPU, openpi venv).  Selection rule, fixed here and in the prereg row before any
probe rollout: for each policy ONE canonical measured row per new task = the row with the largest pre-registered state set in
reports/predictor_targets.md Table A / A1 (scene 1) or Table B (scene 2); ties go to the latest SR section.  Per-task success is the mean
over that row's episodes in the per-task client file <dir>/<prefix>_<tid>.json (tids N1-N6 = 16 18 20 22 24 26).  Every policy's pooled
N1-N6 mean is checked against the number printed in the SR table (tolerance .002); a mismatch is reported, not repaired.

    python experiments/steer_targets.py [--out results/flawed/steer/targets.json]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL = os.path.join(ROOT, "results", "flawed")
NEW = dict(N1=16, N2=18, N3=20, N4=22, N5=24, N6=26)
# policy -> (family, directory, file prefix, state set as written in Table A / A1, n states per task, SR section, pooled mean printed in the SR)
S1 = {
    "pi05_I":     ("pi05",   "tvpb_rg",  "pi05_I",    "new k116-155", 40, "55", .454),
    "groot_I":    ("groot",  "tv_g",     "groot_I",   "new k56-115", 60, "49", .219),
    "pi0_I":      ("pi0",    "s1_cross", "pi0_I",     "new k236-275", 40, "65", .021),
    "smolvla_I":  ("smolvla", "s1_cross", "smolvla_I", "new k236-275", 40, "65", .021),
    "pi05_tv":    ("pi05",   "tv_fresh", "pi05_tv",   "fresh k36-55 (ties main 4-23 at 20; later section)", 20, "47/48", .700),
    "pi05_tvp":   ("pi05",   "tvpb_rg",  "pi05_tvp",  "new k116-155", 40, "55", .838),
    "pi05_tvpb":  ("pi05",   "tvpb_rg",  "pi05_tvpb", "new k116-155", 40, "55", .938),
    "groot_tv":   ("groot",  "tv_g",     "groot_tv",  "new k56-115", 60, "49", .406),
    "groot_tvb":  ("groot",  "groot_box", "groot_tvb", "new k156-195", 40, "56", .383),
    "r0":         ("pi05",   "sc_t37",   "r0_main",   "dev 4-23 familiar", 20, "37.1", .025),
    "r1":         ("pi05",   "sc_t37",   "r1_main",   "dev 4-23 familiar", 20, "37.1", .225),
    "r2":         ("pi05",   "sc_t37",   "r2_main",   "dev 4-23 familiar", 20, "37.1", .425),
    "r2b":        ("pi05",   "sc_t37",   "r2b_main",  "dev 4-23 familiar", 20, "37.2", .150),
    "r2w":        ("pi05",   "sc_t37",   "r2w_main",  "dev 4-23 familiar", 20, "37.2", .242),
    "r2l":        ("pi05",   "sc_t37",   "r2l_main",  "dev 4-23 familiar", 20, "43.1", .417),
}
# policies with no measured new-pairing row: probed anyway, excluded from T0-T2
NO_TARGET = {"xvla_I": ("xvla", "SR 45.3 has X-VLA only on the F7 bowl family and in the atlas excess table; no N1-N6 success")}
# scene 2, as printed in Table B (N1, N2 per task are in the SR; no job needed): policy -> (family, state set, n, SR section, N1, N2)
S2 = {
    "pi05_I":   ("pi05", "new seeds 1000000 (tie at 60; latest section)", 60, "72", .000, .000),
    "pi0_I":    ("pi0",  "new seeds 990000 (tie at 60; latest section with pi0 I)", 60, "69", .000, .000),
    "pi05_tvp": ("pi05", "k4-63", 60, "50/54", .000, .000),
}


def succ(path):
    d = json.load(open(path, encoding="utf-8"))
    eps = [e for v in d["tasks"].values() for e in v["episodes"]]
    return float(np.mean([bool(e["success"]) for e in eps])), len(eps)


def main():
    out = os.path.join(FL, "steer", "targets.json")
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    assert not os.path.exists(out), "never overwrite " + out
    cells, checks, missing = [], {}, []
    for pol, (fam, d, pre, sset, n, sec, printed) in S1.items():
        vals = []
        for task, tid in NEW.items():
            f = os.path.join(FL, d, "%s_%d.json" % (pre, tid))
            if not os.path.exists(f):
                missing.append(f)
                continue
            s, ne = succ(f)
            vals.append(s)
            cells.append(dict(scene=1, policy=pol, family=fam, task=task, success=s, n_episodes=ne, state_set=sset, n_states=n, sr=sec,
                              source=os.path.relpath(f, ROOT)))
        pooled = float(np.mean(vals)) if len(vals) == 6 else None
        checks[pol] = dict(pooled=pooled, printed=printed, ok=None if pooled is None else abs(pooled - printed) <= .002)
    for pol, (fam, sset, n, sec, a, b) in S2.items():
        for task, v in (("N1", a), ("N2", b)):
            cells.append(dict(scene=2, policy=pol, family=fam, task=task, success=v, n_episodes=None, state_set=sset, n_states=n, sr=sec,
                              source="reports/predictor_targets.md Table B (printed)"))
    res = dict(rule=__doc__.split("    python")[0].strip(), cells=cells, checks=checks, missing=missing,
               no_target={k: dict(family=v[0], why=v[1]) for k, v in NO_TARGET.items()})
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for pol, c in checks.items():
        print("%-10s pooled %s printed %.3f %s" % (pol, "NA" if c["pooled"] is None else "%.3f" % c["pooled"], c["printed"],
                                                   "OK" if c["ok"] else "MISMATCH"), flush=True)
    print("missing files: %s" % missing, flush=True)
    print("wrote %s (%d cells)" % (out, len(cells)), flush=True)


if __name__ == "__main__":
    main()

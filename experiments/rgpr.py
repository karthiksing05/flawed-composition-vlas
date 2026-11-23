"""RGPR (docs/notes/rgpr.md; m3_prereg tag rgpr): "ReGuide plus our referent resolver" on the relational F7 families.  Evaluation only, no training.

The arm rgpr is rg_comp.py's rgp (ReGuide RG-perc, SR 52 / 70, unchanged) whose referent bowl comes from the F7R resolver R (fx_r.py,
unchanged) instead of rgp's positional noun rule.  Everything else is rgp's: the VLA gets the task's ORIGINAL instruction (no unique-name
prompt) and UNMODIFIED images (no role-view paint, no memory slot), ReGuide's perception, registration and transport are the same code, the
destination is rg_roles' noun.  It differs from rc (SR 70) in exactly the VLA-facing parts of rc: rc also rewrites the prompt, paints the
other bowl magenta in both cameras, and (when the destination is the plate) runs the memory slot.  rg_comp.py is NOT edited: this file
imports it and patches two names in-process (Ctx, fx_r.Front), so rgp / rc / every existing arm behave exactly as before.

    python experiments/rgpr.py client --arm rgpr <rg_comp.py client arguments>     (LIBERO venv; one rgc task per process)
    python experiments/rgpr.py read --phase smoke|main --out X.json [--roll-rgpr DIR] [--roll-base DIR]    (CPU)

Read: paired clustered bootstrap (B = 2000, seed 0; the 40 F7 states resampled together across the four F7 tasks, each state keeping its
(rgp, rgpr, rc) outcomes).  r1 = rgpr - rgp and r2 = rc - rgpr on the F7 pool (mean of relL, relR, pos, neg), frozen pi0.5.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import rg_comp as RC                                                          # noqa: E402

ROLL_BASE = os.path.join(ROOT, "results", "flawed", "rg_comp_roll")
ROLL_RGPR = os.path.join(ROOT, "results", "flawed", "rgpr_roll")
F7_TIDS = (0, 1, 2, 3)
EQ = 0.10


def _patch():
    import fx_r as XR

    class _Front(XR.Front):
        """The resolver without the VLA-facing paint: Front only paints for arm in (f7r, oracle)."""

        def __init__(self, env, arm, parsed, ckpt=""):
            super().__init__(env, "rgpr", parsed, ckpt)
            assert not self.paint_on

    class _Ctx(RC.Ctx):
        def __init__(self, arm, task, meta):
            super().__init__("rc", task, meta)                              # parsed, front_on, rg_on, roles: as rc
            self.prompt = task["language"]                                    # the ORIGINAL instruction (rc: F7R's unique-name prompt)
            self.slot_on, self.Slot = False, None                             # no memory slot (rc: on when the destination is the plate)
            assert self.front_on and self.rg_on and self.prompt == task["language"]
    XR.Front = _Front
    RC.Ctx = _Ctx


def client():
    i = sys.argv.index("--arm")
    assert sys.argv[i + 1] == "rgpr", sys.argv[i + 1]
    sys.argv[i + 1] = "rc"                                                    # the registered (arm, task) cells of rg_comp include rc on 0-3
    tid = int(sys.argv[sys.argv.index("--tasks") + 1])
    assert tid in F7_TIDS, tid
    _patch()
    return RC.client_main()


def _lab1(ci):
    if ci[1] is None:
        return "MISSING"
    return "RESOLVER HELPS" if ci[1] > 0 else ("RESOLVER HURTS" if ci[2] < 0 else "NO CLEAR DIFFERENCE")


def _lab2(ci):
    if ci[1] is None:
        return "MISSING"
    if ci[1] > -EQ and ci[2] < EQ:
        return "EQUIVALENT (+-.10)"
    return "RC LEADS" if ci[1] > 0 else ("RGPR LEADS" if ci[2] < 0 else "NO CLEAR DIFFERENCE")


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    meta = json.load(open(RC.META, encoding="utf-8"))
    rows = meta["states"]["f7"][a.phase]
    n = len(rows)
    roots = {"rgp": a.roll_base, "rc": a.roll_base, "rgpr": a.roll_rgpr}
    D, X = {}, {}
    for arm, root in roots.items():
        for t in F7_TIDS:
            d = RC.load_run(root, "pi05", arm, t, rows)
            D[(arm, t)] = d
            X[(arm, t)] = np.array([float(d[k]["success"]) if (d and k in d) else np.nan for k in rows], float)
    missing = [f"{arm}/{t}" for (arm, t), d in D.items() if d is None or len(d) < n]
    res = dict(phase=a.phase, rolls=roots, B=RC.B, n_states=n, missing=missing, success={}, contrasts={}, mechanism={})
    mean = lambda x, I: float(np.nanmean(x[I]))                               # noqa: E731
    cell = {"relL": (0,), "relR": (1,), "pos": (2,), "neg": (3,), "SP": (0, 1), "NG": (2, 3), "F7pool": (0, 1, 2, 3)}

    def fn(arm, c):
        return lambda I, arm=arm, c=c: float(np.mean([mean(X[(arm, t)], I[0]) for t in cell[c]]))
    for c in cell:
        for arm in roots:
            res["success"][f"{arm}/{c}"] = RC.bootm(fn(arm, c), [n])
    for c in ("F7pool", "SP", "NG", "relL", "relR", "pos", "neg"):
        r1 = RC.bootm(lambda I, c=c: fn("rgpr", c)(I) - fn("rgp", c)(I), [n])
        r2 = RC.bootm(lambda I, c=c: fn("rc", c)(I) - fn("rgpr", c)(I), [n])
        rg = RC.bootm(lambda I, c=c: fn("rc", c)(I) - fn("rgp", c)(I), [n])
        res["contrasts"][c] = dict(r1_rgpr_minus_rgp=r1, r1_label=_lab1(r1), r2_rc_minus_rgpr=r2, r2_label=_lab2(r2), rc_minus_rgp=rg)
    # share of the rc - rgp gap closed by rgpr (point only)
    g, c1 = res["contrasts"]["F7pool"]["rc_minus_rgp"][0], res["contrasts"]["F7pool"]["r1_rgpr_minus_rgp"][0]
    res["gap_closed_share_F7pool"] = (c1 / g) if g else None
    # mechanism: binding, resolver, registration, grasp (descriptive)
    for arm in roots:
        for t in F7_TIDS:
            d = D[(arm, t)] or {}
            sides = [r.get("side") or {} for r in d.values()]
            bound = [(s.get("inst") or {}).get("bound_referent") for s in sides]
            bound = [b for b in bound if b is not None]
            reg = [(s.get("reg") or {}).get("bound_err_cm") for s in sides]
            reg = [x for x in reg if x is not None]
            rs = [((s.get("front") or {}).get("resolve") or {}).get("resolver_correct") for s in sides]
            rs = [x for x in rs if x is not None]
            gr = [r.get("grasped") for r in d.values()]
            nb = [x for x in gr if x in (RC.B1, RC.B2)]
            tr = [(s.get("rg") or {}).get("transports") for s in sides]
            tr = [x for x in tr if x is not None]
            res["mechanism"][f"{arm}/{RC.TASKS[t][0]}"] = dict(
                n=len(d), errors=sum(1 for r in d.values() if r.get("error")),
                bound_referent=(f"{sum(bool(b) for b in bound)}/{len(bound)}" if bound else None),
                resolver_correct=(f"{sum(bool(x) for x in rs)}/{len(rs)}" if rs else None),
                bound_err_cm_median=float(np.median(reg)) if reg else None,
                C_referent_grasp=(f"{sum(x == RC.B1 for x in nb)}/{len(nb)}" if nb else None),
                transports_mean=float(np.mean(tr)) if tr else None,
                prompt_rewritten=None if not sides else None)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    print("[rgpr read] missing:", missing)
    for k, v in res["success"].items():
        print("  %-14s %s" % (k, [round(x, 3) if x is not None else None for x in v]))
    for c, v in res["contrasts"].items():
        print("  %-7s r1 %s %s | r2 %s %s | rc-rgp %s" % (c, [round(x, 3) for x in v["r1_rgpr_minus_rgp"] if x is not None], v["r1_label"],
                                                         [round(x, 3) for x in v["r2_rc_minus_rgpr"] if x is not None], v["r2_label"],
                                                         [round(x, 3) for x in v["rc_minus_rgp"] if x is not None]))
    print("  gap closed share (F7 pool, point):", res["gap_closed_share_F7pool"])
    for k, v in res["mechanism"].items():
        print("  ", k, v)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        sys.argv.pop(1)
        sys.argv.insert(1, "client")
        i = sys.argv.index("--arm")
        assert sys.argv[i + 1] == "rgpr"
        return client()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read",))
    ap.add_argument("--phase", default="main", choices=("smoke", "main"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--roll-rgpr", dest="roll_rgpr", default=ROLL_RGPR)
    ap.add_argument("--roll-base", dest="roll_base", default=ROLL_BASE)
    read(ap.parse_args())


if __name__ == "__main__":
    main()

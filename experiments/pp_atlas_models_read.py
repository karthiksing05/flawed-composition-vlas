"""The break atlas's registered read (experiments/pp_atlas.py read; docs/BREAK_ATLAS.md 2-3; m3_prereg pp_atlas), unchanged, for
any list of VLAs.  pp_atlas.py is imported and never edited: the stage rule (classify), the per-episode tables (stage_table), the
partners (PAIR), the estimator, the bootstrap (B = 2000, rng seed 0, tasks and states drawn jointly) and the mapping are its own.
The only differences from pp_atlas.read: the systems come from --models; agreement with N2's recorded rollouts is computed only where
results/flawed/n2_roll/<model>_main_<tid>.json exists (pi05, groot); the output file is --out.

    LIBERO venv, CPU:  python experiments/pp_atlas_models_read.py --models pi0,xvla [--out results/flawed/pp_atlas_more_read.json]
    check:            --models pi05,groot --out <scratch>.json reproduces results/flawed/pp_atlas_read.json's pi05 / groot entries
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

STEMS = dict(pmin=["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"], servo=["servo_s10_{tid}"])


def read_models(models, out):
    n2 = PA.tasks()
    ks = n2["states"]["main"]
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    NEW, KNOWN, STAGES, PAIR = PA.NEW, PA.KNOWN, PA.STAGES, PA.PAIR
    res = {}
    for sysname in models:
        stems = STEMS.get(sysname, [sysname + "_{tid}"])
        res[sysname] = {}
        for lay in ("familiar", "new"):
            RATE, per, subs, geos, rec_agree = {}, {}, {}, {}, []
            for c in NEW + KNOWN:
                t = tid[(c, lay)]
                A = {s: np.zeros(len(ks)) for s in STAGES}
                subs[c], geos[c] = {}, []
                for st in stems:
                    S, X, G, SU = PA.stage_table(os.path.join(PA.OD, st.format(tid=t["tid"])), t, ks)
                    for j, s in enumerate(S):
                        A[s][j] += 1.0 / len(stems)
                    for x in X:
                        if x:
                            subs[c][x] = subs[c].get(x, 0) + 1
                    geos[c] += G
                    f = os.path.join(PA.OUT, "n2_roll", f"{sysname}_main_{t['tid']}.json")
                    if sysname not in STEMS and os.path.exists(f):
                        Er = {int(e["trial"]): bool(e["success"]) for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]}
                        rec_agree += [float(Er[k] == s) for k, s in zip(ks, SU) if k in Er]
                RATE[c] = A
                per[c] = {s: float(A[s].mean()) for s in STAGES}

            def excess(names, ti, si):
                v = []
                for i in ti:
                    c = NEW[i]
                    ko, kt = PAIR[c]
                    v.append(sum(RATE[c][n][si] - 0.5 * (RATE[ko][n][si] + RATE[kt][n][si]) for n in names).mean())
                return float(np.mean(v))

            def est(names):
                rng = np.random.default_rng(0)
                e = excess(names, np.arange(len(NEW)), np.arange(len(ks)))
                v = [excess(names, rng.choice(len(NEW), len(NEW)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
                lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
                return dict(E=e, ci=[lo, hi], breaks=bool(lo > 0.0 and e > 0.10))

            r = dict(stages={s: est((s,)) for s in STAGES[1:]}, groups={g: est(n) for g, n in PA.GROUP.items()}, per_task=per, subtypes=subs)
            r["verdict"] = {(True, False): "WITHIN-CHUNK", (False, True): "CROSS-CHUNK", (True, True): "BOTH", (False, False): "NEITHER"}[(r["groups"]["within"]["breaks"], r["groups"]["cross"]["breaks"])]
            reached = {}
            for grp, cs in (("new", NEW), ("known", KNOWN)):
                left = float(len(cs))
                reached[grp] = {}
                for s in ("GRASP", "HANDOFF", "TRANSPORT"):
                    x = sum(per[c][s] for c in cs)
                    reached[grp][s] = x / left if left > 0 else None
                    left -= x
                x = sum(per[c]["RELEASE"] + per[c]["PHYSICS"] + per[c]["success"] for c in cs)
                reached[grp]["arrived_or_success"] = x / len(cs)
                y = sum(per[c]["RELEASE"] for c in cs)
                reached[grp]["RELEASE"] = y / x if x > 0 else None
            r["reached"] = reached
            r["release_geometry"] = {}
            for c in NEW + KNOWN:
                G = [g for g in geos[c] if g["stage"] in ("RELEASE", "PHYSICS", "success")]
                if G:
                    r["release_geometry"][c] = dict(n=len(G), ee=np.median([g["ee"] for g in G], 0).tolist(), obj=np.median([g["obj"] for g in G], 0).tolist(), tilt=float(np.median([g["tilt"] for g in G])))
            if rec_agree:
                r["agrees_with_recorded"] = float(np.mean(rec_agree))
            res[sysname][lay] = r
            print(sysname, lay, r["verdict"], json.dumps({k: r[k] for k in ("stages", "groups", "reached")}), flush=True)
            print(sysname, lay, "per_task", json.dumps(per), "agree", r.get("agrees_with_recorded"), flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, flush=True)


def compare(out, ref):
    """Every number in `out` equals `ref`'s for the systems both hold (the pi05 / groot reproduction check)."""
    a, b = json.load(open(out, encoding="utf-8")), json.load(open(ref, encoding="utf-8"))
    bad = []

    def walk(x, y, path):
        if isinstance(x, dict):
            for k in x:
                if k not in y:
                    bad.append(path + "/" + k + " missing")
                else:
                    walk(x[k], y[k], path + "/" + k)
        elif isinstance(x, list):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif isinstance(x, float):
            if not abs(x - y) <= 1e-9:
                bad.append(f"{path}: {x} vs {y}")
        elif x != y:
            bad.append(f"{path}: {x} vs {y}")

    for s in a:
        walk(a[s], b[s], s)
    print("compare", ref, "mismatches", len(bad), bad[:10], flush=True)
    return not bad


def smoke(models, d, tids, ks):
    """Per-episode stages on a smoke directory (no estimator): checks that the tracks exist, line up and classify."""
    T = {t["tid"]: t for t in PA.tasks()["tasks"]}
    for m in models:
        for i in tids:
            S, X, G, SU = PA.stage_table(os.path.join(d, f"{m}_{i}"), T[i], ks)
            print("smoke", m, T[i]["key"], "stages", S, "subtypes", X, flush=True)


if __name__ == "__main__":
    if "--smoke" in sys.argv:          # --smoke DIR --models pi0,xvla --tids 0,16 --states 0,1,2,3
        ap = argparse.ArgumentParser()
        ap.add_argument("--smoke")
        ap.add_argument("--models")
        ap.add_argument("--tids")
        ap.add_argument("--states", default="0,1,2,3")
        a = ap.parse_args()
        smoke(a.models.split(","), a.smoke, [int(x) for x in a.tids.split(",")], [int(x) for x in a.states.split(",")])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated: pi05, groot, pi0, xvla, smolvla, pmin, servo")
    ap.add_argument("--out", default=os.path.join(PA.OUT, "pp_atlas_more_read.json"))
    ap.add_argument("--compare", default="", help="a registered read to check against (pp_atlas_read.json)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert os.path.basename(out) != "pp_atlas_read.json", "never overwrite the registered read"
    read_models([m for m in a.models.split(",") if m], out)
    if a.compare:
        sys.exit(0 if compare(out, a.compare) else 1)

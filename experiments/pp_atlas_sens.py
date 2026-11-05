"""Break atlas sensitivity read (docs/BREAK_ATLAS.md 6; m3_prereg pp_atlas_sens).  CPU, no new episodes.  -> results/flawed/pp_atlas_sens.json
  selftest   the two variants on synthetic episodes
  read       both variants on results/flawed/pp_atlas tracks, with the descriptive table
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                        # noqa: E402

SLACK = 0.02
VARIANTS = ("frozen", "S1_carried", "S2_final")


def carried(held, ee, obj, a6):
    """S1's hold signal: Held, or after the onset the gripper is commanded closed and stays at this episode's own grip distance (+ SLACK)."""
    idx = np.flatnonzero(held)
    if not len(idx):
        return held.copy()
    gap = np.linalg.norm(ee - obj, axis=1)
    g = float(np.median(gap[idx]))
    out = held.copy()
    out[idx[0]:] |= (gap[idx[0]:] <= g + SLACK) & (a6[idx[0]:] > 0)
    return out


def classify(v, r, d, success, bound, oe):
    held = r["held"]
    if v == "S1_carried":
        return PA.classify(carried(held, r["ee"], r["obj"], r["a6"]), d, success, bound, oe)[0]
    if v == "frozen":
        return PA.classify(held, d, success, bound, oe)[0]
    s = PA.classify(held, d, success, bound, oe)[0]
    return "RELEASE" if s in ("HANDOFF", "TRANSPORT") and float(d[-1]) <= PA.ARRIVE else s    # S2: an object that ended within the radius arrived, and was let go away from the target


def selftest():
    T = 20
    on = np.zeros(T, bool)
    on[5:10] = True
    obj = np.zeros((T, 3))
    far = np.full(T, 0.40)
    ee = np.zeros((T, 3))
    a_shut, a_open = np.ones(T), -np.ones(T)
    ct = dict(held=on, ee=ee, obj=obj, a6=a_shut)
    nr = np.where(np.arange(T) < 10, 0.40, 0.05)
    assert classify("frozen", ct, nr, False, "o", "o") == "HANDOFF"
    assert classify("S1_carried", ct, nr, False, "o", "o") == "RELEASE"                # carried on with Held off, still shut at the end
    assert classify("S1_carried", dict(ct, a6=a_open), nr, False, "o", "o") == "HANDOFF"
    away = ee.copy()
    away[10:] = [0, 0, 0.3]
    assert classify("S1_carried", dict(ct, ee=away), nr, False, "o", "o") == "HANDOFF"
    assert classify("S2_final", ct, nr, False, "o", "o") == "RELEASE"
    assert classify("S2_final", ct, far, False, "o", "o") == "HANDOFF"
    assert classify("S2_final", ct, far, True, "o", "o") == "success"
    print("selftest ok 7", flush=True)


def med(x):
    return float(np.median(x)) if len(x) else None


def read():
    n2 = PA.tasks()
    ks = n2["states"]["main"]
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    res = {}
    for sysname, stems in (("pi05", ["pi05_{tid}"]), ("groot", ["groot_{tid}"]), ("pmin", ["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"]), ("servo", ["servo_s10_{tid}"])):
        res[sysname] = {}
        for lay in ("familiar", "new"):
            RATE = {v: {} for v in VARIANTS}
            desc = {}
            for c in PA.NEW + PA.KNOWN:
                t = tid[(c, lay)]
                for v in VARIANTS:
                    RATE[v][c] = {s: np.zeros(len(ks)) for s in PA.STAGES}
                rows = []
                for st in stems:
                    stem = os.path.join(PA.OD, st.format(tid=t["tid"]))
                    R = json.load(open(stem + ".json", encoding="utf-8"))
                    E = R["eps"] if "eps" in R else [e for w in R["tasks"].values() for e in w["episodes"]]
                    for j, (e, r) in enumerate(zip(E, PA.load_track(stem + "_track.npz"))):
                        assert int(e["trial"]) == ks[j]
                        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
                        got = {}
                        for v in VARIANTS:
                            got[v] = classify(v, r, d, bool(e["success"]), e.get("bound"), t["object"])
                            RATE[v][c][got[v]][j] += 1.0 / len(stems)
                        if got["frozen"] in ("HANDOFF", "TRANSPORT"):
                            idx = np.flatnonzero(r["held"])
                            rows.append(dict(d0=float(d[idx[0]]), dmin=float(d[idx].min()), last=float(d[idx[-1]]), final=float(d[-1]), anystep=float(d[idx[0]:].min()),
                                             runs=int(1 + np.sum(np.diff(np.flatnonzero(r["held"])) > 1)), s1=got["S1_carried"], s2=got["S2_final"]))
                desc[c] = dict(n=len(rows), **{k: med([x[k] for x in rows]) for k in ("d0", "dmin", "last", "final", "anystep", "runs")},
                               dmin_le_15=sum(x["dmin"] <= 0.15 for x in rows), dmin_le_20=sum(x["dmin"] <= 0.20 for x in rows), final_le_10=sum(x["final"] <= 0.10 for x in rows),
                               moved_s1=sum(x["s1"] == "RELEASE" for x in rows), moved_s2=sum(x["s2"] == "RELEASE" for x in rows))

            def excess(v, names, ti, si):
                out = []
                for i in ti:
                    c = PA.NEW[i]
                    ko, kt = PA.PAIR[c]
                    out.append(sum(RATE[v][c][n][si] - 0.5 * (RATE[v][ko][n][si] + RATE[v][kt][n][si]) for n in names).mean())
                return float(np.mean(out))

            def est(v, names):
                rng = np.random.default_rng(0)
                e = excess(v, names, np.arange(len(PA.NEW)), np.arange(len(ks)))
                b = [excess(v, names, rng.choice(len(PA.NEW), len(PA.NEW)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
                lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
                return dict(E=e, ci=[lo, hi], breaks=bool(lo > 0.0 and e > 0.10))

            r = dict(desc=desc)
            for v in VARIANTS:
                g = {k: est(v, n) for k, n in PA.GROUP.items()}
                r[v] = dict(groups=g, stages={s: est(v, (s,)) for s in PA.STAGES[1:]},
                            per_task={c: {s: float(RATE[v][c][s].mean()) for s in PA.STAGES} for c in PA.NEW + PA.KNOWN},
                            verdict={(True, False): "WITHIN-CHUNK", (False, True): "CROSS-CHUNK", (True, True): "BOTH", (False, False): "NEITHER"}[(g["within"]["breaks"], g["cross"]["breaks"])])
            if sysname in ("pi05", "groot"):
                same = all(r[v]["verdict"] == r["frozen"]["verdict"] for v in VARIANTS)
                r["call"] = "ROBUST" if same else "INSTRUMENT-DEPENDENT"
            res[sysname][lay] = r
            print(sysname, lay, r.get("call", "no verdict"), {v: r[v]["verdict"] for v in VARIANTS}, flush=True)
            for v in VARIANTS:
                print(" ", v, json.dumps({k: [round(r[v]["groups"][k]["E"], 3), [round(x, 3) for x in r[v]["groups"][k]["ci"]]] for k in ("within", "cross")}), flush=True)
                print("   per_task", json.dumps({c: {s[:4]: round(x, 2) for s, x in r[v]["per_task"][c].items() if x} for c in PA.NEW}), flush=True)
            print("  desc", json.dumps({c: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in desc[c].items()} for c in PA.NEW}), flush=True)
    json.dump(res, open(os.path.join(PA.OUT, "pp_atlas_sens.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    selftest()
    if len(sys.argv) > 1 and sys.argv[1] == "read":
        read()

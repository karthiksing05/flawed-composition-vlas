"""SR 89 step-2 read (docs/notes/scene2_plan.md section 3; docs/notes/sr89_pred.txt = experiments/prereg/s2r_pred.txt; m3_prereg tag s2r).  Numpy only (openpi venv; compute node).
Layout (results/flawed/sr89/main/): <label>/<arm>_<tid>.json for the wa_s2r.py labels  raasls2 (RAA-sL-S2, where_own 0, arm full), w5 (W5-S2, full), w5_noR, w5_noM, w5_noC, step0 (kitchen RAA-sL, where_own 0, full);
ref/pi05_I_<tid>.json, ref/pi05_rgp_<tid>.json (s2rg_roll.sh on the registered states); tids 0-3 = K1 K2 N1 N2; trials 4-63 (60 states, seed 1,100,000 + k).
Estimator: paired bootstrap over STATES only (tasks fixed: only two new pairings), B = 2000, numpy default_rng(0), one index draw shared by every contrast; CI = 2.5 / 97.5 percentile.
    python experiments/s2r_read.py read [--tag T]      -> results/flawed/sr89/s2r_read[_T].json (refuses to overwrite)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "flawed", "sr89", "main")
KEYS = {0: "K1", 1: "K2", 2: "N1", 3: "N2"}
TRIALS = list(range(4, 64))
B = 2000
LABELS = ("raasls2", "w5", "w5_noR", "w5_noM", "w5_noC", "step0")
ARMOF = {"raasls2": "full", "w5": "full", "w5_noR": "noR", "w5_noM": "noM", "w5_noC": "noC", "step0": "full"}


def path(label, tid):
    if label in ("I", "rgp"):
        return os.path.join(BASE, "ref", "pi05_%s_%d.json" % (label, tid))
    return os.path.join(BASE, label, "%s_%d.json" % (ARMOF[label], tid))


def load(label, tid):
    p = path(label, tid)
    if not os.path.exists(p):
        return None
    eps = json.load(open(p, encoding="utf-8"))["tasks"][str(tid)]["episodes"]
    by = {int(e["trial"]): e for e in eps}
    assert sorted(by) == TRIALS, (label, tid, sorted(by))
    return by


def wa(label, tid):
    p = path(label, tid).replace(".json", "_wa.json")
    if not os.path.exists(p):
        return None
    return {int(e["trial"]): e for e in json.load(open(p, encoding="utf-8"))["eps"]}


def succ(label):
    out = {}
    for t in KEYS:
        by = load(label, t)
        if by is None:
            return None
        out[t] = np.array([float(bool(by[k]["success"])) for k in TRIALS])
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


class Boot:
    def __init__(self):
        self.idx = np.random.default_rng(0).integers(0, len(TRIALS), (B, len(TRIALS)))

    def delta(self, S, a, b, tids):
        """paired state-only bootstrap of mean over tids and states of (arm a - arm b)"""
        d = np.mean([S[a][t] - S[b][t] for t in tids], 0)                     # [60]
        v = d[self.idx].mean(1)
        return dict(a=float(np.mean([S[a][t] for t in tids])), b=float(np.mean([S[b][t] for t in tids])), delta=float(d.mean()),
                    ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    def rate(self, x):
        v = np.asarray(x)[self.idx].mean(1)
        return float(np.mean(x)), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def lab_sup(d):
    return "SUPPORTED" if d["ci"][0] > 0 else ("REVERSED" if d["ci"][1] < 0 else "INCONCLUSIVE")


def lab_ni(d, m):
    return "NON-INFERIOR" if d["ci"][0] > -m else "NOT NON-INFERIOR"


def lab_eq(d, m=0.10):
    return "EQUIVALENT" if (d["ci"][0] > -m and d["ci"][1] < m) else "NOT EQUIVALENT"


def read(a):
    sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
    outp = os.path.join(ROOT, "results", "flawed", "sr89", "s2r_read%s.json" % sfx)
    assert not os.path.exists(outp), "never overwrite: " + outp
    S, miss = {}, []
    for l in LABELS + ("I", "rgp"):
        s = succ(l)
        if s is None:
            miss.append(l)
        else:
            S[l] = s
    R = dict(missing=miss, cells={}, contrasts={}, gates={}, descriptive={})
    bt = Boot()
    K, N = (0, 1), (2, 3)
    for l, s in S.items():
        R["cells"][l] = {KEYS[t]: dict(k=int(s[t].sum()), n=len(TRIALS), rate=round(float(s[t].mean()), 3), wilson95=wilson(int(s[t].sum()), len(TRIALS))) for t in KEYS}
        for nm, ts in (("K1+K2", K), ("N1+N2", N)):
            x = np.mean([s[t] for t in ts], 0)
            r, ci = bt.rate(x)
            R["cells"][l][nm] = dict(rate=round(r, 3), ci=[round(c, 3) for c in ci])
    C = R["contrasts"]

    def add(name, a_, b_, ts, rule, **kw):
        if a_ in S and b_ in S:
            d = bt.delta(S, a_, b_, ts)
            d["label"] = rule(d)
            bad = [x for x in (a_, b_) if x in ("w5", "raasls2") and R["cells"][x]["K1+K2"]["rate"] < 0.70]      # the FIT gate (pred file): an arm below .70 on K1+K2 is UNTESTABLE
            if bad and not name.startswith("desc"):
                d["label"] = "UNTESTABLE (fit gate: %s K1+K2 < .70); raw label: %s" % (bad, d["label"])
            C[name] = d
        else:
            C[name] = dict(label="NOT RUN (missing %s)" % [x for x in (a_, b_) if x not in S])
    add("A1 PRIMARY W5-S2 - RAA-sL-S2, N1+N2", "w5", "raasls2", N, lab_sup)
    add("A2a W5-S2 - rgp, N1+N2 (NI at -.10)", "w5", "rgp", N, lambda d: lab_ni(d, 0.10))
    add("A2b W5-S2 - rgp, N1+N2 (beats)", "w5", "rgp", N, lab_sup)
    for arm, nm in (("raasls2", "RAA-sL-S2"), ("w5", "W5-S2")):
        add("V1 %s - I, N1+N2" % nm, arm, "I", N, lab_sup)
        add("V1 %s - I, K1+K2 (NI at -.05)" % nm, arm, "I", K, lambda d: lab_ni(d, 0.05))
        if "V1 %s - I, N1+N2" % nm in C and "V1 %s - I, K1+K2 (NI at -.05)" % nm in C:
            n_ok = C["V1 %s - I, N1+N2" % nm]["ci"][0] > 0
            k_ok = C["V1 %s - I, K1+K2 (NI at -.05)" % nm]["ci"][0] > -0.05
            C["V1 %s" % nm] = dict(label="FIX WORKS" if (n_ok and k_ok) else ("PARTIAL" if (n_ok or k_ok) else "FAILS"))
    add("AK W5-S2 - RAA-sL-S2, K1+K2 (NI at -.05)", "w5", "raasls2", K, lambda d: lab_ni(d, 0.05))
    add("L1 W5-S2 - noC, N1+N2", "w5", "w5_noC", N, lab_sup)
    add("L2 noC - I, N1+N2 (EQUIVALENT iff inside +-.10)", "w5_noC", "I", N, lab_eq)
    add("D1' W5-S2 - noR, N1+N2", "w5", "w5_noR", N, lab_sup)
    add("D2' W5-S2 - noM, N1+N2", "w5", "w5_noM", N, lab_sup)
    add("DK W5-S2 - noR, K1+K2 (NI at -.05)", "w5", "w5_noR", K, lambda d: lab_ni(d, 0.05))
    add("DKM W5-S2 - noM, K1+K2 (NI at -.05)", "w5", "w5_noM", K, lambda d: lab_ni(d, 0.05))
    add("desc RAA-sL-S2 - step-0, N1+N2", "raasls2", "step0", N, lambda d: "descriptive")
    add("desc RAA-sL-S2 - step-0, K1+K2", "raasls2", "step0", K, lambda d: "descriptive")
    # ---- gates and engagement (full = W5-S2)
    G = R["gates"]
    if "w5" in S:
        G["fit K1+K2 >= .70 (W5-S2 full)"] = dict(value=R["cells"]["w5"]["K1+K2"]["rate"], ok=bool(R["cells"]["w5"]["K1+K2"]["rate"] >= 0.70))
    if "raasls2" in S:
        G["fit K1+K2 >= .70 (RAA-sL-S2)"] = dict(value=R["cells"]["raasls2"]["K1+K2"]["rate"], ok=bool(R["cells"]["raasls2"]["K1+K2"]["rate"] >= 0.70))
    # per-episode records of the full arm
    Wq, mshare, cm = {}, {}, []
    for t in KEYS:
        w = wa("w5", t)
        if w is None:
            continue
        Wq[t] = w
        tq = [[q for q in e["queries_diag"] if q["role"] == "target"] for e in (w[k] for k in TRIALS)]
        allq = [q for qs in tq for q in qs]
        mshare[KEYS[t]] = dict(target_queries=len(allq), held_est_share=round(float(np.mean([q["held_est"] for q in allq])) if allq else 0.0, 3),
                               episodes_with_held_target_est=int(sum(any(q["held_est"] for q in qs) for qs in tq)))
        cm += [float(np.median([q["err_chan_cm"] for q in qs if q["err_chan_cm"] is not None])) for qs in tq if any(q["err_chan_cm"] is not None for q in qs)]
    if mshare:
        G["M engaged: target-phase held-estimate share >= .20 in at least one task (else D2' UNTESTABLE)"] = dict(per_task=mshare, ok=bool(any(v["held_est_share"] >= 0.20 for v in mshare.values())))
    if cm:
        G["channel at test: held-target channel median (per-episode medians pooled) < 2 cm"] = dict(value_cm=round(float(np.median(cm)), 2), ok=bool(np.median(cm) < 2.0))
    ncorr = {}
    for t in (2, 3):
        w = wa("w5_noR", t)
        if w is not None:
            ncorr[KEYS[t]] = [bool(w[k]["res"]["correct"]) for k in TRIALS]
    if ncorr:
        diff = 1.0 - float(np.mean([np.mean(v) for v in ncorr.values()]))
        G["R engaged: share of N episodes where noR's plate differs from the parsed plate >= .30 (else D1' trivially empty)"] = dict(value=round(diff, 3), ok=bool(diff >= 0.30), per_task={k: round(1 - float(np.mean(v)), 3) for k, v in ncorr.items()})
    rr = {}
    for l in ("w5", "raasls2", "w5_noM", "w5_noC"):
        for t in KEYS:
            w = wa(l, t)
            if w is not None:
                rr.setdefault(l, []).extend(bool(w[k]["res"]["correct"]) for k in TRIALS)
    R["descriptive"]["resolver_correct_share"] = {l: round(float(np.mean(v)), 3) for l, v in rr.items()}
    # ---- D3a / D3b on episodes where the removed module is not engaged (paired by state)
    def sub(arm_a, arm_b, tids, keep):
        xs, ys = [], []
        for t in tids:
            wa_ = wa(arm_a, t)
            if wa_ is None:
                return None
            for i, k in enumerate(TRIALS):
                if keep(wa_[k]):
                    xs.append(S[arm_a][t][i] - S[arm_b][t][i])
        if len(xs) < 20:
            return dict(label="UNTESTABLE (n = %d < 20 episodes)" % len(xs), n=len(xs))
        x = np.array(xs)
        v = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
        d = dict(delta=float(x.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], n=len(xs))
        d["label"] = lab_eq(d)
        return d
    if "w5_noR" in S and "w5" in S:
        C["D3a noR - W5-S2 on the episodes where noR chose the parsed plate (all 4 tasks)"] = sub("w5_noR", "w5", tuple(KEYS), lambda e: bool(e["res"]["correct"]))
    if "w5_noM" in S and "w5" in S:
        C["D3b noM - W5-S2 on the episodes where noM withheld no query (all 4 tasks)"] = sub("w5_noM", "w5", tuple(KEYS), lambda e: not any(q["withheld"] for q in e["queries_diag"]))
    # ---- gated dissociation
    need = ("D1' W5-S2 - noR, N1+N2", "D2' W5-S2 - noM, N1+N2", "D3a noR - W5-S2 on the episodes where noR chose the parsed plate (all 4 tasks)", "D3b noM - W5-S2 on the episodes where noM withheld no query (all 4 tasks)")
    if all(n in C and C[n] and "label" in C[n] for n in need):
        gok = all(v.get("ok", False) for k, v in G.items() if k.startswith(("M engaged", "R engaged", "fit K1+K2 >= .70 (W5", "channel")))
        d1, d2 = C[need[0]]["label"] == "SUPPORTED", C[need[1]]["label"] == "SUPPORTED"
        e3 = C[need[2]]["label"] == "EQUIVALENT" and C[need[3]]["label"] == "EQUIVALENT"
        if not gok:
            C["GATED DISSOCIATION"] = dict(label="UNTESTABLE (a gate failed: %s)" % [k for k, v in G.items() if not v.get("ok", True)])
        else:
            C["GATED DISSOCIATION"] = dict(label="DISSOCIATION SHOWN" if (d1 and d2 and e3) else ("PARTIAL" if (d1 or d2) else "NOT SHOWN"))
    # ---- descriptive: wrong-plate / right-grasp rates from the tracks are read in docs; here the per-cell tables above
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["read"])
    ap.add_argument("--tag", default="")
    read(ap.parse_args())

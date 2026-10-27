"""PLAN_SOLVE S3a read (docs/PLAN_S3.md; pre-registration experiments/prereg/ps3a.py, ledger m3_prereg tag ps3a).  CPU (chunks env).

Inputs
  results/bind/s3/conf/<set>_<binder>/*.npz   first-frame binder confidence + mask quality vs live oracle (s2_sandbox --conf-only)
  results/bind/s2/roll/<set>_binder           S2's adapter + S1b runs (seeds 0-3)
  results/bind/s0/roll/<set>_{zs,oracle}      S0's runs (seeds 0-7)
  results/bind/s3/roll/<set>_gated            (if run) adapter + S1b + per-call gate, seeds 4-5

1. SELECTION (training cells only; S1b):  for each score in s3_gate.SCORES and each tau in {train unit scores} + {inf}:
     gated train success = per unit: S2 deploy runs if score >= tau else S0 zs runs (zero masks = zs exactly; S0 parity 24/24);
     damage D = paired task-bootstrap diff of gated - zs (seeds 0-3).
   Admissible: D (point) >= -0.02.  Chosen: the admissible (score, tau) with the largest train gated-on fraction; ties -> SCORES
   order, then the larger tau.  coverage 0 (tau = inf only) -> no gate.
2. OFFLINE DEV EFFECT (the chosen gate, per-unit first-frame decision): dev coverage; gated-on vs gated-off first-frame mask
   quality; recombined dev rates (gated-on -> S2 deploy runs, gated-off -> S0 zs runs; seeds 0-3), diffs vs zs, closure vs oracle.
3. CLOSED-LOOP CHECK (only if dev coverage >= 0.10 AND gated-on dev object masks mostly inside the target in >= 0.80 of units):
   seeds 4-5, per-call gate; vs S0 zs / oracle at seeds 4-5.  BARS: B1 train damage >= -0.05; B2 dev binding error <= zs (points).
   Without the closed loop the bars are read on the offline recombination and flagged in-sample for B1.
S1c: first-frame mask quality only (no S1c rollouts exist).

    python experiments/s3a_read.py [--record]  -> results/bind/s3a.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s0_read as S0R  # noqa: E402
import s3_gate as G    # noqa: E402

LIST_KEYS = ("bound", "released", "rel_step", "labels", "success", "steps")
ADMIT = -0.02
RD = os.path.join(ROOT, "results", "bind")


def runs(path, sl):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        for k in LIST_KEYS:
            m[k] = m[k][sl]
        out[m["unit"]] = m
    return out


def conf(set_name, binder):
    out = {}
    for f in sorted(glob.glob(os.path.join(RD, "s3", "conf", f"{set_name}_{binder}", "*.npz"))):
        r = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        out[r["unit"]] = r
    return out


def by_task(units_runs):
    per = {}
    for m in units_runs:
        per.setdefault(m["tid"], []).append(m)
    return per


def quality(rows):
    if not rows:
        return dict(units=0)
    return {key: dict(units=len(rows), majority_inside=float(np.mean([r[key]["inside"] > 0.5 for r in rows])),
                      inside=float(np.mean([r[key]["inside"] for r in rows])), iou=float(np.mean([r[key]["iou"] for r in rows])),
                      one_token=float(np.mean([r[key]["n"] <= 1 for r in rows])), empty=float(np.mean([r[key]["n"] == 0 for r in rows])))
            for key in ("obj", "goal")}


def recombine(units, on, dep, zs):
    return [dep[u] if u in on else zs[u] for u in units]


def rates3(per):
    import d4_read as DR
    return {c: DR.rate(S0R.rates(per, c)) for c in ("success", "binding", "motion")}


def closure(r):
    T = sorted(set(r["zs"]) & set(r["oracle"]) & set(r["gated"]))
    w = np.array([len(r["zs"][t]) for t in T], float)
    z, o, g = (np.average([np.mean(r[k][t]) for t in T], weights=w) for k in ("zs", "oracle", "gated"))
    return float((g - z) / (o - z)) if abs(o - z) > 1e-9 else None


def compare(units, gated_runs, zs, orc):
    """rates + paired diffs vs zs (+ closure) for the arm gated_runs over units; zs/orc dicts unit -> run."""
    arms = dict(zs=by_task([zs[u] for u in units]), gated=by_task(gated_runs))
    if orc:
        arms["oracle"] = by_task([orc[u] for u in units])
    res = dict(rates={k: rates3(v) for k, v in arms.items()}, diffs={}, closure={})
    for c in ("success", "binding", "motion"):
        r = {k: S0R.rates(v, c) for k, v in arms.items()}
        res["diffs"][c] = S0R.paired(r["gated"], r["zs"])
        if orc:
            res["closure"][c] = closure(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    S = slice(0, 4)
    dep = {s: runs(os.path.join(RD, "s2", "roll", f"{s}_binder"), S) for s in ("dev", "train")}
    zs = {s: runs(os.path.join(RD, "s0", "roll", f"{s}_zs"), S) for s in ("dev", "train")}
    orc = runs(os.path.join(RD, "s0", "roll", "dev_oracle"), S)
    cf = {s: conf(s, "s1b") for s in ("dev", "train")}
    cfc = {s: conf(s, "s1c") for s in ("dev", "train")}
    # consistency: the conf pass reproduces S2's first-call masks (same frame, deterministic binder)
    agree = [abs(cf["dev"][u]["obj"]["iou"] - dep["dev"][u]["binder_diag"]["obj"]["iou"]) < 1e-6
             for u in cf["dev"] if u in dep["dev"] and dep["dev"][u].get("binder_diag")]
    res = dict(conf_units={s: len(cf[s]) for s in cf}, conf_matches_s2_first_call=dict(n=len(agree), same=int(sum(agree))))
    # 1. selection on training cells
    tr_units = sorted(set(cf["train"]) & set(dep["train"]) & set(zs["train"]))
    zs_tr = by_task([zs["train"][u] for u in tr_units])
    cand = []
    for name in G.SCORES:
        sc = {u: G.score(*cf["train"][u]["conf"], name) for u in tr_units}
        for tau in sorted(set(sc.values())) + [float("inf")]:
            on = {u for u in tr_units if sc[u] >= tau}
            g = by_task(recombine(tr_units, on, dep["train"], zs["train"]))
            d = S0R.paired(S0R.rates(g, "success"), S0R.rates(zs_tr, "success"))
            cand.append(dict(score=name, tau=tau, coverage=len(on) / len(tr_units), damage=d))
    adm = [c for c in cand if c["damage"]["diff"] >= ADMIT]
    best = max(adm, key=lambda c: (c["coverage"], -G.SCORES.index(c["score"]), c["tau"]))
    res["selection"] = dict(train_units=len(tr_units), chosen=best, admissible_n=len(adm), candidates_n=len(cand),
                            frontier={n: sorted([(c["coverage"], c["damage"]["diff"], c["tau"]) for c in cand if c["score"] == n])
                                      for n in G.SCORES})
    # 2. offline dev effect
    dv_units = sorted(set(cf["dev"]) & set(dep["dev"]) & set(zs["dev"]) & set(orc))
    name, tau = best["score"], best["tau"]
    on = {u for u in dv_units if G.score(*cf["dev"][u]["conf"], name) >= tau}
    res["offline_dev"] = dict(units=len(dv_units), coverage=len(on) / len(dv_units), on_units=len(on),
                              tasks_on=len({cf["dev"][u]["tid"] for u in on}),
                              quality_on=quality([cf["dev"][u] for u in sorted(on)]),
                              quality_off=quality([cf["dev"][u] for u in dv_units if u not in on]),
                              quality_all=quality([cf["dev"][u] for u in dv_units]))
    if on:
        res["offline_dev"].update(compare(dv_units, recombine(dv_units, on, dep["dev"], zs["dev"]), zs["dev"], orc))
    res["offline_train_in_sample"] = best["damage"]
    res["s1c_first_frame"] = dict(dev=quality(list(cfc["dev"].values())), train=quality(list(cfc["train"].values())))
    res["s1b_first_frame_train"] = quality(list(cf["train"].values()))
    qo = res["offline_dev"]["quality_on"]
    res["closed_loop_condition"] = bool(res["offline_dev"]["coverage"] >= 0.10 and qo.get("obj", {}).get("majority_inside", 0) >= 0.80)
    # 3. closed-loop check (seeds 4-5), if run
    S2s = slice(4, 6)
    gl = {s: runs(os.path.join(RD, "s3", "roll", f"{s}_gated"), slice(0, 2)) for s in ("dev", "train")}
    if gl["dev"] and gl["train"]:
        zs45 = {s: runs(os.path.join(RD, "s0", "roll", f"{s}_zs"), S2s) for s in ("dev", "train")}
        orc45 = runs(os.path.join(RD, "s0", "roll", "dev_oracle"), S2s)
        du = sorted(set(gl["dev"]) & set(zs45["dev"]) & set(orc45))
        tu = sorted(set(gl["train"]) & set(zs45["train"]))
        cd = compare(du, [gl["dev"][u] for u in du], zs45["dev"], orc45)
        ct = compare(tu, [gl["train"][u] for u in tu], zs45["train"], None)
        res["closed_loop"] = dict(dev_units=len(du), train_units=len(tu), dev=cd, train=ct)
        b1, b2 = ct["diffs"]["success"]["diff"], cd["rates"]["gated"]["binding"]["mean"] - cd["rates"]["zs"]["binding"]["mean"]
        src = "closed_loop"
    else:
        b1 = best["damage"]["diff"]
        b2 = (res["offline_dev"]["rates"]["gated"]["binding"]["mean"] - res["offline_dev"]["rates"]["zs"]["binding"]["mean"]) if on else 0.0
        src = "offline (B1 in-sample)"
    res["bar"] = dict(source=src, B1_damage=b1, B1_pass=bool(b1 >= -0.05), B2_binding_minus_zs=b2, B2_pass=bool(b2 <= 0))
    res["bar"]["verdict"] = "PASS" if res["bar"]["B1_pass"] and res["bar"]["B2_pass"] else "FAIL"
    json.dump(res, open(os.path.join(RD, "s3a.json"), "w"), indent=1)
    p = lambda k: print(f"[s3a read] {k}: {json.dumps(res[k])}", flush=True)  # noqa: E731
    for k in ("conf_units", "conf_matches_s2_first_call"):
        p(k)
    print(f"[s3a read] selection chosen {json.dumps(best)} (admissible {len(adm)}/{len(cand)})", flush=True)
    for n, fr in res["selection"]["frontier"].items():
        print(f"[s3a read] frontier {n}: " + " ".join(f"({c:.2f},{d:+.3f})" for c, d, _ in fr[::max(1, len(fr) // 12)]), flush=True)
    od = {k: v for k, v in res["offline_dev"].items()}
    print(f"[s3a read] offline dev: {json.dumps(od)}", flush=True)
    for k in ("s1c_first_frame", "s1b_first_frame_train", "closed_loop_condition", "bar"):
        p(k)
    if "closed_loop" in res:
        p("closed_loop")
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s3a_gate", bench="libero_90_c", tag="read", seed=0, command="python experiments/s3a_read.py --record",
                            metrics=dict(bar=res["bar"], selection=dict(chosen=best, train_units=len(tr_units)),
                                         offline_dev={k: v for k, v in res["offline_dev"].items() if k != "frontier"},
                                         closed_loop=res.get("closed_loop"), closed_loop_condition=res["closed_loop_condition"],
                                         s1c_first_frame=res["s1c_first_frame"], conf_matches=res["conf_matches_s2_first_call"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S3a binder confidence gate"), script="experiments/s3a_read.py"))


if __name__ == "__main__":
    main()

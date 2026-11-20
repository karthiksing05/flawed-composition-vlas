"""SR 89 step 0 (EXPLORATORY): read the zero-shot probe outputs of experiments/wa_s2.py (results/flawed/sr89/probe/<arm>/<arm>_<tid>.json + _wa.json).
Prints and writes results/flawed/sr89/probe/probe_read.json: per cell success (n, k, rate, Wilson 95%), right-first-mug rate (the mug the oracle track first
holds), plate each mug ends on, and the channel diagnostics (estimate error vs simulator truth, query counts).  Descriptive only; no label.
    python experiments/sr89_probe_read.py [--dirs raasl w5] [--out probe_read.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PD = os.path.join(ROOT, "results", "flawed", "sr89", "probe")
KEYS = {0: "K1", 1: "K2", 2: "N1", 3: "N2"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["raasl", "w5"])
    ap.add_argument("--arm", default="full")
    ap.add_argument("--out", default=os.path.join(PD, "probe_read.json"))
    a = ap.parse_args()
    assert not os.path.exists(a.out), "never overwrite: %s" % a.out
    res = {}
    for d in a.dirs:
        res[d] = {}
        for tid, key in KEYS.items():
            p = os.path.join(PD, d, "%s_%d.json" % (a.arm, tid))
            if not os.path.exists(p):
                res[d][key] = None
                continue
            j = json.load(open(p, encoding="utf-8"))
            eps = j["tasks"][str(tid)]["episodes"]
            n, k = len(eps), sum(1 for e in eps if e["success"])
            cell = dict(n=n, k=k, rate=round(k / n, 3) if n else None, wilson95=wilson(k, n), trials=[e.get("trial") for e in eps],
                        success=[int(bool(e["success"])) for e in eps])
            w = os.path.join(PD, d, "%s_%d_wa.json" % (a.arm, tid))
            if os.path.exists(w):
                s = json.load(open(w, encoding="utf-8"))
                Q = [q for ep in s["eps"] for q in ep.get("queries_diag", [])]
                eo = [q["err_obj_cm"] for q in Q if q["role"] == "object" and q["err_obj_cm"] is not None]
                ec = [q["err_chan_cm"] for q in Q if q["role"] == "target" and q["err_chan_cm"] is not None]
                eoc = [q["err_chan_cm"] for q in Q if q["role"] == "object" and q["err_chan_cm"] is not None]
                cell["channel"] = dict(queries=len(Q), held_queries=sum(1 for q in Q if q["role"] == "target"),
                                       no_channel_queries=sum(1 for q in Q if not q["has"]),
                                       object_chan_err_cm_median=None if not eoc else round(float(np.median(eoc)), 2),
                                       object_chan_err_cm_p90=None if not eoc else round(float(np.percentile(eoc, 90)), 2),
                                       held_target_chan_err_cm_median=None if not ec else round(float(np.median(ec)), 2),
                                       held_target_chan_err_cm_p90=None if not ec else round(float(np.percentile(ec, 90)), 2),
                                       held_target_within_2cm=None if not ec else round(float(np.mean(np.asarray(ec) < 2.0)), 3),
                                       object_pos_err_cm_median=None if not eo else round(float(np.median(eo)), 2))
            t = os.path.join(PD, d, "%s_%d_track.npz" % (a.arm, tid))
            if os.path.exists(t):
                z = np.load(t)
                on = [np.asarray(z["e%d_on" % i]) for i in range(int(z["n"]))]
                # on = the LIBERO On predicate of every mug on every plate (mugs w, y, r x plates 1, 2), rows in s2.MUGS x PLATES order
                cell["on_final_share"] = np.round(np.mean(np.stack(on).astype(float), 0), 3).tolist()
            res[d][key] = cell
        for g, ks in (("K1+K2", ("K1", "K2")), ("N1+N2", ("N1", "N2"))):
            c = [res[d][k] for k in ks if res[d][k]]
            if len(c) == len(ks):
                n, k = sum(x["n"] for x in c), sum(x["k"] for x in c)
                res[d][g] = dict(n=n, k=k, rate=round(k / n, 3), wilson95=wilson(k, n))
        print("[sr89 probe] %-6s " % d + "  ".join("%s %s" % (k, "-" if res[d].get(k) is None else "%d/%d=%.3f" % (res[d][k]["k"], res[d][k]["n"], res[d][k]["rate"]))
                                                   for k in ("K1", "K2", "N1", "N2", "K1+K2", "N1+N2")), flush=True)
        for k in ("K1", "K2", "N1", "N2"):
            if res[d].get(k) and "channel" in res[d][k]:
                print("[sr89 probe] %-6s %s channel %s" % (d, k, json.dumps(res[d][k]["channel"])), flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(note="EXPLORATORY zero-shot probe (SR 89 step 0); descriptive, no prereg row; states seed 1,090,000 + k, k 0-19", cells=res), fh, indent=1)
    print("[sr89 probe] wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()

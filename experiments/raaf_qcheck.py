"""RAA-F Step 0 (docs/notes/raaf_plan.md 2.4): where the registered quantile fractions come from.  openpi venv, numpy only, sbatch only.
Over rows h0 .. e - 1 of every K1-K6 demo (the rows a copy covers): the fraction outside the norm-stats [q01, q99] per xyz dim and for any dim, for
(a) the ORIGINAL actions, (b) the relabelled actions of every accepted copy, and for the state token (rel, rel + delta; dims 0-2): (c) originals, (d) copies.
Also the per-dim action fractions of ALL frames' kept-row held originals.  Reported so the 2% STOP can be read against the demos' own baseline.
    python experiments/raaf_qcheck.py --prep P --delta D --out O        (O must not exist)
"""
import argparse
import json
import os
import pickle

import numpy as np

K = 2.851209548740752
NORM = "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json"

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--delta", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert not os.path.exists(a.out), a.out
P = pickle.load(open(a.prep, "rb"))
D = json.load(open(a.delta, encoding="utf-8"))
ns = json.load(open(NORM, encoding="utf-8"))["norm_stats"]
aq = (np.asarray(ns["actions"]["q01"][:3]), np.asarray(ns["actions"]["q99"][:3]))
sq = (np.asarray(ns["state"]["q01"][:3]), np.asarray(ns["state"]["q99"][:3]))
by = {(d["task"], d["demo"]): d for d in P["demos"]}


def frac(x, q):
    o = (x < q[0][None]) | (x > q[1][None])
    return o


acc = {k: [] for k in ("act_o", "act_c", "tok_o", "tok_c")}
per = {}
for dd in D["demos"]:
    d = by[(dd["task"], dd["demo"])]
    h0, e = dd["h0"], dd["e"]
    for cp in dd["copies"]:
        dl = np.asarray(cp["delta"])
        ao = d["actions"][h0:e, :3]
        to = d["rel"][h0:e]
        acc["act_o"].append(frac(ao, aq)); acc["act_c"].append(frac(ao + K * dl[None], aq))
        acc["tok_o"].append(frac(to, sq)); acc["tok_c"].append(frac(to + dl[None], sq))
        per.setdefault(dd["task"], []).append(frac(ao + K * dl[None], aq))
out = {}
for k, v in acc.items():
    o = np.concatenate(v)
    out[k] = dict(rows=int(len(o)), any_dim=float(o.any(1).mean()), per_dim=o.mean(0).tolist())
out["actions_copies_per_task_any_dim"] = {c: float(np.concatenate(v).any(1).mean()) for c, v in sorted(per.items())}
out["q01_q99_actions"] = [aq[0].tolist(), aq[1].tolist()]
out["q01_q99_state"] = [sq[0].tolist(), sq[1].tolist()]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("[raaf_qcheck] " + json.dumps(out, indent=1))

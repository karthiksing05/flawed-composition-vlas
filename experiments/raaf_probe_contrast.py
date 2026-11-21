"""RAA-F probe contrast (docs/notes/raaf_sel.md): route - A0 difference of the pooled N + K learned where-share and of in.prefix.
Each route's observations come from its own rollout (the 3rd held query differs per policy), so they are paired by cluster (set, cluster id = the
dev state), not by observation: one cluster resample per set (a2_read.boot_idx's scheme, B = AR.B_BOOT, rng 0) is drawn over the clusters present
in both routes and applied to both.  a2_read is imported, not edited.
    python experiments/raaf_probe_contrast.py --a DIR+DIR --an in.tok --b DIR+DIR --bn in.tok --out results/flawed/raaf/a2/contrast_A_A0.json
Writes once.  Shares: in.tok (A, A0) or in.tokcrop (B)."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import a2_read as AR                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--an", default="in.tok")
    ap.add_argument("--bn", default="in.tok")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), "never overwrite"
    oa, ob = AR.load(a.a.split("+")), AR.load(a.b.split("+"))
    Ea, Ta = AR.table(oa, [a.an, "in.prefix"])
    Eb, Tb = AR.table(ob, [a.bn, "in.prefix"])

    def cl(obs):
        d = {}
        for i, o in enumerate(obs):
            d.setdefault((o["meta"]["set"], str(o["meta"]["cluster"])), []).append(i)
        return d
    ca, cb = cl(oa), cl(ob)
    common = {s: sorted(k for k in ca if k in cb and k[0] == s) for s in AR.SETS}
    print("clusters a %d b %d common %s" % (len(ca), len(cb), {s: len(v) for s, v in common.items()}), flush=True)

    def stat(keys):
        ia = np.array([i for k in keys for i in ca[k]], int)
        ib = np.array([i for k in keys for i in cb[k]], int)
        sa = [np.nanmean(Ea[ia][:, k]) / np.nanmean(Ta[ia]) for k in (0, 1)]
        sb = [np.nanmean(Eb[ib][:, k]) / np.nanmean(Tb[ib]) for k in (0, 1)]
        return sa + sb
    allk = [k for s in AR.SETS for k in common[s]]
    pt = stat(allk)
    rng = np.random.default_rng(0)
    reps = []
    for _ in range(AR.B_BOOT):
        keys = []
        for s in AR.SETS:
            c = common[s]
            keys += [c[j] for j in rng.integers(0, len(c), len(c))] if c else []
        reps.append(stat(keys))
    reps = np.array(reps)
    r = dict(n_a=len(oa), n_b=len(ob), n_common_clusters=len(allk), a=a.a, b=a.b, B=AR.B_BOOT)
    for j, nm in enumerate(["a_where", "a_prefix", "b_where", "b_prefix"]):
        r[nm] = dict(point=pt[j], ci=AR.ci(reps[:, j]))
    r["where_diff"] = dict(point=pt[0] - pt[2], ci=AR.ci(reps[:, 0] - reps[:, 2]))
    r["prefix_diff"] = dict(point=pt[1] - pt[3], ci=AR.ci(reps[:, 1] - reps[:, 3]))
    json.dump(r, open(a.out, "x", encoding="utf-8"), indent=1)
    print(json.dumps(r), flush=True)


main()

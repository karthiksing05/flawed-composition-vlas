"""Read H (docs/HAND_OVER.md 9; m3_prereg pp_read_h): do positions move by more than the 0.02 m budget?  CPU, no rollouts.  -> results/flawed/pp_read_h.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

n2 = PA.tasks()
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
BUDGET = 0.02
HD = os.path.join(PA.OUT, "pp_hand")


def p95(x):
    x = np.asarray(x, np.float64)[:, :2]
    return float(np.percentile(np.linalg.norm(x - x.mean(0), axis=1), 95))


res = {}
for lay in ("familiar", "new"):
    tg, ob = {}, {}
    for c in PA.NEW + PA.KNOWN:
        t = tid[(c, lay)]
        for r in PA.load_track(os.path.join(PA.OD, f"pi05_{t['tid']}_track.npz")):
            tg.setdefault(t["target"], []).append(r["tgt"][0])
            ob.setdefault(t["object"], []).append(r["obj"][0])
    a, b = {k: p95(v) for k, v in tg.items()}, {k: p95(v) for k, v in ob.items()}
    res[lay] = dict(target=a, target_verdict="A TABLE SUFFICES" if max(a.values()) <= BUDGET else "NEEDS PERCEPTION",
                    object=b, object_verdict="A TABLE SUFFICES" if max(b.values()) <= BUDGET else "NEEDS PERCEPTION", offset={})
    for model in ("pi05", "groot"):
        off = {}
        for c in PA.NEW + PA.KNOWN:
            t = tid[(c, lay)]
            z = np.load(os.path.join(HD, f"{model}_H_{t['tid']}_track.npz"))
            for i in range(int(z["n"])):
                k = int(z[f"e{i}_handed"])
                if k >= 0:
                    off.setdefault(t["object"], []).append(z[f"e{i}_obj"][k] - z[f"e{i}_ee"][k])
        res[lay]["offset"][model] = {k: dict(n=len(v), p95=p95(v)) for k, v in off.items()}
    print(lay, json.dumps(res[lay]), flush=True)
json.dump(res, open(os.path.join(PA.OUT, "pp_read_h.json"), "w", encoding="utf-8"), indent=1)

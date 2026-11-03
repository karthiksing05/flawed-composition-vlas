"""PLAN_PARTS round diagnostics (descriptive, demo data only): each part's own support radius under rounds 5-6's rule.
Per object atom: the 99th percentile (and median, max) of the horizontal hand-to-object distance on its not-held demo rows (round 5's
r is the minimum of these over the build's objects).  Per target atom: the same for the horizontal object-to-target distance on its
held demo rows (round 6's r_t is the minimum over targets), plus that distance at the hold's onset (median, max).  Full library
(all 8 known tasks).  CPU.  -> results/flawed/pp_support_diag.json

    python experiments/pp_support_diag.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as PT                                                     # noqa: E402

b, X, _, _, _, _ = PT.rows(torch.device("cpu"), list(FC.KNOWN))
idx = torch.arange(len(b["oi"]))
held = b["held"].numpy() > 0.5
oi, ti = b["oi"].numpy(), b["ti"].numpy()
dxy = np.linalg.norm((b["ee_pos"] - b["pos"][idx, b["oi"]])[:, :2].numpy(), axis=1)
txy = np.linalg.norm((b["pos"][idx, b["oi"]] - b["pos"][idx, b["ti"]])[:, :2].numpy(), axis=1)
onset = held & np.r_[True, ~held[:-1]]
out = dict(objects={}, targets={})
q = lambda v: dict(n=int(len(v)), p99=float(np.percentile(v, 99)), median=float(np.median(v)), max=float(v.max()))   # noqa: E731
for o in np.unique(oi[~held]):
    out["objects"][FC.ENT[o]] = q(dxy[(~held) & (oi == o)])
for t in np.unique(ti[held]):
    r = q(txy[held & (ti == t)])
    v = txy[onset & (ti == t)]
    r.update(onset_median=float(np.median(v)), onset_max=float(v.max()), onset_n=int(len(v)))
    out["targets"][FC.ENT[t]] = r
for k, v in out.items():
    for e, r in v.items():
        print("[support]", k, e, json.dumps({x: (round(y, 4) if isinstance(y, float) else y) for x, y in r.items()}), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "flawed", "pp_support_diag.json"), "w", encoding="utf-8"), indent=1)
